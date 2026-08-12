"""Unified DI resolution module.

Single module with one resolver path for all DI parameters.

Four types/helpers live here:
- ``DIKind`` — enum of 6 DI parameter kinds (no UPSTREAM)
- ``DIBinding`` — a fully resolved DI parameter binding
- ``_unwrap_loop_value`` — Loop append-list unwrap (shared by factory, compiler, DI)
- ``_unwrap_each_dict`` — Each dict-to-list unwrap (shared by factory, DI)
- ``read_upstream`` — THE state-bus read: the two unwraps composed with the read,
  the exact sibling of ``DIBinding.resolve``'s config-side read-then-unwrap
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Union
from typing import get_args as _get_args
from typing import get_origin as _get_origin

import structlog
from pydantic import ValidationError

# --- names di.py imported and RE-EXPORTED before the split; the hydration
# --- cluster was their only local consumer here.
from neograph._content_blocks import (  # noqa: E402,F401
    _first_resource_link_uri,
    _resource_link_uri_for_kind,
)

# --- extracted cluster (neograph-3ffdg.19), re-exported so existing
# --- `from neograph.di import ...` call sites keep resolving unchanged.
# --- _get_configurable is imported BACK: it moved with the hydration cluster but
# --- DIBinding here still uses it.
from neograph._resource_hydration import (  # noqa: E402,F401
    _FETCHER_HINT,
    RESOURCE_FETCHER_KEY,
    RESOURCE_REPLAYER_KEY,
    _enforce_max_bytes,
    _get_configurable,
    _require_fetcher,
    hydrate_resource_ref,
    parse_resource_content,
)
from neograph._state_keys import StateKeys
from neograph._uri_template import _expand_uri
from neograph.errors import ConfigurationError as _ConfigurationError
from neograph.errors import ExecutionError as _ExecutionError
from neograph.errors import NonIdempotentReplayError as _NonIdempotentReplayError  # noqa: E402,F401
from neograph.errors import ResourceExpiredError as _ResourceExpiredError  # noqa: E402,F401
from neograph.naming import field_name_for

if TYPE_CHECKING:
    from neograph._state_bus import StateBus

log = structlog.get_logger()


class DIKind(Enum):
    """How a DI parameter gets its value at runtime.

    Six kinds. Upstream resolution is NOT a DI kind — it stays in
    ``_extract_input`` in the factory layer.
    """

    FROM_INPUT = "from_input"
    FROM_CONFIG = "from_config"
    FROM_INPUT_MODEL = "from_input_model"
    FROM_CONFIG_MODEL = "from_config_model"
    FROM_STATE = "from_state"  # merge_fn only — resolved from graph state
    CONSTANT = "constant"
    # A resource read via config['configurable']['mcp_resource_fetcher']. The
    # fetch is AWAITED, so this kind resolves ONLY through the async twin
    # ``DIBinding.aresolve``; the sync ``resolve()`` fails loud (see below).
    FROM_RESOURCE = "from_resource"


# DI kinds whose resolved value can serve as a prompt template variable on an
# LLM-mode node (keyed by the parameter name) SYNCHRONOUSLY. Excludes CONSTANT (a
# plain default, not run-input context) and FROM_STATE (merge_fn-only, meaningless
# outside an Oracle merge). Also excludes FROM_RESOURCE: a fetched resource is not
# ambient run-input and the SYNC di_inputs injection cannot await its fetch — the
# async twin `_dispatch._ainject_di_inputs` awaits `DIBinding.aresolve` for
# FROM_RESOURCE and stashes it as a template var on the arun() path, while the sync
# path fails loud (neograph-3q6j; see `_dispatch._inject_di_inputs`). Shared by the
# LLM dispatch layer (which resolves and stashes these) and lint (which treats them
# as valid template-ref placeholders when the prompt_compiler accepts di_inputs).
DI_TEMPLATE_KINDS: frozenset[DIKind] = frozenset(
    {
        DIKind.FROM_INPUT,
        DIKind.FROM_CONFIG,
        DIKind.FROM_INPUT_MODEL,
        DIKind.FROM_CONFIG_MODEL,
    }
)


def _configurable_dict(config: Any) -> dict[str, Any]:
    """The whole ``config['configurable']`` mapping (for URI-template interpolation)."""
    cfg = config or {}
    if isinstance(cfg, dict):
        return cfg.get("configurable", {}) or {}
    return getattr(cfg, "configurable", {}) or {}


def read_upstream(
    state: StateBus,
    key: str,
    expected_type: Any,
    *,
    required: bool,
    node_label: str = "",
) -> Any:
    """Read ONE upstream node's output off the state bus, shaped for its consumer.

    THE reader see neograph-13k4i. Reading an upstream is a RULE, not a line:
    resolve the field name, read it, then unwrap the Loop append-list and the
    Each result dict **against the type the consumer declared**. Both unwraps
    no-op unless the stored value and the declared type actually disagree, which
    is why one function serves every caller.

    ``required`` is the only real variation between call sites, so it is a
    parameter rather than a second function: a fan-in upstream is guaranteed by
    ``_validate_node_chain``, so its absence is a wiring bug worth failing loud
    on, while merge-prompt context and loop-bootstrap reads are best-effort and
    treat absence as "nothing to say".

    Lives HERE, beside the two unwraps it composes, rather than in
    ``_input_shape`` where its callers live: ``_oracle`` and ``_input_shape`` are
    both declared leaves of the assembly-cluster import DAG, and ``_input_shape``
    is by guard reserved to node-body executors. A shared helper in a leaf module
    would have meant widening one of those two rules to buy a fix — so the helper
    moved to the module that already owns both halves of the rule and is already
    imported by both callers. It is the state-bus sibling of ``DIBinding.resolve``,
    which does the same read-then-unwrap-against-declared-type over config.

    Before this existed, ``_oracle._build_upstream_context`` re-derived the read
    and knew only the first half, so an Each-produced upstream reached the node's
    own prompt as the declared ``list[X]`` and the merge prompt as the raw Each
    dict. ``tests/test_guards_upstream_read.py`` keeps a second reader from
    reappearing.
    """
    state_key = field_name_for(key)
    # StateBus.get optional (required=False): best-effort callers — merge-prompt
    # context treats an absent upstream as nothing to say, and loop bootstrap has
    # no sibling value on iteration 0.
    value = state.get_required(state_key, node_label=node_label) if required else state.get(state_key)
    value = _unwrap_loop_value(value, expected_type)
    return _unwrap_each_dict(value, expected_type)


def _unwrap_loop_value(val: Any, expected_type: Any) -> Any:
    """Unwrap a Loop append-list to the latest value.

    Loop nodes store list[T] in state via the append reducer. Consumers
    see the latest value ([-1]), not the full list. Empty list (first
    iteration) returns None. If the consumer expects list[T], the value
    passes through unchanged.

    This is the single source of truth for Loop unwrap — used by
    _extract_input, _resolve_merge_args, and loop_router.
    """
    if val is None:
        return None
    if not isinstance(val, list):
        return val
    # Consumer wants a list → pass through
    if _get_origin(expected_type) is list:
        return val
    # Empty list = first iteration → None
    if not val:
        return None
    return val[-1]


def _unwrap_each_dict(val: Any, expected_type: Any) -> Any:
    """Unwrap a dict[str, X] Each result to list[X] when consumer expects list.

    Each fan-out produces dict[str, X] keyed by each.key. A downstream
    consumer declaring list[X] gets list(dict.values()). Non-list consumers
    and non-dict values pass through unchanged.
    """
    if val is None:
        return None
    if not isinstance(val, dict):
        return val
    if _get_origin(expected_type) is not list:
        return val
    return list(val.values())


def _isinstance_safe(val: Any, tp: type) -> bool:
    """Check if val is an instance of tp, handling Union/Optional and generic origins."""
    origin = _get_origin(tp)

    # Union / Optional — unwrap and check each branch
    if origin is Union:
        args = _get_args(tp)
        return any(_isinstance_safe(val, a) for a in args if a is not type(None))

    # Generic origins (list, dict, set, tuple, etc.) — check container type only
    if origin is not None:
        return isinstance(val, origin)

    # Plain type — direct isinstance
    try:
        return isinstance(val, tp)
    except TypeError:
        return False


@dataclass
class DIBinding:
    """A fully resolved DI parameter binding.

    Created at assembly time by ``_classify_di_params``, resolved at
    runtime by ``resolve()``.

    Typed fields per kind (no untagged payload):
    - CONSTANT: ``default_value`` holds the literal default
    - FROM_INPUT_MODEL / FROM_CONFIG_MODEL: ``model_cls`` holds the Pydantic model class
    - FROM_INPUT / FROM_CONFIG: no extra fields needed (``required`` + ``inner_type`` suffice)
    - FROM_STATE: no extra fields needed (``inner_type`` used for loop unwrap)
    """

    name: str
    kind: DIKind
    inner_type: type
    required: bool
    default_value: Any = None  # CONSTANT only: the literal default value
    model_cls: Any = None  # MODEL kinds only: the Pydantic BaseModel subclass
    uri: str | None = None  # FROM_RESOURCE only: the (static or templated) URI
    parse_fn: Any = None  # FROM_RESOURCE only: explicit (content, mime)->model
    resource_mime: str | None = None  # FROM_RESOURCE only: mime hint from the marker
    ref_kind: str | None = None  # FROM_RESOURCE(ref=) only: manifest KIND to hydrate
    max_bytes: int | None = None  # FROM_RESOURCE only: fail-loud size cap at node entry

    def resolve(self, config: Any, *, state: Any = None) -> Any:
        """The ONE resolution path for DI parameters.

        For FROM_STATE (merge_fn only), ``state`` must be provided.
        For all other kinds, reads from ``config['configurable']``.
        """
        if self.kind in (DIKind.FROM_INPUT, DIKind.FROM_CONFIG):
            val = _get_configurable(config, self.name)
            if val is None and self.required:
                source = "input" if self.kind == DIKind.FROM_INPUT else "config"
                raise _ExecutionError.build(
                    f"required DI parameter '{self.name}' (from {source}) is missing from config['configurable']",
                    hint=f"provide it via run(input={{'{self.name}': ...}})",
                )
            if val is not None and self.inner_type is not None:
                if not _isinstance_safe(val, self.inner_type):
                    source = "input" if self.kind == DIKind.FROM_INPUT else "config"
                    raise _ExecutionError.build(
                        f"DI parameter '{self.name}' (from {source}) has the wrong type",
                        expected=self.inner_type.__name__,
                        found=type(val).__name__,
                        hint=f"provide the correct type via run(input={{'{self.name}': ...}})",
                    )
            return val

        if self.kind in (DIKind.FROM_INPUT_MODEL, DIKind.FROM_CONFIG_MODEL):
            model = self.model_cls
            field_values: dict[str, Any] = {}
            for fname in model.model_fields:
                val = _get_configurable(config, fname)
                if val is not None:
                    field_values[fname] = val
            if self.required:
                missing = [f for f in model.model_fields if f not in field_values]
                if missing:
                    source = "input" if self.kind == DIKind.FROM_INPUT_MODEL else "config"
                    raise _ExecutionError.build(
                        f"required DI bundled model '{self.name}' ({model.__name__}) is missing fields from {source}",
                        found=f"missing: {sorted(missing)}",
                        hint="provide them via run(input={...})",
                    )
            try:
                return model(**field_values)
            except (ValidationError, TypeError, ValueError):
                if self.required:
                    source = "input" if self.kind == DIKind.FROM_INPUT_MODEL else "config"
                    raise _ExecutionError.build(
                        f"required DI bundled model '{self.name}' ({model.__name__}) construction failed",
                        hint="provide all fields via run(input={...}) or config['configurable']",
                    ) from None
                log.warning(
                    "DI model construction failed, returning None",
                    model=model.__name__,
                    param=self.name,
                    fields=field_values,
                )
                return None

        if self.kind == DIKind.CONSTANT:
            return self.default_value

        if self.kind == DIKind.FROM_STATE:
            val = getattr(state, self.name, None) if state is not None else None
            val = _unwrap_loop_value(val, self.inner_type)
            return val

        if self.kind == DIKind.FROM_RESOURCE:
            # The fetch is awaited — it CANNOT resolve on the sync path. Fail loud
            # rather than silently drop the value (mirrors the async-only-tool /
            # async-body-under-run guards). Reached directly (e.g. the scripted
            # shim under the sync driver builds an async shim and ScriptedDispatch
            # .execute fails loud first; this is the direct-call safety net).
            raise _ConfigurationError.build(
                f"resource DI parameter '{self.name}' cannot resolve synchronously",
                hint="FromResource fetches are awaited — drive the graph with "
                "arun(). resolve() has no fetcher to await; use aresolve().",
            )

        return None

    async def aresolve(self, config: Any, *, state: Any = None) -> Any:
        """Async resolution twin. Only FROM_RESOURCE needs to await (it fetches);
        every other kind delegates to the synchronous ``resolve``.

        FROM_RESOURCE has two faces neograph-a5nh:

        * **URI mode** (``uri=``, v1 + v2 templated): the URI is interpolated from
          ``config['configurable']`` values (RFC-6570 vars, e.g. ``{deal_id}``
          bound from a sibling ``FromInput``), fetched once via the consumer's
          ``RESOURCE_FETCHER_KEY`` fetcher, size-checked against ``max_bytes``
          (fail loud BEFORE parse so an oversized blob never reaches a prompt),
          then parsed into ``inner_type``.
        * **Manifest mode** (``ref_kind=``, v2): the matching ``ResourceRef`` is
          looked up in the injected manifest (``StateKeys.RESOURCE_MANIFEST_INJECT``,
          populated from the checkpointed manifest channel by
          ``_execute._inject_resource_manifest``) and hydrated with LAYERED EXPIRY
          (read -> replay producing_call -> fail loud) via ``hydrate_resource_ref``.
        """
        if self.kind != DIKind.FROM_RESOURCE:
            return self.resolve(config, state=state)

        if self.ref_kind is not None:
            return await self._aresolve_from_manifest(config)

        fetcher = _require_fetcher(
            config,
            subject=f"resource DI parameter '{self.name}' has no fetcher to resolve from",
            required=self.required,
        )
        if fetcher is None:
            return None
        uri = _expand_uri(self.uri or "", _configurable_dict(config))
        content, mime = await fetcher(uri)
        _enforce_max_bytes(content, self.max_bytes, name=self.name, uri=uri)
        return parse_resource_content(
            content,
            mime,
            self.inner_type,
            self.parse_fn,
            marker_mime=self.resource_mime,
        )

    async def _aresolve_from_manifest(self, config: Any) -> Any:
        """Manifest-driven hydration for a ``FromResource(ref=<kind>)`` binding.

        Selects the FIRST ``ResourceRef`` of ``self.ref_kind`` from the injected
        manifest and hydrates it with layered expiry. A missing kind fails loud
        (naming the param + kind) rather than silently returning ``None`` — a
        confusing far-from-cause miss is worse than a clear one; lint
        (``resource_hydration_kind_unmatched``) catches the static case earlier.
        """
        manifest = _get_configurable(config, StateKeys.RESOURCE_MANIFEST_INJECT) or []
        match = next((r for r in manifest if getattr(r, "kind", None) == self.ref_kind), None)
        if match is None:
            if not self.required:
                return None
            raise _ExecutionError.build(
                f"resource DI parameter '{self.name}' found no ResourceRef of kind '{self.ref_kind}' in the manifest",
                hint="an upstream agent/act node must emit a resource_link of that "
                "kind before this node runs (flat servers that emit no "
                "resource_link fall back to a templated FromResource(uri=...) / "
                "resource_reader tool)",
            )
        return await hydrate_resource_ref(
            match,
            config,
            self.inner_type,
            parse=self.parse_fn,
            marker_mime=self.resource_mime,
            max_bytes=self.max_bytes,
            node=self.name,
        )
