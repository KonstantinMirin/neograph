"""Unified DI resolution module.

Single module with one resolver path for all DI parameters.

Four types/helpers live here:
- ``DIKind`` — enum of 6 DI parameter kinds (no UPSTREAM)
- ``DIBinding`` — a fully resolved DI parameter binding
- ``_unwrap_loop_value`` — Loop append-list unwrap (shared by factory, compiler, DI)
- ``_unwrap_each_dict`` — Each dict-to-list unwrap (shared by factory, DI)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Union
from typing import get_args as _get_args
from typing import get_origin as _get_origin

import structlog
from pydantic import ValidationError

from neograph.errors import ConfigurationError as _ConfigurationError
from neograph.errors import ExecutionError as _ExecutionError

log = structlog.get_logger(__name__)

# Config['configurable'] key holding the consumer-supplied resource fetcher —
# an ``async def fetch(uri) -> (content, mime)`` callable. Consumer-owned, exactly
# like tool factories and the per-run token provider (no session ownership). Shared
# by ``resource_reader`` (tool.py) and ``FromResource`` DI (DIBinding.aresolve).
RESOURCE_FETCHER_KEY = "mcp_resource_fetcher"


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
# LLM-mode node (keyed by the parameter name). Excludes CONSTANT (a plain
# default, not run-input context) and FROM_STATE (merge_fn-only, meaningless
# outside an Oracle merge). Shared by the LLM dispatch layer (which resolves and
# stashes these — see `_dispatch._inject_di_inputs`) and lint (which treats them
# as valid template-ref placeholders when the prompt_compiler accepts di_inputs).
DI_TEMPLATE_KINDS: frozenset[DIKind] = frozenset({
    DIKind.FROM_INPUT, DIKind.FROM_CONFIG,
    DIKind.FROM_INPUT_MODEL, DIKind.FROM_CONFIG_MODEL,
})


def _get_configurable(config: Any, key: str) -> Any:
    """Read a key from config['configurable'], handling dict and attr forms."""
    cfg = config or {}
    if isinstance(cfg, dict):
        return cfg.get("configurable", {}).get(key)
    return getattr(cfg, "configurable", {}).get(key)


def parse_resource_content(
    content: Any,
    mime: str | None,
    output_model: Any,
    parse: Any = None,
    *,
    marker_mime: str | None = None,
) -> Any:
    """Turn a fetched resource blob into the declared type. v1 rules:

    - explicit ``parse(content, mime)`` callable wins (the general escape hatch);
    - ``str`` target -> raw text passthrough (decode bytes);
    - ``application/json`` (or ``*+json``, or an absent mime) into a BaseModel ->
      ``model_validate_json``;
    - any other mime into a BaseModel with no parser -> FAIL LOUD. neograph NEVER
      runs a silent LLM parse inside DI/resource resolution (banned hidden
      cognition + cost). The caller supplies an explicit ``parse=`` for text/*.

    Shared by ``resource_reader`` (tool.py) and ``FromResource`` (aresolve).
    """
    if parse is not None:
        return parse(content, mime)

    if output_model is str:
        return content.decode() if isinstance(content, bytes) else content

    base = (mime or marker_mime or "").split(";")[0].strip().lower()
    if base == "" or base == "application/json" or base.endswith("+json"):
        return output_model.model_validate_json(content)

    model_name = getattr(output_model, "__name__", str(output_model))
    raise _ConfigurationError.build(
        f"resource mime '{mime or marker_mime or '?'}' cannot be parsed into "
        f"{model_name} without an explicit parser",
        hint="pass parse=(content, mime)->model (or type the param as str for raw "
             "text passthrough); neograph never runs a silent LLM parse inside "
             "DI/resource resolution",
    )


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
    model_cls: Any = None      # MODEL kinds only: the Pydantic BaseModel subclass
    uri: str | None = None          # FROM_RESOURCE only: the (static, v1) resource URI
    parse_fn: Any = None            # FROM_RESOURCE only: explicit (content, mime)->model
    resource_mime: str | None = None  # FROM_RESOURCE only: mime hint from the marker

    def resolve(self, config: Any, *, state: Any = None) -> Any:
        """The ONE resolution path for DI parameters.

        For FROM_STATE (merge_fn only), ``state`` must be provided.
        For all other kinds, reads from ``config['configurable']``.
        """
        if self.kind in (DIKind.FROM_INPUT, DIKind.FROM_CONFIG):
            val = _get_configurable(config, self.name)
            if val is None and self.required:
                source = "input" if self.kind == DIKind.FROM_INPUT else "config"
                raise _ExecutionError(
                    f"Required DI parameter '{self.name}' (from {source}) is missing "
                    f"from config['configurable']. Provide it via "
                    f"run(input={{'{self.name}': ...}})."
                )
            if val is not None and self.inner_type is not None:
                if not _isinstance_safe(val, self.inner_type):
                    source = "input" if self.kind == DIKind.FROM_INPUT else "config"
                    raise _ExecutionError(
                        f"DI parameter '{self.name}' (from {source}) expects "
                        f"{self.inner_type.__name__}, got {type(val).__name__}. "
                        f"Provide the correct type via run(input={{'{self.name}': ...}})."
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
                    raise _ExecutionError(
                        f"Required DI bundled model '{self.name}' ({model.__name__}) "
                        f"is missing fields from {source}: {sorted(missing)}. "
                        f"Provide them via run(input={{...}})."
                    )
            try:
                return model(**field_values)
            except (ValidationError, TypeError, ValueError):
                if self.required:
                    source = "input" if self.kind == DIKind.FROM_INPUT_MODEL else "config"
                    raise _ExecutionError(
                        f"Required DI bundled model '{self.name}' ({model.__name__}) "
                        f"construction failed. Provide all fields via "
                        f"run(input={{...}}) or config['configurable']."
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

        FROM_RESOURCE: read the consumer-supplied fetcher from
        ``config['configurable'][RESOURCE_FETCHER_KEY]`` (async
        ``fetch(uri) -> (content, mime)``), await it, and parse the blob into
        ``inner_type`` via ``parse_resource_content`` (v1: static URI; json ->
        model_validate_json; text -> explicit parser or str).
        """
        if self.kind != DIKind.FROM_RESOURCE:
            return self.resolve(config, state=state)

        fetcher = _get_configurable(config, RESOURCE_FETCHER_KEY)
        if fetcher is None:
            if not self.required:
                return None
            raise _ConfigurationError.build(
                f"resource DI parameter '{self.name}' has no fetcher to resolve from",
                hint=f"provide config['configurable']['{RESOURCE_FETCHER_KEY}'] = "
                     "an async 'fetch(uri) -> (content, mime)' callable",
            )
        content, mime = await fetcher(self.uri)
        return parse_resource_content(
            content, mime, self.inner_type, self.parse_fn,
            marker_mime=self.resource_mime,
        )
