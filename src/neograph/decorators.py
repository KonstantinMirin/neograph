"""@node decorator and construct_from_module — Dagster-style pipeline definition.

Ergonomic front-end on top of Node + Construct: the function signature IS the
dependency graph.

    @node(mode="scripted", outputs=Claims)
    def decompose(topic: RawText) -> Claims: ...

    @node(mode="scripted", outputs=Classified)
    def classify(decompose: Claims) -> Classified: ...
    # parameter name 'decompose' matches upstream node 'decompose' → auto-wires

    pipeline = construct_from_module(sys.modules[__name__])

Design notes

* Mirrors `src/neograph/tool.py:89-133` — same two-form call shape (`@node`
  vs `@node(...)`), same function-local factory import to dodge the
  `factory → node → decorators` circular path, same "return a spec instance
  rather than the wrapped function" contract.

* The decorator stores the original function and its parameter-name tuple
  on the Node via Pydantic PrivateAttr fields (_sidecar, _param_res).
  These are preserved by model_copy when modifiers are applied via |.

* Scripted @node functions are dispatched via `register_scripted` — at
  `_build_construct_from_decorated` time a shim closure is synthesized for
  each scripted node and registered under a unique name. The shim reads N
  upstream values by parameter name from `input_data`, resolves DI params
  (FromInput/FromConfig/constant) from `config`, and calls the user
  function with positional args. The factory's unified `_execute_node` path
  picks up the registered shim via `ScriptedDispatch`. Non-scripted modes
  (think / agent / act) use `ThinkDispatch` / `ToolDispatch`; their
  parameter annotations only drive topology + type inference.

* `construct_from_module` walks `vars(mod)` once, keeping every pipeline
  member per the single `_classify_member` predicate (@node Nodes, plain
  `Node(...)` instances, and sub-`Construct`s with an `output=` boundary),
  builds adjacency from each node's parameter-name tuple, DFS
  topological-sorts with a visiting set for cycle detection, and hands the
  sorted list to `Construct(name=..., nodes=...)`. No new validation path:
  the existing `_validate_node_chain` runs via `Construct.__init__`.

* Name convention: function `foo_bar` → node name `'foo-bar'`; a downstream
  parameter `foo_bar: T` looks up the node via `field_name_for(name)`.
  Matches the state-field convention everywhere else in the codebase.

* v1 scope: every parameter must name an upstream `@node` in the module.
  Scalars and run-input kwargs are out of scope (they raise `ConstructError`).
  `*args` / `**kwargs` are rejected at decoration time.
"""

from __future__ import annotations

import ast
import inspect
import secrets
import sys
import textwrap
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass


from neograph._construct_validation import ConstructError
from neograph._di_classify import (  # noqa: F401 — re-exported for backward compat
    FromConfig,
    FromInput,
    FromResource,
    ParamResolution,
    _build_annotation_namespace,
    _classify_di_params,
    _resolve_di_args,
    _resolve_merge_args,
)
from neograph._hints import resolve_hints
from neograph._llm_config import LlmConfig

# --- extracted clusters (neograph-3ffdg.11), re-exported so existing
# --- `from neograph.decorators import ...` call sites keep resolving unchanged.
from neograph._merge_fn_decorator import (  # noqa: E402,F401
    _qualname_site,
    _same_def_site,
)
from neograph._merge_fn_decorator import merge_fn as _merge_fn_impl  # noqa: E402
from neograph._node_modifier_kwargs import (  # noqa: E402,F401
    _apply_eager_oracle_gen_type,
    _build_each_kwargs,
    _build_oracle_kwargs,
    _build_portal_kwargs,
    _is_trivial_body,
)

# Decorator-side shim registration. The decorators emit shims for inline
# body-merge functions (`@node(merge_fn=callable)`), inline interrupt-when
# conditions (`@node(interrupt_when=callable)`), `@merge_fn`, and `@tool`.
# These run at IMPORT time, before compile() exists, so the store lives in the
# leaf `_runtime_registry` module — decorators.py owns ZERO module-level
# mutable dicts (neograph-v3xx HIGH-01). `register_*` are re-exported here so
# existing call sites (and `@tool` in tool.py) keep working unchanged.
from neograph._runtime_registry import (  # noqa: F401 — re-exported registration API
    register_condition,
    register_scripted,
    register_tool_factory,
)
from neograph._sidecar import (  # noqa: F401 — re-exported for backward compat
    _get_node_source,
    _get_param_res,
    _get_sidecar,
    _merge_fn_caller_ns,
    _merge_fn_registry,
    _register_sidecar,
    _set_param_res,
    get_merge_fn_metadata,
    infer_oracle_gen_type,
)
from neograph.describe_type import type_display_name
from neograph.modifiers import Each, Loop, Operator, Oracle, Portal
from neograph.node import Node
from neograph.renderers import Renderer
from neograph.tool import Tool


def node(
    fn: Callable | None = None,
    *,
    mode: Literal["think", "agent", "act", "scripted", "raw"] | None = None,
    inputs: Any = None,
    outputs: Any = None,
    model: str | None = None,
    prompt: str | None = None,
    llm_config: dict[str, Any] | LlmConfig | None = None,
    tools: list[Tool] | None = None,
    name: str | None = None,
    map_over: str | None = None,
    map_key: str | None = None,
    map_on_error: Literal["raise", "collect"] = "raise",
    ensemble_n: int | None = None,
    models: list[str] | None = None,
    merge_fn: str | None = None,
    merge_prompt: str | None = None,
    merge_pre_process: Callable | None = None,
    merge_post_process: Callable | None = None,
    merge_fallback: Callable | None = None,
    merge_model: str | None = None,
    interrupt_when: str | Callable | None = None,
    renderer: Renderer | None = None,
    context: list[str] | None = None,
    skip_when: Callable | None = None,
    skip_value: Callable | None = None,
    gate_tools_when: Callable | str | None = None,
    loop_when: str | Callable | None = None,
    max_iterations: int | None = None,
    on_exhaust: Literal["error", "last", "exit"] | None = None,
    portal: list[str] | None = None,
    route: str | None = None,
    max_hops: int | None = None,
) -> Any:
    """Decorator that turns a function into a Node spec with signature-inferred
    dependencies. Supports both `@node` and `@node(...)` call forms.

    Inference rules:
        * `name`    ← kwarg, else `fn.__name__.replace("_", "-")`
        * `outputs` ← kwarg, else function return annotation.
          When both are present and differ, raises ConstructError
          (dict-form outputs= exempt — multi-output can't be annotated).
        * `inputs`  ← kwarg, else annotation of the first annotated parameter

    Fan-out via Each::

        @node(map_over='make_clusters.groups', map_key='label')
        def verify(cluster: ClusterGroup) -> MatchResult: ...

    When ``map_over`` is set the node is automatically composed with
    ``Each(over=map_over, key=map_key)``. The first parameter whose name does
    NOT match any upstream ``@node`` is treated as the fan-out item receiver;
    ``construct_from_module`` skips it in topology wiring.
    ``map_on_error='collect'`` forwards to ``Each(on_error='collect')``
    (per-item failures collected instead of raised).

    Oracle ensemble::

        @node(mode='produce', outputs=Claims, prompt='rw/decompose', model='reason',
              ensemble_n=3, merge_prompt='rw/decompose-merge')
        def decompose(topic: RawText) -> Claims: ...

    When any of ``ensemble_n``, ``merge_fn``, or ``merge_prompt`` is set the
    node is composed with ``Oracle(n=..., merge_fn=..., merge_prompt=...)``.
    Exactly one of ``merge_fn`` or ``merge_prompt`` is required; ``ensemble_n``
    defaults to 3 if omitted. ``merge_model='<tier>'`` forwards to
    ``Oracle(merge_model=...)`` — the model tier the merge call uses
    (default ``'reason'``; like the programmatic Oracle, it is silently
    ignored on the ``merge_fn`` path).

    Loop self-refinement::

        @node(outputs=Draft, prompt='rw/draft', model='reason',
              loop_when='needs_work', max_iterations=5)
        def draft(topic: RawText) -> Draft: ...

    A self-loop node's output field accumulates every iteration as a list
    (the ``_append_loop_result`` reducer), so ``result[node]`` is the full
    iteration history; the last element is the final value.

    Merge hooks (``merge_prompt`` path only)::

        @node(outputs=Claims, prompt='rw/decompose', model='reason',
              ensemble_n=3, merge_prompt='rw/merge',
              merge_pre_process=tag_variants,     # fn(variants) -> dict
              merge_post_process=validate_result,  # fn(result, variants) -> result
              merge_fallback=deterministic_merge)  # fn(variants, error) -> result

    ``merge_pre_process`` replaces the default input_data construction.
    ``merge_post_process`` transforms the LLM result (skipped on fallback).
    ``merge_fallback`` catches LLM errors and returns a deterministic result.
    All three are optional and invalid with ``merge_fn``.

    Human-in-the-loop via Operator::

        @node(mode='scripted', outputs=ValidationResult,
              interrupt_when='validation_failed')
        def validate(claims: Claims) -> ValidationResult: ...

    When ``interrupt_when`` is set the node is composed with
    ``Operator(when=...)``. The value can be a string (registered condition
    name) or a callable (auto-registered under a synthesized name).

    For `mode='scripted'`, a shim is registered via `register_scripted` at
    `_build_construct_from_decorated` time and dispatched through
    `factory._execute_node` via `ScriptedDispatch`. Supports fan-in
    (>1 parameter) nodes uniformly.
    """
    # Capture the caller's local namespace once, at decoration time.
    # For @node(...) form: node() is called from user code, _getframe(1)
    # is the user's scope. For bare @node: same — node(fn=f) is still
    # called from user code. The closure carries it into decorator(f).
    caller_ns = sys._getframe(1).f_locals  # noqa: SLF001

    def decorator(f: Callable) -> Node:
        # -- Validate map_over / map_key pairing early -----------------------
        if map_over is not None and map_key is None:
            raise ConstructError.build(
                "map_over= requires map_key=",
                node=(name or f.__name__).replace("_", "-"),
                hint="pass map_key='<field>' to specify the dispatch key on each item",
            )
        if map_key is not None and map_over is None:
            raise ConstructError.build(
                "map_key= requires map_over=",
                node=(name or f.__name__).replace("_", "-"),
                hint="pass map_over='<dotted.path>' to specify the collection to fan out over",
            )
        if map_over is not None and loop_when is not None:
            raise ConstructError.build(
                "map_over= (Each) and loop_when= (Loop) cannot be combined on the same node",
                node=(name or f.__name__).replace("_", "-"),
                hint="use a sub-construct with Loop inside an Each fan-out instead",
            )

        # -- Validate Portal peer-mode sugar (portal=/route=/max_hops=) -------
        # ``portal=`` builds a Portal, exactly as ``loop_when=`` builds a Loop.
        # It owns the node's outgoing edge, so it is mutually exclusive with the
        # other edge-shaping modifiers; reject the conflicts at decoration
        # (mirror the map_over/loop_when raise above).
        _km_node = (name or f.__name__).replace("_", "-")
        if portal is not None and map_over is not None:
            raise ConstructError.build(
                "portal= (Portal) and map_over= (Each) cannot be combined on the same node",
                node=_km_node,
                hint="Portal owns the node's outgoing edge; a mesh member cannot also fan out",
            )
        if portal is not None and loop_when is not None:
            raise ConstructError.build(
                "portal= (Portal) and loop_when= (Loop) cannot be combined on the same node",
                node=_km_node,
                hint="Portal owns the node's outgoing edge; a mesh member cannot also self-loop",
            )
        # A peer-mode routing knob with no mesh to attach to is a decoration
        # error (mirror map_key-requires-map_over). ``on_exhaust`` is EXEMPT —
        # it is shared with Loop and routed by trigger below.
        if portal is None and max_hops is not None:
            raise ConstructError.build(
                "max_hops= requires portal=",
                node=_km_node,
                hint="pass portal=[...] to declare a Portal mesh member (max_hops is a mesh budget)",
            )
        if portal is None and route is not None:
            raise ConstructError.build(
                "route= requires portal=",
                node=_km_node,
                hint="pass portal=[...] to declare a Portal mesh member (route is its per-node routing field)",
            )

        # -- Mode inference: if not explicitly set, infer from kwargs ----------
        effective_mode = mode
        if effective_mode is None:
            if prompt is not None or model is not None:
                effective_mode = "think"
            else:
                effective_mode = "scripted"

        node_label = (name or f.__name__).replace("_", "-")

        # -- Decoration-time validation for LLM modes -------------------------
        if effective_mode in ("think", "agent", "act"):
            if prompt is None:
                raise ConstructError.build(
                    f"mode='{effective_mode}' requires prompt=",
                    node=node_label,
                    hint="pass prompt='<template>' or switch to mode='scripted'",
                )
            if model is None:
                raise ConstructError.build(
                    f"mode='{effective_mode}' requires model=",
                    node=node_label,
                    hint="pass model='<model_name>' or switch to mode='scripted'",
                )

            # -- Dead-body warning for LLM modes ------------------------------
            # Check if the function body is non-trivial (more than just `...`,
            # `pass`, or a bare constant/return). Uses AST inspection.
            # Docstrings are stripped before checking — `"""doc""" + ...` is trivial.
            try:
                source = textwrap.dedent(inspect.getsource(f))
                tree = ast.parse(source)
                func_def = next(
                    (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
                    None,
                )
                if func_def is not None:
                    body = func_def.body
                    # Strip leading docstring (string-constant expression).
                    if (
                        body
                        and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)
                    ):
                        body = body[1:]
                    trivial = _is_trivial_body(body)
                    if not trivial:
                        warnings.warn(
                            f"@node '{node_label}': the body of mode='{effective_mode}' "
                            f"functions is not executed; the LLM call via prompt= provides "
                            f"the output. Move this logic into a scripted node, or remove "
                            f"the body and use '...' as placeholder.",
                            UserWarning,
                            stacklevel=3,
                        )
            except (OSError, TypeError):  # pragma: no cover
                # Source not available (e.g. built-in, dynamic) — skip check.
                pass

        sig = inspect.signature(f)

        # -- Raw mode: enforce (state, config) signature ----------------------
        if effective_mode == "raw":
            params = list(sig.parameters.values())
            if len(params) != 2:
                raise ConstructError.build(
                    "mode='raw' requires exactly two parameters (state, config)",
                    node=f.__name__,
                    found=f"{len(params)} parameters",
                )
            if [p.name for p in params] != ["state", "config"]:
                raise ConstructError.build(
                    "mode='raw' parameters must be named 'state' and 'config'",
                    node=f.__name__,
                    found=str([p.name for p in params]),
                )
            # Raw mode: empty param_names — not used for topology.
            param_names: tuple[str, ...] = ()
        else:
            # Reject *args / **kwargs early — they have no sensible mapping to
            # upstream nodes.
            for p in sig.parameters.values():
                if p.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    raise ConstructError.build(
                        f"parameter '{p.name}' is *args/**kwargs, which has no upstream-node mapping",
                        node=f.__name__,
                        hint="use explicit named parameters",
                    )

            param_names = tuple(p.name for p in sig.parameters.values())

        # Classify non-upstream params at decoration time via the shared
        # DI classifier. Handles FromInput[T] / FromConfig[T] including the
        # bundled form FromInput[PydanticModel]. Default-value constants
        # are deferred to construct_from_module (we don't know which param
        # names map to @node upstreams until then).
        param_res: ParamResolution = {}
        if effective_mode != "raw":
            param_res = _classify_di_params(f, sig, caller_ns=caller_ns)

        # Resolve stringified annotations (from __future__ import annotations)
        # ONCE for both output-inference and inputs-inference below. Uses the
        # shared resolve_hints, which isolates unresolvable annotations
        # per-parameter: one bad forward-ref no longer discards the OTHER
        # params' resolved types (the pre-audit all-or-nothing bug, 7ymj).
        extra_ns = _build_annotation_namespace(f, caller_ns=caller_ns)
        resolved_hints = resolve_hints(f, localns=extra_ns, owner=node_label)

        # Output inference: explicit kwarg wins; fall back to return annotation.
        #
        # Mismatch check: when outputs= is explicit AND
        # a return annotation exists, they must agree. Dict-form outputs=
        # is exempt (multi-output can't be expressed as an annotation).
        ret_hint = resolved_hints.get("return")

        inferred_output = outputs
        if inferred_output is None:
            # No explicit outputs= — infer from annotation
            if ret_hint is type(None):
                raise ConstructError.build(
                    "return annotation is None",
                    node=node_label,
                    hint="every node must produce output -- annotate with a concrete type or pass outputs=",
                )
            if ret_hint is not None:
                inferred_output = ret_hint
        elif (
            not isinstance(outputs, dict)
            and ret_hint is not None
            and ret_hint is not type(None)
            and not isinstance(ret_hint, str)  # unresolved string annotation — can't compare
        ):
            # Explicit outputs= AND return annotation — must match.
            # Dict-form outputs= is exempt (multi-output, annotation is partial).
            if outputs is not ret_hint:
                out_name = type_display_name(outputs)
                ret_name = type_display_name(ret_hint)
                raise ConstructError.build(
                    "outputs= differs from return annotation",
                    node=node_label,
                    expected=out_name,
                    found=ret_name,
                    hint="use one or the other -- having both with different types is a bug",
                )

        # Inputs inference: explicit kwarg wins. Otherwise build a dict-form
        # `inputs = {param_name: annotation}` from every typed upstream
        # parameter (neograph-kqd.4). DI params (FromInput/FromConfig/constant)
        # are excluded because they come from config, not state. Fan-out
        # params (Each receivers) are also excluded later at
        # _build_construct_from_decorated time — we can't identify them yet
        # without the full module context.
        #
        # Reuses ``resolved_hints`` (computed once above via resolve_hints) so
        # the dict carries real types, not ForwardRef strings.
        inferred_inputs: Any
        if inputs is not None:
            inferred_inputs = inputs
        elif effective_mode == "raw":
            inferred_inputs = None
        else:
            inputs_dict: dict[str, Any] = {}
            for p in sig.parameters.values():
                if p.name in param_res:
                    continue  # skip from_input / from_config / constant params
                if p.annotation is inspect.Parameter.empty:
                    continue  # unannotated — can't type-check
                # Prefer resolved hint (handles from __future__ annotations).
                hint = resolved_hints.get(p.name, p.annotation)
                inputs_dict[p.name] = hint
            inferred_inputs = inputs_dict if inputs_dict else None

        n = Node(
            name=node_label,
            mode="scripted" if effective_mode == "raw" else effective_mode,
            inputs=inferred_inputs,
            outputs=inferred_output,
            model=model,
            prompt=prompt,
            llm_config=(llm_config if isinstance(llm_config, LlmConfig) else LlmConfig(**(llm_config or {}))),
            tools=tools or [],
            raw_fn=f if effective_mode == "raw" else None,
            renderer=renderer,
            context=context,
            skip_when=skip_when,
            skip_value=skip_value,
            gate_tools_when=gate_tools_when,
        )

        # -- Oracle kwargs detection (needed for combined check) ----------------
        has_oracle_kwarg = (
            ensemble_n is not None or models is not None or merge_fn is not None or merge_prompt is not None
        )

        # -- Each×Oracle fusion: map_over + ensemble -----------
        if map_over is not None and has_oracle_kwarg:
            oracle_kw = _build_oracle_kwargs(
                node_label=node_label,
                f=f,
                merge_fn=merge_fn,
                merge_prompt=merge_prompt,
                models=models,
                ensemble_n=ensemble_n,
                merge_pre_process=merge_pre_process,
                merge_post_process=merge_post_process,
                merge_fallback=merge_fallback,
                merge_model=merge_model,
            )
            n = n | Oracle(**oracle_kw)
            n = n | Each(**_build_each_kwargs(map_over, map_key, map_on_error))
            # Rebind, never return (neograph-s7zt3.10). Both map_over branches used
            # to early-return here, BEFORE the interrupt_when tail below, so
            # ``@node(map_over=..., interrupt_when=...)`` silently classified as
            # EACH / EACH_ORACLE with the Operator dropped and no error — the exact
            # silent-seam class the North Star forbids. Operator is the ONLY
            # modifier the early return dropped: map_over+loop_when and
            # portal+map_over already raise ConstructError above. A DISCARDED
            # rebind is the same silent-drop shape, hence ``n =``, not a bare call.
            n = _apply_eager_oracle_gen_type(n)

        # -- Fan-out via Each when map_over is set (no Oracle) -----------------
        elif map_over is not None:
            # Apply | Each(...) — this creates a new Node via model_copy. The
            # shared re-registration below re-stamps the sidecar on the new
            # instance so construct_from_module can find it.
            n = n | Each(**_build_each_kwargs(map_over, map_key, map_on_error))

        _register_sidecar(n, f, param_names)
        if param_res:
            _set_param_res(n, param_res)

        # -- Oracle ensemble when any ensemble kwarg is set (no Each) ----------
        # Gated on ``map_over is None``: the fused Each×Oracle branch above already
        # piped the Oracle, and re-piping here would double-apply it.
        if has_oracle_kwarg and map_over is None:
            oracle_kw = _build_oracle_kwargs(
                node_label=node_label,
                f=f,
                merge_fn=merge_fn,
                merge_prompt=merge_prompt,
                models=models,
                ensemble_n=ensemble_n,
                merge_pre_process=merge_pre_process,
                merge_post_process=merge_post_process,
                merge_fallback=merge_fallback,
                merge_model=merge_model,
            )
            n = n | Oracle(**oracle_kw)
            _register_sidecar(n, f, param_names)
            if param_res:
                _set_param_res(n, param_res)
            # Rebind: the eager write produces a copy; the fall-through below
            # (Operator/Loop/context handling) must use the rebound node.
            n = _apply_eager_oracle_gen_type(n)

        # -- Operator interrupt when interrupt_when is set --------------------
        if interrupt_when is not None:
            if isinstance(interrupt_when, str):
                condition_name = interrupt_when
            elif callable(interrupt_when):
                condition_name = f"_node_interrupt_{node_label}_{secrets.token_hex(8)}"
                register_condition(condition_name, interrupt_when)
            else:
                raise ConstructError.build(
                    "interrupt_when must be a string (registered condition name) or a callable",
                    node=node_label,
                    found=type(interrupt_when).__name__,
                )

            n = n | Operator(when=condition_name)
            _register_sidecar(n, f, param_names)
            if param_res:
                _set_param_res(n, param_res)

        # -- Loop when loop_when is set -----------------------------------------
        # ``on_exhaust`` is shared with the Portal sugar (below); route it by
        # trigger. On the Loop path Portal's 'exit' is the cross-modifier
        # value — reject it with a clean ConstructError. Any other invalid value
        # falls through to Loop.model_post_init's own ConfigurationError.
        if loop_when is not None:
            if on_exhaust == "exit":
                raise ConstructError.build(
                    "loop_when= (Loop) does not accept on_exhaust='exit'",
                    node=_km_node,
                    found=repr(on_exhaust),
                    hint="'exit' is a Portal (portal=) value; use 'last' to return the last iteration",
                )
            loop_kwargs: dict[str, Any] = {
                "when": loop_when,
                "max_iterations": max_iterations or 10,
            }
            if on_exhaust is not None:
                loop_kwargs["on_exhaust"] = on_exhaust
            n = n | Loop(**loop_kwargs)
            _register_sidecar(n, f, param_names)
            if param_res:
                _set_param_res(n, param_res)

        # -- Portal (peer-mode dynamic handoff) when portal= is set ------------
        # Presence of ``portal=`` builds the Portal, mirroring ``loop_when=`` ->
        # Loop. This is PURE SUGAR: it only builds + pipes the modifier. The
        # normalizer (_ir_normalize) infers handoff_param + handoff_channel on
        # Construct assembly, IDENTICALLY for all three surfaces — decorators.py
        # writes NEITHER IR field (the neograph-ts7 / fan_out_param single-writer
        # rule, guard G3). The reserved 'handoff' param flows through the normal
        # signature-inferred inputs dict with no special-casing.
        if portal is not None:
            if on_exhaust == "last":
                raise ConstructError.build(
                    "portal= (Portal) does not accept on_exhaust='last'",
                    node=_km_node,
                    found=repr(on_exhaust),
                    hint="'last' is a Loop (loop_when=) value; use 'exit' to leave the mesh on budget exhaustion",
                )
            n = n | Portal(**_build_portal_kwargs(portal, route, max_hops, on_exhaust))
            _register_sidecar(n, f, param_names)
            if param_res:
                _set_param_res(n, param_res)

        # Eager scripted-shim registration (do0d9): a bare @node placed DIRECTLY
        # into a declarative ``Construct(nodes=[...])`` — e.g. a Portal mesh
        # member alongside a sub-construct member — never passes through
        # ``_build_construct_from_decorated`` (the construct_from_* path), so its
        # shim would otherwise stay unregistered (``scripted_fn=None``) and
        # ``compile()`` fails with "Scripted function 'None' not registered".
        # Registering here at the @node layer (AGENTS.md: fix @node gaps in
        # decorators.py, NOT the IR) closes that gap. The inputs are already
        # DI-stripped at decoration, so an empty port/fan-out map is correct for
        # the declarative case; ``construct_from_functions`` RE-registers with
        # full port/fan-out context (unconditional in _cleanup_inputs_and_register)
        # and overwrites this default, so the topo-sorted surface is unchanged.
        if n.mode == "scripted" and n.raw_fn is None and n.scripted_fn is None and _get_sidecar(n) is not None:
            synthetic = _register_node_scripted(n)
            if synthetic is not None:
                n = n.model_copy(update={"scripted_fn": synthetic})

        return n

    # Support both @node and @node(...) forms (see tool.py:130-132).
    if fn is not None:
        return decorator(fn)
    return decorator


# ──────────────────────────── @merge_fn ───────────────────────
#
# Registry and inference functions live in _sidecar.py.
# _merge_fn_registry, _merge_fn_caller_ns, get_merge_fn_metadata,
# and infer_oracle_gen_type are re-exported via the import block above.


# Construct-building functions live in _construct_builder.py and its sibling
# helper modules per neograph-3zai. Re-exported here for backward compatibility
# and so __init__.py's existing imports + test imports continue to work.
from neograph._construct_builder import (  # noqa: E402, F401
    _build_construct_from_decorated,
    construct_from_functions,
    construct_from_module,
)
from neograph._construct_graph import (  # noqa: E402, F401
    _resolve_dict_output_param,
    _resolve_loop_self_param,
)
from neograph._scripted_registry import _register_node_scripted  # noqa: E402, F401

# Re-export of the @merge_fn decorator (neograph-3ffdg.11). Bound HERE, after
# node(), because that is where the def used to live: node() takes a keyword
# parameter also called `merge_fn` (the string name of a registered merge
# function), and binding the module-level name before node() makes ruff read that
# parameter as a redefinition. Same public surface, original ordering.
merge_fn = _merge_fn_impl
