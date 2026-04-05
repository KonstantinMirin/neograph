"""@node decorator and construct_from_module — Dagster-style pipeline definition.

Ergonomic front-end on top of Node + Construct: the function signature IS the
dependency graph.

    @node(mode="scripted", output=Claims)
    def decompose(topic: RawText) -> Claims: ...

    @node(mode="scripted", output=Classified)
    def classify(decompose: Claims) -> Classified: ...
    # parameter name 'decompose' matches upstream node 'decompose' → auto-wires

    pipeline = construct_from_module(sys.modules[__name__])

Design notes

* Mirrors `src/neograph/tool.py:89-133` — same two-form call shape (`@node`
  vs `@node(...)`), same function-local factory import to dodge the
  `factory → node → decorators` circular path, same "return a spec instance
  rather than the wrapped function" contract.

* The decorator stashes the original function and its parameter-name tuple in
  a module-level dict keyed by `id(Node)`, with a `weakref.finalize` callback
  that evicts the entry when the Node is garbage-collected. `Node` itself is
  a pydantic BaseModel and is not mutated to add sidecar fields — the beads
  brief explicitly forbids editing `node.py`.

* Scripted @node functions are dispatched via the existing `raw_fn` field on
  `Node` (which the `factory._make_raw_wrapper` branch already supports for
  `raw_node`). Using the raw-fn path — rather than `register_scripted` — lets
  us pass the full state to a closure that reads N upstream values by name,
  without editing `factory.py` (which is out of scope per the brief). Non-
  scripted modes (produce / gather / execute) never see the raw_fn path and
  keep their existing LLM dispatch; their parameter annotations only drive
  topology + type inference.

* `construct_from_module` walks `vars(mod)` once, keeps only Node instances
  that appear in the sidecar (so plain `Node(...)` at module scope is
  ignored), builds adjacency from each node's parameter-name tuple, DFS
  topological-sorts with a visiting set for cycle detection, and hands the
  sorted list to `Construct(name=..., nodes=...)`. No new validation path:
  the existing `_validate_node_chain` runs via `Construct.__init__`.

* Name convention: function `foo_bar` → node name `'foo-bar'`; a downstream
  parameter `foo_bar: T` looks up the node via `name.replace("-", "_")`.
  Matches the state-field convention everywhere else in the codebase.

* v1 scope: every parameter must name an upstream `@node` in the module.
  Scalars and run-input kwargs are out of scope (they raise `ConstructError`).
  `*args` / `**kwargs` are rejected at decoration time.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
import textwrap
import warnings
import weakref
from typing import Annotated, Any, Callable, Literal, get_args, get_origin


class FromInput:
    """Dependency-injection marker: parameter value comes from ``run(input=...)``.

    Use as a marker inside ``typing.Annotated``. The primary annotation is
    the real type of the parameter; ``FromInput`` tells neograph where the
    value comes from at runtime::

        from typing import Annotated
        from neograph import node, FromInput

        @node(output=Result)
        def my_node(topic: Annotated[str, FromInput]) -> Result: ...

    ``topic`` is resolved from ``config["configurable"]["topic"]`` — ``run()``
    injects every key of ``input=`` into ``configurable`` for you. If the
    key is absent, ``None`` is passed.

    Pydantic models work the same way: ``Annotated[MyModel, FromInput]``
    constructs an instance by pulling each of the model's declared fields
    from ``config["configurable"]`` under that field's name. This is how
    you bundle pipeline metadata (``node_id``, ``project_root``, ...) into
    a single typed context argument.
    """


class FromConfig:
    """Dependency-injection marker: parameter value comes from ``config['configurable']``.

    Use as a marker inside ``typing.Annotated``::

        from typing import Annotated
        from neograph import node, FromConfig

        @node(output=Result)
        def my_node(limiter: Annotated[RateLimiter, FromConfig]) -> Result: ...

    ``limiter`` is resolved from ``config["configurable"]["limiter"]`` at
    runtime. This is the standard path for shared infrastructure (rate
    limiters, trace providers, DB connections) that you pass in via
    ``run(graph, config={"configurable": {...}})``.

    Pydantic models work the same way: ``Annotated[Shared, FromConfig]``
    constructs an instance from per-field ``configurable`` entries. Use
    this when your shared resources are a typed bundle.
    """

from neograph._construct_validation import (
    ConstructError,
    _fmt_type,
    _types_compatible,
    effective_producer_type,
)
from neograph.construct import Construct
from neograph.modifiers import Each, Operator, Oracle
from neograph.node import Node
from neograph.tool import Tool


# Param resolution: how each parameter gets its value at runtime.
# 'upstream'          — read from state by param name (existing @node output)
# 'from_input'        — read from config["configurable"][param_name]  (run input kwarg)
# 'from_config'       — read from config["configurable"][param_name]  (shared resource)
# 'from_input_model'  — FromInput[PydanticModel]: construct the model by pulling
#                       each model field name from config["configurable"][field_name]
# 'from_config_model' — FromConfig[PydanticModel]: same mechanic, different semantic
# 'constant'          — use the captured default value
# Second tuple element carries the default (for 'constant') OR the Pydantic
# model class (for '*_model' kinds) OR None.
ParamResolution = dict[str, tuple[str, Any]]

# Sidecar: id(Node) -> (original_fn, param_names_tuple, fan_out_param).
# Keyed by id() so `Node` is not mutated. A `weakref.finalize` callback evicts
# entries when the Node is garbage-collected, so the dict cannot leak.
# fan_out_param is the parameter name that receives Each items (None if no fan-out).
_node_sidecar: dict[int, tuple[Callable, tuple[str, ...], str | None]] = {}

# Separate storage for param resolutions — keyed by id(Node), same lifecycle
# as _node_sidecar. Kept separate to preserve the 3-element sidecar tuple
# contract that existing code depends on.
_param_resolutions: dict[int, ParamResolution] = {}


def _register_sidecar(
    n: Node, fn: Callable, param_names: tuple[str, ...], fan_out_param: str | None = None,
) -> None:
    node_id = id(n)
    _node_sidecar[node_id] = (fn, param_names, fan_out_param)
    weakref.finalize(n, _node_sidecar.pop, node_id, None)


def _register_param_resolutions(n: Node, resolutions: ParamResolution) -> None:
    """Store param resolution metadata for a Node, separate from the sidecar."""
    node_id = id(n)
    _param_resolutions[node_id] = resolutions
    weakref.finalize(n, _param_resolutions.pop, node_id, None)


def _get_param_resolutions(n: Node) -> ParamResolution:
    """Get param resolution metadata for a Node."""
    return _param_resolutions.get(id(n), {})


def _get_sidecar(n: Node) -> tuple[Callable, tuple[str, ...], str | None] | None:
    return _node_sidecar.get(id(n))


def _classify_di_params(
    f: Callable,
    sig: inspect.Signature,
    frame_depth: int = 2,
) -> ParamResolution:
    """Classify a function's parameters by FromInput/FromConfig markers.

    The DI surface uses ``typing.Annotated`` with ``FromInput`` /
    ``FromConfig`` as markers — the FastAPI dependency-injection pattern.
    The primary annotation is the real type; the marker tells neograph
    where the value comes from at runtime::

        topic: Annotated[str, FromInput]            # scalar per-param
        ctx:   Annotated[RunCtx, FromInput]         # bundle from BaseModel fields
        limit: Annotated[RateLimiter, FromConfig]   # shared resource per-param

    Under ``from __future__ import annotations`` the annotation arrives as
    a string, so we walk the caller's frame stack at decoration time to
    capture locally-defined classes (e.g. ``class RunCtx`` inside a test
    method) that aren't in the function's globals or closure. This is the
    same technique Pydantic uses for forward-ref resolution.

    frame_depth: how many frames up from this helper to the user's call
    site. For @node's ``decorator(f)`` → ``_classify_di_params(...)``
    chain, that's 2. @merge_fn is the same.
    """
    from pydantic import BaseModel as _BaseModel

    # Build a resolution namespace: markers, function closure, caller locals.
    extra_locals: dict[str, Any] = {
        "FromInput": FromInput,
        "FromConfig": FromConfig,
        "Annotated": Annotated,
    }
    try:
        cv = inspect.getclosurevars(f)
        extra_locals.update(cv.globals)
        extra_locals.update(cv.nonlocals)
    except (TypeError, ValueError):
        pass
    try:
        caller = sys._getframe(frame_depth)  # noqa: SLF001
        hops = 0
        while caller is not None and hops < 8:
            for k, v in caller.f_locals.items():
                if not k.startswith("_") and k not in extra_locals:
                    extra_locals[k] = v
            caller = caller.f_back
            hops += 1
    except Exception:
        pass

    try:
        import typing as _typing
        # include_extras=True preserves the Annotated marker metadata.
        resolved = _typing.get_type_hints(
            f, localns=extra_locals, include_extras=True,
        )
    except Exception:
        resolved = {}

    param_res: ParamResolution = {}
    for p in sig.parameters.values():
        ann = resolved.get(p.name)
        if ann is None or ann is inspect.Parameter.empty:
            continue
        if get_origin(ann) is not Annotated:
            continue
        args = get_args(ann)
        if len(args) < 2:
            continue
        inner_type, *markers = args

        # Match the first FromInput/FromConfig marker we find. Users can
        # stack other Annotated metadata (docs, validators) alongside —
        # we only care about DI markers.
        kind_base: str | None = None
        for marker in markers:
            if marker is FromInput:
                kind_base = "from_input"
                break
            if marker is FromConfig:
                kind_base = "from_config"
                break
        if kind_base is None:
            continue

        # Pydantic BaseModel → bundled form (build instance from scattered
        # fields). Everything else → per-parameter lookup by name.
        if isinstance(inner_type, type) and issubclass(inner_type, _BaseModel):
            param_res[p.name] = (f"{kind_base}_model", inner_type)
        else:
            param_res[p.name] = (kind_base, None)

    return param_res


def _resolve_di_value(
    kind: str,
    payload: Any,
    pname: str,
    config: Any,
) -> Any:
    """Resolve a single DI-classified parameter value from a runtime config.
    Shared between @node raw_adapter and @merge_fn wrapper.

    Returns the value to pass into the user function (or None on failure).
    """
    def _get_configurable(key: str) -> Any:
        cfg = config or {}
        if isinstance(cfg, dict):
            return cfg.get("configurable", {}).get(key)
        return getattr(cfg, "configurable", {}).get(key)

    if kind in ("from_input", "from_config"):
        return _get_configurable(pname)
    if kind in ("from_input_model", "from_config_model"):
        model_cls = payload
        field_values: dict[str, Any] = {}
        for fname in model_cls.model_fields:
            val = _get_configurable(fname)
            if val is not None:
                field_values[fname] = val
        try:
            return model_cls(**field_values)
        except Exception:
            return None
    if kind == "constant":
        return payload
    return None


def _get_node_source(n: Node) -> str | None:
    """Return 'basename.py:lineno' for the @node-decorated function, or None."""
    sidecar = _get_sidecar(n)
    if sidecar is None:
        return None
    fn = sidecar[0]
    try:
        fname = os.path.basename(fn.__code__.co_filename)
        lineno = fn.__code__.co_firstlineno
        return f"{fname}:{lineno}"
    except (AttributeError, TypeError):
        return None


def node(
    fn: Callable | None = None,
    *,
    mode: Literal["produce", "gather", "execute", "scripted", "raw"] | None = None,
    input: Any = None,
    output: Any = None,
    model: str | None = None,
    prompt: str | None = None,
    llm_config: dict[str, Any] | None = None,
    tools: list[Tool] | None = None,
    name: str | None = None,
    map_over: str | None = None,
    map_key: str | None = None,
    ensemble_n: int | None = None,
    merge_fn: str | None = None,
    merge_prompt: str | None = None,
    interrupt_when: str | Callable | None = None,
) -> Any:
    """Decorator that turns a function into a Node spec with signature-inferred
    dependencies. Supports both `@node` and `@node(...)` call forms.

    Inference rules — explicit kwargs always win over annotations:
        * `name`   ← kwarg, else `fn.__name__.replace("_", "-")`
        * `output` ← kwarg, else function return annotation
        * `input`  ← kwarg, else annotation of the first annotated parameter

    Fan-out via Each::

        @node(mode='scripted', output=MatchResult,
              map_over='make_clusters.groups', map_key='label')
        def verify(cluster: ClusterGroup) -> MatchResult: ...

    When ``map_over`` is set the node is automatically composed with
    ``Each(over=map_over, key=map_key)``. The first parameter whose name does
    NOT match any upstream ``@node`` is treated as the fan-out item receiver;
    ``construct_from_module`` skips it in topology wiring.

    Oracle ensemble::

        @node(mode='produce', output=Claims, prompt='rw/decompose', model='reason',
              ensemble_n=3, merge_prompt='rw/decompose-merge')
        def decompose(topic: RawText) -> Claims: ...

    When any of ``ensemble_n``, ``merge_fn``, or ``merge_prompt`` is set the
    node is composed with ``Oracle(n=..., merge_fn=..., merge_prompt=...)``.
    Exactly one of ``merge_fn`` or ``merge_prompt`` is required; ``ensemble_n``
    defaults to 3 if omitted.

    Human-in-the-loop via Operator::

        @node(mode='scripted', output=ValidationResult,
              interrupt_when='validation_failed')
        def validate(claims: Claims) -> ValidationResult: ...

    When ``interrupt_when`` is set the node is composed with
    ``Operator(when=...)``. The value can be a string (registered condition
    name) or a callable (auto-registered under a synthesized name).

    For `mode='scripted'`, the function is executed via `Node.raw_fn` set at
    `construct_from_module` time — this keeps `factory.py` untouched and
    supports fan-in (>1 parameter) nodes uniformly.
    """

    def decorator(f: Callable) -> Node:
        # -- Validate map_over / map_key pairing early -----------------------
        if map_over is not None and map_key is None:
            raise ConstructError(
                f"@node '{(name or f.__name__).replace('_', '-')}': "
                f"map_over= requires map_key=. Pass map_key='<field>' to "
                f"specify the dispatch key on each item."
            )
        if map_key is not None and map_over is None:
            raise ConstructError(
                f"@node '{(name or f.__name__).replace('_', '-')}': "
                f"map_key= requires map_over=. Pass map_over='<dotted.path>' "
                f"to specify the collection to fan out over."
            )

        # -- Mode inference: if not explicitly set, infer from kwargs ----------
        effective_mode = mode
        if effective_mode is None:
            if prompt is not None or model is not None:
                effective_mode = "produce"
            else:
                effective_mode = "scripted"

        node_label = (name or f.__name__).replace("_", "-")

        # -- Decoration-time validation for LLM modes -------------------------
        if effective_mode in ("produce", "gather", "execute"):
            if prompt is None:
                raise ConstructError(
                    f"@node '{node_label}' uses mode='{effective_mode}' "
                    f"which requires prompt=. Pass prompt='<template>' or "
                    f"switch to mode='scripted'."
                )
            if model is None:
                raise ConstructError(
                    f"@node '{node_label}' uses mode='{effective_mode}' "
                    f"which requires model=. Pass model='<model_name>' or "
                    f"switch to mode='scripted'."
                )

            # -- Dead-body warning for LLM modes ------------------------------
            # Check if the function body is non-trivial (more than just `...`,
            # `pass`, or a bare constant/return). Uses AST inspection.
            try:
                source = textwrap.dedent(inspect.getsource(f))
                tree = ast.parse(source)
                func_def = next(
                    (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
                    None,
                )
                if func_def is not None:
                    body = func_def.body
                    trivial = False
                    if len(body) == 1:
                        stmt = body[0]
                        # `...` / `pass` / bare constant
                        if isinstance(stmt, ast.Pass):
                            trivial = True
                        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                            trivial = True
                    if not trivial:
                        warnings.warn(
                            f"@node '{node_label}': the body of mode='{effective_mode}' "
                            f"functions is not executed; the LLM call via prompt= provides "
                            f"the output. Move this logic into a scripted node, or remove "
                            f"the body and use '...' as placeholder.",
                            UserWarning,
                            stacklevel=3,
                        )
            except (OSError, TypeError):
                # Source not available (e.g. built-in, dynamic) — skip check.
                pass

        sig = inspect.signature(f)

        # -- Raw mode: enforce (state, config) signature ----------------------
        if effective_mode == "raw":
            params = list(sig.parameters.values())
            if len(params) != 2:
                raise ConstructError(
                    f"@node(mode='raw') '{f.__name__}' must have exactly two "
                    f"parameters (state, config); got {len(params)}."
                )
            if [p.name for p in params] != ["state", "config"]:
                raise ConstructError(
                    f"@node(mode='raw') '{f.__name__}' parameters must be "
                    f"named 'state' and 'config'; got {[p.name for p in params]}."
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
                    msg = (
                        f"@node decorator on '{f.__name__}': parameter '{p.name}' "
                        f"is *args/**kwargs, which has no upstream-node mapping. "
                        f"Use explicit named parameters."
                    )
                    raise ConstructError(msg)

            param_names = tuple(p.name for p in sig.parameters.values())

        # Classify non-upstream params at decoration time via the shared
        # DI classifier. Handles FromInput[T] / FromConfig[T] including the
        # bundled form FromInput[PydanticModel]. Default-value constants
        # are deferred to construct_from_module (we don't know which param
        # names map to @node upstreams until then).
        param_res: ParamResolution = {}
        if effective_mode != "raw":
            # frame_depth=2: from inside _classify_di_params, frame 0 is the
            # helper itself, frame 1 is decorator(f) here, and frame 2 is
            # the user code applying @node. We walk up from there to pick
            # up locally-defined classes in the enclosing scope.
            param_res = _classify_di_params(f, sig, frame_depth=2)

        # Output inference: explicit kwarg wins; fall back to return annotation.
        inferred_output = output
        if inferred_output is None:
            ret = sig.return_annotation
            if ret is not inspect.Signature.empty:
                inferred_output = ret

        # Input inference: explicit kwarg wins. Otherwise use the annotation
        # of the first annotated parameter that is an upstream dependency
        # (skip FromInput/FromConfig/constant params). The per-node `.input`
        # field only tracks a single type (matches the existing Node contract);
        # fan-in still drives topology via param names, only the first
        # annotated upstream param feeds `.input`.
        inferred_input = input
        if inferred_input is None:
            for p in sig.parameters.values():
                if p.name in param_res:
                    continue  # skip from_input / from_config params
                if p.annotation is not inspect.Parameter.empty:
                    inferred_input = p.annotation
                    break

        n = Node(
            name=node_label,
            mode="scripted" if effective_mode == "raw" else effective_mode,
            input=inferred_input,
            output=inferred_output,
            model=model,
            prompt=prompt,
            llm_config=llm_config or {},
            tools=tools or [],
            raw_fn=f if effective_mode == "raw" else None,
        )

        # -- Fan-out via Each when map_over is set ---------------------------
        if map_over is not None:
            # Apply | Each(...) — this creates a new Node via model_copy.
            n_mapped = n | Each(over=map_over, key=map_key)  # type: ignore[arg-type]
            # The model_copy produced a new id(); re-register the sidecar on
            # the new instance so construct_from_module can find it.
            _register_sidecar(n_mapped, f, param_names)
            if param_res:
                _register_param_resolutions(n_mapped, param_res)
            return n_mapped

        _register_sidecar(n, f, param_names)
        if param_res:
            _register_param_resolutions(n, param_res)

        # -- Oracle ensemble when any ensemble kwarg is set --------------------
        if ensemble_n is not None or merge_fn is not None or merge_prompt is not None:
            if merge_fn is None and merge_prompt is None:
                raise ConstructError(
                    f"@node '{node_label}' sets ensemble_n={ensemble_n} but "
                    f"neither merge_fn nor merge_prompt. One is required."
                )
            if merge_fn is not None and merge_prompt is not None:
                raise ConstructError(
                    f"@node '{node_label}' sets both merge_fn and merge_prompt. "
                    f"Choose exactly one."
                )
            n_copies = ensemble_n if ensemble_n is not None else 3
            if n_copies < 2:
                raise ConstructError(
                    f"@node '{node_label}' ensemble_n must be >= 2, got {n_copies}."
                )
            n = n | Oracle(n=n_copies, merge_fn=merge_fn, merge_prompt=merge_prompt)
            _register_sidecar(n, f, param_names)
            if param_res:
                _register_param_resolutions(n, param_res)

        # -- Operator interrupt when interrupt_when is set --------------------
        if interrupt_when is not None:
            from neograph.factory import register_condition

            if isinstance(interrupt_when, str):
                condition_name = interrupt_when
            elif callable(interrupt_when):
                condition_name = f"_node_interrupt_{node_label}_{id(f):x}"
                register_condition(condition_name, interrupt_when)
            else:
                raise ConstructError(
                    f"@node '{node_label}' interrupt_when must be a string "
                    f"(registered condition name) or a callable; got "
                    f"{type(interrupt_when).__name__}"
                )

            n = n | Operator(when=condition_name)
            _register_sidecar(n, f, param_names)
            if param_res:
                _register_param_resolutions(n, param_res)

        return n

    # Support both @node and @node(...) forms (see tool.py:130-132).
    if fn is not None:
        return decorator(fn)
    return decorator


# ──────────────────────────── @merge_fn (neograph-9zj) ───────────────────────
#
# Registry keyed by the function name that Oracle.merge_fn references. Maps
# to (original_fn, param_resolutions). factory.make_oracle_merge_fn consults
# this dict: if a lookup_scripted result is also in _merge_fn_registry, the
# factory uses the DI-aware call path (variants + resolved DI params)
# instead of the plain (variants, config) legacy signature.
_merge_fn_registry: dict[str, tuple[Callable, ParamResolution]] = {}


def get_merge_fn_metadata(name: str) -> tuple[Callable, ParamResolution] | None:
    """Public lookup used by neograph.factory to detect @merge_fn-decorated
    merge functions and resolve their DI parameters at runtime."""
    return _merge_fn_registry.get(name)


def merge_fn(
    fn: Callable | None = None,
    *,
    name: str | None = None,
) -> Any:
    """Decorator for Oracle merge functions with FromInput/FromConfig DI.

    Usage::

        @merge_fn
        def combine(
            variants: list[Claims],
            shared: FromConfig[SharedResources],
            node_id: FromInput[str],
        ) -> Claims:
            ...

        node | Oracle(n=3, merge_fn="combine")

    The decorated function is auto-registered via ``register_scripted`` so
    existing ``Oracle(merge_fn="combine")`` lookups still work. At runtime,
    ``neograph.factory.make_oracle_merge_fn`` detects the decorator's
    metadata and calls the function with resolved DI parameters. Functions
    without this decorator (plain ``(variants, config) -> X`` signatures)
    continue to work unchanged.

    The first parameter of a merge function always receives the list of
    variants produced by the Oracle generators; every subsequent parameter
    must be annotated with ``FromInput[T]`` or ``FromConfig[T]``. Positional
    defaults are not supported.
    """

    def decorator(f: Callable) -> Callable:
        sig = inspect.signature(f)
        params = list(sig.parameters.values())
        if not params:
            msg = (
                f"@merge_fn '{f.__name__}' must accept at least one parameter "
                f"(the variants list)."
            )
            raise ConstructError(msg)

        # Skip the first parameter (variants); classify the rest for DI.
        rest_params = params[1:]
        rest_sig = sig.replace(parameters=rest_params)
        # frame_depth=2: inside decorator(f), the caller is the user's @merge_fn
        # site. Same rationale as @node.
        param_res = _classify_di_params(f, rest_sig, frame_depth=2)

        fn_name = name or f.__name__
        _merge_fn_registry[fn_name] = (f, param_res)

        # Auto-register via register_scripted so Oracle's existing string
        # lookup path finds the function. The factory wrapper we return
        # here is a legacy-compatible (variants, config) shim that falls
        # back to calling the user function with positional args if the
        # factory hasn't hooked into the DI path. In practice the factory
        # always checks _merge_fn_registry first (see
        # factory.make_oracle_merge_fn) so this shim is rarely invoked.
        from neograph.factory import register_scripted

        def legacy_shim(variants: Any, config: Any) -> Any:
            resolved_args = [variants]
            for pname, (kind, payload) in param_res.items():
                resolved_args.append(
                    _resolve_di_value(kind, payload, pname, config)
                )
            return f(*resolved_args)

        legacy_shim.__name__ = fn_name
        register_scripted(fn_name, legacy_shim)

        return f

    if fn is not None:
        return decorator(fn)
    return decorator


def construct_from_module(
    mod: Any,
    name: str | None = None,
    *,
    llm_config: dict[str, Any] | None = None,
) -> Construct:
    """Walk a module's @node-built Nodes, sort topologically, return a Construct.

    Walks `vars(mod)`, keeping only Node instances that appear in the @node
    sidecar (plain `Node(...)` instances at module scope are ignored). Builds
    adjacency from each node's parameter-name tuple: a parameter named `foo_bar`
    is treated as a dependency on the node whose state-field form is `foo_bar`
    (i.e. the node named `'foo-bar'`). Unknown parameter names raise
    `ConstructError`; cycles raise `ConstructError`.

    The returned Construct is a regular Construct — compile/run operate on it
    unchanged. The existing `_validate_node_chain` walker runs via
    `Construct.__init__`, so type-compatibility is enforced as usual.

    Args:
        mod: The module to walk.
        name: Construct name. Default: module's short name.
        llm_config: Default LLM config inherited by every node. Per-node
            llm_config merges over this (node wins on conflicts).
    """
    decorated: dict[str, Node] = {}
    source_label = f"module '{mod.__name__}'"
    for attr in vars(mod).values():
        if isinstance(attr, Node) and _get_sidecar(attr) is not None:
            field_name = attr.name.replace("-", "_")
            if field_name in decorated:
                existing = decorated[field_name]
                msg = (
                    f"@node name collision: two nodes resolve to field name "
                    f"'{field_name}' in {source_label}. "
                    f"One is '{existing.name}', another is '{attr.name}'. "
                    f"Fix: pass explicit name= to @node on one of them."
                )
                raise ConstructError(msg)
            decorated[field_name] = attr

    construct_name = name or mod.__name__.split(".")[-1]
    return _build_construct_from_decorated(
        decorated, construct_name, source_label, llm_config
    )


def construct_from_functions(
    name: str,
    functions: list[Any],
    *,
    llm_config: dict[str, Any] | None = None,
) -> Construct:
    """Build a Construct from an explicit list of @node-decorated functions.

    Use this when multiple pipelines share a file — `construct_from_module()`
    walks the whole module and cannot partition @nodes into separate
    Constructs. Pass the subset explicitly:

        pipelineA = construct_from_functions("A", [fn1, fn2, fn3])
        pipelineB = construct_from_functions("B", [fn4, fn5])

    Same topological sort, validation, and error messages as
    `construct_from_module()`. The returned Construct is a regular Construct.

    Args:
        name: Construct name.
        functions: List of @node-decorated functions (in any order —
            topological sort handles ordering). Each element must be a Node
            instance returned by @node; plain callables raise ConstructError.
        llm_config: Default LLM config inherited by every node. Per-node
            llm_config merges over this (node wins on conflicts).
    """
    decorated: dict[str, Node] = {}
    source_label = f"construct '{name}'"
    for item in functions:
        if not isinstance(item, Node) or _get_sidecar(item) is None:
            got = type(item).__name__
            msg = (
                f"construct_from_functions('{name}'): argument is not "
                f"decorated with @node (got {got}). Every list element must "
                f"be a function decorated with @node."
            )
            raise ConstructError(msg)
        field_name = item.name.replace("-", "_")
        if field_name in decorated:
            existing = decorated[field_name]
            msg = (
                f"@node name collision: two nodes resolve to field name "
                f"'{field_name}' in {source_label}. "
                f"One is '{existing.name}', another is '{item.name}'. "
                f"Fix: pass explicit name= to @node on one of them."
            )
            raise ConstructError(msg)
        decorated[field_name] = item

    return _build_construct_from_decorated(
        decorated, name, source_label, llm_config
    )


def _build_construct_from_decorated(
    decorated: dict[str, Node],
    construct_name: str,
    source_label: str,
    llm_config: dict[str, Any] | None,
) -> Construct:
    """Core pipeline builder shared by construct_from_module and
    construct_from_functions. Takes the already-discovered {field_name: Node}
    dict and runs adjacency + topo sort + validation + raw_fn attach.
    """
    if not decorated:
        return Construct(name=construct_name, nodes=[], llm_config=llm_config or {})

    # Build adjacency: for each node, which other nodes does it depend on?
    # Identify fan-out parameters: for nodes with Each modifier, params that
    # don't match any @node name are Each item receivers (fan-out params) and
    # must be skipped in adjacency wiring.
    fan_out_params: dict[str, set[str]] = {}
    for field_name, n in decorated.items():
        if n.has_modifier(Each):
            sidecar = _get_sidecar(n)
            assert sidecar is not None
            _, pnames, _ = sidecar
            fan_out_params[field_name] = {p for p in pnames if p not in decorated}

    # Classify default-value constants: params with defaults that don't match
    # any decorated @node and aren't already classified as from_input/from_config.
    for field_name, n in decorated.items():
        sidecar = _get_sidecar(n)
        assert sidecar is not None
        fn, pnames, _ = sidecar
        param_res = _get_param_resolutions(n)
        sig = inspect.signature(fn)
        updated = False
        for pname in pnames:
            if pname in param_res:
                continue
            if pname in fan_out_params.get(field_name, set()):
                continue
            if pname not in decorated:
                p = sig.parameters.get(pname)
                if p is not None and p.default is not inspect.Parameter.empty:
                    param_res[pname] = ("constant", p.default)
                    updated = True
        if updated:
            _register_param_resolutions(n, param_res)

    adjacency: dict[str, list[str]] = {k: [] for k in decorated}
    for field_name, n in decorated.items():
        sidecar = _get_sidecar(n)
        assert sidecar is not None
        _, param_names, _ = sidecar
        param_res = _get_param_resolutions(n)
        skip = fan_out_params.get(field_name, set())
        seen_deps: set[str] = set()
        for pname in param_names:
            if pname in skip:
                continue
            if pname in param_res:
                continue
            if pname not in decorated:
                src = _get_node_source(n)
                src_suffix = f"\n  @node defined at {src}" if src else ""
                msg = (
                    f"@node '{n.name}' parameter '{pname}' does not match any "
                    f"@node in {source_label}. All parameters must "
                    f"name an upstream @node, use FromInput/FromConfig annotation, "
                    f"or have a default value.\n"
                    f"  available @nodes: {sorted(decorated.keys())}"
                    f"{src_suffix}"
                )
                raise ConstructError(msg)
            if pname == field_name:
                src = _get_node_source(n)
                src_suffix = f"\n  @node defined at {src}" if src else ""
                msg = (
                    f"@node '{n.name}' has a parameter '{pname}' that refers "
                    f"to itself — self-dependency is not allowed."
                    f"{src_suffix}"
                )
                raise ConstructError(msg)
            if pname not in seen_deps:
                adjacency[field_name].append(pname)
                seen_deps.add(pname)

    # Topological sort via DFS with a visiting set for cycle detection.
    ordered: list[Node] = []
    marks: dict[str, str] = {}

    def visit(field: str) -> None:
        state = marks.get(field)
        if state == "black":
            return
        if state == "gray":
            src = _get_node_source(decorated[field])
            src_suffix = f"\n  @node defined at {src}" if src else ""
            msg = (
                f"@node cycle detected in {source_label} involving '{field}'. "
                f"Cyclical parameter-name dependencies are not allowed."
                f"{src_suffix}"
            )
            raise ConstructError(msg)
        marks[field] = "gray"
        for dep in adjacency[field]:
            visit(dep)
        marks[field] = "black"
        ordered.append(decorated[field])

    for field in decorated:
        visit(field)

    # Validate fan-in parameter types.
    _validate_fan_in_types(decorated)

    # Install raw_fn adapters for scripted @node functions.
    for n in ordered:
        if n.mode == "scripted" and n.raw_fn is None:
            field = n.name.replace("-", "_")
            _attach_scripted_raw_fn(n, fan_out_params.get(field, set()))

    return Construct(name=construct_name, nodes=ordered, llm_config=llm_config or {})


def _attach_scripted_raw_fn(n: Node, fan_out: set[str] | None = None) -> None:
    """Wrap the original user function in a raw-node adapter that reads
    upstream values from LangGraph state by parameter name.

    The adapter reads each `param_name` from state (dict or BaseModel form),
    calls the user function positionally, and returns a state update dict
    keyed by this node's state-field name. If the user function produces
    `None`, nothing is written — matching `factory._make_scripted_wrapper`
    semantics.

    For fan-out params (``fan_out`` set), the value is read from
    ``neo_each_item`` in state instead of looking up by parameter name.
    """
    sidecar = _get_sidecar(n)
    if sidecar is None:
        return
    fn, param_names, _ = sidecar
    param_res = _get_param_resolutions(n)
    field_name = n.name.replace("-", "_")
    _fan_out = fan_out or set()
    each_mod = n.get_modifier(Each)

    def raw_adapter(state: Any, config: Any) -> dict:
        def _get(key: str) -> Any:
            if isinstance(state, dict):
                return state.get(key)
            return getattr(state, key, None)

        def _get_configurable(key: str) -> Any:
            cfg = config or {}
            if isinstance(cfg, dict):
                return cfg.get("configurable", {}).get(key)
            # RunnableConfig object
            return getattr(cfg, "configurable", {}).get(key)

        args = []
        for pname in param_names:
            resolution = param_res.get(pname)
            if resolution is not None:
                kind, payload = resolution
                if kind in ("from_input", "from_config"):
                    # Read from config["configurable"] (run() injects all
                    # input fields there via _inject_input_to_config).
                    args.append(_get_configurable(pname))
                elif kind in ("from_input_model", "from_config_model"):
                    # Bundle resolution: build the Pydantic model by pulling
                    # each of its declared fields from config["configurable"].
                    # Missing fields fall through to the model's own default
                    # or None — Pydantic handles the rest.
                    model_cls = payload
                    field_values: dict[str, Any] = {}
                    for fname in model_cls.model_fields:
                        val = _get_configurable(fname)
                        if val is not None:
                            field_values[fname] = val
                    try:
                        args.append(model_cls(**field_values))
                    except Exception:
                        # Validation failure (required field missing, type
                        # coercion error) — pass None so the user code can
                        # handle it rather than crashing the whole run.
                        args.append(None)
                elif kind == "constant":
                    args.append(payload)
                else:
                    args.append(_get(pname))
            elif pname in _fan_out:
                # Fan-out param: value comes from neo_each_item (set by Send)
                args.append(_get("neo_each_item"))
            else:
                # Upstream @node: read from state
                args.append(_get(pname))
        result = fn(*args)

        if result is None:
            return {}

        # Each fan-out: wrap result in dict keyed by item's key field
        if each_mod and _fan_out:
            each_item = _get("neo_each_item")
            if each_item is not None:
                key_val = getattr(each_item, each_mod.key, str(each_item))
                return {field_name: {key_val: result}}

        return {field_name: result}

    raw_adapter.__name__ = field_name

    # Node is a pydantic v2 BaseModel without `frozen=True`, so direct
    # attribute assignment is supported. `raw_fn` is a declared field on
    # Node (used by the existing @raw_node decorator) so assigning it does
    # not mutate the schema — only the field value.
    n.raw_fn = raw_adapter


def _validate_fan_in_types(decorated: dict[str, Node]) -> None:
    """Check every fan-in parameter's type annotation against the upstream
    node's declared output. Raises ConstructError on the first mismatch.

    Unannotated parameters are silently skipped. ForwardRef / string
    annotations are resolved via ``typing.get_type_hints``; resolution
    failures are skipped (same pattern as ``_resolve_field_annotation``).
    """
    import inspect as _inspect
    from typing import ForwardRef as _ForwardRef, get_type_hints as _get_type_hints

    # Build a localns dict from output/input types of all decorated nodes.
    # This lets get_type_hints resolve string annotations produced by
    # `from __future__ import annotations` when those types aren't in the
    # function's __globals__ (e.g. types defined in a local scope).
    localns: dict[str, Any] = {}
    for nd in decorated.values():
        for tp in (nd.output, nd.input):
            if isinstance(tp, type) and hasattr(tp, "__name__"):
                localns.setdefault(tp.__name__, tp)

    for field_name, n in decorated.items():
        sidecar = _get_sidecar(n)
        if sidecar is None:
            continue
        fn, param_names, _ = sidecar
        param_res = _get_param_resolutions(n)

        # Resolve annotations, handling `from __future__ import annotations`
        # which turns all annotations into strings / ForwardRef objects.
        try:
            resolved_hints = _get_type_hints(fn, localns=localns)
        except Exception:
            resolved_hints = {}

        sig = _inspect.signature(fn)
        for pname in param_names:
            if pname in param_res:
                continue  # from_input / from_config / constant — not upstream
            if pname not in decorated:
                continue  # unknown-param error already raised earlier

            param = sig.parameters.get(pname)
            if param is None or param.annotation is _inspect.Parameter.empty:
                continue  # unannotated — skip

            # Prefer resolved hint (handles ForwardRef); fall back to raw annotation.
            expected = resolved_hints.get(pname, param.annotation)
            if isinstance(expected, (str, _ForwardRef)):
                continue  # unresolvable ForwardRef — skip

            upstream = decorated[pname]
            # Shared helper from _construct_validation.py — single source
            # of truth for modifier-aware producer types. Handles Each
            # dict[str, X] wrapping and any future modifier that reshapes
            # state. Do NOT inline modifier checks here.
            actual = effective_producer_type(upstream)
            if actual is None:
                continue

            if not _types_compatible(actual, expected):
                src = _get_node_source(n)
                src_suffix = f"\n  @node defined at {src}" if src else ""
                msg = (
                    f"@node '{n.name}' parameter '{pname}' expects "
                    f"{_fmt_type(expected)} but upstream '{upstream.name}' "
                    f"produces {_fmt_type(actual)}."
                    f"{src_suffix}"
                )
                raise ConstructError(msg)
