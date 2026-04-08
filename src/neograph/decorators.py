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

* The decorator stashes the original function and its parameter-name tuple in
  a module-level dict keyed by `id(Node)`, with a `weakref.finalize` callback
  that evicts the entry when the Node is garbage-collected. `Node` itself is
  a pydantic BaseModel and is not mutated to add sidecar fields — the beads
  brief explicitly forbids editing `node.py`.

* Scripted @node functions are dispatched via `register_scripted` — at
  `_build_construct_from_decorated` time a shim closure is synthesized for
  each scripted node and registered under a unique name. The shim reads N
  upstream values by parameter name from `input_data`, resolves DI params
  (FromInput/FromConfig/constant) from `config`, and calls the user
  function with positional args. `factory._make_scripted_wrapper` picks up
  the registered shim via `Node.scripted_fn`. Non-scripted modes (produce /
  gather / execute) keep their existing LLM dispatch; their parameter
  annotations only drive topology + type inference.

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
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, get_args, get_origin

if TYPE_CHECKING:
    from pydantic import BaseModel

import structlog


class FromInput:
    """Dependency-injection marker: parameter value comes from ``run(input=...)``.

    Use as a marker inside ``typing.Annotated``. The primary annotation is
    the real type of the parameter; ``FromInput`` tells neograph where the
    value comes from at runtime::

        from typing import Annotated
        from neograph import node, FromInput

        @node(outputs=Result)
        def my_node(topic: Annotated[str, FromInput]) -> Result: ...

    ``topic`` is resolved from ``config["configurable"]["topic"]`` — ``run()``
    injects every key of ``input=`` into ``configurable`` for you. If the
    key is absent, ``None`` is passed.

    Use ``FromInput(required=True)`` to raise ``ExecutionError`` at runtime
    when the key is absent instead of passing ``None``::

        @node(outputs=Result)
        def my_node(topic: Annotated[str, FromInput(required=True)]) -> Result: ...

    Pydantic models work the same way: ``Annotated[MyModel, FromInput]``
    constructs an instance by pulling each of the model's declared fields
    from ``config["configurable"]`` under that field's name. This is how
    you bundle pipeline metadata (``node_id``, ``project_root``, ...) into
    a single typed context argument.
    """

    def __init__(self, *, required: bool = False) -> None:
        self.required = required


class FromConfig:
    """Dependency-injection marker: parameter value comes from ``config['configurable']``.

    Use as a marker inside ``typing.Annotated``::

        from typing import Annotated
        from neograph import node, FromConfig

        @node(outputs=Result)
        def my_node(limiter: Annotated[RateLimiter, FromConfig]) -> Result: ...

    ``limiter`` is resolved from ``config["configurable"]["limiter"]`` at
    runtime. This is the standard path for shared infrastructure (rate
    limiters, trace providers, DB connections) that you pass in via
    ``run(graph, config={"configurable": {...}})``.

    Use ``FromConfig(required=True)`` to raise ``ExecutionError`` at runtime
    when the key is absent::

        @node(outputs=Result)
        def my_node(key: Annotated[str, FromConfig(required=True)]) -> Result: ...

    Pydantic models work the same way: ``Annotated[Shared, FromConfig]``
    constructs an instance from per-field ``configurable`` entries. Use
    this when your shared resources are a typed bundle.
    """

    def __init__(self, *, required: bool = False) -> None:
        self.required = required

from neograph._construct_validation import ConstructError
from neograph.construct import Construct
from neograph.modifiers import Each, Loop, Operator, Oracle
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

# Sidecar: id(Node) -> (original_fn, param_names_tuple).
# Keyed by id() so `Node` is not mutated. A `weakref.finalize` callback evicts
# entries when the Node is garbage-collected, so the dict cannot leak.
_node_sidecar: dict[int, tuple[Callable, tuple[str, ...]]] = {}

# Separate storage for param resolutions — keyed by id(Node), same lifecycle
# as _node_sidecar.
_param_resolutions: dict[int, ParamResolution] = {}


def _register_sidecar(
    n: Node, fn: Callable, param_names: tuple[str, ...],
) -> None:
    node_id = id(n)
    _node_sidecar[node_id] = (fn, param_names)
    weakref.finalize(n, _node_sidecar.pop, node_id, None)


def _register_param_resolutions(n: Node, resolutions: ParamResolution) -> None:
    """Store param resolution metadata for a Node, separate from the sidecar."""
    node_id = id(n)
    _param_resolutions[node_id] = resolutions
    weakref.finalize(n, _param_resolutions.pop, node_id, None)


def _get_param_resolutions(n: Node) -> ParamResolution:
    """Get param resolution metadata for a Node."""
    return _param_resolutions.get(id(n), {})


def _get_sidecar(n: Node) -> tuple[Callable, tuple[str, ...]] | None:
    return _node_sidecar.get(id(n))


def _build_annotation_namespace(
    f: Callable,
    frame_depth: int = 2,
) -> dict[str, Any]:
    """Build a namespace dict for resolving string annotations on *f*.

    Collects DI markers (``FromInput``, ``FromConfig``, ``Annotated``),
    the function's closure variables, and locals walked up the caller's
    frame stack (up to 8 hops). The returned dict is suitable as the
    ``localns`` argument to ``typing.get_type_hints``.

    Under ``from __future__ import annotations`` all annotations arrive as
    strings, so we need a namespace that includes locally-defined classes
    (e.g. ``class RunCtx`` inside a test method) that aren't in the
    function's globals or closure. This is the same technique Pydantic
    uses for forward-ref resolution.

    *frame_depth* counts from **this helper** to the user's call site.
    """
    ns: dict[str, Any] = {
        "FromInput": FromInput,
        "FromConfig": FromConfig,
        "Annotated": Annotated,
    }
    try:
        cv = inspect.getclosurevars(f)
        ns.update(cv.globals)
        ns.update(cv.nonlocals)
    except (TypeError, ValueError):
        pass
    try:
        caller = sys._getframe(frame_depth)  # noqa: SLF001
        hops = 0
        while caller is not None and hops < 8:
            for k, v in caller.f_locals.items():
                if not k.startswith("_") and k not in ns:
                    ns[k] = v
            caller = caller.f_back
            hops += 1
    except Exception:
        pass
    return ns


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

    frame_depth: how many frames up from this helper to the user's call
    site. For @node's ``decorator(f)`` → ``_classify_di_params(...)``
    chain, that's 2. @merge_fn is the same.
    """
    from pydantic import BaseModel as _BaseModel

    # +1 because _build_annotation_namespace is one frame deeper.
    extra_locals = _build_annotation_namespace(f, frame_depth=frame_depth + 1)

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
        # we only care about DI markers. Both the bare class (FromInput)
        # and instances (FromInput(required=True)) are supported.
        kind_base: str | None = None
        required: bool = False
        for marker in markers:
            if marker is FromInput or isinstance(marker, FromInput):
                kind_base = "from_input"
                required = getattr(marker, "required", False)
                break
            if marker is FromConfig or isinstance(marker, FromConfig):
                kind_base = "from_config"
                required = getattr(marker, "required", False)
                break
        if kind_base is None:
            continue

        # Pydantic BaseModel → bundled form (build instance from scattered
        # fields). Everything else → per-parameter lookup by name.
        # Payload carries the required flag alongside type info:
        #   scalar: payload = required (bool)
        #   model:  payload = (model_cls, required)
        if isinstance(inner_type, type) and issubclass(inner_type, _BaseModel):
            param_res[p.name] = (f"{kind_base}_model", (inner_type, required))
        else:
            param_res[p.name] = (kind_base, required)

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
    Raises ExecutionError when a required parameter is missing.
    """
    from neograph.errors import ExecutionError as _ExecutionError

    def _get_configurable(key: str) -> Any:
        cfg = config or {}
        if isinstance(cfg, dict):
            return cfg.get("configurable", {}).get(key)
        return getattr(cfg, "configurable", {}).get(key)

    if kind in ("from_input", "from_config"):
        required = bool(payload)  # payload is the required flag
        val = _get_configurable(pname)
        if val is None and required:
            source = "input" if kind == "from_input" else "config"
            raise _ExecutionError(
                f"Required DI parameter '{pname}' (from {source}) is missing "
                f"from config['configurable']. Provide it via "
                f"run(input={{'{pname}': ...}})."
            )
        return val
    if kind in ("from_input_model", "from_config_model"):
        model_cls, _required = payload  # payload is (model_cls, required)
        field_values: dict[str, Any] = {}
        for fname in model_cls.model_fields:
            val = _get_configurable(fname)
            if val is not None:
                field_values[fname] = val
        try:
            return model_cls(**field_values)
        except Exception:
            log = structlog.get_logger(__name__)
            log.warning(
                "DI model construction failed, returning None",
                model=model_cls.__name__,
                param=pname,
                fields=field_values,
            )
            return None
    if kind == "constant":
        return payload
    return None


def _resolve_di_args(param_res: ParamResolution, config: Any) -> list[Any]:
    """Resolve all DI-classified parameters into a positional args list.

    Shared between the @merge_fn legacy shim and factory.make_oracle_merge_fn.
    """
    return [
        _resolve_di_value(kind, payload, pname, config)
        for pname, (kind, payload) in param_res.items()
    ]


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
    mode: Literal["think", "agent", "act", "scripted", "raw"] | None = None,
    inputs: Any = None,
    outputs: Any = None,
    model: str | None = None,
    prompt: str | None = None,
    llm_config: dict[str, Any] | None = None,
    tools: list[Tool] | None = None,
    name: str | None = None,
    map_over: str | None = None,
    map_key: str | None = None,
    ensemble_n: int | None = None,
    models: list[str] | None = None,
    merge_fn: str | None = None,
    merge_prompt: str | None = None,
    interrupt_when: str | Callable | None = None,
    renderer: Any = None,
    context: list[str] | None = None,
    skip_when: Callable | None = None,
    skip_value: Callable | None = None,
    loop_when: str | Callable | None = None,
    max_iterations: int | None = None,
    on_exhaust: Literal["error", "last"] | None = None,
) -> Any:
    """Decorator that turns a function into a Node spec with signature-inferred
    dependencies. Supports both `@node` and `@node(...)` call forms.

    Inference rules — explicit kwargs always win over annotations:
        * `name`   ← kwarg, else `fn.__name__.replace("_", "-")`
        * `outputs` ← kwarg, else function return annotation
        * `inputs` ← kwarg, else annotation of the first annotated parameter

    Fan-out via Each::

        @node(mode='scripted', outputs=MatchResult,
              map_over='make_clusters.groups', map_key='label')
        def verify(cluster: ClusterGroup) -> MatchResult: ...

    When ``map_over`` is set the node is automatically composed with
    ``Each(over=map_over, key=map_key)``. The first parameter whose name does
    NOT match any upstream ``@node`` is treated as the fan-out item receiver;
    ``construct_from_module`` skips it in topology wiring.

    Oracle ensemble::

        @node(mode='produce', outputs=Claims, prompt='rw/decompose', model='reason',
              ensemble_n=3, merge_prompt='rw/decompose-merge')
        def decompose(topic: RawText) -> Claims: ...

    When any of ``ensemble_n``, ``merge_fn``, or ``merge_prompt`` is set the
    node is composed with ``Oracle(n=..., merge_fn=..., merge_prompt=...)``.
    Exactly one of ``merge_fn`` or ``merge_prompt`` is required; ``ensemble_n``
    defaults to 3 if omitted.

    Human-in-the-loop via Operator::

        @node(mode='scripted', outputs=ValidationResult,
              interrupt_when='validation_failed')
        def validate(claims: Claims) -> ValidationResult: ...

    When ``interrupt_when`` is set the node is composed with
    ``Operator(when=...)``. The value can be a string (registered condition
    name) or a callable (auto-registered under a synthesized name).

    For `mode='scripted'`, a shim is registered via `register_scripted` at
    `_build_construct_from_decorated` time and dispatched through
    `factory._make_scripted_wrapper`. Supports fan-in (>1 parameter) nodes
    uniformly.
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
        if map_over is not None and loop_when is not None:
            raise ConstructError(
                f"@node '{(name or f.__name__).replace('_', '-')}': "
                f"map_over= (Each) and loop_when= (Loop) cannot be combined "
                f"on the same node. Use a sub-construct with Loop inside an "
                f"Each fan-out instead."
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
        # Uses get_type_hints to resolve stringified annotations from
        # `from __future__ import annotations`.
        inferred_output = outputs
        if inferred_output is None:
            try:
                import typing as _typing
                extra_ns = _build_annotation_namespace(f, frame_depth=2)
                all_hints = _typing.get_type_hints(
                    f, localns=extra_ns, include_extras=False,
                )
                ret = all_hints.get("return")
            except Exception:
                ret = sig.return_annotation
                if ret is inspect.Signature.empty:
                    ret = None
            if ret is not None:
                inferred_output = ret

        # Inputs inference: explicit kwarg wins. Otherwise build a dict-form
        # `inputs = {param_name: annotation}` from every typed upstream
        # parameter (neograph-kqd.4). DI params (FromInput/FromConfig/constant)
        # are excluded because they come from config, not state. Fan-out
        # params (Each receivers) are also excluded later at
        # _build_construct_from_decorated time — we can't identify them yet
        # without the full module context.
        #
        # Resolve string annotations (from __future__ import annotations)
        # via typing.get_type_hints so the dict carries real types, not
        # ForwardRef strings. Include locals from the caller's frame to
        # catch class definitions inside test methods, same trick as
        # _classify_di_params.
        inferred_inputs: Any
        if inputs is not None:
            inferred_inputs = inputs
        elif effective_mode == "raw":
            inferred_inputs = None
        else:
            resolved_hints: dict[str, Any] = {}
            try:
                import typing as _typing
                # frame_depth=2: frame 0 = _build_annotation_namespace,
                # frame 1 = decorator(f) here, frame 2 = user call site.
                extra_ns = _build_annotation_namespace(f, frame_depth=2)
                resolved_hints = _typing.get_type_hints(
                    f, localns=extra_ns, include_extras=False,
                )
            except Exception:
                resolved_hints = {}

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
            llm_config=llm_config or {},
            tools=tools or [],
            raw_fn=f if effective_mode == "raw" else None,
            renderer=renderer,
            context=context,
            skip_when=skip_when,
            skip_value=skip_value,
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
        has_oracle_kwarg = (
            ensemble_n is not None or models is not None
            or merge_fn is not None or merge_prompt is not None
        )
        if has_oracle_kwarg:
            # Body-as-merge: when models= is set without merge_fn/merge_prompt,
            # the function body IS the merge function.
            effective_merge_fn = merge_fn
            effective_merge_prompt = merge_prompt
            if models is not None and merge_fn is None and merge_prompt is None:
                # Body-as-merge: the function's parameter type annotation declares
                # the upstream type (for pipeline wiring), but at runtime the body
                # receives list[OutputType] (the collected Oracle variants). This
                # is intentional — the annotation serves compile-time wiring, not
                # runtime type checking. Static analysers will flag a mismatch;
                # that is an accepted tradeoff for the DX win of one definition.
                body_merge_name = f"_body_merge_{node_label}_{id(f):x}"
                from neograph.factory import register_scripted as _reg_scripted

                def _make_body_merge(user_fn: Callable) -> Callable:
                    def body_merge(variants: list, config: Any) -> Any:
                        return user_fn(variants)
                    return body_merge

                _reg_scripted(body_merge_name, _make_body_merge(f))
                effective_merge_fn = body_merge_name

            if effective_merge_fn is None and effective_merge_prompt is None:
                raise ConstructError(
                    f"@node '{node_label}' sets ensemble_n={ensemble_n} but "
                    f"neither merge_fn nor merge_prompt. One is required."
                )
            if effective_merge_fn is not None and effective_merge_prompt is not None:
                raise ConstructError(
                    f"@node '{node_label}' sets both merge_fn and merge_prompt. "
                    f"Choose exactly one."
                )
            oracle_kwargs: dict[str, Any] = {
                "merge_fn": effective_merge_fn,
                "merge_prompt": effective_merge_prompt,
            }
            if models is not None:
                oracle_kwargs["models"] = models
            if ensemble_n is not None:
                oracle_kwargs["n"] = ensemble_n
                if ensemble_n < 2:
                    raise ConstructError(
                        f"@node '{node_label}' ensemble_n must be >= 2, got {ensemble_n}."
                    )
            n = n | Oracle(**oracle_kwargs)
            _register_sidecar(n, f, param_names)
            if param_res:
                _register_param_resolutions(n, param_res)
            # Infer per-generator output type from merge_fn signature
            if merge_fn is not None:
                gen_type = infer_oracle_gen_type(merge_fn)
                if gen_type is not None and gen_type is not n.outputs:
                    n.oracle_gen_type = gen_type

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

        # -- Loop when loop_when is set -----------------------------------------
        if loop_when is not None:
            from neograph.modifiers import Loop as _Loop

            loop_kwargs: dict[str, Any] = {
                "when": loop_when,
                "max_iterations": max_iterations or 10,
            }
            if on_exhaust is not None:
                loop_kwargs["on_exhaust"] = on_exhaust
            n = n | _Loop(**loop_kwargs)
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


def infer_oracle_gen_type(merge_fn_name: str) -> Any | None:
    """Infer the per-generator output type from a merge_fn's first parameter.

    The merge_fn's first parameter is ``list[T]`` where T is the type each
    Oracle generator should produce. Returns T, or None if inference fails.

    Uses ``typing.get_type_hints()`` to resolve string annotations from
    ``from __future__ import annotations``.

    Used by ``@node(ensemble_n=..., merge_fn=...)`` and the compiler to set
    ``Node.oracle_gen_type`` so generators use the correct LLM schema.
    """
    import typing

    # Check @merge_fn registry first (has the original function)
    meta = _merge_fn_registry.get(merge_fn_name)
    if meta is not None:
        fn, _ = meta
    else:
        # Fall back to scripted registry (plain merge functions).
        # May not be registered yet at decoration time — return None.
        try:
            from neograph.factory import lookup_scripted
            fn = lookup_scripted(merge_fn_name)
        except Exception:
            return None
        if fn is None:
            return None

    # Use get_type_hints to resolve string annotations (from __future__)
    # with the function's local namespace for locally-defined classes.
    try:
        extra_ns = _build_annotation_namespace(fn, frame_depth=0)
        hints = typing.get_type_hints(fn, localns=extra_ns, include_extras=False)
    except Exception:
        hints = {}

    if not hints:
        # Fallback: try raw signature annotations
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if not params or params[0].annotation is inspect.Parameter.empty:
            return None
        first_ann = params[0].annotation
    else:
        # First parameter's resolved hint
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if not params:
            return None
        first_ann = hints.get(params[0].name)
        if first_ann is None:
            return None

    # Extract T from list[T]
    origin = typing.get_origin(first_ann)
    if origin is list:
        args = typing.get_args(first_ann)
        if args:
            return args[0]
    return None


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
            return f(variants, *_resolve_di_args(param_res, config))

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
    input: type[BaseModel] | None = None,
    output: type[BaseModel] | None = None,
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
        input: Input type for sub-construct boundary.
        output: Output type for sub-construct boundary.
    """
    nodes: list[Node] = []
    source_label = f"module '{mod.__name__}'"
    for attr in vars(mod).values():
        if isinstance(attr, Node) and _get_sidecar(attr) is not None:
            nodes.append(attr)

    construct_name = name or mod.__name__.split(".")[-1]
    return _build_construct_from_decorated(
        nodes, construct_name, source_label, llm_config,
        construct_input=input, construct_output=output,
    )


def construct_from_functions(
    name: str,
    functions: list[Any],
    *,
    llm_config: dict[str, Any] | None = None,
    input: type[BaseModel] | None = None,
    output: type[BaseModel] | None = None,
) -> Construct:
    """Build a Construct from an explicit list of @node-decorated functions.

    Use this when multiple pipelines share a file — `construct_from_module()`
    walks the whole module and cannot partition @nodes into separate
    Constructs. Pass the subset explicitly:

        pipelineA = construct_from_functions("A", [fn1, fn2, fn3])
        pipelineB = construct_from_functions("B", [fn4, fn5])

    When building a sub-construct, pass ``input=`` / ``output=`` to define the
    state boundary:

        sub = construct_from_functions("verify", [explore, score],
                                       input=VerifyClaim, output=ClaimResult)

    Same topological sort, validation, and error messages as
    `construct_from_module()`. The returned Construct is a regular Construct.

    Args:
        name: Construct name.
        functions: List of @node-decorated functions (in any order —
            topological sort handles ordering). Each element must be a Node
            instance returned by @node; plain callables raise ConstructError.
        llm_config: Default LLM config inherited by every node. Per-node
            llm_config merges over this (node wins on conflicts).
        input: Input type for sub-construct boundary. When set, the Construct
            receives an isolated state with this type at ``neo_subgraph_input``.
        output: Output type for sub-construct boundary.
    """
    source_label = f"construct '{name}'"
    nodes: list[Node] = []
    sub_constructs: list[Construct] = []
    for item in functions:
        if isinstance(item, Construct):
            if item.output is None:
                msg = (
                    f"construct_from_functions('{name}'): Construct '{item.name}' "
                    f"has no output type. Sub-constructs must declare output= "
                    f"so downstream @nodes can resolve dependencies."
                )
                raise ConstructError(msg)
            sub_constructs.append(item)
        elif isinstance(item, Node) and _get_sidecar(item) is not None:
            nodes.append(item)
        else:
            got = type(item).__name__
            msg = (
                f"construct_from_functions('{name}'): argument is not "
                f"decorated with @node or a Construct (got {got}). "
                f"Every list element must be a function decorated with "
                f"@node or a Construct with declared output."
            )
            raise ConstructError(msg)

    return _build_construct_from_decorated(
        nodes, name, source_label, llm_config,
        construct_input=input, construct_output=output,
        sub_constructs=sub_constructs,
    )


def _resolve_dict_output_param(
    pname: str,
    decorated: dict[str, Node],
) -> str | None:
    """If pname is {upstream}_{output_key} for a dict-output upstream, return the upstream name.

    Tries longest-prefix matching against decorated node names with dict outputs.
    Returns None if no match.
    """
    for upstream_name, upstream_node in decorated.items():
        prefix = f"{upstream_name}_"
        if pname.startswith(prefix) and isinstance(upstream_node.outputs, dict):
            output_key = pname[len(prefix):]
            if output_key in upstream_node.outputs:
                return upstream_name
    return None


def _resolve_loop_self_param(
    node: Node,
    pname: str,
    decorated: dict[str, Node],
    sub_by_field: dict[str, Any],
) -> str | None:
    """For a Loop node, resolve a param by type when name doesn't match upstream.

    Returns the upstream field_name if exactly one upstream produces a compatible
    type. Returns None if no match. Raises ConstructError if ambiguous (multiple
    matches).
    """
    from neograph._construct_validation import effective_producer_type, _types_compatible

    if not isinstance(node.inputs, dict):
        return None
    param_type = node.inputs.get(pname)
    if param_type is None:
        return None

    field_name = node.name.replace("-", "_")
    candidates: list[str] = []
    all_upstreams = {**decorated, **sub_by_field}
    for up_field, upstream in all_upstreams.items():
        if up_field == field_name:
            continue  # skip self
        up_type = effective_producer_type(upstream)
        if up_type is not None and _types_compatible(up_type, param_type):
            candidates.append(up_field)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        src = _get_node_source(node)
        src_suffix = f"\n  @node defined at {src}" if src else ""
        msg = (
            f"@node '{node.name}' loop self-reference param '{pname}' "
            f"matches multiple upstreams by type: {sorted(candidates)}. "
            f"Name the param after the specific upstream to disambiguate."
            f"{src_suffix}"
        )
        raise ConstructError(msg)
    return None


def _build_construct_from_decorated(
    nodes: list[Node],
    construct_name: str,
    source_label: str,
    llm_config: dict[str, Any] | None,
    construct_input: type[BaseModel] | None = None,
    construct_output: type[BaseModel] | None = None,
    sub_constructs: list[Construct] | None = None,
) -> Construct:
    """Core pipeline builder shared by construct_from_module and
    construct_from_functions. Builds {field_name: Node} with collision
    checking, then runs adjacency + topo sort + validation + scripted shim
    registration.
    """
    _sub_constructs = sub_constructs or []
    if not nodes and not _sub_constructs:
        return Construct(name=construct_name, nodes=[], llm_config=llm_config or {})

    # Build field_name → Construct dict for sub-constructs (separate from
    # decorated @nodes — Option B from architect review). Sub-constructs
    # participate in adjacency as producers but skip all sidecar processing.
    sub_by_field: dict[str, Construct] = {}
    for sc in _sub_constructs:
        field_name = sc.name.replace("-", "_")
        sub_by_field[field_name] = sc

    # Build field_name → Node dict with collision detection.
    decorated: dict[str, Node] = {}
    for n in nodes:
        field_name = n.name.replace("-", "_")
        if field_name in decorated or field_name in sub_by_field:
            existing_name = decorated[field_name].name if field_name in decorated else sub_by_field[field_name].name
            msg = (
                f"name collision: two items resolve to field name "
                f"'{field_name}' in {source_label}. "
                f"One is '{existing_name}', another is '{n.name}'. "
                f"Fix: pass explicit name= to @node on one of them."
            )
            raise ConstructError(msg)
        decorated[field_name] = n

    # Port param identification: when construct_input is set, params whose
    # type matches it are "port params" — they read from neo_subgraph_input,
    # not from a peer @node. Identified before fan-out/constant/adjacency.
    port_params: dict[str, set[str]] = {}  # field_name → {param_names}
    if construct_input is not None:
        for field_name, n in decorated.items():
            if not isinstance(n.inputs, dict):
                continue
            ports: set[str] = set()
            for pname, ptype in n.inputs.items():
                if pname in decorated:
                    continue  # peer @node takes priority
                try:
                    if isinstance(ptype, type) and issubclass(ptype, construct_input):
                        ports.add(pname)
                except TypeError:
                    pass  # generic types fail issubclass — skip
            if len(ports) > 1:
                msg = (
                    f"@node '{n.name}' has {len(ports)} parameters matching "
                    f"construct input type {construct_input.__name__}: "
                    f"{sorted(ports)}. Ambiguous — only one port param is "
                    f"allowed per node. Rename one or use FromInput annotation."
                )
                raise ConstructError(msg)
            if ports:
                port_params[field_name] = ports

    # Build adjacency: for each node, which other nodes does it depend on?
    # Identify fan-out parameters: for nodes with Each modifier, params that
    # don't match any @node name are Each item receivers (fan-out params) and
    # must be skipped in adjacency wiring.
    fan_out_params: dict[str, set[str]] = {}
    for field_name, n in decorated.items():
        if n.has_modifier(Each):
            sidecar = _get_sidecar(n)
            if sidecar is None:
                raise ConstructError(
                    f"@node '{n.name}' lost its sidecar metadata (function + param names). "
                    f"This usually means a modifier was applied via | without re-registering "
                    f"the sidecar on the new Node copy. See AGENTS.md '@node sidecar pattern'."
                )
            _, pnames = sidecar
            di_params = set(_get_param_resolutions(n))
            _ports = port_params.get(field_name, set())
            fan_out_params[field_name] = {p for p in pnames if p not in decorated and p not in di_params and p not in _ports}

    # Classify default-value constants: params with defaults that don't match
    # any decorated @node and aren't already classified as from_input/from_config.
    for field_name, n in decorated.items():
        sidecar = _get_sidecar(n)
        if sidecar is None:
            raise ConstructError(
                f"@node '{n.name}' lost its sidecar metadata (function + param names). "
                f"This usually means a modifier was applied via | without re-registering "
                f"the sidecar on the new Node copy. See AGENTS.md '@node sidecar pattern'."
            )
        fn, pnames = sidecar
        param_res = _get_param_resolutions(n)
        sig = inspect.signature(fn)
        updated = False
        for pname in pnames:
            if pname in param_res:
                continue
            if pname in fan_out_params.get(field_name, set()):
                continue
            if pname in port_params.get(field_name, set()):
                continue
            if pname not in decorated and pname not in sub_by_field:
                p = sig.parameters.get(pname)
                if p is not None and p.default is not inspect.Parameter.empty:
                    param_res[pname] = ("constant", p.default)
                    updated = True
        if updated:
            _register_param_resolutions(n, param_res)

    # Sub-constructs participate in adjacency as producers with no deps of
    # their own (they're self-contained — internal wiring is already validated).
    # Track loop self-reference param renames: {field_name: {orig_param: resolved_upstream}}
    loop_param_renames: dict[str, dict[str, str]] = {}
    all_known = {**{k: None for k in decorated}, **{k: None for k in sub_by_field}}
    adjacency: dict[str, list[str]] = {k: [] for k in all_known}
    for field_name, n in decorated.items():
        sidecar = _get_sidecar(n)
        if sidecar is None:
            raise ConstructError(
                f"@node '{n.name}' lost its sidecar metadata (function + param names). "
                f"This usually means a modifier was applied via | without re-registering "
                f"the sidecar on the new Node copy. See AGENTS.md '@node sidecar pattern'."
            )
        _, param_names = sidecar
        param_res = _get_param_resolutions(n)
        skip = fan_out_params.get(field_name, set())
        _ports = port_params.get(field_name, set())
        seen_deps: set[str] = set()
        for pname in param_names:
            if pname in skip:
                continue
            if pname in param_res:
                continue
            if pname in _ports:
                continue  # port param — reads from neo_subgraph_input, not a peer
            if pname in sub_by_field:
                # Param names a sub-construct — wire dependency edge.
                if pname not in seen_deps:
                    adjacency[field_name].append(pname)
                    seen_deps.add(pname)
                continue
            if pname not in decorated:
                # Check if pname matches {upstream}_{output_key} for a dict-output
                # upstream (neograph-1bp.5). E.g. "analyze_summary" matches upstream
                # "analyze" with outputs={"summary": ...}.
                resolved_upstream = _resolve_dict_output_param(pname, decorated)
                if resolved_upstream is not None:
                    if resolved_upstream not in seen_deps:
                        adjacency[field_name].append(resolved_upstream)
                        seen_deps.add(resolved_upstream)
                    continue
                # Loop self-reference: when loop_when is set and param type
                # matches exactly one upstream, resolve by type (neograph-0zk2).
                if n.has_modifier(Loop):
                    loop_upstream = _resolve_loop_self_param(n, pname, decorated, sub_by_field)
                    if loop_upstream is not None:
                        if loop_upstream not in seen_deps:
                            adjacency[field_name].append(loop_upstream)
                            seen_deps.add(loop_upstream)
                        # Track the rename so inputs cleanup + shim can handle it
                        loop_param_renames.setdefault(field_name, {})[pname] = loop_upstream
                        continue
                src = _get_node_source(n)
                src_suffix = f"\n  @node defined at {src}" if src else ""
                all_names = sorted(set(decorated.keys()) | set(sub_by_field.keys()))
                msg = (
                    f"@node '{n.name}' parameter '{pname}' does not match any "
                    f"@node or sub-construct in {source_label}. All parameters must "
                    f"name an upstream @node/Construct, use FromInput/FromConfig annotation, "
                    f"or have a default value.\n"
                    f"  available items: {all_names}"
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
    # ordered contains both Node and Construct items in dependency order.
    ordered: list[Any] = []
    marks: dict[str, str] = {}

    def visit(field: str) -> None:
        state = marks.get(field)
        if state == "black":
            return
        if state == "gray":
            item = decorated.get(field) or sub_by_field.get(field)
            src = _get_node_source(item) if isinstance(item, Node) else ""
            src_suffix = f"\n  defined at {src}" if src else ""
            msg = (
                f"Cycle detected in {source_label} involving '{field}'. "
                f"Cyclical dependencies are not allowed."
                f"{src_suffix}"
            )
            raise ConstructError(msg)
        marks[field] = "gray"
        for dep in adjacency[field]:
            visit(dep)
        marks[field] = "black"
        # Append the actual item: Node or Construct
        if field in decorated:
            ordered.append(decorated[field])
        elif field in sub_by_field:
            ordered.append(sub_by_field[field])

    for field in all_known:
        visit(field)

    # Clean up n.inputs: strip DI params and constants (non-upstream) but
    # KEEP fan-out params (they need to be in the dict so
    # factory._extract_input can route them to neo_each_item via
    # node.fan_out_param). KEEP port params (rewritten to neo_subgraph_input).
    # Set fan_out_param on Each nodes.
    for field, n in decorated.items():
        if not isinstance(n.inputs, dict):
            continue
        skip = fan_out_params.get(field, set())
        _ports = port_params.get(field, set())
        # Keep keys that are upstream @nodes, fan-out receivers, port params,
        # dict-output references ({upstream}_{output_key}, neograph-1bp.5),
        # or loop self-reference params (renamed to resolved upstream).
        renames = loop_param_renames.get(field, {})
        filtered: dict[str, Any] = {}
        for k, v in n.inputs.items():
            if k in _ports:
                # Port param: rewrite key to neo_subgraph_input so
                # factory._extract_input reads from the right state field.
                filtered["neo_subgraph_input"] = v
            elif k in renames:
                # Loop self-reference: rewrite key to the resolved upstream
                # so _extract_input finds it by name on the first iteration.
                filtered[renames[k]] = v
            elif (
                (k in decorated and k != field)
                or k in sub_by_field
                or k in skip
                or _resolve_dict_output_param(k, decorated) is not None
            ):
                filtered[k] = v
        if filtered != n.inputs:
            n.inputs = filtered
        # Mark the fan-out param on the Node so factory._extract_input and
        # the validator know which key reads from neo_each_item.
        if skip:
            fan_out_name = next(iter(skip))  # typically one fan-out param
            n.fan_out_param = fan_out_name

    # Register scripted shims for @node functions via register_scripted
    # so they dispatch through factory._make_scripted_wrapper — one path
    # for all scripted nodes (neograph-kqd.8).
    for n in ordered:
        if not isinstance(n, Node):
            continue  # sub-construct — no shim needed
        if n.mode == "scripted" and n.raw_fn is None:
            field = n.name.replace("-", "_")
            _register_node_scripted(
                n, fan_out_params.get(field, set()),
                port_param_map={p: "neo_subgraph_input" for p in port_params.get(field, set())},
                loop_renames=loop_param_renames.get(field),
            )

    # Deferred oracle_gen_type inference: at decoration time, the merge_fn
    # may not be registered yet (defined below @node in the file). Retry
    # inference here — all @merge_fn decorators have run by construct assembly.
    from neograph.modifiers import Oracle as _Oracle
    for n in ordered:
        if isinstance(n, Node) and n.has_modifier(_Oracle) and n.oracle_gen_type is None:
            oracle_mod = next((m for m in n.modifiers if isinstance(m, _Oracle)), None)
            if oracle_mod is not None and oracle_mod.merge_fn is not None:
                gen_type = infer_oracle_gen_type(oracle_mod.merge_fn)
                if gen_type is not None and gen_type is not n.outputs:
                    n.oracle_gen_type = gen_type

    return Construct(
        name=construct_name,
        nodes=ordered,
        llm_config=llm_config or {},
        input=construct_input,
        output=construct_output,
    )


def _register_node_scripted(
    n: Node,
    fan_out: set[str] | None = None,
    port_param_map: dict[str, str] | None = None,
    loop_renames: dict[str, str] | None = None,
) -> None:
    """Register a scripted shim for a @node-decorated function via
    ``register_scripted`` so it dispatches through
    ``factory._make_scripted_wrapper`` — the single path for all scripted
    nodes (neograph-kqd.8).

    The shim receives ``(input_data, config)`` from the factory wrapper:
      - ``input_data`` is the dict returned by ``factory._extract_input``,
        which reads upstream values from state by key and routes the
        ``fan_out_param`` key to ``neo_each_item``.
      - ``config`` is the LangGraph ``RunnableConfig``.

    The shim resolves DI params (``FromInput``/``FromConfig``/constant)
    from ``config``, reads upstream values from ``input_data``, and calls
    the user function with positional args in parameter order.

    When ``port_param_map`` is set, port params (whose keys were rewritten
    in ``n.inputs`` to ``neo_subgraph_input``) are looked up under the
    rewritten key in ``input_data``.

    When ``loop_renames`` is set, loop self-reference params (whose keys
    were rewritten in ``n.inputs`` from original name to resolved upstream)
    are looked up under the rewritten key in ``input_data``.
    """
    from neograph.factory import register_scripted

    sidecar = _get_sidecar(n)
    if sidecar is None:
        return
    fn, param_names = sidecar
    param_res = _get_param_resolutions(n)
    _fan_out = fan_out or set()
    _port_map = port_param_map or {}
    _loop_map = loop_renames or {}

    # Synthesize a unique name for the registered shim. Use id(n) so
    # parallel pipelines with the same node names don't collide.
    synthetic_name = f"_node_{n.name}_{id(n):x}"

    def scripted_shim(input_data: Any, config: Any) -> Any:
        """Adapter: (input_data, config) → fn(*positional_args)."""
        args = []
        for pname in param_names:
            resolution = param_res.get(pname)
            if resolution is not None:
                kind, payload = resolution
                args.append(_resolve_di_value(kind, payload, pname, config))
            else:
                # Port param or loop rename: key was rewritten
                # (e.g. "claim" → "neo_subgraph_input", or
                # "draft" → "seed" for loop self-ref). Look up rewritten key.
                lookup_key = _port_map.get(pname, _loop_map.get(pname, pname))
                # Fan-out or upstream param — both are already in
                # input_data under the param name (fan-out via
                # node.fan_out_param → neo_each_item, upstream via
                # factory._extract_input).
                args.append(
                    input_data.get(lookup_key)
                    if isinstance(input_data, dict)
                    else input_data
                )
        return fn(*args)

    scripted_shim.__name__ = n.name.replace("-", "_")
    register_scripted(synthetic_name, scripted_shim)
    n.scripted_fn = synthetic_name


# _validate_fan_in_types was deleted in neograph-kqd.4. Fan-in validation
# now flows through Construct.__init__ → _validate_node_chain →
# _check_fan_in_inputs (see src/neograph/_construct_validation.py). The
# @node decoration emits dict-form inputs {param_name: annotation} at
# decoration time, which the validator walks by upstream name. One walker,
# not two.
