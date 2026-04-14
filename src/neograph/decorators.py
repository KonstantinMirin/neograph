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

* `construct_from_module` walks `vars(mod)` once, keeps only Node instances
  that appear in the sidecar (so plain `Node(...)` at module scope is
  ignored), builds adjacency from each node's parameter-name tuple, DFS
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
    ParamResolution,
    _build_annotation_namespace,
    _classify_di_params,
    _resolve_di_args,
    _resolve_merge_args,
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
from neograph.di import DIBinding, DIKind
from neograph.modifiers import Each, Loop, Operator, Oracle
from neograph.node import Node
from neograph.tool import Tool


def _is_trivial_body(body: list[ast.stmt]) -> bool:
    """Check if a function body (docstring already stripped) is a placeholder.

    Trivial patterns: empty (docstring-only), single `...`, `pass`,
    bare constant, `return`, or `return None`.
    """
    if not body:
        return True
    if len(body) != 1:
        return False
    stmt = body[0]
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        return True
    if isinstance(stmt, ast.Return):
        # `return` (no value) or `return None`
        if stmt.value is None:
            return True
        if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
            return True
    return False


def _build_oracle_kwargs(
    *,
    node_label: str,
    f: Callable,
    merge_fn: str | None,
    merge_prompt: str | None,
    models: list[str] | None,
    ensemble_n: int | None,
) -> dict[str, Any]:
    """Build and validate Oracle modifier kwargs from @node decorator arguments.

    Shared between Each+Oracle fusion and Oracle-only paths. Handles:
    - Body-as-merge detection + warning + shim registration
    - All validations (requires merge strategy, both set, ensemble_n >= 2)
    - Oracle kwargs dict construction
    """
    effective_merge_fn = merge_fn
    effective_merge_prompt = merge_prompt

    # Body-as-merge: models= set without merge_fn/merge_prompt
    if models is not None and merge_fn is None and merge_prompt is None:
        warnings.warn(
            f"@node '{node_label}': body used as both generator and merge function. "
            f"The first parameter receives list[OutputType] at merge time, not the "
            f"annotated upstream type. Consider adding an explicit merge_fn or merge_prompt.",
            UserWarning,
            stacklevel=4,
        )
        body_merge_name = f"_body_merge_{node_label}_{id(f):x}"
        from neograph.factory import register_scripted as _reg_scripted

        def _make_body_merge(user_fn: Callable) -> Callable:
            def body_merge(variants: list, config: Any) -> Any:
                return user_fn(variants)
            return body_merge

        _reg_scripted(body_merge_name, _make_body_merge(f))
        effective_merge_fn = body_merge_name

    if effective_merge_fn is None and effective_merge_prompt is None:
        raise ConstructError.build(
            f"ensemble_n={ensemble_n} requires merge_fn or merge_prompt",
            node=node_label,
            hint="pass merge_fn='<name>' or merge_prompt='<template>'",
        )
    if effective_merge_fn is not None and effective_merge_prompt is not None:
        raise ConstructError.build(
            "both merge_fn and merge_prompt are set",
            node=node_label,
            hint="choose exactly one",
        )
    if ensemble_n is not None and ensemble_n < 2:
        raise ConstructError.build(
            "ensemble_n must be >= 2",
            node=node_label,
            found=str(ensemble_n),
        )

    oracle_kw: dict[str, Any] = {
        "merge_fn": effective_merge_fn,
        "merge_prompt": effective_merge_prompt,
    }
    if models is not None:
        oracle_kw["models"] = models
    if ensemble_n is not None:
        oracle_kw["n"] = ensemble_n
    return oracle_kw


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

        # Output inference: explicit kwarg wins; fall back to return annotation.
        # Uses get_type_hints to resolve stringified annotations from
        # `from __future__ import annotations`.
        #
        # Mismatch check: when outputs= is explicit AND
        # a return annotation exists, they must agree. Dict-form outputs=
        # is exempt (multi-output can't be expressed as an annotation).
        try:
            import typing as _typing
            extra_ns = _build_annotation_namespace(f, caller_ns=caller_ns)
            all_hints = _typing.get_type_hints(
                f, localns=extra_ns, include_extras=False,
            )
            ret_hint = all_hints.get("return")
        except (NameError, AttributeError, TypeError):
            ret_hint = sig.return_annotation
            if ret_hint is inspect.Signature.empty:
                ret_hint = None

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
                out_name = getattr(outputs, "__name__", str(outputs))
                ret_name = getattr(ret_hint, "__name__", str(ret_hint))
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
                extra_ns = _build_annotation_namespace(f, caller_ns=caller_ns)
                resolved_hints = _typing.get_type_hints(
                    f, localns=extra_ns, include_extras=False,
                )
            except (NameError, AttributeError, TypeError):
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

        # -- Oracle kwargs detection (needed for combined check) ----------------
        has_oracle_kwarg = (
            ensemble_n is not None or models is not None
            or merge_fn is not None or merge_prompt is not None
        )

        # -- Each×Oracle fusion: map_over + ensemble -----------
        if map_over is not None and has_oracle_kwarg:
            oracle_kw = _build_oracle_kwargs(
                node_label=node_label, f=f, merge_fn=merge_fn,
                merge_prompt=merge_prompt, models=models, ensemble_n=ensemble_n,
            )
            n = n | Oracle(**oracle_kw)
            n = n | Each(over=map_over, key=map_key)  # type: ignore[arg-type]
            _register_sidecar(n, f, param_names)
            if param_res:
                _set_param_res(n, param_res)
            if merge_fn is not None:
                gen_type = infer_oracle_gen_type(merge_fn)
                if gen_type is not None and gen_type is not n.outputs:
                    n.oracle_gen_type = gen_type
            return n

        # -- Fan-out via Each when map_over is set (no Oracle) -----------------
        if map_over is not None:
            # Apply | Each(...) — this creates a new Node via model_copy.
            n_mapped = n | Each(over=map_over, key=map_key)  # type: ignore[arg-type]
            # The model_copy produced a new id(); re-register the sidecar on
            # the new instance so construct_from_module can find it.
            _register_sidecar(n_mapped, f, param_names)
            if param_res:
                _set_param_res(n_mapped, param_res)
            return n_mapped

        _register_sidecar(n, f, param_names)
        if param_res:
            _set_param_res(n, param_res)

        # -- Oracle ensemble when any ensemble kwarg is set (no Each) ----------
        if has_oracle_kwarg:
            oracle_kw = _build_oracle_kwargs(
                node_label=node_label, f=f, merge_fn=merge_fn,
                merge_prompt=merge_prompt, models=models, ensemble_n=ensemble_n,
            )
            n = n | Oracle(**oracle_kw)
            _register_sidecar(n, f, param_names)
            if param_res:
                _set_param_res(n, param_res)
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
        if loop_when is not None:
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
    # Capture the caller's local namespace once. Same rationale as @node:
    # both @merge_fn and @merge_fn(...) call merge_fn() from user code.
    caller_ns = sys._getframe(1).f_locals  # noqa: SLF001

    def decorator(f: Callable) -> Callable:
        sig = inspect.signature(f)
        params = list(sig.parameters.values())
        if not params:
            raise ConstructError.build(
                "must accept at least one parameter (the variants list)",
                node=f.__name__,
            )

        # Skip the first parameter (variants); classify the rest for DI.
        rest_params = params[1:]
        rest_sig = sig.replace(parameters=rest_params)
        param_res = _classify_di_params(f, rest_sig, caller_ns=caller_ns)

        # Auto-wire non-DI params from state by name.
        # Params without FromInput/FromConfig markers that have type
        # annotations are treated as state params — resolved from graph
        # state at merge time, matching @node's upstream wiring pattern.
        # Rebuild param_res in function signature order so positional
        # args match the function's parameter order.
        try:
            import typing as _typing
            extra_ns = _build_annotation_namespace(f, caller_ns=caller_ns)
            all_hints = _typing.get_type_hints(
                f, localns=extra_ns, include_extras=False,
            )
        except (NameError, AttributeError, TypeError):
            all_hints = {}

        ordered_res: ParamResolution = {}
        for p in rest_params:
            if p.name in param_res:
                ordered_res[p.name] = param_res[p.name]
            else:
                hint = all_hints.get(p.name, p.annotation)
                if hint is inspect.Parameter.empty:
                    continue
                if p.default is not inspect.Parameter.empty:
                    ordered_res[p.name] = DIBinding(
                        name=p.name, kind=DIKind.CONSTANT,
                        inner_type=type(p.default), required=False,
                        default_value=p.default,
                    )
                else:
                    ordered_res[p.name] = DIBinding(
                        name=p.name, kind=DIKind.FROM_STATE,
                        inner_type=hint if hint is not None else type(None),
                        required=False,
                    )
        param_res = ordered_res

        fn_name = name or f.__name__
        _merge_fn_registry[fn_name] = (f, param_res)
        _merge_fn_caller_ns[fn_name] = caller_ns

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


# Construct-building functions live in _construct_builder.py. Re-exported
# here for backward compatibility and so __init__.py's existing imports work.


# Construct-building functions live in _construct_builder.py. Re-exported
# here so __init__.py and test imports continue to work.
from neograph._construct_builder import (  # noqa: E402, F401
    _build_construct_from_decorated,
    _register_node_scripted,
    _resolve_dict_output_param,
    _resolve_loop_self_param,
    construct_from_functions,
    construct_from_module,
)
