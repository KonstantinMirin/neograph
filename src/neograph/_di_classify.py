"""DI parameter classification and resolution helpers.

Extracted from decorators.py — these functions classify function parameters
by their Annotated[T, FromInput/FromConfig] markers and resolve them at
runtime from config/state.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Annotated, Any, get_args, get_origin

from neograph._construct_validation import ConstructError
from neograph.di import DIBinding, DIKind

ParamResolution = dict[str, DIBinding]


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
    key is absent, ``ExecutionError`` is raised at runtime.

    Use ``FromInput(required=False)`` to pass ``None`` instead of raising
    when the key is absent::

        @node(outputs=Result)
        def my_node(topic: Annotated[str, FromInput(required=False)]) -> Result: ...

    Pydantic models work the same way: ``Annotated[MyModel, FromInput]``
    constructs an instance by pulling each of the model's declared fields
    from ``config["configurable"]`` under that field's name. This is how
    you bundle pipeline metadata (``node_id``, ``project_root``, ...) into
    a single typed context argument.
    """

    def __init__(self, *, required: bool = True) -> None:
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

    If the key is absent, ``ExecutionError`` is raised at runtime. Use
    ``FromConfig(required=False)`` to pass ``None`` instead::

        @node(outputs=Result)
        def my_node(key: Annotated[str, FromConfig(required=False)]) -> Result: ...

    Pydantic models work the same way: ``Annotated[Shared, FromConfig]``
    constructs an instance from per-field ``configurable`` entries. Use
    this when your shared resources are a typed bundle.
    """

    def __init__(self, *, required: bool = True) -> None:
        self.required = required


def _build_annotation_namespace(
    f: Callable,
    caller_ns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a namespace dict for resolving string annotations on *f*.

    Collects DI markers (``FromInput``, ``FromConfig``, ``Annotated``),
    the function's closure variables, and — when provided — the caller's
    local namespace. The returned dict is suitable as the ``localns``
    argument to ``typing.get_type_hints``.

    Under ``from __future__ import annotations`` all annotations arrive as
    strings, so we need a namespace that includes locally-defined classes
    (e.g. ``class RunCtx`` inside a test method) that aren't in the
    function's globals or closure.

    *caller_ns*, when not None, is merged into the namespace. The caller
    captures ``sys._getframe(1).f_locals`` once at decoration time and
    passes it explicitly — no fragile frame-depth arithmetic needed.
    When ``caller_ns`` is None only closure vars are used (suitable for
    helpers like ``infer_oracle_gen_type`` that are frame-independent).
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
    if caller_ns is not None:
        for k, v in caller_ns.items():
            if not k.startswith("_") and k not in ns:
                ns[k] = v
    return ns


def _classify_di_params(
    f: Callable,
    sig: inspect.Signature,
    caller_ns: dict[str, Any] | None = None,
) -> ParamResolution:
    """Classify a function's parameters by FromInput/FromConfig markers.

    The DI surface uses ``typing.Annotated`` with ``FromInput`` /
    ``FromConfig`` as markers --- the FastAPI dependency-injection pattern.
    The primary annotation is the real type; the marker tells neograph
    where the value comes from at runtime::

        topic: Annotated[str, FromInput]            # scalar per-param
        ctx:   Annotated[RunCtx, FromInput]         # bundle from BaseModel fields
        limit: Annotated[RateLimiter, FromConfig]   # shared resource per-param

    *caller_ns*: the caller's local namespace (captured once via
    ``sys._getframe(1).f_locals`` at decoration time). Passed through to
    ``_build_annotation_namespace`` for resolving locally-defined classes.
    """
    from pydantic import BaseModel as _BaseModel

    extra_locals = _build_annotation_namespace(f, caller_ns=caller_ns)

    try:
        import typing as _typing
        # include_extras=True preserves the Annotated marker metadata.
        resolved = _typing.get_type_hints(
            f, localns=extra_locals, include_extras=True,
        )
    except (NameError, AttributeError, TypeError):
        resolved = {}

    param_res: ParamResolution = {}
    for p in sig.parameters.values():
        ann = resolved.get(p.name)
        if ann is None or ann is inspect.Parameter.empty:
            continue
        if get_origin(ann) is not Annotated:
            continue
        args = get_args(ann)
        if len(args) < 2:  # pragma: no cover --- Annotated requires >= 2 args
            continue
        inner_type, *markers = args

        # Match the first FromInput/FromConfig marker we find. Users can
        # stack other Annotated metadata (docs, validators) alongside ---
        # we only care about DI markers. Both the bare class (FromInput)
        # and instances (FromInput(required=True)) are supported.
        # Reject ambiguous double DI markers.
        di_markers = [
            m for m in markers
            if m is FromInput or isinstance(m, FromInput)
            or m is FromConfig or isinstance(m, FromConfig)
        ]
        if len(di_markers) > 1:
            raise ConstructError.build(
                f"parameter '{p.name}' has multiple DI markers",
                found=str([type(m).__name__ for m in di_markers]),
                hint="use exactly one of FromInput or FromConfig",
            )
        kind_base: str | None = None
        required: bool = True
        for marker in markers:
            if marker is FromInput or isinstance(marker, FromInput):
                kind_base = "from_input"
                required = getattr(marker, "required", True)
                break
            if marker is FromConfig or isinstance(marker, FromConfig):
                kind_base = "from_config"
                required = getattr(marker, "required", True)
                break
        if kind_base is None:
            continue

        # Pydantic BaseModel -> bundled form (build instance from scattered
        # fields). Everything else -> per-parameter lookup by name.
        _KIND_MAP = {
            "from_input": DIKind.FROM_INPUT,
            "from_config": DIKind.FROM_CONFIG,
            "from_input_model": DIKind.FROM_INPUT_MODEL,
            "from_config_model": DIKind.FROM_CONFIG_MODEL,
        }
        if isinstance(inner_type, type) and issubclass(inner_type, _BaseModel):
            di_kind = _KIND_MAP[f"{kind_base}_model"]
            param_res[p.name] = DIBinding(
                name=p.name, kind=di_kind, inner_type=inner_type,
                required=required, model_cls=inner_type,
            )
        else:
            di_kind = _KIND_MAP[kind_base]
            param_res[p.name] = DIBinding(
                name=p.name, kind=di_kind, inner_type=inner_type,
                required=required,
            )

    return param_res


def _resolve_di_args(param_res: ParamResolution, config: Any) -> list[Any]:
    """Resolve all DI-classified parameters into a positional args list.

    Shared between the @merge_fn legacy shim and factory.make_oracle_merge_fn.
    Does NOT resolve from_state params --- use _resolve_merge_args for those.
    """
    return [
        binding.resolve(config)
        for binding in param_res.values()
        if binding.kind != DIKind.FROM_STATE
    ]


def _resolve_merge_args(
    param_res: ParamResolution,
    config: Any,
    state: Any,
) -> list[Any]:
    """Resolve all @merge_fn parameters: DI + state params.

    For from_state params, resolves from graph state.
    For DI params (from_input, from_config, etc.), resolves from config.
    All use DIBinding.resolve() directly.
    """
    return [
        binding.resolve(config, state=state)
        for binding in param_res.values()
    ]
