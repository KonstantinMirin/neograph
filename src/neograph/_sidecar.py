"""Sidecar storage for @node and @merge_fn metadata.

Extracted from decorators.py to break the circular import between
decorators.py and _construct_builder.py.

Import graph after extraction:
    decorators.py → _sidecar.py ← _construct_builder.py
    (one-way, no cycles)

Node._sidecar and Node._param_res are Pydantic PrivateAttr fields,
preserved by model_copy when modifiers are applied via |.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from typing import Any

from neograph._di_classify import _build_annotation_namespace
from neograph.di import DIBinding
from neograph.errors import ConfigurationError
from neograph.node import Node

# Type alias for DI parameter resolution dicts.
ParamResolution = dict[str, DIBinding]


# ─── Node sidecar accessors ──────────────────────────────────────────

def _register_sidecar(
    n: Node, fn: Callable, param_names: tuple[str, ...],
) -> None:
    n._sidecar = (fn, param_names)


def _set_param_res(n: Node, resolutions: ParamResolution) -> None:
    n._param_res = resolutions


def _get_param_res(n: Node) -> ParamResolution:
    return n._param_res or {}


def _get_sidecar(n: Node) -> tuple[Callable, tuple[str, ...]] | None:
    return n._sidecar


def _get_node_source(n: Node) -> str | None:
    """Return 'basename.py:lineno' for the @node-decorated function, or None."""
    sidecar = n._sidecar
    if sidecar is None:
        return None
    fn = sidecar[0]
    try:
        fname = os.path.basename(fn.__code__.co_filename)
        lineno = fn.__code__.co_firstlineno
        return f"{fname}:{lineno}"
    except (AttributeError, TypeError):
        return None


# ─── @merge_fn registry ──────────────────────────────────────────────
#
# Keyed by the function name that Oracle.merge_fn references. Maps
# to (original_fn, param_resolutions). factory.make_oracle_merge_fn
# consults this dict.

_merge_fn_registry: dict[str, tuple[Callable, ParamResolution]] = {}
_merge_fn_caller_ns: dict[str, dict[str, Any]] = {}


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
        except (ImportError, ConfigurationError):
            # May not be registered yet at decoration time.
            return None
        if fn is None:  # pragma: no cover — lookup_scripted raises, never returns None
            return None

    # Use get_type_hints to resolve string annotations (from __future__)
    # with the function's local namespace for locally-defined classes.
    # If the merge_fn was decorated with @merge_fn, use the caller_ns
    # captured at decoration time. Otherwise fall back to closure vars only.
    stored_ns = _merge_fn_caller_ns.get(merge_fn_name)
    try:
        extra_ns = _build_annotation_namespace(fn, caller_ns=stored_ns)
        hints = typing.get_type_hints(fn, localns=extra_ns, include_extras=False)
    except (NameError, AttributeError, TypeError):
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
