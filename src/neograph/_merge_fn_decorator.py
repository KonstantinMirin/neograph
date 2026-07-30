"""``@merge_fn`` — the standalone merge-function decorator.

Extracted from ``decorators.py`` (neograph-3ffdg.11) as a pure file split — the
functions below are unchanged, only their home moved. ``decorators.py``
re-exports them, so existing ``from neograph.decorators import merge_fn`` call
sites keep working.

Why the registry is name-keyed (unchanged by this split, restated here because
this is now its home): ``@merge_fn`` decorates a standalone function and returns
the bare function, which is referenced from a DIFFERENT object
(``Oracle(merge_fn='combine')``) purely by STRING NAME — no Node/Oracle is in
scope at decoration time to self-store on, unlike ``@node``. Same-name
collisions between DIFFERENT definition sites FAIL LOUD; the same definition site
is idempotent via ``_same_def_site``. The registry itself lives in
``_sidecar.py``.
"""

from __future__ import annotations

import inspect
import os
import sys
from collections.abc import Callable
from typing import Any

from neograph._construct_validation import ConstructError
from neograph._di_classify import (
    ParamResolution,
    _build_annotation_namespace,
    _classify_di_params,
    _resolve_di_args,
)
from neograph._hints import resolve_hints
from neograph._runtime_registry import register_scripted
from neograph._sidecar import _merge_fn_caller_ns, _merge_fn_registry
from neograph.di import DIBinding, DIKind


def _qualname_site(f: Callable) -> str:
    """Human-readable definition site for a function: ``module.qualname (file.py:lineno)``.

    Used to name both sides of a @merge_fn collision so the error points at the
    two competing definitions rather than just the shared registry name.
    """
    module = getattr(f, "__module__", None) or "<unknown>"
    qualname = getattr(f, "__qualname__", None) or getattr(f, "__name__", repr(f))
    label = f"{module}.{qualname}"
    code = getattr(f, "__code__", None)
    if code is not None:
        label += f" ({os.path.basename(code.co_filename)}:{code.co_firstlineno})"
    return label


def _same_def_site(a: Callable, b: Callable) -> bool:
    """True when two callables originate from the same ``def`` statement.

    A collision is two *distinct* definitions competing for one registry name;
    re-executing the same ``def`` is not one. This is the same object (``is``),
    the same code object (a ``def`` re-run in a loop / hypothesis example — new
    function object, shared code object), or the same source site with a matching
    qualname (a module reload recompiles the code object but keeps file/line/name).
    """
    if a is b:
        return True
    ca, cb = getattr(a, "__code__", None), getattr(b, "__code__", None)
    if ca is not None and ca is cb:
        return True
    if ca is None or cb is None:
        return False
    return (
        getattr(a, "__qualname__", None) == getattr(b, "__qualname__", None)
        and ca.co_filename == cb.co_filename
        and ca.co_firstlineno == cb.co_firstlineno
    )


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
        # Rebuild param_res in function signature order so positional args match
        # the function's parameter order. Per-annotation resolution (7ymj): one
        # unresolvable annotation no longer drops the OTHER params' hints and
        # mis-types their FROM_STATE bindings.
        extra_ns = _build_annotation_namespace(f, caller_ns=caller_ns)
        all_hints = resolve_hints(f, localns=extra_ns, owner=getattr(f, "__name__", None))

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
                        name=p.name,
                        kind=DIKind.CONSTANT,
                        inner_type=type(p.default),
                        required=False,
                        default_value=p.default,
                    )
                else:
                    ordered_res[p.name] = DIBinding(
                        name=p.name,
                        kind=DIKind.FROM_STATE,
                        inner_type=hint if hint is not None else type(None),
                        required=False,
                    )
        param_res = ordered_res

        fn_name = name or f.__name__
        # Fail loud on a same-name collision between two *different* functions.
        # Oracle references a merge_fn by string name, so a silent overwrite in
        # _merge_fn_registry lets two modules with a common helper name
        # (merge/combine) corrupt each other's Oracles with zero signal.
        # Re-registering the identical function object — as a module reload /
        # re-import does — stays idempotent.
        existing = _merge_fn_registry.get(fn_name)
        if existing is not None and not _same_def_site(existing[0], f):
            prior = existing[0]
            raise ConstructError.build(
                f"merge_fn name '{fn_name}' is already registered by a different function",
                found=f"{_qualname_site(prior)} then {_qualname_site(f)}",
                hint=(
                    "Two @merge_fn functions cannot share a registry name. Give one a "
                    "distinct name via @merge_fn(name='...'), or rename the function."
                ),
            )
        _merge_fn_registry[fn_name] = (f, param_res)
        _merge_fn_caller_ns[fn_name] = caller_ns

        # Auto-register via register_scripted so Oracle's existing string
        # lookup path finds the function. The factory wrapper we return
        # here is a legacy-compatible (variants, config) shim that falls
        # back to calling the user function with positional args if the
        # factory hasn't hooked into the DI path. In practice the factory
        # always checks _merge_fn_registry first (see
        # factory.make_oracle_merge_fn) so this shim is rarely invoked.
        def legacy_shim(variants: Any, config: Any) -> Any:
            return f(variants, *_resolve_di_args(param_res, config))

        legacy_shim.__name__ = fn_name
        register_scripted(fn_name, legacy_shim)

        return f

    if fn is not None:
        return decorator(fn)
    return decorator
