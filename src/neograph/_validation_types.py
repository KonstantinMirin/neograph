"""Type-compatibility primitives and shared vocabulary for construct validation.

Leaf module of the validation cluster (see ``_construct_validation.py`` for the
orchestrator + public seam). Holds the producer/consumer assignability rules
(``_types_compatible``), the type-introspection helpers they lean on, the
shared ``Producer``/``ProducerMap``/``NodeItem`` vocabulary, the
modifier-aware producer-side type rule (``effective_producer_type``), and the
user-frame source-location helper used to decorate error messages.

No intra-cluster dependencies — every other validation module imports FROM
here, never the reverse.
"""

from __future__ import annotations

import os
import sys
import types
from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    ForwardRef,
    TypeGuard,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from neograph._ir_protocols import ConstructItem, ConstructLike
from neograph.node import Node, TypeSpecStatic


@dataclass(frozen=True)
class Producer:
    """A producer registered during construct validation.

    effective_type is user-declared and therefore opaque from neograph's
    perspective — see docs/design/architecture-decisions.md §5 for the
    boundary rationale. label is rendered verbatim in error messages.
    """
    field_name: str
    effective_type: TypeSpecStatic
    label: str


# Items that appear in Construct.nodes — Node, Construct, or the _BranchNode
# sentinel. Aliased to the structural Protocol so this module's helpers share
# one polymorphic shape with Construct.nodes (see neograph._ir_protocols).
NodeItem = ConstructItem

# Producers collected during a single construct walk, keyed by state-field
# name. The dict makes the field_name → producer lookup O(1) (no per-call
# rebuild in _check_fan_in_inputs) and the ordering contract explicit.
ProducerMap = OrderedDict[str, Producer]


def _is_construct_like(item: NodeItem) -> TypeGuard[ConstructLike]:
    """Narrow a ``ConstructItem`` to a container ``ConstructLike``.

    True for sub-constructs (have both ``.input`` and ``.nodes``); False for
    ``Node`` and the ``_BranchNode`` sentinel. Replaces the inline
    ``is_sub_construct`` boolean plus the untyped escape-hatch cast at the
    recursive call site with a type-safe narrowing — no runtime ``Construct``
    import.
    """
    return (
        not isinstance(item, Node)
        and getattr(item, "input", None) is not None
        and getattr(item, "nodes", None) is not None
    )


def effective_producer_type(item: NodeItem) -> TypeSpecStatic:
    """Return the type this producer writes to the state bus, accounting
    for modifiers.

    This is the **single source of truth** for the "producer side" of
    type compatibility. The sole validator walker
    (``_validate_node_chain``) consults it, so a new modifier that
    reshapes state only needs to teach this one function about the new
    rule — the walker picks up the change automatically.

    Current rules:
      - ``Each`` modifier → ``dict[str, raw_output]`` (aggregated fan-out
        results land as a dict keyed by ``each.key``; see
        ``state.py:_add_output_field`` for the state builder side of
        this rule).
      - Everything else → the item's declared output (Node ``.outputs``,
        Construct ``.output``) unchanged.

    Returns ``None`` when the item has no declared output.
    """
    # Node uses .outputs (plural); Construct uses .output (singular).
    output = item.outputs if isinstance(item, Node) else getattr(item, "output", None)
    if output is None:
        return None
    ms = getattr(item, "modifier_set", None)
    if ms is None:
        return output
    if ms.each is not None:
        return dict[str, output]  # type: ignore[valid-type]
    return output


# Sentinel distinguishing "field absent" from "field present but None-valued" —
# needed because a Pydantic field annotation could legitimately be None, and
# we want a returnable value that never collides with user types.
_MISSING = object()


def _resolve_field_annotation(model_class: TypeSpecStatic, field_name: str) -> TypeSpecStatic:
    """Return the fully-resolved annotation for a field, or _MISSING if absent.

    Tries `typing.get_type_hints` first to unwrap ForwardRefs and string
    annotations introduced by `from __future__ import annotations`, then
    falls back to `model_fields[name].annotation`. If the final result is
    still unresolved (ForwardRef or bare string), returns `_MISSING` rather
    than leaking it to callers — otherwise `_extract_list_element` silently
    returns None on an unresolved annotation and the validation appears to
    pass when it should have flagged a resolution failure.

    Callers check ``is _MISSING`` before using the result; the cast keeps the
    sentinel sharing the return slot without widening to ``object``.
    """
    model_fields = getattr(model_class, "model_fields", None) or {}
    if field_name not in model_fields:
        return cast(TypeSpecStatic, _MISSING)
    try:
        hints = get_type_hints(model_class)
    except (NameError, AttributeError, TypeError):
        hints = {}
    ann = hints.get(field_name, model_fields[field_name].annotation)
    if ann is None or isinstance(ann, (str, ForwardRef)):
        return cast(TypeSpecStatic, _MISSING)
    return ann


def _types_compatible(producer: TypeSpecStatic, target: TypeSpecStatic) -> bool:
    """True if a value of type `producer` can satisfy a consumer of `target`.

    Handles parameterized generics (e.g. dict[str, X]) as well as plain classes.
    """
    if producer is target:
        return True
    producer_origin = get_origin(producer)
    target_origin = get_origin(target)
    # Unwrap Union/Optional types.
    # typing.Optional[X] = Union[X, None], PEP 604 X | None = UnionType.
    if producer_origin is Union or producer_origin is types.UnionType:
        prod_args = [a for a in get_args(producer) if a is not type(None)]
        return bool(prod_args) and all(_types_compatible(a, target) for a in prod_args)
    if target_origin is Union or target_origin is types.UnionType:
        target_args = [a for a in get_args(target) if a is not type(None)]
        return bool(target_args) and any(_types_compatible(producer, a) for a in target_args)
    # Parameterized generic producer (e.g. dict[str, X]):
    # compatible with raw origin class (dict) or exact parameterized match.
    if producer_origin is not None:
        # dict[str, X] vs dict → compatible (runtime isinstance handles it)
        if isinstance(target, type) and issubclass(producer_origin, target):
            return True
        # dict[str, X] vs dict[str, Y] → compare origin + args recursively
        if target_origin is not None and producer_origin is target_origin:
            p_args, t_args = get_args(producer), get_args(target)
            if len(p_args) != len(t_args):
                return False
            return all(_types_compatible(p, t) for p, t in zip(p_args, t_args, strict=True))
        # dict[str, X] producer ↔ list[Y] consumer — merge-after-fanout
        # (neograph-kqd.2). A downstream node consuming an Each-fanned-out
        # result as list[Y] gets the runtime unwrap via dict.values() in
        # step 5 (factory._extract_input). Element-type compatibility is
        # checked recursively so subclass rules apply consistently.
        if producer_origin is dict and target_origin is list:
            dict_args = get_args(producer)     # (str, X)
            list_args = get_args(target)       # (Y,)
            if len(dict_args) == 2 and len(list_args) == 1:
                return _types_compatible(dict_args[1], list_args[0])
        return False
    if not (isinstance(producer, type) and isinstance(target, type)):
        return False
    try:
        return issubclass(producer, target)
    except TypeError:
        return False


def _extract_list_element(tp: TypeSpecStatic) -> TypeSpecStatic:
    """If tp is list[X], Optional[list[X]], or list[X] | None, return X."""
    origin = get_origin(tp)
    if origin is list:
        args = get_args(tp)
        return args[0] if args else None
    # Handle Union / Optional / X | None. `requires-python >= 3.11` guarantees
    # types.UnionType exists, so no hasattr guard needed.
    if origin is Union or origin is types.UnionType:
        for arg in get_args(tp):
            if arg is type(None):
                continue
            element = _extract_list_element(arg)
            if element is not None:
                return element
    return None


def _fmt_type(tp: TypeSpecStatic) -> str:
    if tp is None:
        return "None"
    if hasattr(tp, "__name__"):
        return tp.__name__
    return repr(tp)


def _source_location() -> str | None:
    """Return 'file.py:line' for the user-code frame that assembled this Construct.

    Walks frames via `sys._getframe` past neograph and pydantic internals to
    find the first user frame — typically the `Construct(...)` call site.
    `sys._getframe` is ~50× cheaper than `inspect.stack()` because it doesn't
    materialize source context for every frame.

    Filters by module name (`frame.f_globals['__name__']`) rather than path
    substring, because user tests/examples often live under a `neograph/`
    directory that would otherwise get mis-filtered.
    """
    try:
        frame: types.FrameType | None = sys._getframe(1)
        while frame is not None:
            module_name = frame.f_globals.get("__name__", "")
            if not (
                module_name == "neograph"
                or module_name.startswith("neograph.")
                or module_name.startswith("pydantic")
            ):
                fname = frame.f_code.co_filename
                if fname and not fname.startswith("<"):
                    return f"{os.path.basename(fname)}:{frame.f_lineno}"
            frame = frame.f_back
    except (AttributeError, TypeError, ValueError):  # noqa: bare-except — frame walk best-effort
        return None
    return None
