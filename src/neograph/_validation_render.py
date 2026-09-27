"""How a refused read is EXPLAINED: rendering the resolver's verdict.

Split from ``_validation_inputs``, which decides WHETHER a read is satisfied. The
seam became sharp when the decision moved out of this cluster entirely: the
resolver in ``_ir_normalize`` resolves each read once, before validation, and what
is left here is to say -- well, and in the author's vocabulary -- what it found.

The rule these functions exist to keep: RENDER the ``Unresolved`` the resolver
produced; never re-run the search behind it. Re-probing was how a refusal came to
advise "have every arm produce a compatible value" to an author whose every arm
already did.
"""

from __future__ import annotations

from typing import get_args, get_origin

from neograph._ir_protocols import ConstructLike
from neograph._ir_source import Unresolved
from neograph._validation_types import (
    _MISSING,
    NodeItem,
    ProducerMap,
    _extract_list_element,
    _fmt_type,
    _resolve_field_annotation,
    _source_location,
    _types_compatible,
)
from neograph.errors import ConstructError, NeographError
from neograph.node import Node, TypeSpecStatic

__all__ = ["_build_no_producer_error"]


def _build_no_producer_error(
    construct: ConstructLike,
    item: NodeItem,
    input_type: TypeSpecStatic,
    producers: ProducerMap,
    all_producers: ProducerMap | None = None,
    *,
    refusal: Unresolved | None = None,
) -> NeographError:
    """Render the resolver's ``Unresolved``; never re-run the search behind it.

    ``refusal.candidates`` carry the reason the resolver computed, so the message
    says which producers it considered and why each lost. Re-probing here with a
    predicate of this module's own is what let the verdict and its explanation
    disagree (design 7.4).
    """
    authored = getattr(item, "input_from", None)
    if authored is not None and refusal is not None and refusal.candidates:
        # The author NAMED a port and the name did not resolve. Lead with the name,
        # because that is the one thing they wrote and the one thing to fix; the
        # reason comes from the resolver, which is why it can say "no such producer"
        # and "wrong type" without this module re-deciding which it was.
        why = refusal.candidates[0]
        return ConstructError.build(
            f"declares input_from={authored!r}, which {why.reason}",
            expected=f"a member producing {_fmt_type(input_type)}",
            found=f"input_from={authored!r}",
            hint=(
                "input_from names a MEMBER, or 'member.output' for one of a dict-form node's keys, "
                "declared before this node (a branch arm's member counts, and means that arm's value)"
            ),
            node=item.name,
            construct=construct.name,
            location=_source_location(),
        )
    arm_candidates = [c for c in (refusal.candidates if refusal else ()) if "branch arm" in c.reason]
    if arm_candidates:
        named = ", ".join(sorted(c.ref.member for c in arm_candidates))
        return ConstructError.build(
            f"declares "
            f"{'inputs' if isinstance(item, Node) else 'input'}="
            f"{_fmt_type(input_type)}, which a branch arm produces: {named}",
            expected="a read that says WHICH producer it means",
            found=f"compatible producers on branch arms: {named}; visible here: {sorted(producers) or '(none)'}",
            hint=(
                "which arm ran is a runtime fact, so this cannot be resolved by type: name the producer "
                "with input_from='<member>', move one above the branch, or have every arm append to a "
                "shared Accumulate[...] channel and read that"
            ),
            node=item.name,
            construct=construct.name,
            location=_source_location(),
        )
    if producers:
        producer_summary = "\n".join(f"    - {p.label}: {_fmt_type(p.effective_type)}" for p in producers.values())
    else:
        producer_summary = "    (no upstream producers)"

    return ConstructError.build(
        f"declares "
        f"{'inputs' if isinstance(item, Node) else 'input'}="
        f"{_fmt_type(input_type)} but no upstream produces a "
        f"compatible value",
        found=f"upstream producers:\n{producer_summary}",
        hint=_suggest_hint(input_type, producers),
        node=item.name,
        construct=construct.name,
        location=_source_location(),
    )


def _suggest_hint(
    input_type: TypeSpecStatic,
    producers: ProducerMap,
) -> str | None:
    """Scan producer outputs for actionable suggestions."""
    # Check for Each dict[str, X] → raw X mismatch first.
    for p in producers.values():
        p_origin = get_origin(p.effective_type)
        if p_origin is dict:
            p_args = get_args(p.effective_type)
            if p_args and len(p_args) == 2:
                element_type = p_args[1]
                if isinstance(input_type, type) and isinstance(element_type, type):
                    try:
                        match = issubclass(element_type, input_type) or issubclass(input_type, element_type)
                    except TypeError:
                        match = False
                    if match:
                        return (
                            f"upstream produces dict[str, {_fmt_type(element_type)}] "
                            f"via Each — consume the whole dict with input=dict "
                            f"or input=dict[str, {_fmt_type(element_type)}]"
                        )

    # Fallback: scan for list[input_type] fields and suggest .map().
    for p in producers.values():
        model_fields = getattr(p.effective_type, "model_fields", None) or {}
        for fname in model_fields:
            resolved = _resolve_field_annotation(p.effective_type, fname)
            if resolved is _MISSING:
                continue
            element = _extract_list_element(resolved)
            if element is not None and _types_compatible(element, input_type):
                return f"did you forget to fan out? try .map(lambda s: s.{p.field_name}.{fname}, key='...')"
    return None
