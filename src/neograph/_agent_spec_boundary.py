"""EndNode output-boundary wiring — shared by every Construct export
(neograph-qtfof.9).

Extracted as its own module because ``_agent_spec.py`` sits at its exact
file-size ceiling (``tests/test_guards_file_size.py``); this is also the
shared helper ``neograph-qtfof.11``/``neograph-qtfof.12`` are specified to
reuse, so a standalone module is the right shape regardless.

Root cause this closes: a synthetic ``EndNode`` (the boundary every
``to_agent_spec()`` call synthesizes — outermost AND nested) declared its
outputs from ``construct.output`` alone, with NO ``DataFlowEdge`` ever
feeding it (``_emit_input_edges`` only ever targets real construct ITEMS; an
``EndNode`` is not an item at any nesting level). Failure mode (a): the
outermost ``construct.output`` is always ``None`` by design, so
``end_props`` was empty and a third-party ``invoke()`` unconditionally
returned ``{}``. Failure mode (b): whenever the ``EndNode`` DID declare
outputs, the loader raised ``Expected node to have a value for property X``
since nothing fed it.

SCOPE BOUNDARY (deliberate, not an oversight): wiring is only attempted when
the terminal producer's exported SpecNode declares
``outputs=_properties_for(item.outputs)`` verbatim -- BARE, and ORACLE
(verified against ``_lower_oracle``: both its ``merge_prompt`` LlmNode and
``merge_fn`` ToolNode declare exactly that). Each/Loop/Portal are NOT in this
set -- e.g. a MapNode infers its outputs from its inner sub-Flow's EndNode, a
different, wrapped/titled shape (this is exactly neograph-qtfof.7's
still-open MapNode ``iterated_item`` gap). Reproducing that resolution
correctly for every modifier is real, separate scope — attempted naively here
it silently emitted a DataFlowEdge naming a Property the exported node does
not have, which pydantic's own ``DataFlowEdge`` validation rejects (verified:
broke the existing GREEN ``act-each-single`` cell). An unwirable terminal
item therefore falls back to today's behavior (``construct.output`` for a
nested sub-construct, empty for the outermost) — narrower than a full fix,
but every case this narrows away was ALREADY broken (mode (a) or (b)) before
this ticket, so nothing regresses; only BARE/ORACLE, which is what both of
this ticket's own reproduction tests exercise, is fixed here. See this
module's own test coverage and the ticket's follow-up for extending to
Each/Loop/Portal terminals.

``resolve_end_node_sources`` computes the terminal producer's output
Properties AND the ``(source_node, property_title)`` pairs the caller wires
via its own ``_emit_input_edges`` closure (that closure captures per-call
state and cannot be extracted here without changing what it closes over —
see the caller). R2 (multi-exit branch): the terminal producer is
``construct.nodes[-1]``; if it is a ``_BranchNode``, both arms' own last item
must converge on a type-compatible BARE output, or export fails LOUD
(``ConfigurationError``) rather than guessing a union — maintainer decision.
Multi-exit-branch Agent Spec export support itself is a documented follow-up,
not implemented here.
"""

from __future__ import annotations

from typing import Any, cast

from neograph._agent_spec_placeholders import _item_outputs, _properties_for
from neograph._ir_branch import _BranchNode
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.modifiers import PrimaryShape, primary_shape
from neograph.node import Node

#: Modifier shapes whose exported SpecNode declares outputs=
#: _properties_for(_item_outputs(item)) -- IDENTICAL to the bare case, so the
#: terminal-producer wiring below is correct for them too (verified against
#: _lower_oracle: both the merge_prompt LlmNode and merge_fn ToolNode declare
#: outputs = _properties_for(_item_outputs(node)) verbatim). EACH is
#: deliberately excluded -- a MapNode infers its own outputs from its inner
#: sub-Flow's EndNode, a different shape (see module docstring).
_WIRABLE_SHAPES = (PrimaryShape.BARE, PrimaryShape.ORACLE)


def _is_wirable(item: Any) -> bool:
    return primary_shape(item) in _WIRABLE_SHAPES


def resolve_end_node_sources(
    construct: Construct,
    data_node_by_item_name: dict[str, Any],
) -> tuple[list[Any], list[tuple[Any, str]]]:
    """Return ``(end_output_properties, sources)`` for ``construct``'s
    synthetic ``EndNode``, where ``sources`` is a list of
    ``(source_spec_node, property_title)`` pairs the caller wires via its own
    ``_emit_input_edges``. Raises ``ConfigurationError`` (R2) if the terminal
    item is a ``_BranchNode`` whose arms do not converge on a compatible
    output. Falls back to ``(_properties_for(construct.output), [])`` --
    today's pre-fix shape -- when the terminal producer is modifier-wrapped
    (see module docstring's scope boundary).
    """
    fallback: tuple[list[Any], list[tuple[Any, str]]] = (_properties_for(construct.output), [])
    last = construct.nodes[-1]

    if isinstance(last, _BranchNode):
        meta = last._neo_branch_meta
        true_last = meta.true_arm_nodes[-1] if meta.true_arm_nodes else None
        false_last = meta.false_arm_nodes[-1] if meta.false_arm_nodes else None
        if true_last is None or false_last is None:
            raise ConfigurationError.build(
                f"construct {construct.name!r}'s terminal branch has an empty arm",
                expected="both branch arms non-empty with a convergent terminal output",
                found="true_arm_nodes empty" if true_last is None else "false_arm_nodes empty",
                hint="neograph-qtfof.9: the Agent Spec EndNode needs exactly one converged "
                "terminal output; an empty arm has none to converge on",
            )
        if not (_is_wirable(true_last) and _is_wirable(false_last)):
            return fallback
        true_type = _item_outputs(true_last)
        false_type = _item_outputs(false_last)
        if not (
            isinstance(true_type, type)
            and isinstance(false_type, type)
            and (issubclass(true_type, false_type) or issubclass(false_type, true_type))
        ):
            raise ConfigurationError.build(
                f"construct {construct.name!r}'s branch arms terminate in incompatible output types",
                expected="both arms' terminal producer to converge on a compatible output type",
                found=f"true arm: {true_type!r}, false arm: {false_type!r}",
                hint="neograph-qtfof.9 (R2): a multi-exit-branch Construct needs one EndNode "
                "with one converged output; diverging terminal types have no single Agent Spec "
                "representation today (see the multi-exit-branch export follow-up ticket)",
            )
        props = _properties_for(true_type)
        sources = [
            (source_node, prop.title)
            for item in (true_last, false_last)
            if (source_node := data_node_by_item_name.get(item.name)) is not None
            for prop in props
        ]
        return props, sources

    if not _is_wirable(last):
        return fallback

    # _is_wirable's BARE/ORACLE shapes only ever apply to a Node/Construct item
    # (the _BranchNode case already returned above) -- primary_shape/_is_wirable
    # take Any (they must, to classify a raw ConstructItem before narrowing), so
    # this cast documents the narrowing mypy cannot derive from that.
    wirable_last = cast("Node | Construct", last)
    props = _properties_for(_item_outputs(wirable_last))
    name = getattr(wirable_last, "name", None)
    source_node = data_node_by_item_name.get(name) if isinstance(name, str) else None
    sources = [(source_node, prop.title) for prop in props] if source_node is not None else []
    return props, sources
