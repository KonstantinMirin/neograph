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
from neograph._ir_branch import _BranchNode, iter_with_arms
from neograph._ir_fields import boundary_member_name
from neograph._ir_normalize import resolve_output_from
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.modifiers import PrimaryShape, primary_shape
from neograph.node import Node

#: Modifier shapes whose exported SpecNode declares outputs=
#: _properties_for(_item_outputs(item)) -- IDENTICAL to the bare case, so the
#: terminal-producer wiring below is correct for them too (verified against
#: _lower_oracle: both the merge_prompt LlmNode and merge_fn ToolNode declare
#: outputs = _properties_for(_item_outputs(node)) verbatim).
_WIRABLE_SHAPES = (PrimaryShape.BARE, PrimaryShape.ORACLE)

#: Shapes whose exported SpecNode declares outputs pyagentspec INFERS rather than
#: outputs neograph declares -- so the IR is the WRONG source and the lowered node
#: is read instead (``_inferred_output_props``). EACH: a MapNode's outputs are
#: ``collected_{inner title}``, list-wrapped by its default APPEND reducer
#: (``MapNode._get_inferred_outputs`` + ``_get_default_reducers``, reading
#: ``subflow.outputs`` <- the sub-Flow's EndNode). neograph-qtfof.11 made those
#: non-empty (``_lower_each`` now declares + feeds that EndNode); this admits them
#: to the wiring. The ``collected_`` prefix and the list-wrapping stay
#: pyagentspec's -- re-deriving either here would be the disease that ticket's
#: codebase scan pinned at zero instances.
_INFERRED_SHAPES = (PrimaryShape.EACH,)


def _is_wirable(item: Any) -> bool:
    return primary_shape(item) in _WIRABLE_SHAPES


def _inferred_output_props(item: Any, data_node_by_item_name: dict[str, Any]) -> tuple[list[Any], Any] | None:
    """``(properties, source_node)`` read off ``item``'s LOWERED SpecNode, or
    ``None`` when ``item``'s shape is not one pyagentspec infers outputs for.

    ``ComponentWithIO.model_post_init`` materialises ``.outputs`` from
    ``_get_inferred_outputs`` at construction, so by the time the caller holds the
    lowered node the inferred Properties are already there to be read.
    """
    if primary_shape(item) not in _INFERRED_SHAPES:
        return None
    name = getattr(item, "name", None)
    source_node = data_node_by_item_name.get(name) if isinstance(name, str) else None
    if source_node is None:
        return None
    return list(getattr(source_node, "outputs", None) or []), source_node


def _sub_flow_end_outputs(terminal: Any) -> list[Any]:
    """The output Properties a sub-Flow's synthetic ``EndNode`` must declare:
    exactly those its terminal producer exposes (neograph-qtfof.11).

    Read off the LOWERED SpecNode rather than recomputed from the IR, so a
    placeholder-translated (LLM-mode) body -- whose declared Properties are the
    flat ``${var}``-derived names, not the dotted IR ones -- stays correct with no
    second rule. ``EndNode`` mirrors ``outputs`` into ``inputs``
    (``_get_inferred_inputs``), which is what makes the edges below wirable.
    """
    return list(getattr(terminal, "outputs", None) or [])


def _sub_flow_boundary_edges(
    prefix: str,
    start_node: Any,
    body_nodes: list[Any],
    terminal: Any,
    end_node: Any,
    edges_mod: Any,
    existing: list[Any],
) -> list[Any]:
    """Every ``DataFlowEdge`` a sub-Flow's own boundary needs: ``StartNode`` ->
    each body input it can fill, then ``terminal`` -> ``EndNode``.

    **Why the StartNode half is not separable from the EndNode half.** A Flow's
    same-title data edges are synthesised by the loader ONLY while
    ``data_flow_connections is None`` (``_langgraphconverter``, all-or-nothing).
    A sub-Flow that shipped ``None`` was relying on that synthesis for its body's
    INPUT; the first explicit edge -- the EndNode one this ticket adds -- switches
    it off for the WHOLE sub-Flow. Emitting both halves together is what keeps
    that flip from turning a missing-result bug into a missing-input one.

    **Why this is not a blanket replay of the loader's rule.** The loader pairs
    EVERY same-titled ``(source output, destination input)``; replayed verbatim on
    a fused Each x Oracle body that would put BOTH variants AND the merge onto the
    EndNode's single input -- many sources, one destination, last writer wins. So
    only the two safe directions are emitted: one source (the StartNode) fanning
    out to many destinations, and the single terminal producer feeding the
    EndNode. ``existing`` (the modifier's own already-emitted edges) is honoured,
    never duplicated.
    """
    fed = {(edge.destination_node.name, edge.destination_input) for edge in existing}
    edges: list[Any] = []
    start_titles = {prop.title for prop in (getattr(start_node, "outputs", None) or [])}
    for body in body_nodes:
        for prop in getattr(body, "inputs", None) or []:
            if prop.title not in start_titles or (body.name, prop.title) in fed:
                continue
            fed.add((body.name, prop.title))
            edges.append(
                edges_mod.DataFlowEdge(
                    name=f"{prefix}__start_data_{body.name}_{prop.title}",
                    source_node=start_node,
                    source_output=prop.title,
                    destination_node=body,
                    destination_input=prop.title,
                )
            )
    for prop in getattr(end_node, "outputs", None) or []:
        if (end_node.name, prop.title) in fed:
            continue
        edges.append(
            edges_mod.DataFlowEdge(
                name=f"{prefix}__end_data_{prop.title}",
                source_node=terminal,
                source_output=prop.title,
                destination_node=end_node,
                destination_input=prop.title,
            )
        )
    return edges


def close_sub_flow(
    kind: str,
    name: str,
    start_node: Any,
    body_nodes: list[Any],
    body_control: list[Any],
    body_data: list[Any],
    flow_classes: tuple[Any, Any, Any],
) -> Any:
    """Build a modifier's sub-``Flow``, CLOSED at its terminal boundary
    (neograph-qtfof.11): the ``EndNode`` declares its terminal producer's outputs
    and every boundary ``DataFlowEdge`` is emitted.

    ``kind`` names the modifier for the synthesized node/edge names
    (``{name}__{kind}_end``); ``flow_classes`` is the caller's already-imported
    ``(nodes_mod, flow_mod, edges_mod)`` triple, so this module keeps its
    import-free relationship with pyagentspec.

    ``body_nodes[-1]`` is the terminal producer by construction -- the plain body,
    or ``_lower_oracle``'s trailing merge -- i.e. exactly the node the end control
    edge has always departed from. The caller's ``body_control`` order is preserved
    with the end edge APPENDED, so lowering order is unchanged.

    Why the whole boundary lives here rather than at the call site: a synthetic
    boundary node with no declared I/O was the one disease three separate export
    sites shared (neograph-qtfof.9 outermost, .11 Each, .12 Portal), and the
    ``data_flow_connections``-flip trap ``_sub_flow_boundary_edges`` documents is
    not something each caller should have to remember independently.
    """
    nodes_mod, flow_mod, edges_mod = flow_classes
    terminal = body_nodes[-1]
    end_node = nodes_mod.EndNode(name=f"{name}__{kind}_end", outputs=_sub_flow_end_outputs(terminal) or None)
    edges = _sub_flow_boundary_edges(name, start_node, body_nodes, terminal, end_node, edges_mod, body_data)
    return flow_mod.Flow(
        name=f"{name}__{kind}_body",
        start_node=start_node,
        nodes=[start_node, *body_nodes, end_node],
        control_flow_connections=[
            *body_control,
            edges_mod.ControlFlowEdge(name=f"{name}__{kind}_end_edge", from_node=terminal, to_node=end_node),
        ],
        data_flow_connections=[*body_data, *edges] or None,
    )


def _declared_terminal(construct: Any) -> Any | None:
    """The member a construct's ``output_from`` names, or ``None`` when unnamed.

    Reads the declaration through ``resolve_output_from`` -- the same reader
    validation and the runtime use -- so the exported artifact and the run cannot
    answer "which member is the boundary" differently. Deleted along with this
    comment if it ever stops being a call: design 7.5 asks for parity BY CALLING,
    never by asserting.
    """
    ref = resolve_output_from(construct)
    if ref is not None:
        for item in iter_with_arms(construct):
            if getattr(item, "name", None) == ref.member:
                return item
        return None

    # UNDECLARED boundary: ask the SHARED declaration-level derivation, which gives
    # the same answer the runtime does (last declared eligible member). Reading
    # construct.nodes[-1] blindly is not merely a wrong edge -- when the last member
    # is not the boundary producer it builds a DataFlowEdge between mismatched
    # properties and to_agent_spec CRASHES with a pydantic ValidationError on a
    # construct that runs fine. The positional rule was never a defensible default;
    # it held only while the last member happened to also be the producer.
    name = boundary_member_name(construct)
    if name is None:
        return None
    for item in iter_with_arms(construct):
        if getattr(item, "name", None) == name:
            return item
    return None


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
    # The DECLARED boundary port wins over position. Until neograph-9axw6.3 this read
    # construct.nodes[-1] unconditionally and never looked at output_from, while the
    # runtime honoured it -- so an exported Flow wired b -> EndNode for a construct
    # whose run returned a's output. Measured on a two-member pair: the documents
    # differed only in an inert metadata marker while the data-flow edges were
    # identical, which is why the differential harness asserts on the edge SET rather
    # than the document. That divergence is neograph-fnlrx / neograph-avmx4.
    #
    # A construct with no declared port keeps the positional rule exactly, so nothing
    # regresses where nothing was declared.
    last = _declared_terminal(construct) or construct.nodes[-1]

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

    # neograph-qtfof.11: an EACH terminal's Properties are pyagentspec's inferred
    # ``collected_*``, not neograph's declared ones -- read the lowered MapNode.
    # Branch arms are deliberately NOT extended to this: two MapNodes converging on
    # one EndNode input is the many-sources-one-destination shape the arm case above
    # only gets away with because both arms write the SAME converged Property.
    inferred = _inferred_output_props(last, data_node_by_item_name)
    if inferred is not None:
        props, source_node = inferred
        return (props, [(source_node, prop.title) for prop in props]) if props else fallback

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
