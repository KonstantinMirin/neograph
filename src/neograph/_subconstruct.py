"""Builds the per-sub-construct Callable that LangGraph adds to the parent StateGraph; encodes the sub-construct's input/output boundary semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel

from neograph._ir_branch import iter_with_arms
from neograph._ir_fields import item_field_names
from neograph._ir_normalize import resolve_output_from
from neograph._ir_source import HandoffChannel, LastPresent, LoopCarry, Source, source_channel_kind
from neograph._oracle import _inject_oracle_config
from neograph._read_source import read_source
from neograph._state_bus import StateBus, adapt_state
from neograph._state_keys import StateKeys
from neograph.construct import Construct
from neograph.di import _unwrap_loop_value
from neograph.errors import ExecutionError, StateMissingError
from neograph.modifiers import (
    COMBO_DECOMPOSITION,
    PrimaryShape,
    classify_modifiers,
)
from neograph.naming import field_name_for

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

log = structlog.get_logger()


def _scan_subgraph_output(sub_result: dict[str, Any], sub_output_type: type, *, eligible: list[str]) -> Any:
    """Resolve a sub-construct's boundary value from the child's final state.

    ``eligible`` restricts the candidates to the fields the
    construct's own declared items write, LAST-DECLARED-ITEM FIRST, so a value the
    child was merely handed can never win. Measured over the full suite at the time
    of the change: 270 boundary resolutions, 26 with more than one type match, and
    this rule changes the answer on 2 -- both of them the reported bug. The other 24
    are ordinary same-typed chains (``review``->``revise``, ``write``->``improve``)
    and keep resolving exactly as before, which is why ORDERING was chosen over
    refusing (a refusal would have broken the canonical refine sub-construct).

    ``eligible`` is REQUIRED: there is no whole-state caller. Portal mode-(b)
    dispatch resolves its flow's boundary over the same set, because the dispatched
    flow is built through ``from_agent_spec`` and normalized before it is invoked
    -- neograph-5zl3c.

    Unwraps loop append-lists (``list[T]`` from the reducer) by checking ``val[-1]``
    against the declared output type.
    """
    for field in reversed(eligible):
        if field not in sub_result:
            continue
        check_val = _unwrap_loop_value(sub_result[field], object)
        if isinstance(check_val, sub_output_type):
            return check_val
    return None


def _absence_is_a_defect(source: Source) -> bool:
    """Whether reading nothing through ``source`` means something is WRONG.

    Two channels are legitimately empty at a known moment, and both were documented
    before this: a ``LoopCarry`` on iteration 0 (there is no previous iteration) and a
    mesh ``HandoffChannel`` on the entry's first activation (no hop has run). Every
    other rung names a field that something wrote before this node, so reading nothing
    through it means the address is broken -- which used to omit the port key and hand
    the child's first node ``None``.

    An ABSENT stamp is not a defect either: it means nothing in the parent could feed
    the port, which is ``neograph-xejyn`` and still tolerated -- the caller checks that
    before asking here.
    """
    rungs = source.rungs if isinstance(source, LastPresent) else (source,)
    return any(not isinstance(rung, (LoopCarry, HandoffChannel)) for rung in rungs)


def make_subgraph_fn(
    sub: Construct, sub_graph: CompiledStateGraph, *, handoff_channel: str | None = None
) -> RunnableLambda:
    """Create a Runnable that runs a sub-Construct in isolation.

    Extracts input from parent state by type, runs sub_graph, extracts output
    by type, returns {field_name: output}.

    Dual-path (driver-selected, neograph-expi): returns
    ``RunnableLambda(subgraph_node, afunc=asubgraph_node)`` so the DRIVER picks
    the path — ``graph.invoke`` runs the sync twin (``sub_graph.invoke``),
    ``graph.ainvoke`` runs the async twin (``await sub_graph.ainvoke``). Without
    the afunc twin, LangGraph threadpools the sync closure under ``ainvoke`` and
    the ENTIRE child runs synchronously, blocking the loop and silently
    downgrading any async-only leaf inside the child (Phase-1 H2 invariant: async
    must propagate through every nesting level). The two twins share the same
    input-extraction (``_build_sub_input``) and update-shaping
    (``_build_update``) helpers so the sync/async paths cannot drift.

    ``handoff_channel`` (do0d9 site 7): when this sub-construct is a Portal mesh
    MEMBER, its boundary input MUST be sourced DETERMINISTICALLY from the routed
    parent handoff channel (``StateKeys.handoff_payload(entry_field)``) — NOT the
    port read below. In a uniform-payload mesh every member field
    AND the channel hold the same payload type, so a blind scan can feed the
    WRONG instance (a silent mis-route the North Star forbids). This mirrors the
    atomic member's reserved-``handoff`` read (``_input_shape.py:119-123``). The
    default (``None``) keeps every NON-mesh sub-construct on the general blind
    scan; only ``make_portal_subgraph_fn`` passes the channel key.
    """
    sub_log = log.bind(subgraph=sub.name)
    field_name = field_name_for(sub.name)

    sub_combo, _ = classify_modifiers(sub)
    sub_decomp = COMBO_DECOMPOSITION[sub_combo]
    sub_shape = sub_decomp.primary
    has_loop = sub_shape is PrimaryShape.LOOP
    # EACH-shaped but NOT the Each x Oracle fusion: EACH_ORACLE decomposes to
    # primary=EACH, so decomp.fused preserves the exclusion the old two-member
    # combo tuple had. It is load-bearing, not defensive — this function is
    # called at compiler.py:512, BEFORE _add_subgraph's
    # SUB_CONSTRUCT_UNSUPPORTED_COMBOS gate rejects the fusion, so the line
    # really does execute for a fused sub-construct.
    has_each = sub_shape is PrimaryShape.EACH and not sub_decomp.fused

    def _build_sub_input(
        state: BaseModel | dict[str, Any], config: RunnableConfig
    ) -> tuple[dict[str, Any], StateBus, RunnableConfig]:
        """Shared pre-invoke logic: extract input, forward context, inject config.

        Returns ``(sub_input, bus, config)``. Identical for both twins — the only
        difference between sync and async is the ``invoke`` vs ``ainvoke`` call.
        """
        bus = adapt_state(state)

        # ONE stamped Source, ONE read. This was a five-rung ladder -- the child's own
        # carry list, the fanned item, the mesh channel, the parent's port, the
        # resolved peer -- each tried by PRESENCE, two of them with an extra
        # isinstance probe, and when every rung missed the port key was OMITTED so the
        # child's first node read nothing and its body was handed None on a green run.
        #
        # The rungs are the same and in the same precedence order; what changed is
        # that the ORDER is now written down at assembly (resolve_port_source) instead
        # of being the order of five if-statements, and an absence is reported.
        port_source = sub.port_source
        input_data = (
            read_source(bus, port_source, sub.input, label=field_name)
            if sub.input is not None and port_source is not None
            else None
        )
        if (
            sub.input is not None
            and port_source is not None
            and input_data is None
            and _absence_is_a_defect(port_source)
        ):
            # A STAMPED address that holds nothing is a broken address: assembly said
            # which field feeds this port, and that field was never written. Omitting
            # the port key -- the old behaviour -- handed the child's first node None
            # on a green run.
            #
            # An ABSENT stamp is a different question and is deliberately not raised
            # here: it means nothing in the parent could feed the port, which is
            # neograph-xejyn, and a ported child fed from OUTSIDE through run(input=)
            # has no other spelling until xejyn's first half lands. A synthesized
            # fan-agent port with no fields is the other legitimate case -- there is
            # nothing to read.
            # StateMissingError, not a bespoke ExecutionError: this IS a required
            # state read that missed, which is the error's stated purpose, and the
            # canonical form names the reading node the way every other required read
            # does. The channel the address names stands in for the key when the source
            # has more than one rung.
            raise StateMissingError.build(
                key=source_channel_kind(port_source),
                node_label=sub.name,
            )

        # Run sub-graph with isolated state.
        # StateBus.get optional: framework — node_id is a DI-style context key
        # that may not be present; empty-string default propagates to sub-graph.
        sub_input: dict[str, Any] = {StateKeys.NODE_ID: bus.get(StateKeys.NODE_ID, "")}
        if input_data is not None:
            sub_input[StateKeys.SUBGRAPH_INPUT] = input_data

        # Forward context fields from parent state into sub-construct.
        # iter_with_arms so a context node living inside a branch arm of the
        # sub-construct gets its context field forwarded, not resolved to None.
        # See neograph-vn5f (site 11).
        for n in iter_with_arms(sub):
            if hasattr(n, "context") and n.context:
                for ctx_name in n.context:
                    ctx_field = field_name_for(ctx_name)
                    # StateBus.get optional: context forwarding is best-effort;
                    # missing context propagates as None to sub-node, whose own
                    # _extract_context read enforces required-ness (see §7 Q3).
                    val = bus.get(ctx_field)
                    if val is not None:
                        sub_input[ctx_field] = val

        # Forward Oracle gen_id + model override from parent state into config
        config = _inject_oracle_config(bus, config)
        return sub_input, bus, config

    def _build_update(sub_result: dict[str, Any], bus: StateBus) -> dict[str, Any]:
        """Shared post-invoke logic: extract declared output, shape state update.

        Identical for both twins.
        """
        # Extract the declared output type from sub result.
        output_val = None
        if sub.output is not None:
            # GH #17: a declared port wins outright; otherwise the boundary
            # is the LAST DECLARED ITEM that produced the type -- never a forwarded
            # context value or another field the child was merely handed.
            # The member->field hop goes through PortRef.field, the ONE place it is
            # computed. Spelling it here was wrong for a dotted address: step 1 made
            # output_from="settle.result" legal at assembly, and field_name_for on the
            # whole string yields "settle.result", a field nothing writes -- so a
            # correctly-named port assembled and then died below at "no internal node
            # produced a compatible output value" neograph-fx3j7.
            ref = resolve_output_from(sub)
            eligible = [ref.field] if ref is not None else item_field_names(sub)
            output_val = _scan_subgraph_output(sub_result, sub.output, eligible=eligible)

        # Runtime defense: if no internal node produced a compatible output,
        # fail loud instead of writing None silently.
        #
        # This carried `# pragma: no cover -- defensive` until neograph-9axw6.2, which
        # was false: with a NAMED port whose type mismatched, `eligible` is that one
        # field and nothing else, so this branch was the ordinary outcome rather than
        # an unreachable guard -- reached, and with a message naming the wrong cause.
        # Step 1 refuses that construct at ASSEMBLY, which is what finally makes this
        # branch the genuine defense the pragma claimed it already was. The pragma is
        # removed rather than re-worded: it asserted a property it did not have.
        if output_val is None and sub.output is not None:
            raise ExecutionError.build(
                "No internal node produced a compatible output value",
                expected=sub.output.__name__,
                hint="Check that at least one node writes the declared output type",
                construct=sub.name,
            )

        sub_log.info("subgraph_complete")
        update: dict[str, Any] = {field_name: output_val}
        if has_loop:
            count_field = StateKeys.loop_count(field_name)
            # Counter bootstrap (absent/None -> 0) lives in StateBus.get_counter.
            current = bus.get_counter(count_field)
            update[count_field] = current + 1
        return update

    def subgraph_node(state: BaseModel | dict[str, Any], config: RunnableConfig) -> dict:
        sub_log.info("subgraph_start")
        sub_input, bus, config = _build_sub_input(state, config)
        # No strip: the child compile declared output_schema=non-neo_ fields, so
        # sub_graph.invoke() already returns neo_-free results. See neograph-pjqe.
        # The reverse-scan in _build_update sees the same dict _strip_internals made.
        sub_result = sub_graph.invoke(sub_input, config=config)
        return _build_update(sub_result, bus)

    async def asubgraph_node(state: BaseModel | dict[str, Any], config: RunnableConfig) -> dict:
        sub_log.info("subgraph_start")
        sub_input, bus, config = _build_sub_input(state, config)
        # Async twin: await the child's ainvoke so a sub-construct under the async
        # driver propagates async selection into the child graph, instead of
        # blocking the loop on sub_graph.invoke. See neograph-expi.
        # No strip: child output_schema filters ainvoke() results. See neograph-pjqe.
        sub_result = await sub_graph.ainvoke(sub_input, config=config)
        return _build_update(sub_result, bus)

    # Driver-selected dual path. __name__ stays informational; routing is the
    # graph.add_node(name, fn) argument (always sub.name/item.name). See
    # neograph-y20i. Trace-span naming per neograph-3fm1 is applied by the
    # graph-assembly layer (`named(...)` at the add_node sites in compiler.py and
    # _wiring._add_arm_nodes) so this factory keeps returning the bare
    # dual-path RunnableLambda the async guard pins.
    # `name=` is set HERE rather than by `named(...)` at the assembly layer:
    # this RunnableLambda closes over a compiled Pregel, and `named` calls
    # `.with_config(...)`, which returns a RunnableBinding that LangGraph's
    # subgraph walker cannot see through. See neograph-xunot / GH #6.
    return RunnableLambda(subgraph_node, afunc=asubgraph_node, name=sub.name)
