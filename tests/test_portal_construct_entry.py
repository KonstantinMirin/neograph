"""Regression (TDD red) for neograph-s7zt3.5 — Portal Construct-as-mesh-ENTRY.

Sibling of ``test_portal_cross_subconstruct.py`` (do0d9), which put a Construct
in a NON-entry Portal position. This file pins the harder shape the master
architecture doc (§2/§5) and the design-gap probe
(docs/design/architecture-audit-phase1-design-gap-probe-2026-07-27.md) call out:
a Portal-modified ``Construct`` as the mesh **ENTRY** (``members[0]``).

Three compiler sites must land together (probe Findings 1/2/3):

1. ``compiler.py``'s ``isinstance(item, Node)`` mesh-detection gate misroutes a
   Portal-Construct entry into ``_add_subgraph`` (unconditional CompileError).
2. ``state.py``'s Portal state-field builder sources ``portal_members`` from
   ``nodes_only`` only, so a Construct entry is excluded and the FIRST Node peer
   is mis-declared as the entry field (``neo_handoff_<peer>`` instead of
   ``neo_handoff_<entry_sub>``) — silent state-key divergence.
3. ``_ir_normalize.py``'s own ``portal_members`` collector (sole writer of
   ``Node.handoff_channel``) has the identical ``isinstance(item, Node)`` gate,
   so it stamps the WRONG ``handoff_channel`` on the Node peers — at RUNTIME the
   peer's ``handoff`` input resolves to ``None`` (reads the wrong, empty channel).

The acceptance test is NOT 'does not raise': the real failure mode of a partial
fix is silent state-key divergence, so the tests assert the declared Portal
field names MATCH ``_wiring.py``'s ``entry_field`` for the Construct entry, and a
runtime-invoking test proves the routed payload actually reaches the Node peer.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from pydantic import BaseModel

from neograph import (
    HANDOFF_END,
    Construct,
    FromInput,
    Portal,
    arun,
    compile,
    construct_from_functions,
    node,
    run,
)
from neograph._state_keys import StateKeys
from neograph.naming import field_name_for
from neograph.state import compile_state_model


# ── the uniform mesh payload (also the entry sub-construct's declared output) ──
class Handoff(BaseModel, frozen=True):
    goto: str  # a parent peer name, or HANDOFF_END
    subject: str
    trail: list[str] = []
    resolution: str | None = None


class SubDecision(BaseModel, frozen=True):
    subject: str
    kind: Literal["local", "escalate"]
    trail: list[str]


# ── the ENTRY sub-construct's internal members ────────────────────────────────
# ``ticket: Handoff`` is a boundary PORT param (its type == input=Handoff), so on
# the mesh entry's first activation it is fed by the deterministic type-scan of
# parent state (the ``origin`` node's Handoff); on a hop it reads the channel.
def _make_entry_sub() -> Construct:
    @node(outputs=SubDecision)
    def sub_intake(ticket: Handoff) -> SubDecision:
        kind: Literal["local", "escalate"] = (
            "escalate" if "escalate" in ticket.subject.lower() else "local"
        )
        return SubDecision(subject=ticket.subject, kind=kind, trail=[*ticket.trail, "sub_intake"])

    @node(outputs=Handoff)
    def sub_decide(sub_intake: SubDecision) -> Handoff:
        trail = [*sub_intake.trail, "sub_decide"]
        if sub_intake.kind == "local":
            return Handoff(
                goto=HANDOFF_END,
                subject=sub_intake.subject,
                trail=trail,
                resolution="resolved inside entry sub",
            )
        # ESCAPE to the Node peer at the parent level.
        return Handoff(goto="closer", subject=sub_intake.subject, trail=trail)

    return construct_from_functions(
        "entry_sub", [sub_intake, sub_decide], input=Handoff, output=Handoff
    )


# ── the parent mesh: the Construct is the ENTRY (first Portal member) ──────────
def _make_parent() -> Construct:
    @node(outputs=Handoff)
    def origin(subject: Annotated[str, FromInput] = "") -> Handoff:
        # Ordinary upstream: seeds the initial Handoff into parent state so the
        # ENTRY sub-construct's first-activation type-scan picks it up.
        return Handoff(goto="entry_sub", subject=subject, trail=["origin"])

    entry_sub = _make_entry_sub()

    @node(outputs=Handoff, portal=[])
    def closer(handoff: Handoff) -> Handoff:
        return Handoff(
            goto=HANDOFF_END,
            subject=handoff.subject,
            trail=[*handoff.trail, "closer"],
            resolution="resolved by parent closer",
        )

    # THE ACCEPTANCE-CRITICAL SHAPE: the Construct is members[0] of the mesh.
    return Construct(
        "parent_entry_mesh",
        nodes=[origin, entry_sub | Portal(to=["closer"], max_hops=8), closer],
    )


def _handoffs(result) -> list[Handoff]:
    if isinstance(result, Handoff):
        return [result]
    if isinstance(result, dict):
        return [v for v in result.values() if isinstance(v, Handoff)]
    return []


def _final(result) -> Handoff:
    hs = _handoffs(result)
    assert hs, f"parent mesh produced no Handoff payload: {result!r}"
    resolved = [h for h in hs if h.resolution is not None]
    return max(resolved or hs, key=lambda h: len(h.trail))


# =============================================================================
# SITE 2 — the declared Portal state fields are keyed off the CONSTRUCT entry.
# Isolates state.py: assert field names MATCH _wiring.py's entry_field
# (field_name_for(entry.name)) — NOT the first Node peer.
# =============================================================================
def test_portal_state_fields_keyed_off_construct_entry_not_peer():
    parent = _make_parent()
    state_model = compile_state_model(parent)
    fields = set(state_model.model_fields)

    entry_field = field_name_for("entry_sub")  # _wiring.py's real entry_field
    peer_field = field_name_for("closer")

    # The channel + hop counter must be keyed off the Construct entry ...
    assert StateKeys.handoff_payload(entry_field) in fields, (
        f"entry-keyed handoff payload field missing; a Construct entry was "
        f"excluded from portal_members. fields={sorted(fields)}"
    )
    assert StateKeys.handoff_hops(entry_field) in fields, sorted(fields)

    # ... and NOT off the first Node peer (the silent-divergence failure mode).
    assert StateKeys.handoff_payload(peer_field) not in fields, (
        "handoff payload was mis-keyed off the Node peer 'closer' — the "
        "Construct entry was excluded from the state-field builder"
    )
    assert StateKeys.handoff_hops(peer_field) not in fields, sorted(fields)


# =============================================================================
# SITE 3 (IR) — the Node peer's handoff_channel points at the CONSTRUCT entry's
# channel, not its own. Directly pins _ir_normalize.py's collector.
# =============================================================================
def test_node_peer_handoff_channel_keyed_off_construct_entry():
    parent = _make_parent()
    closer = next(n for n in parent.nodes if getattr(n, "name", None) == "closer")

    expected = StateKeys.handoff_payload(field_name_for("entry_sub"))
    wrong = StateKeys.handoff_payload(field_name_for("closer"))

    assert closer.handoff_channel == expected, (
        f"Node peer 'closer' stamped with the WRONG handoff_channel "
        f"{closer.handoff_channel!r}; expected the Construct entry's channel "
        f"{expected!r}. _ir_normalize.py excluded the Construct entry."
    )
    assert closer.handoff_channel != wrong, closer.handoff_channel


# =============================================================================
# SITE 1 — compile() must not misroute the Portal-Construct entry to
# _add_subgraph's unconditional CompileError; every participant must wire in.
# =============================================================================
def test_construct_entry_mesh_compiles_and_wires_all_participants():
    parent = _make_parent()
    graph = compile(parent)  # must not raise CompileError
    compiled_nodes = set(graph.get_graph().nodes)
    for participant in ("origin", "entry_sub", "closer"):
        assert participant in compiled_nodes, (
            f"mesh participant {participant!r} not wired into the compiled "
            f"Construct-entry mesh; compiled nodes: {sorted(compiled_nodes)}"
        )


# =============================================================================
# SITE 3 (RUNTIME) — the routed payload actually reaches the Node peer. A
# compile-only test cannot see the handoff_channel mis-stamp; only a run can.
# =============================================================================
def test_routed_payload_reaches_node_peer_from_construct_entry_sync():
    parent = compile(_make_parent())
    result = run(parent, input={"subject": "please escalate this login crash"})

    final = _final(result)
    assert "closer" in final.trail, (
        f"escape from the Construct entry did not reach the Node peer 'closer'; "
        f"handoff resolved to None (wrong channel). trail={final.trail}"
    )
    assert final.resolution == "resolved by parent closer", (
        f"silent seam / wrong resolver: {final.resolution!r}"
    )
    assert "sub_intake" in final.trail and "sub_decide" in final.trail, final.trail
    # site-7 proxy: closer ran on the routed subject, not a mis-picked payload.
    assert final.subject == "please escalate this login crash", (
        f"wrong payload fed across the boundary: {final.subject!r}"
    )


def test_entry_sub_local_finish_exits_cleanly_sync():
    parent = compile(_make_parent())
    result = run(parent, input={"subject": "a simple refund, handle locally"})

    final = _final(result)
    assert final.resolution == "resolved inside entry sub", (
        f"local finish from the Construct entry broken: {final.resolution!r}"
    )
    assert "closer" not in final.trail, "unexpected escape to the Node peer"


def test_routed_payload_reaches_node_peer_from_construct_entry_async():
    parent = compile(_make_parent())
    result = asyncio.run(arun(parent, input={"subject": "please escalate this login crash"}))

    final = _final(result)
    assert "closer" in final.trail, f"async escape did not reach 'closer': {final.trail}"
    assert final.resolution == "resolved by parent closer", final.resolution
    assert "sub_decide" in final.trail, final.trail
