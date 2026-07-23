"""Regression (TDD red) for neograph-do0d9 — cross-subconstruct Portal handoff.

do0d9 makes an ACTUAL ``Construct`` object a first-class Portal mesh member: a
parent Construct carries a Portal mesh; ONE peer is a real sub-construct sitting
in the ``Portal`` member list; a routing decision made INSIDE the sub-construct
reaches a DIFFERENT parent-level peer, routed by ordinary parent Portal
machinery (``Command(goto)``), preserving isolated-invoke state hygiene.

Authoritative spec: docs/design/do0d9-parent-scoped-bubbleup-2026-07-23.md
(Core Invariant; §3 mechanism; §3.1 the seven implementation sites; §4 Q1-Q7;
§5 worked trace; §8 finding 5).

Why these tests FAIL today (the RIGHT red reason): a bare ``Construct`` in a
``Portal`` member list is REJECTED at assembly by ``_check_portal_mesh``
(``_validation_portal.py:89`` — "Portal mesh member '<name>' is a Construct").
The existing spike (docs/design/spikes/do0d9_bubbleup_spike.py) does NOT exercise
this shape — its ``worker`` is a plain ``@node`` whose BODY hand-invokes the
sub-construct, which already works. These tests put an ACTUAL ``Construct`` into
the member list end-to-end, exercising the real Node-typed sites (site 2
admission, site 4 Construct-aware wiring, and — the acceptance-critical one —
site 7 deterministic channel-sourced boundary input).

NOTE (site-7 honesty): the fully-adversarial multi-live-payload determinism
proof (``test_site7_boundary_reads_routed_channel_payload_not_decoy``) pins the
CORRECT observable outcome, but it cannot be *run* today (the Construct-member
path is rejected at assembly), so it fails at ``Construct(...)`` build like the
others. Post-do0d9 it becomes the deterministic-read (site 7) proof: only the
routed channel payload (not a stale same-typed decoy) may feed the boundary.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel

from neograph import (
    HANDOFF_END,
    Construct,
    FromInput,
    Node,
    Portal,
    arun,
    compile,
    construct_from_functions,
    node,
    run,
)


# ── the uniform mesh payload (also the sub-construct's declared output) ───────
class Handoff(BaseModel, frozen=True):
    goto: str  # a parent peer name, or HANDOFF_END
    subject: str
    trail: list[str] = []
    resolution: str | None = None


class SubDecision(BaseModel, frozen=True):
    """The sub-construct's own internal working type."""

    subject: str
    kind: Literal["local", "escalate"]
    trail: list[str]


# ── the sub-construct's internal members ──────────────────────────────────────
# ``ticket: Handoff`` is a boundary PORT param (its type == the sub-construct's
# ``input=Handoff``), so it is fed the routed handoff at the boundary — NOT a
# peer @node named "ticket".
def _make_sub_from_functions() -> Construct:
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
                resolution="resolved inside sub",
            )
        # ESCAPE: name a PARENT-level peer, not a sub-local node.
        return Handoff(goto="specialist", subject=sub_intake.subject, trail=trail)

    return construct_from_functions(
        "resolver_sub", [sub_intake, sub_decide], input=Handoff, output=Handoff
    )


def _make_sub_declarative() -> Construct:
    """Same sub-construct via the declarative Construct(input=, output=, nodes=[...])."""
    from neograph._state_keys import StateKeys
    from tests.fakes import register_scripted

    def _fn_intake(input_data, _config):
        ticket = input_data[StateKeys.SUBGRAPH_INPUT]  # the routed boundary payload
        kind = "escalate" if "escalate" in ticket.subject.lower() else "local"
        return SubDecision(subject=ticket.subject, kind=kind, trail=[*ticket.trail, "sub_intake"])

    def _fn_decide(input_data, _config):
        d = input_data["sub_intake"]
        trail = [*d.trail, "sub_decide"]
        if d.kind == "local":
            return Handoff(goto=HANDOFF_END, subject=d.subject, trail=trail, resolution="resolved inside sub")
        return Handoff(goto="specialist", subject=d.subject, trail=trail)

    register_scripted("_cx_fn_intake", _fn_intake)
    register_scripted("_cx_fn_decide", _fn_decide)
    return Construct(
        "resolver_sub",
        input=Handoff,
        output=Handoff,
        nodes=[
            Node.scripted("sub_intake", fn="_cx_fn_intake", inputs={StateKeys.SUBGRAPH_INPUT: Handoff}, outputs=SubDecision),
            Node.scripted("sub_decide", fn="_cx_fn_decide", inputs={"sub_intake": SubDecision}, outputs=Handoff),
        ],
    )


# ── the parent mesh with the sub-construct as an ACTUAL Portal member ──────────
def _make_parent(sub: Construct, *, async_mode: bool = False) -> Construct:
    @node(outputs=Handoff, portal=["resolver_sub"], max_hops=8)
    def dispatcher(handoff: Handoff, subject: Annotated[str, FromInput] = "") -> Handoff:
        if handoff is None:  # first activation: seed from FromInput, route into the sub-construct
            return Handoff(goto="resolver_sub", subject=subject, trail=["dispatcher"])
        return Handoff(goto="resolver_sub", subject=handoff.subject, trail=[*handoff.trail, "dispatcher"])

    @node(outputs=Handoff, portal=[])
    def specialist(handoff: Handoff) -> Handoff:
        return Handoff(
            goto=HANDOFF_END,
            subject=handoff.subject,
            trail=[*handoff.trail, "specialist"],
            resolution="resolved by parent specialist",
        )

    @node(outputs=Handoff, portal=[])
    def closer(handoff: Handoff) -> Handoff:
        return Handoff(
            goto=HANDOFF_END,
            subject=handoff.subject,
            trail=[*handoff.trail, "closer"],
            resolution="closed",
        )

    # THE ACCEPTANCE-CRITICAL SHAPE: a bare Construct in the Portal member list.
    return Construct(
        "parent_mesh",
        nodes=[dispatcher, sub | Portal(to=["specialist", "closer"]), specialist, closer],
    )


# ── result helpers (robust to final-state field naming) ───────────────────────
def _handoffs(result) -> list[Handoff]:
    if isinstance(result, Handoff):
        return [result]
    if isinstance(result, dict):
        return [v for v in result.values() if isinstance(v, Handoff)]
    return []


def _final(result) -> Handoff:
    """The terminal resolved payload — the one carrying a resolution, deepest trail."""
    hs = _handoffs(result)
    assert hs, f"parent mesh produced no Handoff payload: {result!r}"
    resolved = [h for h in hs if h.resolution is not None]
    return max(resolved or hs, key=lambda h: len(h.trail))


def _any_trail_contains(result, name: str) -> bool:
    return any(name in h.trail for h in _handoffs(result))


# =============================================================================
# 1 — HAPPY PATH (sync): inner sub-construct decision reaches a parent peer.
# =============================================================================
def test_inner_subconstruct_decision_reaches_parent_peer_when_escalating_sync():
    """A decision made INSIDE the sub-construct (sub_decide -> goto='specialist')
    routes to the parent peer 'specialist' via ordinary parent Portal machinery;
    'closer' is never hit and 'specialist' runs on the CORRECT routed payload."""
    parent = compile(_make_parent(_make_sub_from_functions()))
    result = run(parent, input={"subject": "please escalate this login crash"})

    final = _final(result)
    # escape crossed the boundary to the named parent peer
    assert "specialist" in final.trail, f"escape did not reach parent peer 'specialist': {final.trail}"
    assert final.resolution == "resolved by parent specialist", f"silent seam / wrong resolver: {final.resolution!r}"
    # the inner members genuinely ran (the decision was made INSIDE the child)
    assert "sub_intake" in final.trail and "sub_decide" in final.trail, final.trail
    # not mis-routed to the wrong parent peer
    assert not _any_trail_contains(result, "closer"), "mis-routed to 'closer'"
    # site-7 proxy: specialist ran on the routed subject, not a mis-picked payload
    assert final.subject == "please escalate this login crash", f"wrong payload fed across boundary: {final.subject!r}"


def test_clean_local_finish_exits_mesh_with_no_escape_sync():
    """When the sub-construct resolves locally (goto=HANDOFF_END), the mesh exits
    cleanly — no parent peer is activated, no escape leaks."""
    parent = compile(_make_parent(_make_sub_from_functions()))
    result = run(parent, input={"subject": "a simple refund, handle locally"})

    final = _final(result)
    assert final.resolution == "resolved inside sub", f"local finish broken: {final.resolution!r}"
    assert not _any_trail_contains(result, "specialist"), "unexpected escape to parent peer"
    assert not _any_trail_contains(result, "closer"), "unexpected escape to parent peer"


# =============================================================================
# 2 — ASYNC PARITY (design §4 Q7 / Phase-1 H2).
# =============================================================================
def test_inner_subconstruct_decision_reaches_parent_peer_when_escalating_async():
    """The same cross-boundary escape under ``arun`` — async selection must
    propagate into the child sub-construct's isolated invoke."""
    parent = compile(_make_parent(_make_sub_from_functions(), async_mode=True))
    result = asyncio.run(arun(parent, input={"subject": "please escalate this login crash"}))

    final = _final(result)
    assert "specialist" in final.trail, f"async escape did not reach 'specialist': {final.trail}"
    assert final.resolution == "resolved by parent specialist", final.resolution
    assert "sub_decide" in final.trail, final.trail
    assert not _any_trail_contains(result, "closer"), "async mis-route to 'closer'"


# =============================================================================
# 3 — THREE-SURFACE PARITY (AGENTS.md; ForwardConstruct EXEMPT per D-FORWARD-EXEMPT).
# =============================================================================
@pytest.mark.parametrize(
    "make_sub",
    [
        pytest.param(_make_sub_from_functions, id="construct_from_functions"),
        pytest.param(_make_sub_declarative, id="declarative_Construct"),
    ],
)
def test_construct_as_portal_member_assembles_and_compiles_across_surfaces(make_sub):
    """A Construct-as-Portal-member mesh must ASSEMBLE and COMPILE on every
    in-scope surface: the sub built via construct_from_functions(@node) OR the
    declarative Construct(input=,output=,nodes=[...]) form, attached via the
    programmatic ``sub | Portal(to=[...])`` pipe and placed declaratively."""
    parent = _make_parent(make_sub())  # Construct(...) assembly must not raise
    graph = compile(parent)  # compile must not raise
    # Stronger than a bare `is not None`: the mesh must actually wire every
    # participant into the compiled LangGraph — the three sibling Node peers AND
    # the sub-construct boundary node `resolver_sub` (a Construct member's parent
    # peer maps to its boundary node name). compile() also raises on any invalid
    # mesh, so reaching here already proves _check_portal_mesh admitted the
    # Construct member.
    compiled_nodes = set(graph.get_graph().nodes)
    for participant in ("dispatcher", "resolver_sub", "specialist", "closer"):
        assert participant in compiled_nodes, (
            f"mesh participant {participant!r} was not wired into the compiled "
            f"cross-subconstruct mesh; compiled nodes: {sorted(compiled_nodes)}"
        )


# =============================================================================
# 4 — ASSEMBLY VALIDATION: an invalid cross-boundary target is unrepresentable.
# =============================================================================
def test_subconstruct_member_naming_nonexistent_parent_peer_raises_at_assembly():
    """A Construct member whose Portal names a NON-member/non-existent parent
    target must fail LOUD at Construct(...) assembly (ConstructError), before any
    run — the same 'invalid target unrepresentable' guarantee sibling Nodes get."""
    from neograph.errors import ConstructError

    sub = _make_sub_from_functions()

    @node(outputs=Handoff, portal=["resolver_sub"], max_hops=8)
    def dispatcher(handoff: Handoff, subject: Annotated[str, FromInput] = "") -> Handoff:
        if handoff is None:
            return Handoff(goto="resolver_sub", subject=subject, trail=["dispatcher"])
        return Handoff(goto="resolver_sub", subject=handoff.subject, trail=handoff.trail)

    @node(outputs=Handoff, portal=[])
    def specialist(handoff: Handoff) -> Handoff:
        return Handoff(goto=HANDOFF_END, subject=handoff.subject, trail=[*handoff.trail, "specialist"])

    with pytest.raises(ConstructError, match=r"does not exist|unknown peer|ghost_peer"):
        Construct(
            "parent_mesh_bad",
            # resolver_sub names 'ghost_peer' — not a sibling of the parent mesh.
            nodes=[dispatcher, sub | Portal(to=["ghost_peer"]), specialist],
        )


# =============================================================================
# 5 — SITE 7: deterministic channel-sourced boundary input (§3.1 site 7 / §8 f5).
# =============================================================================
def test_site7_boundary_reads_routed_channel_payload_not_decoy():
    """The boundary handoff-IN for a Construct mesh member MUST read the routed
    parent-channel payload, NOT a blind reverse-type-scan that could pick a
    stale same-typed decoy left in another member's field.

    Topology forces the ambiguity: ``dispatcher`` leaves a decoy Handoff
    (subject 'DECOY-resolve-locally') in its own field, then routes to
    ``prefill``, which OVERRIDES the subject to 'escalate NOW' and routes to the
    sub-construct. When the boundary runs, TWO differently-subjected Handoffs are
    live in parent state; only the routed channel payload ('escalate NOW') must
    feed the sub-construct. A blind scan that grabs the decoy would send the sub
    'DECOY-resolve-locally' -> it resolves LOCALLY -> 'specialist' never runs: a
    SILENT MIS-ROUTE the North Star forbids. Asserting 'specialist' resolved the
    'escalate NOW' payload pins the deterministic read."""
    sub = _make_sub_from_functions()

    @node(outputs=Handoff, portal=["prefill"], max_hops=8)
    def dispatcher(handoff: Handoff) -> Handoff:
        # leaves a DECOY in the 'dispatcher' field, routes onward to prefill
        return Handoff(goto="prefill", subject="DECOY-resolve-locally", trail=["dispatcher"])

    @node(outputs=Handoff, portal=["resolver_sub"])
    def prefill(handoff: Handoff) -> Handoff:
        # the payload that MUST feed the sub-construct across the boundary
        return Handoff(goto="resolver_sub", subject="escalate NOW", trail=[*handoff.trail, "prefill"])

    @node(outputs=Handoff, portal=[])
    def specialist(handoff: Handoff) -> Handoff:
        return Handoff(
            goto=HANDOFF_END,
            subject=handoff.subject,
            trail=[*handoff.trail, "specialist"],
            resolution="resolved by parent specialist",
        )

    parent = compile(
        Construct(
            "parent_mesh_site7",
            nodes=[dispatcher, prefill, sub | Portal(to=["specialist"]), specialist],
        )
    )
    result = run(parent, input={})

    final = _final(result)
    assert "specialist" in final.trail, (
        f"site-7 mis-route: boundary did not feed the routed channel payload "
        f"('escalate NOW'); the sub resolved locally instead. trail={final.trail}"
    )
    # the routed subject (not the decoy) must be what reached specialist
    assert final.subject == "escalate NOW", f"boundary fed the WRONG (decoy) payload: {final.subject!r}"
    assert final.resolution == "resolved by parent specialist", final.resolution
