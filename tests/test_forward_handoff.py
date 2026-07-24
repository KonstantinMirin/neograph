"""Regression test for neograph-a37vk: ForwardConstruct self.handoff(...) builder.

Pins the Core Invariant from the refined design (bd show a37vk): self.handoff's
emitted IR must be BYTE-FOR-BYTE identical to the declarative mesh IR --
``Construct(nodes=[member | Portal(...), ...])`` with no extra nesting level --
and must satisfy the entry-only kwargs rule enforced by
``_validation_portal._check_portal_mesh`` (max_hops/on_exhaust settable ONLY on
the entry member).

TDD red: ``self.handoff`` does not exist yet on ``_ForwardSelf`` (forward.py).
This test is expected to FAIL with an AttributeError until neograph-a37vk's
implementation lands.

Mirrors ``examples/28_portal_swarm.py::demo_surface_parity`` (the declarative
mesh IR self.handoff must reproduce) and
``examples/27_forward_agent_wiring.py`` (self.interrupt/self.ensemble builder
shape) for the ForwardConstruct-side wiring pattern.
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import HANDOFF_END, Construct, ForwardConstruct, Node, Portal, compile, run


class Handoff(BaseModel, frozen=True):
    goto: str
    subject: str
    body: str
    resolution: str | None = None
    hops: int = 0


def fn_triage(input_data, _config):
    incoming = input_data.get("handoff") if isinstance(input_data, dict) else None
    if incoming is None:
        return Handoff(goto="billing", subject="Double charge", body="charged twice", hops=1)
    return Handoff(goto=HANDOFF_END, subject=incoming.subject, body=incoming.body, hops=incoming.hops + 1)


def fn_billing(input_data, _config):
    incoming = input_data["handoff"]
    return Handoff(
        goto=HANDOFF_END,
        subject=incoming.subject,
        body=incoming.body,
        resolution="Refund issued.",
        hops=incoming.hops + 1,
    )


class HandoffSwarm(ForwardConstruct):
    """Two-member mesh built with self.handoff -- the imperative twin of
    examples/28_portal_swarm.py::demo_surface_parity's declarative mesh."""

    triage = Node.scripted("triage", fn="fn_triage", outputs=Handoff)
    billing = Node.scripted("billing", fn="fn_billing", inputs={"handoff": Handoff}, outputs=Handoff)

    def forward(self, _seed):
        # Whole-graph mesh (D-FORWARD-EXEMPT): no proxy in/out, matching the
        # committed signature `self.handoff(members=..., to=..., max_hops=...,
        # on_exhaust=..., entry=None)`.
        self.handoff(
            members=[self.triage, self.billing],
            to={"triage": ["billing"], "billing": []},
            max_hops=6,
            on_exhaust="error",
        )


def _declarative_reference_mesh() -> Construct:
    """The declarative twin IR self.handoff's output must match byte-for-byte."""
    triage = Node.scripted("triage", fn="fn_triage", outputs=Handoff) | Portal(to=["billing"], max_hops=6)
    billing = Node.scripted("billing", fn="fn_billing", inputs={"handoff": Handoff}, outputs=Handoff) | Portal(
        to=[]
    )
    return Construct("swarm-declarative", nodes=[triage, billing])


def test_self_handoff_runs_end_to_end_matching_declarative_mesh():
    """self.handoff(...) must compile and RUN identically to the declarative
    Node | Portal(...) mesh: triage routes to billing, billing resolves and
    exits the mesh via HANDOFF_END."""
    pipeline = HandoffSwarm()
    graph = compile(pipeline, scripted={"fn_triage": fn_triage, "fn_billing": fn_billing})
    result = run(graph, input={})

    resolved = result["billing"]
    assert resolved.goto == HANDOFF_END
    assert resolved.resolution == "Refund issued."
    assert resolved.hops == 2

    # Round-trip against the declarative reference mesh -- same inputs, same
    # observable output through the same public run() surface.
    ref_graph = compile(_declarative_reference_mesh(), scripted={"fn_triage": fn_triage, "fn_billing": fn_billing})
    ref_result = run(ref_graph, input={})
    assert ref_result["billing"].model_dump() == resolved.model_dump()


def test_self_handoff_emits_flat_mesh_ir_with_entry_only_portal_kwargs():
    """Core Invariant: self.handoff's IR is FLAT -- pipeline.nodes holds the two
    bare mesh members directly (no wrapping sub-Construct), exactly like the
    declarative mesh's Construct(nodes=[triage, billing]). Also pins the
    entry-only kwargs rule: only the entry member (triage, recorded first) may
    carry max_hops/on_exhaust in its Portal's model_fields_set; the non-entry
    member (billing) must carry neither, or assembly would raise per
    _validation_portal._check_portal_mesh."""
    pipeline = HandoffSwarm()

    # No nesting: exactly the two members, not a Construct-wrapping-a-Construct.
    assert len(pipeline.nodes) == 2
    names = [n.name for n in pipeline.nodes]
    assert names == ["triage", "billing"]
    assert all(isinstance(n, Node) for n in pipeline.nodes)

    triage_node, billing_node = pipeline.nodes
    triage_portal = triage_node.get_modifier(Portal)
    billing_portal = billing_node.get_modifier(Portal)
    assert triage_portal is not None
    assert billing_portal is not None

    # Entry member: max_hops/on_exhaust are set explicitly.
    assert "max_hops" in triage_portal.model_fields_set
    assert "on_exhaust" in triage_portal.model_fields_set
    assert triage_portal.max_hops == 6
    assert triage_portal.to == ["billing"]

    # Non-entry member: neither knob may be in model_fields_set, or the
    # entry-only validator (_check_portal_mesh) would reject the mesh.
    assert "max_hops" not in billing_portal.model_fields_set
    assert "on_exhaust" not in billing_portal.model_fields_set
    assert billing_portal.to == []
