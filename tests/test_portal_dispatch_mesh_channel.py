"""A dispatch-mode Portal must not hijack a peer mesh's handoff channel (dgbqv.12).

``route="decide"`` makes a Portal a STANDALONE linear node, not a mesh member. Two of
the three places that answer "which items are mesh members" agree on that -- the
validator (``_validation_portal``) and the wiring layer (``_wiring``) both skip
``PortalMemberClass.DISPATCH``. ``_ir_normalize`` did not: it collected on
``modifier_set.portal is not None``, so a dispatch node sitting before a mesh became
``members[0]`` and the mesh channel got keyed off a non-member.

That is a WRITE/READ split, which is why nothing raised:
  * the runtime WRITES the payload to ``MeshContext.channel_key`` (from the real entry)
  * each member READS ``node.handoff_channel`` (``_input_shape``), stamped by the normalizer
so the members read a state key nothing ever writes and see ``None``.
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import Construct, Node
from neograph._ir_normalize import normalize_ir
from neograph._portal import _group_portal_members
from neograph._portal_member import PortalMemberClass, portal_member_class
from neograph._portal_route import MeshContext
from neograph.modifiers import Portal
from tests.fakes import register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


class Spec(BaseModel, frozen=True):
    spec: str


register_scripted("dgbqv12_fn", lambda i, c: Handoff(goto="__end__"))


def _mesh(*, with_dispatch: bool) -> Construct:
    triage = Node.scripted("triage", fn="dgbqv12_fn", inputs=Handoff, outputs=Handoff) | Portal(
        to=["billing"], max_hops=5
    )
    billing = Node.scripted("billing", fn="dgbqv12_fn", inputs=Handoff, outputs=Handoff) | Portal(
        to=["triage"]
    )
    items: list = [triage, billing]
    if with_dispatch:
        planner = Node.scripted("planner", fn="dgbqv12_fn", inputs=Spec, outputs=Handoff) | Portal(
            route="decide",
            spec_field="spec",
            input_field="dispatch_input",
            output=Handoff,
            max_depth=2,
        )
        items.insert(0, planner)
    return Construct("dgbqv12", nodes=items)


def _runtime_write_key(construct: Construct) -> str:
    """The key the WIRING layer writes the handoff payload to.

    Mirrors _wiring's own filter, which skips DISPATCH -- that is precisely the
    filter _ir_normalize was missing.
    """
    members = [
        item
        for item in construct.nodes
        if portal_member_class(item) not in (None, PortalMemberClass.DISPATCH)
    ]
    return MeshContext.build(_group_portal_members(members)[None]).channel_key


class TestDispatchPortalDoesNotHijackTheMeshChannel:
    def test_mesh_channel_is_unchanged_by_a_preceding_dispatch_portal(self):
        """The control and the with-dispatch case must key the mesh identically --
        a standalone dispatch node is not a member and must not move the entry."""
        control = _mesh(with_dispatch=False)
        withdisp = _mesh(with_dispatch=True)
        normalize_ir(control)
        normalize_ir(withdisp)

        control_keys = {n.name: n.handoff_channel for n in control.nodes}
        withdisp_keys = {n.name: n.handoff_channel for n in withdisp.nodes}

        for member in ("triage", "billing"):
            assert withdisp_keys[member] == control_keys[member], (
                f"{member}'s handoff channel moved to {withdisp_keys[member]!r} because a "
                f"dispatch-mode Portal preceded the mesh; it must stay "
                f"{control_keys[member]!r} (dgbqv.12)"
            )

    def test_the_stamped_read_key_matches_the_runtime_write_key(self):
        """The bug's actual bite: members read a channel nothing writes."""
        construct = _mesh(with_dispatch=True)
        normalize_ir(construct)
        stamped = {n.name: n.handoff_channel for n in construct.nodes}

        write_key = _runtime_write_key(construct)
        assert stamped["triage"] == write_key, (
            f"members READ {stamped['triage']!r} but the runtime WRITES {write_key!r} -- "
            "the handoff payload is delivered to a key nobody reads, so the member sees "
            "None and nothing raises (dgbqv.12)"
        )

    def test_a_dispatch_portal_is_not_given_a_mesh_channel_at_all(self):
        """A standalone dispatch node has no mesh to hand off through."""
        construct = _mesh(with_dispatch=True)
        normalize_ir(construct)
        planner = next(n for n in construct.nodes if n.name == "planner")
        assert planner.handoff_channel is None, (
            "a route='decide' Portal is a standalone linear node, not a mesh member, so it "
            f"must carry no mesh channel (got {planner.handoff_channel!r})"
        )
