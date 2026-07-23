# CHECK_ERROR: does not exist|unknown peer|ghost_peer
# A Construct Portal mesh member (neograph-do0d9) whose Portal names a peer that
# is NOT a sibling of the parent mesh — a cross-boundary handoff to a
# non-existent parent target must fail LOUD at assembly (ConstructError), the
# same "invalid target unrepresentable" guarantee sibling Nodes get.
from pydantic import BaseModel

from neograph import Construct, Node, Portal
from tests.fakes import register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


register_scripted("_cxfx_resolve", lambda i, c: Handoff(goto="ghost_peer"))
register_scripted("_cxfx_entry", lambda i, c: Handoff(goto="resolver_sub"))

resolver_sub = Construct(
    "resolver_sub",
    input=Handoff,
    output=Handoff,
    nodes=[Node.scripted("resolve", fn="_cxfx_resolve", outputs=Handoff)],
)

pipeline = Construct(
    "parent_mesh_bad",
    nodes=[
        Node.scripted("entry", fn="_cxfx_entry", outputs=Handoff) | Portal(to=["resolver_sub"], max_hops=6),
        # 'ghost_peer' is not a sibling of the parent mesh — unknown peer.
        resolver_sub | Portal(to=["ghost_peer"]),
    ],
)
