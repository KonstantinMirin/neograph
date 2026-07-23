# A Construct as a first-class Portal mesh member (neograph-do0d9). The parent
# mesh entry routes to a sub-construct peer whose declared output is the uniform
# mesh payload; the sub-construct routes on to `closer`. This ASSEMBLES and
# COMPILES cleanly once do0d9's Construct-member wiring lands (site 2 admission +
# site 4 Construct-aware dispatch). Today it is REJECTED at assembly
# ("Portal mesh member '...' is a Construct") — the TDD-red state.
from pydantic import BaseModel

from neograph import HANDOFF_END, Construct, Node, Portal
from tests.fakes import register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


register_scripted("_cxf_resolve", lambda i, c: Handoff(goto="closer"))
register_scripted("_cxf_entry", lambda i, c: Handoff(goto="resolver_sub"))
register_scripted("_cxf_close", lambda i, c: Handoff(goto=HANDOFF_END))

resolver_sub = Construct(
    "resolver_sub",
    input=Handoff,
    output=Handoff,
    nodes=[Node.scripted("resolve", fn="_cxf_resolve", outputs=Handoff)],
)

pipeline = Construct(
    "parent_mesh",
    nodes=[
        Node.scripted("entry", fn="_cxf_entry", outputs=Handoff) | Portal(to=["resolver_sub"], max_hops=6),
        resolver_sub | Portal(to=["closer"]),
        Node.scripted("closer", fn="_cxf_close", inputs={"handoff": Handoff}, outputs=Handoff) | Portal(to=[]),
    ],
)
