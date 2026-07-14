# CHECK_ERROR: [Ll]oop
# Portal + Loop on the same node is forbidden (slot conflict, design §5.6).
from pydantic import BaseModel

from neograph import Loop, Node, Portal
from tests.fakes import register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


register_scripted("km_f", lambda i, c: Handoff(goto="__end__"))

pipeline_node = (
    Node.scripted("member", fn="km_f", outputs=Handoff)
    | Portal(to=["member"])
    | Loop(when=lambda d: False, max_iterations=2)
)
