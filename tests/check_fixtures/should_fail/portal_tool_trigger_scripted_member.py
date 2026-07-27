# CHECK_ERROR: trigger="tool" requires an agent/act member
# Portal trigger="tool" (design portal-tool-triggered-handoff §4) requires a
# member with a ReAct tool-call turn to emit the synthesized transfer_to_<peer>
# tool call from. A scripted (atomic) member has no such turn, so
# Portal(to=[...], trigger="tool") on it is a ConstructError at assembly time --
# the same narrowed-rejection style as the dict-form / Operator-mode checks.
from pydantic import BaseModel

from neograph import Construct, Node, Portal
from tests.fakes import register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


register_scripted("cf_tooltrigger_scripted", lambda i, c: Handoff(goto="__end__"))

pipeline = Construct(
    "portal-tool-trigger-scripted",
    nodes=[
        Node.scripted("triage", fn="cf_tooltrigger_scripted", outputs=Handoff)
        | Portal(to=["billing"], trigger="tool", max_hops=6),
        Node.scripted("billing", fn="cf_tooltrigger_scripted", inputs={"handoff": Handoff}, outputs=Handoff)
        | Portal(to=["triage"]),
    ],
)
