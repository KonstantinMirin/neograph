# Portal (PEER mode) + Operator is now LEGAL (neograph-kdr1u, D4 lift): a
# human-approval gate spliced onto the dynamic Command(goto) path. Was a
# should_fail fixture pre-kdr1u ("CHECK_ERROR: [Oo]perator") -- flipped here
# per the design's step 3 instruction, not deleted (pins the ban-lift).
from pydantic import BaseModel

from neograph import Construct, Node, Operator, Portal
from tests.fakes import register_condition, register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


register_scripted("cf_km_triage", lambda i, c: Handoff(goto="billing"))
register_scripted("cf_km_billing", lambda i, c: Handoff(goto="__end__"))
register_condition("cf_km_gate", lambda d: None)

triage = (
    Node.scripted("triage", fn="cf_km_triage", outputs=Handoff)
    | Portal(to=["billing"])
    | Operator(when="cf_km_gate")
)
billing = Node.scripted("billing", fn="cf_km_billing", inputs={"handoff": Handoff}, outputs=Handoff) | Portal(
    to=["triage"]
)

pipeline = Construct("portal-operator-combo", nodes=[triage, billing])
