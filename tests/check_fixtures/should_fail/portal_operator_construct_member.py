# CHECK_ERROR: sub-construct
# Portal+Operator's approval-node splice (neograph-kdr1u) is implemented for
# ATOMIC members (scripted/think/raw) only -- a sub-Construct mesh member
# carrying an Operator gate is narrowed-rejected, never silently accepted
# wider than the splice actually covers.
from pydantic import BaseModel

from neograph import Construct, Node, Operator, Portal
from tests.fakes import register_condition, register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


register_scripted("cf_construct_member_inner", lambda i, c: Handoff(goto="__end__"))
register_scripted("cf_construct_member_billing", lambda i, c: Handoff(goto="__end__"))
register_condition("cf_construct_member_gate", lambda d: None)

inner = Construct(
    "triage-sub",
    input=Handoff,
    output=Handoff,
    nodes=[Node.scripted("inner", fn="cf_construct_member_inner", inputs=Handoff, outputs=Handoff)],
) | Portal(to=["billing"]) | Operator(when="cf_construct_member_gate")
billing = Node.scripted(
    "billing", fn="cf_construct_member_billing", inputs={"handoff": Handoff}, outputs=Handoff
) | Portal(to=["triage-sub"])

pipeline = Construct("portal-operator-construct-member", nodes=[inner, billing])
