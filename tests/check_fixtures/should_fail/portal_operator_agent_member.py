# CHECK_ERROR: agent-mode
# Portal+Operator's approval-node splice (neograph-kdr1u) is implemented for
# ATOMIC members (scripted/think/raw) only -- an agent/act mesh member
# carrying an Operator gate is narrowed-rejected, never silently accepted
# wider than the splice actually covers.
from pydantic import BaseModel

from neograph import Construct, Node, Operator, Portal
from tests.fakes import register_condition, register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


register_scripted("cf_agent_member_billing", lambda i, c: Handoff(goto="__end__"))
register_condition("cf_agent_member_gate", lambda d: None)

triage = Node(
    name="triage",
    mode="agent",
    model="fast",
    prompt="rw/triage",
    outputs=Handoff,
) | Portal(to=["billing"]) | Operator(when="cf_agent_member_gate")
billing = Node.scripted("billing", fn="cf_agent_member_billing", inputs={"handoff": Handoff}, outputs=Handoff) | Portal(
    to=["triage"]
)

pipeline = Construct("portal-operator-agent-member", nodes=[triage, billing])
