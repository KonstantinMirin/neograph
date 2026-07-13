# CHECK_ERROR: [Oo]perator
# Keymaker + Operator is ILLEGAL in v1 (D-NO-OPERATOR-COMBO, design §5.6).
from pydantic import BaseModel

from neograph import Keymaker, Node, Operator
from tests.fakes import register_condition, register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


register_scripted("km_f", lambda i, c: Handoff(goto="__end__"))
register_condition("km_gate", lambda d: True)

pipeline_node = (
    Node.scripted("member", fn="km_f", outputs=Handoff)
    | Keymaker(peers=["member"])
    | Operator(when="km_gate")
)
