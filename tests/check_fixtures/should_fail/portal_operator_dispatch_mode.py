# CHECK_ERROR: dispatch mode
# Portal DISPATCH mode (route="decide") + Operator is still illegal
# (neograph-kdr1u): dispatch has no "peer" to approve a handoff TO, and no
# mesh-exit analog for a rejection to route to. Only PEER mode (to=[...])
# combines with Operator.
from pydantic import BaseModel

from neograph import Node, Operator, Portal
from tests.fakes import register_condition, register_scripted


class Emitted(BaseModel, frozen=True):
    spec: dict
    dispatch_input: dict


class Summary(BaseModel, frozen=True):
    text: str


register_scripted("cf_dispatch_op_fn", lambda i, c: Emitted(spec={}, dispatch_input={}))
register_condition("cf_dispatch_op_gate", lambda d: None)

pipeline_node = Node.scripted("planner", fn="cf_dispatch_op_fn", outputs=Emitted) | Portal(
    route="decide", spec_field="spec", input_field="dispatch_input", output=Summary, max_depth=5
) | Operator(when="cf_dispatch_op_gate")
