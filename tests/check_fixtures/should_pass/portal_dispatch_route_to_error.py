# A legal Portal DISPATCH node with on_invalid='route_to_error' (neograph-4pr04):
# error_handler names a REAL sibling Node, so it ASSEMBLES and COMPILES cleanly
# (_add_portal_dispatch wires a Command(goto=...) success/error split with a
# synthetic pass-through exit node, mirroring the mesh's exit-node pattern).
from pydantic import BaseModel

from neograph import Construct, Node, Portal
from tests.fakes import register_scripted


class Emitted(BaseModel, frozen=True):
    spec: dict
    dispatch_input: dict


class Summary(BaseModel, frozen=True):
    text: str


class Final(BaseModel, frozen=True):
    text: str


register_scripted("km_dispatch_rte_pass", lambda i, c: Emitted(spec={}, dispatch_input={}))
register_scripted("km_handler_rte_pass", lambda i, c: Final(text="handled"))

pipeline = Construct(
    "dispatch-route-to-error-pass",
    nodes=[
        Node.scripted("planner", fn="km_dispatch_rte_pass", outputs=Emitted)
        | Portal(
            route="decide",
            spec_field="spec",
            input_field="dispatch_input",
            output=Summary,
            max_depth=5,
            on_invalid="route_to_error",
            error_handler="handler",
        ),
        Node.scripted("handler", fn="km_handler_rte_pass", outputs=Final),
    ],
)
