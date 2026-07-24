# CHECK_ERROR: does_not_exist
# Portal on_invalid='route_to_error' with an error_handler naming NO sibling
# Node -- fails at Construct(...) assembly (neograph-4pr04, mirrors the mesh
# peer-existence check in _validation_portal.py).
from pydantic import BaseModel

from neograph import Construct, Node, Portal
from tests.fakes import register_scripted


class Emitted(BaseModel, frozen=True):
    spec: dict
    dispatch_input: dict


class Summary(BaseModel, frozen=True):
    text: str


register_scripted("km_dispatch_rte_fail", lambda i, c: Emitted(spec={}, dispatch_input={}))

pipeline = Construct(
    "dispatch-route-to-error-fail",
    nodes=[
        Node.scripted("planner", fn="km_dispatch_rte_fail", outputs=Emitted)
        | Portal(
            route="decide",
            spec_field="spec",
            input_field="dispatch_input",
            output=Summary,
            max_depth=5,
            on_invalid="route_to_error",
            error_handler="does_not_exist",
        ),
    ],
)
