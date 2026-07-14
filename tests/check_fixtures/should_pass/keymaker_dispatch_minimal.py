# A legal minimal Keymaker DISPATCH node (route="decide", design §4.2): it
# ASSEMBLES cleanly (M1 — _validation_keymaker excludes route="decide" from the
# mesh-member set, so a lone dispatch node is NOT treated as a degenerate mesh) AND
# compiles cleanly now that T6 (neograph-f27xo) lowering has landed
# (_add_keymaker_dispatch wires it as a plain LINEAR node, NO Command). The
# check-fixtures harness compiles every should_pass Construct, so this pins the
# dispatch lowering end to end. The emitted spec/input dicts are inert here (the
# node body only runs at run() time — compile just builds the graph), and
# ``output`` is a class, so no runtime type registration is needed to compile.
from pydantic import BaseModel

from neograph import Construct, Keymaker, Node
from tests.fakes import register_scripted


class Emitted(BaseModel, frozen=True):
    spec: dict
    dispatch_input: dict


class Summary(BaseModel, frozen=True):
    text: str


register_scripted("km_dispatch", lambda i, c: Emitted(spec={}, dispatch_input={}))

pipeline = Construct(
    "dispatch",
    nodes=[
        Node.scripted("planner", fn="km_dispatch", outputs=Emitted)
        | Keymaker(route="decide", spec_field="spec", input_field="dispatch_input", output=Summary),
    ],
)
