# neograph-2j208 (review finding 4): a Construct-level Portal mesh member,
# declared entirely via YAML -- ConstructSpec.portal is materially different
# from NodeSpec.portal (_build_sub_construct auto-wires sequential inputs for
# any node lacking an explicit `inputs:`), so it needs its own coverage rather
# than inheriting NodeSpec's. The inner "resolve" node deliberately has NO
# explicit inputs -- sequential auto-wiring assigns it the sub-construct's own
# boundary input; "closer" (the top-level node AFTER the mesh member) is the
# one that must declare `inputs: {handoff: Handoff}` explicitly, since it
# consumes the mesh handoff channel, not a normal upstream producer.
from pydantic import BaseModel

from neograph import HANDOFF_END
from neograph.loader import load_spec
from neograph.spec_types import register_type
from tests.fakes import register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


register_type("Handoff", Handoff)
register_scripted("spec_cm_entry", lambda i, c: Handoff(goto="resolver_sub"))
register_scripted("spec_cm_resolve", lambda i, c: Handoff(goto="closer"))
register_scripted("spec_cm_close", lambda i, c: Handoff(goto=HANDOFF_END))

pipeline = load_spec(
    {
        "name": "spec-portal-construct-member",
        "nodes": [
            {
                "name": "entry",
                "mode": "scripted",
                "scripted_fn": "spec_cm_entry",
                "outputs": "Handoff",
                "portal": {"to": ["resolver_sub"], "max_hops": 6},
            },
            {
                "name": "resolve",
                "mode": "scripted",
                "scripted_fn": "spec_cm_resolve",
                "outputs": "Handoff",
            },
            {
                "name": "closer",
                "mode": "scripted",
                "scripted_fn": "spec_cm_close",
                "outputs": "Handoff",
                "inputs": {"handoff": "Handoff"},
                "portal": {"to": []},
            },
        ],
        "constructs": [
            {
                "name": "resolver_sub",
                "input": "Handoff",
                "output": "Handoff",
                "nodes": ["resolve"],
                "portal": {"to": ["closer"]},
            }
        ],
        "pipeline": {"nodes": ["entry", "resolver_sub", "closer"]},
    }
)
