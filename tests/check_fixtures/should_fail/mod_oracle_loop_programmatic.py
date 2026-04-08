# CHECK_ERROR: Cannot combine Oracle and Loop|Oracle and Loop
# Bypass the | operator by constructing modifiers list directly.
# Belt-and-suspenders check in _validate_node_chain should catch this.
from neograph import Construct, Node
from neograph.modifiers import Oracle, Loop
from neograph.factory import register_scripted
from pydantic import BaseModel


class Draft(BaseModel, frozen=True):
    text: str


register_scripted("molp_proc", lambda i, c: Draft(text="ok"))
register_scripted("molp_merge", lambda variants, c: variants[0])

# Construct modifiers list directly, bypassing __or__ mutual exclusion
bad_node = Node(
    name="proc",
    mode="scripted",
    scripted_fn="molp_proc",
    outputs=Draft,
    modifiers=[
        Oracle(n=3, merge_fn="molp_merge"),
        Loop(when=lambda d: False, max_iterations=1),
    ],
)

pipeline = Construct("broken", nodes=[bad_node])
