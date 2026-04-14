# CHECK_ERROR: Cannot combine Oracle and Loop|Oracle and Loop
# Bypass the | operator by constructing ModifierSet directly.
# ModifierSet.model_post_init rejects illegal combos at construction time.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted
from neograph.modifiers import Loop, ModifierSet, Oracle


class Draft(BaseModel, frozen=True):
    text: str


register_scripted("molp_proc", lambda i, c: Draft(text="ok"))
register_scripted("molp_merge", lambda variants, c: variants[0])

# Construct ModifierSet directly — model_post_init should reject this
bad_node = Node(
    name="proc",
    mode="scripted",
    scripted_fn="molp_proc",
    outputs=Draft,
    modifier_set=ModifierSet(
        oracle=Oracle(n=3, merge_fn="molp_merge"),
        loop=Loop(when=lambda d: False, max_iterations=1),
    ),
)

pipeline = Construct("broken", nodes=[bad_node])
