# CHECK_ERROR: Cannot combine Each and Loop|Each and Loop
# Bypass the | operator by constructing ModifierSet directly.
# ModifierSet.model_post_init rejects illegal combos at construction time.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted
from neograph.modifiers import Each, Loop, ModifierSet


class Item(BaseModel, frozen=True):
    x: str


register_scripted("melp_proc", lambda i, c: Item(x="done"))

# Construct ModifierSet directly — model_post_init should reject this
bad_node = Node(
    name="proc",
    mode="scripted",
    scripted_fn="melp_proc",
    outputs=Item,
    modifier_set=ModifierSet(
        each=Each(over="source.items", key="x"),
        loop=Loop(when=lambda d: False, max_iterations=1),
    ),
)

pipeline = Construct("broken", nodes=[bad_node])
