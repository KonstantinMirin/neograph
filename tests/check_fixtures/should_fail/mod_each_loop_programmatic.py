# CHECK_ERROR: Cannot combine Each and Loop|Each and Loop
# Bypass the | operator by constructing modifiers list directly.
# Belt-and-suspenders check in _validate_node_chain should catch this.
from neograph import Construct, Node
from neograph.modifiers import Each, Loop
from neograph.factory import register_scripted
from pydantic import BaseModel


class Item(BaseModel, frozen=True):
    x: str


register_scripted("melp_proc", lambda i, c: Item(x="done"))

# Construct modifiers list directly, bypassing __or__ mutual exclusion
bad_node = Node(
    name="proc",
    mode="scripted",
    scripted_fn="melp_proc",
    outputs=Item,
    modifiers=[
        Each(over="source.items", key="x"),
        Loop(when=lambda d: False, max_iterations=1),
    ],
)

pipeline = Construct("broken", nodes=[bad_node])
