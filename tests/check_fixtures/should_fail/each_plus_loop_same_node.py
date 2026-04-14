# CHECK_ERROR: Cannot combine Each and Loop
# Each + Loop on the same item is forbidden
from pydantic import BaseModel

from neograph import Node
from neograph.factory import register_scripted
from neograph.modifiers import Each, Loop


class Item(BaseModel, frozen=True):
    x: str

register_scripted("el_proc", lambda i, c: Item(x="done"))

pipeline_node = (
    Node.scripted("proc", fn="el_proc", outputs=Item)
    | Each(over="source.items", key="x")
    | Loop(when=lambda d: False, max_iterations=1)
)
