# CHECK_ERROR: Each.*already|duplicate.*Each|multiple.*Each
# Two Each modifiers on the same node. Should this be caught?
# Each wiring assumes one Each per node.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted
from neograph.modifiers import Each


class Item(BaseModel, frozen=True):
    x: str


class Container(BaseModel, frozen=True):
    items: list[Item]
    other: list[Item]


class Result(BaseModel, frozen=True):
    value: str


register_scripted("mde_container", lambda i, c: Container(items=[Item(x="a")], other=[Item(x="b")]))
register_scripted("mde_proc", lambda i, c: Result(value="ok"))

pipeline = Construct("broken", nodes=[
    Node.scripted("container", fn="mde_container", outputs=Container),
    Node.scripted("proc", fn="mde_proc", inputs=Item, outputs=Result)
    | Each(over="container.items", key="x")
    | Each(over="container.other", key="x"),
])
