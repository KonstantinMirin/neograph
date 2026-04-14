# CHECK_ERROR: Each.*element type mismatch[\s\S]*DifferentItem[\s\S]*list\[Item\]
# Each on sub-construct where sub-construct.input doesn't match item type
from pydantic import BaseModel

from neograph import Construct, Each, Node
from neograph.factory import register_scripted


class Item(BaseModel, frozen=True):
    label: str

class DifferentItem(BaseModel, frozen=True):
    code: int

class Batch(BaseModel, frozen=True):
    items: list[Item]  # items are Item type

class Result(BaseModel, frozen=True):
    score: int

register_scripted("sei_batch", lambda i, c: Batch(items=[Item(label="a")]))
register_scripted("sei_proc", lambda i, c: Result(score=1))

sub = Construct(
    "process",
    input=DifferentItem,  # expects DifferentItem, but Each fans out Item
    output=Result,
    nodes=[Node.scripted("proc", fn="sei_proc", outputs=Result)],
)

pipeline = Construct("each-type-mismatch", nodes=[
    Node.scripted("make", fn="sei_batch", outputs=Batch),
    sub | Each(over="make.items", key="label"),
])
