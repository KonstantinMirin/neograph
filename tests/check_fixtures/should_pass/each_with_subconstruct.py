# Valid: Each modifier on a sub-construct with proper types
from pydantic import BaseModel

from neograph import Construct, Each, Node
from neograph.factory import register_scripted


class Item(BaseModel, frozen=True):
    label: str

class Batch(BaseModel, frozen=True):
    items: list[Item]

class Result(BaseModel, frozen=True):
    label: str
    score: int

register_scripted("ep_batch", lambda i, c: Batch(items=[Item(label="a")]))
register_scripted("ep_proc", lambda i, c: Result(label="x", score=1))

sub = Construct(
    "process",
    input=Item,
    output=Result,
    nodes=[Node.scripted("proc", fn="ep_proc", outputs=Result)],
)

pipeline = Construct("valid-each", nodes=[
    Node.scripted("make", fn="ep_batch", outputs=Batch),
    sub | Each(over="make.items", key="label"),
])
