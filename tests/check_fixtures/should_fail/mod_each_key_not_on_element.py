# CHECK_ERROR: no field.*badkey|has no field 'badkey'
# Each.key references a field that doesn't exist on the element type.
# Should fail at assembly-time validation.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted
from neograph.modifiers import Each


class Item(BaseModel, frozen=True):
    name: str
    value: int


class Container(BaseModel, frozen=True):
    items: list[Item]


class Result(BaseModel, frozen=True):
    output: str


register_scripted("mek_container", lambda i, c: Container(items=[Item(name="a", value=1)]))
register_scripted("mek_proc", lambda i, c: Result(output="ok"))

pipeline = Construct("broken", nodes=[
    Node.scripted("container", fn="mek_container", outputs=Container),
    Node.scripted("proc", fn="mek_proc", inputs=Item, outputs=Result)
    | Each(over="container.items", key="badkey"),
])
