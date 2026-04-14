# CHECK_ERROR: ghost_id|no field
# Scenario: Each.key references a field that doesn't exist on a Pydantic item type
from pydantic import BaseModel

from neograph import Construct, Each, Node
from neograph.factory import register_scripted


class Item(BaseModel, frozen=True):
    label: str
    value: int

class ItemList(BaseModel, frozen=True):
    items: list[Item]

register_scripted("ek_make", lambda i, c: ItemList(items=[Item(label="a", value=1)]))
register_scripted("ek_proc", lambda i, c: Item(label="x", value=2))

pipeline = Construct("broken", nodes=[
    Node.scripted("make", fn="ek_make", outputs=ItemList),
    Node.scripted("proc", fn="ek_proc", inputs=Item, outputs=Item)
    | Each(over="make.items", key="ghost_id"),  # Item has no "ghost_id" field
])
