# CHECK_ERROR: conditions.*not found|has no field.*conditions
# Scenario 5: dotted map_over path references a field that doesn't exist
from neograph import Construct, Node, Each
from neograph.factory import register_scripted
from pydantic import BaseModel

class FilterResult(BaseModel, frozen=True):
    uncovered: list[str]  # field is "uncovered", not "conditions"

class Item(BaseModel, frozen=True):
    ec_id: str

register_scripted("s5_filter", lambda i, c: FilterResult(uncovered=["a"]))
register_scripted("s5_process", lambda i, c: Item(ec_id="x"))

pipeline = Construct("broken", nodes=[
    Node.scripted("filter", fn="s5_filter", outputs=FilterResult),
    Node.scripted("process", fn="s5_process", inputs=Item, outputs=Item)
    | Each(over="filter.conditions", key="ec_id"),  # "conditions" doesn't exist
])
