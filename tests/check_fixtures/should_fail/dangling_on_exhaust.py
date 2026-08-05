# CHECK_ERROR: on_exhaust= requires one of: loop_when=, portal=
# on_exhaust= is a SHARED satellite of both Loop and Portal -- dangling
# without either loop_when= or portal= (neograph Phase 3 strictness gate).
# Programmatic @node fixture -- only the line-1 CHECK_ERROR comment SHAPE
# transfers from each_plus_loop_same_node.py, not an @node precedent.
from pydantic import BaseModel

from neograph import node


class Item(BaseModel, frozen=True):
    x: str


@node(outputs=Item, on_exhaust="last")
def dangling() -> Item:
    return Item(x="v")
