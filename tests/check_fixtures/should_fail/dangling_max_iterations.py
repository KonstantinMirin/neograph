# CHECK_ERROR: max_iterations= requires one of: loop_when=
# max_iterations= is a Loop satellite -- dangling without loop_when= (neograph
# Phase 3 strictness gate). Programmatic @node fixture -- the shape exemplar
# (each_plus_loop_same_node.py) is a Node.scripted|Each|Loop fixture; only its
# line-1 CHECK_ERROR comment SHAPE transfers here, not an @node precedent.
from pydantic import BaseModel

from neograph import node


class Item(BaseModel, frozen=True):
    x: str


@node(outputs=Item, max_iterations=5)
def dangling() -> Item:
    return Item(x="v")
