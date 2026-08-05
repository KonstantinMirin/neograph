# CHECK_ERROR: map_on_error= requires one of: map_over=
# map_on_error= is an Each satellite -- dangling without map_over= (neograph
# Phase 3 strictness gate). map_on_error='raise' is node()'s live default and
# stays silently accepted (value-vs-default, not is-not-None); 'collect' is a
# genuinely-passed non-default value. Programmatic @node fixture -- only the
# line-1 CHECK_ERROR comment SHAPE transfers from each_plus_loop_same_node.py.
from pydantic import BaseModel

from neograph import node


class Item(BaseModel, frozen=True):
    x: str


@node(outputs=Item, map_on_error="collect")
def dangling() -> Item:
    return Item(x="v")
