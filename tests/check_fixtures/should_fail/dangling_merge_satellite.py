# CHECK_ERROR: merge_pre_process= requires one of: ensemble_n=, merge_fn=, merge_prompt=, models=
# merge_pre_process= is an Oracle satellite -- dangling without any Oracle
# trigger (ensemble_n/models/merge_fn/merge_prompt) present (neograph Phase 3
# strictness gate). Programmatic @node fixture -- only the line-1 CHECK_ERROR
# comment SHAPE transfers from each_plus_loop_same_node.py, not a precedent.
from pydantic import BaseModel

from neograph import node


class Item(BaseModel, frozen=True):
    x: str


@node(outputs=Item, merge_pre_process=lambda v: {"x": v})
def dangling() -> Item:
    return Item(x="v")
