# CHECK_ERROR: FromConfig.*merge|merge.*FromConfig|DI.*merge_fn
# Attack vector 6: FromConfig in a @merge_fn — lint() only walks Construct
# nodes via _get_param_resolutions(). Merge functions are stored in
# _merge_fn_registry, which lint() never checks. This means DI bindings
# in merge functions are invisible to all static validation.
#
# Uses scripted mode to avoid unrelated LLM config errors. The Oracle
# modifier is applied programmatically to a scripted node, so compile
# should succeed unless the validator checks merge_fn DI bindings.
from typing import Annotated

from pydantic import BaseModel

from neograph import Construct, FromConfig, merge_fn, node
from neograph.decorators import construct_from_functions
from neograph.factory import register_scripted
from neograph.modifiers import Oracle
from neograph.node import Node


class Claims(BaseModel, frozen=True):
    items: list[str]


class SharedResources(BaseModel):
    db_url: str = "sqlite://"
    api_key: str = "test"


@merge_fn
def combine(
    variants: list[Claims],
    shared: Annotated[SharedResources, FromConfig],
) -> Claims:
    return variants[0]


# Use the programmatic API to avoid @node + LLM mode complications.
register_scripted("di_merge_producer", lambda i, c: Claims(items=["a"]))

pipeline = Construct("merge-di", nodes=[
    Node.scripted("producer", fn="di_merge_producer", outputs=Claims)
    | Oracle(n=3, merge_fn="combine"),
])
