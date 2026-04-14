# CHECK_ERROR: self.reference|not available|no.*producer|does not exist|own output
# Attack vector: @merge_fn param named after the Oracle node itself.
# The merge barrier runs BEFORE the node's output is written to state,
# so referencing the node's own field is a temporal self-reference that
# can never resolve.

from pydantic import BaseModel

from neograph import merge_fn, node
from neograph.decorators import construct_from_functions


class Claims(BaseModel, frozen=True):
    items: list[str]


@merge_fn
def combine(variants: list[Claims], generate: Claims) -> Claims:
    # 'generate' matches the Oracle node's own name — but that field
    # doesn't exist yet when the merge barrier fires.
    return variants[0]


@node(outputs=Claims, ensemble_n=3, merge_fn="combine")
def generate() -> Claims: ...


pipeline = construct_from_functions("self-ref", [generate])
