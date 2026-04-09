# Valid: FromInput(required=True) on a bundled BaseModel compiles fine.
# The required flag is enforced at runtime, not compile time.
from typing import Annotated

from pydantic import BaseModel

from neograph import FromInput, node
from neograph.decorators import construct_from_functions


class RunCtx(BaseModel):
    node_id: str
    project_root: str


class Output(BaseModel, frozen=True):
    result: str


@node(outputs=Output)
def my_node(ctx: Annotated[RunCtx, FromInput(required=True)]) -> Output: ...


pipeline = construct_from_functions("required-bundled", [my_node])
