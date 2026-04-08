# CHECK_ERROR: required|bundled|model|missing
# Attack vector: FromInput(required=True) on a bundled BaseModel.
# The required flag should propagate to all model fields, but does
# _classify_di_params correctly capture (model_cls, required=True)?
# At compile time this should arguably flag since a required bundled
# model has no config to resolve from.
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
