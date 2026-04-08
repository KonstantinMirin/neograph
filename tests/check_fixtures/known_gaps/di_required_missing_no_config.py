# CHECK_ERROR: required|missing|FromInput
# Attack vector 4: FromInput(required=True) with no config provided.
# lint() flags this, but compile() does not. A required DI binding that
# cannot possibly be satisfied should arguably be a compile-time error.
from typing import Annotated

from pydantic import BaseModel

from neograph import Construct, FromInput, node
from neograph.decorators import construct_from_functions


class Output(BaseModel, frozen=True):
    result: str


@node(outputs=Output)
def my_node(topic: Annotated[str, FromInput(required=True)]) -> Output: ...


pipeline = construct_from_functions("required-di", [my_node])
