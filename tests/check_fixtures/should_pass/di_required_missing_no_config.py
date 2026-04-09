# Valid: FromInput(required=True) compiles fine without config.
# The required flag is enforced at runtime, not compile time. lint() flags it.
from typing import Annotated

from pydantic import BaseModel

from neograph import Construct, FromInput, node
from neograph.decorators import construct_from_functions


class Output(BaseModel, frozen=True):
    result: str


@node(outputs=Output)
def my_node(topic: Annotated[str, FromInput(required=True)]) -> Output: ...


pipeline = construct_from_functions("required-di", [my_node])
