# CHECK_ERROR: ambig|conflict|both.*FromInput.*FromConfig|FromInput.*FromConfig
# Attack vector: Both FromInput AND FromConfig on the same param.
# Annotated[str, FromInput, FromConfig] — which marker wins? The classifier
# picks the first one it finds. This is ambiguous and should arguably be
# flagged at decoration time, but the classifier silently picks FromInput.
from typing import Annotated

from pydantic import BaseModel

from neograph import FromConfig, FromInput, node
from neograph.decorators import construct_from_functions


class Output(BaseModel, frozen=True):
    result: str


@node(outputs=Output)
def my_node(topic: Annotated[str, FromInput, FromConfig]) -> Output: ...


pipeline = construct_from_functions("double-marker", [my_node])
