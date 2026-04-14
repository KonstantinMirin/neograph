"""@node(outputs=X) with -> X (matching) should not raise."""
from pydantic import BaseModel

from neograph import node


class Claims(BaseModel, frozen=True):
    items: list[str]


@node(outputs=Claims)
def extract(topic: str) -> Claims:
    return Claims(items=["a"])


assert extract.outputs is Claims
