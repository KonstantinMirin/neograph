"""@node with -> T and no outputs= should infer outputs=T."""
from pydantic import BaseModel

from neograph import node


class Claims(BaseModel, frozen=True):
    items: list[str]


@node
def extract(topic: str) -> Claims:
    return Claims(items=["a"])


assert extract.outputs is Claims
