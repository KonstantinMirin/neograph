# CHECK_ERROR: outputs=.*differs from return annotation
from pydantic import BaseModel

from neograph import node


class Claims(BaseModel, frozen=True):
    items: list[str]

class Scores(BaseModel, frozen=True):
    values: list[float]


@node(outputs=Scores)
def extract(topic: str) -> Claims:
    return Claims(items=["a"])
