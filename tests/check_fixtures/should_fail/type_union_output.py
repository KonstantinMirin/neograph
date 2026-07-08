# CHECK_ERROR: declares inputs=Scores but no upstream produces a compatible value[\s\S]*Claims \| str
# Attack vector 6: Union type output — Claims | str
# Validator should reject union outputs because the consumer can't be sure
# which branch it gets at runtime.
from pydantic import BaseModel

from neograph import Construct, Node
from tests.fakes import register_scripted


class Claims(BaseModel, frozen=True):
    text: str


class Scores(BaseModel, frozen=True):
    value: float


register_scripted("union_out", lambda i, c: Claims(text="ok"))
register_scripted("union_in", lambda i, c: Scores(value=1.0))

pipeline = Construct(
    "broken",
    nodes=[
        Node.scripted("first", fn="union_out", outputs=Claims | str),
        Node.scripted("second", fn="union_in", inputs=Scores, outputs=Scores),
    ],
)
