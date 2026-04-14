# CHECK_ERROR: type.*compatible|no upstream produces
# Attack vector: two unrelated BaseModel subclasses. Claims is not Scores.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class Claims(BaseModel, frozen=True):
    text: str

class Scores(BaseModel, frozen=True):
    value: float

register_scripted("unr_a", lambda i, c: Claims(text="ok"))
register_scripted("unr_b", lambda i, c: Scores(value=1.0))

pipeline = Construct("broken", nodes=[
    Node.scripted("first", fn="unr_a", outputs=Claims),
    Node.scripted("second", fn="unr_b", inputs=Scores, outputs=Scores),
])
