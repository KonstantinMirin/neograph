# CHECK_ERROR: type.*compatible|type.*mismatch|expected.*got
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class TypeA(BaseModel, frozen=True):
    x: str

class TypeB(BaseModel, frozen=True):
    y: int

register_scripted("tm_a", lambda i, c: TypeA(x="hello"))
register_scripted("tm_b", lambda i, c: TypeB(y=1))

pipeline = Construct("broken", nodes=[
    Node.scripted("first", fn="tm_a", outputs=TypeA),
    Node.scripted("second", fn="tm_b", inputs=TypeB, outputs=TypeB),
])
