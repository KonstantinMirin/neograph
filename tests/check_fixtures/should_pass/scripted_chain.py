# Valid: three scripted nodes in a chain
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class A(BaseModel, frozen=True):
    x: str

class B(BaseModel, frozen=True):
    y: int

register_scripted("sp_a", lambda i, c: A(x="hello"))
register_scripted("sp_b", lambda i, c: B(y=1))

pipeline = Construct("valid-chain", nodes=[
    Node.scripted("first", fn="sp_a", outputs=A),
    Node.scripted("second", fn="sp_b", inputs=A, outputs=B),
])
