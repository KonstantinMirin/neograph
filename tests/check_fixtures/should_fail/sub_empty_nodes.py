# CHECK_ERROR: output.*TypeB.*no.*node.*produces|boundary.*contract
# Sub-construct with empty nodes list — no output producer
from neograph import Construct, Node
from neograph.factory import register_scripted
from pydantic import BaseModel

class TypeA(BaseModel, frozen=True):
    x: str

class TypeB(BaseModel, frozen=True):
    y: int

register_scripted("se_first", lambda i, c: TypeA(x="hello"))

pipeline = Construct("empty-sub", nodes=[
    Node.scripted("first", fn="se_first", outputs=TypeA),
    Construct(
        "empty",
        input=TypeA,
        output=TypeB,
        nodes=[],  # no internal nodes at all
    ),
])
