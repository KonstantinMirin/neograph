# CHECK_ERROR: output.*TypeD.*no.*node.*produces|boundary.*contract
# Sub-construct output=X, 2 internal nodes, neither produces X
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class TypeA(BaseModel, frozen=True):
    x: str

class TypeB(BaseModel, frozen=True):
    y: int

class TypeC(BaseModel, frozen=True):
    z: float

class TypeD(BaseModel, frozen=True):
    w: bool

register_scripted("so2_a", lambda i, c: TypeB(y=1))
register_scripted("so2_b", lambda i, c: TypeC(z=3.14))

pipeline = Construct("output-no-prod-2", nodes=[
    Node.scripted("first", fn="so2_a", outputs=TypeA),
    Construct(
        "sub",
        input=TypeA,
        output=TypeD,  # declares TypeD output but nobody produces it
        nodes=[
            Node.scripted("inner1", fn="so2_a", outputs=TypeB),
            Node.scripted("inner2", fn="so2_b", inputs=TypeB, outputs=TypeC),
        ],
    ),
])
