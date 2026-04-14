# CHECK_ERROR: TypeB.*no upstream
# Sub-construct input type doesn't match upstream producer
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class TypeA(BaseModel, frozen=True):
    x: str

class TypeB(BaseModel, frozen=True):
    y: int

class TypeC(BaseModel, frozen=True):
    z: float

register_scripted("sim_a", lambda i, c: TypeA(x="hello"))
register_scripted("sim_c", lambda i, c: TypeC(z=3.14))

pipeline = Construct("input-mismatch", nodes=[
    Node.scripted("first", fn="sim_a", outputs=TypeA),
    Construct(
        "sub",
        input=TypeB,   # expects TypeB
        output=TypeC,
        nodes=[Node.scripted("inner", fn="sim_c", outputs=TypeC)],
    ),
    # upstream "first" produces TypeA, but sub expects TypeB
    # These are unrelated types — not parent-child
])
