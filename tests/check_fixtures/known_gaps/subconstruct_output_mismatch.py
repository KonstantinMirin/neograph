# CHECK_ERROR: output.*TypeA.*no.*node.*produces|boundary.*contract
# Scenario 7 variant: sub-construct declares output=TypeA but no internal node produces TypeA
from neograph import Construct, Node
from neograph.factory import register_scripted
from pydantic import BaseModel

class TypeA(BaseModel, frozen=True):
    x: str

class TypeB(BaseModel, frozen=True):
    y: int

register_scripted("so_inner", lambda i, c: TypeB(y=1))

pipeline = Construct("broken", nodes=[
    Construct(
        "sub",
        input=TypeA,
        output=TypeA,  # declares TypeA output
        nodes=[
            Node.scripted("inner", fn="so_inner", outputs=TypeB),  # produces TypeB
        ],
    ),
])
