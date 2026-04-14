# Valid: Nested sub-constructs 3 levels deep — compile recurses correctly
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class TypeA(BaseModel, frozen=True):
    x: str

class TypeB(BaseModel, frozen=True):
    y: int

class TypeC(BaseModel, frozen=True):
    z: float

register_scripted("n3_a", lambda i, c: TypeA(x="hello"))
register_scripted("n3_b", lambda i, c: TypeB(y=1))
register_scripted("n3_c", lambda i, c: TypeC(z=3.14))

# Level 3 (innermost): input=TypeB, output=TypeC
inner = Construct(
    "inner",
    input=TypeB,
    output=TypeC,
    nodes=[Node.scripted("inner-node", fn="n3_c", outputs=TypeC)],
)

# Level 2: input=TypeA, output=TypeC — but inner expects TypeB from upstream
# The inner sub-construct takes TypeB input, but the mid-level only has TypeA producer
mid = Construct(
    "mid",
    input=TypeA,
    output=TypeC,
    nodes=[
        Node.scripted("mid-node", fn="n3_b", outputs=TypeB),
        inner,
    ],
)

# Level 1 (top): wraps mid
pipeline = Construct("top", nodes=[
    Node.scripted("seed", fn="n3_a", outputs=TypeA),
    mid,
])
