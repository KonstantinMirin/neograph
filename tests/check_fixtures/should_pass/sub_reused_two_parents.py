# Valid: Same Construct object reused in two parent Constructs — state isolation OK
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class TypeA(BaseModel, frozen=True):
    x: str

class TypeB(BaseModel, frozen=True):
    y: int

register_scripted("sr_a", lambda i, c: TypeA(x="hello"))
register_scripted("sr_b", lambda i, c: TypeB(y=1))

# Shared sub-construct instance
shared_sub = Construct(
    "shared-sub",
    input=TypeA,
    output=TypeB,
    nodes=[Node.scripted("sub-node", fn="sr_b", outputs=TypeB)],
)

# Parent 1 uses it
parent1 = Construct("parent1", nodes=[
    Node.scripted("seed1", fn="sr_a", outputs=TypeA),
    shared_sub,
])

# Parent 2 also uses the exact same object
parent2 = Construct("parent2", nodes=[
    Node.scripted("seed2", fn="sr_a", outputs=TypeA),
    shared_sub,
])
