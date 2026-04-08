# Attack vector 5: Subclass compatibility (Liskov)
# TypeB inherits from TypeA. Node produces TypeB, consumer expects TypeA.
# Should pass — Liskov substitution principle.
from neograph import Construct, Node
from neograph.factory import register_scripted
from pydantic import BaseModel

class TypeA(BaseModel, frozen=True):
    x: str

class TypeB(TypeA, frozen=True):
    y: int = 0

register_scripted("sub_b", lambda i, c: TypeB(x="hello", y=1))
register_scripted("sub_a", lambda i, c: TypeA(x="done"))

pipeline = Construct("valid-subclass", nodes=[
    Node.scripted("first", fn="sub_b", outputs=TypeB),
    Node.scripted("second", fn="sub_a", inputs=TypeA, outputs=TypeA),
])
