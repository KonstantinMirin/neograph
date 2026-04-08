# CHECK_ERROR: type.*compatible|no upstream produces
# Attack vector 5b: Reverse Liskov — producer is superclass, consumer expects subclass.
# TypeA is NOT a TypeB. Should fail.
from neograph import Construct, Node
from neograph.factory import register_scripted
from pydantic import BaseModel

class TypeA(BaseModel, frozen=True):
    x: str

class TypeB(TypeA, frozen=True):
    y: int = 0

register_scripted("super_a", lambda i, c: TypeA(x="hello"))
register_scripted("super_b", lambda i, c: TypeB(x="ok", y=1))

pipeline = Construct("broken", nodes=[
    Node.scripted("first", fn="super_a", outputs=TypeA),
    Node.scripted("second", fn="super_b", inputs=TypeB, outputs=TypeB),
])
