# CHECK_ERROR: declares inputs=TypeB but no upstream produces a compatible value[\s\S]*node 'first': TypeA
# Attack vector 5b: Reverse Liskov — producer is superclass, consumer expects subclass.
# TypeA is NOT a TypeB. Should fail.
from pydantic import BaseModel

from neograph import Construct, Node
from tests.fakes import register_scripted


class TypeA(BaseModel, frozen=True):
    x: str


class TypeB(TypeA, frozen=True):
    y: int = 0


register_scripted("super_a", lambda i, c: TypeA(x="hello"))
register_scripted("super_b", lambda i, c: TypeB(x="ok", y=1))

pipeline = Construct(
    "broken",
    nodes=[
        Node.scripted("first", fn="super_a", outputs=TypeA),
        Node.scripted("second", fn="super_b", inputs=TypeB, outputs=TypeB),
    ],
)
