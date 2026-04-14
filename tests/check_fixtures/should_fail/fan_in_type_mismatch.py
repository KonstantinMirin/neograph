# CHECK_ERROR: TypeB.*produces TypeC|type.*compatible
# Scenario 15: fan-in where one upstream produces wrong type
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class TypeA(BaseModel, frozen=True):
    x: str

class TypeB(BaseModel, frozen=True):
    y: int

class TypeC(BaseModel, frozen=True):
    z: float

register_scripted("fi_a", lambda i, c: TypeA(x="hello"))
register_scripted("fi_b", lambda i, c: TypeC(z=3.14))  # produces TypeC, not TypeB
register_scripted("fi_merge", lambda i, c: "done")

pipeline = Construct("broken", nodes=[
    Node.scripted("source_a", fn="fi_a", outputs=TypeA),
    Node.scripted("source_b", fn="fi_b", outputs=TypeC),
    # Consumer expects TypeA + TypeB, but source_b produces TypeC
    Node.scripted("merge", fn="fi_merge",
                  inputs={"source_a": TypeA, "source_b": TypeB},
                  outputs=str),
])
