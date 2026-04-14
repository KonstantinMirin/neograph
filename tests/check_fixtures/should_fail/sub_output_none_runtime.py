# CHECK_ERROR: output.*TypeA.*no.*node.*produces
# Known gap: sub-construct output=TypeA, input=TypeA, inner node produces TypeB.
# The input port satisfies the output boundary check (input==output), so assembly
# passes. At runtime, make_subgraph_fn iterates sub_result.values() looking for
# isinstance(X, sub.output) — no match → returns None silently.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class TypeA(BaseModel, frozen=True):
    x: str

class TypeB(BaseModel, frozen=True):
    y: int

register_scripted("sonr_a", lambda i, c: TypeA(x="hello"))
register_scripted("sonr_b", lambda i, c: TypeB(y=1))

pipeline = Construct("output-none-rt", nodes=[
    Node.scripted("first", fn="sonr_a", outputs=TypeA),
    Construct(
        "sub",
        input=TypeA,
        output=TypeA,  # declares TypeA output
        nodes=[
            Node.scripted("inner", fn="sonr_b", outputs=TypeB),  # produces TypeB!
        ],
    ),
])
