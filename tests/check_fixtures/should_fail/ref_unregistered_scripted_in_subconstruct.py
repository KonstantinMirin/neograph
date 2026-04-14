# CHECK_ERROR: Scripted function.*not registered
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class Input(BaseModel, frozen=True):
    text: str


class Output(BaseModel, frozen=True):
    result: str


register_scripted("ref_outer_fn", lambda i, c: Input(text="ok"))

# Outer scripted node is registered. Inner sub-construct scripted node is NOT.
sub = Construct("inner", input=Input, output=Output, nodes=[
    Node.scripted("inner-step", fn="ghost_inner_fn", outputs=Output),
])

pipeline = Construct("outer", nodes=[
    Node.scripted("first", fn="ref_outer_fn", outputs=Input),
    sub,
])
