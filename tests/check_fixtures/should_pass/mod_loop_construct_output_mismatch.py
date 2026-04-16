# Loop on a Construct where output type differs from input type.
# Since neograph-vt4y, this is the produce+validate pattern — the loop
# re-reads original inputs on each iteration.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted
from neograph.modifiers import Loop


class Draft(BaseModel, frozen=True):
    text: str


class Report(BaseModel, frozen=True):
    summary: str
    score: float


register_scripted("mlc_inner", lambda i, c: Report(summary="ok", score=0.9))

sub = Construct(
    "refine",
    input=Draft,
    output=Report,
    nodes=[
        Node.scripted("inner", fn="mlc_inner", outputs=Report),
    ],
)

# Allowed: Loop with input != output (produce+validate pattern).
pipeline_node = sub | Loop(when=lambda d: d.score < 0.8, max_iterations=5)
