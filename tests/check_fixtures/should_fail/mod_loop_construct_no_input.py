# CHECK_ERROR: Loop.*requires both input.*and output|input.*output.*declared
# Loop on a Construct that has output but no input declared.
# Back-edge can't wire without both.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted
from neograph.modifiers import Loop


class Draft(BaseModel, frozen=True):
    text: str
    score: float = 0.0


register_scripted("mlcni_inner", lambda i, c: Draft(text="ok", score=0.5))

sub = Construct(
    "refine",
    output=Draft,  # output declared
    # input=... deliberately missing
    nodes=[
        Node.scripted("inner", fn="mlcni_inner", outputs=Draft),
    ],
)

pipeline_node = sub | Loop(when=lambda d: d.score < 0.8, max_iterations=5)
