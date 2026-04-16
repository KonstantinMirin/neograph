# Loop on a Construct where output type differs from input type.
# Since neograph-vt4y, this is allowed — the loop re-reads original inputs
# from parent state on each iteration instead of feeding output back.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted
from neograph.modifiers import Loop


class Draft(BaseModel, frozen=True):
    text: str

class Review(BaseModel, frozen=True):
    score: float

register_scripted("lt_review", lambda i, c: Review(score=0.5))

pipeline = Construct("valid", nodes=[
    Construct(
        "refine",
        input=Draft,
        output=Review,  # Review != Draft — produce+validate pattern
        nodes=[
            Node.scripted("review", fn="lt_review", outputs=Review),
        ],
    ) | Loop(when=lambda r: r.score < 0.8, max_iterations=3),
])
