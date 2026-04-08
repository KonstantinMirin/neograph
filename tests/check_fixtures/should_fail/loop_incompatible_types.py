# CHECK_ERROR: loop.*compatible|output.*not compatible.*input
# Loop on a Construct where output type doesn't match input type
from neograph import Construct, Node
from neograph.modifiers import Loop
from neograph.factory import register_scripted
from pydantic import BaseModel

class Draft(BaseModel, frozen=True):
    text: str

class Review(BaseModel, frozen=True):
    score: float

register_scripted("lt_review", lambda i, c: Review(score=0.5))

pipeline = Construct("broken", nodes=[
    Construct(
        "refine",
        input=Draft,
        output=Review,  # Review != Draft — can't loop
        nodes=[
            Node.scripted("review", fn="lt_review", outputs=Review),
        ],
    ) | Loop(when=lambda r: r.score < 0.8, max_iterations=3),
])
