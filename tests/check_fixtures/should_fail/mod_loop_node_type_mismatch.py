# CHECK_ERROR: loop.*back-edge|output type.*not compatible.*input
# Loop on a Node where output type differs from input type.
# Self-loop requires output -> input compatibility.
from neograph import Node
from neograph.modifiers import Loop
from neograph.factory import register_scripted
from pydantic import BaseModel


class Input(BaseModel, frozen=True):
    text: str


class Output(BaseModel, frozen=True):
    score: float


register_scripted("mln_proc", lambda i, c: Output(score=0.5))

# Output(score) -> Input(text) is not compatible
pipeline_node = (
    Node.scripted("proc", fn="mln_proc", inputs=Input, outputs=Output)
    | Loop(when=lambda d: d.score < 0.8, max_iterations=5)
)
