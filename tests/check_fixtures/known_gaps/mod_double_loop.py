# CHECK_ERROR: Loop.*already|duplicate.*Loop|multiple.*Loop
# Two Loop modifiers on the same node. Should this be caught?
from neograph import Node
from neograph.modifiers import Loop
from neograph.factory import register_scripted
from pydantic import BaseModel


class Draft(BaseModel, frozen=True):
    text: str
    score: float = 0.5


register_scripted("mdl_proc", lambda i, c: Draft(text="ok", score=0.9))

pipeline_node = (
    Node.scripted("proc", fn="mdl_proc", inputs=Draft, outputs=Draft)
    | Loop(when=lambda d: d.score < 0.8, max_iterations=5)
    | Loop(when=lambda d: d.score < 0.9, max_iterations=3)
)
