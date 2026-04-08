# CHECK_ERROR: Condition.*not registered
from pydantic import BaseModel
from neograph import Construct, Node
from neograph.modifiers import Loop
from neograph.factory import register_scripted


class Draft(BaseModel, frozen=True):
    text: str


register_scripted("ref_loop_draft", lambda i, c: Draft(text="v1"))

pipeline = Construct("broken", nodes=[
    Node.scripted("refine", fn="ref_loop_draft", inputs=Draft,
                  outputs=Draft) | Loop(when="ghost_loop_condition", max_iterations=3),
])
