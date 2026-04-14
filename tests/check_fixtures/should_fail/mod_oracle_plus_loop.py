# CHECK_ERROR: Cannot combine Oracle and Loop
# Oracle + Loop on the same node is forbidden (mutual exclusion).
from pydantic import BaseModel

from neograph import Node
from neograph.factory import register_scripted
from neograph.modifiers import Loop, Oracle


class Draft(BaseModel, frozen=True):
    text: str


register_scripted("mol_proc", lambda i, c: Draft(text="ok"))
register_scripted("mol_merge", lambda variants, c: variants[0])

pipeline_node = (
    Node.scripted("proc", fn="mol_proc", outputs=Draft)
    | Oracle(n=3, merge_fn="mol_merge")
    | Loop(when=lambda d: False, max_iterations=1)
)
