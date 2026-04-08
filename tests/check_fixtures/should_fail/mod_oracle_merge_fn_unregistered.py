# CHECK_ERROR: not registered|merge_fn.*not found|no.*merge|Scripted function.*not registered
# Oracle with merge_fn pointing to a function name that was never registered.
# Uses a scripted node to avoid the configure_llm() guard masking the real check.
from neograph import Construct, Node
from neograph.modifiers import Oracle
from neograph.factory import register_scripted
from pydantic import BaseModel


class Draft(BaseModel, frozen=True):
    text: str


register_scripted("omfu_gen", lambda i, c: Draft(text="hello"))

pipeline = Construct("broken", nodes=[
    Node.scripted("generate", fn="omfu_gen", outputs=Draft)
    | Oracle(n=3, merge_fn="this_merge_fn_does_not_exist"),
])
