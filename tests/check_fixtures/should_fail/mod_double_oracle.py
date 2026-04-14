# CHECK_ERROR: Oracle.*already|duplicate.*Oracle|multiple.*Oracle
# Two Oracle modifiers on the same node. Should this be caught?
# Oracle fan-out + merge wiring assumes one Oracle per node.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted
from neograph.modifiers import Oracle


class Draft(BaseModel, frozen=True):
    text: str


register_scripted("mdo_proc", lambda i, c: Draft(text="ok"))
register_scripted("mdo_merge1", lambda variants, c: variants[0])
register_scripted("mdo_merge2", lambda variants, c: variants[0])

pipeline = Construct("broken", nodes=[
    Node.scripted("proc", fn="mdo_proc", outputs=Draft)
    | Oracle(n=3, merge_fn="mdo_merge1")
    | Oracle(n=2, merge_fn="mdo_merge2"),
])
