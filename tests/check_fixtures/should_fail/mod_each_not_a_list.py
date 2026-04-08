# CHECK_ERROR: not a list|resolves to.*not.*list
# Each.over path resolves to a plain string field, not a list.
# Should fail at assembly-time validation.
from neograph import Construct, Node
from neograph.modifiers import Each
from neograph.factory import register_scripted
from pydantic import BaseModel


class Info(BaseModel, frozen=True):
    title: str
    count: int


class Result(BaseModel, frozen=True):
    value: str


register_scripted("menl_info", lambda i, c: Info(title="test", count=5))
register_scripted("menl_proc", lambda i, c: Result(value="ok"))

pipeline = Construct("broken", nodes=[
    Node.scripted("info_node", fn="menl_info", outputs=Info),
    Node.scripted("proc", fn="menl_proc", inputs=str, outputs=Result)
    | Each(over="info_node.title", key="x"),
])
