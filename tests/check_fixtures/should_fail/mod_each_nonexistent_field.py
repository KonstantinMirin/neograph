# CHECK_ERROR: Each\(over='cluster\.nonexistent_field'\) path 'cluster\.nonexistent_field' does not resolve
# Each.over path where the dotted segment points to a field that doesn't exist
# on the upstream output model. Should fail at assembly-time validation.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.modifiers import Each
from tests.fakes import register_scripted


class Clusters(BaseModel, frozen=True):
    groups: list[str]


class Result(BaseModel, frozen=True):
    value: str


register_scripted("me_cluster", lambda i, c: Clusters(groups=["a"]))
register_scripted("me_proc", lambda i, c: Result(value="ok"))

pipeline = Construct(
    "broken",
    nodes=[
        Node.scripted("cluster", fn="me_cluster", outputs=Clusters),
        Node.scripted("proc", fn="me_proc", inputs=str, outputs=Result)
        | Each(over="cluster.nonexistent_field", key="x"),
    ],
)
