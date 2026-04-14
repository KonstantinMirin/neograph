# Valid: list[X] consumer of Each-produced dict[str, X] (Scenario 8)
from pydantic import BaseModel

from neograph import Construct, Each, Node
from neograph.factory import register_scripted


class ClusterGroup(BaseModel, frozen=True):
    label: str
    claim_ids: list[str]

class Clusters(BaseModel, frozen=True):
    groups: list[ClusterGroup]

class VerifyResult(BaseModel, frozen=True):
    cluster_label: str
    coverage_pct: int

register_scripted("lc_clusters", lambda i, c: Clusters(groups=[
    ClusterGroup(label="auth", claim_ids=["1"]),
]))
register_scripted("lc_verify", lambda i, c: VerifyResult(cluster_label="auth", coverage_pct=85))
register_scripted("lc_collect", lambda i, c: "done")

pipeline = Construct("valid-list-consumer", nodes=[
    Node.scripted("clusters", fn="lc_clusters", outputs=Clusters),
    Node.scripted("verify", fn="lc_verify", inputs=ClusterGroup, outputs=VerifyResult)
    | Each(over="clusters.groups", key="label"),
    # list[VerifyResult] consumer of dict[str, VerifyResult] Each producer
    Node.scripted("collect", fn="lc_collect", inputs={"verify": list[VerifyResult]}, outputs=str),
])
