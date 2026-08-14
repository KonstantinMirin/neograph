# A node inside a branch arm consumes an EARLIER node's output from the SAME
# arm -- must still validate and compile after neograph-ftnxl.2's arm-scoping
# fix (same-arm visibility is preserved; only cross-arm/post-join visibility
# was removed).
from pydantic import BaseModel

from neograph import Construct, Node
from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
from tests.fakes import register_scripted


class ArmSeed(BaseModel, frozen=True):
    text: str


class Intermediate(BaseModel, frozen=True):
    value: str


class Final(BaseModel, frozen=True):
    value: str


register_scripted("sapc_seed", lambda i, c: ArmSeed(text="hi"))
register_scripted("sapc_first", lambda i, c: Intermediate(value="mid"))
register_scripted("sapc_second", lambda i, c: Final(value="done"))
register_scripted("sapc_false", lambda i, c: Final(value="skip"))

seed = Node.scripted("seed", fn="sapc_seed", outputs=ArmSeed)
first = Node.scripted("arm-first", fn="sapc_first", outputs=Intermediate)
# Consumes 'arm-first' by name -- legal because it is in the SAME (true) arm.
second = Node.scripted(
    "arm-second",
    fn="sapc_second",
    inputs={"arm_first": Intermediate},
    outputs=Final,
)
false_producer = Node.scripted("arm-alt", fn="sapc_false", outputs=Final)

meta = _BranchMeta(
    condition_spec=_ConditionSpec(
        source_node=seed,
        attr_chain=["text"],
        op_fn=lambda v, _t: bool(v),
        op_str="route",
        threshold=None,
    ),
    true_arm_nodes=[first, second],
    false_arm_nodes=[false_producer],
)

pipeline = Construct("same-arm-chain", nodes=[seed, _BranchNode(meta, 0)])
