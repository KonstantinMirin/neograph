# CHECK_ERROR: produced only on a branch arm this node cannot reach
from pydantic import BaseModel

from neograph import Construct, Node
from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
from tests.fakes import register_scripted


class ArmSeed(BaseModel, frozen=True):
    text: str


class TrueArmOutput(BaseModel, frozen=True):
    value: str


class FalseArmOutput(BaseModel, frozen=True):
    value: str


register_scripted("cca_seed", lambda i, c: ArmSeed(text="hi"))
register_scripted("cca_true", lambda i, c: TrueArmOutput(value="t"))
register_scripted("cca_false", lambda i, c: FalseArmOutput(value="f"))

seed = Node.scripted("seed", fn="cca_seed", outputs=ArmSeed)
true_producer = Node.scripted("arm-a", fn="cca_true", outputs=TrueArmOutput)
# The false arm declares a dependency on the TRUE arm's producer by name —
# mutually exclusive at runtime, so this can never resolve when the false
# arm actually runs.
false_consumer = Node.scripted(
    "arm-b",
    fn="cca_false",
    inputs={"arm_a": TrueArmOutput},
    outputs=FalseArmOutput,
)

meta = _BranchMeta(
    condition_spec=_ConditionSpec(
        source_node=seed,
        attr_chain=["text"],
        op_fn=lambda v, _t: bool(v),
        op_str="route",
        threshold=None,
    ),
    true_arm_nodes=[true_producer],
    false_arm_nodes=[false_consumer],
)

pipeline = Construct("cross-arm-consumption", nodes=[seed, _BranchNode(meta, 0)])
