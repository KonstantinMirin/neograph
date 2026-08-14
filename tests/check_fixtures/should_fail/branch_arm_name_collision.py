# CHECK_ERROR: node name collision[\s\S]*both map to state field
# Two nodes with colliding field_name_for() output, one placed in each
# branch arm. state.py's seen_fields collision walk (compile_state_model)
# only scans nodes_only + sub_constructs, never branch-arm items, so this
# currently compiles silently instead of raising the same collision error
# a top-level pair gets (neograph-ftnxl.13).
from pydantic import BaseModel

from neograph import Construct, Node
from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
from tests.fakes import register_scripted


class ArmSeed(BaseModel, frozen=True):
    text: str


class ArmOutput(BaseModel, frozen=True):
    value: str


register_scripted("bnc_seed", lambda i, c: ArmSeed(text="hi"))
register_scripted("bnc_true", lambda i, c: ArmOutput(value="t"))
register_scripted("bnc_false", lambda i, c: ArmOutput(value="f"))

seed = Node.scripted("seed", fn="bnc_seed", outputs=ArmSeed)
# "arm-x" and "arm_x" both normalize to state field "arm_x" — collide
# across the true/false arm boundary.
true_producer = Node.scripted("arm-x", fn="bnc_true", outputs=ArmOutput)
false_producer = Node.scripted("arm_x", fn="bnc_false", outputs=ArmOutput)

meta = _BranchMeta(
    condition_spec=_ConditionSpec(
        source_node=seed,
        attr_chain=["text"],
        op_fn=lambda v, _t: bool(v),
        op_str="route",
        threshold=None,
    ),
    true_arm_nodes=[true_producer],
    false_arm_nodes=[false_producer],
)

pipeline = Construct("branch-arm-name-collision", nodes=[seed, _BranchNode(meta, 0)])
