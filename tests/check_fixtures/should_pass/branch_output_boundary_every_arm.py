# A sub-construct's declared output= is satisfied on EVERY arm of an internal
# branch by two DIFFERENTLY-NAMED arm nodes, each producing a compatible type
# (neograph-ftnxl.2's should_pass twin for the type-based output-boundary
# check). Two arm nodes can never share a literal .name (LangGraph's
# add_node requires graph-wide unique names), so "every arm satisfies the
# boundary" is only reachable this way -- not via same-name promotion.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
from tests.fakes import register_scripted


class ArmSeed(BaseModel, frozen=True):
    text: str


class BoundaryResult(BaseModel, frozen=True):
    value: str


register_scripted("obea_seed", lambda i, c: ArmSeed(text="hi"))
register_scripted("obea_true", lambda i, c: BoundaryResult(value="t"))
register_scripted("obea_false", lambda i, c: BoundaryResult(value="f"))

seed = Node.scripted("seed", fn="obea_seed", outputs=ArmSeed)
# Different names, same compatible TYPE -- reachable and compilable.
true_producer = Node.scripted("arm-true-result", fn="obea_true", outputs=BoundaryResult)
false_producer = Node.scripted("arm-false-result", fn="obea_false", outputs=BoundaryResult)

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

sub = Construct(
    "boundary-sub",
    input=ArmSeed,
    nodes=[seed, _BranchNode(meta, 0)],
    output=BoundaryResult,
)

pipeline = Construct("boundary-parent", nodes=[sub])
