"""Branch-arm cross-arm reachability leakage (neograph-ftnxl.2).

``_validate_node_chain`` (``_construct_validation.py``) walks
``iter_with_arms(construct)``, which flattens a ``_BranchNode``'s
``true_arm_nodes`` then ``false_arm_nodes`` into one producer accumulation
pass with no per-arm membership tracking (see the comment at
``_construct_validation.py:141-150``). Because the walk is a single ordered
pass, a producer registered by a TRUE-arm node is visible to a FALSE-arm node
declared later in the same pass -- even though the two arms are mutually
exclusive at runtime (only one arm's nodes ever execute, so the FALSE-arm
node's ``inputs`` reference can never actually resolve when its own arm is
taken).

This is a soundness gap in the validator, not merely a missing nice-to-have:
the validator's whole purpose is to answer "is this guaranteed present on
every path that reaches this node", but for branch arms it actually answers
the weaker "was this declared upstream in source order". A false-arm
consumer of a true-arm producer should raise ``ConstructError`` at
construction time -- exactly like any other unsatisfied ``inputs`` reference.
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import Construct, ConstructError, Node
from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
from tests.fakes import register_scripted


class ArmSeed(BaseModel, frozen=True):
    text: str


class TrueArmOutput(BaseModel, frozen=True):
    value: str


class FalseArmOutput(BaseModel, frozen=True):
    value: str


def _seed_node() -> Node:
    register_scripted("cross_arm_seed", lambda _in, _cfg: ArmSeed(text="hi"))
    return Node.scripted("seed", fn="cross_arm_seed", outputs=ArmSeed)


def test_false_arm_node_consuming_true_arm_output_raises_construct_error():
    """A false-arm node declaring inputs={'arm-a': TrueArmOutput} references a
    producer that only exists on the (mutually exclusive) true arm. This must
    raise ConstructError at Construct construction time -- it currently does
    NOT, because iter_with_arms flattens both arms into one producer map
    before the false-arm node's inputs are checked (see module docstring).
    """
    seed = _seed_node()

    register_scripted("cross_arm_a", lambda _in, _cfg: TrueArmOutput(value="t"))
    node_a = Node.scripted("arm-a", fn="cross_arm_a", outputs=TrueArmOutput)

    register_scripted("cross_arm_b", lambda _in, _cfg: FalseArmOutput(value="f"))
    node_b = Node.scripted(
        "arm-b",
        fn="cross_arm_b",
        inputs={"arm_a": TrueArmOutput},
        outputs=FalseArmOutput,
    )

    cond = _ConditionSpec(
        source_node=seed,
        attr_chain=["text"],
        op_fn=lambda value, _t: bool(value),
        op_str="route",
        threshold=None,
    )
    meta = _BranchMeta(
        condition_spec=cond,
        true_arm_nodes=[node_a],
        false_arm_nodes=[node_b],
    )

    try:
        Construct("parent", nodes=[seed, _BranchNode(meta, 0)])
    except ConstructError:
        return
    raise AssertionError(
        "expected ConstructError: node 'arm-b' (false arm) consumes "
        "'arm_a', a producer that only exists on the true arm"
    )
