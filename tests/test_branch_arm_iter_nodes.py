"""Branch-arm nodes must be visible to iter_nodes tree walks (neograph-tdbb).

``iter_nodes`` (construct.py) is the single source of truth for the IR
node-tree walk that feeds ``_collect_scripted_shims``, ``_collect_required_di``,
and node/schema fingerprint collection. It skips ``_BranchNode`` sentinels, so
every node inside a branch arm (``_BranchMeta.true_arm_nodes`` /
``false_arm_nodes``) is invisible to those walks.

Concrete symptom: a scripted ``@node`` sub-construct placed inside a branch arm
fails to compile with ``ConfigurationError: Scripted function '<name>' not
registered`` -- its ``_scripted_shim`` PrivateAttr is never collected into the
per-compile scripted dict, and the recursive arm compile inherits the parent's
incomplete lookup (``_scripted_lookup=`` is threaded down, never re-collected).

This is distinct from neograph-faf8 (checkpointer/conditions threading into arm
sub-construct compile). faf8's tests pass only because they used
``Node.scripted(fn=...)`` + an explicit ``scripted=`` dict rather than the
``@node`` shim-collection path that this bug breaks.

Three-surface parity: branches only arise from ForwardConstruct tracing or the
programmatic ``_BranchNode`` sentinel, so the branch itself is built
programmatically. The arm content is a ``@node`` sub-construct
(``construct_from_functions`` with a port param) -- the exact surface the bug
makes invisible.
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import (
    Construct,
    Node,
    compile,
    construct_from_functions,
    node,
    run,
)
from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
from tests.fakes import build_test_compile_kwargs, register_scripted


class ArmClaim(BaseModel, frozen=True):
    claim_id: str
    text: str


class ArmResult(BaseModel, frozen=True):
    claim_id: str
    disposition: str


def _node_subconstruct() -> Construct:
    """A @node sub-construct whose single scripted node carries a
    ``_scripted_shim`` PrivateAttr (attached by the @node assembly path) and a
    port param (``claim: ArmClaim`` matches ``input=ArmClaim``)."""

    @node(outputs=ArmResult)
    def gate(claim: ArmClaim) -> ArmResult:
        return ArmResult(claim_id=claim.claim_id, disposition="confirmed")

    return construct_from_functions(
        "gate-sub", [gate], input=ArmClaim, output=ArmResult,
    )


def _branch_parent_with_arm_sub(sub: Construct, *, arm: str) -> tuple[Construct, dict]:
    """Feed ``sub`` from a scripted seed through the chosen branch arm.

    The op_fn routes unconditionally to the arm that holds the sub-construct.
    Returns (parent, compile_kwargs).
    """
    register_scripted(
        "arm_seed", lambda _in, _cfg: ArmClaim(claim_id="c1", text="evidence"),
    )
    seed = Node.scripted("seed", fn="arm_seed", outputs=ArmClaim)

    take_true = arm == "true"
    cond = _ConditionSpec(
        source_node=seed,
        attr_chain=["text"],
        op_fn=(lambda value, _t: bool(value)) if take_true else (lambda value, _t: not value),
        op_str="route",
        threshold=None,
    )
    meta = _BranchMeta(
        condition_spec=cond,
        true_arm_nodes=[sub] if take_true else [],
        false_arm_nodes=[] if take_true else [sub],
    )
    parent = Construct("parent", nodes=[seed, _BranchNode(meta, 0)])
    return parent, build_test_compile_kwargs()


def test_node_subconstruct_in_true_arm_compiles_and_runs():
    """A @node sub-construct in the TRUE arm must compile (its shim is
    collected) and run (the arm output surfaces under the sub's state field).

    Before the fix: compile raised ``ConfigurationError: Scripted function
    'gate-sub.gate' not registered`` because iter_nodes skipped the branch arm
    and the shim was never collected.
    """
    sub = _node_subconstruct()
    parent, kwargs = _branch_parent_with_arm_sub(sub, arm="true")

    graph = compile(parent, **kwargs)
    result = run(graph, input={"node_id": "tdbb-true"})

    assert result["gate_sub"].disposition == "confirmed"
    assert result["gate_sub"].claim_id == "c1"


def test_node_subconstruct_in_false_arm_compiles_and_runs():
    """Same coverage for the FALSE arm (the second arm-iteration site)."""
    sub = _node_subconstruct()
    parent, kwargs = _branch_parent_with_arm_sub(sub, arm="false")

    graph = compile(parent, **kwargs)
    result = run(graph, input={"node_id": "tdbb-false"})

    assert result["gate_sub"].disposition == "confirmed"
    assert result["gate_sub"].claim_id == "c1"


def test_iter_nodes_yields_arm_nodes_exactly_once():
    """Direct unit test of the fix: iter_nodes descends into both arms and
    yields each arm node exactly once (guards against double-yield -- arm nodes
    are arm-exclusive, so no node appears at top level AND in an arm)."""
    from neograph.construct import iter_nodes

    @node(outputs=ArmResult)
    def true_gate(claim: ArmClaim) -> ArmResult:
        return ArmResult(claim_id=claim.claim_id, disposition="t")

    @node(outputs=ArmResult)
    def false_gate(claim: ArmClaim) -> ArmResult:
        return ArmResult(claim_id=claim.claim_id, disposition="f")

    true_sub = construct_from_functions("t-sub", [true_gate], input=ArmClaim, output=ArmResult)
    false_sub = construct_from_functions("f-sub", [false_gate], input=ArmClaim, output=ArmResult)

    register_scripted("seed2", lambda _in, _cfg: ArmClaim(claim_id="c", text="x"))
    seed = Node.scripted("seed", fn="seed2", outputs=ArmClaim)
    cond = _ConditionSpec(
        source_node=seed, attr_chain=["text"],
        op_fn=lambda value, _t: bool(value), op_str="route", threshold=None,
    )
    meta = _BranchMeta(
        condition_spec=cond,
        true_arm_nodes=[true_sub], false_arm_nodes=[false_sub],
    )
    parent = Construct("parent", nodes=[seed, _BranchNode(meta, 0)])

    names = [n.name for n in iter_nodes(parent)]
    # seed (top level) + true-gate (true arm) + false-gate (false arm), once
    # each. @node hyphenates the function name into the node name.
    assert sorted(names) == ["false-gate", "seed", "true-gate"]
    assert len(names) == len(set(names)), "an arm node was yielded more than once"


def test_arm_node_output_type_is_fingerprinted():
    """compute_node_fingerprints must include arm nodes (M1) -- otherwise a
    changed arm-node output type would not invalidate the checkpoint."""
    from neograph.state import compute_node_fingerprints

    sub = _node_subconstruct()
    parent, _ = _branch_parent_with_arm_sub(sub, arm="true")

    fingerprints = compute_node_fingerprints(parent)
    # The arm sub-construct's output surfaces under its field name.
    assert "gate_sub" in fingerprints, (
        "branch-arm sub-construct missing from node fingerprints -- checkpoint "
        "auto-rewind would be blind to arm-node output-type changes"
    )
