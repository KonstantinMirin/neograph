"""Branch-arm nodes must be visible to the five NON-``iter_nodes`` tree walks
(neograph-vn5f).

Adjacent to neograph-tdbb, which fixed ``iter_nodes`` (construct.py) — the
single source of truth for the IR node-tree walk feeding scripted-shim / DI /
fingerprint collection. That fix does NOT reach five *hand-rolled*
``construct.nodes`` walks that never route through ``iter_nodes``:

1. ``_construct_validation._validate_node_chain`` — bare arm Nodes are never
   producer-registered nor type-validated at the parent level.
2. ``_ir_normalize.normalize_ir`` — ``if not isinstance(item, Node): continue``
   skips ``_BranchNode``, so ``fan_out_param`` / ``oracle_gen_type`` inference
   never runs for arm nodes.
3. ``lint._walk`` — ``_BranchNode`` falls through the ``isinstance(item, Node)``
   guard, so DI-binding + template lint never inspect arm nodes.
4. ``verify._has_llm_nodes`` / ``verify._walk`` — arm nodes skipped by the
   verify tree walk.
5. ``runner._build_producer_consumer_adjacency`` — arm nodes contribute no
   edges to the checkpoint-invalidation adjacency map.

Each root cause is identical: ``_BranchNode`` sentinels carry
``_BranchMeta.true_arm_nodes`` / ``false_arm_nodes`` that every non-arm-aware
walk ignores. Branches only arise from ForwardConstruct tracing or a
programmatic ``_BranchNode`` sentinel, so the branch itself is built
programmatically (mirroring ``test_branch_arm_iter_nodes.py``); the arm content
is a bare ``Node`` — the exact surface these walks make invisible.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import Construct, Each, Node, Operator, Tool, compile
from neograph._ir_branch import (
    _BranchMeta,
    _BranchNode,
    _ConditionSpec,
    iter_item_slots,
    iter_with_arms,
)
from neograph.construct import ConstructError
from neograph.errors import CompileError, ConfigurationError
from tests.fakes import (
    build_test_compile_kwargs,
    configure_fake_llm,
    register_scripted,
)


class ArmClaim(BaseModel, frozen=True):
    claim_id: str
    text: str


class ArmResult(BaseModel, frozen=True):
    claim_id: str
    disposition: str


class ArmGroup(BaseModel, frozen=True):
    label: str


def _seed() -> Node:
    """A scripted seed that produces ``ArmClaim`` under state field ``seed``."""
    return Node.scripted("seed", fn="arm_seed", outputs=ArmClaim)


def _branch_parent(arm_node: Node | Construct, *, arm: str = "true") -> Construct:
    """Wrap ``arm_node`` in the chosen arm of a ``_BranchNode`` under a parent
    Construct whose first node is the scripted seed.

    ``arm`` selects the true or false arm — both arms are distinct iteration
    sites in every migrated walk, so tests parametrize over both.
    """
    seed = _seed()
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
        true_arm_nodes=[arm_node] if take_true else [],
        false_arm_nodes=[] if take_true else [arm_node],
    )
    return Construct("parent", nodes=[seed, _BranchNode(meta, 0)])


def _arm_meta(parent: Construct) -> _BranchMeta:
    """The ``_BranchMeta`` of the single ``_BranchNode`` in ``parent``."""
    for item in parent.nodes:
        if isinstance(item, _BranchNode):
            return item._neo_branch_meta
    raise AssertionError("no _BranchNode in parent")


# ═══════════════════════════════════════════════════════════════════════════
# Site 1 — _construct_validation._validate_node_chain (producer registration
# + type validation)
# ═══════════════════════════════════════════════════════════════════════════

def test_arm_node_input_type_mismatch_is_caught_at_assembly():
    """A bare arm Node whose dict-form ``inputs`` type mismatches its upstream
    producer must raise ``ConstructError`` at parent assembly.

    ``seed`` produces ``ArmClaim`` (field ``seed``); the arm node declares
    ``inputs={'seed': ArmResult}`` — a type mismatch that the parent-level
    validator catches for a top-level node but silently skips for an arm node.
    """
    bad_arm = Node.scripted(
        "gate", fn="f", inputs={"seed": ArmResult}, outputs=ArmResult,
    )
    with pytest.raises(ConstructError):
        _branch_parent(bad_arm)


# ═══════════════════════════════════════════════════════════════════════════
# Site 2 — _ir_normalize.normalize_ir (fan_out_param inference)
# ═══════════════════════════════════════════════════════════════════════════

def test_arm_each_node_gets_fan_out_param_normalized():
    """An Each + dict-form arm Node must have ``fan_out_param`` inferred by
    ``normalize_ir`` (which runs in ``Construct.__init__``).

    The fan-out receiver is the single input key naming neither a peer producer
    nor the node itself — here ``group``. Because ``normalize_ir`` skips
    ``_BranchNode``, the arm node's ``fan_out_param`` stays ``None``.
    """
    each_arm = Node.scripted(
        "canonicalize", fn="f",
        inputs={"group": ArmGroup},
        outputs=ArmResult,
    ) | Each(over="seed.items", key="label")
    parent = _branch_parent(each_arm)

    arm_node = _arm_meta(parent).true_arm_nodes[0]
    assert arm_node.fan_out_param == "group", (
        "branch-arm Each node missed fan_out_param normalization — the fan-out "
        "receiver was never resolved, so the runtime extractor reads the wrong "
        "input"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Site 3 — lint._walk (DI binding checks)
# ═══════════════════════════════════════════════════════════════════════════

def test_arm_node_di_binding_gap_is_linted():
    """A DI binding on a bare arm Node must be inspected by ``lint()``.

    The @node function declares a ``FromInput`` param; linting with a config
    that omits that key must surface a ``from_input`` LintIssue. Because
    ``lint._walk`` skips ``_BranchNode``, the arm node is never inspected.
    """
    from typing import Annotated

    from neograph import FromInput, lint, node

    @node(outputs=ArmResult)
    def gate(seed: ArmClaim, topic: Annotated[str, FromInput]) -> ArmResult:
        return ArmResult(claim_id=seed.claim_id, disposition=topic)

    arm_node = gate  # the decorated Node instance
    parent = _branch_parent(arm_node)

    issues = lint(parent, config={"node_id": "x", "project_root": "/p"})
    topic_issues = [i for i in issues if i.param == "topic"]
    assert topic_issues, (
        "branch-arm node's FromInput binding was never linted — the missing "
        "'topic' config key produced no LintIssue"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Site 4 — verify._has_llm_nodes / verify._walk
# ═══════════════════════════════════════════════════════════════════════════

def test_llm_node_in_arm_is_seen_by_has_llm_nodes():
    """``_has_llm_nodes`` must return True when an LLM-mode node lives only in a
    branch arm — otherwise verify skips LLM-kwargs checks for arm-only LLM
    pipelines."""
    from neograph.verify import _has_llm_nodes

    llm_arm = Node(
        name="think-arm", mode="think",
        inputs={"seed": ArmClaim}, outputs=ArmResult,
    )
    parent = _branch_parent(llm_arm)

    assert _has_llm_nodes(parent) is True, (
        "LLM-mode node inside a branch arm was invisible to _has_llm_nodes"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Site 5 — runner._build_producer_consumer_adjacency
# ═══════════════════════════════════════════════════════════════════════════

def test_arm_node_contributes_edge_to_adjacency():
    """A bare arm Node's dict-form ``inputs`` must contribute a producer→consumer
    edge to the checkpoint-invalidation adjacency map — otherwise a change to
    ``seed`` would not mark the arm consumer for re-execution."""
    from neograph.runner import _build_producer_consumer_adjacency

    arm_node = Node.scripted(
        "gate", fn="f", inputs={"seed": ArmClaim}, outputs=ArmResult,
    )
    parent = _branch_parent(arm_node)

    adjacency = _build_producer_consumer_adjacency(parent)
    assert "gate" in adjacency.get("seed", set()), (
        "branch-arm consumer contributed no edge — 'seed' -> 'gate' is missing "
        "from the adjacency map, so checkpoint invalidation is blind to it"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Primitive properties — iter_with_arms / iter_item_slots
# ═══════════════════════════════════════════════════════════════════════════

def test_iter_with_arms_is_identity_when_no_branch():
    """Zero-regression property: over a construct with no _BranchNode,
    iter_with_arms yields exactly construct.nodes, in order. This is what makes
    migrating every read-only walk to the primitive safe."""
    a = Node.scripted("a", fn="f", outputs=ArmClaim)
    b = Node.scripted("b", fn="f", inputs={"a": ArmClaim}, outputs=ArmResult)
    parent = Construct("plain", nodes=[a, b])

    assert list(iter_with_arms(parent)) == parent.nodes


def test_iter_with_arms_yields_both_arms_once_each():
    """iter_with_arms expands a _BranchNode into true-arm then false-arm items,
    each exactly once (arm nodes are arm-exclusive)."""
    seed = _seed()
    true_node = Node.scripted("t", fn="f", inputs={"seed": ArmClaim}, outputs=ArmResult)
    false_node = Node.scripted("fa", fn="f", inputs={"seed": ArmClaim}, outputs=ArmResult)
    cond = _ConditionSpec(
        source_node=seed, attr_chain=["text"],
        op_fn=lambda v, _t: bool(v), op_str="route", threshold=None,
    )
    meta = _BranchMeta(condition_spec=cond, true_arm_nodes=[true_node], false_arm_nodes=[false_node])
    parent = Construct("parent", nodes=[seed, _BranchNode(meta, 0)])

    names = [n.name for n in iter_with_arms(parent)]
    assert names == ["seed", "t", "fa"]
    assert len(names) == len(set(names))


def test_iter_item_slots_write_back_targets_the_arm_meta_list():
    """A model_copy through an iter_item_slots slot must land in the ORIGINAL
    arm meta-list slot (meta.true_arm_nodes[j] / false_arm_nodes[j]), not a
    detached copy — otherwise the compiled arm never sees the rewrite."""
    seed = _seed()
    t = Node.scripted("t", fn="f", inputs={"seed": ArmClaim}, outputs=ArmResult)
    fa = Node.scripted("fa", fn="f", inputs={"seed": ArmClaim}, outputs=ArmResult)
    cond = _ConditionSpec(
        source_node=seed, attr_chain=["text"],
        op_fn=lambda v, _t: bool(v), op_str="route", threshold=None,
    )
    meta = _BranchMeta(condition_spec=cond, true_arm_nodes=[t], false_arm_nodes=[fa])
    parent = Construct("parent", nodes=[seed, _BranchNode(meta, 0)])

    # Rewrite every Node slot; assert the arm slots were mutated in place.
    for container, idx in iter_item_slots(parent):
        item = container[idx]
        if isinstance(item, Node):
            container[idx] = item.model_copy(update={"scripted_fn": "rewritten"})

    assert meta.true_arm_nodes[0].scripted_fn == "rewritten"
    assert meta.false_arm_nodes[0].scripted_fn == "rewritten"


# ═══════════════════════════════════════════════════════════════════════════
# False-arm parity — the second iteration site in every migrated walk
# ═══════════════════════════════════════════════════════════════════════════

def test_false_arm_each_node_gets_fan_out_param_normalized():
    """normalize_ir must reach the FALSE arm too (write-back into
    meta.false_arm_nodes)."""
    each_arm = Node.scripted(
        "canonicalize", fn="f", inputs={"group": ArmGroup}, outputs=ArmResult,
    ) | Each(over="seed.items", key="label")
    parent = _branch_parent(each_arm, arm="false")

    arm_node = _arm_meta(parent).false_arm_nodes[0]
    assert arm_node.fan_out_param == "group"


def test_false_arm_llm_node_is_seen_by_has_llm_nodes():
    """_has_llm_nodes must reach the FALSE arm too."""
    from neograph.verify import _has_llm_nodes

    llm_arm = Node(name="think-arm", mode="think", inputs={"seed": ArmClaim}, outputs=ArmResult)
    parent = _branch_parent(llm_arm, arm="false")

    assert _has_llm_nodes(parent) is True


# ═══════════════════════════════════════════════════════════════════════════
# Sites 6-9 — compile-time validation gaps for bare arm Nodes
# ═══════════════════════════════════════════════════════════════════════════

def test_arm_operator_unregistered_condition_raises_at_compile():
    """Site 6: a bare arm Operator node with a string condition that is not
    registered must raise ConfigurationError at compile()."""
    register_scripted("arm_seed", lambda _in, _cfg: ArmClaim(claim_id="c1", text="x"))
    register_scripted("f", lambda _in, _cfg: ArmResult(claim_id="c1", disposition="d"))
    op_arm = Node.scripted(
        "gate", fn="f", inputs={"seed": ArmClaim}, outputs=ArmResult,
    ) | Operator(when="unregistered_cond_9513")
    parent = _branch_parent(op_arm)

    with pytest.raises(ConfigurationError, match="unregistered_cond_9513"):
        compile(parent, **build_test_compile_kwargs())


def test_arm_operator_without_checkpointer_raises_at_compile():
    """Site 7 (highest-value): a bare arm Operator node with a REGISTERED
    condition still requires a checkpointer — compile() with checkpointer=None
    must raise CompileError, not defer the failure to runtime."""
    register_scripted("arm_seed", lambda _in, _cfg: ArmClaim(claim_id="c1", text="x"))
    register_scripted("f", lambda _in, _cfg: ArmResult(claim_id="c1", disposition="d"))
    op_arm = Node.scripted(
        "gate", fn="f", inputs={"seed": ArmClaim}, outputs=ArmResult,
    ) | Operator(when="reg_cond")
    parent = _branch_parent(op_arm)

    kwargs = build_test_compile_kwargs(conditions={"reg_cond": lambda d: True})
    with pytest.raises(CompileError, match="checkpointer"):
        compile(parent, checkpointer=None, **kwargs)


def test_arm_agent_node_missing_tool_factory_raises_at_compile():
    """Site 8: a bare arm agent-mode node with a tool that has no registered
    factory must raise CompileError at compile()."""
    register_scripted("arm_seed", lambda _in, _cfg: ArmClaim(claim_id="c1", text="x"))
    llm_kw = configure_fake_llm(lambda tier: None)
    agent_arm = Node(
        name="researcher", mode="agent",
        inputs={"seed": ArmClaim}, outputs=ArmResult,
        model="fast", prompt="test",
        tools=[Tool("missing_tool_9513", budget=3)],
    )
    parent = _branch_parent(agent_arm)

    with pytest.raises(CompileError, match="missing_tool_9513"):
        compile(parent, **build_test_compile_kwargs(), **llm_kw)


def test_arm_llm_node_inherits_parent_llm_config():
    """Site 9: a bare arm LLM node must inherit the parent Construct's
    llm_config override (Construct.__init__ inheritance pass), otherwise it runs
    with the wrong output_strategy."""
    llm_arm = Node(name="think-arm", mode="think", inputs={"seed": ArmClaim}, outputs=ArmResult)
    seed = _seed()
    cond = _ConditionSpec(
        source_node=seed, attr_chain=["text"],
        op_fn=lambda v, _t: bool(v), op_str="route", threshold=None,
    )
    meta = _BranchMeta(condition_spec=cond, true_arm_nodes=[llm_arm], false_arm_nodes=[])
    # Parent declares an llm_config override that children must inherit.
    parent = Construct(
        "parent", nodes=[seed, _BranchNode(meta, 0)],
        llm_config={"output_strategy": "json_mode"},
    )

    arm_node = _arm_meta(parent).true_arm_nodes[0]
    assert arm_node.llm_config.output_strategy == "json_mode", (
        "branch-arm LLM node did not inherit the parent Construct's llm_config"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Documented limitation — cross-arm producer leakage is NOT caught
# ═══════════════════════════════════════════════════════════════════════════

def test_cross_arm_producer_leakage_is_not_flagged():
    """Pins the documented site-1 limitation: arms are flattened without
    recording which arm each producer belongs to, so a false-arm node reading a
    true-arm node's output is NOT rejected. This is a known limitation (not a
    regression — it was uncaught before arm nodes were validated at all); this
    test documents it so a future reader does not assume completeness."""
    seed = _seed()
    true_producer = Node.scripted("tprod", fn="f", inputs={"seed": ArmClaim}, outputs=ArmResult)
    # false-arm node reads the TRUE arm's output field 'tprod' — conditionally
    # unsatisfiable at runtime, but the flattened validator accepts it.
    false_consumer = Node.scripted(
        "fcons", fn="f", inputs={"tprod": ArmResult}, outputs=ArmResult,
    )
    cond = _ConditionSpec(
        source_node=seed, attr_chain=["text"],
        op_fn=lambda v, _t: bool(v), op_str="route", threshold=None,
    )
    meta = _BranchMeta(
        condition_spec=cond,
        true_arm_nodes=[true_producer],
        false_arm_nodes=[false_consumer],
    )
    # Must NOT raise — documents the flattening limitation.
    Construct("parent", nodes=[seed, _BranchNode(meta, 0)])
