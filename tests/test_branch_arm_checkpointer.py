"""Branch-arm sub-construct checkpointer + condition threading (neograph-faf8).

The main sub-construct path (``compiler.py`` -> ``_add_subgraph``) recursively
compiles a sub-construct with ``checkpointer=checkpointer`` AND
``conditions=condition_lookup`` -- the codebase's own theory is that the main
path threads them *because they are needed*. The branch-arm path
(``_wiring.py::_add_branch_to_graph``) used to compile arm sub-constructs with
``checkpointer=None`` and no ``conditions=``, so a sub-construct carrying an
Operator (interrupt) inside a branch arm had NO persistence.

BEHAVIORAL, not consistency-only: the drop is caught at *compile time* by the
``compiler.py`` guard "Operator modifier requires a checkpointer" -- an Operator
sub-construct in a branch arm failed to compile at all, while the identical
sub-construct on the main path compiled and ran. After the fix it compiles,
interrupts mid-arm, and RESUMES from the interrupt (the pre-interrupt node runs
exactly once across the interrupt/resume boundary).

Real file-backed ``SqliteSaver`` per project convention -- checkpoint tests
never use in-memory fakes.

Three-surface parity note: branches only arise from ForwardConstruct tracing or
the programmatic ``_BranchNode`` sentinel, so the branch itself is always built
programmatically. The arm sub-construct is built declaratively
(``Node.scripted() | Operator``) with the interrupt condition supplied through
the per-compile ``conditions=`` dict -- this simultaneously exercises the
checkpointer threading AND the ``conditions=`` threading the fix added. The
``@node`` surface for arm sub-constructs is blocked by an INDEPENDENT gap
(``iter_nodes`` / ``_collect_scripted_shims`` skip ``_BranchNode`` sentinels, so
a scripted ``@node`` inside a branch arm is invisible to shim collection);
that is filed separately and is out of scope for faf8's checkpointer threading.
"""

from __future__ import annotations

from langgraph.checkpoint.sqlite import SqliteSaver

from neograph import (
    Construct,
    Node,
    Operator,
    compile,
    run,
)
from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
from tests.fakes import build_test_compile_kwargs, register_condition, register_scripted
from tests.schemas import Claims, ValidationResult


def _operator_subconstruct(name: str, probe_calls: list[str]) -> Construct:
    """A declarative sub-construct: probe -> gate, gate carrying an Operator
    interrupt whose condition lives ONLY in the per-compile ``conditions=``
    dict (never the global decoration registry), so it resolves iff the
    branch-arm compile threads ``conditions=``.
    """
    probe_fn_name = f"arm_probe_{name}"
    gate_fn_name = f"arm_gate_{name}"
    cond_name = f"arm_failed_{name}"

    def _probe(_in, _cfg):
        probe_calls.append("probe")
        return Claims(items=["probed"])

    register_scripted(probe_fn_name, _probe)
    register_scripted(
        gate_fn_name,
        lambda _in, _cfg: ValidationResult(passed=False, issues=["needs review"]),
    )
    register_condition(
        cond_name,
        lambda state: (
            {"issues": state.gate.issues}
            if getattr(state, "gate", None) and not state.gate.passed
            else None
        ),
    )

    probe_node = Node.scripted(
        "probe", fn=probe_fn_name, inputs=Claims, outputs=Claims,
    )
    gate_node = Node.scripted(
        "gate", fn=gate_fn_name, inputs=Claims, outputs=ValidationResult,
    ) | Operator(when=cond_name)
    return Construct(
        name, input=Claims, output=ValidationResult, nodes=[probe_node, gate_node],
    )


def _branch_parent(seed: Node, sub: Construct, *, arm: str) -> Construct:
    """Wrap ``sub`` in the chosen (``"true"``/``"false"``) arm of a programmatic
    branch fed by ``seed``. The op_fn is chosen so the branch always routes to
    the arm holding the sub-construct.
    """
    take_true = arm == "true"
    cond = _ConditionSpec(
        source_node=seed,
        attr_chain=["items"],
        # Route to whichever arm holds the sub: truthy seed output -> true arm
        # when arm=="true"; negated -> false arm when arm=="false".
        op_fn=(lambda value, _t: bool(value)) if take_true else (lambda value, _t: not value),
        op_str="route",
        threshold=None,
    )
    meta = _BranchMeta(
        condition_spec=cond,
        true_arm_nodes=[sub] if take_true else [],
        false_arm_nodes=[] if take_true else [sub],
    )
    return Construct("parent", nodes=[seed, _BranchNode(meta, 0)])


def _run_interrupt_resume(parent: Construct, sub_name: str, probe_calls: list[str], tmp_path, thread: str):
    """Compile with a real file-backed SqliteSaver, interrupt, resume, and
    assert the arm continued from the interrupt (probe ran exactly once)."""
    db = str(tmp_path / f"{thread}.db")
    config = {"configurable": {"thread_id": thread}}
    with SqliteSaver.from_conn_string(db) as saver:
        # Compiles: before the fix this raised CompileError "Operator modifier
        # requires a checkpointer" (checkpointer=None on the recursive
        # branch-arm compile).
        graph = compile(parent, checkpointer=saver, **build_test_compile_kwargs())

        paused = run(graph, input={"node_id": thread}, config=config)
        assert "__interrupt__" in paused
        assert probe_calls == ["probe"]

        resumed = run(graph, resume={"approved": True}, config=config)

    assert "__interrupt__" not in resumed
    # The sub-construct output surfaces at the parent under its name field.
    result = resumed[sub_name]
    assert isinstance(result, ValidationResult)
    assert result.passed is False
    assert probe_calls == ["probe"], "arm restarted instead of resuming"


def test_true_arm_operator_subconstruct_compiles_interrupts_and_resumes(tmp_path):
    """Operator sub-construct in the TRUE arm: compiles (checkpointer threaded),
    resolves its string interrupt condition (conditions threaded), interrupts
    mid-arm, and resumes from the interrupt."""
    probe_calls: list[str] = []
    sub = _operator_subconstruct("arm_true", probe_calls)
    register_scripted("seed_true", lambda _in, _cfg: Claims(items=["data"]))
    seed = Node.scripted("seed", fn="seed_true", outputs=Claims)
    parent = _branch_parent(seed, sub, arm="true")
    _run_interrupt_resume(parent, "arm_true", probe_calls, tmp_path, "faf8-true")


def test_false_arm_operator_subconstruct_compiles_interrupts_and_resumes(tmp_path):
    """Same as the true-arm case but the sub-construct lives in the FALSE arm,
    covering the second edited call site in ``_add_branch_to_graph``."""
    probe_calls: list[str] = []
    sub = _operator_subconstruct("arm_false", probe_calls)
    register_scripted("seed_false", lambda _in, _cfg: Claims(items=["data"]))
    seed = Node.scripted("seed", fn="seed_false", outputs=Claims)
    parent = _branch_parent(seed, sub, arm="false")
    _run_interrupt_resume(parent, "arm_false", probe_calls, tmp_path, "faf8-false")


def test_arm_operator_subconstruct_compiles_like_main_path(tmp_path):
    """Parity pin: the identical Operator sub-construct compiles both on the
    main path (as a direct child) AND inside a branch arm. Before the fix the
    branch-arm form raised while the main-path form compiled -- documenting the
    inconsistency the fix removes."""
    db = str(tmp_path / "parity.db")
    with SqliteSaver.from_conn_string(db) as saver:
        # Main path: Operator sub-construct as a direct child.
        main_probe: list[str] = []
        main_sub = _operator_subconstruct("main_sub", main_probe)
        register_scripted("seed_main", lambda _in, _cfg: Claims(items=["data"]))
        main_parent = Construct(
            "parent", nodes=[Node.scripted("seed", fn="seed_main", outputs=Claims), main_sub],
        )
        compile(main_parent, checkpointer=saver, **build_test_compile_kwargs())

        # Branch arm: same Operator sub-construct inside a true arm.
        arm_probe: list[str] = []
        arm_sub = _operator_subconstruct("arm_only", arm_probe)
        register_scripted("seed_arm", lambda _in, _cfg: Claims(items=["data"]))
        seed = Node.scripted("seed", fn="seed_arm", outputs=Claims)
        arm_parent = _branch_parent(seed, arm_sub, arm="true")
        # Must not raise -- this is the assertion.
        compile(arm_parent, checkpointer=saver, **build_test_compile_kwargs())
