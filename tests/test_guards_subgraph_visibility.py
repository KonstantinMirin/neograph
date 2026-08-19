"""Structural guard: a runnable that HOLDS a compiled Pregel is never config-bound.

neograph-xunot / GH issue #6. LangGraph discovers nesting by walking INTO a
node's runnable -- ``langgraph.pregel._utils.find_subgraph_pregel`` follows
``RunnableSequence.steps``, ``RunnableLambda.deps`` and ``RunnableCallable``
nonlocals. It has no branch for ``RunnableBinding.bound``, and
``Runnable.with_config(...)`` -- LangChain's only way to attach tags/run_name --
returns exactly a ``RunnableBinding``. So binding a sub-construct's node runnable
hid its interior from ``get_subgraphs()`` / ``get_graph(xray=True)`` /
``to_json()``, and therefore from Studio and Langfuse's agent-graph view.

The failure is SILENT: the graph still runs correctly and produces correct
results. Only introspection degrades, which is why nothing in the suite noticed
for several releases. That is the whole argument for pinning it.

The ratchet is BEHAVIOURAL rather than a grep for ``named(``. The disease is
about what LangGraph can REACH at runtime, not about which text appears at a call
site: a future refactor could re-introduce the binding through a helper, a
decorator, or a ``RunnableSequence``, and a text scan would pass while nesting
went invisible again. So this guard compiles real nested graphs and asks
LangGraph itself.
"""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.pregel._utils import find_subgraph_pregel
from pydantic import BaseModel

from neograph import Construct, Loop, Node, compile, construct_from_functions, node
from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
from tests.fakes import build_test_compile_kwargs, register_scripted


class Claim(BaseModel):
    text: str


class Finding(BaseModel):
    note: str


def _sub() -> Construct:
    @node(outputs=Finding)
    def explore(claim: Claim) -> Finding:
        return Finding(note="e")

    return construct_from_functions("verify", [explore], input=Claim, output=Finding)


def _cascade(modifier=None) -> Construct:
    @node(outputs=Claim)
    def seed() -> Claim:
        return Claim(text="c")

    verify = _sub()
    if modifier is not None:
        verify = verify | modifier

    @node(outputs=Finding)
    def report(verify: Finding) -> Finding:
        return Finding(note="r")

    return construct_from_functions("cascade", [seed, verify, report])


def _branch_parent() -> Construct:
    register_scripted("guard_seed", lambda _in, _cfg: Claim(text="c"))
    register_scripted("guard_probe", lambda _in, _cfg: Finding(note="p"))

    seed = Node.scripted("seed", fn="guard_seed", outputs=Claim)
    arm_sub = Construct(
        "armsub",
        input=Claim,
        output=Finding,
        nodes=[Node.scripted("probe", fn="guard_probe", inputs=Claim, outputs=Finding)],
    )
    meta = _BranchMeta(
        condition_spec=_ConditionSpec(
            source_node=seed,
            attr_chain=["text"],
            op_fn=lambda value, _t: bool(value),
            op_str="route",
            threshold=None,
        ),
        true_arm_nodes=[arm_sub],
        false_arm_nodes=[],
    )
    return Construct("parent", nodes=[seed, _BranchNode(meta, 0)])


# (label, construct factory, the sub-construct name LangGraph must enumerate)
PLACEMENTS = [
    ("top-level", lambda: _cascade(), "verify"),
    ("loop-modified", lambda: _cascade(Loop(when=lambda d: False, max_iterations=2)), "verify"),
    ("branch-arm", _branch_parent, "armsub"),
]


class TestNestedPregelStaysReachable:
    """Every placement of a sub-construct must remain visible to LangGraph."""

    @pytest.mark.parametrize(("label", "factory", "expected"), PLACEMENTS, ids=[p[0] for p in PLACEMENTS])
    def test_sub_construct_is_enumerated_by_langgraph(self, label: str, factory, expected: str):
        graph = compile(factory(), **build_test_compile_kwargs())

        found = [name for name, _ in graph.graph.get_subgraphs()]

        assert found == [expected], (
            f"the {label} sub-construct is invisible to LangGraph (got {found}). "
            "Something re-wrapped the node runnable that holds the nested Pregel -- "
            "`.with_config(...)` returns a RunnableBinding, which "
            "`find_subgraph_pregel` cannot see through. Attach run_name via "
            "`RunnableLambda(name=...)` and metadata via `add_node(metadata=...)` "
            "instead. See neograph-xunot / GH #6."
        )

    # --- meta-tests: prove the detector actually detects ---

    def test_detector_sees_an_unbound_pregel_holder(self):
        """Positive meta-test: the shape neograph now produces IS reachable, and
        what the walker reaches is the child graph ITSELF -- not merely
        something truthy."""
        runnable, compiled = _make_holder(bind=False, want_child=True)

        found = find_subgraph_pregel(runnable)

        assert found is compiled.graph, (
            f"walker reached {found!r}, expected the compiled sub-graph itself"
        )
        assert sorted(found.nodes) == ["__start__", "explore"], (
            f"the reached graph is not the 'verify' sub-construct: {sorted(found.nodes)}"
        )

    def test_detector_loses_a_bound_pregel_holder(self):
        """Negative meta-test, and the whole mechanism in three lines: the ONLY
        difference is `.with_config(...)`, and it is enough to hide the graph."""
        runnable = _make_holder(bind=True)
        assert find_subgraph_pregel(runnable) is None

    def test_binding_is_the_shape_langgraph_cannot_walk(self):
        """'Would-be-missed' meta-test: pin WHY, not just THAT. A guard that only
        knew `bound -> None` would still pass if LangGraph started unwrapping
        bindings, silently making itself vacuous. This asserts the wrapper type
        and that its `.bound` still holds the reachable runnable, so the day
        upstream grows a RunnableBinding branch this test fails and tells the
        reader the workaround can be deleted."""
        from langchain_core.runnables import RunnableBinding

        bound = _make_holder(bind=True)
        assert isinstance(bound, RunnableBinding)
        assert find_subgraph_pregel(bound.bound) is not None, (
            "the Pregel is reachable one level in; only the wrapper hides it"
        )


def _make_holder(*, bind: bool, want_child: bool = False):
    """A RunnableLambda closing over a compiled sub-graph -- neograph's shape.

    With ``want_child`` also returns the compiled child, so a meta-test can
    assert the walker reached THAT object rather than merely something truthy.
    """
    from neograph._subconstruct import make_subgraph_fn

    sub = _sub()
    compiled = compile(sub, **build_test_compile_kwargs())
    fn: RunnableLambda = make_subgraph_fn(sub, compiled.graph)
    holder = fn.with_config(tags=["t"]) if bind else fn
    return (holder, compiled) if want_child else holder
