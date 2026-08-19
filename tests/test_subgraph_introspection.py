"""Sub-constructs must be visible to LangGraph's own introspection.

neograph-xunot (GH issue #6). A neograph sub-construct compiles to a real nested
Pregel, but every consumer that discovers nesting through LangGraph's API --
``get_subgraphs()``, ``get_graph(xray=True)``, ``Graph.to_json()``, LangGraph
Studio, Langfuse's agent-graph view -- rendered it as one opaque box.

The cause is on neograph's side: ``_trace.named()`` wraps the sub-construct's
runnable in ``.with_config(...)``, producing a ``RunnableBinding``, and
``langgraph.pregel._utils.find_subgraph_pregel`` walks ``RunnableSequence.steps``,
``RunnableLambda.deps`` and ``RunnableCallable`` nonlocals but never unwraps
``RunnableBinding.bound`` -- so the nested Pregel is unreachable from the parent.

These tests drive the LangGraph API a Studio/Langfuse consumer actually calls.
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import Construct, Loop, Node, compile, construct_from_functions, node
from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
from tests.fakes import build_test_compile_kwargs, register_scripted


class Claim(BaseModel):
    text: str


class Finding(BaseModel):
    note: str


def _cascade():
    """seed -> [verify: explore] -> report, the GH issue's reproduction shape."""

    @node(outputs=Claim)
    def seed() -> Claim:
        return Claim(text="c")

    @node(outputs=Finding)
    def explore(claim: Claim) -> Finding:
        return Finding(note=f"explored {claim.text}")

    verify = construct_from_functions("verify", [explore], input=Claim, output=Finding)

    @node(outputs=Finding)
    def report(verify: Finding) -> Finding:
        return Finding(note=verify.note)

    return construct_from_functions("cascade", [seed, verify, report])


class TestSubgraphIntrospection:
    """LangGraph's nesting-discovery API must see a neograph sub-construct."""

    def test_get_subgraphs_enumerates_the_sub_construct(self):
        graph = compile(_cascade(), **build_test_compile_kwargs())

        names = [name for name, _ in graph.graph.get_subgraphs()]

        assert names == ["verify"], (
            "get_subgraphs() did not find the 'verify' sub-construct. The parent's "
            "node runnable is wrapped so LangGraph cannot reach the nested Pregel "
            "(neograph-xunot / GH issue #6)."
        )

    def test_xray_expands_the_sub_construct_interior(self):
        graph = compile(_cascade(), **build_test_compile_kwargs())

        nodes = set(graph.graph.get_graph(xray=True).nodes)

        assert "verify:explore" in nodes, (
            f"xray did not expand 'verify' -- it rendered as one opaque box. Nodes: {sorted(nodes)}"
        )
        assert "verify" not in nodes, (
            "'verify' should be replaced by its expanded interior under xray, not kept as a box"
        )

    def test_to_json_carries_the_expanded_topology(self):
        """The serialized form Studio and other tooling consume."""
        graph = compile(_cascade(), **build_test_compile_kwargs())

        payload = graph.graph.get_graph(xray=True).to_json()

        rendered = repr(payload)
        assert "verify:explore" in rendered, (
            f"to_json() lost the sub-construct interior: {rendered[:400]}"
        )


class TestSubgraphIntrospectionAcrossPlacements:
    """The fix must cover every place a sub-construct can be wired, not only the
    motivating one -- the scan found a second direct site (branch arms) and a
    family of modifier'd sites that wrap the subgraph runnable one level further."""

    def test_branch_arm_sub_construct_is_enumerated(self):
        """Second direct site: ``_wiring._add_arm_nodes``. Fixing only the
        top-level path in ``compiler.py`` would leave this invisible."""
        register_scripted("intro_arm_probe", lambda _in, _cfg: Finding(note="p"))
        register_scripted("intro_seed", lambda _in, _cfg: Claim(text="c"))

        seed = Node.scripted("seed", fn="intro_seed", outputs=Claim)
        arm_sub = Construct(
            "armsub",
            input=Claim,
            output=Finding,
            nodes=[Node.scripted("probe", fn="intro_arm_probe", inputs=Claim, outputs=Finding)],
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
        parent = Construct("parent", nodes=[seed, _BranchNode(meta, 0)])

        graph = compile(parent, **build_test_compile_kwargs())

        assert [name for name, _ in graph.graph.get_subgraphs()] == ["armsub"], (
            "a sub-construct inside a branch arm is still invisible -- _wiring's "
            "add_node site binds it. See neograph-xunot / GH #6."
        )

    def test_loop_modified_sub_construct_is_enumerated(self):
        """Indirect site: a modifier wraps the subgraph runnable in a redirect
        closure. The scan proved this was equally broken before the fix; the
        prediction was that unbinding the inner runnable makes the closure
        walkable again. This asserts the prediction instead of assuming it."""

        @node(outputs=Claim)
        def seed() -> Claim:
            return Claim(text="c")

        @node(outputs=Finding)
        def explore(claim: Claim) -> Finding:
            return Finding(note="e")

        verify = construct_from_functions("verify", [explore], input=Claim, output=Finding)
        looped = verify | Loop(when=lambda d: False, max_iterations=2)

        @node(outputs=Finding)
        def report(verify: Finding) -> Finding:
            return Finding(note="r")

        graph = compile(
            construct_from_functions("cascade", [seed, looped, report]),
            **build_test_compile_kwargs(),
        )

        assert [name for name, _ in graph.graph.get_subgraphs()] == ["verify"], (
            "a Loop-modified sub-construct is still invisible -- the redirect "
            "closure captures a bound runnable. See neograph-xunot / GH #6."
        )


class TestSubgraphTraceHygiene:
    """The fix must not silently pay the price the ticket wrongly feared."""

    def test_span_run_name_is_still_the_construct_name(self):
        """Unbinding the node runnable removes the ``run_name`` that ``named``
        used to set, so ``make_subgraph_fn`` sets it via ``RunnableLambda(name=)``
        instead. Without that the span would read ``subgraph_node`` -- neograph's
        internals leaking into every consumer's trace tree, the exact hygiene
        debt neograph-3fm1 paid off."""
        from langchain_core.callbacks import BaseCallbackHandler

        from neograph import run

        seen: list[str] = []

        class Recorder(BaseCallbackHandler):
            def on_chain_start(self, serialized, inputs, **kw):
                name = kw.get("name") or (serialized or {}).get("name")
                node_name = (kw.get("metadata") or {}).get("langgraph_node")
                if node_name == "verify" and name:
                    seen.append(name)

        graph = compile(_cascade(), **build_test_compile_kwargs())
        run(graph, input={"node_id": "x"}, config={"callbacks": [Recorder()]})

        assert seen, "no span recorded for the sub-construct node"
        assert "subgraph_node" not in seen, (
            f"the inner function name leaked into the trace tree: {seen}"
        )
        assert "verify" in seen, f"span run_name is not the construct name: {seen}"

    def test_sub_construct_node_carries_neograph_metadata(self):
        """``named`` used to attach the neograph_* metadata; it now rides
        ``add_node(metadata=...)``. Tags are a KNOWN GAP (add_node accepts none),
        so metadata is the only channel a backend can index this node by."""
        graph = compile(_cascade(), **build_test_compile_kwargs())

        meta = graph.graph.nodes["verify"].metadata or {}

        assert meta.get("neograph_node") == "verify"
        assert meta.get("neograph_mode") == "subgraph"
        assert meta.get("neograph_output_type") == "Finding"
