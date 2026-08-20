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

import pytest
from pydantic import BaseModel

from neograph import Construct, Each, Loop, Node, Oracle, compile, construct_from_functions, node
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


# ═══════════════════════════════════════════════════════════════════════════
# Every modifier placement (GH #6 follow-up, neograph-4o1cn)
# ═══════════════════════════════════════════════════════════════════════════
#
# 0.7.7 fixed the two make_subgraph_fn call sites and PROVED the modifier'd
# paths with a single Loop case, then generalised to the whole family. Loop
# recovers because _add_subgraph_loop passes subgraph_fn through unwrapped;
# Each does NOT -- it re-wraps at its own `named()` site in _wire_each, which is
# shared between the Node and Construct paths. So Each-modified sub-constructs
# stayed invisible through 0.7.7.
#
# The lesson is the parametrization: prove EVERY placement, not a representative.


class Group(BaseModel):
    label: str


class Groups(BaseModel):
    groups: list[Group]


def _sub(name: str, fn_key: str) -> Construct:
    from tests.fakes import register_scripted

    register_scripted(fn_key, lambda _in, _cfg: Finding(note="n"))
    return Construct(
        name,
        input=Group,
        output=Finding,
        nodes=[Node.scripted(f"{name}-inner", fn=fn_key, outputs=Finding)],
    )


def _each_placement() -> tuple[Construct, str]:
    """A top-level Each-modified sub-construct -- the reporter's `fan_outer`."""
    from tests.fakes import register_scripted

    register_scripted("intro_src", lambda _in, _cfg: Groups(groups=[Group(label="a")]))
    sub = _sub("faneach", "intro_each_inner") | Each(over="source.groups", key="label")
    return (
        Construct(
            "each-outer",
            nodes=[Node.scripted("source", fn="intro_src", outputs=Groups), sub],
        ),
        "faneach",
    )


def _oracle_placement() -> tuple[Construct, str]:
    """An Oracle-modified sub-construct -- the sibling redirect path, which the
    0.7.7 sweep also never probed."""
    from tests.fakes import register_scripted

    register_scripted("intro_ora_seed", lambda _in, _cfg: Group(label="a"))
    register_scripted("intro_ora_merge", lambda variants, _cfg: Finding(note="merged"))
    sub = _sub("oraclesub", "intro_oracle_inner") | Oracle(n=2, merge_fn="intro_ora_merge")
    return (
        Construct(
            "oracle-outer",
            nodes=[Node.scripted("seed", fn="intro_ora_seed", outputs=Group), sub],
        ),
        "oraclesub",
    )


def _loop_placement() -> tuple[Construct, str]:
    from tests.fakes import register_scripted

    register_scripted("intro_loop_seed", lambda _in, _cfg: Group(label="a"))
    register_scripted("intro_loop_inner", lambda _in, _cfg: Group(label="a"))
    sub = Construct(
        "loopsub",
        input=Group,
        output=Group,
        nodes=[Node.scripted("loopsub-inner", fn="intro_loop_inner", outputs=Group)],
    ) | Loop(when=lambda d: False, max_iterations=2)
    return (
        Construct(
            "loop-outer",
            nodes=[Node.scripted("seed", fn="intro_loop_seed", outputs=Group), sub],
        ),
        "loopsub",
    )


def _plain_placement() -> tuple[Construct, str]:
    from tests.fakes import register_scripted

    register_scripted("intro_plain_seed", lambda _in, _cfg: Group(label="a"))
    return (
        Construct(
            "plain-outer",
            nodes=[
                Node.scripted("seed", fn="intro_plain_seed", outputs=Group),
                _sub("plainsub", "intro_plain_inner"),
            ],
        ),
        "plainsub",
    )


PLACEMENTS = [
    ("plain", _plain_placement),
    ("loop", _loop_placement),
    ("each", _each_placement),
    ("oracle", _oracle_placement),
]


class TestEveryModifierPlacementStaysVisible:
    """A sub-construct must be enumerable wherever it is wired.

    Parametrized deliberately: 0.7.7 shipped a regression precisely because one
    placement was proved and the family was reported (neograph-4o1cn / GH #6).
    """

    @pytest.mark.parametrize(
        ("label", "factory"), PLACEMENTS, ids=[p[0] for p in PLACEMENTS]
    )
    def test_sub_construct_is_enumerated(self, label: str, factory):
        pipeline, expected = factory()

        graph = compile(pipeline, **build_test_compile_kwargs())
        found = [name for name, _ in graph.graph.get_subgraphs()]

        assert expected in found, (
            f"the {label}-modified sub-construct {expected!r} is invisible to "
            f"LangGraph (get_subgraphs -> {found}). Something on that wiring path "
            "config-binds the runnable that holds the nested Pregel."
        )


class TestNestedAndRecursiveDiscovery:
    """The reporter's exact shape on GH #6: an Each nested inside a Loop
    sub-construct, plus a second Each at top level. ``recurse=True`` must reach
    the inner one -- 0.7.7 returned only the Loop."""

    def _nested(self) -> Construct:
        from tests.fakes import register_scripted

        register_scripted("nest_src", lambda _in, _cfg: Groups(groups=[Group(label="a")]))
        register_scripted("nest_inner", lambda _in, _cfg: Group(label="a"))
        register_scripted("nest_leaf", lambda _in, _cfg: Group(label="a"))

        # fan_inner: an Each sub-construct living INSIDE loop_sub.
        fan_inner = Construct(
            "fan_inner",
            input=Group,
            output=Group,
            nodes=[Node.scripted("fan-inner-leaf", fn="nest_leaf", outputs=Group)],
        ) | Each(over="neo_subgraph_input.groups", key="label")

        loop_sub = Construct(
            "loop_sub",
            input=Groups,
            output=Groups,
            nodes=[Node.scripted("loop-head", fn="nest_inner", outputs=Groups)],
        ) | Loop(when=lambda d: False, max_iterations=2)

        fan_outer = Construct(
            "fan_outer",
            input=Group,
            output=Group,
            nodes=[Node.scripted("fan-outer-leaf", fn="nest_leaf", outputs=Group)],
        ) | Each(over="source.groups", key="label")

        return Construct(
            "nested-outer",
            nodes=[
                Node.scripted("source", fn="nest_src", outputs=Groups),
                loop_sub,
                fan_outer,
            ],
        )

    def test_top_level_each_and_loop_are_both_enumerated(self):
        graph = compile(self._nested(), **build_test_compile_kwargs())

        found = {name for name, _ in graph.graph.get_subgraphs()}

        assert {"loop_sub", "fan_outer"} <= found, (
            f"expected both the Loop and the Each sub-construct, got {sorted(found)}"
        )

    def test_recursive_discovery_reaches_them(self):
        """``get_subgraphs(recurse=True)`` returned only the Loop in 0.7.7."""
        graph = compile(self._nested(), **build_test_compile_kwargs())

        found = {name for name, _ in graph.graph.get_subgraphs(recurse=True)}

        assert "fan_outer" in found, (
            f"recursive discovery missed the Each sub-construct: {sorted(found)}"
        )

    def test_xray_expands_the_each_interior(self):
        graph = compile(self._nested(), **build_test_compile_kwargs())

        nodes = set(graph.graph.get_graph(xray=True).nodes)

        assert any("fan-outer-leaf" in n for n in nodes), (
            f"the Each fan-out's interior is still an opaque box: {sorted(nodes)}"
        )
