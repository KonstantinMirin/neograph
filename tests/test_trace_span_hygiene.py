"""Trace-span hygiene: the engine's own callback spans read as the user's node
names, carry node metadata, and never leak neograph's internal wrapper function
names (neograph-3fm1).

LangGraph already fires ``on_chain_start``/``on_chain_end`` per node to any
handler in ``config["callbacks"]`` — a Langfuse / OTEL-bridge handler already
receives the full DAG tree. Before neograph-3fm1 every explicitly-constructed
wrapper ``RunnableLambda`` emitted a CHILD span named after its inner function
(``node_wrapper``, ``subgraph_node``, ``oracle_redirect_fn``, ``agent_sync`` …),
bleeding neograph internals into every consumer's trace tree and carrying none of
the node attributes neograph already computes.

These tests are the committed form of the ad-hoc ``cb_span_probe.py``: a
``BaseCallbackHandler`` records every chain-start span's name + tags + metadata,
then asserts (1) NO internal wrapper name appears anywhere, (2) the user's node
names ARE present, (3) node metadata (mode / output type / id) is attached for
backend indexing — across scripted / think / agent modes, a branch, and a
sub-construct. A final test pins that the naming is a STATIC config binding
(``.with_config``), so there is zero runtime cost when no callbacks are attached.

This is a driver/runtime tracing feature (a late config binding on the graph-node
runnables), NOT an IR-shape change — so the three-surface parity rule does not
apply here, the same exemption documented in ``test_async_observability.py``.
"""

from __future__ import annotations

import types
from typing import Annotated

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import RunnableBinding
from pydantic import BaseModel

from neograph import (
    Construct,
    FromInput,
    Node,
    Tool,
    compile,
    construct_from_module,
    node,
    run,
)
from neograph._trace import NODE_TAG, named
from tests.fakes import (
    FakeTool,
    ReActFake,
    StructuredFake,
    build_test_compile_kwargs,
    configure_fake_llm,
)

# --------------------------------------------------------------------------- #
# Internal wrapper function names that must NEVER surface in a consumer's trace #
# tree. Each is the ``__name__`` of a RunnableLambda body that gets added as a  #
# graph node somewhere in the compiler/factory/wiring layer.                   #
# --------------------------------------------------------------------------- #
LEAKING_WRAPPER_NAMES = frozenset(
    {
        "node_wrapper",
        "anode_wrapper",
        "raw_node_wrapper",
        "araw_node_wrapper",
        "subgraph_node",
        "asubgraph_node",
        "oracle_redirect_fn",
        "aoracle_redirect_fn",
        "eachoracle_redirect_fn",
        "each_redirect_fn",
        "merge_fn",
        "amerge_fn",
        "group_merge_barrier",
        "agroup_merge_barrier",
        "agent_sync",
        "agent_async",
        "tools_sync",
        "tools_async",
        "parse_sync",
        "parse_async",
    }
)


# --------------------------------------------------------------------------- #
# Local schemas (self-contained — this suite owns its models)                  #
# --------------------------------------------------------------------------- #
class Doc(BaseModel):
    text: str


class Analysis(BaseModel):
    summary: str


class SubIn(BaseModel):
    payload: str


class SubOut(BaseModel):
    enriched: str


class Confidence(BaseModel):
    score: float


class HighResult(BaseModel):
    label: str


class LowResult(BaseModel):
    label: str


# --------------------------------------------------------------------------- #
# Recording callback handler                                                   #
# --------------------------------------------------------------------------- #
class SpanProbe(BaseCallbackHandler):
    """Captures every chain-start span as ``(name, tags, metadata)``.

    Mirrors what a real Langfuse / OTEL-bridge callback handler receives from
    LangGraph's per-node ``on_chain_start`` events plus the wrapper runnables'
    own child spans.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[str | None, list, dict]] = []

    def on_chain_start(self, serialized, inputs, **kwargs):  # noqa: ANN001
        name = (serialized or {}).get("name") or kwargs.get("name")
        self.rows.append((name, kwargs.get("tags") or [], kwargs.get("metadata") or {}))

    @property
    def names(self) -> list[str]:
        return [n for (n, _t, _m) in self.rows if n]

    def metadata_for(self, name: str) -> dict:
        """Merged metadata across every span with the given name.

        A neograph node surfaces as two spans sharing the name: LangGraph's outer
        Pregel node span (``langgraph_*`` metadata) and our inner ``named``
        binding span (``neograph_*`` metadata). Merge so the caller sees both.
        """
        merged: dict = {}
        for n, _t, m in self.rows:
            if n == name:
                merged.update(m)
        return merged

    def tags_for(self, name: str) -> list:
        tags: list = []
        for n, t, _m in self.rows:
            if n == name:
                tags.extend(t)
        return tags


def _assert_no_wrapper_leak(probe: SpanProbe) -> None:
    leaked = LEAKING_WRAPPER_NAMES.intersection(probe.names)
    assert not leaked, f"internal wrapper names leaked into trace spans: {sorted(leaked)}"


# --------------------------------------------------------------------------- #
# Scripted mode                                                                #
# --------------------------------------------------------------------------- #
def test_scripted_node_span_named_and_carries_metadata_when_run_with_callbacks():
    """A scripted node's engine span reads as the node name (not ``node_wrapper``)
    and carries mode + declared output type as metadata."""

    @node(outputs=Analysis)
    def summarize(text: Annotated[str, FromInput]) -> Analysis:
        return Analysis(summary=f"summary::{text}")

    mod = types.ModuleType("trace_scripted_mod")
    mod.summarize = summarize
    graph = compile(construct_from_module(mod, name="scripted-trace"), **build_test_compile_kwargs())

    probe = SpanProbe()
    result = run(graph, input={"text": "hello"}, config={"callbacks": [probe]})

    assert result["summarize"].summary == "summary::hello"
    _assert_no_wrapper_leak(probe)
    assert "summarize" in probe.names

    meta = probe.metadata_for("summarize")
    assert meta.get("neograph_node") == "summarize"
    assert meta.get("neograph_mode") == "scripted"
    assert meta.get("neograph_output_type") == "Analysis"
    assert NODE_TAG in probe.tags_for("summarize")


# --------------------------------------------------------------------------- #
# Think mode                                                                   #
# --------------------------------------------------------------------------- #
def test_think_node_span_named_when_run_with_callbacks():
    """A think (LLM) node's engine span reads as the node name and is tagged with
    its LLM mode — no ``node_wrapper`` leak from the dead-body wrapper."""
    llm_kw = configure_fake_llm(
        lambda tier: StructuredFake(lambda m: m(summary="fake-analysis"))
    )

    @node(mode="think", outputs=Analysis, model="fast", prompt="test")
    def analyze(text: Annotated[str, FromInput]) -> Analysis: ...

    mod = types.ModuleType("trace_think_mod")
    mod.analyze = analyze
    graph = compile(
        construct_from_module(mod, name="think-trace"),
        **build_test_compile_kwargs(),
        **llm_kw,
    )

    probe = SpanProbe()
    result = run(graph, input={"text": "hi"}, config={"callbacks": [probe]})

    assert result["analyze"].summary == "fake-analysis"
    _assert_no_wrapper_leak(probe)
    assert "analyze" in probe.names
    assert probe.metadata_for("analyze").get("neograph_mode") == "think"


# --------------------------------------------------------------------------- #
# Agent mode (inline ReAct cycle)                                              #
# --------------------------------------------------------------------------- #
def test_agent_cycle_spans_named_when_run_with_callbacks():
    """The agent cycle's three bodies surface as ``{node}__agent`` /
    ``{node}__tools`` / ``{node}__parse`` — not the leaking ``agent_sync`` etc."""
    from tests.fakes import register_tool_factory

    search = FakeTool("search", response="found")
    register_tool_factory("search", lambda config, tool_config: search)

    fake = ReActFake(
        tool_calls=[
            [{"name": "search", "args": {"q": "x"}, "id": "c1"}],
            [],  # stop -> forced final
        ],
        final=lambda m: m(summary="agent-done"),
    )
    llm_kw = configure_fake_llm(lambda tier: fake)

    @node(mode="agent", outputs=Analysis, model="reason", prompt="test", tools=[Tool(name="search", budget=2)])
    def explore() -> Analysis: ...

    mod = types.ModuleType("trace_agent_mod")
    mod.explore = explore
    graph = compile(
        construct_from_module(mod, name="agent-trace"),
        **build_test_compile_kwargs(),
        **llm_kw,
    )

    probe = SpanProbe()
    run(graph, input={"node_id": "agent-trace-1"}, config={"callbacks": [probe]})

    _assert_no_wrapper_leak(probe)
    # The three ReAct-cycle parent nodes read as the user's node name, suffixed.
    assert "explore__agent" in probe.names
    assert "explore__tools" in probe.names
    assert "explore__parse" in probe.names


# --------------------------------------------------------------------------- #
# Branch (ForwardConstruct if/else)                                            #
# --------------------------------------------------------------------------- #
def test_branch_arm_spans_named_when_run_with_callbacks():
    """Both branch-arm node runnables read as their node names — no wrapper leak
    from the arm-descent node-add path (`_add_arm_nodes`)."""
    from neograph.forward import ForwardConstruct
    from tests.fakes import register_scripted

    register_scripted("br_check", lambda input_data, config: Confidence(score=0.9))
    register_scripted("br_high", lambda input_data, config: HighResult(label="high"))
    register_scripted("br_low", lambda input_data, config: LowResult(label="low"))

    class BranchPipeline(ForwardConstruct):
        check = Node.scripted("br-check", fn="br_check", outputs=Confidence)
        high_path = Node.scripted("br-high", fn="br_high", outputs=HighResult)
        low_path = Node.scripted("br-low", fn="br_low", outputs=LowResult)

        def forward(self, topic):
            result = self.check(topic)
            if result.score > 0.5:
                return self.high_path(result)
            else:
                return self.low_path(result)

    graph = compile(BranchPipeline(), **build_test_compile_kwargs())

    probe = SpanProbe()
    result = run(graph, input={"node_id": "branch-trace-1"}, config={"callbacks": [probe]})

    assert result["br_high"].label == "high"
    _assert_no_wrapper_leak(probe)
    assert "br-check" in probe.names
    assert "br-high" in probe.names  # taken arm


# --------------------------------------------------------------------------- #
# Sub-construct (subgraph)                                                     #
# --------------------------------------------------------------------------- #
def test_subgraph_span_named_when_run_with_callbacks():
    """A sub-construct's engine span reads as the construct name — not the leaking
    ``subgraph_node`` wrapper name."""
    from tests.fakes import register_scripted

    register_scripted("seed", lambda input_data, config: SubIn(payload="p"))
    register_scripted("enrich", lambda input_data, config: SubOut(enriched=f"e::{input_data.payload}"))

    sub = Construct(
        "enricher",
        input=SubIn,
        output=SubOut,
        nodes=[Node.scripted("enrich", fn="enrich", inputs=SubIn, outputs=SubOut)],
    )
    seed = Node.scripted("seed", fn="seed", outputs=SubIn)
    parent = Construct("parent", nodes=[seed, sub])
    graph = compile(parent, **build_test_compile_kwargs())

    probe = SpanProbe()
    result = run(graph, input={"node_id": "subgraph-trace-1"}, config={"callbacks": [probe]})

    assert isinstance(result["enricher"], SubOut)
    _assert_no_wrapper_leak(probe)
    assert "enricher" in probe.names
    assert probe.metadata_for("enricher").get("neograph_mode") == "subgraph"


# --------------------------------------------------------------------------- #
# Zero-cost / static-binding guarantee                                         #
# --------------------------------------------------------------------------- #
def test_named_is_a_static_config_binding_with_no_runtime_branching():
    """`named` is a late STATIC ``.with_config`` binding — the run_name/tags/
    metadata live on the binding's config, not in a per-invoke wrapper. This is
    what makes it zero-cost when no callbacks are attached."""

    def body(state, config):  # noqa: ANN001
        return {"x": 1}

    bound = named(RunnableLambda(body), "my-node", mode="scripted", output_type="Analysis")

    # A RunnableBinding: config is merged at build time, not evaluated per call.
    assert isinstance(bound, RunnableBinding)
    assert bound.config.get("run_name") == "my-node"
    assert NODE_TAG in bound.config.get("tags", [])
    md = bound.config.get("metadata", {})
    assert md.get("neograph_node") == "my-node"
    assert md.get("neograph_mode") == "scripted"
    assert md.get("neograph_output_type") == "Analysis"


def test_no_wrapper_leak_and_correct_result_when_no_callbacks_attached():
    """The naming binding is inert without callbacks: the graph runs and returns
    the correct result with no callbacks configured at all."""

    @node(outputs=Analysis)
    def summarize(text: Annotated[str, FromInput]) -> Analysis:
        return Analysis(summary=text.upper())

    mod = types.ModuleType("trace_nocb_mod")
    mod.summarize = summarize
    graph = compile(construct_from_module(mod, name="nocb-trace"), **build_test_compile_kwargs())

    result = run(graph, input={"text": "quiet"})
    assert result["summarize"].summary == "QUIET"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
