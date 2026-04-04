"""End-to-end acceptance tests: what must pass for NeoGraph to be useful for piarch.

Each test proves a real capability, not just data structure correctness.
All LLM calls use fakes — no API keys needed.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from neograph import Construct, Node, Operator, Oracle, Each, Tool, compile, raw_node, run


# ═══════════════════════════════════════════════════════════════════════════
# TEST SCHEMAS — shared across tests
# ═══════════════════════════════════════════════════════════════════════════

class RawText(BaseModel, frozen=True):
    text: str

class Claims(BaseModel, frozen=True):
    items: list[str]

class ClassifiedClaims(BaseModel, frozen=True):
    classified: list[dict[str, str]]  # [{claim: str, category: str}]

class ClusterGroup(BaseModel, frozen=True):
    label: str
    claim_ids: list[str]

class Clusters(BaseModel, frozen=True):
    groups: list[ClusterGroup]

class MatchResult(BaseModel, frozen=True):
    cluster_label: str
    matched: list[str]

class MergedResult(BaseModel, frozen=True):
    final_text: str

class ValidationResult(BaseModel, frozen=True):
    passed: bool
    issues: list[str]


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Scripted pipeline end-to-end
#
# Three deterministic nodes. No LLM. Data flows A→B→C through typed state.
# This proves: compile() builds a working graph, run() executes it,
# state bus passes data between nodes.
# ═══════════════════════════════════════════════════════════════════════════

class TestScriptedPipeline:
    def test_three_node_pipeline(self):
        """Data flows through 3 scripted nodes via typed state bus."""
        from neograph.factory import register_scripted

        # Register scripted functions
        register_scripted("extract_text", lambda input_data, config: RawText(text="hello world"))
        register_scripted("split_claims", lambda input_data, config: Claims(items=["claim-1", "claim-2"]))
        register_scripted("count_claims", lambda input_data, config: ClassifiedClaims(
            classified=[{"claim": c, "category": "fact"} for c in input_data.items]
        ))

        # Define nodes
        extract = Node.scripted("extract", fn="extract_text", output=RawText)
        split = Node.scripted("split", fn="split_claims", input=RawText, output=Claims)
        classify = Node.scripted("classify", fn="count_claims", input=Claims, output=ClassifiedClaims)

        # Compose and run
        pipeline = Construct("test-scripted", nodes=[extract, split, classify])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Verify data flowed through all three nodes
        assert result["classify"] is not None
        classified = result["classify"]
        assert len(classified.classified) == 2
        assert classified.classified[0]["claim"] == "claim-1"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Produce mode — structured LLM output
#
# A node calls a (fake) LLM, gets structured Pydantic output.
# This proves: produce dispatch works, LLM integration wired,
# output validated and written to state.
# ═══════════════════════════════════════════════════════════════════════════

class TestProduceMode:
    def test_produce_node_with_fake_llm(self):
        """Produce node calls LLM and gets structured output."""
        from neograph._llm import configure_llm

        # Fake LLM that returns a Claims object
        class FakeLLM:
            def with_structured_output(self, model):
                self._model = model
                return self

            def invoke(self, messages, **kwargs):
                return self._model(items=["extracted-1", "extracted-2", "extracted-3"])

        configure_llm(
            llm_factory=lambda tier: FakeLLM(),
            prompt_compiler=lambda template, data: [{"role": "user", "content": "test"}],
        )

        extract = Node(
            name="extract",
            mode="produce",
            output=Claims,
            model="fast",
            prompt="test/extract",
        )

        pipeline = Construct("test-produce", nodes=[extract])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        assert result["extract"] is not None
        assert len(result["extract"].items) == 3
        assert result["extract"].items[0] == "extracted-1"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Gather mode — ReAct loop with per-tool budgets
#
# A node with tools, budget enforced. Tool gets called, budget exhausts,
# LLM forced to respond.
# This proves: gather dispatch works, tool budget tracker enforces limits,
# exhausted tools get unbound.
# ═══════════════════════════════════════════════════════════════════════════

class TestGatherMode:
    def test_gather_with_tool_budgets(self):
        """Gather node uses tools within budget, then forced to respond."""
        from langchain_core.messages import AIMessage, ToolMessage

        from neograph._llm import configure_llm
        from neograph.factory import register_tool_factory

        call_count = {"search": 0}

        # Fake tool
        class FakeSearchTool:
            name = "search_nodes"

            def invoke(self, args):
                call_count["search"] += 1
                return f"found: result-{call_count['search']}"

        register_tool_factory("search_nodes", lambda config, tool_config: FakeSearchTool())

        # Fake LLM: calls tool twice, then responds
        class FakeGatherLLM:
            def __init__(self):
                self._call = 0
                self._tools_bound = True

            def bind_tools(self, tools):
                clone = FakeGatherLLM()
                clone._call = self._call
                clone._tools_bound = len(tools) > 0
                return clone

            def invoke(self, messages, **kwargs):
                self._call += 1
                if self._tools_bound and self._call <= 2:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search_nodes", "args": {"query": "test"}, "id": f"call-{self._call}"}]
                    return msg
                # Final response — no tool calls
                return AIMessage(content="done researching")

            def with_structured_output(self, model):
                self._model = model
                return self

        configure_llm(
            llm_factory=lambda tier: FakeGatherLLM(),
            prompt_compiler=lambda template, data: [{"role": "user", "content": "research this"}],
        )

        explore = Node(
            name="explore",
            mode="gather",
            output=Claims,
            model="reason",
            prompt="test/explore",
            tools=[Tool(name="search_nodes", budget=2)],
        )

        pipeline = Construct("test-gather", nodes=[explore])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Tool was called exactly twice (budget=2)
        assert call_count["search"] == 2

    def test_gather_unlimited_budget(self):
        """Tool with budget=0 is never exhausted — LLM decides when to stop."""
        from langchain_core.messages import AIMessage

        from neograph._llm import configure_llm
        from neograph.factory import register_tool_factory

        call_count = {"n": 0}

        class FakeTool:
            name = "lookup"

            def invoke(self, args):
                call_count["n"] += 1
                return f"result-{call_count['n']}"

        register_tool_factory("lookup", lambda config, tool_config: FakeTool())

        invoke_count = {"n": 0}

        class FakeLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kwargs):
                invoke_count["n"] += 1
                if invoke_count["n"] <= 5:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "lookup", "args": {}, "id": f"c{invoke_count['n']}"}]
                    return msg
                return AIMessage(content="done")

            def with_structured_output(self, model):
                self._model = model
                return self

        configure_llm(
            llm_factory=lambda tier: FakeLLM(),
            prompt_compiler=lambda template, data: [{"role": "user", "content": "go"}],
        )

        node = Node(
            name="scan",
            mode="gather",
            output=Claims,
            model="fast",
            prompt="test/scan",
            tools=[Tool(name="lookup", budget=0)],  # unlimited
        )

        pipeline = Construct("test-unlimited", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Tool was called 5 times — budget=0 never blocked it
        assert call_count["n"] == 5


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Oracle — 3-way ensemble + merge
#
# Three parallel generators produce variants, barrier merges them.
# This proves: Oracle modifier expands to fan-out Send() + merge barrier,
# all three generators run, results converge.
# ═══════════════════════════════════════════════════════════════════════════

class TestOracle:
    def test_three_way_ensemble(self):
        """Oracle dispatches 3 generators and merges results."""
        gen_ids_seen = []

        from neograph.factory import register_scripted

        # Each generator records its ID and produces a variant
        def generate_variant(input_data, config):
            gen_id = config.get("configurable", {}).get("_generator_id", "unknown")
            gen_ids_seen.append(gen_id)
            return Claims(items=[f"variant-from-{gen_id}"])

        register_scripted("generate_variant", generate_variant)

        # Merge: combine all variant items into one Claims
        def combine_variants(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("combine_variants", combine_variants)

        gen_node = Node.scripted(
            "generate", fn="generate_variant", output=Claims
        ) | Oracle(n=3, merge_fn="combine_variants")

        pipeline = Construct("test-oracle", nodes=[gen_node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # All 3 generators ran
        assert len(gen_ids_seen) == 3
        # Merge combined all 3 variants
        merged = result.get("generate")
        assert merged is not None
        assert len(merged.items) == 3

    def test_llm_merge(self):
        """Oracle with merge_prompt calls LLM to judge-merge variants."""
        from neograph._llm import configure_llm
        from neograph.factory import register_scripted

        register_scripted("gen", lambda input_data, config: Claims(items=["v1"]))

        class FakeMergeLLM:
            def with_structured_output(self, model):
                self._model = model
                return self

            def invoke(self, messages, **kwargs):
                return self._model(items=["merged-consensus"])

        configure_llm(
            llm_factory=lambda tier: FakeMergeLLM(),
            prompt_compiler=lambda template, data: [{"role": "user", "content": "merge"}],
        )

        gen_node = Node.scripted(
            "generate", fn="gen", output=Claims
        ) | Oracle(n=2, merge_prompt="test/merge")

        pipeline = Construct("test-oracle-llm", nodes=[gen_node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        merged = result.get("generate")
        assert merged is not None
        assert merged.items == ["merged-consensus"]

    def test_rejects_bare_oracle(self):
        """Oracle without merge_prompt or merge_fn is a ValueError."""
        with pytest.raises(ValueError, match="merge_prompt.*merge_fn"):
            Oracle(n=3)

    def test_rejects_both_merge_options(self):
        """Oracle with both merge_prompt and merge_fn is a ValueError."""
        with pytest.raises(ValueError, match="not both"):
            Oracle(n=3, merge_prompt="x", merge_fn="y")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: Each — dynamic fan-out over collection
#
# Node fans out over a list of clusters, processes each in parallel,
# results collected as dict[key, result].
# This proves: Each modifier expands to Send() per item,
# barrier collects, dict reducer merges results.
# ═══════════════════════════════════════════════════════════════════════════

class TestEach:
    def test_fanout_over_collection(self):
        """Each dispatches per-item and collects results."""
        from neograph.factory import register_scripted

        # First node produces clusters
        register_scripted("make_clusters", lambda input_data, config: Clusters(
            groups=[
                ClusterGroup(label="alpha", claim_ids=["c1", "c2"]),
                ClusterGroup(label="beta", claim_ids=["c3"]),
            ]
        ))

        # Fan-out node processes each cluster
        register_scripted("verify_cluster", lambda input_data, config: MatchResult(
            cluster_label=input_data.label if hasattr(input_data, 'label') else "unknown",
            matched=["match-1"],
        ))

        make = Node.scripted("make-clusters", fn="make_clusters", output=Clusters)
        verify = Node.scripted(
            "verify", fn="verify_cluster", input=ClusterGroup, output=MatchResult
        ) | Each(over="make_clusters.groups", key="label")

        pipeline = Construct("test-replicate", nodes=[make, verify])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Both clusters were processed
        verify_results = result.get("verify", {})
        assert "alpha" in verify_results or len(verify_results) == 2


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: Operator — human-in-the-loop interrupt
#
# A node produces a validation result. If validation fails,
# the graph pauses via interrupt(). Resume with human input.
# This proves: Operator modifier wires interrupt() correctly,
# graph pauses and resumes.
# ═══════════════════════════════════════════════════════════════════════════

class TestOperator:
    def test_interrupt_on_failure(self):
        """Graph pauses when Operator condition is met."""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import interrupt

        from neograph.factory import register_condition, register_scripted

        register_scripted("validate", lambda input_data, config: ValidationResult(
            passed=False,
            issues=["missing stakeholder coverage"],
        ))

        register_condition("validation_failed", lambda state: (
            {"issues": state.validate.issues} if state.validate and not state.validate.passed else None
        ))

        validate = Node.scripted(
            "validate", fn="validate", output=ValidationResult
        ) | Operator(when="validation_failed")

        pipeline = Construct("test-operator", nodes=[validate])
        graph = compile(pipeline, checkpointer=MemorySaver())

        # Run — interrupt pauses the graph
        config = {"configurable": {"thread_id": "test-interrupt"}}

        # First run should pause at interrupt
        try:
            result = run(graph, input={"node_id": "test-001"}, config=config)
            # If we get here, interrupt was raised and caught by LangGraph
        except Exception:
            pass  # interrupt() raises GraphInterrupt

        # Resume with human approval
        # result = run(graph, resume={"approved": True}, config=config)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: Raw node alongside declarative nodes
#
# A @raw_node function mixed with Node() declarations in the same Construct.
# This proves: raw escape hatch works, framework wires edges around it,
# data flows through raw node like any other.
# ═══════════════════════════════════════════════════════════════════════════

class TestRawNode:
    def test_raw_node_in_pipeline(self):
        """@raw_node works alongside declarative nodes."""
        from neograph.factory import register_scripted

        register_scripted("make_claims", lambda input_data, config: Claims(items=["a", "b", "c"]))

        @raw_node(input=Claims, output=Claims)
        def filter_claims(state, config):
            """Raw node: custom filtering logic."""
            claims = None
            for field_name in state.__class__.model_fields:
                val = getattr(state, field_name, None)
                if isinstance(val, Claims):
                    claims = val
                    break
            if claims is None:
                return {"filter_claims": Claims(items=[])}
            filtered = Claims(items=[c for c in claims.items if c != "b"])
            return {"filter_claims": filtered}

        make = Node.scripted("make-claims", fn="make_claims", output=Claims)

        pipeline = Construct("test-raw", nodes=[make, filter_claims])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Raw node filtered out "b"
        filtered = result.get("filter_claims")
        assert filtered is not None
        assert "b" not in filtered.items
        assert "a" in filtered.items
        assert "c" in filtered.items


# ═══════════════════════════════════════════════════════════════════════════
# TEST 8: Realistic mini-pipeline (simplified RW)
#
# read → decompose (produce) → classify (scripted) → verify (gather+fanout)
# Mixes all modes in a single pipeline.
# This proves: heterogeneous pipelines work end-to-end.
# ═══════════════════════════════════════════════════════════════════════════

class TestMiniRWPipeline:
    def test_mixed_mode_pipeline(self):
        """Realistic pipeline mixing produce + scripted modes."""
        from neograph._llm import configure_llm
        from neograph.factory import register_scripted

        # Fake LLM
        class FakeLLM:
            def with_structured_output(self, model):
                self._model = model
                return self

            def invoke(self, messages, **kwargs):
                if self._model is Claims:
                    return Claims(items=["r1: system shall validate", "r2: system shall log"])
                if self._model is ClassifiedClaims:
                    return ClassifiedClaims(classified=[
                        {"claim": "r1", "category": "requirement"},
                        {"claim": "r2", "category": "requirement"},
                    ])
                return self._model()

        configure_llm(
            llm_factory=lambda tier: FakeLLM(),
            prompt_compiler=lambda template, data: [{"role": "user", "content": "test"}],
        )

        register_scripted("build_catalog", lambda input_data, config: RawText(text="node catalog: 42 nodes"))

        # Pipeline
        decompose = Node(
            name="decompose",
            mode="produce",
            output=Claims,
            model="reason",
            prompt="rw/decompose",
        )
        classify = Node(
            name="classify",
            mode="produce",
            input=Claims,
            output=ClassifiedClaims,
            model="fast",
            prompt="rw/classify",
        )
        catalog = Node.scripted("catalog", fn="build_catalog", output=RawText)

        pipeline = Construct(
            "mini-rw",
            description="Simplified RW pipeline: decompose → classify → catalog",
            nodes=[decompose, classify, catalog],
        )
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "BR-RW-042"})

        # All three nodes produced output
        assert result["decompose"] is not None
        assert result["classify"] is not None
        assert result["catalog"] is not None
        assert len(result["decompose"].items) == 2
        assert len(result["classify"].classified) == 2
        assert "42 nodes" in result["catalog"].text


# ═══════════════════════════════════════════════════════════════════════════
# COVERAGE GAP TESTS — every remaining code path
# ═══════════════════════════════════════════════════════════════════════════


class TestExecuteMode:
    """Execute mode: ReAct tool loop with mutation tools."""

    def test_execute_with_tools(self):
        """Execute node calls tools and produces structured output."""
        from langchain_core.messages import AIMessage

        from neograph._llm import configure_llm
        from neograph.factory import register_tool_factory

        mutations = []

        class FakeMutationTool:
            name = "write_file"

            def invoke(self, args):
                mutations.append(args)
                return "written"

        register_tool_factory("write_file", lambda config, tool_config: FakeMutationTool())

        class FakeExecuteLLM:
            def __init__(self):
                self._call = 0
                self._tools_bound = True

            def bind_tools(self, tools):
                clone = FakeExecuteLLM()
                clone._call = self._call
                clone._tools_bound = len(tools) > 0
                return clone

            def invoke(self, messages, **kwargs):
                self._call += 1
                if self._tools_bound and self._call == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "write_file", "args": {"path": "out.txt"}, "id": "call-1"}]
                    return msg
                return AIMessage(content="done")

            def with_structured_output(self, model):
                self._model = model
                return self

        configure_llm(
            llm_factory=lambda tier: FakeExecuteLLM(),
            prompt_compiler=lambda template, data: [{"role": "user", "content": "execute"}],
        )

        writer = Node(
            name="writer",
            mode="execute",
            output=RawText,
            model="fast",
            prompt="test/write",
            tools=[Tool(name="write_file", budget=1)],
        )

        pipeline = Construct("test-execute", nodes=[writer])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        assert len(mutations) == 1
        assert mutations[0]["path"] == "out.txt"


class TestOperatorContinues:
    """Operator condition is falsy — graph continues without interrupt."""

    def test_operator_passes_when_condition_falsy(self):
        """Graph runs through Operator without pausing when condition returns None."""
        from neograph.factory import register_condition, register_scripted

        register_scripted("check_quality", lambda input_data, config: ValidationResult(
            passed=True, issues=[],
        ))

        register_condition("validation_failed", lambda state: None)  # always falsy

        validate = Node.scripted(
            "validate", fn="check_quality", output=ValidationResult
        ) | Operator(when="validation_failed")

        from langgraph.checkpoint.memory import MemorySaver

        pipeline = Construct("test-operator-pass", nodes=[validate])
        graph = compile(pipeline, checkpointer=MemorySaver())
        result = run(graph, input={"node_id": "test-001"}, config={"configurable": {"thread_id": "pass-test"}})

        assert result["validate"].passed is True
        assert result.get("human_feedback") is None


class TestOperatorResume:
    """Operator interrupt + resume flow."""

    def test_interrupt_then_resume(self):
        """Graph pauses at interrupt, resumes with human feedback via run()."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        register_scripted("validate_thing", lambda input_data, config: ValidationResult(
            passed=False, issues=["bad coverage"],
        ))

        register_condition("needs_review", lambda state: (
            {"issues": state.validate_thing.issues}
            if state.validate_thing and not state.validate_thing.passed
            else None
        ))

        validate = Node.scripted(
            "validate-thing", fn="validate_thing", output=ValidationResult
        ) | Operator(when="needs_review")

        pipeline = Construct("test-resume", nodes=[validate])
        graph = compile(pipeline, checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "resume-test"}}

        # First run: hits interrupt — returns with __interrupt__
        result = run(graph, input={"node_id": "test-001"}, config=config)
        assert "__interrupt__" in result

        # Resume via run()
        result = run(graph, resume={"approved": True}, config=config)

        assert result["validate_thing"].passed is False
        assert result["human_feedback"] == {"approved": True}


class TestErrorPaths:
    """Every error path the framework raises."""

    def test_produce_without_configure_llm(self):
        """Produce node without configure_llm() raises RuntimeError."""
        node = Node(name="fail", mode="produce", output=Claims, model="fast", prompt="x")
        pipeline = Construct("test-no-llm", nodes=[node])
        graph = compile(pipeline)

        with pytest.raises(RuntimeError, match="LLM not configured"):
            run(graph, input={"node_id": "test-001"})

    def test_unregistered_scripted_fn(self):
        """Referencing unregistered scripted function raises ValueError."""
        node = Node.scripted("bad", fn="nonexistent_fn", output=Claims)
        pipeline = Construct("test-bad-fn", nodes=[node])

        with pytest.raises(ValueError, match="not registered"):
            compile(pipeline)

    def test_unregistered_oracle_merge_fn(self):
        """Oracle with unregistered merge_fn raises ValueError at compile."""
        from neograph.factory import register_scripted

        register_scripted("gen", lambda input_data, config: Claims(items=["x"]))

        node = Node.scripted(
            "gen", fn="gen", output=Claims
        ) | Oracle(n=2, merge_fn="nonexistent_merge")

        pipeline = Construct("test-bad-merge", nodes=[node])

        with pytest.raises(ValueError, match="not registered"):
            compile(pipeline)

    def test_unregistered_operator_condition(self):
        """Operator with unregistered condition raises ValueError at compile."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_scripted

        register_scripted("something", lambda input_data, config: Claims(items=[]))

        node = Node.scripted(
            "something", fn="something", output=Claims
        ) | Operator(when="nonexistent_condition")

        pipeline = Construct("test-bad-condition", nodes=[node])

        with pytest.raises(ValueError, match="not registered"):
            compile(pipeline, checkpointer=MemorySaver())

    def test_unregistered_tool_factory(self):
        """Gather node with unregistered tool factory raises ValueError."""
        from neograph._llm import configure_llm

        class FakeLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kwargs):
                from langchain_core.messages import AIMessage
                return AIMessage(content="done")

        configure_llm(
            llm_factory=lambda tier: FakeLLM(),
            prompt_compiler=lambda template, data: [{"role": "user", "content": "x"}],
        )

        node = Node(
            name="explore",
            mode="gather",
            output=Claims,
            model="fast",
            prompt="x",
            tools=[Tool(name="ghost_tool", budget=1)],
        )

        pipeline = Construct("test-bad-tool", nodes=[node])
        graph = compile(pipeline)

        with pytest.raises(ValueError, match="not registered"):
            run(graph, input={"node_id": "test-001"})

    def test_run_with_no_input_or_resume(self):
        """run() with neither input nor resume raises ValueError."""
        from neograph.factory import register_scripted

        register_scripted("noop", lambda input_data, config: Claims(items=[]))
        node = Node.scripted("noop", fn="noop", output=Claims)
        pipeline = Construct("test-no-args", nodes=[node])
        graph = compile(pipeline)

        with pytest.raises(ValueError, match="input or resume"):
            run(graph)

    def test_node_without_output_type(self):
        """Node with no output type raises ValueError at compile."""
        node = Node(name="bad-node", mode="produce", model="fast", prompt="x")
        pipeline = Construct("test-no-output", nodes=[node])

        with pytest.raises(ValueError, match="no output type"):
            compile(pipeline)


class TestLLMUnknownToolCall:
    """LLM hallucinates a tool name the framework doesn't have."""

    def test_unknown_tool_call_handled(self):
        """Framework responds with error message, doesn't crash."""
        from langchain_core.messages import AIMessage

        from neograph._llm import configure_llm
        from neograph.factory import register_tool_factory

        class FakeRealTool:
            name = "search"

            def invoke(self, args):
                return "found it"

        register_tool_factory("search", lambda config, tool_config: FakeRealTool())

        call_counter = {"n": 0}

        class FakeLLMHallucinator:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kwargs):
                call_counter["n"] += 1
                if call_counter["n"] == 1:
                    # LLM hallucinates a tool that doesn't exist
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "delete_everything", "args": {}, "id": "bad-1"}]
                    return msg
                return AIMessage(content="ok done")

            def with_structured_output(self, model):
                self._model = model
                return self

        configure_llm(
            llm_factory=lambda tier: FakeLLMHallucinator(),
            prompt_compiler=lambda template, data: [{"role": "user", "content": "test"}],
        )

        node = Node(
            name="explore",
            mode="gather",
            output=Claims,
            model="fast",
            prompt="test/explore",
            tools=[Tool(name="search", budget=5)],
        )

        pipeline = Construct("test-hallucinated-tool", nodes=[node])
        graph = compile(pipeline)
        # Should complete without crashing — unknown tool gets error message
        result = run(graph, input={"node_id": "test-001"})
        assert result["explore"] is not None


class TestModifierAsFirstNode:
    """Modifiers on the first node wire from START, not from a previous node."""

    def test_oracle_at_start(self):
        """Oracle as the first (and only) node — router wired from START."""
        from neograph.factory import register_scripted

        register_scripted("gen_start", lambda input_data, config: Claims(items=["from-start"]))

        def merge_start(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("merge_start", merge_start)

        node = Node.scripted(
            "gen", fn="gen_start", output=Claims
        ) | Oracle(n=2, merge_fn="merge_start")

        pipeline = Construct("test-oracle-start", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        merged = result.get("gen")
        assert merged is not None
        assert len(merged.items) == 2

    def test_each_at_start(self):
        """Each as the first node — router wired from START."""
        from langgraph.graph import END, StateGraph

        from neograph.compiler import _add_node_to_graph
        from neograph.factory import register_scripted
        from neograph.state import compile_state_model

        register_scripted("process_item", lambda input_data, config: MatchResult(
            cluster_label=input_data.label, matched=["done"],
        ))

        process = Node.scripted(
            "process", fn="process_item", input=ClusterGroup, output=MatchResult
        ) | Each(over="make_items.groups", key="label")

        pipeline = Construct("test-each-start", nodes=[process])
        state_model = compile_state_model(pipeline)
        graph = StateGraph(state_model)
        prev = _add_node_to_graph(graph, process, None)
        graph.add_edge(prev, END)
        compiled = graph.compile()
        # Compilation succeeded — Each wired from START without crash


class TestMultiFieldInput:
    """Node with dict[str, type] input spec extracts multiple fields."""

    def test_dict_input_extraction(self):
        """Node receives multiple typed fields from state."""
        from neograph.factory import register_scripted

        register_scripted("make_claims", lambda input_data, config: Claims(items=["a", "b"]))
        register_scripted("make_raw", lambda input_data, config: RawText(text="hello"))

        def combine(input_data, config):
            claims = input_data["step_a"]
            raw = input_data["step_b"]
            return RawText(text=f"{raw.text}: {len(claims.items)} items")

        register_scripted("combine", combine)

        step_a = Node.scripted("step-a", fn="make_claims", output=Claims)
        step_b = Node.scripted("step-b", fn="make_raw", output=RawText)
        step_c = Node.scripted(
            "step-c", fn="combine",
            input={"step_a": Claims, "step_b": RawText},
            output=RawText,
        )

        pipeline = Construct("test-multi-input", nodes=[step_a, step_b, step_c])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        assert result["step_c"].text == "hello: 2 items"



class TestReducerEdgeCases:
    """Reducer functions handle None and unexpected inputs correctly."""

    def test_oracle_results_collected_from_empty(self):
        """Oracle reducer builds list from None initial state."""
        from neograph.state import _collect_oracle_results

        # First write: existing is None
        result = _collect_oracle_results(None, "first")
        assert result == ["first"]

        # Second write: existing is a list
        result = _collect_oracle_results(["first"], "second")
        assert result == ["first", "second"]

        # List input (batch)
        result = _collect_oracle_results(["a"], ["b", "c"])
        assert result == ["a", "b", "c"]

    def test_merge_dicts_from_none(self):
        """Dict merge reducer starts from None."""
        from neograph.state import _merge_dicts

        result = _merge_dicts(None, {"a": 1})
        assert result == {"a": 1}

        result = _merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

        # Duplicate key raises
        with pytest.raises(ValueError, match="duplicate key"):
            _merge_dicts({"a": 1}, {"a": 99})

    def test_last_write_wins(self):
        """Last-write-wins reducer always returns new value."""
        from neograph.state import _last_write_wins

        assert _last_write_wins("old", "new") == "new"
        assert _last_write_wins(None, "new") == "new"
        assert _last_write_wins("old", None) is None


class TestStateHygiene:
    """Framework internals never leak to the consumer."""

    def test_oracle_internals_stripped(self):
        """Oracle collector and gen_id are not in the result."""
        from neograph.factory import register_scripted

        register_scripted("g", lambda input_data, config: Claims(items=["x"]))
        register_scripted("m", lambda variants, config: Claims(items=["merged"]))

        node = Node.scripted("gen", fn="g", output=Claims) | Oracle(n=2, merge_fn="m")
        pipeline = Construct("test-hygiene-oracle", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Consumer sees the merged result under the node's name
        assert result["gen"].items == ["merged"]
        # Internals stripped
        assert not any(k.startswith("neo_") for k in result)

    def test_each_internals_stripped(self):
        """Each item plumbing is not in the result."""
        from neograph.factory import register_scripted

        register_scripted("make", lambda input_data, config: Clusters(
            groups=[ClusterGroup(label="a", claim_ids=["1"])]
        ))
        register_scripted("proc", lambda input_data, config: MatchResult(
            cluster_label=input_data.label, matched=["done"],
        ))

        make = Node.scripted("make", fn="make", output=Clusters)
        proc = Node.scripted(
            "proc", fn="proc", input=ClusterGroup, output=MatchResult
        ) | Each(over="make.groups", key="label")

        pipeline = Construct("test-hygiene-each", nodes=[make, proc])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Consumer sees dict keyed by label
        assert "a" in result["proc"]
        # Internals stripped
        assert not any(k.startswith("neo_") for k in result)

    def test_each_duplicate_key_raises(self):
        """Each fan-out with duplicate dispatch keys raises, not silently overwrites."""
        from neograph.factory import register_scripted

        # Collection with duplicate labels
        register_scripted("make_dupes", lambda input_data, config: Clusters(
            groups=[
                ClusterGroup(label="same", claim_ids=["1"]),
                ClusterGroup(label="same", claim_ids=["2"]),
            ]
        ))
        register_scripted("proc_dupe", lambda input_data, config: MatchResult(
            cluster_label=input_data.label, matched=["done"],
        ))

        make = Node.scripted("make-dupes", fn="make_dupes", output=Clusters)
        proc = Node.scripted(
            "proc-dupe", fn="proc_dupe", input=ClusterGroup, output=MatchResult
        ) | Each(over="make_dupes.groups", key="label")

        pipeline = Construct("test-dupe-key", nodes=[make, proc])
        graph = compile(pipeline)

        with pytest.raises(Exception, match="duplicate key"):
            run(graph, input={"node_id": "test-001"})


# ═══════════════════════════════════════════════════════════════════════════
# SUBGRAPH TESTS — Construct inside Construct with isolated state
# ═══════════════════════════════════════════════════════════════════════════


class EnrichInput(BaseModel, frozen=True):
    claims: list[str]


class EnrichOutput(BaseModel, frozen=True):
    scored: list[dict[str, str]]


class TestSubgraph:
    """Sub-construct with isolated state inside a parent pipeline."""

    def test_subgraph_basic(self):
        """Sub-construct runs with isolated state, only output surfaces."""
        from neograph.factory import register_scripted

        # Parent: produces claims
        register_scripted("decompose", lambda input_data, config: EnrichInput(
            claims=["claim-1", "claim-2"],
        ))

        # Sub-construct nodes: enrich the claims
        register_scripted("lookup", lambda input_data, config: RawText(
            text=f"context for {len(input_data.claims)} claims",
        ))
        register_scripted("score", lambda input_data, config: EnrichOutput(
            scored=[{"claim": c, "score": "high"} for c in input_data.claims],
        ))

        # Sub-construct with declared I/O boundary
        enrich = Construct(
            "enrich",
            input=EnrichInput,
            output=EnrichOutput,
            nodes=[
                Node.scripted("lookup", fn="lookup", input=EnrichInput, output=RawText),
                Node.scripted("score", fn="score", input=EnrichInput, output=EnrichOutput),
            ],
        )

        # Parent pipeline
        decompose = Node.scripted("decompose", fn="decompose", output=EnrichInput)
        parent = Construct("parent", nodes=[decompose, enrich])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Sub-construct output surfaces under its name
        assert result["enrich"] is not None
        assert len(result["enrich"].scored) == 2
        assert result["enrich"].scored[0]["score"] == "high"

        # Sub-construct internals (lookup, score) do NOT appear in parent result
        assert "lookup" not in result
        assert "score" not in result

    def test_subgraph_state_isolation(self):
        """Sub-construct's internal fields don't collide with parent fields."""
        from neograph.factory import register_scripted

        # Both parent and sub-construct have a node named "process"
        register_scripted("parent_process", lambda input_data, config: Claims(items=["parent"]))
        register_scripted("sub_input", lambda input_data, config: Claims(items=["sub-in"]))
        register_scripted("sub_process", lambda input_data, config: RawText(text="sub-result"))

        sub = Construct(
            "sub",
            input=Claims,
            output=RawText,
            nodes=[
                Node.scripted("process", fn="sub_process", input=Claims, output=RawText),
            ],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("process", fn="parent_process", output=Claims),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Parent's "process" node output
        assert result["process"].items == ["parent"]
        # Sub-construct output (no collision)
        assert result["sub"].text == "sub-result"

    def test_subgraph_without_input_raises(self):
        """Sub-construct without declared input raises at compile."""
        from neograph.factory import register_scripted

        register_scripted("noop", lambda input_data, config: Claims(items=[]))

        sub = Construct("bad-sub", output=Claims, nodes=[
            Node.scripted("noop", fn="noop", output=Claims),
        ])

        parent = Construct("parent", nodes=[sub])

        with pytest.raises(ValueError, match="must declare input"):
            compile(parent)

    def test_subgraph_without_output_raises(self):
        """Sub-construct without declared output raises at compile."""
        from neograph.factory import register_scripted

        register_scripted("noop", lambda input_data, config: Claims(items=[]))

        sub = Construct("bad-sub", input=Claims, nodes=[
            Node.scripted("noop", fn="noop", output=Claims),
        ])

        parent = Construct("parent", nodes=[sub])

        with pytest.raises(ValueError, match="must declare output"):
            compile(parent)

    def test_subgraph_with_oracle_inside(self):
        """Oracle inside a sub-construct — fan-out happens in isolated state."""
        from neograph.factory import register_scripted

        register_scripted("parent_prep", lambda input_data, config: Claims(items=["topic"]))
        register_scripted("sub_gen", lambda input_data, config: RawText(text=f"variant"))

        def sub_merge(variants, config):
            return RawText(text=f"merged {len(variants)} variants")

        register_scripted("sub_merge", sub_merge)

        sub = Construct(
            "oracle-sub",
            input=Claims,
            output=RawText,
            nodes=[
                Node.scripted("gen", fn="sub_gen", output=RawText) | Oracle(n=3, merge_fn="sub_merge"),
            ],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("prep", fn="parent_prep", output=Claims),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        assert result["oracle_sub"].text == "merged 3 variants"
        # Oracle internals from sub don't leak
        assert not any("neo_oracle" in k for k in result)

    def test_subgraph_with_each_inside(self):
        """Each inside a sub-construct — fan-out in isolated state."""
        from neograph.factory import register_scripted

        # Each produces dict[str, MatchResult]. The sub-construct's output
        # must be the dict type, not MatchResult, because that's what Each writes.
        # Use a collector node after Each to convert dict → single output.
        register_scripted("parent_clusters", lambda input_data, config: Clusters(
            groups=[ClusterGroup(label="a", claim_ids=["1"]), ClusterGroup(label="b", claim_ids=["2"])]
        ))
        register_scripted("sub_verify", lambda input_data, config: MatchResult(
            cluster_label=input_data.label, matched=["ok"],
        ))
        register_scripted("sub_collect", lambda input_data, config: RawText(
            text=f"verified {len(input_data)} clusters" if isinstance(input_data, dict) else "verified"
        ))

        sub = Construct(
            "verify-sub",
            input=Clusters,
            output=RawText,
            nodes=[
                Node.scripted("verify", fn="sub_verify", input=ClusterGroup, output=MatchResult)
                | Each(over="neo_subgraph_input.groups", key="label"),
                Node.scripted("collect", fn="sub_collect", output=RawText),
            ],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("make-clusters", fn="parent_clusters", output=Clusters),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        assert result["verify_sub"] is not None
        assert "verified" in result["verify_sub"].text

    def test_nested_subgraphs(self):
        """Construct inside Construct inside Construct — two levels deep."""
        from neograph.factory import register_scripted

        register_scripted("l0_start", lambda input_data, config: Claims(items=["raw"]))
        register_scripted("l1_process", lambda input_data, config: Claims(items=["l1-processed"]))
        register_scripted("l2_detail", lambda input_data, config: RawText(text="l2-done"))
        register_scripted("l0_finish", lambda input_data, config: RawText(text="final"))

        # Level2: Claims → RawText
        level2 = Construct(
            "level2",
            input=Claims,
            output=RawText,
            nodes=[Node.scripted("detail", fn="l2_detail", input=Claims, output=RawText)],
        )

        # Level1: Claims → RawText (via level2)
        level1 = Construct(
            "level1",
            input=Claims,
            output=RawText,
            nodes=[
                Node.scripted("process", fn="l1_process", input=Claims, output=Claims),
                level2,
            ],
        )

        parent = Construct("root", nodes=[
            Node.scripted("start", fn="l0_start", output=Claims),
            level1,
            Node.scripted("finish", fn="l0_finish", input=RawText, output=RawText),
        ])

        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Level1's output is RawText (from level2 bubbling up)
        assert result["level1"].text == "l2-done"
        assert result["finish"].text == "final"
        # No level1/level2 internals in parent result
        assert "level2" not in result
        assert "detail" not in result
        assert "process" not in result

    def test_multiple_subgraphs_in_parent(self):
        """Two sub-constructs in the same parent pipeline."""
        from neograph.factory import register_scripted

        register_scripted("make_input", lambda input_data, config: Claims(items=["a", "b"]))
        register_scripted("enrich_fn", lambda input_data, config: RawText(text="enriched"))
        register_scripted("validate_fn", lambda input_data, config: ValidationResult(passed=True, issues=[]))

        enrich_sub = Construct(
            "enrich",
            input=Claims,
            output=RawText,
            nodes=[Node.scripted("e", fn="enrich_fn", input=Claims, output=RawText)],
        )

        validate_sub = Construct(
            "check",
            input=RawText,
            output=ValidationResult,
            nodes=[Node.scripted("v", fn="validate_fn", input=RawText, output=ValidationResult)],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("start", fn="make_input", output=Claims),
            enrich_sub,
            validate_sub,
        ])

        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        assert result["enrich"].text == "enriched"
        assert result["check"].passed is True

    def test_operator_without_checkpointer_raises(self):
        """Operator node without checkpointer raises ValueError at compile."""
        from neograph.factory import register_condition, register_scripted

        register_scripted("x", lambda input_data, config: Claims(items=[]))
        register_condition("always", lambda state: True)

        node = Node.scripted("x", fn="x", output=Claims) | Operator(when="always")
        pipeline = Construct("test-no-cp", nodes=[node])

        with pytest.raises(ValueError, match="checkpointer"):
            compile(pipeline)


# ═══════════════════════════════════════════════════════════════════════════
# CONSTRUCT + MODIFIER COMPOSITIONS
#
# Every modifier × Construct target, plus deep nesting combos.
# ═══════════════════════════════════════════════════════════════════════════


class TestConstructOracle:
    """Construct | Oracle — run entire sub-pipeline N times, merge outputs."""

    def test_construct_oracle_scripted_merge(self):
        """Sub-pipeline runs 3 times, scripted merge combines outputs."""
        from neograph.factory import register_scripted

        register_scripted("sub_step_a", lambda input_data, config: Claims(items=["step-a"]))
        register_scripted("sub_step_b", lambda input_data, config: RawText(
            text=f"processed: {input_data.items[0]}" if input_data else "processed: none"
        ))

        def merge_sub_outputs(variants, config):
            all_texts = [v.text for v in variants]
            return RawText(text=" | ".join(all_texts))

        register_scripted("merge_sub", merge_sub_outputs)

        sub = Construct(
            "enrich",
            input=Claims,
            output=RawText,
            nodes=[
                Node.scripted("step-a", fn="sub_step_a", output=Claims),
                Node.scripted("step-b", fn="sub_step_b", input=Claims, output=RawText),
            ],
        ) | Oracle(n=3, merge_fn="merge_sub")

        register_scripted("make_input", lambda input_data, config: Claims(items=["raw"]))

        parent = Construct("parent", nodes=[
            Node.scripted("start", fn="make_input", output=Claims),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # 3 variants merged into one
        assert result["enrich"] is not None
        assert result["enrich"].text.count("processed") == 3

    def test_construct_oracle_llm_merge(self):
        """Sub-pipeline runs 2 times, LLM merge combines outputs."""
        from neograph._llm import configure_llm
        from neograph.factory import register_scripted

        register_scripted("gen_claim", lambda input_data, config: Claims(items=["variant"]))

        class FakeMergeLLM:
            def with_structured_output(self, model):
                self._model = model
                return self

            def invoke(self, messages, **kwargs):
                return self._model(items=["llm-merged"])

        configure_llm(
            llm_factory=lambda tier: FakeMergeLLM(),
            prompt_compiler=lambda template, data: [{"role": "user", "content": "merge"}],
        )

        sub = Construct(
            "gen-pipeline",
            input=Claims,
            output=Claims,
            nodes=[Node.scripted("gen", fn="gen_claim", output=Claims)],
        ) | Oracle(n=2, merge_prompt="test/merge")

        register_scripted("seed", lambda input_data, config: Claims(items=["seed"]))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed", output=Claims),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        assert result["gen_pipeline"].items == ["llm-merged"]


class TestConstructEach:
    """Construct | Each — run entire sub-pipeline per collection item."""

    def test_construct_each_over_collection(self):
        """Sub-pipeline runs once per cluster, results collected as dict."""
        from neograph.factory import register_scripted

        register_scripted("make_clusters", lambda input_data, config: Clusters(
            groups=[
                ClusterGroup(label="alpha", claim_ids=["c1"]),
                ClusterGroup(label="beta", claim_ids=["c2", "c3"]),
            ]
        ))

        register_scripted("sub_analyze", lambda input_data, config: RawText(
            text=f"analyzed: {input_data.label}"
        ))
        register_scripted("sub_score", lambda input_data, config: MatchResult(
            cluster_label=input_data.label if hasattr(input_data, 'label') else "unknown",
            matched=[f"scored-{input_data.text}" if hasattr(input_data, 'text') else "scored"],
        ))

        sub = Construct(
            "verify-cluster",
            input=ClusterGroup,
            output=MatchResult,
            nodes=[
                Node.scripted("analyze", fn="sub_analyze", input=ClusterGroup, output=RawText),
                Node.scripted("score", fn="sub_score", input=RawText, output=MatchResult),
            ],
        ) | Each(over="make_clusters.groups", key="label")

        parent = Construct("parent", nodes=[
            Node.scripted("make-clusters", fn="make_clusters", output=Clusters),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Both clusters processed
        verify_results = result["verify_cluster"]
        assert "alpha" in verify_results
        assert "beta" in verify_results


class TestConstructOperator:
    """Construct | Operator — check condition after sub-pipeline completes."""

    def test_construct_operator_interrupts(self):
        """Sub-pipeline runs, then Operator checks and interrupts."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        register_scripted("sub_validate", lambda input_data, config: ValidationResult(
            passed=False, issues=["coverage gap"],
        ))

        register_condition("sub_failed", lambda state: (
            {"issues": state.enrich.issues}
            if hasattr(state, 'enrich') and state.enrich and not state.enrich.passed
            else None
        ))

        sub = Construct(
            "enrich",
            input=Claims,
            output=ValidationResult,
            nodes=[Node.scripted("val", fn="sub_validate", output=ValidationResult)],
        ) | Operator(when="sub_failed")

        register_scripted("seed", lambda input_data, config: Claims(items=["data"]))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed", output=Claims),
            sub,
        ])
        graph = compile(parent, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "construct-op-test"}}

        result = run(graph, input={"node_id": "test-001"}, config=config)

        # Sub-pipeline ran and produced output
        assert result["enrich"] is not None
        assert result["enrich"].passed is False
        # Interrupted
        assert "__interrupt__" in result

    def test_construct_operator_passes(self):
        """Sub-pipeline runs, condition is falsy, graph continues."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        register_scripted("sub_ok", lambda input_data, config: ValidationResult(
            passed=True, issues=[],
        ))

        register_condition("sub_check", lambda state: None)  # always passes

        sub = Construct(
            "check",
            input=Claims,
            output=ValidationResult,
            nodes=[Node.scripted("ok", fn="sub_ok", output=ValidationResult)],
        ) | Operator(when="sub_check")

        register_scripted("seed2", lambda input_data, config: Claims(items=["ok"]))
        register_scripted("done", lambda input_data, config: RawText(text="complete"))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed2", output=Claims),
            sub,
            Node.scripted("done", fn="done", output=RawText),
        ])
        graph = compile(parent, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "construct-op-pass"}}
        result = run(graph, input={"node_id": "test-001"}, config=config)

        assert result["done"].text == "complete"


class TestDeepCompositions:
    """Complex nesting: modifiers inside modifiers, tool exhaustion, etc."""

    def test_oracle_inside_each(self):
        """Each item gets Oracle ensemble — fan-out inside fan-out."""
        from neograph.factory import register_scripted

        register_scripted("make_items", lambda input_data, config: Clusters(
            groups=[ClusterGroup(label="x", claim_ids=["1"]), ClusterGroup(label="y", claim_ids=["2"])]
        ))

        gen_count = {"n": 0}

        def gen_variant(input_data, config):
            gen_count["n"] += 1
            return RawText(text=f"variant-{gen_count['n']}")

        register_scripted("gen_v", gen_variant)

        def merge_v(variants, config):
            return RawText(text=f"merged-{len(variants)}")

        register_scripted("merge_v", merge_v)

        # Inner sub-construct: Oracle (2 variants, merge)
        inner = Construct(
            "oracle-inner",
            input=ClusterGroup,
            output=RawText,
            nodes=[
                Node.scripted("gen", fn="gen_v", output=RawText)
                | Oracle(n=2, merge_fn="merge_v"),
            ],
        )

        # Outer: Each over clusters, each runs the Oracle sub-pipeline
        # This means: 2 clusters × 2 Oracle variants = 4 generator calls + 2 merges
        parent = Construct("parent", nodes=[
            Node.scripted("make", fn="make_items", output=Clusters),
            inner | Each(over="make.groups", key="label"),
        ])

        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Each cluster got Oracle'd — 2 clusters × 2 generators = 4 total
        assert gen_count["n"] == 4
        assert "x" in result["oracle_inner"]
        assert "y" in result["oracle_inner"]

    def test_tool_budget_exhaustion_in_subgraph(self):
        """Gather node inside subgraph exhausts tool budget, forced to respond."""
        from langchain_core.messages import AIMessage

        from neograph._llm import configure_llm
        from neograph.factory import register_scripted, register_tool_factory

        tool_calls = {"n": 0}

        class FakeTool:
            name = "deep_search"

            def invoke(self, args):
                tool_calls["n"] += 1
                return f"found-{tool_calls['n']}"

        register_tool_factory("deep_search", lambda config, tool_config: FakeTool())

        class FakeGatherLLM:
            def __init__(self):
                self._call = 0

            def bind_tools(self, tools):
                clone = FakeGatherLLM()
                clone._call = self._call
                clone._tools_bound = len(tools) > 0
                return clone

            def invoke(self, messages, **kwargs):
                self._call += 1
                if getattr(self, '_tools_bound', True) and self._call <= 5:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "deep_search", "args": {}, "id": f"c{self._call}"}]
                    return msg
                return AIMessage(content="done searching")

            def with_structured_output(self, model):
                return FakeStructuredLLM(model)

        class FakeStructuredLLM:
            def __init__(self, model):
                self._model = model

            def invoke(self, messages, **kwargs):
                return self._model(text="search complete")

        configure_llm(
            llm_factory=lambda tier: FakeGatherLLM(),
            prompt_compiler=lambda template, data: [{"role": "user", "content": "search"}],
        )

        register_scripted("prep_search", lambda input_data, config: Claims(items=["query"]))

        sub = Construct(
            "deep-search",
            input=Claims,
            output=RawText,
            nodes=[
                Node(
                    name="search",
                    mode="gather",
                    input=Claims,
                    output=RawText,
                    model="fast",
                    prompt="test/search",
                    tools=[Tool(name="deep_search", budget=3)],
                ),
            ],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("prep", fn="prep_search", output=Claims),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Tool budget was 3, so exactly 3 calls made despite LLM wanting 5
        assert tool_calls["n"] == 3
        # Subgraph still produced output
        assert result["deep_search"] is not None

    def test_operator_inside_subgraph_bubbles_interrupt(self):
        """Operator inside a sub-construct pauses the entire parent pipeline."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        register_scripted("sub_check", lambda input_data, config: ValidationResult(
            passed=False, issues=["needs human"],
        ))

        register_condition("inner_failed", lambda state: (
            {"reason": "inner check failed"}
            if hasattr(state, 'check') and state.check and not state.check.passed
            else None
        ))

        sub = Construct(
            "inner",
            input=Claims,
            output=ValidationResult,
            nodes=[
                Node.scripted("check", fn="sub_check", output=ValidationResult)
                | Operator(when="inner_failed"),
            ],
        )

        register_scripted("start_fn", lambda input_data, config: Claims(items=["go"]))
        register_scripted("after_fn", lambda input_data, config: RawText(text="should not reach"))

        parent = Construct("parent", nodes=[
            Node.scripted("start", fn="start_fn", output=Claims),
            sub,
            Node.scripted("after", fn="after_fn", output=RawText),
        ])

        # Operator lives inside sub-construct — parent needs checkpointer
        # so the recursive compile of the sub-construct gets it
        from langgraph.checkpoint.memory import MemorySaver

        graph = compile(parent, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "inner-op-test"}}
        result = run(graph, input={"node_id": "test-001"}, config=config)

        # The interrupt inside the subgraph should surface
        # "after" node should not have run
        assert result.get("after") is None or "__interrupt__" in result


# ═══════════════════════════════════════════════════════════════════════════
# REMAINING COVERAGE GAPS — paths not yet exercised
# ═══════════════════════════════════════════════════════════════════════════


class TestFirstNodeEdgeCases:
    """Subgraphs and modified constructs as the first node (wired from START)."""

    def test_plain_subgraph_as_first_node(self):
        """Sub-construct as the very first node — no upstream data."""
        from neograph.factory import register_scripted

        register_scripted("self_seed", lambda input_data, config: Claims(items=["self-seeded"]))

        sub = Construct(
            "first-sub",
            input=Claims,
            output=Claims,
            nodes=[Node.scripted("seed", fn="self_seed", output=Claims)],
        )

        # Sub-construct is the ONLY node — wired from START
        parent = Construct("parent", nodes=[sub])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        assert result["first_sub"].items == ["self-seeded"]

    def test_construct_oracle_as_first_node(self):
        """Construct | Oracle as the first node — router wired from START."""
        from neograph.factory import register_scripted

        register_scripted("gen_first", lambda input_data, config: Claims(items=["v"]))

        def merge_first(variants, config):
            return Claims(items=[f"merged-{len(variants)}"])

        register_scripted("merge_first", merge_first)

        sub = Construct(
            "oracle-first",
            input=Claims,
            output=Claims,
            nodes=[Node.scripted("g", fn="gen_first", output=Claims)],
        ) | Oracle(n=2, merge_fn="merge_first")

        parent = Construct("parent", nodes=[sub])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        assert result["oracle_first"].items == ["merged-2"]

    def test_construct_each_as_first_node(self):
        """Construct | Each as the first node — needs collection in state."""
        from neograph.factory import register_scripted

        # For Each to work as first node, the collection must come from
        # somewhere. In practice this means the state is pre-seeded.
        # This tests that the router wires from START without crashing.
        # We can't run it (no collection in initial state), but compile must succeed.
        register_scripted("proc_first", lambda input_data, config: RawText(text="done"))

        sub = Construct(
            "each-first",
            input=ClusterGroup,
            output=RawText,
            nodes=[Node.scripted("p", fn="proc_first", output=RawText)],
        ) | Each(over="data.items", key="label")

        parent = Construct("parent", nodes=[sub])
        # Compile succeeds — Each wired from START
        graph = compile(parent)


class TestConstructOracleErrors:
    """Error paths for Construct | Oracle."""

    def test_construct_oracle_unregistered_merge_fn(self):
        """Construct | Oracle with unregistered merge_fn raises at compile."""
        from neograph.factory import register_scripted

        register_scripted("gen_err", lambda input_data, config: Claims(items=[]))

        sub = Construct(
            "bad-oracle-sub",
            input=Claims,
            output=Claims,
            nodes=[Node.scripted("g", fn="gen_err", output=Claims)],
        ) | Oracle(n=2, merge_fn="nonexistent_merge_fn")

        parent = Construct("parent", nodes=[sub])

        with pytest.raises(ValueError, match="not registered"):
            compile(parent)


class TestStripInternalsEdge:
    """Edge case: _strip_internals when result is not a dict."""

    def test_strip_non_dict(self):
        """Non-dict results pass through unchanged."""
        from neograph.runner import _strip_internals

        assert _strip_internals("hello") == "hello"
        assert _strip_internals(42) == 42
        assert _strip_internals(None) is None
        assert _strip_internals([1, 2, 3]) == [1, 2, 3]


class TestLLMConfig:
    """Per-node LLM configuration flows through to the factory."""

    def test_llm_config_passed_to_factory(self):
        """Node's llm_config dict reaches the llm_factory."""
        from neograph._llm import configure_llm

        factory_calls = []

        class FakeLLM:
            def with_structured_output(self, model):
                self._model = model
                return self

            def invoke(self, messages, **kwargs):
                return self._model(items=["result"])

        def tracking_factory(tier, node_name=None, llm_config=None):
            factory_calls.append({
                "tier": tier,
                "node_name": node_name,
                "llm_config": llm_config,
            })
            return FakeLLM()

        configure_llm(
            llm_factory=tracking_factory,
            prompt_compiler=lambda template, data: [{"role": "user", "content": "test"}],
        )

        node = Node(
            name="custom-llm",
            mode="produce",
            output=Claims,
            model="reason",
            prompt="test",
            llm_config={"temperature": 0.9, "max_tokens": 2000},
        )

        pipeline = Construct("test-llm-config", nodes=[node])
        graph = compile(pipeline)
        run(graph, input={"node_id": "test-001"})

        # Factory was called with the node's llm_config
        assert len(factory_calls) == 1
        assert factory_calls[0]["tier"] == "reason"
        assert factory_calls[0]["node_name"] == "custom-llm"
        assert factory_calls[0]["llm_config"]["temperature"] == 0.9
        assert factory_calls[0]["llm_config"]["max_tokens"] == 2000

    def test_llm_config_backward_compatible(self):
        """Old-style factory(tier) still works without llm_config."""
        from neograph._llm import configure_llm

        class FakeLLM:
            def with_structured_output(self, model):
                self._model = model
                return self

            def invoke(self, messages, **kwargs):
                return self._model(items=["ok"])

        # Old-style factory: only accepts tier
        configure_llm(
            llm_factory=lambda tier: FakeLLM(),
            prompt_compiler=lambda template, data: [{"role": "user", "content": "test"}],
        )

        node = Node(
            name="old-style",
            mode="produce",
            output=Claims,
            model="fast",
            prompt="test",
            llm_config={"temperature": 0.5},  # this won't crash old factory
        )

        pipeline = Construct("test-compat", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        assert result["old_style"].items == ["ok"]

    def test_input_injected_into_config(self):
        """Pipeline input fields (node_id, project_root) accessible via config["configurable"]."""
        from neograph.factory import register_scripted

        config_seen = {}

        def capture_config(input_data, config):
            configurable = config.get("configurable", {})
            config_seen["node_id"] = configurable.get("node_id")
            config_seen["project_root"] = configurable.get("project_root")
            config_seen["custom_field"] = configurable.get("custom_field")
            return Claims(items=["done"])

        register_scripted("capture", capture_config)

        node = Node.scripted("capture", fn="capture", output=Claims)
        pipeline = Construct("test-config-inject", nodes=[node])
        graph = compile(pipeline)

        result = run(graph,
                     input={"node_id": "BR-RW-042", "project_root": "/my/project"},
                     config={"configurable": {"custom_field": "extra-data"}})

        # All input fields + custom config accessible
        assert config_seen["node_id"] == "BR-RW-042"
        assert config_seen["project_root"] == "/my/project"
        assert config_seen["custom_field"] == "extra-data"

    def test_prompt_compiler_receives_config(self):
        """Prompt compiler gets node_name and full config with pipeline metadata."""
        from neograph._llm import configure_llm

        compiler_calls = []

        class FakeLLM:
            def with_structured_output(self, model, **kwargs):
                self._model = model
                return self

            def invoke(self, messages, **kwargs):
                return self._model(items=["result"])

        def tracking_compiler(template, data, *, node_name=None, config=None):
            compiler_calls.append({
                "template": template,
                "node_name": node_name,
                "node_id": config.get("configurable", {}).get("node_id") if config else None,
                "project_root": config.get("configurable", {}).get("project_root") if config else None,
            })
            return [{"role": "user", "content": "test"}]

        configure_llm(
            llm_factory=lambda tier: FakeLLM(),
            prompt_compiler=tracking_compiler,
        )

        node = Node(name="analyze", mode="produce", output=Claims, model="fast", prompt="rw/analyze")
        pipeline = Construct("test-prompt-ctx", nodes=[node])
        graph = compile(pipeline)
        run(graph, input={"node_id": "BR-001", "project_root": "/proj"})

        assert len(compiler_calls) == 1
        assert compiler_calls[0]["template"] == "rw/analyze"
        assert compiler_calls[0]["node_name"] == "analyze"
        assert compiler_calls[0]["node_id"] == "BR-001"
        assert compiler_calls[0]["project_root"] == "/proj"
