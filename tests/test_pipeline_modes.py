"""Pipeline mode tests — scripted, produce, gather, execute, raw modes,
output strategies, LLM config, config injection, error paths, and
migration-era rename tests (pending deletion in P7).
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from neograph import Construct, ConstructError, Node, Operator, Oracle, Each, Tool, compile, run, tool
from tests.fakes import FakeTool, ReActFake, StructuredFake, TextFake, configure_fake_llm


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
        import types as _types

        from neograph import construct_from_module, node

        mod = _types.ModuleType("test_scripted_pipeline_mod")

        @node(mode="scripted", outputs=RawText)
        def extract() -> RawText:
            return RawText(text="hello world")

        @node(mode="scripted", outputs=Claims)
        def split(extract: RawText) -> Claims:
            return Claims(items=["claim-1", "claim-2"])

        @node(mode="scripted", outputs=ClassifiedClaims)
        def classify(split: Claims) -> ClassifiedClaims:
            return ClassifiedClaims(
                classified=[{"claim": c, "category": "fact"} for c in split.items]
            )

        mod.extract = extract
        mod.split = split
        mod.classify = classify

        pipeline = construct_from_module(mod, name="test-scripted")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Verify data flowed through all three nodes
        assert isinstance(result["classify"], ClassifiedClaims)
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
        import types as _types

        from neograph import construct_from_module, node

        configure_fake_llm(
            lambda tier: StructuredFake(
                lambda m: m(items=["extracted-1", "extracted-2", "extracted-3"])
            )
        )

        mod = _types.ModuleType("test_produce_mode_mod")

        @node(mode="produce", outputs=Claims, model="fast", prompt="test/extract")
        def extract() -> Claims: ...

        mod.extract = extract

        pipeline = construct_from_module(mod, name="test-produce")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        assert isinstance(result["extract"], Claims)
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
        import types as _types

        from neograph import construct_from_module, node
        from neograph.factory import register_tool_factory

        search_tool = FakeTool("search_nodes", response="found")
        register_tool_factory("search_nodes", lambda config, tool_config: search_tool)

        # Fake LLM: calls tool twice, then responds (budget will cap at 2 anyway)
        fake = ReActFake(
            tool_calls=[
                [{"name": "search_nodes", "args": {"query": "test"}, "id": "call-1"}],
                [{"name": "search_nodes", "args": {"query": "test"}, "id": "call-2"}],
                [],  # stop — forced response
            ],
            final=lambda m: m(items=["done researching"]),
        )
        configure_fake_llm(lambda tier: fake)

        mod = _types.ModuleType("test_gather_budget_mod")

        @node(
            mode="gather",
            outputs=Claims,
            model="reason",
            prompt="test/explore",
            tools=[Tool(name="search_nodes", budget=2)],
        )
        def explore() -> Claims: ...

        mod.explore = explore

        pipeline = construct_from_module(mod, name="test-gather")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Tool was called exactly twice (budget=2)
        assert len(search_tool.calls) == 2

    def test_gather_unlimited_budget(self):
        """Tool with budget=0 is never exhausted — LLM decides when to stop."""
        import types as _types

        from neograph import construct_from_module, node
        from neograph.factory import register_tool_factory

        lookup_tool = FakeTool("lookup", response="result")
        register_tool_factory("lookup", lambda config, tool_config: lookup_tool)

        fake = ReActFake(
            tool_calls=[
                [{"name": "lookup", "args": {}, "id": "c1"}],
                [{"name": "lookup", "args": {}, "id": "c2"}],
                [{"name": "lookup", "args": {}, "id": "c3"}],
                [{"name": "lookup", "args": {}, "id": "c4"}],
                [{"name": "lookup", "args": {}, "id": "c5"}],
                [],  # LLM stops on its own
            ],
            final=lambda m: m(items=["done"]),
        )
        configure_fake_llm(lambda tier: fake)

        mod = _types.ModuleType("test_gather_unlimited_mod")

        @node(
            mode="gather",
            outputs=Claims,
            model="fast",
            prompt="test/scan",
            tools=[Tool(name="lookup", budget=0)],  # unlimited
        )
        def scan() -> Claims: ...

        mod.scan = scan

        pipeline = construct_from_module(mod, name="test-unlimited")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Tool was called 5 times — budget=0 never blocked it
        assert len(lookup_tool.calls) == 5


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Oracle — 3-way ensemble + merge
#
# Three parallel generators produce variants, barrier merges them.
# This proves: Oracle modifier expands to fan-out Send() + merge barrier,
# all three generators run, results converge.
# ═══════════════════════════════════════════════════════════════════════════

class TestRawNode:
    def test_raw_node_in_pipeline(self):
        """@node(mode='raw') works alongside declarative nodes."""
        import types as _types

        from neograph import construct_from_module, node

        mod = _types.ModuleType("test_raw_node_mod")

        @node(mode="scripted", outputs=Claims)
        def make_claims() -> Claims:
            return Claims(items=["a", "b", "c"])

        @node(mode="raw", inputs=Claims, outputs=Claims)
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

        mod.make_claims = make_claims
        mod.filter_claims = filter_claims

        pipeline = construct_from_module(mod, name="test-raw")
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
        import types as _types

        from neograph import construct_from_module, node

        def respond(model):
            if model is Claims:
                return Claims(items=["r1: system shall validate", "r2: system shall log"])
            if model is ClassifiedClaims:
                return ClassifiedClaims(classified=[
                    {"claim": "r1", "category": "requirement"},
                    {"claim": "r2", "category": "requirement"},
                ])
            return model()

        configure_fake_llm(lambda tier: StructuredFake(respond))

        mod = _types.ModuleType("test_mini_rw_mod")

        @node(mode="produce", outputs=Claims, model="reason", prompt="rw/decompose")
        def decompose() -> Claims: ...

        @node(mode="produce", outputs=ClassifiedClaims, model="fast", prompt="rw/classify")
        def classify(decompose: Claims) -> ClassifiedClaims: ...

        @node(mode="scripted", outputs=RawText)
        def catalog() -> RawText:
            return RawText(text="node catalog: 42 nodes")

        mod.decompose = decompose
        mod.classify = classify
        mod.catalog = catalog

        pipeline = construct_from_module(mod, name="mini-rw")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "BR-RW-042"})

        # All three nodes produced output
        assert isinstance(result["decompose"], Claims)
        assert isinstance(result["classify"], ClassifiedClaims)
        assert isinstance(result["catalog"], RawText)
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
        import types as _types

        from neograph import construct_from_module, node
        from neograph.factory import register_tool_factory

        write_tool = FakeTool("write_file", response="written")
        register_tool_factory("write_file", lambda config, tool_config: write_tool)

        fake = ReActFake(
            tool_calls=[
                [{"name": "write_file", "args": {"path": "out.txt"}, "id": "call-1"}],
                [],  # stop
            ],
            final=lambda m: m(text="done"),
        )
        configure_fake_llm(lambda tier: fake)

        mod = _types.ModuleType("test_execute_mode_mod")

        @node(
            mode="execute",
            outputs=RawText,
            model="fast",
            prompt="test/write",
            tools=[Tool(name="write_file", budget=1)],
        )
        def writer() -> RawText: ...

        mod.writer = writer

        pipeline = construct_from_module(mod, name="test-execute")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        assert len(write_tool.calls) == 1
        assert write_tool.calls[0]["path"] == "out.txt"


class TestErrorPaths:
    """Every error path the framework raises."""

    def test_produce_without_configure_llm(self):
        """Produce node without configure_llm() raises RuntimeError."""
        node = Node(name="fail", mode="produce", outputs=Claims, model="fast", prompt="x")
        pipeline = Construct("test-no-llm", nodes=[node])
        graph = compile(pipeline)

        with pytest.raises(ValueError, match="LLM not configured"):
            run(graph, input={"node_id": "test-001"})

    def test_unregistered_scripted_fn(self):
        """Referencing unregistered scripted function raises ValueError."""
        node = Node.scripted("bad", fn="nonexistent_fn", outputs=Claims)
        pipeline = Construct("test-bad-fn", nodes=[node])

        with pytest.raises(ValueError, match="not registered"):
            compile(pipeline)

    def test_unregistered_oracle_merge_fn(self):
        """Oracle with unregistered merge_fn raises ValueError at compile."""
        from neograph.factory import register_scripted

        register_scripted("gen", lambda input_data, config: Claims(items=["x"]))

        node = Node.scripted(
            "gen", fn="gen", outputs=Claims
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
            "something", fn="something", outputs=Claims
        ) | Operator(when="nonexistent_condition")

        pipeline = Construct("test-bad-condition", nodes=[node])

        with pytest.raises(ValueError, match="not registered"):
            compile(pipeline, checkpointer=MemorySaver())

    def test_unregistered_tool_factory(self):
        """Gather node with unregistered tool factory raises ValueError."""
        configure_fake_llm(lambda tier: ReActFake(tool_calls=[[]]))

        node = Node(
            name="explore",
            mode="gather",
            outputs=Claims,
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
        node = Node.scripted("noop", fn="noop", outputs=Claims)
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

        search_tool = FakeTool("search", response="found it")
        register_tool_factory("search", lambda config, tool_config: search_tool)

        call_counter = {"n": 0}

        class FakeLLMHallucinator:
            def __init__(self):
                self._structured = False

            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kwargs):
                if self._structured:
                    return self._model(items=["recovered"])
                call_counter["n"] += 1
                if call_counter["n"] == 1:
                    # LLM hallucinates a tool that doesn't exist
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "delete_everything", "args": {}, "id": "bad-1"}]
                    return msg
                return AIMessage(content="ok done")

            def with_structured_output(self, model, **kwargs):
                clone = FakeLLMHallucinator()
                clone._structured = True
                clone._model = model
                return clone

        configure_llm(
            llm_factory=lambda tier: FakeLLMHallucinator(),
            prompt_compiler=lambda template, data: [{"role": "user", "content": "test"}],
        )

        node = Node(
            name="explore",
            mode="gather",
            outputs=Claims,
            model="fast",
            prompt="test/explore",
            tools=[Tool(name="search", budget=5)],
        )

        pipeline = Construct("test-hallucinated-tool", nodes=[node])
        graph = compile(pipeline)
        # Should complete without crashing — unknown tool gets error message
        result = run(graph, input={"node_id": "test-001"})
        assert isinstance(result["explore"], Claims)
        assert result["explore"].items == ["recovered"]


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
            nodes=[Node.scripted("seed", fn="self_seed", outputs=Claims)],
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
            nodes=[Node.scripted("g", fn="gen_first", outputs=Claims)],
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
            nodes=[Node.scripted("p", fn="proc_first", outputs=RawText)],
        ) | Each(over="data.items", key="label")

        parent = Construct("parent", nodes=[sub])
        # Compile succeeds — Each wired from START
        graph = compile(parent)


# ═══════════════════════════════════════════════════════════════════════════
# Assembly-time type validation — compile errors surface at Construct(...)
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMConfig:
    """Per-node LLM configuration flows through to the factory."""

    def test_llm_config_passed_to_factory(self):
        """Node's llm_config dict reaches the llm_factory."""
        factory_calls = []

        def tracking_factory(tier, node_name=None, llm_config=None):
            factory_calls.append({
                "tier": tier,
                "node_name": node_name,
                "llm_config": llm_config,
            })
            return StructuredFake(lambda m: m(items=["result"]))

        configure_fake_llm(tracking_factory)

        node = Node(
            name="custom-llm",
            mode="produce",
            outputs=Claims,
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
        # Old-style factory: only accepts tier
        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["ok"])))

        node = Node(
            name="old-style",
            mode="produce",
            outputs=Claims,
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

        node = Node.scripted("capture", fn="capture", outputs=Claims)
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
        compiler_calls = []

        def tracking_compiler(template, data, *, node_name=None, config=None):
            compiler_calls.append({
                "template": template,
                "node_name": node_name,
                "node_id": config.get("configurable", {}).get("node_id") if config else None,
                "project_root": config.get("configurable", {}).get("project_root") if config else None,
            })
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            lambda tier: StructuredFake(lambda m: m(items=["result"])),
            prompt_compiler=tracking_compiler,
        )

        node = Node(name="analyze", mode="produce", outputs=Claims, model="fast", prompt="rw/analyze")
        pipeline = Construct("test-prompt-ctx", nodes=[node])
        graph = compile(pipeline)
        run(graph, input={"node_id": "BR-001", "project_root": "/proj"})

        assert len(compiler_calls) == 1
        assert compiler_calls[0]["template"] == "rw/analyze"
        assert compiler_calls[0]["node_name"] == "analyze"
        assert compiler_calls[0]["node_id"] == "BR-001"
        assert compiler_calls[0]["project_root"] == "/proj"


class TestRunIsolated:
    """Node.run_isolated() — direct invocation for unit testing."""

    def test_scripted_node_run_isolated(self):
        """Scripted nodes can be tested without compile()/run()."""
        from neograph import register_scripted

        register_scripted("upper", lambda input_data, config: RawText(
            text=input_data.text.upper() if input_data else "NONE"
        ))

        upper_node = Node.scripted("upper", fn="upper", inputs=RawText, outputs=RawText)

        # Direct invocation — no pipeline, no compile, no run
        result = upper_node.run_isolated(input=RawText(text="hello"))

        assert isinstance(result, RawText)
        assert result.text == "HELLO"

    def test_produce_node_run_isolated(self):
        """Produce nodes can be tested with a fake LLM."""
        configure_fake_llm(lambda tier: StructuredFake(
            lambda model: model(items=["isolated-result"])
        ))

        decompose = Node(
            "decompose", mode="produce", outputs=Claims, model="fast", prompt="test"
        )

        result = decompose.run_isolated()

        assert isinstance(result, Claims)
        assert result.items == ["isolated-result"]

    def test_run_isolated_with_config(self):
        """run_isolated passes config through to the node function."""
        from neograph import register_scripted

        seen_config = {}
        def fn(input_data, config):
            seen_config.update(config.get("configurable", {}))
            return Claims(items=["ok"])

        register_scripted("cfg_test", fn)
        node = Node.scripted("cfg-test", fn="cfg_test", outputs=Claims)

        result = node.run_isolated(
            config={"configurable": {"node_id": "TEST-001", "env": "staging"}}
        )

        assert result.items == ["ok"]
        assert seen_config["node_id"] == "TEST-001"
        assert seen_config["env"] == "staging"


class TestConfigInjectionPatterns:
    """Real-world config injection patterns from piarch consumer."""

    def test_shared_resources_in_config(self):
        """Consumer passes shared infrastructure (rate limiter, tracer) via config."""
        from neograph.factory import register_scripted

        class FakeRateLimiter:
            def __init__(self):
                self.calls = 0

            def call(self):
                self.calls += 1

        limiter = FakeRateLimiter()

        def node_with_resources(input_data, config):
            rl = config["configurable"]["rate_limiter"]
            rl.call()
            return Claims(items=[f"calls={rl.calls}"])

        register_scripted("resourced", node_with_resources)

        pipeline = Construct("test-resources", nodes=[
            Node.scripted("step", fn="resourced", outputs=Claims),
        ])
        graph = compile(pipeline)
        result = run(
            graph,
            input={"node_id": "test"},
            config={"configurable": {"rate_limiter": limiter}},
        )

        assert limiter.calls == 1
        assert result["step"].items == ["calls=1"]

    def test_config_available_in_subgraph(self):
        """Config flows through to sub-construct node functions."""
        from neograph.factory import register_scripted

        seen_in_sub = {}

        def sub_node(input_data, config):
            seen_in_sub["node_id"] = config.get("configurable", {}).get("node_id")
            seen_in_sub["custom"] = config.get("configurable", {}).get("custom")
            return RawText(text="sub-done")

        register_scripted("sub_node", sub_node)
        register_scripted("parent_seed", lambda input_data, config: Claims(items=["x"]))

        sub = Construct(
            "sub",
            input=Claims,
            output=RawText,
            nodes=[Node.scripted("inner", fn="sub_node", outputs=RawText)],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="parent_seed", outputs=Claims),
            sub,
        ])
        graph = compile(parent)
        run(graph,
            input={"node_id": "BR-042"},
            config={"configurable": {"custom": "my-value"}})

        # Sub-construct node function received both pipeline input and custom config
        assert seen_in_sub["node_id"] == "BR-042"
        assert seen_in_sub["custom"] == "my-value"

    def test_config_available_in_oracle_generators(self):
        """Each Oracle generator sees config with node_id and custom fields."""
        from neograph.factory import register_scripted

        gen_configs = []

        def gen_fn(input_data, config):
            gen_configs.append({
                "node_id": config.get("configurable", {}).get("node_id"),
                "project_root": config.get("configurable", {}).get("project_root"),
                "gen_id": config.get("configurable", {}).get("_generator_id"),
            })
            return Claims(items=["v"])

        def merge_fn(variants, config):
            return Claims(items=[f"merged-{len(variants)}"])

        register_scripted("cfg_gen", gen_fn)
        register_scripted("cfg_merge", merge_fn)

        node = Node.scripted(
            "gen", fn="cfg_gen", outputs=Claims
        ) | Oracle(n=3, merge_fn="cfg_merge")

        pipeline = Construct("test-oracle-config", nodes=[node])
        graph = compile(pipeline)
        run(graph,
            input={"node_id": "BR-099", "project_root": "/proj"},
            config={})

        # All 3 generators saw the pipeline metadata
        assert len(gen_configs) == 3
        for gc in gen_configs:
            assert gc["node_id"] == "BR-099"
            assert gc["project_root"] == "/proj"
            assert gc["gen_id"] is not None  # generator ID also present

    def test_config_available_in_each_fanout(self):
        """Each fan-out node sees config with node_id."""
        from neograph.factory import register_scripted

        each_configs = []

        register_scripted("make_groups", lambda input_data, config: Clusters(
            groups=[ClusterGroup(label="a", claim_ids=["1"]),
                    ClusterGroup(label="b", claim_ids=["2"])]
        ))

        def verify_with_config(input_data, config):
            each_configs.append({
                "node_id": config.get("configurable", {}).get("node_id"),
                "label": input_data.label if hasattr(input_data, "label") else "?",
            })
            return MatchResult(cluster_label=input_data.label, matched=["ok"])

        register_scripted("verify_cfg", verify_with_config)

        make = Node.scripted("make", fn="make_groups", outputs=Clusters)
        verify = Node.scripted(
            "verify", fn="verify_cfg", inputs=ClusterGroup, outputs=MatchResult
        ) | Each(over="make.groups", key="label")

        pipeline = Construct("test-each-config", nodes=[make, verify])
        graph = compile(pipeline)
        run(graph, input={"node_id": "BR-050"})

        # Both fan-out invocations saw node_id
        assert len(each_configs) == 2
        assert all(c["node_id"] == "BR-050" for c in each_configs)
        labels = {c["label"] for c in each_configs}
        assert labels == {"a", "b"}

    def test_multi_node_pipeline_all_see_config(self):
        """Every node in a multi-step pipeline sees the same config metadata."""
        from neograph.factory import register_scripted

        nodes_seen = []

        def tracking_fn(input_data, config):
            nodes_seen.append({
                "node_id": config.get("configurable", {}).get("node_id"),
                "env": config.get("configurable", {}).get("env"),
            })
            return Claims(items=["done"])

        register_scripted("track_a", tracking_fn)
        register_scripted("track_b", tracking_fn)
        register_scripted("track_c", tracking_fn)

        pipeline = Construct("test-all-config", nodes=[
            Node.scripted("a", fn="track_a", outputs=Claims),
            Node.scripted("b", fn="track_b", outputs=Claims),
            Node.scripted("c", fn="track_c", outputs=Claims),
        ])
        graph = compile(pipeline)
        run(graph,
            input={"node_id": "REQ-001"},
            config={"configurable": {"env": "staging"}})

        assert len(nodes_seen) == 3
        for seen in nodes_seen:
            assert seen["node_id"] == "REQ-001"
            assert seen["env"] == "staging"


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT STRATEGIES — structured, json_mode, text
# ═══════════════════════════════════════════════════════════════════════════


class TestOutputStrategyStructured:
    """Default strategy: llm.with_structured_output(model). Current behavior."""

    def test_structured_output_default(self):
        """Produce node uses with_structured_output by default."""
        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["via-structured"])))

        node = Node(name="extract", mode="produce", outputs=Claims, model="fast", prompt="test")
        pipeline = Construct("test-structured", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert result["extract"].items == ["via-structured"]


class TestOutputStrategyJsonMode:
    """json_mode strategy: inject schema into prompt, parse response as JSON."""

    def test_json_mode_injects_schema_and_parses(self):
        """json_mode: schema injected into prompt, LLM returns raw JSON, framework parses."""
        compiler_calls = []

        def tracking_compiler(template, data, **kw):
            compiler_calls.append(kw)
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            lambda tier: TextFake('{"items": ["via-json-mode"]}'),
            prompt_compiler=tracking_compiler,
        )

        node = Node(
            name="extract",
            mode="produce",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-json-mode", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        # Parsed correctly from raw JSON
        assert result["extract"].items == ["via-json-mode"]

    def test_json_mode_handles_markdown_fences(self):
        """json_mode: strips markdown code fences before parsing."""
        configure_fake_llm(lambda tier: TextFake('```json\n{"items": ["fenced"]}\n```'))

        node = Node(
            name="extract",
            mode="produce",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-json-fence", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert result["extract"].items == ["fenced"]


class TestOutputStrategyText:
    """text strategy: LLM returns plain text, consumer's prompt_compiler handles schema."""

    def test_text_mode_parses_json_from_response(self):
        """text mode: LLM returns text containing JSON, framework extracts and parses."""
        configure_fake_llm(
            lambda tier: TextFake('Here is my analysis:\n{"items": ["from-text"]}\nDone.')
        )

        node = Node(
            name="extract",
            mode="produce",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "text"},
        )
        pipeline = Construct("test-text", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert result["extract"].items == ["from-text"]


class TestOutputStrategyOnGather:
    """Output strategy also applies to the final parse in gather/execute modes."""

    def test_gather_json_mode_parses_final_response(self):
        """Gather node with json_mode parses the final structured output from raw JSON."""
        from langchain_core.messages import AIMessage

        from neograph._llm import configure_llm
        from neograph.factory import register_tool_factory

        lookup_tool = FakeTool("lookup", response="found")
        register_tool_factory("lookup", lambda config, tool_config: lookup_tool)

        # Custom fake: json_mode gather requires specific JSON text as final response,
        # which ReActFake's hardcoded "done" text can't provide.
        call_n = {"n": 0}

        class FakeGatherLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kwargs):
                call_n["n"] += 1
                if call_n["n"] == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "lookup", "args": {}, "id": "c1"}]
                    return msg
                return AIMessage(content='{"items": ["gathered-json"]}')

        configure_llm(
            llm_factory=lambda tier: FakeGatherLLM(),
            prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": "go"}],
        )

        node = Node(
            name="research",
            mode="gather",
            outputs=Claims,
            model="fast",
            prompt="test",
            tools=[Tool(name="lookup", budget=1)],
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-gather-json", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert result["research"].items == ["gathered-json"]


class TestPromptCompilerReceivesOutputModel:
    """prompt_compiler must receive output_model and llm_config for json_mode prompts."""

    def test_compiler_receives_output_model_in_produce(self):
        """Produce node's prompt compiler sees the output Pydantic model."""
        compiler_calls = []

        def tracking_compiler(template, data, **kw):
            compiler_calls.append(kw)
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            lambda tier: StructuredFake(lambda m: m(items=["ok"])),
            prompt_compiler=tracking_compiler,
        )

        node = Node(name="x", mode="produce", outputs=Claims, model="fast", prompt="test")
        pipeline = Construct("test", nodes=[node])
        graph = compile(pipeline)
        run(graph, input={"node_id": "test"})

        assert len(compiler_calls) == 1
        assert compiler_calls[0].get("output_model") is Claims

    def test_compiler_receives_llm_config(self):
        """Prompt compiler sees the node's llm_config including output_strategy."""
        compiler_calls = []

        def tracking_compiler(template, data, **kw):
            compiler_calls.append(kw)
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            lambda tier: TextFake('{"items": ["ok"]}'),
            prompt_compiler=tracking_compiler,
        )

        node = Node(
            name="x", mode="produce", outputs=Claims, model="fast", prompt="test",
            llm_config={"output_strategy": "json_mode", "temperature": 0.5},
        )
        pipeline = Construct("test", nodes=[node])
        graph = compile(pipeline)
        run(graph, input={"node_id": "test"})

        assert len(compiler_calls) == 1
        assert compiler_calls[0].get("llm_config") == {"output_strategy": "json_mode", "temperature": 0.5}
        assert compiler_calls[0].get("output_model") is Claims

    def test_json_mode_compiler_can_inject_schema(self):
        """End-to-end: compiler injects JSON schema into prompt for json_mode."""
        import json

        injected_prompts = []

        def schema_injecting_compiler(template, data, **kw):
            output_model = kw.get("output_model")
            llm_config = kw.get("llm_config", {})
            strategy = llm_config.get("output_strategy", "structured")

            prompt = f"Analyze: {template}"
            if strategy in ("json_mode", "text") and output_model:
                schema = json.dumps(output_model.model_json_schema(), indent=2)
                prompt += f"\n\nReturn a JSON object matching this schema:\n{schema}"

            injected_prompts.append(prompt)
            return [{"role": "user", "content": prompt}]

        configure_fake_llm(
            lambda tier: TextFake('{"items": ["schema-injected"]}'),
            prompt_compiler=schema_injecting_compiler,
        )

        node = Node(
            name="x", mode="produce", outputs=Claims, model="fast", prompt="decompose",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert result["x"].items == ["schema-injected"]
        # Verify schema was injected into the prompt
        assert "json_schema" in injected_prompts[0] or "items" in injected_prompts[0]
        assert "Return a JSON object" in injected_prompts[0]


# ═══════════════════════════════════════════════════════════════════════════
# TestNodeDecorator — @node + construct_from_module (Dagster-style signatures)
#
# Parameter names in the decorated function name the upstream nodes. The
# decorator produces a plain Node; construct_from_module walks a module's
# @node-built nodes and topologically sorts them into a Construct. No new
# IR path — compile()/run() handle the result unchanged.
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeInputsFieldRename:
    def test_node_accepts_inputs_kwarg_single_type(self):
        """Node(inputs=SomeType) creates a node with .inputs == SomeType."""
        n = Node("t", mode="scripted", inputs=Claims, outputs=RawText)
        assert n.inputs == Claims

    def test_node_accepts_inputs_kwarg_dict_form(self):
        """Node(inputs={'a': A, 'b': B}) stores the dict on .inputs."""
        n = Node(
            "t",
            mode="scripted",
            inputs={"claims": Claims, "clusters": Clusters},
            outputs=MatchResult,
        )
        assert n.inputs == {"claims": Claims, "clusters": Clusters}

    def test_node_accepts_inputs_none_default(self):
        """Node with no inputs kwarg keeps the default (None)."""
        n = Node("t", mode="scripted", outputs=RawText)
        assert n.inputs is None

    def test_node_scripted_classmethod_accepts_inputs_kwarg(self):
        """Node.scripted(..., inputs=X) propagates to .inputs."""
        n = Node.scripted("verify", fn="noop", inputs=ClusterGroup, outputs=MatchResult)
        assert n.inputs == ClusterGroup
        assert n.mode == "scripted"

    def test_node_has_no_legacy_input_attribute(self):
        """.input attribute no longer exists on Node instances."""
        n = Node("t", mode="scripted", inputs=Claims, outputs=RawText)
        assert not hasattr(n, "input"), (
            "Node still exposes legacy .input attribute — rename is incomplete."
        )


# ═══════════════════════════════════════════════════════════════════════════
# TestExtractInputListUnwrap (neograph-kqd.3)
#
# Factory runtime unwrap: dict[str, X] state value → list[X] for consumers
# whose inputs dict declares a list[X] expected type. This is the runtime
# half of the merge-after-fanout pattern (validator side landed in kqd.2).
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractInputListUnwrap:
    def test_list_consumer_unwraps_dict_state_to_list(self):
        """inputs={'upstream': list[MatchResult]} + dict state value
        → list of values in dict insertion order."""
        from neograph.factory import _extract_input

        consumer = Node.scripted(
            "consumer", fn="f",
            inputs={"make_clusters": list[MatchResult]},
            outputs=MergedResult,
        )
        mr1 = MatchResult(cluster_label="a", matched=["x"])
        mr2 = MatchResult(cluster_label="b", matched=["y"])
        state = {"make_clusters": {"a": mr1, "b": mr2}}
        result = _extract_input(state, consumer)
        assert isinstance(result, dict)
        assert "make_clusters" in result
        assert result["make_clusters"] == [mr1, mr2]

    def test_dict_consumer_keeps_dict_state(self):
        """inputs={'upstream': dict[str, MatchResult]} + dict state value
        → dict pass-through, no unwrap."""
        from neograph.factory import _extract_input

        consumer = Node.scripted(
            "consumer", fn="f",
            inputs={"make_clusters": dict[str, MatchResult]},
            outputs=MergedResult,
        )
        mr1 = MatchResult(cluster_label="a", matched=["x"])
        mr2 = MatchResult(cluster_label="b", matched=["y"])
        state = {"make_clusters": {"a": mr1, "b": mr2}}
        result = _extract_input(state, consumer)
        assert result["make_clusters"] == {"a": mr1, "b": mr2}

    def test_list_consumer_over_none_state_returns_none(self):
        """Missing state field (None) is not unwrapped — passthrough None."""
        from neograph.factory import _extract_input

        consumer = Node.scripted(
            "consumer", fn="f",
            inputs={"make_clusters": list[MatchResult]},
            outputs=MergedResult,
        )
        state = {}  # no make_clusters field
        result = _extract_input(state, consumer)
        assert result["make_clusters"] is None


# ═══════════════════════════════════════════════════════════════════════════
# TestNodeDecoratorDictInputs (neograph-kqd.4)
#
# @node decoration now emits dict-form inputs={param_name: annotation, ...}
# for all typed upstream params. This is the metadata shift that lets
# step-2's validator catch fan-in mismatches via _check_fan_in_inputs.
# Fan-out params (Each) are stripped from inputs at construct-assembly time.
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeOutputsRename:
    """Node.output → Node.outputs rename (neograph-1bp.1).

    The field on Node becomes `outputs` (plural), mirroring `inputs`.
    Construct.output stays singular — different semantics (boundary port).
    """

    def test_node_constructor_outputs_kwarg(self):
        """Node(..., outputs=X) sets the field."""
        n = Node("a", mode="produce", outputs=Claims, model="fast", prompt="p")
        assert n.outputs is Claims

    def test_node_scripted_outputs_kwarg(self):
        """Node.scripted(..., outputs=X) sets the field."""
        n = Node.scripted("a", fn="f", outputs=Claims)
        assert n.outputs is Claims

    def test_node_outputs_field_access(self):
        """node.outputs returns the declared output type."""
        n = Node("a", outputs=RawText, model="fast", prompt="p")
        assert n.outputs is RawText

    def test_decorator_outputs_kwarg(self):
        """@node(outputs=X) passes through to Node.outputs."""
        from neograph import node

        @node(outputs=Claims, mode="produce", model="fast", prompt="p")
        def my_node(seed: RawText) -> Claims: ...

        assert my_node.outputs is Claims

    def test_construct_output_stays_singular(self):
        """Construct.output stays as 'output' (singular), not renamed."""
        c = Construct("p", output=Claims, input=RawText, nodes=[
            Node.scripted("a", fn="f", outputs=Claims),
        ])
        assert c.output is Claims

    def test_effective_producer_type_reads_outputs(self):
        """effective_producer_type reads .outputs from Node items."""
        from neograph._construct_validation import effective_producer_type
        n = Node("a", outputs=Claims)
        assert effective_producer_type(n) is Claims

    def test_state_compiler_reads_outputs(self):
        """compile_state_model reads node.outputs for state field types."""
        from neograph.state import compile_state_model
        n = Node.scripted("extract", fn="f", outputs=RawText)
        c = Construct("p", nodes=[n])
        state_model = compile_state_model(c)
        # The state field should have the right type annotation
        assert "extract" in state_model.model_fields


class TestGatherToolCollection:
    """invoke_with_tools collects ToolInteraction records (neograph-1bp.6)."""

    def test_tool_interaction_model(self):
        """ToolInteraction has the expected fields."""
        from neograph import ToolInteraction
        ti = ToolInteraction(tool_name="search", args={"q": "test"}, result="found", duration_ms=42)
        assert ti.tool_name == "search"
        assert ti.args == {"q": "test"}
        assert ti.result == "found"
        assert ti.duration_ms == 42

    def test_tool_interactions_collected(self):
        """Gather node with dict outputs writes tool_log to state."""
        from neograph import node, construct_from_module, compile, run, tool, ToolInteraction
        from neograph.factory import register_tool_factory
        from tests.fakes import FakeTool, ReActFake, configure_fake_llm
        import types

        configure_fake_llm(
            lambda tier: ReActFake(
                tool_calls=[
                    [{"name": "search", "args": {"q": "test"}, "id": "tc1"}],
                    [],  # final response
                ],
                final=lambda m: Claims(items=["result"]),
            )
        )

        # Register a tool factory that returns a fake tool
        fake_tool = FakeTool("search", response="found it")
        register_tool_factory("search", lambda config, tool_config: fake_tool)

        mod = types.ModuleType("test_tool_collection_mod")

        search_tool = Tool("search", budget=3)

        @node(
            mode="gather",
            outputs={"result": Claims, "tool_log": list[ToolInteraction]},
            model="fast",
            prompt="test",
            tools=[search_tool],
        )
        def explore() -> Claims: ...

        mod.explore = explore
        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        result = run(graph, input={})
        assert result.get("explore_result") == Claims(items=["result"])
        tool_log = result.get("explore_tool_log")
        assert tool_log is not None
        assert len(tool_log) == 1
        assert tool_log[0].tool_name == "search"
        assert tool_log[0].result == "found it"
