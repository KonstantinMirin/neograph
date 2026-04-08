"""Pipeline mode tests — scripted, produce, gather, execute, raw modes,
output strategies, LLM config, config injection, error paths, and
migration-era rename tests (pending deletion in P7).
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest

from neograph import Construct, ConstructError, Node, Operator, Oracle, Each, Tool, compile, run, tool
from neograph import CompileError, ConfigurationError, ExecutionError
from tests.fakes import FakeTool, ReActFake, StructuredFake, TextFake, configure_fake_llm
from tests.schemas import RawText, Claims, ClassifiedClaims, ClusterGroup, Clusters, MatchResult, MergedResult, ValidationResult


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Scripted pipeline end-to-end
#
# Three deterministic nodes. No LLM. Data flows A→B→C through typed state.
# This proves: compile() builds a working graph, run() executes it,
# state bus passes data between nodes.
# ═══════════════════════════════════════════════════════════════════════════

class TestScriptedPipeline:
    def test_data_flows_through_when_three_scripted_nodes_chained(self):
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
    def test_structured_output_returned_when_produce_mode_with_fake_llm(self):
        """Produce node calls LLM and gets structured output."""
        import types as _types

        from neograph import construct_from_module, node

        configure_fake_llm(
            lambda tier: StructuredFake(
                lambda m: m(items=["extracted-1", "extracted-2", "extracted-3"])
            )
        )

        mod = _types.ModuleType("test_produce_mode_mod")

        @node(mode="think", outputs=Claims, model="fast", prompt="test/extract")
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
    def test_tool_called_within_budget_when_gather_mode(self):
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
            mode="agent",
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

    def test_tool_called_five_times_when_budget_unlimited(self):
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
            mode="agent",
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
    def test_raw_node_filters_when_mixed_with_declarative(self):
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
    def test_all_outputs_present_when_produce_and_scripted_mixed(self):
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

        @node(mode="think", outputs=Claims, model="reason", prompt="rw/decompose")
        def decompose() -> Claims: ...

        @node(mode="think", outputs=ClassifiedClaims, model="fast", prompt="rw/classify")
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

    def test_tool_called_when_execute_mode_with_mutation_tools(self):
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
            mode="act",
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

    def test_runtime_raises_when_llm_not_configured(self):
        """Produce node without configure_llm() raises RuntimeError."""
        node = Node(name="fail", mode="think", outputs=Claims, model="fast", prompt="x")
        pipeline = Construct("test-no-llm", nodes=[node])
        graph = compile(pipeline)

        with pytest.raises(ConfigurationError, match="LLM not configured"):
            run(graph, input={"node_id": "test-001"})

    def test_compile_raises_when_scripted_fn_not_registered(self):
        """Referencing unregistered scripted function raises ConfigurationError."""
        node = Node.scripted("bad", fn="nonexistent_fn", outputs=Claims)
        pipeline = Construct("test-bad-fn", nodes=[node])

        with pytest.raises(ConfigurationError, match="not registered"):
            compile(pipeline)

    def test_compile_raises_when_oracle_merge_fn_not_registered(self):
        """Oracle with unregistered merge_fn raises ConfigurationError at compile."""
        from neograph.factory import register_scripted

        register_scripted("gen", lambda input_data, config: Claims(items=["x"]))

        node = Node.scripted(
            "gen", fn="gen", outputs=Claims
        ) | Oracle(n=2, merge_fn="nonexistent_merge")

        pipeline = Construct("test-bad-merge", nodes=[node])

        with pytest.raises(ConfigurationError, match="not registered"):
            compile(pipeline)

    def test_compile_raises_when_operator_condition_not_registered(self):
        """Operator with unregistered condition raises ConfigurationError at compile."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_scripted

        register_scripted("something", lambda input_data, config: Claims(items=[]))

        node = Node.scripted(
            "something", fn="something", outputs=Claims
        ) | Operator(when="nonexistent_condition")

        pipeline = Construct("test-bad-condition", nodes=[node])

        with pytest.raises(ConfigurationError, match="not registered"):
            compile(pipeline, checkpointer=MemorySaver())

    def test_runtime_raises_when_tool_factory_not_registered(self):
        """Gather node with unregistered tool factory raises ConfigurationError."""
        configure_fake_llm(lambda tier: ReActFake(tool_calls=[[]]))

        node = Node(
            name="explore",
            mode="agent",
            outputs=Claims,
            model="fast",
            prompt="x",
            tools=[Tool(name="ghost_tool", budget=1)],
        )

        pipeline = Construct("test-bad-tool", nodes=[node])
        graph = compile(pipeline)

        with pytest.raises(ConfigurationError, match="not registered"):
            run(graph, input={"node_id": "test-001"})

    def test_run_raises_when_neither_input_nor_resume(self):
        """run() with neither input nor resume raises ValueError."""
        from neograph.factory import register_scripted

        register_scripted("noop", lambda input_data, config: Claims(items=[]))
        node = Node.scripted("noop", fn="noop", outputs=Claims)
        pipeline = Construct("test-no-args", nodes=[node])
        graph = compile(pipeline)

        with pytest.raises(ValueError, match="input or resume"):
            run(graph)

    def test_compile_raises_when_node_missing_output_type(self):
        """Node with no output type raises CompileError at compile."""
        node = Node(name="bad-node", mode="think", model="fast", prompt="x")
        pipeline = Construct("test-no-output", nodes=[node])

        with pytest.raises(CompileError, match="no output type"):
            compile(pipeline)


class TestLLMUnknownToolCall:
    """LLM hallucinates a tool name the framework doesn't have."""

    def test_framework_recovers_when_llm_hallucinates_tool_name(self):
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
            mode="agent",
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

    def test_sub_construct_runs_when_first_node_in_parent(self):
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

    def test_oracle_construct_runs_when_first_node_in_parent(self):
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

    def test_each_construct_compiles_when_first_node_in_parent(self):
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

    def test_factory_receives_config_when_llm_config_set(self):
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
            mode="think",
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

    def test_old_factory_works_when_llm_config_present(self):
        """Old-style factory(tier) still works without llm_config."""
        # Old-style factory: only accepts tier
        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["ok"])))

        node = Node(
            name="old-style",
            mode="think",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"temperature": 0.5},  # this won't crash old factory
        )

        pipeline = Construct("test-compat", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        assert result["old_style"].items == ["ok"]

    def test_configurable_has_fields_when_input_provided(self):
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

    def test_compiler_sees_metadata_when_node_name_and_config_provided(self):
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

        node = Node(name="analyze", mode="think", outputs=Claims, model="fast", prompt="rw/analyze")
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

    def test_result_returned_when_scripted_node_run_isolated(self):
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

    def test_result_returned_when_produce_node_run_isolated(self):
        """Produce nodes can be tested with a fake LLM."""
        configure_fake_llm(lambda tier: StructuredFake(
            lambda model: model(items=["isolated-result"])
        ))

        decompose = Node(
            "decompose", mode="think", outputs=Claims, model="fast", prompt="test"
        )

        result = decompose.run_isolated()

        assert isinstance(result, Claims)
        assert result.items == ["isolated-result"]

    def test_config_passed_through_when_run_isolated_with_config(self):
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

    def test_resource_invoked_when_passed_via_config(self):
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

    def test_sub_node_sees_config_when_parent_passes_it(self):
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

    def test_all_generators_see_config_when_oracle_with_metadata(self):
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

    def test_each_invocation_sees_config_when_metadata_present(self):
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

    def test_every_node_sees_config_when_multi_step_pipeline(self):
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

    def test_structured_output_used_when_no_strategy_specified(self):
        """Produce node uses with_structured_output by default."""
        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["via-structured"])))

        node = Node(name="extract", mode="think", outputs=Claims, model="fast", prompt="test")
        pipeline = Construct("test-structured", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert result["extract"].items == ["via-structured"]


class TestOutputStrategyJsonMode:
    """json_mode strategy: inject schema into prompt, parse response as JSON."""

    def test_json_parsed_when_json_mode_strategy_set(self):
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
            mode="think",
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

    def test_fences_stripped_when_json_response_wrapped_in_markdown(self):
        """json_mode: strips markdown code fences before parsing."""
        configure_fake_llm(lambda tier: TextFake('```json\n{"items": ["fenced"]}\n```'))

        node = Node(
            name="extract",
            mode="think",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-json-fence", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert result["extract"].items == ["fenced"]

    def test_error_feedback_retry_recovers_when_first_response_is_garbage(self):
        """json_mode: when first response is unparseable, the framework retries
        with the error fed back to the LLM. If the second attempt succeeds,
        the node returns normally."""
        from langchain_core.messages import AIMessage

        call_n = {"n": 0}

        class RetryableFake:
            def invoke(self, messages, **kwargs):
                call_n["n"] += 1
                if call_n["n"] == 1:
                    # First call: garbage response
                    return AIMessage(content="I think the answer is maybe some claims")
                # Second call (after error feedback): valid JSON
                return AIMessage(content='{"items": ["recovered"]}')

        configure_fake_llm(lambda tier: RetryableFake())

        node = Node(
            name="extract",
            mode="think",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-retry-feedback", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert result["extract"].items == ["recovered"]
        assert call_n["n"] == 2  # first call failed, second succeeded


class TestOutputStrategyText:
    """text strategy: LLM returns plain text, consumer's prompt_compiler handles schema."""

    def test_json_extracted_when_text_strategy_with_embedded_json(self):
        """text mode: LLM returns text containing JSON, framework extracts and parses."""
        configure_fake_llm(
            lambda tier: TextFake('Here is my analysis:\n{"items": ["from-text"]}\nDone.')
        )

        node = Node(
            name="extract",
            mode="think",
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

    def test_json_parsed_when_gather_mode_with_json_strategy(self):
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
            mode="agent",
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

    def test_output_model_passed_when_produce_node_compiled(self):
        """Produce node's prompt compiler sees the output Pydantic model."""
        compiler_calls = []

        def tracking_compiler(template, data, **kw):
            compiler_calls.append(kw)
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            lambda tier: StructuredFake(lambda m: m(items=["ok"])),
            prompt_compiler=tracking_compiler,
        )

        node = Node(name="x", mode="think", outputs=Claims, model="fast", prompt="test")
        pipeline = Construct("test", nodes=[node])
        graph = compile(pipeline)
        run(graph, input={"node_id": "test"})

        assert len(compiler_calls) == 1
        assert compiler_calls[0].get("output_model") is Claims

    def test_llm_config_passed_when_produce_node_compiled(self):
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
            name="x", mode="think", outputs=Claims, model="fast", prompt="test",
            llm_config={"output_strategy": "json_mode", "temperature": 0.5},
        )
        pipeline = Construct("test", nodes=[node])
        graph = compile(pipeline)
        run(graph, input={"node_id": "test"})

        assert len(compiler_calls) == 1
        assert compiler_calls[0].get("llm_config") == {"output_strategy": "json_mode", "temperature": 0.5}
        assert compiler_calls[0].get("output_model") is Claims

    def test_schema_injected_when_compiler_uses_json_mode(self):
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
            name="x", mode="think", outputs=Claims, model="fast", prompt="decompose",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert result["x"].items == ["schema-injected"]
        # Verify schema was injected into the prompt
        assert "json_schema" in injected_prompts[0] or "items" in injected_prompts[0]
        assert "Return a JSON object" in injected_prompts[0]


class TestGatherToolCollection:
    """invoke_with_tools collects ToolInteraction records (neograph-1bp.6)."""

    def test_tool_interaction_has_expected_fields(self):
        """ToolInteraction has the expected fields."""
        from neograph import ToolInteraction
        ti = ToolInteraction(tool_name="search", args={"q": "test"}, result="found", duration_ms=42)
        assert ti.tool_name == "search"
        assert ti.args == {"q": "test"}
        assert ti.result == "found"
        assert ti.duration_ms == 42

    def test_tool_log_written_when_gather_with_dict_outputs(self):
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
            mode="agent",
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

    def test_typed_result_preserved_when_tool_returns_pydantic_model(self):
        """Tool returns a Pydantic model. typed_result preserves it,
        result has JSON rendering (not repr). E2E: compile + run."""
        from pydantic import BaseModel
        from neograph import node, construct_from_module, compile, run, ToolInteraction
        from neograph.factory import register_tool_factory
        from tests.fakes import ReActFake, configure_fake_llm
        import types

        class SearchHit(BaseModel, frozen=True):
            node_id: str
            score: float

        # FakeTool that returns a TYPED model, not a string
        class TypedFakeTool:
            name = "typed_search"
            def invoke(self, args):
                return SearchHit(node_id="UC-001", score=0.95)

        configure_fake_llm(
            lambda tier: ReActFake(
                tool_calls=[
                    [{"name": "typed_search", "args": {"q": "test"}, "id": "tc1"}],
                    [],
                ],
                final=lambda m: Claims(items=["found"]),
            )
        )
        register_tool_factory("typed_search", lambda cfg, tc: TypedFakeTool())

        mod = types.ModuleType("test_typed_tool_mod")

        @node(
            mode="agent",
            outputs={"result": Claims, "tool_log": list[ToolInteraction]},
            model="fast", prompt="test",
            tools=[Tool("typed_search", budget=3)],
        )
        def explore() -> Claims: ...

        mod.explore = explore
        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        result = run(graph, input={})

        tool_log = result.get("explore_tool_log")
        assert tool_log is not None
        assert len(tool_log) == 1
        # typed_result preserves the original Pydantic model
        assert tool_log[0].typed_result is not None, "typed_result should not be None"
        assert isinstance(tool_log[0].typed_result, SearchHit)
        assert tool_log[0].typed_result.node_id == "UC-001"
        assert tool_log[0].typed_result.score == 0.95
        # result is BAML-rendered with field descriptions, not repr
        assert "node_id:" in tool_log[0].result, (
            f"Expected BAML notation in result, got: {tool_log[0].result}"
        )

    def test_typed_result_holds_string_when_tool_returns_string(self):
        """Tool returns plain string. typed_result holds the string itself
        (not None). Backward compat. E2E: compile + run."""
        from neograph import node, construct_from_module, compile, run, ToolInteraction
        from neograph.factory import register_tool_factory
        from tests.fakes import FakeTool, ReActFake, configure_fake_llm
        import types

        configure_fake_llm(
            lambda tier: ReActFake(
                tool_calls=[
                    [{"name": "str_tool", "args": {}, "id": "tc1"}],
                    [],
                ],
                final=lambda m: Claims(items=["done"]),
            )
        )
        register_tool_factory("str_tool", lambda cfg, tc: FakeTool("str_tool", response="plain text"))

        mod = types.ModuleType("test_str_tool_mod")

        @node(
            mode="agent",
            outputs={"result": Claims, "tool_log": list[ToolInteraction]},
            model="fast", prompt="test",
            tools=[Tool("str_tool", budget=2)],
        )
        def gather() -> Claims: ...

        mod.gather = gather
        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        result = run(graph, input={})

        tool_log = result.get("gather_tool_log")
        assert tool_log is not None
        assert len(tool_log) == 1
        assert tool_log[0].typed_result == "plain text"
        assert tool_log[0].result == "plain text"

    def test_tool_result_rendered_with_schema_when_pydantic_model(self):
        """Typed tool result in ToolMessage uses describe_type schema header +
        renderer instance, not raw model_dump_json. E2E: compile + run.
        neograph-oky0."""
        from pydantic import BaseModel, Field
        from neograph import node, construct_from_module, compile, run, ToolInteraction, XmlRenderer
        from neograph.factory import register_tool_factory
        from tests.fakes import ReActFake
        import types

        class SearchHit(BaseModel, frozen=True):
            node_id: str = Field(description="Graph node identifier")
            score: float = Field(description="Relevance score 0-1")

        class TypedTool:
            name = "schema_search"
            def invoke(self, args):
                return SearchHit(node_id="UC-042", score=0.9)

        # Capture what the LLM sees in the ToolMessage
        tool_messages_seen: list[str] = []

        class CapturingReActFake(ReActFake):
            def invoke(self, messages, **kwargs):
                # Capture ToolMessage contents
                for m in messages:
                    if hasattr(m, 'tool_call_id'):
                        tool_messages_seen.append(m.content)
                return super().invoke(messages, **kwargs)

        from neograph import configure_llm
        configure_llm(
            llm_factory=lambda tier: CapturingReActFake(
                tool_calls=[[{"name": "schema_search", "args": {}, "id": "t1"}], []],
                final=lambda m: Claims(items=["done"]),
            ),
            prompt_compiler=lambda t, d, **kw: [{"role": "user", "content": "test"}],
        )
        register_tool_factory("schema_search", lambda cfg, tc: TypedTool())

        mod = types.ModuleType("test_schema_render_mod")

        @node(
            mode="agent",
            outputs={"result": Claims, "tool_log": list[ToolInteraction]},
            model="fast", prompt="test",
            tools=[Tool("schema_search", budget=2)],
            renderer=XmlRenderer(),
        )
        def explore() -> Claims: ...

        mod.explore = explore
        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        run(graph, input={})

        # The ToolMessage content should have describe_type schema + rendered instance
        assert len(tool_messages_seen) >= 1, "No ToolMessage captured"
        content = tool_messages_seen[0]
        # Schema header from describe_type: field descriptions as comments
        assert "node_id" in content
        assert "score" in content
        # describe_type produces "// Graph node identifier" style comments
        assert "Graph node identifier" in content, (
            f"Expected describe_type schema with field descriptions, got:\n{content}"
        )
        # Rendered instance (not raw JSON dump)
        assert "UC-042" in content


# ═══════════════════════════════════════════════════════════════════════════
# TEST: @node context= verbatim state injection (neograph-p4hw)
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeContext:
    """@node context= injects verbatim state fields into prompt (neograph-p4hw)."""

    def test_context_passed_to_prompt_compiler_when_declared(self):
        """Prompt compiler receives context dict with raw state values. E2E."""
        from neograph import node, construct_from_module, compile, run
        from neograph.factory import register_scripted
        from tests.fakes import StructuredFake, configure_fake_llm
        import types

        configure_fake_llm(
            lambda tier: StructuredFake(lambda m: m(items=["done"])),
        )

        captured = {}

        def capturing_compiler(template, data, **kw):
            captured[template] = {"data": data, "kw": kw}
            return [{"role": "user", "content": "test"}]

        from neograph import configure_llm
        configure_llm(
            llm_factory=lambda tier: StructuredFake(lambda m: m(items=["done"])),
            prompt_compiler=capturing_compiler,
        )

        mod = types.ModuleType("test_context_mod")

        @node(outputs=RawText)
        def build_catalog() -> RawText:
            return RawText(text="<catalog>UC-001,UC-002,UC-003</catalog>")

        @node(
            outputs=Claims,
            mode="think",
            model="fast",
            prompt="with-context",
            context=["build_catalog"],
        )
        def analyze(build_catalog: RawText) -> Claims: ...

        mod.build_catalog = build_catalog
        mod.analyze = analyze
        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        run(graph, input={"node_id": "ctx-test"})

        assert "with-context" in captured
        kw = captured["with-context"]["kw"]
        assert "context" in kw, f"Expected 'context' in prompt compiler kwargs, got: {list(kw.keys())}"
        assert "build_catalog" in kw["context"]
        # Verbatim — the raw RawText, not BAML-rendered
        ctx_val = kw["context"]["build_catalog"]
        assert hasattr(ctx_val, "text"), f"Expected raw RawText model, got {type(ctx_val)}"
        assert ctx_val.text == "<catalog>UC-001,UC-002,UC-003</catalog>"

    def test_no_context_kwarg_when_node_has_no_context(self):
        """Prompt compiler does NOT receive context when node doesn't declare it."""
        from neograph import node, construct_from_module, compile, run
        from tests.fakes import StructuredFake
        import types

        captured = {}

        def capturing_compiler(template, data, **kw):
            captured[template] = kw
            return [{"role": "user", "content": "test"}]

        from neograph import configure_llm
        configure_llm(
            llm_factory=lambda tier: StructuredFake(lambda m: m(items=["x"])),
            prompt_compiler=capturing_compiler,
        )

        mod = types.ModuleType("test_no_ctx_mod")

        @node(outputs=Claims, mode="think", model="fast", prompt="no-context")
        def simple() -> Claims: ...

        mod.simple = simple
        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        run(graph, input={})

        assert "no-context" in captured
        assert "context" not in captured["no-context"]

    def test_context_works_with_agent_mode(self):
        """Agent node with context= alongside tools. E2E."""
        from neograph import node, construct_from_module, compile, run, ToolInteraction
        from neograph.factory import register_tool_factory
        from tests.fakes import FakeTool, ReActFake
        import types

        captured = {}

        def capturing_compiler(template, data, **kw):
            captured[template] = {"data": data, "kw": kw}
            return [{"role": "user", "content": "test"}]

        from neograph import configure_llm
        configure_llm(
            llm_factory=lambda tier: ReActFake(
                tool_calls=[[{"name": "ctx_tool", "args": {}, "id": "t1"}], []],
                final=lambda m: m(items=["found"]),
            ),
            prompt_compiler=capturing_compiler,
        )
        register_tool_factory("ctx_tool", lambda cfg, tc: FakeTool("ctx_tool", response="ok"))

        mod = types.ModuleType("test_agent_ctx_mod")

        @node(outputs=RawText)
        def catalog() -> RawText:
            return RawText(text="graph-catalog-data")

        @node(
            mode="agent",
            outputs={"result": Claims, "tool_log": list[ToolInteraction]},
            model="fast", prompt="agent-with-ctx",
            tools=[Tool("ctx_tool", budget=2)],
            context=["catalog"],
        )
        def explore(catalog: RawText) -> Claims: ...

        mod.catalog = catalog
        mod.explore = explore
        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        run(graph, input={})

        assert "agent-with-ctx" in captured
        kw = captured["agent-with-ctx"]["kw"]
        assert "context" in kw, f"Expected 'context' kwarg, got: {list(kw.keys())}"
        assert kw["context"]["catalog"].text == "graph-catalog-data"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Tool registration error in invoke_with_tools (neograph-rdu.2)
#
# When a gather/execute node references a tool name that has no registered
# factory, the error should be clear and mention the unregistered tool name.
# ═══════════════════════════════════════════════════════════════════════════

class TestToolRegistrationError:
    def test_clear_error_raised_when_tool_not_registered(self):
        """Gather node with unregistered tool raises ConfigurationError naming the tool."""
        import types as _types

        from neograph import construct_from_module, node
        from neograph.factory import register_tool_factory

        fake = ReActFake(
            tool_calls=[[], []],
            final=lambda m: m(items=["x"]),
        )
        configure_fake_llm(lambda tier: fake)

        mod = _types.ModuleType("test_unreg_tool_mod")

        @node(
            mode="agent",
            outputs=Claims,
            model="fast",
            prompt="test",
            tools=[Tool(name="nonexistent_tool", budget=3)],
        )
        def searcher() -> Claims: ...

        mod.searcher = searcher

        pipeline = construct_from_module(mod, name="test-unreg-tool")
        graph = compile(pipeline)

        with pytest.raises(ConfigurationError, match="nonexistent_tool"):
            run(graph, input={})

    def test_clear_error_raised_when_execute_tool_not_registered(self):
        """Execute node with unregistered tool also raises ConfigurationError."""
        import types as _types

        from neograph import construct_from_module, node

        fake = ReActFake(
            tool_calls=[[], []],
            final=lambda m: m(text="x"),
        )
        configure_fake_llm(lambda tier: fake)

        mod = _types.ModuleType("test_unreg_exec_mod")

        @node(
            mode="act",
            outputs=RawText,
            model="fast",
            prompt="test",
            tools=[Tool(name="missing_exec_tool", budget=1)],
        )
        def writer() -> RawText: ...

        mod.writer = writer

        pipeline = construct_from_module(mod, name="test-unreg-exec")
        graph = compile(pipeline)

        with pytest.raises(ConfigurationError, match="missing_exec_tool"):
            run(graph, input={})


# ═══════════════════════════════════════════════════════════════════════════
# TEST: skip_when on gather/execute nodes (neograph-rdu.8)
#
# skip_when is tested on produce nodes but never on gather/execute.
# The same code path in factory._make_tool_fn should work.
# ═══════════════════════════════════════════════════════════════════════════

class TestSkipWhenOnToolNodes:
    def test_node_skipped_when_skip_when_true_on_gather(self):
        """Gather node with skip_when=True skips LLM and returns skip_value."""
        import types as _types

        from neograph import construct_from_module, node
        from neograph.factory import register_tool_factory

        # Register tool factory so it doesn't fail on missing tool
        fake_tool = FakeTool("lookup", response="found")
        register_tool_factory("lookup", lambda config, tool_config: fake_tool)

        # LLM should NOT be called — if it is, the test will still pass
        # but we verify via the skip_value output
        configure_fake_llm(
            lambda tier: ReActFake(
                tool_calls=[[], []],
                final=lambda m: m(items=["should-not-appear"]),
            )
        )

        mod = _types.ModuleType("test_skip_gather_mod")

        @node(outputs=Claims)
        def seed() -> Claims:
            return Claims(items=["only-one"])

        @node(
            mode="agent",
            outputs=MergedResult,
            model="fast",
            prompt="test",
            tools=[Tool(name="lookup", budget=3)],
            skip_when=lambda inp: len(inp.items) == 1,
            skip_value=lambda inp: MergedResult(final_text=inp.items[0]),
        )
        def gatherer(seed: Claims) -> MergedResult: ...

        mod.seed = seed
        mod.gatherer = gatherer

        pipeline = construct_from_module(mod, name="test-skip-gather")
        graph = compile(pipeline)
        result = run(graph, input={})

        # Node was skipped — skip_value produced the output
        assert result["gatherer"].final_text == "only-one"
        # Tool was never called
        assert len(fake_tool.calls) == 0

    def test_node_runs_when_skip_when_false_on_gather(self):
        """Gather node runs normally when skip_when returns False."""
        import types as _types

        from neograph import construct_from_module, node
        from neograph.factory import register_tool_factory

        fake_tool = FakeTool("lookup", response="result")
        register_tool_factory("lookup", lambda config, tool_config: fake_tool)

        configure_fake_llm(
            lambda tier: ReActFake(
                tool_calls=[
                    [{"name": "lookup", "args": {"q": "test"}, "id": "tc1"}],
                    [],  # final
                ],
                final=lambda m: m(final_text="llm-produced"),
            )
        )

        mod = _types.ModuleType("test_noskip_gather_mod")

        @node(outputs=Claims)
        def seed() -> Claims:
            return Claims(items=["a", "b"])

        @node(
            mode="agent",
            outputs=MergedResult,
            model="fast",
            prompt="test",
            tools=[Tool(name="lookup", budget=3)],
            skip_when=lambda inp: len(inp.items) == 1,
            skip_value=lambda inp: MergedResult(final_text=inp.items[0]),
        )
        def gatherer(seed: Claims) -> MergedResult: ...

        mod.seed = seed
        mod.gatherer = gatherer

        pipeline = construct_from_module(mod, name="test-noskip-gather")
        graph = compile(pipeline)
        result = run(graph, input={})

        # skip_when was False → LLM ran, tool was called
        assert result["gatherer"].final_text == "llm-produced"
        assert len(fake_tool.calls) == 1


# ═══════════════════════════════════════════════════════════════════════════
# TEST: _extract_json regex edge cases (neograph-rdu.9)
#
# _extract_json in _llm.py parses JSON from LLM text responses.
# Test edge cases: plain JSON, markdown fences, multiple objects,
# nested braces, no JSON.
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractJsonEdgeCases:
    def test_plain_json_parsed_when_no_wrapping(self):
        """Plain JSON string is returned as-is."""
        from neograph._llm import _extract_json

        result = _extract_json('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_json_extracted_when_wrapped_in_markdown_fences(self):
        """JSON wrapped in ```json ... ``` is extracted."""
        from neograph._llm import _extract_json

        text = '```json\n{"key": "value"}\n```'
        result = _extract_json(text)
        assert '"key"' in result
        assert '"value"' in result
        # Verify it's valid JSON
        import json
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_first_object_extracted_when_multiple_objects_in_text(self):
        """When multiple JSON objects exist, brace-counting extracts the first balanced one."""
        from neograph._llm import _extract_json
        import json

        text = 'Here is result: {"first": 1} and also {"second": 2}'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed == {"first": 1}

    def test_nested_braces_parsed_when_json_has_nested_objects(self):
        """JSON with nested braces parses correctly."""
        from neograph._llm import _extract_json
        import json

        text = '{"outer": {"inner": "value"}}'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == "value"

    def test_text_returned_when_no_json_present(self):
        """When no JSON is present, the cleaned text is returned."""
        from neograph._llm import _extract_json

        result = _extract_json("no json here at all")
        assert result == "no json here at all"

    def test_json_extracted_when_surrounded_by_prose(self):
        """JSON embedded in prose text is extracted."""
        from neograph._llm import _extract_json
        import json

        text = 'The answer is: {"items": ["a", "b"]} as shown above.'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["items"] == ["a", "b"]


# ═══════════════════════════════════════════════════════════════════════════
# TEST: _parse_json_response lenient parsing (neograph-hqhw)
#
# _parse_json_response must recover from common LLM JSON malformations:
# control characters, trailing commas, single quotes, truncated fields.
# Currently it fails because model_validate_json uses strict json.loads.
# ═══════════════════════════════════════════════════════════════════════════

class TestParseJsonResponseLenientParsing:
    """_parse_json_response should handle malformed LLM JSON output."""

    def test_control_characters_in_strings_parsed(self):
        """Control characters inside JSON string values should not crash parsing."""
        from pydantic import BaseModel
        from neograph._llm import _parse_json_response

        class SimpleModel(BaseModel):
            content: str

        # Embedded null character — not valid in JSON strings
        text = '{"content": "hello\x00world"}'
        result = _parse_json_response(text, SimpleModel)
        assert "hello" in result.content

    def test_trailing_comma_parsed(self):
        """Trailing commas in JSON objects should not crash parsing."""
        from pydantic import BaseModel
        from neograph._llm import _parse_json_response

        class SimpleModel(BaseModel):
            name: str
            value: int

        text = '{"name": "test", "value": 42,}'
        result = _parse_json_response(text, SimpleModel)
        assert result.name == "test"
        assert result.value == 42

    def test_single_quotes_parsed(self):
        """Single-quoted JSON strings should not crash parsing."""
        from pydantic import BaseModel
        from neograph._llm import _parse_json_response

        class SimpleModel(BaseModel):
            name: str

        text = "{'name': 'test'}"
        result = _parse_json_response(text, SimpleModel)
        assert result.name == "test"

    def test_unescaped_newlines_in_strings_parsed(self):
        """Literal newlines inside JSON string values should not crash parsing."""
        from pydantic import BaseModel
        from neograph._llm import _parse_json_response

        class SimpleModel(BaseModel):
            content: str

        text = '{"content": "line1\nline2"}'
        result = _parse_json_response(text, SimpleModel)
        assert "line1" in result.content
        assert "line2" in result.content


# ═══════════════════════════════════════════════════════════════════════════
# TEST: _call_structured TypeError fallback for include_raw (neograph-rdu.11)
#
# _call_structured() passes include_raw=True by default. Some LLMs don't
# support this kwarg and raise TypeError. The code catches this and retries
# without include_raw.
# ═══════════════════════════════════════════════════════════════════════════

class TestCallStructuredFallback:
    def test_fallback_succeeds_when_include_raw_raises_type_error(self):
        """_call_structured retries without include_raw when TypeError is raised."""
        from unittest.mock import MagicMock

        from neograph._llm import _call_structured

        expected = Claims(items=["fallback-result"])

        # Mock LLM: with_structured_output(model, include_raw=True) raises TypeError
        # but with_structured_output(model) alone succeeds
        mock_llm = MagicMock()

        call_count = {"n": 0}

        def with_structured_output_side_effect(model, **kwargs):
            call_count["n"] += 1
            if kwargs.get("include_raw", False):
                raise TypeError("unexpected keyword argument 'include_raw'")
            # Return a mock structured LLM that returns our expected result
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = expected
            return mock_structured

        mock_llm.with_structured_output.side_effect = with_structured_output_side_effect

        config = {"configurable": {}}
        result, usage = _call_structured(mock_llm, [], Claims, "structured", config)

        assert result == expected
        # Called twice: first with include_raw=True (fails), then without
        assert call_count["n"] == 2

    def test_result_correct_when_include_raw_supported(self):
        """_call_structured returns parsed result when include_raw works."""
        from unittest.mock import MagicMock

        from neograph._llm import _call_structured

        expected = Claims(items=["direct-result"])

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"parsed": expected, "raw": None}
        mock_llm.with_structured_output.return_value = mock_structured

        config = {"configurable": {}}
        result, usage = _call_structured(mock_llm, [], Claims, "structured", config)

        assert result == expected
        # Called once — include_raw=True worked
        mock_llm.with_structured_output.assert_called_once_with(Claims, include_raw=True)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: RetryPolicy support (neograph-o0qw)
# ═══════════════════════════════════════════════════════════════════════════


class TestRetryPolicy:
    """compile(retry_policy=) applies retry to LLM nodes only (neograph-o0qw)."""

    def test_pipeline_compiles_and_runs_when_retry_policy_set(self):
        """compile(retry_policy=...) produces a working graph. E2E."""
        from langgraph.types import RetryPolicy
        from neograph import node, construct_from_module, compile, run
        from tests.fakes import StructuredFake, configure_fake_llm
        import types

        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["done"])))

        mod = types.ModuleType("test_retry_mod")

        @node(outputs=Claims, mode="think", model="fast", prompt="test")
        def analyze() -> Claims: ...

        mod.analyze = analyze
        pipeline = construct_from_module(mod)
        graph = compile(pipeline, retry_policy=RetryPolicy(max_attempts=3))
        result = run(graph, input={"node_id": "retry-test"})

        assert result["analyze"].items == ["done"]

    def test_retry_policy_inherited_by_sub_constructs(self):
        """Sub-constructs inherit retry_policy from parent compile(). E2E."""
        from langgraph.types import RetryPolicy
        from neograph import node, construct_from_functions, compile, run
        from neograph.factory import register_scripted
        from tests.fakes import StructuredFake, configure_fake_llm

        configure_fake_llm(lambda tier: StructuredFake(
            lambda m: m(items=["sub-done"]),
        ))

        @node(mode="think", outputs=Claims, model="fast", prompt="score")
        def score(input_text: RawText) -> Claims: ...

        sub = construct_from_functions(
            "scorer", [score], input=RawText, output=Claims,
        )

        register_scripted("retry_seed", lambda _in, _cfg: RawText(text="test"))
        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="retry_seed", outputs=RawText),
            sub,
        ])
        graph = compile(parent, retry_policy=RetryPolicy(max_attempts=2))
        result = run(graph, input={"node_id": "retry-sub"})

        assert result["scorer"].items == ["sub-done"]

    def test_scripted_nodes_not_affected_by_retry_policy(self):
        """Scripted nodes should work fine with retry_policy set (no crash). E2E."""
        from langgraph.types import RetryPolicy
        from neograph import node, construct_from_module, compile, run
        import types

        mod = types.ModuleType("test_retry_scripted_mod")

        @node(outputs=Claims)
        def scripted_node() -> Claims:
            return Claims(items=["scripted"])

        mod.scripted_node = scripted_node
        pipeline = construct_from_module(mod)
        graph = compile(pipeline, retry_policy=RetryPolicy(max_attempts=3))
        result = run(graph, input={})

        assert result["scripted_node"].items == ["scripted"]


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH VISUALIZATION — get_graph().edges completeness
# ═══════════════════════════════════════════════════════════════════════════


class TestGetGraphEdges:
    """get_graph().edges must report edges for all nodes, including those
    wired through Send()-based conditional routing (Oracle, Each)."""

    def test_oracle_edges_visible_in_get_graph(self):
        """Nodes after an Oracle fan-out must appear in get_graph().edges.

        Regression: without path_map on add_conditional_edges, LangGraph
        cannot resolve Send() targets statically, so get_graph() shows only
        the first chain terminating at __end__.
        """
        from neograph.factory import register_scripted
        from neograph.decorators import _merge_fn_registry

        register_scripted("viz_pre", lambda _in, _cfg: RawText(text="input"))
        register_scripted("viz_gen", lambda _in, _cfg: Claims(items=["x"]))

        def merge_fn(results):
            return Claims(items=["merged"])

        _merge_fn_registry["viz_merge"] = (merge_fn, None)

        register_scripted("viz_post", lambda _in, _cfg: RawText(text="done"))

        pre = Node.scripted("pre", fn="viz_pre", outputs=RawText)
        gen = Node.scripted("gen", fn="viz_gen", inputs=RawText, outputs=Claims) | Oracle(n=2, merge_fn="viz_merge")
        post = Node.scripted("post", fn="viz_post", inputs=Claims, outputs=RawText)

        pipeline = Construct("test-viz-oracle", nodes=[pre, gen, post])
        graph = compile(pipeline)

        dg = graph.get_graph()
        edge_set = {(e.source, e.target) for e in dg.edges}

        # post node must be reachable — not orphaned
        assert ("merge_gen", "post") in edge_set, (
            f"Edge merge_gen -> post missing from get_graph().edges: {edge_set}"
        )
        assert ("post", "__end__") in edge_set, (
            f"Edge post -> __end__ missing from get_graph().edges: {edge_set}"
        )

    def test_each_edges_visible_in_get_graph(self):
        """Nodes after an Each fan-out must appear in get_graph().edges."""
        from neograph.factory import register_scripted

        register_scripted("viz_make", lambda _in, _cfg: Clusters(
            groups=[ClusterGroup(label="a", claim_ids=["1"])]))
        register_scripted("viz_verify", lambda _in, _cfg: MatchResult(
            cluster_label="a", matched=["ok"]))
        register_scripted("viz_summary", lambda _in, _cfg: RawText(text="done"))

        make = Node.scripted("make", fn="viz_make", outputs=Clusters)
        verify = Node.scripted(
            "verify", fn="viz_verify", inputs=ClusterGroup, outputs=MatchResult,
        ) | Each(over="make.groups", key="label")
        summary = Node.scripted(
            "summary", fn="viz_summary",
            inputs=dict[str, MatchResult], outputs=RawText,
        )

        pipeline = Construct("test-viz-each", nodes=[make, verify, summary])
        graph = compile(pipeline)

        dg = graph.get_graph()
        edge_set = {(e.source, e.target) for e in dg.edges}

        # summary node must be reachable after the Each barrier
        assert ("assemble_verify", "summary") in edge_set, (
            f"Edge assemble_verify -> summary missing: {edge_set}"
        )
        assert ("summary", "__end__") in edge_set, (
            f"Edge summary -> __end__ missing: {edge_set}"
        )


# =============================================================================
# BUG REGRESSION: neograph-irv3
# R1 emits tool-call XML in content after budget exhaustion
# =============================================================================


class TestR1XmlAfterBudgetExhaustion:
    """When tool budgets are exhausted and the bare LLM responds with XML
    tool-call markup instead of JSON, _parse_json_response must handle it
    instead of crashing with a ValidationError."""

    def test_xml_tool_call_in_content_parsed_when_json_mode_gather(self):
        """After budget exhaustion, R1-style XML in content should not crash.

        The LLM emits '<｜DSML｜function_calls>...' in message.content instead
        of valid JSON. The framework should detect this and either strip it
        or raise a clear ExecutionError — not a raw ValidationError.
        """
        from langchain_core.messages import AIMessage

        from neograph._llm import configure_llm
        from neograph.factory import register_tool_factory

        lookup_tool = FakeTool("lookup", response="found")
        register_tool_factory("lookup", lambda config, tool_config: lookup_tool)

        # Fake LLM: first call uses tool (exhausts budget=1),
        # second call (bare, no tools) emits XML instead of JSON.
        call_n = {"n": 0}
        XML_CONTENT = (
            '<\uff5cDSML\uff5cfunction_calls>'
            '<\uff5cDSML\uff5cinvoke name="read_artifact">'
            '<\uff5cDSML\uff5cparameter name="path">test.py'
            '</\uff5cDSML\uff5cparameter>'
            '</\uff5cDSML\uff5cinvoke>'
            '</\uff5cDSML\uff5cfunction_calls>'
        )

        class FakeR1:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kwargs):
                call_n["n"] += 1
                if call_n["n"] == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "lookup", "args": {}, "id": "c1"}]
                    return msg
                # After budget exhaustion: XML instead of JSON
                return AIMessage(content=XML_CONTENT)

        configure_llm(
            llm_factory=lambda tier: FakeR1(),
            prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": "go"}],
        )

        node = Node(
            name="research",
            mode="agent",
            outputs=Claims,
            model="fast",
            prompt="test",
            tools=[Tool(name="lookup", budget=1)],
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-r1-xml", nodes=[node])
        graph = compile(pipeline)

        # Should raise ExecutionError (clear message), not ValidationError (cryptic)
        with pytest.raises(ExecutionError, match="(?i)structured output|json|xml"):
            run(graph, input={"node_id": "test"})


# =============================================================================
# BUG REGRESSION: neograph-g1ip
# Compiler should register output model types with LangGraph msgpack allowlist
# =============================================================================


class TestMsgpackTypeRegistration:
    """When compile() receives a checkpointer, it should register all node
    output types with the checkpointer's serializer so LangGraph doesn't
    emit 'Deserializing unregistered type' warnings on resume."""

    def test_output_types_registered_when_compiled_with_checkpointer(self):
        """compile() with MemorySaver should convert the serde from
        allow-all (True) to an explicit allowlist containing the node
        output Pydantic models."""
        from langgraph.checkpoint.memory import MemorySaver
        from neograph.factory import register_scripted

        register_scripted("dummy_a", lambda input_data, config: Claims(items=["x"]))
        register_scripted("dummy_b", lambda input_data, config: MatchResult(
            cluster_label="test", coverage_pct=90, gaps=[],
        ))

        pipeline = Construct("test-msgpack", nodes=[
            Node.scripted("a", fn="dummy_a", outputs=Claims),
            Node.scripted("b", fn="dummy_b", inputs=Claims, outputs=MatchResult),
        ])

        checkpointer = MemorySaver()

        # Before compile: serde allows all (True = warn mode)
        assert checkpointer.serde._allowed_msgpack_modules is True

        graph = compile(pipeline, checkpointer=checkpointer)

        # After compile: serde should have an explicit allowlist (set/frozenset),
        # not True. The allowlist must include our output model types.
        allowlist = checkpointer.serde._allowed_msgpack_modules
        assert allowlist is not True, (
            "compile() should convert the serde from allow-all (True) to an "
            "explicit allowlist containing node output types"
        )
        # The allowlist should contain tuples of (module, classname)
        type_names = {name for (_mod, name) in allowlist}
        assert "Claims" in type_names, f"Claims not in allowlist: {type_names}"
        assert "MatchResult" in type_names, f"MatchResult not in allowlist: {type_names}"
