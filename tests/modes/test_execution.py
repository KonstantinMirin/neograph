"""Pipeline mode tests — execute mode, error paths, LLM config, run_isolated, config injection"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import (
    CompileError,
    ConfigurationError,
    Construct,
    Each,
    ExecutionError,
    Node,
    Operator,
    Oracle,
    Tool,
    compile,
    run,
)
from tests.fakes import FakeTool, ReActFake, StructuredFake, configure_fake_llm
from tests.schemas import (
    Claims,
    ClusterGroup,
    Clusters,
    MatchResult,
    RawText,
)

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

    def test_compile_raises_when_llm_not_configured(self):
        """Produce node without configure_llm() raises CompileError at compile()."""
        node = Node(name="fail", mode="think", outputs=Claims, model="fast", prompt="x")
        pipeline = Construct("test-no-llm", nodes=[node])

        with pytest.raises(CompileError, match="configure_llm"):
            compile(pipeline)

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

        node = Node.scripted("gen", fn="gen", outputs=Claims) | Oracle(n=2, merge_fn="nonexistent_merge")

        pipeline = Construct("test-bad-merge", nodes=[node])

        with pytest.raises(ConfigurationError, match="not registered"):
            compile(pipeline)

    def test_compile_raises_when_operator_condition_not_registered(self):
        """Operator with unregistered condition raises ConfigurationError at compile."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_scripted

        register_scripted("something", lambda input_data, config: Claims(items=[]))

        node = Node.scripted("something", fn="something", outputs=Claims) | Operator(when="nonexistent_condition")

        pipeline = Construct("test-bad-condition", nodes=[node])

        with pytest.raises(ConfigurationError, match="not registered"):
            compile(pipeline, checkpointer=MemorySaver())

    def test_compile_raises_when_tool_factory_not_registered(self):
        """Agent node with unregistered tool factory raises CompileError at compile()."""
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

        with pytest.raises(CompileError, match="ghost_tool"):
            compile(pipeline)

    def test_run_without_input_or_checkpoint_raises(self):
        """run() with no input, no resume, no checkpoint raises LangGraph error.

        BUG neograph-zs0r: previously raised ValueError from neograph. Now
        passes through to LangGraph which raises EmptyInputError when there's
        no checkpoint to resume from.
        """
        from langgraph.errors import EmptyInputError

        from neograph.factory import register_scripted

        register_scripted("noop_resume", lambda input_data, config: Claims(items=[]))
        node = Node.scripted("noop", fn="noop_resume", outputs=Claims)
        pipeline = Construct("test-no-args", nodes=[node])
        graph = compile(pipeline)

        # No checkpointer, no input → LangGraph raises EmptyInputError
        with pytest.raises(EmptyInputError):
            run(graph)

    def test_compile_raises_when_node_missing_output_type(self):
        """Node with no output type raises CompileError at compile."""
        configure_fake_llm(lambda tier: StructuredFake(lambda m: m()))
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
            factory_calls.append(
                {
                    "tier": tier,
                    "node_name": node_name,
                    "llm_config": llm_config,
                }
            )
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

        result = run(
            graph,
            input={"node_id": "BR-RW-042", "project_root": "/my/project"},
            config={"configurable": {"custom_field": "extra-data"}},
        )

        # All input fields + custom config accessible
        assert config_seen["node_id"] == "BR-RW-042"
        assert config_seen["project_root"] == "/my/project"
        assert config_seen["custom_field"] == "extra-data"

    def test_compiler_sees_metadata_when_node_name_and_config_provided(self):
        """Prompt compiler gets node_name and full config with pipeline metadata."""
        compiler_calls = []

        def tracking_compiler(template, data, *, node_name=None, config=None):
            compiler_calls.append(
                {
                    "template": template,
                    "node_name": node_name,
                    "node_id": config.get("configurable", {}).get("node_id") if config else None,
                    "project_root": config.get("configurable", {}).get("project_root") if config else None,
                }
            )
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


class TestCheckpointResume:
    """run() crash recovery — resume from checkpoint without new input.

    BUG neograph-zs0r: run(graph, input={...}) with existing checkpoint
    restarts from beginning instead of resuming.
    """

    def test_resume_from_checkpoint_skips_completed_nodes(self):
        """run(graph, config=...) with no input resumes from checkpoint."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_scripted

        call_log = []
        register_scripted("ckpt_a", lambda _i, _c: (call_log.append("a"), RawText(text="a"))[1])
        register_scripted("ckpt_b", lambda _i, _c: (call_log.append("b"), Claims(items=["b"]))[1])

        pipeline = Construct("ckpt-test", nodes=[
            Node.scripted("a", fn="ckpt_a", outputs=RawText),
            Node.scripted("b", fn="ckpt_b", inputs=RawText, outputs=Claims),
        ])

        checkpointer = MemorySaver()
        graph = compile(pipeline, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-resume"}}

        # First run — both nodes execute
        result = run(graph, input={"node_id": "test"}, config=config)
        assert call_log == ["a", "b"]
        assert result["b"].items == ["b"]

        # Second run — resume with no input.
        # Since graph completed successfully, invoke(None) returns final state.
        call_log.clear()
        result2 = run(graph, config=config)
        # Completed graph: no nodes re-execute (already at terminal state)
        assert call_log == []

    def test_resume_after_crash_skips_completed_nodes(self):
        """Crash mid-pipeline, resume skips completed nodes.

        This is the actual crash-recovery scenario: node A succeeds,
        node B crashes. On resume, node A should NOT re-execute.
        """
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_scripted

        call_log = []
        crash_on_first = [True]

        def node_a(input_data, config):
            call_log.append("a")
            return RawText(text="from-a")

        def node_b(input_data, config):
            call_log.append("b")
            if crash_on_first[0]:
                crash_on_first[0] = False
                raise RuntimeError("simulated crash")
            return Claims(items=["from-b"])

        register_scripted("crash_a", node_a)
        register_scripted("crash_b", node_b)

        pipeline = Construct("crash-test", nodes=[
            Node.scripted("a", fn="crash_a", outputs=RawText),
            Node.scripted("b", fn="crash_b", inputs=RawText, outputs=Claims),
        ])

        checkpointer = MemorySaver()
        graph = compile(pipeline, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-crash-resume"}}

        # First run — A succeeds, B crashes
        import pytest as _pytest
        with _pytest.raises(RuntimeError, match="simulated crash"):
            run(graph, input={"node_id": "test"}, config=config)
        assert call_log == ["a", "b"]

        # Resume — pass input (for DI), but checkpoint should be detected.
        # A should NOT re-execute, B should retry.
        call_log.clear()
        result = run(graph, input={"node_id": "test"}, config=config)
        assert "a" not in call_log, "Node A re-executed on resume — checkpoint not working"
        assert "b" in call_log
        assert result["b"].items == ["from-b"]

    def test_resume_without_di_values_raises_preflight_error(self):
        """Crash recovery without DI values gives clear preflight error.

        BUG neograph-mxn3: _neo_input stash lost across process restarts.
        Fix: preflight check runs on crash-recovery path too.
        """
        from typing import Annotated

        from langgraph.checkpoint.memory import MemorySaver

        from neograph import FromInput, node
        from neograph.factory import register_scripted

        @node(outputs=Claims)
        def needs_di(topic: Annotated[str, FromInput]) -> Claims:
            return Claims(items=[topic])

        from neograph import construct_from_functions
        pipeline = construct_from_functions("di-resume", [needs_di])

        checkpointer = MemorySaver()
        graph = compile(pipeline, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-di-resume"}}

        # First run with DI values — succeeds
        result = run(graph, input={"node_id": "test", "topic": "hello"}, config=config)

        # Simulate process restart: new config with thread_id but NO DI values
        fresh_config = {"configurable": {"thread_id": "test-di-resume-2"}}
        with pytest.raises(ExecutionError, match="Required DI"):
            run(graph, config=fresh_config)


    def test_operator_resume_without_neo_input_stash(self):
        """Operator resume when _neo_input is absent from config (path 1b).

        Simulates a caller that provides resume= but didn't stash _neo_input
        (e.g., config reconstructed from external storage).
        """
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        call_log = []
        register_scripted("op_a", lambda _i, _c: (call_log.append("a"), RawText(text="a"))[1])
        register_condition("always_interrupt", lambda result: {"needs_review": True})

        pipeline = Construct("op-resume-test", nodes=[
            Node.scripted("a", fn="op_a", outputs=RawText)
            | Operator(when="always_interrupt"),
        ])

        checkpointer = MemorySaver()
        graph = compile(pipeline, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-op-resume"}}

        # First run hits the interrupt
        result = run(graph, input={"node_id": "test"}, config=config)
        assert call_log == ["a"]
        assert "__interrupt" in str(result) or result.get("a") is not None

        # Resume WITHOUT _neo_input in config — should not crash
        clean_config = {"configurable": {"thread_id": "test-op-resume"}}
        call_log.clear()
        result2 = run(graph, resume={"approved": True}, config=clean_config)
        # Operator resume completes the graph
        assert result2.get("a") is not None

    def test_checkpoint_check_returns_false_without_checkpointer(self):
        """_has_existing_checkpoint returns False when no checkpointer (path 2c)."""
        from neograph.runner import _has_existing_checkpoint
        from neograph.factory import register_scripted

        register_scripted("no_ckpt", lambda _i, _c: RawText(text="x"))
        pipeline = Construct("no-ckpt", nodes=[
            Node.scripted("a", fn="no_ckpt", outputs=RawText),
        ])
        graph = compile(pipeline)  # no checkpointer
        assert not _has_existing_checkpoint(graph, {"configurable": {"thread_id": "x"}})

    def test_checkpoint_check_handles_broken_checkpointer(self):
        """_has_existing_checkpoint returns False on exception (exception path)."""
        from neograph.runner import _has_existing_checkpoint

        class BrokenCheckpointer:
            def get_tuple(self, config):
                raise TypeError("broken")

        class FakeGraph:
            checkpointer = BrokenCheckpointer()

        assert not _has_existing_checkpoint(FakeGraph(), {"configurable": {"thread_id": "x"}})

    def test_run_with_config_none_and_no_input(self):
        """run(graph) with config=None skips preflight, passes to LangGraph (path 3c)."""
        from langgraph.errors import EmptyInputError

        from neograph.factory import register_scripted

        register_scripted("p3c", lambda _i, _c: RawText(text="x"))
        pipeline = Construct("p3c-test", nodes=[
            Node.scripted("a", fn="p3c", outputs=RawText),
        ])
        graph = compile(pipeline)

        # No checkpointer + no input + config=None → LangGraph EmptyInputError
        with pytest.raises(EmptyInputError):
            run(graph, config=None)

    def test_new_execution_with_input_when_no_checkpoint(self):
        """run(graph, input={...}) without checkpoint is a normal new run (path 2b)."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_scripted

        call_log = []
        register_scripted("new_a", lambda _i, _c: (call_log.append("a"), RawText(text="a"))[1])

        pipeline = Construct("new-exec", nodes=[
            Node.scripted("a", fn="new_a", outputs=RawText),
        ])
        checkpointer = MemorySaver()
        graph = compile(pipeline, checkpointer=checkpointer)

        # Fresh thread_id — no checkpoint exists
        config = {"configurable": {"thread_id": "brand-new-thread"}}
        result = run(graph, input={"node_id": "test"}, config=config)
        assert call_log == ["a"]
        assert result["a"].text == "a"


class TestRunIsolated:
    """Node.run_isolated() — direct invocation for unit testing."""

    def test_result_returned_when_scripted_node_run_isolated(self):
        """Scripted nodes can be tested without compile()/run()."""
        from neograph import register_scripted

        register_scripted(
            "upper", lambda input_data, config: RawText(text=input_data.text.upper() if input_data else "NONE")
        )

        upper_node = Node.scripted("upper", fn="upper", inputs=RawText, outputs=RawText)

        # Direct invocation — no pipeline, no compile, no run
        result = upper_node.run_isolated(input=RawText(text="hello"))

        assert isinstance(result, RawText)
        assert result.text == "HELLO"

    def test_result_returned_when_produce_node_run_isolated(self):
        """Produce nodes can be tested with a fake LLM."""
        configure_fake_llm(lambda tier: StructuredFake(lambda model: model(items=["isolated-result"])))

        decompose = Node("decompose", mode="think", outputs=Claims, model="fast", prompt="test")

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

        result = node.run_isolated(config={"configurable": {"node_id": "TEST-001", "env": "staging"}})

        assert result.items == ["ok"]
        assert seen_config["node_id"] == "TEST-001"
        assert seen_config["env"] == "staging"

    def test_dict_input_seeded_to_state_when_run_isolated(self):
        """run_isolated with dict input updates state directly (line 163)."""
        from neograph import register_scripted

        received_data = {}

        def fn(input_data, config):
            if isinstance(input_data, dict):
                received_data.update(input_data)
            return Claims(items=["from_dict"])

        register_scripted("dict_test", fn)
        node = Node.scripted("dict-test", fn="dict_test", outputs=Claims)

        result = node.run_isolated(input={"raw_text": "hello", "count": 5})
        assert isinstance(result, Claims)
        assert result.items == ["from_dict"]

    def test_missing_configurable_key_in_config_when_run_isolated(self):
        """run_isolated adds configurable key if missing from config (line 170)."""
        from neograph import register_scripted

        seen_config = {}

        def fn(input_data, config):
            seen_config.update(config)
            return Claims(items=["ok"])

        register_scripted("no_cfg", fn)
        node = Node.scripted("no-cfg", fn="no_cfg", outputs=Claims)

        # Pass config without 'configurable' key
        result = node.run_isolated(config={"some_other_key": "val"})
        assert isinstance(result, Claims)
        assert "configurable" in seen_config


class TestStateGet:
    """_state_get() — dual-form state access (dict vs Pydantic model).

    BUG neograph-l5g0: dict path was untested. These tests cover both
    forms directly to ensure identical behavior.
    """

    def test_returns_value_from_dict(self):
        """Dict-form state: key present → return value."""
        from neograph.factory import _state_get
        assert _state_get({"foo": 42}, "foo") == 42

    def test_returns_none_for_missing_dict_key(self):
        """Dict-form state: key absent → None (not KeyError)."""
        from neograph.factory import _state_get
        assert _state_get({"foo": 42}, "bar") is None

    def test_returns_value_from_pydantic_model(self):
        """Pydantic-form state: field present → return value."""
        from neograph.factory import _state_get
        assert _state_get(RawText(text="hello"), "text") == "hello"

    def test_returns_none_for_missing_pydantic_field(self):
        """Pydantic-form state: field absent → None (not AttributeError)."""
        from neograph.factory import _state_get
        assert _state_get(RawText(text="hello"), "nonexistent") is None


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

        pipeline = Construct(
            "test-resources",
            nodes=[
                Node.scripted("step", fn="resourced", outputs=Claims),
            ],
        )
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

        parent = Construct(
            "parent",
            nodes=[
                Node.scripted("seed", fn="parent_seed", outputs=Claims),
                sub,
            ],
        )
        graph = compile(parent)
        run(graph, input={"node_id": "BR-042"}, config={"configurable": {"custom": "my-value"}})

        # Sub-construct node function received both pipeline input and custom config
        assert seen_in_sub["node_id"] == "BR-042"
        assert seen_in_sub["custom"] == "my-value"

    def test_all_generators_see_config_when_oracle_with_metadata(self):
        """Each Oracle generator sees config with node_id and custom fields."""
        from neograph.factory import register_scripted

        gen_configs = []

        def gen_fn(input_data, config):
            gen_configs.append(
                {
                    "node_id": config.get("configurable", {}).get("node_id"),
                    "project_root": config.get("configurable", {}).get("project_root"),
                    "gen_id": config.get("configurable", {}).get("_generator_id"),
                }
            )
            return Claims(items=["v"])

        def merge_fn(variants, config):
            return Claims(items=[f"merged-{len(variants)}"])

        register_scripted("cfg_gen", gen_fn)
        register_scripted("cfg_merge", merge_fn)

        node = Node.scripted("gen", fn="cfg_gen", outputs=Claims) | Oracle(n=3, merge_fn="cfg_merge")

        pipeline = Construct("test-oracle-config", nodes=[node])
        graph = compile(pipeline)
        run(graph, input={"node_id": "BR-099", "project_root": "/proj"}, config={})

        # All 3 generators saw the pipeline metadata
        assert len(gen_configs) == 3
        for gc in gen_configs:
            assert gc["node_id"] == "BR-099"
            assert gc["project_root"] == "/proj"
            assert isinstance(gc["gen_id"], str) and gc["gen_id"], "gen_id should be a non-empty string"

    def test_each_invocation_sees_config_when_metadata_present(self):
        """Each fan-out node sees config with node_id."""
        from neograph.factory import register_scripted

        each_configs = []

        register_scripted(
            "make_groups",
            lambda input_data, config: Clusters(
                groups=[ClusterGroup(label="a", claim_ids=["1"]), ClusterGroup(label="b", claim_ids=["2"])]
            ),
        )

        def verify_with_config(input_data, config):
            each_configs.append(
                {
                    "node_id": config.get("configurable", {}).get("node_id"),
                    "label": input_data.label,
                }
            )
            return MatchResult(cluster_label=input_data.label, matched=["ok"])

        register_scripted("verify_cfg", verify_with_config)

        make = Node.scripted("make", fn="make_groups", outputs=Clusters)
        verify = Node.scripted("verify", fn="verify_cfg", inputs=ClusterGroup, outputs=MatchResult) | Each(
            over="make.groups", key="label"
        )

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
            nodes_seen.append(
                {
                    "node_id": config.get("configurable", {}).get("node_id"),
                    "env": config.get("configurable", {}).get("env"),
                }
            )
            return Claims(items=["done"])

        register_scripted("track_a", tracking_fn)
        register_scripted("track_b", tracking_fn)
        register_scripted("track_c", tracking_fn)

        pipeline = Construct(
            "test-all-config",
            nodes=[
                Node.scripted("a", fn="track_a", outputs=Claims),
                Node.scripted("b", fn="track_b", outputs=Claims),
                Node.scripted("c", fn="track_c", outputs=Claims),
            ],
        )
        graph = compile(pipeline)
        run(graph, input={"node_id": "REQ-001"}, config={"configurable": {"env": "staging"}})

        assert len(nodes_seen) == 3
        for seen in nodes_seen:
            assert seen["node_id"] == "REQ-001"
            assert seen["env"] == "staging"


class TestCheckpointSchemaValidation:
    """Checkpoint resume must detect schema changes and refuse silent coercion.

    TASK neograph-05lv: Pipeline "succeeded" with 0 LLM calls because Pydantic
    silently coerced old-format state. The resume path must fingerprint the
    state schema and raise on mismatch.
    """

    def test_schema_change_raises_on_resume(self):
        """After changing a node's output model fields, resume must raise."""
        from langgraph.checkpoint.memory import MemorySaver
        from neograph.errors import CheckpointSchemaError
        from neograph.factory import register_scripted

        class V1Output(BaseModel):
            content: str

        class V2Output(BaseModel):
            content: str
            score: float = 0.0  # new field

        register_scripted("sv1_a", lambda _i, _c: V1Output(content="hello"))

        # Run 1: compile + run with V1 schema
        checkpointer = MemorySaver()
        config = {"configurable": {"thread_id": "schema-test-1"}}

        pipe_v1 = Construct("sv-pipe", nodes=[
            Node.scripted("a", fn="sv1_a", outputs=V1Output),
        ])
        graph_v1 = compile(pipe_v1, checkpointer=checkpointer)
        run(graph_v1, input={"node_id": "test"}, config=config)

        # Run 2: compile with V2 schema (added field), same thread_id
        register_scripted("sv2_a", lambda _i, _c: V2Output(content="hello", score=0.9))
        pipe_v2 = Construct("sv-pipe", nodes=[
            Node.scripted("a", fn="sv2_a", outputs=V2Output),
        ])
        graph_v2 = compile(pipe_v2, checkpointer=checkpointer)

        with pytest.raises(CheckpointSchemaError, match="schema.*changed|fingerprint"):
            run(graph_v2, input={"node_id": "test"}, config=config)

    def test_identical_schema_resumes_normally(self):
        """Same schema on both runs — resume proceeds without error."""
        from langgraph.checkpoint.memory import MemorySaver
        from neograph.factory import register_scripted

        class StableOutput(BaseModel):
            content: str

        call_count = [0]
        def counting_fn(_i, _c):
            call_count[0] += 1
            return StableOutput(content="hello")

        register_scripted("stable_a", counting_fn)

        checkpointer = MemorySaver()
        config = {"configurable": {"thread_id": "schema-test-2"}}

        pipe = Construct("stable-pipe", nodes=[
            Node.scripted("a", fn="stable_a", outputs=StableOutput),
        ])

        # Run 1
        graph = compile(pipe, checkpointer=checkpointer)
        run(graph, input={"node_id": "test"}, config=config)
        assert call_count[0] == 1

        # Run 2 — same schema, should resume (node already complete)
        graph2 = compile(pipe, checkpointer=checkpointer)
        run(graph2, input={"node_id": "test"}, config=config)
        # Node should NOT re-execute (already checkpointed)
        assert call_count[0] == 1

    def test_class_rename_detected(self):
        """Renaming the output class (same fields) still triggers detection."""
        from langgraph.checkpoint.memory import MemorySaver
        from neograph.errors import CheckpointSchemaError
        from neograph.factory import register_scripted

        class OriginalName(BaseModel):
            content: str

        class RenamedClass(BaseModel):
            content: str  # same fields, different class name

        register_scripted("rn_a1", lambda _i, _c: OriginalName(content="hello"))
        register_scripted("rn_a2", lambda _i, _c: RenamedClass(content="hello"))

        checkpointer = MemorySaver()
        config = {"configurable": {"thread_id": "schema-test-3"}}

        pipe1 = Construct("rn-pipe", nodes=[
            Node.scripted("a", fn="rn_a1", outputs=OriginalName),
        ])
        graph1 = compile(pipe1, checkpointer=checkpointer)
        run(graph1, input={"node_id": "test"}, config=config)

        pipe2 = Construct("rn-pipe", nodes=[
            Node.scripted("a", fn="rn_a2", outputs=RenamedClass),
        ])
        graph2 = compile(pipe2, checkpointer=checkpointer)

        with pytest.raises(CheckpointSchemaError):
            run(graph2, input={"node_id": "test"}, config=config)


