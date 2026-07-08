"""Single source of truth for neograph's user-observable log event schema.

Operators monitoring on these event names + payload fields can rely on this
contract. Changes here are deliberate breaking changes to that contract.

When a behavioral test elsewhere needs to assert "the recovery succeeded",
that test should NOT pin the event name — it should assert on the user-visible
result (parsed answer, returned value, etc.). The structlog event names are
covered here, in one place, so a rename is a single deliberate edit.

Events covered:
- ``react_max_iterations_exceeded`` — ReAct loop hit the max_iterations cap
- ``react_token_budget_exceeded`` — ReAct loop hit the token_budget cap
- ``trailing_tool_call_markup`` — DSML/XML markup detected in final response
- ``react_guard_forced_break`` — safety break when guard set but rogue tool_calls
- ``auto_resume_schema_change`` — checkpoint schema fingerprint mismatch
- ``node_start`` / ``node_complete`` — agent/act node lifecycle (PAT-02 shape:
  ``input_type``/``output_type`` via ``type_display_name``; ``duration_s``)
"""

from __future__ import annotations

from pydantic import BaseModel
from structlog.testing import capture_logs

from neograph import (
    Construct,
    Node,
    Tool,
    compile,
    run,
)
from tests.fakes import (
    FakeTool,
    GuardFake,
    build_fake_runtime,
    build_fake_tool_lookup,
    build_test_compile_kwargs,
    configure_fake_llm,
)
from tests.schemas import Claims


class TestReActMaxIterationsExceededEvent:
    """Contract: ReAct loop emits ``react_max_iterations_exceeded`` at warning
    level when the max_iterations guard fires, with payload fields describing
    the loop state (loops, tool_calls, max_iterations)."""

    def test_event_name_emitted_when_max_iterations_hit(self):
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = GuardFake()
        _llm_kw = configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=0)]
        tracker = ToolBudgetTracker(tools)

        with capture_logs() as logs:
            invoke_with_tools(
                runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
                model_tier="fast",
                prompt_template="test",
                input_data="test",
                output_model=Claims,
                tools=tools,
                budget_tracker=tracker,
                config={"configurable": {}},
                llm_config={"max_iterations": 3},
                tool_factory_lookup=build_fake_tool_lookup(),
            )

        events = [e for e in logs if e.get("event") == "react_max_iterations_exceeded"]
        assert events, (
            f"contract: ``react_max_iterations_exceeded`` must fire when the "
            f"max_iterations guard trips. Events observed: "
            f"{[e.get('event') for e in logs]}"
        )

    def test_event_payload_includes_max_iterations_and_loop_state(self):
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = GuardFake()
        _llm_kw = configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=0)]
        tracker = ToolBudgetTracker(tools)

        with capture_logs() as logs:
            invoke_with_tools(
                runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
                model_tier="fast",
                prompt_template="test",
                input_data="test",
                output_model=Claims,
                tools=tools,
                budget_tracker=tracker,
                config={"configurable": {}},
                llm_config={"max_iterations": 3},
                tool_factory_lookup=build_fake_tool_lookup(),
            )

        events = [e for e in logs if e.get("event") == "react_max_iterations_exceeded"]
        assert events, "contract event not emitted"
        evt = events[0]
        # Payload contract: operators rely on these fields for debugging
        # runaway-loop scenarios.
        assert "max_iterations" in evt
        assert "loops" in evt
        assert "tool_calls" in evt
        # Severity contract: this is a warning, not info/debug.
        assert evt.get("log_level") == "warning"


class TestReActTokenBudgetExceededEvent:
    """Contract: ReAct loop emits ``react_token_budget_exceeded`` at warning
    level when the token_budget guard fires, with payload describing token
    consumption (cumulative_input_tokens, token_budget)."""

    def test_event_name_emitted_when_token_budget_hit(self):
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = GuardFake(input_tokens_per_call=1000)
        _llm_kw = configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=0)]
        tracker = ToolBudgetTracker(tools)

        with capture_logs() as logs:
            invoke_with_tools(
                runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
                model_tier="fast",
                prompt_template="test",
                input_data="test",
                output_model=Claims,
                tools=tools,
                budget_tracker=tracker,
                config={"configurable": {}},
                llm_config={"max_iterations": 50, "token_budget": 2500},
                tool_factory_lookup=build_fake_tool_lookup(),
            )

        events = [e for e in logs if e.get("event") == "react_token_budget_exceeded"]
        assert events, (
            f"contract: ``react_token_budget_exceeded`` must fire when the "
            f"token_budget guard trips. Events observed: "
            f"{[e.get('event') for e in logs]}"
        )

    def test_event_payload_includes_token_budget_state(self):
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = GuardFake(input_tokens_per_call=1000)
        _llm_kw = configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=0)]
        tracker = ToolBudgetTracker(tools)

        with capture_logs() as logs:
            invoke_with_tools(
                runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
                model_tier="fast",
                prompt_template="test",
                input_data="test",
                output_model=Claims,
                tools=tools,
                budget_tracker=tracker,
                config={"configurable": {}},
                llm_config={"max_iterations": 50, "token_budget": 2500},
                tool_factory_lookup=build_fake_tool_lookup(),
            )

        events = [e for e in logs if e.get("event") == "react_token_budget_exceeded"]
        assert events, "contract event not emitted"
        evt = events[0]
        # Payload contract: operators rely on these fields for debugging
        # token-budget violations.
        assert "token_budget" in evt
        assert "cumulative_input_tokens" in evt
        assert "loops" in evt
        # Severity contract.
        assert evt.get("log_level") == "warning"


class TestTrailingToolCallMarkupEvent:
    """Contract: when the final response contains DSML/XML tool-call markup,
    the framework emits ``trailing_tool_call_markup`` at warning level before
    attempting a targeted retry."""

    def test_event_name_emitted_when_dsml_markup_in_final_response(self):
        from langchain_core.messages import AIMessage

        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Answer(BaseModel):
            answer: str

        call_count = [0]
        dsml_payload = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="search">\n'
            '<｜DSML｜parameter name="q">more</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

        class DSMLFake:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                call_count[0] += 1
                if call_count[0] == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search", "args": {"q": "x"}, "id": "c1"}]
                    return msg
                if call_count[0] == 2:
                    return AIMessage(content=dsml_payload)
                return AIMessage(content='{"answer": "recovered"}')

            def with_structured_output(self, model, **kw):
                return self

        def search_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            return StructuredTool.from_function(lambda q: f"found {q}", name="search", description="search")

        register_tool_factory("search", search_factory)

        tools = [Tool(name="search", description="search")]
        budget = ToolBudgetTracker(tools)

        with capture_logs() as logs:
            invoke_with_tools(
                model_tier="fast",
                prompt_template="test",
                input_data={},
                output_model=Answer,
                config={"configurable": {}},
                tools=tools,
                budget_tracker=budget,
                llm_config={"output_strategy": "json_mode", "max_iterations": 1},
                runtime=build_fake_runtime(lambda tier: DSMLFake()),
                tool_factory_lookup=build_fake_tool_lookup(),
            )

        events = [e for e in logs if e.get("event") == "trailing_tool_call_markup"]
        assert events, (
            f"contract: ``trailing_tool_call_markup`` must fire when DSML/XML "
            f"markup is detected in the final response. Events observed: "
            f"{[e.get('event') for e in logs]}"
        )

    def test_event_severity_is_warning(self):
        from langchain_core.messages import AIMessage

        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Answer(BaseModel):
            answer: str

        call_count = [0]
        dsml_payload = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="search">\n'
            '<｜DSML｜parameter name="q">more</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

        class DSMLFake:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                call_count[0] += 1
                if call_count[0] == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search", "args": {"q": "x"}, "id": "c1"}]
                    return msg
                if call_count[0] == 2:
                    return AIMessage(content=dsml_payload)
                return AIMessage(content='{"answer": "recovered"}')

            def with_structured_output(self, model, **kw):
                return self

        def search_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            return StructuredTool.from_function(lambda q: f"found {q}", name="search", description="search")

        register_tool_factory("search", search_factory)

        tools = [Tool(name="search", description="search")]
        budget = ToolBudgetTracker(tools)

        with capture_logs() as logs:
            invoke_with_tools(
                model_tier="fast",
                prompt_template="test",
                input_data={},
                output_model=Answer,
                config={"configurable": {}},
                tools=tools,
                budget_tracker=budget,
                llm_config={"output_strategy": "json_mode", "max_iterations": 1},
                runtime=build_fake_runtime(lambda tier: DSMLFake()),
                tool_factory_lookup=build_fake_tool_lookup(),
            )

        events = [e for e in logs if e.get("event") == "trailing_tool_call_markup"]
        assert events, "contract event not emitted"
        # Severity contract: warning, not info/debug.
        assert events[0].get("log_level") == "warning"


class TestReActGuardForcedBreakEvent:
    """Contract: when the guard has fired and the LLM continues to emit tool
    calls (rogue dispatch), the framework emits ``react_guard_forced_break`` at
    warning level and forces the loop to exit. Payload includes loops and
    tool_calls counts."""

    def test_event_name_emitted_on_safety_break(self):
        from langchain_core.messages import AIMessage
        from langchain_core.tools import StructuredTool

        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Result(BaseModel):
            answer: str

        class RogueFake:
            def __init__(self):
                self.calls = 0

            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                self.calls += 1
                if self.calls == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search", "args": {"q": "x"}, "id": "c1"}]
                    return msg
                msg = AIMessage(content='{"answer": "ok"}')
                msg.tool_calls = [{"name": "search", "args": {"q": "again"}, "id": "c2"}]
                return msg

            def with_structured_output(self, model, **kw):
                return self

        register_tool_factory(
            "search",
            lambda c, tc: StructuredTool.from_function(lambda q="": f"r {q}", name="search", description="search tool"),
        )

        tools = [Tool(name="search", description="search tool")]
        budget = ToolBudgetTracker(tools)

        with capture_logs() as logs:
            invoke_with_tools(
                model_tier="fast",
                prompt_template="test",
                input_data={},
                output_model=Result,
                config={"configurable": {}},
                tools=tools,
                budget_tracker=budget,
                llm_config={"max_iterations": 1, "output_strategy": "json_mode"},
                runtime=build_fake_runtime(lambda tier: RogueFake()),
                tool_factory_lookup=build_fake_tool_lookup(),
            )

        events = [e for e in logs if e.get("event") == "react_guard_forced_break"]
        assert events, (
            f"contract: ``react_guard_forced_break`` must fire when guard is "
            f"set but the LLM emits rogue tool calls. Events observed: "
            f"{[e.get('event') for e in logs]}"
        )

    def test_event_payload_includes_loops_and_tool_calls(self):
        from langchain_core.messages import AIMessage
        from langchain_core.tools import StructuredTool

        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Result(BaseModel):
            answer: str

        class RogueFake:
            def __init__(self):
                self.calls = 0

            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                self.calls += 1
                if self.calls == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search", "args": {"q": "x"}, "id": "c1"}]
                    return msg
                msg = AIMessage(content='{"answer": "ok"}')
                msg.tool_calls = [{"name": "search", "args": {"q": "again"}, "id": "c2"}]
                return msg

            def with_structured_output(self, model, **kw):
                return self

        register_tool_factory(
            "search",
            lambda c, tc: StructuredTool.from_function(lambda q="": f"r {q}", name="search", description="search tool"),
        )

        tools = [Tool(name="search", description="search tool")]
        budget = ToolBudgetTracker(tools)

        with capture_logs() as logs:
            invoke_with_tools(
                model_tier="fast",
                prompt_template="test",
                input_data={},
                output_model=Result,
                config={"configurable": {}},
                tools=tools,
                budget_tracker=budget,
                llm_config={"max_iterations": 1, "output_strategy": "json_mode"},
                runtime=build_fake_runtime(lambda tier: RogueFake()),
                tool_factory_lookup=build_fake_tool_lookup(),
            )

        events = [e for e in logs if e.get("event") == "react_guard_forced_break"]
        assert events, "contract event not emitted"
        evt = events[0]
        # Payload contract: operators monitor these fields to detect
        # provider-side regressions where the LLM ignores tool-unbinding.
        assert "loops" in evt
        assert "tool_calls" in evt
        # Severity contract.
        assert evt.get("log_level") == "warning"


class TestAutoResumeSchemaChangeEvent:
    """Contract: when the checkpoint schema fingerprint differs from the
    current graph's fingerprint and ``auto_resume=True``, the framework emits
    ``auto_resume_schema_change`` at info level with the invalidated nodes."""

    def test_event_name_emitted_on_schema_divergence(self):
        from langgraph.checkpoint.memory import MemorySaver

        from tests.fakes import register_scripted

        class TypeAV1(BaseModel):
            val: str = "a1"

        class TypeAV2(BaseModel):
            val: str = "a2"
            extra: int = 0

        register_scripted("ocs_a1", lambda _i, _c: TypeAV1())
        register_scripted("ocs_a2", lambda _i, _c: TypeAV2())

        checkpointer = MemorySaver()
        config = {"configurable": {"thread_id": "obs-contract-schema"}}

        pipe_v1 = Construct(
            "obs-contract-pipe",
            nodes=[
                Node.scripted("a", fn="ocs_a1", outputs=TypeAV1),
            ],
        )
        graph_v1 = compile(pipe_v1, checkpointer=checkpointer, **build_test_compile_kwargs())
        run(graph_v1, input={"node_id": "test"}, config=config)

        pipe_v2 = Construct(
            "obs-contract-pipe",
            nodes=[
                Node.scripted("a", fn="ocs_a2", outputs=TypeAV2),
            ],
        )
        graph_v2 = compile(pipe_v2, checkpointer=checkpointer, **build_test_compile_kwargs())

        with capture_logs() as logs:
            run(graph_v2, input={"node_id": "test"}, config=config, auto_resume=True)

        events = [e for e in logs if e.get("event") == "auto_resume_schema_change"]
        assert events, (
            f"contract: ``auto_resume_schema_change`` must fire when the "
            f"checkpoint schema fingerprint mismatches. Events observed: "
            f"{[e.get('event') for e in logs]}"
        )

    def test_event_payload_includes_invalidated_and_fingerprints(self):
        from langgraph.checkpoint.memory import MemorySaver

        from tests.fakes import register_scripted

        class TypeBV1(BaseModel):
            val: str = "b1"

        class TypeBV2(BaseModel):
            val: str = "b2"
            extra: int = 0

        register_scripted("ocs_b1", lambda _i, _c: TypeBV1())
        register_scripted("ocs_b2", lambda _i, _c: TypeBV2())

        checkpointer = MemorySaver()
        config = {"configurable": {"thread_id": "obs-contract-schema-2"}}

        pipe_v1 = Construct(
            "obs-contract-pipe-2",
            nodes=[
                Node.scripted("b", fn="ocs_b1", outputs=TypeBV1),
            ],
        )
        graph_v1 = compile(pipe_v1, checkpointer=checkpointer, **build_test_compile_kwargs())
        run(graph_v1, input={"node_id": "test"}, config=config)

        pipe_v2 = Construct(
            "obs-contract-pipe-2",
            nodes=[
                Node.scripted("b", fn="ocs_b2", outputs=TypeBV2),
            ],
        )
        graph_v2 = compile(pipe_v2, checkpointer=checkpointer, **build_test_compile_kwargs())

        with capture_logs() as logs:
            run(graph_v2, input={"node_id": "test"}, config=config, auto_resume=True)

        events = [e for e in logs if e.get("event") == "auto_resume_schema_change"]
        assert events, "contract event not emitted"
        evt = events[0]
        # Payload contract: operators rely on these fields to diagnose
        # schema divergence and identify which nodes will re-execute.
        assert "invalidated" in evt
        assert "stored_fp" in evt
        assert "current_fp" in evt
        # Severity contract: info, not warning — schema change is a normal
        # development workflow event.
        assert evt.get("log_level") == "info"


class TestAgentNodeLifecycleEvents:
    """Contract: agent/act nodes emit ``node_start`` and ``node_complete`` with
    the PAT-02 observability shape (neograph-ykun).

    ``node_start`` routes ``input_type``/``output_type`` through the shared
    ``type_display_name`` renderer, so a node declaring dict-form outputs
    (``{"result": ..., "tool_log": ...}``) reports a real ``output_type`` instead
    of the pre-PAT-02 hard-coded ``None`` (the old ``node.outputs.__name__ if
    isinstance(node.outputs, type) else None`` collapsed dict-form to ``None``).
    ``node_complete`` carries ``duration_s`` derived from the turn's stashed
    ``t0``. Both assertions below fail against the pre-PAT-02 shape.
    """

    def _drive_clean_completion(self):
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import ReActFake
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools

        # First turn returns no tool calls -> the loop parses the final answer,
        # so the cycle runs node_start (first turn) through node_complete (parse).
        fake = ReActFake(
            tool_calls=[[]],
            final=lambda m: m(items=["done"]),
            output_model=Claims,
        )
        _llm_kw = configure_fake_llm(lambda tier: fake)
        tools: list = []

        with capture_logs() as logs:
            invoke_with_tools(
                runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
                model_tier="fast",
                prompt_template="test",
                input_data="test",
                output_model=Claims,
                tools=tools,
                budget_tracker=ToolBudgetTracker(tools),
                config={"configurable": {}},
                node_name="research",
                tool_factory_lookup=build_fake_tool_lookup(),
            )
        return logs

    def test_node_start_reports_dict_form_output_type_via_type_display_name(self):
        logs = self._drive_clean_completion()
        starts = [e for e in logs if e.get("event") == "node_start"]
        assert starts, (
            f"contract: agent node must emit ``node_start``. Events observed: {[e.get('event') for e in logs]}"
        )
        evt = starts[0]
        # PAT-02: dict-form outputs now render a real ``output_type`` string;
        # the pre-PAT-02 code emitted None for a non-``type`` outputs value.
        assert evt.get("output_type") is not None, (
            "PAT-02 regression: node_start output_type is None for a node with "
            "dict-form outputs — the pre-PAT-02 hard-coded shape."
        )
        assert isinstance(evt.get("output_type"), str)

    def test_node_complete_includes_duration_s(self):
        logs = self._drive_clean_completion()
        completes = [e for e in logs if e.get("event") == "node_complete"]
        assert completes, (
            f"contract: agent node must emit ``node_complete``. Events observed: {[e.get('event') for e in logs]}"
        )
        evt = completes[0]
        # PAT-02: node_complete now carries duration_s (absent pre-PAT-02).
        assert "duration_s" in evt, "PAT-02 regression: node_complete dropped duration_s."
        assert isinstance(evt["duration_s"], (int, float))


# A note on the audit's ``token_budget_exhausted`` and ``auto_resume_rewind``
# names: the audit doc listed these as proposed event names, but the actual
# production code emits ``react_token_budget_exceeded`` (via the dynamic
# ``f"react_{reason}_exceeded"`` formatter in _tool_loop.py:328) and
# ``auto_resume_schema_change`` (runner.py:124). This contract file pins the
# names as they actually fire today; a future event-name refactor would
# update both source and this file in a single deliberate change.
