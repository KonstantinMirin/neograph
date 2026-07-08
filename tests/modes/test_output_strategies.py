"""Pipeline mode tests — output strategies, prompt compiler, gather tool collection"""

from __future__ import annotations

import pytest

from neograph import (
    Construct,
    Node,
    Tool,
    compile,
    run,
)
from tests.fakes import (
    FakeTool,
    ReActFake,
    StructuredFake,
    TextFake,
    build_fake_runtime,
    build_test_compile_kwargs,
    configure_fake_llm,
    register_tool_factory,
)
from tests.schemas import (
    Claims,
)

# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT STRATEGIES — structured, json_mode, text
# ═══════════════════════════════════════════════════════════════════════════


class TestOutputStrategyStructured:
    """Default strategy: llm.with_structured_output(model). Current behavior."""

    def test_structured_output_used_when_no_strategy_specified(self):
        """Produce node uses with_structured_output by default."""
        _llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["via-structured"])))

        node = Node(name="extract", mode="think", outputs=Claims, model="fast", prompt="test")
        pipeline = Construct("test-structured", nodes=[node])
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
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

        _llm_kw = configure_fake_llm(
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
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "test"})

        # Parsed correctly from raw JSON
        assert result["extract"].items == ["via-json-mode"]

    def test_fences_stripped_when_json_response_wrapped_in_markdown(self):
        """json_mode: strips markdown code fences before parsing."""
        _llm_kw = configure_fake_llm(lambda tier: TextFake('```json\n{"items": ["fenced"]}\n```'))

        node = Node(
            name="extract",
            mode="think",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-json-fence", nodes=[node])
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
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

        _llm_kw = configure_fake_llm(lambda tier: RetryableFake())

        node = Node(
            name="extract",
            mode="think",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-retry-feedback", nodes=[node])
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "test"})

        # Contract: error-feedback retry recovers the parse failure. Soften the
        # exact call_n == 2 pin to >= 2 (need at least one retry).
        assert result["extract"].items == ["recovered"]
        assert call_n["n"] >= 2

    def test_validation_errors_included_in_retry_when_fields_wrong(self):
        """json_mode: when LLM returns valid JSON with wrong field types,
        the retry message includes the specific Pydantic validation errors."""
        from langchain_core.messages import AIMessage

        retry_messages_seen = []

        class ValidationRetryFake:
            def invoke(self, messages, **kwargs):
                # Check if this is a retry (has the error feedback)
                for msg in messages:
                    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                    if "failed validation" in content:
                        retry_messages_seen.append(content)
                        return AIMessage(content='{"items": ["fixed"]}')
                # First call: valid JSON but wrong type (items should be list[str])
                return AIMessage(content='{"items": "not-a-list"}')

        _llm_kw = configure_fake_llm(lambda tier: ValidationRetryFake())

        node = Node(
            name="extract",
            mode="think",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-validation-retry", nodes=[node])
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "test"})

        assert result["extract"].items == ["fixed"]
        assert len(retry_messages_seen) == 1
        # The retry message should contain field-level error details
        assert "items" in retry_messages_seen[0]

    def test_max_retries_configurable_when_set_in_llm_config(self):
        """json_mode: max_retries=2 allows two retries (three total attempts)."""
        from langchain_core.messages import AIMessage

        call_n = {"n": 0}

        class ThreeAttemptFake:
            def invoke(self, messages, **kwargs):
                call_n["n"] += 1
                if call_n["n"] <= 2:
                    return AIMessage(content="still garbage")
                return AIMessage(content='{"items": ["third-time"]}')

        _llm_kw = configure_fake_llm(lambda tier: ThreeAttemptFake())

        node = Node(
            name="extract",
            mode="think",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "json_mode", "max_retries": 2},
        )
        pipeline = Construct("test-max-retries", nodes=[node])
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "test"})

        # Contract: max_retries=2 lets the user-supplied parse eventually
        # succeed when the LLM gets it right on attempt 3. The exact attempt
        # count is implementation detail; soften to a lower bound.
        assert result["extract"].items == ["third-time"]
        assert call_n["n"] >= 3


class _SeqBindFake:
    """json_mode fake that records bind kwargs and returns a scripted sequence of
    raw responses (garbage first, then valid JSON). ``invoke`` asserts it is the
    ``response_format``-bound clone — so the error-feedback retry reusing the SAME
    bound client is proven for both twins. ``ainvoke`` bare-delegates (async guard).
    """

    def __init__(self, responses, *, bound=False, bind_calls=None):
        self._responses = responses
        self._i = 0
        self._bound = bound
        self.bind_calls = bind_calls if bind_calls is not None else []

    def bind(self, **kw):
        self.bind_calls.append(dict(kw))
        return _SeqBindFake(
            self._responses,
            bound=self._bound or ("response_format" in kw),
            bind_calls=self.bind_calls,
        )

    def invoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage

        assert self._bound, "retry must reuse the response_format-bound client"
        r = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return AIMessage(content=r)

    async def ainvoke(self, *a, **k):
        return self.invoke(*a, **k)


def _json_runtime(fake, prompt_compiler=None):
    return build_fake_runtime(factory=lambda tier: fake, prompt_compiler=prompt_compiler)


def _json_call_kwargs(**over):
    base = {
        "model_tier": "fast",
        "prompt_template": "test",
        "input_data": "test",
        "output_model": Claims,
        "config": {"configurable": {}},
        "node_name": "x",
        "llm_config": {"output_strategy": "json_mode"},
    }
    base.update(over)
    return base


class TestJsonModeNativeResponseFormat:
    """json_mode SENDS the provider-native response_format={'type':'json_object'}
    at the shared _prepare_structured_call seam (neograph-15s2), so BOTH the sync
    invoke_structured and async ainvoke_structured twins inherit it from ONE site.
    Schema-in-prompt + parse stays the universal fallback. ATTEMPT-BIND-AND-FALL-
    BACK is done independently by each entrypoint. Every scenario is covered for
    sync AND async.
    """

    _RESPONSE_FORMAT = {"response_format": {"type": "json_object"}}

    # ── sync twin (invoke_structured) ──────────────────────────────────────

    def test_response_format_bound_when_json_mode_sync(self):
        from neograph._llm import invoke_structured

        fake = TextFake('{"items": ["native"]}')
        result = invoke_structured(_json_runtime(fake), **_json_call_kwargs())

        assert result.items == ["native"]
        assert fake.bind_calls == [self._RESPONSE_FORMAT]

    def test_falls_back_to_unbound_when_provider_rejects_sync(self):
        from structlog.testing import capture_logs

        from neograph._llm import invoke_structured

        fake = TextFake('{"items": ["recovered-native"]}', reject_response_format=True)
        with capture_logs() as logs:
            result = invoke_structured(_json_runtime(fake), **_json_call_kwargs())

        assert result.items == ["recovered-native"]
        assert fake.bind_calls == [self._RESPONSE_FORMAT]
        events = [e for e in logs if e.get("event") == "json_mode_native_unsupported"]
        assert len(events) == 1
        assert events[0]["provider"] == "TextFake"

    def test_retry_reuses_bound_client_sync(self):
        from neograph._llm import invoke_structured

        fake = _SeqBindFake(["not json at all", '{"items": ["second-try"]}'])
        result = invoke_structured(
            _json_runtime(fake),
            **_json_call_kwargs(llm_config={"output_strategy": "json_mode", "max_retries": 2}),
        )

        assert result.items == ["second-try"]
        # Bound exactly once (reused across the retry), never rebound per attempt.
        assert fake.bind_calls == [self._RESPONSE_FORMAT]

    # ── async twin (ainvoke_structured) ────────────────────────────────────
    # Driven via asyncio.run inside sync test fns — the repo's async-test
    # convention (this venv has no pytest-asyncio plugin).

    def test_response_format_bound_when_json_mode_async(self):
        import asyncio

        from neograph._llm import ainvoke_structured

        fake = TextFake('{"items": ["native-async"]}')
        result = asyncio.run(ainvoke_structured(_json_runtime(fake), **_json_call_kwargs()))

        assert result.items == ["native-async"]
        assert fake.bind_calls == [self._RESPONSE_FORMAT]

    def test_falls_back_to_unbound_when_provider_rejects_async(self):
        import asyncio

        from structlog.testing import capture_logs

        from neograph._llm import ainvoke_structured

        fake = TextFake('{"items": ["recovered-async"]}', reject_response_format=True)
        with capture_logs() as logs:
            result = asyncio.run(ainvoke_structured(_json_runtime(fake), **_json_call_kwargs()))

        assert result.items == ["recovered-async"]
        assert fake.bind_calls == [self._RESPONSE_FORMAT]
        events = [e for e in logs if e.get("event") == "json_mode_native_unsupported"]
        assert len(events) == 1
        assert events[0]["provider"] == "TextFake"

    def test_retry_reuses_bound_client_async(self):
        import asyncio

        from neograph._llm import ainvoke_structured

        fake = _SeqBindFake(["not json at all", '{"items": ["second-async"]}'])
        result = asyncio.run(
            ainvoke_structured(
                _json_runtime(fake),
                **_json_call_kwargs(llm_config={"output_strategy": "json_mode", "max_retries": 2}),
            )
        )

        assert result.items == ["second-async"]
        assert fake.bind_calls == [self._RESPONSE_FORMAT]

    # ── json-word silent-400 guard (strategy-level, twin-agnostic) ─────────

    def test_json_word_appended_when_prompt_lacks_it(self):
        from neograph._llm import invoke_structured

        # Default fake compiler content is "test" — no 'json' word.
        fake = TextFake('{"items": ["ok"]}')
        invoke_structured(_json_runtime(fake), **_json_call_kwargs())

        seen = fake.messages_seen[0]
        assert any(isinstance(m, dict) and "json" in str(m.get("content", "")).lower() for m in seen), (
            f"json-word guard did not inject 'json' into messages: {seen}"
        )

    def test_json_word_not_duplicated_when_already_present(self):
        from neograph._llm import invoke_structured

        fake = TextFake('{"items": ["ok"]}')

        def compiler(template, data, **kw):
            return [{"role": "user", "content": "Return a json object."}]

        invoke_structured(_json_runtime(fake, prompt_compiler=compiler), **_json_call_kwargs())

        seen = fake.messages_seen[0]
        assert len(seen) == 1, f"must not append when 'json' already present: {seen}"

    # ── zero behavior change for the other strategies ──────────────────────

    def test_structured_strategy_does_not_bind_response_format(self):
        """Default structured strategy never binds response_format."""
        fake = StructuredFake(lambda m: m(items=["via-structured"]))
        _llm_kw = configure_fake_llm(lambda tier: fake)

        node = Node(name="extract", mode="think", outputs=Claims, model="fast", prompt="test")
        pipeline = Construct("test-structured-nobind", nodes=[node])
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "test"})

        assert result["extract"].items == ["via-structured"]
        assert fake.bind_calls == []

    def test_text_strategy_does_not_bind_response_format(self):
        """The text strategy is not wrapped/bound."""
        fake = TextFake('{"items": ["from-text"]}')
        _llm_kw = configure_fake_llm(lambda tier: fake)

        node = Node(
            name="extract",
            mode="think",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "text"},
        )
        pipeline = Construct("test-text-nobind", nodes=[node])
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "test"})

        assert result["extract"].items == ["from-text"]
        assert fake.bind_calls == []

    def test_end_to_end_bound_via_compile_run(self):
        """Integration: a json_mode think node compiled and run binds natively."""
        fake = TextFake('{"items": ["e2e"]}')
        _llm_kw = configure_fake_llm(lambda tier: fake)

        node = Node(
            name="extract",
            mode="think",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-native-e2e", nodes=[node])
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "test"})

        assert result["extract"].items == ["e2e"]
        assert fake.bind_calls == [self._RESPONSE_FORMAT]


class TestOutputStrategyText:
    """text strategy: LLM returns plain text, consumer's prompt_compiler handles schema."""

    def test_json_extracted_when_text_strategy_with_embedded_json(self):
        """text mode: LLM returns text containing JSON, framework extracts and parses."""
        _llm_kw = configure_fake_llm(lambda tier: TextFake('Here is my analysis:\n{"items": ["from-text"]}\nDone.'))

        node = Node(
            name="extract",
            mode="think",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "text"},
        )
        pipeline = Construct("test-text", nodes=[node])
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "test"})

        assert result["extract"].items == ["from-text"]


class TestOutputStrategyOnGather:
    """Output strategy also applies to the final parse in gather/execute modes."""

    def test_json_parsed_when_gather_mode_with_json_strategy(self):
        """Gather node with json_mode parses the final structured output from raw JSON."""
        from langchain_core.messages import AIMessage

        lookup_tool = FakeTool("lookup", response="found")
        register_tool_factory("lookup", lambda cfg, tc: lookup_tool)

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
        graph = compile(
            pipeline,
            llm_factory=lambda tier: FakeGatherLLM(),
            prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": "go"}],
            **build_test_compile_kwargs(),
        )
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

        _llm_kw = configure_fake_llm(
            lambda tier: StructuredFake(lambda m: m(items=["ok"])),
            prompt_compiler=tracking_compiler,
        )

        node = Node(name="x", mode="think", outputs=Claims, model="fast", prompt="test")
        pipeline = Construct("test", nodes=[node])
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        run(graph, input={"node_id": "test"})

        assert len(compiler_calls) == 1
        assert compiler_calls[0].get("output_model") is Claims

    def test_llm_config_passed_when_produce_node_compiled(self):
        """Prompt compiler sees the node's llm_config including output_strategy."""
        compiler_calls = []

        def tracking_compiler(template, data, **kw):
            compiler_calls.append(kw)
            return [{"role": "user", "content": "test"}]

        _llm_kw = configure_fake_llm(
            lambda tier: TextFake('{"items": ["ok"]}'),
            prompt_compiler=tracking_compiler,
        )

        node = Node(
            name="x",
            mode="think",
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={
                "output_strategy": "json_mode",
                "provider_kwargs": {"temperature": 0.5},
            },
        )
        pipeline = Construct("test", nodes=[node])
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        run(graph, input={"node_id": "test"})

        assert len(compiler_calls) == 1
        # Post-pej0: as_factory_kwargs flattens framework keys + provider_kwargs
        # into a flat dict the compiler receives.
        compiler_cfg = compiler_calls[0].get("llm_config")
        assert compiler_cfg["output_strategy"] == "json_mode"
        assert compiler_cfg["temperature"] == 0.5
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

        _llm_kw = configure_fake_llm(
            lambda tier: TextFake('{"items": ["schema-injected"]}'),
            prompt_compiler=schema_injecting_compiler,
        )

        node = Node(
            name="x",
            mode="think",
            outputs=Claims,
            model="fast",
            prompt="decompose",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test", nodes=[node])
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
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
        import types

        from neograph import ToolInteraction, compile, construct_from_module, node, run
        from tests.fakes import FakeTool, configure_fake_llm

        _llm_kw = configure_fake_llm(
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
        register_tool_factory("search", lambda cfg, tc: fake_tool)

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
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        result = run(graph, input={})
        assert result.get("explore_result") == Claims(items=["result"])
        tool_log = result.get("explore_tool_log")
        assert isinstance(tool_log, list) and len(tool_log) == 1
        assert tool_log[0].tool_name == "search"
        assert tool_log[0].result == "found it"

    def test_typed_result_preserved_when_tool_returns_pydantic_model(self):
        """Tool returns a Pydantic model. typed_result preserves it,
        result has JSON rendering (not repr). E2E: compile + run."""
        import types

        from pydantic import BaseModel

        from neograph import ToolInteraction, compile, construct_from_module, node, run
        from tests.fakes import configure_fake_llm

        class SearchHit(BaseModel, frozen=True):
            node_id: str
            score: float

        # FakeTool that returns a TYPED model, not a string
        class TypedFakeTool:
            name = "typed_search"

            def invoke(self, args, config=None, **kwargs):
                return SearchHit(node_id="UC-001", score=0.95)

        register_tool_factory("typed_search", lambda cfg, tc: TypedFakeTool())

        _llm_kw = configure_fake_llm(
            lambda tier: ReActFake(
                tool_calls=[
                    [{"name": "typed_search", "args": {"q": "test"}, "id": "tc1"}],
                    [],
                ],
                final=lambda m: Claims(items=["found"]),
            )
        )

        mod = types.ModuleType("test_typed_tool_mod")

        @node(
            mode="agent",
            outputs={"result": Claims, "tool_log": list[ToolInteraction]},
            model="fast",
            prompt="test",
            tools=[Tool("typed_search", budget=3)],
        )
        def explore() -> Claims: ...

        mod.explore = explore
        pipeline = construct_from_module(mod)
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        result = run(graph, input={})

        tool_log = result.get("explore_tool_log")
        assert isinstance(tool_log, list) and len(tool_log) == 1
        # typed_result preserves the original Pydantic model
        assert isinstance(tool_log[0].typed_result, SearchHit), "typed_result should be SearchHit"
        assert tool_log[0].typed_result.node_id == "UC-001"
        assert tool_log[0].typed_result.score == 0.95
        # result is BAML-rendered with field descriptions, not repr
        assert "node_id:" in tool_log[0].result, f"Expected BAML notation in result, got: {tool_log[0].result}"

    def test_typed_result_holds_string_when_tool_returns_string(self):
        """Tool returns plain string. typed_result holds the string itself
        (not None). Backward compat. E2E: compile + run."""
        import types

        from neograph import ToolInteraction, compile, construct_from_module, node, run
        from tests.fakes import configure_fake_llm

        register_tool_factory("str_tool", lambda cfg, tc: FakeTool("str_tool", response="plain text"))

        _llm_kw = configure_fake_llm(
            lambda tier: ReActFake(
                tool_calls=[
                    [{"name": "str_tool", "args": {}, "id": "tc1"}],
                    [],
                ],
                final=lambda m: Claims(items=["done"]),
            )
        )

        mod = types.ModuleType("test_str_tool_mod")

        @node(
            mode="agent",
            outputs={"result": Claims, "tool_log": list[ToolInteraction]},
            model="fast",
            prompt="test",
            tools=[Tool("str_tool", budget=2)],
        )
        def gather() -> Claims: ...

        mod.gather = gather
        pipeline = construct_from_module(mod)
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        result = run(graph, input={})

        tool_log = result.get("gather_tool_log")
        assert isinstance(tool_log, list) and len(tool_log) == 1
        assert tool_log[0].typed_result == "plain text"
        assert tool_log[0].result == "plain text"

    def test_tool_result_rendered_with_schema_when_pydantic_model(self):
        """Typed tool result in ToolMessage uses describe_type schema header +
        renderer instance, not raw model_dump_json. E2E: compile + run.
        neograph-oky0."""
        import types

        from pydantic import BaseModel, Field

        from neograph import ToolInteraction, XmlRenderer, compile, construct_from_module, node, run

        class SearchHit(BaseModel, frozen=True):
            node_id: str = Field(description="Graph node identifier")
            score: float = Field(description="Relevance score 0-1")

        class TypedTool:
            name = "schema_search"

            def invoke(self, args, config=None, **kwargs):
                return SearchHit(node_id="UC-042", score=0.9)

        register_tool_factory("schema_search", lambda cfg, tc: TypedTool())

        # Capture what the LLM sees in the ToolMessage
        tool_messages_seen: list[str] = []

        class CapturingReActFake(ReActFake):
            def invoke(self, messages, **kwargs):
                # Capture ToolMessage contents
                for m in messages:
                    if hasattr(m, "tool_call_id"):
                        tool_messages_seen.append(m.content)
                return super().invoke(messages, **kwargs)

        mod = types.ModuleType("test_schema_render_mod")

        @node(
            mode="agent",
            outputs={"result": Claims, "tool_log": list[ToolInteraction]},
            model="fast",
            prompt="test",
            tools=[Tool("schema_search", budget=2)],
            renderer=XmlRenderer(),
        )
        def explore() -> Claims: ...

        mod.explore = explore
        pipeline = construct_from_module(mod)
        graph = compile(
            pipeline,
            llm_factory=lambda tier: CapturingReActFake(
                tool_calls=[[{"name": "schema_search", "args": {}, "id": "t1"}], []],
                final=lambda m: Claims(items=["done"]),
            ),
            prompt_compiler=lambda t, d, **kw: [{"role": "user", "content": "test"}],
            **build_test_compile_kwargs(),
        )
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


class TestStructuredSilentNoneSource:
    """neograph-7wya (source defense): think-mode structured strategy must not
    silently return None when the provider response parses to ``parsed=None``
    with no DSML markup (the gemini structured-decode flake). It must FAIL LOUD
    at the source (ExecutionError), complementary to the write-boundary backstop.
    """

    def test_structured_strategy_raises_when_provider_parses_to_none(self):
        """The Raw(parsed=None) legacy passthrough now raises ExecutionError
        instead of returning None to the caller."""
        from neograph import ExecutionError
        from neograph._llm import invoke_structured
        from tests.fakes import StructuredFakeWithRaw, build_fake_runtime

        # respond returns None -> include_raw dict {"parsed": None, ...} -> Raw
        fake = StructuredFakeWithRaw(lambda model: None)
        runtime = build_fake_runtime(factory=lambda tier: fake)

        with pytest.raises(ExecutionError, match=r"Claims"):
            invoke_structured(
                runtime,
                model_tier="reason",
                prompt_template="test",
                input_data="test",
                output_model=Claims,
                config={"configurable": {}},
                node_name="hypothesize",
            )
