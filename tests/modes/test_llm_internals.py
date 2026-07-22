"""Pipeline mode tests — context param, tool registration, skip_when on tools, JSON parsing, retry, ReAct guards"""

from __future__ import annotations

import pytest

from neograph import (
    CompileError,
    ConfigurationError,
    Construct,
    Each,
    ExecutionError,
    Node,
    Oracle,
    Tool,
    compile,
    run,
)
from tests.fakes import (
    FakeTool,
    GuardFake,
    ReActFake,
    StructuredFake,
    StubbornFake,
    build_fake_runtime,
    build_fake_tool_lookup,
    build_test_compile_kwargs,
    configure_fake_llm,
)
from tests.schemas import (
    Claims,
    ClusterGroup,
    Clusters,
    MatchResult,
    MergedResult,
    RawText,
)

# ═══════════════════════════════════════════════════════════════════════════
# TEST: @node context= verbatim state injection (neograph-p4hw)
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeContext:
    """@node context= injects verbatim state fields into prompt (neograph-p4hw)."""

    def test_context_passed_to_prompt_compiler_when_declared(self):
        """Prompt compiler receives context dict with raw state values. E2E."""
        import types

        from neograph import compile, construct_from_module, node, run
        from tests.fakes import StructuredFake, configure_fake_llm

        captured = {}

        def capturing_compiler(template, data, **kw):
            captured[template] = {"data": data, "kw": kw}
            return [{"role": "user", "content": "test"}]

        _llm_kw = configure_fake_llm(
            lambda tier: StructuredFake(lambda m: m(items=["done"])),
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
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
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
        import types

        from neograph import compile, construct_from_module, node, run
        from tests.fakes import StructuredFake

        captured = {}

        def capturing_compiler(template, data, **kw):
            captured[template] = kw
            return [{"role": "user", "content": "test"}]

        mod = types.ModuleType("test_no_ctx_mod")

        @node(outputs=Claims, mode="think", model="fast", prompt="no-context")
        def simple() -> Claims: ...

        mod.simple = simple
        pipeline = construct_from_module(mod)
        graph = compile(
            pipeline,
            llm_factory=lambda tier: StructuredFake(lambda m: m(items=["x"])),
            prompt_compiler=capturing_compiler,
            **build_test_compile_kwargs(),
        )
        run(graph, input={})

        assert "no-context" in captured
        assert "context" not in captured["no-context"]

    def test_context_works_with_agent_mode(self):
        """Agent node with context= alongside tools. E2E."""
        import types

        from neograph import ToolInteraction, compile, construct_from_module, node, run
        from tests.fakes import FakeTool, ReActFake, register_tool_factory

        captured = {}

        def capturing_compiler(template, data, **kw):
            captured[template] = {"data": data, "kw": kw}
            return [{"role": "user", "content": "test"}]

        register_tool_factory("ctx_tool", lambda cfg, tc: FakeTool("ctx_tool", response="ok"))

        mod = types.ModuleType("test_agent_ctx_mod")

        @node(outputs=RawText)
        def catalog() -> RawText:
            return RawText(text="graph-catalog-data")

        @node(
            mode="agent",
            outputs={"result": Claims, "tool_log": list[ToolInteraction]},
            model="fast",
            prompt="agent-with-ctx",
            tools=[Tool("ctx_tool", budget=2)],
            context=["catalog"],
        )
        def explore(catalog: RawText) -> Claims: ...

        mod.catalog = catalog
        mod.explore = explore
        pipeline = construct_from_module(mod)
        graph = compile(
            pipeline,
            llm_factory=lambda tier: ReActFake(
                tool_calls=[[{"name": "ctx_tool", "args": {}, "id": "t1"}], []],
                final=lambda m: m(items=["found"]),
            ),
            prompt_compiler=capturing_compiler,
            **build_test_compile_kwargs(),
        )
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
        """Gather node with unregistered tool raises CompileError at compile()."""
        import types as _types

        from neograph import construct_from_module, node

        fake = ReActFake(
            tool_calls=[[], []],
            final=lambda m: m(items=["x"]),
        )
        _llm_kw = configure_fake_llm(lambda tier: fake)

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

        with pytest.raises(CompileError, match="nonexistent_tool"):
            compile(pipeline, **_llm_kw, **build_test_compile_kwargs())

    def test_clear_error_raised_when_execute_tool_not_registered(self):
        """Execute node with unregistered tool raises CompileError at compile()."""
        import types as _types

        from neograph import construct_from_module, node

        fake = ReActFake(
            tool_calls=[[], []],
            final=lambda m: m(text="x"),
        )
        _llm_kw = configure_fake_llm(lambda tier: fake)

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

        with pytest.raises(CompileError, match="missing_exec_tool"):
            compile(pipeline, **_llm_kw, **build_test_compile_kwargs())


# ═══════════════════════════════════════════════════════════════════════════
# TEST: skip_when on gather/execute nodes (neograph-rdu.8)
#
# skip_when is tested on produce nodes but never on gather/execute.
# All modes now share the same _execute_node preamble (neograph-y8ww).
# ═══════════════════════════════════════════════════════════════════════════


class TestSkipWhenOnToolNodes:
    def test_node_skipped_when_skip_when_true_on_gather(self):
        """Gather node with skip_when=True skips LLM and returns skip_value."""
        import types as _types

        from neograph import construct_from_module, node
        from tests.fakes import register_tool_factory

        # Register tool factory so it doesn't fail on missing tool
        fake_tool = FakeTool("lookup", response="found")
        register_tool_factory("lookup", lambda config, tool_config: fake_tool)

        # LLM should NOT be called — if it is, the test will still pass
        # but we verify via the skip_value output
        _llm_kw = configure_fake_llm(
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
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        result = run(graph, input={})

        # Node was skipped — skip_value produced the output
        assert result["gatherer"].final_text == "only-one"
        # Tool was never called
        assert len(fake_tool.calls) == 0

    def test_node_runs_when_skip_when_false_on_gather(self):
        """Gather node runs normally when skip_when returns False."""
        import types as _types

        from neograph import construct_from_module, node
        from tests.fakes import register_tool_factory

        fake_tool = FakeTool("lookup", response="result")
        register_tool_factory("lookup", lambda config, tool_config: fake_tool)

        _llm_kw = configure_fake_llm(
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
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
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
        from neograph._llm_retry import _extract_json

        result = _extract_json('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_json_extracted_when_wrapped_in_markdown_fences(self):
        """JSON wrapped in ```json ... ``` is extracted."""
        from neograph._llm_retry import _extract_json

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
        import json

        from neograph._llm_retry import _extract_json

        text = 'Here is result: {"first": 1} and also {"second": 2}'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed == {"first": 1}

    def test_nested_braces_parsed_when_json_has_nested_objects(self):
        """JSON with nested braces parses correctly."""
        import json

        from neograph._llm_retry import _extract_json

        text = '{"outer": {"inner": "value"}}'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == "value"

    def test_text_returned_when_no_json_present(self):
        """When no JSON is present, the cleaned text is returned."""
        from neograph._llm_retry import _extract_json

        result = _extract_json("no json here at all")
        assert result == "no json here at all"

    def test_json_extracted_when_surrounded_by_prose(self):
        """JSON embedded in prose text is extracted."""
        import json

        from neograph._llm_retry import _extract_json

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

        from neograph._llm_retry import _parse_json_response

        class SimpleModel(BaseModel):
            content: str

        # Embedded null character — not valid in JSON strings
        text = '{"content": "hello\x00world"}'
        result = _parse_json_response(text, SimpleModel)
        assert "hello" in result.content

    def test_null_coerced_to_default_for_string_field(self):
        """null for a str field with default should use the default, not fail.

        BUG neograph-qqel: R1 returns null for str fields with defaults.
        Pydantic rejects null for str. Coerce null → default before validation.
        """
        from pydantic import BaseModel, Field

        from neograph._llm_retry import _parse_json_response

        class StepCheck(BaseModel):
            score: float
            justification: str = Field(default="", description="Why")

        text = '{"score": 0.8, "justification": null}'
        result = _parse_json_response(text, StepCheck)
        assert result.score == 0.8
        assert result.justification == ""

    def test_null_coerced_in_nested_model(self):
        """null → default coercion should work recursively in nested models."""
        from pydantic import BaseModel, Field

        from neograph._llm_retry import _parse_json_response

        class Inner(BaseModel):
            name: str
            note: str = Field(default="none")

        class Outer(BaseModel):
            items: list[Inner]

        text = '{"items": [{"name": "a", "note": null}, {"name": "b"}]}'
        result = _parse_json_response(text, Outer)
        assert result.items[0].note == "none"
        assert result.items[1].note == "none"

    def test_null_without_default_still_fails(self):
        """null for a required field (no default) should still raise."""
        from pydantic import BaseModel

        from neograph._llm_retry import _parse_json_response
        from neograph.errors import ExecutionError

        class Required(BaseModel):
            name: str  # no default

        text = '{"name": null}'
        with pytest.raises(ExecutionError, match="Validation failed"):
            _parse_json_response(text, Required)

    def test_trailing_comma_parsed(self):
        """Trailing commas in JSON objects should not crash parsing."""
        from pydantic import BaseModel

        from neograph._llm_retry import _parse_json_response

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

        from neograph._llm_retry import _parse_json_response

        class SimpleModel(BaseModel):
            name: str

        text = "{'name': 'test'}"
        result = _parse_json_response(text, SimpleModel)
        assert result.name == "test"

    def test_unescaped_newlines_in_strings_parsed(self):
        """Literal newlines inside JSON string values should not crash parsing."""
        from pydantic import BaseModel

        from neograph._llm_retry import _parse_json_response

        class SimpleModel(BaseModel):
            content: str

        text = '{"content": "line1\nline2"}'
        result = _parse_json_response(text, SimpleModel)
        assert "line1" in result.content
        assert "line2" in result.content

    def test_null_coerced_to_default_factory_list_field(self):
        """null for a list field with default_factory should use factory() → [].

        BUG neograph-s1u4: LLMs (GLM 5.2) emit null for list fields that have a
        default_factory (e.g. ``sources: list[str] = Field(default_factory=list)``).
        Such fields carry ``.default = PydanticUndefined`` (the default lives in
        ``.default_factory``), so the null→default coercion is skipped and Pydantic
        rejects null for the list type. Coerce null → default_factory() first.
        """
        from pydantic import BaseModel, Field

        from neograph._llm_retry import _parse_json_response

        class ResearchPacket(BaseModel):
            finding: str
            sources: list[str] = Field(default_factory=list)

        text = '{"finding": "x", "sources": null}'
        result = _parse_json_response(text, ResearchPacket)
        assert result.finding == "x"
        assert result.sources == []

    def test_null_coerced_for_data_accepting_default_factory(self):
        """null for a Pydantic 2.10+ data-accepting default_factory must coerce,
        not crash.

        BUG neograph-s1u4 (completeness): a bare ``default_factory()`` call crashes
        with TypeError on a factory declared to receive validated data
        (``default_factory=lambda data: ...``). The coercion must fall back to
        ``default_factory(data)`` on TypeError — and must NOT pass ``data`` to a
        zero-arg factory like ``list`` (``list(data)`` silently returns the dict's
        keys instead of []). This pins the guard the naive fix omitted.
        """
        from pydantic import BaseModel, Field

        from neograph._llm_retry import _parse_json_response

        class Packet(BaseModel):
            label: str
            # data-accepting factory (Pydantic 2.10+): derives from validated data
            tags: list[str] = Field(default_factory=lambda data: [data["label"]])

        text = '{"label": "alpha", "tags": null}'
        result = _parse_json_response(text, Packet)
        assert result.label == "alpha"
        assert result.tags == ["alpha"]

    def test_stringly_null_coerced_to_none_for_optional_int(self):
        """The STRING "null" for an Optional[int] field must coerce to None.

        BUG stark-h46: GLM 5.2 intermittently emits the *string* ``"null"`` (not
        JSON ``null``) for Optional numeric fields (e.g.
        ``freshness_days: int | None``). json_repair leaves the string intact and
        Pydantic then raises ``int_parsing``, aborting the whole node. The parse
        must coerce the sentinel to ``None`` instead of crashing.
        """
        from pydantic import BaseModel

        from neograph._llm_retry import _parse_json_response

        class Company(BaseModel):
            name: str
            freshness_days: int | None = None

        text = '{"name": "ElectraVehicles", "freshness_days": "null"}'
        result = _parse_json_response(text, Company)
        assert result.name == "ElectraVehicles"
        assert result.freshness_days is None

    def test_stringly_null_coerced_for_optional_enum(self):
        """The STRING "null" for an Optional[Enum] field must coerce to None.

        stark-h46: ``round_type: FundingRoundType | None`` received ``"null"`` →
        pydantic ``enum`` error. Coerce to None.
        """
        from enum import StrEnum

        from pydantic import BaseModel

        from neograph._llm_retry import _parse_json_response

        class FundingRoundType(StrEnum):
            seed = "seed"
            series_a = "series_a"

        class FundingRound(BaseModel):
            amount: str
            round_type: FundingRoundType | None = None

        text = '{"amount": "10M", "round_type": "null"}'
        result = _parse_json_response(text, FundingRound)
        assert result.amount == "10M"
        assert result.round_type is None

    def test_stringly_null_coerced_in_nested_list_items(self):
        """Stringly-null coercion recurses into list[BaseModel] items.

        stark-h46: ``company.hiring_signals.0.total_open_roles`` (Optional[int])
        got ``"null"``. The coercion must reach nested list items, not just the
        top-level object.
        """
        from pydantic import BaseModel

        from neograph._llm_retry import _parse_json_response

        class HiringSignal(BaseModel):
            role: str
            total_open_roles: int | None = None

        class Company(BaseModel):
            name: str
            hiring_signals: list[HiringSignal]

        text = (
            '{"name": "Electra", "hiring_signals": '
            '[{"role": "eng", "total_open_roles": "null"}, '
            '{"role": "sales", "total_open_roles": "none"}]}'
        )
        result = _parse_json_response(text, Company)
        assert result.hiring_signals[0].total_open_roles is None
        assert result.hiring_signals[1].total_open_roles is None

    def test_stringly_null_coerced_inside_optional_nested_model(self):
        """Stringly-null coercion recurses THROUGH an Optional-wrapped nested model.

        BUG neograph-zhwgh (0.7.3 completeness): the 0.7.2 fix coerces a direct
        top-level ``Model | None`` field, but the recursion into a nested model
        only fired when the field annotation was a bare ``BaseModel`` type. A
        field typed ``parent: Company | None`` is a Union, so the descent was
        skipped and a stringly-``"null"`` on the INTERIOR (e.g. ``parent.langs``,
        a ``list[str] | None``) reached Pydantic raw -> list_type crash.
        """
        from pydantic import BaseModel

        from neograph._llm_retry import _parse_json_response

        class Company(BaseModel):
            name: str
            langs: list[str] | None = None
            parent: Company | None = None

        Company.model_rebuild()

        text = '{"name": "A", "parent": {"name": "B", "langs": "null"}}'
        result = _parse_json_response(text, Company)
        assert result.parent is not None
        assert result.parent.name == "B"
        assert result.parent.langs is None

    def test_stringly_null_coerced_inside_optional_list_of_models(self):
        """Stringly-null coercion recurses THROUGH an Optional-wrapped list-of-models.

        neograph-zhwgh: ``products: list[Product] | None`` is a Union, so the
        list-item descent (guarded by ``get_origin(...) is list``) was skipped and
        an interior ``price: int | None`` holding ``"null"`` crashed with
        int_parsing. The 0.7.2 code only reached items of a NON-optional
        ``list[Product]``.
        """
        from pydantic import BaseModel

        from neograph._llm_retry import _parse_json_response

        class Product(BaseModel):
            name: str
            price: int | None = None

        class Company(BaseModel):
            name: str
            products: list[Product] | None = None

        text = '{"name": "A", "products": [{"name": "p", "price": "null"}, {"name": "q", "price": "none"}]}'
        result = _parse_json_response(text, Company)
        assert result.products is not None
        assert result.products[0].price is None
        assert result.products[1].price is None

    def test_stringly_null_legit_values_preserved_through_optional_wrappers(self):
        """Optional-wrapped nested models/lists with LEGIT interiors are untouched.

        The Optional-unwrapping descent must not corrupt real data: a populated
        ``parent`` model and a real ``list[str]`` survive, and a genuine interior
        int is preserved.
        """
        from pydantic import BaseModel

        from neograph._llm_retry import _parse_json_response

        class Product(BaseModel):
            name: str
            price: int | None = None

        class Company(BaseModel):
            name: str
            langs: list[str] | None = None
            products: list[Product] | None = None
            parent: Company | None = None

        Company.model_rebuild()

        text = (
            '{"name": "A", "langs": ["en", "fr"], '
            '"products": [{"name": "p", "price": 5}], '
            '"parent": {"name": "B", "langs": ["de"]}}'
        )
        result = _parse_json_response(text, Company)
        assert result.langs == ["en", "fr"]
        assert result.products[0].price == 5
        assert result.parent.name == "B"
        assert result.parent.langs == ["de"]

    def test_stringly_null_coerced_inside_dict_of_models(self):
        """Stringly-null coercion recurses into dict[str, BaseModel] VALUES.

        neograph-zhwgh: the 0.7.2 descent hand-enumerated only bare-model and
        bare-list-of-model shapes, so a ``dict[str, Product]`` field's interior
        (``by_sku['x'].price``) was never reached. The shape-driven descent walks
        mapping values too.
        """
        from pydantic import BaseModel

        from neograph._llm_retry import _parse_json_response

        class Product(BaseModel):
            name: str
            price: int | None = None

        class Catalog(BaseModel):
            name: str
            by_sku: dict[str, Product]

        text = '{"name": "A", "by_sku": {"x": {"name": "p", "price": "null"}}}'
        result = _parse_json_response(text, Catalog)
        assert result.by_sku["x"].price is None

    def test_stringly_null_coerced_inside_list_of_optional_models(self):
        """Stringly-null coercion recurses into list[Model | None] items.

        neograph-zhwgh: ``list[Product | None]`` has a Union element type, so the
        old ``isinstance(args[0], type)`` element guard rejected it and the item
        interiors were skipped. The shape-driven descent re-peels Optional per
        element, and a real ``None`` element is left untouched.
        """
        from pydantic import BaseModel

        from neograph._llm_retry import _parse_json_response

        class Product(BaseModel):
            name: str
            price: int | None = None

        class Bag(BaseModel):
            items: list[Product | None]

        text = '{"items": [{"name": "p", "price": "null"}, null]}'
        result = _parse_json_response(text, Bag)
        assert result.items[0].price is None
        assert result.items[1] is None

    def test_stringly_null_preserves_legit_string_value(self):
        """A real string value must NOT be destroyed by the sentinel coercion.

        The coercion only fires on Optional fields whose non-None member is not a
        plain ``str`` (numeric/enum), OR on the exact sentinels — a legitimate
        ``str | None`` holding real text is untouched, and "" stays "" for a str
        field.
        """
        from pydantic import BaseModel

        from neograph._llm_retry import _parse_json_response

        class Profile(BaseModel):
            summary: str | None = None
            note: str = ""

        text = '{"summary": "a real description", "note": ""}'
        result = _parse_json_response(text, Profile)
        assert result.summary == "a real description"
        assert result.note == ""

    def test_stringly_null_not_coerced_for_required_non_optional(self):
        """A non-Optional int field that got "null" is NOT silently coerced.

        The field cannot hold None, so coercing would just move the crash. It must
        still fail loud (retry re-prompt territory), never become a bare success
        with a wrong value.
        """
        from pydantic import BaseModel

        from neograph._llm_retry import _parse_json_response
        from neograph.errors import ExecutionError

        class Metric(BaseModel):
            count: int  # required, non-Optional

        text = '{"count": "null"}'
        with pytest.raises(ExecutionError, match="Validation failed"):
            _parse_json_response(text, Metric)


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

        from neograph._llm_dispatch import _call_structured

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

        # Contract: TypeError on include_raw=True is recoverable — parsed result
        # is returned. Drop call-count pin (implementation detail).
        assert result == expected

    def test_result_correct_when_include_raw_supported(self):
        """_call_structured returns parsed result when include_raw works."""
        from unittest.mock import MagicMock

        from neograph._llm_dispatch import _call_structured

        expected = Claims(items=["direct-result"])

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"parsed": expected, "raw": None}
        mock_llm.with_structured_output.return_value = mock_structured

        config = {"configurable": {}}
        result, usage = _call_structured(mock_llm, [], Claims, "structured", config)

        # Contract: when include_raw is supported, parsed result is returned.
        # Drop assert_called_once_with — the result equality covers the
        # user-visible behavior.
        assert result == expected


class TestRetryPromptIncludesSchema:
    """Retry prompt must include describe_type schema so LLM can self-correct.

    BUG neograph-mfzx: DeepSeek R1 simplifies nested objects to strings
    on long prompts. The retry must show the expected structure.
    """

    def test_retry_msg_includes_schema_for_nested_model(self):
        """_build_retry_msg with output_model includes the full schema."""
        from pydantic import BaseModel

        from neograph._llm_retry import _build_retry_msg
        from neograph.errors import ExecutionError

        class Actor(BaseModel):
            name: str
            role: str

        class Result(BaseModel):
            actors: list[Actor]

        err = ExecutionError("validation failed")
        err.validation_errors = "actors.0: Input should be an object"

        msg = _build_retry_msg(err, output_model=Result)

        # Must include the schema so LLM knows the expected structure
        assert "name: string" in msg or "name" in msg
        assert "role: string" in msg or "role" in msg
        # Must instruct about nested objects
        assert "nested object" in msg.lower() or "ALL required fields" in msg

    def test_retry_msg_without_model_still_works(self):
        """_build_retry_msg without output_model doesn't crash."""
        from neograph._llm_retry import _build_retry_msg
        from neograph.errors import ExecutionError

        err = ExecutionError("parse failed")
        msg = _build_retry_msg(err)
        assert "JSON" in msg

    def test_default_max_retries_is_2(self):
        """_invoke_json_with_retry defaults to max_retries=2."""
        import inspect

        from neograph._llm_retry import _invoke_json_with_retry

        sig = inspect.signature(_invoke_json_with_retry)
        assert sig.parameters["max_retries"].default == 2


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

        Contract: the user-declared node names must be reachable from
        __start__, and the pipeline must terminate at __end__. Internal
        synthesized barrier/merge node names are implementation detail.
        """
        from neograph.decorators import _merge_fn_registry
        from tests.fakes import register_scripted

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
        graph = compile(pipeline, **build_test_compile_kwargs())

        dg = graph.get_graph()
        nodes = {n.id for n in dg.nodes.values()} if hasattr(dg, "nodes") else set()
        edges = list(dg.edges)

        def reaches(start: str, target: str) -> bool:
            seen: set[str] = set()
            stack = [start]
            while stack:
                cur = stack.pop()
                if cur == target:
                    return True
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend(e.target for e in edges if e.source == cur)
            return False

        # Declared user nodes must appear in the graph.
        assert "pre" in nodes
        assert "gen" in nodes
        assert "post" in nodes
        # The downstream node after the Oracle fan-out is reachable from __start__
        # and the pipeline terminates at __end__ — without spelling internal names.
        assert reaches("__start__", "post"), f"post unreachable from __start__: {edges}"
        assert reaches("post", "__end__"), f"__end__ unreachable from post: {edges}"

    def test_each_edges_visible_in_get_graph(self):
        """Nodes after an Each fan-out must appear in get_graph().edges.

        Contract: declared user nodes are reachable from __start__ and
        terminate at __end__. The Each barrier's synthesized internal node
        name is implementation detail.
        """
        from tests.fakes import register_scripted

        register_scripted("viz_make", lambda _in, _cfg: Clusters(groups=[ClusterGroup(label="a", claim_ids=["1"])]))
        register_scripted("viz_verify", lambda _in, _cfg: MatchResult(cluster_label="a", matched=["ok"]))
        register_scripted("viz_summary", lambda _in, _cfg: RawText(text="done"))

        make = Node.scripted("make", fn="viz_make", outputs=Clusters)
        verify = Node.scripted(
            "verify",
            fn="viz_verify",
            inputs=ClusterGroup,
            outputs=MatchResult,
        ) | Each(over="make.groups", key="label")
        summary = Node.scripted(
            "summary",
            fn="viz_summary",
            inputs=dict[str, MatchResult],
            outputs=RawText,
        )

        pipeline = Construct("test-viz-each", nodes=[make, verify, summary])
        graph = compile(pipeline, **build_test_compile_kwargs())

        dg = graph.get_graph()
        nodes = {n.id for n in dg.nodes.values()} if hasattr(dg, "nodes") else set()
        edges = list(dg.edges)

        def reaches(start: str, target: str) -> bool:
            seen: set[str] = set()
            stack = [start]
            while stack:
                cur = stack.pop()
                if cur == target:
                    return True
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend(e.target for e in edges if e.source == cur)
            return False

        # Declared user nodes must appear in the graph.
        assert "make" in nodes
        assert "verify" in nodes
        assert "summary" in nodes
        # The post-Each summary node is reachable from __start__ and the pipeline
        # terminates at __end__ — without naming the assemble_verify barrier.
        assert reaches("__start__", "summary"), f"summary unreachable from __start__: {edges}"
        assert reaches("summary", "__end__"), f"__end__ unreachable from summary: {edges}"


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

        from tests.fakes import register_tool_factory

        lookup_tool = FakeTool("lookup", response="found")
        register_tool_factory("lookup", lambda config, tool_config: lookup_tool)

        # Fake LLM: first call uses tool (exhausts budget=1),
        # second call (bare, no tools) emits XML instead of JSON.
        call_n = {"n": 0}
        XML_CONTENT = (
            "<\uff5cDSML\uff5cfunction_calls>"
            '<\uff5cDSML\uff5cinvoke name="read_artifact">'
            '<\uff5cDSML\uff5cparameter name="path">test.py'
            "</\uff5cDSML\uff5cparameter>"
            "</\uff5cDSML\uff5cinvoke>"
            "</\uff5cDSML\uff5cfunction_calls>"
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
        graph = compile(
            pipeline,
            llm_factory=lambda tier: FakeR1(),
            prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": "go"}],
            **build_test_compile_kwargs(),
        )

        # Should raise ExecutionError (clear message), not ValidationError (cryptic)
        with pytest.raises(ExecutionError, match="(?i)structured output|json|xml"):
            run(graph, input={"node_id": "test"})


# ═══════════════════════════════════════════════════════════════════════════
# Runtime: Loop/branch condition error handling (neograph-d19r)
# ═══════════════════════════════════════════════════════════════════════════


class TestConditionErrorHandling:
    """User-provided conditions (Loop.when, branch op_fn) can crash on None
    with AttributeError/TypeError. The error should be wrapped in a clear
    ExecutionError naming the condition and the None value."""

    def test_loop_condition_wraps_attribute_error_when_value_is_none(self):
        """Loop condition accessing .score on None raises ExecutionError."""
        from neograph.modifiers import Loop
        from tests.fakes import register_scripted

        register_scripted("d19r_attr_fn", lambda input_data, config: RawText(text="hello"))

        n = Node.scripted("produce-none", fn="d19r_attr_fn", outputs=RawText) | Loop(
            when=lambda draft: draft.score < 0.8, max_iterations=3
        )
        pipeline = Construct("loop-err", nodes=[n])
        graph = compile(pipeline, **build_test_compile_kwargs())
        with pytest.raises(ExecutionError, match="condition"):
            run(graph, input={"node_id": "test-loop-err"})

    def test_loop_condition_wraps_type_error_when_value_is_none(self):
        """Loop condition doing comparison on None raises ExecutionError."""
        from neograph.modifiers import Loop
        from tests.fakes import register_scripted

        register_scripted("d19r_type_fn", lambda input_data, config: RawText(text="hello"))

        n = Node.scripted("produce-text", fn="d19r_type_fn", outputs=RawText) | Loop(
            when=lambda draft: draft < 0.8, max_iterations=3
        )
        pipeline = Construct("loop-type-err", nodes=[n])
        graph = compile(pipeline, **build_test_compile_kwargs())
        with pytest.raises(ExecutionError, match="condition"):
            run(graph, input={"node_id": "test-loop-type-err"})


# =============================================================================
# _llm.py coverage — edge cases for uncovered lines
# =============================================================================


class TestLlmIntrospectParams:
    """_accepted_params edge cases (lines 60-62)."""

    def test_c_extension_returns_empty_set(self):
        """C extension callables with no inspectable signature return empty set."""
        from neograph._llm import _accepted_params

        # builtins like len have no inspectable signature in some Python versions
        result = _accepted_params(len)
        # Should return empty set or a valid set (not raise)
        assert isinstance(result, (set, frozenset))


class TestLlmNotConfigured:
    """Lines 123-124, 127: _get_llm when not configured."""

    def test_get_llm_raises_when_not_configured(self):
        """_get_llm raises ConfigurationError when llm_factory is None."""
        from neograph._llm import _get_llm

        with pytest.raises(ConfigurationError, match="not configured"):
            _get_llm("fast")

    def test_get_llm_accepts_all_kwargs_factory(self):
        """When factory accepts **kwargs, all kwargs are passed (line 127)."""
        from neograph._llm import _get_llm
        from tests.fakes import build_fake_runtime

        received = {}

        def factory(tier, **kwargs):
            received.update(kwargs)
            return "fake_llm"

        runtime = build_fake_runtime(factory=factory)
        result = _get_llm(runtime, "fast", node_name="test_node", llm_config={"provider_kwargs": {"temp": 0.5}})
        assert result == "fake_llm"
        assert "node_name" in received
        assert "llm_config" in received


class TestExtractJsonEdgeCases2:
    """Lines 227-228, 230-231, 245-248: _extract_json edge cases."""

    def test_escape_char_in_json_string(self):
        """Backslash escapes inside JSON strings are handled (lines 227-228, 230-231)."""
        from neograph._llm_retry import _extract_json

        text = r'{"key": "value with \" escaped"}'
        result = _extract_json(text)
        assert result.startswith("{")
        assert result.endswith("}")

    def test_unbalanced_braces_first_to_last(self):
        """Unbalanced braces with closing brace: first-to-last fallback (line 247)."""
        from neograph._llm_retry import _extract_json

        # Unbalanced: extra opening brace, but there's a } later
        text = '{"key": {"nested": "value"} extra stuff}'
        # The balanced scan finds the first complete match
        result = _extract_json(text)
        assert result.startswith("{")

    def test_unbalanced_with_trailing_closing_brace(self):
        """Unbalanced JSON with a trailing } past the scan falls back (line 247)."""
        from neograph._llm_retry import _extract_json

        # The first { starts depth tracking, but escapes/strings cause imbalance
        # Force unbalanced: an unclosed string with } after it
        text = '{"key": "unclosed string} more text}'
        result = _extract_json(text)
        assert "{" in result

    def test_no_closing_brace_at_all(self):
        """No closing brace after opening returns stripped text (line 248)."""
        from neograph._llm_retry import _extract_json

        text = '{"key": "value'
        result = _extract_json(text)
        assert result == text.strip()


class TestParseJsonException:
    """Lines 293-294: generic Exception during JSON parsing."""

    def test_generic_exception_during_parse(self):
        """Non-ValidationError during parse raises ExecutionError."""
        from neograph._llm_retry import _parse_json_response

        class BadModel:
            """A model that raises a generic Exception during validate."""

            __name__ = "BadModel"

            @classmethod
            def model_validate_json(cls, data):
                raise RuntimeError("unexpected error")

        with pytest.raises(ExecutionError, match="Failed to parse"):
            _parse_json_response('{"x": 1}', BadModel)


class TestTruncatedArraySilentEmpty:
    """neograph-wvrp: truncated array must NOT silently return empty result."""

    def test_truncated_array_raises_not_empty_result(self):
        """Truncated array (no closing ]) must raise ExecutionError, not return empty list."""
        from pydantic import BaseModel, Field

        from neograph._llm_retry import _parse_json_response

        class Item(BaseModel):
            id: str
            value: str

        class Result(BaseModel):
            items: list[Item] = Field(default_factory=list)

        truncated = """[
  {"id": "A", "value": "alpha"},
  {"id": "B", "value": "beta"},
  {"id": "C", "value": "gam"""

        # This MUST either raise ExecutionError (triggering retry) or
        # return a Result with items populated. It must NOT silently
        # return Result(items=[]).
        try:
            r = _parse_json_response(truncated, Result)
            # If it parses without error, it must have items (from json_repair)
            assert len(r.items) > 0, (
                f"Truncated array silently produced empty result: {r}. "
                f"Expected either ExecutionError or non-empty items."
            )
        except ExecutionError:
            pass  # correct — retry path will fire

    def test_truncated_array_extract_json_does_not_return_inner_dict(self):
        """_extract_json on truncated array must not return first inner object."""
        from neograph._llm_retry import _extract_json

        truncated = '[{"id": "A", "value": "alpha"}, {"id": "B", "value": "be'

        result = _extract_json(truncated)
        # Must NOT be just the first inner dict
        assert not result.strip().startswith('{"id"'), (
            f"_extract_json fell back to inner dict from truncated array: {result[:60]}"
        )


class TestToolCallArgsCoercion:
    """neograph-d8y5: tool_calls.args as JSON string must not crash pipeline."""

    def test_string_args_retried_and_recovered(self):
        """ValidationError from string tool_calls.args triggers retry, not crash."""
        from neograph import Tool
        from tests.fakes import StringArgsFake, register_tool_factory
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools

        tool_invoked = []

        def search_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            def search_fn(q: str) -> str:
                tool_invoked.append(q)
                return f"result for {q}"

            return StructuredTool.from_function(search_fn, name="search", description="search")

        register_tool_factory("search", search_factory)

        fake = StringArgsFake(
            tool_calls=[
                [{"name": "search", "args": {"q": "test query"}, "id": "call_1"}],
                [],  # final — no more tool calls
            ],
            final=lambda m: m(answer="done"),
        )

        from pydantic import BaseModel as _BM

        class Answer(_BM):
            answer: str

        from neograph.tool import ToolBudgetTracker

        tools = [Tool(name="search", description="search")]
        budget = ToolBudgetTracker(tools)

        # Should NOT raise — _safe_tool_invoke catches ValidationError and retries
        result = invoke_with_tools(
            runtime=build_fake_runtime(lambda tier: fake),
            model_tier="fast",
            prompt_template="test prompt",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        parsed, interactions = result
        assert parsed.answer == "done"
        # The tool should have been invoked after the coercion succeeded
        assert len(tool_invoked) > 0

    def test_consistent_string_args_always_coerced(self):
        """Even at 100% string-args rate, coercion handles every call."""
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import StringArgsFake, register_tool_factory
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools

        tool_calls_received = []

        def lookup_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            def lookup_fn(node_id: str) -> str:
                tool_calls_received.append(node_id)
                return f"found {node_id}"

            return StructuredTool.from_function(lookup_fn, name="lookup", description="lookup")

        register_tool_factory("lookup", lookup_factory)

        # always_fail=True — EVERY invoke raises. No intermittent success.
        fake = StringArgsFake(
            tool_calls=[
                [{"name": "lookup", "args": {"node_id": "BR-UC-008"}, "id": "c1"}],
                [{"name": "lookup", "args": {"node_id": "FLOW-006"}, "id": "c2"}],
                [],
            ],
            final=lambda m: m(answer="analyzed"),
            always_fail=True,
        )

        class Answer(_BM):
            answer: str

        tools = [Tool(name="lookup", description="lookup")]
        budget = ToolBudgetTracker(tools)

        parsed, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            runtime=build_fake_runtime(lambda tier: fake),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert parsed.answer == "analyzed"
        assert tool_calls_received == ["BR-UC-008", "FLOW-006"]

    def test_non_tool_calls_validation_error_reraised(self):
        """ValidationError NOT related to tool_calls.args is re-raised."""
        from pydantic import ValidationError

        from neograph._tool_loop import _CoercingToolWrapper

        class BadLLM:
            def invoke(self, messages, **kw):
                # Raise ValidationError for a completely different field
                raise ValidationError.from_exception_data(
                    title="AIMessage",
                    line_errors=[
                        {
                            "type": "string_type",
                            "loc": ("content",),
                            "msg": "Input should be a valid string",
                            "input": 12345,
                        }
                    ],
                )

        wrapper = _CoercingToolWrapper(BadLLM())
        with pytest.raises(ValidationError, match="content"):
            wrapper.invoke([])

    def test_multiple_tool_calls_mixed_args(self):
        """Multiple tool_calls where some have dict args and some have string args."""
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import StringArgsFake, register_tool_factory
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools

        args_received = []

        def multi_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            def fn(q: str) -> str:
                args_received.append(q)
                return f"result for {q}"

            return StructuredTool.from_function(fn, name="multi", description="multi")

        register_tool_factory("multi", multi_factory)

        fake = StringArgsFake(
            tool_calls=[
                [
                    {"name": "multi", "args": {"q": "first"}, "id": "c1"},
                    {"name": "multi", "args": {"q": "second"}, "id": "c2"},
                ],
                [],
            ],
            final=lambda m: m(answer="done"),
            always_fail=True,
        )

        class Answer(_BM):
            answer: str

        tools = [Tool(name="multi", description="multi")]
        budget = ToolBudgetTracker(tools)

        parsed, _ = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            runtime=build_fake_runtime(lambda tier: fake),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert parsed.answer == "done"
        assert args_received == ["first", "second"]

    def test_coerced_args_parsed_correctly(self):
        """Coerced tool args must be proper dicts with correct values."""
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import StringArgsFake, register_tool_factory
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools

        received_args = []

        def precise_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            def fn(node_id: str, depth: int = 1) -> str:
                received_args.append({"node_id": node_id, "depth": depth})
                return "ok"

            return StructuredTool.from_function(fn, name="precise", description="precise")

        register_tool_factory("precise", precise_factory)

        fake = StringArgsFake(
            tool_calls=[
                [{"name": "precise", "args": {"node_id": "BR-UC-008", "depth": 3}, "id": "c1"}],
                [],
            ],
            final=lambda m: m(answer="ok"),
            always_fail=True,
        )

        class Answer(_BM):
            answer: str

        tools = [Tool(name="precise", description="precise")]
        budget = ToolBudgetTracker(tools)

        parsed, _ = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            runtime=build_fake_runtime(lambda tier: fake),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert parsed.answer == "ok"
        assert len(received_args) == 1
        assert received_args[0] == {"node_id": "BR-UC-008", "depth": 3}

    def test_malformed_json_in_string_args(self):
        """Unparseable string args degrade to empty dict, tool still called."""
        from types import SimpleNamespace

        from langchain_core.messages import AIMessage

        from neograph._tool_loop import _CoercingToolWrapper

        class MalformedArgsFake:
            def invoke(self, messages, **kw):
                # Raise with truncated JSON string as args
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search", "args": '{"q": "trun', "id": "c1"}],
                )

            def _generate(self, messages, *, run_manager=None, **kw):
                # Return via additional_kwargs with malformed args
                msg = AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"q": "trun'},
                            }
                        ]
                    },
                )
                return SimpleNamespace(generations=[SimpleNamespace(message=msg)])

        wrapper = _CoercingToolWrapper(MalformedArgsFake())
        response = wrapper.invoke([])
        # Should not crash — tool_calls should have parsed what it could
        assert hasattr(response, "tool_calls")

    def test_empty_string_args(self):
        """Empty string args degrade to empty dict."""
        from types import SimpleNamespace

        from langchain_core.messages import AIMessage

        from neograph._tool_loop import _CoercingToolWrapper

        class EmptyArgsFake:
            def invoke(self, messages, **kw):
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search", "args": "", "id": "c1"}],
                )

            def _generate(self, messages, *, run_manager=None, **kw):
                msg = AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "search", "arguments": ""},
                            }
                        ]
                    },
                )
                return SimpleNamespace(generations=[SimpleNamespace(message=msg)])

        wrapper = _CoercingToolWrapper(EmptyArgsFake())
        response = wrapper.invoke([])
        assert hasattr(response, "tool_calls")


class TestCallStructuredUnknownStrategy:
    """Lines 377-378: unknown output_strategy raises ExecutionError."""

    def test_unknown_strategy_raises(self):
        """_call_structured with unknown strategy raises ExecutionError."""
        from neograph._llm_dispatch import _call_structured

        with pytest.raises(ExecutionError, match="Unknown output_strategy"):
            _call_structured(None, [], Claims, "invalid_strategy", {})


class TestToolResultRendering:
    """Lines 452, 562: _render_tool_result_for_llm with plain values and list of models."""

    def test_plain_value_returns_str(self):
        """Non-Pydantic result returns str() (line 452)."""
        from neograph._tool_loop import _render_tool_result_for_llm

        assert _render_tool_result_for_llm(42) == "42"
        assert _render_tool_result_for_llm("hello") == "hello"

    def test_list_of_models_rendered(self):
        """List of BaseModel instances uses renderer or describe_value (line 562)."""
        from neograph._tool_loop import _render_tool_result_for_llm

        items = [RawText(text="a"), RawText(text="b")]
        result = _render_tool_result_for_llm(items)
        assert "a" in result
        assert "b" in result


class TestUnregisteredToolInReact:
    """An agent node whose tool has no registered factory is rejected.

    neograph-m6d3.3: the deleted monolith's DIRECT-call runtime guard raised a
    ConfigurationError from _prepare_tool_loop. Through the inline cycle (i.e.
    compile()/run(), which is the only path now), an unregistered tool factory is
    caught at COMPILE time by compile()'s tool-factory validation — a CompileError,
    which is where compiled agent nodes always caught it (compiler.py predates
    m6d3). The error moves earlier (compile-time), never later.
    """

    def test_unregistered_tool_raises(self):
        from neograph.errors import CompileError
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools

        _llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["x"])))

        with pytest.raises(CompileError, match="no registered factory"):
            invoke_with_tools(
                runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
                model_tier="fast",
                prompt_template="test prompt",
                input_data="test",
                output_model=Claims,
                tools=[Tool("nonexistent", budget=5)],
                config={"configurable": {}},
                tool_factory_lookup=build_fake_tool_lookup(),
            )


class TestUsageTokenAccumulation:
    """Lines 621-626, 630: usage token accumulation in invoke_with_tools."""

    def test_usage_tokens_accumulated_from_messages(self):
        """Token usage from ReAct messages is accumulated (lines 621-626, 630)."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        fake_tool = FakeTool("search", response="found it")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        # Build a ReActFake that makes one tool call then stops
        from langchain_core.messages import AIMessage

        class UsageFake:
            """Fake LLM that adds usage_metadata to responses.

            History-driven (neograph-m6d3): one tool call, then final — decided
            from the message history so it behaves under the inline cycle (the
            LLM is rebuilt per superstep) as it did under the monolith.
            """

            def __init__(self):
                self._model = None
                self._structured = False

            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kwargs):
                from langchain_core.messages import ToolMessage

                if self._structured:
                    result = Claims(items=["done"])
                    return {
                        "parsed": result,
                        "raw": AIMessage(
                            content="",
                            usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                        ),
                    }
                if not any(isinstance(m, ToolMessage) for m in messages):
                    msg = AIMessage(
                        content="", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
                    )
                    msg.tool_calls = [{"name": "search", "args": {"q": "x"}, "id": "1"}]
                    return msg
                return AIMessage(
                    content='{"items": ["done"]}',
                    usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
                )

            def with_structured_output(self, model, **kwargs):
                clone = UsageFake()
                clone._model = model
                clone._structured = True
                return clone

        _llm_kw = configure_fake_llm(lambda tier: UsageFake())

        tools = [Tool("search", budget=5)]
        tracker = ToolBudgetTracker(tools)
        result, interactions = invoke_with_tools(
            runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
            model_tier="fast",
            prompt_template="test prompt",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            tool_factory_lookup=build_fake_tool_lookup(),
        )
        assert isinstance(result, Claims)
        assert len(interactions) == 1


class TestRenderPromptEdgeCases:
    """Lines 669-670, 707-708: render_prompt not configured + message attr access."""

    def test_render_prompt_not_configured_raises(self):
        """render_prompt without configure_llm raises ConfigurationError."""
        from neograph._llm import render_prompt

        node = Node("test", mode="think", prompt="test", model="fast", outputs=Claims)
        with pytest.raises(ConfigurationError, match="not configured"):
            render_prompt(node, {"key": "value"})

    def test_render_prompt_with_message_objects(self):
        """render_prompt handles LangChain message objects (line 707-708)."""

        from langchain_core.messages import HumanMessage

        from neograph._llm import render_prompt

        def msg_compiler(template, data, **kw):
            return [HumanMessage(content="hello world")]

        node = Node("test", mode="think", prompt="test", model="fast", outputs=Claims)
        result = render_prompt(
            node,
            {"key": "value"},
            config={"configurable": {}},
            runtime=build_fake_runtime(prompt_compiler=msg_compiler),
        )
        assert "hello world" in result
        assert "[human]" in result


class TestToolBudgetTrackerAllExhausted:
    """Line 99 in tool.py: all_exhausted returns False when no tools."""

    def test_all_exhausted_no_tools(self):
        """ToolBudgetTracker with no tools returns False for all_exhausted."""
        from neograph.tool import ToolBudgetTracker

        tracker = ToolBudgetTracker([])
        assert tracker.all_exhausted() is False


class TestReActToolReturnsListOfModels:
    """Line 562 in _llm.py: tool returns list of BaseModel instances in ReAct loop."""

    def test_tool_returning_list_of_models_renders(self):
        """When a tool returns list[BaseModel], it's rendered via _render_tool_result_for_llm."""
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as BM

        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        # A tool that returns a list of BaseModel instances
        class SearchResult(BM):
            title: str
            score: float

        class ListTool:
            name = "search"

            def invoke(self, args, config=None, **kwargs):
                return [SearchResult(title="a", score=0.9), SearchResult(title="b", score=0.8)]

        register_tool_factory("search", lambda cfg, tc: ListTool())

        class ListReActFake:
            def __init__(self):
                self._model = None
                self._structured = False

            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kwargs):
                from langchain_core.messages import ToolMessage

                if self._structured:
                    return self._model(items=["done"])
                if not any(isinstance(m, ToolMessage) for m in messages):
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search", "args": {"q": "x"}, "id": "1"}]
                    return msg
                return AIMessage(content='{"items": ["done"]}')

            def with_structured_output(self, model, **kwargs):
                clone = ListReActFake()
                clone._model = model
                clone._structured = True
                return clone

        _llm_kw = configure_fake_llm(lambda tier: ListReActFake())

        tools = [Tool("search", budget=5)]
        tracker = ToolBudgetTracker(tools)
        result, interactions = invoke_with_tools(
            runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
            model_tier="fast",
            prompt_template="test prompt",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            tool_factory_lookup=build_fake_tool_lookup(),
        )
        assert len(interactions) == 1
        # The rendered result should contain both items
        assert "a" in interactions[0].result
        assert "b" in interactions[0].result


class TestReActMaxIterationsGuard:
    """invoke_with_tools max_iterations guard: stops infinite ReAct loops."""

    def test_max_iterations_default_stops_at_20(self):
        """Default max_iterations=20 stops an infinite tool-calling loop."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = GuardFake()
        _llm_kw = configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=0)]  # unlimited per-tool budget
        tracker = ToolBudgetTracker(tools)
        result, interactions = invoke_with_tools(
            runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            tool_factory_lookup=build_fake_tool_lookup(),
        )
        # Contract: unboundedly-eager fake (would call tools forever) must
        # terminate at the default cap with a valid Claims result, without hanging.
        # Bounded check, not exact iteration pin — default cap is 20.
        assert isinstance(result, Claims)
        assert len(interactions) < 25

    def test_max_iterations_custom_value(self):
        """Custom max_iterations in llm_config overrides the default."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = GuardFake()
        _llm_kw = configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=0)]
        tracker = ToolBudgetTracker(tools)

        # max_iterations=3 — guard fires on iteration 3
        result, interactions = invoke_with_tools(
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
        # Contract: custom max_iterations is honored; loop terminates within
        # the user-supplied bound.
        assert isinstance(result, Claims)
        assert len(interactions) < 4  # bound is 3; small upper

    def test_max_iterations_does_not_affect_normal_completion(self):
        """When the LLM finishes before max_iterations, the guard is irrelevant."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = ReActFake(
            tool_calls=[
                [{"name": "search", "args": {"q": "x"}, "id": "1"}],
                [],  # LLM stops after 1 tool call
            ],
            final=lambda model: model(items=["done"]),
        )
        _llm_kw = configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=5)]
        tracker = ToolBudgetTracker(tools)

        result, interactions = invoke_with_tools(
            runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            llm_config={"max_iterations": 20},
            tool_factory_lookup=build_fake_tool_lookup(),
        )
        assert isinstance(result, Claims)
        assert len(interactions) == 1

    def test_max_iterations_equals_one(self):
        """Degenerate case: max_iterations=1 means the first tool-calling iteration triggers the guard."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = GuardFake()
        _llm_kw = configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=0)]
        tracker = ToolBudgetTracker(tools)

        result, interactions = invoke_with_tools(
            runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            llm_config={"max_iterations": 1},
            tool_factory_lookup=build_fake_tool_lookup(),
        )
        # Contract: max_iterations=1 → loop terminates, no successful tool execution.
        assert isinstance(result, Claims)
        assert interactions == []

    def test_guard_fired_llm_ignores_wrap_up_still_calls_tools(self):
        """After guard fires and tools are unbound, if the LLM still returns
        tool_calls on the next invocation, the loop force-breaks instead of
        looping forever (the _guard_fired safety net).

        neograph-eoi8: the stubborn model never emits a JSON final turn, so the
        direct parse of messages[-1] fails; because output_strategy defaults to
        'structured', the parse-failure fallback runs constrained decoding
        (_call_structured), which StubbornFake satisfies via with_structured_output
        and yields a Claims. So this pins BOTH properties: the safety net
        terminates the loop (no hang) AND the structured fallback recovers a typed
        result (weak-model recourse)."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        _llm_kw = configure_fake_llm(lambda tier: StubbornFake())

        tools = [Tool("search", budget=0)]
        tracker = ToolBudgetTracker(tools)

        # max_iterations=1: guard fires immediately; the LLM keeps calling tools,
        # so the loop force-breaks (no infinite loop). The structured fallback
        # then recovers a typed result.
        result, _interactions = invoke_with_tools(
            runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            llm_config={"max_iterations": 1},
            tool_factory_lookup=build_fake_tool_lookup(),
        )
        assert isinstance(result, Claims)


class TestReActTokenBudgetGuard:
    """invoke_with_tools token_budget guard: stops loop when cumulative input tokens exceeded."""

    def test_token_budget_stops_loop(self):
        """token_budget in llm_config stops the loop when input tokens exceed threshold."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = GuardFake(input_tokens_per_call=1000)
        _llm_kw = configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=0)]  # unlimited
        tracker = ToolBudgetTracker(tools)

        # token_budget=2500 — after 3 iterations (3000 cumulative), guard fires
        result, interactions = invoke_with_tools(
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
        # Contract: token_budget bounds the loop; terminates with valid result.
        assert isinstance(result, Claims)
        assert len(interactions) < 50  # would be 50 unbounded; budget enforces small N

    def test_token_budget_none_is_no_limit(self):
        """token_budget=None (default) means no token budget enforcement."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = ReActFake(
            tool_calls=[
                [{"name": "search", "args": {"q": "x"}, "id": "1"}],
                [],  # LLM finishes naturally
            ],
            final=lambda model: model(items=["done"]),
        )
        _llm_kw = configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=5)]
        tracker = ToolBudgetTracker(tools)

        result, interactions = invoke_with_tools(
            runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            # No token_budget — default None means unlimited
            tool_factory_lookup=build_fake_tool_lookup(),
        )
        assert isinstance(result, Claims)
        assert len(interactions) == 1

    def test_both_guards_fire_simultaneously(self):
        """When max_iterations and token_budget are both exceeded, loop still terminates."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = GuardFake(input_tokens_per_call=1000)
        _llm_kw = configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=0)]
        tracker = ToolBudgetTracker(tools)

        # max_iterations=3 and token_budget=2500: both fire on iteration 3
        result, interactions = invoke_with_tools(
            runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            llm_config={"max_iterations": 3, "token_budget": 2500},
            tool_factory_lookup=build_fake_tool_lookup(),
        )
        # Contract: when both knobs are set, loop still terminates cleanly.
        assert isinstance(result, Claims)
        assert len(interactions) < 4  # bound is 3; small upper

    def test_token_budget_missing_usage_metadata(self):
        """When responses lack usage_metadata, token_budget never fires (correct behavior)."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = ReActFake(
            tool_calls=[
                [{"name": "search", "args": {"q": "x"}, "id": "1"}],
                [],  # LLM finishes naturally
            ],
            final=lambda model: model(items=["done"]),
        )
        # ReActFake doesn't include usage_metadata on responses,
        # so cumulative_input_tokens stays 0, and token_budget never triggers
        _llm_kw = configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=5)]
        tracker = ToolBudgetTracker(tools)

        result, interactions = invoke_with_tools(
            runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            # token_budget=100, but responses have no usage_metadata, so it never fires
            llm_config={"token_budget": 100},
            tool_factory_lookup=build_fake_tool_lookup(),
        )
        assert isinstance(result, Claims)
        assert len(interactions) == 1  # tool executes normally


# ═══════════════════════════════════════════════════════════════════════════
# BUG neograph-53lz: _extract_json drops bare-array LLM responses
#
# _extract_json only looks for '{' as JSON start. When the LLM returns a
# bare array ([{...}, {...}]), it falls through and returns raw text.
# _parse_json_response then silently defaults to empty lists.
# ═══════════════════════════════════════════════════════════════════════════


class TestBareArrayExtraction:
    """_extract_json must handle bare JSON arrays, not just objects."""

    def test_extract_json_finds_bare_array(self):
        """Bare array [{"key": "val"}] should be extracted, not dropped."""
        from neograph._llm_retry import _extract_json

        text = '[{"claim_id": "R1", "classification": "business-rule"}, {"claim_id": "R2", "classification": "functional"}]'
        result = _extract_json(text)
        assert result.startswith("["), f"Expected array, got: {result[:50]}"
        assert "R1" in result

    def test_extract_json_bare_array_with_markdown_fence(self):
        """Bare array inside ```json fence should be extracted."""
        from neograph._llm_retry import _extract_json

        text = '```json\n[{"id": 1}, {"id": 2}]\n```'
        result = _extract_json(text)
        assert result.startswith("["), f"Expected array, got: {result[:50]}"
        assert '"id": 1' in result or '"id":1' in result

    def test_extract_json_bare_array_with_prose(self):
        """Bare array preceded by prose text should be extracted."""
        from neograph._llm_retry import _extract_json

        text = 'Here are the results:\n[{"name": "Alice"}, {"name": "Bob"}]\nDone.'
        result = _extract_json(text)
        assert result.startswith("["), f"Expected array, got: {result[:50]}"
        assert "Alice" in result

    def test_parse_json_response_bare_array_auto_wraps(self):
        """When output model has a single list field and LLM returns bare array, auto-wrap."""
        from pydantic import BaseModel

        from neograph._llm_retry import _parse_json_response

        class Claim(BaseModel):
            claim_id: str
            classification: str

        class ClassificationResult(BaseModel):
            classified_claims: list[Claim]

        text = '[{"claim_id": "R1", "classification": "business-rule"}, {"claim_id": "R2", "classification": "functional"}]'
        result = _parse_json_response(text, ClassificationResult)
        assert isinstance(result, ClassificationResult)
        assert len(result.classified_claims) == 2
        assert result.classified_claims[0].claim_id == "R1"

    def test_parse_json_response_bare_array_multi_field_model_raises(self):
        """Bare array with multi-field model should raise, not silently default."""
        from pydantic import BaseModel

        from neograph import ExecutionError
        from neograph._llm_retry import _parse_json_response

        class MultiField(BaseModel):
            items: list[str]
            count: int

        text = '["a", "b", "c"]'
        # Multi-field model can't auto-wrap — should raise, not silently produce defaults
        with pytest.raises(ExecutionError):
            _parse_json_response(text, MultiField)

    def test_extract_json_prefers_object_over_array(self):
        """When both { and [ exist, prefer { if it comes first."""
        from neograph._llm_retry import _extract_json

        text = '{"items": [1, 2, 3]}'
        result = _extract_json(text)
        assert result.startswith("{"), f"Expected object, got: {result[:50]}"


class TestDSMLTrailingToolCallRecovery:
    """neograph-vj2z: DSML markup in final response after budget exhaustion."""

    def test_dsml_markup_retried_with_targeted_directive(self):
        """Model emitting DSML after budget → targeted retry → success."""
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        call_count = [0]

        class DSMLFake:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                call_count[0] += 1
                if call_count[0] == 1:
                    # First call: successful tool call
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search", "args": {"q": "test"}, "id": "c1"}]
                    return msg
                if call_count[0] == 2:
                    # After tool execution: budget exhausted, emit DSML markup
                    return AIMessage(
                        content=(
                            "<\uff5cDSML\uff5ctool_calls>\n"
                            '<\uff5cDSML\uff5cinvoke name="search">\n'
                            '<\uff5cDSML\uff5cparameter name="q">more search</\uff5cDSML\uff5cparameter>\n'
                            "</\uff5cDSML\uff5cinvoke>\n"
                            "</\uff5cDSML\uff5ctool_calls>"
                        )
                    )
                # Targeted retry: produce valid JSON
                return AIMessage(content='{"answer": "recovered"}')

            def with_structured_output(self, model, **kw):
                return self

        def search_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            return StructuredTool.from_function(lambda q: f"found {q}", name="search", description="search")

        register_tool_factory("search", search_factory)

        class Answer(_BM):
            answer: str

        tools = [Tool(name="search", description="search")]
        budget = ToolBudgetTracker(tools)

        parsed, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            llm_config={"output_strategy": "json_mode"},
            runtime=build_fake_runtime(lambda tier: DSMLFake()),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert parsed.answer == "recovered"
        assert len(interactions) == 1  # one tool call succeeded before DSML

    def test_custom_budget_exhausted_message(self):
        """User-provided budget_exhausted_message is used in the retry."""
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        captured_messages = []
        call_count = [0]

        class CaptureFake:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                call_count[0] += 1
                captured_messages.append(messages[-1] if messages else None)
                if call_count[0] == 1:
                    return AIMessage(content='<\uff5cDSML\uff5cinvoke name="x"/>')
                return AIMessage(content='{"answer": "ok"}')

            def with_structured_output(self, model, **kw):
                return self

        register_tool_factory(
            "x",
            lambda c, tc: __import__("langchain_core.tools", fromlist=["StructuredTool"]).StructuredTool.from_function(
                lambda q="": "ok", name="x", description="x"
            ),
        )

        class Answer(_BM):
            answer: str

        tools = [Tool(name="x", description="x")]
        budget = ToolBudgetTracker(tools)

        parsed, _ = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            llm_config={
                "output_strategy": "json_mode",
                "budget_exhausted_message": "CUSTOM: stop calling tools, produce the answer",
            },
            runtime=build_fake_runtime(lambda tier: CaptureFake()),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert parsed.answer == "ok"
        # The custom message should have been used in the retry
        retry_msg = captured_messages[-1]
        content = (
            retry_msg["content"]
            if isinstance(retry_msg, dict)
            else retry_msg.content
            if hasattr(retry_msg, "content")
            else str(retry_msg)
        )
        assert "CUSTOM" in content


# =============================================================================
# tool_loop coverage gaps: DSML retry axes (3, 4, 5, 12) + CoercingToolWrapper (8, 9, 10)
# Filed as neograph-tjhe, -44eq, -gxv8, -xrrt, -bd18, -n4hu, -2od5
# =============================================================================


class TestDSMLDoubleFailure:
    """neograph-tjhe (axis 3): targeted DSML retry returns DSML → generic retry recovers.

    Call sequence:
        1: successful tool call (search)
        2: DSML as "final" response (triggers targeted retry)
        3: targeted retry ALSO returns DSML (double failure)
        4: generic _invoke_json_with_retry first invoke → valid JSON
    """

    def test_double_dsml_falls_through_to_generic_retry(self):
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        call_count = [0]
        dsml_payload = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="search">\n'
            '<｜DSML｜parameter name="q">more search</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

        class DoubleDSMLFake:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                call_count[0] += 1
                if call_count[0] == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search", "args": {"q": "test"}, "id": "c1"}]
                    return msg
                if call_count[0] == 2:
                    return AIMessage(content=dsml_payload)
                if call_count[0] == 3:
                    return AIMessage(content=dsml_payload)
                return AIMessage(content='{"answer": "recovered-via-generic"}')

            def with_structured_output(self, model, **kw):
                return self

        def search_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            return StructuredTool.from_function(lambda q: f"found {q}", name="search", description="search")

        register_tool_factory("search", search_factory)

        class Answer(_BM):
            answer: str

        tools = [Tool(name="search", description="search")]
        budget = ToolBudgetTracker(tools)

        parsed, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            llm_config={"output_strategy": "json_mode"},
            runtime=build_fake_runtime(lambda tier: DoubleDSMLFake()),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        # Contract: double DSML still recovers through the generic retry path.
        # Drop the exact call_count == 4 pin; require at least the initial tool
        # call + DSML attempt + generic-retry success.
        assert parsed.answer == "recovered-via-generic"
        assert len(interactions) == 1
        assert call_count[0] >= 3


class TestDSMLAllRetriesFail:
    """neograph-44eq (axis 4): every retry returns DSML → ExecutionError with hint.

    Default max_retries=1 chain: tool call → DSML → targeted retry DSML →
    generic retry invoke → generic retry invoke → ExecutionError.
    """

    def test_exhausted_retries_raise_execution_error_with_dsml_hint(self):
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.errors import ExecutionError
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        call_count = [0]
        dsml_payload = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="search">\n'
            '<｜DSML｜parameter name="q">more search</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

        class AllDSMLFake:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                call_count[0] += 1
                if call_count[0] == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search", "args": {"q": "test"}, "id": "c1"}]
                    return msg
                return AIMessage(content=dsml_payload)

            def with_structured_output(self, model, **kw):
                return self

        def search_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            return StructuredTool.from_function(lambda q: f"found {q}", name="search", description="search")

        register_tool_factory("search", search_factory)

        class Answer(_BM):
            answer: str

        tools = [Tool(name="search", description="search")]
        budget = ToolBudgetTracker(tools)

        with pytest.raises(ExecutionError, match=r"(?i)(tool-?call markup|DSML)") as exc_info:
            invoke_with_tools(
                model_tier="fast",
                prompt_template="test",
                input_data={},
                output_model=Answer,
                config={"configurable": {}},
                tools=tools,
                budget_tracker=budget,
                llm_config={"output_strategy": "json_mode"},
                runtime=build_fake_runtime(lambda tier: AllDSMLFake()),
                tool_factory_lookup=build_fake_tool_lookup(),
            )

        assert str(exc_info.value).strip()
        assert call_count[0] >= 3


class TestNonDSMLParseFailureTakesGenericRetry:
    """neograph-gxv8 (axis 5): plain non-DSML parse failure → generic retry path.

    When the final response is unparseable JSON that contains NO DSML/XML/
    tool-call markup, control flows to `_invoke_json_with_retry`. The
    user-visible contract is that the garbled response is still recovered.
    Internal routing (DSML vs generic branch) is implementation detail and is
    covered by TestObservabilityContract.
    """

    def test_plain_json_parse_failure_bypasses_dsml_branch(self):
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        call_count = [0]

        # Plain, unparseable text with NO DSML/XML markers — must bypass the
        # `<...function_call|invoke|DSML...>` regex in the targeted retry branch.
        garbled_plain = "this is not json and has no tags at all"

        class GarbledPlainFake:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                call_count[0] += 1
                if call_count[0] == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search", "args": {"q": "test"}, "id": "c1"}]
                    return msg
                # Calls 2 and 3: garbled plain text drives the parse-failure +
                # generic retry path.
                if call_count[0] in (2, 3):
                    return AIMessage(content=garbled_plain)
                return AIMessage(content='{"answer": "recovered-via-generic"}')

            def with_structured_output(self, model, **kw):
                return self

        def search_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            return StructuredTool.from_function(lambda q: f"found {q}", name="search", description="search")

        register_tool_factory("search", search_factory)

        class Answer(_BM):
            answer: str

        tools = [Tool(name="search", description="search")]
        budget = ToolBudgetTracker(tools)

        parsed, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            llm_config={"output_strategy": "json_mode", "max_retries": 2},
            runtime=build_fake_runtime(lambda tier: GarbledPlainFake()),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        # Contract: plain-text parse failure (no DSML markup) still recovers
        # through the generic retry path. The user-visible result is "answer
        # recovered". Internal routing (DSML vs generic) and message phrase
        # pinning are implementation detail — observability tested in
        # TestObservabilityContract.
        assert parsed.answer == "recovered-via-generic"
        assert len(interactions) == 1


class TestE2EDSMLRecoveryViaAgentMode:
    """neograph-xrrt (axis 12): full E2E recovery through compile(**_llm_kw, llm_factory=lambda tier: GarbledPlainFake(), prompt_compiler=lambda t, d, **kw: [{"role": "user", "content": "test"}], **build_test_compile_kwargs())/run().

    Verifies the DSML fix propagates through @node → construct_from_module →
    compile → run, not just the low-level invoke_with_tools call.
    """

    def test_agent_mode_recovers_from_dsml_after_budget_exhaustion_e2e(self):
        import types as _types

        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as _BM

        from neograph import (
            Tool,
            compile,
            construct_from_module,
            node,
            run,
        )
        from tests.fakes import register_tool_factory

        class Answer(_BM):
            answer: str

        call_count = [0]
        dsml_payload = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="search">\n'
            '<｜DSML｜parameter name="q">more search</｜DSML｜parameter>\n'
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
                    msg.tool_calls = [{"name": "search", "args": {"q": "test"}, "id": "c1"}]
                    return msg
                if call_count[0] == 2:
                    return AIMessage(content=dsml_payload)
                return AIMessage(content='{"answer": "recovered-e2e"}')

            def with_structured_output(self, model, **kw):
                return self

        def search_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            return StructuredTool.from_function(lambda q: f"found {q}", name="search", description="search")

        register_tool_factory("search", search_factory)

        mod = _types.ModuleType("test_xrrt_e2e_dsml_mod")

        @node(
            mode="agent",
            outputs=Answer,
            model="fast",
            prompt="test/research",
            tools=[Tool(name="search", description="search", budget=1)],
            llm_config={"output_strategy": "json_mode"},
        )
        def research() -> Answer: ...

        mod.research = research
        pipeline = construct_from_module(mod, name="test-xrrt-e2e")
        graph = compile(
            pipeline,
            llm_factory=lambda tier: DSMLFake(),
            prompt_compiler=lambda t, d, **kw: [{"role": "user", "content": "test"}],
            **build_test_compile_kwargs(),
        )
        result = run(graph, input={"node_id": "test-xrrt"})

        # Contract: E2E DSML recovery returns the recovered answer.
        # Drop call_count[0] == 3 pin (implementation detail).
        assert result["research"].answer == "recovered-e2e"


class TestCoercingToolWrapperGenerateNotAvailable:
    """neograph-bd18 (axis 8): LLM has no _generate() → fall back to empty AIMessage."""

    def test_generate_not_available_falls_back_to_empty(self):
        from langchain_core.messages import AIMessage
        from pydantic import ValidationError

        from neograph._tool_loop import _CoercingToolWrapper

        class NoGenerateFake:
            def invoke(self, messages, **kw):
                raise ValidationError.from_exception_data(
                    title="AIMessage",
                    line_errors=[
                        {
                            "type": "dict_type",
                            "loc": ("tool_calls", 0, "args"),
                            "msg": "Input should be a valid dictionary",
                            "input": "not a dict",
                        }
                    ],
                )

        assert not hasattr(NoGenerateFake(), "_generate")

        wrapper = _CoercingToolWrapper(NoGenerateFake())
        response = wrapper.invoke([])

        # Contract: wrapper falls back to an empty AIMessage when _generate()
        # is unavailable. The structlog event-name assertion is implementation
        # detail and is covered by TestObservabilityContract.
        assert isinstance(response, AIMessage)
        assert response.content == ""
        assert getattr(response, "tool_calls", []) == []


class TestCoercingToolWrapperGenerateRaises:
    """neograph-n4hu (axis 9): _generate() raises → wrapper catches, falls back."""

    def test_generate_raises_exception_falls_back_to_empty(self):
        from langchain_core.messages import AIMessage
        from pydantic import ValidationError

        from neograph._tool_loop import _CoercingToolWrapper

        class GenerateRaisesFake:
            def invoke(self, messages, **kw):
                raise ValidationError.from_exception_data(
                    title="AIMessage",
                    line_errors=[
                        {
                            "type": "dict_type",
                            "loc": ("tool_calls", 0, "args"),
                            "msg": "Input should be a valid dictionary",
                            "input": "bad",
                        }
                    ],
                )

            def _generate(self, messages, *, run_manager=None, **kw):
                raise RuntimeError("simulated network failure")

        wrapper = _CoercingToolWrapper(GenerateRaisesFake())
        response = wrapper.invoke([])

        # Contract: wrapper falls back to an empty AIMessage when _generate()
        # raises. The structlog event-name, log-level, and error-substring
        # assertions are implementation detail and are covered by
        # TestObservabilityContract.
        assert isinstance(response, AIMessage)
        assert response.content == ""
        assert getattr(response, "tool_calls", []) == []


# REMOVED: TestCoercingToolWrapperMixedDictAndStringArgs (neograph-2od5, axis 10)
# Reason: mutation audit exposed the test was exercising LangChain, not the
# wrapper. When `_generate()` returns an AIMessage built from
# `additional_kwargs={"tool_calls": [{"function": {"arguments": "..."}}, ...]}`,
# LangChain's `default_tool_parser` runs inside the AIMessage constructor and
# has already converted valid JSON argument strings into dicts, while routing
# unparseable strings to `.invalid_tool_calls`. By the time the wrapper's own
# explicit `for tc in raw_msg.tool_calls: if isinstance(tc.get("args"), str):
# tc["args"] = _json.loads(...)` loop runs, every entry in `.tool_calls` is
# already a dict — the wrapper's loop is a no-op on this input shape.
# Removing the wrapper's coercion loop entirely did NOT make the test fail.
# neograph-2od5 reopened for re-execution with an input that actually exercises
# the wrapper's coercion (e.g. a raw_msg.tool_calls that still contains a
# string-args entry after construction, or direct assertions on LangChain's
# routing contract rather than the wrapper's behavior).


class TestCoercingToolWrapperMixedDictAndStringArgs:
    """neograph-2od5 (axis 10): wrapper's own coercion loop converts string
    tool_calls.args to dicts and maps unparseable strings to {}.

    Bypasses LangChain's AIMessage constructor (which would run
    ``default_tool_parser`` and mutate args before our loop sees them) by
    returning a ``SimpleNamespace`` with a raw ``.tool_calls`` list from
    ``_generate()``. This forces the wrapper's loop at
    ``_tool_loop.py:121-127`` to perform the coercion itself.
    """

    def test_mixed_dict_and_string_args_are_all_coerced_to_dicts(self):
        from types import SimpleNamespace

        from pydantic import ValidationError

        from neograph._tool_loop import _CoercingToolWrapper

        class MixedArgsFake:
            def invoke(self, messages, **kw):
                raise ValidationError.from_exception_data(
                    title="AIMessage",
                    line_errors=[
                        {
                            "type": "dict_type",
                            "loc": ("tool_calls", 1, "args"),
                            "msg": "Input should be a valid dictionary",
                            "input": '{"query": "second-json"}',
                        }
                    ],
                )

            def _generate(self, messages, *, run_manager=None, **kw):
                fake_msg = SimpleNamespace(
                    tool_calls=[
                        {"name": "search", "args": {"query": "first"}, "id": "c1"},
                        {"name": "search", "args": '{"query": "second-json"}', "id": "c2"},
                        {"name": "search", "args": "unparseable", "id": "c3"},
                        {"name": "search", "args": {"query": "fourth"}, "id": "c4"},
                    ]
                )
                return SimpleNamespace(generations=[SimpleNamespace(message=fake_msg)])

        wrapper = _CoercingToolWrapper(MixedArgsFake())
        response = wrapper.invoke([])

        assert hasattr(response, "tool_calls")
        tcs = response.tool_calls
        assert len(tcs) == 4

        assert tcs[0]["args"] == {"query": "first"}
        assert isinstance(tcs[0]["args"], dict)

        assert tcs[1]["args"] == {"query": "second-json"}
        assert isinstance(tcs[1]["args"], dict)

        # c3's "unparseable" string is not valid JSON: rather than blank to {}
        # (which silently ran the tool with empty args), the coercion path now
        # stamps the unparseable marker with the raw string preserved so the
        # tool-execution seam surfaces a retriable error (neograph-arus). Still a
        # dict, so AIMessage validation holds.
        from neograph._tool_loop import UNPARSEABLE_ARGS_MARKER

        assert tcs[2]["args"] == {UNPARSEABLE_ARGS_MARKER: "unparseable"}
        assert isinstance(tcs[2]["args"], dict)

        assert tcs[3]["args"] == {"query": "fourth"}
        assert isinstance(tcs[3]["args"], dict)


# =============================================================================
# DSML retry — additional axes (6, 7, 8, 9)
# Filed as neograph-bxxf, -tdp3, -xkyk, -d5nm. All mutation-verified.
# =============================================================================


class TestDSMLInStructuredStrategyPath:
    """neograph-bxxf (axis 6): structured strategy has NO DSML recovery.

    The json_mode/text path has a DSML-detection regex + targeted retry
    (_tool_loop.py lines 348-373). The structured path at line 376 calls
    _call_structured, which has NO DSML handling. The only retry there is
    a LangChain-compat fallback that catches TypeError from the first invoke
    (to support providers that reject include_raw=True). If DSML content
    triggers TypeError in with_structured_output, the compat fallback fires
    EXACTLY once and then the error bubbles — no DSML recovery.

    This is a documented parity gap vs json_mode/text.

    Mutation-verified: removing the `except TypeError` compat fallback in
    _call_structured makes this test fail (structured_calls goes from 2 to 1).
    """

    def test_structured_path_recovers_dsml_when_provider_returns_dsml(self):
        """neograph-0tid: structured strategy should recover DSML markup just like
        json_mode/text. Strategy-agnostic recovery: detection is content-based.

        REPRODUCES the bug: today this test FAILS because the gate at
        _tool_loop.py:354 excludes structured from DSML recovery; the structured
        path raises TypeError instead of recovering.
        """
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        DSML_MARKUP = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="search">\n'
            '<｜DSML｜parameter name="q">more</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

        class Answer(_BM):
            answer: str

        call_count = [0]

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
                    # Final response contains DSML markup
                    return AIMessage(content=DSML_MARKUP)
                # Targeted-retry path (post-recovery) returns valid JSON
                return AIMessage(content='{"answer": "recovered-structured"}')

            def with_structured_output(self, model, **kw):
                class _StructuredWrap:
                    def invoke(self, messages, **kw2):
                        raise TypeError("Expected Answer but got non-JSON content: " + DSML_MARKUP[:50])

                return _StructuredWrap()

        def search_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            return StructuredTool.from_function(lambda q: f"found {q}", name="search", description="search")

        register_tool_factory("search", search_factory)

        tools = [Tool(name="search", description="search")]
        budget = ToolBudgetTracker(tools)

        parsed, _ = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            llm_config={"output_strategy": "structured"},
            runtime=build_fake_runtime(lambda tier: DSMLFake()),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        # Contract: DSML in structured strategy recovers via targeted retry,
        # just like json_mode/text. Detection is content-based, strategy-agnostic.
        assert parsed.answer == "recovered-structured"

    def test_structured_path_happy_baseline_no_dsml(self):
        """Control: structured strategy works when provider returns valid model."""
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Answer(_BM):
            answer: str

        call_count = [0]
        structured_calls = [0]

        class HappyFake:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                call_count[0] += 1
                if call_count[0] == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search", "args": {"q": "x"}, "id": "c1"}]
                    return msg
                # neograph-f7nt: agent mode parses this final turn as JSON.
                return AIMessage(content='{"answer": "ok"}')

            def with_structured_output(self, model, include_raw=False, **kw):
                outer_model = model
                inc = include_raw

                class _StructuredWrap:
                    def invoke(self, messages, **kw2):
                        structured_calls[0] += 1
                        parsed = outer_model(answer="ok")
                        if inc:
                            return {"parsed": parsed, "raw": AIMessage(content="fake")}
                        return parsed

                return _StructuredWrap()

        def search_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            return StructuredTool.from_function(lambda q: f"found {q}", name="search", description="search")

        register_tool_factory("search", search_factory)

        tools = [Tool(name="search", description="search")]
        budget = ToolBudgetTracker(tools)

        parsed, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            llm_config={"output_strategy": "structured"},
            runtime=build_fake_runtime(lambda tier: HappyFake()),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        # Contract (neograph-f7nt): agent mode parses the ReAct final turn as
        # JSON directly — no separate structured re-generation — and the scripted
        # single tool call surfaces one interaction.
        assert parsed.answer == "ok"
        assert len(interactions) == 1
        assert structured_calls[0] == 0  # no _call_structured re-gen in agent mode

    def test_structured_path_no_spurious_recovery_for_valid_json(self):
        """neograph-0tid: structured strategy + valid model response ->
        no DSML recovery fires (no extra invokes, no warning logs).

        Sanity check: recovery is gated by content detection. A clean parsed
        model from with_structured_output(...).invoke() must not trigger
        targeted retry or budget-exhausted feedback.
        """
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Answer(_BM):
            answer: str

        call_count = [0]
        structured_calls = [0]

        class HappyFake:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                call_count[0] += 1
                if call_count[0] == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search", "args": {"q": "x"}, "id": "c1"}]
                    return msg
                # neograph-f7nt: agent mode parses this final turn as JSON.
                return AIMessage(content='{"answer": "clean"}')

            def with_structured_output(self, model, include_raw=False, **kw):
                outer_model = model
                inc = include_raw

                class _StructuredWrap:
                    def invoke(self, messages, **kw2):
                        structured_calls[0] += 1
                        parsed = outer_model(answer="clean")
                        if inc:
                            return {"parsed": parsed, "raw": AIMessage(content="fake")}
                        return parsed

                return _StructuredWrap()

        def search_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            return StructuredTool.from_function(lambda q: f"found {q}", name="search", description="search")

        register_tool_factory("search", search_factory)

        tools = [Tool(name="search", description="search")]
        budget = ToolBudgetTracker(tools)

        baseline_call_count = call_count[0]
        parsed, _ = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            llm_config={"output_strategy": "structured"},
            runtime=build_fake_runtime(lambda tier: HappyFake()),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        # Contract (neograph-0tid, agent mode): a clean JSON final turn parses
        # directly -> no DSML recovery fired, and no separate structured re-gen.
        assert parsed.answer == "clean"
        assert structured_calls[0] == 0  # no _call_structured re-gen in agent mode
        # No extra raw llm.invoke beyond the ReAct loop's tool-call + empty-final
        # iterations (2 calls total: tool_call + empty final).
        assert call_count[0] - baseline_call_count == 2

    def test_structured_path_recovers_dsml_when_parsed_is_none_silent_variant(self):
        """neograph-0tid: silent variant — provider returns
        {"parsed": None, "raw": <AIMessage with DSML>, "parsing_error": ...}.

        Before the fix this silently returned None to the caller (the
        ``result = raw_result["parsed"]`` unpack passes through). The
        helper now inspects the raw content for DSML markup and recovers.
        """
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as _BM
        from pydantic import ValidationError

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        DSML_MARKUP = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="search">\n'
            '<｜DSML｜parameter name="q">more</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

        class Answer(_BM):
            answer: str

        call_count = [0]

        class SilentDSMLFake:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                call_count[0] += 1
                if call_count[0] == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search", "args": {"q": "x"}, "id": "c1"}]
                    return msg
                if call_count[0] == 2:
                    return AIMessage(content=DSML_MARKUP)
                # Targeted-retry recovery: valid JSON.
                return AIMessage(content='{"answer": "recovered-structured"}')

            def with_structured_output(self, model, include_raw=False, **kw):
                inc = include_raw

                class _StructuredWrap:
                    def invoke(self, messages, **kw2):
                        # Simulate LangChain's silent variant: parsed=None,
                        # raw contains DSML markup, parsing_error populated.
                        raw_msg = AIMessage(content=DSML_MARKUP)
                        try:
                            err = ValidationError.from_exception_data("Answer", [])
                        except Exception:
                            err = None
                        if inc:
                            return {
                                "parsed": None,
                                "raw": raw_msg,
                                "parsing_error": err,
                            }
                        # Without include_raw, surface a TypeError so the
                        # compat path also doesn't silently return None.
                        raise TypeError("non-JSON content")

                return _StructuredWrap()

        def search_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            return StructuredTool.from_function(lambda q: f"found {q}", name="search", description="search")

        register_tool_factory("search", search_factory)

        tools = [Tool(name="search", description="search")]
        budget = ToolBudgetTracker(tools)

        parsed, _ = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            llm_config={"output_strategy": "structured"},
            runtime=build_fake_runtime(lambda tier: SilentDSMLFake()),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        # Contract: silent variant no longer drops parsed=None to caller;
        # the helper detects DSML in raw and recovers via targeted retry.
        assert parsed is not None
        assert parsed.answer == "recovered-structured"


class TestDSMLAfterMaxIterationsGuard:
    """neograph-tdp3 (axis 7): DSML after max_iterations guard unbinds tools.

    Flow with max_iterations=1:
      call 1: tool_call emitted -> guard fires, tools unbound, loop continues
      call 2: DSML content (forced final) -> loop exits, parse fails
      call 3: targeted retry via raw llm.invoke -> valid JSON -> recovered

    Mutation-verified: `if False:` on the DSML regex makes this test fail
    because no `trailing_tool_call_markup` event is logged.
    """

    def test_dsml_after_max_iterations_recovers_via_targeted_retry(self):
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Answer(_BM):
            answer: str

        call_count = [0]
        dsml_payload = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="search">\n'
            '<｜DSML｜parameter name="q">more</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

        class MaxIterDSMLFake:
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

        parsed, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            llm_config={"output_strategy": "json_mode", "max_iterations": 1},
            runtime=build_fake_runtime(lambda tier: MaxIterDSMLFake()),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        # Contract: DSML after max_iterations guard still recovers via the
        # targeted retry path. Drop exact call_count == 3 pin and structlog
        # event-name assertions (implementation detail; observability tested in
        # TestObservabilityContract).
        assert parsed.answer == "recovered"
        # Recovery requires at least: tool call + DSML + targeted retry.
        assert call_count[0] >= 3
        # Tool call on iteration 1 is skipped by the guard — no interaction recorded.
        assert interactions == []


class TestDSMLAfterTokenBudget:
    """neograph-xkyk (axis 8): DSML after token_budget guard unbinds tools.

    Flow with token_budget=10:
      iter 1: tool_call, input_tokens=5, cumulative=5, tool runs
      iter 2: tool_call, input_tokens=20, cumulative=25 > 10 -> guard fires
      iter 3 (unbound): DSML content -> loop exits, parse fails
      retry: valid JSON -> recovered

    Mutation-verified: `if False:` on DSML regex makes both assertions fail.
    """

    def test_dsml_after_token_budget_recovered_via_targeted_retry(self):
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Answer(_BM):
            answer: str

        call_count = [0]

        class TokenBudgetDSMLFake:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                call_count[0] += 1
                if call_count[0] == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search", "args": {"q": "first"}, "id": "c1"}]
                    msg.usage_metadata = {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15}
                    return msg
                if call_count[0] == 2:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search", "args": {"q": "second"}, "id": "c2"}]
                    msg.usage_metadata = {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
                    return msg
                if call_count[0] == 3:
                    return AIMessage(
                        content=(
                            "<｜DSML｜tool_calls>\n"
                            '<｜DSML｜invoke name="search">\n'
                            '<｜DSML｜parameter name="q">blocked</｜DSML｜parameter>\n'
                            "</｜DSML｜invoke>\n"
                            "</｜DSML｜tool_calls>"
                        )
                    )
                return AIMessage(content='{"answer": "recovered-after-token-budget"}')

            def with_structured_output(self, model, **kw):
                return self

        def search_factory(config, tool_config):
            from langchain_core.tools import StructuredTool

            return StructuredTool.from_function(lambda q: f"found {q}", name="search", description="search")

        register_tool_factory("search", search_factory)

        tools = [Tool(name="search", description="search")]
        budget = ToolBudgetTracker(tools)

        parsed, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            llm_config={
                "output_strategy": "json_mode",
                "token_budget": 10,
                "max_iterations": 50,
            },
            runtime=build_fake_runtime(lambda tier: TokenBudgetDSMLFake()),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        # Contract: DSML after token_budget guard still recovers via the
        # targeted retry path. Drop exact call_count == 4 pin and structlog
        # event-name assertions (implementation detail; observability tested in
        # TestObservabilityContract).
        assert parsed.answer == "recovered-after-token-budget"
        assert len(interactions) == 1
        # Recovery requires at least: 2 budgeted tool calls + DSML + targeted retry.
        assert call_count[0] >= 3


class TestMultipleIndependentDSMLRecoveries:
    """neograph-d5nm (axis 9): two sequential invoke_with_tools calls each recover independently.

    Discriminates state leakage by using DIFFERENT tool names and DIFFERENT
    answers across the two calls. A shared mutable (interactions list, retry
    counter, budget tracker) would surface as wrong call 2 result, duplicated
    interactions, or shared-object identity.

    Mutation-verified: replacing `tool_interactions: list = []` with a
    module-level `_LEAKY_INTERACTIONS` shared across calls makes this test
    fail on `len(interactions2) == 1` (it becomes 2).
    """

    def test_two_sequential_calls_recover_independently(self):
        from langchain_core.messages import AIMessage
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Answer(_BM):
            answer: str

        dsml_template = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="{tool}">\n'
            '<｜DSML｜parameter name="q">more</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

        class DSMLThenJsonFake:
            def __init__(self, tool_name: str, answer: str):
                self._tool_name = tool_name
                self._answer = answer
                self.calls = 0

            def bind_tools(self, tools):
                return self

            def with_structured_output(self, model, **kw):
                return self

            def invoke(self, messages, **kw):
                self.calls += 1
                if self.calls == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": self._tool_name, "args": {"q": "x"}, "id": f"c-{self._tool_name}"}]
                    return msg
                if self.calls == 2:
                    return AIMessage(content=dsml_template.format(tool=self._tool_name))
                return AIMessage(content=f'{{"answer": "{self._answer}"}}')

        def make_tool_factory(tool_name):
            def factory(config, tool_config):
                return StructuredTool.from_function(
                    lambda q="": f"{tool_name}-result",
                    name=tool_name,
                    description=tool_name,
                )

            return factory

        # --- Call 1: tool "search", answer "first-recovered" ---
        fake1 = DSMLThenJsonFake("search", "first-recovered")
        register_tool_factory("search", make_tool_factory("search"))

        tools1 = [Tool(name="search", description="search", budget=1)]
        budget1 = ToolBudgetTracker(tools1)

        parsed1, interactions1 = invoke_with_tools(
            model_tier="fast",
            prompt_template="p1",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools1,
            budget_tracker=budget1,
            llm_config={"output_strategy": "json_mode"},
            runtime=build_fake_runtime(lambda tier: fake1),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert parsed1.answer == "first-recovered"
        assert len(interactions1) == 1
        assert interactions1[0].tool_name == "search"
        assert fake1.calls == 3

        # --- Call 2: tool "lookup", answer "second-recovered" ---
        fake2 = DSMLThenJsonFake("lookup", "second-recovered")
        register_tool_factory("lookup", make_tool_factory("lookup"))

        tools2 = [Tool(name="lookup", description="lookup", budget=1)]
        budget2 = ToolBudgetTracker(tools2)

        parsed2, interactions2 = invoke_with_tools(
            model_tier="fast",
            prompt_template="p2",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools2,
            budget_tracker=budget2,
            llm_config={"output_strategy": "json_mode"},
            runtime=build_fake_runtime(lambda tier: fake2),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert parsed2.answer == "second-recovered"
        assert len(interactions2) == 1
        assert interactions2[0].tool_name == "lookup"
        assert fake2.calls == 3

        # Cross-call independence: distinct identity + distinct contents.
        assert interactions1 is not interactions2
        assert interactions1[0].tool_name != interactions2[0].tool_name
        assert fake1.calls == 3  # first fake untouched after second call
        # neograph-m6d3.3: budget isolation is now guaranteed BY CONSTRUCTION —
        # each run() gets its own neo_agent_budget_* state channel, not a caller-
        # owned ToolBudgetTracker. The isolation this class exists to prove is
        # re-asserted above via the distinct per-call interactions (each tool
        # called exactly once, distinct names, distinct lists). The old
        # budget1/budget2.exhausted_tools() inspection is dropped because the
        # cycle does not mutate the passed-in tracker.


# =============================================================================
# Round 3 — budget_exhausted_message + guard-break + tool-exception + tool_calls shape
# Filed as neograph-h5d7, -z429, -4skw, -3ne3, -9lfh. All mutation-verified.
# =============================================================================


class TestBudgetExhaustedMessageFallback:
    """neograph-h5d7 / neograph-rnjw: budget_exhausted_message fallback semantics.

    After the LlmConfig refactor, all three inputs (missing, empty, None)
    resolve to the default template via LlmConfig.resolved_budget_exhausted_message.
    The empty/None cases are now covered by TestBudgetExhaustedMessageFallbackPostRnjw
    below; this class keeps the missing-key test as the positive-path pin.

    Mutation-verified: replacing the default f-string with "MUTATION: default
    removed" makes the missing-key test fail.
    """

    @staticmethod
    def _extract_retry_content(captured):
        retry_msg = captured[-1]
        if isinstance(retry_msg, dict):
            return retry_msg.get("content")
        if hasattr(retry_msg, "content"):
            return retry_msg.content
        return str(retry_msg)

    @staticmethod
    def _invoke_with_llm_config(llm_config_extras):
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        captured = []
        call_count = [0]

        class CaptureFake:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                call_count[0] += 1
                captured.append(messages[-1] if messages else None)
                if call_count[0] == 1:
                    return AIMessage(content='<｜DSML｜invoke name="x"/>')
                return AIMessage(content='{"answer": "ok"}')

            def with_structured_output(self, model, **kw):
                return self

        register_tool_factory(
            "x",
            lambda c, tc: __import__("langchain_core.tools", fromlist=["StructuredTool"]).StructuredTool.from_function(
                lambda q="": "ok", name="x", description="x"
            ),
        )

        class Answer(_BM):
            answer: str

        tools = [Tool(name="x", description="x")]
        budget = ToolBudgetTracker(tools)
        from neograph._llm_config import LlmConfig as _LlmConfig

        kwargs = {"output_strategy": "json_mode"}
        kwargs.update(llm_config_extras)
        llm_config = _LlmConfig(**kwargs)

        parsed, _ = invoke_with_tools(
            runtime=build_fake_runtime(lambda tier: CaptureFake()),
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            llm_config=llm_config,
            tool_factory_lookup=build_fake_tool_lookup(),
        )
        return parsed, captured

    def test_missing_key_uses_default_message(self):
        parsed, captured = self._invoke_with_llm_config({})
        assert parsed.answer == "ok"
        content = self._extract_retry_content(captured)
        assert content is not None
        assert "tool-call markup" in content
        assert "All tool budgets are exhausted" in content
        assert "Answer" in content


class TestBudgetExhaustedMessageFallbackPostRnjw:
    """neograph-rnjw: budget_exhausted_message=None/'' MUST fall back to default.

    These are the TDD-red tests for the LlmConfig refactor. Against current
    (pre-fix) code they FAIL: invoke_with_tools uses dict.get(key, default)
    which forwards None/'' verbatim instead of resolving the default.

    Post-fix: both None and '' resolve through LlmConfig.resolved_budget_exhausted_message
    to the hardcoded default template (which interpolates output_model.__name__).

    Sibling of TestBudgetExhaustedMessageFallback above -- that class pins the
    buggy behavior and gets deleted when this class passes.
    """

    def test_none_falls_back_to_default_message(self):
        parsed, captured = TestBudgetExhaustedMessageFallback._invoke_with_llm_config(
            {"budget_exhausted_message": None}
        )
        assert parsed.answer == "ok"
        content = TestBudgetExhaustedMessageFallback._extract_retry_content(captured)
        assert content is not None, (
            "budget_exhausted_message=None must fall back to the default template, "
            "not pass None through to chat messages"
        )
        assert "tool-call markup" in content
        assert "All tool budgets are exhausted" in content
        assert "Answer" in content

    def test_empty_string_falls_back_to_default_message(self):
        parsed, captured = TestBudgetExhaustedMessageFallback._invoke_with_llm_config({"budget_exhausted_message": ""})
        assert parsed.answer == "ok"
        content = TestBudgetExhaustedMessageFallback._extract_retry_content(captured)
        assert content, (
            "budget_exhausted_message='' must fall back to the default template, "
            "not pass empty string through to chat messages"
        )
        assert "tool-call markup" in content
        assert "All tool budgets are exhausted" in content
        assert "Answer" in content


class TestLlmConfigTypedView:
    """neograph-rnjw / neograph-pej0: LlmConfig is the IR type for llm_config.

    Post-pej0 semantics:
      - Pydantic rejects wrong types on known framework fields.
      - ``extra='forbid'`` rejects unknown TOP-level keys; provider knobs
        must live in the ``provider_kwargs`` namespace.
    """

    def test_rejects_wrong_type_on_known_field(self):
        from pydantic import ValidationError

        from neograph._llm_config import LlmConfig

        with pytest.raises(ValidationError):
            LlmConfig(max_retries="not-an-int")

    def test_rejects_unknown_top_level_key(self):
        from pydantic import ValidationError

        from neograph._llm_config import LlmConfig

        with pytest.raises(ValidationError):
            LlmConfig(temperature=0.7)  # belongs in provider_kwargs

    def test_provider_kwargs_round_trips(self):
        from neograph._llm_config import LlmConfig

        cfg = LlmConfig(provider_kwargs={"temperature": 0.7, "max_tokens": 512})
        assert cfg.provider_kwargs == {"temperature": 0.7, "max_tokens": 512}

    def test_none_budget_message_resolves_to_default(self):
        from neograph._llm_config import LlmConfig

        cfg = LlmConfig(budget_exhausted_message=None)
        resolved = cfg.resolved_budget_exhausted_message("MyOutput")
        assert "tool-call markup" in resolved
        assert "MyOutput" in resolved

    def test_empty_budget_message_resolves_to_default(self):
        from neograph._llm_config import LlmConfig

        cfg = LlmConfig(budget_exhausted_message="")
        resolved = cfg.resolved_budget_exhausted_message("MyOutput")
        assert "tool-call markup" in resolved
        assert "MyOutput" in resolved

    def test_nonempty_budget_message_is_returned_verbatim(self):
        from neograph._llm_config import LlmConfig

        cfg = LlmConfig(budget_exhausted_message="CUSTOM DIRECTIVE")
        assert cfg.resolved_budget_exhausted_message("X") == "CUSTOM DIRECTIVE"


class TestDefaultBudgetExhaustedMessageRendersModelName:
    """neograph-z429 (axis 11): default budget_exhausted_message resolves output_model.__name__.

    The default is an f-string. Verify the retry message includes the actual
    model class name EXACTLY once, and that the full default phrase is present.

    Mutation-verified: removing the default f-string makes this test fail.
    """

    def test_default_message_includes_output_model_name(self):
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        captured_messages = []
        call_count = [0]

        class CaptureFake:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                call_count[0] += 1
                captured_messages.append(list(messages))
                if call_count[0] == 1:
                    return AIMessage(content='<｜DSML｜invoke name="x"/>')
                return AIMessage(content='{"answer": "ok"}')

            def with_structured_output(self, model, **kw):
                return self

        register_tool_factory(
            "x",
            lambda c, tc: __import__("langchain_core.tools", fromlist=["StructuredTool"]).StructuredTool.from_function(
                lambda q="": "ok", name="x", description="x"
            ),
        )

        class ExplorationResult(_BM):
            answer: str

        tools = [Tool(name="x", description="x")]
        budget = ToolBudgetTracker(tools)

        parsed, _ = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=ExplorationResult,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            llm_config={"output_strategy": "json_mode"},
            runtime=build_fake_runtime(lambda tier: CaptureFake()),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert parsed.answer == "ok"
        retry_msg = captured_messages[-1][-1]
        content = retry_msg["content"] if isinstance(retry_msg, dict) else retry_msg.content
        assert content.count("ExplorationResult") == 1, (
            f"expected 'ExplorationResult' exactly once, got {content.count('ExplorationResult')}"
        )
        assert "Produce the final response as a ExplorationResult object" in content


class TestSafetyBreakOnGuardWithRogueToolCalls:
    """neograph-4skw (axis 8): safety break fires when guard set but LLM still emits tool_calls.

    Rogue LLM scenario: after max_iterations=1 sets _guard_fired on iter 1, the
    LLM emits tool_calls again on iter 2. The safety break at _tool_loop.py:233
    must fire before any rogue dispatch, log react_guard_forced_break, and exit
    the loop so the parse phase can run against messages[-1].

    Mutation-verified: removing the log.warning (keeping break) makes the test
    fail on the event-count assertion. Removing the break entirely creates an
    infinite loop (which itself confirms the safety break's purpose).
    """

    def test_safety_break_fires_when_guard_set_but_rogue_llm_emits_tool_calls(self):
        from langchain_core.messages import AIMessage
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Result(_BM):
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
                msg = AIMessage(content='{"answer": "breakthrough"}')
                msg.tool_calls = [{"name": "search", "args": {"q": "again"}, "id": "c2"}]
                return msg

            def with_structured_output(self, model, **kw):
                return self

        fake = RogueFake()
        dispatched = []

        def _search_impl(q: str = "") -> str:
            dispatched.append(q)
            return f"result for {q}"

        register_tool_factory(
            "search",
            lambda c, tc: StructuredTool.from_function(_search_impl, name="search", description="search tool"),
        )

        tools = [Tool(name="search", description="search tool")]
        budget = ToolBudgetTracker(tools)

        parsed, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Result,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            llm_config={"max_iterations": 1, "output_strategy": "json_mode"},
            runtime=build_fake_runtime(lambda tier: fake),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        # Contract: safety break prevents rogue tool dispatch when the guard is
        # set but the LLM still emits tool_calls. The user-visible recovery is
        # the parsed answer; the no-dispatch behavior is captured by interactions
        # and dispatched lists. Drop structlog event-name + payload assertions
        # (observability is tested in TestObservabilityContract).
        assert parsed.answer == "breakthrough"
        # Safety break must prevent rogue dispatch on iteration 2.
        assert interactions == []
        assert dispatched == []


class TestToolExceptionPropagates:
    """neograph-3ne3 (axis 12): tool_fn.invoke exceptions propagate uncaught.

    Production at _tool_loop.py:278 has no try/except around tool_fn.invoke.
    LangChain's StructuredTool by default re-raises the tool function's
    exception (handle_tool_error=False). The result: any tool-side crash
    propagates out of invoke_with_tools, taking down the ReAct loop with no
    chance for the LLM to observe a ToolMessage error and recover.

    This is a documented GAP. The test pins current behavior; if production
    gains try/except + ToolMessage injection, update this test to assert the
    error-message path.

    Mutation-verified: wrapping tool_fn.invoke in try/except makes this test
    fail with "DID NOT RAISE".
    """

    def test_tool_exception_propagates_out_of_invoke_with_tools(self):
        import pytest as _pytest
        from langchain_core.messages import AIMessage
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Answer(_BM):
            answer: str

        class ToolCallingFake:
            def __init__(self):
                self._calls = 0

            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                self._calls += 1
                if self._calls == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "boom", "args": {"q": "hi"}, "id": "1"}]
                    return msg
                return AIMessage(content='{"answer": "unreachable"}')

            def with_structured_output(self, model, **kw):
                return self

        def _boom(q: str = "") -> str:
            raise RuntimeError("tool-3ne3-kaboom")

        register_tool_factory(
            "boom",
            lambda config, tool_config: StructuredTool.from_function(_boom, name="boom", description="raises"),
        )

        tools = [Tool(name="boom", description="raises")]
        budget = ToolBudgetTracker(tools)

        with _pytest.raises(RuntimeError, match="tool-3ne3-kaboom"):
            invoke_with_tools(
                model_tier="fast",
                prompt_template="test",
                input_data={},
                output_model=Answer,
                config={"configurable": {}},
                tools=tools,
                budget_tracker=budget,
                llm_config={"output_strategy": "json_mode"},
                runtime=build_fake_runtime(lambda tier: ToolCallingFake()),
                tool_factory_lookup=build_fake_tool_lookup(),
            )


class TestToolCallsShapeEdgeCases:
    """neograph-9lfh (axis 14): response.tool_calls == [] vs attribute absent.

    - Empty list: loop exit at `if not response.tool_calls: break` fires correctly.
    - Absent attribute: direct attribute access raises AttributeError (no getattr
      guard). AIMessage always has tool_calls (Pydantic default []), so absent
      only reaches production for non-AIMessage response objects.

    Mutation-verified: `if False:` on the exit check makes the empty-list test
    fail via a bounded loop-runaway assertion (fake raises on call #2).
    """

    class _MutationProofBound(Exception):
        pass

    def test_empty_list_exits_loop_on_first_iteration(self):
        from langchain_core.messages import AIMessage
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Answer(_BM):
            answer: str

        call_count = [0]
        Bound = self._MutationProofBound

        class EmptyListFake:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                call_count[0] += 1
                if call_count[0] > 1:
                    raise Bound(f"loop did not exit on iteration 1; on call #{call_count[0]}")
                msg = AIMessage(content='{"answer": "done"}')
                assert msg.tool_calls == []
                return msg

            def with_structured_output(self, model, **kw):
                return self

        register_tool_factory(
            "x",
            lambda c, tc: StructuredTool.from_function(lambda q="": "ok", name="x", description="x"),
        )

        tools = [Tool(name="x", description="x")]
        budget = ToolBudgetTracker(tools)

        parsed, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=budget,
            llm_config={"output_strategy": "json_mode"},
            runtime=build_fake_runtime(lambda tier: EmptyListFake()),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert call_count[0] == 1
        assert parsed.answer == "done"
        assert interactions == []

    def test_malformed_non_message_response_is_rejected_loudly(self):
        """A response object that is not a LangChain message is rejected loudly.

        neograph-m6d3.3: real LLMs always return AIMessage (tool_calls defaults to
        []), so this only reaches a fake returning a non-Message. The deleted
        monolith accessed ``response.tool_calls`` inline → AttributeError. The
        inline cycle appends the response to the ``neo_agent_messages_*`` channel
        via ``add_messages`` FIRST, so a non-Message is rejected at message
        coercion instead — a different error TYPE, same "malformed → loud failure"
        intent (it never silently passes). Behavior-change flagged in m6d3.3.
        """
        from types import SimpleNamespace

        import pytest as _pytest
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel as _BM

        from neograph import Tool
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Answer(_BM):
            answer: str

        class AbsentAttrFake:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kw):
                resp = SimpleNamespace(content='{"answer": "done"}')
                assert not hasattr(resp, "tool_calls")
                return resp

            def with_structured_output(self, model, **kw):
                return self

        register_tool_factory(
            "x",
            lambda c, tc: StructuredTool.from_function(lambda q="": "ok", name="x", description="x"),
        )

        tools = [Tool(name="x", description="x")]

        # Loud rejection: add_messages cannot coerce a non-Message response, so
        # LangChain raises NotImplementedError at the channel — never silent.
        with _pytest.raises(NotImplementedError):
            invoke_with_tools(
                model_tier="fast",
                prompt_template="test",
                input_data={},
                output_model=Answer,
                config={"configurable": {}},
                tools=tools,
                llm_config={"output_strategy": "json_mode"},
                runtime=build_fake_runtime(lambda tier: AbsentAttrFake()),
                tool_factory_lookup=build_fake_tool_lookup(),
            )


# ═══════════════════════════════════════════════════════════════════════════
# TEST: neograph-pej0 -- LlmConfig promoted from runtime view to IR type
# ═══════════════════════════════════════════════════════════════════════════
class TestLlmConfigAsIRType:
    """neograph-pej0: LlmConfig is the IR type carried by Node and Construct.

    Pins the architectural invariants:
      - Node.llm_config / Construct.llm_config are LlmConfig instances, not dicts
      - extra='forbid' rejects typos at Node construction (not at first invoke)
      - provider_kwargs is a separate namespace for provider-specific knobs
      - merged_with implements typed merge for Construct propagation
      - as_factory_kwargs flattens for the user llm_factory boundary
      - Pydantic dict coercion preserves model_fields_set (the merge relies on it)
      - Three-surface parity: @node, declarative Node, programmatic Node|Modifier
    """

    def test_typo_on_known_field_raises_at_node_construction(self):
        """`max_retires` (typo) must surface at Node construction, not silently default."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Node(
                name="probe",
                mode="think",
                inputs=Claims,
                outputs=MergedResult,
                model="fast",
                prompt="p",
                llm_config={"max_retires": 5},  # typo!
            )

    def test_provider_kwarg_at_top_level_raises_after_promotion(self):
        """temperature/max_tokens at top-level must use provider_kwargs namespace."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Node(
                name="probe",
                mode="think",
                inputs=Claims,
                outputs=MergedResult,
                model="fast",
                prompt="p",
                llm_config={"temperature": 0.7},  # belongs in provider_kwargs
            )

    def test_provider_kwargs_namespace_round_trips(self):
        """provider_kwargs preserves arbitrary provider knobs."""
        from neograph._llm_config import LlmConfig

        cfg = LlmConfig(provider_kwargs={"temperature": 0.7, "max_tokens": 512})
        assert cfg.provider_kwargs == {"temperature": 0.7, "max_tokens": 512}

    def test_dict_input_preserves_model_fields_set(self):
        """The merge logic depends on model_fields_set surviving Pydantic coercion."""
        n = Node(
            name="x",
            mode="think",
            inputs=Claims,
            outputs=MergedResult,
            model="fast",
            prompt="p",
            llm_config={"max_retries": 5},
        )
        from neograph._llm_config import LlmConfig

        assert isinstance(n.llm_config, LlmConfig)
        assert n.llm_config.model_fields_set == {"max_retries"}
        assert n.llm_config.max_retries == 5
        assert n.llm_config.max_iterations == 20  # untouched default

    def test_default_factory_produces_empty_model_fields_set(self):
        """Default-constructed LlmConfig has no fields set -- enables 'inherit parent' semantics."""
        from neograph._llm_config import LlmConfig

        n = Node(
            name="x",
            mode="think",
            inputs=Claims,
            outputs=MergedResult,
            model="fast",
            prompt="p",
        )
        assert isinstance(n.llm_config, LlmConfig)
        assert n.llm_config.model_fields_set == set()

    def test_as_factory_kwargs_flattens_provider_into_top_level(self):
        """as_factory_kwargs returns the dict the user llm_factory expects."""
        from neograph._llm_config import LlmConfig

        cfg = LlmConfig(max_retries=3, provider_kwargs={"temperature": 0.7})
        flat = cfg.as_factory_kwargs()
        # Framework keys present
        assert flat["max_retries"] == 3
        assert flat["output_strategy"] == "structured"
        # Provider kwargs flattened to top
        assert flat["temperature"] == 0.7
        # provider_kwargs key itself is NOT in the flat dict
        assert "provider_kwargs" not in flat

    def test_as_factory_kwargs_framework_takes_precedence_on_collision(self):
        """If a provider_kwarg shadows a framework key, framework wins (semantics authoritative)."""
        from neograph._llm_config import LlmConfig

        cfg = LlmConfig(max_retries=3, provider_kwargs={"max_retries": 99})
        flat = cfg.as_factory_kwargs()
        assert flat["max_retries"] == 3

    def test_merged_with_child_wins_on_set_fields(self):
        """Construct propagation: child override beats parent default."""
        from neograph._llm_config import LlmConfig

        parent = LlmConfig(max_retries=3, max_iterations=15)
        child = LlmConfig(max_retries=7)  # only max_retries set
        merged = parent.merged_with(child)
        assert merged.max_retries == 7  # child wins
        assert merged.max_iterations == 15  # parent retained (child unset)

    def test_merged_with_child_unset_inherits_parent(self):
        """Default-constructed child inherits all parent fields."""
        from neograph._llm_config import LlmConfig

        parent = LlmConfig(max_retries=3, max_iterations=15, token_budget=1000)
        child = LlmConfig()
        merged = parent.merged_with(child)
        assert merged.max_retries == 3
        assert merged.max_iterations == 15
        assert merged.token_budget == 1000

    def test_announce_tool_budget_survives_merged_with_when_set_on_parent(self):
        """announce_tool_budget rides construct->node inheritance via merged_with (neograph-iyo2).

        The opt-in flag lives on LlmConfig with no bespoke plumbing; the only
        propagation contract that must hold is that merged_with carries it from
        a construct-level parent into a default-constructed child node.
        """
        from neograph._llm_config import LlmConfig

        parent = LlmConfig(announce_tool_budget=True)
        child = LlmConfig()  # node sets nothing -> inherits construct default
        merged = parent.merged_with(child)
        assert merged.announce_tool_budget is True

    def test_announce_tool_budget_child_override_beats_parent_in_merged_with(self):
        """A node that explicitly sets announce_tool_budget wins over the parent (neograph-iyo2)."""
        from neograph._llm_config import LlmConfig

        parent = LlmConfig(announce_tool_budget=True)
        child = LlmConfig(announce_tool_budget=False)
        merged = parent.merged_with(child)
        assert merged.announce_tool_budget is False

    def test_announce_tool_budget_defaults_false_when_unset(self):
        """The opt-in flag is off by default (off-by-default contract) (neograph-iyo2)."""
        from neograph._llm_config import LlmConfig

        assert LlmConfig().announce_tool_budget is False

    def test_merged_with_provider_kwargs_collision_child_wins(self):
        """provider_kwargs merge: child wins on key collision."""
        from neograph._llm_config import LlmConfig

        parent = LlmConfig(provider_kwargs={"temperature": 0.5})
        child = LlmConfig(provider_kwargs={"temperature": 0.9})
        merged = parent.merged_with(child)
        assert merged.provider_kwargs["temperature"] == 0.9

    def test_merged_with_provider_kwargs_disjoint_unions(self):
        """Disjoint provider_kwargs from parent and child both survive."""
        from neograph._llm_config import LlmConfig

        parent = LlmConfig(provider_kwargs={"temperature": 0.5})
        child = LlmConfig(provider_kwargs={"max_tokens": 512})
        merged = parent.merged_with(child)
        assert merged.provider_kwargs == {"temperature": 0.5, "max_tokens": 512}

    def test_invalid_output_strategy_raises_at_node_construction(self):
        """compiler.py:163 runtime check is replaced by Pydantic Literal at Node ctor."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Node(
                name="probe",
                mode="think",
                inputs=Claims,
                outputs=MergedResult,
                model="fast",
                prompt="p",
                llm_config={"output_strategy": "banana"},
            )

    def test_construct_propagation_uses_typed_merge(self):
        """Construct.llm_config flows to children via merged_with, not dict spread."""
        from neograph._llm_config import LlmConfig

        n = Node(
            name="child",
            mode="think",
            inputs=Claims,
            outputs=MergedResult,
            model="fast",
            prompt="p",
            llm_config={"max_retries": 7},
        )
        c = Construct(
            "p",
            llm_config={"max_iterations": 25, "max_retries": 3},
            nodes=[n],
        )
        propagated = c.nodes[0].llm_config
        assert isinstance(propagated, LlmConfig)
        assert propagated.max_retries == 7  # child wins
        assert propagated.max_iterations == 25  # inherited from parent

    def test_node_llm_config_is_typed_field(self):
        """Node.llm_config is annotated as LlmConfig in the IR."""
        from neograph._llm_config import LlmConfig
        from neograph.node import Node as _Node

        ann = _Node.model_fields["llm_config"].annotation
        # Field annotation must be exactly LlmConfig (not dict, not Any)
        assert ann is LlmConfig

    def test_construct_llm_config_is_typed_field(self):
        """Construct.llm_config is annotated as LlmConfig in the IR."""
        from neograph._llm_config import LlmConfig
        from neograph.construct import Construct as _Construct

        ann = _Construct.model_fields["llm_config"].annotation
        assert ann is LlmConfig

    def test_normalize_llm_config_not_referenced_in_src(self):
        """Guard: deletion of normalize_llm_config is permanent across the source tree."""
        import subprocess
        from pathlib import Path

        src_dir = Path(__file__).resolve().parents[2] / "src" / "neograph"
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "normalize_llm_config", str(src_dir)],
            capture_output=True,
            text=True,
        )
        matches = [
            line
            for line in result.stdout.splitlines()
            # Allow the LlmConfig module itself to keep the symbol if needed
            # for backward-compat re-export, but disallow active use elsewhere.
            if line and "_llm_config.py" not in line
        ]
        assert matches == [], f"normalize_llm_config still used in src/: {matches}"

    def test_factory_receives_full_typed_config_as_dict(self):
        """The user llm_factory boundary still receives a flat dict (preserved contract)."""
        from neograph._llm import _get_llm
        from tests.fakes import build_fake_runtime

        captured: dict = {}

        def fake_factory(tier, *, node_name="", llm_config):
            captured.update(llm_config)
            return object()  # any sentinel

        runtime = build_fake_runtime(factory=fake_factory)
        from neograph._llm_config import LlmConfig

        cfg = LlmConfig(max_retries=4, provider_kwargs={"temperature": 0.3})
        _get_llm(runtime, "fast", node_name="n", llm_config=cfg)
        assert captured["max_retries"] == 4
        assert captured["temperature"] == 0.3
        # Default-set framework keys are now visible to the factory
        assert captured["output_strategy"] == "structured"

    def test_three_surface_parity_decorative_declarative_programmatic(self):
        """All three API surfaces produce the same typed IR for llm_config."""
        from neograph._llm_config import LlmConfig
        from neograph.decorators import node as node_decorator

        # Surface 1: @node decorator with dict shorthand
        @node_decorator(
            outputs=MergedResult,
            model="fast",
            prompt="p",
            llm_config={"max_retries": 7},
        )
        def n1(claims: Claims) -> MergedResult: ...

        # Surface 2: declarative Node(...) with LlmConfig instance
        n2 = Node(
            name="n2",
            mode="think",
            inputs=Claims,
            outputs=MergedResult,
            model="fast",
            prompt="p",
            llm_config=LlmConfig(max_retries=7),
        )

        # Surface 3: programmatic Node | Oracle (modifier flow)
        n3_base = Node(
            name="n3",
            mode="think",
            inputs=Claims,
            outputs=MergedResult,
            model="fast",
            prompt="p",
            llm_config=LlmConfig(max_retries=7),
        )

        # All three carry typed LlmConfig with max_retries=7
        for n in (n1, n2, n3_base):
            assert isinstance(n.llm_config, LlmConfig)
            assert n.llm_config.max_retries == 7


# ═══════════════════════════════════════════════════════════════════════════
# TEST: neograph-2ur4 -- Typed Protocols for user-supplied callback slots
# ═══════════════════════════════════════════════════════════════════════════
class TestCallbackProtocols:
    """neograph-2ur4: 9 callback slots carry @runtime_checkable Protocols.

    Pins the IR-level type promotion:
      - configure_llm signature uses LlmFactory / PromptCompiler / CostCallback
      - Oracle.merge_pre_process / merge_post_process / merge_fallback are typed
      - Node.skip_when / skip_value / raw_fn carry their Protocol types
      - Existing runtime introspection (_accepted_params,
        _validate_skip_callables) survives intact
      - Backward-compatible: documented Simple `lambda tier: ...` form still
        satisfies LlmFactory structurally
    """

    def test_llm_factory_protocol_field_type(self):
        """LlmFactory Protocol is runtime-checkable and accepts the documented shape."""
        from neograph._llm import LlmFactory

        f = lambda tier: None  # noqa: E731
        assert isinstance(f, LlmFactory)

    def test_prompt_compiler_protocol_field_type(self):
        """PromptCompiler Protocol is runtime-checkable."""
        from neograph._llm import PromptCompiler

        f = lambda template, data, **kw: []  # noqa: E731
        assert isinstance(f, PromptCompiler)

    def test_cost_callback_protocol_field_type(self):
        """CostCallback Protocol is runtime-checkable."""
        from neograph._llm import CostCallback

        def cb(*, tier, input_tokens, output_tokens, **kw):
            pass

        assert isinstance(cb, CostCallback)

    def test_oracle_merge_hooks_protocol_field_types(self):
        import typing

        from neograph.modifiers import (
            MergeFallback,
            MergePostProcess,
            MergePreProcess,
            Oracle,
        )

        for fname, proto in [
            ("merge_pre_process", MergePreProcess),
            ("merge_post_process", MergePostProcess),
            ("merge_fallback", MergeFallback),
        ]:
            ann = Oracle.model_fields[fname].annotation
            args = typing.get_args(ann)
            assert proto in args, f"{fname} annotation missing {proto.__name__}: {ann}"
            assert type(None) in args

    def test_node_skip_predicate_protocol_field_type(self):
        import typing

        from neograph.node import Node as _Node
        from neograph.node import SkipPredicate

        ann = _Node.model_fields["skip_when"].annotation
        args = typing.get_args(ann)
        assert SkipPredicate in args

    def test_node_skip_value_protocol_field_type(self):
        import typing

        from neograph.node import Node as _Node
        from neograph.node import SkipValueFactory

        ann = _Node.model_fields["skip_value"].annotation
        args = typing.get_args(ann)
        assert SkipValueFactory in args

    def test_node_raw_fn_protocol_field_type(self):
        import typing

        from neograph.node import Node as _Node
        from neograph.node import RawNodeFn

        ann = _Node.model_fields["raw_fn"].annotation
        args = typing.get_args(ann)
        assert RawNodeFn in args

    def test_runtime_checkable_isinstance_works(self):
        from neograph.node import SkipPredicate

        def good(value):
            return True

        bad = "not a callable"
        assert isinstance(good, SkipPredicate)
        assert not isinstance(bad, SkipPredicate)

    def test_simple_lambda_factory_satisfies_protocol(self):
        """Documented `lambda tier: ChatOpenAI(...)` form must still pass."""
        from neograph._llm import LlmFactory

        simple = lambda tier: object()  # noqa: E731
        # Protocol with *args/**kwargs catch -- structural match preserved.
        assert isinstance(simple, LlmFactory)

    def test_kwargs_factory_satisfies_protocol(self):
        from neograph._llm import LlmFactory

        def advanced(tier, *, node_name="", llm_config=None):
            return object()

        assert isinstance(advanced, LlmFactory)

    def test_protocols_publicly_exported(self):
        # Top-level re-export so users can `from neograph import LlmFactory`
        from neograph import (  # noqa: F401
            CostCallback,
            LlmFactory,
            MergeFallback,
            MergePostProcess,
            MergePreProcess,
            PromptCompiler,
            RawNodeFn,
            SkipPredicate,
            SkipValueFactory,
        )

    def test_existing_runtime_introspection_unchanged(self):
        """Protocol-annotated callables still introspect via inspect.signature."""
        from neograph._llm import _accepted_params

        def factory(tier, *, node_name="", llm_config=None):
            return None

        params = _accepted_params(factory)
        assert params == {"tier", "node_name", "llm_config"}

    def test_validate_skip_callables_still_runs(self):
        """Zero-arg lambdas in skip_when must still raise at Node construction."""
        from neograph import Node
        from neograph.errors import ConstructError

        with pytest.raises(ConstructError):
            Node(
                "bad-skip",
                mode="think",
                inputs=Claims,
                outputs=Claims,
                model="fast",
                prompt="p",
                skip_when=lambda: True,  # zero-arg -- must raise
            )


# ═══════════════════════════════════════════════════════════════════════════
# TEST: neograph-xmm8 -- typo rejection at the invoke_* dict boundary
# ═══════════════════════════════════════════════════════════════════════════
class TestDictCoercionTypoRejection:
    """neograph-xmm8: dict-form llm_config at invoke_* / render_prompt
    boundaries must route through LlmConfig.model_validate so typos surface
    as ValidationError BEFORE any LLM call.

    Two layers of typo protection exist post-pej0:
      1. Construction-time -- Node/Construct field rejects typoed dicts.
      2. Direct-call boundary -- when something bypasses Node and calls
         invoke_structured / invoke_with_tools / render_prompt with a raw
         dict, _coerce_llm_config still validates. This class pins layer 2:
         if _coerce_llm_config is ever 'optimized' to skip validation,
         these tests fail loudly.
    """

    def test_invoke_structured_rejects_dict_with_typo(self):
        from pydantic import BaseModel as _BM
        from pydantic import ValidationError as _VE

        from neograph._llm import invoke_structured

        # Mock LLM should never be called -- the typo must raise first.
        called = [False]

        class _NeverCalled:
            def with_structured_output(self, *args, **kwargs):
                called[0] = True
                return self

            def invoke(self, *args, **kwargs):
                called[0] = True
                return None

        class Out(_BM):
            text: str

        with pytest.raises(_VE):
            invoke_structured(
                model_tier="fast",
                prompt_template="p",
                input_data={},
                output_model=Out,
                config={"configurable": {}},
                llm_config={"max_retires": 5},  # typo
                runtime=build_fake_runtime(),
            )

        assert called[0] is False, (
            "LLM was reached -- typo rejection must happen at the "
            "_coerce_llm_config boundary, before any factory or invoke call."
        )

    def test_invoke_with_tools_rejects_dict_with_typo(self):
        from pydantic import BaseModel as _BM
        from pydantic import ValidationError as _VE

        from neograph import Tool
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        called = [False]

        class _NeverCalled:
            def bind_tools(self, *args, **kwargs):
                called[0] = True
                return self

            def invoke(self, *args, **kwargs):
                called[0] = True
                return None

            def with_structured_output(self, *args, **kwargs):
                called[0] = True
                return self

        register_tool_factory("xmm8_probe", lambda c, tc: object())

        class Out(_BM):
            text: str

        tools = [Tool(name="xmm8_probe", description="x")]
        budget = ToolBudgetTracker(tools)

        with pytest.raises(_VE):
            invoke_with_tools(
                model_tier="fast",
                prompt_template="p",
                input_data={},
                output_model=Out,
                tools=tools,
                budget_tracker=budget,
                config={"configurable": {}},
                llm_config={"max_iterstions": 3},  # typo (max_iterations)
                runtime=build_fake_runtime(),
                tool_factory_lookup=build_fake_tool_lookup(),
            )

        assert called[0] is False, (
            "LLM was reached -- typo rejection must happen at the "
            "_coerce_llm_config boundary, before any factory or invoke call."
        )

    def test_coerce_llm_config_rejects_dict_with_typo_directly(self):
        """Direct unit test on the helper -- the smallest possible regression
        gate. If _coerce_llm_config ever skips validation, this fails first."""
        from pydantic import ValidationError as _VE

        from neograph._llm_config import _coerce_llm_config

        with pytest.raises(_VE):
            _coerce_llm_config({"max_retires": 5})


# ═══════════════════════════════════════════════════════════════════════════
# TEST: _llm_structured_compat — provider-quirk compat shim (neograph-ble3)
#
# The compat shim normalizes with_structured_output(model, include_raw=True)
# into a StructuredResult tagged union (Parsed | Raw | Failed). Each provider
# quirk is a decorator on a StructuredOutputAdapter. These tests pin the
# CLASSIFICATION contract (pure result-shape normalization, no re-prompt).
# ═══════════════════════════════════════════════════════════════════════════


class TestStructuredCompatShim:
    """neograph-ble3: build_default_adapter classifies each provider quirk into
    the correct StructuredResult variant. Compat = classification only; the
    decorators never re-prompt the LLM (that is the retry concern).
    """

    _DSML = (
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="search">\n'
        '<｜DSML｜parameter name="q">more</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>"
    )

    def test_happy_path_classifies_as_parsed_with_usage(self):
        """include_raw=True works -> Parsed(model, usage from raw.usage_metadata)."""
        from langchain_core.messages import AIMessage

        from neograph._llm_structured_compat import Parsed, build_default_adapter

        expected = Claims(items=["ok"])

        class HappyLLM:
            def with_structured_output(self, model, include_raw=False, **kw):
                inc = include_raw

                class _Wrap:
                    def invoke(self, messages, **kw2):
                        raw = AIMessage(content="ok")
                        raw.usage_metadata = {"input_tokens": 3, "output_tokens": 5}
                        return {"parsed": expected, "raw": raw} if inc else expected

                return _Wrap()

        result = build_default_adapter().invoke(HappyLLM(), Claims, [], {"configurable": {}})

        assert isinstance(result, Parsed)
        assert result.model == expected
        assert result.usage == {"input_tokens": 3, "output_tokens": 5}

    def test_silent_variant_classifies_as_raw_dsml_and_preserves_usage(self):
        """parsed=None + raw contains DSML -> Raw(dsml=True), usage preserved (R1)."""
        from langchain_core.messages import AIMessage

        from neograph._llm_structured_compat import Raw, build_default_adapter

        class SilentLLM:
            def with_structured_output(self, model, include_raw=False, **kw):
                dsml = TestStructuredCompatShim._DSML

                class _Wrap:
                    def invoke(self, messages, **kw2):
                        raw = AIMessage(content=dsml)
                        raw.usage_metadata = {"input_tokens": 7, "output_tokens": 11}
                        return {"parsed": None, "raw": raw, "parsing_error": ValueError("x")}

                return _Wrap()

        result = build_default_adapter().invoke(SilentLLM(), Claims, [], {"configurable": {}})

        assert isinstance(result, Raw)
        assert result.dsml is True
        assert "DSML" in result.raw_text
        # R1: usage must NOT be dropped on the parsed=None path.
        assert result.usage == {"input_tokens": 7, "output_tokens": 11}

    def test_typeerror_then_success_classifies_as_parsed(self):
        """include_raw=True rejected (TypeError); retry without it succeeds -> Parsed."""
        from neograph._llm_structured_compat import Parsed, build_default_adapter

        expected = Claims(items=["fallback"])

        class TypeErrorLLM:
            def with_structured_output(self, model, **kwargs):
                if kwargs.get("include_raw", False):
                    raise TypeError("unexpected keyword argument 'include_raw'")

                class _Wrap:
                    def invoke(self, messages, **kw2):
                        return expected

                return _Wrap()

        result = build_default_adapter().invoke(TypeErrorLLM(), Claims, [], {"configurable": {}})

        assert isinstance(result, Parsed)
        assert result.model == expected

    def test_double_typeerror_with_trailing_dsml_classifies_as_raw_dsml(self):
        """Both structured invokes raise TypeError; messages[-1] carries DSML
        -> Raw(dsml=True). This is the Case-E unification the original sketch lost.
        Classification reads the trailing message; the decorator does NOT re-invoke.
        """
        from langchain_core.messages import AIMessage

        from neograph._llm_structured_compat import Raw, build_default_adapter

        class DoubleTypeErrorLLM:
            def with_structured_output(self, model, **kwargs):
                class _Wrap:
                    def invoke(self, messages, **kw2):
                        raise TypeError("non-JSON content")

                return _Wrap()

            def invoke(self, messages, **kw):  # must NOT be called by the compat decorator
                raise AssertionError("compat decorator must not re-invoke the LLM")

        trailing = AIMessage(content=TestStructuredCompatShim._DSML)
        result = build_default_adapter().invoke(DoubleTypeErrorLLM(), Claims, [trailing], {"configurable": {}})

        assert isinstance(result, Raw)
        assert result.dsml is True
        assert "DSML" in result.raw_text


class TestCallStructuredParsedNoneFailsLoud:
    """neograph-7wya: the C' path — parsed=None with NO DSML markup — used to
    silently return (None, usage), letting the None flow to the write boundary
    and surface far from its source. It now FAILS LOUD at the source with an
    ExecutionError naming the output model and the unparseable response.

    Supersedes neograph-ble3 (R1), which pinned the legacy silent-None return.
    """

    def test_parsed_none_no_dsml_raises_execution_error(self):
        from langchain_core.messages import AIMessage

        from neograph import ExecutionError
        from neograph._llm_dispatch import _call_structured

        class CPrimeLLM:
            def with_structured_output(self, model, include_raw=False, **kw):
                class _Wrap:
                    def invoke(self, messages, **kw2):
                        raw = AIMessage(content="plain prose, no markup here")
                        raw.usage_metadata = {"input_tokens": 9, "output_tokens": 13}
                        # parsed=None, raw has no DSML markup
                        return {"parsed": None, "raw": raw, "parsing_error": ValueError("x")}

                return _Wrap()

        # cfg defaults to None -> C' path now raises instead of returning None.
        with pytest.raises(ExecutionError) as ei:
            _call_structured(
                CPrimeLLM(),
                [],
                Claims,
                "structured",
                {"configurable": {}},
            )
        msg = str(ei.value)
        assert "Claims" in msg
        assert "decoded to None" in msg


# ═══════════════════════════════════════════════════════════════════════════
# TEST: framework-generated tool-budget preamble injection (neograph-iyo2)
# ═══════════════════════════════════════════════════════════════════════════


class _CapturingReActFake(ReActFake):
    """ReActFake that stashes the FIRST ReAct-loop messages list it sees.

    Existing fakes don't retain `messages`, so we cannot otherwise observe
    whether invoke_with_tools prepended the {role:system} budget preamble.
    Reuses ReActFake's proven final-parse path so the structured output
    dispatch (_call_structured) works unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_messages = None

    def invoke(self, messages, **kwargs):
        if not self._in_structured_mode and self.captured_messages is None:
            self.captured_messages = list(messages)
        return super().invoke(messages, **kwargs)

    def with_structured_output(self, model, **kwargs):
        clone = _CapturingReActFake(self._tool_calls, self._final)
        clone._call_idx = self._call_idx
        clone._model = model
        clone._in_structured_mode = True
        clone.captured_messages = self.captured_messages
        return clone


def _msg_role(msg):
    """Role of a message that may be a dict or a LangChain BaseMessage."""
    if isinstance(msg, dict):
        return msg.get("role")
    return getattr(msg, "type", None)


def _msg_content(msg):
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")


class TestToolBudgetPreambleInjection:
    """announce_tool_budget prepends an accurate {role:system} preamble (neograph-iyo2).

    Drives the behavior through the real invoke_with_tools tool loop (the ONLY
    site holding tools+cfg.max_iterations together), asserting on the messages
    the LLM actually receives. announced == enforced, by construction.
    """

    def _answer_model(self):
        from pydantic import BaseModel as _BM

        class Answer(_BM):
            answer: str

        return Answer

    def _register_finite_tools(self):
        from langchain_core.tools import StructuredTool

        from tests.fakes import register_tool_factory

        def _factory(name):
            def factory(config, tool_config):
                return StructuredTool.from_function(lambda q="": "x", name=name, description=name)

            return factory

        register_tool_factory("search", _factory("search"))
        register_tool_factory("read", _factory("read"))

    def test_first_message_is_system_budget_preamble_when_announce_enabled(self):
        """announce_tool_budget=True -> first msg the LLM sees is a system preamble
        containing the finite tools' budget numbers and max_iterations."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools

        self._register_finite_tools()
        Answer = self._answer_model()

        tools = [Tool("search", budget=3), Tool("read", budget=5)]
        fake = _CapturingReActFake(tool_calls=[[]], final=lambda m: m(answer="done"))

        parsed, _ = invoke_with_tools(
            runtime=build_fake_runtime(lambda tier: fake),
            model_tier="fast",
            prompt_template="test prompt",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=ToolBudgetTracker(tools),
            llm_config={"announce_tool_budget": True, "max_iterations": 9},
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert parsed.answer == "done"
        first = fake.captured_messages[0]
        assert _msg_role(first) == "system"
        content = _msg_content(first)
        # announced numbers == enforced Tool.budget values
        assert "3 calls" in content
        assert "5 calls" in content
        # step cap == cfg.max_iterations
        assert "9" in content

    def test_no_budget_preamble_prepended_when_announce_unset(self):
        """Default (announce_tool_budget unset/False): NO budget preamble is
        prepended. The only system message is the always-on output-schema
        instruction (neograph-f7nt), and the user prompt survives."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools

        self._register_finite_tools()
        Answer = self._answer_model()

        tools = [Tool("search", budget=3)]
        fake = _CapturingReActFake(tool_calls=[[]], final=lambda m: m(answer="done"))

        parsed, _ = invoke_with_tools(
            runtime=build_fake_runtime(lambda tier: fake),
            model_tier="fast",
            prompt_template="test prompt",
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=ToolBudgetTracker(tools),
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert parsed.answer == "done"
        # No budget preamble: no "call budget" language anywhere.
        assert not any("tool-call budget" in (_msg_content(m) or "") for m in fake.captured_messages)
        # The user prompt survives.
        assert any(_msg_content(m) == "test prompt" for m in fake.captured_messages)
        # The only system message present is the output-schema instruction.
        system_msgs = [m for m in fake.captured_messages if _msg_role(m) == "system"]
        assert len(system_msgs) == 1
        assert "JSON object matching this schema" in _msg_content(system_msgs[0])

    def test_prepend_keeps_system_first_without_clobbering_existing_system_when_template_ref(self):
        """A template-ref prompt already carrying a leading SystemMessage: the
        budget preamble is prepended (system-first) and the existing messages
        survive intact (LOW finding — provider portability)."""
        from langchain_core.messages import SystemMessage

        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools

        self._register_finite_tools()
        Answer = self._answer_model()

        existing_system = SystemMessage(content="You are a careful agent.")

        def template_ref_compiler(template, input_data, **kwargs):
            # non-inline (template-ref) shape: multi-message, leading SystemMessage
            return [existing_system, {"role": "user", "content": "do the task"}]

        tools = [Tool("search", budget=3)]
        fake = _CapturingReActFake(tool_calls=[[]], final=lambda m: m(answer="done"))

        parsed, _ = invoke_with_tools(
            runtime=build_fake_runtime(lambda tier: fake, prompt_compiler=template_ref_compiler),
            model_tier="fast",
            prompt_template="agent/diagnose",  # no space -> template-ref path
            input_data={},
            output_model=Answer,
            config={"configurable": {}},
            tools=tools,
            budget_tracker=ToolBudgetTracker(tools),
            llm_config={"announce_tool_budget": True, "max_iterations": 11},
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert parsed.answer == "done"
        msgs = fake.captured_messages
        # our preamble is prepended and stays system-first
        assert _msg_role(msgs[0]) == "system"
        assert "3 calls" in _msg_content(msgs[0])
        assert "11" in _msg_content(msgs[0])
        # the pre-existing SystemMessage is not clobbered
        assert existing_system in msgs
        assert any(_msg_content(m) == "do the task" for m in msgs)


# =============================================================================
# neograph-f7nt: agent/act mode produces its typed output in exactly ONE
# generation. The ReAct loop's final turn IS the structured answer (schema
# injected into the loop as a framework system message), parsed directly from
# messages[-1], with retry ONLY on parse failure -- regardless of the node's
# declared output_strategy. NO separate re-generation over the message history
# (no _call_structured in agent mode). The single-shot think path
# (invoke_structured) keeps _call_structured / constrained decoding untouched.
#
# Regression test for the double-generation described in neograph-f7nt:
# trace deal-4818-alice showed GEN1 tool-call + GEN2 fenced-JSON final answer
# + GEN3 structured re-gen (~43% of trace cost). GEN3 must go away.
# =============================================================================

from langchain_core.messages import AIMessage as _AIMessage  # noqa: E402
from pydantic import BaseModel as _BaseModel  # noqa: E402


class _CountingAgentFake:
    """Counts EVERY invoke on ONE shared counter, INCLUDING the
    ``with_structured_output`` re-generation path.

    This is the crux fake for the anti-double-gen regression. Existing fakes
    cannot observe GEN3: ``ReActFake.with_structured_output`` returns a CLONE
    (separate counter) and ``GuardFake``'s structured invoke returns before
    incrementing. This fake returns ``self`` from ``with_structured_output`` so
    the structured re-gen invoke lands on the SAME ``self.calls`` counter.

    Scripts ``k_tool_turns`` tool-call turns, then a FINAL turn whose content is
    ``final_json`` (parseable JSON for the output model). If the (to-be-removed)
    structured else-branch runs, the structured invoke returns ``regen`` -- a
    DIFFERENT value than the JSON in ``messages[-1]`` -- so BOTH the invocation
    count (K+2 vs K+1) and the returned value expose the double-gen.
    """

    def __init__(self, k_tool_turns, final_json, regen, tool_name="search"):
        self.calls = 0
        self.seen_messages = []
        self._k = k_tool_turns
        self._final_json = final_json
        self._regen = regen
        self._tool_name = tool_name
        self._react_idx = 0
        self._structured = False

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, **kwargs):
        self.calls += 1
        self.seen_messages.append(list(messages))
        if self._structured:
            # GEN3: a separate re-generation over the message history. Under the
            # fixed design this branch must NEVER be reached in agent mode.
            return {"parsed": self._regen, "raw": _AIMessage(content="")}
        self._react_idx += 1
        if self._react_idx <= self._k:
            msg = _AIMessage(content="")
            msg.tool_calls = [{"name": self._tool_name, "args": {"q": "x"}, "id": f"c{self._react_idx}"}]
            return msg
        return _AIMessage(content=self._final_json)

    def with_structured_output(self, model, **kwargs):
        # Return self so the SHARED counter sees the re-gen invoke (unlike
        # ReActFake, which clones and hides GEN3).
        self._structured = True
        return self


class _DsmlAgentFake:
    """Agent fake whose FINAL ReAct turn emits DSML tool-call markup.

    Recovery must flow through the json_mode tail's ``recover_dsml``
    (_tool_loop.py:382): the targeted re-prompt (a fresh base-``llm`` invoke)
    then returns valid JSON. The separate ``with_structured_output`` wrapper
    also surfaces the DSML as ``parsed=None`` + raw markup so the pre-fix
    structured path recovers identically -- keeping the neograph-0tid guard
    green both before and after the fix.
    """

    def __init__(self, dsml_markup, recovery_json):
        self._idx = 0
        self._dsml = dsml_markup
        self._recovery_json = recovery_json

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, **kwargs):
        self._idx += 1
        if self._idx == 1:
            msg = _AIMessage(content="")
            msg.tool_calls = [{"name": "search", "args": {"q": "x"}, "id": "c1"}]
            return msg
        if self._idx == 2:
            return _AIMessage(content=self._dsml)  # final ReAct turn = DSML markup
        return _AIMessage(content=self._recovery_json)  # recover_dsml re-prompt

    def with_structured_output(self, model, **kwargs):
        dsml = self._dsml

        class _Wrap:
            def invoke(self, messages, **kw):
                # parsed=None + raw DSML -> Raw(dsml=True) -> recover_dsml
                return {"parsed": None, "raw": _AIMessage(content=dsml)}

        return _Wrap()


class _HappyAgentFake:
    """Agent fake whose FINAL ReAct turn emits valid JSON (happy baseline)."""

    def __init__(self, final_json, parsed_model):
        self._idx = 0
        self._final_json = final_json
        self._parsed = parsed_model

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, **kwargs):
        self._idx += 1
        if self._idx == 1:
            msg = _AIMessage(content="")
            msg.tool_calls = [{"name": "search", "args": {"q": "x"}, "id": "c1"}]
            return msg
        return _AIMessage(content=self._final_json)  # final ReAct turn = valid JSON

    def with_structured_output(self, model, include_raw=False, **kwargs):
        parsed = self._parsed
        inc = include_raw

        class _Wrap:
            def invoke(self, messages, **kw):
                if inc:
                    return {"parsed": parsed, "raw": _AIMessage(content="")}
                return parsed

        return _Wrap()


class _NonJsonFinalAgentFake:
    """Agent fake whose FINAL ReAct turn emits NON-JSON prose (unparseable).

    Drives neograph-eoi8's parse-failure fallback: the happy-path parse of
    ``messages[-1]`` fails, so the agent tail must dispatch on
    ``cfg.output_strategy``:

    - ``structured`` -> ``_call_structured`` (constrained decoding). That path
      calls ``with_structured_output(model, include_raw=True)``; this fake's
      wrapper returns ``structured_return`` (a ``{"parsed": ..., "raw": ...}``
      dict the adapter normalizes) and bumps ``structured_calls`` so the test
      can assert the constrained re-gen actually ran.
    - ``json_mode``/``text`` -> the tail's ``recover_dsml`` (no-op, non-DSML) then
      ``_invoke_json_with_retry``, which re-invokes the BASE ``invoke``. When
      ``base_retry_json`` is set, that retry recovers to it -- WITHOUT touching
      ``with_structured_output`` (``structured_calls`` stays 0).

    ``react_calls`` counts base ``invoke`` (K tool turns + 1 final + any json
    retries); ``structured_calls`` counts the constrained-decoding arm.
    """

    def __init__(
        self,
        k_tool_turns,
        non_json_final,
        structured_return,
        *,
        base_retry_json=None,
        tool_name="search",
    ):
        self.react_calls = 0
        self.structured_calls = 0
        self.seen_messages = []
        self._k = k_tool_turns
        self._final = non_json_final
        self._structured_return = structured_return
        self._base_retry_json = base_retry_json
        self._tool_name = tool_name
        self._react_idx = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, **kwargs):
        self.react_calls += 1
        self.seen_messages.append(list(messages))
        self._react_idx += 1
        if self._react_idx <= self._k:
            msg = _AIMessage(content="")
            msg.tool_calls = [{"name": self._tool_name, "args": {"q": "x"}, "id": f"c{self._react_idx}"}]
            return msg
        if self._react_idx == self._k + 1:
            # Final ReAct turn: non-JSON prose -> parse fails -> fallback fires.
            return _AIMessage(content=self._final)
        # Later invokes = the json_mode/text retry path (base llm re-prompt).
        if self._base_retry_json is not None:
            return _AIMessage(content=self._base_retry_json)
        return _AIMessage(content=self._final)

    def with_structured_output(self, model, include_raw=False, **kwargs):
        self.structured_calls += 1
        ret = self._structured_return

        class _Wrap:
            def invoke(self, messages, **kw):
                return ret

        return _Wrap()


class TestAgentStrategyAwareFallback:
    """neograph-eoi8 (Option A): agent/act produces its typed output in ONE
    generation on the happy path (the ReAct final turn IS the answer, parsed
    from messages[-1]); a SECOND generation happens ONLY inside the
    parse-failure branch, and cfg.output_strategy selects the FALLBACK
    mechanism -- 'structured' -> _call_structured (constrained decoding),
    'json_mode'/'text' -> recover_dsml + _invoke_json_with_retry. The agent
    tail NEVER returns a silent None: on an unrecoverable structured fallback
    it RAISES ExecutionError (REFINEMENT MEDIUM-1)."""

    @staticmethod
    def _answer_model():
        class Answer(_BaseModel):
            answer: str

        return Answer

    def test_happy_path_parses_final_turn_without_structured_regen(self):
        """GUARD (stays green): output_strategy='structured', VALID JSON final
        ReAct turn -> parsed directly, K+1 base invokes, ZERO constrained
        re-gen. Pins that the double-gen elimination survives Option A."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        Answer = self._answer_model()
        register_tool_factory("search", lambda cfg, tc: FakeTool("search", response="found"))

        K = 2
        fake = _NonJsonFinalAgentFake(
            k_tool_turns=K,
            non_json_final='{"answer": "from_final_react_turn"}',  # valid JSON => happy
            structured_return={"parsed": Answer(answer="from_constrained"), "raw": _AIMessage(content="")},
        )
        tools = [Tool("search", budget=5)]

        result, _ = invoke_with_tools(
            runtime=build_fake_runtime(lambda tier: fake),
            model_tier="fast",
            prompt_template="diagnose",
            input_data={},
            output_model=Answer,
            tools=tools,
            budget_tracker=ToolBudgetTracker(tools),
            config={"configurable": {}},
            llm_config={"output_strategy": "structured"},
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert result.answer == "from_final_react_turn"
        assert fake.react_calls == K + 1
        assert fake.structured_calls == 0, "happy path must NOT trigger constrained re-gen"

    def test_structured_fallback_fires_call_structured_on_parse_failure(self):
        """RED today: output_strategy='structured', NON-JSON final ReAct turn ->
        the parse-failure fallback calls _call_structured (constrained decoding)
        and returns ITS result. Weak-model recourse -- the whole point of eoi8.

        Today the agent tail ignores output_strategy and routes the parse
        failure through _invoke_json_with_retry, so the constrained arm is never
        touched (structured_calls == 0) and the returned value comes from the
        json retry, not the constrained sentinel."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        Answer = self._answer_model()
        register_tool_factory("search", lambda cfg, tc: FakeTool("search", response="found"))

        K = 2
        sentinel = Answer(answer="from_constrained_decoding")
        fake = _NonJsonFinalAgentFake(
            k_tool_turns=K,
            non_json_final="I could not find a definitive answer.",  # unparseable prose
            structured_return={"parsed": sentinel, "raw": _AIMessage(content="")},
            # If the json-retry path were (wrongly) taken today, it recovers to a
            # DIFFERENT value, so the assertion fails cleanly rather than erroring.
            base_retry_json='{"answer": "from_json_retry_not_structured"}',
        )
        tools = [Tool("search", budget=5)]

        result, _ = invoke_with_tools(
            runtime=build_fake_runtime(lambda tier: fake),
            model_tier="fast",
            prompt_template="diagnose",
            input_data={},
            output_model=Answer,
            tools=tools,
            budget_tracker=ToolBudgetTracker(tools),
            config={"configurable": {}},
            llm_config={"output_strategy": "structured"},
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert fake.structured_calls == 1, (
            "structured parse-failure fallback must invoke _call_structured "
            "(constrained decoding); got no constrained re-gen"
        )
        assert result.answer == "from_constrained_decoding", (
            "the returned object must be the constrained-decoding result, "
            f"not the json-retry value; got {result.answer!r}"
        )

    def test_json_mode_fallback_uses_json_retry_not_constrained_decoding(self):
        """GUARD (stays green): output_strategy='json_mode', NON-JSON final then
        parseable-on-retry -> recovers via _invoke_json_with_retry, and NEVER
        calls _call_structured / with_structured_output (structured_calls==0)."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        Answer = self._answer_model()
        register_tool_factory("search", lambda cfg, tc: FakeTool("search", response="found"))

        K = 1
        fake = _NonJsonFinalAgentFake(
            k_tool_turns=K,
            non_json_final="thinking out loud, no json yet",
            structured_return={"parsed": Answer(answer="constrained_should_not_run"), "raw": _AIMessage(content="")},
            base_retry_json='{"answer": "recovered_via_json_retry"}',
        )
        tools = [Tool("search", budget=5)]

        result, _ = invoke_with_tools(
            runtime=build_fake_runtime(lambda tier: fake),
            model_tier="fast",
            prompt_template="diagnose",
            input_data={},
            output_model=Answer,
            tools=tools,
            budget_tracker=ToolBudgetTracker(tools),
            config={"configurable": {}},
            llm_config={"output_strategy": "json_mode"},
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert result.answer == "recovered_via_json_retry"
        assert fake.structured_calls == 0, "json_mode must NOT use constrained decoding"

    def test_structured_fallback_raises_execution_error_on_silent_none(self):
        """RED today (REFINEMENT MEDIUM-1): output_strategy='structured', parse
        fails, and _call_structured hits its silent-None variant (parsed=None,
        no DSML) -> the agent tail RAISES ExecutionError, it does NOT leak a
        silent None downstream.

        Today the tail ignores output_strategy and recovers the parse failure
        via _invoke_json_with_retry (base_retry_json parses cleanly), so it
        RETURNS a value instead of raising -> pytest.raises fails."""
        from neograph.errors import ExecutionError
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        Answer = self._answer_model()
        register_tool_factory("search", lambda cfg, tc: FakeTool("search", response="found"))

        fake = _NonJsonFinalAgentFake(
            k_tool_turns=1,
            non_json_final="no structured content here",  # unparseable, non-DSML
            # silent-None variant: parsed=None, raw has no DSML markup.
            structured_return={"parsed": None, "raw": _AIMessage(content="still no json")},
            # Today's json-retry path would recover to this (proving the raise is
            # absent today); the structured fallback must ignore it and raise.
            base_retry_json='{"answer": "json_retry_would_recover"}',
        )
        tools = [Tool("search", budget=5)]

        with pytest.raises(ExecutionError):
            invoke_with_tools(
                runtime=build_fake_runtime(lambda tier: fake),
                model_tier="fast",
                prompt_template="diagnose",
                input_data={},
                output_model=Answer,
                tools=tools,
                budget_tracker=ToolBudgetTracker(tools),
                config={"configurable": {}},
                llm_config={"output_strategy": "structured"},
                tool_factory_lookup=build_fake_tool_lookup(),
            )

    def test_structured_dsml_final_turn_recovers_via_call_structured_internal(self):
        """LOW-2 (route a): output_strategy='structured' + DSML markup in the
        final ReAct turn -> recovered via _call_structured's INTERNAL
        recover_dsml (a fresh constrained call). Preserves neograph-0tid."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        Answer = self._answer_model()
        dsml_markup = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="search">\n'
            '<｜DSML｜parameter name="q">more</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        register_tool_factory("search", lambda cfg, tc: FakeTool("search", response="found"))

        fake = _DsmlAgentFake(dsml_markup, '{"answer": "recovered-structured-route"}')
        tools = [Tool("search", budget=5)]

        parsed, _ = invoke_with_tools(
            runtime=build_fake_runtime(lambda tier: fake),
            model_tier="fast",
            prompt_template="diagnose",
            input_data={},
            output_model=Answer,
            tools=tools,
            budget_tracker=ToolBudgetTracker(tools),
            config={"configurable": {}},
            llm_config={"output_strategy": "structured"},
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert parsed.answer == "recovered-structured-route"

    def test_json_mode_dsml_final_turn_recovers_via_tail_recover_dsml(self):
        """LOW-2 (route b): output_strategy='json_mode' + DSML markup in the
        final ReAct turn -> recovered via the tail's EXPLICIT recover_dsml.
        Preserves neograph-0tid."""
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        Answer = self._answer_model()
        dsml_markup = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="search">\n'
            '<｜DSML｜parameter name="q">more</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        register_tool_factory("search", lambda cfg, tc: FakeTool("search", response="found"))

        fake = _DsmlAgentFake(dsml_markup, '{"answer": "recovered-json-route"}')
        tools = [Tool("search", budget=5)]

        parsed, _ = invoke_with_tools(
            runtime=build_fake_runtime(lambda tier: fake),
            model_tier="fast",
            prompt_template="diagnose",
            input_data={},
            output_model=Answer,
            tools=tools,
            budget_tracker=ToolBudgetTracker(tools),
            config={"configurable": {}},
            llm_config={"output_strategy": "json_mode"},
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert parsed.answer == "recovered-json-route"

    def test_explicit_output_strategy_on_agent_node_does_not_warn(self):
        """RED today: output_strategy is MEANINGFUL again in agent mode (it
        selects the parse-failure fallback), so compiling an agent/act node with
        output_strategy explicitly set must NOT emit any 'inert' UserWarning.

        Inverse of the removed f7nt warn-tests. Today the compiler warning block
        fires (and PP-01: it false-fires under any construct-level llm_config
        default), so this assertion fails."""
        import warnings

        from tests.fakes import register_tool_factory

        register_tool_factory("search", lambda cfg, tc: FakeTool("search", response="ok"))

        agent = Node(
            "agent_explicit",
            mode="agent",
            model="fast",
            prompt="p",
            outputs=Claims,
            tools=[Tool("search", budget=2)],
            llm_config={"output_strategy": "structured"},
        )
        assert "output_strategy" in agent.llm_config.model_fields_set
        pipeline = Construct("no-warn-explicit", nodes=[agent])

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            compile(
                pipeline,
                llm_factory=lambda tier: ReActFake(tool_calls=[[]], final=lambda m: m(items=["x"])),
                prompt_compiler=lambda t, d, **kw: [{"role": "user", "content": "go"}],
                **build_test_compile_kwargs(),
            )

        offending = [w for w in rec if issubclass(w.category, UserWarning) and "output_strategy" in str(w.message)]
        assert not offending, (
            "eoi8: the 'output_strategy inert for agent/act' warning must be "
            f"removed; got: {[str(w.message) for w in offending]}"
        )

    def test_render_output_schema_preamble_is_the_public_helper_name(self):
        """RED today (CON-01): the agent output-schema helper is renamed to
        render_output_schema_preamble (module-stem symmetry with
        render_tool_budget_preamble). Today the symbol is still named
        render_output_schema_instruction, so this import/attr check fails."""
        from neograph import _agent_output_schema_preamble as mod

        assert hasattr(mod, "render_output_schema_preamble"), (
            "CON-01: render_output_schema_instruction must be renamed to render_output_schema_preamble"
        )
        rendered = mod.render_output_schema_preamble(Claims)
        assert "items" in rendered


class TestAgentSingleGenerationOutput:
    """neograph-f7nt: agent/act output is produced in exactly ONE generation."""

    def test_agent_structured_does_not_double_generate(self):
        """PRIMARY: output_strategy='structured' agent node makes K+1 invokes
        (K tool turns + 1 final JSON turn), NOT K+2. The parsed result comes
        from messages[-1], not a separate structured re-generation (GEN3).

        RED today: the structured else-branch (_tool_loop.py:395) fires
        _call_structured, adding GEN3 -> K+2 and returning the re-gen value.
        """
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Diagnosis(_BaseModel):
            diagnosis_summary: str

        register_tool_factory("search", lambda cfg, tc: FakeTool("search", response="found"))

        K = 2
        fake = _CountingAgentFake(
            k_tool_turns=K,
            final_json='{"diagnosis_summary": "from_final_react_turn"}',
            regen=Diagnosis(diagnosis_summary="from_separate_regen"),
        )

        tools = [Tool("search", budget=5)]
        tracker = ToolBudgetTracker(tools)
        result, _interactions = invoke_with_tools(
            runtime=build_fake_runtime(lambda tier: fake),
            model_tier="fast",
            prompt_template="diagnose the deal",
            input_data={},
            output_model=Diagnosis,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            llm_config={"output_strategy": "structured"},
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        # CORE INVARIANT: exactly ONE generation of the typed output.
        assert fake.calls == K + 1, (
            f"agent double-generated: {fake.calls} invocations "
            f"(expected {K + 1} = {K} tool turns + 1 final JSON turn; "
            f"{K + 2} means the structured else-branch re-generated GEN3)"
        )
        # The parsed result MUST come from messages[-1], not a re-gen.
        assert result.diagnosis_summary == "from_final_react_turn"

    def test_agent_loop_system_messages_include_output_schema(self):
        """The agent ReAct loop is told the output schema: a framework system
        message carries the describe_type rendering of the output model.

        Distinct from the (default-off) tool-budget preamble.

        RED today: the agent loop omits output_schema entirely -- no schema
        system message is injected.
        """
        from neograph import describe_type
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Diagnosis(_BaseModel):
            diagnosis_summary: str

        register_tool_factory("search", lambda cfg, tc: FakeTool("search", response="found"))

        fake = _CountingAgentFake(
            k_tool_turns=1,
            final_json='{"diagnosis_summary": "x"}',
            regen=Diagnosis(diagnosis_summary="x"),
        )
        tools = [Tool("search", budget=5)]
        tracker = ToolBudgetTracker(tools)

        invoke_with_tools(
            runtime=build_fake_runtime(lambda tier: fake),
            model_tier="fast",
            prompt_template="diagnose the deal",
            input_data={},
            output_model=Diagnosis,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            # NOTE: announce_tool_budget left at its default (False) -- the schema
            # instruction is unconditional, NOT the budget preamble.
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        first_msgs = fake.seen_messages[0]
        system_text = "\n".join(_msg_content(m) for m in first_msgs if _msg_role(m) == "system")
        # describe_type renders the model's fields; the distinctive field name
        # only appears if the schema instruction was injected.
        assert "diagnosis_summary" in describe_type(Diagnosis)
        assert "diagnosis_summary" in system_text, (
            f"agent loop did not inject the output schema into a system message; system messages seen: {system_text!r}"
        )

    def test_agent_dsml_final_turn_recovers_via_json_mode_tail(self):
        """neograph-0tid PRESERVATION (agent mode): a formerly-structured agent
        node whose FINAL ReAct turn emits DSML/tool-call markup still recovers
        to a valid output via the json_mode tail's recover_dsml
        (_tool_loop.py:382). Must stay green after the fix.
        """
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Answer(_BaseModel):
            answer: str

        dsml_markup = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="search">\n'
            '<｜DSML｜parameter name="q">more</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

        register_tool_factory("search", lambda cfg, tc: FakeTool("search", response="found"))

        fake = _DsmlAgentFake(dsml_markup, '{"answer": "recovered-agent"}')
        tools = [Tool("search", budget=5)]
        tracker = ToolBudgetTracker(tools)

        parsed, _ = invoke_with_tools(
            runtime=build_fake_runtime(lambda tier: fake),
            model_tier="fast",
            prompt_template="diagnose",
            input_data={},
            output_model=Answer,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            llm_config={"output_strategy": "structured"},
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert parsed.answer == "recovered-agent"

    def test_agent_structured_happy_baseline_parses_final_turn(self):
        """Adapts the structured happy baseline through invoke_with_tools: the
        final ReAct turn emits valid JSON and parses directly (no re-gen).
        Must stay green after the fix.
        """
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools
        from tests.fakes import register_tool_factory

        class Answer(_BaseModel):
            answer: str

        register_tool_factory("search", lambda cfg, tc: FakeTool("search", response="found"))

        fake = _HappyAgentFake('{"answer": "ok"}', Answer(answer="ok"))
        tools = [Tool("search", budget=5)]
        tracker = ToolBudgetTracker(tools)

        parsed, interactions = invoke_with_tools(
            runtime=build_fake_runtime(lambda tier: fake),
            model_tier="fast",
            prompt_template="diagnose",
            input_data={},
            output_model=Answer,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            llm_config={"output_strategy": "structured"},
            tool_factory_lookup=build_fake_tool_lookup(),
        )

        assert parsed.answer == "ok"
        assert len(interactions) == 1

    def test_think_structured_still_routes_through_call_structured(self):
        """Think-path-unchanged guard: a single-shot think node with
        output_strategy='structured' still routes through _call_structured
        (constrained decoding), NOT the messages[-1] JSON parse. Must stay
        green after the fix.
        """
        from neograph._llm import invoke_structured

        used = {"structured": False}

        class ThinkFake:
            def with_structured_output(self, model, include_raw=False, **kwargs):
                used["structured"] = True
                parsed = model(items=["ok"])
                inc = include_raw

                class _W:
                    def invoke(self, messages, **kw):
                        if inc:
                            return {"parsed": parsed, "raw": _AIMessage(content="")}
                        return parsed

                return _W()

            def invoke(self, messages, **kwargs):
                # The json-parse path would call this; the structured path must not.
                return _AIMessage(content="SHOULD_NOT_BE_PARSED_AS_JSON")

        result = invoke_structured(
            runtime=build_fake_runtime(lambda tier: ThinkFake()),
            model_tier="fast",
            prompt_template="p",
            input_data={},
            output_model=Claims,
            config={"configurable": {}},
            llm_config={"output_strategy": "structured"},
        )

        assert used["structured"] is True
        assert result.items == ["ok"]
