"""resource_reader() typed domain-reader helper + read_blob escape hatch (neograph-2dtk).

R1: a generic ``read_resource(uri) -> bytes`` tool is the untyped-tool trap.
``resource_reader(name, uri_template=..., output_model=..., ...)`` turns a
*(uri template + output model + name)* into a properly typed async ``BaseTool``:

- Args schema is derived from the uri_template's RFC 6570 placeholders, so the
  agent (or caller) supplies typed parameters.
- At call time it resolves the consumer-owned fetcher from
  ``config['configurable']['mcp_resource_fetcher']`` (async), reads the
  interpolated URI, and parses the blob into ``output_model``.
- It emits a ``StructuredTool`` with a ``coroutine`` and NO ``func`` — already
  async-only, so ``is_async_only_tool()`` fires and the existing
  ``tool_requires_async_driver`` lint covers it for FREE. ZERO new tool infra.
- ``ToolInteraction.typed_result`` carries the parsed model (not a repr string),
  so a downstream node consuming ``tool_log`` gets structured data.

``read_blob(uri) -> BlobResult`` is the typed escape hatch for genuinely opaque
content — the exception, not the default.
"""

from __future__ import annotations

import asyncio
import types as _types

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

import neograph
from neograph import Node, compile, construct_from_module, lint, node, run
from neograph.tool import (
    BlobResult,
    ToolInteraction,
    is_async_only_tool,
    read_blob,
    resource_reader,
)
from tests.fakes import (
    ReActFake,
    build_test_compile_kwargs,
    configure_fake_llm,
)
from tests.schemas import Claims


class Deal(BaseModel):
    deal_id: str
    stage: str


def _json_fetcher(sink: list[str], payload=b'{"deal_id": "42", "stage": "won"}',
                  mime="application/json"):
    async def _fetch(uri: str):
        await asyncio.sleep(0)
        sink.append(uri)
        return payload, mime

    return _fetch


class TestResourceReaderShape:
    """The emitted tool is a typed, async-only StructuredTool."""

    def test_emits_async_only_structured_tool(self):
        t = resource_reader(
            "read_deal", uri_template="crm://deals/{deal_id}",
            output_model=Deal, description="Read one CRM deal.",
        )
        assert isinstance(t, StructuredTool)
        assert t.name == "read_deal"
        assert t.coroutine is not None and t.func is None
        assert is_async_only_tool(t) is True

    def test_args_schema_derived_from_uri_template_placeholders(self):
        t = resource_reader(
            "read_emails", uri_template="crm://deals/{deal_id}/emails?range={range}",
            output_model=Deal, description="Read a slice of a deal's emails.",
        )
        assert set(t.args_schema.model_fields) == {"deal_id", "range"}


class TestResourceReaderInvoke:
    """Directly driving the tool coroutine fetches + parses into output_model."""

    def test_ainvoke_fetches_interpolated_uri_and_parses_json(self):
        sink: list[str] = []
        t = resource_reader(
            "read_deal", uri_template="crm://deals/{deal_id}",
            output_model=Deal, description="Read one CRM deal.",
        )
        cfg = {"configurable": {"mcp_resource_fetcher": _json_fetcher(sink)}}

        got = asyncio.run(t.ainvoke({"deal_id": "42"}, config=cfg))

        assert sink == ["crm://deals/42"], "URI template was not interpolated"
        assert got == Deal(deal_id="42", stage="won")

    def test_explicit_parse_callable_wins(self):
        t = resource_reader(
            "read_deal", uri_template="crm://deals/{deal_id}",
            output_model=Deal, description="d",
            parse=lambda content, mime: Deal(deal_id="X", stage="parsed"),
        )
        sink: list[str] = []
        cfg = {"configurable": {"mcp_resource_fetcher": _json_fetcher(sink)}}

        got = asyncio.run(t.ainvoke({"deal_id": "9"}, config=cfg))

        assert got == Deal(deal_id="X", stage="parsed")

    def test_missing_fetcher_fails_loud(self):
        from neograph.errors import ConfigurationError

        t = resource_reader(
            "read_deal", uri_template="crm://deals/{deal_id}",
            output_model=Deal, description="d",
        )
        with pytest.raises(ConfigurationError, match="mcp_resource_fetcher"):
            asyncio.run(t.ainvoke({"deal_id": "1"}, config={"configurable": {}}))


class TestReadBlob:
    """read_blob is the typed escape hatch — a BlobResult, never raw bytes."""

    def test_read_blob_is_async_only_tool_with_uri_arg(self):
        assert is_async_only_tool(read_blob) is True
        assert set(read_blob.args_schema.model_fields) == {"uri"}

    def test_read_blob_wraps_text_content(self):
        async def _fetch(uri: str):
            await asyncio.sleep(0)
            return "hello world", "text/plain"

        cfg = {"configurable": {"mcp_resource_fetcher": _fetch}}
        got = asyncio.run(read_blob.ainvoke({"uri": "crm://x/raw"}, config=cfg))

        assert isinstance(got, BlobResult)
        assert got.uri == "crm://x/raw"
        assert got.text == "hello world"
        assert got.mime == "text/plain"
        assert got.size == len("hello world")


class TestResourceReaderLintCoverage:
    """The async-only shape means the existing tool_requires_async_driver lint
    fires for free — no new lint rule needed."""

    def test_lint_flags_resource_reader_as_async_driver_required(self):
        t = resource_reader(
            "read_deal", uri_template="crm://deals/{deal_id}",
            output_model=Deal, description="d",
        )
        pipeline = Node("scan", mode="agent", outputs=Claims, model="fast",
                        prompt="test/scan", tools=[t])
        from neograph import Construct

        issues = lint(Construct("p", nodes=[pipeline]))

        kinds = {(i.param, i.kind) for i in issues}
        assert ("read_deal", "tool_requires_async_driver") in kinds


class TestResourceReaderAgentModeEndToEnd:
    """Agent-mode hydration: the LLM calls the reader; typed_result carries the
    parsed model so a downstream tool_log consumer gets structured data."""

    def test_typed_result_carries_model_under_arun(self):
        sink: list[str] = []
        t = resource_reader(
            "read_deal", uri_template="crm://deals/{deal_id}",
            output_model=Deal, description="Read one CRM deal.",
        )
        mod = _types.ModuleType("test_resource_reader_agent_mod")

        @node(mode="agent",
              outputs={"result": Claims, "tool_log": list[ToolInteraction]},
              model="fast", prompt="test/scan", tools=[t])
        def scan() -> Claims: ...

        mod.scan = scan
        pipeline = construct_from_module(mod, name="p_reader")

        react = ReActFake(
            tool_calls=[[{"name": "read_deal", "args": {"deal_id": "42"}, "id": "c1"}], []],
            final=lambda m: m(items=["done"]),
            output_model=Claims,
        )
        _llm_kw = configure_fake_llm(lambda tier: react)
        cfg = {"configurable": {"mcp_resource_fetcher": _json_fetcher(sink)}}
        graph = compile(pipeline, **build_test_compile_kwargs(), **_llm_kw)

        result = asyncio.run(neograph.arun(graph, input={"node_id": "n1"}, config=cfg))

        assert sink == ["crm://deals/42"]
        tool_log = result["scan_tool_log"]
        assert tool_log, "expected a ToolInteraction recorded for read_deal"
        assert any(isinstance(ti.typed_result, Deal) and ti.typed_result.stage == "won"
                   for ti in tool_log), "typed_result did not carry the parsed Deal model"

    def test_async_only_reader_under_sync_run_fails_loud(self):
        t = resource_reader(
            "read_deal", uri_template="crm://deals/{deal_id}",
            output_model=Deal, description="d",
        )
        mod = _types.ModuleType("test_resource_reader_sync_mod")

        @node(mode="agent", outputs=Claims, model="fast", prompt="test/scan", tools=[t])
        def scan() -> Claims: ...

        mod.scan = scan
        pipeline = construct_from_module(mod, name="p_reader_sync")
        react = ReActFake(
            tool_calls=[[{"name": "read_deal", "args": {"deal_id": "42"}, "id": "c1"}], []],
            final=lambda m: m(items=["done"]),
            output_model=Claims,
        )
        _llm_kw = configure_fake_llm(lambda tier: react)
        cfg = {"configurable": {"mcp_resource_fetcher": _json_fetcher([])}}
        graph = compile(pipeline, **build_test_compile_kwargs(), **_llm_kw)

        from neograph.errors import NeographError

        with pytest.raises(NeographError, match="arun"):
            run(graph, input={"node_id": "n1"}, config=cfg)


# ── Idempotency / replay-safety annotation (neograph-lhc6) ─────────────────────
#
# Replaying a producing tool call to re-derive an expired resource is only safe
# for idempotent/read-only tools. `Tool.idempotent` is the replay-safety gate.
# A bare Tool defaults to the CONSERVATIVE non-idempotent (must not be replayed);
# `resource_reader()` (read-only by nature) defaults idempotent=True.


class TestToolIdempotencyAnnotation:
    """`Tool.idempotent` defaults conservative; resource_reader defaults safe."""

    def test_bare_tool_defaults_non_idempotent(self):
        from neograph import Tool

        assert Tool("mutate_deal").idempotent is False

    def test_tool_idempotent_flag_round_trips_through_pipe(self):
        from neograph import Tool

        spec = Tool("read_deal", idempotent=True)
        assert spec.idempotent is True
        # frozen model_copy preserves the field through modifier chains
        assert spec.model_copy().idempotent is True

    def test_resource_reader_marks_tool_idempotent(self):
        t = resource_reader(
            "read_deal", uri_template="crm://deals/{deal_id}",
            output_model=Deal, description="d",
        )
        # The reader stashes its idempotency in metadata so from_base_tool lifts it.
        assert (t.metadata or {}).get("ng_idempotent") is True

    def test_from_base_tool_lifts_idempotent_from_resource_reader(self):
        from neograph import Tool

        t = resource_reader(
            "read_deal", uri_template="crm://deals/{deal_id}",
            output_model=Deal, description="d",
        )
        spec = Tool.from_base_tool(t)
        assert spec.idempotent is True

    def test_from_base_tool_defaults_non_idempotent_for_bare_base_tool(self):
        from langchain_core.tools import StructuredTool
        from pydantic import create_model

        from neograph import Tool

        bare = StructuredTool(
            name="bare", description="d",
            args_schema=create_model("bare_Args", x=(str, ...)),
            coroutine=None, func=lambda x: x,
        )
        assert Tool.from_base_tool(bare).idempotent is False


class TestNonIdempotentReplayError:
    """The typed refusal hydration replay raises for a non-idempotent producer."""

    def test_error_type_exists_and_is_raisable(self):
        from neograph.errors import NeographError, NonIdempotentReplayError

        assert issubclass(NonIdempotentReplayError, NeographError)
        err = NonIdempotentReplayError.of("mutate_deal", node="writer")
        assert err.tool_name == "mutate_deal"
        assert err.node == "writer"
        with pytest.raises(NonIdempotentReplayError, match="mutate_deal"):
            raise err

    def test_error_is_exported_from_package_root(self):
        import neograph

        assert hasattr(neograph, "NonIdempotentReplayError")
