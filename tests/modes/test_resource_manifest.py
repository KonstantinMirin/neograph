"""ResourceRef manifest — Layer 1 checkpointed channel + Layer 2 resource-link lift.

neograph-4atf (hgpt Child 3). Pattern: **Typed Resource Manifest with Ephemeral
Hydration**. A typed, frozen ``ResourceRef`` (carrying its producing call) is
lifted from ``resource_link`` tool-result blocks co-located with the existing
``ToolInteraction`` collection, and parked in a ``neo_``-prefixed CHECKPOINTED
state channel that is EXCLUDED from user-facing run output.

Scope pinned here (hydration/expiry is the separate neograph-a5nh):
  * ``_lift_resource_refs(result, tc)`` extracts a ``ResourceRef`` from a result
    containing a ``resource_link`` block and attaches the producing call.
  * a result without ``resource_link`` blocks yields no refs.
  * refs are collected in BOTH the sync ``tools_body`` and async ``atools_body``.
  * the manifest channel is CHECKPOINTED (visible via ``get_state``) yet EXCLUDED
    from ``run()`` output (``neo_`` prefix -> ``_strip_internals``), exactly like
    the sibling ``agent_tool_log`` channel.
"""

from __future__ import annotations

import asyncio
import types

import pydantic
import pytest

import neograph
from neograph import ResourceRef, Tool, compile, construct_from_module, node, run
from neograph._agent_cycle import _lift_resource_refs
from neograph._state_keys import StateKeys, _strip_internals
from neograph.naming import field_name_for
from tests.fakes import (
    FakeTool,
    ReActFake,
    build_test_compile_kwargs,
    configure_fake_llm,
    register_tool_factory,
)
from tests.schemas import Claims


def _resource_link_result(uri: str = "crm://deals/42/emails") -> list:
    """An MCP tool result carrying a text block + a resource_link block, the
    shape langchain-mcp-adapters yields for a resource-emitting tool."""
    return [
        {"type": "text", "text": "here is the email history"},
        {
            "type": "resource_link",
            "uri": uri,
            "name": "email-history",
            "mimeType": "application/json",
            "size": 2048,
        },
    ]


# ── Layer 2: the lift helper (pure, surface-independent) ───────────────────


class TestLiftResourceRefs:
    def test_extracts_ref_with_producing_call_from_resource_link(self):
        tc = {"name": "list_emails", "args": {"deal_id": 42}, "id": "tc1"}
        refs = _lift_resource_refs(_resource_link_result(), tc)

        assert len(refs) == 1
        ref = refs[0]
        assert isinstance(ref, ResourceRef)
        assert ref.uri == "crm://deals/42/emails"
        assert ref.kind == "email-history"
        assert ref.mime == "application/json"
        assert ref.size == 2048
        # THE re-derivation path: the producing call travels with the ref.
        assert ref.producing_call.tool_name == "list_emails"
        assert ref.producing_call.args == {"deal_id": 42}

    def test_ref_is_frozen(self):
        tc = {"name": "list_emails", "args": {}, "id": "tc1"}
        ref = _lift_resource_refs(_resource_link_result(), tc)[0]
        with pytest.raises(pydantic.ValidationError):
            ref.uri = "mutated"  # type: ignore[misc]

    def test_no_refs_when_result_has_no_resource_link(self):
        tc = {"name": "search", "args": {"q": "x"}, "id": "tc1"}
        assert _lift_resource_refs("just a plain string", tc) == []
        assert _lift_resource_refs([{"type": "text", "text": "no links"}], tc) == []


# ── Layer 1 + Layer 2 E2E: collected in both bodies, checkpointed, excluded ─


def _build_agent_pipeline(tool_result):
    register_tool_factory(
        "list_emails", lambda cfg, tc: FakeTool("list_emails", response=tool_result)
    )
    _llm_kw = configure_fake_llm(
        lambda tier: ReActFake(
            tool_calls=[
                [{"name": "list_emails", "args": {"deal_id": 42}, "id": "tc1"}],
                [],  # final turn
            ],
            final=lambda m: Claims(items=["done"]),
        )
    )
    mod = types.ModuleType("test_resource_manifest_mod")

    @node(
        mode="agent",
        outputs=Claims,
        model="fast",
        prompt="test",
        tools=[Tool("list_emails", budget=3)],
    )
    def research() -> Claims: ...

    mod.research = research
    return construct_from_module(mod), _llm_kw


@pytest.mark.parametrize("is_async", [False, True], ids=["run", "arun"])
def test_manifest_checkpointed_and_excluded_from_user_output(is_async: bool):
    from langgraph.checkpoint.memory import MemorySaver

    pipeline, _llm_kw = _build_agent_pipeline(_resource_link_result())
    graph = compile(
        pipeline, checkpointer=MemorySaver(), **build_test_compile_kwargs(), **_llm_kw
    )
    cfg = {"configurable": {"thread_id": f"manifest-{is_async}"}}

    if is_async:
        result = asyncio.run(neograph.arun(graph, input={"node_id": "n1"}, config=cfg))
        state_values = asyncio.run(graph.aget_state(cfg)).values
    else:
        result = run(graph, input={"node_id": "n1"}, config=cfg)
        state_values = graph.get_state(cfg).values

    manifest_key = StateKeys.resource_manifest(field_name_for("research"))

    # CHECKPOINTED: the manifest lives in the persisted checkpoint state.
    manifest = state_values.get(manifest_key)
    assert manifest, f"resource manifest not found in checkpoint: {sorted(state_values)}"
    assert len(manifest) == 1
    ref = manifest[0]
    assert isinstance(ref, ResourceRef)
    assert ref.uri == "crm://deals/42/emails"
    assert ref.producing_call.tool_name == "list_emails"
    assert ref.producing_call.args == {"deal_id": 42}

    # EXCLUDED from user output: the neo_-prefixed channel is stripped from run().
    assert manifest_key not in result, "manifest channel leaked into user-facing output"
    assert not any(k.startswith("neo_") for k in result)
    # And the strip mechanism the channel relies on genuinely removes it.
    assert manifest_key not in _strip_internals(state_values)
