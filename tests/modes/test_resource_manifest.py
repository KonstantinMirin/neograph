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
    register_tool_factory("list_emails", lambda cfg, tc: FakeTool("list_emails", response=tool_result))
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
    graph = compile(pipeline, checkpointer=MemorySaver(), **build_test_compile_kwargs(), **_llm_kw)
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


# ── Layer 3: hydration replay must re-derive by ref.kind (neograph-m9sj) ────


def _multi_link_replay_result() -> list:
    """A producing call (``get_deal``) that emits TWO ``resource_link`` blocks —
    activity FIRST, then email — the real multi-corpus CRM shape that exposed
    neograph-m9sj. The FIRST link is NOT the email ref we are healing."""
    return [
        {"type": "text", "text": "deal summary"},
        {"type": "resource_link", "uri": "crm://deals/42/activity/fresh", "name": "activity-history"},
        {"type": "resource_link", "uri": "crm://deals/42/emails/fresh", "name": "email-history"},
    ]


class _CorpusPage(pydantic.BaseModel):
    corpus: str


class TestHydrationReDerivationIsKindAware:
    """neograph-m9sj: on the replay path an expired ``ResourceRef`` must
    re-derive to the ``resource_link`` whose KIND matches ``ref.kind`` — never
    blindly the first link. A multi-link producer (activity BEFORE email) used to
    heal an email ref to the activity blob and silently parse it into the wrong
    model.

    Hydration is surface-agnostic — ``FromResource`` DI resolves identically for
    the @node, declarative, and programmatic surfaces — so this single
    hydration-level regression covers all three."""

    def test_uri_for_kind_selects_matching_link_not_first(self):
        from neograph._content_blocks import _first_resource_link_uri, _resource_link_uri_for_kind

        result = _multi_link_replay_result()
        # first-link (kind-blind) lands on ACTIVITY — the old, wrong target.
        assert _first_resource_link_uri(result) == "crm://deals/42/activity/fresh"
        # kind-aware selection lands on the correct corpus for each kind.
        assert _resource_link_uri_for_kind(result, "email-history") == "crm://deals/42/emails/fresh"
        assert _resource_link_uri_for_kind(result, "activity-history") == "crm://deals/42/activity/fresh"
        assert _resource_link_uri_for_kind(result, "no-such-kind") is None

    def test_multi_link_producer_heals_to_the_ref_kind_not_first_link(self):
        from neograph.di import RESOURCE_FETCHER_KEY, RESOURCE_REPLAYER_KEY, hydrate_resource_ref
        from neograph.tool import ProducingCall

        expired_uri = "crm://deals/42/emails"  # the email ref's stale uri
        fresh = {
            "crm://deals/42/emails/fresh": ('{"corpus": "emails"}', "application/json"),
            "crm://deals/42/activity/fresh": ('{"corpus": "activity"}', "application/json"),
        }

        async def fetcher(uri):
            if uri == expired_uri:
                raise RuntimeError("resource_link expired")  # any read failure = candidate expiry
            return fresh[uri]

        async def replayer(tool_name, args):
            assert tool_name == "get_deal"
            return _multi_link_replay_result()

        ref = ResourceRef(
            uri=expired_uri,
            kind="email-history",
            server="crm",
            producing_call=ProducingCall(tool_name="get_deal", args={"deal_id": 42}, producer_idempotent=True),
        )
        config = {"configurable": {RESOURCE_FETCHER_KEY: fetcher, RESOURCE_REPLAYER_KEY: replayer}}

        healed = asyncio.run(hydrate_resource_ref(ref, config, _CorpusPage))
        # Pre-fix: replay took the FIRST link (activity) and healed to the WRONG
        # corpus. Kind-aware re-derivation reads the email link.
        assert healed.corpus == "emails"

    def test_heals_when_replay_blocks_carry_anyurl_uris(self):
        """The SHIPPED replayer (mcp_resource_fetcher) returns RAW mcp
        ``ResourceLink`` OBJECTS whose ``.uri`` is a pydantic ``AnyUrl`` — NOT a
        str. The kind derivation must not assume a str uri (``"://" in AnyUrl``
        raises TypeError, which the replay try-block would mask as a spurious
        ResourceExpiredError). Faithful to the real replay-path block shape, which
        the dict-based cases above do not exercise (neograph-m9sj)."""
        from pydantic import AnyUrl

        from neograph.di import RESOURCE_FETCHER_KEY, RESOURCE_REPLAYER_KEY, hydrate_resource_ref
        from neograph.tool import ProducingCall

        expired_uri = "crm://deals/42/emails"
        fresh = {
            "crm://deals/42/emails/fresh": ('{"corpus": "emails"}', "application/json"),
            "crm://deals/42/activity/fresh": ('{"corpus": "activity"}', "application/json"),
        }

        async def fetcher(uri):
            if uri == expired_uri:
                raise RuntimeError("resource_link expired")
            return fresh[uri]  # keyed by str — the re-derived uri must be coerced

        def _link(uri: str, name: str):
            # object block (getattr access) with an AnyUrl uri — the raw session shape.
            return types.SimpleNamespace(type="resource_link", uri=AnyUrl(uri), name=name)

        async def replayer(tool_name, args):
            return [
                types.SimpleNamespace(type="text", text="deal summary"),
                _link("crm://deals/42/activity/fresh", "activity-history"),
                _link("crm://deals/42/emails/fresh", "email-history"),
            ]

        ref = ResourceRef(
            uri=expired_uri,
            kind="email-history",
            server="crm",
            producing_call=ProducingCall(tool_name="get_deal", args={"deal_id": 42}, producer_idempotent=True),
        )
        config = {"configurable": {RESOURCE_FETCHER_KEY: fetcher, RESOURCE_REPLAYER_KEY: replayer}}

        healed = asyncio.run(hydrate_resource_ref(ref, config, _CorpusPage))
        assert healed.corpus == "emails"
