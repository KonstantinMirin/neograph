"""Manifest-driven hydration + layered expiry + templated URIs (neograph-a5nh).

hgpt Child 4. Builds on the v2a manifest channel (neograph-4atf) and the tool
idempotency annotation (neograph-lhc6). Behaviors under test:

  * **Templated URIs** — ``FromResource("crm://deals/{deal_id}?range={range}")``
    interpolated at node entry from ``FromInput`` values (borrows the v1a
    ``resource_reader`` RFC-6570 machinery).
  * **Layered expiry** — hydrating a ``ResourceRef`` does read -> replay the
    producing call -> fail loud (``ResourceExpiredError``). Silent staleness is
    worse than a loud failure.
  * **HARD GATE** — replay is legal ONLY for an idempotent producer. A
    non-idempotent producing tool refuses replay with ``NonIdempotentReplayError``
    (a read may replay, a mutation may not).
  * **max_bytes** — ``FromResource(text, max_bytes=N)`` fails loud at node entry
    BEFORE the blob reaches a prompt.
  * **manifest lint** — a node hydrating a ``kind`` no upstream agent/act node can
    produce fails loud at lint (flat servers fall back to v1a readers).

Three-surface parity: DI markers are ``Annotated`` metadata classified ONLY by
the @node/@merge_fn decorators. Declarative / programmatic surfaces carry no
``_param_res`` and are EXEMPT by construction (same property as vx9a/3q6j).
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import pytest
from pydantic import BaseModel

from neograph import (
    FromInput,
    FromResource,
    ProducingCall,
    ResourceRef,
    Tool,
    lint,
    node,
)
from neograph._sidecar import _get_param_res
from neograph._state_keys import StateKeys
from neograph.di import (
    RESOURCE_FETCHER_KEY,
    RESOURCE_REPLAYER_KEY,
    DIBinding,
    DIKind,
    hydrate_resource_ref,
)
from neograph.errors import (
    ConfigurationError,
    NonIdempotentReplayError,
    ResourceExpiredError,
)


class Doc(BaseModel):
    title: str
    body: str


def _fetcher(mapping: dict[str, tuple], sink: list[str] | None = None):
    """Fetcher over a uri->(content, mime) map; a missing uri raises (expired)."""
    async def _fetch(uri: str):
        await asyncio.sleep(0)
        if sink is not None:
            sink.append(uri)
        if uri not in mapping:
            raise RuntimeError(f"-32002 resource not found: {uri}")
        return mapping[uri]

    return _fetch


def _cfg(**configurable):
    return {"configurable": configurable}


DOC_JSON = b'{"title": "CONTRACT", "body": "B"}'


# ── Templated URIs ──────────────────────────────────────────────────────────


class TestTemplatedUri:
    """A FROM_RESOURCE uri with RFC-6570 vars interpolates from config values."""

    def test_aresolve_interpolates_uri_from_from_input_values(self):
        sink: list[str] = []
        binding = DIBinding(
            name="doc", kind=DIKind.FROM_RESOURCE, inner_type=Doc, required=True,
            uri="crm://deals/{deal_id}/emails{?range}",
        )
        cfg = _cfg(**{
            RESOURCE_FETCHER_KEY: _fetcher(
                {"crm://deals/42/emails?range=1-5": (DOC_JSON, "application/json")}, sink),
            "deal_id": 42, "range": "1-5",
        })
        out = asyncio.run(binding.aresolve(cfg))
        assert out == Doc(title="CONTRACT", body="B")
        assert sink == ["crm://deals/42/emails?range=1-5"]

    def test_static_uri_unchanged_when_no_vars(self):
        binding = DIBinding(
            name="doc", kind=DIKind.FROM_RESOURCE, inner_type=Doc, required=True,
            uri="crm://deals/42/contract",
        )
        cfg = _cfg(**{RESOURCE_FETCHER_KEY: _fetcher(
            {"crm://deals/42/contract": (DOC_JSON, "application/json")})})
        assert asyncio.run(binding.aresolve(cfg)) == Doc(title="CONTRACT", body="B")


# ── max_bytes fail-loud at node entry ─────────────────────────────────────────


class TestMaxBytes:
    """A max_bytes guard fails loud BEFORE the blob is parsed / reaches a prompt."""

    def test_oversized_text_resource_fails_loud_before_parse(self):
        big = b"x" * 5000
        binding = DIBinding(
            name="history", kind=DIKind.FROM_RESOURCE, inner_type=str, required=True,
            uri="crm://deals/42/emails", max_bytes=1000,
        )
        cfg = _cfg(**{RESOURCE_FETCHER_KEY: _fetcher(
            {"crm://deals/42/emails": (big, "text/plain")})})
        with pytest.raises(ConfigurationError) as ei:
            asyncio.run(binding.aresolve(cfg))
        assert "max_bytes" in str(ei.value)
        assert "5000" in str(ei.value)

    def test_within_limit_passes(self):
        binding = DIBinding(
            name="history", kind=DIKind.FROM_RESOURCE, inner_type=str, required=True,
            uri="crm://deals/42/emails", max_bytes=1000,
        )
        cfg = _cfg(**{RESOURCE_FETCHER_KEY: _fetcher(
            {"crm://deals/42/emails": (b"short", "text/plain")})})
        assert asyncio.run(binding.aresolve(cfg)) == "short"


# ── Layered expiry: read -> replay -> fail loud ───────────────────────────────


def _ref(*, idempotent: bool, uri="crm://deals/42/emails", tool="list_emails"):
    return ResourceRef(
        uri=uri, kind="email-history", server="crm",
        producing_call=ProducingCall(
            tool_name=tool, args={"deal_id": 42}, producer_idempotent=idempotent),
    )


def _replay_link_result(fresh_uri: str) -> list:
    return [
        {"type": "text", "text": "re-derived"},
        {"type": "resource_link", "uri": fresh_uri, "name": "email-history",
         "mimeType": "application/json"},
    ]


class TestLayeredExpiry:
    def test_read_success_returns_parsed_model(self):
        cfg = _cfg(**{RESOURCE_FETCHER_KEY: _fetcher(
            {"crm://deals/42/emails": (DOC_JSON, "application/json")})})
        out = asyncio.run(hydrate_resource_ref(_ref(idempotent=True), cfg, Doc))
        assert out == Doc(title="CONTRACT", body="B")

    def test_expired_ref_replays_producing_call_and_reads_fresh_link(self):
        fresh = "crm://deals/42/emails?token=NEW"
        calls: list[tuple] = []

        async def replay(tool_name, args):
            calls.append((tool_name, args))
            return _replay_link_result(fresh)

        cfg = _cfg(**{
            RESOURCE_FETCHER_KEY: _fetcher({fresh: (DOC_JSON, "application/json")}),
            RESOURCE_REPLAYER_KEY: replay,
        })
        out = asyncio.run(hydrate_resource_ref(_ref(idempotent=True), cfg, Doc))
        assert out == Doc(title="CONTRACT", body="B")
        assert calls == [("list_emails", {"deal_id": 42})]

    def test_non_idempotent_producer_refuses_replay_loud(self):
        async def replay(tool_name, args):  # pragma: no cover - must NOT be called
            raise AssertionError("replay attempted on a non-idempotent producer")

        cfg = _cfg(**{
            RESOURCE_FETCHER_KEY: _fetcher({}),  # read fails -> would trigger replay
            RESOURCE_REPLAYER_KEY: replay,
        })
        with pytest.raises(NonIdempotentReplayError) as ei:
            asyncio.run(hydrate_resource_ref(
                _ref(idempotent=False), cfg, Doc, node="assess"))
        assert ei.value.tool_name == "list_emails"
        assert ei.value.node == "assess"

    def test_fail_loud_when_replay_absent(self):
        cfg = _cfg(**{RESOURCE_FETCHER_KEY: _fetcher({})})  # no replayer
        with pytest.raises(ResourceExpiredError) as ei:
            asyncio.run(hydrate_resource_ref(
                _ref(idempotent=True), cfg, Doc, node="assess"))
        assert ei.value.ref.uri == "crm://deals/42/emails"
        assert ei.value.node == "assess"

    def test_fail_loud_when_replay_itself_fails(self):
        async def replay(tool_name, args):
            raise RuntimeError("producing tool gone")

        cfg = _cfg(**{
            RESOURCE_FETCHER_KEY: _fetcher({}),
            RESOURCE_REPLAYER_KEY: replay,
        })
        with pytest.raises(ResourceExpiredError):
            asyncio.run(hydrate_resource_ref(_ref(idempotent=True), cfg, Doc))

    def test_read_parse_error_does_not_trigger_replay(self):
        # Fetch SUCCEEDS but content is invalid JSON: parse error must surface,
        # NOT be masked by a replay attempt.
        async def replay(tool_name, args):  # pragma: no cover
            raise AssertionError("replay attempted on a parse failure")

        cfg = _cfg(**{
            RESOURCE_FETCHER_KEY: _fetcher(
                {"crm://deals/42/emails": (b"not json", "application/json")}),
            RESOURCE_REPLAYER_KEY: replay,
        })
        with pytest.raises(Exception) as ei:  # pydantic ValidationError
            asyncio.run(hydrate_resource_ref(_ref(idempotent=True), cfg, Doc))
        assert not isinstance(ei.value, (ResourceExpiredError, NonIdempotentReplayError))


# ── FromResource(ref=...) marker + manifest lookup ───────────────────────────


class TestManifestDrivenMarker:
    def test_ref_marker_classifies_to_from_resource_with_ref_kind(self):
        @node(outputs=Doc)
        async def assess(
            doc: Annotated[Doc, FromResource(ref="email-history")],
        ) -> Doc:
            return doc

        binding = _get_param_res(assess)["doc"]
        assert binding.kind is DIKind.FROM_RESOURCE
        assert binding.ref_kind == "email-history"
        assert binding.uri is None

    def test_marker_requires_exactly_one_of_uri_or_ref(self):
        with pytest.raises(ValueError):
            FromResource()
        with pytest.raises(ValueError):
            FromResource("crm://x", ref="email-history")

    def test_aresolve_ref_path_looks_up_manifest_and_hydrates(self):
        ref = _ref(idempotent=True)
        binding = DIBinding(
            name="doc", kind=DIKind.FROM_RESOURCE, inner_type=Doc, required=True,
            ref_kind="email-history",
        )
        cfg = _cfg(**{
            RESOURCE_FETCHER_KEY: _fetcher(
                {"crm://deals/42/emails": (DOC_JSON, "application/json")}),
            StateKeys.RESOURCE_MANIFEST_INJECT: [ref],
        })
        assert asyncio.run(binding.aresolve(cfg)) == Doc(title="CONTRACT", body="B")

    def test_aresolve_ref_path_fails_loud_when_kind_absent(self):
        binding = DIBinding(
            name="doc", kind=DIKind.FROM_RESOURCE, inner_type=Doc, required=True,
            ref_kind="email-history",
        )
        cfg = _cfg(**{
            RESOURCE_FETCHER_KEY: _fetcher({}),
            StateKeys.RESOURCE_MANIFEST_INJECT: [],
        })
        with pytest.raises(Exception) as ei:
            asyncio.run(binding.aresolve(cfg))
        assert "email-history" in str(ei.value)


# ── _inject_resource_manifest: state -> config side-channel ───────────────────


class TestInjectResourceManifest:
    def test_collects_refs_from_all_manifest_channels(self):
        from pydantic import create_model

        from neograph._execute import _inject_resource_manifest

        ref = _ref(idempotent=True)
        StateModel = create_model(
            "S",
            **{StateKeys.resource_manifest("agent1"): (list, [ref])},
        )
        state = StateModel()

        @node(outputs=Doc)
        async def assess(doc: Annotated[Doc, FromResource(ref="email-history")]) -> Doc:
            return doc

        cfg = _cfg()
        out = _inject_resource_manifest(state, assess, cfg)
        assert out["configurable"][StateKeys.RESOURCE_MANIFEST_INJECT] == [ref]

    def test_no_injection_when_node_has_no_ref_binding(self):
        from pydantic import create_model

        from neograph._execute import _inject_resource_manifest

        StateModel = create_model("S", **{StateKeys.resource_manifest("a"): (list, [])})

        @node(outputs=Doc)
        def plain() -> Doc:
            return Doc(title="x", body="y")

        cfg = _cfg(foo=1)
        assert _inject_resource_manifest(StateModel(), plain, cfg) is cfg


# ── _lift_resource_refs stamps idempotency + ttl_ms ───────────────────────────


class TestLiftStampsProducerMetadata:
    def _result(self, ttl=None):
        block = {"type": "resource_link", "uri": "crm://deals/42/emails",
                 "name": "email-history", "mimeType": "application/json"}
        if ttl is not None:
            block["ttlMs"] = ttl
        return [block]

    def test_producer_idempotent_stamped_from_tool_flag(self):
        from neograph._agent_cycle import _lift_resource_refs

        tc = {"name": "list_emails", "args": {"deal_id": 42}, "id": "tc1"}
        idem = _lift_resource_refs(self._result(), tc, idempotent=True)[0]
        assert idem.producing_call.producer_idempotent is True
        non = _lift_resource_refs(self._result(), tc, idempotent=False)[0]
        assert non.producing_call.producer_idempotent is False

    def test_default_lift_is_conservative_non_idempotent(self):
        from neograph._agent_cycle import _lift_resource_refs

        tc = {"name": "mutate", "args": {}, "id": "tc1"}
        ref = _lift_resource_refs(self._result(), tc)[0]
        assert ref.producing_call.producer_idempotent is False

    def test_ttl_ms_lifted_when_present(self):
        from neograph._agent_cycle import _lift_resource_refs

        tc = {"name": "list_emails", "args": {}, "id": "tc1"}
        ref = _lift_resource_refs(self._result(ttl=60000), tc, idempotent=True)[0]
        assert ref.ttl_ms == 60000


# ── Lint: manifest hydration kind unmatched ───────────────────────────────────


class TestManifestHydrationLint:
    def test_error_when_no_upstream_producer_of_resource_links(self):
        @node(outputs=Doc)
        async def assess(doc: Annotated[Doc, FromResource(ref="email-history")]) -> Doc:
            return doc

        from neograph import construct_from_functions

        c = construct_from_functions("p", [assess])
        issues = lint(c)
        kinds = {i.kind for i in issues}
        assert "resource_hydration_kind_unmatched" in kinds
        issue = next(i for i in issues if i.kind == "resource_hydration_kind_unmatched")
        assert issue.required is True
        assert "email-history" in issue.message

    def test_no_error_when_agent_producer_present(self):
        search = Tool("search_crm", budget=3)

        @node(outputs=Doc, mode="agent", prompt="find ${x}", model="m", tools=[search])
        async def acquire(x: Annotated[str, FromInput]) -> Doc: ...

        @node(outputs=Doc)
        async def assess(
            acquire: Doc,
            doc: Annotated[Doc, FromResource(ref="email-history")],
        ) -> Doc:
            return doc

        from neograph import construct_from_functions

        c = construct_from_functions("p", [acquire, assess])
        issues = lint(c)
        assert "resource_hydration_kind_unmatched" not in {i.kind for i in issues}
