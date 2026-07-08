"""Tests for neograph.di — unified DI resolution module.

Tests cover DIKind enum and DIBinding dataclass.
Written TDD-first before implementation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

# ── Schemas for testing ────────────────────────────────────────────────


class RunCtx(BaseModel):
    node_id: str
    project_root: str


class StrictModel(BaseModel):
    x: int


# ── DIKind enum ────────────────────────────────────────────────────────


class TestDIKind:
    """DIKind enum has exactly 7 values (6 original + from_resource), no UPSTREAM."""

    def test_has_seven_values(self):
        from neograph.di import DIKind

        assert len(DIKind) == 7

    def test_values(self):
        from neograph.di import DIKind

        expected = {
            "from_input",
            "from_config",
            "from_input_model",
            "from_config_model",
            "from_state",
            "constant",
            "from_resource",
        }
        assert {k.value for k in DIKind} == expected

    def test_from_resource_excluded_from_template_kinds(self):
        """A fetched resource is not run-input ambient context — it must NOT be a
        prompt template var (no sync di_inputs injection can resolve it anyway)."""
        from neograph.di import DI_TEMPLATE_KINDS, DIKind

        assert DIKind.FROM_RESOURCE not in DI_TEMPLATE_KINDS

    def test_no_upstream(self):
        from neograph.di import DIKind

        values = {k.value for k in DIKind}
        assert "upstream" not in values


# ── DIBinding dataclass ───────────────────────────────────────────────


class TestDIBinding:
    """DIBinding holds name, kind, inner_type, required, payload."""

    def test_construction(self):
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="topic",
            kind=DIKind.FROM_INPUT,
            inner_type=str,
            required=True,
        )
        assert b.name == "topic"
        assert b.kind == DIKind.FROM_INPUT
        assert b.inner_type is str
        assert b.required is True

    def test_resolve_from_input_scalar(self):
        """FROM_INPUT reads from config['configurable'][name]."""
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="topic",
            kind=DIKind.FROM_INPUT,
            inner_type=str,
            required=False,
        )
        config = {"configurable": {"topic": "climate"}}
        assert b.resolve(config) == "climate"

    def test_resolve_from_config_scalar(self):
        """FROM_CONFIG reads from config['configurable'][name]."""
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="rate_limit",
            kind=DIKind.FROM_CONFIG,
            inner_type=int,
            required=False,
        )
        config = {"configurable": {"rate_limit": 42}}
        assert b.resolve(config) == 42

    def test_resolve_from_input_required_missing_raises(self):
        """Required FROM_INPUT param missing from config raises ExecutionError."""
        from neograph.di import DIBinding, DIKind
        from neograph.errors import ExecutionError

        b = DIBinding(
            name="topic",
            kind=DIKind.FROM_INPUT,
            inner_type=str,
            required=True,
        )
        config = {"configurable": {}}
        with pytest.raises(ExecutionError, match="required DI parameter 'topic'"):
            b.resolve(config)

    def test_resolve_from_config_required_missing_raises(self):
        """Required FROM_CONFIG param missing from config raises ExecutionError."""
        from neograph.di import DIBinding, DIKind
        from neograph.errors import ExecutionError

        b = DIBinding(
            name="limiter",
            kind=DIKind.FROM_CONFIG,
            inner_type=object,
            required=True,
        )
        config = {"configurable": {}}
        with pytest.raises(ExecutionError, match="required DI parameter 'limiter'"):
            b.resolve(config)

    def test_resolve_from_input_optional_missing_returns_none(self):
        """Optional FROM_INPUT param missing from config returns None."""
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="topic",
            kind=DIKind.FROM_INPUT,
            inner_type=str,
            required=False,
        )
        config = {"configurable": {}}
        assert b.resolve(config) is None

    def test_resolve_from_input_model_bundled(self):
        """FROM_INPUT_MODEL constructs a BaseModel from scattered config fields."""
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="ctx",
            kind=DIKind.FROM_INPUT_MODEL,
            inner_type=RunCtx,
            required=False,
            model_cls=RunCtx,
        )
        config = {"configurable": {"node_id": "doc-1", "project_root": "/tmp"}}
        result = b.resolve(config)
        assert isinstance(result, RunCtx)
        assert result.node_id == "doc-1"
        assert result.project_root == "/tmp"

    def test_resolve_from_input_model_required_missing_fields_raises(self):
        """Required bundled model with missing fields raises ExecutionError."""
        from neograph.di import DIBinding, DIKind
        from neograph.errors import ExecutionError

        b = DIBinding(
            name="ctx",
            kind=DIKind.FROM_INPUT_MODEL,
            inner_type=RunCtx,
            required=True,
            model_cls=RunCtx,
        )
        config = {"configurable": {}}
        with pytest.raises(ExecutionError, match="missing fields"):
            b.resolve(config)

    def test_resolve_from_input_model_construction_failure_required_raises(self):
        """Required model construction failure raises ExecutionError."""
        from neograph.di import DIBinding, DIKind
        from neograph.errors import ExecutionError

        b = DIBinding(
            name="ctx",
            kind=DIKind.FROM_INPUT_MODEL,
            inner_type=StrictModel,
            required=True,
            model_cls=StrictModel,
        )
        config = {"configurable": {"x": "not_an_int_fail"}}
        with pytest.raises(ExecutionError, match="construction failed"):
            b.resolve(config)

    def test_resolve_from_input_model_construction_failure_optional_returns_none(self):
        """Optional model construction failure returns None."""
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="ctx",
            kind=DIKind.FROM_INPUT_MODEL,
            inner_type=StrictModel,
            required=False,
            model_cls=StrictModel,
        )
        config = {"configurable": {"x": "not_an_int_fail"}}
        assert b.resolve(config) is None

    def test_resolve_constant(self):
        """CONSTANT returns the payload directly."""
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="max_items",
            kind=DIKind.CONSTANT,
            inner_type=int,
            required=False,
            default_value=10,
        )
        assert b.resolve({}) == 10

    def test_resolve_from_state(self):
        """FROM_STATE reads from state by param name."""
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="field",
            kind=DIKind.FROM_STATE,
            inner_type=str,
            required=False,
        )
        state = SimpleNamespace(field="hello")
        assert b.resolve({}, state=state) == "hello"

    def test_resolve_from_state_none_state_returns_none(self):
        """FROM_STATE with state=None returns None."""
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="field",
            kind=DIKind.FROM_STATE,
            inner_type=str,
            required=False,
        )
        assert b.resolve({}, state=None) is None

    def test_resolve_from_state_loop_unwrap(self):
        """FROM_STATE unwraps loop append-lists to latest value."""
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="field",
            kind=DIKind.FROM_STATE,
            inner_type=str,
            required=False,
        )
        state = SimpleNamespace(field=["old", "latest"])
        assert b.resolve({}, state=state) == "latest"

    def test_resolve_from_state_empty_list_returns_none(self):
        """FROM_STATE with empty list (first loop iteration) returns None."""
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="field",
            kind=DIKind.FROM_STATE,
            inner_type=str,
            required=False,
        )
        state = SimpleNamespace(field=[])
        assert b.resolve({}, state=None) is None

    def test_resolve_attr_style_config(self):
        """Config with attr-style configurable (not dict) works."""
        from neograph.di import DIBinding, DIKind

        class AttrConfig:
            configurable = {"my_key": "attr_val"}

        b = DIBinding(
            name="my_key",
            kind=DIKind.FROM_CONFIG,
            inner_type=str,
            required=False,
        )
        assert b.resolve(AttrConfig()) == "attr_val"

    def test_resolve_none_config_returns_none(self):
        """None config returns None for non-required params."""
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="topic",
            kind=DIKind.FROM_INPUT,
            inner_type=str,
            required=False,
        )
        assert b.resolve(None) is None

    def test_resolve_uses_specific_exceptions_not_bare_except(self):
        """Bundled model catch uses (ValidationError, TypeError, ValueError), not bare except."""
        from neograph.di import DIBinding, DIKind
        from neograph.errors import ExecutionError

        # StrictModel requires x: int, "not_an_int_fail" fails pydantic validation
        # which is a ValidationError — this should be caught specifically
        b = DIBinding(
            name="ctx",
            kind=DIKind.FROM_INPUT_MODEL,
            inner_type=StrictModel,
            required=True,
            model_cls=StrictModel,
        )
        config = {"configurable": {"x": "not_an_int_fail"}}
        with pytest.raises(ExecutionError):
            b.resolve(config)


# ── Typed payload fields (neograph-xklx) ─────────────────────────────


class TestDIBindingTypedFields:
    """DIBinding must not use untagged payload: object.

    After xklx, construction uses typed fields: default_value for CONSTANT,
    model_cls for MODEL kinds. The payload field must not exist.
    """

    def test_no_payload_field(self):
        """DIBinding must not have a 'payload' field after xklx."""
        import dataclasses

        from neograph.di import DIBinding

        field_names = {f.name for f in dataclasses.fields(DIBinding)}
        assert "payload" not in field_names, "DIBinding still has 'payload' — should be replaced with typed fields"

    def test_constant_uses_default_value_field(self):
        """CONSTANT kind stores value in default_value, not payload."""
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="max_items",
            kind=DIKind.CONSTANT,
            inner_type=int,
            required=False,
            default_value=10,
        )
        assert b.resolve({}) == 10

    def test_model_uses_model_cls_field(self):
        """MODEL kind stores model class in model_cls, not payload[0]."""
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="ctx",
            kind=DIKind.FROM_INPUT_MODEL,
            inner_type=RunCtx,
            required=False,
            model_cls=RunCtx,
        )
        config = {"configurable": {"node_id": "n1", "project_root": "/p"}}
        result = b.resolve(config)
        assert isinstance(result, RunCtx)

    def test_from_input_no_extra_fields_needed(self):
        """FROM_INPUT needs only name, kind, inner_type, required."""
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="topic",
            kind=DIKind.FROM_INPUT,
            inner_type=str,
            required=True,
        )
        config = {"configurable": {"topic": "test"}}
        assert b.resolve(config) == "test"

    def test_from_state_no_extra_fields_needed(self):
        """FROM_STATE uses inner_type for unwrap, no extra payload."""
        from neograph.di import DIBinding, DIKind

        b = DIBinding(
            name="field",
            kind=DIKind.FROM_STATE,
            inner_type=str,
            required=False,
        )
        state = SimpleNamespace(field="hello")
        assert b.resolve({}, state=state) == "hello"


# ── FromResource marker + async resolution twin (neograph-vx9a) ───────


class Doc(BaseModel):
    title: str
    body: str


def _make_fetcher(content, mime):
    """Build an async fetcher matching the config['configurable']['mcp_resource_fetcher']
    contract: ``async def fetch(uri) -> (content, mime)``."""

    async def _fetch(uri: str):
        import asyncio as _a

        await _a.sleep(0)
        return content, mime

    return _fetch


class TestFromResourceClassification:
    """FromResource is classified once in _classify_di_params → all three
    surfaces inherit (in practice only @node can carry an Annotated marker)."""

    def test_from_resource_param_classified_as_from_resource_kind(self):
        import inspect

        from neograph._di_classify import FromResource, _classify_di_params
        from neograph.di import DIKind

        def f(doc: Annotated[Doc, FromResource("crm://deals/42/contract")]): ...

        from typing import Annotated  # noqa: F401 — used by the string annotation

        sig = inspect.signature(f)
        param_res = _classify_di_params(f, sig, caller_ns=locals())

        assert "doc" in param_res
        b = param_res["doc"]
        assert b.kind is DIKind.FROM_RESOURCE
        assert b.uri == "crm://deals/42/contract"
        assert b.inner_type is Doc

    def test_from_resource_carries_marker_mime_and_parse(self):
        import inspect

        from neograph._di_classify import FromResource, _classify_di_params

        def myparse(content, mime):
            return content

        def f(hist: Annotated[str, FromResource("crm://x", mime="text", parse=myparse)]): ...

        from typing import Annotated  # noqa: F401

        sig = inspect.signature(f)
        param_res = _classify_di_params(f, sig, caller_ns={**locals(), "myparse": myparse})

        b = param_res["hist"]
        assert b.resource_mime == "text"
        assert b.parse_fn is myparse


class TestFromResourceResolve:
    """Sync resolve() FAILS LOUD; aresolve() awaits the fetcher + parses."""

    def _binding(self, inner_type=Doc, *, parse_fn=None, resource_mime=None, uri="crm://x"):
        from neograph.di import DIBinding, DIKind

        return DIBinding(
            name="doc",
            kind=DIKind.FROM_RESOURCE,
            inner_type=inner_type,
            required=True,
            uri=uri,
            parse_fn=parse_fn,
            resource_mime=resource_mime,
        )

    def test_sync_resolve_fails_loud_with_configuration_error(self):
        from neograph.errors import ConfigurationError

        b = self._binding()
        with pytest.raises(ConfigurationError, match="arun"):
            b.resolve({"configurable": {}})

    def test_aresolve_json_validates_into_model(self):
        import asyncio

        fetcher = _make_fetcher(b'{"title": "T", "body": "B"}', "application/json")
        b = self._binding()
        config = {"configurable": {"mcp_resource_fetcher": fetcher}}

        got = asyncio.run(b.aresolve(config))

        assert isinstance(got, Doc)
        assert got == Doc(title="T", body="B")

    def test_aresolve_text_with_explicit_parser(self):
        import asyncio

        def _p(content, mime):
            text = content.decode() if isinstance(content, bytes) else content
            return Doc(title=text.split("|")[0], body=text.split("|")[1])

        fetcher = _make_fetcher("HELLO|WORLD", "text/plain")
        b = self._binding(parse_fn=_p)
        config = {"configurable": {"mcp_resource_fetcher": fetcher}}

        got = asyncio.run(b.aresolve(config))

        assert got == Doc(title="HELLO", body="WORLD")

    def test_aresolve_text_into_str_param_passes_through(self):
        import asyncio

        fetcher = _make_fetcher("raw history text", "text/plain")
        b = self._binding(inner_type=str)
        config = {"configurable": {"mcp_resource_fetcher": fetcher}}

        got = asyncio.run(b.aresolve(config))

        assert got == "raw history text"

    def test_aresolve_text_into_model_without_parser_fails_loud(self):
        """text/* into a BaseModel with no parser must FAIL LOUD — no silent
        LLM parse inside DI resolution (the explicitly-banned hidden cognition)."""
        import asyncio

        from neograph.errors import ConfigurationError

        fetcher = _make_fetcher("some prose", "text/plain")
        b = self._binding()  # inner_type=Doc, no parse_fn
        config = {"configurable": {"mcp_resource_fetcher": fetcher}}

        with pytest.raises(ConfigurationError, match="parse"):
            asyncio.run(b.aresolve(config))

    def test_aresolve_missing_fetcher_fails_loud(self):
        import asyncio

        from neograph.errors import ConfigurationError

        b = self._binding()
        with pytest.raises(ConfigurationError, match="mcp_resource_fetcher"):
            asyncio.run(b.aresolve({"configurable": {}}))


class TestManifestPathNamesHydratingNode:
    """The manifest-driven hydration path (``ref_kind=``) must thread the
    parameter name into ``hydrate_resource_ref`` so an expiry error on the
    MANIFEST path names the hydrating node — parity with the URI-direct path,
    which already carries it (Wave-7 Low; neograph-yc38)."""

    def test_manifest_expiry_error_names_the_param(self):
        import asyncio

        from neograph._state_keys import StateKeys
        from neograph.di import DIBinding, DIKind
        from neograph.errors import ResourceExpiredError
        from neograph.tool import ProducingCall, ResourceRef

        async def _failing_fetch(uri: str):  # read fails -> candidate expiry
            raise RuntimeError("link expired")

        ref = ResourceRef(
            uri="crm://deals/42/emails",
            kind="email-history",
            server="crm",
            producing_call=ProducingCall(
                tool_name="list_emails",
                args={"deal_id": 42},
                producer_idempotent=True,
            ),
        )
        binding = DIBinding(
            name="doc",
            kind=DIKind.FROM_RESOURCE,
            inner_type=Doc,
            required=True,
            ref_kind="email-history",
        )
        # No replayer configured -> read-fail falls through to fail-loud.
        config = {
            "configurable": {
                "mcp_resource_fetcher": _failing_fetch,
                StateKeys.RESOURCE_MANIFEST_INJECT: [ref],
            }
        }

        with pytest.raises(ResourceExpiredError, match="doc") as ei:
            asyncio.run(binding.aresolve(config))
        assert ei.value.node == "doc"


# ── Import path ───────────────────────────────────────────────────────


class TestImportPath:
    """Verify the public import path works."""

    def test_import_from_neograph_di(self):
        from neograph.di import DIBinding, DIKind

        # Verify they're the real types, not re-export stubs
        assert {m.name for m in DIKind} >= {"FROM_INPUT", "FROM_CONFIG", "CONSTANT"}
        assert "resolve" in dir(DIBinding), "DIBinding must expose resolve()"
