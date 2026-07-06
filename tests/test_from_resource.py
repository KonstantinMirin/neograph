"""FromResource DI marker — async resolution twin, end-to-end (neograph-vx9a).

A ``FromResource`` param on a scripted @node fetches + parses a resource at node
entry via the consumer-supplied ``config['configurable']['mcp_resource_fetcher']``.
Because the fetch is awaited, resolution is async: the scripted shim becomes a
coroutine when a resource binding is present, so:

- ``arun()`` awaits the shim → the fetcher runs, the param is parsed into the
  declared type, and the body sees a real value.
- ``run()`` (sync driver) FAILS LOUD via the existing ``ScriptedDispatch.execute``
  awaitable guard (neograph-khff) — a ``NeographError`` naming ``arun()``. The
  direct ``DIBinding.resolve()`` sync fail-loud (a ``ConfigurationError``) is
  pinned separately in ``tests/test_di.py::TestFromResourceResolve``.

Three-surface parity (AGENTS.md neograph-ts7): DI markers are ``Annotated``
metadata classified ONLY by the @node/@merge_fn decorators' ``_classify_di_params``
(same decorator-only property as di_inputs). Declarative ``Node(...)`` and
programmatic ``Node() | Modifier`` carry no ``_param_res``, so they cannot express
a ``FromResource`` binding and are EXEMPT by construction — asserted below.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import pytest
from pydantic import BaseModel

import neograph
from neograph import FromResource, Node, compile, node, run
from neograph._sidecar import _get_param_res
from neograph.di import DIKind
from neograph.errors import NeographError
from tests.fakes import build_test_compile_kwargs
from tests.schemas import Claims, RawText


class Doc(BaseModel):
    title: str
    body: str


def _make_fetcher(sink: list[str], content=b'{"title": "CONTRACT", "body": "B"}',
                  mime="application/json"):
    async def _fetch(uri: str):
        await asyncio.sleep(0)
        sink.append(uri)
        return content, mime

    return _fetch


def _pipeline():
    @node(outputs=RawText)
    def seed() -> RawText:
        return RawText(text="hi")

    @node(outputs=Claims)
    async def assess(
        seed: RawText,
        doc: Annotated[Doc, FromResource("crm://deals/42/contract")],
    ) -> Claims:
        return Claims(items=[seed.text, doc.title])

    from neograph import construct_from_functions

    return construct_from_functions("p", [seed, assess])


class TestFromResourceAsyncResolution:
    """@node surface (the only surface that can carry the Annotated marker)."""

    def test_arun_awaits_fetcher_and_body_sees_parsed_model(self):
        sink: list[str] = []
        cfg = {"configurable": {"mcp_resource_fetcher": _make_fetcher(sink)}}
        graph = compile(_pipeline(), **build_test_compile_kwargs())

        result = asyncio.run(neograph.arun(graph, input={"node_id": "t"}, config=cfg))

        assert sink == ["crm://deals/42/contract"], "fetcher was not awaited with the URI"
        assert result["assess"] == Claims(items=["hi", "CONTRACT"]), (
            "FromResource value did not reach the body parsed into Doc"
        )

    def test_run_sync_fails_loud_naming_arun(self):
        sink: list[str] = []
        cfg = {"configurable": {"mcp_resource_fetcher": _make_fetcher(sink)}}
        graph = compile(_pipeline(), **build_test_compile_kwargs())

        with pytest.raises(NeographError, match="arun"):
            run(graph, input={"node_id": "t"}, config=cfg)


class TestThreeSurfaceExemption:
    """Declarative + programmatic surfaces cannot express FromResource (marker is
    decorator-only) — they carry NO resource binding, so the async shim branch is
    a no-op for them. This is the documented exemption, not a gap."""

    def test_node_decorator_surface_carries_resource_binding(self):
        pipeline = _pipeline()
        assess = next(n for n in pipeline.nodes if getattr(n, "name", "") == "assess")
        binding = _get_param_res(assess).get("doc")
        assert binding is not None and binding.kind is DIKind.FROM_RESOURCE

    def test_declarative_surface_has_no_resource_binding(self):
        n = Node("plain", mode="scripted", scripted_fn="x", inputs=RawText, outputs=Claims)
        assert _get_param_res(n) == {}, (
            "declarative Node cannot carry a FromResource binding — marker is "
            "an Annotated decorator-layer concept"
        )

    def test_programmatic_surface_has_no_resource_binding(self):
        spec = {"mode": "scripted", "scripted_fn": "x", "inputs": RawText, "outputs": Claims}
        n = Node("plain", **spec)
        assert _get_param_res(n) == {}
