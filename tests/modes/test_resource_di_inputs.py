"""FROM_RESOURCE as an LLM-mode template var — async di_inputs twin (neograph-3q6j).

Follow-on to neograph-vx9a. An LLM-mode node (think/agent/act) never runs its
body, so a ``FromResource`` param can only ever serve as a prompt template var
(e.g. ``{history}`` fed from a fetched resource). The value reaches the template
through the ``di_inputs`` config side-channel — but the fetch is AWAITED, so:

- ``arun()`` (async driver) routes through ``_ainject_di_inputs`` which awaits
  ``DIBinding.aresolve`` for FROM_RESOURCE bindings and stashes the fetched value
  under ``StateKeys.DI_INPUTS`` — the resource text reaches the prompt compiler.
- ``run()`` (sync driver) routes through ``_inject_di_inputs`` which CANNOT await;
  it FAILS LOUD (``ConfigurationError`` naming the param + node) rather than
  silently dropping the template var (the R2 silent-no-op hole this ticket closes).

All three LLM modes are in lockstep: think, agent, and act each WORK on arun()
and FAIL LOUD on run(). FROM_RESOURCE stays excluded from ``DI_TEMPLATE_KINDS``
(the sync set); the async twin gates on DI_TEMPLATE_KINDS-or-FROM_RESOURCE.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import pytest
from pydantic import BaseModel

import neograph
from neograph import (
    FromInput,
    FromResource,
    Tool,
    compile,
    construct_from_functions,
    node,
    run,
)
from neograph._dispatch import _ainject_di_inputs, _inject_di_inputs
from neograph._sidecar import _get_param_res
from neograph._state_keys import StateKeys
from neograph.errors import ConfigurationError
from tests.fakes import (
    ReActFake,
    StructuredFake,
    build_fake_llm_kwargs,
    build_test_compile_kwargs,
    register_tool_factory,
)


class Out(BaseModel):
    ok: bool = True


def _fetcher(text="FETCHED HISTORY", mime="text/plain"):
    async def _fetch(uri: str):
        await asyncio.sleep(0)
        return text, mime

    return _fetch


def _cfg(**extra):
    return {"configurable": {"mcp_resource_fetcher": _fetcher(), "node_id": "t", **extra}}


def _recording_compiler(sink: list):
    """A **kwargs prompt compiler (accepts di_inputs via _ACCEPT_ALL) that records
    the di_inputs it was handed so a test can prove the resource text arrived."""

    def compiler(template, input_data, **kw):
        sink.append(kw.get("di_inputs"))
        return [{"role": "user", "content": str(kw.get("di_inputs"))}]

    return compiler


def _think_node(prompt="tmpl"):
    @node(outputs=Out, mode="think", model="fast", prompt=prompt)
    def analyze(history: Annotated[str, FromResource("crm://history")]) -> Out: ...

    return analyze


def _think_pipeline(prompt="tmpl"):
    return construct_from_functions("p", [_think_node(prompt)])


def _node_of(pipeline, name):
    return next(n for n in pipeline.nodes if getattr(n, "name", "") == name)


# ── Injector-level units (shared by think + agent/act seams) ────────────────


class TestSyncInjectorFailsLoudOnResource:
    """The sync ``_inject_di_inputs`` cannot await a fetch — a FROM_RESOURCE
    binding on an LLM-mode node must FAIL LOUD naming the param and the node,
    never silently drop it. This one injector guards think-sync AND agent/act-sync
    (both seams call it on the sync driver)."""

    def test_sync_inject_raises_naming_param_and_node(self):
        analyze = _node_of(_think_pipeline(), "analyze")
        assert _get_param_res(analyze)["history"].kind.value == "from_resource"

        with pytest.raises(ConfigurationError) as exc:
            _inject_di_inputs(analyze, {"configurable": {}})

        msg = str(exc.value)
        assert "history" in msg, "fail-loud must name the offending param"
        assert "analyze" in msg, "fail-loud must name the node"
        assert "arun" in msg, "fail-loud must point at the async driver"

    def test_sync_inject_still_passes_from_input_through(self):
        """Regression: a node with ONLY template-safe kinds still injects them
        synchronously — the resource guard must not break the euyh path."""

        @node(outputs=Out, mode="think", model="fast", prompt="tmpl")
        def leaf(domain: Annotated[str, FromInput]) -> Out: ...

        pipe = construct_from_functions("p", [leaf])
        n = _node_of(pipe, "leaf")
        out = _inject_di_inputs(n, {"configurable": {"domain": "oncology"}})
        assert out["configurable"][StateKeys.DI_INPUTS] == {"domain": "oncology"}


class TestAsyncInjectorResolvesResource:
    """The async twin awaits FROM_RESOURCE and stashes it under DI_INPUTS."""

    def test_async_inject_stashes_fetched_value(self):
        analyze = _node_of(_think_pipeline(), "analyze")
        out = asyncio.run(_ainject_di_inputs(analyze, _cfg()))
        assert out["configurable"][StateKeys.DI_INPUTS] == {"history": "FETCHED HISTORY"}

    def test_async_inject_copy_not_mutate(self):
        analyze = _node_of(_think_pipeline(), "analyze")
        original = _cfg()
        out = asyncio.run(_ainject_di_inputs(analyze, original))
        assert StateKeys.DI_INPUTS not in original["configurable"], (
            "injection must copy-not-mutate (idempotent per superstep)"
        )
        assert StateKeys.DI_INPUTS in out["configurable"]

    def test_async_inject_mixes_resource_and_from_input(self):
        @node(outputs=Out, mode="think", model="fast", prompt="tmpl")
        def mix(
            domain: Annotated[str, FromInput],
            history: Annotated[str, FromResource("crm://history")],
        ) -> Out: ...

        pipe = construct_from_functions("p", [mix])
        n = _node_of(pipe, "mix")
        out = asyncio.run(_ainject_di_inputs(n, _cfg(domain="oncology")))
        di = out["configurable"][StateKeys.DI_INPUTS]
        assert di == {"domain": "oncology", "history": "FETCHED HISTORY"}


# ── Think mode end-to-end ────────────────────────────────────────────────────


class TestThinkModeResourceTemplateVar:
    def test_arun_think_resource_reaches_template(self):
        """WORK path: under arun() the fetched text is injected as di_inputs and
        reaches the (di_inputs-aware) prompt compiler for a think node."""
        sink: list = []
        llm_kw = build_fake_llm_kwargs(
            lambda tier: StructuredFake(lambda m: m()),
            _recording_compiler(sink),
        )
        graph = compile(_think_pipeline(), **llm_kw, **build_test_compile_kwargs())

        result = asyncio.run(neograph.arun(graph, input={"node_id": "t"}, config=_cfg()))

        assert result["analyze"] == Out(ok=True)
        assert sink and sink[-1] == {"history": "FETCHED HISTORY"}, (
            "the fetched resource text did not reach the prompt compiler di_inputs"
        )

    def test_run_think_resource_fails_loud(self):
        """FAIL-LOUD path: the sync driver cannot await the fetch."""
        sink: list = []
        llm_kw = build_fake_llm_kwargs(
            lambda tier: StructuredFake(lambda m: m()),
            _recording_compiler(sink),
        )
        graph = compile(_think_pipeline(), **llm_kw, **build_test_compile_kwargs())

        with pytest.raises(ConfigurationError, match="history"):
            run(graph, input={"node_id": "t"}, config=_cfg())


# ── Agent / act mode end-to-end ──────────────────────────────────────────────


class TestAgentActModeResourceTemplateVar:
    """Agent/act bypass ThinkDispatch (they compile to the ReAct cycle), but the
    cycle's turn-prep twins now split injection: sync -> _inject_di_inputs (fail
    loud), async -> _ainject_di_inputs (await + stash). Same lockstep as think."""

    def _agent_pipeline(self, mode="agent"):
        register_tool_factory("noop", lambda cfg, tc: _FakeNoopTool())

        @node(
            mode=mode,
            outputs=Out,
            model="fast",
            prompt="tmpl",
            tools=[Tool(name="noop", budget=1)],
        )
        def act_node(history: Annotated[str, FromResource("crm://history")]) -> Out: ...

        return construct_from_functions("p", [act_node])

    def test_arun_agent_resource_reaches_template(self):
        sink: list = []
        fake = ReActFake(tool_calls=[[]], final=lambda m: m())
        llm_kw = build_fake_llm_kwargs(lambda tier: fake, _recording_compiler(sink))
        graph = compile(self._agent_pipeline("agent"), **llm_kw, **build_test_compile_kwargs())

        asyncio.run(neograph.arun(graph, input={"node_id": "t"}, config=_cfg()))

        assert any(d == {"history": "FETCHED HISTORY"} for d in sink if d), (
            "the fetched resource text did not reach the agent cycle prompt compiler"
        )

    def test_run_act_resource_fails_loud(self):
        sink: list = []
        fake = ReActFake(tool_calls=[[]], final=lambda m: m())
        llm_kw = build_fake_llm_kwargs(lambda tier: fake, _recording_compiler(sink))
        graph = compile(self._agent_pipeline("act"), **llm_kw, **build_test_compile_kwargs())

        with pytest.raises(ConfigurationError, match="history"):
            run(graph, input={"node_id": "t"}, config=_cfg())


class _FakeNoopTool:
    name = "noop"
    description = "noop"

    def invoke(self, *a, **k):
        return "ok"
