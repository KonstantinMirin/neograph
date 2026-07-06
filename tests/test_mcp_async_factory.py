"""Async tool factories on the arun() path — per-run MCP identity (w74k.3.1).

Pins the Core Invariant of neograph-w74k.3.1: a registered tool *factory* that is
a coroutine function (``async def factory(config, tool_config)``) must be AWAITED
on the async driver (``arun()``) and yield working per-run tools, while the SAME
async factory driven by the SYNC driver (``run()``) must FAIL LOUD with
``ConfigurationError`` (driver/config mismatch — mirroring the async-only-tool
error at ``_agent_cycle.py:352``).

There is exactly ONE tool-factory call site, ``_tool_loop.py`` inside the shared
prep. The fix factors it into a sync ``_instantiate_tools`` (fails loud on a
coroutine / awaitable-returning factory) + an async ``_ainstantiate_tools``
(awaits it), reached via ``_prepare_tool_loop`` / ``_aprepare_tool_loop`` and the
``_build_turn_prep`` / ``_abuild_turn_prep`` twins.

Coverage:
- ``TestAsyncToolFactoryDualPath`` — the 6-cell three-surface × run/arun matrix
  (AGENTS.md neograph-ts7): the async-factory path reachable from @node(tools=),
  declarative ``Node(tools=)``, and programmatic ``Node(...)`` construction, each
  × {arun -> success, run -> fail-loud}.
- ``TestTokenNeverLeaks`` — guards (a)+(b): a per-run auth token minted from
  ``config['configurable']`` never enters checkpoint state, schema fingerprint,
  captured logs, or ToolInteraction args/result (never-passthrough).
- ``TestPerRunIdentityBinding`` — integration: two run-context identities each
  carry the correct per-identity token to the tool.

Driven end-to-end through the outermost surface (``compile()`` + ``run()`` /
``arun()``), NOT the tool-loop internals — the LLM is the only mocked external
(``ReActFake``); the tool factory and returned tool are real.
"""

from __future__ import annotations

import asyncio
import types as _types

import pytest
import structlog
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import neograph
from neograph import (
    Construct,
    Node,
    Tool,
    compile,
    construct_from_module,
    node,
    run,
)
from neograph.errors import ConfigurationError
from neograph.tool import ToolInteraction
from tests.fakes import (
    ReActFake,
    build_test_compile_kwargs,
    configure_fake_llm,
    register_tool_factory,
)
from tests.schemas import Claims

SENTINEL_TOKEN = "SENTINEL-JWT-8f3c1a9e-never-persist"


# ── tool factories ────────────────────────────────────────────────────────────

def _make_recording_async_factory(name: str, calls: list[str]):
    """A COROUTINE-FUNCTION tool factory (``async def``) — the MCP-style shape.

    A real per-run MCP factory awaits a token provider / builds a
    ``MultiServerMCPClient`` connection before returning tools; here the factory
    is async and returns a plain working ``StructuredTool`` whose invocation is
    recorded, so a successful ``arun()`` proves the factory was AWAITED into a
    real tool and that tool was actually driven.
    """

    async def _factory(config, tool_config):
        await asyncio.sleep(0)  # models the async work (token mint / client build)

        def _run(text: str) -> str:
            calls.append(text)
            return f"echo:{text}"

        return StructuredTool.from_function(
            func=_run, name=name, description="per-run MCP-style tool"
        )

    return _factory


def _make_token_reading_sync_factory(name: str, observed: list[str]):
    """Sync twin of the token-reading factory (for the sync-driver leak test).
    The token-never-leaks invariant is driver-agnostic; a sync factory exercises
    the run() path (an async factory would fail loud under run())."""

    def _factory(config, tool_config):
        token = (config or {}).get("configurable", {}).get("mcp_auth", {}).get("server", "")
        observed.append(token)

        def _run(text: str) -> str:
            _ = token
            return f"echo:{text}"

        return StructuredTool.from_function(
            func=_run, name=name, description="per-run MCP-style tool (auth-bound, sync)"
        )

    return _factory


def _make_token_reading_async_factory(name: str, observed: list[str]):
    """Async factory that MINTS per-run identity: reads the auth token from
    ``config['configurable']['mcp_auth']['server']`` and closes over it. The
    returned tool records the token it was built with (proving per-run binding)
    but its RESULT never contains the token (neograph carries, does not expose).
    """

    async def _factory(config, tool_config):
        await asyncio.sleep(0)
        token = (config or {}).get("configurable", {}).get("mcp_auth", {}).get("server", "")
        observed.append(token)

        def _run(text: str) -> str:
            # The token authenticates the (stubbed) upstream call; the tool result
            # returned to the LLM/state must NOT carry it.
            _ = token
            return f"echo:{text}"

        return StructuredTool.from_function(
            func=_run, name=name, description="per-run MCP-style tool (auth-bound)"
        )

    return _factory


# ── surface builders (three construction surfaces) ────────────────────────────

def _build_node_surface(tool_name: str, outputs=Claims):
    """Surface 1 — ``@node`` decorator (runs through _build_construct_from_decorated)."""
    mod = _types.ModuleType(f"test_async_factory_node_{tool_name}_mod")

    @node(mode="agent", outputs=outputs, model="fast", prompt="test/scan",
          tools=[Tool(tool_name, budget=0)])
    def scan() -> Claims: ...

    mod.scan = scan
    return construct_from_module(mod, name="p_node")


def _build_declarative_surface(tool_name: str, outputs=Claims):
    """Surface 2 — declarative ``Node(...)`` assembled directly into a Construct."""
    return Construct("p_declarative", nodes=[
        Node("scan", mode="agent", outputs=outputs, model="fast",
             prompt="test/scan", tools=[Tool(tool_name, budget=0)]),
    ])


def _build_programmatic_surface(tool_name: str, outputs=Claims):
    """Surface 3 — programmatic/runtime construction (LLM-driven kwargs splat)."""
    spec = {
        "mode": "agent", "outputs": outputs, "model": "fast",
        "prompt": "test/scan", "tools": [Tool(tool_name, budget=0)],
    }
    n = Node("scan", **spec)
    return Construct("p_programmatic", nodes=[n])


_SURFACES = {
    "node": _build_node_surface,
    "declarative": _build_declarative_surface,
    "programmatic": _build_programmatic_surface,
}


def _react_fake(tool_name: str) -> ReActFake:
    """Script one tool call to ``tool_name`` then a final structured answer."""
    return ReActFake(
        tool_calls=[
            [{"name": tool_name, "args": {"text": "hi"}, "id": "c1"}],
            [],
        ],
        final=lambda m: m(items=["done"]),
        output_model=Claims,
    )


@pytest.mark.parametrize("surface", list(_SURFACES))
class TestAsyncToolFactoryDualPath:
    """6-cell three-surface parity. Core Invariant: an async (coroutine) tool
    factory is AWAITED under arun() and yields working per-run tools; the same
    factory under sync run() fails loud with ConfigurationError pointing at arun().
    """

    def test_async_tool_factory_awaited_on_arun_yields_working_tools(self, surface):
        """arun() awaits the coroutine factory and drives the real per-run tool."""
        calls: list[str] = []
        register_tool_factory("mcp_echo", _make_recording_async_factory("mcp_echo", calls))
        pipeline = _SURFACES[surface]("mcp_echo")
        _llm_kw = configure_fake_llm(lambda tier: _react_fake("mcp_echo"))
        graph = compile(pipeline, **build_test_compile_kwargs(), **_llm_kw)

        result = asyncio.run(neograph.arun(graph, input={"node_id": "n1"}))

        assert calls == ["hi"], (
            f"[{surface}] async tool factory was not awaited into a working per-run "
            "tool (coroutine object passed to bind_tools instead)"
        )
        assert result["scan"] == Claims(items=["done"])

    def test_async_tool_factory_under_sync_run_fails_loud(self, surface):
        """The SAME async factory under sync run() raises a clear ConfigurationError
        naming arun() — on ALL THREE sync surface cells, not a silent coroutine tool."""
        calls: list[str] = []
        register_tool_factory("mcp_echo", _make_recording_async_factory("mcp_echo", calls))
        pipeline = _SURFACES[surface]("mcp_echo")
        _llm_kw = configure_fake_llm(lambda tier: _react_fake("mcp_echo"))
        graph = compile(pipeline, **build_test_compile_kwargs(), **_llm_kw)

        with pytest.raises(ConfigurationError, match="arun"):
            run(graph, input={"node_id": "n1"})


class TestTokenNeverLeaks:
    """Guards (a)+(b): a per-run auth token minted from config never enters
    checkpoint state, schema fingerprint, captured logs, or ToolInteraction
    args/result. Runtime-sentinel test (not a vacuous AST guard)."""

    def _assert_no_leak(self, *, observed, result, state_values, cap_logs):
        # The token WAS live and non-empty (positive control — else the absence
        # assertions below would be vacuous). The factory is re-invoked per
        # superstep (two-lifetime rule §5), so observed has one entry per prep.
        assert observed and all(t == SENTINEL_TOKEN for t in observed), (
            f"factory did not receive the per-run token from config: {observed!r}"
        )
        # Non-vacuity meta-check: a planted token WOULD be caught by repr-scan.
        assert SENTINEL_TOKEN in repr({"planted": SENTINEL_TOKEN})

        state_dump = repr(state_values)  # channels incl. _neo_*_fingerprint
        assert SENTINEL_TOKEN not in state_dump, "token leaked into checkpoint state / fingerprint"
        assert SENTINEL_TOKEN not in repr(cap_logs), "token leaked into structlog output"
        assert SENTINEL_TOKEN not in repr(result), "token leaked into the run result"

        # ToolInteraction args/result carry no token (never-passthrough, guard b).
        tool_log = result.get("scan_tool_log")
        assert tool_log, "expected a ToolInteraction to have been recorded"
        for interaction in tool_log:
            assert SENTINEL_TOKEN not in repr(interaction.args)
            assert SENTINEL_TOKEN not in repr(interaction.result)
            assert SENTINEL_TOKEN not in repr(getattr(interaction, "typed_result", None))

    def test_token_absent_across_persisted_surfaces_sync(self, tmp_path):
        observed: list[str] = []
        register_tool_factory("mcp_echo", _make_token_reading_sync_factory("mcp_echo", observed))
        outputs = {"result": Claims, "tool_log": list[ToolInteraction]}
        pipeline = _build_node_surface("mcp_echo", outputs=outputs)
        _llm_kw = configure_fake_llm(lambda tier: _react_fake("mcp_echo"))
        cfg = {"configurable": {"thread_id": "leak-run", "mcp_auth": {"server": SENTINEL_TOKEN}}}

        with SqliteSaver.from_conn_string(str(tmp_path / "leak-run.db")) as saver:
            graph = compile(pipeline, checkpointer=saver, **build_test_compile_kwargs(), **_llm_kw)
            with structlog.testing.capture_logs() as cap_logs:
                result = run(graph, input={"node_id": "n1"}, config=cfg)
            state_values = graph.get_state(cfg).values

        self._assert_no_leak(observed=observed, result=result,
                             state_values=state_values, cap_logs=cap_logs)

    def test_token_absent_across_persisted_surfaces_async(self, tmp_path):
        observed: list[str] = []
        register_tool_factory("mcp_echo", _make_token_reading_async_factory("mcp_echo", observed))
        outputs = {"result": Claims, "tool_log": list[ToolInteraction]}
        pipeline = _build_node_surface("mcp_echo", outputs=outputs)
        _llm_kw = configure_fake_llm(lambda tier: _react_fake("mcp_echo"))
        cfg = {"configurable": {"thread_id": "leak-arun", "mcp_auth": {"server": SENTINEL_TOKEN}}}

        async def _drive():
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "leak-arun.db")) as saver:
                graph = compile(pipeline, checkpointer=saver, **build_test_compile_kwargs(), **_llm_kw)
                with structlog.testing.capture_logs() as cap_logs:
                    result = await neograph.arun(graph, input={"node_id": "n1"}, config=cfg)
                state_values = (await graph.aget_state(cfg)).values
            return result, state_values, cap_logs

        result, state_values, cap_logs = asyncio.run(_drive())
        self._assert_no_leak(observed=observed, result=result,
                             state_values=state_values, cap_logs=cap_logs)


class TestPerRunIdentityBinding:
    """Integration: two distinct run-context identities each carry the correct
    per-identity token to the tool (per-run MCP identity, the whole point)."""

    def test_two_identities_each_carry_their_own_token(self, tmp_path):
        per_run: list[tuple[str, str]] = []  # (token_the_factory_saw, echoed_text)

        def _factory_for(observed_token_sink):
            async def _factory(config, tool_config):
                await asyncio.sleep(0)
                token = (config or {}).get("configurable", {}).get("mcp_auth", {}).get("server", "")
                observed_token_sink.append(token)

                def _run(text: str) -> str:
                    per_run.append((token, text))
                    return f"echo:{text}"

                return StructuredTool.from_function(
                    func=_run, name="mcp_echo", description="auth-bound"
                )
            return _factory

        sink: list[str] = []
        register_tool_factory("mcp_echo", _factory_for(sink))
        _llm_kw = configure_fake_llm(lambda tier: _react_fake("mcp_echo"))

        token_a = "TOKEN-operator-A"
        token_b = "TOKEN-operator-B"

        async def _drive_identity(thread, token):
            db = str(tmp_path / f"{thread}.db")
            cfg = {"configurable": {"thread_id": thread, "mcp_auth": {"server": token}}}
            async with AsyncSqliteSaver.from_conn_string(db) as saver:
                pipeline = _build_node_surface("mcp_echo")
                graph = compile(pipeline, checkpointer=saver, **build_test_compile_kwargs(), **_llm_kw)
                await neograph.arun(graph, input={"node_id": thread}, config=cfg)

        for thread, token in (("id-a", token_a), ("id-b", token_b)):
            asyncio.run(_drive_identity(thread, token))

        # Tool-level binding: each operator's tool call carried that operator's
        # token, with NO cross-contamination.
        assert (token_a, "hi") in per_run, f"operator A's token not carried: {per_run!r}"
        assert (token_b, "hi") in per_run, f"operator B's token not carried: {per_run!r}"
        assert {t for t, _ in per_run} == {token_a, token_b}, (
            f"tool saw an unexpected/cross-contaminated token: {per_run!r}"
        )
        # Factory-level (re-invoked per superstep): identity A's factory calls all
        # saw A, identity B's all saw B — the two runs never bled tokens.
        assert set(sink) == {token_a, token_b}, f"per-run token binding wrong: {sink!r}"
        assert token_b not in sink[: sink.index(token_b)], "operator B's token appeared during run A"
        assert token_a not in sink[sink.index(token_b):], "operator A's token appeared during run B"


class TestCancellationDuringAsyncFactory:
    """Re-scoped neograph-51tr: an SSE consumer disconnecting mid-astream cancels
    the run and leaves the checkpoint consistent — for an agent whose MCP tool is
    built by an async tool factory (the w74k.3.1 path).

    NO live-MCP-session-disposal assertion: w74k.3.1 established neograph owns no
    MCP session (guard tests/test_guards_mcp_session_ownership.py), so there is
    nothing for neograph to dispose on cancel. The remaining invariant is that
    cancelling mid-factory-await raises CancelledError cleanly AND does not corrupt
    the checkpoint — a fresh re-arun of the same thread_id still completes.
    """

    def test_cancel_mid_async_factory_leaves_checkpoint_consistent(self, tmp_path):
        gate = asyncio.Event()
        entered: list[bool] = []

        def _make_gated_factory():
            async def _factory(config, tool_config):
                # Park here — models a slow per-run token mint / MCP client build.
                entered.append(True)
                await gate.wait()
                return StructuredTool.from_function(
                    func=lambda text: f"echo:{text}", name="mcp_echo", description="x"
                )
            return _factory

        db = str(tmp_path / "cancel-mcp.db")
        cfg = {"configurable": {"thread_id": "cancel-mcp"}}

        async def _drive():
            register_tool_factory("mcp_echo", _make_gated_factory())
            _llm_kw = configure_fake_llm(lambda tier: _react_fake("mcp_echo"))

            async with AsyncSqliteSaver.from_conn_string(db) as saver:
                graph = compile(
                    _build_node_surface("mcp_echo"),
                    checkpointer=saver, **build_test_compile_kwargs(), **_llm_kw,
                )
                task = asyncio.create_task(
                    neograph.arun(graph, input={"node_id": "cx"}, config=cfg)
                )

                # Poll until arun parks INSIDE the async tool factory's await.
                for _ in range(300):
                    if entered or task.done():
                        break
                    await asyncio.sleep(0.005)
                assert entered and not task.done(), (
                    "arun did not park in the async tool factory — cannot test a "
                    "mid-factory cancellation"
                )

                # (a) SSE disconnect == cancel the consuming task: raises cleanly.
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=2.0)
                gate.set()  # release the (already-cancelled) gated coroutine

                # (b) Checkpoint consistent/resumable: a fresh ungated factory
                # re-arun-s the SAME thread_id under the SAME open saver to
                # completion (no corrupt checkpoint, no torn saver connection).
                calls: list[str] = []
                register_tool_factory(
                    "mcp_echo", _make_recording_async_factory("mcp_echo", calls)
                )
                resume_kw = configure_fake_llm(lambda tier: _react_fake("mcp_echo"))
                resume_graph = compile(
                    _build_node_surface("mcp_echo"),
                    checkpointer=saver, **build_test_compile_kwargs(), **resume_kw,
                )
                return await asyncio.wait_for(
                    neograph.arun(resume_graph, input={"node_id": "cx"}, config=cfg),
                    timeout=5.0,
                )

        completed = asyncio.run(_drive())
        assert completed["scan"] == Claims(items=["done"]), (
            "re-arun of the same thread_id after a mid-factory cancel did not "
            "complete cleanly — the cancel left the checkpoint inconsistent"
        )
