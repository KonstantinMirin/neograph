"""MCP tool-call transport resilience (neograph-uixi1) — TDD red.

Pins the ``_with_transport_resilience`` coroutine wrapper contract in
``src/neograph_mcp/_client.py`` (the battery, NOT core — core stays MCP-free and
retry_policy-free per the ownership guards):

1. A flaky-transport tool (pre-send CONNECT failure N-1 times, then success)
   retries with bounded backoff and succeeds within ``max_attempts``.
2. An isError/tool-domain result (``ToolException`` — the adapter raises
   ``_MCPToolExecutionError``, a ``ToolException`` subclass, for server
   ``isError=True``) surfaces on FIRST return: never retried, never classified
   as transport.
3. A NON-idempotent tool is NOT replayed after an AMBIGUOUS post-send failure
   (read timeout / mid-flight — the call may have partially executed).
   Idempotency reaches the wrapper via the FACTORY-CALL CHANNEL — the
   ``tool_config`` the factory already receives from core's
   ``factory(config, tool_spec.config)`` — single-source, no duplicated flag.
   Missing/unknown idempotency = NON-idempotent (never replay).
4. A retry is the SAME logical call: ``ToolBudgetTracker`` counts it ONCE
   (retries inside the wrapped coroutine are invisible to the budget).
5. A per-call timeout bounds a hung tool call (parity with ``mcp_session``'s
   ``timeout=30``; the factory path today inherits the adapter's 300s).

Conventions follow ``tests/test_mcp_battery.py`` (extra-gated skipif) and
``tests/test_mcp_tools.py`` (StructuredTool doubles; agent drive via
``graph.graph.ainvoke`` for async-only tools). The wrapper does not exist yet:
imports stay in test bodies so collection succeeds and each test FAILS (red).
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
import types as _types
from typing import Any

import pytest
from langchain_core.tools import StructuredTool, ToolException

from neograph import Tool, compile, construct_from_module, node
from tests.fakes import (
    ReActFake,
    build_fake_llm_kwargs,
    build_test_compile_kwargs,
    register_tool_factory,
)
from tests.schemas import Claims

_HAS_MCP = bool(importlib.util.find_spec("mcp")) and bool(importlib.util.find_spec("langchain_mcp_adapters"))
requires_mcp = pytest.mark.skipif(not _HAS_MCP, reason="requires the mcp extra (mcp + langchain-mcp-adapters)")

_CFG = {"configurable": {}}
_INPUT = {"node_id": "t"}


# ── doubles (builders, not inline construction) ───────────────────────────────


def _failing_then_ok_tool(
    name: str,
    *,
    failures: int,
    exc_factory: Any,
    result: str = "ok",
) -> tuple[StructuredTool, dict[str, int]]:
    """An async-only StructuredTool whose coroutine raises ``exc_factory()`` for
    the first ``failures`` underlying attempts, then returns ``result``.

    Models a langchain-mcp-adapters tool over a flaky transport. Returns the
    tool and a mutable ``{"n": attempts}`` counter so tests can assert the exact
    underlying attempt count (logical-call vs transport-attempt distinction).
    """
    attempts = {"n": 0}

    async def _run() -> str:
        attempts["n"] += 1
        if attempts["n"] <= failures:
            raise exc_factory()
        return result

    tool = StructuredTool.from_function(coroutine=_run, name=name, description="flaky-transport MCP-style double")
    return tool, attempts


def _hung_tool(name: str, *, hang_seconds: float = 30.0) -> tuple[StructuredTool, dict[str, int]]:
    """An async-only StructuredTool whose coroutine never returns within the
    per-call timeout — models a wedged MCP server / dead read."""
    attempts = {"n": 0}

    async def _run() -> str:
        attempts["n"] += 1
        await asyncio.sleep(hang_seconds)
        return "never"

    tool = StructuredTool.from_function(coroutine=_run, name=name, description="hung MCP-style double")
    return tool, attempts


def _alternating_flaky_tool(name: str, *, result: str = "found") -> tuple[StructuredTool, dict[str, int]]:
    """Fails every ODD underlying attempt with a pre-send connect error and
    succeeds on every EVEN one — so EVERY logical call needs exactly one retry."""
    attempts = {"n": 0}

    async def _run() -> str:
        attempts["n"] += 1
        if attempts["n"] % 2 == 1:
            raise ConnectionRefusedError("transport blip: connection refused")
        return result

    tool = StructuredTool.from_function(coroutine=_run, name=name, description="alternating flaky double")
    return tool, attempts


def _wrap(tool: Any, **kwargs: Any) -> Any:
    """Import-and-apply the wrapper under test. Import lives HERE (test body
    path) so the missing symbol is a test FAILURE, not a collection error."""
    from neograph_mcp._client import _with_transport_resilience

    return _with_transport_resilience(tool, **kwargs)


# ── 1. flaky transport: bounded retry then success ────────────────────────────


@requires_mcp
class TestTransportRetryWithinBound:
    """A pre-send CONNECT-phase transport failure retries (regardless of
    idempotency — nothing was sent) up to ``max_attempts``, with backoff."""

    def test_flaky_connect_error_retries_within_bound_and_succeeds(self):
        tool, attempts = _failing_then_ok_tool(
            "flaky_echo", failures=2, exc_factory=lambda: ConnectionRefusedError("connection refused")
        )
        # tool_config carries NO idempotency: connect-phase failures retry anyway.
        wrapped = _wrap(tool, timeout=5.0, max_attempts=3, backoff=0.01, tool_config={})

        result = asyncio.run(wrapped.ainvoke({}))

        assert result == "ok"
        assert attempts["n"] == 3, f"expected 2 failed + 1 successful attempt, saw {attempts['n']}"

    def test_transport_failure_beyond_bound_surfaces_after_max_attempts(self):
        tool, attempts = _failing_then_ok_tool(
            "always_down", failures=99, exc_factory=lambda: ConnectionRefusedError("connection refused")
        )
        wrapped = _wrap(tool, timeout=5.0, max_attempts=3, backoff=0.01, tool_config={})

        with pytest.raises(ConnectionError):
            asyncio.run(wrapped.ainvoke({}))

        assert attempts["n"] == 3, f"retry must stop at max_attempts=3, saw {attempts['n']}"


# ── 2. isError / tool-domain results are NEVER retried ────────────────────────


@requires_mcp
class TestIsErrorNeverRetried:
    """A server isError raises ``_MCPToolExecutionError`` (a ``ToolException``)
    from the adapter coroutine — a real RESULT the model must see. The wrapper
    must re-raise it immediately: call count stays 1."""

    def _tool_exception(self) -> ToolException:
        return ToolException("server said isError: bad args")

    def _adapter_is_error(self) -> Exception:
        from langchain_mcp_adapters.tools import _MCPToolExecutionError

        # The adapter's constructor takes the CONVERTED CONTENT BLOCKS of the
        # isError result (tools.py:116) — not a message string; it derives the
        # message via _summarize_tool_error.
        return _MCPToolExecutionError([{"type": "text", "text": "server said isError: boom"}])

    @pytest.mark.parametrize("exc_name", ["tool_exception", "adapter_is_error"])
    def test_is_error_surfaces_on_first_return_without_retry(self, exc_name: str):
        exc_factory = getattr(self, f"_{exc_name}")
        tool, attempts = _failing_then_ok_tool("domain_error", failures=99, exc_factory=exc_factory)
        # Even an IDEMPOTENT declaration must not cause an isError retry:
        # classification (result vs transport) trumps idempotency.
        wrapped = _wrap(tool, timeout=5.0, max_attempts=3, backoff=0.01, tool_config={"idempotent": True})

        with pytest.raises(ToolException):
            asyncio.run(wrapped.ainvoke({}))

        assert attempts["n"] == 1, f"isError result must surface on first return, saw {attempts['n']} attempts"


# ── 3. idempotency gate on AMBIGUOUS post-send failures ───────────────────────


@requires_mcp
class TestAmbiguousFailureIdempotencyGate:
    """A call-time ``TimeoutError`` (read timeout / mid-flight) is AMBIGUOUS —
    the call may have partially executed. Replay only when the factory-call
    channel (``tool_config``) declares ``idempotent=True``; missing/unknown
    means NON-idempotent and the failure surfaces on the first attempt."""

    @pytest.mark.parametrize(
        "tool_config",
        [
            pytest.param({}, id="missing-idempotency"),
            pytest.param(None, id="no-tool-config"),
            pytest.param({"idempotent": None}, id="unknown-idempotency"),
            pytest.param({"idempotent": False}, id="declared-non-idempotent"),
        ],
    )
    def test_non_idempotent_tool_is_not_replayed_after_ambiguous_failure(self, tool_config: Any):
        tool, attempts = _failing_then_ok_tool(
            "mutating_call", failures=1, exc_factory=lambda: TimeoutError("read timed out mid-flight")
        )
        wrapped = _wrap(tool, timeout=5.0, max_attempts=3, backoff=0.01, tool_config=tool_config)

        with pytest.raises(TimeoutError):
            asyncio.run(wrapped.ainvoke({}))

        assert attempts["n"] == 1, (
            f"a non-idempotent tool must NEVER be replayed after an ambiguous "
            f"post-send failure, saw {attempts['n']} attempts"
        )

    def test_idempotent_tool_is_replayed_after_ambiguous_failure(self):
        tool, attempts = _failing_then_ok_tool(
            "safe_read", failures=1, exc_factory=lambda: TimeoutError("read timed out mid-flight")
        )
        wrapped = _wrap(tool, timeout=5.0, max_attempts=3, backoff=0.01, tool_config={"idempotent": True})

        result = asyncio.run(wrapped.ainvoke({}))

        assert result == "ok"
        assert attempts["n"] == 2, f"idempotent tool should retry the ambiguous failure once, saw {attempts['n']}"


# ── 4. a retry is the SAME logical call: budget counts it once ────────────────


@requires_mcp
class TestBudgetCountsLogicalCallOnce:
    """Two logical tool calls, each needing one internal transport retry
    (4 underlying attempts), against ``budget=2``. If retries consumed budget,
    the first logical call would exhaust it and the second would be denied
    (underlying attempts would stop at 2). Both must execute."""

    def test_budget_of_two_admits_two_retried_logical_calls(self):
        tool, attempts = _alternating_flaky_tool("flaky_search")
        wrapped = _wrap(tool, timeout=5.0, max_attempts=2, backoff=0.01, tool_config={})
        register_tool_factory("flaky_search", lambda config, tool_config: wrapped)

        fake = ReActFake(
            tool_calls=[
                [{"name": "flaky_search", "args": {}, "id": "c1"}],
                [{"name": "flaky_search", "args": {}, "id": "c2"}],
                [],  # stop — final structured turn
            ],
            final=lambda m: m(items=["done"]),
            output_model=Claims,
        )

        @node(
            mode="agent",
            outputs=Claims,
            model="fast",
            prompt="test/scan",
            tools=[Tool(name="flaky_search", budget=2)],
        )
        def explore() -> Claims: ...

        mod = _types.ModuleType("test_transport_budget_mod")
        mod.explore = explore
        pipeline = construct_from_module(mod, name="test-transport-budget")

        graph = compile(
            pipeline,
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: fake),
        )

        # Async-only tool -> async loop (the sync driver rejects coroutine-only tools).
        result = asyncio.run(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)))

        assert result["explore"] == Claims(items=["done"])
        assert attempts["n"] == 4, (
            f"2 logical calls x (1 fail + 1 success) = 4 underlying attempts; saw {attempts['n']} — "
            "a retry must not consume tool budget (budget counts the LOGICAL call once)"
        )


# ── 5. per-call timeout bounds a hung tool ────────────────────────────────────


@requires_mcp
class TestPerCallTimeout:
    """The factory path today inherits the adapter's 300s read timeout. The
    wrapper must bound each attempt with its own per-call timeout (parity with
    ``mcp_session``'s timeout=30) — a hung tool fails fast, and (non-idempotent
    default) the ambiguous timeout is not replayed."""

    def test_per_call_timeout_bounds_hung_tool_call(self):
        tool, attempts = _hung_tool("wedged_tool", hang_seconds=30.0)
        wrapped = _wrap(tool, timeout=0.1, max_attempts=3, backoff=0.01, tool_config={})

        start = time.monotonic()
        with pytest.raises(TimeoutError):
            asyncio.run(wrapped.ainvoke({}))
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"hung call must be cut at the per-call timeout, took {elapsed:.1f}s"
        assert attempts["n"] == 1, (
            f"a timeout is AMBIGUOUS (post-send): the non-idempotent default must not replay, "
            f"saw {attempts['n']} attempts"
        )
