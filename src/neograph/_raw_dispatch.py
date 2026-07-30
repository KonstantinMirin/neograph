"""Raw-mode node wrappers — the (state, config) escape hatch.

Extracted from ``factory.py`` (neograph-3ffdg.6) as a pure file split — the two
factories below are unchanged, only their home moved. ``factory.py`` re-exports
them, so existing imports keep resolving.

Raw mode hands the user the LangGraph state and config directly, so these
wrappers do almost nothing beyond timing, type-checking the return, and pairing
a sync factory with its async twin (the pairing is pinned by the async-dispatch
twin guard, which is why both live in one module).

Deliberately does NOT contain the portal-dispatch factory. That cluster
constructs LangGraph ``Command`` objects, which guard G1 confines to
``factory.py`` and ``runner.py``; moving it would take that monopoly from two
modules to three. See neograph-3ffdg.6's notes.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph._execute import _type_name
from neograph.errors import ExecutionError
from neograph.node import Node

log = structlog.get_logger()


def _make_raw_wrapper(node: Node) -> Callable:
    """Wrap a raw_fn dispatch with observability (node_start/node_complete).

    Only used for explicit ``mode='raw'`` escape-hatch nodes. Raw nodes
    bypass the unified _execute_node path — no DI/input/output wrapping,
    only logging.
    """
    assert node.raw_fn is not None, f"node '{node.name}' has mode='raw' but no raw_fn"
    raw_fn = node.raw_fn

    def raw_node_wrapper(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        node_log = log.bind(node=node.name, mode="raw")
        node_log.info("node_start", input_type=_type_name(node.inputs), output_type=_type_name(node.outputs))
        t0 = time.monotonic()

        result = raw_fn(state, config)
        if inspect.isawaitable(result):
            # An `async def` raw body under the SYNC driver: we cannot await here,
            # and returning the coroutine would flow un-awaited into state (silent
            # wrong behavior). Fail loud — araw_node_wrapper awaits correctly.
            if hasattr(result, "close"):
                result.close()  # suppress the "never awaited" RuntimeWarning
            raise ExecutionError.build(
                "async node body invoked under sync run(); use arun()",
                node=node.name,
                hint="An `async def` raw body requires the async driver. "
                "Call arun(graph, ...) / graph.ainvoke instead of run() / graph.invoke.",
            )

        elapsed = time.monotonic() - t0
        node_log.info("node_complete", duration_s=round(elapsed, 3))
        return result

    # __name__ stays informational; routing is the add_node argument (y20i).
    return raw_node_wrapper


def _make_araw_wrapper(node: Node) -> Callable:
    """Async twin of :func:`_make_raw_wrapper` (Phase 1b).

    Same observability/timing as the sync wrapper; the only divergence is that
    an ``async def`` raw body is awaited. Detection is at the call boundary
    (``inspect.isawaitable``), identical to ScriptedDispatch.aexecute, so a sync
    raw body under ``ainvoke`` is simply not awaited (LangGraph threadpools it).
    """
    assert node.raw_fn is not None, f"node '{node.name}' has mode='raw' but no raw_fn"
    raw_fn = node.raw_fn

    async def araw_node_wrapper(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        node_log = log.bind(node=node.name, mode="raw")
        node_log.info("node_start", input_type=_type_name(node.inputs), output_type=_type_name(node.outputs))
        t0 = time.monotonic()

        result = raw_fn(state, config)
        if inspect.isawaitable(result):
            result = await result

        elapsed = time.monotonic() - t0
        node_log.info("node_complete", duration_s=round(elapsed, 3))
        return result

    return araw_node_wrapper
