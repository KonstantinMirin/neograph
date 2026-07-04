"""Node-function construction — turns Node definitions into LangGraph callables."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from typing import Any

import structlog
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel

from neograph._dispatch import _dispatch_for_mode
from neograph._execute import _aexecute_node, _execute_node, _type_name
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph.errors import ConfigurationError, ExecutionError
from neograph.node import Node

log = structlog.get_logger()


def make_node_fn(
    node: Node,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
) -> Runnable:
    """Create a LangGraph node function from a Node definition.

    This is the core of neograph — the generic factory that eliminates
    the 70% boilerplate from every hand-coded node.

    Raw nodes get a minimal observability wrapper. All other modes
    (scripted, think, agent, act) go through _execute_node with a
    mode-specific ModeDispatch that captures the supplied LlmRuntime
    and per-compile scripted lookup.

    Args:
        node: Node IR definition.
        runtime: LLM runtime bundle closure-captured by LLM-mode dispatches.
            Scripted nodes ignore this. Defaults to EMPTY_RUNTIME so
            scripted-only constructs compile without LLM kwargs.
        scripted_lookup: per-compile `{name: shim_fn}` dict built by
            `compile()` from `node._scripted_shim` on each scripted Node.
            Falls back to the deprecated module-level fallback registry
            if not supplied — for direct callers like `Node.run_isolated`.
    """
    # Raw node — wrap with observability so node_start/node_complete fire.
    # Dual-path RunnableLambda (uniform return type; direct callers .invoke()):
    # graph.invoke -> sync raw wrapper, graph.ainvoke -> async raw wrapper that
    # awaits an `async def` raw body (Phase 1b).
    if node.raw_fn is not None:
        return RunnableLambda(_make_raw_wrapper(node), afunc=_make_araw_wrapper(node))

    # Validate scripted registration early
    if node.mode == "scripted":
        per_compile = scripted_lookup or {}
        if node.scripted_fn not in per_compile:
            raise ConfigurationError.build(
                f"Scripted function '{node.scripted_fn}' not registered",
                hint=f"Pass scripted={{'{node.scripted_fn}': fn}} to compile().",
                node=node.name,
            )

    dispatch = _dispatch_for_mode(
        node,
        runtime=runtime,
        scripted_lookup=scripted_lookup,
        tool_factory_lookup=tool_factory_lookup,
    )

    def node_wrapper(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        return _execute_node(node, state, config, dispatch)

    async def anode_wrapper(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        return await _aexecute_node(node, state, config, dispatch)

    # Driver-selected dual path: graph.invoke -> node_wrapper (sync),
    # graph.ainvoke -> anode_wrapper (async). Routing identity is the explicit
    # graph.add_node(name, fn) argument, not this closure's __name__ (which stays
    # informational). Display labels come from node.name. See neograph-y20i.
    return RunnableLambda(node_wrapper, afunc=anode_wrapper)


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
