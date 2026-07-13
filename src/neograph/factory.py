"""Node-function construction — turns Node definitions into LangGraph callables."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from typing import Any

import structlog
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.types import Command
from pydantic import BaseModel

from neograph._dispatch import _dispatch_for_mode
from neograph._execute import _aexecute_node, _execute_node, _type_name
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._normalize import primary_output_field
from neograph._state_bus import adapt_state
from neograph._state_keys import StateKeys
from neograph._trace import named
from neograph.errors import ConfigurationError, ExecutionError
from neograph.modifiers import HANDOFF_END, Keymaker
from neograph.naming import field_name_for
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
        raw = RunnableLambda(_make_raw_wrapper(node), afunc=_make_araw_wrapper(node))
        return named(raw, node.name, mode="raw", output_type=_type_name(node.outputs))

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
    #
    # `named` binds run_name=node.name so the engine's callback span reads as the
    # user's node (not the leaking `node_wrapper` __name__) and carries the node's
    # mode + declared output type as span metadata. See neograph-3fm1.
    wrapper = RunnableLambda(node_wrapper, afunc=anode_wrapper)
    return named(wrapper, node.name, mode=node.mode, output_type=_type_name(node.outputs))


def make_keymaker_fn(
    node: Node,
    keymaker: Keymaker,
    entry_field: str,
    exit_name: str,
    *,
    max_hops: int,
    on_exhaust: str,
    entry_name: str,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
) -> Runnable:
    """Build a Keymaker mesh-member function (design §4.1, decision D3/D10).

    Wraps the standard :func:`make_node_fn` result: the inner node runs normally
    and returns its state-update dict, then this wrapper reads the routing field
    off the member's payload output, validates the target, and returns a
    ``Command(goto=..., update=...)`` so LangGraph derives control flow from the
    member's runtime decision. The payload is also written to the shared,
    entry-keyed mesh channel so the next member reads it via the reserved
    ``handoff`` inputs key (design §3.3).

    INVARIANT (the durability pitch's one actively-false spot): a route target
    outside ``peers ∪ {HANDOFF_END}`` raises ``ExecutionError`` HERE — before the
    goto is emitted — instead of LangGraph silently dropping the update
    (``_algo.py:312``, the research's #1 constraint).

    ``entry_field`` keys the shared channel (``StateKeys.handoff_payload``), so
    EVERY member of the mesh WRITES the SAME channel regardless of its own field
    (one mesh per level — D-SINGLE-MESH). The READ side is symmetric but
    node-self-contained: the normalizer stamps the same key onto each member's
    ``handoff_channel`` and ``_extract_input`` reads it there (decision D10), so
    no channel key is threaded into ``make_node_fn`` here. ``HANDOFF_END`` is
    byte-identical to LangGraph's ``END`` sentinel, so it is mapped to the
    pass-through ``exit_name`` node rather than terminating the whole graph.

    HOP BUDGET (design §3.4, decision D11/D12): ``max_hops``/``on_exhaust`` are
    ENTRY-only knobs, but this wrapper runs per MEMBER — so ``_add_keymaker_mesh``
    sources them from the mesh entry (``members[0]``) plus the entry's name (for
    the error ``node=``) and threads them into EVERY member's wrapper as closure
    params (same closure-capture threading as ``entry_field``/``exit_name``). A
    "hop" is a member routing to a PEER (mesh continuation); routing to
    ``HANDOFF_END`` leaves the mesh cleanly and is NOT budget-gated and does NOT
    increment the counter. The shared, entry-keyed counter
    ``StateKeys.handoff_hops(entry_field)`` is read from the INCOMING ``state``
    (not the local ``update`` dict) so hops accumulate across DIFFERENT members;
    the check is BEFORE emitting the peer goto (``count >= max_hops`` — Loop
    parity), so ``max_hops=N`` allows exactly N peer-hops. ``on_exhaust=="error"``
    raises ``ExecutionError`` naming the entry (no new exception class, Loop
    parity); ``"exit"`` routes to ``exit_name`` with the last payload on the bus.
    """
    channel_key = StateKeys.handoff_payload(entry_field)
    count_field = StateKeys.handoff_hops(entry_field)
    payload_field = primary_output_field(field_name_for(node.name), node.outputs)
    route_field = keymaker.route
    valid_targets = set(keymaker.peers or ()) | {HANDOFF_END}

    inner = make_node_fn(
        node,
        runtime=runtime,
        scripted_lookup=scripted_lookup,
        tool_factory_lookup=tool_factory_lookup,
    )

    def _to_command(update: dict[str, Any], state: BaseModel) -> Command:
        payload = update[payload_field]
        target = getattr(payload, route_field)
        if target not in valid_targets:
            raise ExecutionError.build(
                "Keymaker route target is not a declared peer",
                expected=f"one of {sorted(valid_targets)}",
                found=f"route field '{route_field}'={target!r}",
                node=node.name,
                hint="a mesh member may route only to a declared peer or HANDOFF_END",
            )
        # HANDOFF_END is a clean mesh exit — never budget-gated, never counted.
        if target == HANDOFF_END:
            return Command(goto=exit_name, update={**update, channel_key: payload})
        # Peer continuation: enforce the entry's hop budget BEFORE emitting the
        # goto. Counter bootstrap (absent/None -> 0) lives in StateBus.get_counter;
        # read the SHARED counter from incoming state so hops accumulate across
        # members (the update dict never carries it — a from-update read would
        # always bootstrap 0 and break accumulation).
        current = adapt_state(state).get_counter(count_field)
        if current >= max_hops:
            if on_exhaust == "exit":
                return Command(
                    goto=exit_name,
                    update={**update, channel_key: payload, count_field: current},
                )
            raise ExecutionError.build(
                "Keymaker handoff exceeded max_hops",
                expected=f"convergence within {max_hops} hops",
                found=f"{max_hops} hops exhausted",
                node=entry_name,
                hint="raise the entry's max_hops or route to HANDOFF_END sooner",
            )
        return Command(
            goto=target,
            update={**update, channel_key: payload, count_field: current + 1},
        )

    def keymaker_wrapper(state: BaseModel, config: RunnableConfig) -> Command:
        return _to_command(inner.invoke(state, config), state)

    async def akeymaker_wrapper(state: BaseModel, config: RunnableConfig) -> Command:
        return _to_command(await inner.ainvoke(state, config), state)

    wrapper = RunnableLambda(keymaker_wrapper, afunc=akeymaker_wrapper)
    return named(wrapper, node.name, mode="keymaker", output_type=_type_name(node.outputs))


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
