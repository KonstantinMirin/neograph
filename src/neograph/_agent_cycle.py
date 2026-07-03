"""Inline agent-cycle expander bodies — agent/act mode as a subgraph of supersteps.

An agent/act node compiles to three parent nodes plus conditional routing
(the ``create_react_agent`` shape), NOT a ``while True`` loop inside one node
body:

    {node}__agent  — one ReAct turn: bind tools (budget-aware), call the LLM,
                     append the response to the message channel, bump counters.
    {node}__tools  — execute the requested tool calls, append ToolMessages +
                     ToolInteraction records, advance per-tool budget; on
                     exhaustion it injects a "final answer now" nudge and sets
                     the forced-final flag instead of executing.
    {node}__parse  — read the full message channel and produce the node's typed
                     output via the shared final-parse + fallback cluster.

Router after {node}__agent (3-way with loopback):
    forced-final flag set            -> {node}__parse   (exhaustion/guard path)
    last turn has no tool calls      -> {node}__parse   (happy path)
    otherwise                        -> {node}__tools -> back to {node}__agent

Message history, tool_log, and budget/iteration counters live in ``neo_``-prefixed
state channels (``StateKeys.agent_*``), so every turn is a checkpointed superstep:
a mid-loop interrupt pauses at a turn boundary and resumes without re-executing
prior turns (turn-boundary idempotency by construction).

This module owns only the node *bodies* and the router; ``_wiring._add_agent_cycle``
owns the topology (add_node / conditional edges). No engine execution verb
(``.invoke``/``.astream`` on a compiled graph) appears here — the LLM call is
``llm.invoke(messages)``, Layer-2 node-internal cognition.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import structlog
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph._dispatch import _render_input, _resolve_primary_output, _shape_tool_output
from neograph._input_shape import _extract_context, _extract_input
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._normalize import normalize_outputs
from neograph._state_bus import adapt_state
from neograph._state_keys import StateKeys
from neograph._state_write import _apply_skip_when, _build_state_update
from neograph._tool_loop import (
    _aparse_final_turn,
    _CoercingToolWrapper,
    _finish_tool_loop,
    _parse_final_turn,
    _prepare_tool_loop,
    _render_tool_result_for_llm,
)
from neograph.errors import ConfigurationError
from neograph.naming import field_name_for
from neograph.node import Node
from neograph.tool import Tool, ToolBudgetTracker, ToolInteraction

log = structlog.get_logger()


@dataclass
class AgentCycleNames:
    """The three parent-node names an agent/act node expands into."""

    agent: str
    tools: str
    parse: str


def cycle_names(node_name: str) -> AgentCycleNames:
    return AgentCycleNames(
        agent=f"{node_name}__agent",
        tools=f"{node_name}__tools",
        parse=f"{node_name}__parse",
    )


@dataclass
class _TurnPrep:
    """Per-superstep rebuild of the tool-loop preamble (llm, tool instances,
    seed messages, cfg, guards) — reused verbatim by agent/tools/parse bodies."""

    prep: Any  # _ToolLoopPrep
    output_model: Any
    effective_model: str
    effective_renderer: Any


def _build_turn_prep(
    node: Node,
    runtime: LlmRuntime,
    tool_factory_lookup: dict[str, Callable],
    state: BaseModel,
    config: RunnableConfig,
) -> _TurnPrep:
    """Rebuild the tool-loop preamble for one superstep. Factories are
    re-invocable (two-lifetime rule §5), so rebuilding per superstep is correct
    on resume (a fresh process re-mints tool instances)."""
    bus = adapt_state(state)
    raw_input = _extract_input(bus, node)
    rendered = _render_input(node, raw_input, runtime=runtime)
    context = _extract_context(bus, node)

    output_model, primary_key = _resolve_primary_output(node)
    no = normalize_outputs(node.outputs)
    gen_type = output_model
    if no.is_dict_form and primary_key is not None:
        gen_type = no.all_keys[primary_key]

    effective_model = config.get("configurable", {}).get("_oracle_model", node.model) or ""
    effective_renderer = node.renderer or runtime.renderer

    prep = _prepare_tool_loop(
        runtime=runtime,
        model_tier=effective_model,
        prompt_template=node.prompt or "",
        input_data=rendered,
        output_model=gen_type,
        tools=node.tools,
        config=config,
        node_name=node.name,
        llm_config=node.llm_config,
        context=context,
        tool_factory_lookup=tool_factory_lookup,
    )
    return _TurnPrep(prep=prep, output_model=gen_type, effective_model=effective_model,
                     effective_renderer=effective_renderer)


def _init_budget(existing: Any) -> dict[str, Any]:
    budget = dict(existing or {})
    budget.setdefault("iteration", 0)
    budget.setdefault("cumulative_input_tokens", 0)
    budget.setdefault("calls", {})
    budget.setdefault("forced_final", False)
    budget.setdefault("t0", time.monotonic())
    return budget


def _maybe_skip(node: Node, bus: Any, field: str, budget: dict[str, Any]) -> dict[str, Any] | None:
    """First-turn skip_when check. Mirrors ``_execute_node``: if the predicate
    fires, write the skip output and mark the cycle skipped so tools/parse become
    no-ops. Returns the state update (incl. budget) or None to proceed."""
    if node.skip_when is None:
        return None
    raw_input = _extract_input(bus, node)
    node_log = log.bind(node=node.name, mode=node.mode)
    skip = _apply_skip_when(node, raw_input, field, time.monotonic(), node_log, bus)
    if skip is None:
        return None
    budget["skipped"] = True
    return {**skip, StateKeys.agent_budget(field): budget}


def _tracker_from_budget(node: Node, budget: dict[str, Any]) -> ToolBudgetTracker:
    # node.tools is declared list[Tool | BaseTool], but _normalize_raw_base_tools
    # (node.py) converts every BaseTool -> Tool at construction, so it is always
    # list[Tool] here. Cast documents that invariant rather than widening the
    # tracker signature (which would mask it). See neograph-m6d3.4 refine.
    tracker = ToolBudgetTracker(cast(list[Tool], node.tools))
    tracker._counts = dict(budget.get("calls", {}))
    return tracker


def _agent_caller(prep_prep: Any, node: Node, budget: dict[str, Any]) -> Any:
    """Bind the LLM for one agent turn. Unbound when forced-final (exhaustion/
    guard) so the model must produce a final answer; otherwise bound to the tools
    that still have budget."""
    if budget.get("forced_final"):
        return prep_prep.llm
    tracker = _tracker_from_budget(node, budget)
    active = [prep_prep.tool_instances[t.name] for t in node.tools if tracker.can_call(t.name)]
    if not active:
        return prep_prep.llm
    return _CoercingToolWrapper(prep_prep.llm.bind_tools(active))


def _agent_working_messages(prep_prep: Any, channel_msgs: list) -> tuple[list, list]:
    """(messages to send the LLM, messages to seed into the channel). The seed
    (system preambles + compiled prompt) enters the channel only on the first
    turn; later turns read the accumulated history."""
    if not channel_msgs:
        return list(prep_prep.messages), list(prep_prep.messages)
    return list(channel_msgs), []


def _record_turn_usage(response: Any, budget: dict[str, Any]) -> None:
    usage = getattr(response, "usage_metadata", None) or {}
    budget["iteration"] += 1
    budget["cumulative_input_tokens"] += usage.get("input_tokens", 0)


def _total_calls(budget: dict[str, Any]) -> int:
    return sum(budget.get("calls", {}).values())


def _emit_limit_event(tp: _TurnPrep, budget: dict[str, Any], max_iter_hit: bool, budget_hit: bool) -> None:
    """Emit the ReAct loop-guard observability event (contract; preserved from the
    monolith): ``react_{reason}_exceeded`` at warning level with the loop state."""
    reason = (
        "max_iterations+token_budget"
        if max_iter_hit and budget_hit
        else ("max_iterations" if max_iter_hit else "token_budget")
    )
    tp.prep.llm_log.warning(
        f"react_{reason}_exceeded",
        max_iterations=tp.prep.max_iterations,
        token_budget=tp.prep.token_budget,
        cumulative_input_tokens=budget.get("cumulative_input_tokens", 0),
        loops=budget.get("iteration", 0),
        tool_calls=_total_calls(budget),
    )


def _emit_guard_forced_break(tp: _TurnPrep, budget: dict[str, Any]) -> None:
    """Emit ``react_guard_forced_break`` (contract): the forced-final turn ran
    tools-unbound but the model still returned tool_calls (rogue dispatch)."""
    tp.prep.llm_log.warning(
        "react_guard_forced_break",
        loops=budget.get("iteration", 0),
        tool_calls=_total_calls(budget),
    )


def make_agent_cycle_bodies(
    node: Node,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    tool_factory_lookup: dict[str, Callable] | None = None,
) -> dict[str, Any]:
    """Build the three node bodies + router for an agent/act node's inline cycle.

    Returns a dict with sync+async callables for agent/tools/parse plus the
    router, ready for ``_wiring._add_agent_cycle`` to attach to the graph.
    """
    tfl = tool_factory_lookup or {}
    field = field_name_for(node.name)
    msgs_key = StateKeys.agent_messages(field)
    tlog_key = StateKeys.agent_tool_log(field)
    budget_key = StateKeys.agent_budget(field)
    names = cycle_names(node.name)

    # ── {node}__agent ─────────────────────────────────────────────────────
    def agent_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        bus = adapt_state(state)
        channel_msgs = bus.get(msgs_key) or []
        budget = _init_budget(bus.get(budget_key))
        if not channel_msgs:
            skipped = _maybe_skip(node, bus, field, budget)
            if skipped is not None:
                return skipped
            log.bind(node=node.name, mode=node.mode).info(
                "node_start", input_type=None, output_type=node.outputs.__name__
                if isinstance(node.outputs, type) else None)
        was_forced = budget.get("forced_final", False)
        tp = _build_turn_prep(node, runtime, tfl, state, config)
        working, seed = _agent_working_messages(tp.prep, channel_msgs)
        caller = _agent_caller(tp.prep, node, budget)
        response = caller.invoke(working, config=config)
        _record_turn_usage(response, budget)
        if was_forced and getattr(response, "tool_calls", None):
            _emit_guard_forced_break(tp, budget)
        return {msgs_key: [*seed, response], budget_key: budget}

    async def aagent_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        bus = adapt_state(state)
        channel_msgs = bus.get(msgs_key) or []
        budget = _init_budget(bus.get(budget_key))
        if not channel_msgs:
            skipped = _maybe_skip(node, bus, field, budget)
            if skipped is not None:
                return skipped
            log.bind(node=node.name, mode=node.mode).info(
                "node_start", input_type=None, output_type=node.outputs.__name__
                if isinstance(node.outputs, type) else None)
        was_forced = budget.get("forced_final", False)
        tp = _build_turn_prep(node, runtime, tfl, state, config)
        working, seed = _agent_working_messages(tp.prep, channel_msgs)
        caller = _agent_caller(tp.prep, node, budget)
        response = await caller.ainvoke(working, config=config)
        _record_turn_usage(response, budget)
        if was_forced and getattr(response, "tool_calls", None):
            _emit_guard_forced_break(tp, budget)
        return {msgs_key: [*seed, response], budget_key: budget}

    # ── router after {node}__agent ────────────────────────────────────────
    def router(state: BaseModel) -> str:
        bus = adapt_state(state)
        channel_msgs = bus.get(msgs_key) or []
        budget = bus.get(budget_key) or {}
        if budget.get("skipped") or budget.get("forced_final"):
            return names.parse
        last = channel_msgs[-1] if channel_msgs else None
        tool_calls = getattr(last, "tool_calls", None) if last is not None else None
        if not tool_calls:
            return names.parse
        return names.tools

    # ── {node}__tools ─────────────────────────────────────────────────────
    def _tools_guards(tp: _TurnPrep, budget: dict[str, Any]) -> tuple[bool, bool]:
        max_iter_hit = budget.get("iteration", 0) >= tp.prep.max_iterations
        token_budget = tp.prep.token_budget
        budget_hit = token_budget is not None and budget.get("cumulative_input_tokens", 0) > token_budget
        return max_iter_hit, budget_hit

    def _limit_messages(tool_calls: list, max_iter_hit: bool) -> list:
        reason = "max iterations" if max_iter_hit else "token budget"
        return [
            ToolMessage(
                content=(f"React loop limit reached ({reason}). Provide your final answer now."),
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]

    def tools_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        bus = adapt_state(state)
        channel_msgs = bus.get(msgs_key) or []
        budget = _init_budget(bus.get(budget_key))
        tp = _build_turn_prep(node, runtime, tfl, state, config)
        last = channel_msgs[-1] if channel_msgs else None
        tool_calls = list(getattr(last, "tool_calls", None) or [])

        max_iter_hit, budget_hit = _tools_guards(tp, budget)
        if max_iter_hit or budget_hit:
            _emit_limit_event(tp, budget, max_iter_hit, budget_hit)
            budget["forced_final"] = True
            return {msgs_key: _limit_messages(tool_calls, max_iter_hit), budget_key: budget}

        tracker = _tracker_from_budget(node, budget)
        new_msgs: list = []
        interactions: list = []
        for tc in tool_calls:
            name = tc["name"]
            if not tracker.can_call(name):
                new_msgs.append(ToolMessage(content=f"Tool '{name}' budget exhausted. Use remaining tools or respond.", tool_call_id=tc["id"]))
                continue
            tool_fn = tp.prep.tool_instances.get(name)
            if tool_fn is None:
                new_msgs.append(ToolMessage(content=f"Unknown tool: {name}", tool_call_id=tc["id"]))
                continue
            t0 = time.monotonic()
            try:
                result = tool_fn.invoke(tc["args"])
            except NotImplementedError as exc:
                raise ConfigurationError.build(
                    f"Tool '{name}' does not support synchronous invocation",
                    expected="an async driver (arun())",
                    found="sync run() driving an async-only tool",
                    hint=(
                        "This tool is async-only (e.g. an MCP tool). Drive the graph "
                        "with arun() instead of run() so the async tool loop (ainvoke) "
                        "is used."
                    ),
                    node=node.name or None,
                ) from exc
            elapsed = time.monotonic() - t0
            tracker.record_call(name)
            rendered = _render_tool_result_for_llm(result, tp.effective_renderer)
            interactions.append(ToolInteraction(tool_name=name, args=tc.get("args", {}), result=rendered, typed_result=result, duration_ms=int(elapsed * 1000)))
            new_msgs.append(ToolMessage(content=rendered, tool_call_id=tc["id"]))

        budget["calls"] = dict(tracker._counts)
        if tracker.all_exhausted():
            budget["forced_final"] = True
        return {msgs_key: new_msgs, tlog_key: interactions, budget_key: budget}

    async def atools_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        bus = adapt_state(state)
        channel_msgs = bus.get(msgs_key) or []
        budget = _init_budget(bus.get(budget_key))
        tp = _build_turn_prep(node, runtime, tfl, state, config)
        last = channel_msgs[-1] if channel_msgs else None
        tool_calls = list(getattr(last, "tool_calls", None) or [])

        max_iter_hit, budget_hit = _tools_guards(tp, budget)
        if max_iter_hit or budget_hit:
            _emit_limit_event(tp, budget, max_iter_hit, budget_hit)
            budget["forced_final"] = True
            return {msgs_key: _limit_messages(tool_calls, max_iter_hit), budget_key: budget}

        tracker = _tracker_from_budget(node, budget)
        new_msgs = []
        interactions = []
        for tc in tool_calls:
            name = tc["name"]
            if not tracker.can_call(name):
                new_msgs.append(ToolMessage(content=f"Tool '{name}' budget exhausted. Use remaining tools or respond.", tool_call_id=tc["id"]))
                continue
            tool_fn = tp.prep.tool_instances.get(name)
            if tool_fn is None:
                new_msgs.append(ToolMessage(content=f"Unknown tool: {name}", tool_call_id=tc["id"]))
                continue
            t0 = time.monotonic()
            result = await tool_fn.ainvoke(tc["args"])
            elapsed = time.monotonic() - t0
            tracker.record_call(name)
            rendered = _render_tool_result_for_llm(result, tp.effective_renderer)
            interactions.append(ToolInteraction(tool_name=name, args=tc.get("args", {}), result=rendered, typed_result=result, duration_ms=int(elapsed * 1000)))
            new_msgs.append(ToolMessage(content=rendered, tool_call_id=tc["id"]))

        budget["calls"] = dict(tracker._counts)
        if tracker.all_exhausted():
            budget["forced_final"] = True
        return {msgs_key: new_msgs, tlog_key: interactions, budget_key: budget}

    # ── {node}__parse ─────────────────────────────────────────────────────
    def _finish_and_shape(state, config, tp, channel_msgs, tool_interactions, budget, parse_result, fallback_usage):
        result, _ = _finish_tool_loop(
            messages=channel_msgs, fallback_usage=fallback_usage, parse_result=parse_result,
            tool_interactions=tool_interactions, loop_count=budget.get("iteration", 0),
            total_tool_calls=len(tool_interactions), t0=budget.get("t0", time.monotonic()),
            llm_log=tp.prep.llm_log, runtime=runtime, model_tier=tp.effective_model,
            node_name=node.name, output_model=tp.output_model,
        )
        no = normalize_outputs(node.outputs)
        _, primary_key = _resolve_primary_output(node)
        output = _shape_tool_output(result, tool_interactions, no, primary_key)
        update = _build_state_update(node, field, output.value, adapt_state(state))
        log.bind(node=node.name, mode=node.mode).info("node_complete", loops=budget.get("iteration", 0))
        return update

    def parse_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        bus = adapt_state(state)
        budget = _init_budget(bus.get(budget_key))
        if budget.get("skipped"):
            return {}  # output already written by the skip update
        channel_msgs = list(bus.get(msgs_key) or [])
        tool_interactions = list(bus.get(tlog_key) or [])
        tp = _build_turn_prep(node, runtime, tfl, state, config)
        parse_result, fallback_usage = _parse_final_turn(
            messages=channel_msgs, output_model=tp.output_model, cfg=tp.prep.cfg,
            config=config, llm=tp.prep.llm,
        )
        return _finish_and_shape(state, config, tp, channel_msgs, tool_interactions, budget, parse_result, fallback_usage)

    async def aparse_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        bus = adapt_state(state)
        budget = _init_budget(bus.get(budget_key))
        if budget.get("skipped"):
            return {}  # output already written by the skip update
        channel_msgs = list(bus.get(msgs_key) or [])
        tool_interactions = list(bus.get(tlog_key) or [])
        tp = _build_turn_prep(node, runtime, tfl, state, config)
        parse_result, fallback_usage = await _aparse_final_turn(
            messages=channel_msgs, output_model=tp.output_model, cfg=tp.prep.cfg,
            config=config, llm=tp.prep.llm,
        )
        return _finish_and_shape(state, config, tp, channel_msgs, tool_interactions, budget, parse_result, fallback_usage)

    return {
        "names": names,
        "agent": (agent_body, aagent_body),
        "tools": (tools_body, atools_body),
        "parse": (parse_body, aparse_body),
        "router": router,
    }


