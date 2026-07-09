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

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn, cast

import structlog
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import BaseModel

from neograph._content_blocks import _block_field, _iter_content_blocks, _resource_link_kind
from neograph._dispatch import (
    _ainject_di_inputs,
    _inject_di_inputs,
    _render_input,
    _resolve_primary_output,
    _shape_tool_output,
)
from neograph._input_shape import _extract_context, _extract_input
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._normalize import normalize_outputs
from neograph._state_bus import adapt_state
from neograph._state_keys import StateKeys
from neograph._state_write import _apply_skip_when, _build_state_update
from neograph._tool_loop import (
    _aparse_final_turn,
    _aprepare_tool_loop,
    _CoercingToolWrapper,
    _finish_tool_loop,
    _parse_final_turn,
    _prepare_tool_loop,
    _render_tool_result_for_llm,
    _unparseable_args_raw,
)
from neograph.describe_type import type_display_name
from neograph.errors import ConfigurationError
from neograph.naming import field_name_for
from neograph.node import Node, TypeSpecStatic
from neograph.tool import (
    ProducingCall,
    ResourceRef,
    Tool,
    ToolBudgetTracker,
    ToolInteraction,
)

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


def _turn_prep_kwargs(
    node: Node,
    runtime: LlmRuntime,
    tool_factory_lookup: dict[str, Callable],
    state: BaseModel,
    config: RunnableConfig,
) -> tuple[dict[str, Any], Any, str, Any]:
    """Shared pre-prep for both turn-prep twins: extract + render input, resolve
    the generation type, and assemble the kwargs passed to (a)prepare_tool_loop.
    Returns (prepare_kwargs, gen_type, effective_model, effective_renderer).

    ``config`` MUST already carry the di_inputs injection — each twin runs its own
    driver-matched injector (sync ``_inject_di_inputs`` / async ``_ainject_di_inputs``)
    before calling here, mirroring the ``_prepare_tool_loop`` / ``_aprepare_tool_loop``
    sync/async split. FROM_RESOURCE template vars need the awaited async injector,
    which cannot live in this sync helper. See neograph-3q6j."""
    bus = adapt_state(state)
    raw_input = _extract_input(bus, node)
    rendered = _render_input(node, raw_input, runtime=runtime)
    context = _extract_context(bus, node)

    output_model, primary_key = _resolve_primary_output(node)
    no = normalize_outputs(node.outputs)
    gen_type = output_model
    if no.is_dict_form and primary_key is not None:
        gen_type = no.all_keys[primary_key]

    effective_model = config.get("configurable", {}).get(StateKeys.ORACLE_MODEL_OVERRIDE, node.model) or ""
    effective_renderer = node.renderer or runtime.renderer

    prepare_kwargs = {
        "runtime": runtime,
        "model_tier": effective_model,
        "prompt_template": node.prompt or "",
        "input_data": rendered,
        "output_model": gen_type,
        "tools": node.tools,
        "config": config,
        "node_name": node.name,
        "llm_config": node.llm_config,
        "context": context,
        "tool_factory_lookup": tool_factory_lookup,
    }
    return prepare_kwargs, gen_type, effective_model, effective_renderer


def _build_turn_prep(
    node: Node,
    runtime: LlmRuntime,
    tool_factory_lookup: dict[str, Callable],
    state: BaseModel,
    config: RunnableConfig,
) -> _TurnPrep:
    """Rebuild the tool-loop preamble for one superstep. Factories are
    re-invocable (two-lifetime rule §5), so rebuilding per superstep is correct
    on resume (a fresh process re-mints tool instances). Sync driver path — an
    async tool factory fails loud (drive with arun())."""
    # Sync injector: fails loud on a FROM_RESOURCE template var (its fetch is
    # awaited); resolves FromInput/FromConfig into config before _compile_prompt.
    config = _inject_di_inputs(node, config)
    prepare_kwargs, gen_type, effective_model, effective_renderer = _turn_prep_kwargs(
        node, runtime, tool_factory_lookup, state, config
    )
    prep = _prepare_tool_loop(**prepare_kwargs)
    return _TurnPrep(
        prep=prep, output_model=gen_type, effective_model=effective_model, effective_renderer=effective_renderer
    )


async def _abuild_turn_prep(
    node: Node,
    runtime: LlmRuntime,
    tool_factory_lookup: dict[str, Callable],
    state: BaseModel,
    config: RunnableConfig,
) -> _TurnPrep:
    """Async twin of _build_turn_prep: awaits _aprepare_tool_loop so an async
    tool factory (per-run token mint / MCP client build) is native on the arun()
    path. All pre-prep work is shared with the sync twin via _turn_prep_kwargs."""
    # Async injector twin: awaits FROM_RESOURCE bindings so a fetched resource's
    # text reaches the cycle's _compile_prompt as a template var. See neograph-3q6j.
    config = await _ainject_di_inputs(node, config)
    prepare_kwargs, gen_type, effective_model, effective_renderer = _turn_prep_kwargs(
        node, runtime, tool_factory_lookup, state, config
    )
    prep = await _aprepare_tool_loop(**prepare_kwargs)
    return _TurnPrep(
        prep=prep, output_model=gen_type, effective_model=effective_model, effective_renderer=effective_renderer
    )


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


# ── shared agent-turn skeleton (sync/async twins differ only at the seam) ──


def _obs_type_name(t: TypeSpecStatic) -> str | None:
    """Render a type for a structlog field via ``type_display_name`` (the single
    renderer, dict-form/generic aware), adapting ``None -> None`` so the field is
    omitted when absent. Mirrors ``_execute._type_name``; kept local because
    ``_execute`` is walled to a single importer."""
    return type_display_name(t) if t is not None else None


def _agent_start_log(node: Node) -> None:
    """Single-site the first-turn ``node_start`` event and route it through the
    shared ``type_display_name`` renderer (PAT-02): agent/act nodes now log the
    real ``input_type`` and a dict-form-aware ``output_type`` instead of the
    previous hard-coded ``input_type=None`` + inline ``__name__`` form."""
    log.bind(node=node.name, mode=node.mode).info(
        "node_start",
        input_type=_obs_type_name(node.inputs),
        output_type=_obs_type_name(node.outputs),
    )


def _agent_turn_prelude(
    node: Node,
    bus: Any,
    field: str,
    msgs_key: str,
    budget_key: str,
) -> tuple[dict[str, Any] | None, list, dict[str, Any], bool]:
    """Pure preamble shared by both agent-turn twins: read the channel + budget,
    run the first-turn skip check, emit node_start once. Returns
    ``(early_return, channel_msgs, budget, was_forced)`` — a non-None
    ``early_return`` means the skip predicate fired and the caller must return it
    verbatim (skip output already written)."""
    channel_msgs = bus.get(msgs_key) or []
    budget = _init_budget(bus.get(budget_key))
    if not channel_msgs:
        skipped = _maybe_skip(node, bus, field, budget)
        if skipped is not None:
            return skipped, channel_msgs, budget, False
        _agent_start_log(node)
    was_forced = budget.get("forced_final", False)
    return None, channel_msgs, budget, was_forced


def _agent_turn_finalize(
    tp: _TurnPrep,
    response: Any,
    budget: dict[str, Any],
    was_forced: bool,
    seed: list,
    msgs_key: str,
    budget_key: str,
) -> dict[str, Any]:
    """Pure postamble shared by both agent-turn twins: record usage, emit the
    guard-forced-break warning on rogue dispatch, assemble the state update."""
    _record_turn_usage(response, budget)
    if was_forced and getattr(response, "tool_calls", None):
        _emit_guard_forced_break(tp, budget)
    return {msgs_key: [*seed, response], budget_key: budget}


# ── shared per-tool-call handling (the DRY-01 extraction) ──


def _raise_sync_tool_async(node_name: str, tool_name: str, exc: Exception) -> NoReturn:
    """Fail loud when a sync run() drove an async-only tool. Mirrors
    ``_tool_loop._raise_async_factory_error`` (same driver/config mismatch);
    ``from exc`` preserves the NotImplementedError cause the inline raise had."""
    raise ConfigurationError.build(
        f"Tool '{tool_name}' does not support synchronous invocation",
        expected="an async driver (arun())",
        found="sync run() driving an async-only tool",
        hint=(
            "This tool is async-only (e.g. an MCP tool). Drive the graph "
            "with arun() instead of run() so the async tool loop (ainvoke) "
            "is used."
        ),
        node=node_name or None,
    ) from exc


def _tool_call_precheck(
    tc: dict,
    tracker: ToolBudgetTracker,
    tool_instances: dict,
) -> tuple[str, Any]:
    """Pure pre-invoke check for one tool call. Returns ``("msg", ToolMessage)``
    to short-circuit (unparseable args / budget exhausted / unknown tool) or
    ``("run", tool_fn)``. Single-sites the short-circuit ToolMessage builders
    across the twins."""
    name = tc["name"]
    # Unparseable args neograph-arus: the coercion path could not JSON-parse the
    # provider's args string, so it stamped the marker rather than blanking to {}.
    # Emit a RETRIABLE error to the LLM (so it can re-emit valid args) instead of
    # running the tool with empty args. Checked first + no budget consumed: a
    # malformed call is a re-emit, not a spent turn.
    raw_args = _unparseable_args_raw(tc)
    if raw_args is not None:
        return "msg", ToolMessage(
            content=(
                f"error: could not parse tool args for '{name}': {raw_args!r}. "
                "Re-emit the tool call with valid JSON arguments."
            ),
            tool_call_id=tc["id"],
        )
    if not tracker.can_call(name):
        return "msg", ToolMessage(
            content=f"Tool '{name}' budget exhausted. Use remaining tools or respond.",
            tool_call_id=tc["id"],
        )
    tool_fn = tool_instances.get(name)
    if tool_fn is None:
        return "msg", ToolMessage(content=f"Unknown tool: {name}", tool_call_id=tc["id"])
    return "run", tool_fn


def _build_tool_interaction(
    tc: dict,
    result: Any,
    elapsed_ms: int,
    renderer: Any,
) -> tuple[ToolInteraction, ToolMessage]:
    """Pure result rendering: render the tool result, build the ToolInteraction +
    ToolMessage. Does NOT touch the tracker — budget accounting is caller-owned.
    The sync twin advances the tracker THEN builds (via _record_tool_result); the
    async twin PRE-RESERVES the budget in tool_call order before gathering, then
    builds here (so it never double-counts). See neograph-dyy7."""
    name = tc["name"]
    rendered = _render_tool_result_for_llm(result, renderer)
    interaction = ToolInteraction(
        tool_name=name,
        args=tc.get("args", {}),
        result=rendered,
        typed_result=result,
        duration_ms=elapsed_ms,
    )
    return interaction, ToolMessage(content=rendered, tool_call_id=tc["id"])


def _record_tool_result(
    tc: dict,
    result: Any,
    elapsed_ms: int,
    tracker: ToolBudgetTracker,
    renderer: Any,
) -> tuple[ToolInteraction, ToolMessage]:
    """Pure post-invoke recording for one tool call: advance the tracker, then
    render + build the ToolInteraction + ToolMessage. The sync twin's single-turn
    call path — advance-then-build."""
    tracker.record_call(tc["name"])
    return _build_tool_interaction(tc, result, elapsed_ms, renderer)


async def _ainvoke_tool_timed(tool_fn: Any, tc: dict, config: RunnableConfig) -> tuple[Any, int]:
    """Await one tool's ``ainvoke`` and return ``(result, elapsed_ms)``. Each call
    times its OWN duration so a gathered batch records per-call latency, not the
    batch wall-clock. Used by the async tools superstep's concurrent gather."""
    t0 = time.monotonic()
    result = await tool_fn.ainvoke(tc["args"], config=config)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return result, elapsed_ms


def _lift_resource_refs(result: Any, tc: dict, idempotent: bool = False) -> list[ResourceRef]:
    """Lift typed ``ResourceRef``s from ``resource_link`` blocks in a tool result.

    Called from ``tools_body``/``atools_body`` INSIDE the per-tool-call loop —
    the ONLY point where the producing call (``tc`` name + args) and the raw
    ``result`` are both in scope. Lifting in ``_finish_tool_loop`` would lose the
    producing call (out of scope there), and the producing call is the sole
    protocol-reliable re-derivation path for an MCP ``resource_link`` (which
    carries no lifetime contract). Co-located with the ``ToolInteraction`` the
    ref corresponds to.

    Scans the result for MCP ``resource_link`` content blocks and emits one
    frozen ``ResourceRef`` per block, stamped with the producing call. A result
    without such blocks (a plain string / model — the common case) yields ``[]``.
    ``fetched_at`` stays ``None``: this is a lift, not a hydration.

    ``idempotent`` is the producing tool's ``Tool.idempotent`` flag neograph-lhc6,
    stamped onto ``ProducingCall.producer_idempotent`` — the hard gate hydration
    replay neograph-a5nh consults so a non-idempotent producer refuses replay.
    Defaults to the conservative ``False`` so a bare/unknown producer is never
    replay-eligible. A ``ttlMs`` hint (SEP-2549) is surfaced onto ``ref.ttl_ms``.
    """
    refs: list[ResourceRef] = []
    for block in _iter_content_blocks(result):
        if _block_field(block, "type") != "resource_link":
            continue
        uri = _block_field(block, "uri")
        if not uri:
            continue
        scheme = uri.split("://", 1)[0] if "://" in uri else ""
        refs.append(
            ResourceRef(
                uri=uri,
                kind=_resource_link_kind(block),
                server=scheme,
                producing_call=ProducingCall(
                    tool_name=tc["name"],
                    args=tc.get("args", {}) or {},
                    producer_idempotent=idempotent,
                ),
                mime=_block_field(block, "mimeType"),
                size=_block_field(block, "size"),
                ttl_ms=_block_field(block, "ttlMs"),
            )
        )
    return refs


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
    manifest_key = StateKeys.resource_manifest(field)
    budget_key = StateKeys.agent_budget(field)
    names = cycle_names(node.name)
    # Per-tool idempotency neograph-lhc6 stamped onto each lifted ref's producing
    # call so hydration replay neograph-a5nh can gate on it. A raw BaseTool with
    # no Tool spec is conservatively non-idempotent.
    idempotent_by_tool = {spec.name: bool(getattr(spec, "idempotent", False)) for spec in (node.tools or [])}

    # ── {node}__agent ─────────────────────────────────────────────────────
    def agent_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        bus = adapt_state(state)
        early, channel_msgs, budget, was_forced = _agent_turn_prelude(node, bus, field, msgs_key, budget_key)
        if early is not None:
            return early
        tp = _build_turn_prep(node, runtime, tfl, state, config)
        working, seed = _agent_working_messages(tp.prep, channel_msgs)
        caller = _agent_caller(tp.prep, node, budget)
        response = caller.invoke(working, config=config)
        return _agent_turn_finalize(tp, response, budget, was_forced, seed, msgs_key, budget_key)

    async def aagent_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        bus = adapt_state(state)
        early, channel_msgs, budget, was_forced = _agent_turn_prelude(node, bus, field, msgs_key, budget_key)
        if early is not None:
            return early
        tp = await _abuild_turn_prep(node, runtime, tfl, state, config)
        working, seed = _agent_working_messages(tp.prep, channel_msgs)
        caller = _agent_caller(tp.prep, node, budget)
        response = await caller.ainvoke(working, config=config)
        return _agent_turn_finalize(tp, response, budget, was_forced, seed, msgs_key, budget_key)

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

    # ── extract-then-thin: the tools superstep's shared skeleton ──
    # Both twins share the SAME preamble (read pending tool_calls + run the two
    # loop guards) and the SAME postamble (persist call counts, set forced-final
    # on exhaustion, assemble the update). Divergence is confined to the tp-build
    # seam (sync ``_build_turn_prep`` vs async ``_abuild_turn_prep``) and the
    # execution seam (``_run_tool_calls`` vs ``_arun_tool_calls``) — the sync path
    # runs sequentially, the async path pre-reserves budget then gathers.
    def _tools_prelude(bus: Any, tp: _TurnPrep, budget: dict[str, Any]) -> tuple[dict[str, Any] | None, list, Any]:
        """Read the pending tool_calls and run the two loop guards. Returns
        ``(early_return, tool_calls, tracker)``; a non-None ``early_return`` means
        a guard fired (forced-final) and the caller returns it verbatim, in which
        case ``tracker`` is None."""
        channel_msgs = bus.get(msgs_key) or []
        last = channel_msgs[-1] if channel_msgs else None
        tool_calls = list(getattr(last, "tool_calls", None) or [])
        max_iter_hit, budget_hit = _tools_guards(tp, budget)
        if max_iter_hit or budget_hit:
            _emit_limit_event(tp, budget, max_iter_hit, budget_hit)
            budget["forced_final"] = True
            return {msgs_key: _limit_messages(tool_calls, max_iter_hit), budget_key: budget}, tool_calls, None
        return None, tool_calls, _tracker_from_budget(node, budget)

    def _tools_result(
        new_msgs: list, interactions: list, refs: list, tracker: Any, budget: dict[str, Any]
    ) -> dict[str, Any]:
        """Persist per-tool call counts, force-final on full exhaustion, assemble
        the state update. Shared postamble for both twins."""
        budget["calls"] = dict(tracker._counts)
        if tracker.all_exhausted():
            budget["forced_final"] = True
        return {msgs_key: new_msgs, tlog_key: interactions, manifest_key: refs, budget_key: budget}

    def _run_tool_calls(
        tool_calls: list, tracker: Any, tp: _TurnPrep, config: RunnableConfig
    ) -> tuple[list, list, list]:
        """Sync execution seam: precheck → invoke → advance-then-record, one call
        at a time in tool_call order. Divergent twin of ``_arun_tool_calls``
        (which pre-reserves budget before a concurrent gather); the sync path has
        no gather, so it advances the tracker inline per successful call."""
        new_msgs: list = []
        interactions: list = []
        refs: list = []
        for tc in tool_calls:
            kind, payload = _tool_call_precheck(tc, tracker, tp.prep.tool_instances)
            if kind == "msg":
                new_msgs.append(payload)
                continue
            t0 = time.monotonic()
            try:
                result = payload.invoke(tc["args"], config=config)
            except NotImplementedError as exc:
                _raise_sync_tool_async(node.name, tc["name"], exc)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            interaction, msg = _record_tool_result(tc, result, elapsed_ms, tracker, tp.effective_renderer)
            interactions.append(interaction)
            refs.extend(_lift_resource_refs(result, tc, idempotent_by_tool.get(tc["name"], False)))
            new_msgs.append(msg)
        return new_msgs, interactions, refs

    async def _arun_tool_calls(
        tool_calls: list, tracker: Any, tp: _TurnPrep, config: RunnableConfig
    ) -> tuple[list, list, list]:
        """Async execution seam — the ONLY divergence from ``_run_tool_calls`` is
        concurrency neograph-dyy7. CRITICAL: Phase 1 pre-reserves each runnable
        call's budget SEQUENTIALLY, in tool_call order, BEFORE the gather — do NOT
        move ``record_call`` inside the gather. Reserving up front keeps per-tool
        budget enforcement identical to the sync twin: two parallel calls to a
        budget=1 tool see the first's reservation, so the second short-circuits. A
        plain gather-then-record would let both through because their can_call
        checks would race ahead of any record_call. ``plan`` preserves the
        original tool_call order so the ToolMessage / ToolInteraction message
        history holds regardless of which coroutine finishes first."""
        # Phase 1 (sequential, in tool_call order): precheck + PRE-RESERVE budget.
        plan: list[tuple[str, Any]] = []  # ("msg", ToolMessage) | ("run", tc)
        coros = []
        for tc in tool_calls:
            kind, payload = _tool_call_precheck(tc, tracker, tp.prep.tool_instances)
            if kind == "msg":
                plan.append(("msg", payload))
                continue
            tracker.record_call(tc["name"])  # pre-reserve so parallel calls honor budget
            plan.append(("run", tc))
            coros.append(_ainvoke_tool_timed(payload, tc, config))

        # Phase 2 (concurrent): await all runnable tool calls together.
        results = await asyncio.gather(*coros) if coros else []

        # Phase 3 (sequential, in original order): render + assemble.
        new_msgs: list = []
        interactions: list = []
        refs: list = []
        result_iter = iter(results)
        for kind, payload in plan:
            if kind == "msg":
                new_msgs.append(payload)
                continue
            tc = payload
            result, elapsed_ms = next(result_iter)
            interaction, msg = _build_tool_interaction(tc, result, elapsed_ms, tp.effective_renderer)
            interactions.append(interaction)
            refs.extend(_lift_resource_refs(result, tc, idempotent_by_tool.get(tc["name"], False)))
            new_msgs.append(msg)
        return new_msgs, interactions, refs

    def tools_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        bus = adapt_state(state)
        budget = _init_budget(bus.get(budget_key))
        tp = _build_turn_prep(node, runtime, tfl, state, config)
        early, tool_calls, tracker = _tools_prelude(bus, tp, budget)
        if early is not None:
            return early
        new_msgs, interactions, refs = _run_tool_calls(tool_calls, tracker, tp, config)
        return _tools_result(new_msgs, interactions, refs, tracker, budget)

    async def atools_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        bus = adapt_state(state)
        budget = _init_budget(bus.get(budget_key))
        tp = await _abuild_turn_prep(node, runtime, tfl, state, config)
        early, tool_calls, tracker = _tools_prelude(bus, tp, budget)
        if early is not None:
            return early
        new_msgs, interactions, refs = await _arun_tool_calls(tool_calls, tracker, tp, config)
        return _tools_result(new_msgs, interactions, refs, tracker, budget)

    # ── {node}__parse ─────────────────────────────────────────────────────
    def _finish_and_shape(state, config, tp, channel_msgs, tool_interactions, budget, parse_result, fallback_usage):
        result, _ = _finish_tool_loop(
            messages=channel_msgs,
            fallback_usage=fallback_usage,
            parse_result=parse_result,
            tool_interactions=tool_interactions,
            loop_count=budget.get("iteration", 0),
            total_tool_calls=len(tool_interactions),
            t0=budget.get("t0", time.monotonic()),
            llm_log=tp.prep.llm_log,
            runtime=runtime,
            model_tier=tp.effective_model,
            node_name=node.name,
            output_model=tp.output_model,
        )
        no = normalize_outputs(node.outputs)
        _, primary_key = _resolve_primary_output(node)
        output = _shape_tool_output(result, tool_interactions, no, primary_key)
        update = _build_state_update(node, field, output.value, adapt_state(state))
        elapsed = time.monotonic() - budget.get("t0", time.monotonic())
        log.bind(node=node.name, mode=node.mode).info(
            "node_complete", loops=budget.get("iteration", 0), duration_s=round(elapsed, 3)
        )
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
            messages=channel_msgs,
            output_model=tp.output_model,
            cfg=tp.prep.cfg,
            config=config,
            llm=tp.prep.llm,
        )
        return _finish_and_shape(
            state, config, tp, channel_msgs, tool_interactions, budget, parse_result, fallback_usage
        )

    async def aparse_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        bus = adapt_state(state)
        budget = _init_budget(bus.get(budget_key))
        if budget.get("skipped"):
            return {}  # output already written by the skip update
        channel_msgs = list(bus.get(msgs_key) or [])
        tool_interactions = list(bus.get(tlog_key) or [])
        tp = await _abuild_turn_prep(node, runtime, tfl, state, config)
        parse_result, fallback_usage = await _aparse_final_turn(
            messages=channel_msgs,
            output_model=tp.output_model,
            cfg=tp.prep.cfg,
            config=config,
            llm=tp.prep.llm,
        )
        return _finish_and_shape(
            state, config, tp, channel_msgs, tool_interactions, budget, parse_result, fallback_usage
        )

    return {
        "names": names,
        "agent": (agent_body, aagent_body),
        "tools": (tools_body, atools_body),
        "parse": (parse_body, aparse_body),
        "router": router,
    }


def _gate_approved(human_input: Any) -> bool:
    """Fail-closed approval test for a tool-gate resume value; see neograph-whq0.

    Only an explicit ``{"approved": True}`` approves. A missing key, a non-dict
    payload, or any other value is treated as a DENY — a safety control must not
    execute a side-effecting tool on an ambiguous resume.
    """
    return isinstance(human_input, dict) and human_input.get("approved") is True


def make_tool_gate_bodies(node: Node, gate_condition: Callable) -> dict[str, Any]:
    """Build the tool-gate node body + decision router for ``gate_tools_when``.

    The gate sits on the agent cycle's tools arm (see ``_wiring._add_agent_cycle``).
    It pauses BEFORE the ``{node}__tools`` superstep so a human approves before the
    tool's side effects run, then HONORS the decision per neograph-whq0:

    - condition falsy (gate does not fire) → leave the pending tool_calls
      unanswered → router routes to ``{node}__tools`` (proceed).
    - approve → leave the pending tool_calls unanswered → router routes to
      ``{node}__tools`` (the tool runs).
    - deny (fail-closed default) → append one denial ``ToolMessage`` per pending
      tool_call to the messages channel so the LLM sees why → router routes back
      to ``{node}__agent`` so the loop continues reasoning to a final answer.

    The decision is encoded in the message channel itself (deny answers the
    pending tool_calls; approve/no-fire leaves them pending), so the router reads
    a FRESH per-visit signal — never a persistent channel that can go stale across
    turns. The routing decision lives in the Layer-1 conditional edge; only the
    denial-message synthesis (agent-cycle domain, where ``msgs_key`` lives) sits
    here — no in-body if-approved check inside the tools node.
    """
    field = field_name_for(node.name)
    msgs_key = StateKeys.agent_messages(field)
    names = cycle_names(node.name)

    def _pending_tool_calls(channel_msgs: list) -> list:
        last = channel_msgs[-1] if channel_msgs else None
        return list(getattr(last, "tool_calls", None) or [])

    def _denial_messages(channel_msgs: list, human_input: Any) -> list:
        reason = human_input.get("reason") if isinstance(human_input, dict) else None
        detail = f" Reason: {reason}" if reason else ""
        return [
            ToolMessage(
                content=(
                    f"Tool call '{tc['name']}' was denied by a human reviewer and did "
                    f"not run.{detail} Do not retry it; continue with the information "
                    f"you have and provide your final answer."
                ),
                tool_call_id=tc["id"],
            )
            for tc in _pending_tool_calls(channel_msgs)
        ]

    def gate_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        payload = gate_condition(state)
        if not payload:
            return {}  # gate did not fire → proceed to tools
        human_input = interrupt(payload)
        if _gate_approved(human_input):
            return {StateKeys.HUMAN_FEEDBACK: human_input}
        # Deny (fail-closed): answer the pending tool_calls with denials so the
        # agent sees the rejection, and record the decision for observability.
        bus = adapt_state(state)
        channel_msgs = bus.get(msgs_key) or []
        return {
            msgs_key: _denial_messages(channel_msgs, human_input),
            StateKeys.HUMAN_FEEDBACK: human_input,
        }

    def gate_router(state: BaseModel) -> str:
        # Deny appended ToolMessages answering the pending tool_calls → the last
        # message is a ToolMessage → route back to the agent. Approve / no-fire
        # left the AIMessage's tool_calls pending → route to tools.
        bus = adapt_state(state)
        channel_msgs = bus.get(msgs_key) or []
        last = channel_msgs[-1] if channel_msgs else None
        if isinstance(last, ToolMessage):
            return names.agent
        return names.tools

    return {"gate": gate_body, "router": gate_router}
