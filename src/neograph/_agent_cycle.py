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

# --- names _agent_cycle.py imported and RE-EXPORTED before the split; the moved
# --- clusters were their only local consumers here.
import json  # noqa: E402,F401
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    Any,
    NoReturn,  # noqa: E402,F401
    cast,
)

import structlog
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt  # noqa: E402,F401
from pydantic import BaseModel

# --- extracted clusters (neograph-3ffdg.7), re-exported so existing
# --- `from neograph._agent_cycle import ...` call sites keep resolving.
from neograph._agent_cycle_names import AgentCycleNames, cycle_names  # noqa: E402,F401
from neograph._agent_gate import _gate_approved, make_tool_gate_bodies  # noqa: E402,F401
from neograph._agent_tool_calls import (  # noqa: E402,F401
    _ainvoke_tool_timed,
    _build_tool_interaction,
    _handoff_ack,
    _idempotent_repeat_key,
    _lift_resource_refs,
    _raise_sync_tool_async,
    _record_tool_result,
    _seed_repeat_cache,
    _tool_call_precheck,
)
from neograph._content_blocks import (  # noqa: E402,F401
    _block_field,
    _iter_content_blocks,
    _resource_link_kind,
)
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
from neograph._tool_loop import (  # noqa: E402,F401
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
from neograph.errors import ConfigurationError  # noqa: E402,F401
from neograph.modifiers import Portal
from neograph.naming import field_name_for
from neograph.node import Node, TypeSpecStatic
from neograph.tool import (  # noqa: E402,F401
    ProducingCall,
    ResourceRef,
    Tool,
    ToolBudgetTracker,
    ToolInteraction,
)

log = structlog.get_logger()


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


# Reserved prefix for synthesized tool-triggered-handoff tools (Portal
# trigger='tool', design portal-tool-triggered-handoff §3.1). One
# ``transfer_to_<peer>`` tool per declared peer — never persisted into
# ``Node.tools`` / any IR field, never registered via register_tool_factory.
_HANDOFF_TOOL_PREFIX = "transfer_to_"


def _handoff_targets(handoff_portal: Portal | None) -> dict[str, str]:
    """Map ``transfer_to_<peer>`` tool name -> peer name for a tool-triggered
    Portal member (design §3.2). Empty when the member is not tool-triggered, so
    every detection site short-circuits to the ordinary tool-call path."""
    if handoff_portal is None:
        return {}
    return {f"{_HANDOFF_TOOL_PREFIX}{peer}": peer for peer in (handoff_portal.to or ())}


def _synthesize_handoff_tools(handoff_portal: Portal) -> dict[str, Any]:
    """Build the ephemeral ``transfer_to_<peer>`` handoff tools bound for one
    tool-triggered agent turn (design §3.1). One StructuredTool per declared peer
    in ``handoff_portal.to`` — NEVER added to ``node.tools``, never registered
    via ``register_tool_factory``, never round-tripped through the IR. Mirrors
    ``tool.py``'s ``resource_reader``/``_build_read_blob`` ad-hoc StructuredTool
    pattern.

    The tool body is a trivial stub whose result is never observed:
    :func:`_tool_call_precheck`'s handoff branch intercepts the call by NAME
    before ``tool_instances.get(name)`` is ever reached, so the call is a pure
    routing signal, not a computation. StructuredTool requires a non-None
    ``func``/``coroutine`` to construct, so the stub stands in for "no body". The
    optional ``reason`` arg is model self-explanation only — routing reads the
    tool NAME, never its args.
    """
    from langchain_core.tools import StructuredTool
    from pydantic import create_model

    tools: dict[str, Any] = {}
    for peer in handoff_portal.to or ():
        name = f"{_HANDOFF_TOOL_PREFIX}{peer}"
        args_schema = create_model(f"{name}_Args", reason=(str, ""))
        tools[name] = StructuredTool(
            name=name,
            description=f"Transfer control to the '{peer}' agent to continue handling the request.",
            args_schema=args_schema,
            func=lambda **_: "transfer requested",
            coroutine=None,
        )
    return tools


def _agent_caller(prep_prep: Any, node: Node, budget: dict[str, Any], handoff_portal: Portal | None = None) -> Any:
    """Bind the LLM for one agent turn. Unbound when forced-final (exhaustion/
    guard) so the model must produce a final answer; otherwise bound to the tools
    that still have budget.

    ``handoff_portal`` (non-None only for a tool-triggered Portal member) folds
    the synthesized ``transfer_to_<peer>`` tools into the bound set so the model
    can actually emit a handoff call. Handoff tools are bound UNCONDITIONALLY
    (never gated by ``tracker.can_call`` — a handoff is a control-flow action,
    not budget-metered work) and are absent from ``node.tools``, so every budget/
    idempotency mechanism keyed off ``node.tools`` skips them for free."""
    if budget.get("forced_final"):
        return prep_prep.llm
    tracker = _tracker_from_budget(node, budget)
    active = [prep_prep.tool_instances[t.name] for t in node.tools if tracker.can_call(t.name)]
    if handoff_portal is not None:
        active = active + list(_synthesize_handoff_tools(handoff_portal).values())
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


def make_agent_cycle_bodies(
    node: Node,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    tool_factory_lookup: dict[str, Callable] | None = None,
    handoff_portal: Portal | None = None,
) -> dict[str, Any]:
    """Build the three node bodies + router for an agent/act node's inline cycle.

    Returns a dict with sync+async callables for agent/tools/parse plus the
    router, ready for ``_wiring._add_agent_cycle`` to attach to the graph.

    ``handoff_portal`` (non-None only for a tool-triggered Portal mesh member,
    design §3.1) folds the synthesized ``transfer_to_<peer>`` tools into every
    agent turn's bound tool set and makes the tools superstep detect a handoff
    call, stamping the chosen peer onto a transient sentinel key the factory
    Command-builder reads. ``None`` (every non-tool-triggered node — the two
    existing call sites) is fully zero-behavior-change: no handoff tools are
    bound and the sentinel is never written.
    """
    tfl = tool_factory_lookup or {}
    field = field_name_for(node.name)
    msgs_key = StateKeys.agent_messages(field)
    tlog_key = StateKeys.agent_tool_log(field)
    manifest_key = StateKeys.resource_manifest(field)
    budget_key = StateKeys.agent_budget(field)
    handoff_target_key = StateKeys.handoff_tool_target(field)
    handoff_targets = _handoff_targets(handoff_portal)
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
        caller = _agent_caller(tp.prep, node, budget, handoff_portal)
        response = caller.invoke(working, config=config)
        return _agent_turn_finalize(tp, response, budget, was_forced, seed, msgs_key, budget_key)

    async def aagent_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        bus = adapt_state(state)
        early, channel_msgs, budget, was_forced = _agent_turn_prelude(node, bus, field, msgs_key, budget_key)
        if early is not None:
            return early
        tp = await _abuild_turn_prep(node, runtime, tfl, state, config)
        working, seed = _agent_working_messages(tp.prep, channel_msgs)
        caller = _agent_caller(tp.prep, node, budget, handoff_portal)
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
        new_msgs: list,
        interactions: list,
        refs: list,
        tracker: Any,
        budget: dict[str, Any],
        handoff_target: str | None = None,
    ) -> dict[str, Any]:
        """Persist per-tool call counts, force-final on full exhaustion, assemble
        the state update. Shared postamble for both twins.

        ``handoff_target`` (non-None only for a tool-triggered member that emitted
        a ``transfer_to_<peer>`` call) stamps the TRANSIENT sentinel key the
        factory Command-builder pops back out before it constructs the
        ``Command(update=...)`` — so it never enters LangGraph state."""
        budget["calls"] = dict(tracker._counts)
        if tracker.all_exhausted():
            budget["forced_final"] = True
        update = {msgs_key: new_msgs, tlog_key: interactions, manifest_key: refs, budget_key: budget}
        if handoff_target is not None:
            update[handoff_target_key] = handoff_target
        return update

    def _run_tool_calls(
        tool_calls: list, tracker: Any, tp: _TurnPrep, config: RunnableConfig, repeat_cache: dict[str, str]
    ) -> tuple[list, list, list, str | None]:
        """Sync execution seam: precheck → repeat-guard → invoke → advance-then-record, one call
        at a time in tool_call order. Divergent twin of ``_arun_tool_calls``
        (which pre-reserves budget before a concurrent gather); the sync path has
        no gather, so it advances the tracker inline per successful call.

        Returns ``(new_msgs, interactions, refs, handoff_target)``. A detected
        ``transfer_to_<peer>`` call answers its own ``tool_call_id`` and records
        a ToolInteraction; the FIRST such call in tool_call order wins the routing
        target (design §3.2), while any remaining calls in the batch are still
        answered so the LLM turn is never left with an unanswered id."""
        new_msgs: list = []
        interactions: list = []
        refs: list = []
        handoff_target: str | None = None
        for tc in tool_calls:
            kind, payload = _tool_call_precheck(tc, tracker, tp.prep.tool_instances, handoff_targets)
            if kind == "handoff":
                interaction, msg = _handoff_ack(tc, payload)
                interactions.append(interaction)
                new_msgs.append(msg)
                if handoff_target is None:
                    handoff_target = payload
                continue
            if kind == "msg":
                new_msgs.append(payload)
                continue
            repeat_key = _idempotent_repeat_key(tc, idempotent_by_tool)
            if repeat_key is not None and repeat_key in repeat_cache:
                # Idempotent repeat (8ko.34): serve the cycle's own prior render — no re-invoke,
                # no budget spend, no duplicate ToolInteraction.
                new_msgs.append(ToolMessage(content=repeat_cache[repeat_key], tool_call_id=tc["id"]))
                continue
            t0 = time.monotonic()
            try:
                result = payload.invoke(tc["args"], config=config)
            except NotImplementedError as exc:
                _raise_sync_tool_async(node.name, tc["name"], exc)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            interaction, msg = _record_tool_result(tc, result, elapsed_ms, tracker, tp.effective_renderer)
            if repeat_key is not None:
                repeat_cache[repeat_key] = interaction.result
            interactions.append(interaction)
            refs.extend(_lift_resource_refs(result, tc, idempotent_by_tool.get(tc["name"], False)))
            new_msgs.append(msg)
        return new_msgs, interactions, refs, handoff_target

    async def _arun_tool_calls(
        tool_calls: list, tracker: Any, tp: _TurnPrep, config: RunnableConfig, repeat_cache: dict[str, str]
    ) -> tuple[list, list, list, str | None]:
        """Async execution seam — the ONLY divergence from ``_run_tool_calls`` is
        concurrency neograph-dyy7. CRITICAL: Phase 1 pre-reserves each runnable
        call's budget SEQUENTIALLY, in tool_call order, BEFORE the gather — do NOT
        move ``record_call`` inside the gather. Reserving up front keeps per-tool
        budget enforcement identical to the sync twin: two parallel calls to a
        budget=1 tool see the first's reservation, so the second short-circuits. A
        plain gather-then-record would let both through because their can_call
        checks would race ahead of any record_call. ``plan`` preserves the
        original tool_call order so the ToolMessage / ToolInteraction message
        history holds regardless of which coroutine finishes first. A
        ``transfer_to_<peer>`` handoff call is planned (never gathered — it runs
        no coroutine) and the FIRST one wins the routing target in Phase 3."""
        # Phase 1 (sequential, in tool_call order): precheck + repeat-guard + PRE-RESERVE budget.
        plan: list[tuple[str, Any]] = []  # ("handoff", (tc, peer)) | ("msg", ToolMessage) | ("run", tc)
        coros = []
        for tc in tool_calls:
            kind, payload = _tool_call_precheck(tc, tracker, tp.prep.tool_instances, handoff_targets)
            if kind == "handoff":
                plan.append(("handoff", (tc, payload)))
                continue
            if kind == "msg":
                plan.append(("msg", payload))
                continue
            repeat_key = _idempotent_repeat_key(tc, idempotent_by_tool)
            if repeat_key is not None and repeat_key in repeat_cache:
                # Idempotent repeat (8ko.34): serve the prior render — no invoke, no budget.
                plan.append(("msg", ToolMessage(content=repeat_cache[repeat_key], tool_call_id=tc["id"])))
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
        handoff_target: str | None = None
        result_iter = iter(results)
        for kind, payload in plan:
            if kind == "handoff":
                tc, peer = payload
                interaction, msg = _handoff_ack(tc, peer)
                interactions.append(interaction)
                new_msgs.append(msg)
                if handoff_target is None:
                    handoff_target = peer
                continue
            if kind == "msg":
                new_msgs.append(payload)
                continue
            tc = payload
            result, elapsed_ms = next(result_iter)
            interaction, msg = _build_tool_interaction(tc, result, elapsed_ms, tp.effective_renderer)
            repeat_key = _idempotent_repeat_key(tc, idempotent_by_tool)
            if repeat_key is not None:
                repeat_cache[repeat_key] = interaction.result
            interactions.append(interaction)
            refs.extend(_lift_resource_refs(result, tc, idempotent_by_tool.get(tc["name"], False)))
            new_msgs.append(msg)
        return new_msgs, interactions, refs, handoff_target

    def tools_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        bus = adapt_state(state)
        budget = _init_budget(bus.get(budget_key))
        tp = _build_turn_prep(node, runtime, tfl, state, config)
        early, tool_calls, tracker = _tools_prelude(bus, tp, budget)
        if early is not None:
            return early
        repeat_cache = _seed_repeat_cache(bus.get(tlog_key), idempotent_by_tool)
        new_msgs, interactions, refs, handoff_target = _run_tool_calls(tool_calls, tracker, tp, config, repeat_cache)
        return _tools_result(new_msgs, interactions, refs, tracker, budget, handoff_target)

    async def atools_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        bus = adapt_state(state)
        budget = _init_budget(bus.get(budget_key))
        tp = await _abuild_turn_prep(node, runtime, tfl, state, config)
        early, tool_calls, tracker = _tools_prelude(bus, tp, budget)
        if early is not None:
            return early
        repeat_cache = _seed_repeat_cache(bus.get(tlog_key), idempotent_by_tool)
        new_msgs, interactions, refs, handoff_target = await _arun_tool_calls(
            tool_calls, tracker, tp, config, repeat_cache
        )
        return _tools_result(new_msgs, interactions, refs, tracker, budget, handoff_target)

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
