"""Per-tool-call helpers for the inline ReAct cycle.

Extracted from ``_agent_cycle.py`` (neograph-3ffdg.7) as a pure file split — the
functions below are unchanged, only their home moved. ``_agent_cycle.py``
re-exports them, so existing imports keep resolving.

These are the pure per-call steps the cycle body runs for each tool invocation:
precheck and arg parsing, idempotent-repeat detection, timing, result recording,
handoff acknowledgement, and lifting MCP resource references. The file already
labelled this the DRY-01 extraction; it has zero coupling to the cycle's
closures, which is why it moves cleanly.
"""

from __future__ import annotations

import json
import time
from typing import Any, NoReturn

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from neograph._content_blocks import _block_field, _iter_content_blocks, _resource_link_kind
from neograph._tool_loop import _render_tool_result_for_llm, _unparseable_args_raw
from neograph.di import read_upstream
from neograph.errors import ConfigurationError
from neograph.node import Node
from neograph.tool import ProducingCall, ResourceRef, ToolBudgetTracker, ToolInteraction


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


def _resolve_bound_args(node: Node, bus: Any) -> dict[str, dict[str, Any]]:
    """``{tool_name: {arg: value}}`` for every ``Tool.bound_args`` on *node*.

    Resolved ONCE per turn from state, so a path is read at the point the run
    reached rather than per tool call. A dotted path navigates the resolved
    VALUE after the first segment, matching how ``Each(over=...)`` reads its
    collection -- one navigation rule, not two.

    Sits beside ``_apply_bound_args`` so one module owns the whole rule:
    where a framework-owned tool argument comes from, and that a model cannot
    replace it.
    """
    bound: dict[str, dict[str, Any]] = {}
    for tool_spec in node.tools or []:
        # A raw LangChain BaseTool carries no neograph declaration, so there is
        # nothing to bind on it.
        bindings = getattr(tool_spec, "bound_args", None)
        if not bindings:
            continue
        resolved: dict[str, Any] = {}
        for arg_name, path in bindings.items():
            root, _, rest = path.partition(".")
            # read_upstream, not bus.get: reading an upstream is a RULE, not a
            # line -- it unwraps a Loop append-list and an Each result dict. A
            # bound argument off a Loop-modified producer would otherwise
            # resolve to the whole history rather than the latest value.
            # required=False: assembly already proved the root has a producer,
            # so absence here means an untaken branch arm, not a wiring bug.
            value = read_upstream(bus, root, object, required=False, node_label=getattr(node, "name", ""))
            for part in rest.split(".") if rest else []:
                if value is None:
                    break
                value = getattr(value, part, None)
            resolved[arg_name] = value
        bound[getattr(tool_spec, "name", "")] = resolved
    return bound

def _apply_bound_args(tc: dict, bound_by_tool: dict[str, dict[str, Any]]) -> None:
    """Overwrite the framework-owned arguments the model composed.

    Applied inside the pre-invoke check so BOTH twins get it from one site --
    the sync and async invokes are two sinks and a fix at each would drift.
    It must also land BEFORE ``_idempotent_repeat_key`` reads ``tc["args"]``,
    or two calls differing only in an overridden argument collapse in the
    repeat cache: a correctness fix turning into a caching bug.

    Arguments not named in ``bound_args`` are left exactly as the model wrote
    them. Resolution lives in ``_agent_cycle._resolve_bound_args``, which holds
    the state bus.
    """
    bound = bound_by_tool.get(tc.get("name", ""))
    if not bound:
        return
    args = tc.get("args")
    if isinstance(args, dict):
        args.update(bound)


def _tool_call_precheck(
    tc: dict,
    tracker: ToolBudgetTracker,
    tool_instances: dict,
    handoff_targets: dict[str, str] | None = None,
    bound_by_tool: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, Any]:
    """Pure pre-invoke check for one tool call. Returns ``("handoff", peer)`` for
    a synthesized ``transfer_to_<peer>`` call (design §3.2), ``("msg",
    ToolMessage)`` to short-circuit (unparseable args / budget exhausted /
    unknown tool), or ``("run", tool_fn)``. Single-sites the short-circuit
    ToolMessage builders across the twins."""
    name = tc["name"]
    # Tool-triggered-handoff detection FIRST (design §3.2): a synthesized
    # transfer_to_<peer> call is a pure routing signal, never a real tool — it is
    # intercepted by NAME before any arg-parse / budget / tool_instances lookup,
    # so it never touches the budget tracker or the idempotency cache.
    if handoff_targets and name in handoff_targets:
        return "handoff", handoff_targets[name]
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
    # Last thing before the call is cleared to run, and before the caller takes
    # the repeat key: a framework-owned argument is not the model's to choose.
    if bound_by_tool:
        _apply_bound_args(tc, bound_by_tool)
    return "run", tool_fn


def _idempotent_repeat_key(tc: dict, idempotent_by_tool: dict[str, bool]) -> str | None:
    """Canonical (tool, args) key for the repeat-call guard — ONLY for ``Tool(idempotent=True)``
    tools (the same opt-in flag that gates transport/hydration replay). A repeated identical call
    within one node is a re-emit (typically a parse-failure retry re-issuing a served batch with
    fresh tool_call ids), not new work: serving it from the cycle's own history keeps the budget
    for UNSWEPT work (the ox-troubleshooting-demo 8ko.34 coverage crowd-out). None disables."""
    if not idempotent_by_tool.get(tc.get("name", "")):
        return None
    try:
        args_canon = json.dumps(tc.get("args") or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None
    return f"{tc['name']}::{args_canon}"


def _seed_repeat_cache(prior_interactions: Any, idempotent_by_tool: dict[str, bool]) -> dict[str, str]:
    """Rebuild the repeat cache from the node's ACCUMULATED tool_log channel, so the guard holds
    across supersteps (each tools superstep sees every prior turn's served results)."""
    cache: dict[str, str] = {}
    for interaction in prior_interactions or []:
        key = _idempotent_repeat_key(
            {"name": getattr(interaction, "tool_name", ""), "args": getattr(interaction, "args", {})},
            idempotent_by_tool,
        )
        if key is not None:
            cache[key] = getattr(interaction, "result", "")
    return cache


def _build_tool_interaction(
    tc: dict,
    result: Any,
    elapsed_ms: int,
    renderer: Any,
    ordinal: int = 0,
) -> tuple[ToolInteraction, ToolMessage]:
    """Pure result rendering: render the tool result, build the ToolInteraction +
    ToolMessage. Does NOT touch the tracker — budget accounting is caller-owned.
    The sync twin advances the tracker THEN builds (via _record_tool_result); the
    async twin PRE-RESERVES the budget in tool_call order before gathering, then
    builds here (so it never double-counts). See neograph-dyy7.

    ``ordinal`` is threaded in, not recomputed here — the caller is the one that
    called ``tracker.record_call(...)`` and knows the ordinal it got back."""
    name = tc["name"]
    rendered = _render_tool_result_for_llm(result, renderer)
    interaction = ToolInteraction(
        tool_name=name,
        args=tc.get("args", {}),
        result=rendered,
        typed_result=result,
        duration_ms=elapsed_ms,
        ordinal=ordinal,
    )
    return interaction, ToolMessage(content=rendered, tool_call_id=tc["id"])


def _handoff_ack(tc: dict, peer: str) -> tuple[ToolInteraction, ToolMessage]:
    """Acknowledge a detected ``transfer_to_<peer>`` call (design §3.2): the
    LLM-visible confirmation ``ToolMessage`` (mirrors the reference swarm's
    "Successfully transferred to {peer}") + a ``ToolInteraction`` for tool_log/
    observability parity. No tool ran (the call is a pure routing signal), so
    ``duration_ms=0`` and ``typed_result=None``. Answering the ``tool_call_id``
    keeps the LLM turn well-formed even when other calls share the batch."""
    content = f"Successfully transferred to {peer}"
    interaction = ToolInteraction(
        tool_name=tc["name"],
        args=tc.get("args", {}),
        result=content,
        typed_result=None,
        duration_ms=0,
    )
    return interaction, ToolMessage(content=content, tool_call_id=tc["id"])


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
    ordinal = tracker.record_call(tc["name"])
    return _build_tool_interaction(tc, result, elapsed_ms, renderer, ordinal=ordinal)


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
