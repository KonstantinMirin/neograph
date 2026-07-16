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

from neograph._agent_cycle import make_agent_cycle_bodies
from neograph._dispatch import _dispatch_for_mode
from neograph._execute import _aexecute_node, _execute_node, _type_name
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._normalize import primary_output_field
from neograph._state_bus import adapt_state
from neograph._state_keys import StateKeys
from neograph._subconstruct import _scan_subgraph_output
from neograph._trace import named
from neograph.errors import ConfigurationError, ConstructError, ExecutionError
from neograph.loader import load_spec
from neograph.modifiers import HANDOFF_END, Portal
from neograph.naming import field_name_for, output_field_name
from neograph.node import Node
from neograph.spec_types import lookup_type

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


def make_portal_fn(
    node: Node,
    portal: Portal,
    entry_field: str,
    exit_name: str,
    *,
    max_hops: int,
    on_exhaust: str,
    entry_name: str,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
    target_resolve: dict[str, str] | None = None,
) -> Runnable:
    """Build a Portal mesh-member function (design §4.1, decision D3/D10).

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
    ENTRY-only knobs, but this wrapper runs per MEMBER — so ``_add_portal_mesh``
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
    route_field = portal.route
    valid_targets = set(portal.to or ()) | {HANDOFF_END}

    inner = make_node_fn(
        node,
        runtime=runtime,
        scripted_lookup=scripted_lookup,
        tool_factory_lookup=tool_factory_lookup,
    )

    def portal_wrapper(state: BaseModel, config: RunnableConfig) -> Command:
        return _portal_route_to_command(
            inner.invoke(state, config),
            state,
            payload_field=payload_field,
            route_field=route_field,
            valid_targets=valid_targets,
            channel_key=channel_key,
            count_field=count_field,
            max_hops=max_hops,
            on_exhaust=on_exhaust,
            exit_name=exit_name,
            node_name=node.name,
            entry_name=entry_name,
            target_resolve=target_resolve,
        )

    async def aportal_wrapper(state: BaseModel, config: RunnableConfig) -> Command:
        return _portal_route_to_command(
            await inner.ainvoke(state, config),
            state,
            payload_field=payload_field,
            route_field=route_field,
            valid_targets=valid_targets,
            channel_key=channel_key,
            count_field=count_field,
            max_hops=max_hops,
            on_exhaust=on_exhaust,
            exit_name=exit_name,
            node_name=node.name,
            entry_name=entry_name,
            target_resolve=target_resolve,
        )

    wrapper = RunnableLambda(portal_wrapper, afunc=aportal_wrapper)
    return named(wrapper, node.name, mode="portal", output_type=_type_name(node.outputs))


def _portal_route_to_command(
    update: dict[str, Any],
    state: BaseModel,
    *,
    payload_field: str,
    route_field: str,
    valid_targets: set[str],
    channel_key: str,
    count_field: str,
    max_hops: int,
    on_exhaust: str,
    exit_name: str,
    node_name: str,
    entry_name: str,
    target_resolve: dict[str, str] | None = None,
) -> Command:
    """Shared Portal routing decision: state-update dict -> ``Command(goto=...)``.

    Extracted from :func:`make_portal_fn`'s former inline ``_to_command``
    closure so the SAME target-validation / hop-budget /
    mesh-channel-write logic is reused by both the atomic mesh-member wrapper
    (above) and the agent/act-cycle wrapper (``make_portal_agent_cycle_fn``,
    below) — no second, divergent implementation of Portal's routing decision.
    Still lives in factory.py only, per guard G1
    (``TestCommandConstructionMonopoly``).

    ``target_resolve`` maps a DX-visible peer name to its real LangGraph node
    name (the entry-label map, design portal-addressability-2026-07-15.md
    mechanism 1) — atomic peers map to themselves; an agent/act peer's real
    entry is ``{peer}__agent``. Defaults to identity when omitted (v1 atomic
    mesh, unaffected by nnds9).
    """
    payload = update[payload_field]
    target = getattr(payload, route_field)
    if target not in valid_targets:
        raise ExecutionError.build(
            "Portal route target is not a declared peer",
            expected=f"one of {sorted(valid_targets)}",
            found=f"route field '{route_field}'={target!r}",
            node=node_name,
            hint="a mesh member may route only to a declared peer or HANDOFF_END",
        )
    # HANDOFF_END is a clean mesh exit — never budget-gated, never counted.
    if target == HANDOFF_END:
        return Command(goto=exit_name, update={**update, channel_key: payload})
    resolved_target = (target_resolve or {}).get(target, target)
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
            "Portal handoff exceeded max_hops",
            expected=f"convergence within {max_hops} hops",
            found=f"{max_hops} hops exhausted",
            node=entry_name,
            hint="raise the entry's max_hops or route to HANDOFF_END sooner",
        )
    return Command(
        goto=resolved_target,
        update={**update, channel_key: payload, count_field: current + 1},
    )


def make_portal_agent_cycle_fn(
    node: Node,
    portal: Portal,
    entry_field: str,
    exit_name: str,
    *,
    max_hops: int,
    on_exhaust: str,
    entry_name: str,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    tool_factory_lookup: dict[str, Callable] | None = None,
    target_resolve: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build an agent/act Portal mesh-member's ReAct-cycle bodies.

    The mesh-member counterpart of :func:`make_portal_fn` for a member whose
    ``mode`` is ``agent``/``act``: such a node compiles to THREE parent nodes
    (``{node}__agent``/``__tools``/``__parse``, built by
    ``_agent_cycle.make_agent_cycle_bodies`` — imported function-locally here;
    no reverse dependency from factory.py to ``_agent_cycle`` exists at module
    scope, so this stays the sole new import site). Only the terminal
    ``__parse`` hop changes: its returned state-update dict is piped through
    the SAME :func:`_portal_route_to_command` routing decision
    ``make_portal_fn`` uses, so the mesh's target-validation / hop-budget /
    channel-write logic is single-sourced across atomic and agent/act
    members. The agent/tools nodes and the 3-way router are returned
    UNCHANGED — Mechanism 2 (mesh-transparent exit,
    portal-addressability-2026-07-15.md) only touches the exit node.

    Returns a dict shaped like ``_agent_cycle.make_agent_cycle_bodies``'s
    result (``names``/``agent``/``tools``/``router``) PLUS ``parse`` replaced
    by Command-returning (sync, async) callables, ready for
    ``_wiring._add_portal_agent_cycle_member``.

    Any ``Command(`` construction stays HERE (factory.py), per guard G1 — the
    agent/tools/parse bodies themselves (``_agent_cycle.py``) never construct
    one.
    """
    channel_key = StateKeys.handoff_payload(entry_field)
    count_field = StateKeys.handoff_hops(entry_field)
    payload_field = primary_output_field(field_name_for(node.name), node.outputs)
    route_field = portal.route
    valid_targets = set(portal.to or ()) | {HANDOFF_END}

    parts = make_agent_cycle_bodies(node, runtime=runtime, tool_factory_lookup=tool_factory_lookup)
    parse_sync, parse_async = parts["parse"]

    def parse_and_route(state: BaseModel, config: RunnableConfig) -> Command:
        return _portal_route_to_command(
            parse_sync(state, config),
            state,
            payload_field=payload_field,
            route_field=route_field,
            valid_targets=valid_targets,
            channel_key=channel_key,
            count_field=count_field,
            max_hops=max_hops,
            on_exhaust=on_exhaust,
            exit_name=exit_name,
            node_name=node.name,
            entry_name=entry_name,
            target_resolve=target_resolve,
        )

    async def aparse_and_route(state: BaseModel, config: RunnableConfig) -> Command:
        return _portal_route_to_command(
            await parse_async(state, config),
            state,
            payload_field=payload_field,
            route_field=route_field,
            valid_targets=valid_targets,
            channel_key=channel_key,
            count_field=count_field,
            max_hops=max_hops,
            on_exhaust=on_exhaust,
            exit_name=exit_name,
            node_name=node.name,
            entry_name=entry_name,
            target_resolve=target_resolve,
        )

    return {
        "names": parts["names"],
        "agent": parts["agent"],
        "tools": parts["tools"],
        "parse": (parse_and_route, aparse_and_route),
        "router": parts["router"],
    }


def make_portal_dispatch_fn(
    node: Node,
    portal: Portal,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
) -> Runnable:
    """Build a Portal DISPATCH-mode wrapper (``route="decide"``, design §3.5/§4.2).

    Mode (b), reduced v1. Wraps the standard :func:`make_node_fn` result: the
    dispatcher body runs and emits, as its OWN typed output, a neograph spec dict
    (``portal.spec_field``) and a dispatch input dict (``portal.input_field``).
    This wrapper then, per the design's four steps:

    1. ``load_spec(spec_dict)`` -> ``Construct(...)`` — THE validation gate. This is
       the SAME eager ``_validate_node_chain`` (``construct.py:194``) hand-written
       pipelines pass through (ANTI-BAND-AID: no bespoke validator, no schema
       subset). A bad spec raises ``ConstructError``/``ConfigurationError`` HERE,
       BEFORE anything executes; we re-raise it WRAPPED in ``ExecutionError`` naming
       the spec (``on_invalid="raise"``, §3.5) with the ``ConstructError`` chained as
       ``__cause__``.
    2. Output-contract check: if the built flow declares an ``output`` boundary, it
       must equal ``portal.output`` (resolved via ``lookup_type`` when a str) —
       ``ExecutionError`` on mismatch, before compile. Top-level emitted flows carry
       no ``output`` boundary; their contract is enforced at step 4 by the typed
       result scan (a flow that produces the wrong type yields no assignable value).
    3. ``compile(sub, scripted=portal.scripted, conditions=portal.conditions)`` —
       the emitted flow may wire ONLY the pre-registered building blocks
       (D-DISPATCH-REGISTRIES); an unknown ``scripted_fn`` fails loud at compile. NO
       checkpointer is passed — mode-(b) durability is documented-opaque (§7; Tier-2
       is neograph-mrb2y).
    4. Invoke the compiled flow with ``input_field``'s dict and extract the value
       assignable to ``portal.output`` via the shared :func:`_scan_subgraph_output`
       (the same typed-output scan sub-constructs use); ``None`` (nothing produced
       the required type) raises ``ExecutionError``. The result is written to a new
       regular (fingerprinted) state field ``{node_field}_dispatch``.

    Reduced v1 is a LINEAR arm: this wrapper returns a plain state-update dict (NOT a
    ``Command``), so ``_add_portal_dispatch`` wires it with a static next edge and
    the G1 Command-construction monopoly stays narrow.

    Sync/async parity: the ``_prepare`` (load/contract/compile) and ``_finish``
    (scan/write) steps are shared; only ``compiled.invoke`` vs ``await
    compiled.ainvoke`` differs between the twins (mirrors :func:`make_subgraph_fn`).
    """
    field_name = field_name_for(node.name)
    payload_field = primary_output_field(field_name, node.outputs)
    dispatch_field = output_field_name(field_name, "dispatch")
    spec_field = portal.spec_field
    input_field = portal.input_field
    assert spec_field is not None and input_field is not None  # dispatch-mode invariant (T1 validation)

    inner = make_node_fn(
        node,
        runtime=runtime,
        scripted_lookup=scripted_lookup,
        tool_factory_lookup=tool_factory_lookup,
    )

    def _resolve_expected() -> type[BaseModel]:
        out = portal.output
        if isinstance(out, str):
            return lookup_type(out)
        assert out is not None  # dispatch-mode invariant (T1 validation)
        return out

    def _prepare(update: dict[str, Any]) -> tuple[Any, type[BaseModel], str, Any]:
        """Shared pre-invoke: read the emitted spec/input, run the SAME gate, compile.

        Returns ``(compiled, expected_output, spec_name, dispatch_input)``. Contains
        NO invoke — the sync/async twins supply that so the gate + compile logic
        cannot drift between them.
        """
        # `compile` is the ONE function-local import here: compiler.py imports
        # _wiring -> factory, so a module-level `from neograph.compiler import
        # compile` would cycle. load_spec / _scan_subgraph_output / lookup_type
        # are module-level (their modules do not import factory).
        from neograph.compiler import compile as compile_construct

        decision = update[payload_field]
        spec_dict = getattr(decision, spec_field)
        dispatch_input = getattr(decision, input_field)
        expected = _resolve_expected()
        spec_name = spec_dict.get("name", "<unnamed>") if isinstance(spec_dict, dict) else "<unnamed>"

        try:
            sub = load_spec(spec_dict)
        except (ConstructError, ConfigurationError) as gate_error:
            # The emitted spec failed the SAME Construct(...) gate as a hand-written
            # pipeline — surface it wrapped, naming the spec, BEFORE anything runs.
            raise ExecutionError.build(
                "dispatched flow spec is invalid",
                construct=spec_name,
                found=str(gate_error),
                node=node.name,
                hint="the emitted spec failed the same Construct(...) validation gate as a hand-written pipeline",
            ) from gate_error

        if sub.output is not None and sub.output is not expected:
            raise ExecutionError.build(
                "dispatched flow output-contract mismatch",
                expected=getattr(expected, "__name__", str(expected)),
                found=f"flow '{spec_name}' declares output {getattr(sub.output, '__name__', sub.output)}",
                node=node.name,
                hint="the emitted flow's declared output must equal Portal.output",
            )

        compiled = compile_construct(sub, scripted=portal.scripted, conditions=portal.conditions)
        return compiled, expected, spec_name, dispatch_input

    def _finish(
        update: dict[str, Any], result: dict[str, Any], expected: type[BaseModel], spec_name: str
    ) -> dict[str, Any]:
        """Shared post-invoke: extract the typed output, write ``{node}_dispatch``."""
        out = _scan_subgraph_output(result, expected)
        if out is None:
            raise ExecutionError.build(
                "dispatched flow did not produce the required output type",
                expected=getattr(expected, "__name__", str(expected)),
                found=f"flow '{spec_name}' produced no value assignable to it",
                node=node.name,
                hint="a route='decide' flow must produce Portal.output",
            )
        return {**update, dispatch_field: out}

    def dispatch_wrapper(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        update = inner.invoke(state, config)
        compiled, expected, spec_name, dispatch_input = _prepare(update)
        result = compiled.invoke(dispatch_input, config=config)
        return _finish(update, result, expected, spec_name)

    async def adispatch_wrapper(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        update = await inner.ainvoke(state, config)
        compiled, expected, spec_name, dispatch_input = _prepare(update)
        result = await compiled.ainvoke(dispatch_input, config=config)
        return _finish(update, result, expected, spec_name)

    wrapper = RunnableLambda(dispatch_wrapper, afunc=adispatch_wrapper)
    return named(wrapper, node.name, mode="portal-dispatch", output_type=_type_name(node.outputs))


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
