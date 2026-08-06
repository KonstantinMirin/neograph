"""Node-function construction — turns Node definitions into LangGraph callables."""

from __future__ import annotations

# --- stdlib names factory.py imported and RE-EXPORTED before the split; the
# --- moved raw wrappers were their only local consumer here.
import inspect  # noqa: E402,F401
import time  # noqa: E402,F401
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from neograph._agent_cycle import make_agent_cycle_bodies
from neograph._agent_spec_dispatch import make_dispatch_gate
from neograph._dispatch import _dispatch_for_mode
from neograph._execute import _aexecute_node, _execute_node, _type_name
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._normalize import primary_output_field
from neograph._portal_route import MeshContext, PortalRouteSpec

# --- extracted cluster (neograph-3ffdg.6), re-exported so existing
# --- `from neograph.factory import ...` call sites keep resolving unchanged.
from neograph._raw_dispatch import _make_araw_wrapper, _make_raw_wrapper  # noqa: E402,F401
from neograph._state_bus import adapt_state
from neograph._state_keys import StateKeys
from neograph._subconstruct import make_subgraph_fn
from neograph._trace import named
from neograph.errors import ConfigurationError, ExecutionError
from neograph.modifiers import HANDOFF_END, Operator, Portal
from neograph.naming import field_name_for, output_field_name
from neograph.node import Node

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from neograph.construct import Construct

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
    spec: PortalRouteSpec,
    ctx: MeshContext,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
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

    ``spec``/``ctx`` (neograph-dgbqv.4, P9) are the frozen routing bundles
    ``_add_portal_mesh`` builds ONCE per mesh (``ctx``) and once per member
    (``spec``) — see ``_portal_route.py``. Everything this wrapper used to
    derive itself (``StateKeys.handoff_payload``/``handoff_hops``, the
    payload/route field names, the valid-target set, the proposed-field key)
    now lives on one of those two bundles.
    """
    inner = make_node_fn(
        node,
        runtime=runtime,
        scripted_lookup=scripted_lookup,
        tool_factory_lookup=tool_factory_lookup,
    )

    def portal_wrapper(state: BaseModel, config: RunnableConfig) -> Command:
        return _portal_route_to_command(inner.invoke(state, config), state, spec, ctx)

    async def aportal_wrapper(state: BaseModel, config: RunnableConfig) -> Command:
        return _portal_route_to_command(await inner.ainvoke(state, config), state, spec, ctx)

    wrapper = RunnableLambda(portal_wrapper, afunc=aportal_wrapper)
    return named(wrapper, node.name, mode="portal", output_type=_type_name(node.outputs))


def _portal_route_to_command(
    update: dict[str, Any],
    state: BaseModel,
    spec: PortalRouteSpec,
    ctx: MeshContext,
) -> Command:
    """Shared Portal routing decision: state-update dict -> ``Command(goto=...)``.

    Extracted from :func:`make_portal_fn`'s former inline ``_to_command``
    closure so the SAME target-validation / hop-budget /
    mesh-channel-write logic is reused by both the atomic mesh-member wrapper
    (above) and the agent/act-cycle wrapper (``make_portal_agent_cycle_fn``,
    below) — no second, divergent implementation of Portal's routing decision.
    Still lives in factory.py only, per guard G1
    (``TestCommandConstructionMonopoly``).

    ``ctx.entry_label_map`` maps a DX-visible peer name to its real LangGraph
    node name (design portal-addressability-2026-07-15.md mechanism 1) —
    atomic peers map to themselves; an agent/act peer's real entry is
    ``{peer}__agent``.

    ``spec.approve_name``/``spec.proposed_field``: when the member carries an
    Operator approval gate, a PEER route (never HANDOFF_END — "leaving the
    mesh is not a handoff") detours to ``{member}__approve`` INSTEAD of the
    peer directly, carrying the proposed (already hop-budget-checked) target
    on ``proposed_field`` — the hop counter is NOT incremented here;
    ``make_portal_approval_fn`` increments it ONLY on approval, so a rejected
    hop costs nothing.
    """
    payload = update[spec.payload_field]
    target = getattr(payload, spec.route_field)
    if target not in spec.valid_targets:
        raise ExecutionError.build(
            "Portal route target is not a declared peer",
            expected=f"one of {sorted(spec.valid_targets)}",
            found=f"route field '{spec.route_field}'={target!r}",
            node=spec.node_name,
            hint="a mesh member may route only to a declared peer or HANDOFF_END",
        )
    # HANDOFF_END is a clean mesh exit — never budget-gated, never counted,
    # and never approval-guarded (Operator guards the HANDOFF, not the exit).
    if target == HANDOFF_END:
        return Command(goto=ctx.exit_name, update={**update, ctx.channel_key: payload})
    resolved_target = ctx.entry_label_map.get(target, target)
    # Peer continuation: enforce the entry's hop budget BEFORE emitting the
    # goto (or the approval detour). Counter bootstrap (absent/None -> 0) lives
    # in StateBus.get_counter; read the SHARED counter from incoming state so
    # hops accumulate across members (the update dict never carries it — a
    # from-update read would always bootstrap 0 and break accumulation).
    current = adapt_state(state).get_counter(ctx.count_field)
    if current >= ctx.max_hops:
        if ctx.on_exhaust == "exit":
            return Command(
                goto=ctx.exit_name,
                update={**update, ctx.channel_key: payload, ctx.count_field: current},
            )
        raise ExecutionError.build(
            "Portal handoff exceeded max_hops",
            expected=f"convergence within {ctx.max_hops} hops",
            found=f"{ctx.max_hops} hops exhausted",
            node=ctx.entry_name,
            hint="raise the entry's max_hops or route to HANDOFF_END sooner",
        )
    if spec.approve_name is not None:
        assert spec.proposed_field is not None
        return Command(
            goto=spec.approve_name,
            update={**update, ctx.channel_key: payload, spec.proposed_field: resolved_target},
        )
    return Command(
        goto=resolved_target,
        update={**update, ctx.channel_key: payload, ctx.count_field: current + 1},
    )


def _tool_handoff_to_command(
    update: dict[str, Any],
    state: BaseModel,
    spec: PortalRouteSpec,
    ctx: MeshContext,
) -> Command:
    """Tool-triggered-handoff routing decision (design portal-tool-triggered-handoff
    §3.3): a tool-triggered agent member's ``{node}__tools`` update dict ->
    ``Command(goto=...)``.

    The narrower sibling of :func:`_portal_route_to_command` — no
    ``payload_field``/``route_field`` because there is NO typed payload to read a
    route off of: the routing target is a synthesized ``transfer_to_<peer>`` tool
    call the tools body already resolved to a peer name and stamped onto the
    TRANSIENT ``spec.handoff_target_key`` sentinel. This function pops that
    sentinel back OUT before building the ``Command(update=...)``, so it never
    enters LangGraph state (design §6). Absent a handoff call (``target is
    None``) it emits ``Command(goto=spec.loopback_target)`` — the ordinary
    ReAct loopback, expressed as a dynamic goto because a tool-triggered tools
    node has NO static out-edge (LangGraph would silently double-execute both,
    design §3.4).

    Reuses the SAME hop-budget / ``HANDOFF_END`` / entry-label machinery as
    :func:`_portal_route_to_command`: a peer hop is budget-gated
    (``count >= max_hops`` before emitting the goto — Loop parity), resolved
    through ``ctx.entry_label_map`` to the peer's real entry node. Confined to
    factory.py per guard G1 (``TestCommandConstructionMonopoly``).
    """
    assert spec.handoff_target_key is not None
    assert spec.loopback_target is not None
    target = update.pop(spec.handoff_target_key, None)
    if target is None:
        # No handoff this turn — the ordinary ReAct loopback (design §3.4).
        return Command(goto=spec.loopback_target, update=update)
    # HANDOFF_END parity with _portal_route_to_command (a clean mesh exit, never
    # budget-gated). A synthesized handoff tool only ever names a real peer, so
    # this stays a defensive mirror of the sibling function's shape.
    if target == HANDOFF_END:
        return Command(goto=ctx.exit_name, update=update)
    resolved_target = ctx.entry_label_map.get(target, target)
    # Peer continuation: enforce the entry's hop budget BEFORE emitting the goto.
    # Read the SHARED counter from incoming state so hops accumulate across
    # members (the update dict never carries it).
    current = adapt_state(state).get_counter(ctx.count_field)
    if current >= ctx.max_hops:
        if ctx.on_exhaust == "exit":
            return Command(goto=ctx.exit_name, update=update)
        raise ExecutionError.build(
            "Portal handoff exceeded max_hops",
            expected=f"convergence within {ctx.max_hops} hops",
            found=f"{ctx.max_hops} hops exhausted",
            node=ctx.entry_name,
            hint="raise the entry's max_hops or route to HANDOFF_END sooner",
        )
    return Command(goto=resolved_target, update={**update, ctx.count_field: current + 1})


def make_portal_approval_fn(
    node_name: str,
    operator: Operator,
    *,
    count_field: str,
    proposed_field: str,
    exit_name: str,
    condition_lookup: dict[str, Callable] | None = None,
) -> Callable[[BaseModel, RunnableConfig], Command]:
    """Build the ``{member}__approve`` node (neograph-kdr1u, Portal+Operator
    D4 lift) -- the ONLY ``interrupt()`` site on an approval-guarded member's
    dynamic path.

    ANTI-BAND-AID (the D4 invariant): the interrupt lives ONLY here, never in
    the member's own wrapper (:func:`make_portal_fn`) -- that naive shape
    re-runs the member's LLM/tool spend on resume (LangGraph resumes an
    interrupted node FROM THE TOP), proven failing by the spike
    (``tests/test_spike_portal_operator_approval.py::TestNaiveInWrapperShapeFailsTheCrux``).
    Splicing this cheap, side-effect-free node onto the ``Command(goto)`` path
    means a resume re-runs ONLY this node.

    ``operator.when`` gates whether the node interrupts at all -- a falsy
    predicate passes through to the proposed target WITHOUT pausing (mirrors
    ``_add_operator_check``'s conditional-interrupt shape). On resume, the
    decision is expected to be a dict with an ``"approved"`` key (the
    ``run(graph, resume={"approved": bool}, ...)`` convention): truthy ->
    ``Command(goto=proposed_target)`` with the hop counter incremented
    (an approved hop costs exactly one -- the SAME accounting a direct,
    unguarded hop would have used, just deferred past the approval); falsy ->
    ``Command(goto=exit_name)`` WITHOUT incrementing (a rejected hop is free).

    Confined to factory.py per guard G1 (``TestCommandConstructionMonopoly``).
    """
    # Cycle-avoidance function-local import: _wiring.py imports factory.py
    # (make_portal_fn et al.), so a module-level import here would cycle.
    # Mirrors _prepare's `from neograph.compiler import compile` pattern.
    from neograph._wiring import _resolve_condition

    condition_fn = _resolve_condition(operator.when, condition_lookup)

    def approval_check(state: BaseModel, config: RunnableConfig) -> Command:
        should_pause = condition_fn(state)
        if should_pause:
            decision = interrupt(should_pause)
            approved = isinstance(decision, dict) and bool(decision.get("approved"))
        else:
            approved = True
        proposed_target = getattr(state, proposed_field)
        if not approved:
            return Command(goto=exit_name, update={})
        current = adapt_state(state).get_counter(count_field)
        return Command(goto=proposed_target, update={count_field: current + 1})

    return approval_check


def make_portal_subgraph_fn(
    sub: Construct,
    sub_graph: CompiledStateGraph,
    portal: Portal,
    spec: PortalRouteSpec,
    ctx: MeshContext,
) -> Runnable:
    """Build a sub-construct Portal mesh-member function (do0d9, §3.1 site 1).

    The sub-construct counterpart of :func:`make_portal_fn` for a mesh member
    that is a whole ``Construct``. Wraps the standard :func:`make_subgraph_fn`
    runnable (both sync/async twins) and pipes its returned update dict through
    the SAME :func:`_portal_route_to_command` routing decision atomic and
    agent/act members use — so the target-validation / hop-budget / mesh-channel
    write logic is single-sourced across every member class (§3 steps 3-4).

    A routing decision made INSIDE the isolated sub-construct is carried OUT as
    the sub-construct's declared-output payload (``Construct.output``); the
    parent mesh routes on it exactly as a same-level handoff. This is the SOLE
    new ``Command(``-adjacent site, and it delegates the actual ``Command`` build
    to the already-existing ``_portal_route_to_command`` — so NO new ``Command(``
    literal is added at all (guard G1 satisfied, §4 Q5).

    ``spec.payload_field`` is keyed off :func:`make_subgraph_fn`'s update dict —
    ``{field_name_for(sub.name): payload}`` — NOT ``node.outputs`` (a ``Construct``
    has no ``.outputs``; its boundary output is ``.output``, singular), built by
    :meth:`PortalRouteSpec.for_sub_construct`. The boundary INPUT is sourced
    deterministically from the parent handoff channel
    (``make_subgraph_fn(handoff_channel=...)``, §3.1 site 7), never a blind
    reverse type-scan that could feed a same-typed decoy.

    Sync/async parity: wraps BOTH ``make_subgraph_fn`` twins and pipes each
    through the sync/async pair already present in ``_portal_route_to_command``'s
    callers (mirrors :func:`make_portal_fn`).
    """
    inner = make_subgraph_fn(sub, sub_graph, handoff_channel=ctx.channel_key)

    def portal_subgraph_wrapper(state: BaseModel, config: RunnableConfig) -> Command:
        return _portal_route_to_command(inner.invoke(state, config), state, spec, ctx)

    async def aportal_subgraph_wrapper(state: BaseModel, config: RunnableConfig) -> Command:
        return _portal_route_to_command(await inner.ainvoke(state, config), state, spec, ctx)

    wrapper = RunnableLambda(portal_subgraph_wrapper, afunc=aportal_subgraph_wrapper)
    output_name = sub.output.__name__ if sub.output is not None else None
    return named(wrapper, sub.name, mode="portal-subgraph", output_type=output_name)


def make_portal_agent_cycle_fn(
    node: Node,
    portal: Portal,
    spec: PortalRouteSpec,
    ctx: MeshContext,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    tool_factory_lookup: dict[str, Callable] | None = None,
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
    parts = make_agent_cycle_bodies(node, runtime=runtime, tool_factory_lookup=tool_factory_lookup)
    parse_sync, parse_async = parts["parse"]

    def parse_and_route(state: BaseModel, config: RunnableConfig) -> Command:
        return _portal_route_to_command(parse_sync(state, config), state, spec, ctx)

    async def aparse_and_route(state: BaseModel, config: RunnableConfig) -> Command:
        return _portal_route_to_command(await parse_async(state, config), state, spec, ctx)

    return {
        "names": parts["names"],
        "agent": parts["agent"],
        "tools": parts["tools"],
        "parse": (parse_and_route, aparse_and_route),
        "router": parts["router"],
    }


def make_portal_agent_cycle_tool_handoff_fn(
    node: Node,
    portal: Portal,
    spec: PortalRouteSpec,
    ctx: MeshContext,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    tool_factory_lookup: dict[str, Callable] | None = None,
) -> dict[str, Any]:
    """Build a tool-triggered agent/act Portal mesh member's ReAct-cycle bodies
    (design portal-tool-triggered-handoff §3.3).

    The ``trigger="tool"`` counterpart of :func:`make_portal_agent_cycle_fn`. It
    threads ``handoff_portal=portal`` into ``make_agent_cycle_bodies`` so every
    agent turn binds the synthesized ``transfer_to_<peer>`` tools, then wraps TWO
    exits with a ``Command``-returning body:

    - ``tools``/``atools`` -> :func:`_tool_handoff_to_command`: on a detected
      handoff call, route to the peer's entry (hop-budget-gated); absent one,
      emit the ordinary ReAct loopback ``Command(goto={node}__agent)`` (there is
      NO static ``tools -> agent`` edge for a tool-triggered member — LangGraph
      would silently double-execute, design §3.4).
    - ``parse``/``aparse`` -> :func:`_portal_route_to_command`: wired EXACTLY like
      today's ``trigger="output"`` member, the reconverging exit for the ordinary
      "no more tool calls" completion path (a tool-triggered member may still
      complete normally without ever calling a handoff tool — both exits coexist).

    ``spec`` (built by :meth:`PortalRouteSpec.for_tool_member`) carries BOTH
    the routing fields ``parse``/``aparse`` need (payload_field/route_field/
    valid_targets) AND the tool-handoff fields ``tools``/``atools`` need
    (handoff_target_key/loopback_target) — one bundle serves both exits.

    Any ``Command(`` construction stays HERE (factory.py) per guard G1.
    """
    parts = make_agent_cycle_bodies(
        node, runtime=runtime, tool_factory_lookup=tool_factory_lookup, handoff_portal=portal
    )
    tools_sync, tools_async = parts["tools"]
    parse_sync, parse_async = parts["parse"]

    def tools_and_route(state: BaseModel, config: RunnableConfig) -> Command:
        return _tool_handoff_to_command(tools_sync(state, config), state, spec, ctx)

    async def atools_and_route(state: BaseModel, config: RunnableConfig) -> Command:
        return _tool_handoff_to_command(await tools_async(state, config), state, spec, ctx)

    def parse_and_route(state: BaseModel, config: RunnableConfig) -> Command:
        return _portal_route_to_command(parse_sync(state, config), state, spec, ctx)

    async def aparse_and_route(state: BaseModel, config: RunnableConfig) -> Command:
        return _portal_route_to_command(await parse_async(state, config), state, spec, ctx)

    return {
        "names": parts["names"],
        "agent": parts["agent"],
        "tools": (tools_and_route, atools_and_route),
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
    exit_name: str | None = None,
) -> Runnable:
    """Build a Portal DISPATCH-mode wrapper (``route="decide"``, design §3.5/§4.2).

    Mode (b), reduced v1. Wraps the standard :func:`make_node_fn` result: the
    dispatcher body runs and emits, as its OWN typed output, a neograph-flavored
    Agent Spec dict (``portal.spec_field``) and a dispatch input dict
    (``portal.input_field``). This wrapper then, per the design's four steps:

    1. ``AgentSpecDeserializer().from_dict(spec_dict)`` -> ``Flow`` ->
       ``from_agent_spec(flow)`` -> ``Construct(...)`` — THE validation gate. This
       is the SAME single modifier-aware runtime spec-loading path ``to_agent_spec``
       (export) and a mode-(b) planner (dispatch) share — never a second, bespoke
       native-``Spec``-dict serializer. It is also the SAME eager
       ``_validate_node_chain`` (``construct.py:194``) hand-written pipelines pass
       through (ANTI-BAND-AID: no bespoke validator, no schema subset). A bad spec
       raises ``ConstructError``/``ConfigurationError``/``ValidationError`` HERE,
       BEFORE anything executes; we re-raise it WRAPPED in ``ExecutionError`` naming
       the spec (``on_invalid="raise"``, §3.5) with the original chained as
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

    Sync/async parity: steps 1-3 (load/contract/compile) and step 4's write half
    (scan/write) are the shared, no-Command gate built by
    :func:`neograph._agent_spec_dispatch.make_dispatch_gate` (neograph-jtawq.9) and
    consumed here via its ``prepare``/``finish``/``check_and_increment_depth``
    handle; only ``compiled.invoke`` vs ``await compiled.ainvoke`` differs between
    the twins (mirrors :func:`make_subgraph_fn`).
    """
    field_name = field_name_for(node.name)
    payload_field = primary_output_field(field_name, node.outputs)
    dispatch_field = output_field_name(field_name, "dispatch")

    inner = make_node_fn(
        node,
        runtime=runtime,
        scripted_lookup=scripted_lookup,
        tool_factory_lookup=tool_factory_lookup,
    )

    # The no-Command half of the gate (validate the emitted spec through the
    # SAME from_agent_spec + compile() path a hand-written pipeline passes,
    # extract the typed result, bound recursion depth) lives in
    # _agent_spec_dispatch.py (neograph-jtawq.9) so G1's Command-construction
    # monopoly stays exactly {factory.py, runner.py} — this module never
    # re-derives the gate logic, only calls it and consumes the handle.
    gate = make_dispatch_gate(node, portal, payload_field=payload_field, dispatch_field=dispatch_field)

    error_field = StateKeys.dispatch_error(field_name) if portal.on_invalid == "route_to_error" else None

    def dispatch_wrapper(state: BaseModel, config: RunnableConfig) -> dict[str, Any] | Command:
        child_config = gate.check_and_increment_depth(config)
        update = inner.invoke(state, config)
        compiled, expected, spec_name, dispatch_input, gate_error_msg = gate.prepare(update)
        if gate_error_msg is not None:
            assert error_field is not None and exit_name is not None  # route_to_error invariant
            return Command(goto=portal.error_handler, update={**update, error_field: gate_error_msg})
        result = compiled.invoke(dispatch_input, config=child_config)
        final_update = gate.finish(update, result, expected, spec_name)
        if portal.on_invalid == "route_to_error":
            assert exit_name is not None
            return Command(goto=exit_name, update=final_update)
        return final_update

    async def adispatch_wrapper(state: BaseModel, config: RunnableConfig) -> dict[str, Any] | Command:
        child_config = gate.check_and_increment_depth(config)
        update = await inner.ainvoke(state, config)
        compiled, expected, spec_name, dispatch_input, gate_error_msg = gate.prepare(update)
        if gate_error_msg is not None:
            assert error_field is not None and exit_name is not None  # route_to_error invariant
            return Command(goto=portal.error_handler, update={**update, error_field: gate_error_msg})
        result = await compiled.ainvoke(dispatch_input, config=child_config)
        final_update = gate.finish(update, result, expected, spec_name)
        if portal.on_invalid == "route_to_error":
            assert exit_name is not None
            return Command(goto=exit_name, update=final_update)
        return final_update

    wrapper = RunnableLambda(dispatch_wrapper, afunc=adispatch_wrapper)
    return named(wrapper, node.name, mode="portal-dispatch", output_type=_type_name(node.outputs))
