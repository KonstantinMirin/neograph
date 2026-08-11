"""Portal tool-triggered handoff (``trigger="tool"``) — design
portal-tool-triggered-handoff-2026-07-27.md (neograph-s7zt3.14).

Proves the first-class native capability end to end:

- an agent mesh member that calls a synthesized ``transfer_to_<peer>`` tool hands
  control to the peer's real entry node (``{peer}__agent``) via a
  ``Command(goto=...)`` emitted from its ``{node}__tools`` superstep — never a
  typed routing field;
- the tool-triggered ``{node}__tools`` node has NO static outgoing edge (guard a
  — LangGraph would silently double-execute a static edge + a
  ``destinations=``-registered Command target);
- ``trigger="tool"`` on a non-agent/act member fails LOUD at assembly (guard b);
- ``trigger`` is peer-mode-only (``model_post_init``);
- three-surface parity (the modifier is author-set, so it must survive every
  Node-construction surface unchanged).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import Construct, Node, Portal, arun, compile, node, run
from neograph.errors import ConfigurationError, ConstructError
from tests.fakes import ReActFake, build_fake_llm_kwargs, build_test_compile_kwargs


class Handoff(BaseModel, frozen=True):
    goto: str


def _tool_trigger_mesh() -> Construct:
    """A two-agent tool-triggered mesh: ``triage`` (model tier 'router') calls
    ``transfer_to_researcher``; ``researcher`` (tier 'worker') completes normally
    and leaves the mesh via goto=HANDOFF_END."""
    triage = Node(
        name="triage",
        mode="agent",
        model="router",
        prompt="test/triage",
        inputs={"handoff": Handoff},
        outputs=Handoff,
        tools=[],
    ) | Portal(to=["researcher"], trigger="tool", max_hops=6)
    researcher = Node(
        name="researcher",
        mode="agent",
        model="worker",
        prompt="test/research",
        inputs={"handoff": Handoff},
        outputs=Handoff,
        tools=[],
    ) | Portal(to=["triage"], trigger="tool")
    return Construct("tool-trigger-mesh", nodes=[triage, researcher])


def _tool_trigger_fakes() -> dict[str, ReActFake]:
    return {
        # triage: turn 0 emits the synthesized transfer_to_researcher call.
        "router": ReActFake(
            tool_calls=[[{"name": "transfer_to_researcher", "args": {}, "id": "tr1"}]],
            final=lambda m: m(goto="__end__"),
            output_model=Handoff,
        ),
        # researcher: no tool call — completes and routes to HANDOFF_END.
        "worker": ReActFake(tool_calls=[[]], final=lambda m: m(goto="__end__"), output_model=Handoff),
    }


class TestToolTriggeredHandoffRouting:
    """A transfer_to_<peer> tool call routes to the peer's real entry node."""

    def test_transfer_tool_call_routes_to_peer_entry_sync(self):
        fakes = _tool_trigger_fakes()
        graph = compile(
            _tool_trigger_mesh(),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: fakes[tier]),
        )
        result = run(graph, input={})
        # researcher ran (received the handoff) and completed with goto=HANDOFF_END.
        assert result.get("researcher") == Handoff(goto="__end__")
        # triage handed off via the tool BEFORE producing typed output, so it wrote
        # nothing to its own output field.
        assert result.get("triage") is None

    @pytest.mark.asyncio
    async def test_transfer_tool_call_routes_to_peer_entry_async(self):
        fakes = _tool_trigger_fakes()
        graph = compile(
            _tool_trigger_mesh(),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: fakes[tier]),
        )
        result = await arun(graph, input={})
        assert result.get("researcher") == Handoff(goto="__end__")
        assert result.get("triage") is None


class TestToolTriggeredToolsNodeHasNoStaticEdge:
    """Guard (a): a tool-triggered ``{node}__tools`` node routes ONLY via dynamic
    Command targets (destinations=) — never a static ``tools -> agent`` edge, which
    LangGraph silently double-executes alongside the Command target."""

    def test_tool_triggered_tools_node_has_no_static_outgoing_edge(self):
        graph = compile(
            _tool_trigger_mesh(),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: None),
        )
        static_from_tools = [
            (e.source, e.target)
            for e in graph.get_graph().edges
            if e.source.endswith("__tools") and not e.conditional
        ]
        assert static_from_tools == [], (
            "a tool-triggered {node}__tools node must have NO static outgoing edge "
            f"(LangGraph double-executes it alongside the Command target); found: {static_from_tools}"
        )

    def test_output_triggered_member_keeps_its_static_tools_loopback(self):
        """Contrast: a plain trigger='output' agent mesh member DOES keep the
        static tools->agent loopback (the destinations= replacement is scoped to
        tool-triggered members only)."""
        triage = Node(
            name="triage",
            mode="agent",
            model="router",
            prompt="test/triage",
            inputs={"handoff": Handoff},
            outputs=Handoff,
            tools=[],
        ) | Portal(to=["researcher"], max_hops=6)  # trigger defaults to "output"
        researcher = Node(
            name="researcher",
            mode="agent",
            model="worker",
            prompt="test/research",
            inputs={"handoff": Handoff},
            outputs=Handoff,
            tools=[],
        ) | Portal(to=["triage"])
        graph = compile(
            Construct("output-trigger-mesh", nodes=[triage, researcher]),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: None),
        )
        static_from_tools = {
            (e.source, e.target)
            for e in graph.get_graph().edges
            if e.source.endswith("__tools") and not e.conditional
        }
        assert ("triage__tools", "triage__agent") in static_from_tools
        assert ("researcher__tools", "researcher__agent") in static_from_tools


class TestToolTriggerRequiresAgentMember:
    """Guard (b): trigger='tool' requires an agent/act member (validated at
    assembly, mirrored by a check-fixture pair)."""

    def test_trigger_tool_on_scripted_member_raises_at_assembly(self):
        from tests.fakes import register_scripted

        register_scripted("tt_scripted", lambda i, c: Handoff(goto="__end__"))
        with pytest.raises(ConstructError, match=r'trigger="tool" requires an agent/act member'):
            Construct(
                "bad-tool-trigger",
                nodes=[
                    Node.scripted("triage", fn="tt_scripted", outputs=Handoff)
                    | Portal(to=["billing"], trigger="tool", max_hops=6),
                    Node.scripted("billing", fn="tt_scripted", inputs={"handoff": Handoff}, outputs=Handoff)
                    | Portal(to=["triage"]),
                ],
            )

    def test_trigger_tool_on_agent_member_assembles(self):
        # No raise — an agent member is legal (the mesh itself is built in
        # _tool_trigger_mesh and compiled elsewhere).
        _tool_trigger_mesh()


class TestTriggerIsPeerModeOnly:
    """trigger is a peer-mode knob — forbidden in dispatch mode (model_post_init)."""

    def test_trigger_with_dispatch_route_raises(self):
        with pytest.raises(ConfigurationError, match="dispatch mode forbids peer-mode knobs"):
            Portal(
                route="decide",
                trigger="tool",
                spec_field="spec",
                input_field="inp",
                output=Handoff,
                max_depth=3,
            )


class TestToolTriggerThreeSurfaceParity:
    """The trigger sub-mode is an author-set modifier field, so it must survive
    every Node-construction surface unchanged (no decorator-only sourcing). The
    programmatic pipe form ``Node() | Portal()`` is the SAME object path as the
    declarative form, so exercising declarative + @node covers all three."""

    def test_declarative_node_pipe_preserves_trigger(self):
        member = Node(
            name="triage", mode="agent", model="router", prompt="p", outputs=Handoff, tools=[]
        ) | Portal(to=["researcher"], trigger="tool")
        assert member.modifier_set.portal is not None
        assert member.modifier_set.portal.is_tool_triggered

    def test_node_decorator_pipe_preserves_trigger(self):
        @node(mode="agent", model="router", prompt="p", outputs=Handoff, tools=[])
        def triage(handoff: Handoff) -> Handoff: ...

        member = triage | Portal(to=["researcher"], trigger="tool")
        assert member.modifier_set.portal is not None
        assert member.modifier_set.portal.is_tool_triggered

    def test_both_surfaces_compile_a_tool_triggered_mesh(self):
        @node(mode="agent", model="router", prompt="p", inputs={"handoff": Handoff}, outputs=Handoff, tools=[])
        def triage(handoff: Handoff) -> Handoff: ...

        @node(mode="agent", model="worker", prompt="p", inputs={"handoff": Handoff}, outputs=Handoff, tools=[])
        def researcher(handoff: Handoff) -> Handoff: ...

        decorated = Construct(
            "decorated-tool-trigger",
            nodes=[
                triage | Portal(to=["researcher"], trigger="tool", max_hops=6),
                researcher | Portal(to=["triage"], trigger="tool"),
            ],
        )
        for c in (decorated, _tool_trigger_mesh()):
            graph = compile(c, **build_test_compile_kwargs(), **build_fake_llm_kwargs(lambda tier: None))
            # Both surfaces lower to the same tool-triggered mesh shape: each
            # member's {node}__tools node routes ONLY via dynamic Command targets
            # (no static tools->agent edge), proving trigger='tool' wiring landed
            # regardless of authoring surface.
            node_names = set(graph.get_graph().nodes)
            assert {"triage__tools", "triage__agent", "researcher__tools"} <= node_names
            static_from_tools = [
                (e.source, e.target)
                for e in graph.get_graph().edges
                if e.source.endswith("__tools") and not e.conditional
            ]
            assert static_from_tools == []


class TestToolTriggeredToolsNodeDeclaresTheMeshExit:
    """neograph-dgbqv.7: a tool-triggered ``{node}__tools`` node must DECLARE the
    mesh exit among its Command targets, because it can actually emit one.

    ``_tool_handoff_to_command`` returns ``Command(goto=ctx.exit_name)`` on
    ``HANDOFF_END`` and again when the hop budget is exhausted under
    ``on_exhaust='exit'`` -- but ``_wiring`` built ``tools_destinations`` as peers
    + the agent loopback only, while the sibling ``parse_destinations`` is
    documented as "declared peers UNION {exit} for BOTH trigger kinds".

    WHY THIS IS A DECLARATION TEST AND NOT A RUNTIME ONE. Verified by isolated
    experiment: LangGraph validates ``destinations=`` neither at compile time nor
    at run time, so the undeclared goto executes happily today and NO runtime
    failure exists to assert on. The acceptance therefore genuinely lives in the
    declared graph -- a rendered diagram omits the transition, any static analysis
    over ``destinations=`` is wrong, and the day LangGraph adds enforcement this
    becomes a silent break. Asserting the declaration is the matching locus here,
    not a weaker substitute for a behavioural test.
    """

    @staticmethod
    def _targets_from(graph: object, source_suffix: str) -> set[str]:
        return {e.target for e in graph.get_graph().edges if e.source.endswith(source_suffix)}

    def test_tools_node_declares_the_same_exit_its_parse_sibling_does(self):
        graph = compile(
            _tool_trigger_mesh(),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: None),
        )
        parse_targets = self._targets_from(graph, "triage__parse")
        tools_targets = self._targets_from(graph, "triage__tools")

        exits = {t for t in parse_targets if t.startswith("__handoff_exit")}
        assert exits, "precondition: the parse node must declare the mesh exit"

        assert exits <= tools_targets, (
            "a tool-triggered {node}__tools node emits Command(goto=exit_name) on "
            "HANDOFF_END and on on_exhaust='exit', so it must DECLARE that exit "
            "among its destinations -- its parse sibling already does. Missing: "
            f"{sorted(exits - tools_targets)} (tools declares {sorted(tools_targets)})"
        )


class TestMixedTriggerMeshRoundTripsPerMember:
    """neograph-dgbqv.8: a mesh mixing a tool-triggered agent member with a
    non-tool member must not convert the second member on re-import.

    ``Swarm.handoff`` is a MESH-level field, so the importer applied one trigger
    to every member; a member with no ReAct turn then had ``trigger='tool'``
    forced onto it, which ``_check_portal_mesh`` rejects -- except the importer
    also hard-coded ``mode='agent'``, so the two wrongs cancelled and produced a
    working-but-wrong round trip.

    The trigger is derived PER MEMBER from a structural signal every Agent Spec
    carries -- whether the member has tools at all -- so this holds for a FOREIGN
    Swarm with no neograph markers just as much as for one we exported. A member
    with no tools has no tool-call turn to trigger a handoff from, so it can only
    route by typed output regardless of the mesh's HandoffMode.
    """

    @staticmethod
    def _mixed_mesh():
        from neograph.tool import Tool

        def _fn(**kwargs):
            return "x"

        triage = Node(
            name="triage",
            mode="agent",
            model="router",
            prompt="test/triage",
            inputs={"handoff": Handoff},
            outputs=Handoff,
            tools=[Tool(name="lookup", description="d", fn=_fn)],
        ) | Portal(to=["scribe"], trigger="tool", max_hops=4)
        scribe = Node(
            name="scribe",
            mode="think",
            model="worker",
            prompt="test/scribe",
            inputs={"handoff": Handoff},
            outputs=Handoff,
        ) | Portal(to=["triage"])
        return Construct("mixed-trigger-mesh", nodes=[triage, scribe])

    def test_a_toolless_member_does_not_come_back_tool_triggered(self):
        from neograph._agent_spec import to_agent_spec
        from neograph.loader import from_agent_spec

        rebuilt = from_agent_spec(to_agent_spec(self._mixed_mesh()))
        triggers = {
            n.name: n.modifier_set.portal.trigger for n in rebuilt.nodes if n.modifier_set.portal is not None
        }
        modes = {n.name: n.mode for n in rebuilt.nodes if n.modifier_set.portal is not None}
        toolless = next(name for name in triggers if name.startswith("scribe"))
        assert modes[toolless] == "think", (
            "the think member must survive as a think member -- becoming an agent "
            "member changes what it DOES at runtime, not just how it is labelled "
            f"(got mode={modes[toolless]!r})"
        )
        assert triggers[toolless] == "output", (
            "a mesh member with no tools has no ReAct tool-call turn to hand off from, "
            "so the mesh-level HandoffMode must not force trigger='tool' onto it "
            f"(got {triggers[toolless]!r} for {toolless!r})"
        )

    def test_the_tool_triggered_member_keeps_its_trigger(self):
        """The contrast: deriving the trigger per member must not flatten the
        member that legitimately IS tool-triggered."""
        from neograph._agent_spec import to_agent_spec
        from neograph.loader import from_agent_spec

        rebuilt = from_agent_spec(to_agent_spec(self._mixed_mesh()))
        triggers = {
            n.name: n.modifier_set.portal.trigger for n in rebuilt.nodes if n.modifier_set.portal is not None
        }
        tooled = next(name for name in triggers if name.startswith("triage"))
        assert triggers[tooled] == "tool", f"the tool-bearing member must stay tool-triggered (got {triggers[tooled]!r})"
