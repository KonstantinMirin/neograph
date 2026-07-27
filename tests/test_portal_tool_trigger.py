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
