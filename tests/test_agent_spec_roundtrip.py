"""Round-trip fidelity tests for Agent Spec interop (neograph-csdcl).

Cite docs/design/agent-spec-interop-2026-07-09.md s6/s6a. Proves the fidelity
claims that i3zsh/i3zsh.1/01i0g/0la8v's own unit-level tests don't already
cover end to end:

  (a) A golden/foreign Agent Spec YAML (no neograph markers) imports and RUNS.
  (b) Each/Oracle/Loop modifiers survive export -> import -> compile -> RUN
      with behavior preserved (not just structural marker presence).
  (c) Callable-valued-field export errors -- already covered by
      tests/test_agent_spec_export.py::TestToAgentSpecRejectsUnrepresentableFields
      and ::TestToAgentSpecLowersModifiers::test_callable_loop_when_is_rejected;
      not duplicated here.
  (d) should_pass/should_fail check_fixtures -- NOT added here. The
      tests/check_fixtures/ harness (test_check_fixtures.py) has no mechanism
      to skip a fixture requiring an optional dependency ([agent-spec] extra),
      unlike the MCP examples (gated by a separate `--extra mcp-examples` test
      command). Adding an Agent-Spec-dependent fixture there would break the
      bare `uv run pytest` suite for any environment without the extra
      installed -- violating 'core stays dependency-light'. This file (gated
      by module-level pytest.importorskip, the SAME safe pattern
      test_agent_spec_export.py/test_agent_spec_import.py already use) is the
      equivalent rustc-style should_pass/should_fail proof, just not routed
      through that specific harness.
  (e) Three-surface parity -- N/A, this is IR-level free-function testing
      (to_agent_spec/from_agent_spec), not @node-layer, matching i3zsh's own
      exemption reasoning.

Round-trip-marker items (doc s6a):
  1. Oracle/Each/Loop reconstruct losslessly via markers -- proven here via
     actual compile+run, not just structural presence (test_agent_spec_import.py
     already pins Oracle's structural reconstruction; this file adds Each/Loop
     and end-to-end behavior for all three).
  2. A foreign Agent Spec (no markers) imports as primitives -- proven by
     stripping neograph/* metadata from an exported Flow and re-importing.
  3. A hand-edited Flow with a stale marker does NOT silently reconstruct --
     proven by corrupting an Oracle group's variant count and asserting
     fallback-to-primitive (with a warning), not a wrong reconstruction.
  4. Exported markers live only in metadata; a foreign runtime ignoring them
     still runs the primitives correctly -- implied by (2): stripping the
     markers still yields a runnable, behaviorally-correct primitive import.

Run with::

    uv run --extra dev --extra agent-spec pytest tests/test_agent_spec_roundtrip.py
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import BaseModel

pytest.importorskip("pyagentspec")

from neograph import Construct, Node, compile, run  # noqa: E402
from neograph._agent_spec import to_agent_spec  # noqa: E402
from neograph.loader import from_agent_spec  # noqa: E402
from neograph.modifiers import Each, Loop, ModifierCombo, Oracle, classify_modifiers  # noqa: E402
from neograph.tool import Tool  # noqa: E402
from tests.fakes import build_fake_llm_kwargs, build_test_compile_kwargs, register_scripted  # noqa: E402


class Bag(BaseModel, frozen=True):
    items: list[Tagged]


class Tagged(BaseModel, frozen=True):
    label: str


class Result(BaseModel, frozen=True):
    value: str


class Claims(BaseModel, frozen=True):
    items: list[str]


class Draft(BaseModel, frozen=True):
    content: str
    iteration: int
    score: float


class TestEachOracleLoopRoundTripPreservesBehavior:
    """Item (b): each modifier survives export -> import -> compile -> run
    with the SAME behavior as the original, non-round-tripped Construct."""

    def test_each_round_trips_and_runs(self):
        def seed_fn(input_data, config):
            return Bag(items=[Tagged(label="a"), Tagged(label="b")])

        register_scripted("rt_each_seed", seed_fn)

        def each_fn(input_data, config):
            return Result(value=f"tagged-{input_data.label}")

        register_scripted("rt_each_step", each_fn)

        seed = Node.scripted("seed", fn="rt_each_seed", outputs=Bag)
        each_node = Node.scripted("each_step", fn="rt_each_step", inputs=Tagged, outputs=Result) | Each(
            over="seed.items", key="label"
        )
        pipeline = Construct("each-roundtrip", nodes=[seed, each_node])

        flow = to_agent_spec(pipeline)
        imported = from_agent_spec(flow)

        graph = compile(imported, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "each-rt"})

        collected = result["each_step"]
        assert set(collected) == {"a", "b"}, collected
        assert collected["a"].value == "tagged-a"
        assert collected["b"].value == "tagged-b"

    def test_loop_round_trips_and_runs(self):
        call_count = [0]

        def seed_fn(input_data, config):
            return Draft(content="v0", iteration=0, score=0.0)

        register_scripted("rt_loop_seed", seed_fn)

        def refine_fn(input_data, config):
            # from_agent_spec always reconstructs inputs as dict-form
            # ({upstream_name: type}), even for an originally-single-type
            # input -- the upstream/reentry value lives under the "seed" key.
            call_count[0] += 1
            prev = input_data["seed"]
            return Draft(content=f"v{call_count[0]}", iteration=prev.iteration + 1, score=prev.score + 0.3)

        register_scripted("rt_loop_refine", refine_fn)

        seed = Node.scripted("seed", fn="rt_loop_seed", outputs=Draft)
        refine = Node.scripted("refine", fn="rt_loop_refine", inputs=Draft, outputs=Draft) | Loop(
            when="score < 0.8", max_iterations=10
        )
        pipeline = Construct("loop-roundtrip", nodes=[seed, refine])

        flow = to_agent_spec(pipeline)
        imported = from_agent_spec(flow)

        graph = compile(imported, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "loop-rt"})

        assert call_count[0] == 3
        assert isinstance(result["refine"], list)
        assert result["refine"][-1].score >= 0.8
        assert result["refine"][-1].iteration == 3

    def test_oracle_round_trips_and_runs(self):
        # Oracle round-trips only for "think" mode today: _lower_oracle
        # unconditionally lowers variants to LlmNode (and from_agent_spec's
        # reconstruction unconditionally sets mode="think") regardless of
        # the ORIGINAL node's mode -- a scripted-mode Oracle node round-trips
        # into a broken think-mode reconstruction. Filed as a follow-up
        # (neograph-aa5gq note); this test exercises the currently-correct
        # think-mode path with a FakeLLM, matching what
        # test_agent_spec_export.py's own Oracle test already covers on the
        # export side.
        from neograph.testing.fakes import StructuredFake

        def merge_fn(variants, config):
            # The reconstructed output type is a freshly-synthesized class
            # (Property lists carry no back-reference to the original
            # "Claims" name) -- a real merge_fn operates on whatever type it
            # actually receives, not a hardcoded original class.
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return type(variants[0])(items=all_items)

        register_scripted("rt_oracle_merge", merge_fn)

        node = Node(name="ensemble", mode="think", model="fast", outputs=Claims, prompt="rw/ensemble") | Oracle(
            n=3, merge_fn="rt_oracle_merge"
        )
        pipeline = Construct("oracle-roundtrip", nodes=[node])

        flow = to_agent_spec(pipeline)
        imported = from_agent_spec(flow)

        fake_llm = StructuredFake(lambda m: m(items=["variant"]))
        graph = compile(
            imported, **build_test_compile_kwargs(), **build_fake_llm_kwargs(lambda tier: fake_llm)
        )
        result = run(graph, input={"node_id": "oracle-rt"})

        merged = result["ensemble"]
        assert type(merged).__name__.startswith("AgentSpecType_")
        assert len(merged.items) == 3


class TestForeignAgentSpecImportsAsPrimitives:
    """Item (a) + round-trip-marker item (2)+(4): a Flow with no neograph/*
    metadata (simulating a foreign/third-party Agent Spec, or a 'golden doc'
    authored by hand/another tool) imports as plain primitives and still
    runs correctly -- markers are purely additive, never required for
    correctness. A hand-authored low-level pyagentspec dict was deliberately
    NOT used for this (too fragile/version-coupled to author correctly by
    hand across every required Component field); stripping a REAL exported
    Flow's markers is an equally valid, far more robust proxy for 'a foreign
    spec with no neograph provenance'."""

    def test_markerless_flow_imports_and_runs(self):
        def seed_fn(input_data, config):
            return Claims(items=["hello"])

        register_scripted("rt_foreign_seed", seed_fn)

        def consume_fn(input_data, config):
            # dict-form reconstruction -- see refine_fn's comment above.
            return Result(value=f"got: {input_data['seed'].items[0]}")

        register_scripted("rt_foreign_consume", consume_fn)

        seed = Node.scripted("seed", fn="rt_foreign_seed", outputs=Claims)
        consumer = Node.scripted("consumer", fn="rt_foreign_consume", inputs=Claims, outputs=Result)
        pipeline = Construct("foreign-flow", nodes=[seed, consumer])

        flow = to_agent_spec(pipeline)
        # Strip every neograph/* marker to simulate a truly foreign spec --
        # from_agent_spec must not depend on them for a plain primitive import.
        for node in flow.nodes:
            if node.metadata:
                for key in list(node.metadata):
                    if key.startswith("neograph/"):
                        del node.metadata[key]

        imported = from_agent_spec(flow)
        graph = compile(imported, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "foreign-rt"})

        assert result["consumer"].value == "got: hello"


class TestDictFormFanInRoundTrip:
    """Pins ozxqw's acceptance criterion: a real two-node @node-shaped
    pipeline using the PRIMARY dict-form fan-in (``inputs={'seed': Claims}``,
    the shape @node's decoration always produces for a typed upstream param)
    must export -> import -> compile -> RUN with behavior preserved -- not
    just structurally export without raising (that narrower claim is already
    pinned in test_agent_spec_export.py::TestToAgentSpecExportsDictFormFanIn).
    """

    def test_dict_form_fan_in_round_trips_and_runs(self):
        def seed_fn(input_data, config):
            return Claims(items=["hello", "world"])

        register_scripted("rt_dictfanin_seed", seed_fn)

        def consume_fn(input_data, config):
            return Result(value=f"got: {input_data['seed'].items[0]}")

        register_scripted("rt_dictfanin_consume", consume_fn)

        seed = Node.scripted("seed", fn="rt_dictfanin_seed", outputs=Claims)
        consumer = Node.scripted(
            "consumer", fn="rt_dictfanin_consume", inputs={"seed": Claims}, outputs=Result
        )
        pipeline = Construct("dict-fanin-flow", nodes=[seed, consumer])

        flow = to_agent_spec(pipeline)
        imported = from_agent_spec(flow)
        graph = compile(imported, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "dict-fanin-rt"})

        assert result["consumer"].value == "got: hello"


class TestStaleMarkerDoesNotSilentlyReconstruct:
    """Round-trip-marker item (3): a hand-edited Flow whose Oracle marker no
    longer matches the actual primitives around it (variant count mismatch)
    must NOT be silently reconstructed into a wrong Oracle -- fall back to
    primitive-level import for that group, with a warning."""

    def test_oracle_group_with_stale_variant_count_falls_back_to_primitives(self):
        from neograph.modifiers import ModifierCombo, classify_modifiers

        node = Node(name="ensemble", mode="think", model="fast", outputs=Claims, prompt="rw/ensemble")
        node = node | Oracle(n=3, merge_fn="combine")
        pipeline = Construct("stale-oracle", nodes=[node])

        flow = to_agent_spec(pipeline)

        # Corrupt the marker: REMOVE one variant node entirely from the Flow
        # (not just its metadata -- deleting group_id would KeyError inside
        # the grouping walk, which reads THIS node's own group_id first). Now
        # only 2 of the 3 declared variants are actually present, while the
        # merge node's neograph/oracle_spec still claims n=3.
        variant_nodes = [
            n for n in flow.nodes if n.metadata and n.metadata.get("neograph/modifier") == "oracle" and "__variant_" in n.name
        ]
        assert len(variant_nodes) == 3
        flow.nodes.remove(variant_nodes[0])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            imported = from_agent_spec(flow)

        assert any("stale" in str(w.message) for w in caught), [str(w.message) for w in caught]

        # Falls back to bare primitives -- no single "ensemble" Oracle-combo
        # item; the group's nodes import individually instead.
        # Falls back to bare primitives -- no item is classified as an
        # Oracle combo (one of the fallback bare nodes happens to still be
        # NAMED "ensemble", since that was the merge node's own name -- name
        # alone isn't the right check; the modifier classification is).
        for item in imported.nodes:
            combo, _ = classify_modifiers(item)
            assert combo == ModifierCombo.BARE, (
                f"expected every fallback item to be BARE (stale marker must not reconstruct "
                f"an Oracle), got {combo.name} for {getattr(item, 'name', item)!r}"
            )


# ── neograph-aa5gq gap 1: AgentNode -> agent/act Node LOSSLESS round trip ────


class TestAgentNodeRoundTripLosslessness:
    """Gap 1 (neograph-aa5gq): a neograph agent/act Node exported via
    ``to_agent_spec`` (a real ``AgentNode``+``Agent``+``ServerTool`` composite
    carrying the ``neograph/agent_spec`` marker, neograph-i3zsh.1) must import
    back via ``from_agent_spec`` as the EXACT same agent/act Node -- mode,
    prompt, model, tools (including each Tool's budget/config/idempotent),
    gate_tools_when, and context all preserved.

    Today ``from_agent_spec`` FAILS LOUD (``ConfigurationError``) on any
    ``AgentNode`` (loader.py::_reconstruct_primitive_node) -- so every
    losslessness assertion below currently errors at the ``from_agent_spec``
    call, confirmed by running pytest (TDD red). Once aa5gq inverts the
    fail-loud into the marker-driven inversion, these pass.
    """

    def _exported_agent_flow(self, mode):
        # Mirror test_agent_spec_export.py's agent-marker shape exactly: an
        # upstream context producer + an agent/act node with a fully-populated
        # tool (budget/config/idempotent), a string gate, and a context list.
        notes = Node.scripted("notes", fn="rt_agent_notes", outputs=Claims)
        agent = Node(
            name="explore",
            mode=mode,
            model="research",
            prompt="explore the codebase",
            outputs=Result,
            tools=[Tool("search_code", budget=5, idempotent=True, config={"depth": 2})],
            gate_tools_when="always",
            context=["notes"],
        )
        pipeline = Construct("agent-pipeline", nodes=[notes, agent])
        return to_agent_spec(pipeline)

    @pytest.mark.parametrize("mode", ["agent", "act"])
    def test_agent_node_reconstructs_exact_node_from_marker(self, mode):
        flow = self._exported_agent_flow(mode)
        imported = from_agent_spec(flow)

        assert isinstance(imported, Construct)
        reconstructed = next(n for n in imported.nodes if getattr(n, "name", None) == "explore")

        # Not silently downgraded to a scripted/think ToolNode stand-in --
        # the ReAct tool-loop mode survives.
        assert reconstructed.mode == mode
        assert reconstructed.prompt == "explore the codebase"
        assert reconstructed.model == "research"
        assert reconstructed.gate_tools_when == "always"
        assert reconstructed.context == ["notes"]

        # Tool budget/config/idempotent survive via the flat marker tools list
        # (l7gvy deferred slice -- the tool_spec marker survival is pinned
        # separately below).
        assert reconstructed.tools is not None and len(reconstructed.tools) == 1
        tool = reconstructed.tools[0]
        assert isinstance(tool, Tool)
        assert tool.name == "search_code"
        assert tool.budget == 5
        assert tool.idempotent is True
        assert tool.config == {"depth": 2}

    def test_tool_spec_marker_survives_to_dict_from_dict(self):
        """l7gvy's deferred import+round-trip slice: each ServerTool attached
        to the exported Agent carries an independent ``neograph/tool_spec``
        marker (budget/config/idempotent) that must survive a full
        ``to_dict -> from_dict`` serialization cycle -- the metadata-survival
        concern (ratification §6.1) that gates lossless reconstruction.

        FAILS NOW: the assertion that the reconstructed neograph ``Tool``
        carries the round-tripped budget/config/idempotent cannot run because
        ``from_agent_spec`` rejects the AgentNode first.
        """
        from pyagentspec.flows.flow import Flow

        flow = self._exported_agent_flow("agent")

        # Full serialization cycle -- proves the marker rides the wire format,
        # not just the in-memory object graph.
        rebuilt = Flow.from_dict(flow.to_dict())

        agent_node = next(n for n in rebuilt.nodes if type(n).__name__ == "AgentNode")
        server_tool = next(t for t in agent_node.agent.tools if t.name == "search_code")
        tool_spec = server_tool.metadata["neograph/tool_spec"]
        assert tool_spec["budget"] == 5
        assert tool_spec["idempotent"] is True
        assert tool_spec["config"] == {"depth": 2}

        # And the reconstruction reads it back into a real neograph Tool.
        imported = from_agent_spec(rebuilt)
        reconstructed = next(n for n in imported.nodes if getattr(n, "name", None) == "explore")
        tool = reconstructed.tools[0]
        assert (tool.budget, tool.idempotent, tool.config) == (5, True, {"depth": 2})


# ── neograph-aa5gq gap 2: foreign Swarm -> native Portal mesh + warning ──────


class TestSwarmImportsOntoPortalMesh:
    """Gap 2 (neograph-aa5gq): a FOREIGN pyagentspec ``Swarm`` (whose ``Agent``
    members carry NO neograph marker) imports via ``from_agent_spec`` onto a
    native Portal mesh -- sibling agent-mode member Nodes with
    ``inputs={'handoff': Payload}`` piped through ``Portal(to=[peers])`` -- and,
    per the Core Invariant's no-silent-downgrade arm (refinement MEDIUM-3),
    emits a ``warnings.warn`` documenting the route-only/name-bound downgrade.

    FAILS NOW: ``from_agent_spec`` assumes a ``Flow`` (``.nodes``) and has no
    top-level ``Swarm`` branch, so the call errors before any mesh is built.
    """

    def _foreign_swarm(self):
        from pyagentspec.agent import Agent
        from pyagentspec.llms.llmconfig import LlmConfig
        from pyagentspec.swarm import Swarm

        def mk(name):
            return Agent(
                name=name,
                llm_config=LlmConfig(name=f"{name}-llm", model_id="fast"),
                system_prompt=f"you are {name}",
                tools=[],
            )

        triage, billing, technical = mk("triage"), mk("billing"), mk("technical")
        # Single connected mesh reachable from first_agent=triage; billing can
        # hand BACK to triage (a genuine cycle), technical is terminal.
        return Swarm(
            name="support-swarm",
            first_agent=triage,
            relationships=[(triage, billing), (triage, technical), (billing, triage)],
        )

    def test_foreign_swarm_imports_onto_portal_mesh_with_warning(self):
        swarm = self._foreign_swarm()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            imported = from_agent_spec(swarm)

        assert isinstance(imported, Construct)
        members = imported.nodes
        assert len(members) == 3

        # first_agent must be the entry (nodes[0]) so max_hops/on_exhaust ride
        # the entry Portal.
        assert getattr(members[0], "name", None) == "triage"

        by_name = {getattr(m, "name", None): m for m in members}
        for m in members:
            combo, mods = classify_modifiers(m)
            assert combo == ModifierCombo.PORTAL, (
                f"expected every Swarm member to be a Portal-modified mesh node, "
                f"got {combo.name} for {getattr(m, 'name', m)!r}"
            )

        # Directed relationships map to per-node Portal.to successors.
        _, triage_mods = classify_modifiers(by_name["triage"])
        assert set(triage_mods["portal"].to) == {"billing", "technical"}
        _, billing_mods = classify_modifiers(by_name["billing"])
        assert set(billing_mods["portal"].to) == {"triage"}

        # SINGLE shared payload instance (identity) reused as every member's
        # outputs AND inputs['handoff'] -- _check_portal_mesh checks by `is`.
        payload = members[0].outputs
        assert "goto" in payload.model_fields, "synthesized payload needs a 'goto' route field"
        for m in members:
            assert m.outputs is payload, "all members must share ONE payload model instance"
            assert isinstance(m.inputs, dict) and m.inputs.get("handoff") is payload

        # No-silent-downgrade: a warning documents the route-only/name-bound
        # best-effort nature of the Swarm import.
        messages = [str(w.message).lower() for w in caught]
        assert any(
            "swarm" in msg or "downgrade" in msg or "name-bound" in msg or "route" in msg
            for msg in messages
        ), f"expected a Swarm-downgrade warning, got {messages!r}"

    def test_imported_swarm_mesh_compiles_with_fake_llm(self):
        """The reconstructed mesh must be a real, compilable Portal mesh --
        it goes through the SAME _check_portal_mesh assembly validation a
        hand-written mesh gets (Core Invariant), and its agent-mode members
        bind to a live LLM at compile. Compiling with a FakeLLM proves the
        structure is valid without needing network."""
        from neograph.testing.fakes import StructuredFake

        swarm = self._foreign_swarm()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imported = from_agent_spec(swarm)

        fake_llm = StructuredFake(lambda m: m())
        graph = compile(imported, **build_fake_llm_kwargs(lambda tier: fake_llm))
        # Stronger than a bare `is not None`: every reconstructed Swarm member
        # is an agent-mode Portal member, so each wires into the compiled mesh
        # as its ReAct-cycle entry node `{member}__agent` (the mesh member's
        # real LangGraph entry). compile() also raises on any invalid mesh, so
        # reaching here already proves _check_portal_mesh passed.
        compiled_nodes = set(graph.get_graph().nodes)
        for member in ("triage", "billing", "technical"):
            assert f"{member}__agent" in compiled_nodes, (
                f"Swarm member {member!r} was not wired into the compiled mesh; "
                f"compiled nodes: {sorted(compiled_nodes)}"
            )


# ── neograph-aa5gq gap 3: RemoteAgent/A2AAgent best-effort import + warning ──


class TestRemoteAgentBestEffortImport:
    """Gap 3 (neograph-aa5gq, ratification §3b): an ``AgentNode`` whose
    ``.agent`` is a client-initiated ``RemoteAgent``/``A2AAgent`` (carrying NO
    neograph marker) imports best-effort to a name-bound scripted Node WITH a
    ``warnings.warn`` -- never a silent drop, never a fail-loud (fail-loud is
    reserved for orchestrator-side ServerTool-as-agent).

    FAILS NOW: the AgentNode branch fails loud (``ConfigurationError``) for
    ALL agents regardless of type.
    """

    def _flow_with_a2a_agent(self):
        from pyagentspec.a2aagent import A2AAgent, A2AConnectionConfig
        from pyagentspec.flows.flow import Flow

        # Build a valid agent-node Flow via the real exporter, then swap the
        # AgentNode's Agent for a foreign A2AAgent and strip the neograph
        # marker -- a robust proxy for a foreign spec authored by another tool.
        agent = Node(name="remote_helper", mode="agent", model="fast", prompt="help", outputs=Result)
        flow = to_agent_spec(Construct("remote-pipeline", nodes=[agent]))

        # pyagentspec 26.1.2 enforces AgentNode.outputs == agent.outputs at
        # Flow.from_dict (AgentNode._get_inferred_outputs = agent.outputs), so
        # the swapped-in A2AAgent must carry the SAME outputs the exported
        # AgentNode declares -- otherwise the fixture builds an inconsistent
        # Flow that fails deserialization before from_agent_spec is ever
        # reached. This is fixture consistency only; the behavioral assertions
        # below (mode/scripted_fn/warning) are unchanged.
        agent_node = next(n for n in flow.nodes if type(n).__name__ == "AgentNode")
        remote = A2AAgent(
            name="remote_helper",
            agent_url="http://svc.example/agent",
            connection_config=A2AConnectionConfig(name="conn"),
            outputs=agent_node.outputs,
        )
        new_nodes = [
            n.model_copy(update={"agent": remote, "metadata": {}})
            if type(n).__name__ == "AgentNode"
            else n
            for n in flow.nodes
        ]
        return Flow.from_dict(flow.model_copy(update={"nodes": new_nodes}).to_dict())

    def test_a2a_agent_imports_to_name_bound_scripted_node_with_warning(self):
        flow = self._flow_with_a2a_agent()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            imported = from_agent_spec(flow)

        assert isinstance(imported, Construct)
        member = next(n for n in imported.nodes if getattr(n, "name", None) == "remote_helper")

        # Name-bound scripted stand-in -- the runtime binds the endpoint at
        # compile time (same best-effort contract as any ToolNode import).
        assert isinstance(member, Node)
        assert member.mode == "scripted"
        assert member.scripted_fn == "remote_helper"

        messages = [str(w.message).lower() for w in caught]
        assert any(
            "remote" in msg or "best-effort" in msg or "best effort" in msg or "name-bound" in msg
            for msg in messages
        ), f"expected a best-effort RemoteAgent warning, got {messages!r}"

