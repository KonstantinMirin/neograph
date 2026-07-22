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
from neograph.modifiers import Each, Loop, Oracle  # noqa: E402
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

