"""TDD-red regression for neograph-mrb2y: durable Tier-2 hot-swap helper.

Cite docs/design/agent-spec-interop-2026-07-09.md s1a (motivating use case) and
docs/design/dynamic-handoff-research-2026-07-13.md (Tier-2 compose note).

``resume_from_agent_spec`` (+ its async twin ``aresume_from_agent_spec``) is a
THIN public helper that COMPOSES the three primitives that already exist:

    from_agent_spec(flow)  ->  Construct.__init__ (_validate_node_chain, the
                                type-channel gate that rejects a machine-authored
                                spec BEFORE any node runs)
    compile(construct, checkpointer=<same>)  ->  schema/node fingerprints
    run(graph, config=<same thread_id>, auto_resume=True)  ->  the EXISTING
                                _auto_resume_from_divergence rewind re-runs only
                                fingerprint-invalidated nodes, reusing state.

These tests are RED because the helper does NOT exist yet: every test resolves
``resume_from_agent_spec`` / ``aresume_from_agent_spec`` on its FIRST line, so an
absent-helper ``ImportError`` fails each test uniformly (a call-time failure, not
a collection error). Once neograph-mrb2y lands they exercise the real behavior.

Proofs pinned here (mrb2y deliverables + hhgnz.30 refinement addendum):
  (a) VALIDATION BEFORE EXECUTION -- a type-mismatched emitted Flow makes the
      helper's internal ``from_agent_spec`` -> ``Construct(...)`` gate raise the
      RAW ``ConstructError``/``ConfigurationError`` (NOT an ExecutionError
      wrapper, unlike the in-graph analog at factory.py:440-490) while a
      module-level sentinel proves NO node body ran.
  (b) SELECTIVE RE-RUN AFTER TOPOLOGY CHANGE -- v1 runs to a checkpoint; a v2
      whose ONE changed node has a new OUTPUT TYPE (so the schema fingerprint
      diverges and rewind fires) resumes via the helper on the same thread_id;
      per-node execution counters prove EXACTLY the changed node + its transitive
      downstream re-ran while untouched upstream did not, and the final result
      reflects the v2 schema.
  (3) FAIL LOUD ON MISSING DURABILITY -- ``checkpointer=None`` OR a missing
      thread_id raises ``ConfigurationError`` (durable resume is definitional;
      the checkpointer-less short-circuit at runner.py:289 is the silent seam
      this kills).
  (4) ASYNC PARITY -- proof (b) mirrored through ``aresume_from_agent_spec`` /
      the ``arun`` path, matching test_checkpoint_auto_rewind's sync+async style.

Run with::

    uv run --extra dev --extra agent-spec pytest tests/test_hot_swap_agent_spec.py
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

pytest.importorskip("pyagentspec")

from neograph import Construct, Node, arun, compile, run  # noqa: E402
from neograph._agent_spec import to_agent_spec  # noqa: E402
from neograph.errors import ConfigurationError, ConstructError  # noqa: E402
from neograph.loader import from_agent_spec  # noqa: E402
from neograph.modifiers import Each  # noqa: E402
from neograph.spec_types import model_to_agent_spec_properties  # noqa: E402
from tests.fakes import build_test_compile_kwargs, register_scripted  # noqa: E402

# ── absent-helper resolvers (first line of every test -> uniform RED) ─────────
# Resolving the public symbol here (rather than at module import) keeps the RED
# failure a call-time ImportError inside each test body, not a collection error.


def _resume_from_agent_spec():
    from neograph import resume_from_agent_spec  # noqa: PLC0415  (absent today)

    return resume_from_agent_spec


def _aresume_from_agent_spec():
    from neograph import aresume_from_agent_spec  # noqa: PLC0415  (absent today)

    return aresume_from_agent_spec


# ── shared schemas ───────────────────────────────────────────────────────────


class Tagged(BaseModel, frozen=True):
    label: str


class Bag(BaseModel, frozen=True):
    items: list[Tagged]


class Result(BaseModel, frozen=True):
    value: str


class MismatchedBag(BaseModel, frozen=True):
    # NO ``items: list[Tagged]`` -- its ``items`` is list[int], so an Each
    # consumer declaring ``Tagged`` per-item clashes: the reconstructed Construct
    # rejects it at assembly. This is the deliberate OUTPUT-TYPE mutation.
    items: list[int]


class Doc(BaseModel, frozen=True):
    text: str


class Enriched1(BaseModel, frozen=True):
    score: int


class Enriched2(BaseModel, frozen=True):
    # v2 output type: SAME node name + field set, ``score``'s TYPE widened
    # int -> float. That changes the schema fingerprint (so auto-rewind fires)
    # while staying forward-COERCIBLE, so the stored int checkpoint materializes
    # into the new float schema during the history walk -- the exact
    # test_checkpoint_auto_rewind._make_mid_model(int)->float shape. (An ADDED
    # field would be non-coercible here: the Agent-Spec round-trip drops field
    # defaults, making the reconstructed field REQUIRED -> the old checkpoint
    # can't materialize -> CheckpointSchemaError instead of a clean rewind.)
    score: float


class Report(BaseModel, frozen=True):
    body: str


# A module-level sentinel: any node body that runs appends its label. Proof (a)
# asserts this stays EMPTY -- the gate rejected the spec before execution.
_EXECUTED: list[str] = []


def _selective_constructs(counters: dict[str, int]) -> tuple[Construct, Construct]:
    """Build the (v1, v2) three-node scripted chain ``ingest -> enrich -> report``
    used by both selective-rerun proofs. ``enrich``'s OUTPUT TYPE is the only
    thing that changes between versions; every body increments a shared counter
    so re-execution is observable.
    """

    def ingest_fn(input_data, config):  # noqa: ANN001, ARG001
        counters["ingest"] += 1
        return Doc(text="hi")

    def enrich_v1_fn(input_data, config):  # noqa: ANN001, ARG001
        counters["enrich"] += 1
        return Enriched1(score=1)

    def enrich_v2_fn(input_data, config):  # noqa: ANN001, ARG001
        counters["enrich"] += 1
        return Enriched2(score=2.0)

    def report_fn(input_data, config):  # noqa: ANN001, ARG001
        counters["report"] += 1
        # from_agent_spec always reconstructs inputs as dict-form keyed by the
        # upstream node name (see test_agent_spec_roundtrip.refine_fn).
        return Report(body=f"score={input_data['enrich'].score}")

    register_scripted("hs_ingest", ingest_fn)
    register_scripted("hs_enrich_v1", enrich_v1_fn)
    register_scripted("hs_enrich_v2", enrich_v2_fn)
    register_scripted("hs_report", report_fn)

    def build(enrich_fn_name: str, enrich_out: type) -> Construct:
        ingest = Node.scripted("ingest", fn="hs_ingest", outputs=Doc)
        enrich = Node.scripted("enrich", fn=enrich_fn_name, inputs=Doc, outputs=enrich_out)
        report = Node.scripted("report", fn="hs_report", inputs=enrich_out, outputs=Report)
        return Construct("hotswap-selective", nodes=[ingest, enrich, report])

    return build("hs_enrich_v1", Enriched1), build("hs_enrich_v2", Enriched2)


# ── Proof (a): validation before execution ───────────────────────────────────


class TestHotSwapValidatesBeforeExecuting:
    """Deliverable (a): a machine-authored (mutated) Flow whose emitted output
    type clashes with a downstream consumer is rejected by the helper's internal
    ``from_agent_spec`` -> ``Construct(...)`` gate BEFORE any node runs -- and the
    gate error surfaces RAW (ConstructError/ConfigurationError), never wrapped in
    an ExecutionError (contrast the in-graph sibling at factory.py:440-490)."""

    def test_type_mismatched_flow_rejected_and_no_node_runs(self):
        resume_from_agent_spec = _resume_from_agent_spec()  # RED: absent today
        _EXECUTED.clear()

        def seed_fn(input_data, config):  # noqa: ANN001, ARG001
            _EXECUTED.append("seed")
            return Bag(items=[Tagged(label="a"), Tagged(label="b")])

        register_scripted("hs_bad_seed", seed_fn)

        def each_fn(input_data, config):  # noqa: ANN001, ARG001
            _EXECUTED.append("each")
            return Result(value=f"tagged-{input_data.label}")

        register_scripted("hs_bad_each", each_fn)

        seed = Node.scripted("seed", fn="hs_bad_seed", outputs=Bag)
        each_node = Node.scripted(
            "each_step", fn="hs_bad_each", inputs=Tagged, outputs=Result
        ) | Each(over="seed.items", key="label")
        pipeline = Construct("hotswap-bad", nodes=[seed, each_node])

        # A VALID pipeline exports cleanly; then mutate the seed's OUTPUT
        # properties (via the public exporter, not a hand-authored dict) so the
        # collection element type no longer matches the Each consumer's declared
        # ``Tagged`` per-item input -- a genuine emitted-output-type clash.
        flow = to_agent_spec(pipeline)
        bad_props = model_to_agent_spec_properties(MismatchedBag)
        idx = next(i for i, n in enumerate(flow.nodes) if getattr(n, "name", None) == "seed")
        flow.nodes[idx] = flow.nodes[idx].model_copy(update={"outputs": bad_props})

        # Durable inputs are all VALID here, so the ONLY thing that can fail is
        # the type-channel gate inside from_agent_spec -> Construct(...).
        with pytest.raises((ConstructError, ConfigurationError)):
            resume_from_agent_spec(
                flow,
                checkpointer=MemorySaver(),
                config={"configurable": {"thread_id": "hs-bad"}},
                **build_test_compile_kwargs(),
            )

        # The gate fired BEFORE compile/run: no node body executed.
        assert _EXECUTED == []


# ── Proof (b): selective re-run after topology change ────────────────────────


class TestHotSwapSelectiveReRun:
    """Deliverable (b): resume after a topology change re-runs ONLY the changed
    node + its transitive downstream, reusing checkpointed upstream state -- the
    helper COMPOSES the existing auto-rewind, it does not rebuild it."""

    def test_only_changed_node_and_downstream_reexecute(self):
        resume_from_agent_spec = _resume_from_agent_spec()  # RED: absent today

        counters = {"ingest": 0, "enrich": 0, "report": 0}
        c_v1, c_v2 = _selective_constructs(counters)

        saver = MemorySaver()
        thread = {"configurable": {"thread_id": "hs-selective-sync"}}

        # v1 established via the SAME round-trip the helper uses, so unchanged
        # nodes reconstruct to identical synthesized types across versions (only
        # ``enrich``'s type differs) -- otherwise every node's fingerprint would
        # drift and the whole graph would invalidate.
        graph_v1 = compile(
            from_agent_spec(to_agent_spec(c_v1)), checkpointer=saver, **build_test_compile_kwargs()
        )
        first = run(graph_v1, input={"node_id": "hs"}, config=thread)
        assert counters == {"ingest": 1, "enrich": 1, "report": 1}
        assert "score=1" in first["report"].body  # v1: int

        # Hot-swap to v2 via the helper: emit -> validate -> recompile -> resume.
        second = resume_from_agent_spec(
            to_agent_spec(c_v2),
            checkpointer=saver,
            config=thread,
            **build_test_compile_kwargs(),
        )

        # Changed node + its transitive downstream re-ran ...
        assert counters["enrich"] == 2, "changed node must re-execute on hot-swap rewind"
        assert counters["report"] == 2, "downstream of the changed node must re-execute"
        # ... untouched upstream did NOT (no over-rewind).
        assert counters["ingest"] == 1, "upstream of the change must NOT re-execute"
        # ... and the result reflects the NEW v2 schema, not a stale tip.
        assert "score=2.0" in second["report"].body


# ── Proof (3): fail loud on missing durability ───────────────────────────────


class TestHotSwapFailsLoudWithoutDurability:
    """Refinement MEDIUM-1: durable resume is DEFINITIONAL. ``run(..., auto_resume=True)``
    with no checkpointer/thread_id silently full-re-runs with zero reuse
    (runner.py:289 short-circuit) -- the exact silent seam Tier-2 must not have.
    The helper MUST raise ``ConfigurationError`` instead."""

    def _valid_flow(self):
        def only_fn(input_data, config):  # noqa: ANN001, ARG001
            _EXECUTED.append("only")
            return Result(value="x")

        register_scripted("hs_dur_only", only_fn)
        return to_agent_spec(Construct("hotswap-dur", nodes=[Node.scripted("only", fn="hs_dur_only", outputs=Result)]))

    def test_none_checkpointer_raises(self):
        resume_from_agent_spec = _resume_from_agent_spec()  # RED: absent today
        _EXECUTED.clear()
        flow = self._valid_flow()

        with pytest.raises(ConfigurationError):
            resume_from_agent_spec(
                flow,
                checkpointer=None,
                config={"configurable": {"thread_id": "hs-dur"}},
                **build_test_compile_kwargs(),
            )
        assert _EXECUTED == []

    def test_missing_thread_id_raises(self):
        resume_from_agent_spec = _resume_from_agent_spec()  # RED: absent today
        _EXECUTED.clear()
        flow = self._valid_flow()

        with pytest.raises(ConfigurationError):
            resume_from_agent_spec(
                flow,
                checkpointer=MemorySaver(),
                config={"configurable": {}},
                **build_test_compile_kwargs(),
            )
        assert _EXECUTED == []


# ── Proof (4): async parity (mirror of proof b) ──────────────────────────────


class TestHotSwapSelectiveReRunAsync:
    """Refinement LOW: the ``aresume_from_agent_spec`` / ``arun`` twin inherits
    the identical selective-rerun contract -- mirroring
    test_checkpoint_auto_rewind's sync+async style."""

    async def test_only_changed_node_and_downstream_reexecute_async(self):
        aresume_from_agent_spec = _aresume_from_agent_spec()  # RED: absent today

        counters = {"ingest": 0, "enrich": 0, "report": 0}
        c_v1, c_v2 = _selective_constructs(counters)

        saver = MemorySaver()
        thread = {"configurable": {"thread_id": "hs-selective-async"}}

        graph_v1 = compile(
            from_agent_spec(to_agent_spec(c_v1)), checkpointer=saver, **build_test_compile_kwargs()
        )
        first = await arun(graph_v1, input={"node_id": "hs"}, config=thread)
        assert counters == {"ingest": 1, "enrich": 1, "report": 1}
        assert "score=1" in first["report"].body

        second = await aresume_from_agent_spec(
            to_agent_spec(c_v2),
            checkpointer=saver,
            config=thread,
            **build_test_compile_kwargs(),
        )

        assert counters["enrich"] == 2, "changed node must re-execute on async hot-swap rewind"
        assert counters["report"] == 2, "downstream of the changed node must re-execute"
        assert counters["ingest"] == 1, "upstream of the change must NOT re-execute"
        assert "score=2.0" in second["report"].body
