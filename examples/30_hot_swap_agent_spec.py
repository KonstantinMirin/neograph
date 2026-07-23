"""Example 30: Durable Tier-2 hot-swap via Agent Spec (emit -> validate ->
recompile -> resume).

Cite docs/design/agent-spec-interop-2026-07-09.md §1a (motivating use case) and
docs/design/dynamic-handoff-research-2026-07-13.md (Tier-2 compose note).

Scenario: a running 3-node scripted pipeline (ingest -> enrich -> report) is
re-emitted as an Open Agent Spec ``Flow`` at runtime, then a NEW topology (the
``enrich`` node's output field ``score`` widened int -> float) is hot-swapped in
and resumed durably on the SAME thread_id. Because the changed node's OUTPUT
TYPE differs, the schema fingerprint diverges and neograph's existing
checkpoint auto-rewind re-executes ONLY the changed node + its downstream
(``enrich`` + ``report``), reusing the checkpointed ``ingest`` result.

This is Example 19's checkpoint auto-resume, but the v2 pipeline arrives via the
Agent Spec ROUND-TRIP (``to_agent_spec`` -> ``resume_from_agent_spec`` ->
``from_agent_spec``) instead of being hand-written — the machine-authored-graph
payoff. ``resume_from_agent_spec`` is a THIN compose over
``from_agent_spec`` (validates before execution) -> ``compile`` (same
checkpointer) -> ``run`` (same thread_id, auto_resume). See its in-graph analog
at factory.py:440-490 (``make_portal_dispatch_fn._prepare``).

Boundary notes:
  - The topology change must change a node OUTPUT TYPE (or field types) for the
    fingerprint to diverge and rewind to fire. A same-type re-route would be
    Tier-1 (data-driven routing in one compiled superset graph), not Tier-2.
  - SCOPE: think / scripted (+ Oracle/Each/Loop/Operator) pipelines only.
    ``from_agent_spec`` fails loud on an ``AgentNode`` (agent/act mode), so a
    hot-swap of an agent/act mesh is a separate concern, out of scope here.
  - int -> float widening (NOT an added field) is used because the Agent Spec
    round-trip drops field defaults: an added field reconstructs as REQUIRED, so
    the old checkpoint could not materialize into it. Widening stays coercible.

Keyless: all-scripted nodes, no LLM, no API keys. Requires the [agent-spec]
extra for the Agent Spec round-trip.

Run:
    uv run --extra agent-spec python examples/30_hot_swap_agent_spec.py
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from neograph import (
    Construct,
    Node,
    compile,
    resume_from_agent_spec,
    run,
    to_agent_spec,
)
from neograph.loader import from_agent_spec

# -- Schemas ------------------------------------------------------------------


class Doc(BaseModel, frozen=True):
    text: str


class EnrichedV1(BaseModel, frozen=True):
    score: int


class EnrichedV2(BaseModel, frozen=True):
    # v2: SAME node + field set, ``score``'s TYPE widened int -> float. This
    # shifts the schema fingerprint (so auto-rewind fires) while staying
    # forward-COERCIBLE, so the stored int checkpoint materializes into the new
    # float schema during the history walk.
    score: float


class Report(BaseModel, frozen=True):
    body: str


# -- Scripted node bodies (keyless) -------------------------------------------
# Each prints when it runs so the selective re-execution is visible.


def ingest_fn(input_data, config):  # noqa: ANN001, ARG001
    print("  [ingest] running")
    return Doc(text="hi")


def enrich_v1_fn(input_data, config):  # noqa: ANN001, ARG001
    print("  [enrich] running (v1 -- int)")
    return EnrichedV1(score=1)


def enrich_v2_fn(input_data, config):  # noqa: ANN001, ARG001
    print("  [enrich] running (v2 -- float)")
    return EnrichedV2(score=2.0)


def report_fn(input_data, config):  # noqa: ANN001, ARG001
    print("  [report] running")
    # from_agent_spec reconstructs inputs as dict-form keyed by upstream name.
    return Report(body=f"score={input_data['enrich'].score}")


# A name->callable scripted map handed to compile() / the helper so the
# reconstructed ToolNodes resolve their bodies.
SCRIPTED = {
    "hs_ingest": ingest_fn,
    "hs_enrich_v1": enrich_v1_fn,
    "hs_enrich_v2": enrich_v2_fn,
    "hs_report": report_fn,
}


def _build(enrich_fn: str, enrich_out: type) -> Construct:
    ingest = Node.scripted("ingest", fn="hs_ingest", outputs=Doc)
    enrich = Node.scripted("enrich", fn=enrich_fn, inputs=Doc, outputs=enrich_out)
    report = Node.scripted("report", fn="hs_report", inputs=enrich_out, outputs=Report)
    return Construct("hotswap-demo", nodes=[ingest, enrich, report])


def demo() -> None:
    c_v1 = _build("hs_enrich_v1", EnrichedV1)
    c_v2 = _build("hs_enrich_v2", EnrichedV2)

    saver = MemorySaver()
    config = {"configurable": {"thread_id": "hot-swap-demo"}}

    # -- Run 1: v1 via the SAME Agent Spec round-trip the helper uses, so
    # unchanged nodes reconstruct to identical synthesized types across versions
    # (only ``enrich``'s output type differs) -- otherwise every fingerprint
    # would drift and the whole graph would invalidate.
    print("=" * 60)
    print("Run 1: full pipeline (v1, score: int)")
    print("-" * 60)
    graph_v1 = compile(from_agent_spec(to_agent_spec(c_v1)), checkpointer=saver, scripted=SCRIPTED)
    first = run(graph_v1, input={"node_id": "hs"}, config=config)
    print(f"\n  Result: {first['report'].body}")

    # -- Hot-swap: emit v2 -> validate -> recompile -> resume on same thread_id.
    print("\n" + "=" * 60)
    print("Hot-swap: emit v2 Agent Spec -> validate -> recompile -> resume")
    print("-" * 60)
    print("  Expected: ingest PRESERVED; enrich + report re-execute (v2 schema)")
    second = resume_from_agent_spec(
        to_agent_spec(c_v2),
        checkpointer=saver,
        config=config,
        scripted=SCRIPTED,
    )
    print(f"\n  Result: {second['report'].body}")
    assert "score=2.0" in second["report"].body, second["report"].body
    print("\n  OK: only the changed node + its downstream re-ran; upstream reused.")


if __name__ == "__main__":
    import logging

    import structlog

    logging.getLogger().setLevel(logging.WARNING)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

    demo()
