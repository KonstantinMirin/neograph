"""Resume-stream protocol E2E: astream <-> interrupt <-> astream (q8ec / §6.2).

The AG-UI payoff, as its own end-to-end test: ONE logical stream that carries
custom progress events, PAUSES at a node-boundary interrupt (the Operator
modifier), surfaces the interrupt payload to the consumer, ends — and then
RESUMES the same logical stream when the consumer calls ``astream`` again with
``resume=``.

Typed contract for how an interrupt appears in the event sequence (verified
against LangGraph 1.x):

  * progress events arrive as ``("custom", <envelope>)`` tuples;
  * the interrupt arrives as an ``("updates", {"__interrupt__": (Interrupt(
    value=<payload>, id=...),)})`` tuple and is the LAST meaningful chunk of the
    first stream;
  * calling ``astream(graph, resume=<answer>, config=<same>)`` re-enters the
    same run and yields the post-interrupt updates.

Drives REAL ``astream`` against a REAL ``InMemorySaver`` — no mocks.
"""

from __future__ import annotations

import types as _types

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from neograph import astream, compile, construct_from_module, emit_progress, node
from tests.fakes import build_test_compile_kwargs
from tests.schemas import Claims, ValidationResult


class Milestone(BaseModel):
    stage: str


def _interrupt_stream_pipeline():
    """gate (emits progress, fails validation -> Operator interrupt) -> finalize."""
    mod = _types.ModuleType("test_resume_stream_mod")

    @node(
        mode="scripted",
        outputs=ValidationResult,
        interrupt_when=lambda state: {"issues": state.gate.issues} if state.gate and not state.gate.passed else None,
    )
    def gate() -> ValidationResult:
        emit_progress(Milestone(stage="gating"))
        return ValidationResult(passed=False, issues=["needs review"])

    @node(mode="scripted", outputs=Claims)
    def finalize(gate: ValidationResult) -> Claims:
        return Claims(items=["done"])

    mod.gate = gate
    mod.finalize = finalize
    return construct_from_module(mod, name="resume-stream")


def _is_interrupt_chunk(chunk) -> bool:
    return (
        isinstance(chunk, tuple)
        and len(chunk) == 2
        and chunk[0] == "updates"
        and isinstance(chunk[1], dict)
        and "__interrupt__" in chunk[1]
    )


async def test_astream_pauses_at_interrupt_then_resumes_same_stream():
    graph = compile(
        _interrupt_stream_pipeline(),
        checkpointer=InMemorySaver(),
        **build_test_compile_kwargs(),
    )
    config = {"configurable": {"thread_id": "resume-stream-1"}}

    # ── Leg 1: stream until the node-boundary interrupt ──
    first_customs: list = []
    interrupt_payload = None
    async for chunk in astream(graph, input={"node_id": "rs-001"}, config=config, stream_mode=["custom", "updates"]):
        if isinstance(chunk, tuple) and chunk[0] == "custom":
            first_customs.append(chunk[1])
        if _is_interrupt_chunk(chunk):
            interrupt_payload = chunk[1]["__interrupt__"][0].value

    # Progress surfaced before the pause...
    assert first_customs == [{"neograph_event": "progress", "event_type": "Milestone", "data": {"stage": "gating"}}]
    # ...and the interrupt payload reached the consumer with its typed shape.
    assert interrupt_payload == {"issues": ["needs review"]}

    # ── Leg 2: resume the SAME logical stream with the human's answer ──
    resumed_updates: list = []
    async for chunk in astream(graph, resume={"approved": True}, config=config, stream_mode=["custom", "updates"]):
        if isinstance(chunk, tuple) and chunk[0] == "updates":
            resumed_updates.append(chunk[1])

    # finalize ran on resume — the stream continued past the interrupt.
    finalize_deltas = [u.get("finalize") for u in resumed_updates if "finalize" in u]
    assert finalize_deltas, f"finalize never ran on resume; got {resumed_updates}"
    assert finalize_deltas[0]["finalize"] == Claims(items=["done"])
