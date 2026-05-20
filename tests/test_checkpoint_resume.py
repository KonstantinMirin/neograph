"""Checkpoint roundtrip regression guard (neograph-5s00).

Backfill test for neograph-u8cg, which deleted `_register_msgpack_types`
from `compiler.py`. The deletion assumed LangGraph's default "warn-all"
behavior is safe for Pydantic state shapes containing the types the
removed helper used to allowlist (datetime, custom Pydantic BaseModel,
Enum, nested model).

This test exercises the full save -> resume path through MemorySaver
with a state model containing those types and asserts the loaded state
equals the saved state.

Outcome (GREEN) = warn-all is safe; u8cg validated.
Outcome (RED)   = u8cg regressed; file P0 and restore the allowlist via
                  a public LangGraph API.
"""

from __future__ import annotations

import types as _types
from datetime import datetime
from enum import StrEnum

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from neograph import compile, construct_from_module, node, run
from tests.fakes import build_test_compile_kwargs


class Priority(StrEnum):
    LOW = "low"
    HIGH = "high"


class Tag(BaseModel, frozen=True):
    key: str
    value: str


class CheckpointPayload(BaseModel, frozen=True):
    """State output covering every type the deleted allowlist helper used
    to register: datetime, Enum, custom Pydantic BaseModel, nested model."""

    label: str
    created_at: datetime
    priority: Priority
    tag: Tag
    tags: list[Tag]


def test_checkpoint_roundtrip_with_pydantic_state_after_serde_helper_removed():
    """Save state with datetime/Enum/Pydantic/nested types, resume via
    same thread_id, and assert the deserialized state equals the saved
    state. Regression guard against deletion of _register_msgpack_types
    (neograph-u8cg)."""

    mod = _types.ModuleType("test_checkpoint_resume_mod")

    expected_payload = CheckpointPayload(
        label="alpha",
        created_at=datetime(2026, 5, 19, 12, 30, 45),
        priority=Priority.HIGH,
        tag=Tag(key="env", value="prod"),
        tags=[Tag(key="region", value="us-east"), Tag(key="tier", value="gold")],
    )

    @node(mode="scripted", outputs=CheckpointPayload)
    def produce_payload() -> CheckpointPayload:
        return expected_payload

    mod.produce_payload = produce_payload

    pipeline = construct_from_module(mod, name="checkpoint-roundtrip")
    checkpointer = MemorySaver()
    graph = compile(pipeline, checkpointer=checkpointer, **build_test_compile_kwargs())

    config = {"configurable": {"thread_id": "roundtrip-test"}}

    # Phase 1: execute and let MemorySaver persist the state.
    first = run(graph, input={"node_id": "ckpt-001"}, config=config)
    assert first["produce_payload"] == expected_payload

    # Phase 2: read the persisted snapshot directly from the checkpointer.
    # This is the path that exercises msgpack deserialization for every
    # type in the state model.
    snapshot = graph.get_state(config)
    loaded = snapshot.values["produce_payload"]

    # Type identity survived the roundtrip (not a dict or stringified blob).
    assert isinstance(loaded, CheckpointPayload)
    assert isinstance(loaded.created_at, datetime)
    assert isinstance(loaded.priority, Priority)
    assert isinstance(loaded.tag, Tag)
    assert all(isinstance(t, Tag) for t in loaded.tags)

    # Value equality across the full payload.
    assert loaded == expected_payload
