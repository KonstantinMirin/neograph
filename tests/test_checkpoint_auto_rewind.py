"""Auto-rewind fail-loud contract (neograph-v63o / review 080726 PAT-03, MED-01).

``_auto_resume_from_divergence`` (and its async twin) walk the thread's
``get_state_history`` for the OLDEST checkpoint whose ``.next`` intersects the
invalidated node set — that is the rewind point. Historically, when NO snapshot
matched (history pruned, or every invalidated node already executed to
completion), ``rewind_checkpoint_id`` stayed ``None``, the function returned
having done nothing, and the caller's subsequent ``invoke(None)`` resumed from
the tip — silently SKIPPING the changed nodes and handing back stale results.

Fail-loud house style: a non-empty invalidated set with no rewind point must
raise ``CheckpointSchemaError`` (carrying ``invalidated_nodes``), never silently
resume from the tip. These tests pin that for both the sync and async twins, and
guard that the happy path (a rewind point exists) still mutates the config.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from neograph.errors import CheckpointSchemaError
from neograph.runner import _aauto_resume_from_divergence, _auto_resume_from_divergence


def _snapshot(next_nodes: tuple[str, ...], checkpoint_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        next=next_nodes,
        config={"configurable": {"checkpoint_id": checkpoint_id}},
    )


class _FakeGraph:
    """Minimal stand-in exposing the sync + async state-history generators the
    rewind helpers consume. Only ``get_state_history`` / ``aget_state_history``
    are touched by the code under test."""

    def __init__(self, snapshots: list[SimpleNamespace]):
        self._snapshots = snapshots

    def get_state_history(self, config):  # noqa: ANN001
        return iter(self._snapshots)

    async def aget_state_history(self, config):  # noqa: ANN001
        for snap in self._snapshots:
            yield snap


def test_auto_resume_raises_when_no_rewind_point_found():
    """Sync: invalidated set non-empty but no snapshot has an invalidated node in
    ``.next`` -> raise instead of silently resuming from the tip."""
    graph = _FakeGraph([_snapshot(("x",), "ck-2"), _snapshot(("y",), "ck-1")])
    config = {"configurable": {"thread_id": "t"}}

    with pytest.raises(CheckpointSchemaError) as exc_info:
        _auto_resume_from_divergence(graph, config, {"a", "b"})

    assert exc_info.value.invalidated_nodes == {"a", "b"}
    # The config must NOT have been mutated to resume from the tip.
    assert "checkpoint_id" not in config["configurable"]


def test_auto_resume_sets_rewind_point_when_found():
    """Sync happy path: a snapshot whose ``.next`` intersects the invalidated set
    exists -> its checkpoint_id is written into config, no raise."""
    graph = _FakeGraph([_snapshot(("a",), "ck-before-a"), _snapshot(("z",), "ck-tip")])
    config = {"configurable": {"thread_id": "t"}}

    _auto_resume_from_divergence(graph, config, {"a"})

    assert config["configurable"]["checkpoint_id"] == "ck-before-a"


def test_auto_resume_noop_when_invalidated_empty():
    """An empty invalidated set is a genuine no-op (nothing changed) — must not
    raise and must not touch config."""
    graph = _FakeGraph([_snapshot(("a",), "ck-before-a")])
    config = {"configurable": {"thread_id": "t"}}

    _auto_resume_from_divergence(graph, config, set())

    assert "checkpoint_id" not in config["configurable"]


async def test_aauto_resume_raises_when_no_rewind_point_found():
    """Async twin: same fail-loud contract."""
    graph = _FakeGraph([_snapshot(("x",), "ck-2"), _snapshot(("y",), "ck-1")])
    config = {"configurable": {"thread_id": "t"}}

    with pytest.raises(CheckpointSchemaError) as exc_info:
        await _aauto_resume_from_divergence(graph, config, {"a", "b"})

    assert exc_info.value.invalidated_nodes == {"a", "b"}
    assert "checkpoint_id" not in config["configurable"]


async def test_aauto_resume_sets_rewind_point_when_found():
    """Async happy path still mutates config."""
    graph = _FakeGraph([_snapshot(("a",), "ck-before-a"), _snapshot(("z",), "ck-tip")])
    config = {"configurable": {"thread_id": "t"}}

    await _aauto_resume_from_divergence(graph, config, {"a"})

    assert config["configurable"]["checkpoint_id"] == "ck-before-a"
