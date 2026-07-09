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

Beyond those unit twins, ``TestAutoRewindEndToEnd`` (neograph-oitm / review
080726 TQ-01) drives the WHOLE seam against a real file-backed checkpointer: it
runs a 3-node pipeline to completion, structurally mutates ONE node's output
model (same ``__qualname__``, one field's type changed — the v63o
``_type_signature`` depth-fold path), resumes on the same ``thread_id``, and
asserts via per-node execution counters that EXACTLY the changed node + its
transitive downstream re-ran while untouched upstream did not, and that the
final result reflects the NEW schema. That end-to-end assertion is the one that
would fail if ``_verify_checkpoint_schema`` stopped calling
``_auto_resume_from_divergence`` — the three disjoint unit tests it replaces all
still pass under a broken seam.
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

import neograph
from neograph import compile, construct_from_module, node, run
from neograph.errors import CheckpointSchemaError
from neograph.runner import _aauto_resume_from_divergence, _auto_resume_from_divergence
from tests.fakes import build_test_compile_kwargs


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


# =============================================================================
# End-to-end auto-rewind proof (neograph-oitm / review 080726 TQ-01)
# =============================================================================
#
# The three unit twins above exercise the rewind helper against a hand-built
# ``_FakeGraph``; ``_compute_invalidated_nodes`` is pinned separately in
# ``test_checkpoint_resume_transitive.py``; persistence roundtrip lives in
# ``test_checkpoint_sqlite_async.py``. NONE of them drives the full path — real
# pipeline -> real checkpointer -> schema mutation -> resume -> observe which
# nodes actually re-executed. If ``_verify_checkpoint_schema`` stopped invoking
# ``_auto_resume_from_divergence`` (or the rewind became a no-op), all three
# still pass. The tests below close that gap with per-node execution counters.


class _Doc(BaseModel):
    """Head-node output. Stable across both compiles — this is the UPSTREAM node
    whose counter must NOT tick on resume."""

    text: str = "hello"


class _Report(BaseModel):
    """Tail-node output. Its ``summary`` encodes the enrich value, so a stale
    (non-rewound) resume is visible in the returned result."""

    summary: str


def _make_mid_model(score_type: type) -> type:
    """Two ``_Enriched`` models with an IDENTICAL ``__qualname__``
    (``_make_mid_model.<locals>._Enriched``) differing only in the TYPE of the
    ``score`` field. Recompiling with the other one exercises the v63o
    ``_type_signature`` depth-fold: the per-node fingerprint must change even
    though the class name is unchanged, or auto-rewind never triggers.

    The change used by the tests is ``int -> float`` — a field-TYPE change (so
    ``_type_signature`` sees a different ``str(annotation)`` for the ``score``
    field and the fingerprint diverges) that is ALSO forward-coercible: the
    rewind walks ``get_state_history``, which materializes every historical
    snapshot into the CURRENT state schema, so the pre-change checkpoint's
    ``score=int`` must still validate against ``score: float`` (pydantic widens
    ``1 -> 1.0``). An incompatible change (e.g. ``int -> str``) raises a raw
    pydantic ``ValidationError`` inside the history walk BEFORE the rewind; that
    is now translated into a clean ``CheckpointSchemaError(invalidated_nodes=...)``
    (neograph-1gdw) and pinned by the non-coercible cells below."""

    class _Enriched(BaseModel):
        score: score_type  # type: ignore[valid-type]

    return _Enriched


def _counting_pipeline(counters: dict[str, int], mid_model: type, score_value: object):
    """3-node scripted chain ``ingest -> enrich -> report`` where every body
    increments a shared counter (so re-execution is observable).

    ``enrich`` is the middle node whose output model (``mid_model``) is the one
    swapped between compiles. ``report`` consumes it, so a change to ``enrich``
    must transitively invalidate ``report`` but never ``ingest``.
    """
    mod = types.ModuleType("test_e2e_rewind_mod")

    @node(mode="scripted", outputs=_Doc)
    def ingest() -> _Doc:
        counters["ingest"] += 1
        return _Doc(text="hello")

    @node(mode="scripted", outputs=mid_model)
    def enrich(ingest: _Doc):
        counters["enrich"] += 1
        return mid_model(score=score_value)

    @node(mode="scripted", outputs=_Report)
    def report(enrich: mid_model) -> _Report:
        counters["report"] += 1
        return _Report(summary=f"score={enrich.score}")

    mod.ingest = ingest
    mod.enrich = enrich
    mod.report = report
    return construct_from_module(mod, name="e2e-rewind")


class TestAutoRewindEndToEnd:
    """The headline auto-rewind feature, proven end-to-end with execution
    counters against a real file-backed Sqlite checkpointer.

    Scenario (both sync + async cells): run ``ingest -> enrich -> report`` to
    completion; recompile with ``enrich``'s output model structurally changed
    (same qualname, ``score`` field int -> float); resume on the SAME thread_id.

    The load-bearing assertions and what each catches if the seam breaks:
      (a) ``enrich`` and ``report`` re-ran (counters +1) — FAILS if the rewind
          is a no-op, because ``invoke(None)`` would resume from the completed
          tip and re-execute nothing.
      (b) ``ingest`` did NOT re-run (counter unchanged) — FAILS if the rewind
          over-rewinds to the very start instead of the checkpoint just before
          the earliest invalidated node.
      (c) the final ``report.summary`` reflects the NEW schema value — FAILS if
          a stale tip result is handed back instead of a genuine re-execution.
    """

    def test_sync_auto_rewind_reexecutes_exactly_the_invalidated_subgraph(self, tmp_path):
        db = str(tmp_path / "rewind.db")
        thread = {"configurable": {"thread_id": "e2e-rewind-sync"}}
        counters = {"ingest": 0, "enrich": 0, "report": 0}

        with SqliteSaver.from_conn_string(db) as saver:
            # v1: score is an int. Run the whole chain to completion.
            mid_v1 = _make_mid_model(int)
            graph_v1 = compile(
                _counting_pipeline(counters, mid_v1, 1), checkpointer=saver, **build_test_compile_kwargs()
            )
            first = run(graph_v1, input={"kick": "off"}, config=thread)
            assert counters == {"ingest": 1, "enrich": 1, "report": 1}
            assert first["report"].summary == "score=1"

            # v2: SAME qualname, score is now a float. Resume the same thread_id.
            mid_v2 = _make_mid_model(float)
            assert mid_v1.__qualname__ == mid_v2.__qualname__, "precondition: identical qualname"
            graph_v2 = compile(
                _counting_pipeline(counters, mid_v2, 2.0), checkpointer=saver, **build_test_compile_kwargs()
            )
            second = run(graph_v2, input={"kick": "off"}, config=thread, auto_resume=True)

        # (a) changed node + its transitive downstream re-ran.
        assert counters["enrich"] == 2, "changed node must re-execute on auto-rewind"
        assert counters["report"] == 2, "downstream of the changed node must re-execute"
        # (b) untouched upstream did NOT re-run.
        assert counters["ingest"] == 1, "upstream of the change must NOT re-execute (over-rewind)"
        # (c) the result reflects the NEW schema, not a stale tip.
        assert second["report"].summary == "score=2.0"

    async def test_async_auto_rewind_reexecutes_exactly_the_invalidated_subgraph(self, tmp_path):
        db = str(tmp_path / "rewind_async.db")
        thread = {"configurable": {"thread_id": "e2e-rewind-async"}}
        counters = {"ingest": 0, "enrich": 0, "report": 0}

        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            mid_v1 = _make_mid_model(int)
            graph_v1 = compile(
                _counting_pipeline(counters, mid_v1, 1), checkpointer=saver, **build_test_compile_kwargs()
            )
            first = await neograph.arun(graph_v1, input={"kick": "off"}, config=thread)
            assert counters == {"ingest": 1, "enrich": 1, "report": 1}
            assert first["report"].summary == "score=1"

            mid_v2 = _make_mid_model(float)
            graph_v2 = compile(
                _counting_pipeline(counters, mid_v2, 2.0), checkpointer=saver, **build_test_compile_kwargs()
            )
            second = await neograph.arun(graph_v2, input={"kick": "off"}, config=thread, auto_resume=True)

        assert counters["enrich"] == 2, "changed node must re-execute on async auto-rewind"
        assert counters["report"] == 2, "downstream of the changed node must re-execute"
        assert counters["ingest"] == 1, "upstream of the change must NOT re-execute (over-rewind)"
        assert second["report"].summary == "score=2.0"

    def test_sync_non_coercible_change_raises_clean_checkpoint_error(self, tmp_path):
        """Non-coercible field-type change (int -> str) with ``auto_resume=True``:
        the resume must raise a CLEAN ``CheckpointSchemaError(invalidated_nodes=...)``
        surfaced BEFORE any node re-executes — NOT a raw pydantic ``ValidationError``
        bubbling from inside the ``get_state_history`` walk.

        ``get_state_history`` re-materializes every historical snapshot into the
        CURRENT state schema. A coercible widening (int -> float, the TQ-01 cell)
        validates cleanly and rewinds; a non-coercible change (int -> str) makes
        pydantic reject the stored ``score=1`` and the raw ``ValidationError`` used
        to bubble before the rewind decision ever ran. See neograph-1gdw.
        """
        db = str(tmp_path / "rewind_incompat.db")
        thread = {"configurable": {"thread_id": "e2e-rewind-incompat-sync"}}
        counters = {"ingest": 0, "enrich": 0, "report": 0}

        with SqliteSaver.from_conn_string(db) as saver:
            # v1: score is an int. Run the whole chain to completion.
            graph_v1 = compile(
                _counting_pipeline(counters, _make_mid_model(int), 1),
                checkpointer=saver,
                **build_test_compile_kwargs(),
            )
            run(graph_v1, input={"kick": "off"}, config=thread)
            assert counters == {"ingest": 1, "enrich": 1, "report": 1}

            # v2: SAME qualname, score is now a str — a NON-coercible change.
            mid_v2 = _make_mid_model(str)
            graph_v2 = compile(
                _counting_pipeline(counters, mid_v2, "high"),
                checkpointer=saver,
                **build_test_compile_kwargs(),
            )
            with pytest.raises(CheckpointSchemaError) as exc_info:
                run(graph_v2, input={"kick": "off"}, config=thread, auto_resume=True)

        # The clean error carries the invalidated node set (changed node + downstream).
        assert exc_info.value.invalidated_nodes == {"enrich", "report"}
        # Surfaced BEFORE any re-execution: no counter ticked twice.
        assert counters == {"ingest": 1, "enrich": 1, "report": 1}

    async def test_async_non_coercible_change_raises_clean_checkpoint_error(self, tmp_path):
        """Async twin of the non-coercible cell: the raw ``ValidationError`` used to
        bubble from inside the ``aget_state_history`` walk; the resume must raise a
        clean ``CheckpointSchemaError(invalidated_nodes=...)`` instead. neograph-1gdw."""
        db = str(tmp_path / "rewind_incompat_async.db")
        thread = {"configurable": {"thread_id": "e2e-rewind-incompat-async"}}
        counters = {"ingest": 0, "enrich": 0, "report": 0}

        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            graph_v1 = compile(
                _counting_pipeline(counters, _make_mid_model(int), 1),
                checkpointer=saver,
                **build_test_compile_kwargs(),
            )
            await neograph.arun(graph_v1, input={"kick": "off"}, config=thread)
            assert counters == {"ingest": 1, "enrich": 1, "report": 1}

            mid_v2 = _make_mid_model(str)
            graph_v2 = compile(
                _counting_pipeline(counters, mid_v2, "high"),
                checkpointer=saver,
                **build_test_compile_kwargs(),
            )
            with pytest.raises(CheckpointSchemaError) as exc_info:
                await neograph.arun(graph_v2, input={"kick": "off"}, config=thread, auto_resume=True)

        assert exc_info.value.invalidated_nodes == {"enrich", "report"}
        assert counters == {"ingest": 1, "enrich": 1, "report": 1}

    def test_auto_resume_false_raises_with_exact_invalidated_nodes(self, tmp_path):
        """Negative cell: with ``auto_resume=False`` the schema mismatch is a hard
        error whose ``invalidated_nodes`` is EXACTLY the changed node + its
        downstream — never the untouched upstream."""
        db = str(tmp_path / "rewind_strict.db")
        thread = {"configurable": {"thread_id": "e2e-rewind-strict"}}
        counters = {"ingest": 0, "enrich": 0, "report": 0}

        with SqliteSaver.from_conn_string(db) as saver:
            graph_v1 = compile(
                _counting_pipeline(counters, _make_mid_model(int), 1),
                checkpointer=saver,
                **build_test_compile_kwargs(),
            )
            run(graph_v1, input={"kick": "off"}, config=thread)

            graph_v2 = compile(
                _counting_pipeline(counters, _make_mid_model(float), 2.0),
                checkpointer=saver,
                **build_test_compile_kwargs(),
            )
            with pytest.raises(CheckpointSchemaError) as exc_info:
                run(graph_v2, input={"kick": "off"}, config=thread, auto_resume=False)

        assert exc_info.value.invalidated_nodes == {"enrich", "report"}
        # The strict path raised BEFORE any re-execution: no counter ticked twice.
        assert counters == {"ingest": 1, "enrich": 1, "report": 1}
