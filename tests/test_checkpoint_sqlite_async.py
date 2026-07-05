"""Phase 1e — file-backed Sqlite mixed-driver resume + crash-recovery E2Es.

Pins neograph-w74k.2.5. Core Invariant: ``arun``'s async checkpoint helpers
(``aget_tuple``/``aget_state_history``) and ``run``'s sync helpers
(``get_tuple``/``get_state_history``) read and write ONE identical file-backed
sqlite checkpoint schema, so a thread persisted by either driver resumes under
the other on the same ``thread_id`` — the checkpointer choice is an I/O-driver
detail, never a data-format fork.

CHARACTERIZATION, not TDD red. Phase 1d already shipped ``neograph.arun`` + the
async checkpoint helpers, and ``compile()`` threads the checkpointer through
``graph.compile()`` unchanged — so mixed-driver resume ALREADY WORKS. These
tests are the sanctioned "guard existing behavior" case (the user DECISION on
w74k.2.5): they LOCK the already-working capability against regression. There
is NO production red-green here — the phase is dependency + tests only, so these
tests PASS on first run. A FAIL means a real mixed-driver bug, report it loudly.

NO MOCKS. Every leg drives REAL ``run``/``arun`` against a REAL file-backed
``SqliteSaver`` / ``AsyncSqliteSaver`` on ``tmp_path``. Each test owns its own
db file + unique ``thread_id``; savers are opened INSIDE the test via
``with`` / ``async with`` and kept open across the run/arun call. Never share an
``AsyncSqliteSaver`` across event loops.

Reuse:
- ``_trivial_pipeline`` (fetch -> process, RawText -> Claims) from
  ``tests/test_async_runner.py``.
- the save -> resume assertion shape from ``tests/test_checkpoint_resume.py``.
- the ``interrupt_when`` mid-flight pause from ``tests/modifiers/test_operator.py``.

The ``Deserializing unregistered type`` msgpack ``UserWarning`` on cross-driver
resume is non-fatal (pyproject ``filterwarnings`` has no ``error``); do NOT set
``LANGGRAPH_STRICT_MSGPACK``.
"""

from __future__ import annotations

import types as _types

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import neograph
from neograph import compile, construct_from_module, node, run
from neograph.errors import ConfigurationError
from tests.fakes import build_test_compile_kwargs
from tests.schemas import Claims, RawText, ValidationResult


def _trivial_pipeline():
    """Minimal two-node scripted pipeline: fetch -> process.

    Mirrors ``tests/test_async_runner.py::_trivial_pipeline`` — no LLM, the
    construct itself carries no checkpointer dependency.
    """
    mod = _types.ModuleType("test_ckpt_sqlite_trivial_mod")

    @node(mode="scripted", outputs=RawText)
    def fetch() -> RawText:
        return RawText(text="hello")

    @node(mode="scripted", outputs=Claims)
    def process(fetch: RawText) -> Claims:
        return Claims(items=[fetch.text.upper()])

    mod.fetch = fetch
    mod.process = process
    return construct_from_module(mod, name="ckpt-sqlite-trivial")


def _interrupt_pipeline():
    """Two-node pipeline that pauses MID-FLIGHT via ``interrupt_when``.

    ``gate`` produces a failing ``ValidationResult`` and its ``interrupt_when``
    fires (Operator wiring pauses the graph AFTER ``gate`` writes its output but
    BEFORE ``finalize`` runs). A partial checkpoint is persisted. On resume,
    ``finalize`` runs and produces the completed ``Claims`` output. Mirrors the
    ``interrupt_when`` + resume pattern in ``tests/modifiers/test_operator.py``.
    """
    mod = _types.ModuleType("test_ckpt_sqlite_interrupt_mod")

    @node(
        mode="scripted",
        outputs=ValidationResult,
        interrupt_when=lambda state: (
            {"issues": state.gate.issues}
            if state.gate and not state.gate.passed
            else None
        ),
    )
    def gate() -> ValidationResult:
        return ValidationResult(passed=False, issues=["needs review"])

    @node(mode="scripted", outputs=Claims)
    def finalize(gate: ValidationResult) -> Claims:
        return Claims(items=["done"])

    mod.gate = gate
    mod.finalize = finalize
    return construct_from_module(mod, name="ckpt-sqlite-interrupt")


async def test_async_resume_same_driver_file_backed(tmp_path):
    """arun the SAME thread_id twice against ONE file-backed AsyncSqliteSaver.

    The second arun hits the checkpoint-EXISTS branch — ``await aget_tuple``
    inside ``_ahas_existing_checkpoint`` / ``_averify_checkpoint_schema`` — this
    time against a REAL sqlite FILE (the file-backed twin of
    ``test_async_runner.py::TestArunCheckpointResume``). Both calls complete.
    """
    db = str(tmp_path / "ckpt.db")
    config = {"configurable": {"thread_id": "async-same-driver-1e"}}

    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        graph = compile(
            _trivial_pipeline(),
            checkpointer=saver,
            **build_test_compile_kwargs(),
        )

        first = await neograph.arun(
            graph, input={"node_id": "same-driver-001"}, config=config
        )
        assert first["process"] == Claims(items=["HELLO"])

        # Second arun on the SAME thread_id + SAME saver: checkpoint-EXISTS
        # branch awaits aget_tuple against the real file, resumes, completes.
        second = await neograph.arun(
            graph, input={"node_id": "same-driver-001"}, config=config
        )
        assert second["process"] == Claims(items=["HELLO"])


async def test_mixed_driver_sync_write_async_resume(tmp_path):
    """Sync ``run`` (SqliteSaver) writes a COMPLETED thread to a file; async
    ``arun`` (AsyncSqliteSaver) on the SAME file + thread_id reads back the
    final state.

    Proves cross-driver READ + schema/serde compatibility: the async driver
    deserializes what the sync driver wrote. Result equals a plain
    (non-checkpointed) run.
    """
    db = str(tmp_path / "ckpt.db")
    config = {"configurable": {"thread_id": "mixed-fwd-1e"}}

    # Reference: a plain, non-checkpointed run of the same pipeline.
    plain = run(
        compile(_trivial_pipeline(), **build_test_compile_kwargs()),
        input={"node_id": "plain-ref"},
    )

    # Sync driver writes the completed thread to the file.
    with SqliteSaver.from_conn_string(db) as sync_saver:
        sync_graph = compile(
            _trivial_pipeline(),
            checkpointer=sync_saver,
            **build_test_compile_kwargs(),
        )
        wrote = run(sync_graph, input={"node_id": "mixed-001"}, config=config)
        assert wrote["process"] == Claims(items=["HELLO"])

    # Async driver reads the SAME file + thread_id back to final state.
    async with AsyncSqliteSaver.from_conn_string(db) as async_saver:
        async_graph = compile(
            _trivial_pipeline(),
            checkpointer=async_saver,
            **build_test_compile_kwargs(),
        )
        resumed = await neograph.arun(async_graph, config=config)

    assert resumed["process"] == plain["process"] == Claims(items=["HELLO"])


async def test_mixed_driver_reverse_async_write_sync_resume(tmp_path):
    """Symmetry: async ``arun`` (AsyncSqliteSaver) writes the thread; sync
    ``run`` (SqliteSaver) on the SAME file + thread_id reads it back.

    The reverse direction of the cross-driver invariant — proves the two
    drivers share ONE checkpoint schema in BOTH directions.
    """
    db = str(tmp_path / "ckpt.db")
    config = {"configurable": {"thread_id": "mixed-rev-1e"}}

    # Async driver writes the completed thread to the file.
    async with AsyncSqliteSaver.from_conn_string(db) as async_saver:
        async_graph = compile(
            _trivial_pipeline(),
            checkpointer=async_saver,
            **build_test_compile_kwargs(),
        )
        wrote = await neograph.arun(
            async_graph, input={"node_id": "rev-001"}, config=config
        )
        assert wrote["process"] == Claims(items=["HELLO"])

    # Sync driver reads the SAME file + thread_id back to final state.
    with SqliteSaver.from_conn_string(db) as sync_saver:
        sync_graph = compile(
            _trivial_pipeline(),
            checkpointer=sync_saver,
            **build_test_compile_kwargs(),
        )
        resumed = run(sync_graph, config=config)

    assert resumed["process"] == Claims(items=["HELLO"])


async def test_mixed_driver_sync_write_mid_flight_async_resume(tmp_path):
    """THE DEFINITIVE M6b TEST (refinement MED-2): sync-write -> async-resume
    of a MID-FLIGHT (interrupted) checkpoint.

    Sync ``run`` + an Operator ``interrupt_when`` node writes a PARTIAL
    checkpoint (gate output persisted, ``finalize`` not yet run) to a file-backed
    SqliteSaver, and the run returns with ``__interrupt__``. Then AsyncSqliteSaver
    on the SAME file + thread_id ``arun(resume=...)`` COMPLETES the pipeline.
    This is the literal 'resumes under the other driver' — the sharpest form of
    the Core Invariant, which composing two weaker (completed-thread + same-driver)
    tests only approximates.
    """
    db = str(tmp_path / "ckpt.db")
    config = {"configurable": {"thread_id": "midflight-1e"}}

    # Sync driver runs until the mid-flight interrupt and persists a partial
    # checkpoint to the file.
    with SqliteSaver.from_conn_string(db) as sync_saver:
        sync_graph = compile(
            _interrupt_pipeline(),
            checkpointer=sync_saver,
            **build_test_compile_kwargs(),
        )
        paused = run(sync_graph, input={"node_id": "mf-001"}, config=config)
        assert "__interrupt__" in paused
        assert paused["gate"].passed is False

    # Async driver reopens the SAME file + thread_id and resumes the partial
    # checkpoint to completion — finalize runs under the OTHER driver.
    async with AsyncSqliteSaver.from_conn_string(db) as async_saver:
        async_graph = compile(
            _interrupt_pipeline(),
            checkpointer=async_saver,
            **build_test_compile_kwargs(),
        )
        completed = await neograph.arun(
            async_graph, resume={"approved": True}, config=config
        )

    assert completed["finalize"] == Claims(items=["done"])


async def test_crash_recovery_async_across_reopened_saver(tmp_path):
    """Crash recovery under the async driver: a mid-flight interrupt persists a
    partial checkpoint under AsyncSqliteSaver #1; a SEPARATE AsyncSqliteSaver #2
    (new connection = simulated crash / new process) reopens the SAME file and
    ``arun(resume=...)`` completes.

    The two ``async with`` blocks are DISTINCT connections, so completion proves
    FILE durability — the resume reads persisted bytes, not in-connection memory.
    """
    db = str(tmp_path / "ckpt.db")
    config = {"configurable": {"thread_id": "crash-async-1e"}}

    # Block A: interrupt + persist a partial checkpoint, then close the saver
    # (simulated crash on exit of the async-with).
    async with AsyncSqliteSaver.from_conn_string(db) as saver_a:
        graph_a = compile(
            _interrupt_pipeline(),
            checkpointer=saver_a,
            **build_test_compile_kwargs(),
        )
        paused = await neograph.arun(
            graph_a, input={"node_id": "crash-001"}, config=config
        )
        assert "__interrupt__" in paused
        assert paused["gate"].passed is False

    # Block B: a FRESH connection to the SAME file resumes from the persisted
    # partial checkpoint and completes.
    async with AsyncSqliteSaver.from_conn_string(db) as saver_b:
        graph_b = compile(
            _interrupt_pipeline(),
            checkpointer=saver_b,
            **build_test_compile_kwargs(),
        )
        completed = await neograph.arun(
            graph_b, resume={"approved": True}, config=config
        )

    assert completed["finalize"] == Claims(items=["done"])


class TestWrongDriverCheckpointerGuard:
    """neograph-dqt5 — wrong-driver checkpointer misuse must FAIL LOUD.

    Passing an async-only saver (AsyncSqliteSaver) to the SYNCHRONOUS driver
    (``run``/``stream``) or a sync-only saver (SqliteSaver) to the ASYNC driver
    (``arun``/``astream``) is user error. Before the guard, the wrong-driver
    checkpoint probe either BLOCKED forever (sync ``run`` -> ``AsyncSqliteSaver.
    get_tuple`` bridges to a non-running loop via ``run_coroutine_threadsafe``)
    or raised a raw ``NotImplementedError`` (async ``arun`` ->
    ``SqliteSaver.aget_tuple``); the ``_has_existing_checkpoint`` swallow could
    even discard the failure and SILENTLY start a fresh run that IGNORES an
    existing checkpoint. The guard detects the mismatch at run/arun ENTRY and
    raises a clear ``ConfigurationError`` naming the right driver.
    """

    def _graph_with(self, saver):
        return compile(
            _trivial_pipeline(),
            checkpointer=saver,
            **build_test_compile_kwargs(),
        )

    async def test_sync_run_rejects_async_only_saver(self, tmp_path):
        """``run`` with an ``AsyncSqliteSaver`` raises a clear ConfigurationError
        at entry instead of blocking on the bridged ``get_tuple``."""
        db = str(tmp_path / "ckpt.db")
        config = {"configurable": {"thread_id": "wrong-driver-async-saver"}}
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            graph = self._graph_with(saver)
            with pytest.raises(ConfigurationError) as exc:
                run(graph, input={"node_id": "x"}, config=config)
        msg = str(exc.value)
        assert "async" in msg.lower()
        # The hint must point the user at the async driver.
        assert "arun" in msg

    async def test_arun_rejects_sync_only_saver(self, tmp_path):
        """``arun`` with a ``SqliteSaver`` raises a clear ConfigurationError at
        entry instead of a raw ``NotImplementedError`` from ``aget_tuple``."""
        db = str(tmp_path / "ckpt.db")
        config = {"configurable": {"thread_id": "wrong-driver-sync-saver"}}
        with SqliteSaver.from_conn_string(db) as saver:
            graph = self._graph_with(saver)
            with pytest.raises(ConfigurationError) as exc:
                await neograph.arun(graph, input={"node_id": "x"}, config=config)
        msg = str(exc.value)
        assert "sync" in msg.lower()
        # The hint must point the user at the sync driver.
        assert "run" in msg

    async def test_sync_stream_rejects_async_only_saver(self, tmp_path):
        """The guard lives in the shared ``_prepare`` brain, so ``stream``
        (sync) rejects an async-only saver too."""
        db = str(tmp_path / "ckpt.db")
        config = {"configurable": {"thread_id": "wrong-driver-async-stream"}}
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            graph = self._graph_with(saver)
            with pytest.raises(ConfigurationError):
                # stream is lazy; the guard runs in _prepare BEFORE the first
                # chunk, so merely starting iteration must raise.
                list(neograph.stream(graph, input={"node_id": "x"}, config=config))

    async def test_dual_capable_memory_saver_accepted_by_both_drivers(self, tmp_path):
        """No false positives: a dual-capable ``MemorySaver`` (both get_tuple and
        aget_tuple genuinely implemented, no bound event loop) is accepted by
        BOTH ``run`` and ``arun``."""
        saver = MemorySaver()
        graph = self._graph_with(saver)

        sync_out = run(
            graph,
            input={"node_id": "dual-sync"},
            config={"configurable": {"thread_id": "dual-sync-1"}},
        )
        assert sync_out["process"] == Claims(items=["HELLO"])

        async_out = await neograph.arun(
            graph,
            input={"node_id": "dual-async"},
            config={"configurable": {"thread_id": "dual-async-1"}},
        )
        assert async_out["process"] == Claims(items=["HELLO"])
