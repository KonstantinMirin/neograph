"""Phase 1d driver-twin tests — ``neograph.arun`` (async twin of ``run``).

Pins neograph-w74k.2.4. Core Invariant: ``arun()`` is a driver-parallel twin of
``run()`` over ONE shared pure core — it mirrors ``run``'s three-mode branch
structure and diverges ONLY where the sync API touches the graph/checkpointer
(``graph.invoke`` -> ``await graph.ainvoke``; ``checkpointer.get_tuple`` ->
``await aget_tuple``; ``graph.get_state_history`` -> ``async for
aget_state_history``).

These tests drive the RUNNER surface (``neograph.arun``), not the graph layer —
that is the whole point of Phase 1d. They MUST fail now because ``neograph.arun``
does not exist yet (``AttributeError``/``ImportError``); they pass once arun +
the async checkpoint twins land.

MED-3 (the load-bearing test): ``test_arun_resume_completes_on_existing_checkpoint``
compiles WITH a langgraph ``InMemorySaver`` and arun-s the SAME thread_id twice so
the second call hits the checkpoint-EXISTS branch. That branch executes the async
await-shapes — ``await checkpointer.aget_tuple(...)`` inside
``_ahas_existing_checkpoint`` / ``_averify_checkpoint_schema`` — which the sole
harness-flip test (no checkpointer) never reaches. ``InMemorySaver.aget_tuple`` is
a real coroutine, so this needs NO ``AsyncSqliteSaver`` (real file-backed
mixed-driver resume is Phase 1e per the user DECISION). Mirrors the sync
save->resume pattern in ``tests/test_checkpoint_resume.py``.
"""

from __future__ import annotations

import asyncio
import types as _types

from langgraph.checkpoint.memory import InMemorySaver

import neograph
from neograph import compile, construct_from_module, node, run
from tests.fakes import build_test_compile_kwargs
from tests.schemas import Claims, RawText


def _trivial_pipeline():
    """Minimal two-node scripted pipeline: fetch -> process.

    Mirrors ``tests/test_async_harness.py::_trivial_pipeline`` — no LLM, no
    checkpointer dependency in the construct itself.
    """
    mod = _types.ModuleType("test_async_runner_mod")

    @node(mode="scripted", outputs=RawText)
    def fetch() -> RawText:
        return RawText(text="hello")

    @node(mode="scripted", outputs=Claims)
    def process(fetch: RawText) -> Claims:
        return Claims(items=[fetch.text.upper()])

    mod.fetch = fetch
    mod.process = process
    return construct_from_module(mod, name="async-runner")


class TestArunNewExecution:
    """arun's new-execution path — the async twin of ``run(graph, input=...)``.

    RED now: ``neograph.arun`` does not exist -> ``AttributeError``.
    """

    def test_arun_returns_pipeline_result_when_new_execution(self):
        """``asyncio.run(arun(graph, input=...))`` returns the same result dict
        shape ``run`` does: ``result['process'] == Claims(items=['HELLO'])``."""
        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())

        result = asyncio.run(neograph.arun(graph, input={"node_id": "async-runner-001"}))

        assert result["process"] == Claims(items=["HELLO"])


class TestArunRunParity:
    """arun == run parity — locks the driver twin: the two drivers cannot
    behaviorally fork on the new-execution path.

    RED now: ``neograph.arun`` does not exist -> ``AttributeError``.
    """

    def test_arun_result_equals_run_result_on_same_graph(self):
        """Same construct, same input: ``run`` and ``arun`` return equal results.

        Compiled twice (each ``compile`` produces a fresh graph) from the same
        construct so neither driver mutates state the other observes.
        """
        sync_graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        async_graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())

        sync_result = run(sync_graph, input={"node_id": "parity-001"})
        async_result = asyncio.run(
            neograph.arun(async_graph, input={"node_id": "parity-001"})
        )

        assert async_result["process"] == sync_result["process"] == Claims(items=["HELLO"])
        assert async_result == sync_result


class TestArunCheckpointResume:
    """MED-3 — the InMemorySaver arun-resume smoke that EXECUTES the async
    checkpoint await-shapes.

    arun the SAME thread_id twice against an ``InMemorySaver``-checkpointed
    graph. The second arun hits the checkpoint-EXISTS branch, exercising
    ``await checkpointer.aget_tuple(...)`` inside ``_ahas_existing_checkpoint``
    and ``_averify_checkpoint_schema`` — the async await-shapes that the sole
    harness-flip test (no checkpointer) never reaches.

    RED now: ``neograph.arun`` does not exist -> ``AttributeError``.
    """

    def test_arun_resume_completes_on_existing_checkpoint(self):
        """Second arun on the same thread_id resumes from the checkpoint the
        first arun persisted, completes without error, and returns the pipeline
        result — proving the async checkpoint twins' await-shapes are correct."""
        checkpointer = InMemorySaver()
        graph = compile(
            _trivial_pipeline(),
            checkpointer=checkpointer,
            **build_test_compile_kwargs(),
        )
        config = {"configurable": {"thread_id": "arun-resume-x"}}

        # First arun: new execution, InMemorySaver persists the state.
        first = asyncio.run(
            neograph.arun(graph, input={"node_id": "ckpt-async-001"}, config=config)
        )
        assert first["process"] == Claims(items=["HELLO"])

        # Second arun on the SAME thread_id: hits the checkpoint-EXISTS branch,
        # awaiting checkpointer.aget_tuple inside the async checkpoint twins.
        second = asyncio.run(
            neograph.arun(graph, input={"node_id": "ckpt-async-001"}, config=config)
        )
        assert second["process"] == Claims(items=["HELLO"])
