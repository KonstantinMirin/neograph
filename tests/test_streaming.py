"""Streaming runner verbs — ``stream``/``astream`` + ``_finalize_chunk`` (q8ec).

TDD red-first: these drive the NOT-YET-IMPLEMENTED streaming surface. The three
genuinely-new pieces are covered:

  * ``_finalize_chunk(chunk, stream_mode)`` — the ONLY place a bug can silently
    leak ``neo_*`` framework keys or corrupt a user payload. Per-mode unit tests
    (``values`` strips top-level, ``updates`` strips one level down,
    ``custom``/``messages``/``debug`` pass through untouched, list stream_mode
    yields ``(mode, chunk)`` tuples finalized per their own mode).
  * ``stream``/``astream`` — thin verbs over ``_prepare``/``_aprepare`` +
    ``_finalize_chunk``. Driven end-to-end against a real compiled graph.
  * The refactor is behavior-preserving: ``run``/``arun`` still return
    ``neo_*``-free results.
"""

from __future__ import annotations

import types as _types

from neograph import compile, construct_from_module, node, run
from neograph._state_keys import StateKeys
from tests.fakes import build_test_compile_kwargs
from tests.schemas import Claims, RawText


def _trivial_pipeline():
    mod = _types.ModuleType("test_streaming_trivial_mod")

    @node(mode="scripted", outputs=RawText)
    def fetch() -> RawText:
        return RawText(text="hello")

    @node(mode="scripted", outputs=Claims)
    def process(fetch: RawText) -> Claims:
        return Claims(items=[fetch.text.upper()])

    mod.fetch = fetch
    mod.process = process
    return construct_from_module(mod, name="streaming-trivial")


# ═══════════════════════════════════════════════════════════════════════════
# _finalize_chunk — the load-bearing stripping logic
# ═══════════════════════════════════════════════════════════════════════════
class TestFinalizeChunkValuesMode:
    """``values`` chunks are full state dicts — strip ``neo_*`` at the top."""

    def test_strips_neo_keys_from_values_chunk(self):
        from neograph.runner import _finalize_chunk

        chunk = {
            "process": Claims(items=["HELLO"]),
            StateKeys.SCHEMA_FINGERPRINT: "abc123",
            StateKeys.NODE_FINGERPRINTS: {"process": "def"},
        }
        out = _finalize_chunk(chunk, "values")
        assert out == {"process": Claims(items=["HELLO"])}

    def test_leaves_user_keys_intact(self):
        from neograph.runner import _finalize_chunk

        chunk = {"a": 1, "b": 2}
        assert _finalize_chunk(chunk, "values") == {"a": 1, "b": 2}


class TestFinalizeChunkUpdatesMode:
    """``updates`` chunks are ``{node: delta}`` — strip one level down; a
    per-node delta can carry fingerprints (e.g. the first node's write)."""

    def test_strips_neo_keys_inside_each_node_delta(self):
        from neograph.runner import _finalize_chunk

        chunk = {
            "process": {
                "process": Claims(items=["HELLO"]),
                StateKeys.SCHEMA_FINGERPRINT: "abc",
                StateKeys.NODE_FINGERPRINTS: {"process": "x"},
            }
        }
        out = _finalize_chunk(chunk, "updates")
        assert out == {"process": {"process": Claims(items=["HELLO"])}}

    def test_interrupt_update_passes_through(self):
        from neograph.runner import _finalize_chunk

        # ``__interrupt__`` is not a node delta dict; it must survive unharmed.
        chunk = {"__interrupt__": ("payload",)}
        assert _finalize_chunk(chunk, "updates") == {"__interrupt__": ("payload",)}


class TestFinalizeChunkPassthroughModes:
    """``custom``/``messages``/``debug`` are user payloads / token tuples —
    NEVER stripped, even if they happen to contain a ``neo_``-prefixed key."""

    def test_custom_chunk_passes_through_untouched(self):
        from neograph.runner import _finalize_chunk

        payload = {"stage": "emit", "neo_looking_like_internal": "keep me"}
        assert _finalize_chunk(payload, "custom") is payload

    def test_messages_chunk_passes_through_untouched(self):
        from neograph.runner import _finalize_chunk

        token_tuple = ("token", {"neo_meta": 1})
        assert _finalize_chunk(token_tuple, "messages") is token_tuple

    def test_debug_chunk_passes_through_untouched(self):
        from neograph.runner import _finalize_chunk

        payload = {"type": "task", "neo_x": 1}
        assert _finalize_chunk(payload, "debug") is payload


class TestFinalizeChunkListMode:
    """A list ``stream_mode`` makes LangGraph yield ``(mode, chunk)`` tuples;
    each tuple is finalized by ITS OWN mode."""

    def test_values_tuple_is_stripped(self):
        from neograph.runner import _finalize_chunk

        chunk = ("values", {"process": Claims(items=["HELLO"]), StateKeys.SCHEMA_FINGERPRINT: "abc"})
        mode, payload = _finalize_chunk(chunk, ["values", "custom"])
        assert mode == "values"
        assert payload == {"process": Claims(items=["HELLO"])}

    def test_custom_tuple_passes_through(self):
        from neograph.runner import _finalize_chunk

        chunk = ("custom", {"stage": "emit"})
        out = _finalize_chunk(chunk, ["values", "custom"])
        assert out == ("custom", {"stage": "emit"})

    def test_updates_tuple_strips_one_level_down(self):
        from neograph.runner import _finalize_chunk

        chunk = ("updates", {"process": {"process": Claims(items=["HELLO"]), StateKeys.NODE_FINGERPRINTS: {}}})
        mode, payload = _finalize_chunk(chunk, ["updates", "custom"])
        assert mode == "updates"
        assert payload == {"process": {"process": Claims(items=["HELLO"])}}


# ═══════════════════════════════════════════════════════════════════════════
# stream / astream — thin verbs, driven end-to-end
# ═══════════════════════════════════════════════════════════════════════════
class TestSyncStreamVerb:
    def test_stream_values_yields_neo_free_final_state(self):
        from neograph import stream

        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        chunks = list(stream(graph, input={"node_id": "s-001"}, stream_mode="values"))

        assert chunks, "stream produced no chunks"
        final = chunks[-1]
        assert final["process"] == Claims(items=["HELLO"])
        # No framework keys leaked into any streamed values chunk.
        for ch in chunks:
            assert not any(k.startswith(StateKeys.FRAMEWORK_PREFIX) for k in ch)

    def test_stream_updates_strips_fingerprints_from_deltas(self):
        from neograph import stream

        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        chunks = list(stream(graph, input={"node_id": "s-002"}, stream_mode="updates"))

        for ch in chunks:
            for delta in ch.values():
                if isinstance(delta, dict):
                    assert not any(
                        k.startswith(StateKeys.FRAMEWORK_PREFIX) for k in delta
                    )


class TestAsyncStreamVerb:
    async def test_astream_values_yields_neo_free_final_state(self):
        from neograph import astream

        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        chunks = [
            ch
            async for ch in astream(graph, input={"node_id": "as-001"}, stream_mode="values")
        ]

        assert chunks, "astream produced no chunks"
        final = chunks[-1]
        assert final["process"] == Claims(items=["HELLO"])
        for ch in chunks:
            assert not any(k.startswith(StateKeys.FRAMEWORK_PREFIX) for k in ch)

    async def test_astream_result_matches_run(self):
        from neograph import astream

        graph_a = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        graph_b = compile(_trivial_pipeline(), **build_test_compile_kwargs())

        batch = run(graph_a, input={"node_id": "parity"})
        chunks = [
            ch
            async for ch in astream(graph_b, input={"node_id": "parity"}, stream_mode="values")
        ]
        assert chunks[-1]["process"] == batch["process"]


# ═══════════════════════════════════════════════════════════════════════════
# run/arun still batch verbs (behavior preserved through the refactor)
# ═══════════════════════════════════════════════════════════════════════════
class TestBatchVerbsUnchanged:
    def test_run_still_returns_neo_free_result(self):
        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "b-001"})
        assert result["process"] == Claims(items=["HELLO"])
        assert not any(k.startswith(StateKeys.FRAMEWORK_PREFIX) for k in result)


# ═══════════════════════════════════════════════════════════════════════════
# Early-close eviction — the finally is inside the generator body (yc38)
# ═══════════════════════════════════════════════════════════════════════════
class TestStreamEarlyCloseEvictsRunCache:
    """``stream``/``astream`` wire ``_evict_run_cache`` into a ``finally`` INSIDE
    the generator body, so a consumer that abandons the stream early (``.close()``
    / GC) still fires ``GeneratorExit`` through that finally and drops the run's
    per-run cache entries — a loop-bound handle never lingers past its run. Only
    normal-completion eviction was covered; this pins the early-close path."""

    def test_sync_stream_close_after_one_chunk_evicts(self, monkeypatch):
        import neograph.runner as _runner

        evicted: list[str] = []
        monkeypatch.setattr(_runner, "evict_run", lambda run_id: evicted.append(run_id))

        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        gen = _runner.stream(graph, input={"node_id": "ec-001"}, stream_mode="values")
        next(gen)  # consume exactly one chunk, then abandon the stream
        gen.close()

        assert evicted, "early .close() did not fire the eviction finally"

    async def test_astream_close_after_one_chunk_evicts(self, monkeypatch):
        import neograph.runner as _runner

        evicted: list[str] = []
        monkeypatch.setattr(_runner, "evict_run", lambda run_id: evicted.append(run_id))

        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        agen = _runner.astream(graph, input={"node_id": "ec-002"}, stream_mode="values")
        await agen.__anext__()  # one chunk
        await agen.aclose()

        assert evicted, "early .aclose() did not fire the eviction finally"


# ═══════════════════════════════════════════════════════════════════════════
# di_inputs anti-leak — _neo_di_inputs is a config-only side-channel (yc38)
# ═══════════════════════════════════════════════════════════════════════════
class TestDiInputsNeverLeaks:
    """``_neo_di_inputs`` (``StateKeys.DI_INPUTS``) is a config-only side-channel
    that carries resolved ``FromInput``/``FromConfig`` values to an LLM-mode
    node's prompt compiler — it NEVER enters state (mirrors ``STREAM_CUSTOM``).
    Locked by name here: a run whose think node HAS di_inputs injected must still
    surface neither in the ``run()`` result dict nor in any streamed chunk."""

    def _think_pipeline(self):
        from typing import Annotated

        from neograph import FromInput

        mod = _types.ModuleType("test_streaming_dileak_mod")

        @node(outputs=Claims, mode="think", model="fast", prompt="tmpl")
        def analyze(domain: Annotated[str, FromInput]) -> Claims: ...

        mod.analyze = analyze
        return construct_from_module(mod, name="streaming-dileak")

    def test_di_inputs_absent_from_run_result_and_stream(self):
        from neograph import stream
        from tests.fakes import StructuredFake, build_fake_llm_kwargs

        llm_kw = build_fake_llm_kwargs(lambda tier: StructuredFake(lambda m: m(items=["ok"])))
        cfg = {"configurable": {"domain": "oncology"}}

        graph = compile(self._think_pipeline(), **llm_kw, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "d-001"}, config=dict(cfg))
        assert StateKeys.DI_INPUTS not in result

        graph2 = compile(self._think_pipeline(), **llm_kw, **build_test_compile_kwargs())
        chunks = list(
            stream(graph2, input={"node_id": "d-002"}, config=dict(cfg), stream_mode="values")
        )
        assert chunks, "stream produced no chunks"
        for ch in chunks:
            assert StateKeys.DI_INPUTS not in ch


# ═══════════════════════════════════════════════════════════════════════════
# Facade delegations completed for the claimed capability set (three-layer §2.2)
# ═══════════════════════════════════════════════════════════════════════════
class TestFacadeDelegations:
    """``stream`` (sync), ``astream_events``, ``update_state``/``aupdate_state``
    fell off the facade allowlist by accident (violation E). Pin that each is
    now reachable THROUGH the facade (single-seam) and actually delegates."""

    def test_facade_exposes_new_methods(self):
        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        for name in ("stream", "astream_events", "update_state", "aupdate_state"):
            assert hasattr(graph, name), f"facade missing {name}"

    def test_facade_sync_stream_delegates(self):
        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        chunks = list(graph.stream({"node_id": "f-001"}, stream_mode="updates"))
        # raw facade delegation is un-finalized; the node deltas are present.
        node_names = {n for ch in chunks for n in ch}
        assert "process" in node_names

    async def test_facade_astream_events_delegates(self):
        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        events = [
            ev
            async for ev in graph.astream_events({"node_id": "f-002"}, version="v2")
        ]
        assert any(ev.get("event") == "on_chain_start" for ev in events)

    def test_facade_update_state_delegates(self):
        from langgraph.checkpoint.memory import InMemorySaver

        graph = compile(
            _trivial_pipeline(), checkpointer=InMemorySaver(), **build_test_compile_kwargs()
        )
        config = {"configurable": {"thread_id": "f-003"}}
        run(graph, input={"node_id": "f-003"}, config=config)
        # update_state through the facade returns a config referencing a new checkpoint.
        new_config = graph.update_state(config, {"fetch": RawText(text="patched")})
        assert new_config["configurable"]["thread_id"] == "f-003"

    async def test_facade_aupdate_state_delegates(self):
        from langgraph.checkpoint.memory import InMemorySaver

        graph = compile(
            _trivial_pipeline(), checkpointer=InMemorySaver(), **build_test_compile_kwargs()
        )
        config = {"configurable": {"thread_id": "f-004"}}
        run(graph, input={"node_id": "f-004"}, config=config)
        new_config = await graph.aupdate_state(config, {"fetch": RawText(text="patched")})
        assert new_config["configurable"]["thread_id"] == "f-004"
