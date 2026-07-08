"""``emit_progress`` — typed custom progress/domain events from nodes (q8ec).

TDD red-first for the emit surface (sequencing doc §6.2). Contract:

  * A node emits a typed event via ``emit_progress(model)``; under a streaming
    driver consuming ``custom`` it surfaces in the stream with a stable envelope.
  * Under a NON-streaming driver (``run``/``arun``, or ``stream`` without
    ``custom``) it must NOT vanish silently — it warns once (review L1), it does
    NOT raise, and the pipeline result is unaffected.
  * Custom payloads never enter state and never touch the schema fingerprint.
"""

from __future__ import annotations

import types as _types
import warnings

import pytest
from pydantic import BaseModel

from neograph import compile, construct_from_module, emit_progress, node, run, stream
from neograph._state_keys import StateKeys
from neograph.errors import ConfigurationError
from tests.fakes import build_test_compile_kwargs
from tests.schemas import Claims, RawText


class Milestone(BaseModel):
    stage: str
    detail: str = ""


def _emitting_pipeline(n_emits: int = 1):
    """fetch -> worker, where worker emits ``n_emits`` progress milestones."""
    mod = _types.ModuleType(f"test_emit_progress_mod_{n_emits}")

    @node(mode="scripted", outputs=RawText)
    def fetch() -> RawText:
        return RawText(text="hello")

    @node(mode="scripted", outputs=Claims)
    def worker(fetch: RawText) -> Claims:
        for i in range(n_emits):
            emit_progress(Milestone(stage=f"step-{i}", detail=fetch.text))
        return Claims(items=[fetch.text.upper()])

    mod.fetch = fetch
    mod.worker = worker
    return construct_from_module(mod, name=f"emit-{n_emits}")


def _reset_warn_state():
    from neograph import progress

    progress._reset_warned()


# ═══════════════════════════════════════════════════════════════════════════
# Emitted events surface via the custom stream
# ═══════════════════════════════════════════════════════════════════════════
class TestEmitSurfacesInStream:
    def test_emitted_milestone_appears_with_typed_envelope(self):
        _reset_warn_state()
        graph = compile(_emitting_pipeline(1), **build_test_compile_kwargs())

        customs = list(stream(graph, input={"node_id": "e-001"}, stream_mode="custom"))

        assert customs, "no custom events surfaced"
        evt = customs[0]
        assert evt == {
            "neograph_event": "progress",
            "event_type": "Milestone",
            "data": {"stage": "step-0", "detail": "hello"},
        }

    def test_all_emitted_events_surface_in_order(self):
        _reset_warn_state()
        graph = compile(_emitting_pipeline(3), **build_test_compile_kwargs())

        customs = list(stream(graph, input={"node_id": "e-002"}, stream_mode="custom"))

        stages = [c["data"]["stage"] for c in customs]
        assert stages == ["step-0", "step-1", "step-2"]

    def test_emit_does_not_warn_under_custom_stream(self):
        _reset_warn_state()
        graph = compile(_emitting_pipeline(1), **build_test_compile_kwargs())

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any UserWarning becomes an error
            list(stream(graph, input={"node_id": "e-003"}, stream_mode="custom"))


# ═══════════════════════════════════════════════════════════════════════════
# Non-streaming driver — warn, don't vanish (review L1)
# ═══════════════════════════════════════════════════════════════════════════
class TestEmitUnderNonStreamingDriver:
    def test_run_warns_once_and_still_completes(self):
        _reset_warn_state()
        graph = compile(_emitting_pipeline(3), **build_test_compile_kwargs())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run(graph, input={"node_id": "e-004"})

        # Pipeline unaffected...
        assert result["worker"] == Claims(items=["HELLO"])
        # ...and exactly ONE progress warning despite 3 emits (dedup per run).
        progress_warnings = [
            w for w in caught if issubclass(w.category, UserWarning) and "progress" in str(w.message).lower()
        ]
        assert len(progress_warnings) == 1

    def test_stream_without_custom_mode_warns(self):
        _reset_warn_state()
        graph = compile(_emitting_pipeline(1), **build_test_compile_kwargs())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            list(stream(graph, input={"node_id": "e-005"}, stream_mode="values"))

        progress_warnings = [
            w for w in caught if issubclass(w.category, UserWarning) and "progress" in str(w.message).lower()
        ]
        assert len(progress_warnings) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Typed contract — a non-BaseModel event is a clear configuration error
# ═══════════════════════════════════════════════════════════════════════════
class TestEmitRejectsUntypedEvent:
    def test_non_basemodel_event_raises_configuration_error(self):
        _reset_warn_state()
        with pytest.raises(ConfigurationError, match="requires a Pydantic BaseModel"):
            emit_progress({"stage": "step-0"})  # a dict is not a typed event

    def test_error_names_the_offending_type(self):
        _reset_warn_state()
        with pytest.raises(ConfigurationError, match="dict"):
            emit_progress({"stage": "step-0"})


# ═══════════════════════════════════════════════════════════════════════════
# Custom events never enter state / fingerprint (checkpoint safety, review L2)
# ═══════════════════════════════════════════════════════════════════════════
class TestEmitDoesNotTouchState:
    def test_emitted_payload_not_in_pipeline_result(self):
        _reset_warn_state()
        graph = compile(_emitting_pipeline(2), **build_test_compile_kwargs())
        with warnings.catch_warnings():  # the L1 non-stream warning is not under test here
            warnings.simplefilter("ignore")
            result = run(graph, input={"node_id": "e-006"})

        # The milestone data must not have leaked into any state field.
        for value in result.values():
            assert not (isinstance(value, dict) and value.get("neograph_event") == "progress")
        assert "neograph_event" not in result

    def test_schema_fingerprint_identical_with_and_without_emit(self):
        _reset_warn_state()
        emitting = compile(_emitting_pipeline(1), **build_test_compile_kwargs())

        # A structurally-identical pipeline that does NOT emit.
        mod = _types.ModuleType("test_emit_progress_noemit")

        @node(mode="scripted", outputs=RawText)
        def fetch() -> RawText:
            return RawText(text="hello")

        @node(mode="scripted", outputs=Claims)
        def worker(fetch: RawText) -> Claims:
            return Claims(items=[fetch.text.upper()])

        mod.fetch = fetch
        mod.worker = worker
        non_emitting = compile(construct_from_module(mod, name="emit-1"), **build_test_compile_kwargs())

        # emit_progress is a stream-side effect only — the type-based fingerprint
        # (state fields, neo_* excluded) is identical.
        assert emitting.schema_fingerprint == non_emitting.schema_fingerprint

    def test_emit_output_is_not_a_neo_state_key(self):
        # The envelope key is deliberately NOT neo_*-prefixed — it is a user
        # payload, and _finalize_chunk must never strip it (asserted in
        # test_streaming). This pins that the envelope is not framework state.
        assert not "neograph_event".startswith(StateKeys.FRAMEWORK_PREFIX)
