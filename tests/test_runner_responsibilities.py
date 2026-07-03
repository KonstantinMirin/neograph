"""Guard-first pins for the four weakly-pinned runner responsibilities (q8ec).

Written BEFORE the ``_prepare``/``_aprepare`` extraction (three-layer-principle
doc §2.3) so the refactor cannot silently drop or reorder a pre-engine step.
Each responsibility below was either entirely unpinned (the defensive input
copy) or only transitively pinned (CONFIG_INPUT stash/re-inject, fingerprint
injection ordering, preflight-DI ordering). These tests PASS on today's source
and MUST keep passing after the collapse into ``_prepare``/``_aprepare``.

They intentionally pin at two levels:
  * behavioral (through ``run``/``arun``) for the observable no-leak and
    fail-before-invoke contracts, which survive any internal restructuring; and
  * helper-level (``_prepare_new_input``/``_prepare_resume_config``) for the
    stash/re-inject round-trip and the inject-into-engine-input-not-caller-dict
    ordering — pinning the shared pure helpers the refactor must keep.
"""

from __future__ import annotations

import asyncio
import types as _types
from typing import Annotated

import pytest

from neograph import FromInput, compile, construct_from_module, node, run
from neograph._state_keys import StateKeys
from neograph.errors import ExecutionError
from neograph.runner import _prepare_new_input, _prepare_resume_config
from tests.fakes import build_test_compile_kwargs
from tests.schemas import Claims, RawText


def _trivial_pipeline():
    """fetch -> process, no LLM, no DI. Compiles with real fingerprints."""
    mod = _types.ModuleType("test_runner_resp_trivial_mod")

    @node(mode="scripted", outputs=RawText)
    def fetch() -> RawText:
        return RawText(text="hello")

    @node(mode="scripted", outputs=Claims)
    def process(fetch: RawText) -> Claims:
        return Claims(items=[fetch.text.upper()])

    mod.fetch = fetch
    mod.process = process
    return construct_from_module(mod, name="runner-resp-trivial")


# ═══════════════════════════════════════════════════════════════════════════
# Responsibility 1 — defensive input copy (was ENTIRELY unpinned)
# ═══════════════════════════════════════════════════════════════════════════
class TestDefensiveInputCopy:
    """The caller's ``input`` dict must never gain framework ``neo_*`` keys.

    An in-place mutation of ``input`` would leak the schema/node fingerprints
    back into the caller's dict — silently, with zero prior test failure.
    """

    def test_run_does_not_mutate_caller_input_dict(self):
        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        caller_input = {"node_id": "resp-001"}

        run(graph, input=caller_input)

        assert StateKeys.SCHEMA_FINGERPRINT not in caller_input
        assert StateKeys.NODE_FINGERPRINTS not in caller_input
        assert caller_input == {"node_id": "resp-001"}

    def test_arun_does_not_mutate_caller_input_dict(self):
        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        caller_input = {"node_id": "resp-002"}

        asyncio.run(_arun_new(graph, caller_input))

        assert StateKeys.SCHEMA_FINGERPRINT not in caller_input
        assert StateKeys.NODE_FINGERPRINTS not in caller_input
        assert caller_input == {"node_id": "resp-002"}


async def _arun_new(graph, caller_input):
    import neograph

    return await neograph.arun(graph, input=caller_input)


# ═══════════════════════════════════════════════════════════════════════════
# Responsibility 2 — CONFIG_INPUT stash / re-inject round-trip
# ═══════════════════════════════════════════════════════════════════════════
class TestConfigInputRoundTrip:
    """New-input stashes the run input under ``CONFIG_INPUT`` in config; resume
    re-injects it into ``configurable`` so post-interrupt FromInput DI resolves.
    """

    def test_new_input_stashes_config_input_in_configurable(self):
        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        original = {"node_id": "resp-003", "topic": "climate"}

        _, config = _prepare_new_input(graph, original, None)

        stashed = config["configurable"][StateKeys.CONFIG_INPUT]
        assert stashed == {"node_id": "resp-003", "topic": "climate"}

    def test_resume_reinjects_stashed_input_into_configurable(self):
        config = {"configurable": {StateKeys.CONFIG_INPUT: {"topic": "climate"}}}

        reinjected = _prepare_resume_config(config)

        assert reinjected["configurable"]["topic"] == "climate"

    def test_stash_then_reinject_round_trips_the_input(self):
        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        original = {"node_id": "resp-004", "topic": "energy"}

        _, after_new = _prepare_new_input(graph, original, None)
        # Simulate a resume that only carries the config forward (input is gone).
        resume_config = {"configurable": dict(after_new["configurable"])}
        reinjected = _prepare_resume_config(resume_config)

        assert reinjected["configurable"]["topic"] == "energy"
        assert reinjected["configurable"]["node_id"] == "resp-004"


# ═══════════════════════════════════════════════════════════════════════════
# Responsibility 3 — fingerprint injection targets the engine input, not caller
# ═══════════════════════════════════════════════════════════════════════════
class TestFingerprintInjectionOrdering:
    """Fingerprints land on the ENGINE input dict (what graph.invoke receives),
    never on the caller's dict — the copy must happen before injection."""

    def test_fingerprints_go_to_engine_input_not_caller_dict(self):
        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        caller_input = {"node_id": "resp-005"}

        engine_input, _ = _prepare_new_input(graph, caller_input, None)

        # Engine input carries both fingerprints...
        assert engine_input[StateKeys.SCHEMA_FINGERPRINT] == graph.schema_fingerprint
        assert engine_input[StateKeys.NODE_FINGERPRINTS] == graph.node_fingerprints
        # ...and it is a DISTINCT object from the caller's dict, which stays clean.
        assert engine_input is not caller_input
        assert StateKeys.SCHEMA_FINGERPRINT not in caller_input
        assert StateKeys.NODE_FINGERPRINTS not in caller_input


# ═══════════════════════════════════════════════════════════════════════════
# Responsibility 4 — preflight DI runs BEFORE any node executes
# ═══════════════════════════════════════════════════════════════════════════
class TestPreflightDiOrdering:
    """A missing required DI param fails at the gate — no node body runs."""

    def _di_pipeline(self, executed: list[str]):
        mod = _types.ModuleType("test_runner_resp_di_mod")

        @node(mode="scripted", outputs=Claims)
        def greet(topic: Annotated[str, FromInput]) -> Claims:
            executed.append("greet")
            return Claims(items=[topic])

        mod.greet = greet
        return construct_from_module(mod, name="runner-resp-di")

    def test_run_raises_before_any_node_when_required_di_missing(self):
        executed: list[str] = []
        graph = compile(self._di_pipeline(executed), **build_test_compile_kwargs())

        with pytest.raises(ExecutionError, match="Required DI parameters not provided"):
            run(graph, input={"node_id": "resp-006"})  # 'topic' absent

        assert executed == []  # preflight fired BEFORE graph.invoke reached greet

    def test_arun_raises_before_any_node_when_required_di_missing(self):
        executed: list[str] = []
        graph = compile(self._di_pipeline(executed), **build_test_compile_kwargs())

        async def _go():
            import neograph

            await neograph.arun(graph, input={"node_id": "resp-007"})

        with pytest.raises(ExecutionError, match="Required DI parameters not provided"):
            asyncio.run(_go())

        assert executed == []
