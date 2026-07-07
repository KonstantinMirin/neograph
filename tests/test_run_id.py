"""Per-run id primitive — ``StateKeys.RUN_ID`` (``_neo_run_id``), neograph-puip.

A framework-minted, config-only per-run id. Minted fresh per execution attempt in
``_prepare`` / ``_aprepare`` (the single pre-engine brains), stable across every
superstep of a run, NEVER entering state (so it cannot touch the schema
fingerprint or persist in checkpoints), and NOT user-supplied.

Two-lifetime-correct BY CONSTRUCTION: resume re-runs ``_prepare`` -> a NEW id, so
resume invalidation is free. The load-bearing test is
``test_run_id_is_reminted_on_resume`` — same ``thread_id``, different
``_neo_run_id`` across the interrupt/resume boundary, driven through a real
file-backed ``SqliteSaver`` per project convention.
"""

from __future__ import annotations

import asyncio

from langgraph.checkpoint.sqlite import SqliteSaver

import neograph
from neograph import Construct, Node, Operator, compile, run
from neograph._state_keys import StateKeys
from neograph.runner import _mint_run_id, stream
from tests.fakes import build_test_compile_kwargs, register_condition, register_scripted
from tests.schemas import Claims, ValidationResult


def _recorder(fn_name: str, seen: list, output):
    """Register a scripted fn that records the config's RUN_ID into ``seen``."""

    def _fn(_in, cfg):
        configurable = (cfg or {}).get("configurable", {})
        seen.append(configurable.get(StateKeys.RUN_ID))
        return output

    register_scripted(fn_name, _fn)
    return fn_name


class TestRunIdPresence:
    def test_run_id_present_in_node_when_run(self):
        """(1) A node sees a non-empty ``_neo_run_id`` in config['configurable']."""
        seen: list = []
        node = Node.scripted(
            "probe", fn=_recorder("probe_fn", seen, Claims(items=["x"])), outputs=Claims
        )
        graph = compile(Construct("p", nodes=[node]), **build_test_compile_kwargs())
        run(graph, input={})
        assert len(seen) == 1
        assert isinstance(seen[0], str) and len(seen[0]) == 32  # uuid4().hex

    def test_run_id_stable_across_supersteps_when_multi_node(self):
        """(2) Every superstep of ONE run observes the SAME id."""
        seen: list = []
        n1 = Node.scripted(
            "n1", fn=_recorder("n1_fn", seen, Claims(items=["a"])), outputs=Claims
        )
        n2 = Node.scripted(
            "n2",
            fn=_recorder("n2_fn", seen, ValidationResult(passed=True, issues=[])),
            inputs=Claims,
            outputs=ValidationResult,
        )
        graph = compile(Construct("p", nodes=[n1, n2]), **build_test_compile_kwargs())
        run(graph, input={})
        assert len(seen) == 2
        assert seen[0] == seen[1]
        assert seen[0] is not None

    def test_run_id_differs_across_two_run_calls(self):
        """(3) Two independent run() calls mint DIFFERENT ids."""
        seen: list = []
        node = Node.scripted(
            "probe", fn=_recorder("probe_fn", seen, Claims(items=["x"])), outputs=Claims
        )
        graph = compile(Construct("p", nodes=[node]), **build_test_compile_kwargs())
        run(graph, input={})
        run(graph, input={})
        assert len(seen) == 2
        assert seen[0] != seen[1]


class TestRunIdTwoLifetime:
    def test_run_id_is_reminted_on_resume(self, tmp_path):
        """(4) THE two-lifetime assertion. Same thread_id, a fresh id after resume.

        n1 records its id then interrupts (Operator); on resume, n2 records the
        id of the SECOND execution attempt. Because resume re-enters _prepare, the
        id is re-minted — different from the pre-interrupt id — with NO change to
        thread_id. Real file-backed SqliteSaver per project convention.
        """
        seen: list = []
        register_condition(
            "puip_gate",
            lambda state: {"needs": "review"} if getattr(state, "n1", None) else None,
        )
        n1 = (
            Node.scripted(
                "n1", fn=_recorder("n1_fn", seen, Claims(items=["a"])), outputs=Claims
            )
            | Operator(when="puip_gate")
        )
        n2 = Node.scripted(
            "n2",
            fn=_recorder("n2_fn", seen, ValidationResult(passed=True, issues=[])),
            inputs=Claims,
            outputs=ValidationResult,
        )
        construct = Construct("p", nodes=[n1, n2])

        db = str(tmp_path / "puip-resume.db")
        config = {"configurable": {"thread_id": "puip-resume"}}
        with SqliteSaver.from_conn_string(db) as saver:
            graph = compile(construct, checkpointer=saver, **build_test_compile_kwargs())
            paused = run(graph, input={}, config=config)
            assert "__interrupt__" in paused
            resumed = run(graph, resume={"approved": True}, config=config)
            assert "__interrupt__" not in resumed

        # seen[0] = lifetime-1 id (n1, pre-interrupt); seen[-1] = lifetime-2 id (n2).
        assert len(seen) == 2
        first_life, second_life = seen[0], seen[-1]
        assert first_life is not None and second_life is not None
        assert first_life != second_life, "run_id must be re-minted on resume"


class TestMintRunIdHelper:
    def test_mint_run_id_returns_fresh_dict_without_mutating_caller(self):
        """(5a) Mirrors _mark_stream_custom: fresh dict, caller untouched."""
        caller = {"configurable": {"thread_id": "t"}}
        out = _mint_run_id(caller)
        assert out is not caller
        assert out["configurable"] is not caller["configurable"]
        # Caller dict is NOT mutated — no RUN_ID leaked back in.
        assert StateKeys.RUN_ID not in caller["configurable"]
        assert out["configurable"][StateKeys.RUN_ID] is not None
        # Pre-existing keys are preserved.
        assert out["configurable"]["thread_id"] == "t"

    def test_mint_run_id_mints_a_distinct_id_each_call(self):
        """(5b) Two mints off the same caller dict yield different ids."""
        caller = {"configurable": {}}
        a = _mint_run_id(caller)
        b = _mint_run_id(caller)
        assert a["configurable"][StateKeys.RUN_ID] != b["configurable"][StateKeys.RUN_ID]

    def test_mint_run_id_handles_none_config(self):
        """None config collapses to a fresh dict carrying the id."""
        out = _mint_run_id(None)
        assert out["configurable"][StateKeys.RUN_ID] is not None

    def test_parallel_arun_share_config_get_distinct_ids(self):
        """(5c) Two arun() calls sharing ONE config dict each mint their own id;
        the shared caller dict is never mutated."""
        seen: list = []
        node = Node.scripted(
            "probe", fn=_recorder("probe_fn", seen, Claims(items=["x"])), outputs=Claims
        )
        graph = compile(Construct("p", nodes=[node]), **build_test_compile_kwargs())
        shared = {"configurable": {}}

        async def _drive():
            await asyncio.gather(
                neograph.arun(graph, input={}, config=shared),
                neograph.arun(graph, input={}, config=shared),
            )

        asyncio.run(_drive())
        assert len(seen) == 2
        assert seen[0] != seen[1]
        assert StateKeys.RUN_ID not in shared["configurable"]


class TestRunIdNeverPersists:
    def test_run_id_absent_from_returned_state(self):
        """(6a) The id never surfaces in the returned state dict."""
        node = Node.scripted(
            "probe", fn=_recorder("probe_fn", [], Claims(items=["x"])), outputs=Claims
        )
        graph = compile(Construct("p", nodes=[node]), **build_test_compile_kwargs())
        result = run(graph, input={})
        assert StateKeys.RUN_ID not in result
        assert not any(k.startswith("_neo_") for k in result)

    def test_run_id_absent_from_stream_chunks(self):
        """(6b) The id never leaks into stream_mode='values' chunks."""
        node = Node.scripted(
            "probe", fn=_recorder("probe_fn", [], Claims(items=["x"])), outputs=Claims
        )
        graph = compile(Construct("p", nodes=[node]), **build_test_compile_kwargs())
        for chunk in stream(graph, input={}, stream_mode="values"):
            assert StateKeys.RUN_ID not in chunk

    def test_run_id_absent_from_checkpoint(self, tmp_path):
        """(6c) The id never persists in the checkpoint channel_values."""
        node = Node.scripted(
            "probe", fn=_recorder("probe_fn", [], Claims(items=["x"])), outputs=Claims
        )
        db = str(tmp_path / "puip-checkpoint.db")
        config = {"configurable": {"thread_id": "puip-cp"}}
        with SqliteSaver.from_conn_string(db) as saver:
            graph = compile(
                Construct("p", nodes=[node]), checkpointer=saver, **build_test_compile_kwargs()
            )
            run(graph, input={}, config=config)
            saved = saver.get_tuple(config)
        channel_values = saved.checkpoint.get("channel_values", {})
        assert StateKeys.RUN_ID not in channel_values
