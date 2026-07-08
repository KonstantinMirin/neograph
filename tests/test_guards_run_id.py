"""Structural guard — the per-run id is minted symmetrically in BOTH pre-engine
brains (neograph-puip).

``_mint_run_id`` mirrors ``_mark_stream_custom``: it must be called in BOTH
``_prepare`` and ``_aprepare`` so the sync and async drivers cannot fork on run
identity. If a future edit adds the mint to one brain but not its twin, resume
correctness (the two-lifetime invariant) would silently hold on one driver and
break on the other. This AST guard pins the symmetry.
"""

from __future__ import annotations

import ast
import inspect

import neograph.runner as runner


def _calls_in_function(func_name: str) -> set[str]:
    """Return the set of function names called (by bare Name) inside ``func_name``
    as defined in runner.py."""
    source = inspect.getsource(runner)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return {
                call.func.id
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            }
    raise AssertionError(f"{func_name} not found in runner.py")


class TestRunIdMintTwinSymmetry:
    def test_prepare_mints_run_id(self):
        assert "_mint_run_id" in _calls_in_function("_prepare")

    def test_aprepare_mints_run_id(self):
        assert "_mint_run_id" in _calls_in_function("_aprepare")

    def test_both_prepare_twins_mint_symmetrically(self):
        """Neither brain may mint without the other (the twin invariant)."""
        sync = "_mint_run_id" in _calls_in_function("_prepare")
        asyncf = "_mint_run_id" in _calls_in_function("_aprepare")
        assert sync == asyncf is True, (
            f"_mint_run_id must be called in BOTH _prepare and _aprepare — _prepare={sync}, _aprepare={asyncf}"
        )
