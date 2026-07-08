"""Unit pins for ``neograph._config_carrier`` — the single-sited helpers that
read/write ``config['configurable']``.

``run_id_of`` is the canonical reader for the framework-minted per-run id. Its
``if not config`` / ``config.get('configurable') or {}`` guards make it total
over the degenerate configs a caller can pass (graph invoked directly, an
explicitly-``None`` configurable, an empty dict). Those guards had zero test
references (Wave-9 verification, neograph-yc38) — pinned here so a refactor that
drops the ``or {}`` fallback fails loudly instead of raising ``AttributeError``
deep in the per-run cache.
"""

from __future__ import annotations

from neograph._config_carrier import run_id_of
from neograph._state_keys import StateKeys


class TestRunIdOfDefensiveForms:
    """``run_id_of`` returns None for every shape that lacks a minted run id,
    never raising — the per-run cache and run-id log binding treat None as
    "no run scope"."""

    def test_none_config_returns_none(self):
        assert run_id_of(None) is None

    def test_empty_config_returns_none(self):
        assert run_id_of({}) is None

    def test_configurable_is_none_returns_none(self):
        # The ``or {}`` guard turns a None configurable into an empty lookup.
        assert run_id_of({"configurable": None}) is None

    def test_empty_configurable_returns_none(self):
        assert run_id_of({"configurable": {}}) is None

    def test_minted_run_id_is_read_back(self):
        cfg = {"configurable": {StateKeys.RUN_ID: "run-abc"}}
        assert run_id_of(cfg) == "run-abc"
