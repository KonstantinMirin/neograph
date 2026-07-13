"""__main__.py dev-mode traceback regression test for neograph-2yi7q(c).

Under NEOGRAPH_DEV=1, cmd_test_scaffold's exception handler prints a traceback
before the FAIL line; under default mode it does not. Verified by forcing
scaffold_tests to raise (deterministic) and toggling the DEV_MODE binding on
neograph.__main__.
"""

from __future__ import annotations

import argparse

import pytest

import neograph.__main__ as mainmod


def _force_fail_args():
    return argparse.Namespace(target="ignored", output=None, overwrite=False)


@pytest.fixture
def _fail_scaffold(monkeypatch):
    """Force cmd_test_scaffold onto its `except Exception` path deterministically."""

    class _Dummy:
        name = "dummy"

    def _boom(*a, **kw):
        raise RuntimeError("forced scaffold failure")

    monkeypatch.setattr(mainmod, "_import_module", lambda target: object())
    monkeypatch.setattr(mainmod, "_discover_constructs", lambda mod: [("c", _Dummy())])
    monkeypatch.setattr("neograph.testing.scaffold_tests", _boom)
    yield


class TestDevModeTraceback:
    def test_traceback_printed_when_dev_mode_enabled(self, monkeypatch, capsys, _fail_scaffold):
        monkeypatch.setattr(mainmod, "DEV_MODE", True)
        rc = mainmod.cmd_test_scaffold(_force_fail_args())
        captured = capsys.readouterr()
        assert rc == 1
        out = captured.out + captured.err
        assert "Traceback" in out, f"expected traceback under DEV_MODE, got:\n{out}"
        assert "FAIL" in captured.out

    def test_no_traceback_when_dev_mode_disabled(self, monkeypatch, capsys, _fail_scaffold):
        monkeypatch.setattr(mainmod, "DEV_MODE", False)
        rc = mainmod.cmd_test_scaffold(_force_fail_args())
        captured = capsys.readouterr()
        assert rc == 1
        out = captured.out + captured.err
        assert "Traceback" not in out, f"no traceback expected in default mode, got:\n{out}"
        assert "FAIL" in captured.out

    def test_dev_mode_and_traceback_imported(self):
        """The feature's dependencies are wired into __main__."""
        import traceback as _tb

        assert hasattr(mainmod, "DEV_MODE") is True
        assert mainmod.traceback is _tb
