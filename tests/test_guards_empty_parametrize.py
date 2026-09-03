"""An empty ``@pytest.mark.parametrize`` must be a COLLECTION ERROR, never a skip.

neograph-e8wiv. ``tests/test_agent_spec_export.py`` carried two tests
parametrized over ``FALLTHROUGH_COMBOS``, a set that is empty BECAUSE every
fusion combo gained a real Agent Spec lowering -- the emptiness was the success
condition. pytest's default renders an empty parameter set as
``SKIPPED ... got empty parameter set for (combo)``, and ``scripts/check_skips.py``
-- correctly, since its allowlist was deleted -- cannot tell "empty because we
succeeded" from "did not run because something broke". Two green-by-accident
skips sat in the gate until someone read the reason text.

The ban is enforced by pytest itself: ``empty_parameter_set_mark = "fail_at_collect"``
in ``pyproject.toml``'s ``[tool.pytest.ini_options]``. That is the single
authority -- this file does NOT re-implement the detection (an AST scan cannot
see it anyway: the offending argvalues was a NAME imported from another module,
empty only after evaluation). What this file does is prove the setting is live
and keyed on real emptiness, so it cannot be dropped in a config tidy-up without
turning something red.

Meta-tests run a synthetic module through a pytest SUBPROCESS pointed at the
repo's own ``pyproject.toml`` (``-c``), which is what makes them a test of the
shipped configuration rather than of a fixture.

Not regex-based -- the mechanism is pytest's own ``get_empty_parameterset_mark``
config branch, so there is no pattern to slip past. The analogous slip case is a
FALSE POSITIVE rather than a false negative, and it is pinned: pytest gives an
empty parameter set the literal id ``NOTSET``, so a naive guard matching test ids
for ``[NOTSET]`` would also condemn ``parametrize("x", ["NOTSET"])`` -- a real,
non-empty, perfectly legal parametrize. ``test_a_literal_notset_value_is_not_a_false_positive``
feeds exactly that and asserts it still collects.

The rule for a genuinely-empty-by-construction set: assert the emptiness
(``assert derived == ()``), which also makes it exercisable against a simulated
non-empty input. See
``test_agent_spec_export.py::TestUnsupportedComboFallthroughRaise``.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tomllib

REPO = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = REPO / "pyproject.toml"


def _collect(tmp_path: pathlib.Path, source: str) -> subprocess.CompletedProcess[str]:
    """Collect a synthetic one-module suite under the REPO's pytest config."""
    module = tmp_path / "test_synthetic_probe.py"
    module.write_text(source)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(module),
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-c",
            str(PYPROJECT),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


class TestEmptyParametrizeIsACollectionError:
    """The setting exists, is spelled correctly, and actually fires."""

    def test_the_setting_is_declared_in_pyproject(self) -> None:
        """Pin the literal value: 'skip' (the default) and 'xfail' are the two
        other legal values and BOTH would silently restore the disease -- 'skip'
        by definition, 'xfail' by making a non-running test look like a tracked
        known-failure."""
        ini = tomllib.loads(PYPROJECT.read_text())["tool"]["pytest"]["ini_options"]

        assert ini.get("empty_parameter_set_mark") == "fail_at_collect", (
            "empty_parameter_set_mark must be 'fail_at_collect' -- an empty parametrize "
            "otherwise becomes a SKIP, which scripts/check_skips.py cannot distinguish "
            "from a test that failed to run (neograph-e8wiv)"
        )

    def test_an_empty_parametrize_fails_collection(self, tmp_path: pathlib.Path) -> None:
        """POSITIVE meta-test: the disease, in the shape it actually shipped in --
        argvalues is a NAME that evaluates to empty, not an inline ``[]``."""
        result = _collect(
            tmp_path,
            "import pytest\n"
            "DERIVED = ()\n"
            "@pytest.mark.parametrize('combo', DERIVED)\n"
            "def test_over_an_empty_set(combo):\n"
            "    pass\n",
        )

        assert result.returncode != 0, result.stdout + result.stderr
        assert "Empty parameter set in 'test_over_an_empty_set'" in result.stdout + result.stderr, (
            result.stdout + result.stderr
        )
        assert "got empty parameter set" not in result.stdout + result.stderr, (
            "the skip-flavoured reason came back -- empty_parameter_set_mark reverted to 'skip'"
        )

    def test_a_populated_parametrize_still_collects(self, tmp_path: pathlib.Path) -> None:
        """NEGATIVE meta-test: the setting must not condemn ordinary parametrize."""
        result = _collect(
            tmp_path,
            "import pytest\n"
            "DERIVED = (1, 2)\n"
            "@pytest.mark.parametrize('combo', DERIVED)\n"
            "def test_over_a_real_set(combo):\n"
            "    pass\n",
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "2 tests collected" in result.stdout, result.stdout

    def test_a_literal_notset_value_is_not_a_false_positive(self, tmp_path: pathlib.Path) -> None:
        """SLIP meta-test. pytest ids an empty parameter set as ``NOTSET``, so a
        guard that matched collected ids for ``[NOTSET]`` -- the obvious way to
        write this check without the ini option -- would also condemn this
        module, which parametrizes over one real value that happens to be the
        string 'NOTSET'. Keying on pytest's own emptiness branch instead means
        the false positive is structurally impossible."""
        result = _collect(
            tmp_path,
            "import pytest\n"
            "@pytest.mark.parametrize('name', ['NOTSET'])\n"
            "def test_over_a_value_called_notset(name):\n"
            "    assert name == 'NOTSET'\n",
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 test collected" in result.stdout, result.stdout
