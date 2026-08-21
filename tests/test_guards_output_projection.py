"""Structural guard: the schema shown to the LLM and the schema that actually
validates the parsed response must never diverge again (neograph-ftnxl.4).

The disease this ticket fixed: ``ExcludeFromOutput``'s only reader used to be
``describe_type()`` (the rendered prompt text) -- ``with_structured_output``
received the DECLARED model unprojected, so an excluded field with no default
was demanded of the provider as ``required`` under the default
``output_strategy="structured"``. The fix is ONE projection
(``project_output_model``) applied before BOTH call sites inside
``invoke_structured``/``ainvoke_structured``. This guard pins that shape so a
future refactor cannot silently re-split the two schemas.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"
LLM_MODULE = SRC_DIR / "_llm.py"
DESCRIBE_TYPE_MODULE = SRC_DIR / "describe_type.py"
DESCRIBE_COUNTING_MODULE = SRC_DIR / "_describe_counting.py"


def _project_output_model_call_count(tree: ast.Module, func_name: str) -> int:
    """Count `project_output_model(...)` calls inside the named top-level function."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return sum(
                1
                for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "project_output_model"
            )
    raise AssertionError(f"function {func_name!r} not found in {LLM_MODULE.name} via AST")


class TestBothStructuredCallTwinsProjectBeforeCalling:
    """Positive: `invoke_structured`/`ainvoke_structured` each call
    `project_output_model` exactly once, rebinding the local `output_model`
    that BOTH `_prepare_structured_call` and `_call_structured` read --
    projecting inside `_prepare_structured_call` instead would fix neither
    call site (its 4-tuple return doesn't carry `output_model`)."""

    def test_invoke_structured_projects(self):
        tree = ast.parse(LLM_MODULE.read_text())
        assert _project_output_model_call_count(tree, "invoke_structured") == 1

    def test_ainvoke_structured_projects(self):
        tree = ast.parse(LLM_MODULE.read_text())
        assert _project_output_model_call_count(tree, "ainvoke_structured") == 1

    def test_prepare_structured_call_itself_does_not_project(self):
        """Would-be-missed case: a regression that moves the projection INTO
        `_prepare_structured_call` (which looks equally plausible at a
        glance) must still be caught -- that function's 4-tuple return
        doesn't carry `output_model` back to the caller, so `_call_structured`
        would see the UNPROJECTED model again."""
        tree = ast.parse(LLM_MODULE.read_text())
        assert _project_output_model_call_count(tree, "_prepare_structured_call") == 0


class TestDescribeTypeUsesTheSharedPredicate:
    """Positive + negative: both internal strip sites call the SHARED
    `output_markers` predicate (covers `Carried` too, not just
    `ExcludeFromOutput`) -- never a re-derived, local marker check.

    The two sites are the renderer's two PASSES, which is why there must be
    exactly two and why they must agree: pass 1 counts which nested classes
    appear, pass 2 emits them, and a class stripped by one but not the other
    is hoisted-but-never-rendered or rendered-but-never-declared. They live in
    separate modules since the counting pass was split out, so the scan spans
    both -- the claim is about the passes, not about a file."""

    def test_two_strip_sites_call_output_markers(self):
        source = DESCRIBE_TYPE_MODULE.read_text() + "\n" + DESCRIBE_COUNTING_MODULE.read_text()
        tree = ast.parse(source)
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "output_markers"
        ]
        assert len(calls) == 2, (
            f"expected exactly 2 output_markers(...) calls across the renderer's two passes "
            f"(_count_classes in _describe_counting.py + _render_model_body in describe_type.py) "
            f"-- found {len(calls)}"
        )

    def test_no_re_derived_marker_check_in_describe_type(self):
        """Would-be-missed case: a local `isinstance(m, ExcludeFromOutput)` or
        `isinstance(m, Carried)` re-check anywhere in describe_type.py would
        silently diverge from output_markers() the moment one side is edited
        and not the other -- the exact schema-fingerprint lesson."""
        tree = ast.parse(DESCRIBE_TYPE_MODULE.read_text())
        offenders = [
            ast.dump(n)
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "isinstance"
            and len(n.args) == 2
            and isinstance(n.args[1], ast.Name)
            and n.args[1].id in ("ExcludeFromOutput", "Carried")
        ]
        assert not offenders, f"re-derived marker isinstance check(s) found: {offenders}"


class TestGuardDetectorMetaTests:
    """Meta-test proving the call-count detector actually counts, not just
    checks non-zero (else the guard could pass with a wrong count silently)."""

    def test_detector_counts_multiple_calls(self):
        src = "def f():\n    project_output_model(x)\n    project_output_model(y)\n"
        tree = ast.parse(src)
        assert _project_output_model_call_count(tree, "f") == 2

    def test_detector_ignores_unrelated_calls(self):
        src = "def f():\n    some_other_call(x)\n"
        tree = ast.parse(src)
        assert _project_output_model_call_count(tree, "f") == 0
