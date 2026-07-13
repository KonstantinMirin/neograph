"""Structural guard: bare module-level logger convention (PAT-01, review 130726).

Every src/neograph/ module that logs binds its logger exactly once at module
scope via bare ``log = structlog.get_logger()`` — no arguments, no inline
resolution inside function bodies.

This guard enforces that convention and flags any drift:
- get_logger calls with arguments (__name__, literal strings, etc.)
- get_logger calls inline inside function bodies
- get_logger calls not bound to a simple module-level Name

The guard is receiver-agnostic: it flags ``structlog.get_logger(...)``,
``sl.get_logger(...)`` for an alias, or any other ``<X>.get_logger(...)`` form.
Only a bare ``get_logger()`` call at module scope, bound to a simple Name via
Assign or AnnAssign, is allowed.

Pure AST guard (no ``re`` module) — exempt from test_guards_meta.py's
named-regex discipline by construction.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"


def _find_non_bare_get_logger_calls(source: str) -> list[tuple[int, str]]:
    """Return ``(lineno, line_text)`` for every get_logger call that is NOT a
    bare module-level binding.

    A bare module-level binding is:
    - An ast.Assign or ast.AnnAssign at module top level
    - Whose value is an ast.Call to ``get_logger`` with NO arguments
    - And the target is a simple ast.Name (not a subscript or attribute)

    Any other get_logger call is flagged:
    - Has arguments (``get_logger(__name__)``, ``get_logger("neograph")``)
    - Is inline inside a function body
    - Is not bound to a simple Name (e.g., attribute access)
    - Receiver-agnostic (structlog.get_logger or alias.get_logger both checked)
    """
    tree = ast.parse(source)
    offenders: list[tuple[int, str]] = []

    # Pass 1: collect module-level bare get_logger() bindings as allowset
    bare_binding_locations: set[tuple[int, int]] = set()  # (lineno, end_lineno)

    for node in ast.iter_child_nodes(tree):
        targets: list[ast.expr] = []
        value: ast.expr | None = None

        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value

        if value is None:
            continue

        # Check if value is a bare get_logger() call
        if isinstance(value, ast.Call):
            func_name = None
            if isinstance(value.func, ast.Name) and value.func.id == "get_logger":
                func_name = "get_logger"
            elif isinstance(value.func, ast.Attribute) and value.func.attr == "get_logger":
                func_name = "get_logger"  # receiver-agnostic: <X>.get_logger()

            if func_name == "get_logger" and len(value.args) == 0 and len(value.keywords) == 0:
                # Bare call (no args) - check if bound to a simple Name at module level
                for tgt in targets:
                    if isinstance(tgt, ast.Name):
                        bare_binding_locations.add((node.lineno, getattr(node, "end_lineno", node.lineno)))

    # Pass 2: find all get_logger calls NOT in the allowset
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Detect get_logger calls (receiver-agnostic)
        is_get_logger = False
        if isinstance(node.func, ast.Name) and node.func.id == "get_logger":
            is_get_logger = True
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "get_logger":
            is_get_logger = True  # matches structlog.get_logger, sl.get_logger, etc.

        if not is_get_logger:
            continue

        # Check if this call is in the allowset (module-level bare binding)
        in_allowset = False
        for start, end in bare_binding_locations:
            if start <= node.lineno <= end:
                in_allowset = True
                break

        if not in_allowset:
            # This is a violation - inline call, has args, or not module-level binding
            line_text = source.splitlines()[node.lineno - 1].strip() if node.lineno <= len(source.splitlines()) else ""
            offenders.append((node.lineno, line_text))

    return offenders


class TestBareModuleLoggerConvention:
    """PAT-01: every src/neograph/ module that logs binds its logger exactly once
    at module scope via bare ``log = structlog.get_logger()``.

    No arguments, no inline resolution inside function bodies, no aliased receivers.
    The guard is receiver-agnostic: ``structlog.get_logger(__name__)`` and
    ``import structlog as sl; sl.get_logger(__name__)`` are both flagged.
    """

    def test_bare_module_level_logger_convention(self) -> None:
        """Every src/neograph/ module uses bare module-level ``log = structlog.get_logger()``."""
        offenders: list[str] = []
        for path in sorted(SRC_DIR.rglob("*.py")):
            for lineno, text in _find_non_bare_get_logger_calls(path.read_text()):
                rel = path.relative_to(SRC_DIR.parent.parent)
                offenders.append(f"{rel}:{lineno}: {text}")

        assert not offenders, (
            "Non-bare module-level get_logger() calls found. "
            "Every module must bind its logger at module scope via "
            "bare 'log = structlog.get_logger()' (no arguments, no inline calls):\n"
            + "\n".join(offenders)
        )

    # --- meta-tests: prove the guard catches regressions ---

    def test_meta_positive_flags_get_logger_with_name_arg(self) -> None:
        """Positive: ``log = structlog.get_logger(__name__)`` is flagged (has arg)."""
        source = "log = structlog.get_logger(__name__)\n"
        hits = _find_non_bare_get_logger_calls(source)
        assert hits == [(1, "log = structlog.get_logger(__name__)")]

    def test_meta_positive_flags_get_logger_with_literal_arg(self) -> None:
        """Positive: ``structlog.get_logger("neograph")`` is flagged (has arg)."""
        source = "structlog.get_logger('neograph')\n"
        hits = _find_non_bare_get_logger_calls(source)
        assert hits == [(1, "structlog.get_logger('neograph')")]

    def test_meta_positive_flags_inline_get_logger_call(self) -> None:
        """Positive: inline ``structlog.get_logger("neograph").warning(...)`` is flagged."""
        source = """
def foo():
    import structlog
    structlog.get_logger("neograph").warning("message")
"""
        hits = _find_non_bare_get_logger_calls(source)
        assert len(hits) == 1 and hits[0][0] == 4  # line 4, the inline call (line 1 is the leading blank)

    def test_meta_positive_flags_aliased_receiver_get_logger(self) -> None:
        """Positive: ``import structlog as sl; sl.get_logger(__name__)`` is flagged.

        Receiver-agnostic: the guard matches the concept of a get_logger call,
        not the spelling of the structlog module. Aliases cannot evade.
        """
        source = "import structlog as sl\nsl.get_logger(__name__)\n"
        hits = _find_non_bare_get_logger_calls(source)
        assert len(hits) == 1 and hits[0][0] == 2  # line 2, the aliased call

    def test_meta_positive_flags_get_logger_in_function_body(self) -> None:
        """Positive: get_logger call inside function body (even bare) is flagged."""
        source = """
def foo():
    log = structlog.get_logger()
    log.warning("message")
"""
        hits = _find_non_bare_get_logger_calls(source)
        assert len(hits) == 1 and hits[0][0] == 3  # line 3, function-local binding (line 1 is the leading blank)

    def test_meta_negative_passes_bare_module_level_binding(self) -> None:
        """Negative: bare ``log = structlog.get_logger()`` at module scope passes."""
        source = "import structlog\nlog = structlog.get_logger()\n"
        hits = _find_non_bare_get_logger_calls(source)
        assert hits == []

    def test_meta_negative_passes_bare_binding_with_ann_assign(self) -> None:
        """Negative: bare ``log: structlog.Logger = structlog.get_logger()`` passes.

        AnnAssign form (annotated assignment) is as valid as Assign for the
        module-level binding convention.
        """
        source = "import structlog\nlog: structlog.Logger = structlog.get_logger()\n"
        hits = _find_non_bare_get_logger_calls(source)
        assert hits == []

    def test_meta_negative_passes_module_with_no_logging(self) -> None:
        """Negative: a module with no get_logger calls at all passes."""
        source = "def foo():\n    return 42\n"
        hits = _find_non_bare_get_logger_calls(source)
        assert hits == []
