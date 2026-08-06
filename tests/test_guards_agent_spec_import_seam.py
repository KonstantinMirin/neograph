"""Structural guard (neograph-s7zt3.11): the Agent Spec IMPORT path recurses
into a sub-flow in exactly ONE place.

Disease. Reconstructing a ``FlowNode`` into a ``Construct`` is a multi-step
procedure -- recurse through ``from_agent_spec``, rename to the item's name, and
restore the declared boundary port -- and it existed TWICE: inlined in the
``PrimaryShape.BARE`` arm, and copied again inside ``_flow_member_to_construct``.
The second copy's docstring asserted it "reuses the SAME from_agent_spec
FlowNode->Construct recursion the bare-FlowNode item path uses" while in fact
re-implementing it.

That is not a cosmetic duplication, and this guard exists because it demonstrably
diverged: BOTH copies dropped the sub-Construct's ``input``/``output``, so a
sub-Construct declared ``input=A, output=B`` came back with both ``None`` and the
reimported parent no longer compiled -- while still satisfying an
is-a-Construct-with-the-right-combo check, so no test saw it. A procedure with
enough steps to diverge must have one owner.

The EXPORT side has one body seam too, but only since **neograph-15rpw** — it
previously had two ``FlowNode`` construction sites, ``_agent_spec.py``'s
``_lower_item_body`` and an inline one in the Construct-variant arm of
``_lower_oracle`` (which needs a distinct per-variant name and metadata, now
stamped on with a shallow ``model_copy`` over the seam's result). This guard
still governs the IMPORT path only and enforces nothing about export.

Scope note: ``from_agent_spec`` is also invoked at genuine ENTRY points
(``_hot_swap.py``, ``_agent_spec_dispatch.py`` -- relocated from ``factory.py``
by neograph-jtawq.9) to load a whole Flow. Those are not sub-flow recursion and
are out of scope -- which is exactly why this guard is scoped to the
import-path modules rather than grepping the tree for the call.

Pure AST, no ``re``, so this module is exempt-by-construction from
``test_guards_meta.py`` Discipline 1.
"""

from __future__ import annotations

import ast
import pathlib

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# The Agent Spec import path. All four are scanned, not just the module that
# currently owns the seam: a recursion re-inlined back into loader.py must fail
# here too, or the guard silently narrows to wherever the code happens to live.
IMPORT_PATH_FILES = [
    _SRC / "loader.py",
    _SRC / "_agent_spec_node_import.py",
    _SRC / "_agent_spec_group_import.py",
    _SRC / "_agent_spec_swarm_import.py",
]

# The ONE function permitted to recurse into a sub-flow.
SEAM = "_construct_from_subflow"

# Names that invoke the recursion: the module-level entry point, and the
# injected-callable parameter it is threaded through as.
RECURSION_CALLEES = frozenset({"from_agent_spec", "from_spec"})


def _recursion_call_sites() -> dict[str, list[int]]:
    """Map enclosing-function name -> lines where it INVOKES the recursion.

    Passing ``from_agent_spec`` as an ARGUMENT (how the callable is injected) is
    deliberately not a call site -- threading the dependency is the sanctioned
    pattern; invoking it is what must stay singular.
    """
    sites: dict[str, list[int]] = {}
    for path in IMPORT_PATH_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in RECURSION_CALLEES:
                    sites.setdefault(fn.name, []).append(node.lineno)
    return sites


class TestSubFlowRecursionHasOneOwner:
    """One seam, and the files that must contain it are really being scanned."""

    def test_scanned_files_all_exist(self):
        """Anti-vacuity: a renamed module would otherwise make this guard pass by
        scanning nothing at all."""
        missing = [p.name for p in IMPORT_PATH_FILES if not p.exists()]
        assert not missing, (
            f"these import-path modules are missing, so this guard would scan a "
            f"shrinking surface and pass vacuously: {missing}. Re-point "
            f"IMPORT_PATH_FILES at the current layout."
        )

    def test_exactly_one_function_recurses_into_a_subflow(self):
        sites = _recursion_call_sites()

        assert SEAM in sites, (
            f"{SEAM} must be the site that recurses into a sub-flow. Found "
            f"recursion in: {sorted(sites)}. If the seam was renamed, update SEAM; "
            f"if it was deleted, the import path has lost its single owner."
        )

        offenders = {name: lines for name, lines in sites.items() if name != SEAM}
        assert not offenders, (
            "the FlowNode->Construct recursion must have exactly ONE owner on the "
            f"import path ({SEAM}); these functions invoke it directly instead of "
            "routing through the seam:\n"
            + "\n".join(f"  {name} (line{'s' if len(v) > 1 else ''} {v})" for name, v in sorted(offenders.items()))
            + f"\nRoute them through {SEAM}. A second copy is how the boundary-drop "
            "defect survived in one path and not the other (neograph-s7zt3.11)."
        )

    def test_the_seam_restores_the_boundary_port(self):
        """The seam's whole reason for existing is the boundary restoration -- a
        seam that only recursed and renamed would satisfy the count check above
        while reproducing the exact defect this ticket fixed."""
        for path in IMPORT_PATH_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for fn in ast.walk(tree):
                if isinstance(fn, ast.FunctionDef) and fn.name == SEAM:
                    written = {
                        node.value
                        for node in ast.walk(fn)
                        if isinstance(node, ast.Constant) and isinstance(node.value, str)
                    }
                    assert {"input", "output"} <= written, (
                        f"{SEAM} must restore BOTH 'input' and 'output' on the "
                        f"reconstructed Construct; found only {sorted(written & {'input', 'output'})}. "
                        "Without it a sub-Construct reimports with null ports and its "
                        "parent stops compiling, while still passing a combo check."
                    )
                    return
        raise AssertionError(f"{SEAM} not found in any scanned import-path module")


class TestDetectorSlips:
    """Slip meta-tests (PROC-2): the detector's boundaries, pinned."""

    @staticmethod
    def _sites(src: str) -> dict[str, list[int]]:
        sites: dict[str, list[int]] = {}
        tree = ast.parse(src)
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in RECURSION_CALLEES:
                    sites.setdefault(fn.name, []).append(node.lineno)
        return sites

    def test_slip_detector_fires_on_a_reinlined_recursion(self):
        """POSITIVE: a hand-rolled copy in a new function is caught."""
        assert self._sites("def sneaky(n):\n    return from_agent_spec(n.subflow)\n") == {"sneaky": [2]}
        assert self._sites("def sneaky(n, from_spec):\n    return from_spec(n.subflow)\n") == {"sneaky": [2]}

    def test_slip_detector_ignores_threading_the_callable(self):
        """NEGATIVE, and the boundary that matters: PASSING ``from_agent_spec``
        as an argument is the sanctioned injection pattern, not a call site.
        A detector that flagged it would make the correct design unwritable."""
        assert self._sites("def ok(n):\n    return _construct_from_subflow(n.subflow, n.name, from_agent_spec)\n") == {}
        # An unrelated call is not a recursion either.
        assert self._sites("def ok(n):\n    return something_else(n)\n") == {}
