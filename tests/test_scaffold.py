"""Tests for neograph.testing.scaffold_tests (CLI test-scaffold code generator).

Until neograph-wpzg this module had ZERO behavioral coverage (only a structural
import guard in test_guards_sidecar_imports.py). These tests pin the observable
contract of the generator: every file it emits must be valid, compilable Python.

The dict-form test is the TDD red artifact for the CON-01 finding: _node_info
rendered ``repr(node.outputs)`` for dict-form outputs, emitting a dict literal
full of ``<class '...'>`` reprs into ``assert node.outputs is {...}`` — invalid
Python that fails to compile. The fix renders the primary output type's name via
``normalize_outputs(node.outputs).primary``.
"""

from __future__ import annotations

from pathlib import Path

from neograph import Construct, Node
from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
from neograph.testing import _collect_edges, _collect_items, scaffold_tests
from tests.schemas import Claims, RawText


def _branch_parent(arm_node: Node, *, arm: str = "true") -> Construct:
    """Wrap ``arm_node`` in the chosen arm of a ``_BranchNode`` under a parent
    Construct whose first node is a scripted seed producing ``RawText``.

    Mirrors the branch-construction helper in ``test_branch_arm_noniter_walks.py``
    (neograph-vn5f); the branch itself is built programmatically because
    ``_BranchNode`` sentinels only arise from ForwardConstruct tracing.
    """
    seed = Node.scripted("seed", fn="scaffold_seed", outputs=RawText)
    take_true = arm == "true"
    cond = _ConditionSpec(
        source_node=seed,
        attr_chain=["text"],
        op_fn=(lambda v, _t: bool(v)) if take_true else (lambda v, _t: not v),
        op_str="route",
        threshold=None,
    )
    meta = _BranchMeta(
        condition_spec=cond,
        true_arm_nodes=[arm_node] if take_true else [],
        false_arm_nodes=[] if take_true else [arm_node],
    )
    return Construct("parent", nodes=[seed, _BranchNode(meta, 0)])


class TestScaffoldGeneratesCompilableCode:
    """Every generated file must be syntactically valid Python."""

    def test_metadata_compiles_when_node_has_dict_form_outputs(self, tmp_path):
        """Dict-form Node.outputs must yield a compilable test_metadata.py.

        RED (neograph-wpzg / CON-01): the old _node_info emitted
        ``repr(node.outputs)`` -> ``{'result': <class 'tests.schemas.RawText'>,
        ...}`` into ``assert explore.outputs is {repr}``, which is not valid
        Python -> SyntaxError at compile(). The fix renders the primary type
        name (RawText), not the dict repr.
        """
        explore = Node("explore", outputs={"result": RawText, "tool_log": Claims})
        construct = Construct("pipe", nodes=[explore])

        scaffold_tests(construct, output_dir=str(tmp_path), overwrite=True)

        meta_src = (tmp_path / "test_metadata.py").read_text()
        # Must be syntactically valid Python (this is what FAILS pre-fix).
        compile(meta_src, "test_metadata.py", "exec")
        # And it must reference the primary type by name, never a class repr.
        assert "RawText" in meta_src
        assert "<class" not in meta_src

    def test_all_generated_files_compile_for_single_output_pipeline(self, tmp_path):
        """Characterization safety net for the T1 codegen refactor.

        Guards existing behavior: a plain single-output pipeline must scaffold
        into files that are all valid Python. Passes today; protects the
        _gen_sync set-literal extraction from regressing the emitted source.
        """
        alpha = Node.scripted("alpha", fn="f", outputs=RawText)
        beta = Node.scripted("beta", fn="f", inputs=RawText, outputs=Claims)
        construct = Construct("simple", nodes=[alpha, beta])

        written = scaffold_tests(construct, output_dir=str(tmp_path), overwrite=True)

        assert written, "scaffold_tests produced no files"
        py_files = [p for p in written if p.endswith(".py")]
        assert py_files, "no .py files generated"
        for path in py_files:
            src = Path(path).read_text()
            compile(src, path, "exec")  # raises SyntaxError on bad codegen


class TestScaffoldSeesBranchArmNodes:
    """Branch-arm nodes and their edges must appear in the scaffold walks.

    RED (neograph-gfoq): ``_collect_items`` and ``_collect_edges`` walked
    ``construct.nodes`` with ``isinstance(item, Node)`` guards and never
    descended into ``_BranchNode`` arms, so generated scaffolds omitted every
    conditionally-executed node and its edges. The fix migrates both walks to
    ``iter_with_arms`` (the vn5f arm-descent primitive).
    """

    def test_collect_items_includes_true_arm_node(self):
        """A bare Node in the true arm must appear in _collect_items' node list."""
        arm = Node.scripted("gate", fn="f", inputs={"seed": RawText}, outputs=Claims)
        parent = _branch_parent(arm)

        nodes, _subs = _collect_items(parent)
        names = [n["name"] for n in nodes]
        assert "gate" in names, "true-arm node 'gate' was invisible to _collect_items — the scaffold omits it entirely"

    def test_collect_items_includes_false_arm_node(self):
        """A bare Node in the false arm must appear in _collect_items' node list."""
        arm = Node.scripted("gate", fn="f", inputs={"seed": RawText}, outputs=Claims)
        parent = _branch_parent(arm, arm="false")

        nodes, _subs = _collect_items(parent)
        names = [n["name"] for n in nodes]
        assert "gate" in names, "false-arm node 'gate' was invisible to _collect_items"

    def test_collect_edges_includes_arm_node_edge(self):
        """A dict-form arm Node's upstream edge must appear in _collect_edges."""
        arm = Node.scripted("gate", fn="f", inputs={"seed": RawText}, outputs=Claims)
        parent = _branch_parent(arm)

        edges = _collect_edges(parent)
        assert ("seed", "gate") in edges, (
            "the arm consumer's 'seed' -> 'gate' edge is missing — the scaffold's "
            "topological-ordering test never covers the branch arm"
        )

    def test_scaffold_metadata_covers_arm_node(self, tmp_path):
        """The generated test_metadata.py must include a stub for the arm node."""
        arm = Node.scripted("gate", fn="f", inputs={"seed": RawText}, outputs=Claims)
        parent = _branch_parent(arm)

        scaffold_tests(parent, output_dir=str(tmp_path), overwrite=True)

        meta_src = (tmp_path / "test_metadata.py").read_text()
        assert "test_gate" in meta_src, "generated metadata scaffold has no test for the branch-arm node"
