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
from neograph.testing import scaffold_tests
from tests.schemas import Claims, RawText


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
