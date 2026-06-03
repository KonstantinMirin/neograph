"""Structural guards — completion invariants for ARCH-4 (neograph-v3xx).

Three migrations were declared complete in prior tickets without a
structurally-enforced completion invariant, so the codebase drifted back
toward the pre-migration shape. Each guard here pins ONE completion invariant
so the migration cannot silently regress:

* HIGH-01 — ``decorators.py`` carries ZERO module-level mutable dicts (the
  scripted/condition shims live per-node; the standalone ``@tool``/``@merge_fn``
  registries live in a leaf registry module, not in ``decorators.py``).
* HIGH-02 — ``compile()`` returns a typed ``CompiledNeograph`` facade; no
  ``_neo_*`` attribute is monkey-patched onto the LangGraph graph and no
  ``getattr(graph, "_neo_*")`` read survives.
* HIGH-08 — ``factory.py`` exposes ZERO test-only ``# noqa: F401`` re-export
  shims; tests import from the real leaf modules.
* HIGH-09 — no function-local import allowlist entry names ``neograph.decorators``
  as the source of a symbol that is actually DEFINED in a leaf module
  (``_sidecar``/``_di_classify``) and merely re-exported by ``decorators``.

Each guard has positive + negative-mutation meta-tests so the scanner cannot rot.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"


def _module_level_dict_assigns(tree: ast.AST) -> list[tuple[int, str]]:
    """Return (lineno, name) for every module-level mutable-dict binding.

    Matches both ``_x: dict[...] = {}`` (AnnAssign) and ``_x = {}`` (Assign)
    whose value is a dict literal — i.e. a mutable module-level registry.
    Only top-level statements count (class/function bodies are excluded).
    """
    hits: list[tuple[int, str]] = []
    body = getattr(tree, "body", [])
    for stmt in body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            ann = stmt.annotation
            is_dict_ann = (
                (isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name) and ann.value.id == "dict")
                or (isinstance(ann, ast.Name) and ann.id == "dict")
            )
            if is_dict_ann:
                hits.append((stmt.lineno, stmt.target.id))
        elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Dict):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    hits.append((stmt.lineno, tgt.id))
    return hits


class TestDecoratorsNoModuleLevelDicts:
    """HIGH-01: ``decorators.py`` must carry zero module-level mutable dicts.

    The decorator-side scripted/condition shims move to per-node PrivateAttrs
    (``_scripted_shim`` / ``_condition_shim``); the standalone ``@tool`` /
    ``@merge_fn`` import-time registries move to a leaf registry module. After
    that, ``decorators.py`` has no module-level dict registry to clear.
    """

    def test_decorators_has_no_module_level_dicts(self):
        tree = ast.parse((SRC_DIR / "decorators.py").read_text())
        hits = _module_level_dict_assigns(tree)
        assert hits == [], (
            "decorators.py still has module-level mutable dict(s):\n"
            + "\n".join(f"  decorators.py:{ln}: {nm}" for ln, nm in hits)
            + "\n\nMove scripted/condition shims to per-node PrivateAttrs and the "
            "standalone @tool/@merge_fn registries to a leaf registry module."
        )

    def test_mutation_module_dict_detected(self):
        tree = ast.parse("_x: dict[str, int] = {}\n_y = {}\n")
        hits = _module_level_dict_assigns(tree)
        assert {nm for _, nm in hits} == {"_x", "_y"}, hits

    def test_function_local_dict_not_flagged(self):
        tree = ast.parse("def f():\n    local: dict[str, int] = {}\n    return local\n")
        assert _module_level_dict_assigns(tree) == []


class TestNoNeoMonkeyPatch:
    """HIGH-02: no ``_neo_*`` attribute is stitched onto a runtime object, and
    ``compile()`` returns the typed ``CompiledNeograph`` facade.

    The pre-migration shape stashed 8 ``compiled._neo_* = ...`` attributes on
    LangGraph's CompiledStateGraph (each with ``# type: ignore[attr-defined]``)
    and read them back via ``getattr(graph, "_neo_*", None)`` in runner.py /
    verify.py. The facade replaces all of that with typed fields.
    """

    @staticmethod
    def _neo_attr_assigns(tree: ast.AST) -> list[int]:
        """Lineno of every ``<expr>._neo_* = ...`` attribute assignment."""
        hits: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Attribute) and tgt.attr.startswith("_neo_"):
                        hits.append(node.lineno)
        return hits

    @staticmethod
    def _neo_getattr_reads(tree: ast.AST) -> list[int]:
        """Lineno of every ``getattr(x, "_neo_...")`` read."""
        hits: list[int] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and node.args[1].value.startswith("_neo_")
            ):
                hits.append(node.lineno)
        return hits

    def test_no_neo_attribute_monkey_patch_in_src(self):
        violations: list[str] = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
            for ln in self._neo_attr_assigns(tree):
                violations.append(f"  {py_file.name}:{ln}: <obj>._neo_* = ...")
        assert violations == [], (
            "\n_neo_* monkey-patch assignment(s) found:\n" + "\n".join(violations)
            + "\n\nUse the typed CompiledNeograph facade fields instead."
        )

    def test_no_neo_getattr_reads_in_src(self):
        violations: list[str] = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
            for ln in self._neo_getattr_reads(tree):
                violations.append(f"  {py_file.name}:{ln}: getattr(_, '_neo_...')")
        assert violations == [], (
            "\ngetattr(_, '_neo_*') read(s) found:\n" + "\n".join(violations)
            + "\n\nRead the typed CompiledNeograph field instead."
        )

    def test_compile_returns_compiled_neograph(self):
        tree = ast.parse((SRC_DIR / "compiler.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "compile":
                ret = node.returns
                name = (
                    ret.id if isinstance(ret, ast.Name)
                    else ret.attr if isinstance(ret, ast.Attribute)
                    else ast.dump(ret) if ret is not None
                    else None
                )
                assert name == "CompiledNeograph", (
                    f"compile() returns {name!r}, expected 'CompiledNeograph'"
                )
                return
        raise AssertionError("compile() not found in compiler.py")

    def test_mutation_neo_assign_detected(self):
        tree = ast.parse("def f(g):\n    g._neo_thing = 1\n")
        assert self._neo_attr_assigns(tree), "scanner missed a _neo_ assignment"

    def test_mutation_neo_getattr_detected(self):
        tree = ast.parse('def f(g):\n    return getattr(g, "_neo_thing", None)\n')
        assert self._neo_getattr_reads(tree), "scanner missed a _neo_ getattr"


class TestFactoryNoTestReexportShims:
    """HIGH-08: ``factory.py`` exposes zero test-only ``# noqa: F401`` re-export
    shims. Tests import internal helpers from their real leaf modules.
    """

    def test_factory_has_no_noqa_f401_shims(self):
        text = (SRC_DIR / "factory.py").read_text()
        offenders = [
            f"  factory.py:{i}: {line.strip()}"
            for i, line in enumerate(text.splitlines(), start=1)
            if "noqa: F401" in line or "noqa:F401" in line
        ]
        assert offenders == [], (
            "\nfactory.py still has test-only re-export shim(s):\n"
            + "\n".join(offenders)
            + "\n\nRepoint the test importers to the real leaf modules and delete "
            "these re-exports."
        )


class TestNoLeafSymbolReexportedViaDecorators:
    """HIGH-09: no function-local import allowlist entry names
    ``neograph.decorators`` as the source of a symbol that is actually DEFINED
    in a leaf module and only re-exported by ``decorators``.

    Such an entry describes an *illusory* cycle: the real symbol home is a leaf
    module (``_sidecar`` / ``_di_classify``), so the import should point there
    (module-level when no cycle, function-local with a truthful justification
    when a genuine cycle exists). Either way the allowlist must not claim
    ``decorators`` owns a symbol it merely re-exports.
    """

    @staticmethod
    def _decorators_defined_names() -> set[str]:
        """Names DEFINED at module level in decorators.py (def/class/assign).

        Imported (re-exported) names are deliberately excluded — that is the
        whole point of the guard.
        """
        tree = ast.parse((SRC_DIR / "decorators.py").read_text())
        defined: set[str] = set()
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(stmt.name)
            elif isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        defined.add(tgt.id)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                defined.add(stmt.target.id)
        return defined

    def test_allowlist_entries_naming_decorators_own_their_symbols(self):
        from tests.test_guards_sidecar_imports import FUNCTION_LOCAL_IMPORT_ALLOWLIST

        defined = self._decorators_defined_names()
        violations: list[str] = []
        for file_rel, module, names in FUNCTION_LOCAL_IMPORT_ALLOWLIST:
            if module != "neograph.decorators":
                continue
            for name in names:
                if name not in defined:
                    violations.append(
                        f"  {file_rel} imports '{name}' from neograph.decorators, "
                        f"but '{name}' is not defined there (re-exported from a leaf)."
                    )
        assert violations == [], (
            "\nIllusory-cycle allowlist entry(ies) found:\n" + "\n".join(violations)
            + "\n\nRepoint the import to the leaf module where the symbol lives "
            "(_sidecar / _di_classify) and fix the allowlist entry."
        )

    def test_mutation_reexported_symbol_detected(self):
        defined = self._decorators_defined_names()
        # get_merge_fn_metadata lives in _sidecar.py and is re-exported by
        # decorators.py — it must NOT count as defined-in-decorators.
        assert "get_merge_fn_metadata" not in defined, (
            "guard premise broken: get_merge_fn_metadata must not be DEFINED in "
            "decorators.py (it lives in _sidecar.py)"
        )
