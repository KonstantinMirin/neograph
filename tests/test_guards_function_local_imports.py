"""Structural guards: function-local factory/llm imports in oracle/dispatch/
decorators, retry-policy signature, StateKeys centralization, no module-level
registration/globals."""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# Error classes that must use .build() instead of direct construction.
ERROR_CLASSES = frozenset({
    "ConstructError",
    "ExecutionError",
    "CompileError",
    "ConfigurationError",
    "NeographError",
})


class TestNoFunctionLocalFactoryImportInOracle:
    """neograph-9occ: break factory <-> _oracle cycle via a Protocol module.

    `_oracle.py` is imported by `factory.py`, so a top-level
    `from neograph.factory import ...` in `_oracle.py` would create a cycle.
    Historically `_oracle.py` worked around this with function-local imports
    (the canonical deferred-import anti-pattern).

    The fix is to extract the shared names into a dedicated Protocol module
    (`_runtime_registry.py`) that both modules import from. After the
    extraction, no function-local `from neograph.factory` import is allowed
    inside `_oracle.py`.

    Mutation-verified: a parametrized tmpdir fixture writes a synthetic
    module with a function-local `from neograph.factory import _state_get`,
    runs the AST scanner, asserts detection.
    """

    @staticmethod
    def _find_function_local_factory_imports(source: str) -> list[tuple[int, str]]:
        """Return (lineno, module) for every function-local `from neograph.factory`
        or `from .factory` ImportFrom node in *source*.
        """
        tree = ast.parse(source)
        offenders: list[tuple[int, str]] = []
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for sub in ast.walk(func):
                if not isinstance(sub, ast.ImportFrom):
                    continue
                module = sub.module or ""
                # Absolute: from neograph.factory ...
                # Relative: from .factory ... (level=1, module="factory")
                is_absolute = module == "neograph.factory"
                is_relative = sub.level >= 1 and module == "factory"
                if is_absolute or is_relative:
                    offenders.append((sub.lineno, module or f".{'.' * (sub.level - 1)}factory"))
        return offenders

    def test_no_function_local_factory_import_in_oracle(self):
        """`_oracle.py` must not contain any function-local `from neograph.factory` import."""
        oracle_path = SRC_DIR / "_oracle.py"
        offenders = self._find_function_local_factory_imports(oracle_path.read_text())
        assert offenders == [], (
            f"\n{len(offenders)} function-local `from neograph.factory` import(s) "
            f"found in {oracle_path.name}:\n"
            + "\n".join(f"  line {lineno}: from {module} import ..." for lineno, module in offenders)
            + "\n\nFix: extract the shared symbols into a Protocol/registry module "
              "(e.g. `_runtime_registry.py`) and import from there at module scope."
        )

    def test_scanner_detects_injected_function_local_factory_import(self, tmp_path: pathlib.Path):
        """Mutation: a synthetic module with a function-local factory import must be flagged."""
        synthetic = tmp_path / "_oracle.py"
        synthetic.write_text(
            "from __future__ import annotations\n"
            "\n"
            "def _lookup(name):\n"
            "    from neograph.factory import lookup_scripted\n"
            "    return lookup_scripted(name)\n"
        )
        offenders = self._find_function_local_factory_imports(synthetic.read_text())
        assert any(m == "neograph.factory" for _, m in offenders), (
            f"scanner failed to detect injected function-local factory import; offenders={offenders}"
        )

    def test_scanner_accepts_module_level_factory_import(self, tmp_path: pathlib.Path):
        """Module-level imports from neograph.factory are not flagged by this scanner.

        (The deferred-import budget guard tracks those separately.)
        """
        synthetic = tmp_path / "_oracle.py"
        synthetic.write_text(
            "from __future__ import annotations\n"
            "from neograph.factory import lookup_scripted\n"
            "\n"
            "def _lookup(name):\n"
            "    return lookup_scripted(name)\n"
        )
        offenders = self._find_function_local_factory_imports(synthetic.read_text())
        assert offenders == [], (
            f"scanner false-positive on module-level import; offenders={offenders}"
        )


class TestNoFunctionLocalLlmOrFactoryImportInDispatch:
    """neograph-7uhx: _dispatch.py must not import _llm/factory inside function bodies.

    Six function-local imports lived inside ThinkDispatch.execute / ToolDispatch.execute /
    _render_input — including a defensive ``try/except ImportError`` guarding the
    ``_get_global_renderer`` read against a phantom circular import. The fix is a
    Protocol module so the types are importable at module top, plus deletion of the
    try/except.

    Mutation-verified: re-introducing `from neograph._llm import _is_inline_prompt`
    inside a function body in _dispatch.py, or wrapping a renderer read in
    `try: ... except ImportError: ...`, makes this test fail naming the offender.
    """

    FORBIDDEN_MODULES = frozenset({
        "neograph._llm",
        "neograph.factory",
        "_llm",
        "factory",
    })

    def _walk_function_bodies(self, tree: ast.AST):
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                for child in ast.walk(fn):
                    if child is fn:
                        continue
                    yield fn, child

    def test_no_function_local_llm_or_factory_import_in_dispatch(self):
        path = SRC_DIR / "_dispatch.py"
        tree = ast.parse(path.read_text(), filename=str(path))

        function_local_imports: list[str] = []
        importerror_handlers: list[str] = []

        for fn, child in self._walk_function_bodies(tree):
            if isinstance(child, ast.ImportFrom):
                module = child.module or ""
                if module in self.FORBIDDEN_MODULES:
                    alias_names = ", ".join(alias.name for alias in child.names)
                    function_local_imports.append(
                        f"  _dispatch.py:{child.lineno} (in {fn.name}): "
                        f"from {module} import {alias_names}"
                    )
            elif isinstance(child, ast.ExceptHandler):
                # ``except ImportError:`` or ``except (ImportError, ...):``
                exc_type = child.type
                exc_names: list[str] = []
                if isinstance(exc_type, ast.Name):
                    exc_names = [exc_type.id]
                elif isinstance(exc_type, ast.Tuple):
                    exc_names = [
                        n.id for n in exc_type.elts if isinstance(n, ast.Name)
                    ]
                if "ImportError" in exc_names:
                    importerror_handlers.append(
                        f"  _dispatch.py:{child.lineno} (in {fn.name}): "
                        f"except {', '.join(exc_names)}"
                    )

        problems = []
        if function_local_imports:
            problems.append(
                f"{len(function_local_imports)} function-local _llm/factory import(s):\n"
                + "\n".join(function_local_imports)
            )
        if importerror_handlers:
            problems.append(
                f"{len(importerror_handlers)} defensive ImportError handler(s):\n"
                + "\n".join(importerror_handlers)
            )

        assert not problems, (
            "\n_dispatch.py has function-local imports / defensive ImportError "
            "handlers that should be hoisted via a Protocol module (neograph-7uhx):\n\n"
            + "\n\n".join(problems)
            + "\n\nFix: extract a Protocol module so the types/interfaces are "
            "importable at module top, then delete the try/except ImportError."
        )

    def test_scanner_detects_mutation(self, tmp_path):
        """Mutation check: synthetic module with function-local import + try/except ImportError
        must be flagged by the scanner logic (we re-execute it via parse → walk)."""
        synthetic = tmp_path / "_dispatch.py"
        synthetic.write_text(
            "from __future__ import annotations\n"
            "\n"
            "def f():\n"
            "    from neograph._llm import invoke_structured\n"
            "    try:\n"
            "        from neograph._llm import _get_global_renderer\n"
            "        return _get_global_renderer()\n"
            "    except ImportError:\n"
            "        return None\n"
        )
        tree = ast.parse(synthetic.read_text(), filename=str(synthetic))

        function_local_imports = []
        importerror_handlers = []
        for _fn, child in self._walk_function_bodies(tree):
            if isinstance(child, ast.ImportFrom):
                if (child.module or "") in self.FORBIDDEN_MODULES:
                    function_local_imports.append(child.lineno)
            elif isinstance(child, ast.ExceptHandler):
                exc_type = child.type
                exc_names: list[str] = []
                if isinstance(exc_type, ast.Name):
                    exc_names = [exc_type.id]
                elif isinstance(exc_type, ast.Tuple):
                    exc_names = [
                        n.id for n in exc_type.elts if isinstance(n, ast.Name)
                    ]
                if "ImportError" in exc_names:
                    importerror_handlers.append(child.lineno)

        assert function_local_imports, "scanner missed function-local _llm import"
        assert importerror_handlers, "scanner missed except ImportError handler"


class TestNoFunctionLocalFactoryImportInDecorators:
    """neograph-q7fh: break decorators ↔ factory cycle via a Protocol module.

    The original cycle was `decorators → factory → _construct_builder → decorators`.
    `_sidecar.py` extraction broke the `decorators ↔ _construct_builder` edge.
    This guard enforces the remaining `decorators ↔ factory` break: the shared
    registration callables (`register_scripted`, `register_condition`) live in
    a leaf Protocol module that both `decorators.py` and `factory.py` import
    from, removing the need for function-local imports inside `decorators.py`.

    A function-local `from neograph.factory import ...` is forbidden because
    it signals a residual cycle: if decorators.py needed nothing from factory.py
    at module-import time, the imports should live at the top of the file.

    Mutation-verified: a parametrized tmpdir fixture writes a synthetic
    module with a function-local `from neograph.factory import register_scripted`,
    runs the AST scanner, asserts detection.
    """

    @staticmethod
    def _find_function_local_factory_imports(source: str) -> list[tuple[int, str]]:
        """Return (lineno, module) for every function-local `from neograph.factory`
        or `from .factory` ImportFrom node in *source*.
        """
        tree = ast.parse(source)
        offenders: list[tuple[int, str]] = []
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for sub in ast.walk(func):
                if not isinstance(sub, ast.ImportFrom):
                    continue
                module = sub.module or ""
                is_absolute = module == "neograph.factory"
                is_relative = sub.level >= 1 and module == "factory"
                if is_absolute or is_relative:
                    offenders.append((sub.lineno, module or f".{'.' * (sub.level - 1)}factory"))
        return offenders

    def test_no_function_local_factory_import_in_decorators(self):
        """`decorators.py` must not contain any function-local `from neograph.factory` import."""
        decorators_path = SRC_DIR / "decorators.py"
        offenders = self._find_function_local_factory_imports(decorators_path.read_text())
        assert offenders == [], (
            f"\n{len(offenders)} function-local `from neograph.factory` import(s) "
            f"found in {decorators_path.name}:\n"
            + "\n".join(f"  line {lineno}: from {module} import ..." for lineno, module in offenders)
            + "\n\nFix: extract the shared symbols into a Protocol/registry module "
              "(e.g. `_runtime_registry.py`) and import from there at module scope."
        )

    def test_scanner_detects_injected_function_local_factory_import_in_decorators(
        self, tmp_path: pathlib.Path
    ):
        """Mutation: a synthetic module with a function-local factory import must be flagged."""
        synthetic = tmp_path / "decorators.py"
        synthetic.write_text(
            "from __future__ import annotations\n"
            "\n"
            "def make_node():\n"
            "    from neograph.factory import register_scripted\n"
            "    register_scripted('x', lambda: None)\n"
        )
        offenders = self._find_function_local_factory_imports(synthetic.read_text())
        assert any(m == "neograph.factory" for _, m in offenders), (
            f"scanner failed to detect injected function-local factory import; offenders={offenders}"
        )

    def test_scanner_accepts_module_level_factory_import_in_decorators(
        self, tmp_path: pathlib.Path
    ):
        """Module-level imports from neograph.factory are not flagged by this scanner.

        (The deferred-import budget guard tracks those separately.)
        """
        synthetic = tmp_path / "decorators.py"
        synthetic.write_text(
            "from __future__ import annotations\n"
            "from neograph.factory import register_scripted\n"
            "\n"
            "def make_node():\n"
            "    register_scripted('x', lambda: None)\n"
        )
        offenders = self._find_function_local_factory_imports(synthetic.read_text())
        assert offenders == [], (
            f"scanner false-positive on module-level import; offenders={offenders}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TEST: §3 - compile() does not accept retry_policy (neograph-s5yk)
# ═══════════════════════════════════════════════════════════════════════════


class TestNoRetryPolicyInCompileSignature:
    """``compile()`` must not expose a ``retry_policy`` kwarg.

    Per ``docs/design/architecture-decisions.md`` §3, LangGraph's
    ``RetryPolicy`` is not a framework-boundary concern. Retry concerns live
    in three separate layers (transient -> ``llm_factory``; output-quality ->
    ``LlmConfig.max_retries``; flaky scripted -> in-function). A compile-level
    knob is also incorrect because it replays already-executed ``act``-mode
    tools.

    Guard covers:
      (a) AST scan of ``compiler.py``: no function named ``compile`` accepts a
          ``retry_policy`` parameter.
      (b) ``inspect.signature(compile)`` confirms it at runtime.
      (c) ``src/neograph/`` is free of ``RetryPolicy`` references.
    """

    def test_compile_signature_has_no_retry_policy_param_ast(self):
        compiler_py = SRC_DIR / "compiler.py"
        tree = ast.parse(compiler_py.read_text(), filename=str(compiler_py))

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            all_args = (
                list(args.posonlyargs)
                + list(args.args)
                + list(args.kwonlyargs)
            )
            for a in all_args:
                assert a.arg != "retry_policy", (
                    f"compiler.py:{node.lineno} {node.name}(...) still declares "
                    "a 'retry_policy' parameter; per §3 this kwarg has been removed"
                )

    def test_compile_runtime_signature_has_no_retry_policy(self):
        import inspect

        from neograph import compile as compile_fn

        sig = inspect.signature(compile_fn)
        assert "retry_policy" not in sig.parameters, (
            f"neograph.compile() still exposes retry_policy: {list(sig.parameters)}"
        )

    def test_src_neograph_does_not_reference_retry_policy(self):
        offenders: list[str] = []
        for py_file in SRC_DIR.rglob("*.py"):
            for lineno, raw in enumerate(py_file.read_text().splitlines(), start=1):
                stripped = raw.strip()
                if stripped.startswith("#"):
                    continue
                if "RetryPolicy" in raw or "retry_policy" in raw:
                    offenders.append(f"{py_file.name}:{lineno}: {stripped[:120]}")
        assert offenders == [], (
            f"\n{len(offenders)} reference(s) to RetryPolicy / retry_policy remain in src/neograph:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nPer §3, this kwarg is removed from the framework boundary."
        )

    def test_scanner_detects_injected_retry_policy_param(self, tmp_path: pathlib.Path):
        """Mutation: a synthetic compile() with retry_policy must be flagged."""
        synthetic = tmp_path / "compiler.py"
        synthetic.write_text(
            "from typing import Any\n"
            "def compile(construct, checkpointer=None, retry_policy=None) -> Any:\n"
            "    return None\n"
        )
        tree = ast.parse(synthetic.read_text(), filename=str(synthetic))
        flagged = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != "compile":
                continue
            args = node.args
            all_args = (
                list(args.posonlyargs)
                + list(args.args)
                + list(args.kwonlyargs)
            )
            if any(a.arg == "retry_policy" for a in all_args):
                flagged = True
        assert flagged, "mutation case: injected retry_policy param was not detected"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: §3 - retry-semantics doc page covers the three layers (neograph-7vls)
# ═══════════════════════════════════════════════════════════════════════════


class TestRetrySemanticsDocPage:
    """The neograph.pro retry-semantics page must cover the three-layer model.

    Per ``docs/design/architecture-decisions.md`` §3, the page must mention
    every layer (transient -> ``llm_factory.with_retry``; output-quality ->
    ``LlmConfig.max_retries``; flaky external in scripted nodes -> in-function),
    state the correctness rationale for not exposing ``retry_policy`` at the
    ``compile()`` boundary (``act``-mode tool replay), and show a concrete
    ``model.with_retry(...)`` example.
    """

    DOC_DIR = pathlib.Path(__file__).resolve().parent.parent / "website" / "src" / "content" / "docs"

    def _find_page(self) -> pathlib.Path:
        candidates = list(self.DOC_DIR.rglob("*retry*.mdx"))
        assert candidates, (
            f"no retry-semantics page found under {self.DOC_DIR}; "
            "expected a file with 'retry' in its name"
        )
        # Prefer concepts/ over reference/ when both exist.
        candidates.sort(key=lambda p: (0 if "concepts" in p.parts else 1, str(p)))
        return candidates[0]

    def test_retry_page_exists_and_covers_three_layers(self):
        page = self._find_page()
        text = page.read_text()

        required = {
            "llm_factory": "transient-API layer reference (user's llm_factory)",
            "with_retry": "concrete model.with_retry(...) example",
            "LlmConfig.max_retries": "output-quality layer reference",
            "scripted": "flaky-external-in-scripted layer reference",
        }
        missing = [k for k in required if k not in text]
        assert missing == [], (
            f"\n{page} is missing required mentions: {missing}\n"
            f"Required terms map: {required}"
        )

    def test_retry_page_states_compile_correctness_rationale(self):
        page = self._find_page()
        text = page.read_text().lower()

        # Either phrasing of the correctness argument: act-mode tools, or
        # double-write / replay reasoning.
        ok = (
            ("act" in text and ("replay" in text or "tool" in text))
            or "double-write" in text
            or "double write" in text
        )
        assert ok, (
            f"{page.name} does not state the correctness rationale for not "
            "exposing retry_policy at compile() (expected mention of 'act'-mode "
            "tool replay / double-write)."
        )


class TestNeoStateKeysCentralized:
    """Guard — `neo_*` state-bus key literals must live in `_state_keys.py`.

    Per `docs/design/architecture-decisions.md` §7 (State bus): `neo_*` state-bus
    keys are framework-internal field names on the LangGraph state dict. A typo
    in any read site is a silent runtime miss — LangGraph returns `None` for an
    unknown key with no error. Centralizing the keys in a single module
    eliminates that whole class of bug.

    This guard AST-scans every `src/neograph/*.py` (except `_state_keys.py`)
    for string-literal fragments matching `^neo_`. Both plain `"neo_..."`
    constants and f-string fragments like `f"neo_loop_count_{field}"` are
    detected.

    If this test fails, replace the literal with the named constant or builder
    from `neograph._state_keys` — do NOT add an allowlist entry.
    """

    PATTERN = re.compile(r"^neo_")

    @staticmethod
    def _string_fragments(node: ast.AST):
        """Yield every literal string fragment in an AST node.

        Covers plain `ast.Constant(str)` and the string parts of an
        `ast.JoinedStr` (f-string). For f-strings, only the *literal* fragments
        are yielded (the surrounding text), so `f"neo_loop_count_{field}"`
        yields `"neo_loop_count_"`.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    yield part.value

    @classmethod
    def _scan(cls, path: pathlib.Path) -> list[tuple[int, str]]:
        """Return a list of (lineno, fragment) for every offending literal."""
        source = path.read_text()
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            return []

        hits: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            for frag in cls._string_fragments(node):
                if cls.PATTERN.match(frag):
                    hits.append((node.lineno, frag))
        return hits

    def test_no_neo_state_key_literals_outside_state_keys_module(self):
        violations: list[str] = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            if py_file.name == "_state_keys.py":
                continue
            for lineno, frag in self._scan(py_file):
                violations.append(f"  {py_file.name}:{lineno}: '{frag}'")

        assert violations == [], (
            f"\n{len(violations)} `neo_*` literal(s) found outside "
            "_state_keys.py:\n"
            + "\n".join(violations)
            + "\n\nReplace each with the named constant or builder from "
            "`neograph._state_keys`."
        )

    def test_guard_detects_injected_violation(self, tmp_path):
        """Mutation case: synthesize a file with a forbidden literal and
        confirm the scanner detects it. Guarantees the guard cannot rot."""
        bad = tmp_path / "synthetic.py"
        bad.write_text('FORBIDDEN = "neo_typo_injected"\n')
        hits = self._scan(bad)
        assert hits, "scanner failed to detect injected `neo_*` literal"
        assert any("neo_typo_injected" in h[1] for h in hits)

    def test_guard_detects_fstring_violation(self, tmp_path):
        """Mutation case: f-string fragment starting with `neo_` is detected."""
        bad = tmp_path / "synthetic_fstring.py"
        bad.write_text('def f(x):\n    return f"neo_synthetic_{x}"\n')
        hits = self._scan(bad)
        assert hits, "scanner failed to detect injected `neo_*` f-string fragment"
        assert any("neo_synthetic_" in h[1] for h in hits)


class TestNoLlmModuleGlobals:
    """LLM runtime configuration must not live in module-level mutable state.

    Per docs/design/architecture-decisions.md §2: compile() reads inputs from
    keyword arguments and closes them over into factory closures. Six
    module-level mutables in `_llm.py` (`_llm_factory`, `_llm_factory_params`,
    `_prompt_compiler`, `_prompt_compiler_params`, `_global_renderer`,
    `_cost_callback`) violated that — two `compile()` calls in the same
    process could collide.

    This guard AST-scans `_llm.py` and asserts that none of those names
    appear as module-level Assign targets.
    """

    FORBIDDEN_NAMES = frozenset({
        "_llm_factory",
        "_llm_factory_params",
        "_prompt_compiler",
        "_prompt_compiler_params",
        "_global_renderer",
        "_cost_callback",
    })

    def test_no_module_level_llm_globals(self):
        llm_path = SRC_DIR / "_llm.py"
        source = llm_path.read_text()
        tree = ast.parse(source, filename=str(llm_path))

        violations: list[str] = []
        for node in tree.body:  # top-level only — not ast.walk
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Name) and tgt.id in self.FORBIDDEN_NAMES:
                    violations.append(f"  _llm.py:{node.lineno}: {tgt.id}")

        assert violations == [], (
            f"\n{len(violations)} module-level LLM global(s) remain in _llm.py:\n"
            + "\n".join(violations)
            + "\n\nMove these into a closure-captured LlmRuntime per §2."
        )

    def test_no_global_in_configure_llm(self):
        """The `global` statement listing those names must also be gone."""
        llm_path = SRC_DIR / "_llm.py"
        source = llm_path.read_text()
        tree = ast.parse(source, filename=str(llm_path))

        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                offenders = [n for n in node.names if n in self.FORBIDDEN_NAMES]
                if offenders:
                    violations.append(
                        f"  _llm.py:{node.lineno}: global {', '.join(offenders)}"
                    )

        assert violations == [], (
            f"\n{len(violations)} `global` statement(s) target forbidden LLM names:\n"
            + "\n".join(violations)
        )


class TestNoModuleLevelRegistration:
    """Public registration API must not exist on the `neograph` package surface.

    Per docs/design/architecture-decisions.md §2: `compile()` is the sole
    entry point for runtime configuration. The legacy module-level helpers
    `configure_llm`, `register_scripted`, `register_condition`, and
    `register_tool_factory` are removed in ticket `neograph-ezqz`.

    This guard AST-scans `src/neograph/` for top-level function definitions
    bearing any of those names AND checks `__init__.py` for re-exports.
    """

    FORBIDDEN_NAMES = frozenset({
        "configure_llm",
        "register_scripted",
        "register_condition",
        "register_tool_factory",
    })

    def test_no_top_level_function_defs(self):
        """No `src/neograph/*.py` should define `configure_llm` at module level.

        `register_scripted`/`register_condition`/`register_tool_factory` MAY
        appear in `decorators.py` (internal use by `@node`/`@tool` decorators
        that need to capture inline shims at decoration time) — they are NOT
        exported from `neograph/__init__.py`. The `test_no_reexport_in_init`
        guard enforces the public-surface boundary instead.
        """
        violations: list[str] = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == "configure_llm":
                        violations.append(f"  {py_file.name}:{node.lineno}: def {node.name}")
        assert violations == [], (
            f"\n{len(violations)} forbidden `configure_llm` definition(s) in src/:\n"
            + "\n".join(violations)
            + "\n\nPass llm_factory/prompt_compiler as kwargs to compile() instead."
        )

    def test_no_reexport_in_init(self):
        """`neograph/__init__.py` must not import or re-export the helpers."""
        path = SRC_DIR / "__init__.py"
        text = path.read_text()
        violations: list[str] = []
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for name in self.FORBIDDEN_NAMES:
                if name in line and ("import" in line or '"' + name + '"' in line):
                    violations.append(f"  __init__.py:{i}: {stripped[:90]}")
                    break
        assert violations == [], (
            f"\n{len(violations)} re-export(s) of removed helpers in __init__.py:\n"
            + "\n".join(violations)
        )

    def test_configure_llm_no_longer_importable(self):
        """`from neograph import configure_llm` must raise ImportError."""
        import importlib

        with pytest.raises((ImportError, AttributeError)):
            mod = importlib.import_module("neograph")
            _ = mod.configure_llm  # noqa


class TestNoGlobalRegistry:
    """`_registry.py` must not hold a process-global singleton.

    Per docs/design/architecture-decisions.md §2: `compile()` builds a
    fresh per-compile Registry; factory closures capture that instance.
    Two `compile()` calls produce two independent registries that cannot
    collide.

    This guard AST-scans `_registry.py` for the module-level
    `registry = Registry()` instantiation. The `Registry` class itself
    may remain as a per-compile container.
    """

    def test_no_module_level_registry_singleton(self):
        path = SRC_DIR / "_registry.py"
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))

        violations: list[str] = []
        for node in tree.body:  # top-level only
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if (
                        isinstance(tgt, ast.Name)
                        and tgt.id == "registry"
                        and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "Registry"
                    ):
                        violations.append(
                            f"  _registry.py:{node.lineno}: registry = Registry()"
                        )

        assert violations == [], (
            f"\n{len(violations)} module-level Registry singleton(s) remain:\n"
            + "\n".join(violations)
            + "\n\nBuild a fresh Registry per `compile()` and thread it via closures."
        )


