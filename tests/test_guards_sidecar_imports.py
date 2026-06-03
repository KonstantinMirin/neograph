"""Structural guards: sidecar module, function-local import allowlist,
tool-loop import graph, langgraph imports, IO polymorphism."""

from __future__ import annotations

import ast
import pathlib
import re

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# Error classes that must use .build() instead of direct construction.
ERROR_CLASSES = frozenset({
    "ConstructError",
    "ExecutionError",
    "CompileError",
    "ConfigurationError",
    "NeographError",
})


class TestSidecarModule:
    """_sidecar.py must exist and _construct_builder.py must NOT import from decorators.py.

    TASK neograph-s0cu: extract sidecar storage to break circular import.
    """

    def test_sidecar_module_exists(self):
        """_sidecar.py must exist as a standalone module."""
        sidecar_path = SRC_DIR / "_sidecar.py"
        assert sidecar_path.exists(), (
            "_sidecar.py does not exist — sidecar storage functions "
            "must be extracted from decorators.py"
        )

    def test_construct_builder_does_not_import_from_decorators(self):
        """_construct_builder.py must import sidecar functions from _sidecar, not decorators."""
        builder_path = SRC_DIR / "_construct_builder.py"
        tree = ast.parse(builder_path.read_text())
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "neograph.decorators":
                names = [a.name for a in node.names]
                violations.append(
                    f"  _construct_builder.py:{node.lineno}: from neograph.decorators import {', '.join(names)}"
                )
        assert violations == [], (
            "\n_construct_builder.py still imports from decorators.py:\n"
            + "\n".join(violations)
            + "\n\nThese should import from neograph._sidecar instead."
        )


# Allowlist of accepted function-local `from neograph...` imports.
#
# Per docs/design/architecture-decisions.md §4: function-local imports break
# import cycles and are tolerated but not preferred. The list shrinks as cycle
# break-up tickets land — it must never grow. Each entry names the cycle (or
# CLI-deferral justification) and the ticket that will retire it.
#
# Key shape: (file_relative_to_src_neograph, imported_module, frozenset_of_names).
# Names are included so a NEW import in an allowlisted file/module pairing
# still trips the guard.
FUNCTION_LOCAL_IMPORT_ALLOWLIST: set[tuple[str, str, frozenset[str]]] = {
    # __main__.py — CLI command bodies defer heavy graph imports to keep
    # `neograph --help` startup fast. Justification: import latency, not cycle.
    ("__main__.py", "neograph.compiler", frozenset({"compile"})),
    ("__main__.py", "neograph.errors", frozenset({"CompileError", "ConstructError"})),
    ("__main__.py", "neograph.lint", frozenset({"lint"})),
    ("__main__.py", "neograph.compiler", frozenset({"classify_modifiers"})),
    ("__main__.py", "neograph.testing", frozenset({"scaffold_tests"})),
    ("__main__.py", "neograph", frozenset({"__version__"})),
    # _construct_builder.py — cycle: _construct_builder imports validation/factory
    # which themselves transitively touch decorator metadata. Break tracked by
    # the §4 epic (neograph-pgso).
    (
        "_construct_builder.py",
        "neograph._construct_validation",
        frozenset({"_types_compatible", "effective_producer_type"}),
    ),
    # _construct_validation.py — REAL cycle: get_merge_fn_metadata lives in the
    # leaf module _sidecar, but a module-level import here cycles via
    # _sidecar -> _di_classify -> _construct_validation (_di_classify imports
    # ConstructError from this module). Function-local import to the leaf is the
    # truthful fix (neograph-v3xx / HIGH-09). Retires when _di_classify stops
    # importing ConstructError from _construct_validation.
    (
        "_construct_validation.py",
        "neograph._sidecar",
        frozenset({"get_merge_fn_metadata"}),
    ),
    # _construct_validation.py — cycle: validation shares the fan-out candidate
    # rule with the normalizer, but _ir_normalize -> _sidecar -> _di_classify ->
    # _construct_validation forms a cycle. Function-local import keeps the shared
    # rule single-sourced without the cycle (neograph-k7bg). Retires when
    # _di_classify no longer imports ConstructError from _construct_validation.
    (
        "_construct_validation.py",
        "neograph._ir_normalize",
        frozenset({"fan_out_candidates"}),
    ),
    # _llm_runtime.py — cycle: check_llm_kwargs_or_raise walks the construct
    # for LLM-mode nodes, but Construct/Node themselves import from _llm_runtime
    # (via factory.py through compiler.py). Function-local imports keep
    # _llm_runtime a leaf. Retires when the construct-walking helper moves
    # to a non-leaf module (e.g., a new _llm_check.py).
    ("_llm_runtime.py", "neograph.construct", frozenset({"Construct"})),
    ("_llm_runtime.py", "neograph.errors", frozenset({"CompileError"})),
    ("_llm_runtime.py", "neograph.node", frozenset({"Node"})),
    # NOTE: _llm.py describe_type / renderers entries retired by neograph-8ne2.
    # The split made both leaf modules importable at module level — render_prompt
    # moved into _llm_render where describe_type + build_rendered_input are
    # safe module-level imports, and the slim _llm.py imports describe_type
    # directly for the json_mode schema branch.
    # _oracle.py — cycle: oracle merges call invoke_structured (_llm). Retires
    # when §2 -lyvi collapses _llm globals.
    ("_oracle.py", "neograph._llm", frozenset({"invoke_structured"})),
    # NOTE (ARCH-4 / neograph-v3xx / HIGH-09): the former _oracle -> decorators
    # function-local imports of get_merge_fn_metadata + _resolve_merge_args were
    # illusory cycles — those symbols live in the leaf modules _sidecar and
    # _di_classify (decorators only re-exported them). _oracle now imports them
    # at MODULE level from the leaves (_oracle -> _sidecar -> _di_classify ->
    # _construct_validation never reaches _oracle, so no cycle). Both allowlist
    # entries deleted.
    # _sidecar.py — infer_oracle_gen_type peeks the decoration-time scripted
    # registry to type-infer Oracle's per-generator output. The registry now
    # lives in the leaf _runtime_registry (neograph-v3xx HIGH-01), so this is no
    # longer a decorators cycle; the import stays function-local only to defer
    # the registry import to call time. Retires if made module-level.
    ("_sidecar.py", "neograph._runtime_registry", frozenset({"_decoration_registry"})),
    # NOTE (ARCH-4 / neograph-v3xx): the _wiring.py -> factory function-local
    # import of make_eachoracle_redirect_fn was removed when factory.py's
    # test-only re-export shims were deleted (HIGH-08). _wiring.py now imports
    # make_eachoracle_redirect_fn at module level from _oracle (already a
    # module-level dependency), so no cycle and no allowlist entry is needed.
    # NOTE (ARCH-1 / neograph-s0iz): the merge_fn metadata + invoke_structured
    # function-local imports left _wiring.py when the Oracle merge algorithm was
    # consolidated into _oracle._merge_variants. _wiring.py now imports that
    # kernel at module level and performs no merge step, so the former
    # decorators/_llm function-local allowlist entries are retired here.
    ("_wiring.py", "neograph.compiler", frozenset({"compile"})),
    # _subconstruct.py — cycle: sub-construct invocation strips internal fields
    # that runner.py owns. Inherited from factory.py when make_subgraph_fn moved
    # out (gm-4). Will retire when runner.py loses its internal-field knowledge.
    ("_subconstruct.py", "neograph.runner", frozenset({"_strip_internals"})),
    # modifiers.py — cycle: Loop validation lives in _construct_validation,
    # which imports modifier types. Function-local import keeps modifiers.py a
    # leaf. Retires when validation rules move out of _construct_validation.
    (
        "modifiers.py",
        "neograph._construct_validation",
        frozenset({"validate_loop_self_edge"}),
    ),
    (
        "modifiers.py",
        "neograph._construct_validation",
        frozenset({"validate_loop_construct"}),
    ),
    # node.py — cycle: Node.compile() ultimately calls make_node_fn; raising a
    # typed ConstructError requires errors.py which transitively re-touches node.
    # Both retire when Node loses its compile()/run() convenience methods.
    ("node.py", "neograph.errors", frozenset({"ConstructError"})),
    ("node.py", "neograph.factory", frozenset({"make_node_fn"})),
    # node.py — Node.run_isolated() falls back to decoration-time defaults for
    # scripted/tool-factory lookups when the caller hasn't passed scripted=/
    # tool_factories= kwargs. The registry now lives in the leaf
    # _runtime_registry (neograph-v3xx HIGH-01); the import stays function-local
    # inside run_isolated alongside its other deferred imports.
    (
        "node.py",
        "neograph._runtime_registry",
        frozenset({"_decoration_registry"}),
    ),
    # state.py — cycle: state.py owns field naming logic which naming.py wraps
    # for legacy callers. Trivial; can be flattened anytime.
    ("state.py", "neograph.naming", frozenset({"field_name_for"})),
    # tool.py — @tool registers the tool factory into the decoration-time
    # registry (leaf _runtime_registry, neograph-v3xx HIGH-01). Function-local
    # import inside the decorator defers the registry import to decoration time.
    ("tool.py", "neograph._runtime_registry", frozenset({"register_tool_factory"})),
}


def _scan_function_local_neograph_imports(
    src_dir: pathlib.Path,
) -> list[tuple[str, int, str, frozenset[str]]]:
    """Walk every .py file under src_dir and return function-local `from neograph...`
    imports as (filename, lineno, module, frozenset_of_imported_names).

    An import is function-local iff it appears inside a (possibly nested)
    FunctionDef or AsyncFunctionDef body. Class-body and module-level imports
    are NOT flagged. TYPE_CHECKING blocks at module top-level are also not
    flagged (they live in If nodes outside FunctionDef).
    """

    class _Scanner(ast.NodeVisitor):
        def __init__(self, filename: str) -> None:
            self.filename = filename
            self._depth = 0
            self.found: list[tuple[str, int, str, frozenset[str]]] = []

        def visit_FunctionDef(self, fn: ast.FunctionDef) -> None:
            self._depth += 1
            try:
                self.generic_visit(fn)
            finally:
                self._depth -= 1

        def visit_AsyncFunctionDef(self, fn: ast.AsyncFunctionDef) -> None:
            self._depth += 1
            try:
                self.generic_visit(fn)
            finally:
                self._depth -= 1

        def visit_ImportFrom(self, imp: ast.ImportFrom) -> None:
            if self._depth > 0 and imp.module and (
                imp.module.startswith("neograph.") or imp.module == "neograph"
            ):
                self.found.append(
                    (
                        self.filename,
                        imp.lineno,
                        imp.module,
                        frozenset(a.name for a in imp.names),
                    )
                )

    out: list[tuple[str, int, str, frozenset[str]]] = []
    for py_file in sorted(src_dir.glob("*.py")):
        scanner = _Scanner(py_file.name)
        scanner.visit(ast.parse(py_file.read_text()))
        out.extend(scanner.found)
    return out


class TestFunctionLocalImportAllowlist:
    """Function-local `from neograph...` imports must be explicitly allowlisted.

    Replaces the old numeric budget guard (which counted indented imports
    without naming them). Per docs/design/architecture-decisions.md §4, every
    function-local `from neograph...` import is an unresolved cycle. The
    allowlist freezes today's snapshot with one entry per (file, module, names)
    tuple and a comment naming the cycle it breaks. Cycle-break tickets in the
    §4 epic shrink the list; new entries are forbidden without an accompanying
    justification comment.

    Mutation-verified: removing an entry from the allowlist while keeping the
    import in src/ makes the main scan fail with that exact tuple in the diff.
    Adding a synthetic function-local `from neograph._llm import X` to a temp
    file (passed directly to the scanner) flags it. The allowlist is the
    backlog: the goal is len == 0.
    """

    def test_no_unallowlisted_function_local_neograph_imports(self):
        """Every function-local `from neograph...` import must be in the allowlist."""
        observed = _scan_function_local_neograph_imports(SRC_DIR)
        observed_keys = {(f, m, n) for (f, _, m, n) in observed}

        unauthorized = sorted(observed_keys - {(f, m, n) for (f, m, n) in FUNCTION_LOCAL_IMPORT_ALLOWLIST})
        stale = sorted({(f, m, n) for (f, m, n) in FUNCTION_LOCAL_IMPORT_ALLOWLIST} - observed_keys)

        msg_parts = []
        if unauthorized:
            msg_parts.append(
                f"{len(unauthorized)} NEW function-local `from neograph...` import(s) "
                "not in the allowlist:\n"
                + "\n".join(
                    f"  {f}: from {m} import {', '.join(sorted(n))}"
                    for (f, m, n) in unauthorized
                )
                + "\n\nFunction-local imports are accepted only to break a cycle. "
                "Either restructure to eliminate the cycle, or add the entry to "
                "FUNCTION_LOCAL_IMPORT_ALLOWLIST with a justification comment "
                "naming the cycle and the ticket that will retire it."
            )
        if stale:
            msg_parts.append(
                f"{len(stale)} stale allowlist entry/entries (no matching import in src/):\n"
                + "\n".join(
                    f"  {f}: from {m} import {', '.join(sorted(n))}"
                    for (f, m, n) in stale
                )
                + "\n\nRemove from FUNCTION_LOCAL_IMPORT_ALLOWLIST — the cycle is broken."
            )
        assert not msg_parts, "\n\n".join(msg_parts)

    def test_scanner_detects_injected_function_local_neograph_import(
        self, tmp_path: pathlib.Path
    ):
        """Mutation: a synthetic module with a function-local import must be flagged."""
        synthetic_dir = tmp_path / "neograph_fake"
        synthetic_dir.mkdir()
        synthetic = synthetic_dir / "fakemod.py"
        synthetic.write_text(
            "from __future__ import annotations\n"
            "\n"
            "def make_node():\n"
            "    from neograph._llm import invoke_structured\n"
            "    return invoke_structured\n"
        )
        found = _scan_function_local_neograph_imports(synthetic_dir)
        assert any(
            m == "neograph._llm" and "invoke_structured" in n for (_, _, m, n) in found
        ), f"scanner failed to detect injected function-local import; found={found}"

    def test_scanner_ignores_module_level_and_type_checking_imports(
        self, tmp_path: pathlib.Path
    ):
        """Module-level imports and TYPE_CHECKING-block imports must NOT be flagged."""
        synthetic_dir = tmp_path / "neograph_fake"
        synthetic_dir.mkdir()
        synthetic = synthetic_dir / "fakemod.py"
        synthetic.write_text(
            "from __future__ import annotations\n"
            "from typing import TYPE_CHECKING\n"
            "from neograph._llm import invoke_structured\n"
            "\n"
            "if TYPE_CHECKING:\n"
            "    from neograph.factory import lookup_scripted\n"
            "\n"
            "def make_node():\n"
            "    return invoke_structured\n"
        )
        found = _scan_function_local_neograph_imports(synthetic_dir)
        assert found == [], (
            f"scanner false-positive on module-level / TYPE_CHECKING imports; found={found}"
        )


class TestNoBareExceptException:
    """No bare 'except Exception' that swallows errors silently.

    TASK neograph-bjin: 13 bare except Exception handlers found.
    Every except must either:
    - Catch a specific exception tuple (NameError, TypeError, etc.)
    - Re-raise as a typed error (except Exception as exc: raise ... from exc)
    - Be explicitly marked with # noqa: bare-except (must be justified)
    """

    def test_no_bare_except_exception_pass(self):
        """No 'except Exception:' followed by pass/return None/empty dict."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            source = py_file.read_text()
            tree = ast.parse(source)
            lines = source.splitlines()
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                # Only flag bare "except Exception" (not "except Exception as exc")
                if (
                    node.type is not None
                    and isinstance(node.type, ast.Name)
                    and node.type.id == "Exception"
                    and node.name is None  # no "as exc"
                ):
                    # Check if the line has a noqa comment
                    line_text = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    if "noqa: bare-except" in line_text:
                        continue
                    violations.append(
                        f"  {py_file.name}:{node.lineno}: bare except Exception (no 'as exc', no specific types)"
                    )
        assert violations == [], (
            f"\n{len(violations)} bare 'except Exception:' handler(s) found.\n"
            + "\n".join(violations)
            + "\n\nFix: catch specific exceptions, or use 'except Exception as exc:' and re-raise/log."
        )


class TestLLMModuleSymbolsMovedToToolLoop:
    """neograph-smjo: confirm _tool_loop.py split is stable.

    After the _llm.py -> _tool_loop.py extraction (commit 63ada61), the
    tool-loop symbols must not reappear as module-level definitions in
    _llm.py. A regression here means someone reintroduced the cycle.

    Mutation-verified: adding `def invoke_with_tools(...)` back to _llm.py
    makes this test fail with the exact symbol name in the error.
    """

    FORBIDDEN = {
        "invoke_with_tools",
        "_CoercingToolWrapper",
        "_render_tool_result_for_llm",
        "_safe_tool_invoke",
    }

    def test_forbidden_names_absent_from_llm_module(self):
        src = pathlib.Path("src/neograph/_llm.py").read_text()
        tree = ast.parse(src)
        top_level_names: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                top_level_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        top_level_names.add(target.id)
        leaked = self.FORBIDDEN & top_level_names
        assert not leaked, (
            f"_llm.py defines symbols that must live in _tool_loop.py: {sorted(leaked)}. "
            f"Move them back to _tool_loop.py or update the guard."
        )


class TestToolLoopImportGraph:
    """neograph-b3nh: _tool_loop.py has a strictly one-way dependency on _llm.py.

    Importing from higher layers (dispatch/factory/compiler/construct) would
    create a cycle and violate the layering documented in AGENTS.md.

    Mutation-verified: adding `from neograph.factory import register_scripted`
    to _tool_loop.py makes this test fail reporting neograph.factory.
    """

    FORBIDDEN_MODULES = {
        "neograph._dispatch",
        "neograph.factory",
        "neograph.compiler",
        "neograph.construct",
    }

    def test_tool_loop_does_not_import_from_higher_layers(self):
        src = pathlib.Path("src/neograph/_tool_loop.py").read_text()
        tree = ast.parse(src)
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
        leaked = self.FORBIDDEN_MODULES & imports
        assert not leaked, (
            f"_tool_loop.py imports from higher layers: {sorted(leaked)}. "
            f"Layer discipline: _tool_loop -> _llm (one-way)."
        )


class TestNoPrivateLanggraphImports:
    """LangGraph is a committed dependency, but only its PUBLIC API is allowed.

    Per docs/design/architecture-decisions.md §1: any `_`-prefixed module
    segment in a `langgraph.*` import path is private and forbidden. Examples:

    Forbidden:
        from langgraph._internal._serde import build_serde_allowlist
        from langgraph.checkpoint.serde._msgpack import SAFE_MSGPACK_TYPES
        import langgraph._internal

    Allowed:
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Send, Command, interrupt
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    Private modules can break or vanish between LangGraph versions without
    notice. If you need behavior only available there, file a beads task to
    upstream a public API or vendor the logic locally — do not import it.
    """

    def test_no_private_langgraph_imports(self):
        violations = []

        for py_file in sorted(SRC_DIR.glob("*.py")):
            source = py_file.read_text()
            try:
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                # `from langgraph.<...> import ...`
                if isinstance(node, ast.ImportFrom) and node.module:
                    if not node.module.startswith("langgraph"):
                        continue
                    segments = node.module.split(".")
                    private = [s for s in segments if s.startswith("_")]
                    if private:
                        violations.append(
                            f"  {py_file.name}:{node.lineno}: "
                            f"from {node.module} import ... "
                            f"(private segment(s): {', '.join(private)})"
                        )
                # `import langgraph.<...>` / `import langgraph.<...> as <...>`
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.name
                        if not name.startswith("langgraph"):
                            continue
                        segments = name.split(".")
                        private = [s for s in segments if s.startswith("_")]
                        if private:
                            violations.append(
                                f"  {py_file.name}:{node.lineno}: "
                                f"import {name} "
                                f"(private segment(s): {', '.join(private)})"
                            )

        assert violations == [], (
            f"\n{len(violations)} private langgraph import(s):\n"
            + "\n".join(violations)
            + "\n\nLangGraph private APIs (`_`-prefixed module segments) are "
            "off-limits — they can break between versions without notice. "
            "Use only public `langgraph.<public>` paths. "
            "See docs/design/architecture-decisions.md §1."
        )


# ═══════════════════════════════════════════════════════════════════════════
# TEST: neograph-l2vr -- Tool.config remains pass-through (no framework reads)
# ═══════════════════════════════════════════════════════════════════════════
class TestToolConfigOnlyPassedPositionally:
    """neograph-l2vr / mlxg: Tool.config is a dynamic dict pass-through to
    user-supplied tool factories.

    Mlxg's audit established that the framework never reads keys from
    tool.config / tool_spec.config -- it forwards the dict verbatim to the
    factory at _tool_loop.py:205. This guard rejects any future regression
    where a helper grows a tool_spec.config[...] / tool.config.get(...)
    read, which would silently re-introduce the same bug class that
    pej0/rnjw closed for llm_config.
    """

    # Named (PROC-2) so the regex set carries a slip meta-test. Both the
    # subscript (`[`) and the `.get(` read forms must be caught for both
    # `tool` and `tool_spec` receivers.
    _TOOL_CONFIG_READ_RES = (
        re.compile(r"\btool_spec\.config\["),
        re.compile(r"\btool_spec\.config\.get\("),
        re.compile(r"\btool\.config\["),
        re.compile(r"\btool\.config\.get\("),
    )

    def test_no_framework_reads_on_tool_spec_config(self):
        from pathlib import Path

        src_dir = Path(__file__).resolve().parents[1] / "src" / "neograph"

        violations: list[str] = []
        for py_file in src_dir.rglob("*.py"):
            text = py_file.read_text()
            for line_no, line in enumerate(text.splitlines(), start=1):
                # Skip docstring-style comments and example blocks.
                stripped = line.lstrip()
                if stripped.startswith(("#", "//", "*", '"', "'")):
                    continue
                for pattern in self._TOOL_CONFIG_READ_RES:
                    if pattern.search(line):
                        violations.append(
                            f"{py_file.name}:{line_no} -- {line.strip()}"
                        )

        assert not violations, (
            "Framework code now reads keys from tool_spec.config / tool.config. "
            "mlxg accepted Tool.config as a dynamic pass-through dict; any "
            "framework-side .get() / [...] access reintroduces the typo / "
            "default-drift bug class that LlmConfig closed for llm_config. "
            f"Violations:\n{chr(10).join(violations)}"
        )

    def test_slip_tool_config_read_res(self):
        """Regex-slip: BOTH read forms (`[` subscript and `.get(`) for BOTH
        receivers (`tool`, `tool_spec`) must be caught, and the `\\b` word
        boundary must not let a longer attribute name (e.g. `mytool.config[`)
        masquerade or be missed. Prove each form matches and an unrelated
        `.config` access on a different object does not."""
        def matched(s: str) -> bool:
            return any(p.search(s) for p in self._TOOL_CONFIG_READ_RES)

        assert matched("x = tool.config['k']")
        assert matched("x = tool.config.get('k')")
        assert matched("x = tool_spec.config['k']")
        assert matched("x = tool_spec.config.get('k')")
        # Word boundary: a different object's .config read is NOT flagged.
        assert not matched("x = llm.config['k']")
        # Pass-through (positional forward of the whole dict) is NOT a key read.
        assert not matched("factory(tool.config)")


# ═══════════════════════════════════════════════════════════════════════════
# TEST: neograph-mqr3 -- Node.outputs / Node.inputs polymorphism is normalized
# ═══════════════════════════════════════════════════════════════════════════
class TestNodeIOPolymorphismNormalized:
    """neograph-mqr3: ``Node.outputs`` / ``Node.inputs`` are polymorphic
    (``type | dict[str, type] | None``). The discrimination must happen in
    exactly one place — ``src/neograph/_normalize.py``. Every other module
    consumes the normalized view (``NormalizedOutputs`` / ``NormalizedInputs``)
    instead of re-deriving the trichotomy via ``isinstance(...,  dict)``.

    Why: 18+ ``isinstance(node.outputs, dict)`` / ``isinstance(node.inputs, dict)``
    sites existed before the normalizer. Each was a place where mypy couldn't
    help (TypeSpec is statically Any) and where a future fourth form would
    silently slip through. Centralizing keeps the polymorphism in one spot.
    """

    def test_no_outputs_dict_isinstance_outside_normalizer(self):
        violations = self._scan_isinstance("outputs")
        assert not violations, (
            f"\n{len(violations)} isinstance(<expr>.outputs, dict) site(s) outside _normalize.py:\n"
            + "\n".join(violations)
            + "\n\nUse `normalize_outputs(node.outputs)` from neograph._normalize "
              "and consume the NormalizedOutputs view (primary / primary_key / "
              "secondary / all_keys / is_dict_form / is_none) instead."
        )

    def test_no_inputs_dict_isinstance_outside_normalizer(self):
        violations = self._scan_isinstance("inputs")
        assert not violations, (
            f"\n{len(violations)} isinstance(<expr>.inputs, dict) site(s) outside _normalize.py:\n"
            + "\n".join(violations)
            + "\n\nUse `normalize_inputs(node.inputs)` from neograph._normalize "
              "and consume the NormalizedInputs view (by_name / single_type / "
              "is_dict_form / is_none) instead."
        )

    @staticmethod
    def _scan_isinstance(attr: str) -> list[str]:
        """AST-scan src/neograph/*.py for isinstance(<expr>.{attr}, dict).

        Skips _normalize.py (the only legitimate location).
        """
        violations: list[str] = []
        for py_file in sorted(SRC_DIR.rglob("*.py")):
            if py_file.name == "_normalize.py":
                continue
            try:
                tree = ast.parse(py_file.read_text(), filename=str(py_file))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                # Only isinstance(...)
                if not (isinstance(func, ast.Name) and func.id == "isinstance"):
                    continue
                # Need exactly two args, where:
                #   arg[0] is an Attribute access ending in `.<attr>`
                #   arg[1] mentions ``dict``
                if len(node.args) != 2:
                    continue
                target, classinfo = node.args
                if not (isinstance(target, ast.Attribute) and target.attr == attr):
                    continue
                if not TestNodeIOPolymorphismNormalized._mentions_dict(classinfo):
                    continue
                violations.append(
                    f"  {py_file.relative_to(SRC_DIR.parent)}:{node.lineno}: "
                    f"isinstance(<expr>.{attr}, dict)"
                )
        return violations

    @staticmethod
    def _mentions_dict(node: ast.expr) -> bool:
        """True if the classinfo arg of isinstance references the builtin ``dict``."""
        if isinstance(node, ast.Name) and node.id == "dict":
            return True
        if isinstance(node, ast.Tuple):
            return any(TestNodeIOPolymorphismNormalized._mentions_dict(e) for e in node.elts)
        return False


# Modules in scope for the Any audit. These are the IR public APIs and the
# dispatch/wiring layers where user-supplied data should be wrapped in typed
# containers (NodeInput / NodeOutput) rather than passed as bare Any.
