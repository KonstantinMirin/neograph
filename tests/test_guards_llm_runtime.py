"""Structural guards: factory kwargs, LLM responsibility discipline, LLM
scenario/cohesion, StateBus.get discipline, runtime fan-out, normalize_ir
field writer, routing-key invariant."""

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


class TestFactoryFunctionsTakeKwargs:
    """Factory functions must not import the removed LLM globals.

    Per docs/design/architecture-decisions.md §2: factory functions
    (`make_node_fn`, `make_subgraph_fn`, `make_oracle_*`) close over the
    `LlmRuntime` bundle passed at compile time instead of reading from
    module-level state in `_llm.py`. This guard AST-scans `factory.py`
    and `_oracle.py` for any import of the six forbidden names.
    """

    FORBIDDEN_NAMES = frozenset({
        "_llm_factory",
        "_llm_factory_params",
        "_prompt_compiler",
        "_prompt_compiler_params",
        "_global_renderer",
        "_cost_callback",
        "_get_global_renderer",
    })

    FACTORY_FILES = ("factory.py", "_oracle.py", "_dispatch.py", "_execute.py",
                     "_state_write.py", "_input_shape.py", "_subconstruct.py",
                     "_wiring.py")

    def test_factory_files_do_not_import_forbidden_names(self):
        violations: list[str] = []
        for fname in self.FACTORY_FILES:
            path = SRC_DIR / fname
            if not path.exists():  # pragma: no cover
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and "neograph._llm" in node.module:
                        for alias in node.names:
                            if alias.name in self.FORBIDDEN_NAMES:
                                violations.append(
                                    f"  {fname}:{node.lineno}: "
                                    f"from {node.module} import {alias.name}"
                                )

        assert violations == [], (
            f"\n{len(violations)} factory file(s) still import removed LLM globals:\n"
            + "\n".join(violations)
            + "\n\nClose over the LlmRuntime passed at compile time instead."
        )

    def test_factory_files_do_not_use_get_global_renderer(self):
        """No factory file should call `_get_global_renderer()` — that name is gone."""
        violations: list[str] = []
        for fname in self.FACTORY_FILES:
            path = SRC_DIR / fname
            if not path.exists():  # pragma: no cover
                continue
            text = path.read_text()
            if "_get_global_renderer" in text:
                for i, line in enumerate(text.splitlines(), 1):
                    if "_get_global_renderer" in line and not line.lstrip().startswith("#"):
                        violations.append(f"  {fname}:{i}: {line.strip()[:80]}")

        assert violations == [], (
            f"\n{len(violations)} reference(s) to `_get_global_renderer` remain:\n"
            + "\n".join(violations)
            + "\n\nUse `runtime.renderer` (closure-captured) instead."
        )

    def test_no_id_keying_in_construct_builder(self):
        """`_register_node_scripted` must NOT use `id(n)` to mint shim keys.

        `id(n)` is recycled by Python and silently shadows shims when two
        Nodes happen to land at the same memory address. Use a fresh
        `secrets.token_hex` value per shim (or a per-compile dict — which
        eliminates the collision risk entirely).
        """
        path = SRC_DIR / "_construct_builder.py"
        text = path.read_text()
        violations: list[str] = []
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # Reject `id(n)` or `id(node)` patterns used to mint shim names.
            if "id(n)" in line or "id(node)" in line:
                violations.append(f"  _construct_builder.py:{i}: {stripped[:80]}")
        assert violations == [], (
            f"\n{len(violations)} `id()`-keyed shim name(s) remain:\n"
            + "\n".join(violations)
            + "\n\nUse a per-compile dict or `secrets.token_hex` for shim names."
        )

    def test_factory_files_do_not_reach_into_compat_slot(self):
        """No factory file should read from `_legacy_config.get_compat_runtime`.

        Factory functions must receive their runtime as a parameter, not
        consult the legacy compat slot directly. Reading from `get_compat_runtime`
        in a factory closure would re-introduce the same multi-tenant
        contamination risk the §2 work eliminated — even if the call sits
        downstream of a closure.

        Only `compile()` itself (in `compiler.py`) and the deprecated public
        wrappers (`Node.run_isolated`, `render_prompt`) may consult the slot —
        everything else should take a runtime parameter.
        """
        violations: list[str] = []
        for fname in self.FACTORY_FILES:
            path = SRC_DIR / fname
            if not path.exists():  # pragma: no cover
                continue
            text = path.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                if "get_compat_runtime" in line and not line.lstrip().startswith("#"):
                    violations.append(f"  {fname}:{i}: {line.strip()[:80]}")

        assert violations == [], (
            f"\n{len(violations)} factory file(s) reach into the compat runtime "
            "slot:\n" + "\n".join(violations)
            + "\n\nThread the runtime as a parameter instead — the compat slot "
              "is reserved for the deprecated `configure_llm()` bridge."
        )


class TestLlmResponsibilityDiscipline:
    """neograph-8ne2 / §4: _llm.py split into responsibility-aligned modules.

    Per docs/design/architecture-decisions.md §4 (Module and function
    responsibilities): _llm.py mixed six responsibilities (cost callback,
    LLM resolution, prompt rendering, JSON parsing/retry, output-strategy
    dispatch, produce-mode orchestration). This guard pins the post-split
    layout so the god-module cannot re-form by accretion.

    Change-axis clusters (each scenario forces exactly one file):

    | Scenario                              | Owning file       |
    |---------------------------------------|-------------------|
    | Add new output_strategy               | _llm_dispatch.py  |
    | Add new JSON retry rule / repair      | _llm_retry.py     |
    | Add new prompt-render output (image,  | _llm_render.py    |
    |   streaming, etc.)                    |                   |
    | Add new produce-mode orchestration    | _llm.py           |
    |   step (cost, telemetry, LLM resolve) |                   |

    Mutation-verified: moving `_extract_json` back into `_llm.py` makes
    `test_llm_module_only_defines_allowed_names` fail naming `_extract_json`.
    """

    # Single-responsibility name allowlist per module. Names that appear here
    # MUST be defined at module top-level in the named file; names that don't
    # appear MUST NOT be top-level definitions there.
    ALLOWED_NAMES = {
        "_llm.py": frozenset({
            # produce-mode orchestrator + runtime adapters it owns
            "invoke_structured",
            "_get_llm",
            "_notify_cost",
            # neograph-w74k.2.3 (Phase 1c): async twin + the pure preamble/
            # postamble helpers both orchestrators share (anti-drift extraction).
            "ainvoke_structured",
            "_prepare_structured_call",
            "_finish_structured_call",
        }),
        "_llm_dispatch.py": frozenset({
            "_call_structured",
            # neograph-w74k.2.3: async twin of the strategy dispatch.
            "_acall_structured",
        }),
        # neograph-ble3: DSML detection extracted to its own pure leaf module.
        "_dsml.py": frozenset({
            "contains_dsml",
            "message_text",
        }),
        # neograph-ble3: provider-quirk compat shim — StructuredResult tagged
        # union + Protocol-based adapter chain. New provider quirks are new
        # decorator classes HERE, never new branches in _call_structured.
        "_llm_structured_compat.py": frozenset({
            "Parsed",
            "Raw",
            "Failed",
            "StructuredOutputAdapter",
            "LangChainStructuredAdapter",
            "IncludeRawCompatDecorator",
            "DsmlClassifierDecorator",
            "build_default_adapter",
            # neograph-w74k.2.3: pure classifiers shared by the sync/async adapters.
            "_classify_lc_result",
            "_reclassify_dsml",
        }),
        "_llm_retry.py": frozenset({
            "_extract_json",
            "_extract_balanced",
            "_is_list_annotation",
            "_apply_null_defaults",
            "_parse_json_response",
            "_build_retry_msg",
            "_invoke_json_with_retry",
            # neograph-ble3: DSML recovery (re-prompt + re-parse) is the RETRY
            # side of the DSML story; detection moved to _dsml. Renamed from
            # _attempt_dsml_recovery to recover_dsml.
            "recover_dsml",
            # neograph-w74k.2.3: async twins of the retry + DSML-recovery seams.
            "_ainvoke_json_with_retry",
            "arecover_dsml",
        }),
        "_llm_render.py": frozenset({
            "_is_inline_prompt",
            "_compile_multimodal_prompt",
            "_resolve_var",
            "_resolve_var_raw",
            "_substitute_vars",
            "_compile_prompt",
            "render_prompt",
        }),
    }

    # Coarse line-count budget. Not the load-bearing assertion (the name set
    # is); a proxy that catches accretion that escapes name-level review.
    # neograph-w74k.2.3 (Phase 1c): budgets raised to accommodate the async
    # twins across the LLM vertical. The twins are thin over shared pure
    # preamble/postamble/classify helpers (anti-drift), but adding an awaiting
    # mirror of each orchestrator is a real, reviewed line increase — not
    # accretion. The name-set assertion remains the load-bearing guard.
    LINE_BUDGETS = {
        "_llm.py": 290,
        # neograph-ble3: tightened 130 -> 115. The 5-path include_raw try/except
        # ladder collapsed to a match on the StructuredResult variant; the
        # provider-quirk wiring moved to the compat shim. Locks the deletion.
        "_llm_dispatch.py": 200,
        # neograph-ble3: tightened 365 -> 360. _DSML_PATTERN regex moved to
        # _dsml.py; recover_dsml is detection-free. Locks the deletion.
        # neograph-s1u4: 360 -> 375. _apply_null_defaults gained a guarded
        # default_factory coercion branch (a real fix, not accretion).
        "_llm_retry.py": 480,
        "_llm_render.py": 310,
        # neograph-ble3: new pure-leaf detection module.
        "_dsml.py": 55,
        # neograph-ble3: compat shim — sum-type + Protocol + 3 adapters + factory.
        "_llm_structured_compat.py": 220,
    }

    def _top_level_defs(self, path: pathlib.Path) -> set[str]:
        """Collect top-level function / class names. Module-level data
        assignments (regex constants, logger handles, sentinels) are not
        responsibilities — they're configuration that supports the named
        functions and may live anywhere those functions need them.
        """
        tree = ast.parse(path.read_text())
        names: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(node.name)
        return names

    def test_all_split_modules_exist(self):
        missing = [name for name in self.ALLOWED_NAMES if not (SRC_DIR / name).exists()]
        assert not missing, (
            f"Split modules do not exist yet: {missing}. "
            "neograph-8ne2 requires each responsibility module to be created."
        )

    def test_llm_module_only_defines_allowed_names(self):
        for fname, allowed in self.ALLOWED_NAMES.items():
            path = SRC_DIR / fname
            if not path.exists():
                pytest.fail(f"{fname} does not exist (neograph-8ne2 split not complete).")
            top = self._top_level_defs(path)
            extra = top - allowed
            missing = allowed - top
            assert not extra, (
                f"{fname} defines unexpected top-level names: {sorted(extra)}. "
                f"Allowed: {sorted(allowed)}. "
                "Move them to their change-axis cluster module."
            )
            assert not missing, (
                f"{fname} is missing required top-level names: {sorted(missing)}. "
                f"Required: {sorted(allowed)}."
            )

    def test_line_count_budgets(self):
        violations: list[str] = []
        for fname, budget in self.LINE_BUDGETS.items():
            path = SRC_DIR / fname
            if not path.exists():
                violations.append(f"  {fname}: file missing")
                continue
            n = len(path.read_text().splitlines())
            if n > budget:
                violations.append(f"  {fname}: {n} lines (budget {budget})")
        assert not violations, (
            "Line-count budgets exceeded — coarse proxy for accretion:\n"
            + "\n".join(violations)
        )

    def test_no_cycles_among_split_modules(self):
        """The five _llm* modules must form a DAG.

        Allowed edges (a -> b means a imports from b):
            _llm.py        -> _llm_dispatch, _llm_render, _llm_config,
                              _llm_protocols, _llm_runtime
            _llm_dispatch  -> _llm_retry, _llm_structured_compat
            _llm_structured_compat -> _dsml (pure leaf, not _llm*-prefixed)
            _llm_retry     -> _dsml (no other _llm* deps except possibly _llm_config)
            _llm_render    -> _llm_config, _llm_protocols, _llm_runtime
            _llm_protocols -> (no _llm* deps)
            _llm_runtime   -> (no _llm* deps; uses TYPE_CHECKING for protocols)
            _llm_config    -> (no _llm* deps)

        neograph-ble3: _dsml.py is a pure leaf (no neograph._llm* imports) so it
        cannot participate in any cycle; it is intentionally omitted from the
        _llm*-prefixed collector below. _llm_structured_compat.py IS tracked.
        """
        modules = [
            "_llm.py",
            "_llm_dispatch.py",
            "_llm_structured_compat.py",
            "_llm_retry.py",
            "_llm_render.py",
            "_llm_protocols.py",
            "_llm_runtime.py",
            "_llm_config.py",
        ]
        def _collect_runtime_imports(tree: ast.AST) -> set[str]:
            """Walk imports, skipping anything inside an `if TYPE_CHECKING:` block."""
            tc_blocks: list[ast.If] = []
            for n in ast.walk(tree):
                if isinstance(n, ast.If):
                    test = n.test
                    is_tc = (
                        (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
                        or (
                            isinstance(test, ast.Attribute)
                            and test.attr == "TYPE_CHECKING"
                        )
                    )
                    if is_tc:
                        tc_blocks.append(n)

            def in_tc(node: ast.AST) -> bool:
                for blk in tc_blocks:
                    for sub in ast.walk(blk):
                        if sub is node:
                            return True
                return False

            deps: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if not node.module.startswith("neograph._llm"):
                        continue
                    if in_tc(node):
                        continue
                    leaf = node.module.split(".")[-1] + ".py"
                    deps.add(leaf)
            return deps

        graph: dict[str, set[str]] = {}
        for fname in modules:
            path = SRC_DIR / fname
            if not path.exists():
                continue
            tree = ast.parse(path.read_text())
            deps = {d for d in _collect_runtime_imports(tree) if d in modules and d != fname}
            graph[fname] = deps

        # Cycle detection via DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color = dict.fromkeys(graph, WHITE)
        cycle: list[str] = []

        def dfs(node: str, path: list[str]) -> bool:
            color[node] = GRAY
            for dep in graph.get(node, set()):
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    idx = path.index(dep) if dep in path else 0
                    cycle.extend(path[idx:] + [dep])
                    return True
                if color[dep] == WHITE and dfs(dep, path + [dep]):
                    return True
            color[node] = BLACK
            return False

        for m in graph:
            if color[m] == WHITE and dfs(m, [m]):
                break
        assert not cycle, f"Cycle detected among _llm* modules: {' -> '.join(cycle)}"

    def test_no_function_local_llm_split_imports(self):
        """No spin-off module may use function-local `from neograph._llm*` imports.

        Mutation-verified: adding `from neograph._llm_retry import x` inside
        a function in _llm_dispatch.py makes this guard fire — symptomatic
        of a hidden cycle the layout was supposed to prevent.
        """
        violations: list[str] = []
        for fname in (
            "_llm_dispatch.py", "_llm_retry.py", "_llm_render.py",
            "_llm_structured_compat.py",  # neograph-ble3
        ):
            path = SRC_DIR / fname
            if not path.exists():
                continue
            tree = ast.parse(path.read_text())
            for func in ast.walk(tree):
                if not isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                for sub in ast.walk(func):
                    if isinstance(sub, ast.ImportFrom) and sub is not func:
                        if sub.module and sub.module.startswith("neograph._llm"):
                            # The walk above includes the function's own toplevel
                            # imports; filter to ones inside the body specifically.
                            if sub in func.body:
                                continue  # toplevel of function body (rare)
                            violations.append(
                                f"  {fname}:{sub.lineno}: from {sub.module} import ..."
                            )
        assert not violations, (
            "Function-local `from neograph._llm*` import detected in spin-off module:\n"
            + "\n".join(violations)
            + "\n\nResolve the underlying cycle by reshaping the layout, "
              "not by hiding the import inside a function body."
        )


class TestLlmScenarioTouchpoints:
    """neograph-8ne2 / §4: Cohesion property test — scenario walkthrough.

    Architectural teeth for the responsibility-cluster split: each scenario
    enumerates which files MUST change and which files MAY change. If a
    future change forces a file outside the may_touch set, the cluster has
    leaked responsibility.

    The point is not to forbid touching other files (tests, importers will
    always need updates for renames). The point is to bound the *source*
    files implementing the new behaviour.

    Mutation-verified by `test_mutation_excess_scenario_touchpoints`.
    """

    SCENARIO_TOUCHPOINTS = {
        # "Add new output_strategy (e.g., dsml, function_calling)"
        # → strategy dispatch only.
        "add_new_output_strategy": {
            "must_touch": {"_llm_dispatch.py"},
            "may_touch": {"_llm_config.py"},
            "max_touch": 2,
        },
        # "Add new retry rule (e.g., schema mismatch → reformat)"
        # → retry module only.
        "add_new_retry_rule": {
            "must_touch": {"_llm_retry.py"},
            "may_touch": {"_llm_config.py"},
            "max_touch": 2,
        },
        # "Add new prompt rendering output (e.g., audio block, streaming)"
        # → render module only.
        "add_new_prompt_render_output": {
            "must_touch": {"_llm_render.py"},
            "may_touch": set(),
            "max_touch": 1,
        },
        # "Add new produce-mode orchestration step (e.g., pre-call hook)"
        # → orchestrator only.
        "add_new_orchestration_step": {
            "must_touch": {"_llm.py"},
            "may_touch": set(),
            "max_touch": 1,
        },
        # neograph-ble3: "Add new provider quirk (e.g., provider returns empty
        # string on overload)" → a new StructuredOutputAdapter decorator in the
        # compat shim ONLY. This is the Open/Closed proof: a new quirk is a new
        # decorator class, never a new branch in _call_structured.
        "add_new_provider_quirk": {
            "must_touch": {"_llm_structured_compat.py"},
            "may_touch": set(),
            "max_touch": 1,
        },
    }

    def test_all_must_touch_files_exist(self):
        missing: list[str] = []
        for scenario, spec in self.SCENARIO_TOUCHPOINTS.items():
            for fname in (spec["must_touch"] | spec["may_touch"]):
                if not (SRC_DIR / fname).exists():
                    missing.append(f"  {scenario}: {fname} does not exist")
        assert not missing, "\n".join(missing)

    def test_max_touch_bounded(self):
        for scenario, spec in self.SCENARIO_TOUCHPOINTS.items():
            total = len(spec["must_touch"]) + len(spec["may_touch"])
            assert total <= spec["max_touch"], (
                f"Scenario '{scenario}' touches {total} files; max {spec['max_touch']}."
            )

    def test_mutation_excess_scenario_touchpoints(self):
        """Mutation case — synthesize an oversized scenario, scanner detects."""
        bad = {
            "must_touch": {"a.py", "b.py", "c.py"},
            "may_touch": set(),
            "max_touch": 1,
        }
        total = len(bad["must_touch"]) + len(bad["may_touch"])
        assert total > bad["max_touch"]


class TestLlmCohesionFanOut:
    """neograph-8ne2 / §4: Cohesion fan-out for the _llm* spin-off modules.

    For each spin-off module, count distinct external src/neograph modules
    that import from it. Spin-off modules are leaf-ish helpers; they should
    have few importers. A bloated import set signals the spin-off has
    become a kitchen sink.

    Mutation-verified: synthesizing 4 importers of a fictitious leaf module
    exceeds a ceiling of 2.
    """

    FAN_OUT_CEILING = {
        # _llm_dispatch is a leaf strategy table; called by _llm + _tool_loop.
        "_llm_dispatch.py": 3,
        # _llm_retry is a leaf parser; called by _llm_dispatch + _tool_loop.
        "_llm_retry.py": 4,
        # _llm_render owns prompt compilation + public render_prompt;
        # called by _llm, _tool_loop, _dispatch (for _is_inline_prompt),
        # and re-exported via __init__.
        "_llm_render.py": 5,
        # neograph-ble3: _dsml is the shared detection leaf — imported by
        # _llm_dispatch, _llm_structured_compat, _llm_retry, _tool_loop.
        "_dsml.py": 5,
        # neograph-ble3: compat shim, called only by the dispatch table.
        "_llm_structured_compat.py": 2,
    }

    def _count_importers(self, target_module_basename: str) -> list[str]:
        target = f"neograph.{target_module_basename[:-3]}"
        importers: list[str] = []
        for py in sorted(SRC_DIR.glob("*.py")):
            if py.name == target_module_basename:
                continue
            text = py.read_text()
            if (
                f"from {target} " in text
                or f"from {target}\n" in text
                or f"import {target}\n" in text
            ):
                importers.append(py.name)
        return importers

    def test_fan_out_under_ceiling(self):
        violations: list[str] = []
        for mod, ceiling in self.FAN_OUT_CEILING.items():
            if not (SRC_DIR / mod).exists():
                continue
            importers = self._count_importers(mod)
            if len(importers) > ceiling:
                violations.append(
                    f"  {mod}: {len(importers)} importers (ceiling {ceiling}): "
                    f"{importers}"
                )
        assert not violations, (
            "Spin-off module fan-out exceeded ceiling:\n" + "\n".join(violations)
        )

    def test_mutation_excess_importers_detected(self, tmp_path):
        for i in range(4):
            (tmp_path / f"client_{i}.py").write_text("from neograph.leaf import x\n")
        count = sum(
            1 for p in tmp_path.glob("client_*.py")
            if "from neograph.leaf " in p.read_text()
        )
        assert count > 2



class TestStateBusGetDiscipline:
    """Every .get(...) call on a StateBus receiver in src/neograph/ must
    either use ``.get_required(...)`` (§7 fail-loud) OR carry a
    ``# StateBus.get optional: <reason>`` justification comment on the
    same line or the immediately-preceding line.

    Per neograph-tzzi (izo1-D). Mutation tests verify the scanner.
    """

    # Filenames whose StateBus.get sites have been audited (izo1-B). The
    # scanner only enforces discipline on these files; new modules can be
    # added once their sites are classified per the audit doc.
    IN_SCOPE = frozenset({
        "_input_shape.py",
        "_execute.py",
        "_oracle.py",
        "_subconstruct.py",
        "_wiring.py",
        "_state_write.py",
    })

    # Variable names that we treat as "this expression is a StateBus".
    BUS_NAMES = frozenset({"bus", "state", "_state"})

    OPTIONAL_TAG = "StateBus.get optional:"

    @classmethod
    def _scan(cls, src_dir: pathlib.Path) -> list[str]:
        offenders: list[str] = []
        for py_file in sorted(src_dir.glob("*.py")):
            if py_file.name not in cls.IN_SCOPE:
                continue
            text = py_file.read_text()
            lines = text.splitlines()
            try:
                tree = ast.parse(text, filename=str(py_file))
            except SyntaxError:  # pragma: no cover — file must parse
                continue
            for ast_node in ast.walk(tree):
                if not isinstance(ast_node, ast.Call):
                    continue
                func = ast_node.func
                if not isinstance(func, ast.Attribute):
                    continue
                if func.attr != "get":
                    continue
                value = func.value
                # Receiver must be a bare Name matching the StateBus naming
                # convention used in the codebase.
                if not isinstance(value, ast.Name):
                    continue
                if value.id not in cls.BUS_NAMES:
                    continue
                lineno = ast_node.lineno
                if 0 < lineno <= len(lines) and cls.OPTIONAL_TAG in lines[lineno - 1]:
                    continue
                annotated = False
                cursor = lineno - 2
                while cursor >= 0 and lines[cursor].lstrip().startswith("#"):
                    if cls.OPTIONAL_TAG in lines[cursor]:
                        annotated = True
                        break
                    cursor -= 1
                if annotated:
                    continue
                offenders.append(f"{py_file.name}:{lineno}")
        return offenders

    def test_every_state_bus_get_is_annotated(self):
        offenders = self._scan(SRC_DIR)
        assert offenders == [], (
            f"\n{len(offenders)} unannotated StateBus.get site(s) found:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nEither switch to .get_required(...) (§7 required-getter) "
              "or add a `# StateBus.get optional: <reason>` comment on the "
              "call site or the immediately-preceding line."
        )

    def test_mutation_unannotated_get_detected(self, tmp_path: pathlib.Path):
        """Inject an unannotated `bus.get(k)` into a scoped temp file;
        scanner must flag it."""
        target = tmp_path / "_input_shape.py"
        target.write_text(
            "def f(bus):\n"
            "    return bus.get('k')\n"
        )
        offenders = self._scan(tmp_path)
        assert any(o.endswith(":2") for o in offenders), offenders

    def test_mutation_annotated_get_passes(self, tmp_path: pathlib.Path):
        """`bus.get(k, None)  # StateBus.get optional: test` → scanner skips."""
        target = tmp_path / "_input_shape.py"
        target.write_text(
            "def f(bus):\n"
            "    return bus.get('k', None)  # StateBus.get optional: test reason\n"
        )
        offenders = self._scan(tmp_path)
        assert offenders == [], offenders

    def test_mutation_get_required_passes(self, tmp_path: pathlib.Path):
        """`bus.get_required(k)` (not `.get`) is never flagged."""
        target = tmp_path / "_input_shape.py"
        target.write_text(
            "def f(bus):\n"
            "    return bus.get_required('k')\n"
        )
        offenders = self._scan(tmp_path)
        assert offenders == [], offenders

    def test_mutation_out_of_scope_file_skipped(self, tmp_path: pathlib.Path):
        """A file outside IN_SCOPE is not scanned."""
        target = tmp_path / "not_audited.py"
        target.write_text(
            "def f(bus):\n"
            "    return bus.get('k')\n"
        )
        offenders = self._scan(tmp_path)
        assert offenders == [], offenders


class TestNoRawStateAccessInRoutingModules:
    """ARCH-2 (neograph-aiiz): the routing/boundary modules must read pipeline
    state ONLY through the StateBus protocol. Three raw forms are structurally
    forbidden on a bare ``state`` name:

      1. ``getattr(state, ...)``               -- use ``bus.get`` / ``get_required``
      2. ``state.__class__.model_fields`` (any ``state.__class__`` access)
      3. ``state[...]`` subscript              -- use ``bus.get``

    The full-state snapshot lives in exactly one helper (``snapshot_state`` in
    ``_state_bus.py``, which goes through ``bus.keys()``/``bus.get``); the
    Each-router navigation+dedup lives in exactly one helper
    (``_collect_each_items`` in ``_wiring.py``). Neither re-derives raw access.

    Scope is the four routing modules ONLY -- ``_state_bus.py`` is the StateBus
    implementation and legitimately uses ``getattr(self._state, ...)`` and
    ``self._state.__class__.model_fields``; it is NOT in scope. ``adapt_state(state)``
    call-args are not a banned form (``state`` is an argument, not a receiver).

    Value-navigation (``getattr(item, key)``, ``getattr(value, attr)``) is
    excluded because the scanner keys on ``Name(id='state')`` only.
    """

    IN_SCOPE = frozenset({
        "_wiring.py",
        "_oracle.py",
        "_subconstruct.py",
        "_state_write.py",
    })

    @classmethod
    def _scan(cls, src_dir: pathlib.Path) -> list[str]:
        offenders: list[str] = []
        for py_file in sorted(src_dir.glob("*.py")):
            if py_file.name not in cls.IN_SCOPE:
                continue
            text = py_file.read_text()
            try:
                tree = ast.parse(text, filename=str(py_file))
            except SyntaxError:  # pragma: no cover — file must parse
                continue
            for ast_node in ast.walk(tree):
                # Form 1: getattr(state, ...)
                if (
                    isinstance(ast_node, ast.Call)
                    and isinstance(ast_node.func, ast.Name)
                    and ast_node.func.id == "getattr"
                    and ast_node.args
                    and isinstance(ast_node.args[0], ast.Name)
                    and ast_node.args[0].id == "state"
                ):
                    offenders.append(f"{py_file.name}:{ast_node.lineno} getattr(state,...)")
                # Form 2: state.__class__ (covers state.__class__.model_fields)
                if (
                    isinstance(ast_node, ast.Attribute)
                    and ast_node.attr == "__class__"
                    and isinstance(ast_node.value, ast.Name)
                    and ast_node.value.id == "state"
                ):
                    offenders.append(f"{py_file.name}:{ast_node.lineno} state.__class__")
                # Form 3: state[...] subscript
                if (
                    isinstance(ast_node, ast.Subscript)
                    and isinstance(ast_node.value, ast.Name)
                    and ast_node.value.id == "state"
                ):
                    offenders.append(f"{py_file.name}:{ast_node.lineno} state[...]")
        return offenders

    def test_no_raw_state_access_in_routing_modules(self):
        offenders = self._scan(SRC_DIR)
        assert offenders == [], (
            f"\n{len(offenders)} raw-state-access site(s) found in routing modules:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nRoute reads through StateBus: adapt_state(state) then bus.get / "
              "get_required / get_counter / keys. Use snapshot_state(bus) for the "
              "full snapshot and _collect_each_items(bus, each, fan_out=...) for "
              "Each-router navigation+dedup."
        )

    def test_mutation_getattr_state_detected(self, tmp_path: pathlib.Path):
        """A probe ``getattr(state, 'x')`` in a scoped temp file is flagged."""
        (tmp_path / "_wiring.py").write_text(
            "def f(state):\n"
            "    return getattr(state, 'x')\n"
        )
        offenders = self._scan(tmp_path)
        assert any("getattr(state,...)" in o for o in offenders), offenders

    def test_mutation_model_fields_enumeration_detected(self, tmp_path: pathlib.Path):
        """The regex-slip case: ``state.__class__.model_fields`` enumeration."""
        (tmp_path / "_oracle.py").write_text(
            "def f(state):\n"
            "    return {k: 1 for k in state.__class__.model_fields}\n"
        )
        offenders = self._scan(tmp_path)
        assert any("state.__class__" in o for o in offenders), offenders

    def test_mutation_subscript_detected(self, tmp_path: pathlib.Path):
        """A raw ``state[root]`` subscript is flagged."""
        (tmp_path / "_state_write.py").write_text(
            "def f(state):\n"
            "    return state['root']\n"
        )
        offenders = self._scan(tmp_path)
        assert any("state[...]" in o for o in offenders), offenders

    def test_mutation_statebus_calls_pass(self, tmp_path: pathlib.Path):
        """StateBus method calls and adapt_state(state) args are NOT flagged."""
        (tmp_path / "_subconstruct.py").write_text(
            "def f(state):\n"
            "    bus = adapt_state(state)\n"
            "    a = bus.get('k')\n"
            "    b = bus.get_required('k')\n"
            "    c = getattr(item, 'key', None)\n"
            "    return a, b, c\n"
        )
        offenders = self._scan(tmp_path)
        assert offenders == [], offenders

    def test_mutation_out_of_scope_file_skipped(self, tmp_path: pathlib.Path):
        """``_state_bus.py`` (the StateBus impl) is out of scope and skipped."""
        (tmp_path / "_state_bus.py").write_text(
            "def f(state):\n"
            "    return getattr(state, 'x')\n"
        )
        offenders = self._scan(tmp_path)
        assert offenders == [], offenders


class TestNoRuntimeFanOutDetection:
    """neograph-vgc1: ``neograph._ir_normalize.normalize_ir`` is the single
    source of truth for ``node.fan_out_param``. The runtime extractor at
    ``_input_shape._extract_fan_in_dict`` must NOT detect fan-out via
    ``state.keys()`` scanning — that band-aid existed during the izo1-B
    migration and was removed once the construct-time normalization landed.
    Reintroducing it splits the rule across two layers and re-creates the
    drift class (cf. neograph-8k3, neograph-ayq).
    """

    def test_input_shape_does_not_call_state_keys_for_fan_out_detection(self):
        source = (SRC_DIR / "_input_shape.py").read_text()
        tree = ast.parse(source)

        # Find _extract_fan_in_dict body.
        target_fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_extract_fan_in_dict":
                target_fn = node
                break
        assert target_fn is not None, "_extract_fan_in_dict must exist"

        # Within its body, no `state.keys()` call.
        for sub in ast.walk(target_fn):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "keys"
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id in ("state", "bus")
            ):
                raise AssertionError(
                    f"_extract_fan_in_dict re-introduced state.keys() scan at "
                    f"line {sub.lineno}; fan-out detection belongs in "
                    f"neograph._ir_normalize.normalize_ir (single source of truth, "
                    f"neograph-vgc1)."
                )


class TestNormalizeIrIsSoleIrFieldWriter:
    """neograph-20xq: IR-level inferred fields (fan_out_param, oracle_gen_type)
    are written at exactly one ASSEMBLY-TIME site — neograph._ir_normalize,
    called once from Construct.__init__. This replaces the older
    TestConstructNormalizesEveryAtNodeOnlyIRField, which only policed
    _construct_builder's `updates` dict and required a parallel
    Construct._normalize_<field> method per field (the very drift mechanism
    the epic removed).

    The invariant is structural, not a snapshot: rather than enumerate the
    expected writes, the guard scans ALL of src/neograph for EVERY syntactic
    form that writes one of the IR fields — attribute assignment, subscript
    assignment, a key in a model_copy/dict-literal update, and
    setattr/object.__setattr__ — and asserts the (file, field) write set
    equals a tightly-justified allowlist:

      _ir_normalize.py  → {fan_out_param, oracle_gen_type}  (the canonical site;
            normalize_ir is the SOLE writer of fan_out_param after neograph-k7bg)
      decorators.py → {oracle_gen_type}   (@node DECORATION-time eager
            pre-population so a bare Node carries the type before assembly;
            normalize_ir owns the inference rule via oracle_gen_type_for and
            is idempotent over it)

    The remaining pre-population site (decorators.py) runs BEFORE the Construct
    object exists. Any other
    (file, field) pair — e.g. re-adding oracle_gen_type inference to
    _construct_builder, or a new Construct._normalize_* method — fails the
    guard. So does any write of a NEW IR field anywhere but _ir_normalize.
    """

    # IR-level inferred fields owned by normalize_ir.
    IR_FIELDS = frozenset({"fan_out_param", "oracle_gen_type"})

    # Sanctioned (file, field) pre-population writes outside _ir_normalize.
    # After neograph-k7bg, _construct_builder no longer writes fan_out_param —
    # the normalizer is its sole writer. Only the @node decoration-time eager
    # oracle_gen_type write remains.
    ALLOWED_PREPOP: dict[str, frozenset[str]] = {
        "decorators.py": frozenset({"oracle_gen_type"}),
    }

    @classmethod
    def _scan_ir_field_writes(cls, tree: ast.AST) -> set[str]:
        """Return the set of IR field names WRITTEN anywhere in ``tree``.

        Detects every write form for an IR field on an *instance*:
          - attribute-assign            ``x.field = ...``
          - subscript-assign            ``d["field"] = ...``
          - augmented-assign            ``x.field += ...`` / ``d["field"] += ...``
          - annotated-assign            ``x.field: T = ...`` / ``d["field"]: T = ...``
          - tuple/list-unpack targets   ``x.field, y = a, b``
          - dict-literal keys           ``{"field": ...}`` (covers
            ``model_copy(update={"field": ...})`` and a normalizer's
            ``return {"field": ...}``)
          - dict() keyword args         ``dict(field=...)`` (the kwarg form of
            the same ``model_copy(update=...)`` idiom)
          - ``setattr``/``object.__setattr__`` string-literal targets

        Deliberately NOT matched (not writes to an instance field):
          - bare-``Name`` annotated/plain assignments — these are class-body
            field DECLARATIONS (``fan_out_param: str | None = None`` in
            ``node.py``) or module constants, not instance writes.
          - read forms — ``getattr(x, "field")``, ``x.field`` access.
        """
        written: set[str] = set()

        def _str_const(node: ast.AST) -> str | None:
            return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

        def _check_target(target: ast.AST) -> None:
            """Record an assignment target if it writes an IR field on an
            instance. Recurses tuple/list unpacking. Bare ``Name`` targets are
            ignored (declaration/constant, not an instance write)."""
            if isinstance(target, ast.Attribute):
                if target.attr in cls.IR_FIELDS:
                    written.add(target.attr)
            elif isinstance(target, ast.Subscript):
                key = _str_const(target.slice)
                if key in cls.IR_FIELDS:
                    written.add(key)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    _check_target(elt)

        for node in ast.walk(tree):
            # x.field = ... | d["field"] = ... | a, b = ... (tuple targets)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    _check_target(target)
            # x.field += ... | d["field"] += ...
            elif isinstance(node, ast.AugAssign):
                _check_target(node.target)
            # x.field: T = ... | d["field"]: T = ...  (Name targets ignored above)
            elif isinstance(node, ast.AnnAssign):
                _check_target(node.target)
            # {"field": ...} dict literals (model_copy update=, normalizer return)
            elif isinstance(node, ast.Dict):
                for key_node in node.keys:
                    if key_node is not None:
                        key = _str_const(key_node)
                        if key in cls.IR_FIELDS:
                            written.add(key)
            elif isinstance(node, ast.Call):
                # setattr(x, "field", ...) / object.__setattr__(x, "field", ...)
                if (
                    (isinstance(node.func, ast.Attribute) and node.func.attr == "__setattr__")
                    or (isinstance(node.func, ast.Name) and node.func.id == "setattr")
                ):
                    for arg in node.args:
                        key = _str_const(arg)
                        if key in cls.IR_FIELDS:
                            written.add(key)
                # dict(field=...) — kwarg form of model_copy(update=dict(...))
                elif isinstance(node.func, ast.Name) and node.func.id == "dict":
                    for kw in node.keywords:
                        if kw.arg in cls.IR_FIELDS:
                            written.add(kw.arg)
        return written

    def test_ir_fields_written_only_in_normalize_ir_and_sanctioned_prepop(self):
        offenders: list[str] = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            written = self._scan_ir_field_writes(ast.parse(py_file.read_text()))
            if not written:
                continue
            name = py_file.name
            if name == "_ir_normalize.py":
                continue  # the canonical site may write any IR field
            allowed = self.ALLOWED_PREPOP.get(name, frozenset())
            unexpected = written - allowed
            for field in sorted(unexpected):
                offenders.append(f"{name}: writes IR field '{field}'")

        assert offenders == [], (
            "IR-level inferred fields must be written only by "
            "neograph._ir_normalize (the single assembly-time normalization "
            "site) or by a sanctioned @node pre-population site listed in "
            "ALLOWED_PREPOP.\nUnsanctioned writes:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nIf this is a new IR inference, add it as an IrNormalizer in "
            "_ir_normalize.py — do NOT inline it in an assembly path (that "
            "re-creates the drift class of neograph-vgc1/aqau/20xq)."
        )

    # Named so the regex carries a slip meta-test (PROC-2). Matches the exact
    # set of removed per-field Construct methods.
    _NORMALIZE_FIELD_RE = re.compile(r"_normalize_(fan_out_params|oracle_gen_type)")

    def test_construct_has_no_normalize_field_methods(self):
        """The per-field Construct._normalize_<field> methods are GONE — the
        whole point of the epic. A new one would re-introduce parallel
        inference. Guards against regression to the old shape."""
        construct_src = (SRC_DIR / "construct.py").read_text()
        tree = ast.parse(construct_src)
        offenders = [
            n.name for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and self._NORMALIZE_FIELD_RE.fullmatch(n.name)
        ]
        assert offenders == [], (
            f"construct.py defines {offenders}; IR-field inference belongs in "
            f"_ir_normalize.IrNormalizer implementations, not per-field methods "
            f"on Construct (neograph-20xq)."
        )

    def test_slip_normalize_field_re(self):
        """Regex-slip: fullmatch (not search) is what makes this precise — a
        method merely CONTAINING the name (e.g. a helper or a longer name) must
        NOT be flagged, while the exact removed names must be."""
        assert self._NORMALIZE_FIELD_RE.fullmatch("_normalize_fan_out_params")
        assert self._NORMALIZE_FIELD_RE.fullmatch("_normalize_oracle_gen_type")
        # Slip guard: substring/longer forms would match re.search but NOT
        # fullmatch — proving the fullmatch anchoring is load-bearing.
        assert not self._NORMALIZE_FIELD_RE.fullmatch("_normalize_fan_out_params_v2")
        assert not self._NORMALIZE_FIELD_RE.fullmatch("_normalize_inputs")

    def test_construct_init_calls_normalize_ir(self):
        """Construct.__init__ must delegate to normalize_ir — proves the single
        site is actually wired in, not just present."""
        construct_src = (SRC_DIR / "construct.py").read_text()
        assert "normalize_ir(self)" in construct_src, (
            "Construct.__init__ must call normalize_ir(self) so all three API "
            "surfaces converge on identical IR before validation (neograph-20xq)."
        )

    def test_scanner_detects_every_write_form(self):
        """Mutation: the scanner must catch EVERY write form it claims to, so
        the guard is not a dead snapshot. Each form below must surface the
        field; if any line stops being detected, the real guard would silently
        pass a regression that writes an IR field via that form."""
        forms = {
            "attribute": 'node.fan_out_param = "x"',
            "subscript": 'd["fan_out_param"] = "x"',
            "aug-assign-attr": 'node.fan_out_param += "x"',
            "aug-assign-subscript": 'd["fan_out_param"] += "x"',
            "ann-assign-attr": 'node.fan_out_param: str = "x"',
            "ann-assign-subscript": 'd["fan_out_param"]: str = "x"',
            "tuple-unpack": 'node.fan_out_param, y = a, b',
            "list-unpack": '[node.fan_out_param, y] = a, b',
            "dict-literal": 'node.model_copy(update={"fan_out_param": "y"})',
            "dict-kwarg": 'node.model_copy(update=dict(fan_out_param="y"))',
            "setattr": 'setattr(node, "fan_out_param", "x")',
            "object-setattr": 'object.__setattr__(node, "fan_out_param", T)',
        }
        for label, src in forms.items():
            written = self._scan_ir_field_writes(ast.parse(src))
            assert "fan_out_param" in written, (
                f"scanner missed write form {label!r} ({src!r}); detected "
                f"{sorted(written)}. The guard's docstring claims this form is "
                f"covered — a regression using it would slip past silently."
            )

    def test_scanner_ignores_class_body_field_declarations(self):
        """A class-body field declaration (``fan_out_param: str | None = None``)
        and a module-level annotated constant are NOT instance writes and must
        NOT be flagged — otherwise the guard would fail against node.py's own
        Node field definitions and _ir_normalize's _NORMALIZERS constant."""
        synthetic = (
            "class Node:\n"
            "    fan_out_param: str | None = None\n"
            "    oracle_gen_type: object = None\n"
            "_NORMALIZERS: list = []\n"
        )
        written = self._scan_ir_field_writes(ast.parse(synthetic))
        assert written == set(), (
            f"scanner flagged a class-body declaration as a write: {sorted(written)}; "
            f"bare-Name annotated/plain targets must be ignored."
        )

    def test_scanner_ignores_read_forms(self):
        """getattr/attribute-read of an IR field is NOT a write and must not
        be flagged (the validator legitimately reads fan_out_param via
        getattr)."""
        synthetic = (
            'def f(node):\n'
            '    a = getattr(node, "fan_out_param", None)\n'
            '    b = node.oracle_gen_type\n'
            '    return a, b\n'
        )
        written = self._scan_ir_field_writes(ast.parse(synthetic))
        assert written == set(), (
            f"scanner flagged a read as a write: {sorted(written)}"
        )


class TestRoutingKeyNotLabelInvariant:
    """neograph-y20i / 7df1: LangGraph routing identity is the explicit
    graph.add_node(name, fn) argument — never a wrapper/shim closure's
    __name__. No closure may override its __name__ to a mangled state
    field_name (``= field_name`` or ``= field_name_for(...)``). Such
    overrides conflate routing identity with the display label and are
    dead weight (routing never reads closure __name__).

    Legitimate __name__ assignments that preserve a real user-facing name
    (e.g. ``legacy_shim.__name__ = fn_name`` in decorators.py, which keeps
    the user's function name) are NOT flagged — only mangled-field_name
    overrides are.
    """

    @staticmethod
    def _scan_name_overrides(tree: ast.AST) -> list[int]:
        """Return line numbers of ``<x>.__name__ = field_name`` or
        ``<x>.__name__ = field_name_for(...)`` assignments."""
        offenders: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            # target must be an attribute access ending in `.__name__`
            if not (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "__name__"
            ):
                continue
            val = node.value
            is_field_name = isinstance(val, ast.Name) and val.id == "field_name"
            is_field_name_for = (
                isinstance(val, ast.Call)
                and isinstance(val.func, ast.Name)
                and val.func.id == "field_name_for"
            )
            if is_field_name or is_field_name_for:
                offenders.append(node.lineno)
        return offenders

    def test_no_closure_name_mangled_to_field_name(self):
        offenders: list[str] = []
        for py in SRC_DIR.rglob("*.py"):
            tree = ast.parse(py.read_text())
            for lineno in self._scan_name_overrides(tree):
                offenders.append(f"{py.name}:{lineno}")
        assert offenders == [], (
            f"\n{len(offenders)} closure __name__ override(s) mangle routing "
            f"identity into a state field_name:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nRouting identity is the graph.add_node(name, fn) argument; "
              "do NOT override the closure __name__ to field_name/field_name_for(). "
              "Source display labels from node.name/sub.name via the captured IR "
              "object (see neograph-y20i)."
        )

    def test_scanner_detects_field_name_override(self):
        """Mutation: ``x.__name__ = field_name`` must be flagged."""
        tree = ast.parse("def f():\n    x.__name__ = field_name\n")
        assert self._scan_name_overrides(tree) == [2]

    def test_scanner_detects_field_name_for_override(self):
        """Mutation: ``x.__name__ = field_name_for(n.name)`` must be flagged."""
        tree = ast.parse("def f():\n    x.__name__ = field_name_for(n.name)\n")
        assert self._scan_name_overrides(tree) == [2]

    def test_scanner_ignores_legitimate_name_assignment(self):
        """Mutation: preserving a real user name (fn_name) is NOT flagged."""
        tree = ast.parse("def f():\n    shim.__name__ = fn_name\n")
        assert self._scan_name_overrides(tree) == []


class TestNoOrZeroCounterIdiom:
    """neograph-ylk9: the counter 'None-means-zero' rule lives ONLY in
    ``StateBus.get_counter``. No production site may re-derive it via
    ``.get(<count_field>) or 0`` or ``.get(<count_field>, 0)``.

    Same root cause as ``loop_condition_none_unsafe``: counter zero-default
    semantics must not leak across the codebase. This guard is scoped to the
    StateBus-audited files (TestStateBusGetDiscipline.IN_SCOPE).
    """

    # A .get(...) whose argument list mentions a counter field.
    _COUNTER_GET = re.compile(r"\.get\([^)]*count[^)]*\)")
    # The explicit-default form: .get(<...count...>, 0)
    _COUNTER_GET_DEFAULT = re.compile(r"\.get\([^)]*count[^)]*,\s*0\s*\)")

    @classmethod
    def _is_counter_zero_idiom(cls, line: str) -> bool:
        if not cls._COUNTER_GET.search(line):
            return False
        return ("or 0" in line) or bool(cls._COUNTER_GET_DEFAULT.search(line))

    @classmethod
    def _scan(cls, src_dir) -> list[str]:
        offenders: list[str] = []
        for name in sorted(TestStateBusGetDiscipline.IN_SCOPE):
            py = src_dir / name
            if not py.exists():
                continue
            for lineno, line in enumerate(py.read_text().splitlines(), start=1):
                if cls._is_counter_zero_idiom(line):
                    offenders.append(f"{name}:{lineno}: {line.strip()}")
        return offenders

    def test_no_or_zero_counter_idiom_in_statebus_files(self):
        offenders = self._scan(SRC_DIR)
        assert offenders == [], (
            f"\n{len(offenders)} counter 'or 0' / get(count,0) idiom(s) — "
            f"use StateBus.get_counter(key) instead:\n"
            + "\n".join(f"  {o}" for o in offenders)
        )

    def test_scanner_detects_or_zero_form(self):
        """Mutation: 'bus.get(count_field) or 0' must be flagged."""
        assert self._is_counter_zero_idiom("    current = bus.get(count_field) or 0")

    def test_scanner_detects_guarded_or_zero_form(self):
        """Mutation: the _state_write guarded form must be flagged."""
        line = "        current = (state.get(count_field) if state is not None else None) or 0"
        assert self._is_counter_zero_idiom(line)

    def test_scanner_detects_explicit_zero_default_form(self):
        """Mutation: 'bus.get(count_field, 0)' must be flagged."""
        assert self._is_counter_zero_idiom("        count = bus.get(count_field, 0)")

    def test_scanner_ignores_get_counter(self):
        """get_counter(key) is the sanctioned form — never flagged."""
        assert not self._is_counter_zero_idiom("        count = bus.get_counter(count_field)")

    def test_scanner_ignores_unrelated_or_zero(self):
        """'or 0' on a non-counter .get must NOT be flagged."""
        assert not self._is_counter_zero_idiom("        x = bus.get('score') or 0")
        assert not self._is_counter_zero_idiom("        total = some_list_len or 0")

    def test_slip_counter_get(self):
        """Regex-slip: _COUNTER_GET keys on the 'count' substring inside a .get
        arg list. Prove the boundary: a counter .get matches, a non-counter .get
        does not (the 'count'-substring scoping is load-bearing and documented)."""
        assert self._COUNTER_GET.search("bus.get(loop_count_field)")
        assert self._COUNTER_GET.search('bus.get("neo_loop_count")')
        assert not self._COUNTER_GET.search("bus.get(score_field)")

    def test_slip_counter_get_default(self):
        """Regex-slip: _COUNTER_GET_DEFAULT requires the explicit ', 0' default
        with optional surrounding whitespace. Prove whitespace variants match and
        a no-default counter .get does not (so it can't masquerade as the
        explicit-default form)."""
        assert self._COUNTER_GET_DEFAULT.search("bus.get(loop_count, 0)")
        assert self._COUNTER_GET_DEFAULT.search("bus.get(loop_count,0)")
        assert not self._COUNTER_GET_DEFAULT.search("bus.get(loop_count)")


class TestToolBudgetPreambleSingleSource:
    """The framework tool-budget preamble has exactly ONE producer.

    announced==enforced holds only while a single helper renders the budget
    prose from Tool.budget / cfg.max_iterations. A future PR that hand-rolls a
    second budget announcement in another module would silently reintroduce the
    drift this feature eliminated (the in-repo analogue of the piarch
    max_tool_calls=15 hardcode). This guard pins the single source.
    """

    _PRODUCER_MODULE = "_tool_budget_preamble.py"
    # Distinctive fragment of the locked plan-ahead directive. Only the
    # canonical helper may contain it.
    _DIRECTIVE_MARKER = "you need not use every call"

    @staticmethod
    def _modules_with_marker(corpus: dict[str, str], marker: str) -> list[str]:
        return sorted(name for name, text in corpus.items() if marker in text)

    def _real_corpus(self) -> dict[str, str]:
        return {py.name: py.read_text() for py in SRC_DIR.glob("*.py")}

    def test_directive_lives_in_exactly_one_module(self):
        """The live tree: the directive marker appears in only the producer."""
        hits = self._modules_with_marker(self._real_corpus(), self._DIRECTIVE_MARKER)
        assert hits == [self._PRODUCER_MODULE], (
            "Tool-budget preamble directive must live only in "
            f"{self._PRODUCER_MODULE}; found in: {hits}. Do not hand-roll a "
            "second budget announcement -- call render_tool_budget_preamble so "
            "announced==enforced by construction."
        )

    def test_tool_loop_delegates_to_helper(self):
        """The announce branch delegates to the helper -- numbers are derived
        from tools/cfg, never written as literals."""
        src = (SRC_DIR / "_tool_loop.py").read_text()
        assert "render_tool_budget_preamble(tools, cfg.max_iterations)" in src

    def test_scanner_accepts_single_producer(self):
        """Positive meta-test: one producer -> the sanctioned result."""
        corpus = {
            "_tool_budget_preamble.py": "... you need not use every call ...",
            "other.py": "unrelated",
        }
        assert self._modules_with_marker(corpus, self._DIRECTIVE_MARKER) == [
            self._PRODUCER_MODULE
        ]

    def test_scanner_flags_second_producer(self):
        """Negative meta-test: a rogue second producer diverges from the
        sanctioned single-module result, so the guard fires."""
        corpus = {
            "_tool_budget_preamble.py": "... you need not use every call ...",
            "_rogue.py": "hand-rolled: you need not use every call",
        }
        hits = self._modules_with_marker(corpus, self._DIRECTIVE_MARKER)
        assert hits != [self._PRODUCER_MODULE]
        assert "_rogue.py" in hits


class TestAgentModeNoStructuredRegeneration:
    """Agent/act mode does not UNCONDITIONALLY re-generate its output.

    neograph-eoi8 (refining f7nt): the ReAct loop's final turn is parsed from
    messages[-1] on the happy path (0 extra calls). A constrained-decoding
    re-generation via _call_structured is allowed ONLY as a parse-failure
    fallback (output_strategy='structured'). So any _call_structured call inside
    _tool_loop.py must live inside an `except` block, never on the happy path.

    LIMITS (do not over-trust this AST scan): it is a lexical, single-file
    approximation. It cannot catch a happy-path re-gen introduced via a helper
    in ANOTHER module (the name scan is _tool_loop.py-only), and it treats ANY
    `except` as the parse-failure branch (it does not prove the except catches
    the parse error specifically). The REAL invariant pins are the behavioral
    tests: TestAgentStrategyAwareFallback (happy path = K+1, fallback only on
    parse failure) and the with_structured_output-absent assertion below.
    """

    @staticmethod
    def _call_ids_in_except(func_name: str, tree: ast.AST) -> set[int]:
        ids: set[int] = set()
        for handler in ast.walk(tree):
            if isinstance(handler, ast.ExceptHandler):
                for node in ast.walk(handler):
                    if (isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Name)
                            and node.func.id == func_name):
                        ids.add(id(node))
        return ids

    @staticmethod
    def _all_call_ids(func_name: str, tree: ast.AST) -> set[int]:
        return {
            id(node) for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == func_name
        }

    @classmethod
    def _unconditional_calls(cls, func_name: str, src: str) -> int:
        """Count func_name calls that are NOT inside any except handler."""
        tree = ast.parse(src)
        return len(cls._all_call_ids(func_name, tree) - cls._call_ids_in_except(func_name, tree))

    @classmethod
    def _total_calls(cls, func_name: str, src: str) -> int:
        return len(cls._all_call_ids(func_name, ast.parse(src)))

    def _tool_loop_src(self) -> str:
        return (SRC_DIR / "_tool_loop.py").read_text()

    def test_call_structured_only_in_failure_branch_in_tool_loop(self):
        """Live tree: _call_structured appears in _tool_loop.py (the fallback)
        but every call is inside an except block — none on the happy path."""
        src = self._tool_loop_src()
        assert self._total_calls("_call_structured", src) >= 1, (
            "expected a _call_structured fallback in _tool_loop.py"
        )
        assert self._unconditional_calls("_call_structured", src) == 0, (
            "_call_structured is called OUTSIDE an except block in _tool_loop.py; "
            "agent mode must parse messages[-1] first and only fall back to "
            "constrained decoding ON parse failure (no unconditional re-generation)."
        )

    def test_no_with_structured_output_in_tool_loop(self):
        """AR-03: the loop must not bypass the fallback discipline by calling
        with_structured_output (the primitive _call_structured wraps) directly."""
        assert "with_structured_output" not in self._tool_loop_src()

    def test_scanner_flags_unconditional_call(self):
        """Negative meta-test: a top-level (non-except) call is flagged."""
        src = "def g():\n    return _call_structured(x, y)\n"
        assert self._unconditional_calls("_call_structured", src) == 1

    def test_scanner_accepts_except_only_call(self):
        """Positive meta-test: a call reachable only inside except is accepted."""
        src = (
            "def g():\n"
            "    try:\n"
            "        return parse(m)\n"
            "    except ExecutionError:\n"
            "        return _call_structured(x, y)\n"
        )
        assert self._total_calls("_call_structured", src) == 1
        assert self._unconditional_calls("_call_structured", src) == 0


class TestDefaultFactoryCoercionIsGuarded:
    """The default_factory null-coercion must call the factory inside a
    TypeError guard (neograph-s1u4).

    A bare ``default_factory()`` crashes on Pydantic 2.10+ data-accepting
    factories (``default_factory=lambda data: ...``); conversely, passing the
    data dict to a zero-arg factory like ``list`` silently returns the dict's
    keys instead of ``[]``. So the coercion must be zero-arg-first with an
    ``except TypeError -> factory(data)`` fallback. This guard stops a future PR
    from "simplifying" it back to a bare call.

    LIMITS: a single-function AST check on ``_apply_null_defaults`` — it proves a
    TypeError-guarded try wraps a factory call, not the exact call shape. The
    behavioral pins are TestParseJsonResponseLenientParsing (plain list factory
    AND data-accepting factory).
    """

    @staticmethod
    def _func_default_factory_is_typeerror_guarded(src: str, func_name: str) -> bool:
        tree = ast.parse(src)
        target = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == func_name),
            None,
        )
        if target is None:
            return False
        references_factory = any(
            isinstance(n, ast.Attribute) and n.attr == "default_factory"
            for n in ast.walk(target)
        )
        if not references_factory:
            return False
        for node in ast.walk(target):
            if isinstance(node, ast.Try) and any(
                isinstance(h.type, ast.Name) and h.type.id == "TypeError"
                for h in node.handlers
            ):
                body_has_call = any(isinstance(n, ast.Call) for b in node.body for n in ast.walk(b))
                handler_has_call = any(
                    isinstance(n, ast.Call) for h in node.handlers for n in ast.walk(h)
                )
                if body_has_call and handler_has_call:
                    return True
        return False

    def test_apply_null_defaults_guards_default_factory(self):
        """Live tree: the coercion in _apply_null_defaults is TypeError-guarded."""
        src = (SRC_DIR / "_llm_retry.py").read_text()
        assert self._func_default_factory_is_typeerror_guarded(src, "_apply_null_defaults")

    def test_scanner_flags_bare_factory_call(self):
        """Negative meta-test: a bare default_factory() call is NOT guarded."""
        src = (
            "def _apply_null_defaults(data, model):\n"
            "    fi = model\n"
            "    data['x'] = fi.default_factory()\n"
        )
        assert not self._func_default_factory_is_typeerror_guarded(src, "_apply_null_defaults")

    def test_scanner_accepts_guarded_call(self):
        """Positive meta-test: zero-arg-first + except TypeError fallback passes."""
        src = (
            "def _apply_null_defaults(data, model):\n"
            "    fi = model\n"
            "    factory = fi.default_factory\n"
            "    try:\n"
            "        data['x'] = factory()\n"
            "    except TypeError:\n"
            "        data['x'] = factory(data)\n"
        )
        assert self._func_default_factory_is_typeerror_guarded(src, "_apply_null_defaults")
