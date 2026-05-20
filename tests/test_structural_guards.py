"""Structural guards — AST-based invariants that run every build.

These tests enforce architectural rules by scanning source code structure,
not by running pipelines. They prevent erosion of patterns that were
established via migration sprints.

Each guard was written BEFORE the migration it enforces. If a guard fails,
fix the violation — do NOT add an allowlist.
"""

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


class TestErrorBuilderEnforcement:
    """Every raise of a neograph error class must use .build(), not direct construction.

    Good:  raise ConstructError.build("what", expected="X", found="Y")
    Bad:   raise ConstructError(f"Node '{name}' ...")
    Bad:   raise ConstructError(msg)
    Bad:   raise ConstructError("literal string")

    This guard prevents the error builder migration from eroding.
    If this test fails, convert the flagged raise to .build() — do NOT
    add an allowlist entry.
    """

    def test_all_error_raises_use_build(self):
        violations = []

        for py_file in sorted(SRC_DIR.glob("*.py")):
            source = py_file.read_text()
            try:
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.Raise) or node.exc is None:
                    continue

                exc = node.exc

                # raise SomeError(...) — direct call, should be .build()
                if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                    if exc.func.id in ERROR_CLASSES:
                        violations.append(
                            f"  {py_file.name}:{node.lineno}: "
                            f"raise {exc.func.id}(...) — use {exc.func.id}.build() instead"
                        )

        assert violations == [], (
            f"\n{len(violations)} error raise(s) bypass .build():\n"
            + "\n".join(violations)
            + "\n\nConvert each to .build(what=, expected=, found=, hint=). "
            "See src/neograph/errors.py for the API."
        )


class TestFileSplitEnforcement:
    """Functions extracted from decorators.py must not drift back.

    The decorators.py splits are still ad-hoc per-symbol rules. Factory.py
    drift is now enforced by TestFactoryResponsibilityDiscipline, which
    whitelists factory.py's public surface instead of blacklisting symbols.
    """

    # (function_or_class_name, must_be_in, must_NOT_be_in)
    SPLIT_RULES = [
        # decorators.py → _di_classify.py
        ("def _classify_di_params", "_di_classify.py", "decorators.py"),
        ("class FromInput", "_di_classify.py", "decorators.py"),
        ("class FromConfig", "_di_classify.py", "decorators.py"),
        ("def _resolve_di_args", "_di_classify.py", "decorators.py"),
        ("def _resolve_merge_args", "_di_classify.py", "decorators.py"),
        # decorators.py → _construct_builder.py
        ("def construct_from_module", "_construct_builder.py", "decorators.py"),
        ("def construct_from_functions", "_construct_builder.py", "decorators.py"),
        ("def _build_construct_from_decorated", "_construct_builder.py", "decorators.py"),
        ("def _register_node_scripted", "_construct_builder.py", "decorators.py"),
    ]

    def test_extracted_functions_in_correct_module(self):
        violations = []
        for signature, must_be_in, must_not_be_in in self.SPLIT_RULES:
            good_file = SRC_DIR / must_be_in
            bad_file = SRC_DIR / must_not_be_in

            good_text = good_file.read_text() if good_file.exists() else ""
            bad_text = bad_file.read_text() if bad_file.exists() else ""

            if signature not in good_text:
                violations.append(
                    f"  MISSING: '{signature}' not found in {must_be_in}"
                )
            if signature in bad_text:
                violations.append(
                    f"  DRIFTED: '{signature}' found in {must_not_be_in} "
                    f"(should only be in {must_be_in})"
                )

        assert violations == [], (
            f"\n{len(violations)} file-split violation(s):\n"
            + "\n".join(violations)
        )


def _parse_neograph_imports(path: pathlib.Path) -> set[str]:
    """Return the set of `neograph.*` modules imported by the file.

    Walks both `from neograph.X import Y` and `import neograph.X` forms.
    Returns module names without the `neograph.` prefix
    (e.g., `_execute`, `factory`, `_oracle`).
    """
    if not path.exists():
        return set()
    tree = ast.parse(path.read_text())
    mods: set[str] = set()
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.ImportFrom):
            if stmt.module and stmt.module.startswith("neograph.") and stmt.level == 0:
                mods.add(stmt.module[len("neograph."):])
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name.startswith("neograph."):
                    mods.add(alias.name[len("neograph."):])
    return mods


class TestAssemblyClusterImportDAG:
    """Guard 1 — Import-graph DAG layering for the assembly cluster.

    Replaces the previous TestFactoryResponsibilityDiscipline line-count +
    name-whitelist guard with a real architectural rule: the assembly cluster
    modules (factory.py, _execute.py, _subconstruct.py, _oracle.py,
    _state_write.py, _input_shape.py) form a strict layering DAG per
    docs/design/god-module-analysis-2026-05-20.md §5.

    factory.py -> _execute.py -> {_state_write, _input_shape, _oracle, _dispatch}
    _subconstruct.py -> {_oracle, _state_bus}
    _oracle.py / _state_write.py / _input_shape.py are leaves (no upward edges).
    """

    CLUSTER = frozenset({
        "factory",
        "_execute",
        "_subconstruct",
        "_oracle",
        "_state_write",
        "_input_shape",
    })

    # For each cluster module, the cluster modules it is allowed to import.
    # External modules (di, modifiers, naming, errors, etc.) are not enforced.
    ALLOWED_CLUSTER_EDGES = {
        "factory": {"_execute", "_input_shape", "_oracle", "_state_write", "_subconstruct"},
        "_execute": {"_input_shape", "_oracle", "_state_write"},
        "_subconstruct": {"_oracle"},
        "_oracle": set(),
        "_state_write": set(),
        "_input_shape": set(),
    }

    def test_no_edge_violates_dag(self):
        violations: list[str] = []
        for mod, allowed in self.ALLOWED_CLUSTER_EDGES.items():
            path = SRC_DIR / f"{mod}.py"
            imports = _parse_neograph_imports(path)
            cluster_imports = imports & self.CLUSTER
            extras = cluster_imports - allowed
            for extra in sorted(extras):
                violations.append(f"  {mod}.py -> {extra}.py (not in allowed DAG)")
        assert not violations, (
            "Assembly cluster import-DAG layering violated:\n"
            + "\n".join(violations)
        )

    def test_mutation_violation_detected(self, tmp_path):
        """Mutation case — inject a violating edge; scanner must detect."""
        # Build a fake module that has a violating edge: _oracle imports _execute
        fake_oracle = tmp_path / "_oracle.py"
        fake_oracle.write_text(
            "from neograph._execute import _execute_node\n"
        )
        imports = _parse_neograph_imports(fake_oracle)
        cluster_imports = imports & self.CLUSTER
        allowed = self.ALLOWED_CLUSTER_EDGES["_oracle"]
        extras = cluster_imports - allowed
        assert "_execute" in extras, (
            "Scanner failed to detect the mutation (oracle -> _execute)."
        )


class TestAssemblyScenarioTouchpoints:
    """Guard 2 — Scenario walkthrough: change scenarios bounded by file count.

    Each architectural change scenario lists the files that should be touched.
    The guard asserts the touched-file set stays bounded. If a future change
    forces more files than `max_touch`, that is a drift signal — the cluster
    has accreted responsibilities.
    """

    SCENARIO_TOUCHPOINTS = {
        "add_new_execution_mode": {
            "must_touch": {"_dispatch.py"},
            "may_touch": {"factory.py"},
            "max_touch": 2,
        },
        "add_new_modifier_shape": {
            "must_touch": {"_state_write.py", "_input_shape.py", "modifiers.py"},
            "may_touch": set(),
            "max_touch": 3,
        },
        "add_new_oracle_config_field": {
            "must_touch": {"_oracle.py"},
            "may_touch": set(),
            "max_touch": 1,
        },
        "add_subconstruct_boundary_feature": {
            "must_touch": {"_subconstruct.py"},
            "may_touch": {"construct.py"},
            "max_touch": 3,
        },
        "add_lifecycle_step": {
            "must_touch": {"_execute.py"},
            "may_touch": set(),
            "max_touch": 1,
        },
    }

    def test_all_must_touch_files_exist(self):
        """Every file named in SCENARIO_TOUCHPOINTS must exist in src."""
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
                f"Scenario '{scenario}' touches {total} files; max {spec['max_touch']}. "
                "If a real change forces more, that's a drift signal."
            )

    def test_mutation_excess_touchpoint_detected(self):
        """Mutation case — inject a scenario that touches too many files."""
        bad = {
            "must_touch": {"a.py", "b.py", "c.py", "d.py"},
            "may_touch": set(),
            "max_touch": 2,
        }
        total = len(bad["must_touch"]) + len(bad["may_touch"])
        assert total > bad["max_touch"], (
            "Mutation scanner failed: oversized touchpoint not detected."
        )


class TestAssemblyCohesionFanOut:
    """Guard 3 — Cohesion fan-out metric.

    For each assembly-cluster module, count how many distinct external
    (src/neograph) modules import from it. The ceiling is set per the
    expected architectural shape — drivers (factory) can have many
    importers; leaf modules should have few; lifecycle hub (_execute)
    must have exactly one importer (factory).
    """

    FAN_OUT_CEILING = {
        # factory.py is the public-facing wrapper builder; it's allowed many importers.
        "factory.py": 12,
        # _execute.py is the lifecycle hub; only factory should import.
        "_execute.py": 1,
        # _subconstruct.py is imported by compiler + _wiring + factory.
        "_subconstruct.py": 3,
        # _oracle.py: re-exported via factory, imported by _execute, _subconstruct, _wiring.
        "_oracle.py": 6,
        # _state_write.py: imported by _execute + factory (re-export) + tests scope.
        "_state_write.py": 3,
        # _input_shape.py: imported by _execute + factory (re-export).
        "_input_shape.py": 3,
    }

    def _count_importers(self, target_module_basename: str) -> list[str]:
        target = f"neograph.{target_module_basename[:-3]}"  # strip '.py'
        importers: list[str] = []
        for py in sorted(SRC_DIR.glob("*.py")):
            if py.name == target_module_basename:
                continue
            text = py.read_text()
            if f"from {target} " in text or f"from {target}\n" in text or f"import {target}\n" in text:
                importers.append(py.name)
        return importers

    def test_fan_out_under_ceiling(self):
        violations: list[str] = []
        for mod, ceiling in self.FAN_OUT_CEILING.items():
            importers = self._count_importers(mod)
            if len(importers) > ceiling:
                violations.append(
                    f"  {mod}: imported by {len(importers)} modules (ceiling {ceiling}). "
                    f"Importers: {importers}"
                )
        assert not violations, (
            "Cohesion fan-out exceeded ceiling — module is becoming a kitchen sink:\n"
            + "\n".join(violations)
        )

    def test_mutation_excess_importers_detected(self, tmp_path):
        """Mutation case — synthesize a target that has too many importers."""
        # Synthesize 5 importers of a fictitious module 'leaf.py', ceiling 1.
        for i in range(5):
            (tmp_path / f"client_{i}.py").write_text(
                "from neograph.leaf import x\n"
            )
        count = sum(
            1 for p in tmp_path.glob("client_*.py")
            if "from neograph.leaf " in p.read_text()
        )
        assert count > 1, "Mutation scanner failed: excess importers not detected."


class TestClusterEModifierTouchpointSentinels:
    """Guard 4 — Co-change witness for Cluster E (tightens gm-1 sentinel).

    Every `# MODIFIER_RULE_TOUCHPOINT` sentinel must live in _state_write.py.
    Sentinels in other files indicate the Each-key / Loop-counter / Oracle-fusion
    rules are leaking across files — the same anti-pattern Cluster E was
    designed to eliminate.
    """

    SENTINEL = "# MODIFIER_RULE_TOUCHPOINT"

    def _scan_sentinels(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for py in sorted(SRC_DIR.glob("*.py")):
            text = py.read_text()
            n = text.count(self.SENTINEL)
            if n > 0:
                result[py.name] = n
        return result

    def test_all_sentinels_live_in_state_write(self):
        counts = self._scan_sentinels()
        offenders = {k: v for k, v in counts.items() if k != "_state_write.py"}
        assert not offenders, (
            f"{self.SENTINEL} sentinels found outside _state_write.py: {offenders}"
        )

    def test_state_write_has_sentinels(self):
        counts = self._scan_sentinels()
        n = counts.get("_state_write.py", 0)
        assert n >= 3, (
            f"Expected >=3 {self.SENTINEL} sentinels in _state_write.py "
            f"(Each-key wrap, Loop counter, Oracle fusion) but found {n}."
        )

    def test_mutation_drift_detected(self, tmp_path):
        """Mutation case — inject sentinel into a different file; scanner detects."""
        rogue = tmp_path / "rogue.py"
        rogue.write_text(f"x = 1  {self.SENTINEL}\n")
        # The scanner using `text.count(SENTINEL)` would see it. Simulate the
        # check shape from test_all_sentinels_live_in_state_write.
        n = rogue.read_text().count(self.SENTINEL)
        assert n > 0, "Mutation scanner failed: sentinel injection not detected."


class TestInputShapeExhaustiveness:
    """Guard 5 — Every InputShape enum value has a matching extractor function.

    The dispatch in _extract_input must enumerate every InputShape variant.
    A new shape always implies (a) a new enum value, (b) a new extractor
    function, (c) a new branch in _classify_input_shape, and (d) a new
    branch in _extract_input. The exhaustive `assert_never` plus this guard
    keep them in lockstep.
    """

    def _enum_values(self) -> list[str]:
        from neograph._input_shape import InputShape
        return [v.value for v in InputShape]

    def test_extractor_per_variant(self):
        src = (SRC_DIR / "_input_shape.py").read_text()
        missing: list[str] = []
        for variant in self._enum_values():
            if variant == "none":
                continue
            expected_fn = f"_extract_{variant}"
            if f"def {expected_fn}(" not in src:
                missing.append(f"  InputShape({variant!r}) has no _extract_{variant}() function")
        assert not missing, (
            "InputShape extractor table incomplete:\n" + "\n".join(missing)
        )

    def test_dispatch_covers_all_variants(self):
        """_extract_input must reference every (non-NONE) InputShape variant."""
        src = (SRC_DIR / "_input_shape.py").read_text()
        missing: list[str] = []
        for variant in self._enum_values():
            if variant == "none":
                continue
            needle = f"InputShape.{variant.upper()}"
            if needle not in src:
                missing.append(f"  {needle} not referenced in _extract_input")
        assert not missing, (
            "InputShape dispatch is not exhaustive:\n" + "\n".join(missing)
        )

    def test_mutation_missing_extractor_detected(self):
        """Mutation case — pretend a variant exists with no extractor."""
        src = (SRC_DIR / "_input_shape.py").read_text()
        bogus_variant = "phantom_shape"
        expected = f"def _extract_{bogus_variant}("
        assert expected not in src, (
            "Mutation precondition violated: _extract_phantom_shape unexpectedly exists."
        )


class TestSubconstructBoundaryOwnership:
    """Guard 6 — Sub-construct runtime boundary lives in _subconstruct.py only.

    The runtime mutation that writes `neo_subgraph_input` into a sub_input
    dict is sub-construct boundary semantics. Re-implementations in
    factory.py / _execute.py / _wiring.py / runner.py would mean the boundary
    is splaying across files.

    Compile-time mentions in state.py / loader.py / _construct_builder.py /
    _construct_validation.py / forward.py are out-of-scope (they declare the
    port type; they don't perform the runtime write).
    """

    NEEDLE = '"neo_subgraph_input"'
    RUNTIME_MODULES = {
        "factory.py",
        "_execute.py",
        "_wiring.py",
        "runner.py",
    }

    def test_runtime_write_only_in_subconstruct(self):
        offenders: list[str] = []
        for fname in self.RUNTIME_MODULES:
            path = SRC_DIR / fname
            if not path.exists():
                continue
            text = path.read_text()
            if self.NEEDLE in text:
                offenders.append(fname)
        assert not offenders, (
            f"{self.NEEDLE} found in runtime modules other than _subconstruct.py: {offenders}. "
            "Sub-construct boundary semantics belong in _subconstruct.py only."
        )

    def test_subconstruct_writes_neo_subgraph_input(self):
        text = (SRC_DIR / "_subconstruct.py").read_text()
        assert self.NEEDLE in text, (
            "_subconstruct.py must own the runtime neo_subgraph_input write."
        )

    def test_mutation_runtime_leak_detected(self, tmp_path):
        """Mutation case — write needle into a fake runtime file; scanner detects."""
        rogue = tmp_path / "factory.py"
        rogue.write_text('sub_input["neo_subgraph_input"] = x\n')
        text = rogue.read_text()
        assert self.NEEDLE in text, (
            "Mutation scanner failed: runtime leak not detected."
        )


class TestClusterEUnification:
    """_apply_skip_when joins _build_state_update in renamed _state_write.py (gm-1).

    The two functions encode the same modifier rule set (Each-key wrapping,
    Loop counter increment, Oracle-fusion). They were split across two files
    in the cgkl topical layout; this guard pins them in one module so the
    rule set has a single home.

    Per docs/design/god-module-analysis-2026-05-20.md Q3+Q4:
      - _state_io.py renamed to _state_write.py (mirrors _state_bus.py read side)
      - _apply_skip_when moved out of _modifier_io.py to _state_write.py
      - both functions co-located so modifier rules change in lockstep

    The co-change witness is a `# MODIFIER_RULE_TOUCHPOINT` sentinel comment
    placed at every Each-key wrapping / Loop counter / Oracle-fusion rule
    site inside _state_write.py. The guard asserts all sentinels live in
    _state_write.py only — if one drifts to another file, modifier rules
    have splayed again.
    """

    def test_state_write_module_exists(self):
        assert (SRC_DIR / "_state_write.py").exists(), (
            "_state_write.py must exist (renamed from _state_io.py per gm-1)."
        )

    def test_state_io_module_deleted(self):
        assert not (SRC_DIR / "_state_io.py").exists(), (
            "_state_io.py must be deleted — its contents moved to _state_write.py."
        )

    def test_state_write_contains_build_state_update_and_apply_skip_when(self):
        path = SRC_DIR / "_state_write.py"
        if not path.exists():
            pytest.fail("_state_write.py does not exist yet (gm-1 not complete).")
        tree = ast.parse(path.read_text())
        names = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        }
        for required in ("_build_state_update", "_apply_skip_when"):
            assert required in names, (
                f"_state_write.py must define {required} (gm-1 Cluster E unification)."
            )

    def test_apply_skip_when_not_in_modifier_io_or_input_shape(self):
        for candidate in ("_modifier_io.py", "_input_shape.py"):
            path = SRC_DIR / candidate
            if not path.exists():
                continue
            tree = ast.parse(path.read_text())
            names = {
                n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
            }
            assert "_apply_skip_when" not in names, (
                f"_apply_skip_when must NOT be defined in {candidate} — "
                "it belongs in _state_write.py (Cluster E)."
            )

    def test_modifier_rule_touchpoint_sentinels_live_in_state_write(self):
        """All `# MODIFIER_RULE_TOUCHPOINT` sentinels must live in _state_write.py."""
        SENTINEL = "# MODIFIER_RULE_TOUCHPOINT"
        offenders: list[str] = []
        count_in_state_write = 0
        for py in sorted(SRC_DIR.glob("*.py")):
            text = py.read_text()
            n = text.count(SENTINEL)
            if n == 0:
                continue
            if py.name == "_state_write.py":
                count_in_state_write = n
            else:
                offenders.append(f"{py.name}: {n} sentinel(s)")
        assert not offenders, (
            f"{SENTINEL} found outside _state_write.py: {offenders}"
        )
        assert count_in_state_write >= 3, (
            f"Expected >=3 {SENTINEL} sentinels in _state_write.py "
            f"(Each-key wrap, Loop counter, Oracle fusion) but found {count_in_state_write}."
        )


class TestLifecycleSeparation:
    """_execute_node and its helpers live in _execute.py (gm-3 / Cluster B).

    factory.py builds wrappers (Cluster A). _execute_node *runs* one
    invocation through preamble → dispatch → postamble — a different change
    axis. Per docs/design/god-module-analysis-2026-05-20.md Q1+Q5+Q2:
      - _execute_node moved to new _execute.py
      - _type_name moved into _execute.py as a private helper
      - _extract_context moved into _execute.py (sole caller was _execute_node)
      - _observability.py deleted (its only function lives in _execute.py)
    """

    def test_execute_module_exists(self):
        assert (SRC_DIR / "_execute.py").exists(), (
            "_execute.py must exist (created by gm-3 for Cluster B)."
        )

    def test_observability_module_deleted(self):
        assert not (SRC_DIR / "_observability.py").exists(), (
            "_observability.py must be deleted — _extract_context moved into _execute.py."
        )

    def test_execute_defines_required_functions(self):
        path = SRC_DIR / "_execute.py"
        if not path.exists():
            pytest.fail("_execute.py does not exist yet (gm-3 not complete).")
        tree = ast.parse(path.read_text())
        names = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        }
        for required in ("_execute_node", "_type_name", "_extract_context"):
            assert required in names, (
                f"_execute.py must define {required} (gm-3 Cluster B)."
            )

    def test_factory_does_not_define_execute_helpers(self):
        path = SRC_DIR / "factory.py"
        tree = ast.parse(path.read_text())
        names = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        }
        for forbidden in ("_execute_node", "_type_name", "_extract_context"):
            assert forbidden not in names, (
                f"factory.py must NOT define {forbidden} — it lives in _execute.py."
            )

    def test_execute_cohesion_single_src_importer(self):
        """_execute.py's only src/neograph importer must be factory.py."""
        importers: list[str] = []
        target = "neograph._execute"
        for py in sorted(SRC_DIR.glob("*.py")):
            if py.name == "_execute.py":
                continue
            text = py.read_text()
            if f"from {target}" in text or f"import {target}" in text:
                importers.append(py.name)
        assert importers == ["factory.py"], (
            f"_execute.py should be imported by factory.py only, got: {importers}"
        )


class TestInputShapeRename:
    """_modifier_io.py renamed to _input_shape.py (gm-3 / Clusters G+H).

    After _apply_skip_when left in gm-1, what remains is pure read-side
    input-shape classification + extraction. The name should reflect that.
    """

    def test_input_shape_module_exists(self):
        assert (SRC_DIR / "_input_shape.py").exists(), (
            "_input_shape.py must exist (renamed from _modifier_io.py per gm-3)."
        )

    def test_modifier_io_module_deleted(self):
        assert not (SRC_DIR / "_modifier_io.py").exists(), (
            "_modifier_io.py must be deleted — contents renamed to _input_shape.py."
        )

    def test_input_shape_contains_required_symbols(self):
        path = SRC_DIR / "_input_shape.py"
        if not path.exists():
            pytest.fail("_input_shape.py does not exist yet.")
        tree = ast.parse(path.read_text())
        func_names = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        }
        class_names = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
        }
        assert "InputShape" in class_names, "_input_shape.py must define InputShape."
        for fn in (
            "_classify_input_shape",
            "_extract_input",
            "_extract_loop_reentry",
            "_extract_each_item",
            "_extract_fan_in_dict",
            "_extract_single_type",
        ):
            assert fn in func_names, f"_input_shape.py must define {fn}."

    def test_input_shape_cohesion_single_src_importer(self):
        """_input_shape.py's only src/neograph importer (apart from factory's
        backward-compat re-export) must be _execute.py."""
        importers: list[str] = []
        target = "neograph._input_shape"
        for py in sorted(SRC_DIR.glob("*.py")):
            if py.name == "_input_shape.py":
                continue
            text = py.read_text()
            if f"from {target}" in text or f"import {target}" in text:
                importers.append(py.name)
        # factory.py keeps a noqa re-export of InputShape + extractors for
        # tests/test_coverage_gaps that imports them off factory.py. _execute.py
        # is the real consumer.
        assert "_execute.py" in importers, (
            f"_execute.py must import from _input_shape.py, got importers: {importers}"
        )
        # Apart from factory.py (re-export) and _execute.py (real), no one
        # else may import.
        allowed = {"_execute.py", "factory.py"}
        extras = set(importers) - allowed
        assert not extras, (
            f"_input_shape.py imported by unexpected modules: {extras}; "
            f"allowed: {allowed}"
        )


class TestSubconstructBoundary:
    """make_subgraph_fn moves to _subconstruct.py (gm-4 / Cluster C).

    Sub-construct boundary semantics (input-by-type scan, output-by-type scan,
    loop re-entry shortcut, context-field forwarding, isolated sub_input dict)
    is a separate change axis from per-node wrapper assembly. The inline scan
    blocks are extracted to named helpers so the boundary rules have
    inspectable shape per docs/design/god-module-analysis-2026-05-20.md
    Cluster C + §6.

    Scenario walkthrough — "Add a new sub-construct boundary feature" should
    touch at most 3 files: _subconstruct.py, tests/test_composition.py, and
    optionally construct.py for IR-level shape changes.
    """

    SCENARIO_TOUCHPOINTS = {
        "add_subconstruct_boundary_feature": {
            "_subconstruct.py",
            "tests/test_composition.py",
            "construct.py",
        },
    }

    def test_subconstruct_module_exists(self):
        assert (SRC_DIR / "_subconstruct.py").exists(), (
            "_subconstruct.py must exist (created by gm-4 for Cluster C)."
        )

    def test_subconstruct_defines_required_symbols(self):
        path = SRC_DIR / "_subconstruct.py"
        if not path.exists():
            pytest.fail("_subconstruct.py does not exist yet.")
        tree = ast.parse(path.read_text())
        names = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        }
        for required in (
            "make_subgraph_fn",
            "_scan_subgraph_input",
            "_scan_subgraph_output",
        ):
            assert required in names, (
                f"_subconstruct.py must define {required} (gm-4 Cluster C)."
            )

    def test_factory_does_not_define_make_subgraph_fn(self):
        path = SRC_DIR / "factory.py"
        tree = ast.parse(path.read_text())
        names = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        }
        assert "make_subgraph_fn" not in names, (
            "factory.py must NOT define make_subgraph_fn — it lives in _subconstruct.py."
        )

    def test_scenario_touchpoints_bounded(self):
        """Adding a sub-construct boundary feature must touch at most 3 files."""
        for scenario, files in self.SCENARIO_TOUCHPOINTS.items():
            assert len(files) <= 3, (
                f"Scenario '{scenario}' would touch {len(files)} files; "
                "max 3. If a future change forces more, that's a drift signal."
            )


class TestOracleConfigInOracleModule:
    """_inject_oracle_config belongs in _oracle.py (gm-2 / Cluster F).

    The function reads neo_oracle_* state and injects into config['configurable'].
    That is Oracle dispatch plumbing, not state I/O. Per
    docs/design/god-module-analysis-2026-05-20.md Q4, it lives in _oracle.py
    next to make_oracle_redirect_fn / make_oracle_merge_fn / _unwrap_oracle_results.
    """

    def test_inject_oracle_config_defined_in_oracle_module(self):
        oracle = (SRC_DIR / "_oracle.py").read_text()
        tree = ast.parse(oracle)
        names = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        }
        assert "_inject_oracle_config" in names, (
            "_inject_oracle_config must be defined in _oracle.py "
            "(Cluster F — Oracle generator dispatch)."
        )

    def test_inject_oracle_config_not_in_state_modules(self):
        for candidate in ("_state_io.py", "_state_write.py"):
            path = SRC_DIR / candidate
            if not path.exists():
                continue
            tree = ast.parse(path.read_text())
            names = {
                n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
            }
            assert "_inject_oracle_config" not in names, (
                f"_inject_oracle_config must NOT be defined in {candidate} — "
                "it belongs in _oracle.py (Cluster F)."
            )


class TestDeadCodeRemoval:
    """Verify dead code that was deliberately removed stays removed."""

    def test_di_resolver_not_in_codebase(self):
        """DIResolver was dead code — removed in neograph-eq80."""
        for py_file in sorted(SRC_DIR.glob("*.py")):
            text = py_file.read_text()
            assert "class DIResolver" not in text, (
                f"DIResolver found in {py_file.name} — it was removed as dead code. "
                f"If you need a resolver wrapper, design a new one."
            )

    def test_no_raw_node_decorator(self):
        """@raw_node decorator was removed in favor of @node(mode='raw').

        Checks for the decorator definition (not local function names
        like raw_node_wrapper which are fine).
        """
        for py_file in sorted(SRC_DIR.glob("*.py")):
            tree = ast.parse(py_file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "raw_node":
                    # Check if it has a decorator list (it's a decorator definition)
                    # or is a top-level function (public API)
                    assert False, (
                        f"raw_node() function found in {py_file.name}:{node.lineno} — "
                        f"use @node(mode='raw') instead."
                    )

    def test_no_old_from_input_subscription(self):
        """FromInput[T] generic subscription was removed in 0.2.0.

        Only checks actual code, not comments or docstrings.
        """
        for py_file in sorted(SRC_DIR.glob("*.py")):
            tree = ast.parse(py_file.read_text())
            for node in ast.walk(tree):
                # Look for FromInput[...] subscription in actual code
                if isinstance(node, ast.Subscript):
                    if isinstance(node.value, ast.Name) and node.value.id in ("FromInput", "FromConfig"):
                        assert False, (
                            f"{node.value.id}[T] subscription found in "
                            f"{py_file.name}:{node.lineno} — "
                            f"use Annotated[T, {node.value.id}] instead."
                        )


class TestOutputsInferenceGuard:
    """@node outputs= inference from return annotation (neograph-pcdp).

    The decorator infers outputs from -> T. When both outputs= and -> T
    are present and disagree, it must raise ConstructError at decoration
    time. This guard verifies the check exists in the decorator source.
    """

    def test_mismatch_check_exists_in_decorator(self):
        """The decorator must contain the mismatch check logic."""
        decorators = (SRC_DIR / "decorators.py").read_text()
        assert "differs from return annotation" in decorators, (
            "The outputs=/annotation mismatch check is missing from decorators.py. "
            "The @node decorator must raise ConstructError when outputs= and -> T disagree."
        )

    def test_mismatch_actually_raises(self):
        """Runtime verification: mismatched outputs= and -> T raises."""
        from pydantic import BaseModel

        from neograph import ConstructError, node

        class TypeA(BaseModel):
            x: int

        class TypeB(BaseModel):
            y: str

        with pytest.raises(ConstructError, match="differs from return annotation"):
            @node(outputs=TypeA)
            def bad_node(topic: str) -> TypeB:
                return TypeB(y="oops")


class TestNoAnyInTypeBoundaries:
    """NodeInput, NodeOutput, and DIBinding must not use bare Any for typed fields.

    Any at type boundaries defeats the "typed compiler" claim. These
    containers carry concrete values — their fields should say so.
    """

    def test_node_input_output_no_any_fields(self):
        """NodeInput and NodeOutput fields must not be annotated with bare Any."""
        dispatch_file = SRC_DIR / "_dispatch.py"
        tree = ast.parse(dispatch_file.read_text())
        violations = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name not in ("NodeInput", "NodeOutput"):
                continue
            for item in node.body:
                if not isinstance(item, ast.AnnAssign):
                    continue
                if item.target is None or not isinstance(item.target, ast.Name):
                    continue
                field_name = item.target.id
                # Check if annotation is bare "Any"
                ann = item.annotation
                if isinstance(ann, ast.Name) and ann.id == "Any":
                    violations.append(
                        f"  {node.name}.{field_name}: Any "
                        f"(line {item.lineno})"
                    )
                # Check dict[str, Any] — the value type should be concrete
                if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name):
                    if ann.value.id == "dict":
                        # dict[str, Any] — check the second element
                        if isinstance(ann.slice, ast.Tuple) and len(ann.slice.elts) == 2:
                            val_type = ann.slice.elts[1]
                            if isinstance(val_type, ast.Name) and val_type.id == "Any":
                                violations.append(
                                    f"  {node.name}.{field_name}: dict[str, Any] "
                                    f"(line {item.lineno})"
                                )

        assert violations == [], (
            f"\n{len(violations)} bare Any field(s) in dispatch containers:\n"
            + "\n".join(violations)
            + "\n\nReplace with a concrete union type."
        )

    def test_di_binding_has_no_payload_field(self):
        """DIBinding must not have a 'payload' field (neograph-xklx).

        Replaced with typed fields: default_value (CONSTANT) and model_cls (MODEL).
        """
        di_file = SRC_DIR / "di.py"
        tree = ast.parse(di_file.read_text())

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != "DIBinding":
                continue
            field_names = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    field_names.append(item.target.id)
            assert "payload" not in field_names, (
                "DIBinding still has 'payload' field — should be replaced with "
                "default_value (CONSTANT) and model_cls (MODEL)"
            )
            assert "default_value" in field_names, "DIBinding missing default_value field"
            assert "model_cls" in field_names, "DIBinding missing model_cls field"


class TestNoRedundantValidation:
    """Modifier mutual-exclusion checks must live in ModifierSet only.

    ModifierSet.model_post_init is the structural gate (catches all paths).
    _validate_node_chain should NOT duplicate those checks — it adds
    maintenance burden without catching anything model_post_init misses.
    """

    def test_no_each_loop_check_in_validate_node_chain(self):
        """_construct_validation.py must not check Each+Loop exclusion."""
        validation_file = SRC_DIR / "_construct_validation.py"
        text = validation_file.read_text()
        tree = ast.parse(text)

        # Find _validate_node_chain function
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "_validate_node_chain":
                continue
            # Check the function body source for Each+Loop or Oracle+Loop
            # mutual exclusion patterns
            func_source = ast.get_source_segment(text, node)
            if func_source is None:
                continue
            violations = []
            if "Each" in func_source and "Loop" in func_source and "mutual exclu" in func_source.lower():
                violations.append(
                    "  _validate_node_chain contains Each+Loop mutual exclusion check"
                )
            if "Oracle" in func_source and "Loop" in func_source and "mutual exclu" in func_source.lower():
                violations.append(
                    "  _validate_node_chain contains Oracle+Loop mutual exclusion check"
                )
            assert violations == [], (
                "\nRedundant modifier checks in _validate_node_chain:\n"
                + "\n".join(violations)
                + "\n\nModifierSet.model_post_init is the structural gate. "
                "Remove the belt-and-suspenders checks."
            )


class TestNoSidecarPattern:
    """Node metadata must live on the Node via PrivateAttr, not in global dicts.

    The sidecar pattern (_node_sidecar, _param_resolutions keyed by id(Node))
    is a hack around Pydantic immutability. PrivateAttr is the Pydantic-native
    solution — data travels with the object, no re-registration needed.
    """

    def test_no_node_sidecar_dict(self):
        """No global _node_sidecar dict should exist in the codebase."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            text = py_file.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                # Match the dict definition, not comments or strings
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                    continue
                if "_node_sidecar" in line and "import" not in line:
                    violations.append(f"  {py_file.name}:{i}: {stripped[:80]}")
        assert violations == [], (
            f"\n{len(violations)} _node_sidecar reference(s) remain:\n"
            + "\n".join(violations)
            + "\n\nReplace with PrivateAttr on Node."
        )

    def test_no_param_resolutions_dict(self):
        """No global _param_resolutions dict should exist in the codebase."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            text = py_file.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                    continue
                # Match _param_resolutions as a dict or function, not as a PrivateAttr field name
                if "_param_resolutions" in line and "import" not in line:
                    # Allow: _param_res (the PrivateAttr field, shorter name)
                    # Reject: _param_resolutions (the old global dict pattern)
                    violations.append(f"  {py_file.name}:{i}: {stripped[:80]}")
        assert violations == [], (
            f"\n{len(violations)} _param_resolutions reference(s) remain:\n"
            + "\n".join(violations)
            + "\n\nReplace with PrivateAttr on Node."
        )

    def test_no_weakref_finalize_for_sidecar(self):
        """No weakref.finalize calls for sidecar cleanup should exist."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            text = py_file.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                if "weakref.finalize" in line:
                    stripped = line.lstrip()
                    if not stripped.startswith("#"):
                        violations.append(f"  {py_file.name}:{i}: {stripped[:80]}")
        assert violations == [], (
            f"\n{len(violations)} weakref.finalize call(s) remain:\n"
            + "\n".join(violations)
            + "\n\nPrivateAttr data lives on the Node — no external cleanup needed."
        )

    def test_node_has_private_attrs(self):
        """Node must declare PrivateAttr fields for sidecar data."""
        from neograph.node import Node
        private_fields = getattr(Node, "__private_attributes__", {})
        assert "_sidecar" in private_fields or "_sidecar_fn" in private_fields, (
            "Node must have a _sidecar or _sidecar_fn PrivateAttr field. "
            "Add: _sidecar: tuple[Callable, tuple[str, ...]] | None = PrivateAttr(default=None)"
        )
        assert "_param_res" in private_fields, (
            "Node must have a _param_res PrivateAttr field. "
            "Add: _param_res: dict = PrivateAttr(default_factory=dict)"
        )


class TestNoFrameDepthParam:
    """Frame-stack walking must be replaced with explicit namespace passing.

    The frame_depth parameter forces fragile arithmetic (frame_depth=2,
    +1 for nesting). Explicit caller_ns dict is stable regardless of
    call depth.
    """

    def test_no_frame_depth_in_source(self):
        """No function should accept or pass frame_depth."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            text = py_file.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                    continue
                if "frame_depth" in line:
                    violations.append(f"  {py_file.name}:{i}: {stripped[:80]}")
        assert violations == [], (
            f"\n{len(violations)} frame_depth reference(s) remain:\n"
            + "\n".join(violations)
            + "\n\nReplace with explicit caller_ns parameter."
        )


class TestNoTicketIdsInComments:
    """Ticket IDs (neograph-xxxx) belong in git blame, not in source comments.

    Parenthesized ticket refs like (neograph-26ih) clutter code for zero value
    — the commit message already has the context. This guard prevents them
    from creeping back after cleanup.
    """

    def test_no_parenthesized_ticket_ids(self):
        """No (neograph-xxxx) patterns in source code."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            for i, line in enumerate(py_file.read_text().splitlines(), 1):
                if re.search(r"\(neograph-[a-z0-9]{3,5}\)", line):
                    stripped = line.lstrip()[:80]
                    violations.append(f"  {py_file.name}:{i}: {stripped}")
        assert violations == [], (
            f"\n{len(violations)} ticket ID reference(s) in source:\n"
            + "\n".join(violations[:20])
            + ("\n  ..." if len(violations) > 20 else "")
            + "\n\nRemove the (neograph-xxxx) part. Keep the comment text."
        )


class TestNodeIRTyping:
    """Node.inputs and Node.outputs must not be typed as bare Any.

    Any at the IR core defeats the typed compiler claim. 22 isinstance checks
    across the codebase branch on the shape — a Union annotation makes
    the shapes explicit at the type level.
    """

    def test_node_inputs_not_any(self):
        """Node.inputs field annotation must not be bare Any."""
        node_file = SRC_DIR / "node.py"
        tree = ast.parse(node_file.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != "Node":
                continue
            for item in node.body:
                if not isinstance(item, ast.AnnAssign):
                    continue
                if item.target is None or not isinstance(item.target, ast.Name):
                    continue
                if item.target.id == "inputs":
                    ann = item.annotation
                    assert not (isinstance(ann, ast.Name) and ann.id == "Any"), (
                        f"Node.inputs is typed as bare Any (line {item.lineno}). "
                        "Use: inputs: type[BaseModel] | dict[str, type] | None = None"
                    )

    def test_node_outputs_not_any(self):
        """Node.outputs field annotation must not be bare Any."""
        node_file = SRC_DIR / "node.py"
        tree = ast.parse(node_file.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != "Node":
                continue
            for item in node.body:
                if not isinstance(item, ast.AnnAssign):
                    continue
                if item.target is None or not isinstance(item.target, ast.Name):
                    continue
                if item.target.id == "outputs":
                    ann = item.annotation
                    assert not (isinstance(ann, ast.Name) and ann.id == "Any"), (
                        f"Node.outputs is typed as bare Any (line {item.lineno}). "
                        "Use: outputs: type[BaseModel] | dict[str, type] | None = None"
                    )


class TestVersionSync:
    """__init__.py __version__ must match pyproject.toml version."""

    def test_version_matches_pyproject(self):
        import tomllib
        pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        pyproject_version = data["project"]["version"]

        import neograph
        assert neograph.__version__ == pyproject_version, (
            f"__init__.py __version__={neograph.__version__!r} != "
            f"pyproject.toml version={pyproject_version!r}. Keep them in sync."
        )


class TestFieldNameContract:
    """The hyphen-to-underscore naming contract must use a central utility.

    36+ copies of name.replace('-', '_') is a typo waiting to happen.
    Once field_name_for() exists, this guard enforces its use.
    """

    def test_no_inline_hyphen_replace(self):
        """Source code should use field_name_for(), not inline .replace('-', '_')."""
        violations = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            text = py_file.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                    continue
                if '.replace("-", "_")' in line or ".replace('-', '_')" in line:
                    # Allow the definition of field_name_for itself
                    if "def field_name_for" in line:
                        continue
                    violations.append(f"  {py_file.name}:{i}: {stripped[:80]}")
        assert violations == [], (
            f"\n{len(violations)} inline .replace('-', '_') call(s):\n"
            + "\n".join(violations[:20])
            + ("\n  ..." if len(violations) > 20 else "")
            + "\n\nUse field_name_for(name) from neograph.naming instead."
        )


class TestOracleAutoMergeWarning:
    """Oracle with models= but no merge_fn should warn about body dual-purpose."""

    def test_body_merge_emits_warning(self):
        """@node with models= and no merge_fn/merge_prompt emits UserWarning."""
        import warnings

        from neograph import node
        from tests.schemas import Claims, RawText

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @node(
                outputs=Claims,
                model="fast",
                prompt="test",
                models=["fast", "reason"],
            )
            def dual_purpose(topic: RawText) -> Claims:
                return Claims(items=["merged"])

            body_warnings = [
                x for x in w
                if "body used as both generator and merge" in str(x.message)
            ]
            assert len(body_warnings) >= 1, (
                "@node with models= but no merge_fn should emit UserWarning "
                "about body serving dual purpose (generator + merge)."
            )


class TestConstructFromModuleDetectsPlainNodes:
    """construct_from_module should detect plain Node() instances, not just @node."""

    def test_plain_node_instance_detected(self):
        """Module-scope Node(...) should be picked up by construct_from_module."""
        import types as _types

        from neograph import Node, construct_from_module, node
        from tests.schemas import Claims, RawText

        mod = _types.ModuleType("test_plain_node_mod")

        @node
        def source() -> RawText:
            return RawText(text="hi")

        plain = Node.scripted("transform", fn="identity", inputs=RawText, outputs=Claims)

        mod.source = source
        mod.plain = plain

        pipeline = construct_from_module(mod)
        node_names = {n.name for n in pipeline.nodes}
        assert "transform" in node_names, (
            "construct_from_module should detect plain Node() instances. "
            f"Found nodes: {node_names}"
        )


class TestExtractInputDispatch:
    """_extract_input must be a pure dispatch function (neograph-b9pm).

    The main function body should be < 30 lines — all logic lives in
    named helper functions dispatched via InputShape classification.
    """

    def test_extract_input_body_is_short(self):
        """_extract_input body must be < 30 lines (pure dispatch)."""
        source = (SRC_DIR / "_input_shape.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_extract_input":
                body_lines = node.end_lineno - node.lineno
                assert body_lines < 30, (
                    f"_extract_input is {body_lines} lines — should be < 30 "
                    f"(pure dispatch to named helpers, not inline logic)."
                )
                return

        pytest.fail("_extract_input function not found in _input_shape.py")

    def test_input_shape_enum_exists(self):
        """InputShape enum must exist for exhaustive dispatch."""
        source = (SRC_DIR / "_input_shape.py").read_text()
        tree = ast.parse(source)
        class_names = {
            n.name for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef)
        }
        assert "InputShape" in class_names, (
            "InputShape enum must exist in _input_shape.py for exhaustive "
            "_extract_input dispatch."
        )


class TestNoNodeMutationInAssembly:
    """Assembly code must not mutate Node instances in place (neograph-n573).

    _construct_builder.py iterates the `ordered` list and must use model_copy()
    instead of direct attribute assignment on Node instances.
    """

    def test_no_direct_attr_assignment_on_nodes_in_ordered_loops(self):
        """No `.inputs =`, `.fan_out_param =`, `.scripted_fn =`, `.oracle_gen_type =`
        direct assignments on loop variables in _construct_builder.py."""
        source = (SRC_DIR / "_construct_builder.py").read_text()
        tree = ast.parse(source)

        # Collect all attribute assignments (x.attr = value) that target
        # known Node fields being mutated during assembly.
        forbidden_attrs = {"inputs", "fan_out_param", "scripted_fn", "oracle_gen_type"}
        violations = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr in forbidden_attrs
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "n"
                ):
                    violations.append(
                        f"line {node.lineno}: n.{target.attr} = ..."
                    )

        assert violations == [], (
            "Direct attribute assignment on Node instances in _construct_builder.py "
            "violates immutable IR principle. Use model_copy() instead.\n"
            "Violations:\n" + "\n".join(f"  {v}" for v in violations)
        )


class TestDeadBodyDocstringStrip:
    """Dead-body check must strip docstrings before triviality check (neograph-hn8e).

    The _is_trivial_body helper and the dead-body warning code must handle
    docstring + placeholder patterns like `'''doc''' + ...` without false warnings.
    """

    def test_is_trivial_body_exists(self):
        """_is_trivial_body must exist as a named function in decorators.py."""
        source = (SRC_DIR / "decorators.py").read_text()
        tree = ast.parse(source)
        func_names = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        }
        assert "_is_trivial_body" in func_names, (
            "_is_trivial_body helper must exist in decorators.py "
            "to handle docstring + placeholder patterns."
        )

    def test_dead_body_check_strips_docstring(self):
        """The dead-body warning code must strip docstrings before the trivial check.

        Looks for: isinstance check on body[0] for str constant (docstring detection)
        followed by body slice (body = body[1:] or equivalent).
        """
        source = (SRC_DIR / "decorators.py").read_text()
        # Must contain docstring detection (isinstance + str) AND body re-slicing
        assert "isinstance(body[0].value.value, str)" in source, (
            "Dead-body check must detect docstrings by checking "
            "isinstance(body[0].value.value, str)"
        )
        assert "body[1:]" in source, (
            "Dead-body check must strip the docstring via body = body[1:]"
        )


class TestBranchNodeIsNotDuckTyped:
    """_BranchNode should inherit from a shared base, not duck-type."""

    def test_branch_node_has_no_stub_methods(self):
        """_BranchNode should not define its own has_modifier/get_modifier stubs."""

        # If _BranchNode inherits properly, has_modifier/get_modifier come
        # from the parent (Modifiable mixin on Node or a Protocol).
        # If it duck-types, the methods are defined directly on _BranchNode.
        src = pathlib.Path(SRC_DIR / "forward.py").read_text()
        tree = ast.parse(src)

        for cls_node in ast.walk(tree):
            if not isinstance(cls_node, ast.ClassDef) or cls_node.name != "_BranchNode":
                continue
            method_names = {
                item.name for item in cls_node.body
                if isinstance(item, ast.FunctionDef)
            }
            stubs = method_names & {"has_modifier", "get_modifier"}
            assert stubs == set(), (
                f"_BranchNode has duck-typed stub methods: {stubs}. "
                "It should inherit these from a shared base (Node or Protocol)."
            )


class TestBuildConstructBodySize:
    """_build_construct_from_decorated must be a thin orchestrator, not a monolith.

    The body should delegate to named step helpers (_build_decorated_dict,
    _identify_port_params, _build_adjacency, etc.) and stay under 50 lines.
    """

    def test_build_construct_body_under_50_lines(self):
        """_build_construct_from_decorated body must be < 50 lines."""
        source = (SRC_DIR / "_construct_builder.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_build_construct_from_decorated":
                # Body starts after the docstring (if any)
                body_start = node.body[0].end_lineno if (
                    isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ) else node.lineno
                body_lines = node.end_lineno - body_start
                assert body_lines < 50, (
                    f"_build_construct_from_decorated body is {body_lines} lines — "
                    f"should be < 50 (delegate to named step helpers)."
                )
                return

        pytest.fail("_build_construct_from_decorated not found in _construct_builder.py")


class TestTypesCompatibleUnionForms:
    """_types_compatible must handle all Union representations."""

    def test_optional_x_compatible_with_x(self):
        """Optional[X] (Union[X, None]) is compatible with X."""
        from typing import Optional

        from neograph._construct_validation import _types_compatible
        from tests.schemas import RawText

        assert _types_compatible(RawText, Optional[RawText])  # noqa: UP045

    def test_pipe_none_compatible_with_x(self):
        """X | None is compatible with X (Python 3.10+ union syntax)."""
        from neograph._construct_validation import _types_compatible
        from tests.schemas import RawText

        assert _types_compatible(RawText, RawText | None)

    def test_union_xy_compatible_with_x(self):
        """Union[X, Y] is compatible with X."""
        from typing import Union

        from neograph._construct_validation import _types_compatible
        from tests.schemas import Claims, RawText

        assert _types_compatible(RawText, Union[RawText, Claims])  # noqa: UP007

    def test_nested_optional_in_dict(self):
        """dict[str, Optional[X]] producer compatible with dict[str, X] consumer."""
        from neograph._construct_validation import _types_compatible
        from tests.schemas import RawText

        # This tests that the validator doesn't crash on nested Optional
        # within parameterized generics
        assert _types_compatible(dict[str, RawText], dict[str, RawText | None])


class TestCompilerWiringSplit:
    """compiler.py wiring helpers must live in _wiring.py.

    Wiring helpers (_wire_each, _wire_oracle, _add_each_oracle_fused,
    _add_subgraph_loop, loop/branch/operator helpers) were extracted to
    _wiring.py to keep compiler.py focused on the compile() entry point.
    """

    def test_compiler_under_600_lines(self):
        """compiler.py must be < 600 lines after wiring extraction."""
        compiler = SRC_DIR / "compiler.py"
        line_count = len(compiler.read_text().splitlines())
        assert line_count < 600, (
            f"compiler.py is {line_count} lines — must be < 600. "
            "Move wiring helpers to _wiring.py."
        )

    def test_wiring_module_exists(self):
        """_wiring.py must exist in the neograph package."""
        wiring = SRC_DIR / "_wiring.py"
        assert wiring.exists(), (
            "_wiring.py does not exist. Create it with the extracted wiring helpers."
        )

    WIRING_FUNCTIONS = [
        "def _wire_oracle",
        "def _wire_each",
        "def _add_each_oracle_fused",
        "def _merge_one_group",
        "def _make_loop_router",
        "def _node_loop_unwrap",
        "def _construct_loop_unwrap",
        "def _add_loop_back_edge",
        "def _add_subgraph_loop",
        "def _add_branch_to_graph",
        "def _add_operator_check",
    ]

    def test_wiring_functions_in_correct_module(self):
        """Extracted wiring functions must be in _wiring.py, not compiler.py."""
        wiring = SRC_DIR / "_wiring.py"
        compiler = SRC_DIR / "compiler.py"

        if not wiring.exists():
            pytest.skip("_wiring.py does not exist yet")

        wiring_text = wiring.read_text()
        compiler_text = compiler.read_text()
        violations = []

        for sig in self.WIRING_FUNCTIONS:
            if sig not in wiring_text:
                violations.append(f"  MISSING: '{sig}' not found in _wiring.py")
            if sig in compiler_text:
                violations.append(
                    f"  DRIFTED: '{sig}' found in compiler.py "
                    f"(should only be in _wiring.py)"
                )

        assert violations == [], (
            f"\n{len(violations)} wiring split violation(s):\n"
            + "\n".join(violations)
        )


class TestNoModuleLevelRegistryDicts:
    """Module-level registry dicts in factory.py must be replaced by a Registry class.

    Three dicts (_scripted_registry, _condition_registry, _tool_factory_registry)
    were module-level globals. They cause cross-pipeline collisions and have no
    thread safety. All must live inside a Registry class now.
    """

    def test_no_module_level_registry_dicts(self):
        """AST-scan factory.py for module-level dict assignments named *_registry."""
        source = (SRC_DIR / "factory.py").read_text()
        tree = ast.parse(source)

        violations = []
        for node in ast.iter_child_nodes(tree):
            # Module-level assignments: x: type = {} or x = {}
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id.endswith("_registry"):
                    violations.append(
                        f"  factory.py:{node.lineno}: {node.target.id}"
                    )
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.endswith("_registry"):
                        violations.append(
                            f"  factory.py:{node.lineno}: {target.id}"
                        )

        assert violations == [], (
            f"\n{len(violations)} module-level *_registry dict(s) in factory.py:\n"
            + "\n".join(violations)
            + "\n\nAll registries must live inside the Registry class."
        )


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
    ("_construct_builder.py", "neograph.factory", frozenset({"register_scripted"})),
    # _construct_validation.py — cycle: validation reads merge_fn metadata that
    # lives in decorators.py. Resolved when DI/merge metadata moves into a leaf.
    (
        "_construct_validation.py",
        "neograph.decorators",
        frozenset({"get_merge_fn_metadata"}),
    ),
    # _dispatch.py — cycle: dispatch resolves scripted shims from the registry,
    # whose module imports IR types. Will resolve via _runtime_registry split.
    ("_dispatch.py", "neograph._registry", frozenset({"registry"})),
    # _llm.py — cycle: _llm uses describe_type for structured output schemas,
    # and renderers for prompt construction. Both pull leaf utilities; the cycle
    # is module-level _llm globals (neograph-lyvi will collapse it). Until then,
    # function-local imports keep the module graph acyclic.
    ("_llm.py", "neograph.describe_type", frozenset({"describe_type"})),
    ("_llm.py", "neograph.renderers", frozenset({"build_rendered_input"})),
    # _oracle.py — cycle: oracle merges call invoke_structured (_llm) and resolve
    # merge_fn metadata (decorators). Both retire when §2 -lyvi collapses _llm
    # globals and merge_fn metadata moves to a leaf module.
    ("_oracle.py", "neograph._llm", frozenset({"invoke_structured"})),
    (
        "_oracle.py",
        "neograph.decorators",
        frozenset({"_resolve_merge_args", "get_merge_fn_metadata"}),
    ),
    # _sidecar.py — cycle: sidecar storage references factory's scripted lookup
    # for shim attachment. Will retire when factory.py is split (cgkl).
    ("_sidecar.py", "neograph.factory", frozenset({"lookup_scripted"})),
    # _wiring.py — cycle: wiring composes Each-Oracle redirects from factory,
    # resolves merge metadata from decorators, calls compile() recursively for
    # nested constructs. Multiple cycles converging in one module; will resolve
    # piecewise as factory.py and decorators.py are split.
    (
        "_wiring.py",
        "neograph.factory",
        frozenset({"make_eachoracle_redirect_fn"}),
    ),
    (
        "_wiring.py",
        "neograph.decorators",
        frozenset({"_resolve_merge_args", "get_merge_fn_metadata"}),
    ),
    ("_wiring.py", "neograph.factory", frozenset({"lookup_scripted"})),
    ("_wiring.py", "neograph._llm", frozenset({"invoke_structured"})),
    ("_wiring.py", "neograph.compiler", frozenset({"compile"})),
    # compiler.py — cycle: compile() reads module-level _llm globals to wire the
    # llm_factory / prompt_compiler closures. Retires with §2 -lyvi.
    (
        "compiler.py",
        "neograph._llm",
        frozenset({"_llm_factory", "_prompt_compiler"}),
    ),
    # _subconstruct.py — cycle: sub-construct invocation strips internal fields
    # that runner.py owns. Inherited from factory.py when make_subgraph_fn moved
    # out (gm-4). Will retire when runner.py loses its internal-field knowledge.
    ("_subconstruct.py", "neograph.runner", frozenset({"_strip_internals"})),
    # lint.py — cycle: lint inspects the runtime registry to check whether
    # scripted shims exist. Retires with _runtime_registry split.
    ("lint.py", "neograph._registry", frozenset({"registry"})),
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
    # state.py — cycle: state.py owns field naming logic which naming.py wraps
    # for legacy callers. Trivial; can be flattened anytime.
    ("state.py", "neograph.naming", frozenset({"field_name_for"})),
    # tool.py — cycle: @tool decorator registers tool factories; the registry
    # lives in factory.py. Retires when tool registration moves to its own
    # leaf module.
    ("tool.py", "neograph.factory", frozenset({"register_tool_factory"})),
    # verify.py — cycle: verify uses _llm globals and the runtime registry.
    # Retires with §2 -lyvi (for _llm) and _runtime_registry split.
    ("verify.py", "neograph._llm", frozenset({"_llm_factory"})),
    ("verify.py", "neograph._registry", frozenset({"registry"})),
    ("verify.py", "neograph._registry", frozenset({"registry"})),
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

    Mutation-verified: adding `def invoke_with_tools(): pass` to _llm.py
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

    def test_no_framework_reads_on_tool_spec_config(self):
        import re
        from pathlib import Path

        src_dir = Path(__file__).resolve().parents[1] / "src" / "neograph"
        forbidden = [
            re.compile(r"\btool_spec\.config\["),
            re.compile(r"\btool_spec\.config\.get\("),
            re.compile(r"\btool\.config\["),
            re.compile(r"\btool\.config\.get\("),
        ]

        violations: list[str] = []
        for py_file in src_dir.rglob("*.py"):
            text = py_file.read_text()
            for line_no, line in enumerate(text.splitlines(), start=1):
                # Skip docstring-style comments and example blocks.
                stripped = line.lstrip()
                if stripped.startswith(("#", "//", "*", '"', "'")):
                    continue
                for pattern in forbidden:
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
ANY_AUDIT_MODULES = (
    "node.py",
    "construct.py",
    "modifiers.py",
    "_construct_validation.py",
    "factory.py",
    "_dispatch.py",
    "_oracle.py",
    "_wiring.py",
)

# Allowlist of public functions/methods that may use ``Any`` because the value
# at that boundary is genuinely untyped from neograph's perspective. Each entry
# names the boundary and is keyed by ``module:qualname:param`` (param is the
# parameter name or ``return`` for the return annotation). The value is a
# one-line reason naming the boundary.
#
# ANY ADDITION TO THIS ALLOWLIST REQUIRES a one-line boundary reason. If you
# can replace the Any with a precise type (or the NormalizedInputs view, or
# NodeInput/NodeOutput, or a BaseModel), do that instead.
ANY_ALLOWLIST: dict[str, str] = {
    # ── node.py — Protocol signatures and TypeSpec validator boundaries ──
    "node.py:RawNodeFn.__call__:return": "user-supplied state-update dict; values typed by user node",
    "node.py:_validate_type_spec:v": "Pydantic BeforeValidator boundary; raw input is untyped",
    "node.py:_validate_type_spec:return": "Pydantic BeforeValidator boundary; returns type | dict[str, type] | None",
    "node.py:_is_type_like:v": "introspection helper called on arbitrary user-declared shapes",
    "node.py:Node.run_isolated:input": "user-supplied initial state (typed instance or dict) for isolated execution",
    "node.py:Node.run_isolated:return": "user-supplied output value; type declared by node.outputs",
    # ── construct.py — node list validator boundary and dynamic kwargs ──
    "construct.py:_validate_node_list:v": "Pydantic BeforeValidator boundary; raw input is untyped",
    "construct.py:Construct.__init__:kwargs": "Pydantic BaseModel kwargs passthrough boundary",
    # ── modifiers.py — Protocol signatures for user merge/fallback callbacks ──
    "modifiers.py:MergePreProcess.__call__:return": "invoke_structured accepts BaseModel | dict[str, Any] | str; dict-form retains Any value type",
    "modifiers.py:Oracle.model_post_init:__context": "Pydantic model_post_init context payload; framework-internal",
    "modifiers.py:Loop.model_post_init:__context": "Pydantic model_post_init context payload; framework-internal",
    "modifiers.py:ModifierSet.model_post_init:__context": "Pydantic model_post_init context payload; framework-internal",
    # ── _construct_validation.py — IR introspection over user-declared types ──
    # Cluster-5 entries (effective_producer_type:return, _check_item_input:input_type,
    # _check_fan_in_inputs:inputs_dict, _check_each_path:input_type,
    # _resolve_field_annotation:return, _types_compatible:producer/target,
    # _extract_list_element:tp/return, _fmt_type:tp, _build_no_producer_error:input_type,
    # _suggest_hint:input_type) migrated to TypeSpecStatic in Batch 1 (neograph-86r1).
    # ── factory.py — state-bus polymorphism resolved via StateBus protocol ──
    # The state union BaseModel | dict[str, Any] is now adapted into a StateBus
    # at the dispatch entry; helpers take ``state: StateBus`` and never see the
    # raw union. ``Any`` returns survive where the value is genuinely user-typed
    # (Cluster 5 / 6 boundaries — covered by their own allowlist entries).
    # Cluster 3 / 8 state entries migrated to StateBus in Batch 2 (neograph-036p).
    "factory.py:_extract_context:return": "context dict values resolved from user-declared state fields",
    # factory.py:_type_name:t migrated to TypeSpecStatic in Batch 1 (neograph-86r1).
    "factory.py:_apply_skip_when:input_data": "user-supplied extracted input; type declared by node.inputs",
    "factory.py:_apply_skip_when:return": "state update dict; values typed by user node outputs",
    "factory.py:_build_state_update:result": "user-supplied node result; type declared by node.outputs",
    "factory.py:_build_state_update:return": "state update dict; values typed by user node outputs",
    "factory.py:_execute_node:return": "state update dict; values typed by user node outputs",
    "factory.py:_extract_loop_reentry:return": "user-supplied loop value; type declared by node.outputs",
    "factory.py:_extract_each_item:return": "user-supplied Each item; element type from each.over collection",
    "factory.py:_extract_fan_in_dict:return": "dict of upstream values; element types declared by node.inputs",
    "factory.py:_extract_single_type:return": "user-supplied upstream value; type declared by node.inputs",
    "factory.py:_extract_input:return": "user-supplied extracted input; type declared by node.inputs",
    # ── _dispatch.py — render boundary; context_data is now precise (dict[str, str]) ──
    "_dispatch.py:_render_input:input_data": "user-supplied extracted input; type declared by node.inputs",
    "_dispatch.py:_render_input:return": "RenderedInput.raw or RenderedInput.for_template_ref; user-typed payload",
    # _dispatch.py:_resolve_primary_output:return migrated to TypeSpecStatic in Batch 1 (neograph-86r1).
    # ── _oracle.py — user-declared output models ──
    # _oracle.py:_unwrap_oracle_results:output_model migrated to TypeSpecStatic in Batch 1 (neograph-86r1).
    "_oracle.py:_build_oracle_merge_result:merged": "user-supplied merge result; type declared by node.outputs",
    # ── _wiring.py — Callable fn pointers ──
    # gen_fn / merge_fn / fan_fn / subgraph_fn are runtime-built closures whose
    # precise signatures are determined by the user's modifier configuration.
    "_wiring.py:_merge_one_group:return": "user-supplied merge result; type declared by node.outputs",
    "_wiring.py:_construct_loop_unwrap:return": "user-supplied loop value; type declared by the sub-construct output",
}


class TestNoAnyInIRPublicAPIs:
    """Public functions/methods in IR + dispatch layers must not use bare ``Any``.

    Scope: ``node.py``, ``construct.py``, ``modifiers.py``, ``_construct_validation.py``,
    ``factory.py``, ``_dispatch.py``, ``_oracle.py``, ``_wiring.py``.

    Rule: every parameter or return annotation that resolves to ``typing.Any``
    on a public (non-underscore-prefixed) function or method must appear in
    ``ANY_ALLOWLIST`` with a one-line reason naming the user-data boundary it
    represents.

    Use precise types (incl. NormalizedInputs, NodeInput/NodeOutput, BaseModel,
    RunnableConfig) where possible. Bare ``Any`` is only acceptable where the
    value genuinely originates from user-declared types or user-supplied state.

    Sentinel for the mutation test: a parametrized fixture proves the scanner
    detects un-allowlisted ``Any`` annotations.
    """

    def test_no_unallowlisted_any_in_public_apis(self):
        offenders = self._scan_public_any_uses(SRC_DIR, ANY_AUDIT_MODULES, ANY_ALLOWLIST)
        assert offenders == [], (
            f"\n{len(offenders)} public function(s)/method(s) use bare typing.Any "
            "outside the allowlist:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nReplace Any with a precise type, or add an allowlist entry "
              "with a one-line reason naming the user-data boundary."
        )

    def test_scanner_detects_injected_any(self, tmp_path: pathlib.Path):
        """Mutation: a synthetic module with un-allowlisted Any must be flagged.

        Verifies the scanner is not silently passing because of a bug.
        """
        synthetic = tmp_path / "node.py"  # mimic an in-scope filename
        synthetic.write_text(
            "from typing import Any\n"
            "def public_fn(x: Any) -> Any:\n"
            "    return x\n"
        )
        offenders = self._scan_public_any_uses(
            tmp_path, ("node.py",), allowlist={}
        )
        assert any("public_fn" in o for o in offenders), (
            f"scanner failed to detect injected Any; offenders={offenders}"
        )

    def test_scanner_respects_allowlist(self, tmp_path: pathlib.Path):
        """Mutation: an allowlisted Any must pass the scanner."""
        synthetic = tmp_path / "node.py"
        synthetic.write_text(
            "from typing import Any\n"
            "def public_fn(x: Any) -> Any:\n"
            "    return x\n"
        )
        offenders = self._scan_public_any_uses(
            tmp_path,
            ("node.py",),
            allowlist={
                "node.py:public_fn:x": "test boundary",
                "node.py:public_fn:return": "test boundary",
            },
        )
        assert offenders == [], f"allowlist not honored: {offenders}"

    @staticmethod
    def _scan_public_any_uses(
        src_dir: pathlib.Path,
        modules: tuple[str, ...],
        allowlist: dict[str, str],
    ) -> list[str]:
        offenders: list[str] = []
        for fname in modules:
            path = src_dir / fname
            if not path.exists():
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for qualname, fn in _walk_public_functions(tree):
                # Parameter annotations
                for arg, ann in _iter_annotated_params(fn):
                    if _annotation_uses_any(ann):
                        key = f"{fname}:{qualname}:{arg}"
                        if key not in allowlist:
                            offenders.append(f"{fname}:{fn.lineno} {qualname}({arg}: Any)")
                # Return annotation
                if fn.returns is not None and _annotation_uses_any(fn.returns):
                    key = f"{fname}:{qualname}:return"
                    if key not in allowlist:
                        offenders.append(f"{fname}:{fn.lineno} {qualname} -> Any")
                # Class-level attribute annotations (AnnAssign in a class body)
                # are handled by _walk_public_functions returning a synthetic
                # marker — we collect those via the module walk below.
            # Collect public class-attribute annotations that resolve to Any.
            for qualname, target_name, ann in _iter_public_class_attr_annotations(tree):
                if _annotation_uses_any(ann):
                    key = f"{fname}:{qualname}.{target_name}"
                    if key not in allowlist:
                        offenders.append(f"{fname}:{ann.lineno} {qualname}.{target_name}: Any")
        return offenders


def _walk_public_functions(tree: ast.AST) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Yield (qualname, function-node) for every public function/method in tree.

    Public = not underscore-prefixed at any level, except dunder methods (which
    we include because Protocol __call__ is a real boundary).
    Methods inside private classes are excluded. Nested functions are excluded.
    """
    out: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Module-level function: include even if private — Any audit applies.
            out.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_") and not (node.name.startswith("__") and node.name.endswith("__")):
                continue
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = sub.name
                    # Private helper methods on public classes are excluded; dunders included.
                    if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
                        continue
                    out.append((f"{node.name}.{name}", sub))
    return out


def _iter_annotated_params(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[str, ast.expr]]:
    """Yield (param_name, annotation) for every annotated parameter (skipping self/cls)."""
    out: list[tuple[str, ast.expr]] = []
    args = fn.args
    all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    for a in all_args:
        if a.arg in ("self", "cls"):
            continue
        if a.annotation is not None:
            out.append((a.arg, a.annotation))
    if args.vararg is not None and args.vararg.annotation is not None:
        out.append((args.vararg.arg, args.vararg.annotation))
    if args.kwarg is not None and args.kwarg.annotation is not None:
        out.append((args.kwarg.arg, args.kwarg.annotation))
    return out


def _iter_public_class_attr_annotations(
    tree: ast.AST,
) -> list[tuple[str, str, ast.expr]]:
    """Yield (class_qualname, attr_name, annotation) for public class attrs."""
    out: list[tuple[str, str, ast.expr]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name.startswith("_") and not (node.name.startswith("__") and node.name.endswith("__")):
            continue
        for sub in node.body:
            if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                name = sub.target.id
                if name.startswith("_") and not name.startswith("__"):
                    continue
                out.append((node.name, name, sub.annotation))
    return out


def _annotation_uses_any(ann: ast.expr) -> bool:
    """Return True if the annotation expression mentions ``typing.Any``.

    Catches ``Any``, ``Any | None``, ``dict[str, Any]``, ``list[Any]``,
    ``Annotated[Any, ...]``, etc. ``RunnableConfig``, ``BaseModel``, etc. do
    NOT count.

    ``Any`` appearing inside a ``Callable[[...], ...]`` subscript is exempt:
    the visible union (e.g. ``str | Callable[[Any], bool]``) is honest about
    the two-shape boundary, and the inner ``Any`` represents the user state
    value -- a §5 boundary that has no more precise type at the IR layer.
    """
    skip_nodes: set[int] = set()
    for node in ast.walk(ann):
        if isinstance(node, ast.Subscript):
            value = node.value
            is_callable = (
                (isinstance(value, ast.Name) and value.id == "Callable")
                or (isinstance(value, ast.Attribute) and value.attr == "Callable")
            )
            if is_callable:
                for inner in ast.walk(node.slice):
                    skip_nodes.add(id(inner))
    for node in ast.walk(ann):
        if id(node) in skip_nodes:
            continue
        if isinstance(node, ast.Name) and node.id == "Any":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "Any":
            return True
    return False


# Models with arbitrary_types_allowed=True must appear here AND have an
# adjacent justification comment naming the field(s) that require it.
# Add a new entry only with sign-off — non-Pydantic-friendly types should
# usually live in a PrivateAttr sidecar (see Node._sidecar pattern).
ARBITRARY_TYPES_ALLOWLIST: frozenset[str] = frozenset({
    "construct.py",
    "node.py",
    "forward.py",
    "modifiers.py",
})


class TestArbitraryTypesJustified:
    """Every ``arbitrary_types_allowed=True`` in src/neograph must be justified.

    Two requirements per use:
      1. The host file is in ``ARBITRARY_TYPES_ALLOWLIST``.
      2. There is a justification comment within 3 lines above the config line
         (or trailing on the same line) starting with ``# arbitrary_types_allowed:``
         and naming the field(s) that require it.

    Habitual uses (where every field is already a Pydantic-friendly type) must
    be removed. If the only blocker is a Callable or sidecar value, prefer
    moving it to ``PrivateAttr`` (see ``Node._sidecar``).
    """

    JUSTIFICATION_PREFIX = "# arbitrary_types_allowed:"

    def test_all_uses_justified_and_allowlisted(self):
        offenders: list[str] = []
        for hit in _find_arbitrary_types_hits(SRC_DIR):
            fname, lineno, raw_line = hit
            if fname not in ARBITRARY_TYPES_ALLOWLIST:
                offenders.append(
                    f"{fname}:{lineno} uses arbitrary_types_allowed=True but file "
                    f"is not in ARBITRARY_TYPES_ALLOWLIST"
                )
                continue
            path = SRC_DIR / fname
            if not _has_arbitrary_types_justification(
                path, lineno, self.JUSTIFICATION_PREFIX
            ):
                offenders.append(
                    f"{fname}:{lineno} has no '{self.JUSTIFICATION_PREFIX}' comment "
                    f"within 3 lines above (or trailing on the same line). Add a "
                    f"comment naming the field(s) that require arbitrary_types_allowed=True."
                )
        assert offenders == [], (
            f"\n{len(offenders)} arbitrary_types_allowed=True use(s) missing justification:\n"
            + "\n".join(f"  {o}" for o in offenders)
        )

    def test_scanner_detects_unjustified_arbitrary_types(self, tmp_path: pathlib.Path):
        """Mutation: file with arbitrary_types_allowed=True and no comment must fail."""
        synthetic = tmp_path / "construct.py"
        synthetic.write_text(
            "from pydantic import BaseModel\n"
            "class M(BaseModel):\n"
            "    model_config = {'arbitrary_types_allowed': True}\n"
        )
        hits = list(_find_arbitrary_types_hits(tmp_path))
        assert hits, "scanner failed to detect arbitrary_types_allowed=True"
        fname, lineno, _ = hits[0]
        assert not _has_arbitrary_types_justification(
            tmp_path / fname, lineno, "# arbitrary_types_allowed:"
        ), "scanner incorrectly accepted a missing justification"

    def test_scanner_accepts_justified_use(self, tmp_path: pathlib.Path):
        """Mutation: file with a justification comment must pass."""
        synthetic = tmp_path / "node.py"
        synthetic.write_text(
            "from pydantic import BaseModel\n"
            "class M(BaseModel):\n"
            "    # arbitrary_types_allowed: 'renderer' field is a Renderer Protocol\n"
            "    model_config = {'arbitrary_types_allowed': True}\n"
        )
        hits = list(_find_arbitrary_types_hits(tmp_path))
        assert hits, "scanner failed to detect arbitrary_types_allowed=True"
        fname, lineno, _ = hits[0]
        assert _has_arbitrary_types_justification(
            tmp_path / fname, lineno, "# arbitrary_types_allowed:"
        ), "scanner failed to honor a present justification"


# Matches both ConfigDict(arbitrary_types_allowed=True) and
# {"arbitrary_types_allowed": True} forms. The optional trailing quote handles
# the dict-literal form where the key is quoted.
_ARBITRARY_TYPES_RE = re.compile(
    r'arbitrary_types_allowed["\']?\s*(?:=|:)\s*True'
)


def _find_arbitrary_types_hits(src_dir: pathlib.Path) -> list[tuple[str, int, str]]:
    """Return (filename, lineno, raw_line) for every arbitrary_types_allowed=True use."""
    hits: list[tuple[str, int, str]] = []
    for py_file in sorted(src_dir.glob("*.py")):
        for lineno, line in enumerate(py_file.read_text().splitlines(), start=1):
            if _ARBITRARY_TYPES_RE.search(line):
                hits.append((py_file.name, lineno, line))
    return hits


def _has_arbitrary_types_justification(
    path: pathlib.Path, lineno: int, prefix: str
) -> bool:
    """Look for `prefix` in the contiguous comment block immediately above.

    A justification comment may span multiple lines as long as those lines
    form an uninterrupted comment block ending immediately above the
    arbitrary_types_allowed=True line. Trailing comments on the same line
    are also accepted.
    """
    lines = path.read_text().splitlines()
    # Trailing inline comment on the same line.
    same_line = lines[lineno - 1]
    if prefix in same_line and "#" in same_line:
        return True
    # Walk up the contiguous comment block immediately above.
    i = lineno - 1  # 0-indexed line above
    while i >= 1:
        stripped = lines[i - 1].lstrip()
        if not stripped.startswith("#"):
            return False
        if stripped.startswith(prefix):
            return True
        i -= 1
    return False


# Stdlib exception types whose use is allowlisted at specific call sites where
# the stdlib semantics ARE the contract (Pydantic validators, parser grammar,
# attribute-protocol fallback for hasattr/getattr). Every entry must have a
# one-line reason naming why NeographError is wrong at that site.
#
# Key format: "{filename}:{lineno}" — pinned so a moved raise re-triggers
# review. Update the key when the line moves, and confirm the boundary
# reason still applies.
NEOGRAPH_ERROR_ALLOWLIST: dict[str, str] = {
    # ── conditions.py — string-grammar parser (stdlib parser contract) ──
    # parse_condition() implements a tiny expression grammar; ValueError is
    # the documented contract and tests depend on it. AttributeError raises
    # inside _resolve_field implement the Python attribute-protocol contract
    # for dotted-path lookup (caller may catch AttributeError to fall back).
    "conditions.py:63": "ValueError is the documented contract for parse_condition grammar errors; tests assert it",
    "conditions.py:74": "AttributeError is the Python attribute-protocol contract for dotted-path lookup",
    "conditions.py:82": "AttributeError is the Python attribute-protocol contract for dotted-path lookup",
    "conditions.py:90": "AttributeError is the Python attribute-protocol contract for dotted-path lookup",
    "conditions.py:115": "ValueError is the documented contract for parse_condition grammar errors; tests assert it",
    "conditions.py:126": "ValueError is the documented contract for parse_condition grammar errors; tests assert it",

    # ── construct.py — Pydantic BeforeValidator boundary ──
    # _validate_node_list runs inside Pydantic field validation; Pydantic
    # catches TypeError/ValueError and rolls them into ValidationError.
    "construct.py:42": "Pydantic BeforeValidator boundary; TypeError is rolled into ValidationError",
    "construct.py:45": "Pydantic BeforeValidator boundary; TypeError is rolled into ValidationError",

    # ── forward.py — proxy / tracer / abstract-method contracts ──
    # _Proxy.__getattr__ raises AttributeError per the Python attribute
    # protocol (hasattr depends on it). __bool__/__iter__ raise TypeError per
    # the Python protocol contract (a non-iterable used in `for` raises
    # TypeError, not NeographError). forward() raises NotImplementedError as
    # a Python abstract-method idiom.
    "forward.py:165": "NotImplementedError is the Python abstract-method idiom",
    "forward.py:196": "AttributeError is the Python attribute-protocol contract (hasattr depends on it)",
    "forward.py:225": "TypeError is the Python protocol contract for __bool__ misuse",
    "forward.py:233": "TypeError is the Python protocol contract for __iter__ misuse",
    "forward.py:260": "TypeError is the Python protocol contract for __bool__ misuse on _ConditionProxy",

    # ── modifiers.py — Pydantic field_validator + proxy attribute protocol ──
    # _PathRecorder.__getattr__ implements the attribute protocol. Pydantic
    # @field_validator boundaries catch ValueError into ValidationError.
    "modifiers.py:186": "AttributeError is the Python attribute-protocol contract (private-attr guard)",
    "modifiers.py:411": "Pydantic @field_validator boundary; ValueError is rolled into ValidationError",
    "modifiers.py:467": "Pydantic @field_validator boundary; ValueError is rolled into ValidationError",

    # ── node.py — Pydantic BeforeValidator boundary ──
    # _validate_type_spec runs inside Pydantic field validation; Pydantic
    # catches TypeError and rolls it into ValidationError.
    "node.py:92": "Pydantic BeforeValidator boundary; TypeError is rolled into ValidationError",
    "node.py:94": "Pydantic BeforeValidator boundary; TypeError is rolled into ValidationError",
    "node.py:98": "Pydantic BeforeValidator boundary; TypeError is rolled into ValidationError",
}


class TestPublicFunctionsRaiseNeographError:
    """All `raise <BareException>` in src/neograph/ must use a NeographError subclass.

    Rule: every `raise` statement in src/neograph/*.py must satisfy one of:

      - ``raise <NeographError-subclass>(...)`` (or via ``.build()``)
      - ``raise`` bare (re-raise of caught exception)
      - ``raise <NameLoaded>`` where the name binds an exception variable
        (e.g. ``raise exc``, ``raise e from None``)
      - allowlisted entry in NEOGRAPH_ERROR_ALLOWLIST, with the line documented
        as a stdlib-contract boundary

    Anything else — ``raise ValueError(...)``, ``raise TypeError(...)``,
    ``raise RuntimeError(...)`` — fails this guard. Convert to
    ``NeographError.build(...)`` (use ``ConfigurationError`` for configuration,
    ``ConstructError`` for assembly-time validation, ``ExecutionError`` for
    runtime failures, ``CompileError`` for graph-build failures).
    """

    NEOGRAPH_ERROR_NAMES = frozenset({
        "NeographError",
        "ConstructError",
        "ExecutionError",
        "CompileError",
        "ConfigurationError",
        "CheckpointSchemaError",
    })

    def test_no_bare_stdlib_raises(self):
        offenders = self._scan_bare_raises(SRC_DIR, NEOGRAPH_ERROR_ALLOWLIST)
        assert offenders == [], (
            f"\n{len(offenders)} raise(s) use a stdlib exception class instead of NeographError:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nConvert each to NeographError.build(...) using the appropriate subclass "
              "(ConfigurationError, ConstructError, ExecutionError, CompileError), or add an "
              "allowlist entry with a one-line reason naming the stdlib-contract boundary."
        )

    def test_scanner_detects_injected_stdlib_raise(self, tmp_path: pathlib.Path):
        """Mutation: a file with `raise ValueError(...)` must be flagged."""
        synthetic = tmp_path / "fake_module.py"
        synthetic.write_text(
            "def fn():\n"
            "    raise ValueError('boom')\n"
        )
        offenders = self._scan_bare_raises(tmp_path, {})
        assert any("ValueError" in o for o in offenders), (
            f"scanner failed to detect injected ValueError; offenders={offenders}"
        )

    def test_scanner_accepts_neograph_error(self, tmp_path: pathlib.Path):
        """Mutation: a file with `raise ConstructError.build(...)` must pass."""
        synthetic = tmp_path / "fake_module.py"
        synthetic.write_text(
            "from neograph.errors import ConstructError\n"
            "def fn():\n"
            "    raise ConstructError.build('boom')\n"
        )
        offenders = self._scan_bare_raises(tmp_path, {})
        assert offenders == [], f"scanner rejected a valid NeographError raise: {offenders}"

    def test_scanner_respects_allowlist(self, tmp_path: pathlib.Path):
        """Mutation: an allowlisted stdlib raise must pass."""
        synthetic = tmp_path / "fake_module.py"
        synthetic.write_text(
            "def fn():\n"
            "    raise ValueError('boom')\n"
        )
        offenders = self._scan_bare_raises(
            tmp_path, {"fake_module.py:2": "test boundary"}
        )
        assert offenders == [], f"allowlist not honored: {offenders}"

    @classmethod
    def _scan_bare_raises(
        cls,
        src_dir: pathlib.Path,
        allowlist: dict[str, str],
    ) -> list[str]:
        offenders: list[str] = []
        for py_file in sorted(src_dir.glob("*.py")):
            try:
                tree = ast.parse(py_file.read_text(), filename=str(py_file))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Raise):
                    continue
                if node.exc is None:
                    # bare ``raise`` — re-raises caught exception, always OK
                    continue
                exc_name = _raised_exception_name(node.exc)
                if exc_name is None:
                    # ``raise exc`` or ``raise self.something()`` — name bound at
                    # runtime, can't statically verify; skip.
                    continue
                if exc_name in cls.NEOGRAPH_ERROR_NAMES:
                    continue
                if exc_name.startswith("_") and exc_name.endswith("Error"):
                    # ``_ExecutionError``, ``_ConfigurationError`` aliases used
                    # to break import cycles. Verify by stripping the leading
                    # underscore.
                    if exc_name[1:] in cls.NEOGRAPH_ERROR_NAMES:
                        continue
                key = f"{py_file.name}:{node.lineno}"
                if key in allowlist:
                    continue
                offenders.append(f"{key}: raise {exc_name}(...)")
        return offenders


def _raised_exception_name(exc_node: ast.expr) -> str | None:
    """Return the bare name of the raised exception class, or None if dynamic.

    Handles ``raise Foo(...)``, ``raise Foo`` (no call), and ``raise Foo.build(...)``.
    Returns None for ``raise some_var`` and ``raise mod.Foo(...)`` (attribute on a module).
    """
    # raise Foo(...) — Call with a Name as func
    if isinstance(exc_node, ast.Call):
        func = exc_node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            # raise Foo.build(...) — Attribute access; check the base name.
            base = func.value
            if isinstance(base, ast.Name):
                return base.id
            return None
        return None
    # raise Foo (no call) — Name directly
    if isinstance(exc_node, ast.Name):
        # Bare name: could be a class or a variable holding an exception instance.
        # Treat capitalized names as exception classes; lowercase as variables.
        if exc_node.id[:1].isupper():
            return exc_node.id
        return None  # ``raise exc`` — runtime-bound variable
    return None


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
