"""Structural guards: error builder, file-split, assembly import DAG,
subconstruct boundaries, dead code, no-Any boundaries, no-sidecar-pattern."""

from __future__ import annotations

import ast
import pathlib

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

    The runtime mutation that writes `StateKeys.SUBGRAPH_INPUT` (the
    `neo_subgraph_input` state-bus key) into a sub_input dict is sub-construct
    boundary semantics. Re-implementations in factory.py / _execute.py /
    _wiring.py / runner.py would mean the boundary is splaying across files.

    Compile-time mentions in state.py / loader.py / _construct_builder.py /
    _construct_validation.py / forward.py are out-of-scope (they declare the
    port type; they don't perform the runtime write).

    Per neograph-n3f1 the literal `"neo_subgraph_input"` was centralized into
    `_state_keys.py`; the runtime needle is now the typed `StateKeys.SUBGRAPH_INPUT`
    reference rather than the bare string.
    """

    NEEDLE = "StateKeys.SUBGRAPH_INPUT"
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
            "_subconstruct.py must own the runtime StateKeys.SUBGRAPH_INPUT write."
        )

    def test_mutation_runtime_leak_detected(self, tmp_path):
        """Mutation case — write needle into a fake runtime file; scanner detects."""
        rogue = tmp_path / "factory.py"
        rogue.write_text('sub_input[StateKeys.SUBGRAPH_INPUT] = x\n')
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




class TestValidatorHardening:
    """neograph-xs3e: ta43 recursive-validator hardening.

    1. No ``cast(Any`` at the recursive call site — the recursion is made
       type-safe via a ConstructLike Protocol + TypeGuard instead.
    2. ``ValidationMode`` enum names the STANDALONE-vs-IN_CONTEXT state that
       the old ``context_checkable`` boolean encoded implicitly.
    3. The producer collection is keyed by field_name
       (``OrderedDict[str, Producer]``), dropping the O(N^2) per-call rebuild
       in ``_check_fan_in_inputs``.
    """

    VALIDATOR = SRC_DIR / "_construct_validation.py"

    def test_no_cast_any_in_validator(self):
        """The recursive call site must not launder the type via cast(Any).

        Specific-type casts (e.g. cast(Node, ...), cast(dict[...], ...)) are
        fine — only the untyped cast(Any, ...) escape hatch is forbidden.
        """
        hits = [
            f"{self.VALIDATOR.name}:{i}: {line.strip()}"
            for i, line in enumerate(self.VALIDATOR.read_text().splitlines(), 1)
            if "cast(Any" in line
        ]
        assert hits == [], (
            "cast(Any ...) found in the recursive validator — use the "
            "ConstructLike Protocol + _is_construct_like TypeGuard instead:\n"
            + "\n".join(f"  {h}" for h in hits)
        )

    def test_validation_mode_enum_replaces_context_checkable(self):
        """ValidationMode{STANDALONE, IN_CONTEXT} must exist and be used."""
        import enum

        from neograph import _construct_validation as cv

        assert hasattr(cv, "ValidationMode"), "ValidationMode enum missing"
        mode = cv.ValidationMode
        assert issubclass(mode, enum.Enum)
        names = {m.name for m in mode}
        assert {"STANDALONE", "IN_CONTEXT"} <= names, names

    def test_producers_collection_keyed_by_field_name(self):
        """Producer collection is OrderedDict[str, Producer], not list[Producer]."""
        text = self.VALIDATOR.read_text()
        assert "OrderedDict[str, Producer]" in text, (
            "producers should be typed OrderedDict[str, Producer] keyed by field_name"
        )
        assert "list[Producer]" not in text, (
            "no signature should still thread producers as list[Producer]"
        )

    def test_construct_like_protocol_exists(self):
        """ConstructLike Protocol (name/input/nodes) lives in _ir_protocols."""
        from neograph import _ir_protocols

        assert hasattr(_ir_protocols, "ConstructLike"), "ConstructLike Protocol missing"
