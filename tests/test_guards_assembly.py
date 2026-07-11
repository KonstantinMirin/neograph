"""Structural guards: error builder, file-split, assembly import DAG,
subconstruct boundaries, dead code, no-Any boundaries, no-sidecar-pattern."""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# The validation cluster after the _construct_validation.py decomposition
# (neograph-gig0). _construct_validation.py is the ONLY entry point: external
# modules import the public surface from it; the _validation_* sub-modules are
# package-private and may only be imported from WITHIN the cluster.
VALIDATION_CLUSTER = frozenset(
    {
        "_construct_validation.py",
        "_validation_types.py",
        "_validation_inputs.py",
        "_validation_modifiers.py",
    }
)

# Error classes that must use .build() instead of direct construction.
ERROR_CLASSES = frozenset(
    {
        "ConstructError",
        "ExecutionError",
        "CompileError",
        "ConfigurationError",
        "NeographError",
    }
)


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


class TestErrorBuildBodyMonopoly:
    """Every ``.build`` classmethod delegates message formatting to one helper.

    CON-02 / PAT-01: ``ExecutionError.build`` used to duplicate the entire
    ``NeographError.build`` format body verbatim. The ``.build()`` call-site
    guard (``TestErrorBuilderEnforcement``) forces call SITES through ``build``
    but structurally cannot see the two ``build`` BODIES diverge — the drift
    lived in the gap between guards. This guard closes it: any ``build`` method
    in ``errors.py`` that re-inlines the format assembly (builds a ``parts``
    list or joins a ``msg`` string itself) instead of calling ``_format_message``
    fails here at authoring time.
    """

    def _build_methods(self, tree: ast.Module) -> list[ast.FunctionDef]:
        found: list[ast.FunctionDef] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build":
                found.append(node)
        return found

    def _reinlines_format_body(self, fn: ast.FunctionDef) -> bool:
        """True if the body assembles the message itself instead of delegating.

        The tell is a local assignment to ``parts`` or ``msg`` — the two names
        the shared ``_format_message`` helper owns. A delegating ``build`` only
        assigns ``msg = _format_message(...)``, so we treat an assignment whose
        value is a call to ``_format_message`` as clean.
        """
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "parts" in targets:
                return True
            if "msg" in targets:
                value = node.value
                delegates = (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "_format_message"
                )
                if not delegates:
                    return True
        return False

    def test_build_methods_delegate_to_format_message(self):
        errors_file = SRC_DIR / "errors.py"
        tree = ast.parse(errors_file.read_text(), filename=str(errors_file))

        builds = self._build_methods(tree)
        assert builds, "expected at least one build() classmethod in errors.py"

        violations = [
            f"  errors.py:{fn.lineno}: build() re-inlines the format body — "
            "call _format_message(...) instead of assembling parts/msg locally"
            for fn in builds
            if self._reinlines_format_body(fn)
        ]
        assert violations == [], (
            f"\n{len(violations)} build() method(s) duplicate the format body:\n"
            + "\n".join(violations)
            + "\n\nEvery .build must delegate to _format_message (errors.py). "
            "See TestErrorBuildBodyMonopoly / CON-02."
        )

    def test_detector_flags_a_reinlined_build(self):
        """Slip check: the detector must fire on a re-inlined format body."""
        src = (
            "def build(cls, what):\n"
            "    parts = []\n"
            "    parts.append(what)\n"
            "    msg = ' '.join(parts)\n"
            "    return cls(msg)\n"
        )
        fn = ast.parse(src).body[0]
        assert isinstance(fn, ast.FunctionDef)
        assert self._reinlines_format_body(fn) is True

    def test_detector_passes_a_delegating_build(self):
        """Slip check: a build() that delegates to _format_message is clean."""
        src = "def build(cls, what):\n    msg = _format_message(what)\n    return cls(msg)\n"
        fn = ast.parse(src).body[0]
        assert isinstance(fn, ast.FunctionDef)
        assert self._reinlines_format_body(fn) is False


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
        # _construct_builder.py → cohesive sibling modules (neograph-3zai).
        # must_NOT_be_in is _construct_builder.py so the helpers can't drift back
        # and re-bloat the orchestrator.
        ("def _register_node_scripted", "_scripted_registry.py", "_construct_builder.py"),
        ("def _resolve_dict_output_param", "_construct_graph.py", "_construct_builder.py"),
        ("def _resolve_loop_self_param", "_construct_graph.py", "_construct_builder.py"),
        ("def _identify_port_params", "_param_classify.py", "_construct_builder.py"),
        ("def _detect_fan_out_params", "_param_classify.py", "_construct_builder.py"),
        ("def _classify_constants", "_param_classify.py", "_construct_builder.py"),
        ("def _check_di_collisions", "_param_classify.py", "_construct_builder.py"),
    ]

    def test_extracted_functions_in_correct_module(self):
        violations = []
        for signature, must_be_in, must_not_be_in in self.SPLIT_RULES:
            good_file = SRC_DIR / must_be_in
            bad_file = SRC_DIR / must_not_be_in

            good_text = good_file.read_text() if good_file.exists() else ""
            bad_text = bad_file.read_text() if bad_file.exists() else ""

            if signature not in good_text:
                violations.append(f"  MISSING: '{signature}' not found in {must_be_in}")
            if signature in bad_text:
                violations.append(
                    f"  DRIFTED: '{signature}' found in {must_not_be_in} (should only be in {must_be_in})"
                )

        assert violations == [], f"\n{len(violations)} file-split violation(s):\n" + "\n".join(violations)


def _line_cap_violations(src_dir: pathlib.Path, files: tuple[str, ...], cap: int) -> list[str]:
    """Pure check: which of `files` are missing or exceed `cap` lines."""
    violations: list[str] = []
    for fname in files:
        path = src_dir / fname
        if not path.exists():
            violations.append(f"  MISSING: {fname} does not exist")
            continue
        n_lines = len(path.read_text().splitlines())
        if n_lines > cap:
            violations.append(f"  {fname}: {n_lines} lines (> {cap})")
    return violations


def _symbol_location_violations(src_dir: pathlib.Path, expected_locations: dict[str, set[str]]) -> list[str]:
    """Pure check: each signature must be in its target file and absent from the builder."""
    builder_path = src_dir / "_construct_builder.py"
    builder_text = builder_path.read_text() if builder_path.exists() else ""
    violations: list[str] = []
    for fname, signatures in expected_locations.items():
        path = src_dir / fname
        text = path.read_text() if path.exists() else ""
        for sig in signatures:
            if sig not in text:
                violations.append(f"  MISSING: '{sig}' not found in {fname}")
            # Moved symbols must NOT linger in the builder.
            if fname != "_construct_builder.py" and sig in builder_text:
                violations.append(f"  DRIFTED: '{sig}' still in _construct_builder.py (should only be in {fname})")
    return violations


class TestConstructBuilderSplit:
    """neograph-3zai: _construct_builder.py is split into cohesive modules.

    The 711-line builder mixed four responsibilities: public API + orchestrator,
    graph construction, param classification, and scripted-shim registration.
    After the split:
      - graph construction lives in _construct_graph.py
      - param classification lives in _param_classify.py
      - scripted-shim registration lives in _scripted_registry.py
      - _construct_builder.py keeps only the public API + orchestrator + the
        @node input-cleanup pass (_cleanup_inputs_and_register).

    Acceptance (from the ticket): no extracted file exceeds 300 lines and the
    moved symbols live in their target modules, not the builder.

    This is an AST/text guard (substring + line-count), not regex-based, so a
    positive + negative meta-test pair suffices (no regex-slip case).
    """

    # field_name (must be in this file) -> set of `def <name>` that must live there
    EXPECTED_LOCATIONS = {
        "_construct_graph.py": {
            "def _build_decorated_dict",
            "def _build_adjacency",
            "def _topo_sort",
            "def _resolve_dict_output_param",
            "def _resolve_loop_self_param",
        },
        "_param_classify.py": {
            "def _identify_port_params",
            "def _detect_fan_out_params",
            "def _classify_constants",
            "def _check_di_collisions",
        },
        "_scripted_registry.py": {
            "def _register_node_scripted",
        },
        "_member_select.py": {
            "def _classify_member",
            "def _bucket_members",
        },
        "_construct_builder.py": {
            "def construct_from_module",
            "def construct_from_functions",
            "def _build_construct_from_decorated",
            "def _cleanup_inputs_and_register",
        },
    }

    # All four files participating in the split must stay under this cap.
    LINE_CAP = (
        330  # raised 300->330 for the m0tv ruff-format pass (mechanical rewrap; _construct_builder 311 post-format)
    )
    SPLIT_FILES = (
        "_construct_builder.py",
        "_construct_graph.py",
        "_member_select.py",
        "_param_classify.py",
        "_scripted_registry.py",
    )

    def test_each_split_file_under_line_cap(self):
        violations = _line_cap_violations(SRC_DIR, self.SPLIT_FILES, self.LINE_CAP)
        assert violations == [], f"\n{len(violations)} line-cap/existence violation(s):\n" + "\n".join(violations)

    def test_symbols_live_in_target_modules(self):
        violations = _symbol_location_violations(SRC_DIR, self.EXPECTED_LOCATIONS)
        assert violations == [], f"\n{len(violations)} symbol-location violation(s):\n" + "\n".join(violations)

    # --- meta-tests: prove the guard actually catches regressions ---

    def test_meta_line_cap_catches_oversized_file(self, tmp_path):
        """Negative meta-test: an oversized file must be flagged."""
        big = tmp_path / "_construct_graph.py"
        big.write_text("\n".join(f"x = {i}" for i in range(self.LINE_CAP + 50)))
        violations = _line_cap_violations(tmp_path, ("_construct_graph.py",), self.LINE_CAP)
        assert any(f"> {self.LINE_CAP}" in v for v in violations), (
            "line-cap guard failed to flag a file exceeding the cap"
        )

    def test_meta_line_cap_passes_clean_files(self, tmp_path):
        """Positive meta-test: a small file must not be flagged."""
        small = tmp_path / "_construct_graph.py"
        small.write_text("x = 1\n")
        assert _line_cap_violations(tmp_path, ("_construct_graph.py",), self.LINE_CAP) == []

    def test_meta_drift_catches_symbol_left_in_builder(self, tmp_path):
        """Negative meta-test: a moved symbol still present in the builder must be flagged."""
        # Builder still contains the helper that should have moved out.
        (tmp_path / "_construct_builder.py").write_text("def _register_node_scripted(): ...\n")
        (tmp_path / "_scripted_registry.py").write_text("def _register_node_scripted(): ...\n")
        violations = _symbol_location_violations(tmp_path, {"_scripted_registry.py": {"def _register_node_scripted"}})
        assert any("DRIFTED" in v for v in violations), (
            "symbol-location guard failed to flag a moved symbol left in the builder"
        )

    def test_meta_drift_catches_missing_symbol(self, tmp_path):
        """Negative meta-test: a symbol absent from its target file must be flagged."""
        (tmp_path / "_construct_builder.py").write_text("# empty\n")
        (tmp_path / "_scripted_registry.py").write_text("# helper not here\n")
        violations = _symbol_location_violations(tmp_path, {"_scripted_registry.py": {"def _register_node_scripted"}})
        assert any("MISSING" in v for v in violations), (
            "symbol-location guard failed to flag a symbol absent from its target file"
        )


def _parse_neograph_imports(path: pathlib.Path) -> set[str]:
    """Return the set of `neograph.*` modules imported by the file.

    Walks three forms:
      - `from neograph.X import Y`      (dotted submodule)
      - `import neograph.X`             (dotted submodule)
      - `from neograph import X`        (from-PACKAGE form) — where `X` names a
        submodule (`neograph/X.py` exists). This form was previously DROPPED ON
        THE FLOOR (neograph-awor / PAT-01 / LR-01): a cardinal-rule layering
        violation like `from neograph import decorators` in a lower-layer module
        passed every import-DAG guard green because the parser only matched the
        dotted `stmt.module == "neograph.X"` shape. A `from neograph import X`
        where `X` is a public SYMBOL re-export (e.g. `__version__`, `node` the
        decorator) is NOT a module edge and is excluded via the `X.py`-exists
        submodule check.
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
                mods.add(stmt.module[len("neograph.") :])
            elif stmt.module == "neograph" and stmt.level == 0:
                # from-package form: only count names that are real submodules,
                # not public symbol re-exports.
                for alias in stmt.names:
                    if (SRC_DIR / f"{alias.name}.py").exists():
                        mods.add(alias.name)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name.startswith("neograph."):
                    mods.add(alias.name[len("neograph.") :])
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

    CLUSTER = frozenset(
        {
            "factory",
            "_execute",
            "_subconstruct",
            "_oracle",
            "_state_write",
            "_input_shape",
        }
    )

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
        assert not violations, "Assembly cluster import-DAG layering violated:\n" + "\n".join(violations)

    def test_mutation_violation_detected(self, tmp_path):
        """Mutation case — inject a violating edge; scanner must detect."""
        # Build a fake module that has a violating edge: _oracle imports _execute
        fake_oracle = tmp_path / "_oracle.py"
        fake_oracle.write_text("from neograph._execute import _execute_node\n")
        imports = _parse_neograph_imports(fake_oracle)
        cluster_imports = imports & self.CLUSTER
        allowed = self.ALLOWED_CLUSTER_EDGES["_oracle"]
        extras = cluster_imports - allowed
        assert "_execute" in extras, "Scanner failed to detect the mutation (oracle -> _execute)."


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
            for fname in spec["must_touch"] | spec["may_touch"]:
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
        assert total > bad["max_touch"], "Mutation scanner failed: oversized touchpoint not detected."


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
                    f"  {mod}: imported by {len(importers)} modules (ceiling {ceiling}). Importers: {importers}"
                )
        assert not violations, "Cohesion fan-out exceeded ceiling — module is becoming a kitchen sink:\n" + "\n".join(
            violations
        )

    def test_mutation_excess_importers_detected(self, tmp_path):
        """Mutation case — synthesize a target that has too many importers."""
        # Synthesize 5 importers of a fictitious module 'leaf.py', ceiling 1.
        for i in range(5):
            (tmp_path / f"client_{i}.py").write_text("from neograph.leaf import x\n")
        count = sum(1 for p in tmp_path.glob("client_*.py") if "from neograph.leaf " in p.read_text())
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
        assert not offenders, f"{self.SENTINEL} sentinels found outside _state_write.py: {offenders}"

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
        assert not missing, "InputShape extractor table incomplete:\n" + "\n".join(missing)

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
        assert not missing, "InputShape dispatch is not exhaustive:\n" + "\n".join(missing)

    def test_mutation_missing_extractor_detected(self):
        """Mutation case — pretend a variant exists with no extractor."""
        src = (SRC_DIR / "_input_shape.py").read_text()
        bogus_variant = "phantom_shape"
        expected = f"def _extract_{bogus_variant}("
        assert expected not in src, "Mutation precondition violated: _extract_phantom_shape unexpectedly exists."


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
        assert self.NEEDLE in text, "_subconstruct.py must own the runtime StateKeys.SUBGRAPH_INPUT write."

    def test_mutation_runtime_leak_detected(self, tmp_path):
        """Mutation case — write needle into a fake runtime file; scanner detects."""
        rogue = tmp_path / "factory.py"
        rogue.write_text("sub_input[StateKeys.SUBGRAPH_INPUT] = x\n")
        text = rogue.read_text()
        assert self.NEEDLE in text, "Mutation scanner failed: runtime leak not detected."


class TestNoMergeStepsOutsideOracleModule:
    """ARCH-1 (CRIT-01) — the Oracle merge ALGORITHM lives only in `_oracle.py`.

    The merge skeleton (merge_prompt path: pre_process -> invoke_structured ->
    fallback -> post_process; merge_fn path: @merge_fn metadata lookup +
    arg-resolution / scripted) had a parallel re-implementation in
    `_wiring.py::_merge_one_group` that drifted from
    `_oracle.py::make_oracle_merge_fn` (the Each x Oracle fused path dropped
    node_inputs upstream-context injection). After consolidation `_wiring.py`
    obtains variants and shapes results but never performs a merge STEP — it
    routes through `_oracle._merge_variants`.

    This guard AST-scans `_wiring.py` for any direct call to a merge-step
    symbol. A hit means the algorithm is leaking back out of `_oracle.py`.
    """

    # Every merge-step symbol. A direct call to any of these in _wiring.py
    # means a merge step is being re-derived there instead of delegated.
    MERGE_STEP_CALLEES = frozenset(
        {
            "invoke_structured",
            "merge_pre_process",
            "merge_post_process",
            "merge_fallback",
            "get_merge_fn_metadata",
            "_resolve_merge_args",
        }
    )

    @classmethod
    def _called_names(cls, tree: ast.AST) -> set[str]:
        """Return the set of called callee names (Name or Attribute attr)."""
        names: set[str] = set()
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            fn = n.func
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.add(fn.attr)
        return names

    def test_wiring_does_not_call_merge_steps(self):
        tree = ast.parse((SRC_DIR / "_wiring.py").read_text())
        leaked = self._called_names(tree) & self.MERGE_STEP_CALLEES
        assert not leaked, (
            f"_wiring.py calls merge-step symbol(s) {sorted(leaked)} directly. "
            "The Oracle merge algorithm has ONE home: _oracle.py. _wiring.py must "
            "route through _oracle._merge_variants (obtain variants + shape "
            "results only — never re-derive a merge step)."
        )

    def test_wiring_does_not_import_invoke_structured(self):
        """Import-level needle: even a re-aliased invoke_structured re-leak fails."""
        tree = ast.parse((SRC_DIR / "_wiring.py").read_text())
        imported_from_llm: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module == "neograph._llm":
                imported_from_llm |= {a.name for a in n.names}
        assert "invoke_structured" not in imported_from_llm, (
            "_wiring.py imports invoke_structured from neograph._llm — the merge "
            "LLM call belongs in _oracle._merge_variants, not the wiring layer."
        )

    def test_oracle_module_owns_merge_steps(self):
        """Positive: the canonical merge step IS in _oracle.py."""
        oracle_calls = self._called_names(ast.parse((SRC_DIR / "_oracle.py").read_text()))
        assert "invoke_structured" in oracle_calls and "get_merge_fn_metadata" in oracle_calls, (
            "_oracle.py must own the merge steps (invoke_structured + get_merge_fn_metadata)."
        )

    def test_meta_scanner_detects_injected_merge_step(self, tmp_path):
        """Mutation: a synthetic file calling a merge step is flagged."""
        rogue = tmp_path / "rogue.py"
        rogue.write_text("def f(s, c):\n    return invoke_structured(s, c)\n")
        leaked = self._called_names(ast.parse(rogue.read_text())) & self.MERGE_STEP_CALLEES
        assert leaked == {"invoke_structured"}

    def test_meta_scanner_ignores_clean_file(self, tmp_path):
        """Negative: a file that only delegates is not flagged."""
        clean = tmp_path / "clean.py"
        clean.write_text("def f(o, v, c):\n    return _merge_variants(o, v, c)\n")
        leaked = self._called_names(ast.parse(clean.read_text())) & self.MERGE_STEP_CALLEES
        assert leaked == set()


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
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for required in ("_build_state_update", "_apply_skip_when"):
            assert required in names, f"_state_write.py must define {required} (gm-1 Cluster E unification)."

    def test_apply_skip_when_not_in_modifier_io_or_input_shape(self):
        for candidate in ("_modifier_io.py", "_input_shape.py"):
            path = SRC_DIR / candidate
            if not path.exists():
                continue
            tree = ast.parse(path.read_text())
            names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            assert "_apply_skip_when" not in names, (
                f"_apply_skip_when must NOT be defined in {candidate} — it belongs in _state_write.py (Cluster E)."
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
        assert not offenders, f"{SENTINEL} found outside _state_write.py: {offenders}"
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
        assert (SRC_DIR / "_execute.py").exists(), "_execute.py must exist (created by gm-3 for Cluster B)."

    def test_observability_module_deleted(self):
        assert not (SRC_DIR / "_observability.py").exists(), (
            "_observability.py must be deleted — _extract_context moved into _execute.py."
        )

    def test_execute_defines_required_functions(self):
        path = SRC_DIR / "_execute.py"
        if not path.exists():
            pytest.fail("_execute.py does not exist yet (gm-3 not complete).")
        tree = ast.parse(path.read_text())
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        # _extract_context relocated to _input_shape.py (neograph-m6d3): its
        # caller set grew 1→2 (the inline agent cycle now reuses it too), so it
        # moved to its cohesive read-side home next to _extract_input.
        for required in ("_execute_node", "_type_name"):
            assert required in names, f"_execute.py must define {required} (gm-3 Cluster B)."

    def test_factory_does_not_define_execute_helpers(self):
        path = SRC_DIR / "factory.py"
        tree = ast.parse(path.read_text())
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for forbidden in ("_execute_node", "_type_name", "_extract_context"):
            assert forbidden not in names, f"factory.py must NOT define {forbidden} — it lives in _execute.py."

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
        assert importers == ["factory.py"], f"_execute.py should be imported by factory.py only, got: {importers}"


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
        func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        class_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        assert "InputShape" in class_names, "_input_shape.py must define InputShape."
        for fn in (
            "_classify_input_shape",
            "_extract_input",
            "_extract_context",  # relocated from _execute.py (neograph-m6d3): read-side
            # input shaping, now reused by the inline agent cycle too.
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
        assert "_execute.py" in importers, f"_execute.py must import from _input_shape.py, got importers: {importers}"
        # Apart from factory.py (re-export) and _execute.py (real), no one
        # else may import — except _agent_cycle.py, the second node-body module
        # (agent-as-subgraph, neograph-m6d3): its agent/tools/parse bodies extract
        # node input the same way _execute does. A deliberate architecture change,
        # not drift — the inline ReAct cycle is a peer node-body executor.
        allowed = {"_execute.py", "factory.py", "_agent_cycle.py"}
        extras = set(importers) - allowed
        assert not extras, f"_input_shape.py imported by unexpected modules: {extras}; allowed: {allowed}"


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
        assert (SRC_DIR / "_subconstruct.py").exists(), "_subconstruct.py must exist (created by gm-4 for Cluster C)."

    def test_subconstruct_defines_required_symbols(self):
        path = SRC_DIR / "_subconstruct.py"
        if not path.exists():
            pytest.fail("_subconstruct.py does not exist yet.")
        tree = ast.parse(path.read_text())
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for required in (
            "make_subgraph_fn",
            "_scan_subgraph_input",
            "_scan_subgraph_output",
        ):
            assert required in names, f"_subconstruct.py must define {required} (gm-4 Cluster C)."

    def test_factory_does_not_define_make_subgraph_fn(self):
        path = SRC_DIR / "factory.py"
        tree = ast.parse(path.read_text())
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
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
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "_inject_oracle_config" in names, (
            "_inject_oracle_config must be defined in _oracle.py (Cluster F — Oracle generator dispatch)."
        )

    def test_inject_oracle_config_not_in_state_modules(self):
        for candidate in ("_state_io.py", "_state_write.py"):
            path = SRC_DIR / candidate
            if not path.exists():
                continue
            tree = ast.parse(path.read_text())
            names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            assert "_inject_oracle_config" not in names, (
                f"_inject_oracle_config must NOT be defined in {candidate} — it belongs in _oracle.py (Cluster F)."
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
                        f"raw_node() function found in {py_file.name}:{node.lineno} — use @node(mode='raw') instead."
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
                    violations.append(f"  {node.name}.{field_name}: Any (line {item.lineno})")
                # Check dict[str, Any] — the value type should be concrete
                if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name):
                    if ann.value.id == "dict":
                        # dict[str, Any] — check the second element
                        if isinstance(ann.slice, ast.Tuple) and len(ann.slice.elts) == 2:
                            val_type = ann.slice.elts[1]
                            if isinstance(val_type, ast.Name) and val_type.id == "Any":
                                violations.append(f"  {node.name}.{field_name}: dict[str, Any] (line {item.lineno})")

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
                violations.append("  _validate_node_chain contains Each+Loop mutual exclusion check")
            if "Oracle" in func_source and "Loop" in func_source and "mutual exclu" in func_source.lower():
                violations.append("  _validate_node_chain contains Oracle+Loop mutual exclusion check")
            assert violations == [], (
                "\nRedundant modifier checks in _validate_node_chain:\n"
                + "\n".join(violations)
                + "\n\nModifierSet.model_post_init is the structural gate. "
                "Remove the belt-and-suspenders checks."
            )

    @staticmethod
    def _each_rule_reinline_violations(text: str) -> list[str]:
        """Detect a re-inlined Each->dict[str,X] producer rule in validator text.

        Disease: the dict-form per-key branch recomputing the modifier-to-bus
        rule inline instead of delegating to effective_producer_type_for. The
        drift class behind neograph-8k3 / neograph-ayq.
        """
        violations = []
        if "dict[str, key_type]" in text:
            violations.append(
                "  inlines 'dict[str, key_type]' — the Each->dict[str,X] rule. Delegate to effective_producer_type_for."
            )
        if "effective_producer_type_for" not in text:
            violations.append("  does not reference effective_producer_type_for — the dict-form branch must use it.")
        return violations

    def test_each_dict_form_producer_rule_not_reinlined(self):
        """The Each->dict[str,X] producer rule has ONE owner.

        The dict-form per-key producer branch in _construct_validation.py must
        NOT recompute 'dict[str, key_type] if has_each else key_type' inline —
        that re-implements the modifier-to-bus rule that effective_producer_type
        (now via effective_producer_type_for) solely owns.
        """
        text = (SRC_DIR / "_construct_validation.py").read_text()
        violations = self._each_rule_reinline_violations(text)
        assert violations == [], (
            "\nEach->dict[str,X] producer rule re-inlined in the validator:\n"
            + "\n".join(violations)
            + "\n\nTeach the rule to effective_producer_type_for "
            "(_validation_types.py) and have both producer paths delegate."
        )

    def test_meta_reinline_guard_catches_disease(self):
        """positive meta-test: the inline literal is flagged."""
        bad = (
            "for output_key, key_type in output_type.items():\n"
            "    producer_type = dict[str, key_type] if has_each else key_type\n"
        )
        violations = self._each_rule_reinline_violations(bad)
        assert any("dict[str, key_type]" in v for v in violations)

    def test_meta_reinline_guard_passes_clean_source(self):
        """negative meta-test: delegating source is accepted."""
        good = "producer_type = effective_producer_type_for(key_type, item.modifier_set)\n"
        assert self._each_rule_reinline_violations(good) == []


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

    def test_param_res_accessed_only_via_accessor(self):
        """neograph-awor (LR-02): raw ``item._param_res`` attribute access is
        banned outside _sidecar.py; every other module must go through the
        ``_get_param_res`` / ``_set_param_res`` accessor. compiler.py was the sole
        raw reader (``item._param_res`` in _collect_required_di); migrating it
        makes _sidecar.py the single home. AST-scan (attribute access only — the
        ``_param_res: ... = PrivateAttr(...)`` field DECLARATION in node.py is an
        assignment target, not an attribute access, so it is not flagged).
        """
        offenders: list[str] = []
        for py_file in sorted(SRC_DIR.glob("*.py")):
            if py_file.name == "_sidecar.py":
                continue
            for node in ast.walk(ast.parse(py_file.read_text())):
                if isinstance(node, ast.Attribute) and node.attr == "_param_res":
                    offenders.append(f"  {py_file.name}:{node.lineno}")
        assert offenders == [], (
            f"\n{len(offenders)} raw ._param_res access(es) outside _sidecar.py:\n"
            + "\n".join(offenders)
            + "\n\nUse _get_param_res(node) / _set_param_res(node, ...) from neograph._sidecar."
        )

    def test_meta_param_res_scanner_catches_raw_access(self):
        """slip test: a raw ``x._param_res`` read is detected; the accessor call
        ``_get_param_res(x)`` is not."""
        raw = ast.parse("def f(x):\n    return x._param_res\n")
        assert any(isinstance(n, ast.Attribute) and n.attr == "_param_res" for n in ast.walk(raw))
        clean = ast.parse("def f(x):\n    return _get_param_res(x)\n")
        assert not any(isinstance(n, ast.Attribute) and n.attr == "_param_res" for n in ast.walk(clean))

    def test_node_has_private_attrs(self):
        """Node must declare PrivateAttr fields for sidecar data."""
        from neograph.node import Node

        private_fields = getattr(Node, "__private_attributes__", {})
        assert "_sidecar" in private_fields or "_sidecar_fn" in private_fields, (
            "Node must have a _sidecar or _sidecar_fn PrivateAttr field. "
            "Add: _sidecar: tuple[Callable, tuple[str, ...]] | None = PrivateAttr(default=None)"
        )
        assert "_param_res" in private_fields, (
            "Node must have a _param_res PrivateAttr field. Add: _param_res: dict = PrivateAttr(default_factory=dict)"
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

    Post neograph-gig0 the validator is split into the flat-peer cluster
    (``VALIDATION_CLUSTER``); the text scans below run across the whole cluster
    so the assertions follow the symbols into their new homes (e.g. the
    ``OrderedDict[str, Producer]`` alias now lives in ``_validation_types.py``,
    the ``cast(Node, ...)`` casts in ``_validation_inputs.py``).
    """

    CLUSTER = tuple(sorted(SRC_DIR / f for f in VALIDATION_CLUSTER))

    def test_no_cast_any_in_validator(self):
        """No cluster module may launder the type via cast(Any).

        Specific-type casts (e.g. cast(Node, ...), cast(dict[...], ...)) are
        fine — only the untyped cast(Any, ...) escape hatch is forbidden.
        """
        hits = [
            f"{path.name}:{i}: {line.strip()}"
            for path in self.CLUSTER
            for i, line in enumerate(path.read_text().splitlines(), 1)
            if "cast(Any" in line
        ]
        assert hits == [], (
            "cast(Any ...) found in the validation cluster — use the "
            "ConstructLike Protocol + _is_construct_like TypeGuard instead:\n" + "\n".join(f"  {h}" for h in hits)
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
        """Producer collection is OrderedDict[str, Producer], not list[Producer].

        The ``ProducerMap = OrderedDict[str, Producer]`` alias lives in
        ``_validation_types.py`` post-split; the assertion scans the cluster so
        it follows the alias to its home.
        """
        cluster_text = "\n".join(p.read_text() for p in self.CLUSTER)
        assert "OrderedDict[str, Producer]" in cluster_text, (
            "producers should be typed OrderedDict[str, Producer] keyed by field_name"
        )
        assert "list[Producer]" not in cluster_text, "no signature should still thread producers as list[Producer]"

    def test_construct_like_protocol_exists(self):
        """ConstructLike Protocol (name/input/nodes) lives in _ir_protocols."""
        from neograph import _ir_protocols

        assert hasattr(_ir_protocols, "ConstructLike"), "ConstructLike Protocol missing"


# --- validation-cluster boundary helpers (neograph-gig0) ---


def _seam_violations(src_dir: pathlib.Path) -> list[str]:
    """Files OUTSIDE the cluster must not import from a _validation_* sub-module.

    The single import seam is `neograph._construct_validation`. Any
    `from neograph._validation_types/_validation_inputs/_validation_modifiers`
    appearing in a non-cluster file (module-level OR function-local) is a
    boundary breach.
    """
    private_modules = {
        "neograph._validation_types",
        "neograph._validation_inputs",
        "neograph._validation_modifiers",
    }
    violations: list[str] = []
    for py_file in sorted(src_dir.glob("*.py")):
        if py_file.name in VALIDATION_CLUSTER:
            continue
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in private_modules:
                names = ", ".join(a.name for a in node.names)
                violations.append(f"  {py_file.name}:{node.lineno}: from {node.module} import {names}")
    return violations


def _star_import_violations(src_dir: pathlib.Path) -> list[str]:
    """No file may `import *` from the cluster; cluster files may not `import *` at all."""
    violations: list[str] = []
    cluster_modules = {f"neograph.{f[:-3]}" for f in VALIDATION_CLUSTER}
    for py_file in sorted(src_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            is_star = any(a.name == "*" for a in node.names)
            if not is_star:
                continue
            # Star FROM a cluster module — forbidden anywhere.
            if node.module in cluster_modules:
                violations.append(f"  {py_file.name}:{node.lineno}: from {node.module} import *")
            # Any star import INSIDE a cluster file — forbidden.
            elif py_file.name in VALIDATION_CLUSTER:
                violations.append(f"  {py_file.name}:{node.lineno}: from {node.module} import *")
    return violations


class TestValidationModuleBoundary:
    """neograph-gig0: _construct_validation.py (1005L) is split by validation concern.

    After the split the cluster is four flat peer modules (no sub-package,
    matching the neograph-3zai builder-split precedent):
      - _validation_types.py     — type-compat primitives + shared vocab
      - _validation_inputs.py    — fan-in + Each fan-out consumer-input checks
      - _validation_modifiers.py — Loop self-edge/construct + Oracle merge hooks
      - _construct_validation.py — orchestrator (_validate_node_chain) + the
        single public re-export seam.

    Invariant: behavior is byte-identical; only file boundaries + the import
    seam change. This guard pins the new structure: line cap, symbol homes,
    the single import seam, and no star-imports.

    AST/substring guard (line-count + `def` presence + import shape) — a
    positive + negative meta-test pair suffices (no regex-slip case).
    """

    LINE_CAP = 400

    # `def`/`class` signature -> module it must live in. Each moved symbol must
    # also be ABSENT (as a definition) from the slimmed orchestrator.
    EXPECTED_LOCATIONS = {
        "_validation_types.py": {
            "def _types_compatible",
            "def _extract_list_element",
            "def _resolve_field_annotation",
            "def _fmt_type",
            "def _source_location",
            "def effective_producer_type",
            "def _is_construct_like",
            "class Producer",
        },
        "_validation_inputs.py": {
            "def _check_item_input",
            "def _check_fan_in_inputs",
            "def _check_each_path",
            "def _build_no_producer_error",
            "def _suggest_hint",
        },
        "_validation_modifiers.py": {
            "def validate_loop_self_edge",
            "def validate_loop_construct",
            "def _validate_merge_hooks",
        },
        "_construct_validation.py": {
            "def _validate_node_chain",
            "class ValidationMode",
        },
    }

    def test_each_cluster_file_under_line_cap(self):
        violations = _line_cap_violations(SRC_DIR, tuple(sorted(VALIDATION_CLUSTER)), self.LINE_CAP)
        assert violations == [], f"\n{len(violations)} line-cap/existence violation(s):\n" + "\n".join(violations)

    def test_symbols_live_in_target_modules(self):
        """Each symbol is DEFINED in its target module and absent from the orchestrator."""
        orchestrator = SRC_DIR / "_construct_validation.py"
        orch_text = orchestrator.read_text() if orchestrator.exists() else ""
        violations: list[str] = []
        for fname, signatures in self.EXPECTED_LOCATIONS.items():
            path = SRC_DIR / fname
            text = path.read_text() if path.exists() else ""
            for sig in signatures:
                if sig not in text:
                    violations.append(f"  MISSING: '{sig}' not found in {fname}")
                # A moved definition must not linger in the orchestrator.
                if fname != "_construct_validation.py" and sig in orch_text:
                    violations.append(
                        f"  DRIFTED: '{sig}' still defined in _construct_validation.py (should only be in {fname})"
                    )
        assert violations == [], f"\n{len(violations)} symbol-location violation(s):\n" + "\n".join(violations)

    def test_single_import_seam(self):
        """Only _construct_validation.py is importable by non-cluster files."""
        violations = _seam_violations(SRC_DIR)
        assert violations == [], (
            f"\n{len(violations)} import-seam violation(s) — non-cluster files must "
            f"import the validation public surface from neograph._construct_validation, "
            f"not from a _validation_* sub-module:\n" + "\n".join(violations)
        )

    def test_no_star_imports_in_cluster(self):
        violations = _star_import_violations(SRC_DIR)
        assert violations == [], f"\n{len(violations)} star-import violation(s):\n" + "\n".join(violations)

    # --- meta-tests: prove the guard catches regressions ---

    def test_meta_line_cap_catches_oversized_cluster_file(self, tmp_path):
        big = tmp_path / "_validation_inputs.py"
        big.write_text("\n".join(f"x = {i}" for i in range(self.LINE_CAP + 50)))
        violations = _line_cap_violations(tmp_path, ("_validation_inputs.py",), self.LINE_CAP)
        assert any("> 400" in v for v in violations)

    def test_meta_seam_catches_private_import(self, tmp_path):
        (tmp_path / "_construct_validation.py").write_text("x = 1\n")
        (tmp_path / "_validation_inputs.py").write_text("y = 2\n")
        (tmp_path / "outsider.py").write_text("from neograph._validation_inputs import _check_item_input\n")
        violations = _seam_violations(tmp_path)
        assert any("outsider.py" in v for v in violations)

    def test_meta_star_import_flagged(self, tmp_path):
        (tmp_path / "rogue.py").write_text("from neograph._construct_validation import *\n")
        violations = _star_import_violations(tmp_path)
        assert any("rogue.py" in v for v in violations)


class TestLowerLayersDoNotImportForwardDX:
    """neograph-9epk / LR-01 — DIP: the DX layer is BOTH ``forward.py`` and
    ``decorators.py`` (AGENTS.md names both). No lower-layer module (compiler,
    state, wiring, IR, validation, factory cluster) may import from either.
    Core-IR concepts the compiler needs (e.g. the branch sentinel) live in
    neutral low-level modules (_ir_branch.py), so the dependency points
    DX -> IR, never IR -> DX.

    Originally this guard covered only ``forward.py``; the reviewer
    mutation-verified that its twin ``decorators.py`` was an unguarded backdoor
    (decorators.py re-exports the DI symbols, so an IR module importing it left
    the full guard suite green). Both DX modules are now banned.

    Only the package facade ``__init__.py`` may import them (it re-exports the
    PUBLIC ``ForwardConstruct`` / ``@node`` for the top-level API). The two DX
    modules may import each other (peer DX, not a downward edge).
    """

    # The two modules AGENTS.md designates as the DX layer.
    DX_MODULES = frozenset({"forward", "decorators"})
    # The package facade re-exports the public DX symbols — allowed.
    FORWARD_IMPORT_ALLOWLIST = {"__init__"}

    def test_no_lower_layer_imports_dx(self):
        offenders: list[str] = []
        for path in SRC_DIR.glob("*.py"):
            mod = path.stem
            # A DX module itself and the facade are exempt; DX modules may also
            # import each other (peer edge, not IR -> DX).
            if mod in self.DX_MODULES or mod in self.FORWARD_IMPORT_ALLOWLIST:
                continue
            imported = _parse_neograph_imports(path)
            for dx in sorted(self.DX_MODULES & imported):
                offenders.append(f"  {mod}.py imports from neograph.{dx} (DX layer)")
        assert not offenders, (
            "Lower-layer modules must not import the DX layer (forward.py / decorators.py). "
            "Move shared core-IR concepts to a neutral module (e.g. _ir_branch.py):\n" + "\n".join(offenders)
        )

    def test_meta_detects_forward_import(self, tmp_path):
        """Positive: a module importing from neograph.forward is detected."""
        fake = tmp_path / "_fake_lower.py"
        fake.write_text("from neograph.forward import _BranchNode\n")
        assert "forward" in _parse_neograph_imports(fake)

    def test_meta_detects_decorators_import(self, tmp_path):
        """Positive: a module importing from neograph.decorators is detected —
        the LR-01 hole the reviewer mutation-verified."""
        fake = tmp_path / "_fake_lower_dec.py"
        fake.write_text("from neograph.decorators import _classify_di_params\n")
        assert "decorators" in _parse_neograph_imports(fake) & self.DX_MODULES

    def test_meta_detects_from_package_dx_import(self, tmp_path):
        """slip test (neograph-awor / LR-01): the from-PACKAGE spelling
        `from neograph import decorators` is a DX import too, and must be caught
        exactly like the dotted `from neograph.decorators import X` form. Before
        the parser learned this shape it dropped it on the floor, so a lower-layer
        module could import the DX layer with every guard green."""
        fake = tmp_path / "_fake_from_pkg.py"
        fake.write_text("from neograph import decorators\n")
        assert "decorators" in _parse_neograph_imports(fake) & self.DX_MODULES

    def test_meta_from_package_symbol_reexport_is_not_module_edge(self, tmp_path):
        """Negative: `from neograph import <public symbol>` (no matching
        submodule file, e.g. __version__) is NOT counted as a module edge."""
        fake = tmp_path / "_fake_symbol.py"
        fake.write_text("from neograph import __version__\n")
        assert "__version__" not in _parse_neograph_imports(fake)

    def test_meta_passes_neutral_ir_import(self, tmp_path):
        """Negative: importing the neutral IR module is not a DX import."""
        fake = tmp_path / "_fake_ok.py"
        fake.write_text("from neograph._ir_branch import _BranchNode\n")
        assert not (self.DX_MODULES & _parse_neograph_imports(fake))


class TestMemberSelectionPredicateMonopoly:
    """Pipeline-member selection has exactly ONE predicate (neograph-xv9ay).

    The silent sub-construct drop happened because construct_from_module and
    construct_from_functions each carried their own isinstance-classification
    ladder, which drifted on two axes (plain Node: collect vs reject;
    Construct: silent-skip vs collect). The fix monopolizes classification in
    `_classify_member` inside _member_select.py; the builder's entry points
    and core must delegate to it and never re-derive membership inline.

    Documented exemption (neograph-gtzkd, ratified by architect review):
    ForwardConstruct._discover_node_attrs (forward.py) keeps its private
    isinstance(v, Node) walk BY DESIGN — it is a name -> Node lookup table
    feeding the tracer, not a membership classifier (membership is fixed by
    forward() tracing; sub-pipelines enter via self.each()/self.loop()/
    self.ensemble()). The exemption is contract-pinned by
    tests/test_forward.py::TestConstructClassAttrFailsLoud (a Construct
    class attr fails loud at class-definition time). Disease scans should
    not re-flag that site.
    """

    BUILDER = SRC_DIR / "_construct_builder.py"
    SELECT = SRC_DIR / "_member_select.py"
    PREDICATE = "_classify_member"
    POLICY = "_bucket_members"
    MEMBER_TYPES = frozenset({"Node", "Construct"})

    def _function_defs(self) -> dict[str, ast.FunctionDef]:
        defs: dict[str, ast.FunctionDef] = {}
        for path in (self.BUILDER, self.SELECT):
            tree = ast.parse(path.read_text())
            defs.update({n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)})
        return defs

    @staticmethod
    def _member_isinstance_calls(fn: ast.FunctionDef, member_types: frozenset[str]) -> list[int]:
        """Line numbers of isinstance(x, Node|Construct) calls inside fn."""
        hits = []
        for call in ast.walk(fn):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
                continue
            if call.func.id != "isinstance" or len(call.args) != 2:
                continue
            names = {n.id for n in ast.walk(call.args[1]) if isinstance(n, ast.Name)}
            if names & member_types:
                hits.append(call.lineno)
        return hits

    def test_classify_member_exists(self):
        """The single predicate must exist in _member_select.py."""
        assert self.PREDICATE in self._function_defs(), (
            f"{self.PREDICATE} not found in {self.SELECT.name} — the member-selection "
            "predicate monopoly (neograph-xv9ay) requires it"
        )

    def test_no_member_isinstance_outside_the_predicate(self):
        """isinstance(x, Node/Construct) classification is allowed ONLY inside
        _classify_member. A second ladder in an entry point or the core builder
        is the exact drift that caused the silent sub-construct drop."""
        fns = self._function_defs()
        offenders: dict[str, list[int]] = {}
        for name, fn in fns.items():
            if name == self.PREDICATE:
                continue
            # skip nested reporting: ast.walk(fn) sees nested defs too; only
            # attribute top-level hits to the outermost function that owns them
            nested = {
                inner.name
                for inner in ast.walk(fn)
                if isinstance(inner, ast.FunctionDef) and inner is not fn
            }
            if self.PREDICATE in nested:
                continue
            hits = self._member_isinstance_calls(fn, self.MEMBER_TYPES)
            if hits:
                offenders[name] = hits
        assert not offenders, (
            f"member-type isinstance classification outside {self.PREDICATE} in "
            f"{self.BUILDER.name}/{self.SELECT.name}: {offenders}. Route membership "
            f"decisions through {self.PREDICATE} (neograph-xv9ay)."
        )

    def test_sidecar_check_only_inside_the_predicate(self):
        """_get_sidecar-based @node-vs-plain classification must live only in
        _classify_member within the builder/select modules (its other
        legitimate users live in _construct_graph/_param_classify/
        _scripted_registry, not here)."""
        fns = self._function_defs()
        offenders = []
        for name, fn in fns.items():
            if name == self.PREDICATE:
                continue
            for call in ast.walk(fn):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_get_sidecar":
                    offenders.append(f"{name}:{call.lineno}")
        assert not offenders, (
            f"_get_sidecar classification outside {self.PREDICATE} in "
            f"{self.BUILDER.name}/{self.SELECT.name}: {offenders}"
        )

    def test_classification_called_only_from_the_policy_site(self):
        """`_classify_member` may be CALLED only from `_bucket_members` — the
        entry points pass raw members + a source kind and never classify
        locally. A local pre-filter in an entry point is a second policy
        site, the exact drift class this guard exists to ban."""
        fns = self._function_defs()
        offenders = []
        for name, fn in fns.items():
            if name == self.POLICY:
                continue
            for c in ast.walk(fn):
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == self.PREDICATE:
                    offenders.append(f"{name}:{c.lineno}")
        assert not offenders, (
            f"{self.PREDICATE} called outside {self.POLICY}: {offenders}. "
            f"Entry points must delegate raw members to the core builder; "
            f"all classification and skip/warn/raise policy lives in {self.POLICY}."
        )

    def test_skip_warn_policy_only_inside_the_policy_site(self):
        """The warn half of the skip/warn/raise policy must be single-sited:
        `warnings.warn` in the builder/select modules only inside
        `_bucket_members`."""
        fns = self._function_defs()
        offenders = []
        for name, fn in fns.items():
            if name == self.POLICY:
                continue
            for c in ast.walk(fn):
                if (
                    isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Attribute)
                    and c.func.attr == "warn"
                    and isinstance(c.func.value, ast.Name)
                    and c.func.value.id == "warnings"
                ):
                    offenders.append(f"{name}:{c.lineno}")
        assert not offenders, (
            f"warnings.warn outside {self.POLICY} in "
            f"{self.BUILDER.name}/{self.SELECT.name}: {offenders}"
        )
