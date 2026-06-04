"""Structural guards: helper-monopoly enforcement for neograph-wpzg.

neograph-wpzg extracted three single-source-of-truth helpers after finding the
same logic hand-rolled across >=2 call sites in the DX-layer codegen/tracer
files. These guards pin the monopolies so a future PR cannot silently
re-introduce an inline variant:

  - forward.py   _attr_chain_after_prefix  — strips the ``out_of_<node>`` proxy
                  prefix (the parse idiom ``full_name[len(prefix):]`` may live
                  ONLY inside the helper).
  - forward.py   _build_condition_spec     — builds a _ConditionSpec from a
                  branch condition (the truthy-fallback ``op_str="truthy"`` spec
                  may live ONLY inside the helper).
  - testing.py   _emit_set_block           — emits an ``EXPECTED_* = {...}`` set
                  literal (the set-open codegen idiom may live ONLY inside it).

Each monopoly is enforced two ways:
  1. the helper is called at least N times (AST call-site count), and
  2. its unique inline idiom appears exactly once in the owner file (the def).

The idiom check normalizes whitespace before counting, so a spacing variant
(``[len( prefix ):]``) cannot slip — proven by the regex-slip meta-test.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# helper name -> (owner file, minimum call sites excluding the def)
MONOPOLIES = {
    "_attr_chain_after_prefix": ("forward.py", 2),
    "_build_condition_spec": ("forward.py", 2),
    "_emit_set_block": ("testing.py", 4),
}

# helper name -> the unique inline idiom that must appear exactly once (the def).
# Stored whitespace-stripped; the scanner strips whitespace from the source too.
IDIOM_SIGNATURES = {
    "_attr_chain_after_prefix": "full_name[len(prefix):]",
    "_build_condition_spec": 'op_str="truthy"',
    "_emit_set_block": "lines=[f'    {field_name} = {{']",
}


def _count_calls(source: str, func_name: str) -> int:
    """Count call sites ``func_name(...)`` in source (AST, excludes the def)."""
    tree = ast.parse(source)
    count = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == func_name
        ):
            count += 1
    return count


def _has_def(source: str, func_name: str) -> bool:
    tree = ast.parse(source)
    return any(
        isinstance(n, ast.FunctionDef) and n.name == func_name
        for n in ast.walk(tree)
    )


def _normalized_idiom_count(source: str, idiom: str) -> int:
    """Count whitespace-insensitive occurrences of ``idiom`` in source."""
    squashed = "".join(source.split())
    needle = "".join(idiom.split())
    return squashed.count(needle)


class TestHelperMonopolyWpzg:
    """neograph-wpzg: extracted helpers are the single source for their logic.

    AST/normalized-text guard. Positive + negative + regex-slip meta-tests.
    """

    def test_helpers_are_defined_and_called_at_least_n_times(self):
        violations: list[str] = []
        for helper, (fname, min_calls) in MONOPOLIES.items():
            source = (SRC_DIR / fname).read_text()
            if not _has_def(source, helper):
                violations.append(f"  MISSING DEF: {helper} not defined in {fname}")
                continue
            calls = _count_calls(source, helper)
            if calls < min_calls:
                violations.append(
                    f"  UNDER-USED: {helper} called {calls}x in {fname} "
                    f"(monopoly requires >= {min_calls}); did a call site get inlined?"
                )
        assert violations == [], (
            "\nHelper-monopoly call-site violations:\n" + "\n".join(violations)
        )

    def test_inline_idiom_appears_only_in_its_helper(self):
        violations: list[str] = []
        for helper, idiom in IDIOM_SIGNATURES.items():
            fname = MONOPOLIES[helper][0]
            source = (SRC_DIR / fname).read_text()
            n = _normalized_idiom_count(source, idiom)
            if n != 1:
                violations.append(
                    f"  BYPASS: idiom for {helper} appears {n}x in {fname} "
                    f"(must be exactly 1 — only inside the helper). Idiom: {idiom!r}"
                )
        assert violations == [], (
            "\nHelper-monopoly inline-bypass violations:\n" + "\n".join(violations)
        )

    # --- meta-tests: prove the guards catch regressions ---

    def test_meta_call_count_catches_inlined_call_site(self):
        """Negative: a helper with too few call sites is flagged."""
        source = "def _emit_set_block(a, b):\n    return [a, b]\n\n_emit_set_block(1, 2)\n"
        # only 1 call; monopoly requires >= 4
        assert _count_calls(source, "_emit_set_block") < MONOPOLIES["_emit_set_block"][1]

    def test_meta_idiom_count_catches_duplicate(self):
        """Negative: the same idiom appearing twice is flagged."""
        dup = (
            "def helper():\n"
            "    remainder = full_name[len(prefix):]\n"
            "def bypass():\n"
            "    remainder = full_name[len(prefix):]\n"
        )
        assert _normalized_idiom_count(dup, "full_name[len(prefix):]") == 2

    def test_meta_idiom_count_resists_whitespace_slip(self):
        """Regex-slip: a spacing variant must NOT evade the idiom scanner."""
        spaced = "def bypass():\n    remainder = full_name[ len( prefix ) : ]\n"
        # A naive substring scan of the exact idiom would return 0 here and let
        # the bypass slip; the normalized scanner must still count it as 1.
        assert spaced.count("full_name[len(prefix):]") == 0  # naive scan misses it
        assert _normalized_idiom_count(spaced, "full_name[len(prefix):]") == 1


# Functions that legitimately read their OWN loop-append field's latest value
# (after get_required / shape classification guarantees a non-empty list); these
# are NOT the consumer/scan unwrap that di._unwrap_loop_value owns.
_OWN_FIELD_READ_ALLOWLIST = frozenset({"subgraph_node"})


def _list_latest_bypass_sites(
    source: str, allowlist: frozenset[str]
) -> list[tuple[str, str]]:
    """Find functions that both ``isinstance(X, list)``-test and ``X[-1]``-subscript
    the SAME name — the hand-rolled loop-append latest-unwrap idiom that must
    delegate to ``di._unwrap_loop_value`` instead (neograph-ovx1).

    Per-function scoping avoids cross-function false positives (e.g. a shape
    check on ``own_val`` in one function and a guaranteed own-field ``own_val[-1]``
    in another). Returns (function_name, variable_name) pairs.
    """
    tree = ast.parse(source)
    out: list[tuple[str, str]] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name in allowlist:
            continue
        isinstance_list: set[str] = set()
        neg1: set[str] = set()
        # Analyse this function's OWN body only — nested closures are separate
        # FunctionDefs (visited in their own right), so a nested own-field read
        # isn't mis-attributed to its enclosing builder.
        nested_ids: set[int] = set()
        for child in ast.walk(fn):
            if child is fn:
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                nested_ids.update(id(n) for n in ast.walk(child))
        for node in ast.walk(fn):
            if id(node) in nested_ids:
                continue
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "isinstance"
                and len(node.args) == 2
                and isinstance(node.args[0], ast.Name)
                and isinstance(node.args[1], ast.Name)
                and node.args[1].id == "list"
            ):
                isinstance_list.add(node.args[0].id)
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and isinstance(node.slice, ast.UnaryOp)
                and isinstance(node.slice.op, ast.USub)
                and isinstance(node.slice.operand, ast.Constant)
                and node.slice.operand.value == 1
            ):
                neg1.add(node.value.id)
        for name in sorted(isinstance_list & neg1):
            out.append((fn.name, name))
    return out


class TestUnwrapHelperMonopoly:
    """neograph-ovx1: loop-append / Each-dict unwrap is defined once in di.py.

    No consumer or scan site may hand-roll ``isinstance(X, list) and X`` ->
    ``X[-1]``; it must call ``di._unwrap_loop_value``. Likewise the dict ->
    list(values()) Each unwrap must call ``di._unwrap_each_dict``.

    AST guard (per-function same-name conjunction) + positive monopoly call-site
    assertions. Positive + negative meta-tests (AST guard — no regex-slip case).
    """

    # files the disease lived in (and their migrated helper) — the canonical
    # helper must be referenced here after migration.
    _HELPER_USERS = {
        "_input_shape.py": ("_unwrap_loop_value", "_unwrap_each_dict"),
        "_subconstruct.py": ("_unwrap_loop_value",),
        "_wiring.py": ("_unwrap_loop_value",),
    }

    def test_no_inline_list_latest_unwrap_outside_di(self):
        violations: list[str] = []
        for py in sorted(SRC_DIR.glob("*.py")):
            if py.name == "di.py":
                continue  # the monopoly home
            for func, name in _list_latest_bypass_sites(
                py.read_text(), _OWN_FIELD_READ_ALLOWLIST
            ):
                violations.append(
                    f"  {py.name}::{func} hand-rolls isinstance({name}, list)+{name}[-1]"
                    f" — delegate to di._unwrap_loop_value."
                )
        assert violations == [], (
            "\nInline loop-append latest-unwrap (neograph-ovx1 disease):\n"
            + "\n".join(violations)
            + "\n\ndi._unwrap_loop_value is the single source of truth."
        )

    def test_migrated_files_reference_canonical_helper(self):
        violations: list[str] = []
        for fname, helpers in self._HELPER_USERS.items():
            source = (SRC_DIR / fname).read_text()
            for helper in helpers:
                if _count_calls(source, helper) < 1:
                    violations.append(
                        f"  {fname} does not call {helper} — bypass re-introduced?"
                    )
        assert violations == [], (
            "\nMigrated files must delegate to the di unwrap helpers:\n"
            + "\n".join(violations)
        )

    # --- meta-tests ---

    def test_meta_bypass_scanner_catches_inline_idiom(self):
        """positive: a hand-rolled consumer unwrap is flagged."""
        bad = (
            "def consume(value, expected_type):\n"
            "    if isinstance(value, list) and value:\n"
            "        value = value[-1]\n"
            "    return value\n"
        )
        sites = _list_latest_bypass_sites(bad, frozenset())
        assert ("consume", "value") in sites

    def test_meta_bypass_scanner_passes_helper_delegation(self):
        """negative: delegating to the helper is clean."""
        good = (
            "def consume(value, expected_type):\n"
            "    return _unwrap_loop_value(value, expected_type)\n"
        )
        assert _list_latest_bypass_sites(good, frozenset()) == []

    def test_meta_bypass_scanner_allowlists_own_field_reads(self):
        """own-field reads (allowlisted function) are not flagged."""
        own = (
            "def subgraph_node(own_val):\n"
            "    if isinstance(own_val, list) and own_val:\n"
            "        return own_val[-1]\n"
        )
        assert _list_latest_bypass_sites(own, _OWN_FIELD_READ_ALLOWLIST) == []
        # but the same idiom in a non-allowlisted function IS flagged
        assert _list_latest_bypass_sites(own, frozenset()) == [
            ("subgraph_node", "own_val")
        ]


_COMPARISON_OP_ATTRS = frozenset({"lt", "le", "gt", "ge", "eq", "ne"})


def _comparison_op_table_modules(src_dir: pathlib.Path) -> list[str]:
    """Modules defining a dict literal that maps to >=4 ``operator.<cmp>``
    callables — the comparison-operator->callable table (neograph-e27b).

    Detects values like ``operator.lt`` / ``op_module.ge`` regardless of the
    import alias (matches on the trailing attribute name).
    """
    found: list[str] = []
    for py in sorted(src_dir.glob("*.py")):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            op_values = sum(
                1
                for v in node.values
                if isinstance(v, ast.Attribute) and v.attr in _COMPARISON_OP_ATTRS
            )
            if op_values >= 4:
                found.append(py.name)
                break
    return found


class TestComparisonOperatorTableMonopoly:
    """neograph-e27b: the comparison-operator->callable table lives once.

    conditions.py owns the canonical table (public OPERATORS); forward.py
    imports it rather than re-declaring _OP_MAP. AST guard + positive/negative
    meta-tests (AST — no regex-slip case).
    """

    def test_op_table_defined_in_exactly_one_module(self):
        modules = _comparison_op_table_modules(SRC_DIR)
        assert modules == ["conditions.py"], (
            f"\nComparison-operator table must be defined ONCE (conditions.py); "
            f"found in: {modules}.\nImport conditions.OPERATORS instead of "
            "re-declaring the mapping."
        )

    def test_forward_imports_canonical_operators(self):
        source = (SRC_DIR / "forward.py").read_text()
        assert "OPERATORS" in source and "from neograph.conditions import" in source, (
            "forward.py must import the canonical OPERATORS from neograph.conditions."
        )

    # --- meta-tests ---

    def test_meta_op_table_scanner_catches_duplicate(self, tmp_path):
        """positive: a module re-declaring the table is detected."""
        (tmp_path / "dup.py").write_text(
            "import operator\n"
            "_T = {'<': operator.lt, '>': operator.gt, "
            "'<=': operator.le, '>=': operator.ge}\n"
        )
        assert _comparison_op_table_modules(tmp_path) == ["dup.py"]

    def test_meta_op_table_scanner_ignores_unrelated_dict(self, tmp_path):
        """negative: a non-operator dict is not flagged."""
        (tmp_path / "clean.py").write_text(
            "_M = {'a': 1, 'b': 2, 'c': 3, 'd': 4}\n"
        )
        assert _comparison_op_table_modules(tmp_path) == []


# neograph-x3f0: three shared IR helpers, each defined in exactly one home.
_X3F0_HELPER_HOMES = {
    "iter_nodes": "construct.py",
    "collect_llm_nodes": "_llm_runtime.py",
    "missing_runtime_kwargs": "_llm_runtime.py",
    "_declared_output": "_normalize.py",
}

# The declared-output field-selection ternary (whitespace-normalized).
_DECLARED_OUTPUT_IDIOM = 'item.outputs if isinstance(item, Node) else getattr(item, "output", None)'


def _modules_defining(src_dir: pathlib.Path, func_name: str) -> list[str]:
    return [
        py.name
        for py in sorted(src_dir.glob("*.py"))
        if _has_def(py.read_text(), func_name)
    ]


class TestIrWalkHelperMonopoly:
    """neograph-x3f0: iter_nodes / collect_llm_nodes / missing_runtime_kwargs /
    _declared_output are the single source for IR-tree walk, LLM-node collection,
    and declared-output selection. AST + normalized-text guard with meta-tests.
    """

    def test_helpers_defined_in_exactly_one_home(self):
        violations: list[str] = []
        for helper, home in _X3F0_HELPER_HOMES.items():
            modules = _modules_defining(SRC_DIR, helper)
            if modules != [home]:
                violations.append(
                    f"  {helper} defined in {modules} (expected exactly [{home}])"
                )
        assert violations == [], "\nIR-walk helper home violations:\n" + "\n".join(
            violations
        )

    def test_declared_output_ternary_appears_once(self):
        total = sum(
            _normalized_idiom_count(py.read_text(), _DECLARED_OUTPUT_IDIOM)
            for py in sorted(SRC_DIR.glob("*.py"))
        )
        assert total == 1, (
            f"\nDeclared-output ternary appears {total}x across src "
            "(must be exactly 1 — only inside _declared_output). "
            "Call _declared_output(item) instead of re-inlining."
        )

    def test_no_inline_walk_for_llm(self):
        """The hand-rolled nested _walk_for_llm collectors must be gone."""
        offenders = [
            py.name
            for py in sorted(SRC_DIR.glob("*.py"))
            if "_walk_for_llm" in py.read_text()
        ]
        assert offenders == [], (
            f"\n_walk_for_llm re-inlined in {offenders}; use collect_llm_nodes(construct)."
        )

    def test_llm_node_collection_callers_delegate(self):
        for fname in ("_llm_runtime.py", "lint.py"):
            source = (SRC_DIR / fname).read_text()
            assert _count_calls(source, "collect_llm_nodes") >= 1, (
                f"{fname} must call collect_llm_nodes (LLM-node walk monopoly)."
            )

    def test_compiler_collectors_use_iter_nodes(self):
        source = (SRC_DIR / "compiler.py").read_text()
        assert _count_calls(source, "iter_nodes") >= 2, (
            "compiler.py collectors must iterate via iter_nodes(construct)."
        )

    # --- meta-tests ---

    def test_meta_home_scanner_flags_second_definition(self, tmp_path):
        (tmp_path / "construct.py").write_text("def iter_nodes(c):\n    return []\n")
        (tmp_path / "rogue.py").write_text("def iter_nodes(c):\n    return []\n")
        assert _modules_defining(tmp_path, "iter_nodes") == [
            "construct.py",
            "rogue.py",
        ]

    def test_meta_declared_output_idiom_resists_whitespace_slip(self):
        spaced = 'x = item.outputs  if  isinstance( item , Node )  else getattr(item, "output", None)'
        assert spaced.count(_DECLARED_OUTPUT_IDIOM) == 0  # naive scan misses it
        assert _normalized_idiom_count(spaced, _DECLARED_OUTPUT_IDIOM) == 1


# neograph-hbhi: the short type-display renderer has one home.
_TYPE_DISPLAY_GETATTR_IDIOM = "getattr(t, '__name__', str(t))"
# Files allowed to contain the getattr-__name__-str idiom (the helper home).
_TYPE_DISPLAY_EXEMPT = frozenset({"describe_type.py"})


class TestTypeDisplayNameMonopoly:
    """neograph-hbhi: rendering a TypeSpec to a short display string has ONE
    home -- describe_type.type_display_name. _type_name (_execute) and
    _fmt_type (_validation_types) delegate; decorators / validation messages
    call it directly. AST + normalized-text guard with meta-tests.
    """

    def test_type_display_name_defined_once_in_describe_type(self):
        homes = _modules_defining(SRC_DIR, "type_display_name")
        assert homes == ["describe_type.py"], (
            f"type_display_name must be defined once in describe_type.py; found {homes}."
        )

    def test_delegators_call_type_display_name(self):
        for fname in ("_execute.py", "_validation_types.py"):
            source = (SRC_DIR / fname).read_text()
            assert _count_calls(source, "type_display_name") >= 1, (
                f"{fname} must delegate to type_display_name (type-display monopoly)."
            )

    def test_getattr_name_str_idiom_only_in_home(self):
        offenders = [
            py.name
            for py in sorted(SRC_DIR.glob("*.py"))
            if py.name not in _TYPE_DISPLAY_EXEMPT
            and _normalized_idiom_count(py.read_text(), _TYPE_DISPLAY_GETATTR_IDIOM) > 0
        ]
        assert offenders == [], (
            f"\ngetattr(.,'__name__',str(.)) type-display idiom outside describe_type: "
            f"{offenders}.\nCall type_display_name(t) instead."
        )

    # --- meta-tests ---

    def test_meta_type_display_home_scanner(self, tmp_path):
        (tmp_path / "describe_type.py").write_text(
            "def type_display_name(t):\n    return str(t)\n"
        )
        (tmp_path / "rogue.py").write_text(
            "def type_display_name(t):\n    return str(t)\n"
        )
        assert _modules_defining(tmp_path, "type_display_name") == [
            "describe_type.py",
            "rogue.py",
        ]

    def test_meta_getattr_idiom_resists_whitespace_slip(self):
        spaced = "name = getattr( t ,  '__name__' , str( t ) )"
        assert spaced.count(_TYPE_DISPLAY_GETATTR_IDIOM) == 0  # naive scan misses it
        assert _normalized_idiom_count(spaced, _TYPE_DISPLAY_GETATTR_IDIOM) == 1


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _getattr_output_sites(tree: ast.AST) -> bool:
    """True if the tree contains ``getattr(<x>, 'output', ...)`` — the
    Construct-boundary declared-output selector (neograph-8cqd). Singular
    'output' only; 'input'/'nodes'/'outputs' accesses are different concerns."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "output"
        ):
            return True
    return False


class TestDeclaredOutputSelectorMonopoly:
    """neograph-8cqd: the Node.outputs/Construct.output declared-output selector
    has ONE home (_declared_output in _normalize.py). No other module may
    hand-roll it via ``getattr(item, 'output', None)``. AST guard + meta-tests.
    The irreducible compiler.py sum-type match uses isinstance (not
    getattr-'output') so it is not caught.
    """

    _EXEMPT_FILES = frozenset({"_normalize.py"})

    def test_no_handrolled_getattr_output_selector(self):
        offenders = [
            py.name
            for py in sorted(SRC_DIR.glob("*.py"))
            if py.name not in self._EXEMPT_FILES
            and _getattr_output_sites(ast.parse(py.read_text()))
        ]
        assert offenders == [], (
            f"\n{len(offenders)} hand-rolled getattr(_, 'output', ...) declared-output "
            f"selector(s): {offenders}.\nUse _declared_output(item) from neograph._normalize."
        )

    def test_meta_getattr_output_scanner_catches(self):
        tree = ast.parse("def f(x):\n    return getattr(x, 'output', None)\n")
        assert _getattr_output_sites(tree)

    def test_meta_getattr_output_scanner_ignores_other_attrs(self):
        tree = ast.parse(
            "def f(x):\n    return getattr(x, 'input', None), getattr(x, 'nodes', None)\n"
        )
        assert not _getattr_output_sites(tree)


class TestNoTsPortJustification:
    """neograph-8cqd (maintainer decision 2026-06-04): the TypeScript port is a
    PARALLEL codebase sharing data FORMATS (YAML/JSON-Schema/templates), not
    objects — so it must never be cited as the justification for a Python-
    codebase decision. The phrase 'TypeScript port' is banned from AGENTS.md
    (descriptive 'TypeScript-style notation' for describe_type is a different
    string and stays).
    """

    def test_no_ts_port_justification_in_agents_md(self):
        text = (REPO_ROOT / "AGENTS.md").read_text()
        assert "TypeScript port" not in text, (
            "AGENTS.md cites 'TypeScript port' as a justification. The TS port is a "
            "parallel codebase (shared formats, not objects); remove it as a rationale "
            "and keep the independent reason."
        )

    def test_meta_ts_port_phrase_detected(self):
        sample = "...consumes the same metadata. The TypeScript port (over LangGraphJS)..."
        assert "TypeScript port" in sample  # the phrase the guard bans
        assert "TypeScript port" not in "renders a TypeScript-style notation"  # descriptive stays


# ─────────────────────────────────────────────────────────────────────────────
# neograph-z3ie — fan-in wiring must not assign a raw `.outputs` into a mapping.
# A dict-form Node.outputs is registered by the validator only at per-key fields
# ({base}_{key}); assigning the raw outputs dict at a base field mis-wires the
# fan-in. Producer-field + primary-type selection must go through the monopolized
# _normalize helpers (normalize_outputs / primary_output_field). The precise
# disease signature is a SUBSCRIPT assignment whose value is `<expr>.outputs`
# (e.g. `inputs_dict[prev] = prev_node.outputs`) -- distinct from legitimate raw
# `.outputs` reads (call args, plain `x = node.outputs` for the merge/factory).
# ─────────────────────────────────────────────────────────────────────────────


def _subscript_assigns_raw_outputs(source: str) -> list[int]:
    """Line numbers of `<subscript>[...] = <expr>.outputs` assignments."""
    tree = ast.parse(source)
    hits: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "outputs"
                and any(isinstance(t, ast.Subscript) for t in node.targets)
            ):
                hits.append(node.lineno)
    return hits


class TestNoRawOutputsInFanInWiring:
    """neograph-z3ie: no src module may assign a raw `.outputs` into a mapping
    (fan-in wiring). Use normalize_outputs(...).primary + primary_output_field."""

    def test_no_subscript_raw_outputs_assignment_in_src(self):
        offenders: list[str] = []
        for path in SRC_DIR.rglob("*.py"):
            for lineno in _subscript_assigns_raw_outputs(path.read_text()):
                offenders.append(f"{path.relative_to(SRC_DIR.parent.parent).as_posix()}:{lineno}")
        assert not offenders, (
            "Raw `.outputs` assigned into a mapping (fan-in wiring). Use "
            "primary_output_field(base, outputs) for the key and "
            "normalize_outputs(outputs).primary for the value (neograph-z3ie).\n"
            + "\n".join(offenders)
        )

    def test_meta_catches_raw_outputs_subscript_assign(self):
        """Positive: `d[k] = node.outputs` is flagged."""
        assert _subscript_assigns_raw_outputs("d[k] = node.outputs\n")

    def test_meta_passes_normalized_primary(self):
        """Negative: routing through the helper (.primary) is not flagged."""
        assert not _subscript_assigns_raw_outputs(
            "d[k] = normalize_outputs(node.outputs).primary\n"
        )

    def test_meta_passes_plain_assign_and_call_arg(self):
        """Negative: plain `x = node.outputs` and call-arg `f(node.outputs)` are
        legitimate raw reads (factory output_model, merge_fn arg) -- not the
        fan-in disease."""
        assert not _subscript_assigns_raw_outputs("output_model = node.outputs\n")
        assert not _subscript_assigns_raw_outputs("f(node.outputs)\n")
