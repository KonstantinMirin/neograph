"""Structural meta-guards: the guards that guard the guards (PROC-2).

Two disciplines, enforced structurally so the documentation IS the test:

1. ``TestRegexGuardsHaveSlipMetaTests`` -- every regex in a
   ``tests/test_guards_*.py`` module is a NAMED constant and carries a
   per-constant regex-slip meta-test. Closes MED-06 (a ``_neo_``-style edge
   case slipping a guard regex with no test to catch it) and the two evasions
   the architect review flagged: an empty ``pass`` slip method, and a second
   regex hiding inside an already-covered module.

2. ``TestNoNearDuplicateHelperNames`` -- every helper name across
   ``src/neograph/`` (top-level ``def`` names AND ``from ... import ... as``
   aliases) is canonical. Near-duplicate names (Levenshtein <= 2, both >= 4
   chars) for distinct functions are eliminated or allowlisted-with-reason.
   Closes CON-01 (``_isinstance_safe`` / ``_is_instance_safe``).

This module uses PURE AST (no ``re`` at all) so it is exempt from rule 1 by
construction: it cannot slip its own discipline.
"""

from __future__ import annotations

import ast
import itertools
import pathlib

TESTS_DIR = pathlib.Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src" / "neograph"

# ``re`` functions whose FIRST argument is the pattern. A literal first arg to
# any of these is an inline (un-named) regex -- a naming violation.
_RE_PATTERN_FIRST = frozenset({"search", "match", "fullmatch", "findall", "finditer", "sub", "subn", "split"})


# ════════════════════════════════════════════════════════════════════════════
# Discipline 1 -- every guard regex is a named constant with a slip meta-test
# ════════════════════════════════════════════════════════════════════════════
class TestRegexGuardsHaveSlipMetaTests:
    """Every regex in a guard module is a named constant carrying a slip test.

    A guard regex that lacks an edge-case ("would-be-missed") test is how
    MED-06 happened: the StateKeys guard's ``^neo_`` anchor silently missed
    ``_neo_isolated_input`` and no test would have caught the gap. This guard
    makes that structurally impossible going forward.

    The contract, per regex *binding* (a module/class-level name bound to a
    compiled regex or a collection of them):

    * naming -- regexes must be NAMED constants; an inline ``re.search(r"...")``
      or an unbound ``re.compile(...)`` is rejected (there is no name to key a
      slip test to).
    * slip test -- each binding ``FOO_RE`` requires a test whose name contains
      both ``slip`` and the normalized binding (``foo_re``) and which contains
      at least one ``assert`` (an empty ``pass`` does not satisfy the contract).

    Mutation-verified below with positive + negative meta-tests.
    """

    # ── detector (operates on source text so meta-tests can feed synthetics) ──

    @staticmethod
    def _normalize(binding: str) -> str:
        """Normalize a binding name for slip-method keying."""
        return binding.strip("_").lower()

    @staticmethod
    def _re_call_func(node: ast.Call, re_aliases: set[str], from_re: dict[str, str]) -> str | None:
        """If ``node`` is a call to a ``re`` function, return the function name
        (``compile``/``search``/...) -- regardless of HOW ``re`` was imported
        (``re.x``, ``import re as r`` -> ``r.x``, or ``from re import x`` -> bare
        ``x``). Returns None otherwise. Receiver-agnostic, mirroring the
        StateKeys Layer-B rule: we match the concept of a regex call, not the
        spelling of the ``re`` module."""
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id in re_aliases:
                return func.attr
        elif isinstance(func, ast.Name) and func.id in from_re:
            return from_re[func.id]
        return None

    @classmethod
    def _analyze(cls, source: str) -> tuple[list[str], list[str]]:
        """Return ``(naming_violations, slip_violations)`` for one module's
        source. Pure AST -- no regex."""
        tree = ast.parse(source)

        # Pass 0: resolve how `re` entered this module's namespace. The detector
        # is receiver-agnostic: `import re [as r]` and `from re import f [as g]`
        # are both followed, so an aliased/from-imported regex cannot bypass.
        re_aliases: set[str] = set()
        from_re: dict[str, str] = {}  # local name -> re function name
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "re":
                        re_aliases.add(a.asname or "re")
            elif isinstance(node, ast.ImportFrom) and node.module == "re":
                for a in node.names:
                    from_re[a.asname or a.name] = a.name

        # Pass 1: collect named regex bindings and the id() of every compile
        # call bound to a name (directly or inside a Name-bound list/tuple/set).
        bindings: dict[str, int] = {}  # normalized binding name -> lineno
        bound_compile_ids: set[int] = set()

        def _compiles_in(node: ast.AST):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and cls._re_call_func(sub, re_aliases, from_re) == "compile":
                    yield sub

        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            value: ast.expr | None = None
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            if value is None:
                continue
            compiles = list(_compiles_in(value))
            if not compiles:
                continue
            for tgt in targets:
                if isinstance(tgt, ast.Name):
                    bindings[cls._normalize(tgt.id)] = node.lineno
            for c in compiles:
                bound_compile_ids.add(id(c))

        # Pass 2: naming violations -- unbound compiles + inline-literal calls.
        # A literal first POSITIONAL arg OR a literal `pattern=` kwarg counts as
        # inline (kwarg form cannot sneak past).
        naming: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = cls._re_call_func(node, re_aliases, from_re)
            if fn is None:
                continue
            if fn == "compile":
                if id(node) not in bound_compile_ids:
                    naming.append(f"line {node.lineno}: re.compile(...) not bound to a named constant")
            elif fn in _RE_PATTERN_FIRST:
                pattern_arg = node.args[0] if node.args else None
                for kw in node.keywords:
                    if kw.arg == "pattern":
                        pattern_arg = kw.value
                if isinstance(pattern_arg, (ast.Constant, ast.JoinedStr)):
                    naming.append(
                        f"line {node.lineno}: inline re.{fn}(<literal>) -- assign the pattern to a named constant"
                    )

        # Pass 3: slip-test coverage per binding.
        test_fns: dict[str, bool] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                has_assert = any(isinstance(s, ast.Assert) for s in ast.walk(node))
                test_fns[node.name] = has_assert

        # Assign each slip test to its LONGEST token-matching binding only, so a
        # binding that is a prefix of a longer one (e.g. counter_get vs
        # counter_get_default) cannot be falsely covered by the longer binding's
        # test. Closes the substring-keying hole.
        def _covers(binding: str, name: str) -> bool:
            low = f"_{name.lower()}_"
            return f"_{binding}_" in low or low.endswith(f"_{binding}_")

        covered: dict[str, list[str]] = {b: [] for b in bindings}
        for name in test_fns:
            if "slip" not in name.lower():
                continue
            matches = [b for b in bindings if _covers(b, name)]
            if not matches:
                continue
            best = max(matches, key=len)  # longest binding wins this test
            covered[best].append(name)

        slip: list[str] = []
        for binding, lineno in sorted(bindings.items()):
            named_slips = covered[binding]
            if not named_slips:
                slip.append(
                    f"regex constant matching '{binding}' (line {lineno}) has no "
                    f"slip meta-test (expected a test named like 'test_slip_{binding}')"
                )
            elif not any(test_fns[n] for n in named_slips):
                slip.append(f"slip meta-test(s) for '{binding}' contain no assert (empty-pass evasion): {named_slips}")
        return naming, slip

    @classmethod
    def _guard_modules(cls) -> list[pathlib.Path]:
        """Every ``tests/test_guards_*.py`` module (this meta module included --
        it uses no regex, so it is naturally compliant)."""
        return sorted(TESTS_DIR.glob("test_guards_*.py"))

    # ── the live guard ───────────────────────────────────────────────────────

    def test_every_guard_regex_is_named_with_a_slip_test(self) -> None:
        problems: list[str] = []
        for path in self._guard_modules():
            naming, slip = self._analyze(path.read_text())
            rel = path.name
            problems += [f"{rel}: {m}" for m in naming]
            problems += [f"{rel}: {m}" for m in slip]
        assert not problems, (
            "Regex-guard discipline violated (PROC-2). Every regex in a guard "
            "module must be a NAMED constant with a per-constant slip meta-test "
            "(name contains 'slip' + the constant, body has >=1 assert):\n" + "\n".join(problems)
        )

    # ── mutation meta-tests: prove the analyzer actually analyzes ─────────────

    _GOOD = (
        "import re\n"
        "FOO_RE = re.compile(r'^x')\n"
        "class T:\n"
        "    def test_slip_foo_re(self):\n"
        "        assert FOO_RE.match('x')\n"
    )

    def test_meta_accepts_named_regex_with_slip(self) -> None:
        naming, slip = self._analyze(self._GOOD)
        assert naming == [] and slip == []

    def test_meta_detects_inline_unnamed_regex(self) -> None:
        naming, _ = self._analyze("import re\nx = re.search(r'^y', line)\n")
        assert naming and "inline" in naming[0]

    def test_meta_accepts_compile_bound_inside_collection(self) -> None:
        # A re.compile inside a Name-bound list is bound -> no naming violation.
        src = (
            "import re\n"
            "FOO_RES = [re.compile(r'^z')]\n"
            "class T:\n"
            "    def test_slip_foo_res(self):\n"
            "        assert FOO_RES\n"
        )
        naming, slip = self._analyze(src)
        assert naming == [] and slip == []

    def test_meta_detects_unbound_compile(self) -> None:
        naming, _ = self._analyze("import re\nprint(re.compile(r'^z'))\n")
        assert naming and "not bound" in naming[0]

    def test_meta_detects_missing_slip(self) -> None:
        _, slip = self._analyze("import re\nFOO_RE = re.compile(r'^x')\n")
        assert slip and "slip meta-test" in slip[0]

    def test_meta_detects_empty_pass_slip(self) -> None:
        src = "import re\nFOO_RE = re.compile(r'^x')\nclass T:\n    def test_slip_foo_re(self):\n        pass\n"
        _, slip = self._analyze(src)
        assert slip and "no assert" in slip[0]

    # ── receiver-agnostic re-detection: aliased / from-imported / kwarg forms ──

    def test_meta_detects_aliased_re_import(self) -> None:
        """`import re as r; r.search(r'..')` must NOT bypass the detector."""
        naming, _ = self._analyze("import re as r\nx = r.search(r'^y', line)\n")
        assert naming and "inline" in naming[0]

    def test_meta_detects_from_import_re(self) -> None:
        """`from re import search; search(r'..')` must NOT bypass the detector."""
        naming, _ = self._analyze("from re import search\nx = search(r'^y', line)\n")
        assert naming and "inline" in naming[0]

    def test_meta_detects_unbound_compile_via_alias(self) -> None:
        naming, _ = self._analyze("import re as r\nprint(r.compile(r'^z'))\n")
        assert naming and "not bound" in naming[0]

    def test_meta_detects_kwarg_pattern_form(self) -> None:
        """`re.search(pattern=r'..', string=line)` must NOT bypass via kwargs."""
        naming, _ = self._analyze("import re\nx = re.search(pattern=r'^y', string=line)\n")
        assert naming and "inline" in naming[0]

    def test_meta_prefix_binding_not_falsely_covered(self) -> None:
        """A binding that is a prefix of a longer one (FOO vs FOO_RE) is NOT
        falsely covered by the longer binding's slip test -- longest match wins.
        Closes the substring-keying hole."""
        src = (
            "import re\n"
            "FOO = re.compile(r'^a')\n"
            "FOO_RE = re.compile(r'^b')\n"
            "class T:\n"
            "    def test_slip_foo_re(self):\n"
            "        assert FOO_RE and FOO\n"
        )
        _, slip = self._analyze(src)
        # FOO_RE is covered by test_slip_foo_re; FOO has NO dedicated slip test.
        assert any("'foo'" in s for s in slip)
        assert not any("'foo_re'" in s for s in slip)


# ════════════════════════════════════════════════════════════════════════════
# Discipline 2 -- helper names across src/neograph/ are canonical
# ════════════════════════════════════════════════════════════════════════════
class TestNoNearDuplicateHelperNames:
    """No two helpers in ``src/neograph/`` carry near-duplicate names.

    CON-01: ``_isinstance_safe`` (di.py) was aliased to ``_is_instance_safe``
    (factory.py / _input_shape.py) -- two grep-hostile names for one helper,
    distance 1. This guard pools top-level ``def`` names AND ``import ... as``
    aliases, then flags any pair with Levenshtein <= 2 (both names >= 4 chars)
    that is not on the reasoned allowlist.

    Pooling import aliases (not just defs) is what lets the guard catch CON-01
    and prevent its regression: a re-introduced ``... as _is_instance_safe``
    alias is distance 1 from the canonical ``_isinstance_safe`` def.
    """

    MIN_LEN = 4
    MAX_DISTANCE = 2

    # Allowlist: frozenset({name_a, name_b}) -> reason (includes lev distance).
    ALLOWLIST: dict[frozenset[str], str] = {
        # neograph-hjwv: render_inputs is the ticket-mandated public prompt
        # primitive (the exported dict view of build_rendered_input(...).
        # for_template_ref — ALL inputs, template-ref). It is semantically
        # distinct from render_input/_render_input, which render a SINGLE value's
        # inline view. Different arity, different consumer, ticket-finalized name.
        frozenset({"render_input", "render_inputs"}): (
            "lev=1: render_inputs (prompt.py) exports the template-ref dict of all "
            "inputs; render_input (renderers.py) renders one value's inline view. "
            "Ticket-mandated public name (hjwv)."
        ),
        frozenset({"_render_input", "render_inputs"}): (
            "lev=2: render_inputs (prompt.py, public) vs _render_input (_dispatch.py, "
            "private mode-dispatch helper). Distinct layer + arity; hjwv public name."
        ),
        frozenset({"_aexecute_node", "_execute_node"}): (
            "lev=1: async twin of the sync node executor (_execute.py). The "
            "a-prefix (aexecute/ainvoke/arun) is the deliberate sync/async twin "
            "naming convention for the async foundation, not a duplicate helper."
        ),
        frozenset({"_make_araw_wrapper", "_make_raw_wrapper"}): (
            "lev=1: async twin of the sync raw-node wrapper (factory.py). Same "
            "a-prefix sync/async twin convention as _aexecute_node/_execute_node."
        ),
        # neograph-w74k.2.3 (Phase 1c): a-prefix async twins across the LLM/tool
        # vertical — same deliberate sync/async twin convention.
        frozenset({"_acall_structured", "_call_structured"}): (
            "lev=1: async twin of the structured-output dispatch (_llm_dispatch.py)."
        ),
        frozenset({"ainvoke_structured", "invoke_structured"}): (
            "lev=1: async twin of the think-mode orchestrator (_llm.py)."
        ),
        # neograph-m6d3.3: the invoke_with_tools/ainvoke_with_tools monolith was
        # deleted (agent/act compile to the inline cycle). The surviving _tool_loop
        # twin is the shared final-parse cluster the cycle's parse node reuses.
        frozenset({"_aparse_final_turn", "_parse_final_turn"}): (
            "lev=1: async twin of the ReAct final-parse + fallback cluster."
        ),
        frozenset({"_ainvoke_json_with_retry", "_invoke_json_with_retry"}): (
            "lev=1: async twin of the json-retry loop (_llm_retry.py)."
        ),
        frozenset({"arecover_dsml", "recover_dsml"}): ("lev=1: async twin of DSML recovery (_llm_retry.py)."),
        # neograph-3q6j: the di_inputs injector's async twin awaits FROM_RESOURCE
        # bindings (DIBinding.aresolve) before _compile_prompt — same a-prefix
        # sync/async twin convention; the sync form fails loud on FROM_RESOURCE.
        frozenset({"_ainject_di_inputs", "_inject_di_inputs"}): (
            "lev=1: async twin of the di_inputs injector (_dispatch.py)."
        ),
        # neograph-w74k.2.4 (Phase 1d): a-prefix async twins of the runner's
        # checkpoint helpers (arun path).
        frozenset({"_ahas_existing_checkpoint", "_has_existing_checkpoint"}): (
            "lev=1: async twin of the checkpoint-exists probe (runner.py)."
        ),
        frozenset({"_averify_checkpoint_schema", "_verify_checkpoint_schema"}): (
            "lev=1: async twin of checkpoint-schema verification (runner.py)."
        ),
        frozenset({"_aauto_resume_from_divergence", "_auto_resume_from_divergence"}): (
            "lev=1: async twin of the auto-resume rewind (runner.py)."
        ),
        # neograph-q8ec (Phase 2, streaming): a-prefix async twins of the
        # runner's prepare brain and stream verb — same sync/async twin
        # convention (astream/_aprepare run the vertical on the event loop).
        frozenset({"_aprepare", "_prepare"}): ("lev=1: async twin of the pre-engine prepare brain (runner.py)."),
        frozenset({"astream", "stream"}): ("lev=1: async twin of the sync stream verb (runner.py)."),
        # neograph-p3c7: a-prefix async twins of the Oracle merge barrier — same
        # deliberate sync/async twin convention (merge runs on the loop under arun).
        # neograph-w74k.3.1: a-prefix async twins of the tool-loop factory
        # instantiation — the async path awaits coroutine/awaitable tool factories
        # (per-run MCP identity), the sync twin fails loud. Same twin convention.
        frozenset({"_ainstantiate_tools", "_instantiate_tools"}): (
            "lev=1: async twin of the tool-factory instantiation (_tool_loop.py)."
        ),
        frozenset({"_aprepare_tool_loop", "_prepare_tool_loop"}): (
            "lev=1: async twin of the tool-loop preamble (_tool_loop.py)."
        ),
        frozenset({"_abuild_turn_prep", "_build_turn_prep"}): (
            "lev=1: async twin of the per-superstep turn prep (_agent_cycle.py)."
        ),
        frozenset({"_arun_merge_prompt", "_run_merge_prompt"}): (
            "lev=1: async twin of the LLM-judge merge step (_oracle.py)."
        ),
        frozenset({"_amerge_variants", "_merge_variants"}): (
            "lev=1: async twin of the canonical Oracle merge (_oracle.py)."
        ),
        frozenset({"_amerge_one_group", "_merge_one_group"}): (
            "lev=1: async twin of the Each×Oracle per-group merge (_wiring.py)."
        ),
        frozenset({"_get_param_res", "_set_param_res"}): (
            "lev=1: getter/setter antonym pair (_sidecar.py); the get/set prefix "
            "is intentional API symmetry, not a near-duplicate of one helper."
        ),
        frozenset({"_render_dict_value", "_render_list_value"}): (
            "lev=2: parallel type-dispatch renderers (describe_type.py); the "
            "dict-vs-list distinction is the whole point of the two functions."
        ),
        frozenset({"_render_input", "render_input"}): (
            "lev=1: private mode-dispatch wrapper (_dispatch.py) vs the public "
            "renderer entry (renderers.py); leading-underscore is the "
            "private/public convention, not a duplicate name."
        ),
        # neograph-m6d3.8 / neograph-43do: a-prefix async twin of the per-run
        # cache accessor (_run_cache.py) — the async form awaits the build/fetch
        # callback on a miss. Same deliberate sync/async twin convention.
        frozenset({"aget_or_build", "get_or_build"}): (
            "lev=1: async twin of the per-run handle/resource cache accessor "
            "(_run_cache.py); build callback awaited on a miss."
        ),
        # neograph-hhlr: a-prefix async twin of the per-key single-flight latch
        # minter (_run_cache.py). The async form mints a LOOP-AFFINE asyncio.Lock
        # keyed by running-loop id; the sync form mints a threading.Lock. Same
        # deliberate sync/async twin convention.
        frozenset({"_alatch", "_latch"}): (
            "lev=1: async twin of the sync single-flight latch minter "
            "(_run_cache.py); loop-affine asyncio.Lock vs threading.Lock."
        ),
        # neograph-v569: public compile_prompt is the PROMOTED, ticket-mandated
        # public wrapper over the internal _compile_prompt seam (both in
        # _llm_render.py) — same compiler brain, byte-identical output. The
        # leading-underscore is the private/public convention; the public name IS
        # the API contract (parallel to the render_inputs/render_input entry).
        frozenset({"_compile_prompt", "compile_prompt"}): (
            "lev=1: public compile_prompt (the standalone eval-parity entry) wraps "
            "the private _compile_prompt seam (_llm_render.py); private/public "
            "underscore convention, ticket-mandated public name (v569)."
        ),
    }

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        if a == b:
            return 0
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                cur = dp[j]
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
                prev = cur
        return dp[n]

    @classmethod
    def _collect_names(cls, src_dir: pathlib.Path) -> dict[str, list[str]]:
        """Pool top-level ``def`` names + ``ImportFrom`` aliases -> locations."""
        names: dict[str, list[str]] = {}
        for path in sorted(src_dir.glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in tree.body:  # top-level only
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.setdefault(node.name, []).append(f"{path.name}:{node.lineno}(def)")
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.asname:  # only aliases introduce a new local name
                            names.setdefault(alias.asname, []).append(f"{path.name}:{node.lineno}(alias)")
        return names

    @classmethod
    def _find_near_duplicates(cls, names: dict[str, list[str]]) -> list[str]:
        flagged: list[str] = []
        for a, b in itertools.combinations(sorted(names), 2):
            if len(a) < cls.MIN_LEN or len(b) < cls.MIN_LEN:
                continue
            d = cls._levenshtein(a, b)
            if d > cls.MAX_DISTANCE:
                continue
            if frozenset({a, b}) in cls.ALLOWLIST:
                continue
            flagged.append(f"lev={d}: {a} {names[a]} <-> {b} {names[b]}")
        return flagged

    # ── the live guard ───────────────────────────────────────────────────────

    def test_no_near_duplicate_helper_names(self) -> None:
        flagged = self._find_near_duplicates(self._collect_names(SRC_DIR))
        assert not flagged, (
            "Near-duplicate helper names found in src/neograph/ (PROC-2). Give "
            "the helper ONE canonical name, or add the pair to ALLOWLIST with a "
            "written reason:\n" + "\n".join(flagged)
        )

    def test_allowlist_entries_are_actually_near_duplicates(self) -> None:
        """An allowlist entry that no longer trips the rule is dead -- remove it."""
        dead: list[str] = []
        for pair in self.ALLOWLIST:
            a, b = sorted(pair)
            if len(a) < self.MIN_LEN or len(b) < self.MIN_LEN:
                dead.append(f"{pair}: a name is shorter than MIN_LEN")
            elif self._levenshtein(a, b) > self.MAX_DISTANCE:
                dead.append(f"{pair}: lev > {self.MAX_DISTANCE}")
        assert not dead, "Stale allowlist entries (no longer near-duplicates):\n" + "\n".join(dead)

    # ── mutation meta-tests ───────────────────────────────────────────────────

    def test_meta_positive_flags_near_duplicate(self) -> None:
        flagged = self._find_near_duplicates({"_foobar": ["x.py:1(def)"], "_foobaz": ["y.py:1(def)"]})
        assert len(flagged) == 1 and "_foobar" in flagged[0]

    def test_meta_negative_ignores_distant_and_short(self) -> None:
        # lev > 2:
        assert self._find_near_duplicates({"compile": ["x.py:1(def)"], "validate": ["y.py:1(def)"]}) == []
        # both < MIN_LEN even though lev=1:
        assert self._find_near_duplicates({"get": ["x.py:1(def)"], "set": ["y.py:1(def)"]}) == []

    def test_meta_negative_respects_allowlist(self) -> None:
        flagged = self._find_near_duplicates({"_get_param_res": ["a.py:1(def)"], "_set_param_res": ["a.py:2(def)"]})
        assert flagged == []


# ════════════════════════════════════════════════════════════════════════════
# Slip meta-test for the DX-import guard (LR-01)
# ════════════════════════════════════════════════════════════════════════════
class TestDXImportGuardCoversDecorators:
    """Slip meta-test: the DX-import guard must fire on a ``decorators.py``
    import, not only ``forward.py``.

    LR-01 was an unguarded HOLE: the "lower layers must not import the DX layer"
    guard (``TestLowerLayersDoNotImportForwardDX`` in test_guards_assembly.py)
    was written for forward.py and never extended to its twin decorators.py, so
    an IR module importing decorators left the full guard suite green — the
    reviewer mutation-verified it. This meta-test pins the extension: if a
    future edit drops ``decorators`` from the guard's ``DX_MODULES`` set (or the
    parser stops seeing the import), the guard silently reopens the hole and
    THIS test catches it.
    """

    def _guard(self):
        from tests.test_guards_assembly import TestLowerLayersDoNotImportForwardDX

        return TestLowerLayersDoNotImportForwardDX

    def test_decorators_is_in_the_guards_banned_dx_set(self) -> None:
        guard = self._guard()
        assert "decorators" in guard.DX_MODULES, (
            "decorators.py must be in the DX-import guard's banned set (LR-01). "
            "Removing it reopens the mutation-verified backdoor."
        )
        assert "forward" in guard.DX_MODULES

    def test_guard_logic_flags_a_synthetic_decorators_import(self, tmp_path) -> None:
        """Replay the guard's offender loop against a synthetic IR module that
        imports decorators — it must be flagged as an offender."""
        from tests.test_guards_assembly import _parse_neograph_imports

        guard = self._guard()
        fake = tmp_path / "_fake_ir_module.py"
        fake.write_text("from neograph.decorators import _classify_di_params\n")

        imported = _parse_neograph_imports(fake)
        offenders = sorted(guard.DX_MODULES & imported)
        assert offenders == ["decorators"], (
            f"The DX-import guard must flag an IR module that imports neograph.decorators; got offenders={offenders}"
        )
