"""Structural guard (G-PRP): a Portal mesh's routing plumbing is derived ONCE.

neograph-dgbqv.4 (P9 of the Agent Spec dispatch-vocabulary epic). Written
FAILING, before ``_portal_route.py`` exists, per this repo's guard-first
discipline. Its direct siblings — whose shape this file copies (hand-written
literals, pure-``ast`` scanners, mutation meta-tests, anti-vacuity tripwire) —
are ``tests/test_guards_combo_decomposition_consumers.py`` and
``tests/test_guards_portal_member_class_consumers.py`` (P7, the classifier this
one builds on).

The disease
-----------
Every fact a Portal mesh member needs in order to reach ``Command(goto=...)``
— the shared channel key, the hop counter, the entry/exit names, the hop
budget, the exhaust policy, the entry-label map, the resolved destination
tuple, the proposed-target field — is hand re-derived at each call site
instead of built ONCE into a frozen bundle. Concretely, at HEAD 51b4a44:

* ``StateKeys.handoff_payload`` / ``handoff_hops`` / ``portal_proposed_target``
  / ``handoff_tool_target`` are CALLED 12 times outside the modules that own
  those keys — 10 in ``factory.py`` (:168, :169, :173, :468, :469, :556, :557,
  :642, :643, :644) and 2 in ``_wiring.py`` (:405, :406).
* ``_portal_route_to_command`` carries 13 kw-only params beyond
  ``(update, state)`` and ``_tool_handoff_to_command`` carries 8 — the same
  bundle, threaded by hand through 10 call sites.
* the member-kind dispatch inside ``_add_portal_mesh`` is an ``if/elif`` chain,
  so a sixth ``PortalMemberClass`` can be added without anyone noticing that
  the mesh does not wire it.

The cure
--------
``src/neograph/_portal_route.py``: a frozen ``MeshContext`` built once per mesh
in ``_add_portal_mesh``, a frozen ``PortalRouteSpec`` per member, and a
``PortalMemberClass``-keyed adapter table in ``_wiring.py`` replacing the
chain. The two decision functions KEEP their names and their home in
``factory.py`` and take the spec as one positional — guard G1's ``Command``
monopoly (``tests/test_guards_assembly.py``) must stay byte-identical, so the
spec module is PURE DATA with no ``to_command``.

Why a structural guard is the right (and only) instrument here
--------------------------------------------------------------
P9 is a pure plumbing collapse: its regression evidence is the existing Portal
behavioral suite passing UNCHANGED. That suite cannot see the two things most
likely to go wrong.

1. ``destinations=`` is graph-registration metadata. The architect review
   (neograph-jn555.20) caught a plan that would have routed ``_wiring.py:763``'s
   peers-ONLY ``tools_destinations`` through a peers+exit method, silently
   WIDENING a tool-triggered member's declared goto target set. No behavioral
   test asserts on ``destinations=``, so the suite would have stayed green.
   ``test_mesh_context_exposes_both_peer_shapes`` pins the two-method split
   that makes the mistake unrepresentable.
2. Re-derivation is invisible at runtime by definition — two copies of the same
   derivation agree until the day one is edited.

Written in pure ``ast`` with no ``re``, so it is exempt by construction from
``tests/test_guards_meta.py``'s named-regex/slip-test discipline — the same
move both sibling guards make.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# --- independent literals (hand-written; never derived from the scan) --------

#: The deliverable: a neutral leaf module both ``_wiring.py`` and ``factory.py``
#: can reach, holding the frozen routing bundles. Placement copies
#: ``_agent_cycle_names.py``; shape copies ``_llm_runtime.LlmRuntime``
#: (``@dataclass(frozen=True)`` + a ``build()`` classmethod). NO ``Command``
#: import and no back-import of ``factory`` — that would widen G1 or force a
#: function-local import, both refused by AGENTS.md's file-split ladder.
SPEC_MODULE = "_portal_route.py"

#: The mesh-wide bundle and the per-member bundle. ``MeshDeps`` is deliberately
#: NOT required: the plan carries an explicit fallback in which the compile-time
#: deps stay explicit kwargs on the adapters and no deps bundle is built.
SPEC_SYMBOLS: frozenset[str] = frozenset({"MeshContext", "PortalRouteSpec"})

#: The TWO destination shapes, which are DIFFERENT registered-destination sets:
#: ``resolved_peers`` is peers only (``_wiring.py:745``'s ``peer_targets``,
#: which feeds the AGENT_CYCLE_TOOL node's ``tools_destinations``), and
#: ``destinations_for`` is ``resolved_peers + (exit_name,)`` (``_wiring.py``
#: :358, :415, :424, :746). Collapsing them into one method widens a
#: tool-triggered member's goto target set with a green behavioral suite — see
#: the module docstring.
MESH_CONTEXT_METHODS: frozenset[str] = frozenset({"resolved_peers", "destinations_for"})

#: The four ``StateKeys`` builders that mint a mesh's ``neo_``-prefixed state
#: keys. They remain the ONLY way those keys are built (guard G2); this guard
#: constrains WHERE they may be CALLED, so the result is derived once and
#: cached on the context/spec rather than re-minted per call site.
HANDOFF_KEY_BUILDERS: frozenset[str] = frozenset(
    {"handoff_payload", "handoff_hops", "portal_proposed_target", "handoff_tool_target"}
)

#: Files that may CALL those builders. ``at most``, not ``exactly``: each entry
#: has a stated, non-mesh-wiring reason to mint a key itself.
#:
#: * ``_portal_route.py`` — the cure. ``MeshContext.build`` / the
#:   ``PortalRouteSpec`` classmethods call them ONCE and cache the results.
#: * ``state.py`` — the state-MODEL side (:245, :246, :255). A different layer
#:   with a different lifetime: no mesh is being wired there. Deliberately out
#:   of P9's "read-side only" scope.
#: * ``_agent_cycle.py`` (:460) — the ReAct cycle's own tool-target write.
#: * ``_ir_normalize.py`` (:288) — the sole writer of ``handoff_channel``
#:   (guard G3's single-writer IR-field invariant, untouched by P9).
#:
#: ``factory.py`` and ``_wiring.py`` are the two files this guard removes: they
#: hold all 12 offending call sites listed in the module docstring.
DECLARED_KEY_CALLERS: frozenset[str] = frozenset({SPEC_MODULE, "state.py", "_agent_cycle.py", "_ir_normalize.py"})

#: ``_subconstruct.py`` mentions two of these builders in a DOCSTRING and calls
#: neither, as do ``factory.py``'s own two docstrings. Pinned as a literal so
#: the "why AST, never grep" reason survives: a text scanner would report three
#: permanent false positives and force a bogus entry into the declared set.
DOCSTRING_ONLY_MENTIONS: frozenset[str] = frozenset({"_subconstruct.py"})

#: Where the member-kind dispatch table lives. ``_wiring.py``, not the spec
#: module: the adapters call ``graph.add_node`` and the ``factory`` builders,
#: and the spec module must not import ``factory``.
ADAPTER_TABLE_MODULE = "_wiring"
ADAPTER_TABLE_NAME = "_PORTAL_MEMBER_ADAPTERS"

#: The decision functions stay in ``factory.py`` — G1's ``Command``-construction
#: monopoly is ``{"factory.py", "runner.py"}`` and must not grow to buy this
#: split (AGENTS.md refusal #1).
DECISION_MODULE = "factory.py"
DECISION_FUNCTIONS: frozenset[str] = frozenset({"_portal_route_to_command", "_tool_handoff_to_command"})

#: The collapsed signature: ``(update, state, spec)``. ``update``/``state`` are
#: the per-invocation values (never spec-resident); ``spec`` is the bundle.
SPEC_SIGNATURE_PARAMS: frozenset[str] = frozenset({"update", "state", "spec"})

#: Headroom for a ``ctx`` positional and at most two genuinely per-invocation
#: params. Today the two functions sit at 13 and 8 — the cap is what fails.
MAX_PARAMS_BEYOND_SPEC = 3


# --- scanners (pure ast) ----------------------------------------------------


def _enclosing_functions(tree: ast.Module) -> dict[int, str]:
    """Line number -> name of the INNERMOST enclosing function (or ``<module>``)."""
    spans: dict[int, list[tuple[int, str]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno or node.lineno
            for line in range(node.lineno, end + 1):
                spans.setdefault(line, []).append((end - node.lineno, node.name))
    return {line: min(cands)[1] for line, cands in spans.items()}


def _handoff_key_call_sites(source: str) -> list[tuple[str, int, str]]:
    """``(function, lineno, "base.builder")`` for every CALL of a handoff key
    builder.

    An ``ast.Call`` on an ``ast.Attribute``, never text: ``factory.py`` and
    ``_subconstruct.py`` name these builders in DOCSTRINGS, and a grep-shaped
    scanner would report those as permanent false positives.

    Matched on the ATTRIBUTE NAME rather than on ``base is StateKeys``, which
    is a deliberate strengthening of the ticket's spec: the four names are
    unique in the package, so requiring the base to be spelled ``StateKeys``
    only creates an evasion (``from ... import StateKeys as SK``) without
    buying any precision. The base IS reported, so an unrelated
    ``foo.handoff_hops()`` would be legible in the failure message rather than
    silently conflated.
    """
    tree = ast.parse(source)
    where = _enclosing_functions(tree)
    hits = {
        (
            where.get(node.lineno, "<module>"),
            node.lineno,
            f"{ast.unparse(node.func.value)}.{node.func.attr}",
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in HANDOFF_KEY_BUILDERS
    }
    return sorted(hits)


def _function_params(source: str, name: str) -> list[str] | None:
    """Every parameter name of module-level function ``name``, or ``None``."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            args = node.args
            return [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    return None


def _defined_methods(source: str, class_name: str) -> set[str]:
    """Method names defined directly on ``class_name``."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return set()


def _package_files() -> list[pathlib.Path]:
    """Every .py under src/neograph (recursive -- a subpackage must not escape)."""
    return sorted(p for p in SRC_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(SRC_DIR).as_posix()


# --- the guard --------------------------------------------------------------


class TestPortalRoutePlumbingMonopoly:
    """neograph-dgbqv.4: every Portal mesh member, whatever its
    ``PortalMemberClass``, reaches ``Command(goto=...)`` through ONE derivation
    of the mesh's routing parameters and ONE pair of decision functions."""

    # --- (a) the mesh's state keys are minted once --------------------------

    def test_handoff_state_keys_are_built_only_in_the_declared_owners(self):
        offenders: list[str] = []
        for path in _package_files():
            name = _rel(path)
            if name in DECLARED_KEY_CALLERS:
                continue
            offenders += [
                f"{name}:{lineno}\t{function}\t{call}"
                for function, lineno, call in _handoff_key_call_sites(path.read_text())
            ]
        assert not offenders, (
            f"A Portal mesh key builder ({sorted(HANDOFF_KEY_BUILDERS)}) is CALLED outside "
            f"{sorted(DECLARED_KEY_CALLERS)}.\n"
            "Each of these re-mints a key that MeshContext/PortalRouteSpec already caches "
            "(neograph-dgbqv.4). Read it off the spec instead. The builders themselves stay "
            "the only way to spell the key (guard G2) -- this rule is about WHERE the result "
            "is derived, not how.\n"
            f"Expected to be red today at exactly 12 sites: 10 in factory.py, 2 in _wiring.py.\n"
            f"Found {len(offenders)}:\n" + "\n".join(offenders)
        )

    # --- the spec module ----------------------------------------------------

    def test_route_spec_module_exists_and_defines_the_frozen_bundles(self):
        path = SRC_DIR / SPEC_MODULE
        assert path.is_file(), (
            f"{SPEC_MODULE} does not exist. It is the deliverable of neograph-dgbqv.4 and the "
            "single authority this whole guard is built around."
        )
        defined = {n.name for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ClassDef)}
        assert SPEC_SYMBOLS <= defined, f"{SPEC_MODULE} must define {sorted(SPEC_SYMBOLS)}; found {sorted(defined)}"

    def test_mesh_context_exposes_both_peer_shapes(self):
        """The two-method split, which no behavioral test can defend.

        ``resolved_peers`` (peers only) and ``destinations_for``
        (peers + exit) are DIFFERENT registered-destination sets. One method
        serving both sites appends ``exit_name`` to a tool-triggered member's
        ``tools_destinations``, silently widening its declared goto target set
        — and ``destinations=`` is validation/rendering metadata that no Portal
        behavioral test asserts on, so the suite stays green.
        """
        path = SRC_DIR / SPEC_MODULE
        if not path.is_file():
            return  # covered by test_route_spec_module_exists_and_defines_the_frozen_bundles
        methods = _defined_methods(path.read_text(), "MeshContext")
        assert MESH_CONTEXT_METHODS <= methods, (
            f"MeshContext must expose BOTH {sorted(MESH_CONTEXT_METHODS)}; found {sorted(methods)}.\n"
            "resolved_peers(portal) is peers ONLY (the _wiring.py:745 peer_targets shape, which "
            "feeds an AGENT_CYCLE_TOOL member's tools_destinations); destinations_for(portal) is "
            "resolved_peers + (exit_name,) (the :358/:415/:424/:746 shape). Collapsing them widens "
            "a tool-triggered member's goto target set and NO behavioral test would catch it. "
            "Whether tools_destinations SHOULD carry exit_name is neograph-dgbqv.7 -- a separate, "
            "deliberate decision, not a side effect of a plumbing collapse."
        )

    # --- (b) the dispatch table is total ------------------------------------

    def test_member_class_adapter_table_is_total_over_the_reachable_classes(self):
        """A sixth ``PortalMemberClass`` cannot be added without an adapter.

        ``DISPATCH`` is excluded because it is structurally unreachable inside
        ``_add_portal_mesh``: ``_contiguous_portal_mesh`` breaks on a
        dispatch-mode Portal, so such a node is never a mesh member. That is
        the same partition shape ``COMBO_DECOMPOSITION``'s totality test uses.
        """
        import importlib

        from neograph._portal_member import PortalMemberClass

        module = importlib.import_module(f"neograph.{ADAPTER_TABLE_MODULE}")
        table = getattr(module, ADAPTER_TABLE_NAME, None)
        assert table is not None, (
            f"neograph.{ADAPTER_TABLE_MODULE}.{ADAPTER_TABLE_NAME} does not exist. "
            "neograph-dgbqv.4 replaces _add_portal_mesh's member-kind if/elif chain with a "
            "module-level dict[PortalMemberClass, Adapter] so the dispatch is total by "
            "construction rather than by reading the chain."
        )
        expected = set(PortalMemberClass) - {PortalMemberClass.DISPATCH}
        assert set(table) == expected, (
            f"{ADAPTER_TABLE_NAME} is not total over the reachable member classes.\n"
            f"  missing an adapter: {sorted(c.name for c in expected - set(table))}\n"
            f"  unexpected key:     {sorted(getattr(c, 'name', repr(c)) for c in set(table) - expected)}\n"
            "DISPATCH is excluded by reachability (_contiguous_portal_mesh breaks on it), not by "
            "oversight -- do not add an adapter for it, and do not drop one to make this pass."
        )

    # --- (c) the decision functions collapse --------------------------------

    def test_decision_functions_take_the_spec_instead_of_the_bundle_by_hand(self):
        source = (SRC_DIR / DECISION_MODULE).read_text()
        offenders: list[str] = []
        for name in sorted(DECISION_FUNCTIONS):
            params = _function_params(source, name)
            assert params is not None, (
                f"{DECISION_MODULE} no longer defines {name}. neograph-dgbqv.4 RE-SIGNATURES the "
                "two decision functions; it does not move or rename them -- G1's Command monopoly "
                "is {'factory.py', 'runner.py'} and must stay byte-identical."
            )
            extra = [p for p in params if p not in SPEC_SIGNATURE_PARAMS]
            if len(extra) > MAX_PARAMS_BEYOND_SPEC:
                offenders.append(f"{name}: {len(extra)} params beyond (update, state, spec) -> {extra}")
        assert not offenders, (
            "A Portal decision function still threads the routing bundle by hand.\n"
            "Every one of these fields belongs on PortalRouteSpec, built ONCE per member "
            f"(neograph-dgbqv.4). At most {MAX_PARAMS_BEYOND_SPEC} params may remain beyond "
            "(update, state, spec).\n"
            "Expected to be red today at 13 (_portal_route_to_command) and 8 "
            "(_tool_handoff_to_command).\n" + "\n".join(offenders)
        )

    def test_the_tool_handoff_targets_are_spec_resident(self):
        """``handoff_target_key`` and ``loopback_target`` travel ON the spec.

        This is what makes assertion (a) satisfiable rather than merely moved:
        ``factory.py:644`` derives the tool-target key and ``:648`` the
        loopback. Keeping either as an explicit kwarg would relocate the
        derivation to ``_wiring.py`` and leave the disease intact (architect
        review finding 2, neograph-jn555.20).
        """
        source = (SRC_DIR / DECISION_MODULE).read_text()
        params = _function_params(source, "_tool_handoff_to_command") or []
        leaked = [p for p in params if p in {"handoff_target_key", "loopback_target"}]
        assert not leaked, (
            f"_tool_handoff_to_command still takes {leaked} as explicit param(s). Both are "
            "spec-resident, built by PortalRouteSpec.for_tool_member(...) -- handoff_target_key "
            "from StateKeys.handoff_tool_target(field_name) and loopback_target from "
            "cycle_names(node.name).agent (the existing primitive; no hand-rolled "
            "'{name}__agent' f-string anywhere in the new module)."
        )


class TestHandoffKeyScannerMetaTests:
    """Positive + negative meta-tests for ``_handoff_key_call_sites``. A guard
    whose scanner silently matches nothing is worse than no guard."""

    def test_meta_flags_a_state_keys_call(self):
        src = "def f(entry_field):\n    return StateKeys.handoff_hops(entry_field)\n"
        assert [(fn, call) for fn, _l, call in _handoff_key_call_sites(src)] == [("f", "StateKeys.handoff_hops")]

    def test_meta_flags_every_builder_name(self):
        src = (
            "def f(e, n):\n"
            "    a = StateKeys.handoff_payload(e)\n"
            "    b = StateKeys.handoff_hops(e)\n"
            "    c = StateKeys.portal_proposed_target(n)\n"
            "    d = StateKeys.handoff_tool_target(n)\n"
            "    return a, b, c, d\n"
        )
        assert len(_handoff_key_call_sites(src)) == len(HANDOFF_KEY_BUILDERS)

    def test_meta_flags_an_aliased_import(self):
        """The strengthening over a ``base is StateKeys`` check: an alias must
        not walk through."""
        src = "from neograph._state_keys import StateKeys as SK\ndef f(e):\n    return SK.handoff_hops(e)\n"
        assert [call for _fn, _l, call in _handoff_key_call_sites(src)] == ["SK.handoff_hops"]

    def test_meta_flags_a_conditional_call(self):
        """The real shape at factory.py:173 -- a ternary must not hide it."""
        src = "def f(n, approve_name):\n    return StateKeys.portal_proposed_target(n) if approve_name else None\n"
        assert len(_handoff_key_call_sites(src)) == 1

    def test_meta_reports_the_innermost_enclosing_function(self):
        src = "def outer(e):\n    def inner(x):\n        return StateKeys.handoff_hops(x)\n    return inner\n"
        assert [fn for fn, _l, _c in _handoff_key_call_sites(src)] == ["inner"]

    def test_meta_ignores_docstring_and_comment_mentions(self):
        """WHY this scanner is AST and never grep: three permanent false
        positives live in the tree (factory.py x2, _subconstruct.py)."""
        src = '"""Reads StateKeys.handoff_hops(entry_field) upstream."""\n# and StateKeys.handoff_payload\nX = 1\n'
        assert _handoff_key_call_sites(src) == []

    def test_meta_ignores_a_bare_attribute_reference(self):
        """Handing the builder itself to something is not minting a key."""
        src = "def f():\n    return StateKeys.handoff_hops\n"
        assert _handoff_key_call_sites(src) == []

    def test_meta_ignores_the_healthy_spec_read(self):
        """Negative: the cured form reads a cached field off the spec."""
        src = "def f(update, state, spec):\n    return state.get(spec.count_field)\n"
        assert _handoff_key_call_sites(src) == []

    def test_meta_ignores_an_unrelated_state_keys_builder(self):
        src = "def f(n):\n    return StateKeys.each_item(n)\n"
        assert _handoff_key_call_sites(src) == []

    def test_meta_confirms_the_docstring_only_files_hold_no_call(self):
        """Live control: the literal claims ``_subconstruct.py`` mentions these
        builders in prose and calls none. If that stops being true the reason
        this guard is AST-based has changed and the literal is stale."""
        for name in sorted(DOCSTRING_ONLY_MENTIONS):
            path = SRC_DIR / name
            assert path.is_file(), f"DOCSTRING_ONLY_MENTIONS names a file that does not exist: {name}"
            assert _handoff_key_call_sites(path.read_text()) == [], (
                f"{name} now CALLS a handoff key builder. Either migrate it or move it into "
                "DECLARED_KEY_CALLERS with a stated reason -- do not leave it in "
                "DOCSTRING_ONLY_MENTIONS, which asserts the opposite."
            )


class TestSignatureScannerMetaTests:
    """Positive + negative meta-tests for ``_function_params`` /
    ``_defined_methods``."""

    def test_meta_counts_positional_and_kwonly_params(self):
        src = "def f(update, state, *, a, b):\n    return 1\n"
        assert _function_params(src, "f") == ["update", "state", "a", "b"]

    def test_meta_counts_async_definitions(self):
        """The sync/async twins stay literally duplicated (jtawq.6), so the
        async half must be visible to the same scanner."""
        src = "async def f(update, state, *, a):\n    return 1\n"
        assert _function_params(src, "f") == ["update", "state", "a"]

    def test_meta_returns_none_for_a_missing_function(self):
        assert _function_params("X = 1\n", "f") is None

    def test_meta_reports_the_collapsed_signature_as_zero_extra(self):
        src = "def f(update, state, spec):\n    return 1\n"
        assert [p for p in (_function_params(src, "f") or []) if p not in SPEC_SIGNATURE_PARAMS] == []

    def test_meta_finds_methods_on_the_named_class(self):
        src = "class MeshContext:\n    def resolved_peers(self, p):\n        return ()\n"
        assert _defined_methods(src, "MeshContext") == {"resolved_peers"}

    def test_meta_ignores_methods_on_another_class(self):
        src = "class Other:\n    def destinations_for(self, p):\n        return ()\n"
        assert _defined_methods(src, "MeshContext") == set()


class TestGuardIsNotVacuous:
    """Anti-vacuity, written against the PREDICATES rather than a file list. If
    the scanners ever stop matching, these fail even though the tree is clean —
    so a silently-dead guard cannot masquerade as a passing one."""

    def test_key_scanner_matches_a_synthetic_diseased_module(self):
        src = (
            "def make_portal_fn(node, portal, entry_field, exit_name, *, approve_name=None):\n"
            "    channel_key = StateKeys.handoff_payload(entry_field)\n"
            "    count_field = StateKeys.handoff_hops(entry_field)\n"
            "    proposed = StateKeys.portal_proposed_target(node.name) if approve_name else None\n"
            "    return channel_key, count_field, proposed\n"
        )
        assert len(_handoff_key_call_sites(src)) == 3

    def test_key_scanner_still_sees_live_sites_in_a_permanently_declared_owner(self):
        """Live control from the real tree: ``state.py`` builds these keys on
        the state-MODEL side and is out of P9's scope permanently. If this ever
        returns empty, the scanner has broken, not the codebase."""
        assert _handoff_key_call_sites((SRC_DIR / "state.py").read_text())

    def test_param_scanner_still_sees_the_two_decision_functions(self):
        """Live control: both functions exist in ``factory.py`` today and must
        still exist after the collapse (G1 monopoly unchanged)."""
        source = (SRC_DIR / DECISION_MODULE).read_text()
        for name in sorted(DECISION_FUNCTIONS):
            assert _function_params(source, name), f"{DECISION_MODULE} no longer defines {name}"

    def test_declared_key_callers_are_real_files_except_the_deliverable(self):
        """A typo'd entry would silently exempt nothing — or, worse, read as a
        wildcard to a future maintainer."""
        missing = [n for n in sorted(DECLARED_KEY_CALLERS) if n != SPEC_MODULE and not (SRC_DIR / n).is_file()]
        assert not missing, f"DECLARED_KEY_CALLERS names file(s) that do not exist: {missing}"

    def test_the_declared_owners_are_disjoint_from_the_files_being_cured(self):
        """``factory.py`` / ``_wiring.py`` must never be admitted to the
        declared set — that would make assertion (a) pass by permission rather
        than by cure."""
        cured = {"factory.py", "_wiring.py"}
        assert not (DECLARED_KEY_CALLERS & cured), (
            f"DECLARED_KEY_CALLERS admits {sorted(DECLARED_KEY_CALLERS & cured)}, the very files "
            "neograph-dgbqv.4 removes these calls from. Weakening the literal is not a fix."
        )
