"""Structural guard (G-PMC): "what kind of Portal mesh participant is this item"
has exactly ONE authority.

neograph-dgbqv.3 (P7 of the Agent Spec dispatch-vocabulary epic). Written
FAILING, before the classifier exists, per this repo's guard-first discipline.

The disease: that question is independently re-derived across the package by
hand-written ``portal.is_dispatch`` / ``portal.is_tool_triggered`` reads and
``member.mode in ("agent", "act")`` compares, so adding or reshaping a member
kind means editing every site or silently diverging. It is the same
duplicated-source-of-truth anti-pattern AGENTS.md records for the Portal
rollout, and the direct sibling of
``tests/test_guards_combo_decomposition_consumers.py``, whose shape this file
copies (MIGRATED / PENDING / EXEMPT literals, AST scanners, mutation
meta-tests, anti-vacuity tripwire).

The cure: ``portal_member_class(item) -> PortalMemberClass | None`` in
``src/neograph/_portal_member.py``. Its observable contract — including the
LOSSY precedence ``SUB_CONSTRUCT > AGENT_CYCLE_* > ATOMIC_OPERATOR > ATOMIC``
— is pinned behaviorally by ``tests/test_portal_member_class.py``. This file
pins only the STRUCTURE: that every consumer reads it and none re-derives it.

Two detection axes, both crisp and AST-decidable
------------------------------------------------
The design doc proposed keying on "isinstance-Node-or-Construct-in-portal-
context". That has no decidable AST form (``isinstance(x, Construct)`` appears
all over ``src/`` for non-Portal reasons), so the architect review (yf2ar.28)
required a redesign. What is implemented instead:

**Axis A — discriminator-attribute access** (the load-bearing half). Reading
``.is_dispatch`` or ``.is_tool_triggered`` outside ``_portal.py`` (which DEFINES
them) and the classifier module. Same shape as the already-passing inline
``route == "decide"`` ban in ``tests/test_guards_assembly.py``
(``TestPortalDispatchDiscriminationMonopoly``). Tree-wide and exact: it names
exactly eight files today, with zero false positives.

**Axis B — agent/act mode discrimination**, scoped to MIGRATED files. A
``.mode`` compare against ``"agent"``/``"act"`` is a legitimate, common idiom
elsewhere in the package (tool binding, lint, scaffolding, LLM-node collection),
so this axis is deliberately NOT tree-wide — it would be over-broad and would
turn the guard into noise. Scoping it to the declared consumer inventory is the
"narrower, stated rule" the review asked for. Sites inside a MIGRATED file that
ask a genuinely non-Portal agent/act question are listed in ``EXEMPT`` with a
reason, keyed by ``(file, enclosing function, axis)`` so a line shift never
silently re-arms or dis-arms an exemption.

NOT scanned, deliberately: the node-vs-construct ``isinstance`` axis (not
AST-decidable, per above) and ``loader.py``'s ``type(agent).__name__ == "Flow"``
derivation (a FOREIGN pyagentspec object with no ``.modifier_set``, which the
classifier cannot classify by construction — see ``EXEMPT_FILES``).

Written in pure ``ast`` with no ``re``, so it is exempt by construction from
``tests/test_guards_meta.py``'s named-regex/slip-test discipline — the same move
the combo-decomposition guard makes.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# --- independent literals (hand-written; never derived from the scan) --------

#: The single authority. Resolved by dgbqv.3's plan step 0: NOT ``modifiers.py``
#: (which sits at its exact file-size ceiling, and "raise the number" is not a
#: remedy) and NOT ``_portal.py`` (``modifiers.py`` imports it, so importing
#: ``classify_modifiers``/``COMBO_DECOMPOSITION`` back would cycle). A new leaf
#: module above ``modifiers.py`` and below every consumer.
CLASSIFIER_MODULE = "_portal_member.py"

#: Where ``Portal.is_dispatch`` / ``.is_tool_triggered`` are DEFINED. Scoped out
#: of axis A by construction — reading them here is the definition, not a
#: re-derivation.
DISCRIMINATOR_OWNER = "_portal.py"

#: Files that must READ the classifier and hold ZERO unexempted member-class
#: re-derivation. Seeded from a sweep run at write-test time (2026-08-05,
#: against HEAD dcf0b4f), NOT copied from the ticket's or the design doc's
#: citation lists — both predate three landed siblings and both are declared
#: provisional by their own authors.
MIGRATED: frozenset[str] = frozenset(
    {
        # The consumer BOTH the ticket and docs/design/agent-spec-target-
        # architecture-2026-08-03.md miss: `_member_hop_cost` is already a
        # hand-written four-way member-class dispatch, and `_ensure_agent_
        # recursion_limit` costs agent/act nodes by excluding mesh members.
        "_recursion_budget.py",
        # `_contiguous_portal_mesh`'s dispatch break, `_add_portal_mesh`'s
        # member-kind chain (the fullest hand-written taxonomy in the tree,
        # and the migration exemplar), `_add_portal_agent_cycle_member`.
        "_wiring.py",
        # Member collection (`_check_portal_mesh`) + the dispatch error-handler
        # walk. Its pair-LEGALITY arms are exempted below, not migrated.
        "_validation_portal.py",
        # Dispatch-field producer registration.
        "_construct_validation.py",
        # Swarm export: sub-construct-as-Flow, tool-trigger -> HandoffMode, and
        # `_is_peer_mesh_member` (which collapses to one classifier call).
        "_agent_spec_portal.py",
        # The mesh-entry vs dispatch arm select in `compile`.
        "compiler.py",
        # The private `_is_dispatch` helper, the member filter, and the
        # Operator-guarded proposed-target field (the ATOMIC_OPERATOR axis).
        "state.py",
        # MeshContext.build's entry_label_map derivation -- the classifier
        # call that replaces _wiring.py:315's getattr(member, 'mode')
        # re-derivation (neograph-dgbqv.4, P9).
        "_portal_route.py",
        # neograph-dgbqv.5 (P10): the Swarm-import cluster's `type(agent).
        # __name__ == "Flow"` derivation now reads SWARM_ENCODING[PortalMemberClass.
        # SUB_CONSTRUCT].spec_class instead of a hard-coded literal -- MOVED here
        # from EXEMPT_FILES/NO_DISCRIMINATOR_ATTR_SITES (removed together below),
        # never kept alongside.
        "_agent_spec_swarm_import.py",
        # neograph-dgbqv.5 (P10): the ONE Agent-Spec Swarm <-> PortalMemberClass
        # encoding table -- every SWARM_ENCODING row is keyed by PortalMemberClass,
        # a live import-and-use of the classifier's own enum.
        "_agent_spec_swarm_encoding.py",
        # neograph-dgbqv.12: the IR normalizer's mesh-member collection AND its
        # handoff_channel stamping. Both used `portal.is_not None`, which swept in a
        # standalone route="decide" node -- making it members[0] when it preceded a
        # mesh, so the runtime wrote one channel key and every member read another.
        # Now asks the same classifier _wiring and _validation_portal already did.
        "_ir_normalize.py",
        # neograph-wvp7j: the test scaffold's mesh collector. It asks the classifier
        # twice -- to FILTER construct.nodes down to mesh participants (excluding
        # DISPATCH, which is a standalone linear node and never a member), and to
        # record each member's class in the generated assertion so a lowering change
        # shows up as drift. Both are genuine member-class questions.
        "testing/_scaffold_capture.py",
    }
)

#: Known-diseased files whose migration is sequenced LATER. A RATCHET: it may
#: only shrink. Declared EMPTY from the start — dgbqv.3 migrates every in-scope
#: consumer in one commit-coherent pass, so there is nothing to park here. New
#: member-class dispatch must be written against the classifier, never deferred.
PENDING: frozenset[str] = frozenset()

#: Whole files that ask a DIFFERENT question and are exempt by stated reason.
#: (Site-level exemptions inside MIGRATED files live in ``EXEMPT`` below.)
EXEMPT_FILES: dict[str, str] = {
    # Portal(is_dispatch) x Operator PAIR-LEGALITY (`_DYNAMIC_RULES`). This is
    # the neograph-jtawq.3 / P3 axis: it reads is_dispatch for a legality
    # verdict about a specific PRODUCT, never to name a member class. Migrating
    # it would be a category error (dgbqv.3 disease-scan row 9).
    "modifiers.py": "pair-legality (_DYNAMIC_RULES), the jtawq.3/P3 axis -- not a member-class question",
    # The lint framework-consumer derivation reads is_dispatch to know WHICH
    # attribute holds a field NAME -- spec_field/input_field in dispatch mode,
    # route in peer mode. The answer is a string to look up in the node's own
    # output model, never a member class, and the two peer-mode member classes
    # are indistinguishable to it: both name their routing field the same way.
    # Migrating it to portal_member_class would be a category error, the same
    # one modifiers.py is exempt for.
    "_lint_consumers.py": "which Portal attribute names an output field -- not a member-class question",
}

#: Site-level exemptions inside MIGRATED files, keyed
#: ``(file, enclosing function, axis)`` -> reason. Function-keyed, not
#: line-keyed, so a line shift cannot silently re-arm or dis-arm one.
#: Axis is ``"discriminator"`` (axis A) or ``"mode"`` (axis B).
#:
#: Adding an entry here is a DESIGN decision that requires a written reason, not
#: a way to quiet the guard. The two ``_validation_portal.py`` entries are the
#: architect review's re-disposition (Refined Plan item 3): they answer the same
#: "is this (member-kind, operator-or-trigger) PRODUCT legal" question already
#: exempted at ``modifiers.py``, and the classifier's LOSSY precedence cannot
#: resolve them without discarding the very axis the rule tests.
EXEMPT: dict[tuple[str, str, str], str] = {
    (
        "_validation_portal.py",
        "_check_one_mesh_group",
        "discriminator",
    ): "trigger x member-kind pair-legality (same category as modifiers.py's _DYNAMIC_RULES)",
    (
        "_validation_portal.py",
        "_check_one_mesh_group",
        "mode",
    ): "Operator x member-kind and trigger x member-kind pair-legality; rejects illegal PRODUCTS",
    (
        "compiler.py",
        "compile",
        "mode",
    ): "agent/act TOOL-binding collection -- not a Portal member-class question",
    (
        "compiler.py",
        "_add_node_to_graph",
        "mode",
    ): "agent/act ReAct-cycle lowering for ordinary (non-mesh) nodes",
    (
        "state.py",
        "_add_agent_channels",
        "mode",
    ): "agent/act message-channel state fields -- unrelated to Portal membership",
    (
        "_recursion_budget.py",
        "_ensure_agent_recursion_limit",
        "mode",
    ): "flat per-node agent/act cost for EVERY node in the construct (mesh members already "
    "excluded via the id() set from _portal_mesh_member_ids) -- a standalone agent/act-node "
    "question, not a Portal member-class one; portal_member_class(node) would return None "
    "for a non-mesh agent/act node, which is the wrong answer here",
}

#: Declared consumers that carry NO axis-A site, so the filesystem census below
#: cannot see them until they start importing the classifier. Hand-listed so
#: assertion (c) stays a strict EQUALITY rather than a one-directional subset
#: check (the anti-tautology lesson from tests/test_guards_parity_ratchet.py).
#: SHRINK-ONLY: ``_agent_spec_swarm_import.py`` left this set in neograph-dgbqv.5
#: (P10) -- it now imports PortalMemberClass directly, so DELETE-not-move.
#: Declared EMPTY, matching the general shrink-only-ratchet pattern.
NO_DISCRIMINATOR_ATTR_SITES: frozenset[str] = frozenset()

#: The classifier's public symbols. A MIGRATED file must import-and-USE at
#: least one (a dead import does not count).
CLASSIFIER_SYMBOLS: frozenset[str] = frozenset({"PortalMemberClass", "portal_member_class"})

#: The discriminator properties axis A bans outside owner + classifier.
DISCRIMINATOR_ATTRS: frozenset[str] = frozenset({"is_dispatch", "is_tool_triggered"})

#: The mode literals axis B keys on.
AGENT_CYCLE_MODES: frozenset[str] = frozenset({"agent", "act"})


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


def _discriminator_sites(source: str) -> list[tuple[str, int, str]]:
    """Axis A: ``(function, lineno, attr)`` for every ``.is_dispatch`` /
    ``.is_tool_triggered`` attribute access.

    An ``ast.Attribute`` node, so a docstring or comment mentioning the property
    never trips it — the same precision ``TestPortalDispatchDiscriminationMonopoly``
    gets from matching an ``ast.Compare`` rather than the raw text.
    """
    tree = ast.parse(source)
    where = _enclosing_functions(tree)
    hits = {
        (where.get(node.lineno, "<module>"), node.lineno, node.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in DISCRIMINATOR_ATTRS
    }
    return sorted(hits)


def _reads_mode(operand: ast.expr) -> bool:
    """True if ``operand`` reads a ``.mode`` — as an attribute access OR via
    ``getattr(x, "mode", ...)``.

    The ``getattr`` arm was added by neograph-dgbqv.4 (P9). It closes a hole
    this axis was structurally blind to: ``_wiring.py:315`` built its
    entry-label map with ``getattr(member, "mode", None) in ("agent", "act")``,
    and because a ``getattr`` call is an ``ast.Call`` rather than an
    ``ast.Attribute``, the live disease site inside a MIGRATED file slipped
    straight through. A ``Construct`` has no ``.mode`` field, which is exactly
    why a re-derivation reaches for the defaulted form — so the defaulted form
    is the one most likely to regrow.

    Tightening this cost nothing: a tree-wide sweep finds exactly TWO
    ``getattr(x, "mode")`` sites in ``src/``, and the other is the classifier's
    own definition (``_portal_member.py:94``), which compares against a NAME
    rather than a string literal and sits outside MIGRATED on both counts.
    """
    if isinstance(operand, ast.Attribute) and operand.attr == "mode":
        return True
    return (
        isinstance(operand, ast.Call)
        and isinstance(operand.func, ast.Name)
        and operand.func.id == "getattr"
        and len(operand.args) >= 2
        and isinstance(operand.args[1], ast.Constant)
        and operand.args[1].value == "mode"
    )


def _agent_mode_sites(source: str) -> list[tuple[str, int, tuple[str, ...]]]:
    """Axis B: ``(function, lineno, literals)`` for every ``.mode`` compare naming
    ``"agent"`` or ``"act"``.

    Covers ``node.mode in ("agent", "act")``, ``item.mode == "agent"``,
    ``getattr(m, "mode", None) in ("agent", "act")``, and the ``not in``/``!=``
    negations — one side must READ a mode (see ``_reads_mode``) and the other
    must name an agent-cycle mode, directly or inside a tuple/list/set literal.
    """
    tree = ast.parse(source)
    where = _enclosing_functions(tree)
    hits: set[tuple[str, int, tuple[str, ...]]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in node.ops):
            continue
        operands = [node.left, *node.comparators]
        if not any(_reads_mode(o) for o in operands):
            continue
        literals: set[str] = set()
        for operand in operands:
            if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
                literals.add(operand.value)
            elif isinstance(operand, (ast.Tuple, ast.List, ast.Set)):
                literals.update(
                    e.value for e in operand.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
                )
        if literals & AGENT_CYCLE_MODES:
            hits.add((where.get(node.lineno, "<module>"), node.lineno, tuple(sorted(literals))))
    return sorted(hits)


def _used_classifier_symbols(source: str) -> set[str]:
    """Classifier symbols imported from the classifier module AND actually USED.

    A dead import must not satisfy assertion (b) — the binding has to be
    referenced somewhere (an ``ast.Attribute`` such as
    ``PortalMemberClass.ATOMIC`` contains that ``ast.Name``, so attribute
    references count). Alias-tolerant.
    """
    tree = ast.parse(source)
    module_stem = CLASSIFIER_MODULE.removesuffix(".py")
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[-1] == module_stem:
            for alias in node.names:
                if alias.name in CLASSIFIER_SYMBOLS:
                    imported[alias.asname or alias.name] = alias.name
    if not imported:
        return set()
    return {
        imported[node.id]
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in imported
    }


def _package_files() -> list[pathlib.Path]:
    """Every .py under src/neograph (recursive -- a subpackage must not escape)."""
    return sorted(p for p in SRC_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(SRC_DIR).as_posix()


def _unexempted(name: str, sites: list, axis: str) -> list[str]:
    """Drop sites whose ``(file, function, axis)`` key carries a stated reason."""
    return [
        f"{name}:{lineno}\t{axis}\t{function}\t{detail}"
        for function, lineno, detail in sites
        if (name, function, axis) not in EXEMPT
    ]


# --- the guard --------------------------------------------------------------


class TestPortalMemberClassConsumerMonopoly:
    """neograph-dgbqv.3: "what kind of Portal mesh participant is this item" is
    answered in exactly ONE place — ``portal_member_class`` in
    ``_portal_member.py``. Every consumer READS it; none re-derives the class
    from ``.is_dispatch`` / ``.is_tool_triggered`` / an agent-act mode compare.
    """

    # --- (a) no re-derivation in a migrated file ----------------------------

    def test_no_hand_written_member_class_derivation_in_migrated_files(self):
        offenders: list[str] = []
        for name in sorted(MIGRATED):
            path = SRC_DIR / name
            assert path.is_file(), f"MIGRATED names a file that does not exist: {name}"
            source = path.read_text()
            offenders += _unexempted(name, _discriminator_sites(source), "discriminator")
            offenders += _unexempted(name, _agent_mode_sites(source), "mode")
        assert not offenders, (
            "Hand-written Portal member-class derivation found in a MIGRATED file.\n"
            "Call `portal_member_class(item)` and match on `PortalMemberClass` instead "
            "(neograph-dgbqv.3). If a site asks a genuinely different question -- pair "
            "LEGALITY of a specific product, or a non-Portal agent/act concern -- add it "
            "to EXEMPT with a written reason; do not delete the assertion.\n" + "\n".join(offenders)
        )

    # --- (b) reads the classifier -------------------------------------------

    def test_every_migrated_file_imports_and_uses_a_classifier_symbol(self):
        missing: list[str] = []
        for name in sorted(MIGRATED):
            if not _used_classifier_symbols((SRC_DIR / name).read_text()):
                missing.append(name)
        assert not missing, (
            f"MIGRATED file(s) do not import-and-use any of {sorted(CLASSIFIER_SYMBOLS)} "
            f"from neograph.{CLASSIFIER_MODULE.removesuffix('.py')}. A dead import does not "
            "count -- the binding must actually be referenced.\n" + "\n".join(missing)
        )

    # --- (c) completeness / anti-tautology ----------------------------------

    def test_member_class_consumers_are_exactly_the_declared_inventory(self):
        """Filesystem-derived census == the hand-written literals.

        The two sides come from independent sources, so this cannot pass
        tautologically. A NEW file that grows a discriminator read (or starts
        reading the classifier) must be declared here -- this guard IS the
        consumer inventory.

        Census predicate: holds an axis-A site OR imports-and-uses a classifier
        symbol. That union is what makes the assertion stable ACROSS the
        migration -- pre-migration a consumer qualifies via its
        ``.is_dispatch`` read, post-migration via its classifier import.
        """
        actual = {
            _rel(p)
            for p in _package_files()
            if _discriminator_sites(p.read_text()) or _used_classifier_symbols(p.read_text())
        }
        expected = (set(MIGRATED) | set(PENDING) | set(EXEMPT_FILES) | {CLASSIFIER_MODULE, DISCRIMINATOR_OWNER}) - set(
            NO_DISCRIMINATOR_ATTR_SITES
        )
        assert actual == expected, (
            "The set of src/neograph files answering the Portal member-class question "
            "diverged from the declared inventory.\n"
            f"  undeclared (new consumer -- migrate it, or declare it): {sorted(actual - expected)}\n"
            f"  declared but absent from the census: {sorted(expected - actual)}"
        )

    def test_classifier_module_exists_and_is_the_sole_authority(self):
        path = SRC_DIR / CLASSIFIER_MODULE
        assert path.is_file(), (
            f"{CLASSIFIER_MODULE} does not exist. It is the deliverable of neograph-dgbqv.3 "
            "and the single authority this whole guard is built around."
        )
        source = path.read_text()
        tree = ast.parse(source)
        defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert CLASSIFIER_SYMBOLS <= defined, (
            f"{CLASSIFIER_MODULE} must define {sorted(CLASSIFIER_SYMBOLS)}; found {sorted(defined)}"
        )

    def test_classifier_module_holds_no_node_or_construct_function_local_import(self):
        """The import-discipline rule, scoped to the CLASSIFIER MODULE only.

        The design doc specified this as "modifiers.py contains no module-level
        or function-local Node/Construct import", which is UNSATISFIABLE: that
        file already holds an allowlisted function-local pair for Loop
        validation (tests/test_guards_sidecar_imports.py). Re-scoped to the new
        module, where the real rule is that
        ``FUNCTION_LOCAL_IMPORT_ALLOWLIST`` must not grow: a MODULE-level
        Node/Construct import is fine here (the module sits above them in the
        DAG and nothing low-level imports it), a FUNCTION-local one is not.
        """
        path = SRC_DIR / CLASSIFIER_MODULE
        if not path.is_file():
            return  # covered by test_classifier_module_exists_and_is_the_sole_authority
        tree = ast.parse(path.read_text())
        module_level = {id(n) for n in tree.body}
        offenders = [
            f"line {n.lineno}: {ast.unparse(n)}"
            for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom)) and id(n) not in module_level
        ]
        assert not offenders, (
            f"{CLASSIFIER_MODULE} holds a function-local import. That grows "
            "FUNCTION_LOCAL_IMPORT_ALLOWLIST, which this repo's file-split decision "
            "ladder refuses. Import at module level (the DAG permits it) or relocate.\n" + "\n".join(offenders)
        )

    def test_pending_is_a_ratchet_and_the_literals_are_disjoint(self):
        assert not (MIGRATED & PENDING), f"A file is both MIGRATED and PENDING: {sorted(MIGRATED & PENDING)}"
        assert not (MIGRATED & set(EXEMPT_FILES)), (
            f"A file is both MIGRATED and EXEMPT_FILES: {sorted(MIGRATED & set(EXEMPT_FILES))}"
        )
        assert CLASSIFIER_MODULE not in MIGRATED and DISCRIMINATOR_OWNER not in MIGRATED
        assert PENDING == frozenset(), (
            "PENDING must be EMPTY. dgbqv.3 migrates every in-scope consumer in one pass; "
            "a site that genuinely asks a different question belongs in EXEMPT/EXEMPT_FILES "
            "with a reason, never parked here."
        )

    def test_every_exemption_carries_a_nonempty_reason(self):
        """An exemption without a stated reason is an unratcheted hole."""
        blank = [k for k, reason in {**EXEMPT_FILES, **EXEMPT}.items() if not (reason or "").strip()]
        assert not blank, f"Exemption(s) with no stated reason: {blank}"

    def test_every_exemption_is_live(self):
        """Anti-staleness: an exemption whose site no longer exists must be
        DELETED, not left behind granting silent permission to regrow."""
        stale: list[str] = []
        for (name, function, axis), _reason in EXEMPT.items():
            path = SRC_DIR / name
            assert path.is_file(), f"EXEMPT names a file that does not exist: {name}"
            source = path.read_text()
            sites = _discriminator_sites(source) if axis == "discriminator" else _agent_mode_sites(source)
            if not any(fn == function for fn, _lineno, _detail in sites):
                stale.append(f"{name}::{function} ({axis})")
        for name in EXEMPT_FILES:
            assert (SRC_DIR / name).is_file(), f"EXEMPT_FILES names a file that does not exist: {name}"
        assert not stale, (
            "EXEMPT entr(ies) no longer match any site -- delete them rather than leaving "
            "standing permission for the disease to regrow:\n" + "\n".join(stale)
        )

    def test_no_discriminator_attr_sites_literal_is_accurate(self):
        """The strict-equality escape hatch in (c) must stay honest: a file
        listed there must genuinely hold no axis-A site."""
        wrong = [
            name
            for name in sorted(NO_DISCRIMINATOR_ATTR_SITES)
            if (SRC_DIR / name).is_file() and _discriminator_sites((SRC_DIR / name).read_text())
        ]
        assert not wrong, (
            "NO_DISCRIMINATOR_ATTR_SITES claims these files hold no .is_dispatch/"
            f".is_tool_triggered site, but they do: {wrong}. Remove them from the literal."
        )


class TestDiscriminatorScannerMetaTests:
    """Positive + negative meta-tests for ``_discriminator_sites`` (axis A). A
    guard whose scanner silently matches nothing is worse than no guard."""

    def test_meta_flags_attribute_read(self):
        src = "def f(portal):\n    if portal.is_dispatch:\n        return 1\n    return 0\n"
        assert [(fn, attr) for fn, _l, attr in _discriminator_sites(src)] == [("f", "is_dispatch")]

    def test_meta_flags_tool_trigger_read(self):
        src = "def f(p):\n    return p.is_tool_triggered\n"
        assert [attr for _fn, _l, attr in _discriminator_sites(src)] == ["is_tool_triggered"]

    def test_meta_flags_chained_read(self):
        """The real shape at state.py / _agent_spec_portal.py: a full chain."""
        src = "def f(m):\n    return m.modifier_set.portal.is_dispatch\n"
        assert [attr for _fn, _l, attr in _discriminator_sites(src)] == ["is_dispatch"]

    def test_meta_reports_the_innermost_enclosing_function(self):
        """Exemptions are function-keyed, so an inner def must not be reported
        under its parent's name."""
        src = "def outer(x):\n    def inner(p):\n        return p.is_dispatch\n    return inner\n"
        assert [fn for fn, _l, _a in _discriminator_sites(src)] == ["inner"]

    def test_meta_reports_module_scope_sites(self):
        src = "RULES = [lambda p: p.is_dispatch]\n"
        assert [fn for fn, _l, _a in _discriminator_sites(src)] == ["<module>"]

    def test_meta_ignores_docstring_and_comment_mentions(self):
        src = '"""Reads portal.is_dispatch upstream."""\n# also is_tool_triggered\nX = 1\n'
        assert _discriminator_sites(src) == []

    def test_meta_ignores_the_healthy_classifier_read(self):
        """Negative: the cured form names neither property."""
        src = (
            "from neograph._portal_member import PortalMemberClass, portal_member_class\n"
            "def f(item):\n"
            "    return portal_member_class(item) is PortalMemberClass.DISPATCH\n"
        )
        assert _discriminator_sites(src) == []

    def test_meta_ignores_a_same_named_local_variable(self):
        """A plain name binding is not an attribute access."""
        src = "def f(route):\n    is_dispatch = route == 'decide'\n    return is_dispatch\n"
        assert _discriminator_sites(src) == []

    def test_meta_owner_module_is_the_only_definition_site(self):
        """Control from the real tree: _portal.py DEFINES both properties."""
        found = {attr for _fn, _l, attr in _discriminator_sites((SRC_DIR / DISCRIMINATOR_OWNER).read_text())}
        assert found <= DISCRIMINATOR_ATTRS
        source = (SRC_DIR / DISCRIMINATOR_OWNER).read_text()
        tree = ast.parse(source)
        properties = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert DISCRIMINATOR_ATTRS <= properties, (
            f"{DISCRIMINATOR_OWNER} must remain the definition site of {sorted(DISCRIMINATOR_ATTRS)}"
        )


class TestAgentModeScannerMetaTests:
    """Positive + negative meta-tests for ``_agent_mode_sites`` (axis B)."""

    def test_meta_flags_tuple_membership(self):
        src = 'def f(m):\n    return m.mode in ("agent", "act")\n'
        assert [lits for _fn, _l, lits in _agent_mode_sites(src)] == [("act", "agent")]

    def test_meta_flags_equality(self):
        src = 'def f(m):\n    return m.mode == "act"\n'
        assert [lits for _fn, _l, lits in _agent_mode_sites(src)] == [("act",)]

    def test_meta_flags_set_and_list_literals(self):
        src = 'def f(m, n):\n    return m.mode in {"agent"} or n.mode not in ["act"]\n'
        assert len(_agent_mode_sites(src)) == 2

    def test_meta_flags_compound_condition(self):
        """The real shape at _recursion_budget.py:53 -- an isinstance conjunct
        must not hide the mode compare."""
        src = (
            'from neograph.node import Node\ndef f(m):\n    return isinstance(m, Node) and m.mode in ("agent", "act")\n'
        )
        assert [fn for fn, _l, _lits in _agent_mode_sites(src)] == ["f"]

    def test_meta_flags_the_getattr_membership_form(self):
        """POSITIVE meta-test for the neograph-dgbqv.4 scanner extension.

        The real shape at ``_wiring.py:315``, which the ``ast.Attribute``-only
        scanner could not see. A scanner extension without its own meta-test is
        unpinned.
        """
        src = 'def f(m):\n    return getattr(m, "mode", None) in ("agent", "act")\n'
        assert [(fn, lits) for fn, _l, lits in _agent_mode_sites(src)] == [("f", ("act", "agent"))]

    def test_meta_flags_the_getattr_equality_form(self):
        src = 'def f(m):\n    return getattr(m, "mode", None) == "agent"\n'
        assert [lits for _fn, _l, lits in _agent_mode_sites(src)] == [("agent",)]

    def test_meta_flags_the_getattr_form_inside_a_comprehension(self):
        """``_wiring.py:315`` sits inside a dict comprehension — the walk must
        reach it there, not only in a statement position."""
        src = 'def f(ms):\n    return {m.name: getattr(m, "mode", None) in ("agent", "act") for m in ms}\n'
        assert [fn for fn, _l, _lits in _agent_mode_sites(src)] == ["f"]

    def test_meta_ignores_getattr_on_another_attribute(self):
        src = 'def f(m):\n    return getattr(m, "route", None) == "agent"\n'
        assert _agent_mode_sites(src) == []

    def test_meta_ignores_the_classifier_own_definition_shape(self):
        """The ONLY other ``getattr(x, "mode")`` in ``src/``
        (``_portal_member.py:94``) compares against a NAME, not a string
        literal, so tightening axis B has zero collateral. Pinned so the
        "zero collateral" claim stays true rather than remembered."""
        src = (
            '_AGENT_CYCLE_MODES = frozenset({"agent", "act"})\n'
            "def portal_member_class(item):\n"
            '    return getattr(item, "mode", None) in _AGENT_CYCLE_MODES\n'
        )
        assert _agent_mode_sites(src) == []

    def test_meta_ignores_non_agent_mode_compares(self):
        """``mode == "scripted"`` and the think axis are not member-class
        questions -- axis B keys on the agent-cycle modes only."""
        src = 'def f(m):\n    return m.mode == "scripted" or m.mode == "raw"\n'
        assert _agent_mode_sites(src) == []

    def test_meta_ignores_a_bare_mode_read(self):
        src = "def f(m):\n    return m.mode\n"
        assert _agent_mode_sites(src) == []

    def test_meta_ignores_the_healthy_classifier_match(self):
        src = (
            "from neograph._portal_member import PortalMemberClass, portal_member_class\n"
            "def f(item):\n"
            "    match portal_member_class(item):\n"
            "        case PortalMemberClass.AGENT_CYCLE_TOOL:\n"
            "            return 1\n"
            "        case _:\n"
            "            return 0\n"
        )
        assert _agent_mode_sites(src) == []

    def test_meta_flags_a_think_inclusive_compare_that_still_names_agent(self):
        """``("think", "agent", "act")`` is an LLM-mode question, not a member-
        class one -- but it DOES name the agent-cycle modes, so the scanner
        reports it and the decision is recorded in EXEMPT rather than hidden in
        the scanner. Pinned so a future 'let me just filter think out' tweak is
        a deliberate, visible change."""
        src = 'def f(m):\n    return m.mode in ("think", "agent", "act")\n'
        assert [lits for _fn, _l, lits in _agent_mode_sites(src)] == [("act", "agent", "think")]


class TestClassifierSymbolScannerMetaTests:
    """Positive + negative meta-tests for ``_used_classifier_symbols`` (b)."""

    def test_meta_detects_imported_and_used_symbol(self):
        src = (
            "from neograph._portal_member import PortalMemberClass, portal_member_class\n"
            "def f(i):\n"
            "    return portal_member_class(i) is PortalMemberClass.ATOMIC\n"
        )
        assert _used_classifier_symbols(src) == {"PortalMemberClass", "portal_member_class"}

    def test_meta_rejects_dead_import(self):
        src = "from neograph._portal_member import portal_member_class\ndef f():\n    return 1\n"
        assert _used_classifier_symbols(src) == set()

    def test_meta_detects_aliased_import_and_reports_canonical_name(self):
        src = "from neograph._portal_member import portal_member_class as _pmc\ndef f(i):\n    return _pmc(i)\n"
        assert _used_classifier_symbols(src) == {"portal_member_class"}

    def test_meta_ignores_same_named_symbol_from_another_module(self):
        src = "from somewhere.else_ import portal_member_class\ndef f(i):\n    return portal_member_class(i)\n"
        assert _used_classifier_symbols(src) == set()


class TestGuardIsNotVacuous:
    """Anti-vacuity, written against the PREDICATES rather than a file list
    (Refined Plan item 7). If the scanners ever stop matching, these fail even
    though the tree is clean -- so a silently-dead guard cannot masquerade as a
    passing one."""

    def test_discriminator_scanner_matches_a_synthetic_diseased_module(self):
        src = (
            "def lower(member):\n"
            "    portal = member.modifier_set.portal\n"
            "    if portal is not None and portal.is_dispatch:\n"
            "        return 'dispatch'\n"
            "    if portal.is_tool_triggered:\n"
            "        return 'tool'\n"
            "    return 'atomic'\n"
        )
        assert len(_discriminator_sites(src)) == 2

    def test_agent_mode_scanner_matches_a_synthetic_diseased_module(self):
        src = (
            "from neograph.node import Node\n"
            "def cost(member):\n"
            '    if isinstance(member, Node) and member.mode in ("agent", "act"):\n'
            "        return 43\n"
            "    return 1\n"
        )
        assert len(_agent_mode_sites(src)) == 1

    def test_agent_mode_scanner_matches_the_getattr_disease_form(self):
        """Anti-vacuity for the neograph-dgbqv.4 extension, written against the
        PREDICATE. If the ``getattr`` arm ever stops matching, this fails even
        though the tree is clean."""
        src = (
            "def add_portal_mesh(members):\n"
            "    return {\n"
            '        m.name: (f"{m.name}__agent" if getattr(m, "mode", None) in ("agent", "act") else m.name)\n'
            "        for m in members\n"
            "    }\n"
        )
        assert len(_agent_mode_sites(src)) == 1

    def test_the_owner_module_still_holds_a_site_the_scanner_can_see(self):
        """Live control from the real tree: _portal.py's own
        ``not self.is_dispatch`` guard inside ``is_tool_triggered``. If this
        ever returns empty, the scanner has broken, not the codebase."""
        assert _discriminator_sites((SRC_DIR / DISCRIMINATOR_OWNER).read_text())

    def test_exemption_keys_are_reachable_shapes(self):
        """Every EXEMPT axis label must be one the scanners actually emit --
        a typo'd axis would silently exempt nothing (or, worse, read as a
        wildcard to a future maintainer)."""
        axes = {axis for _f, _fn, axis in EXEMPT}
        assert axes <= {"discriminator", "mode"}, f"unknown exemption axis: {sorted(axes - {'discriminator', 'mode'})}"
