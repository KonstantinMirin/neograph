"""Structural guard (neograph-jtawq.10): the Swarm-import cluster lives in its
own module, reaches ``from_agent_spec`` only as an INJECTED callable, and drags
every guard that scanned it along to its new home.

Disease. ``loader.py`` owned six functions that reconstruct a pyagentspec
``Swarm`` into a native Portal mesh. They are a self-contained cluster with a
single dependency on their own parent module -- the ``from_agent_spec`` entry
point -- which is exactly the shape the codebase already solved once, at
``_agent_spec_group_import.py``'s ``_construct_from_subflow(subflow, name,
from_spec)``: the parent's entry point arrives as a parameter so the child never
imports back up. The cluster stayed in ``loader.py`` only because the ticket
that noticed it predated that seam.

Two halves of the invariant, and the second is the one that fails silently.

**One-way chain, one owner per seam.** The moved module must not import
``loader`` -- not at module level, not deferred inside a function. It receives
``from_spec`` and threads it. A back-import would compile and pass every
behavioural test while re-creating the cycle the injection convention exists to
forbid.

**The guard surface follows the code.** Four sibling inventories are keyed on
the string ``"loader.py"`` for code that is INSIDE this cluster. Three of them
(``test_guards_agent_spec_import_seam``'s ``IMPORT_PATH_FILES``,
``test_guards_agent_spec_markers``'s ``SCANNED``,
``test_guards_portal_member_class_consumers``'s two literals) stay GREEN after
the move while scanning -- or granting a permission to -- a file that no longer
does the thing. That is the vacuous-repoint failure
``docs/file-split-procedure.md`` records as having nearly shipped in an earlier
split, and it is why this guard asserts on the inventories themselves rather
than trusting the gate to notice.

Pure AST plus direct reads of the sibling inventories, no ``re``, so this module
is exempt-by-construction from ``test_guards_meta.py`` Discipline 1.
"""

from __future__ import annotations

import ast
import pathlib

from tests.test_guards_agent_spec_import_seam import IMPORT_PATH_FILES
from tests.test_guards_agent_spec_markers import SCANNED
from tests.test_guards_branch_arm_walks import _ALLOWLIST as BRANCH_ARM_ALLOWLIST
from tests.test_guards_file_size import ALLOWLIST as FILE_SIZE_ALLOWLIST
from tests.test_guards_portal_member_class_consumers import (
    EXEMPT_FILES,
    NO_DISCRIMINATOR_ATTR_SITES,
)
from tests.test_guards_portal_member_class_consumers import (
    MIGRATED as PORTAL_MEMBER_CLASS_MIGRATED,
)

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src" / "neograph"

SWARM_MODULE = "_agent_spec_swarm_import.py"
SWARM_PATH = _SRC / SWARM_MODULE
LOADER_PATH = _SRC / "loader.py"

#: The whole cluster. Contiguous in ``loader.py`` before the move, and moved as
#: one unit -- a partial move leaves the seam with two owners again.
SWARM_CLUSTER = frozenset(
    {
        "_swarm_agents_ordered",
        "_synthesize_swarm_payload",
        "_swarm_trigger",
        "_flow_member_to_construct",
        "_reconstruct_swarm_mesh",
        "_reconstruct_swarm_mesh_with_operator_gates",
    }
)

#: The three that need the parent's entry point. Each threads it one layer
#: deeper: gates -> mesh -> member -> ``_construct_from_subflow``.
THREADING_FUNCTIONS = frozenset(
    {
        "_flow_member_to_construct",
        "_reconstruct_swarm_mesh",
        "_reconstruct_swarm_mesh_with_operator_gates",
    }
)

#: The injected parameter name, matching the exemplar at
#: ``_agent_spec_group_import.py:48``. A SECOND spelling would be a second
#: injection mechanism, which the ticket forbids explicitly.
INJECTED_PARAM = "from_spec"

#: The two dispatch points inside ``loader.from_agent_spec``.
DISPATCH_CALLEES = frozenset({"_reconstruct_swarm_mesh", "_reconstruct_swarm_mesh_with_operator_gates"})


def _defined_functions(path: pathlib.Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _imported_modules(source: str) -> set[str]:
    """Every module name reached by an ``import``, at ANY nesting depth.

    Walking the whole tree (not just module-level body) is the point: a deferred
    ``import`` inside a function is the usual way a back-edge gets re-introduced
    without anyone reading it as one.
    """
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


class TestSwarmClusterHasOneOwner:
    """The cluster moved, wholly, and left no copy behind."""

    def test_the_swarm_module_exists(self):
        assert SWARM_PATH.exists(), (
            f"{SWARM_MODULE} does not exist. The Swarm-import cluster is still "
            "inside loader.py; every assertion below is about its new home."
        )

    def test_the_whole_cluster_lives_in_the_swarm_module(self):
        defined = set(_defined_functions(SWARM_PATH))
        missing = SWARM_CLUSTER - defined
        assert not missing, (
            f"{SWARM_MODULE} must define the WHOLE swarm cluster; missing: "
            f"{sorted(missing)}. A partial move leaves the reconstruction "
            "procedure with two owners, which is the defect the sibling seam "
            "guard (neograph-s7zt3.11) exists to prevent."
        )

    def test_loader_no_longer_defines_the_cluster(self):
        left_behind = SWARM_CLUSTER & set(_defined_functions(LOADER_PATH))
        assert not left_behind, (
            f"loader.py still DEFINES {sorted(left_behind)} after the move. A "
            "re-export (the F401 block) is the sanctioned way to keep the name "
            "importable from neograph.loader; a second def is a second copy."
        )


class TestTheChainStaysOneWay:
    """``from_agent_spec`` arrives as a parameter, never as an import."""

    def test_the_swarm_module_never_imports_loader(self):
        offenders = {
            m for m in _imported_modules(SWARM_PATH.read_text(encoding="utf-8")) if m.split(".")[-1] == "loader"
        }
        assert not offenders, (
            f"{SWARM_MODULE} imports {sorted(offenders)}. The parent's entry "
            f"point must be INJECTED as `{INJECTED_PARAM}`, mirroring "
            "_agent_spec_group_import._construct_from_subflow -- importing it "
            "back up re-creates the cycle the convention forbids, and does so "
            "without failing a single behavioural test."
        )

    def test_the_swarm_module_never_names_the_entry_point_directly(self):
        """Even without an import, a bare ``from_agent_spec`` reference means the
        cluster is reaching for its parent rather than using what it was given."""
        tree = ast.parse(SWARM_PATH.read_text(encoding="utf-8"))
        hits = sorted({n.lineno for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "from_agent_spec"})
        assert not hits, (
            f"{SWARM_MODULE} references `from_agent_spec` by name at lines "
            f"{hits}. It must use the injected `{INJECTED_PARAM}` parameter -- "
            "one injection mechanism, not two."
        )

    def test_the_threading_functions_take_the_injected_callable_last(self):
        defined = _defined_functions(SWARM_PATH)
        for name in sorted(THREADING_FUNCTIONS):
            fn = defined.get(name)
            assert fn is not None, f"{name} is not defined in {SWARM_MODULE}"
            params = [a.arg for a in fn.args.args]
            assert params and params[-1] == INJECTED_PARAM, (
                f"{name} must take `{INJECTED_PARAM}` as its LAST positional "
                f"parameter (got {params}), matching the exemplar signature "
                "_construct_from_subflow(subflow, name, from_spec). A different "
                "position or spelling is a second convention."
            )

    def test_the_pure_functions_gain_no_parameter(self):
        """A mechanical move must not widen signatures it does not need to.
        Threading the callable into a pure helper would make the injection look
        load-bearing where it is not."""
        defined = _defined_functions(SWARM_PATH)
        for name in sorted(SWARM_CLUSTER - THREADING_FUNCTIONS):
            fn = defined.get(name)
            assert fn is not None, f"{name} is not defined in {SWARM_MODULE}"
            params = [a.arg for a in fn.args.args]
            assert INJECTED_PARAM not in params, f"{name} is pure and must NOT take `{INJECTED_PARAM}`; got {params}."

    def test_loader_passes_the_callable_at_both_dispatch_points(self):
        entry = _defined_functions(LOADER_PATH).get("from_agent_spec")
        assert entry is not None, "loader.from_agent_spec disappeared"

        passed: dict[str, bool] = {}
        for node in ast.walk(entry):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DISPATCH_CALLEES:
                args = [a.id for a in node.args if isinstance(a, ast.Name)]
                passed[node.func.id] = "from_agent_spec" in args

        assert set(passed) == DISPATCH_CALLEES, (
            f"from_agent_spec must still dispatch to BOTH swarm reconstructors; found calls to {sorted(passed)}."
        )
        not_injected = sorted(name for name, ok in passed.items() if not ok)
        assert not not_injected, (
            f"{not_injected} are called without passing `from_agent_spec` as the "
            "injected callable. Passing it as an ARGUMENT is the sanctioned "
            "pattern (and is explicitly not a recursion call site under "
            "test_guards_agent_spec_import_seam's detector)."
        )


class TestReExportSurfaceSurvives:
    """Nothing importing these names from ``neograph.loader`` breaks."""

    def test_the_six_names_are_still_importable_from_loader(self):
        import neograph.loader as loader_mod

        missing = sorted(n for n in SWARM_CLUSTER if not hasattr(loader_mod, n))
        assert not missing, (
            f"these names vanished from neograph.loader: {missing}. Add the F401 "
            "re-export block (sibling of the neograph-s7zt3.11 block) -- without "
            "`# noqa: F401` ruff --fix strips the re-exports and the surface "
            "silently shrinks."
        )


class TestGuardSurfaceFollowsTheCode:
    """The half that stays green if skipped. Read the diff, not the gate."""

    def test_import_seam_guard_scans_the_new_module(self):
        names = {p.name for p in IMPORT_PATH_FILES}
        assert SWARM_MODULE in names, (
            f"test_guards_agent_spec_import_seam.IMPORT_PATH_FILES does not "
            f"include {SWARM_MODULE} (has {sorted(names)}). Its anti-vacuity "
            "test only checks that the LISTED files exist, so it CANNOT detect "
            "this omission -- a from_agent_spec recursion re-inlined inside the "
            "swarm module would go undetected."
        )

    def test_marker_literal_guard_scans_the_new_module_without_dropping_loader(self):
        names = {p.name for p in SCANNED}
        assert SWARM_MODULE in names, (
            f"test_guards_agent_spec_markers.SCANNED does not include "
            f"{SWARM_MODULE}, yet the moved cluster is a marker CONSUMER "
            "(_MARK_PORTAL_SPEC, _MARK_PROMPT_SPEC, _MARK_MODIFIER, "
            '_MARK_PORTAL_OPERATOR_SPEC). A re-inlined "neograph/..." literal '
            "there would not be caught."
        )
        assert "loader.py" in names, (
            "SCANNED must be EXTENDED, never repointed -- dropping loader.py "
            "would stop catching a literal re-inlined at the old site, the "
            "shrinking-surface failure its own header comment warns about."
        )

    def test_branch_arm_walk_allowlist_is_rekeyed_not_duplicated(self):
        loader_keys = sorted(snippet for (file, snippet) in BRANCH_ARM_ALLOWLIST if file == "loader.py")
        swarm_keys = sorted(snippet for (file, snippet) in BRANCH_ARM_ALLOWLIST if file == SWARM_MODULE)

        assert not loader_keys, (
            "test_guards_branch_arm_walks._ALLOWLIST still grants loader.py "
            f"permissions for walks that moved: {loader_keys}. Follow that "
            "file's own precedent comment -- the old key is REMOVED, not kept "
            "alongside, or it grants a permission to a file that no longer does "
            "the thing."
        )
        assert len(swarm_keys) == 3, (
            f"expected the three moved swarm walks re-keyed to {SWARM_MODULE}; "
            f"found {len(swarm_keys)}: {swarm_keys}. The snippets are "
            "formatter-sensitive -- copy the POST-`ruff format` text."
        )

    def test_portal_member_class_consumer_literals_are_rekeyed(self):
        assert "loader.py" not in EXEMPT_FILES, (
            "test_guards_portal_member_class_consumers.EXEMPT_FILES still "
            'exempts loader.py, but the `type(agent).__name__ == "Flow"` '
            "derivation it documents moved into the swarm module. This guard "
            "stays GREEN either way (the two literals cancel in its expected "
            "computation), so it must be verified by reading the diff."
        )
        assert "loader.py" not in NO_DISCRIMINATOR_ATTR_SITES, (
            "NO_DISCRIMINATOR_ATTR_SITES still names loader.py; it must be removed."
        )

        # SUPERSEDED by neograph-dgbqv.5 (P10): this ticket's own
        # EXEMPT_FILES/NO_DISCRIMINATOR_ATTR_SITES re-key at jtawq.10-landing-time
        # was always meant to be an INTERIM state -- the "foreign pyagentspec
        # object, no .modifier_set" exemption reason string named dgbqv.5 as its
        # own retirement condition. dgbqv.5 has now landed: the swarm module reads
        # SWARM_ENCODING[PortalMemberClass.SUB_CONSTRUCT].spec_class instead of a
        # hard-coded "Flow" literal, so it is no longer exempt at all -- it is a
        # real classifier consumer and belongs in MIGRATED, not EXEMPT_FILES.
        assert SWARM_MODULE not in EXEMPT_FILES, (
            f"{SWARM_MODULE} is no longer exempt post-neograph-dgbqv.5 -- it directly imports "
            "and uses PortalMemberClass now, so it must be MIGRATED, not EXEMPT_FILES."
        )
        assert SWARM_MODULE not in NO_DISCRIMINATOR_ATTR_SITES, (
            f"{SWARM_MODULE} now carries a real classifier import post-neograph-dgbqv.5, so it "
            "must be removed from NO_DISCRIMINATOR_ATTR_SITES, not kept there."
        )
        assert SWARM_MODULE in PORTAL_MEMBER_CLASS_MIGRATED, (
            f"{SWARM_MODULE} must be declared in MIGRATED post-neograph-dgbqv.5."
        )

    def test_no_test_still_points_at_the_old_line_reference(self):
        """``tests/test_portal_member_class.py`` names ``loader.py:363`` in an
        assertion message for a site that is no longer in loader.py at all."""
        stale = sorted(
            p.name
            for p in (_ROOT / "tests").glob("test_*.py")
            if p.name != pathlib.Path(__file__).name and "loader.py:363" in p.read_text(encoding="utf-8")
        )
        assert not stale, (
            f"these tests still cite `loader.py:363` for a site that moved: "
            f"{stale}. Point the message at {SWARM_MODULE}."
        )

    def test_file_size_bookkeeping(self):
        assert "loader.py" not in FILE_SIZE_ALLOWLIST, (
            "loader.py drops under 500 lines after the move, so its "
            "test_guards_file_size.ALLOWLIST entry must be DELETED entirely -- "
            "never lowered to a sub-500 number, which would quietly grant a "
            "private ceiling to a file the plain 500 rule should govern "
            "(that guard's ANTI-DEAD-ENTRY rule)."
        )
        assert SWARM_MODULE not in FILE_SIZE_ALLOWLIST, (
            f"{SWARM_MODULE} must land under the plain 500-line cap and get NO "
            "allowlist entry. A mechanical move that needs a fresh private "
            "ceiling is not a mechanical move."
        )


class TestDetectorSlips:
    """Slip meta-tests (PROC-2): the detectors' boundaries, pinned."""

    def test_slip_import_detector_catches_a_deferred_back_import(self):
        """POSITIVE: the back-edge is usually re-introduced as a function-local
        import, which a module-level-body-only scan would miss."""
        assert "neograph.loader" in _imported_modules(
            "def f():\n    from neograph.loader import from_agent_spec\n    return from_agent_spec\n"
        )
        assert "neograph.loader" in _imported_modules("import neograph.loader\n")

    def test_slip_import_detector_ignores_sibling_modules(self):
        """NEGATIVE: the cluster's legitimate one-layer-down imports are not
        back-edges. A detector that flagged them would make the move impossible."""
        found = _imported_modules(
            "from neograph._agent_spec_group_import import _construct_from_subflow\n"
            "from neograph.construct import Construct\n"
        )
        assert not {m for m in found if m.split(".")[-1] == "loader"}
