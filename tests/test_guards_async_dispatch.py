"""Structural guards locking the dual-path (sync + async) dispatch invariant.

neograph-w74k.2.7 (Phase 1g, LAST child of the Phase 1 async-foundation epic).

## Core Invariant
Every LLM-mode dispatch class exposes BOTH a sync ``execute`` AND an async
``aexecute`` (with ``aexecute`` being ``async def``), and every async twin is
defined in the SAME module as its sync counterpart — so no future PR can
silently reintroduce a sync-only execution path (the H2 silent-failure mode).

This file is a LOCK over already-correct Phase 1a-1f code: the guards PASS on
today's real source. Non-vacuity is proven by the NEGATIVE meta-tests, which
feed synthetic AST (not the real source) through the same detectors and confirm
they FLAG violations. Mirrors ``tests/test_guards_async_fakes.py`` (Phase 0):
pure-AST, module-level ``ast.parse`` of the target source, small collectors,
per-target asserts, and slip meta-tests with good/bad synthetic sources.

## GUARD A — dual-path dispatch completeness (src/neograph/_dispatch.py)
A "dispatch class" is detected structurally (NOT by ``*Dispatch`` suffix): any
top-level ClassDef whose body defines a method named ``execute``. This enrolls
the ``ModeDispatch`` Protocol + the 3 concrete classes and excludes
``NodeInput``/``NodeOutput`` (which expose only a ``value`` property). Each
dispatch class must ALSO define ``aexecute`` as ``ast.AsyncFunctionDef`` (blocks
a sync stub), and its ``execute`` must be a plain ``ast.FunctionDef``.

## GUARD B — async-twin CO-LOCATION ONLY
For a table of ``{module: [(sync, async), ...]}`` twins NOT covered by the
LLM-vertical allowlist, assert BOTH names are top-level defs (``FunctionDef``
OR ``AsyncFunctionDef``) in the SAME module's AST. This is CO-LOCATION ONLY:
we do NOT assert the twin is ``AsyncFunctionDef``, because ``factory``'s
``_make_araw_wrapper`` is a SYNC wrapper-factory that returns an ``async def``
closure (its top-level node is ``ast.FunctionDef``). Async-ness of the dispatch
path is GUARD A's job; GUARD B only enforces that a twin cannot migrate to an
async-only module.

The 4 LLM-vertical twins (``ainvoke_structured`` / ``_acall_structured`` /
``_ainvoke_json_with_retry`` / ``arecover_dsml``) are already module-pinned by
``tests/test_guards_llm_runtime.py`` ``TestLlmResponsibilityDiscipline``
``ALLOWED_NAMES`` (which fails if a twin leaves its ``_llm*`` module), so GUARD B
EXCLUDES them — no duplication, one place to update on a rename.

No async-only-module import-DAG guard is added: Phase 1a-1f created ZERO
async-only modules (every twin sits in its sync module), so there is nothing to
police, and the existing sidecar/assembly import guards remain sufficient.

PURE AST — this file imports no ``re`` and defines no regex constant, so it is
automatically exempt from ``test_guards_meta.py``'s regex-slip meta-guard.
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"
DISPATCH_PATH = SRC / "_dispatch.py"

# The exact dispatch-class set expected on today's source. Catches detector
# drift (empty detection passes the per-class loop vacuously) AND a future
# 4th class sprouting `execute` (desirable friction — a new dispatch mode
# must consciously acknowledge the dual-path invariant).
EXPECTED_DISPATCH_CLASSES = frozenset(
    {"ModeDispatch", "ScriptedDispatch", "ThinkDispatch", "ToolDispatch"}
)

FuncDef = ast.FunctionDef | ast.AsyncFunctionDef


def _class_methods(source: str) -> dict[str, dict[str, FuncDef]]:
    """Map every top-level ClassDef -> {method_name: def_node} from source."""
    tree = ast.parse(source)
    out: dict[str, dict[str, FuncDef]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods: dict[str, FuncDef] = {}
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    methods[item.name] = item
            out[node.name] = methods
    return out


def _dispatch_classes(source: str) -> dict[str, dict[str, FuncDef]]:
    """Detect dispatch classes structurally: any top-level ClassDef whose body
    defines an ``execute`` method. Excludes NodeInput/NodeOutput (value only)."""
    return {
        name: methods
        for name, methods in _class_methods(source).items()
        if "execute" in methods
    }


def _top_level_defs(source: str) -> dict[str, FuncDef]:
    """Map every top-level (module-scope) def -> def_node."""
    tree = ast.parse(source)
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


class TestDualPathDispatchCompleteness:
    """GUARD A: every dispatch class in _dispatch.py exposes sync execute AND
    async aexecute — the structural counter to the H2 sync-only hazard."""

    def test_detected_dispatch_classes_match_expected_set(self):
        """The structural detector yields EXACTLY the 4 known dispatch classes.

        This anti-vacuity check catches an empty detection (a rename of
        ``execute`` would make the per-class loop pass on zero classes) AND a
        non-dispatch class wrongly sprouting an ``execute`` method."""
        detected = frozenset(_dispatch_classes(DISPATCH_PATH.read_text()))
        assert detected == EXPECTED_DISPATCH_CLASSES, (
            "dispatch-class detection drifted from the known set.\n"
            f"  detected: {sorted(detected)}\n"
            f"  expected: {sorted(EXPECTED_DISPATCH_CLASSES)}\n"
            "A dispatch class = top-level ClassDef with an `execute` method. "
            "If a new dispatch mode was added, extend EXPECTED_DISPATCH_CLASSES "
            "AND ensure it defines both `execute` and `async def aexecute`."
        )

    def test_every_dispatch_class_has_async_aexecute(self):
        """Each dispatch class defines ``aexecute`` as ``async def`` (a sync
        ``def aexecute`` stub is rejected) and ``execute`` as a plain ``def``."""
        classes = _dispatch_classes(DISPATCH_PATH.read_text())
        problems: list[str] = []
        for name in sorted(classes):
            methods = classes[name]
            execute = methods.get("execute")
            aexecute = methods.get("aexecute")

            if aexecute is None:
                problems.append(f"{name}: missing `aexecute` (sync-only dispatch path)")
            elif not isinstance(aexecute, ast.AsyncFunctionDef):
                problems.append(
                    f"{name}.aexecute must be `async def` (found sync `def` stub — "
                    "reintroduces a blocking path under the async driver)"
                )

            if execute is not None and not isinstance(execute, ast.FunctionDef):
                problems.append(
                    f"{name}.execute must be a plain `def` (found `async def` — "
                    "the sync path must stay sync)"
                )

        assert not problems, (
            "dual-path dispatch invariant violated in _dispatch.py:\n"
            + "".join(f"  {p}\n" for p in problems)
            + "\nCore Invariant (neograph-w74k.2.7): every dispatch mode exposes "
            "BOTH a sync `execute` and an `async def aexecute` — no sync-only "
            "execution path (the H2 silent-failure mode)."
        )

    # ── slip meta-tests (mirror test_delegation_detector_is_not_vacuous) ──
    #
    # Feed SYNTHETIC AST through the detector + rules to prove the guards
    # above cannot pass vacuously. POSITIVE enrolls + passes; NEGATIVE-1/2/3
    # must be flagged; EXCLUSION must not enroll.

    def test_meta_positive_class_with_execute_and_async_aexecute_passes(self):
        src = (
            "class X:\n"
            "    def execute(self): ...\n"
            "    async def aexecute(self): ...\n"
        )
        classes = _dispatch_classes(src)
        assert "X" in classes  # enrolled by execute presence
        methods = classes["X"]
        assert isinstance(methods["execute"], ast.FunctionDef)
        assert isinstance(methods["aexecute"], ast.AsyncFunctionDef)

    def test_meta_negative_execute_without_aexecute_is_flagged(self):
        src = "class X:\n    def execute(self): ...\n"
        classes = _dispatch_classes(src)
        assert "X" in classes  # enrolled...
        assert "aexecute" not in classes["X"]  # ...but the async twin is absent

    def test_meta_negative_sync_aexecute_stub_is_flagged(self):
        src = (
            "class X:\n"
            "    def execute(self): ...\n"
            "    def aexecute(self): ...\n"
        )
        classes = _dispatch_classes(src)
        aexecute = classes["X"]["aexecute"]
        # present, but NOT AsyncFunctionDef -> guard's async-ness check flags it
        assert not isinstance(aexecute, ast.AsyncFunctionDef)

    def test_meta_negative_async_execute_is_flagged(self):
        src = (
            "class X:\n"
            "    async def execute(self): ...\n"
            "    async def aexecute(self): ...\n"
        )
        classes = _dispatch_classes(src)
        execute = classes["X"]["execute"]
        # `execute` must be a plain def; an async execute is flagged
        assert not isinstance(execute, ast.FunctionDef)

    def test_meta_exclusion_value_property_class_not_enrolled(self):
        src = (
            "class X:\n"
            "    @property\n"
            "    def value(self): ...\n"
        )
        classes = _dispatch_classes(src)
        assert "X" not in classes  # no execute -> not a dispatch class


class TestAsyncTwinCoLocation:
    """GUARD B: async twins live in the SAME module as their sync counterpart
    (co-location only — no AsyncFunctionDef assertion on the twin).

    Excludes the 4 LLM-vertical twins already module-pinned by
    ``tests/test_guards_llm_runtime.py`` ``ALLOWED_NAMES``."""

    # {module_filename: [(sync_name, async_name), ...]}
    TWIN_TABLE: dict[str, list[tuple[str, str]]] = {
        "_execute.py": [("_execute_node", "_aexecute_node")],
        "_tool_loop.py": [("invoke_with_tools", "ainvoke_with_tools")],
        "runner.py": [
            ("run", "arun"),
            ("_has_existing_checkpoint", "_ahas_existing_checkpoint"),
            ("_verify_checkpoint_schema", "_averify_checkpoint_schema"),
            ("_auto_resume_from_divergence", "_aauto_resume_from_divergence"),
        ],
        "factory.py": [("_make_raw_wrapper", "_make_araw_wrapper")],
    }

    def test_async_twins_are_co_located_with_sync_counterpart(self):
        """Both names in each pair are top-level defs in the SAME module.

        Co-location ONLY: the async twin may be `def` (e.g.
        ``factory._make_araw_wrapper`` is a sync factory returning an async
        closure) or `async def`. A twin migrating to an async-only module
        fails this."""
        problems: list[str] = []
        for filename, pairs in self.TWIN_TABLE.items():
            defs = _top_level_defs((SRC / filename).read_text())
            for sync_name, async_name in pairs:
                if sync_name not in defs:
                    problems.append(f"{filename}: missing sync def `{sync_name}`")
                if async_name not in defs:
                    problems.append(
                        f"{filename}: missing async twin `{async_name}` "
                        "(moved out of its sync counterpart's module?)"
                    )
        assert not problems, (
            "async-twin co-location violated:\n"
            + "".join(f"  {p}\n" for p in problems)
            + "\nGUARD B (neograph-w74k.2.7): each async twin must be a top-level "
            "def in the SAME module as its sync counterpart — no async-only "
            "module for a dispatch twin."
        )

    # ── slip meta-tests: synthetic modules prove co-location is enforced ──

    def test_meta_positive_both_defs_in_one_module_pass(self):
        src = (
            "def _work(): ...\n"
            "async def _awork(): ...\n"
        )
        defs = _top_level_defs(src)
        assert "_work" in defs and "_awork" in defs

    def test_meta_negative_missing_async_twin_is_flagged(self):
        src = "def _work(): ...\n"
        defs = _top_level_defs(src)
        assert "_work" in defs
        assert "_awork" not in defs  # missing twin -> guard flags it

    def test_meta_negative_twin_moved_to_second_module_is_flagged(self):
        module_a = "def _work(): ...\n"
        module_b = "async def _awork(): ...\n"
        defs_a = _top_level_defs(module_a)
        defs_b = _top_level_defs(module_b)
        # the twin is NOT co-located in module_a; guard scanning module_a flags it
        assert "_awork" not in defs_a
        assert "_awork" in defs_b  # it exists, but in the wrong module
