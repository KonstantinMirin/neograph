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


def _func_named(source: str, name: str) -> FuncDef | None:
    """Return the first def named ``name`` anywhere in ``source`` (top-level or
    nested), or None. Used to reach into a factory whose twins are closures."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            return node
    return None


def _returns_runnable_with_afunc(fn: FuncDef) -> bool:
    """True if ``fn`` has a ``return RunnableLambda(..., afunc=...)`` — i.e. it
    hands the DRIVER an async twin, not a sync-only closure."""
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "RunnableLambda"
            and any(kw.arg == "afunc" for kw in node.value.keywords)
        ):
            return True
    return False


def _calls_attr(fn: FuncDef, attr: str) -> bool:
    """True if ``fn`` (or a nested closure) calls ``something.<attr>(...)`` —
    e.g. ``sub_graph.ainvoke(...)``. Proves the async twin actually drives the
    child async instead of a band-aid afunc that re-calls the sync path."""
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr
        ):
            return True
    return False


def _calls_func(fn: FuncDef, name: str) -> bool:
    """True if ``fn`` (or a nested closure) calls a bare ``<name>(...)`` —
    e.g. ``ainvoke_structured(...)``. Distinguishes an async twin that awaits the
    async LLM seam from a band-aid that re-calls the sync ``invoke_structured``."""
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ):
            return True
    return False


def _wraps_with_afunc(source: str, wrapped: str) -> bool:
    """True if ``source`` has a ``RunnableLambda(<wrapped>, afunc=...)`` call —
    i.e. a closure ``wrapped`` handed to the graph with an async twin. Used for
    barriers added via ``graph.add_node(RunnableLambda(fn, afunc=afn), ...)``
    rather than returned from a factory."""
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "RunnableLambda"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == wrapped
            and any(kw.arg == "afunc" for kw in node.keywords)
        ):
            return True
    return False


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
            ("stream", "astream"),
            ("_prepare", "_aprepare"),
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


class TestSubgraphFnDualPath:
    """GUARD C (neograph-expi): ``make_subgraph_fn`` must be dual-path.

    A sub-construct node-callable factory whose twins are CLOSURES (not
    top-level defs, so GUARD B's co-location table can't reach them) is pinned
    here by two structural facts that together defeat the sync-only regression
    AND the band-aid variant:

    1. It returns ``RunnableLambda(..., afunc=...)`` — the DRIVER gets an async
       twin. A bare ``return subgraph_node`` or ``RunnableLambda(subgraph_node)``
       (no afunc) is threadpooled under ``ainvoke`` and runs the whole child on
       the sync path — the exact neograph-expi bug.
    2. Its body calls ``<x>.ainvoke(...)`` — the async twin actually drives the
       child graph asynchronously. This blocks the subtle band-aid: an afunc twin
       that still calls ``sub_graph.invoke`` (afunc present but async selection
       does NOT propagate into the child). This is GUARD C's analog of a
       regex-slip meta-case.

    Non-vacuity is proven by the synthetic negative meta-tests below.
    """

    SUBCONSTRUCT_PATH = SRC / "_subconstruct.py"

    def test_make_subgraph_fn_returns_runnable_with_afunc(self):
        fn = _func_named(self.SUBCONSTRUCT_PATH.read_text(), "make_subgraph_fn")
        assert fn is not None, "make_subgraph_fn not found in _subconstruct.py"
        assert _returns_runnable_with_afunc(fn), (
            "make_subgraph_fn must `return RunnableLambda(subgraph_node, "
            "afunc=asubgraph_node)` — a sync-only return threadpools the child "
            "under ainvoke and runs it on the sync path (neograph-expi)."
        )

    def test_make_subgraph_fn_drives_child_via_ainvoke(self):
        fn = _func_named(self.SUBCONSTRUCT_PATH.read_text(), "make_subgraph_fn")
        assert fn is not None
        assert _calls_attr(fn, "ainvoke"), (
            "make_subgraph_fn's async twin must call sub_graph.ainvoke — an afunc "
            "that still calls sub_graph.invoke does NOT propagate async selection "
            "into the child (band-aid, neograph-expi)."
        )

    # ── slip meta-tests: synthetic factories prove both facts are enforced ──

    def test_meta_positive_dualpath_factory_passes(self):
        src = (
            "def make_x(sub, sub_graph):\n"
            "    def node(s, c):\n"
            "        return sub_graph.invoke(s, config=c)\n"
            "    async def anode(s, c):\n"
            "        return await sub_graph.ainvoke(s, config=c)\n"
            "    return RunnableLambda(node, afunc=anode)\n"
        )
        fn = _func_named(src, "make_x")
        assert fn is not None
        assert _returns_runnable_with_afunc(fn)
        assert _calls_attr(fn, "ainvoke")

    def test_meta_negative_sync_only_return_is_flagged(self):
        src = (
            "def make_x(sub, sub_graph):\n"
            "    def node(s, c):\n"
            "        return sub_graph.invoke(s, config=c)\n"
            "    return node\n"
        )
        fn = _func_named(src, "make_x")
        assert fn is not None
        assert not _returns_runnable_with_afunc(fn)  # no afunc twin -> flagged

    def test_meta_negative_runnable_without_afunc_is_flagged(self):
        src = (
            "def make_x(sub, sub_graph):\n"
            "    def node(s, c):\n"
            "        return sub_graph.invoke(s, config=c)\n"
            "    return RunnableLambda(node)\n"
        )
        fn = _func_named(src, "make_x")
        assert fn is not None
        assert not _returns_runnable_with_afunc(fn)  # RunnableLambda(node) alone -> flagged

    def test_meta_negative_afunc_without_ainvoke_is_flagged(self):
        """Band-aid case: afunc present but the twin re-calls sync ``invoke``."""
        src = (
            "def make_x(sub, sub_graph):\n"
            "    def node(s, c):\n"
            "        return sub_graph.invoke(s, config=c)\n"
            "    async def anode(s, c):\n"
            "        return sub_graph.invoke(s, config=c)\n"
            "    return RunnableLambda(node, afunc=anode)\n"
        )
        fn = _func_named(src, "make_x")
        assert fn is not None
        assert _returns_runnable_with_afunc(fn)  # afunc is present...
        assert not _calls_attr(fn, "ainvoke")  # ...but never drives the child async -> flagged


class TestOracleMergeBarrierDualPath:
    """GUARD D (neograph-p3c7): the Oracle merge BARRIERS must be dual-path.

    The generator redirects were made dual-path in Phase 1a; the merge barriers
    (which invoke a sync LLM merge via ``invoke_structured``) were missed and ran
    threadpooled under ``graph.ainvoke``. Three structural facts pin the fix and
    defeat both the sync-only regression and the band-aid variant:

    1. ``make_oracle_merge_fn`` (single-group) returns ``RunnableLambda(..., afunc=...)``.
    2. ``group_merge_barrier`` (Each×Oracle fused) is handed to the graph wrapped
       as ``RunnableLambda(group_merge_barrier, afunc=...)`` — it is add_node'd,
       not returned, so it is pinned by the wrap-site, not a return.
    3. ``_arun_merge_prompt`` (the async merge_prompt twin) calls
       ``ainvoke_structured`` — an afunc that re-called the sync ``invoke_structured``
       would present an async twin that still blocks the loop (the band-aid).
    """

    ORACLE_PATH = SRC / "_oracle.py"
    WIRING_PATH = SRC / "_wiring.py"

    def test_make_oracle_merge_fn_returns_runnable_with_afunc(self):
        fn = _func_named(self.ORACLE_PATH.read_text(), "make_oracle_merge_fn")
        assert fn is not None, "make_oracle_merge_fn not found in _oracle.py"
        assert _returns_runnable_with_afunc(fn), (
            "make_oracle_merge_fn must return RunnableLambda(merge_fn, afunc=amerge_fn) "
            "— a sync-only merge barrier threadpools the LLM merge under ainvoke "
            "(neograph-p3c7)."
        )

    def test_group_merge_barrier_is_wrapped_with_afunc(self):
        assert _wraps_with_afunc(self.WIRING_PATH.read_text(), "group_merge_barrier"), (
            "group_merge_barrier must be add_node'd as RunnableLambda(group_merge_barrier, "
            "afunc=agroup_merge_barrier) — the Each×Oracle fused merge otherwise runs on "
            "the sync path under ainvoke (neograph-p3c7)."
        )

    def test_async_merge_prompt_twin_awaits_async_llm_seam(self):
        fn = _func_named(self.ORACLE_PATH.read_text(), "_arun_merge_prompt")
        assert fn is not None, "_arun_merge_prompt (async merge twin) not found"
        assert _calls_func(fn, "ainvoke_structured"), (
            "_arun_merge_prompt must call ainvoke_structured — an async merge twin "
            "that re-calls the sync invoke_structured still blocks the loop (band-aid, "
            "neograph-p3c7)."
        )

    # ── slip meta-tests: synthetic sources prove all three facts are enforced ──

    def test_meta_positive_wrapped_barrier_passes(self):
        src = "graph.add_node(n, RunnableLambda(group_merge_barrier, afunc=agroup_merge_barrier), defer=True)\n"
        assert _wraps_with_afunc(src, "group_merge_barrier")

    def test_meta_negative_unwrapped_barrier_is_flagged(self):
        src = "graph.add_node(n, group_merge_barrier, defer=True)\n"
        assert not _wraps_with_afunc(src, "group_merge_barrier")

    def test_meta_negative_wrapped_without_afunc_is_flagged(self):
        src = "graph.add_node(n, RunnableLambda(group_merge_barrier), defer=True)\n"
        assert not _wraps_with_afunc(src, "group_merge_barrier")

    def test_meta_negative_async_twin_calling_sync_seam_is_flagged(self):
        """Band-aid: an async merge twin that re-calls the sync seam."""
        good = "async def _arun(): return await ainvoke_structured(x)\n"
        bad = "async def _arun(): return invoke_structured(x)\n"
        assert _calls_func(_func_named(good, "_arun"), "ainvoke_structured")
        assert not _calls_func(_func_named(bad, "_arun"), "ainvoke_structured")
