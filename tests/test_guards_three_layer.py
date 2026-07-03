"""Structural guard locking the three-layer principle's engine-surface boundary.

Cites ``docs/design/three-layer-principle-2026-07-03.md`` sections 1 and 3.5.

## The invariant (three-layer §1, §3.5)

neograph's value comes from exactly two places — compile-time topology emission
(Layer 1) and a self-contained node-runtime helper library (Layer 2). The engine
surface (Layer 3: scheduling, streaming, checkpointing, interrupts) belongs to
LangGraph, untouched. neograph's entry points are THIN VERBS:
``prepare()`` -> engine verb -> ``finalize()``.

Therefore an *engine execution verb on a compiled graph* may appear ONLY in:

  (a) ``_compiled.py`` — the typed facade's EXPLICIT delegations (§2.2), and
  (b) ``runner.py``    — the top-level ``run``/``arun``/``stream``/``astream`` verbs.

The one allowlisted exception (§1.3) is ``_subconstruct.py``'s wrapper-invoke:
sub-constructs have their OWN schema, and LangGraph's documented rule is
"different schemas -> invoke inside a wrapper function that transforms state".

An engine verb is NOT the same as a Layer-2 node-internal LLM/tool call. A node
body that does ``llm.invoke(...)`` / ``raw_fn.ainvoke(...)`` / ``tool_fn.invoke(...)``
is cognition, not scheduling — those are LangChain ``Runnable`` calls on a
``BaseChatModel``/``BaseTool``, and the audit (§2.1) classified every one of them
(``_llm_*``, ``_tool_loop``, ``_oracle``, ``node.run_isolated``) as Layer 2. The
guard MUST discriminate by RECEIVER so it does not drown in — or wrongly forbid —
those calls.

## How the discrimination works (two verb classes)

1. **Graph-only verbs** (``GRAPH_ONLY_VERBS``): ``get_state``/``update_state``/…
   No LangChain ``Runnable``/``BaseChatModel``/``BaseTool`` exposes these; they
   belong to ``CompiledStateGraph`` and its checkpointer alone. So ANY call with
   one of these attribute names is an engine verb — no receiver check needed.

2. **Runnable-shared verbs** (``RUNNABLE_SHARED_VERBS``): ``invoke``/``ainvoke``/
   ``stream``/``astream``/``astream_events``/``get_graph``/…  A compiled graph
   exposes these, but so does every ``BaseChatModel``/``BaseTool`` (all are
   ``Runnable``s). A call is an engine verb ONLY when the receiver is a
   compiled-graph object. We recognize that structurally: the receiver's dotted
   chain contains a ``graph`` token (``graph``, ``sub_graph``, ``self.graph``).
   Every Layer-2 receiver in the tree — ``llm``, ``raw_fn``, ``tool_fn``,
   ``wrapped``, ``structured``, ``llm_with_tools``, ``node_fn``, ``self._bound``,
   ``self._inner``, ``build_default_adapter()`` — has NO ``graph`` token, so the
   guard excludes them by construction.

Also guarded: the IMPORT DIRECTION (§2.5.3). The run layer (``runner.py``) must
not be imported by any compile-layer module — the ``_subconstruct -> runner``
inversion (function-local ``_strip_internals`` import) that relocation to
``_state_keys`` removed must stay removed.

This is a LOCK over already-correct source: the guards PASS on today's tree
(audit §2.1: engine touchpoints ~95% clean, the confined verbs already sit in
the two sanctioned places). Non-vacuity is proven by (i) the EXPECTED-surface
exact-match test — an empty scan or a drifted detector fails it — and (ii) the
synthetic slip meta-tests, which feed good/bad AST through the same detectors.

PURE AST — this module imports no ``re`` and defines no regex constant, so it is
automatically exempt from ``test_guards_meta.py``'s regex-slip meta-guard.
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# ── Verb classes ─────────────────────────────────────────────────────────────

# Graph-only: exposed by CompiledStateGraph / its checkpointer, NOT by any
# LangChain Runnable/BaseChatModel/BaseTool. A call with one of these attribute
# names is an engine verb regardless of receiver.
GRAPH_ONLY_VERBS = frozenset(
    {
        "get_state",
        "aget_state",
        "get_state_history",
        "aget_state_history",
        "update_state",
        "aupdate_state",
        "bulk_update_state",
        "get_subgraphs",
    }
)

# Runnable-shared: a compiled graph exposes these AND so does every BaseChatModel
# / BaseTool (all Runnables). Only an engine verb when the receiver is a
# compiled-graph object (see _receiver_is_graphlike).
RUNNABLE_SHARED_VERBS = frozenset(
    {
        "invoke",
        "ainvoke",
        "stream",
        "astream",
        "astream_events",
        "astream_log",
        "batch",
        "abatch",
        "get_graph",
    }
)

# ── Module allowlists (with justification, three-layer §§1, 1.3, 2.2) ─────────

# Graph-only verbs may live only in the facade + the runner verbs.
ALLOWED_GRAPH_ONLY_MODULES: dict[str, str] = {
    "_compiled.py": (
        "the typed facade's EXPLICIT delegations to the wrapped LangGraph graph "
        "(three-layer §2.2 — a closed allowlist, not a transparent proxy)"
    ),
    "runner.py": (
        "the top-level run/arun/stream/astream verbs — thin prepare->engine->"
        "finalize entry points (three-layer §1 Layer 3)"
    ),
}

# Runnable-shared verbs on a compiled-graph receiver: the two above PLUS the one
# documented sub-construct exception.
ALLOWED_SHARED_MODULES: dict[str, str] = {
    **ALLOWED_GRAPH_ONLY_MODULES,
    "_subconstruct.py": (
        "the documented LangGraph different-schema subgraph wrapper-invoke "
        "(three-layer §1.3) — sub-constructs are isolated-by-type, forcing the "
        "function-form invoke inside make_subgraph_fn (dual-path, neograph-expi)"
    ),
}

# ── EXPECTED engine surface (anti-vacuity + friction) ─────────────────────────
# The EXACT set of (module, verb, kind) engine touchpoints on today's source.
# Mirrors test_guards_async_dispatch.EXPECTED_DISPATCH_CLASSES: an empty scan (a
# renamed verb making the confinement loop pass on zero hits) fails this, and a
# NEW engine call site anywhere forces a conscious edit here — desirable friction
# for a change that touches the engine boundary.
EXPECTED_ENGINE_SURFACE = frozenset(
    {
        # runner.py — top-level verbs
        ("runner.py", "get_state_history", "graph_only"),
        ("runner.py", "aget_state_history", "graph_only"),
        ("runner.py", "invoke", "shared"),
        ("runner.py", "ainvoke", "shared"),
        ("runner.py", "stream", "shared"),
        ("runner.py", "astream", "shared"),
        # _compiled.py — facade delegations
        ("_compiled.py", "get_state", "graph_only"),
        ("_compiled.py", "get_state_history", "graph_only"),
        ("_compiled.py", "update_state", "graph_only"),
        ("_compiled.py", "aget_state", "graph_only"),
        ("_compiled.py", "aget_state_history", "graph_only"),
        ("_compiled.py", "aupdate_state", "graph_only"),
        ("_compiled.py", "invoke", "shared"),
        ("_compiled.py", "ainvoke", "shared"),
        ("_compiled.py", "stream", "shared"),
        ("_compiled.py", "astream", "shared"),
        ("_compiled.py", "astream_events", "shared"),
        ("_compiled.py", "get_graph", "shared"),
        # _subconstruct.py — the allowlisted wrapper-invoke exception (§1.3)
        ("_subconstruct.py", "invoke", "shared"),
        ("_subconstruct.py", "ainvoke", "shared"),
    }
)

# Compile-layer modules must not import the run layer. Only the package facade
# (__init__.py, the run layer's own public export point) and runner.py itself are
# exempt from the "no neograph.runner import" rule.
IMPORT_DIRECTION_EXEMPT = frozenset({"runner.py", "__init__.py"})


# ── Detectors (operate on source text so meta-tests can feed synthetics) ──────


def _receiver_tokens(expr: ast.expr) -> set[str]:
    """Collect the identifier tokens in a receiver's dotted/call chain.

    ``graph`` -> {'graph'}; ``self.graph`` -> {'self', 'graph'};
    ``build_default_adapter()`` -> {'build_default_adapter'};
    ``self._bound`` -> {'self', '_bound'}.
    """
    tokens: set[str] = set()
    node: ast.expr | None = expr
    while node is not None:
        if isinstance(node, ast.Attribute):
            tokens.add(node.attr)
            node = node.value
        elif isinstance(node, ast.Name):
            tokens.add(node.id)
            node = None
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Subscript):
            node = node.value
        else:
            node = None
    return tokens


def _receiver_is_graphlike(expr: ast.expr) -> bool:
    """True if the receiver names a compiled graph — any token contains ``graph``.

    Matches ``graph`` / ``sub_graph`` / ``self.graph`` / ``compiled_graph``.
    Excludes every Layer-2 receiver (``llm``/``raw_fn``/``tool_fn``/``wrapped``/
    ``structured``/``llm_with_tools``/``node_fn``/``self._bound``/``self._inner``/
    ``build_default_adapter()``), none of which carries a ``graph`` token.
    """
    return any("graph" in t.lower() for t in _receiver_tokens(expr))


def _scan_engine_verbs(source: str) -> list[tuple[str, str]]:
    """Return ``[(verb, kind), ...]`` engine-verb calls in ``source``.

    ``kind`` is ``"graph_only"`` (unambiguous, any receiver) or ``"shared"``
    (Runnable-shared verb whose receiver is compiled-graph-like). Layer-2 LLM/tool
    calls (shared verb on a non-graph receiver) are NOT returned.
    """
    hits: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        verb = node.func.attr
        if verb in GRAPH_ONLY_VERBS:
            hits.append((verb, "graph_only"))
        elif verb in RUNNABLE_SHARED_VERBS and _receiver_is_graphlike(node.func.value):
            hits.append((verb, "shared"))
    return hits


def _imports_runner(source: str) -> bool:
    """True if ``source`` imports ``neograph.runner`` in any spelling.

    Catches ``from neograph.runner import x``, ``from .runner import x``,
    ``from neograph import runner``, ``from . import runner``, and
    ``import neograph.runner``.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "neograph.runner" or mod.endswith(".runner"):
                return True
            if node.level and mod == "runner":
                return True
            # `from neograph import runner` / `from . import runner`
            if (mod == "neograph" or (node.level and mod == "")) and any(a.name == "runner" for a in node.names):
                return True
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "neograph.runner" or a.name.endswith(".runner"):
                    return True
    return False


def _actual_engine_surface() -> set[tuple[str, str, str]]:
    """Scan every ``src/neograph/*.py`` -> set of ``(filename, verb, kind)``."""
    surface: set[tuple[str, str, str]] = set()
    for path in sorted(SRC.glob("*.py")):
        for verb, kind in _scan_engine_verbs(path.read_text()):
            surface.add((path.name, verb, kind))
    return surface


# ═════════════════════════════════════════════════════════════════════════════
# GUARD A — engine execution verbs are confined to the sanctioned modules
# ═════════════════════════════════════════════════════════════════════════════
class TestEngineVerbsConfinedToEngineSurface:
    """Engine execution verbs on a compiled graph live ONLY in ``_compiled.py``
    delegations, ``runner.py`` verbs, and (shared-verb only) the ``_subconstruct.py``
    wrapper-invoke exception (three-layer §1, §1.3, §3.5)."""

    def test_graph_only_verbs_are_confined(self) -> None:
        """``get_state``/``update_state``/… (no Runnable has them) appear only in
        the facade + runner."""
        problems: list[str] = []
        for path in sorted(SRC.glob("*.py")):
            for verb, kind in _scan_engine_verbs(path.read_text()):
                if kind == "graph_only" and path.name not in ALLOWED_GRAPH_ONLY_MODULES:
                    problems.append(f"{path.name}: graph-only engine verb `.{verb}(...)`")
        assert not problems, (
            "graph-only engine verbs escaped the engine surface (three-layer §1):\n"
            + "".join(f"  {p}\n" for p in problems)
            + "\nThese verbs belong to CompiledStateGraph/checkpointer alone. Move "
            "the call behind a runner verb or a _compiled.py delegation."
        )

    def test_shared_engine_verbs_on_graph_receivers_are_confined(self) -> None:
        """``invoke``/``ainvoke``/``stream``/… on a compiled-graph receiver appear
        only in the facade + runner + the _subconstruct.py exception. Layer-2
        LLM/tool calls on non-graph receivers are excluded by construction."""
        problems: list[str] = []
        for path in sorted(SRC.glob("*.py")):
            for verb, kind in _scan_engine_verbs(path.read_text()):
                if kind == "shared" and path.name not in ALLOWED_SHARED_MODULES:
                    problems.append(f"{path.name}: engine verb `<graph>.{verb}(...)`")
        assert not problems, (
            "engine execution verbs on a compiled graph escaped the engine surface "
            "(three-layer §1, §3.5):\n"
            + "".join(f"  {p}\n" for p in problems)
            + "\nDrive the graph through a runner verb (run/arun/stream/astream) or "
            "a _compiled.py delegation. If this is a NEW sub-construct-style "
            "wrapper-invoke, justify it in ALLOWED_SHARED_MODULES (§1.3)."
        )

    def test_engine_surface_matches_expected_exactly(self) -> None:
        """The scanned engine surface equals ``EXPECTED_ENGINE_SURFACE`` exactly.

        Anti-vacuity: an empty scan (renamed verb, drifted receiver detector)
        fails here. Friction: a new engine call site anywhere forces a conscious
        edit to the expected set — a change touching the engine boundary must be
        acknowledged."""
        actual = _actual_engine_surface()
        missing = EXPECTED_ENGINE_SURFACE - actual
        extra = actual - EXPECTED_ENGINE_SURFACE
        assert not missing and not extra, (
            "engine surface drifted from EXPECTED_ENGINE_SURFACE.\n"
            f"  new/unexpected engine call sites: {sorted(extra)}\n"
            f"  expected-but-gone (detector drift?): {sorted(missing)}\n"
            "If you INTENTIONALLY added an engine touchpoint, add it to "
            "EXPECTED_ENGINE_SURFACE and (if in a new module) ALLOWED_*_MODULES "
            "with a written justification tied to three-layer §1/§1.3."
        )

    def test_expected_surface_only_references_allowed_modules(self) -> None:
        """Every module in EXPECTED_ENGINE_SURFACE is an allowlisted host — the
        expected set can't smuggle an unjustified module past GUARD A."""
        problems: list[str] = []
        for filename, verb, kind in EXPECTED_ENGINE_SURFACE:
            allowed = ALLOWED_GRAPH_ONLY_MODULES if kind == "graph_only" else ALLOWED_SHARED_MODULES
            if filename not in allowed:
                problems.append(f"{filename}: {kind} verb `{verb}` not in the {kind} allowlist")
        assert not problems, "EXPECTED_ENGINE_SURFACE references a non-allowlisted module:\n" + "".join(
            f"  {p}\n" for p in problems
        )

    # ── slip meta-tests: synthetic sources prove the detectors are not vacuous ──

    def test_meta_graph_only_verb_flagged_regardless_of_receiver(self) -> None:
        # get_state has no Runnable analog -> flagged even on a bare `x`.
        assert _scan_engine_verbs("x.get_state(cfg)\n") == [("get_state", "graph_only")]

    def test_meta_shared_verb_on_graph_receiver_flagged(self) -> None:
        assert _scan_engine_verbs("graph.invoke(i, config=c)\n") == [("invoke", "shared")]
        assert _scan_engine_verbs("sub_graph.ainvoke(i)\n") == [("ainvoke", "shared")]
        assert _scan_engine_verbs("self.graph.astream(i)\n") == [("astream", "shared")]

    def test_meta_shared_verb_on_llm_or_tool_receiver_not_flagged(self) -> None:
        # Layer-2 cognition: NOT engine verbs, must be excluded by receiver.
        assert _scan_engine_verbs("llm.invoke(messages, config=c)\n") == []
        assert _scan_engine_verbs("raw_fn.ainvoke(state, config)\n") == []
        assert _scan_engine_verbs("tool_fn.invoke(args)\n") == []
        assert _scan_engine_verbs("self._bound.invoke(messages)\n") == []
        assert _scan_engine_verbs("build_default_adapter().invoke(llm, m, msgs, c)\n") == []

    def test_meta_a_violation_in_a_bad_module_would_be_flagged(self) -> None:
        # If a compile-layer module gained `graph.invoke`, GUARD A's loop flags it
        # because the module is not in the shared allowlist.
        hits = _scan_engine_verbs("def n():\n    return graph.invoke(x, config=c)\n")
        assert ("invoke", "shared") in hits
        assert "_wiring.py" not in ALLOWED_SHARED_MODULES  # a real compile-layer module


# ═════════════════════════════════════════════════════════════════════════════
# GUARD B — import direction: the compile layer must not import the run layer
# ═════════════════════════════════════════════════════════════════════════════
class TestRunLayerNotImportedByCompileLayer:
    """No compile-layer module imports ``neograph.runner`` (three-layer §2.5.3).

    The ``_subconstruct -> runner`` inversion (function-local ``_strip_internals``
    import, dodging the circular import it itself created) was removed by relocating
    ``_strip_internals`` to the neutral ``_state_keys`` module. This keeps it gone:
    only the package facade (``__init__.py``, which re-exports run/arun/stream/
    astream) and ``runner.py`` itself may reference the run layer."""

    def test_no_compile_layer_module_imports_runner(self) -> None:
        offenders: list[str] = []
        for path in sorted(SRC.glob("*.py")):
            if path.name in IMPORT_DIRECTION_EXEMPT:
                continue
            if _imports_runner(path.read_text()):
                offenders.append(path.name)
        assert not offenders, (
            "compile-layer module(s) import the run layer (three-layer §2.5.3 layer "
            "inversion):\n"
            + "".join(f"  {o} imports neograph.runner\n" for o in offenders)
            + "\nResult-shaping utilities the run layer shares belong in a neutral "
            "module (e.g. _state_keys), imported by both — not reached UP from a "
            "compile-layer module into runner.py."
        )

    # ── slip meta-tests ──

    def test_meta_detects_from_runner_import(self) -> None:
        assert _imports_runner("from neograph.runner import _strip_internals\n")
        assert _imports_runner("from .runner import _strip_internals\n")
        assert _imports_runner("from neograph import runner\n")
        assert _imports_runner("import neograph.runner\n")

    def test_meta_ignores_neutral_and_non_runner_imports(self) -> None:
        assert not _imports_runner("from neograph._state_keys import StateKeys, _strip_internals\n")
        assert not _imports_runner("from neograph.compiler import compile\n")
        # a comment/docstring mentioning the runner is not an import
        assert not _imports_runner('"""the runner drives this"""\nx = 1\n')


# ═══════════════════════════════════════════════════════════════════════════
# neograph-m6d3.5 — no while-True ReAct loop in a node body
#
# Agent/act cognition compiles to a LangGraph subgraph of supersteps
# (_agent_cycle via _wiring._add_agent_cycle), NOT a `while True` loop driving
# LLM + tool turns inside one node body. The deleted monolith
# (_tool_loop.invoke_with_tools) was exactly that anti-pattern; this guard bans
# its re-introduction. Locks the principle the way the H2 dual-path guard locks
# async: cognition lives at superstep boundaries the checkpointer can see, so a
# mid-loop interrupt pauses at a turn boundary (turn-boundary idempotency).
# ═══════════════════════════════════════════════════════════════════════════

# The modules that host agent-cognition node bodies. A ReAct loop would regress
# HERE. _llm_retry.py's bounded JSON-fix retry loop is a Layer-2 parse helper,
# not agent cognition, and is deliberately out of scope.
_AGENT_COGNITION_MODULES = ("_agent_cycle.py", "_tool_loop.py", "_dispatch.py")


def _while_loops_driving_llm(source: str) -> list[int]:
    """Line numbers of ``while`` loops whose body contains an ``.invoke`` /
    ``.ainvoke`` call — the monolithic ReAct-loop signature (LLM turns driven by
    a Python loop instead of LangGraph supersteps)."""
    tree = ast.parse(source)
    hits: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr in ("invoke", "ainvoke")
                ):
                    hits.append(node.lineno)
                    break
    return hits


class TestNoReActLoopInNodeBody:
    """m6d3.5: no while-loop drives LLM turns in an agent-cognition node body."""

    def test_no_while_loop_drives_llm_in_cognition_modules(self) -> None:
        problems: list[str] = []
        for name in _AGENT_COGNITION_MODULES:
            src = (SRC / name).read_text()
            for ln in _while_loops_driving_llm(src):
                problems.append(
                    f"{name}:{ln}: a `while` loop drives an LLM `.invoke`/`.ainvoke` — "
                    "the monolithic ReAct-loop anti-pattern deleted in m6d3.3"
                )
        assert not problems, (
            "ReAct-loop-in-a-node-body reintroduced (neograph-m6d3.5):\n"
            + "".join(f"  {p}\n" for p in problems)
            + "\nAgent/act cognition must compile to a subgraph of supersteps "
            "(_agent_cycle), not a Python while-True loop, so a mid-loop interrupt "
            "pauses at a turn boundary the checkpointer can see."
        )

    # ── slip meta-tests ──

    def test_meta_synthetic_react_loop_is_flagged(self) -> None:
        src = "def f(llm, msgs):\n    while True:\n        r = llm.invoke(msgs)\n"
        assert _while_loops_driving_llm(src) == [2]

    def test_meta_bounded_retry_loop_without_llm_not_flagged(self) -> None:
        src = "def f():\n    n = 0\n    while n < 3:\n        n += 1\n"
        assert _while_loops_driving_llm(src) == []
