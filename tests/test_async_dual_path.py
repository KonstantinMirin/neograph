"""Phase 1a IR-spine tests — driver-selected async over ONE shared core.

Pins neograph-w74k.2.1 (Phase 1a). The Core Invariant: ``make_node_fn`` returns
a single ``RunnableLambda(node_wrapper, afunc=anode_wrapper)`` so the DRIVER
selects the path — ``graph.invoke`` runs ``node_wrapper`` (sync), ``graph.ainvoke``
runs ``anode_wrapper`` (the async twin ``_aexecute_node`` -> ``dispatch.aexecute``).

Why the async cell is driven at ``compiled.graph.ainvoke`` and NOT ``arun``:
``arun()`` does not exist until Phase 1d, and the ``run_driver`` async cell in
``tests/conftest.py`` is skip-guarded on ``hasattr(neograph, "arun")``. So this
suite drives the graph layer directly (design NOTES on neograph-w74k.2.1); the
invoke==ainvoke assertion is a same-layer parity so any runner-layer behavior is
bypassed EQUALLY on both sides and the equality isolates func-vs-afunc dispatch.

Why result-equality alone is NOT the red assertion (critical): LangGraph today
threadpools a SYNC node under ``ainvoke``, so ``invoke``/``ainvoke`` already
return equal results before Phase 1a lands — a pure equality check is green now
and worthless as a TDD-red. The load-bearing red is a THREAD-IDENTITY
discriminator: under ``ainvoke`` the scripted node must run INLINE on the
event-loop thread (proving the RunnableLambda ``afunc`` async twin fired), not on
a threadpool worker (which is what the un-wired sync ``node_wrapper`` does today).
Empirically confirmed red now: node runs on a worker thread under ``ainvoke``.

Three-surface parity (AGENTS.md neograph-ts7 rule): @node decorator, declarative
``Node.scripted()``, and programmatic ``Node() | Modifier`` — this is the 6-cell
(3 surfaces x sync/async) mandated for an IR-level change.
"""

from __future__ import annotations

import asyncio
import inspect
import threading

import pytest

import neograph
from neograph import (
    Construct,
    Node,
    Oracle,
    compile,
    construct_from_functions,
    node,
)
from neograph.errors import NeographError
from neograph.factory import make_node_fn
from tests.fakes import build_test_compile_kwargs, register_scripted
from tests.schemas import Claims, RawText

_CFG = {"configurable": {}}
_INPUT = {"node_id": "t"}


def _ainvoke_field(graph, field: str):
    """Drive ONLY ``graph.ainvoke`` (async body cells must not touch the sync
    driver — sync-driver + async-body is the out-of-scope footgun neograph-khff)
    and return the named state field.
    """
    result = asyncio.run(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)))
    return result[field]


def _sync_async_node_thread(graph, field: str, node_thread: dict):
    """Run ``graph`` sync then async at the graph layer; return
    ``(sync_field, async_field, loop_thread_id, async_node_thread_id)``.

    ``node_thread`` is a holder the scripted body writes ``threading.get_ident()``
    into. It is cleared right before ``ainvoke`` so the captured id reflects the
    ASYNC run's execution thread (proving inline-on-loop vs threadpooled).
    """
    sync_result = graph.graph.invoke(dict(_INPUT), dict(_CFG))
    loop_thread = threading.get_ident()  # asyncio.run drives the loop on THIS thread
    node_thread.clear()
    async_result = asyncio.run(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)))
    return sync_result[field], async_result[field], loop_thread, node_thread.get("tid")


class TestScriptedDualPathParity:
    """6-cell: the async cell must route through the RunnableLambda ``afunc``
    (scripted node runs INLINE on the loop thread), not threadpool the sync
    ``node_wrapper``. RED now — ``afunc`` is not wired, so ``ainvoke``
    threadpools the sync wrapper and the node runs on a worker thread.
    """

    def test_dual_path_parity_when_declarative_scripted_surface(self):
        """Surface 1 — declarative ``Node.scripted()``."""
        node_thread: dict = {}

        def gen(input_data, config):
            node_thread["tid"] = threading.get_ident()
            return Claims(items=["x"])

        register_scripted("gen", gen)
        graph = compile(
            Construct("p", nodes=[Node.scripted("gen", fn="gen", outputs=Claims)]),
            **build_test_compile_kwargs(),
        )

        sync_v, async_v, loop_tid, node_tid = _sync_async_node_thread(graph, "gen", node_thread)

        assert sync_v == async_v  # same-layer parity (green pre- and post-1a)
        # RED: async twin must run the scripted node inline on the loop thread.
        assert node_tid == loop_tid, (
            "ainvoke threadpooled the sync node_wrapper instead of routing through "
            "the RunnableLambda afunc (async twin) — Phase 1a not implemented"
        )

    def test_dual_path_parity_when_node_decorator_surface(self):
        """Surface 2 — ``@node`` decorator (scripted mode, body runs)."""
        node_thread: dict = {}

        @node(outputs=Claims)
        def gen() -> Claims:
            node_thread["tid"] = threading.get_ident()
            return Claims(items=["x"])

        graph = compile(construct_from_functions("p", [gen]), **build_test_compile_kwargs())

        sync_v, async_v, loop_tid, node_tid = _sync_async_node_thread(graph, "gen", node_thread)

        assert sync_v == async_v
        assert node_tid == loop_tid, (
            "ainvoke threadpooled the sync node_wrapper instead of routing through "
            "the RunnableLambda afunc (async twin) — Phase 1a not implemented"
        )

    def test_dual_path_parity_when_programmatic_oracle_pipe_surface(self):
        """Surface 3 — programmatic ``Node() | Modifier`` (``Node.scripted | Oracle``).

        Doubles as the Oracle-Node async-parity cell: the Oracle router calls the
        gen node's factory return via the redirect fn (``_oracle.py:72/:99``),
        which Phase 1a migrates to ``raw_fn.invoke(state, config)``. Result parity
        of the merged output locks that migration; the inline-thread assertion
        locks the async routing through the afunc.
        """
        node_thread: dict = {}

        def gen(input_data, config):
            node_thread["tid"] = threading.get_ident()
            return Claims(items=["v"])

        def merge(variants, config):
            return Claims(items=[f"m{len(variants)}"])

        register_scripted("gen", gen)
        register_scripted("merge", merge)
        graph = compile(
            Construct(
                "p",
                nodes=[Node.scripted("gen", fn="gen", outputs=Claims) | Oracle(n=2, merge_fn="merge")],
            ),
            **build_test_compile_kwargs(),
        )

        sync_v, async_v, loop_tid, node_tid = _sync_async_node_thread(graph, "gen", node_thread)

        assert sync_v == async_v == Claims(items=["m2"])  # merged output parity (redirect .invoke)
        assert node_tid == loop_tid, (
            "Oracle gen node was threadpooled under ainvoke instead of running "
            "inline via the RunnableLambda afunc — Phase 1a not implemented"
        )


class TestSubConstructOracleWrapSurvivesMigration:
    """Locks the ``compiler.py:412`` ``RunnableLambda(subgraph_fn)`` wrap: a
    sub-construct piped with Oracle feeds ``subgraph_fn`` (a bare closure) to the
    shared ``make_oracle_redirect_fn``. When Phase 1a switches that redirect to
    ``.invoke()``, ``subgraph_fn`` must be RunnableLambda-wrapped or the subgraph
    source breaks. Regression LOCK — must run correctly on BOTH paths.
    """

    def test_subconstruct_oracle_runs_on_both_paths(self):
        register_scripted("sg", lambda input_data, config: Claims(items=["v"]))
        register_scripted("sm", lambda variants, config: Claims(items=[f"m{len(variants)}"]))

        sub = Construct(
            "osub",
            input=Claims,
            output=Claims,
            nodes=[Node.scripted("g", fn="sg", outputs=Claims)],
        ) | Oracle(n=2, merge_fn="sm")
        parent = Construct("parent", nodes=[sub])
        graph = compile(parent, **build_test_compile_kwargs())

        sync_result = graph.graph.invoke(dict(_INPUT), dict(_CFG))
        async_result = asyncio.run(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)))

        assert sync_result["osub"] == Claims(items=["m2"])
        assert sync_result["osub"] == async_result["osub"]


class TestSubConstructAsyncTwin:
    """neograph-expi — a sub-construct under the async driver must run its child
    via ``sub_graph.ainvoke``, not ``sub_graph.invoke``.

    ``make_subgraph_fn`` today (``_subconstruct.py:117``) has ONE sync path:
    ``subgraph_node`` calls ``sub_graph.invoke(...)`` unconditionally, and the
    compiler adds that bare closure to the parent graph (``compiler.py:430``) with
    no ``afunc`` twin. Under ``graph.ainvoke`` LangGraph threadpools the sync
    ``subgraph_node`` onto a worker thread, where ``sub_graph.invoke`` runs the
    ENTIRE child synchronously — blocking the loop and silently downgrading any
    async-only leaf inside the child to the sync path. This violates the Phase-1
    H2 invariant: async is DRIVER-selected and must propagate through every
    nesting level.

    Load-bearing RED (thread-identity, NOT parity): under ``ainvoke`` the child's
    leaf node must run INLINE on the event-loop thread (proving the parent
    subgraph node routed through a RunnableLambda ``afunc`` that awaited
    ``sub_graph.ainvoke``), not on a threadpool worker (today's sync path).
    Result parity is green pre-fix and worthless as the red — see the module
    docstring.

    Three-surface parity (AGENTS.md neograph-ts7): a sub-construct is authored
    declaratively (``Construct(input=, output=, nodes=[Node.scripted(...)])``) or
    from ``@node`` functions (``construct_from_functions(input=, output=)``); both
    lower to the same ``make_subgraph_fn`` path, so both must go green together.
    """

    def test_declarative_subconstruct_child_runs_inline_under_ainvoke(self):
        """Surface 1 — declarative ``Construct(input=, output=, nodes=[...])``."""
        node_thread: dict = {}

        def leaf(input_data, config):
            node_thread["tid"] = threading.get_ident()
            return Claims(items=[input_data.text.upper()])

        register_scripted("leaf", leaf)
        register_scripted("seed", lambda input_data, config: RawText(text="hi"))

        sub = Construct(
            "child",
            input=RawText,
            output=Claims,
            nodes=[Node.scripted("leaf", fn="leaf", inputs=RawText, outputs=Claims)],
        )
        parent = Construct(
            "parent",
            nodes=[Node.scripted("seed", fn="seed", outputs=RawText), sub],
        )
        graph = compile(parent, **build_test_compile_kwargs())

        sync_v, async_v, loop_tid, node_tid = _sync_async_node_thread(graph, "child", node_thread)

        assert sync_v == async_v == Claims(items=["HI"])  # same-layer parity
        assert node_tid == loop_tid, (
            "ainvoke threadpooled the sync subgraph_node instead of routing "
            "through a RunnableLambda afunc that awaits sub_graph.ainvoke — the "
            "child ran on the sync path (neograph-expi)"
        )

    def test_node_subconstruct_child_runs_inline_under_ainvoke(self):
        """Surface 2 — ``@node`` sub-construct via ``construct_from_functions``."""
        node_thread: dict = {}

        @node(outputs=Claims)
        def leaf(port: RawText) -> Claims:
            node_thread["tid"] = threading.get_ident()
            return Claims(items=[port.text.upper()])

        child = construct_from_functions("child", [leaf], input=RawText, output=Claims)
        register_scripted("seed", lambda input_data, config: RawText(text="hi"))
        parent = Construct(
            "parent",
            nodes=[Node.scripted("seed", fn="seed", outputs=RawText), child],
        )
        graph = compile(parent, **build_test_compile_kwargs())

        sync_v, async_v, loop_tid, node_tid = _sync_async_node_thread(graph, "child", node_thread)

        assert sync_v == async_v == Claims(items=["HI"])
        assert node_tid == loop_tid, (
            "ainvoke threadpooled the sync subgraph_node instead of routing "
            "through a RunnableLambda afunc that awaits sub_graph.ainvoke — the "
            "child ran on the sync path (neograph-expi)"
        )


class TestRunIsolatedSurvivesRunnableMigration:
    """Locks ``node.py:422`` ``Node.run_isolated`` (sync) against the make_node_fn
    return-type change. ``run_isolated`` calls the factory return directly; when
    it becomes a RunnableLambda, ``run_isolated`` must ``.invoke()`` it. Regression
    LOCK — the scripted node must still return its output.
    """

    def test_run_isolated_scripted_returns_output(self):
        register_scripted(
            "upper",
            lambda input_data, config: RawText(text=input_data.text.upper()),
        )
        upper = Node.scripted("upper", fn="upper", inputs=RawText, outputs=RawText)

        result = upper.run_isolated(**build_test_compile_kwargs(), input=RawText(text="hello"))

        assert isinstance(result, RawText)
        assert result.text == "HELLO"


class TestMakeNodeFnReturnsRunnable:
    """Structural/behavioral guard (for sweep): ``make_node_fn`` returns a
    Runnable (has ``.invoke``) on ALL paths — scripted AND raw — so every direct
    caller (``_oracle`` redirects, ``run_isolated``) can uniformly ``.invoke()``.
    Calls the factory (behavioral), not a source scan. RED now — both paths
    return a bare closure with no ``.invoke``.
    """

    def test_scripted_node_fn_is_runnable(self):
        register_scripted("g", lambda input_data, config: Claims(items=["x"]))
        scripted = Node.scripted("g", fn="g", outputs=Claims)

        fn = make_node_fn(scripted, scripted_lookup={"g": lambda i, c: Claims(items=["x"])})

        assert hasattr(fn, "invoke"), "make_node_fn(scripted) must return a Runnable"

    def test_raw_node_fn_is_runnable(self):
        @node(mode="raw", outputs=Claims)
        def rawn(state, config):
            return {"rawn": Claims(items=["r"])}

        fn = make_node_fn(rawn)

        assert hasattr(fn, "invoke"), "make_node_fn(raw) must return a Runnable"


class TestAsyncScriptedBody:
    """Phase 1b (neograph-w74k.2.2) — TDD RED.

    ``ScriptedDispatch.aexecute`` must AWAIT an ``async def`` scripted body when
    one is detected via ``inspect.isawaitable(result)`` AFTER calling ``self.fn``.
    Today (post-1a) ``aexecute`` calls ``self.fn(...)`` and returns the result
    WITHOUT awaiting, so an async body's result is an un-awaited COROUTINE written
    into state — ``result[field]`` is a coroutine, not the declared model.

    The load-bearing RED (identical across all three surfaces): without the
    conditional await, ``isinstance(result[field], Claims)`` is False and
    ``inspect.isawaitable(result[field])`` is True. All cells drive
    ``graph.ainvoke`` only (the sync driver + async body is out of 1b scope).

    Three-surface parity (AGENTS.md neograph-ts7 rule): the async body reaches
    ``self.fn`` differently per surface — behind a sync ``scripted_shim`` for
    ``@node`` (so ``iscoroutinefunction(self.fn)`` is BLIND to it), and directly
    for declarative/programmatic ``register_scripted`` fns — but the single
    call-then-``isawaitable`` check is uniform, so all three must go green
    together.
    """

    def test_node_decorator_async_scripted_body_is_awaited(self):
        """Surface 1 — ``@node`` async scripted body. The user body hides behind
        a sync shim; the shim returns an un-awaited coroutine, so aexecute must
        detect+await the awaitable result.
        """
        @node(outputs=RawText)
        def raw_src() -> RawText:
            return RawText(text="hi")

        @node(outputs=Claims)
        async def f(raw_src: RawText) -> Claims:
            await asyncio.sleep(0)
            return Claims(items=[raw_src.text.upper()])

        graph = compile(
            construct_from_functions("p", [raw_src, f]),
            **build_test_compile_kwargs(),
        )

        got = _ainvoke_field(graph, "f")

        assert not inspect.isawaitable(got), (
            "async @node scripted body left un-awaited — result is a coroutine "
            "(ScriptedDispatch.aexecute did not await the awaitable)"
        )
        assert isinstance(got, Claims)
        assert got == Claims(items=["HI"])

    def test_declarative_scripted_async_fn_is_awaited(self):
        """Surface 2 — declarative ``Node.scripted()`` with an ``async def``
        ``(input_data, config)`` fn registered via ``register_scripted``.
        """
        async def g(input_data, config):
            await asyncio.sleep(0)
            return Claims(items=["G"])

        register_scripted("g", g)
        graph = compile(
            Construct("p", nodes=[Node.scripted("g", fn="g", outputs=Claims)]),
            **build_test_compile_kwargs(),
        )

        got = _ainvoke_field(graph, "g")

        assert not inspect.isawaitable(got), (
            "async declarative scripted fn left un-awaited — result is a coroutine"
        )
        assert isinstance(got, Claims)
        assert got == Claims(items=["G"])

    def test_programmatic_node_async_fn_is_awaited(self):
        """Surface 3 — programmatic bare ``Node(...)`` constructor with the same
        registered ``async def`` fn.
        """
        async def h(input_data, config):
            await asyncio.sleep(0)
            return Claims(items=["H"])

        register_scripted("h", h)
        graph = compile(
            Construct(
                "p",
                nodes=[Node(name="h", mode="scripted", scripted_fn="h", outputs=Claims)],
            ),
            **build_test_compile_kwargs(),
        )

        got = _ainvoke_field(graph, "h")

        assert not inspect.isawaitable(got), (
            "async programmatic scripted fn left un-awaited — result is a coroutine"
        )
        assert isinstance(got, Claims)
        assert got == Claims(items=["H"])

    def test_raw_async_body_state_update_is_awaited(self):
        """Surface 4 — ``@node(mode='raw')`` async body under ainvoke. The raw
        path has no ``afunc`` twin today, so the sync raw wrapper returns an
        un-awaited coroutine as the node's state update instead of the dict.
        """
        @node(mode="raw", outputs=Claims)
        async def araw(state, config):
            await asyncio.sleep(0)
            return {"araw": Claims(items=["R"])}

        graph = compile(
            construct_from_functions("p", [araw]),
            **build_test_compile_kwargs(),
        )

        got = _ainvoke_field(graph, "araw")

        assert not inspect.isawaitable(got), (
            "async raw body left un-awaited — state update is a coroutine "
            "(raw path lacks an async afunc twin)"
        )
        assert isinstance(got, Claims)
        assert got == Claims(items=["R"])

    def test_sync_scripted_body_under_ainvoke_is_not_over_awaited(self):
        """Guard (green now AND after 1b) — a SYNC scripted body under ainvoke
        must NOT be awaited: ``isawaitable(result)`` is False, so the 1b
        conditional-await must leave it untouched. Locks against over-awaiting.
        """
        @node(outputs=Claims)
        def sync_body() -> Claims:
            return Claims(items=["S"])

        graph = compile(
            construct_from_functions("p", [sync_body]),
            **build_test_compile_kwargs(),
        )

        got = _ainvoke_field(graph, "sync_body")

        assert not inspect.isawaitable(got)
        assert isinstance(got, Claims)
        assert got == Claims(items=["S"])


class TestAsyncBodyUnderSyncRunFailsLoud:
    """neograph-khff — an ``async def`` scripted/raw body run under the SYNC
    driver (``run()`` / ``graph.invoke``) must FAIL LOUD, not silently store an
    un-awaited coroutine.

    The async driver awaits the awaitable (``ScriptedDispatch.aexecute`` /
    ``araw_node_wrapper`` via ``inspect.isawaitable``). The SYNC twins
    (``ScriptedDispatch.execute``, ``raw_node_wrapper``) cannot await, so instead
    of returning ``self.fn(...)`` directly they must detect the awaitable and
    raise ``NeographError`` telling the user to use ``arun()``. Otherwise a
    coroutine object flows into state where a model is expected — silent wrong
    behavior.

    RED before the fix: today the sync path returns the coroutine and no error is
    raised, so ``pytest.raises(NeographError)`` fails.
    """

    def test_scripted_async_body_under_sync_run_raises(self):
        """Scripted ``@node`` with an ``async def`` body run via sync ``run()``
        raises ``NeographError`` mentioning ``arun()``.
        """
        @node(outputs=RawText)
        def raw_src() -> RawText:
            return RawText(text="hi")

        @node(outputs=Claims)
        async def f(raw_src: RawText) -> Claims:
            await asyncio.sleep(0)
            return Claims(items=[raw_src.text.upper()])

        graph = compile(
            construct_from_functions("p", [raw_src, f]),
            **build_test_compile_kwargs(),
        )

        with pytest.raises(NeographError, match="arun"):
            neograph.run(graph, input={"node_id": "t"})

    def test_raw_async_body_under_sync_run_raises(self):
        """``mode='raw'`` node with an ``async def`` body run via sync ``run()``
        raises ``NeographError`` mentioning ``arun()``.
        """
        @node(mode="raw", outputs=Claims)
        async def araw(state, config):
            await asyncio.sleep(0)
            return {"araw": Claims(items=["R"])}

        graph = compile(
            construct_from_functions("p", [araw]),
            **build_test_compile_kwargs(),
        )

        with pytest.raises(NeographError, match="arun"):
            neograph.run(graph, input={"node_id": "t"})

    def test_scripted_async_body_under_arun_still_works(self):
        """Positive regression: the SAME async scripted body run via ``arun()``
        awaits correctly and produces the declared model — the fail-loud guard is
        sync-path only and must not touch the async path.
        """
        @node(outputs=RawText)
        def raw_src() -> RawText:
            return RawText(text="hi")

        @node(outputs=Claims)
        async def f(raw_src: RawText) -> Claims:
            await asyncio.sleep(0)
            return Claims(items=[raw_src.text.upper()])

        graph = compile(
            construct_from_functions("p", [raw_src, f]),
            **build_test_compile_kwargs(),
        )

        result = asyncio.run(neograph.arun(graph, input={"node_id": "t"}))

        assert result["f"] == Claims(items=["HI"])

    def test_raw_async_body_under_arun_still_works(self):
        """Positive regression: the SAME async raw body run via ``arun()`` awaits
        correctly and produces the declared model.
        """
        @node(mode="raw", outputs=Claims)
        async def araw(state, config):
            await asyncio.sleep(0)
            return {"araw": Claims(items=["R"])}

        graph = compile(
            construct_from_functions("p", [araw]),
            **build_test_compile_kwargs(),
        )

        result = asyncio.run(neograph.arun(graph, input={"node_id": "t"}))

        assert result["araw"] == Claims(items=["R"])
