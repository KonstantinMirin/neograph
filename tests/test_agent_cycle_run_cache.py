"""Per-run handle/resource reuse for the inline agent cycle — neograph-m6d3.8
(within-run live LLM/tool handle reuse) and neograph-43do (per-run FROM_RESOURCE
fetch cache on the async agent path).

The inline agent cycle rebuilds its tool-loop preamble ONCE PER SUPERSTEP (each
ReAct turn — agent/tools/parse — is a separate checkpointed node body). A
stateless rebuild is correct on resume but re-invokes the LLM factory / tool
factories (m6d3.8) and the FROM_RESOURCE fetcher (43do) every turn. Both tickets
cache the live handle/resource for the duration of ONE run, keyed on the
framework-minted ``StateKeys.RUN_ID`` (config-only, minted fresh per run in
``runner._prepare``/``_aprepare``, never persisted into a checkpoint):

- WITHIN one run, the RUN_ID is stable across every superstep -> cache HIT ->
  the handle/resource is reused (built/fetched exactly once).
- ON RESUME, ``_prepare`` re-mints a FRESH RUN_ID -> cache MISS -> a fresh
  rebuild/refetch. This is the §5 two-lifetime rule: a cached entry never
  outlives the run/process it was built for. Resume tests use a REAL file-backed
  Sqlite saver per the project checkpoint-test convention.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import interrupt
from pydantic import BaseModel

from neograph import (
    FromResource,
    Tool,
    arun,
    compile,
    construct_from_functions,
    node,
    run,
)
from neograph._state_keys import StateKeys
from tests.fakes import (
    build_fake_llm_kwargs,
    build_test_compile_kwargs,
    register_tool_factory,
)


class KResult(BaseModel, frozen=True):
    items: list[str]


# ── Stateless, history-driven LLM fakes (mirror the keystone fakes) ──────────


class _MultiTurnFake:
    """Calls ``record`` on the first two turns, then emits the final answer.
    Stateless: decides from message history, so it is reuse-safe when cached."""

    def __init__(self) -> None:
        self._model: type[BaseModel] | None = None
        self._structured = False

    def bind_tools(self, tools: list) -> _MultiTurnFake:
        return self

    def abind_tools(self, *a: Any, **k: Any) -> _MultiTurnFake:
        return self

    def invoke(self, messages: list, **kwargs: Any) -> Any:
        if self._structured:
            assert self._model is not None
            return self._model(items=["done"])
        n_results = sum(isinstance(m, ToolMessage) for m in messages)
        if n_results < 2:
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": "record", "args": {"fact": "x"}, "id": f"r{n_results}"}]
            return msg
        return AIMessage(content='{"items": ["done"]}')

    async def ainvoke(self, *a: Any, **k: Any) -> Any:
        return self.invoke(*a, **k)

    def with_structured_output(self, model: type[BaseModel], **kwargs: Any) -> _MultiTurnFake:
        clone = _MultiTurnFake()
        clone._model = model
        clone._structured = True
        return clone


class _InterruptFake:
    """record (turn 1) -> ask_operator/interrupt (turn 2) -> final (turn 3)."""

    def __init__(self) -> None:
        self._model: type[BaseModel] | None = None
        self._structured = False

    def bind_tools(self, tools: list) -> _InterruptFake:
        return self

    def abind_tools(self, *a: Any, **k: Any) -> _InterruptFake:
        return self

    def invoke(self, messages: list, **kwargs: Any) -> Any:
        if self._structured:
            assert self._model is not None
            return self._model(items=["done"])
        n_results = sum(isinstance(m, ToolMessage) for m in messages)
        if n_results == 0:
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": "record", "args": {"fact": "x"}, "id": "r1"}]
            return msg
        if n_results == 1:
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": "ask_operator", "args": {"q": "decide"}, "id": "a1"}]
            return msg
        return AIMessage(content='{"items": ["done"]}')

    async def ainvoke(self, *a: Any, **k: Any) -> Any:
        return self.invoke(*a, **k)

    def with_structured_output(self, model: type[BaseModel], **kwargs: Any) -> _InterruptFake:
        clone = _InterruptFake()
        clone._model = model
        clone._structured = True
        return clone


class _RecordTool:
    name = "record"

    def __init__(self, counter: list[int]) -> None:
        self._counter = counter

    def invoke(self, args: dict, config: Any = None, **kwargs: Any) -> str:
        self._counter[0] += 1
        return "recorded"

    async def ainvoke(self, *a: Any, **k: Any) -> str:
        return self.invoke(*a, **k)


class _AskTool:
    name = "ask_operator"

    def __init__(self, received: list) -> None:
        self._received = received

    def invoke(self, args: dict, config: Any = None, **kwargs: Any) -> str:
        answer = interrupt({"question": "decide?"})
        self._received.append(answer)
        return f"decided: {answer}"

    async def ainvoke(self, *a: Any, **k: Any) -> str:
        return self.invoke(*a, **k)


def _counting_factory(sink: list, fake_cls: type) -> Any:
    def factory(tier: str, **kw: Any) -> Any:
        sink.append(tier)
        return fake_cls()

    return factory


# ── neograph-m6d3.8: within-run LLM/tool handle reuse ───────────────────────


class TestWithinRunLlmHandleReuse:
    """The live LLM handle is built ONCE per run and reused across the agent
    cycle's supersteps; a resume (fresh RUN_ID) rebuilds it."""

    def test_llm_handle_built_once_across_supersteps_within_one_run(self):
        counter = [0]
        factory_calls: list[str] = []
        register_tool_factory("record", lambda cfg, tc: _RecordTool(counter))

        @node(
            mode="agent",
            outputs=KResult,
            model="reason",
            prompt="test/explore",
            tools=[Tool(name="record", budget=5)],
        )
        def research() -> KResult: ...

        graph = compile(
            construct_from_functions("rc_within", [research]),
            checkpointer=MemorySaver(),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(_counting_factory(factory_calls, _MultiTurnFake)),
        )
        result = run(graph, input={"node_id": "X"}, config={"configurable": {"thread_id": "rc-within"}})

        assert result.get("research") == KResult(items=["done"])
        assert counter[0] == 2, "the two record turns must have executed"
        assert len(factory_calls) == 1, (
            f"llm_factory must be invoked ONCE per run (built handle reused across "
            f"the cycle's supersteps), got {len(factory_calls)}: {factory_calls}"
        )

    def test_llm_handle_rebuilt_on_resume(self, tmp_path):
        counter = [0]
        received: list = []
        factory_calls: list[str] = []
        register_tool_factory("record", lambda cfg, tc: _RecordTool(counter))
        register_tool_factory("ask_operator", lambda cfg, tc: _AskTool(received))

        @node(
            mode="agent",
            outputs=KResult,
            model="reason",
            prompt="test/explore",
            tools=[Tool(name="record", budget=3), Tool(name="ask_operator", budget=3)],
        )
        def research() -> KResult: ...

        db = str(tmp_path / "rc-resume.db")
        config = {"configurable": {"thread_id": "rc-resume"}}
        with SqliteSaver.from_conn_string(db) as saver:
            graph = compile(
                construct_from_functions("rc_resume", [research]),
                checkpointer=saver,
                **build_test_compile_kwargs(),
                **build_fake_llm_kwargs(_counting_factory(factory_calls, _InterruptFake)),
            )
            paused = run(graph, input={"node_id": "REQ"}, config=config)
            assert "__interrupt__" in paused
            # Reused across life-1's supersteps (agent/tools/agent/tools).
            assert len(factory_calls) == 1, f"life-1 must build the handle once, got {factory_calls}"

            resumed = run(graph, resume={"decision": "go"}, config=config)
            assert "__interrupt__" not in resumed

        # Fresh RUN_ID on resume -> cache miss -> exactly one MORE build in life-2
        # (never a stale life-1 handle: the two-lifetime rule).
        assert len(factory_calls) == 2, (
            f"resume must re-mint RUN_ID and rebuild the handle (life-2), got {factory_calls}"
        )


# ── neograph-43do: per-run FROM_RESOURCE fetch cache (async path) ───────────


class TestPerRunResourceFetchCache:
    """A FROM_RESOURCE template var is fetched ONCE per run on the async agent
    cycle and reused across supersteps; a resume (fresh RUN_ID) refetches."""

    async def test_resource_fetched_once_across_supersteps_within_one_run(self):
        counter = [0]
        fetch_count = [0]
        register_tool_factory("record", lambda cfg, tc: _RecordTool(counter))

        async def fetch(uri: str):
            fetch_count[0] += 1
            return "HISTORY", "text/plain"

        @node(
            mode="agent",
            outputs=KResult,
            model="reason",
            prompt="tmpl",
            tools=[Tool(name="record", budget=5)],
        )
        def research(history: Annotated[str, FromResource("crm://history")]) -> KResult: ...

        graph = compile(
            construct_from_functions("rr_within", [research]),
            checkpointer=MemorySaver(),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: _MultiTurnFake()),
        )
        config = {
            "configurable": {
                "mcp_resource_fetcher": fetch,
                "node_id": "t",
                "thread_id": "rr-within",
            }
        }
        result = await arun(graph, input={"node_id": "t"}, config=config)

        assert result.get("research") == KResult(items=["done"])
        assert counter[0] == 2, "the two record turns must have executed"
        assert fetch_count[0] == 1, (
            f"FROM_RESOURCE must be fetched ONCE per run (reused across the cycle's supersteps), got {fetch_count[0]}"
        )

    async def test_resource_refetched_on_resume(self, tmp_path):
        counter = [0]
        received: list = []
        fetch_count = [0]
        register_tool_factory("record", lambda cfg, tc: _RecordTool(counter))
        register_tool_factory("ask_operator", lambda cfg, tc: _AskTool(received))

        async def fetch(uri: str):
            fetch_count[0] += 1
            return "HISTORY", "text/plain"

        @node(
            mode="agent",
            outputs=KResult,
            model="reason",
            prompt="tmpl",
            tools=[Tool(name="record", budget=3), Tool(name="ask_operator", budget=3)],
        )
        def research(history: Annotated[str, FromResource("crm://history")]) -> KResult: ...

        db = str(tmp_path / "rr-resume.db")
        config = {
            "configurable": {
                "mcp_resource_fetcher": fetch,
                "node_id": "t",
                "thread_id": "rr-resume",
            }
        }
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            graph = compile(
                construct_from_functions("rr_resume", [research]),
                checkpointer=saver,
                **build_test_compile_kwargs(),
                **build_fake_llm_kwargs(lambda tier: _InterruptFake()),
            )
            paused = await arun(graph, input={"node_id": "t"}, config=config)
            assert "__interrupt__" in paused
            assert fetch_count[0] == 1, f"life-1 must fetch the resource once, got {fetch_count[0]}"

            resumed = await arun(graph, resume={"decision": "go"}, config=config)
            assert "__interrupt__" not in resumed

        assert fetch_count[0] == 2, (
            f"resume must re-mint RUN_ID and refetch the resource (life-2), got {fetch_count[0]}"
        )


# ── neograph-hhlr: per-key single-flight (sync + async twins) ───────────────


class TestPerKeySingleFlight:
    """Two concurrent misses on ONE (run_id, subkey) must build EXACTLY once —
    the loser blocks on the per-key latch, then reuses the winner's value. Twin
    thinness: the sync and async cases pin the same property on their surface."""

    def test_sync_concurrent_misses_build_the_key_once(self):
        import threading

        from neograph._run_cache import get_or_build

        config = {"configurable": {StateKeys.RUN_ID: "sf-sync"}}
        build_count = [0]
        entered = threading.Semaphore(0)
        proceed = threading.Event()
        results: dict[int, object] = {}

        def build() -> object:
            build_count[0] += 1
            entered.release()  # signal "a build is in-flight, holding the latch"
            proceed.wait(2.0)
            return object()

        def worker(i: int) -> None:
            results[i] = get_or_build(config, "k", build)

        t1 = threading.Thread(target=worker, args=(0,))
        t1.start()
        # Winner is now parked INSIDE build() holding the per-key latch.
        assert entered.acquire(timeout=2.0)

        t2 = threading.Thread(target=worker, args=(1,))
        t2.start()
        # Without single-flight the loser also enters build() (double-build);
        # with it, the loser blocks on the latch and never reaches build().
        proceed.set()
        t1.join(2.0)
        t2.join(2.0)

        assert build_count[0] == 1, (
            f"concurrent misses on one key double-built (no single-flight): build_count={build_count[0]}"
        )
        assert results[0] is results[1], "loser did not reuse the winner's value"

    async def test_async_concurrent_misses_build_the_key_once(self):
        from neograph._run_cache import aget_or_build

        config = {"configurable": {StateKeys.RUN_ID: "sf-async"}}
        build_count = [0]
        entered = asyncio.Event()
        proceed = asyncio.Event()

        async def build() -> object:
            build_count[0] += 1
            entered.set()
            await proceed.wait()
            return object()

        t1 = asyncio.create_task(aget_or_build(config, "k", build))
        await entered.wait()  # winner parked inside build(), holding the latch
        t2 = asyncio.create_task(aget_or_build(config, "k", build))
        await asyncio.sleep(0)  # let the loser reach the latch (miss -> acquire)
        proceed.set()
        r1, r2 = await asyncio.gather(t1, t2)

        assert build_count[0] == 1, (
            f"concurrent misses on one key double-built (no single-flight): build_count={build_count[0]}"
        )
        assert r1 is r2, "loser did not reuse the winner's value"

    def test_async_latch_is_loop_affine_across_separate_loops(self):
        """A miss on the SAME (run_id, subkey) under two DIFFERENT event loops
        must not reuse one ``asyncio.Lock`` — awaiting a lock bound to a dead
        loop raises. The loop-affine latch map mints a fresh lock per loop, so
        the second loop rebuilds cleanly rather than raising."""
        from neograph import _run_cache
        from neograph._run_cache import aget_or_build

        config = {"configurable": {StateKeys.RUN_ID: "loop-affine"}}
        build_count = [0]

        async def build() -> object:
            build_count[0] += 1
            return object()

        async def once() -> object:
            return await aget_or_build(config, "k", build)

        asyncio.run(once())  # loop 1: mints + binds a latch for this key
        # Force a miss on loop 2 WITHOUT dropping the latch map — this is the
        # exact hazard: a shared lock would still carry loop-1's binding.
        with _run_cache._lock:
            _run_cache._cache.clear()
        asyncio.run(once())  # loop 2: must NOT raise "bound to a different loop"

        assert build_count[0] == 2, "loop 2 did not rebuild after the forced miss"


# ── neograph-hhlr: run-end eviction ─────────────────────────────────────────


class TestRunEndEviction:
    """A run's cache entries are dropped the moment its driver verb returns, so
    loop-bound handles do not linger until LRU pressure."""

    def test_evict_run_drops_only_that_runs_keys(self):
        from neograph._run_cache import evict_run, get_or_build

        c1 = {"configurable": {StateKeys.RUN_ID: "r1"}}
        c2 = {"configurable": {StateKeys.RUN_ID: "r2"}}
        get_or_build(c1, "a", lambda: "1a")
        get_or_build(c1, "b", lambda: "1b")
        get_or_build(c2, "a", lambda: "2a")

        evict_run("r1")

        # r1's keys are gone -> a fresh build replaces them; r2 is untouched.
        assert get_or_build(c1, "a", lambda: "REBUILT") == "REBUILT"
        assert get_or_build(c2, "a", lambda: "REBUILT") == "2a"

    def test_run_verb_evicts_its_cache_on_completion(self):
        """After ``run`` returns, no cache entry survives for that run — the
        finalize-seam eviction fired in the verb's ``finally``."""
        from neograph import _run_cache

        counter = [0]
        register_tool_factory("record", lambda cfg, tc: _RecordTool(counter))

        @node(
            mode="agent",
            outputs=KResult,
            model="reason",
            prompt="test/explore",
            tools=[Tool(name="record", budget=5)],
        )
        def research() -> KResult: ...

        graph = compile(
            construct_from_functions("rc_evict", [research]),
            checkpointer=MemorySaver(),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: _MultiTurnFake()),
        )
        _run_cache.clear()
        result = run(graph, input={"node_id": "X"}, config={"configurable": {"thread_id": "rc-evict"}})

        assert result.get("research") == KResult(items=["done"])
        assert len(_run_cache._cache) == 0, (
            f"run() must evict its run's cache entries in finally; leftover: {list(_run_cache._cache)}"
        )
