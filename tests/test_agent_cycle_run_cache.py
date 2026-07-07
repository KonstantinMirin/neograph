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
        result = run(
            graph, input={"node_id": "X"}, config={"configurable": {"thread_id": "rc-within"}}
        )

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
            assert len(factory_calls) == 1, (
                f"life-1 must build the handle once, got {factory_calls}"
            )

            resumed = run(graph, resume={"decision": "go"}, config=config)
            assert "__interrupt__" not in resumed

        # Fresh RUN_ID on resume -> cache miss -> exactly one MORE build in life-2
        # (never a stale life-1 handle: the two-lifetime rule).
        assert len(factory_calls) == 2, (
            f"resume must re-mint RUN_ID and rebuild the handle (life-2), got "
            f"{factory_calls}"
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
            f"FROM_RESOURCE must be fetched ONCE per run (reused across the cycle's "
            f"supersteps), got {fetch_count[0]}"
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
            assert fetch_count[0] == 1, (
                f"life-1 must fetch the resource once, got {fetch_count[0]}"
            )

            resumed = await arun(graph, resume={"decision": "go"}, config=config)
            assert "__interrupt__" not in resumed

        assert fetch_count[0] == 2, (
            f"resume must re-mint RUN_ID and refetch the resource (life-2), got "
            f"{fetch_count[0]}"
        )
