"""Public LLM fakes for testing neograph pipelines (neograph.testing.fakes).

The battle-hardened doubles that survived the f7nt/eoi8/m6d3 agent-cycle
refactors — promoted from tests/fakes.py to a SUPPORTED public API so external
consumers stop hand-rolling AIMessage duck-types that break when the loop
coerces messages through LangChain. tests/fakes.py now re-exports these (one
implementation, two consumers).

Core: real ``langchain_core`` ``AIMessage`` types, the
``bind_tools`` / ``with_structured_output`` / ``bind`` surface, and
empty-tool_calls final-turn semantics — so think / agent / act / oracle-merge
nodes all resolve against the same double.

``FakeLLM(outputs)`` + ``install_fake_llm(monkeypatch, outputs)`` are the canned
entry points; the lower-level doubles are exposed for scenarios that script
tool-call sequences directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage
from pydantic import BaseModel


def _final_json_content(
    final: Callable[[type[BaseModel]], BaseModel] | None,
    output_model: type[BaseModel] | None = None,
) -> str:
    """Render a ReAct fake's FINAL (no-tool-call) turn as parseable JSON.

    neograph-f7nt: agent mode parses the loop's final turn as JSON directly
    (no separate with_structured_output re-gen), so fakes must emit JSON on
    their final turn. Supports both ``final`` conventions without the caller
    passing a model:
        final=lambda m: m(field=...)         -> capture-proxy records the kwargs
        final=lambda m: SomeModel(field=...) -> returns a real BaseModel
    An explicit ``output_model`` is used as a precise override when given.
    Returns the legacy ``"done"`` when there is no ``final``.
    """
    if final is None:
        return "done"
    if output_model is not None:
        return final(output_model).model_dump_json()

    # Each proxy instance holds its OWN kwargs so nested calls
    # (final=lambda m: m(x=m(y=1))) nest correctly instead of flattening or
    # str()-ing an inner proxy (PP-03).
    class _Capture:
        def __init__(self, **kwargs: Any) -> None:
            self._data = kwargs

    result = final(_Capture)  # type: ignore[arg-type]
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    data = result._data if isinstance(result, _Capture) else {}

    def _default(o: Any) -> Any:
        if isinstance(o, _Capture):
            return o._data
        if isinstance(o, BaseModel):
            return o.model_dump(mode="json")
        if isinstance(o, Enum):
            return o.value
        return str(o)

    return json.dumps(data, default=_default)


class StructuredFake:
    """Fake LLM that returns a Pydantic model via with_structured_output.

    Usage:
        fake = StructuredFake(lambda model: model(items=["a", "b"]))
        configure_fake_llm(lambda tier: fake)

    The `respond` callable receives the output model class and must return
    an instance of it.
    """

    def __init__(self, respond: Callable[[type[BaseModel]], BaseModel]):
        self._respond = respond
        self._model: type[BaseModel] | None = None
        # neograph-15s2: records the kwargs any ``bind`` call would forward to the
        # provider, on a SHARED list so a test holding the original fake can assert
        # what was bound. The ``structured`` strategy never binds ``response_format``
        # — a test asserts ``bind_calls`` stays empty of it (zero behavior change).
        self.bind_calls: list[dict[str, Any]] = []

    def with_structured_output(self, model: type[BaseModel], **kwargs) -> StructuredFake:
        clone = StructuredFake(self._respond)
        clone._model = model
        clone.bind_calls = self.bind_calls
        return clone

    def bind(self, **kwargs: Any) -> StructuredFake:
        """Capture provider kwargs the way a real ``BaseChatModel.bind`` would."""
        self.bind_calls.append(dict(kwargs))
        clone = StructuredFake(self._respond)
        clone._model = self._model
        clone.bind_calls = self.bind_calls
        return clone

    def invoke(self, messages: list, **kwargs) -> Any:
        assert self._model is not None, "with_structured_output must be called first"
        return self._respond(self._model)

    async def ainvoke(self, *a, **k) -> Any:
        return self.invoke(*a, **k)


class StructuredFakeWithRaw:
    """Fake LLM that honors include_raw=True in with_structured_output.

    When include_raw=True (the default in _call_structured), returns
    {"parsed": model_instance, "raw": AIMessage(content="fake", response_metadata=...)}
    so the usage metadata extraction path is testable.

    When include_raw=False, behaves like regular StructuredFake.

    Usage:
        fake = StructuredFakeWithRaw(lambda model: model(items=["a", "b"]))
        configure_fake_llm(lambda tier: fake)
    """

    def __init__(
        self,
        respond: Callable[[type[BaseModel]], BaseModel],
        *,
        usage: dict[str, int] | None = None,
    ):
        self._respond = respond
        self._model: type[BaseModel] | None = None
        self._include_raw: bool = False
        self._usage = usage or {"prompt_tokens": 10, "completion_tokens": 20}

    def with_structured_output(self, model: type[BaseModel], **kwargs) -> StructuredFakeWithRaw:
        clone = StructuredFakeWithRaw(self._respond, usage=self._usage)
        clone._model = model
        clone._include_raw = kwargs.get("include_raw", False)
        return clone

    def invoke(self, messages: list, **kwargs) -> Any:
        assert self._model is not None, "with_structured_output must be called first"
        result = self._respond(self._model)
        if self._include_raw:
            raw_msg = AIMessage(
                content="fake",
                response_metadata={"token_usage": self._usage},
                usage_metadata={
                    "input_tokens": self._usage.get("prompt_tokens", 0),
                    "output_tokens": self._usage.get("completion_tokens", 0),
                    "total_tokens": (self._usage.get("prompt_tokens", 0) + self._usage.get("completion_tokens", 0)),
                },
            )
            return {"parsed": result, "raw": raw_msg}
        return result

    async def ainvoke(self, *a, **k) -> Any:
        return self.invoke(*a, **k)


class ReActFake:
    """Fake LLM that scripts a ReAct tool loop.

    Usage:
        fake = ReActFake(
            tool_calls=[
                [{"name": "search", "args": {"q": "x"}, "id": "1"}],  # call 1
                [{"name": "search", "args": {"q": "y"}, "id": "2"}],  # call 2
                [],  # call 3: no tools, final response
            ],
            final=lambda model: model(items=["done"]),
        )
        configure_fake_llm(lambda tier: fake)

    Each element in tool_calls is the list of tool calls for that invocation.
    An empty list means "stop calling tools, final response coming." The
    final callable is invoked when with_structured_output().invoke() runs
    for the final parse.
    """

    def __init__(
        self,
        tool_calls: list[list[dict]],
        final: Callable[[type[BaseModel]], BaseModel] | None = None,
        output_model: type[BaseModel] | None = None,
    ):
        self._tool_calls = tool_calls
        self._final = final
        # neograph-f7nt: agent mode now parses the loop's FINAL turn as JSON
        # (no separate with_structured_output re-gen). When output_model is
        # given, the final turn serializes final(output_model) to JSON so the
        # json_mode-style tail can parse it. Without it, the final turn returns
        # plain "done" (legacy — for tests that drive their own final message).
        self._output_model = output_model
        self._call_idx = 0
        self._model: type[BaseModel] | None = None
        self._in_structured_mode = False

    def bind_tools(self, tools: list) -> ReActFake:
        # Return self so call counter persists across rebinds
        return self

    def abind_tools(self, *a, **k) -> ReActFake:
        return self.bind_tools(*a, **k)

    def _final_message(self) -> AIMessage:
        """The loop's final (no-tool-call) turn — parseable JSON (neograph-f7nt)."""
        return AIMessage(content=_final_json_content(self._final, self._output_model))

    def invoke(self, messages: list, **kwargs) -> Any:
        if self._in_structured_mode:
            assert self._final is not None, "ReActFake needs a final callable for structured output"
            return self._final(self._model)  # type: ignore[arg-type]  # fake: _model set by with_structured_output before structured invoke

        # History-driven turn index (neograph-m6d3): count prior tool-call turns
        # in the message history rather than a persistent counter. This behaves
        # correctly whether the agent runs as one node body (the LLM instance is
        # reused across turns) OR as a cycle of supersteps (the instance is rebuilt
        # each turn and replays the accumulated history) — a real stateless LLM
        # works both ways.
        idx = sum(1 for m in messages if getattr(m, "tool_calls", None) and not isinstance(m, dict))
        self._call_idx = idx  # kept in sync for with_structured_output clones

        if idx >= len(self._tool_calls):
            return self._final_message()

        calls = self._tool_calls[idx]
        if not calls:
            return self._final_message()

        msg = AIMessage(content="")
        msg.tool_calls = calls  # type: ignore[assignment]  # fake emits raw provider-shaped tool-call dicts
        return msg

    async def ainvoke(self, *a, **k) -> Any:
        return self.invoke(*a, **k)

    def with_structured_output(self, model: type[BaseModel], **kwargs) -> ReActFake:
        clone = ReActFake(self._tool_calls, self._final, self._output_model)
        clone._call_idx = self._call_idx
        clone._model = model
        clone._in_structured_mode = True
        return clone


class StringArgsFake:
    """Fake LLM that simulates providers emitting tool_calls.args as JSON strings.

    Every invoke() raises ValidationError (consistent provider behavior).
    Provides _generate() fallback that returns the response via
    additional_kwargs (which handles string args correctly).
    Used to test _CoercingToolWrapper resilience.

    Args:
        tool_calls: list of per-invocation tool call lists
        final: callable for structured output parse
        always_fail: if True, every invoke raises (default); if False,
            first invoke raises, subsequent succeed (intermittent mode)
    """

    def __init__(
        self,
        tool_calls: list[list[dict]],
        final: Callable[[type[BaseModel]], BaseModel] | None = None,
        *,
        always_fail: bool = True,
    ):
        self._tool_calls = tool_calls
        self._final = final
        self._call_idx = 0
        self._model: type[BaseModel] | None = None
        self._in_structured_mode = False
        self._always_fail = always_fail
        self._fail_count = 0

    def bind_tools(self, tools: list) -> StringArgsFake:
        return self

    def abind_tools(self, *a, **k) -> StringArgsFake:
        return self.bind_tools(*a, **k)

    def invoke(self, messages: list, **kwargs) -> Any:
        import json as _json

        if self._in_structured_mode:
            assert self._final is not None
            return self._final(self._model)  # type: ignore[arg-type]  # fake: _model set by with_structured_output before structured invoke

        if self._call_idx >= len(self._tool_calls):
            return AIMessage(content=_final_json_content(self._final))

        calls = self._tool_calls[self._call_idx]

        if not calls:
            self._call_idx += 1
            return AIMessage(content=_final_json_content(self._final))

        should_fail = self._always_fail or self._fail_count == 0
        if should_fail:
            self._fail_count += 1
            from pydantic import ValidationError

            # Raise the exact ValidationError that real providers produce
            try:
                AIMessage(
                    content="",
                    tool_calls=[
                        {**tc, "args": _json.dumps(tc["args"]) if isinstance(tc["args"], dict) else tc["args"]}
                        for tc in calls
                    ],
                )
            except ValidationError:
                raise

        # Non-failing path (intermittent mode, subsequent calls)
        self._call_idx += 1
        msg = AIMessage(content="")
        msg.tool_calls = calls  # type: ignore[assignment]  # fake emits raw provider-shaped tool-call dicts
        return msg

    async def ainvoke(self, *a, **k) -> Any:
        return self.invoke(*a, **k)

    def _generate(self, messages: list, *, run_manager: Any = None, **kwargs) -> Any:
        """Fallback for _CoercingToolWrapper — returns response via additional_kwargs."""
        import json as _json
        from types import SimpleNamespace

        if self._call_idx >= len(self._tool_calls):
            msg = AIMessage(content=_final_json_content(self._final))
            return SimpleNamespace(generations=[SimpleNamespace(message=msg)])

        calls = self._tool_calls[self._call_idx]
        self._call_idx += 1

        if not calls:
            msg = AIMessage(content=_final_json_content(self._final))
            return SimpleNamespace(generations=[SimpleNamespace(message=msg)])

        # Build via additional_kwargs — handles string args correctly
        ak_tool_calls = [
            {
                "id": tc.get("id", f"call_{i}"),
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": _json.dumps(tc["args"]) if isinstance(tc["args"], dict) else tc["args"],
                },
            }
            for i, tc in enumerate(calls)
        ]
        msg = AIMessage(content="", additional_kwargs={"tool_calls": ak_tool_calls})
        return SimpleNamespace(generations=[SimpleNamespace(message=msg)])

    async def _agenerate(self, *a, **k) -> Any:
        return self._generate(*a, **k)

    def with_structured_output(self, model: type[BaseModel], **kwargs) -> StringArgsFake:
        clone = StringArgsFake(self._tool_calls, self._final, always_fail=self._always_fail)
        clone._call_idx = self._call_idx
        clone._fail_count = self._fail_count
        clone._model = model
        clone._in_structured_mode = True
        return clone


class TextFake:
    """Fake LLM that returns plain text via invoke(). No with_structured_output.

    Usage:
        fake = TextFake('{"items": ["a", "b"]}')
        configure_fake_llm(lambda tier: fake)

    The framework's output_strategy="json_mode" or "text" will parse the
    text into the node's output model.
    """

    def __init__(self, text: str, *, reject_response_format: bool = False):
        self._text = text
        # neograph-15s2: simulate a provider that does NOT support the native
        # ``response_format`` kwarg (e.g. ChatAnthropic) — the bound clone raises
        # on invoke so the ATTEMPT-BIND-AND-FALL-BACK path is testable offline.
        self._reject_response_format = reject_response_format
        self._bound_response_format = False
        # Observability for native-binding tests. SHARED across bind clones so a
        # test holding the original fake sees what was bound and what was invoked.
        self.bind_calls: list[dict[str, Any]] = []
        self.messages_seen: list = []

    def bind(self, **kwargs: Any) -> TextFake:
        """Capture native provider kwargs (e.g. ``response_format``) the way a
        real LangChain ``BaseChatModel.bind`` would, returning a bound clone."""
        self.bind_calls.append(dict(kwargs))
        clone = TextFake(self._text, reject_response_format=self._reject_response_format)
        clone.bind_calls = self.bind_calls
        clone.messages_seen = self.messages_seen
        clone._bound_response_format = self._bound_response_format or ("response_format" in kwargs)
        return clone

    def invoke(self, messages: list, **kwargs) -> AIMessage:
        self.messages_seen.append(messages)
        if self._bound_response_format and self._reject_response_format:
            # Mirror a provider whose API rejects the forwarded response_format.
            raise TypeError("Completions.create() got an unexpected keyword argument 'response_format'")
        return AIMessage(content=self._text)

    async def ainvoke(self, *a, **k) -> AIMessage:
        return self.invoke(*a, **k)


class FakeTool:
    """Fake tool that records invocations. Set .name at construction."""

    def __init__(self, name: str, response: Any = "ok"):
        self.name = name
        self.response = response
        self.calls: list[dict] = []

    def invoke(self, args: dict, config: Any = None, **kwargs: Any) -> Any:
        # config= accepted (and ignored) to mirror a real LangChain BaseTool —
        # the agent cycle threads the run config into every tool call (neograph-zmfx).
        self.calls.append(args)
        return self.response

    async def ainvoke(self, *a, **k) -> Any:
        return self.invoke(*a, **k)


class GuardFake:
    """Fake LLM that returns tool calls when tools are bound, plain response otherwise.

    The factory returns a raw (unbound) fake with has_tools=False.
    bind_tools(non_empty_list) creates a bound version with has_tools=True.
    When the ReAct guard unbinds (llm_with_tools = llm), the raw fake returns
    a plain response (no tool calls), which breaks the loop naturally.

    Args:
        input_tokens_per_call: If > 0, each invocation includes this many
            input_tokens in usage_metadata (for token_budget tests).
    """

    def __init__(self, input_tokens_per_call: int = 0):
        from langchain_core.messages import AIMessage

        tool_call = {"name": "search", "args": {"q": "x"}, "id": "1"}
        self._call_counter = [0]
        self._input_tokens = input_tokens_per_call
        self._AIMessage = AIMessage
        self._tool_call = tool_call
        self._has_tools = False
        self._model = None
        self._structured = False

    def bind_tools(self, tools):
        bound = GuardFake(input_tokens_per_call=self._input_tokens)
        bound._call_counter = self._call_counter
        bound._has_tools = bool(tools)
        return bound

    def abind_tools(self, *a, **k):
        return self.bind_tools(*a, **k)

    def invoke(self, messages, **kwargs):
        if self._structured:
            return self._model(items=["done"])
        self._call_counter[0] += 1
        if self._has_tools:
            msg = self._AIMessage(content="")
            msg.tool_calls = [self._tool_call]
            if self._input_tokens:
                msg.usage_metadata = {
                    "input_tokens": self._input_tokens,
                    "output_tokens": 50,
                    "total_tokens": self._input_tokens + 50,
                }
            return msg
        # neograph-f7nt: agent mode parses the final turn as JSON. GuardFake's
        # structured output is always items-shaped (see with_structured_output),
        # so emit matching JSON directly.
        return self._AIMessage(content='{"items": ["done"]}')

    async def ainvoke(self, *a, **k):
        return self.invoke(*a, **k)

    def with_structured_output(self, model, **kwargs):
        clone = GuardFake(input_tokens_per_call=self._input_tokens)
        clone._call_counter = self._call_counter
        clone._has_tools = self._has_tools
        clone._model = model
        clone._structured = True
        return clone

    @property
    def call_count(self):
        return self._call_counter[0]


class StubbornFake:
    """Fake LLM that ALWAYS returns tool calls, ignoring bind_tools / unbinding.

    Used to verify the _guard_fired safety net: after the guard unbinds tools,
    if the LLM still returns tool_calls, the loop force-breaks instead of
    looping forever.
    """

    def __init__(self):
        from langchain_core.messages import AIMessage

        self._AIMessage = AIMessage
        self._model = None
        self._structured = False
        self._call_count = 0

    def bind_tools(self, tools):
        return self  # always returns self — ignores unbinding

    def abind_tools(self, *a, **k):
        return self.bind_tools(*a, **k)

    def invoke(self, messages, **kwargs):
        if self._structured:
            return self._model(items=["done"])
        self._call_count += 1
        msg = self._AIMessage(content="")
        msg.tool_calls = [{"name": "search", "args": {"q": "x"}, "id": "1"}]
        return msg

    async def ainvoke(self, *a, **k):
        return self.invoke(*a, **k)

    def with_structured_output(self, model, **kwargs):
        clone = StubbornFake()
        clone._model = model
        clone._structured = True
        return clone


class GatedAsyncFake:
    """Async-native fake whose ``ainvoke`` parks on a gate until released.

    Enables the two tests parity cannot be:
      * real-concurrency — launch N ``ainvoke`` coroutines on ONE loop, use
        ``wait_entered()`` to confirm they all parked simultaneously (proving
        interleaving/isolation), then ``release()`` and assert results.
      * cancellation E2E — park a run mid-flight and cancel the task, asserting
        clean teardown / checkpoint consistency.

    Async-NATIVE by construction: it has NO sync ``invoke`` twin, so it is
    EXEMPT from the sync/async bare-delegation invariant enforced on the shared
    fakes (test_guards_async_fakes.py only governs the 8 deterministic fakes).
    Computing its response inside ``ainvoke`` is correct here precisely because
    there is no sync surface to drift from.
    """

    def __init__(self, respond: Callable[[type[BaseModel]], BaseModel] | None = None):
        self._respond = respond
        self._model: type[BaseModel] | None = None
        self._gate: Any = None  # asyncio.Event, created lazily on the running loop
        self._entered: Any = None
        self.enter_count = 0

    def _ensure_events(self) -> None:

        if self._gate is None:
            self._gate = asyncio.Event()
        if self._entered is None:
            self._entered = asyncio.Event()

    def with_structured_output(self, model: type[BaseModel], **kwargs) -> GatedAsyncFake:
        self._model = model
        return self

    def bind_tools(self, tools) -> GatedAsyncFake:
        return self

    def abind_tools(self, *a, **k) -> GatedAsyncFake:
        return self.bind_tools(*a, **k)

    def release(self) -> None:
        """Unblock every parked (and future) ``ainvoke`` call."""
        self._ensure_events()
        self._gate.set()

    async def wait_entered(self) -> None:
        """Await until at least one ``ainvoke`` has reached the gate."""
        self._ensure_events()
        await self._entered.wait()

    async def ainvoke(self, *a, **k) -> Any:
        self._ensure_events()
        self.enter_count += 1
        self._entered.set()
        await self._gate.wait()
        if self._model is not None and self._respond is not None:
            return self._respond(self._model)
        return AIMessage(content="done")


@contextlib.asynccontextmanager
async def event_loop_lag_watchdog(threshold_s: float = 0.5, interval_s: float = 0.02):
    """Detect a coroutine/tool that BLOCKS the event loop (M1).

    Parity cannot catch a fake/tool that runs sync-blocking work under ``arun``:
    LangGraph threadpools a truly-sync node, masking the block. This watchdog
    schedules a heartbeat every ``interval_s``; if the loop is blocked so long
    that a heartbeat lands more than its interval late, the excess is recorded as
    lag. A generous default ``threshold_s`` keeps CI non-flaky — a blocking-
    detector test asserts ``handle.max_lag > threshold_s``.

    Yields a handle exposing ``.max_lag`` (seconds). Assertion-light in Phase 0;
    the blocking E2E lands with Phase 1.
    """
    import time

    class _Handle:
        max_lag = 0.0

    handle = _Handle()
    stop = asyncio.Event()

    async def _heartbeat() -> None:
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(interval_s)
            now = time.monotonic()
            lag = (now - last) - interval_s
            if lag > handle.max_lag:
                handle.max_lag = lag
            last = now

    task = asyncio.create_task(_heartbeat())
    # Prime the heartbeat: yield the loop once so it records its baseline and
    # parks on its first interval sleep BEFORE we hand control to the caller.
    # Without this the watchdog is unarmed until its first cycle, and a block
    # that happens immediately after entry would go unmeasured.
    await asyncio.sleep(0)
    try:
        yield handle
    finally:
        stop.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# FakeLLM + install_fake_llm — the canned, loop-compatible public entry points
# ═══════════════════════════════════════════════════════════════════════════


class FakeLLM:
    """A canned ``llm_factory`` mapping ``node_name -> a per-node double``.

    ``FakeLLM({"research": KResult(...), "": Claims(...)})`` returns, for each
    node, a double guaranteed compatible with EVERY LLM path:

    - agent / act (ReAct cycle): ``bind_tools`` + an empty-tool_calls final turn
      whose JSON content parses to the declared output;
    - think / produce and Oracle-merge (``node_name == ''``): the structured path
      ``with_structured_output(...).invoke()`` returns the mapped model.

    A single ``ReActFake(tool_calls=[[]], final=..., output_model=...)`` satisfies
    all three, so the same instance works whether the node is agent, think, or an
    Oracle merge whose node_name arrives as ``''``.

    ``node_name`` is declared explicitly in ``__call__`` so ``_get_llm``'s
    param-filtering (``LlmRuntime.llm_factory_params``) does not strip it — routing
    depends on the node_name reaching the factory.
    """

    def __init__(self, outputs: dict[str, BaseModel | str]):
        self._outputs = outputs

    def _double_for(self, value: BaseModel | str) -> Any:
        if isinstance(value, BaseModel):
            model = value

            def _final(_model: type[BaseModel]) -> BaseModel:
                # Ignore the requested model — the canned instance IS the answer,
                # for both the structured path and the JSON final-turn serializer.
                return model

            return ReActFake(tool_calls=[[]], final=_final, output_model=type(value))
        # A plain string: the final turn / structured parse yields the raw text
        # (json_mode / text strategies parse it downstream).
        return TextFake(str(value))

    def __call__(self, tier: str, *, node_name: str = "", **_kwargs: Any) -> Any:
        if node_name not in self._outputs:
            raise KeyError(f"FakeLLM has no output for node_name={node_name!r}; known keys: {sorted(self._outputs)}")
        return self._double_for(self._outputs[node_name])


def install_fake_llm(monkeypatch: Any, outputs: dict[str, BaseModel | str]) -> FakeLLM:
    """Patch the ``_get_llm`` seam at BOTH binding sites so a graph compiled inside
    app code (that the test cannot reach to pass ``llm_factory=``) resolves fakes.

    ``_get_llm`` is defined in ``neograph._llm`` AND imported into
    ``neograph._tool_loop``'s namespace (the agent/ReAct path calls the latter), so
    patching only ``_llm._get_llm`` would leave the agent path on the real factory.
    This patches both. Consumers who own the ``compile()`` call can instead pass
    ``llm_factory=FakeLLM(outputs)``.

    Returns the ``FakeLLM`` for assertions.
    """
    fake = FakeLLM(outputs)

    def _fake_get_llm(*args: Any, node_name: str = "", **_kw: Any) -> Any:
        # Mirror _get_llm's arg shapes to find the tier (unused by FakeLLM but
        # kept so the signature is call-compatible with both call forms).
        if args and hasattr(args[0], "llm_factory"):
            tier = args[1] if len(args) > 1 else "fast"
        else:
            tier = args[0] if args else "fast"
        return fake(tier, node_name=node_name)

    monkeypatch.setattr("neograph._llm._get_llm", _fake_get_llm)
    monkeypatch.setattr("neograph._tool_loop._get_llm", _fake_get_llm)
    return fake
