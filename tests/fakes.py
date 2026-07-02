"""Shared test fakes — FakeLLM variants and helpers.

Three fake LLM patterns cover all production scenarios:

StructuredFake  — implements with_structured_output() for produce mode.
                  The `respond` callable is called with the output model
                  and should return an instance of it.

ReActFake       — implements bind_tools() + invoke() with a scriptable
                  sequence of responses. For gather/execute mode tests.

TextFake        — returns plain text via invoke(). For json_mode and
                  text output strategies where the framework parses.

Also provides helpers:

configure_fake_llm(factory, prompt="test") — one-line configure_llm
setup with a simple prompt compiler.
"""

from __future__ import annotations

import json
from collections.abc import Callable

# Post-§2 (ticket ezqz): `configure_llm` no longer exists. Tests pass LLM
# configuration as kwargs to `compile()`. Helpers below return those kwargs
# instead of mutating module state.
#
# `register_scripted`/`register_condition`/`register_tool_factory` were also
# removed from src/neograph/. They now live HERE as test-only convenience
# that writes to test-local dicts. compile() reads them via the `scripted=`,
# `conditions=`, `tool_factories=` kwargs (tests splat via
# `compile(c, **build_test_compile_kwargs())` or pass dicts explicitly).
from collections.abc import Callable as _Callable
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage
from pydantic import BaseModel

_TEST_SCRIPTED: dict[str, _Callable] = {}
_TEST_CONDITIONS: dict[str, _Callable] = {}
_TEST_TOOL_FACTORIES: dict[str, _Callable] = {}


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


def register_scripted(name: str, fn: _Callable) -> None:
    """Test-side scripted-shim registration (post-§2).

    Writes into a test-local dict in `tests/fakes.py`; the autouse fixture in
    `tests/conftest.py` resets these dicts between tests. Tests pass the
    accumulated registrations to compile() via `**build_test_compile_kwargs()`.
    """
    _TEST_SCRIPTED[name] = fn


def register_condition(name: str, fn: _Callable) -> None:
    """Test-side condition registration. See `register_scripted` for context."""
    _TEST_CONDITIONS[name] = fn


def register_tool_factory(name: str, fn: _Callable) -> None:
    """Test-side tool-factory registration. See `register_scripted` for context."""
    _TEST_TOOL_FACTORIES[name] = fn


def build_test_compile_kwargs(**extra) -> dict[str, Any]:
    """Build compile() kwargs from the test-local registration dicts.

    Tests that registered scripted/conditions/tool_factories pass the result
    to `compile(c, **build_test_compile_kwargs())`. Extra kwargs (e.g.
    `llm_factory=...`) are merged on top.
    """
    out: dict[str, Any] = {}
    if _TEST_SCRIPTED:
        out["scripted"] = dict(_TEST_SCRIPTED)
    if _TEST_CONDITIONS:
        out["conditions"] = dict(_TEST_CONDITIONS)
    if _TEST_TOOL_FACTORIES:
        out["tool_factories"] = dict(_TEST_TOOL_FACTORIES)
    out.update(extra)
    return out


def reset_test_registry() -> None:
    """Clear the test-local registration dicts. Called by conftest fixture."""
    _TEST_SCRIPTED.clear()
    _TEST_CONDITIONS.clear()
    _TEST_TOOL_FACTORIES.clear()


def lookup_scripted(name: str) -> _Callable:
    """Test-only mirror of the removed src/ helper.

    Looks up in both the test-local dict AND the decoration-time registry
    (which holds inline body-merge / @merge_fn / interrupt shims).
    """
    from neograph._runtime_registry import _decoration_registry
    from neograph.errors import ConfigurationError
    fn = _TEST_SCRIPTED.get(name) or _decoration_registry.scripted.get(name)
    if fn is None:
        raise ConfigurationError.build(
            f"Scripted function '{name}' not registered",
            hint="Use tests.fakes.register_scripted() or pass scripted= to compile().",
        )
    return fn


def lookup_condition(name: str) -> _Callable:
    """Test-only mirror of the removed src/ helper."""
    from neograph._runtime_registry import _decoration_registry
    from neograph.errors import ConfigurationError
    fn = _TEST_CONDITIONS.get(name) or _decoration_registry.condition.get(name)
    if fn is None:
        raise ConfigurationError.build(
            f"Condition '{name}' not registered",
            hint="Use tests.fakes.register_condition() or pass conditions= to compile().",
        )
    return fn


# ═══════════════════════════════════════════════════════════════════════════
# StructuredFake — produce mode
# ═══════════════════════════════════════════════════════════════════════════


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

    def with_structured_output(self, model: type[BaseModel], **kwargs) -> StructuredFake:
        clone = StructuredFake(self._respond)
        clone._model = model
        return clone

    def invoke(self, messages: list, **kwargs) -> Any:
        assert self._model is not None, "with_structured_output must be called first"
        return self._respond(self._model)


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


# ═══════════════════════════════════════════════════════════════════════════
# ReActFake — gather/execute mode
# ═══════════════════════════════════════════════════════════════════════════


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

    def _final_message(self) -> AIMessage:
        """The loop's final (no-tool-call) turn — parseable JSON (neograph-f7nt)."""
        return AIMessage(content=_final_json_content(self._final, self._output_model))

    def invoke(self, messages: list, **kwargs) -> Any:
        if self._in_structured_mode:
            assert self._final is not None, "ReActFake needs a final callable for structured output"
            return self._final(self._model)

        # Normal ReAct invocation
        if self._call_idx >= len(self._tool_calls):
            return self._final_message()

        calls = self._tool_calls[self._call_idx]
        self._call_idx += 1

        if not calls:
            return self._final_message()

        msg = AIMessage(content="")
        msg.tool_calls = calls
        return msg

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

    def invoke(self, messages: list, **kwargs) -> Any:
        import json as _json

        if self._in_structured_mode:
            assert self._final is not None
            return self._final(self._model)

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
                    tool_calls=[{**tc, "args": _json.dumps(tc["args"]) if isinstance(tc["args"], dict) else tc["args"]}
                                for tc in calls],
                )
            except ValidationError:
                raise

        # Non-failing path (intermittent mode, subsequent calls)
        self._call_idx += 1
        msg = AIMessage(content="")
        msg.tool_calls = calls
        return msg

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

    def with_structured_output(self, model: type[BaseModel], **kwargs) -> StringArgsFake:
        clone = StringArgsFake(self._tool_calls, self._final, always_fail=self._always_fail)
        clone._call_idx = self._call_idx
        clone._fail_count = self._fail_count
        clone._model = model
        clone._in_structured_mode = True
        return clone


# ═══════════════════════════════════════════════════════════════════════════
# TextFake — json_mode / text output strategies
# ═══════════════════════════════════════════════════════════════════════════


class TextFake:
    """Fake LLM that returns plain text via invoke(). No with_structured_output.

    Usage:
        fake = TextFake('{"items": ["a", "b"]}')
        configure_fake_llm(lambda tier: fake)

    The framework's output_strategy="json_mode" or "text" will parse the
    text into the node's output model.
    """

    def __init__(self, text: str):
        self._text = text

    def invoke(self, messages: list, **kwargs) -> AIMessage:
        return AIMessage(content=self._text)


# ═══════════════════════════════════════════════════════════════════════════
# FakeTool — shared tool implementation for gather/execute tests
# ═══════════════════════════════════════════════════════════════════════════


class FakeTool:
    """Fake tool that records invocations. Set .name at construction."""

    def __init__(self, name: str, response: Any = "ok"):
        self.name = name
        self.response = response
        self.calls: list[dict] = []

    def invoke(self, args: dict) -> Any:
        self.calls.append(args)
        return self.response


# ═══════════════════════════════════════════════════════════════════════════
# GuardFake — ReAct loop guard tests (max_iterations / token_budget)
# ═══════════════════════════════════════════════════════════════════════════


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

    def invoke(self, messages, **kwargs):
        if self._structured:
            return self._model(items=["done"])
        self._call_count += 1
        msg = self._AIMessage(content="")
        msg.tool_calls = [{"name": "search", "args": {"q": "x"}, "id": "1"}]
        return msg

    def with_structured_output(self, model, **kwargs):
        clone = StubbornFake()
        clone._model = model
        clone._structured = True
        return clone


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _default_fake_prompt_compiler(template, data, **kw):
    """A minimal prompt compiler that returns a single user message.
    Test sites that need a custom compiler pass their own."""
    return [{"role": "user", "content": "test"}]


def build_fake_llm_kwargs(
    factory: Callable,
    prompt_compiler: Callable | None = None,
    *,
    cost_callback: Callable | None = None,
    renderer: Any = None,
) -> dict[str, Any]:
    """Build the LLM kwargs dict that gets splatted into `compile(c, **kwargs)`.

    Replaces the old `configure_fake_llm(...)` (which mutated module state).
    Test sites that previously did::

        configure_fake_llm(lambda tier: fake)
        graph = compile(pipeline)

    now do::

        graph = compile(pipeline, **build_fake_llm_kwargs(lambda tier: fake))
    """
    pc = prompt_compiler if prompt_compiler is not None else _default_fake_prompt_compiler
    kwargs: dict[str, Any] = {
        "llm_factory": factory,
        "prompt_compiler": pc,
    }
    if cost_callback is not None:
        kwargs["cost_callback"] = cost_callback
    if renderer is not None:
        kwargs["renderer"] = renderer
    return kwargs


# Backward-compat alias to ease the migration. New code prefers
# `build_fake_llm_kwargs(...)`. The name `configure_fake_llm` no longer
# mutates anything — it now returns the same kwargs dict.
def configure_fake_llm(
    factory: Callable,
    prompt_compiler: Callable | None = None,
) -> dict[str, Any]:
    """Migration alias for `build_fake_llm_kwargs`.

    Old shape mutated module state via `configure_llm`. New shape returns
    a kwargs dict — every old call site must add `**` to splat the result
    into `compile()` or accept the dict and pass it explicitly.
    """
    return build_fake_llm_kwargs(factory, prompt_compiler)


def build_fake_runtime(
    factory: Callable | None = None,
    prompt_compiler: Callable | None = None,
    *,
    cost_callback: Callable | None = None,
    renderer: Any = None,
) -> Any:
    """Build an `LlmRuntime` backed by fakes — for tests that call helpers
    like `invoke_structured(runtime, ...)` directly without going through
    `compile()`.
    """
    from neograph._llm_runtime import LlmRuntime

    if factory is None:
        factory = lambda tier: StructuredFake(lambda m: m())  # noqa: E731
    if prompt_compiler is None:
        prompt_compiler = _default_fake_prompt_compiler
    return LlmRuntime.build(
        llm_factory=factory,
        prompt_compiler=prompt_compiler,
        cost_callback=cost_callback,
        renderer=renderer,
    )


def build_fake_tool_lookup() -> dict[str, _Callable]:
    """Snapshot of the test-local tool-factory registry.

    For tests that call `invoke_with_tools(...)` directly (without going
    through `compile()`), pass the result as `tool_factory_lookup=`. Mirrors
    what `compile()` would have built from `tool_factories=` kwarg.
    """
    return dict(_TEST_TOOL_FACTORIES)
