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

from typing import Any, Callable

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from neograph import configure_llm


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

    def with_structured_output(self, model: type[BaseModel], **kwargs) -> "StructuredFake":
        clone = StructuredFake(self._respond)
        clone._model = model
        return clone

    def invoke(self, messages: list, **kwargs) -> Any:
        assert self._model is not None, "with_structured_output must be called first"
        return self._respond(self._model)


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
    ):
        self._tool_calls = tool_calls
        self._final = final
        self._call_idx = 0
        self._model: type[BaseModel] | None = None
        self._in_structured_mode = False

    def bind_tools(self, tools: list) -> "ReActFake":
        # Return self so call counter persists across rebinds
        return self

    def invoke(self, messages: list, **kwargs) -> Any:
        if self._in_structured_mode:
            assert self._final is not None, "ReActFake needs a final callable for structured output"
            return self._final(self._model)

        # Normal ReAct invocation
        if self._call_idx >= len(self._tool_calls):
            return AIMessage(content="done")

        calls = self._tool_calls[self._call_idx]
        self._call_idx += 1

        if not calls:
            return AIMessage(content="done")

        msg = AIMessage(content="")
        msg.tool_calls = calls
        return msg

    def with_structured_output(self, model: type[BaseModel], **kwargs) -> "ReActFake":
        clone = ReActFake(self._tool_calls, self._final)
        clone._call_idx = self._call_idx
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
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def configure_fake_llm(
    factory: Callable,
    prompt_compiler: Callable | None = None,
) -> None:
    """Configure NeoGraph with a fake LLM factory and a minimal prompt compiler.

    If prompt_compiler is not provided, a default one is used that returns
    a single user message with "test" content.
    """
    if prompt_compiler is None:
        prompt_compiler = lambda template, data, **kw: [{"role": "user", "content": "test"}]
    configure_llm(llm_factory=factory, prompt_compiler=prompt_compiler)
