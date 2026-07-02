"""Executed async-fake smoke (neograph-w74k.1, Phase 0 — review finding LOW-3).

The structural drift-guard (tests/test_guards_async_fakes.py) INSPECTS the async
mirrors but never runs them; the async driver cell skips today (arun does not
exist). This module EXERCISES ``ainvoke`` directly via ``asyncio.run(...)`` so
the async surface is proven callable, not merely present.

State-safety (architect review MED-1): the scripted fakes mutate call-index
state on each invoke, and AIMessage has no value ``__eq__``. So each assertion
drives TWO FRESH identically-constructed instances (one via sync ``invoke``, one
via ``asyncio.run(ainvoke)``) — never sequential calls on one instance — and
compares a NORMALIZED projection (.content / .tool_calls / model_dump), not raw
object equality.

These are plain sync ``def test_*`` cases calling ``asyncio.run``; they need no
pytest-asyncio collection. Do NOT gate on ``arun`` (Phase 1).

TDD red: FAILS today with ``AttributeError: '<Fake>' object has no attribute
'ainvoke'``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from tests.fakes import ReActFake, StructuredFake


class Items(BaseModel):
    items: list[str]


def _project(result: Any) -> tuple:
    """Normalized, __eq__-safe projection of a fake response.

    AIMessage has no value equality; project to (content, tool_calls). Pydantic
    models project to their dump. This is what parity compares — never the raw
    object.
    """
    if isinstance(result, AIMessage):
        return ("aimessage", result.content, list(getattr(result, "tool_calls", []) or []))
    if isinstance(result, BaseModel):
        return ("model", result.model_dump())
    return ("other", result)


class TestAsyncFakeSmoke:
    """ainvoke is EXECUTED and yields the same normalized result as invoke."""

    def test_structuredfake_ainvoke_matches_invoke(self):
        """StructuredFake.ainvoke (via asyncio.run) projects identically to
        invoke() using two fresh instances."""
        def respond(model: type[BaseModel]) -> BaseModel:
            return model(items=["a", "b"])

        sync_fake = StructuredFake(respond).with_structured_output(Items)
        async_fake = StructuredFake(respond).with_structured_output(Items)

        sync_result = sync_fake.invoke([])
        async_result = asyncio.run(async_fake.ainvoke([]))

        assert _project(sync_result) == _project(async_result)

    def test_reactfake_ainvoke_matches_invoke(self):
        """ReActFake.ainvoke (via asyncio.run) projects identically to invoke()
        using two fresh instances — first scripted turn on each."""
        tool_calls = [[{"name": "search", "args": {"q": "x"}, "id": "1"}], []]

        def final(model: type[BaseModel]) -> BaseModel:
            return model(items=["done"])

        sync_fake = ReActFake(tool_calls, final=final)
        async_fake = ReActFake(tool_calls, final=final)

        sync_result = sync_fake.invoke([])
        async_result = asyncio.run(async_fake.ainvoke([]))

        assert _project(sync_result) == _project(async_result)
