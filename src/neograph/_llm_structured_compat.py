"""Provider-quirk compat shim for LangChain structured output.

``BaseChatModel.with_structured_output(model, include_raw=True)`` is polymorphic
in an awkward way: different providers return parsed models, ``{"parsed": ...,
"raw": ...}`` dicts, ``parsed=None`` with raw markup, or reject ``include_raw=``
outright with a ``TypeError``. This module normalizes that polymorphism at the
boundary into a single tagged union — ``Parsed | Raw | Failed`` — so the
dispatch site (`_llm_dispatch._call_structured`) is a ``match`` on the variant
instead of a try/except ladder.

This is the same boundary-normalization idiom the IR layer uses in
``_normalize.normalize_outputs() -> NormalizedOutputs``: classify the messy
external shape into one typed result, then let callers switch on it.

Each provider quirk is a *decorator* on a ``StructuredOutputAdapter``, composed
at construction time by :func:`build_default_adapter`. Adding support for a new
quirk is a new decorator class here — never a new branch in the dispatch.

Separation of concerns: this shim only CLASSIFIES one call's result shape. It
never re-prompts the LLM. Recovering from a ``Raw`` (re-prompting + re-parsing)
is the *retry* concern and lives in ``_llm_retry.recover_dsml``; the dispatch
layer owns any IO needed to decide on recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from neograph._dsml import contains_dsml, message_text


@dataclass(frozen=True)
class Parsed:
    """The provider returned a usable structured model."""

    model: BaseModel
    usage: Any = None


@dataclass(frozen=True)
class Raw:
    """The provider returned text but structured parsing did not yield a model.

    ``dsml`` marks raw text that contains DSML/XML tool-call markup, eligible
    for targeted content-level recovery. ``raw_text`` holds the full output so
    the retry layer can act on it.
    """

    raw_text: str
    dsml: bool = False
    parsing_error: Exception | None = None
    usage: Any = None


@dataclass(frozen=True)
class Failed:
    """The provider rejected the request entirely (e.g. ``include_raw=True`` not
    supported). ``raw_text`` carries any trailing-message text captured while the
    request failed, so a downstream classifier can still inspect it."""

    error: Exception
    raw_text: str | None = None


StructuredResult = Parsed | Raw | Failed


@runtime_checkable
class StructuredOutputAdapter(Protocol):
    """Invokes an LLM for structured output and returns a typed result.

    Implementations and their decorators encapsulate provider-specific quirks.
    """

    def invoke(
        self, llm: Any, model: type[BaseModel], messages: list, config: Any,
    ) -> StructuredResult: ...

    async def ainvoke(
        self, llm: Any, model: type[BaseModel], messages: list, config: Any,
    ) -> StructuredResult: ...


def _classify_lc_result(result: Any) -> StructuredResult:
    """Pure classifier for a with_structured_output(include_raw=True) return.

    Shared by LangChainStructuredAdapter.invoke and .ainvoke so the sync and
    async paths cannot drift — the only difference between them is the awaited
    ``.ainvoke`` network call that produces ``result``.
    """
    if isinstance(result, dict) and "parsed" in result:
        parsed = result.get("parsed")
        raw_msg = result.get("raw")
        usage = getattr(raw_msg, "usage_metadata", None) if raw_msg is not None else None
        if parsed is not None:
            return Parsed(model=parsed, usage=usage)
        return Raw(
            raw_text=message_text(raw_msg),
            parsing_error=result.get("parsing_error"),
            usage=usage,
        )
    return Parsed(model=result)


class LangChainStructuredAdapter:
    """Canonical adapter: ``with_structured_output(model, include_raw=True)``.

    Normalizes the raw return into ``Parsed``/``Raw``/``Failed`` without any
    content inspection (DSML detection is a separate decorator's job).
    """

    def invoke(self, llm, model, messages, config) -> StructuredResult:
        try:
            structured = llm.with_structured_output(model, include_raw=True)
            result = structured.invoke(messages, config=config)
        except TypeError as exc:
            return Failed(error=exc, raw_text=message_text(messages[-1]) if messages else None)
        return _classify_lc_result(result)

    async def ainvoke(self, llm, model, messages, config) -> StructuredResult:
        try:
            structured = llm.with_structured_output(model, include_raw=True)
            result = await structured.ainvoke(messages, config=config)
        except TypeError as exc:
            return Failed(error=exc, raw_text=message_text(messages[-1]) if messages else None)
        return _classify_lc_result(result)


class IncludeRawCompatDecorator:
    """Quirk: provider rejects ``include_raw=True`` with ``TypeError``.

    On a ``Failed(TypeError)`` from the inner adapter, retries
    ``with_structured_output(model)`` without ``include_raw``. Success yields
    ``Parsed``; a second ``TypeError`` stays ``Failed`` (carrying trailing
    text for the DSML classifier).
    """

    def __init__(self, inner: StructuredOutputAdapter):
        self._inner = inner

    def invoke(self, llm, model, messages, config) -> StructuredResult:
        result = self._inner.invoke(llm, model, messages, config)
        if not (isinstance(result, Failed) and isinstance(result.error, TypeError)):
            return result
        try:
            structured = llm.with_structured_output(model)
            parsed = structured.invoke(messages, config=config)
            return Parsed(model=parsed)
        except TypeError as exc:
            return Failed(error=exc, raw_text=result.raw_text)

    async def ainvoke(self, llm, model, messages, config) -> StructuredResult:
        result = await self._inner.ainvoke(llm, model, messages, config)
        if not (isinstance(result, Failed) and isinstance(result.error, TypeError)):
            return result
        try:
            structured = llm.with_structured_output(model)
            parsed = await structured.ainvoke(messages, config=config)
            return Parsed(model=parsed)
        except TypeError as exc:
            return Failed(error=exc, raw_text=result.raw_text)


def _reclassify_dsml(result: StructuredResult) -> StructuredResult:
    """Pure DSML reclassification of an already-in-hand result (no network).

    Shared by DsmlClassifierDecorator.invoke and .ainvoke.
    """
    if isinstance(result, Raw) and not result.dsml and contains_dsml(result.raw_text):
        return Raw(
            raw_text=result.raw_text,
            dsml=True,
            parsing_error=result.parsing_error,
            usage=result.usage,
        )
    if isinstance(result, Failed) and result.raw_text and contains_dsml(result.raw_text):
        return Raw(raw_text=result.raw_text, dsml=True)
    return result


class DsmlClassifierDecorator:
    """Quirk: provider emits DSML/XML tool-call markup instead of JSON.

    Inspects text ALREADY in hand — never re-invokes the LLM. A ``Raw`` or a
    ``Failed`` whose captured text contains DSML markup is reclassified as
    ``Raw(dsml=True)`` so the dispatch layer can route it to content-level
    recovery. This unifies the "silent ``parsed=None`` + DSML" and the
    "``include_raw`` TypeError caused by DSML content" cases into one path.
    """

    def __init__(self, inner: StructuredOutputAdapter):
        self._inner = inner

    def invoke(self, llm, model, messages, config) -> StructuredResult:
        result = self._inner.invoke(llm, model, messages, config)
        return _reclassify_dsml(result)

    async def ainvoke(self, llm, model, messages, config) -> StructuredResult:
        result = await self._inner.ainvoke(llm, model, messages, config)
        return _reclassify_dsml(result)


def build_default_adapter() -> StructuredOutputAdapter:
    """Standard chain: LangChain base + include_raw compat + DSML classification."""
    return DsmlClassifierDecorator(
        IncludeRawCompatDecorator(
            LangChainStructuredAdapter(),
        ),
    )
