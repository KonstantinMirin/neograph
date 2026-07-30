"""Structured-strategy re-prompt retry on Pydantic ValidationError (neograph-zcxd).

The ``structured`` output strategy did not re-prompt when a provider returned
structurally-valid JSON that failed Pydantic model validation (a weakly-enforced
constrained decode, e.g. z-ai/glm-5.1 via OpenRouter). json_mode routes through
``_invoke_json_with_retry`` and re-prompts; structured surfaced the
ValidationError and killed the node/round. These tests pin the parity fix:

- a validation failure surfaces as ``Failed(error=ValidationError)`` from the
  compat chain (both the include_raw=True dict form AND the no-include_raw
  fallback form),
- the dispatch layer re-prompts the SAME structured adapter with the validation
  error appended as a repair hint, bounded by ``max_retries``,
- exhaustion raises ExecutionError; a first-try Parsed makes zero extra calls,
- provider-rejection Failed (TypeError) keeps its current handling (no retry).

Coverage is pinned at the SHARED seam (``_call_structured`` / ``_acall_structured``
in ``_llm_dispatch``) which BOTH the think-mode primary path and the agent-mode
structured fallback funnel through, plus a think-mode and an agent-mode
integration cell.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from neograph import ExecutionError
from neograph._llm_config import LlmConfig
from neograph._llm_dispatch import _acall_structured, _call_structured
from tests.schemas import Claims


def _make_validation_error(model: type) -> ValidationError:
    """Produce a REAL ValidationError for *model* (missing required fields)."""
    try:
        model.model_validate({})
    except ValidationError as exc:
        return exc
    raise AssertionError(f"{model.__name__} unexpectedly validated an empty dict")


class WeakIncludeRawFake:
    """Provider that ACCEPTS include_raw=True but weakly enforces the schema.

    ``with_structured_output(model, include_raw=True).invoke()`` returns
    ``{"parsed": None, "raw": ..., "parsing_error": ValidationError}`` for the
    first ``fail_times`` calls (structurally-valid JSON that fails validation),
    then a ``{"parsed": <valid model>, "raw": ...}`` dict. Mirrors ChatOpenAI /
    OpenRouter include_raw behavior. Counter + seen-messages are shared across
    ``with_structured_output`` clones so retries accumulate on one object.
    """

    def __init__(
        self,
        valid_factory,
        *,
        fail_times: int = 1,
        counter: list[int] | None = None,
        seen: list | None = None,
        model: type | None = None,
    ):
        self._valid_factory = valid_factory
        self._fail_times = fail_times
        self._counter = counter if counter is not None else [0]
        self.seen = seen if seen is not None else []
        self._model = model

    def with_structured_output(self, model, **kwargs):
        return WeakIncludeRawFake(
            self._valid_factory,
            fail_times=self._fail_times,
            counter=self._counter,
            seen=self.seen,
            model=model,
        )

    @property
    def call_count(self) -> int:
        return self._counter[0]

    def invoke(self, messages, **kwargs):
        self.seen.append(messages)
        n = self._counter[0]
        self._counter[0] += 1
        raw = AIMessage(content='{"wrong": "shape"}')
        if n < self._fail_times:
            return {"parsed": None, "raw": raw, "parsing_error": _make_validation_error(self._model)}
        return {"parsed": self._valid_factory(self._model), "raw": AIMessage(content="ok")}

    async def ainvoke(self, *a, **k):
        return self.invoke(*a, **k)


class WeakNoIncludeRawFake:
    """Provider that REJECTS include_raw=True (TypeError) then weakly enforces.

    The compat ``IncludeRawCompatDecorator`` retries without include_raw; that
    runnable raises ValidationError for the first ``fail_times`` calls, then
    returns a valid model. Exercises the Failed(error=ValidationError) surfacing
    via the no-include_raw fallback.
    """

    def __init__(
        self,
        valid_factory,
        *,
        fail_times: int = 1,
        counter: list[int] | None = None,
        seen: list | None = None,
        model: type | None = None,
    ):
        self._valid_factory = valid_factory
        self._fail_times = fail_times
        self._counter = counter if counter is not None else [0]
        self.seen = seen if seen is not None else []
        self._model = model

    def with_structured_output(self, model, **kwargs):
        if kwargs.get("include_raw"):
            raise TypeError("include_raw=True not supported by this provider")
        return WeakNoIncludeRawFake(
            self._valid_factory,
            fail_times=self._fail_times,
            counter=self._counter,
            seen=self.seen,
            model=model,
        )

    @property
    def call_count(self) -> int:
        return self._counter[0]

    def invoke(self, messages, **kwargs):
        self.seen.append(messages)
        n = self._counter[0]
        self._counter[0] += 1
        if n < self._fail_times:
            raise _make_validation_error(self._model)
        return self._valid_factory(self._model)

    async def ainvoke(self, *a, **k):
        return self.invoke(*a, **k)


class RejectingFake:
    """Provider that rejects include_raw AND the fallback structured call with a
    non-validation TypeError — a genuine provider rejection, NOT a validation
    failure. Must NOT be retried; keeps the existing dispatch-failed handling."""

    def with_structured_output(self, model, **kwargs):
        raise TypeError("with_structured_output not supported at all")

    def invoke(self, messages, **kwargs):
        raise TypeError("bare invoke not supported")

    async def ainvoke(self, *a, **k):
        return self.invoke(*a, **k)


def _cfg(max_retries: int = 1) -> LlmConfig:
    return LlmConfig(output_strategy="structured", max_retries=max_retries)


def _last_user_content(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# Shared seam — include_raw=True surfacing (Case A: Raw/parsing_error path)
# ═══════════════════════════════════════════════════════════════════════════


class TestStructuredRetryIncludeRawSync:
    def test_reprompts_once_and_returns_valid_when_first_decode_fails_validation(self):
        fake = WeakIncludeRawFake(lambda m: m(items=["fixed"]), fail_times=1)
        messages = [{"role": "user", "content": "extract claims"}]

        result, _usage = _call_structured(
            fake,
            messages,
            Claims,
            "structured",
            {"configurable": {}},
            cfg=_cfg(max_retries=1),
            max_retries=1,
        )

        assert isinstance(result, Claims)
        assert result.items == ["fixed"]
        assert fake.call_count == 2, "expected exactly one re-prompt (2 total calls)"
        # The retry carries a repair hint referencing the validation failure.
        assert "validation" in _last_user_content(fake.seen[1]).lower()

    def test_first_try_parsed_makes_zero_extra_calls(self):
        fake = WeakIncludeRawFake(lambda m: m(items=["ok"]), fail_times=0)
        result, _usage = _call_structured(
            fake,
            [{"role": "user", "content": "x"}],
            Claims,
            "structured",
            {"configurable": {}},
            cfg=_cfg(max_retries=2),
            max_retries=2,
        )
        assert isinstance(result, Claims)
        assert fake.call_count == 1, "happy path must not re-prompt"

    def test_max_retries_honored_then_exhaustion_raises(self):
        # Never recovers; max_retries=2 -> 3 total attempts, then ExecutionError.
        fake = WeakIncludeRawFake(lambda m: m(items=["never"]), fail_times=99)
        with pytest.raises(ExecutionError):
            _call_structured(
                fake,
                [{"role": "user", "content": "x"}],
                Claims,
                "structured",
                {"configurable": {}},
                cfg=_cfg(max_retries=2),
                max_retries=2,
            )
        assert fake.call_count == 3, "max_retries=2 -> 1 initial + 2 re-prompts"


class TestStructuredRetryIncludeRawAsync:
    async def test_reprompts_once_and_returns_valid(self):
        fake = WeakIncludeRawFake(lambda m: m(items=["fixed"]), fail_times=1)
        result, _usage = await _acall_structured(
            fake,
            [{"role": "user", "content": "x"}],
            Claims,
            "structured",
            {"configurable": {}},
            cfg=_cfg(max_retries=1),
            max_retries=1,
        )
        assert isinstance(result, Claims)
        assert result.items == ["fixed"]
        assert fake.call_count == 2

    async def test_first_try_parsed_makes_zero_extra_calls(self):
        fake = WeakIncludeRawFake(lambda m: m(items=["ok"]), fail_times=0)
        result, _usage = await _acall_structured(
            fake,
            [{"role": "user", "content": "x"}],
            Claims,
            "structured",
            {"configurable": {}},
            cfg=_cfg(max_retries=2),
            max_retries=2,
        )
        assert isinstance(result, Claims)
        assert fake.call_count == 1

    async def test_max_retries_honored_then_exhaustion_raises(self):
        fake = WeakIncludeRawFake(lambda m: m(items=["never"]), fail_times=99)
        with pytest.raises(ExecutionError):
            await _acall_structured(
                fake,
                [{"role": "user", "content": "x"}],
                Claims,
                "structured",
                {"configurable": {}},
                cfg=_cfg(max_retries=2),
                max_retries=2,
            )
        assert fake.call_count == 3


# ═══════════════════════════════════════════════════════════════════════════
# Shared seam — no-include_raw fallback surfacing (Case B: Failed path)
# ═══════════════════════════════════════════════════════════════════════════


class TestStructuredRetryNoIncludeRaw:
    def test_reprompts_once_sync(self):
        fake = WeakNoIncludeRawFake(lambda m: m(items=["fixed"]), fail_times=1)
        result, _usage = _call_structured(
            fake,
            [{"role": "user", "content": "x"}],
            Claims,
            "structured",
            {"configurable": {}},
            cfg=_cfg(max_retries=1),
            max_retries=1,
        )
        assert isinstance(result, Claims)
        assert result.items == ["fixed"]
        assert fake.call_count == 2

    async def test_reprompts_once_async(self):
        fake = WeakNoIncludeRawFake(lambda m: m(items=["fixed"]), fail_times=1)
        result, _usage = await _acall_structured(
            fake,
            [{"role": "user", "content": "x"}],
            Claims,
            "structured",
            {"configurable": {}},
            cfg=_cfg(max_retries=1),
            max_retries=1,
        )
        assert isinstance(result, Claims)
        assert fake.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# Discrimination — provider rejection (non-validation) is NOT retried
# ═══════════════════════════════════════════════════════════════════════════


class TestProviderRejectionNotRetried:
    def test_typeerror_rejection_raises_dispatch_failed_without_retry_sync(self):
        fake = RejectingFake()
        with pytest.raises(ExecutionError):
            _call_structured(
                fake,
                [{"role": "user", "content": "x"}],
                Claims,
                "structured",
                {"configurable": {}},
                cfg=_cfg(max_retries=3),
                max_retries=3,
            )

    async def test_typeerror_rejection_raises_dispatch_failed_without_retry_async(self):
        fake = RejectingFake()
        with pytest.raises(ExecutionError):
            await _acall_structured(
                fake,
                [{"role": "user", "content": "x"}],
                Claims,
                "structured",
                {"configurable": {}},
                cfg=_cfg(max_retries=3),
                max_retries=3,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Integration — think-mode primary path (invoke_structured / ainvoke_structured)
# ═══════════════════════════════════════════════════════════════════════════


class TestThinkModeStructuredRetry:
    def test_think_mode_reprompts_and_returns_valid_sync(self):
        from neograph._llm import invoke_structured
        from tests.fakes import build_fake_runtime

        fake = WeakIncludeRawFake(lambda m: m(items=["fixed"]), fail_times=1)
        runtime = build_fake_runtime(factory=lambda tier: fake)
        result = invoke_structured(
            runtime,
            model_tier="reason",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            config={"configurable": {}},
            node_name="hypothesize",
            llm_config={"output_strategy": "structured", "max_retries": 1},
        )
        assert isinstance(result, Claims)
        assert result.items == ["fixed"]
        assert fake.call_count == 2

    async def test_think_mode_reprompts_and_returns_valid_async(self):
        from neograph._llm import ainvoke_structured
        from tests.fakes import build_fake_runtime

        fake = WeakIncludeRawFake(lambda m: m(items=["fixed"]), fail_times=1)
        runtime = build_fake_runtime(factory=lambda tier: fake)
        result = await ainvoke_structured(
            runtime,
            model_tier="reason",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            config={"configurable": {}},
            node_name="hypothesize",
            llm_config={"output_strategy": "structured", "max_retries": 1},
        )
        assert isinstance(result, Claims)
        assert result.items == ["fixed"]
        assert fake.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# Integration — agent-mode structured fallback (_parse_final_turn twins)
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentModeStructuredFallbackRetry:
    """The ReAct final-turn parse falls back to the structured strategy when the
    turn text is not valid JSON; that fallback must inherit the re-prompt retry."""

    def test_agent_fallback_reprompts_and_returns_valid_sync(self):
        from neograph._tool_loop import _parse_final_turn

        fake = WeakIncludeRawFake(lambda m: m(items=["fixed"]), fail_times=1)
        messages = [
            {"role": "user", "content": "do the task"},
            AIMessage(content="not json at all"),
        ]
        result, _usage = _parse_final_turn(
            messages=messages,
            output_model=Claims,
            cfg=_cfg(max_retries=1),
            config={"configurable": {}},
            llm=fake,
        )
        assert isinstance(result, Claims)
        assert result.items == ["fixed"]
        assert fake.call_count == 2

    async def test_agent_fallback_reprompts_and_returns_valid_async(self):
        from neograph._tool_loop import _aparse_final_turn

        fake = WeakIncludeRawFake(lambda m: m(items=["fixed"]), fail_times=1)
        messages = [
            {"role": "user", "content": "do the task"},
            AIMessage(content="not json at all"),
        ]
        result, _usage = await _aparse_final_turn(
            messages=messages,
            output_model=Claims,
            cfg=_cfg(max_retries=1),
            config={"configurable": {}},
            llm=fake,
        )
        assert isinstance(result, Claims)
        assert result.items == ["fixed"]
        assert fake.call_count == 2

    def test_agent_fallback_exhaustion_raises_sync(self):
        from neograph._tool_loop import _parse_final_turn

        fake = WeakIncludeRawFake(lambda m: m(items=["never"]), fail_times=99)
        messages = [
            {"role": "user", "content": "do the task"},
            AIMessage(content="not json at all"),
        ]
        with pytest.raises(ExecutionError):
            _parse_final_turn(
                messages=messages,
                output_model=Claims,
                cfg=_cfg(max_retries=2),
                config={"configurable": {}},
                llm=fake,
            )
        assert fake.call_count == 3


# ═══════════════════════════════════════════════════════════════════════════
# Empty / undecodable response — the Raw (parsed=None, no DSML) arm
# (neograph-yqrsz)
# ═══════════════════════════════════════════════════════════════════════════
#
# An EMPTY provider response has nothing to validate, so it never becomes
# Failed(ValidationError) — the compat shim classifies it as Raw(dsml=False)
# and the dispatch layer raised on the FIRST occurrence, never consulting
# max_retries. That gave the single most likely transient provider failure
# ZERO retries while a less likely one (weak constrained decode) got
# max_retries. Downstream this cost 70 of 192 runs of one A/B arm and
# published a wrong conclusion (see the bead).
#
# Fail-loud stays: the budget is bounded and exhaustion still raises. These
# pin that the raise happens only AFTER the same re-prompt budget the
# ValidationError arm already gets.


class EmptyThenValidFake:
    """Provider that returns an EMPTY response, then a valid parse.

    ``with_structured_output(model, include_raw=True).invoke()`` returns
    ``{"parsed": None, "raw": AIMessage(content="")}`` — no ``parsing_error``,
    because an empty body never reached validation — for the first
    ``fail_times`` calls, then a valid ``{"parsed": <model>, ...}``. This is
    the shape a provider emits during an empty-response incident (an empty
    200), reproduced from a real one. Counter + seen-messages are shared
    across ``with_structured_output`` clones so retries accumulate.
    """

    def __init__(
        self,
        valid_factory,
        *,
        fail_times: int = 1,
        empty_content: str = "",
        counter: list[int] | None = None,
        seen: list | None = None,
        model: type | None = None,
    ):
        self._valid_factory = valid_factory
        self._fail_times = fail_times
        self._empty_content = empty_content
        self._counter = counter if counter is not None else [0]
        self.seen = seen if seen is not None else []
        self._model = model

    def with_structured_output(self, model, **kwargs):
        return EmptyThenValidFake(
            self._valid_factory,
            fail_times=self._fail_times,
            empty_content=self._empty_content,
            counter=self._counter,
            seen=self.seen,
            model=model,
        )

    @property
    def call_count(self) -> int:
        return self._counter[0]

    def invoke(self, messages, **kwargs):
        self.seen.append(messages)
        n = self._counter[0]
        self._counter[0] += 1
        if n < self._fail_times:
            return {"parsed": None, "raw": AIMessage(content=self._empty_content)}
        return {"parsed": self._valid_factory(self._model), "raw": AIMessage(content="ok")}

    async def ainvoke(self, *a, **k):
        return self.invoke(*a, **k)


class TestEmptyResponseIsRetriedSync:
    def test_reprompts_once_and_returns_valid_when_first_response_is_empty(self):
        fake = EmptyThenValidFake(lambda m: m(items=["fixed"]), fail_times=1)

        result, _usage = _call_structured(
            fake,
            [{"role": "user", "content": "extract claims"}],
            Claims,
            "structured",
            {"configurable": {}},
            cfg=_cfg(max_retries=1),
            max_retries=1,
        )

        assert isinstance(result, Claims)
        assert result.items == ["fixed"]
        assert fake.call_count == 2, "an empty response must be re-prompted, not raised on"

    def test_max_retries_honored_then_exhaustion_raises(self):
        fake = EmptyThenValidFake(lambda m: m(items=["never"]), fail_times=99)
        with pytest.raises(ExecutionError):
            _call_structured(
                fake,
                [{"role": "user", "content": "x"}],
                Claims,
                "structured",
                {"configurable": {}},
                cfg=_cfg(max_retries=2),
                max_retries=2,
            )
        assert fake.call_count == 3, "max_retries=2 -> 1 initial + 2 re-prompts, THEN raise"

    def test_whitespace_only_response_is_also_retried(self):
        """A whitespace-only body is as undecodable as an empty one."""
        fake = EmptyThenValidFake(lambda m: m(items=["fixed"]), fail_times=1, empty_content="   \n")
        result, _usage = _call_structured(
            fake,
            [{"role": "user", "content": "x"}],
            Claims,
            "structured",
            {"configurable": {}},
            cfg=_cfg(max_retries=1),
            max_retries=1,
        )
        assert isinstance(result, Claims)
        assert fake.call_count == 2

    def test_first_try_parsed_makes_zero_extra_calls(self):
        fake = EmptyThenValidFake(lambda m: m(items=["ok"]), fail_times=0)
        result, _usage = _call_structured(
            fake,
            [{"role": "user", "content": "x"}],
            Claims,
            "structured",
            {"configurable": {}},
            cfg=_cfg(max_retries=2),
            max_retries=2,
        )
        assert isinstance(result, Claims)
        assert fake.call_count == 1, "happy path must not re-prompt"


class TestEmptyResponseIsRetriedAsync:
    async def test_reprompts_once_and_returns_valid_when_first_response_is_empty(self):
        fake = EmptyThenValidFake(lambda m: m(items=["fixed"]), fail_times=1)
        result, _usage = await _acall_structured(
            fake,
            [{"role": "user", "content": "x"}],
            Claims,
            "structured",
            {"configurable": {}},
            cfg=_cfg(max_retries=1),
            max_retries=1,
        )
        assert isinstance(result, Claims)
        assert result.items == ["fixed"]
        assert fake.call_count == 2, "async twin must retry an empty response too"

    async def test_max_retries_honored_then_exhaustion_raises(self):
        fake = EmptyThenValidFake(lambda m: m(items=["never"]), fail_times=99)
        with pytest.raises(ExecutionError):
            await _acall_structured(
                fake,
                [{"role": "user", "content": "x"}],
                Claims,
                "structured",
                {"configurable": {}},
                cfg=_cfg(max_retries=2),
                max_retries=2,
            )
        assert fake.call_count == 3


class TestEmptyResponseRepairFeedbackIsDistinct:
    """The re-prompt must diagnose EMPTINESS, not a malformed/failed validation.

    An empty response never reached validation and never produced JSON to
    malform, so 'your JSON was malformed' / 'could not be parsed' is confusing,
    actively-wrong advice to a model that returned nothing. Without this test
    the wording silently regresses to the ValidationError phrasing.
    """

    def test_retry_prompt_names_emptiness_and_not_a_validation_failure(self):
        fake = EmptyThenValidFake(lambda m: m(items=["fixed"]), fail_times=1)
        _call_structured(
            fake,
            [{"role": "user", "content": "extract claims"}],
            Claims,
            "structured",
            {"configurable": {}},
            cfg=_cfg(max_retries=1),
            max_retries=1,
        )

        feedback = _last_user_content(fake.seen[1]).lower()
        assert "empty" in feedback, "the re-prompt must name the empty response"
        assert "failed validation" not in feedback, "no validation failure occurred here"
        assert "could not be parsed" not in feedback, "nothing was emitted to parse"

    def test_empty_assistant_turn_is_not_echoed_into_the_retry(self):
        """Echoing an empty assistant turn back adds noise and can trip
        providers that reject empty-content messages."""
        fake = EmptyThenValidFake(lambda m: m(items=["fixed"]), fail_times=1)
        _call_structured(
            fake,
            [{"role": "user", "content": "extract claims"}],
            Claims,
            "structured",
            {"configurable": {}},
            cfg=_cfg(max_retries=1),
            max_retries=1,
        )

        retry_messages = fake.seen[1]
        assert not any(
            isinstance(m, dict) and m.get("role") == "assistant" and not str(m.get("content", "")).strip()
            for m in retry_messages
        ), "an empty assistant turn must not be appended to the re-prompt"

    def test_validation_arm_feedback_is_unchanged(self):
        """Discrimination: the ValidationError arm keeps its own wording."""
        fake = WeakIncludeRawFake(lambda m: m(items=["fixed"]), fail_times=1)
        _call_structured(
            fake,
            [{"role": "user", "content": "extract claims"}],
            Claims,
            "structured",
            {"configurable": {}},
            cfg=_cfg(max_retries=1),
            max_retries=1,
        )
        feedback = _last_user_content(fake.seen[1]).lower()
        assert "validation" in feedback
        assert "empty" not in feedback


# ═══════════════════════════════════════════════════════════════════════════
# Integration — empty response through think mode and the agent fallback
# ═══════════════════════════════════════════════════════════════════════════


class TestEmptyResponseThinkModeIntegration:
    """The empty-response retry survives the whole think-mode path, not just the
    dispatch seam it is implemented at."""

    def test_think_mode_reprompts_on_empty_and_returns_valid_sync(self):
        from neograph._llm import invoke_structured
        from tests.fakes import build_fake_runtime

        fake = EmptyThenValidFake(lambda m: m(items=["fixed"]), fail_times=1)
        runtime = build_fake_runtime(factory=lambda tier: fake)
        result = invoke_structured(
            runtime,
            model_tier="reason",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            config={"configurable": {}},
            node_name="hypothesize",
            llm_config={"output_strategy": "structured", "max_retries": 1},
        )
        assert isinstance(result, Claims)
        assert result.items == ["fixed"]
        assert fake.call_count == 2

    async def test_think_mode_reprompts_on_empty_and_returns_valid_async(self):
        from neograph._llm import ainvoke_structured
        from tests.fakes import build_fake_runtime

        fake = EmptyThenValidFake(lambda m: m(items=["fixed"]), fail_times=1)
        runtime = build_fake_runtime(factory=lambda tier: fake)
        result = await ainvoke_structured(
            runtime,
            model_tier="reason",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            config={"configurable": {}},
            node_name="hypothesize",
            llm_config={"output_strategy": "structured", "max_retries": 1},
        )
        assert isinstance(result, Claims)
        assert result.items == ["fixed"]
        assert fake.call_count == 2


class TestEmptyResponseAgentFallbackIntegration:
    """An agent node whose final turn is unparseable falls back to the structured
    path; an empty response there must re-prompt too. This is the exact shape the
    downstream incident hit before it moved to json_mode."""

    def test_agent_fallback_reprompts_on_empty_sync(self):
        from neograph._tool_loop import _parse_final_turn

        fake = EmptyThenValidFake(lambda m: m(items=["fixed"]), fail_times=1)
        result, _usage = _parse_final_turn(
            messages=[
                {"role": "user", "content": "do the task"},
                AIMessage(content="not json at all"),
            ],
            output_model=Claims,
            cfg=_cfg(max_retries=1),
            config={"configurable": {}},
            llm=fake,
        )
        assert isinstance(result, Claims)
        assert result.items == ["fixed"]
        assert fake.call_count == 2

    async def test_agent_fallback_reprompts_on_empty_async(self):
        from neograph._tool_loop import _aparse_final_turn

        fake = EmptyThenValidFake(lambda m: m(items=["fixed"]), fail_times=1)
        result, _usage = await _aparse_final_turn(
            messages=[
                {"role": "user", "content": "do the task"},
                AIMessage(content="not json at all"),
            ],
            output_model=Claims,
            cfg=_cfg(max_retries=1),
            config={"configurable": {}},
            llm=fake,
        )
        assert isinstance(result, Claims)
        assert fake.call_count == 2
