"""Regression tests for neograph-8uoot: json_mode repair failure + truncation.

Two defects observed live (ox-troubleshooting-demo cascade, 2026-07-14):

1. ``repair_json()`` in ``_parse_json_response`` sits OUTSIDE the guarded
   try-block, so a payload json_repair itself chokes on (deep/truncation-driven
   recursion -> ValueError/RecursionError) escapes ``_invoke_json_with_retry``
   uncaught and kills the run, bypassing the error-feedback retry built for
   exactly this malformation class.

2. A ``finish_reason == "length"`` response is KNOWN incomplete. A blind
   re-issue of the same prompt at temperature=0 very likely reproduces the
   same runaway. The correct recovery is CONTINUATION: feed the truncated
   text back and instruct the model to emit ONLY the JSON payload.

Contract pinned here:
- a repair_json failure surfaces as ExecutionError and is retried
- truncated responses with no parseable payload get a continuation re-prompt
  (carries the prior text, instructs emit-only), not the generic repair msg
- neither path lets a non-ExecutionError escape _invoke_json_with_retry
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from neograph._llm_retry import (
    _ainvoke_json_with_retry,
    _invoke_json_with_retry,
    _parse_json_response,
)
from neograph.errors import ExecutionError

# A payload that makes json_repair itself raise (RecursionError locally,
# ValueError in releases that convert it): thousands of unclosed openers,
# the same recursive-descent blowup a 290KB truncated string produced live.
REPAIR_BOMB = "[" * 4000

VALID_JSON = '{"diagnosis": "stage-3 drop", "confidence": 0.9}'

TRUNCATED_PROSE = (
    "Let me work through the funnel arithmetic carefully. Stage 1 shows 4869 "
    "deals entering, stage 2 retains 4312, and the break is clearly at sta"
)


class Diagnosis(BaseModel):
    diagnosis: str
    confidence: float


class SequenceFake:
    """LLM fake returning a scripted sequence of AIMessages, recording calls."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages, config=None, **kwargs):
        self.calls.append(list(messages))
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]

    async def ainvoke(self, messages, config=None, **kwargs):
        return self.invoke(messages, config=config, **kwargs)


def _msgs():
    return [{"role": "user", "content": "diagnose the funnel"}]


def _last_user_content(messages: list) -> str:
    last = messages[-1]
    return last["content"] if isinstance(last, dict) else last.content


def _assistant_contents(messages: list) -> list[str]:
    out = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", None)
        if role in ("assistant", "ai"):
            out.append(m["content"] if isinstance(m, dict) else m.content)
    return out


class TestRepairFailureIsRetryable:
    """Defect 1: a repair_json blowup must become ExecutionError, not escape."""

    def test_parse_json_response_wraps_repair_failure_in_execution_error(self):
        """A payload json_repair chokes on raises ExecutionError, not
        ValueError/RecursionError."""
        with pytest.raises(ExecutionError):
            _parse_json_response(REPAIR_BOMB, Diagnosis)

    def test_repair_failure_triggers_retry_and_recovers_sync(self):
        """The retry loop treats a repair failure like any malformation:
        re-prompts and returns the corrected parse."""
        fake = SequenceFake([
            AIMessage(content=REPAIR_BOMB),
            AIMessage(content=VALID_JSON),
        ])
        result, _usage = _invoke_json_with_retry(fake, _msgs(), Diagnosis, config={})
        assert result.diagnosis == "stage-3 drop"
        assert len(fake.calls) == 2

    async def test_repair_failure_triggers_retry_and_recovers_async(self):
        fake = SequenceFake([
            AIMessage(content=REPAIR_BOMB),
            AIMessage(content=VALID_JSON),
        ])
        result, _usage = await _ainvoke_json_with_retry(fake, _msgs(), Diagnosis, config={})
        assert result.diagnosis == "stage-3 drop"
        assert len(fake.calls) == 2

    def test_exhausted_retries_raise_execution_error_not_recursion_error(self):
        """Even when every attempt is a repair bomb, the loop exits with
        ExecutionError — never a raw ValueError/RecursionError."""
        fake = SequenceFake([AIMessage(content=REPAIR_BOMB)])
        with pytest.raises(ExecutionError):
            _invoke_json_with_retry(fake, _msgs(), Diagnosis, config={}, max_retries=2)

    async def test_exhausted_retries_raise_execution_error_async(self):
        fake = SequenceFake([AIMessage(content=REPAIR_BOMB)])
        with pytest.raises(ExecutionError):
            await _ainvoke_json_with_retry(fake, _msgs(), Diagnosis, config={}, max_retries=2)


class TestTruncationContinuation:
    """Defect 2: finish_reason=length with no parseable payload must get a
    continuation re-prompt (carry prior text + emit-only directive)."""

    def _truncated(self) -> AIMessage:
        return AIMessage(
            content=TRUNCATED_PROSE,
            response_metadata={"finish_reason": "length"},
        )

    def test_truncated_response_gets_continuation_reprompt_sync(self):
        fake = SequenceFake([self._truncated(), AIMessage(content=VALID_JSON)])
        result, _usage = _invoke_json_with_retry(fake, _msgs(), Diagnosis, config={})
        assert result.diagnosis == "stage-3 drop"
        assert len(fake.calls) == 2

        retry_call = fake.calls[1]
        # carries the prior (truncated) text back as the assistant turn
        assert TRUNCATED_PROSE in _assistant_contents(retry_call)
        # instructs emit-only continuation, names the target model,
        # and forbids further analysis
        follow_up = _last_user_content(retry_call)
        assert "cut off" in follow_up.lower()
        assert "ONLY the JSON" in follow_up
        assert "Diagnosis" in follow_up
        assert "do not" in follow_up.lower()

    def test_truncated_reprompt_is_not_the_generic_repair_message_sync(self):
        """A blind re-issue / generic 'could not be parsed' message would
        reproduce the same runaway; the follow-up must be continuation-shaped."""
        fake = SequenceFake([self._truncated(), AIMessage(content=VALID_JSON)])
        _invoke_json_with_retry(fake, _msgs(), Diagnosis, config={})
        follow_up = _last_user_content(fake.calls[1])
        assert "could not be parsed" not in follow_up

    async def test_truncated_response_gets_continuation_reprompt_async(self):
        fake = SequenceFake([self._truncated(), AIMessage(content=VALID_JSON)])
        result, _usage = await _ainvoke_json_with_retry(fake, _msgs(), Diagnosis, config={})
        assert result.diagnosis == "stage-3 drop"
        retry_call = fake.calls[1]
        assert TRUNCATED_PROSE in _assistant_contents(retry_call)
        follow_up = _last_user_content(retry_call)
        assert "cut off" in follow_up.lower()
        assert "ONLY the JSON" in follow_up

    def test_truncated_but_parseable_payload_still_returns_first_try(self):
        """finish_reason=length with a COMPLETE payload is not an error —
        no re-prompt, one call."""
        fake = SequenceFake([
            AIMessage(content=VALID_JSON, response_metadata={"finish_reason": "length"}),
        ])
        result, _usage = _invoke_json_with_retry(fake, _msgs(), Diagnosis, config={})
        assert result.diagnosis == "stage-3 drop"
        assert len(fake.calls) == 1

    def test_non_truncated_parse_failure_keeps_generic_repair_message(self):
        """Normal malformations (finish_reason=stop) keep the existing
        schema-bearing repair re-prompt."""
        fake = SequenceFake([
            AIMessage(content="not json at all", response_metadata={"finish_reason": "stop"}),
            AIMessage(content=VALID_JSON),
        ])
        _invoke_json_with_retry(fake, _msgs(), Diagnosis, config={})
        follow_up = _last_user_content(fake.calls[1])
        assert "cut off" not in follow_up.lower()
