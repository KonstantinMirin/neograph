"""Null-to-default coercion parity between the structured and json_mode strategies.

GH #20 / neograph-5s8f6. A Pydantic default applies only when a key is ABSENT;
a present-and-null value overrides it and fails validation, and models emit
present-and-null routinely. ``_apply_null_defaults`` repairs exactly that, but
it was reachable only from ``_parse_json_response`` -- the json_mode path. The
``structured`` path (the DEFAULT strategy) went straight into LangChain's
validation and raised.

Measured on the reporter's shape before the fix::

    json_mode   -> Claim(root_cause=RootCause(changes=()))   RECOVERS
    structured  -> ValidationError                           FAILS

The house rule from GH #14 is that the two output strategies must agree about
what a declared output can be, so these tests are parametrized over both and
assert the SAME value comes back. They drive the shared dispatch seam
(``_call_structured`` / ``_acall_structured``), which both the think-mode
primary path and the agent-mode structured fallback funnel through.
"""

from __future__ import annotations

import ast
import json
import pathlib
from unittest import mock

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ValidationError

import neograph
from neograph import ExecutionError, _null_defaults
from neograph._llm_dispatch import _acall_structured, _call_structured
from neograph.testing.fakes import TextFake


class Change(BaseModel):
    path: str


class RootCause(BaseModel):
    label: str
    changes: tuple[Change, ...] = ()


class Claim(BaseModel):
    ref: str
    root_cause: RootCause


# The reporter's payload: a defaulted field ONE LEVEL DOWN given an explicit
# null. `changes` has a default, and `tuple[Change, ...]` rejects None.
NULL_PAYLOAD = '{"ref":"c1","root_cause":{"label":"x","changes":null}}'

EXPECTED = Claim(ref="c1", root_cause=RootCause(label="x", changes=()))


def _weak_decode(payload: str, model: type[BaseModel]) -> BaseModel | ValidationError:
    """What a provider's weakly-enforced decode of *payload* yields for *model*.

    A payload the model accepts comes back as the parsed instance (the provider
    would have returned ``parsed=``); anything else comes back as the REAL
    ``ValidationError``, which is what ``include_raw=True`` surfaces as
    ``parsing_error``. Content that is not even a JSON object stands in for the
    generic reject, using the same empty-dict error the rest of the suite uses.
    """
    try:
        data = json.loads(payload)
    except ValueError:
        data = {}
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        return exc


class NullEmittingIncludeRawFake:
    """Provider that accepts ``include_raw=True`` and emits a present-and-null field.

    Mirrors ChatOpenAI/OpenRouter: ``with_structured_output(model,
    include_raw=True).invoke()`` returns ``{"parsed": None, "raw": <AIMessage
    carrying the JSON>, "parsing_error": ValidationError}`` when the decoded
    JSON fails the model's validation. The payload here is well-formed JSON --
    only the null-over-default makes it fail.
    """

    def __init__(self, payload: str = NULL_PAYLOAD, *, in_tool_call: bool = False, counter=None):
        self._payload = payload
        self._in_tool_call = in_tool_call
        self._counter = counter if counter is not None else [0]
        self._model: type[BaseModel] | None = None

    def with_structured_output(self, model, **kwargs):
        clone = NullEmittingIncludeRawFake(self._payload, in_tool_call=self._in_tool_call, counter=self._counter)
        clone._model = model
        return clone

    @property
    def call_count(self) -> int:
        return self._counter[0]

    def invoke(self, messages, **kwargs):
        self._counter[0] += 1
        assert self._model is not None
        if self._in_tool_call:
            # method="function_calling": the payload rides in tool_calls.args and
            # the message content is empty.
            raw = AIMessage(
                content="",
                tool_calls=[{"name": self._model.__name__, "args": json.loads(self._payload), "id": "call_1"}],
            )
        else:
            raw = AIMessage(content=self._payload)
        decoded = _weak_decode(self._payload, self._model)
        if isinstance(decoded, ValidationError):
            return {"parsed": None, "raw": raw, "parsing_error": decoded}
        return {"parsed": decoded, "raw": raw}

    async def ainvoke(self, *a, **k):
        return self.invoke(*a, **k)


def _llm_for(strategy: str):
    return TextFake(NULL_PAYLOAD) if strategy == "json_mode" else NullEmittingIncludeRawFake()


# ═══════════════════════════════════════════════════════════════════════════
# The parity assertion
# ═══════════════════════════════════════════════════════════════════════════


class TestNullDefaultCoercionParity:
    """Both output strategies must recover a present-and-null defaulted field."""

    @pytest.mark.parametrize("strategy", ["structured", "json_mode"])
    def test_nested_null_over_default_recovers_under_every_output_strategy(self, strategy):
        result, _usage = _call_structured(
            _llm_for(strategy),
            [{"role": "user", "content": "go"}],
            Claim,
            strategy,
            {},
            max_retries=1,
        )
        assert result == EXPECTED, f"{strategy} did not coerce the nested null to the field default"

    @pytest.mark.parametrize("strategy", ["structured", "json_mode"])
    @pytest.mark.asyncio
    async def test_async_twin_recovers_under_every_output_strategy(self, strategy):
        result, _usage = await _acall_structured(
            _llm_for(strategy),
            [{"role": "user", "content": "go"}],
            Claim,
            strategy,
            {},
            max_retries=1,
        )
        assert result == EXPECTED, f"async {strategy} did not coerce the nested null to the field default"

    def test_structured_recovers_when_the_payload_rides_in_tool_call_args(self):
        """method="function_calling" leaves message content empty; the JSON is in
        ``tool_calls[0]['args']``. The recovery must read it there too."""
        fake = NullEmittingIncludeRawFake(in_tool_call=True)
        result, _usage = _call_structured(
            fake,
            [{"role": "user", "content": "go"}],
            Claim,
            "structured",
            {},
            max_retries=1,
        )
        assert result == EXPECTED
        assert fake.call_count == 1, "recovery must not cost an extra provider round-trip"

    def test_recovery_costs_no_extra_provider_call(self):
        fake = NullEmittingIncludeRawFake()
        _call_structured(fake, [{"role": "user", "content": "go"}], Claim, "structured", {}, max_retries=1)
        assert fake.call_count == 1


# ═══════════════════════════════════════════════════════════════════════════
# The coercion must not swallow anything else
# ═══════════════════════════════════════════════════════════════════════════


class TestCoercionDoesNotMaskOtherFailures:
    """Recovery fires ONLY when the null-default coercion actually changed the
    payload AND the changed payload validates. Everything else keeps the
    existing Failed -> re-prompt -> exhaust behavior."""

    def test_unrelated_validation_failure_still_re_prompts_and_still_raises(self):
        """A payload that is simply the wrong shape has no null to coerce, so it
        must stay a validation failure -- not be reclassified as a success."""
        fake = NullEmittingIncludeRawFake('{"ref":"c1"}')  # root_cause missing, no nulls
        with pytest.raises(ExecutionError, match="failed validation after"):
            _call_structured(fake, [{"role": "user", "content": "go"}], Claim, "structured", {}, max_retries=1)
        assert fake.call_count == 2, "the re-prompt budget must still be spent"

    def test_a_null_on_a_field_with_no_default_still_fails(self):
        """`ref` has no default, so a null there is unrepairable -- coercion leaves
        it alone and the failure survives."""
        fake = NullEmittingIncludeRawFake('{"ref":null,"root_cause":{"label":"x"}}')
        with pytest.raises(ExecutionError, match="failed validation after"):
            _call_structured(fake, [{"role": "user", "content": "go"}], Claim, "structured", {}, max_retries=1)

    def test_non_json_raw_content_is_left_to_the_existing_path(self):
        fake = NullEmittingIncludeRawFake("I could not produce that.")
        with pytest.raises(ExecutionError, match="failed validation after"):
            _call_structured(fake, [{"role": "user", "content": "go"}], Claim, "structured", {}, max_retries=1)


# ═══════════════════════════════════════════════════════════════════════════
# ONE coercion, not two — the acceptance's structural half
# ═══════════════════════════════════════════════════════════════════════════


class TestStrategiesShareOneCoercionImplementation:
    """The bug was a second path that did not reach the ONE coercion. The failure
    mode a fix like this invites is a second COPY of the coercion, which then
    drifts. Two pins: the semantic one (both strategies agree on a table of
    shapes -- drift shows up as disagreement) and the structural one (the
    coercion is defined exactly once in the package)."""

    # (payload, expected) pairs spanning the shapes _apply_null_defaults handles.
    SHAPES = [
        ('{"ref":"c1","root_cause":{"label":"x","changes":null}}', EXPECTED),
        # top-level default, not nested
        ('{"ref":"c1","root_cause":{"label":"x"}}', EXPECTED),
        # a populated nested tuple is untouched by the coercion
        (
            '{"ref":"c2","root_cause":{"label":"y","changes":[{"path":"a.py"}]}}',
            Claim(ref="c2", root_cause=RootCause(label="y", changes=(Change(path="a.py"),))),
        ),
    ]

    @pytest.mark.parametrize("payload,expected", SHAPES)
    def test_both_strategies_agree_shape_for_shape(self, payload, expected):
        structured, _ = _call_structured(
            NullEmittingIncludeRawFake(payload),
            [{"role": "user", "content": "go"}],
            Claim,
            "structured",
            {},
            max_retries=1,
        )
        json_mode, _ = _call_structured(
            TextFake(payload),
            [{"role": "user", "content": "go"}],
            Claim,
            "json_mode",
            {},
            max_retries=1,
        )
        assert structured == json_mode == expected

    def test_the_coercion_is_defined_exactly_once_in_the_package(self):
        """Structural: `_apply_null_defaults` has ONE definition across src/. A
        second `def` -- the drift this bug's fix would otherwise invite -- fails
        here even if every behavioral test above still passes."""
        src_dir = pathlib.Path(neograph.__file__).parent
        homes = [
            path.relative_to(src_dir).as_posix()
            for path in sorted(src_dir.rglob("*.py"))
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_apply_null_defaults"
        ]
        assert homes == ["_null_defaults.py"], f"_apply_null_defaults must be defined once; found in {homes}"

    def test_the_structured_path_calls_that_one_coercion(self):
        """Behavioral counterpart: neutralize the single implementation and the
        structured recovery must stop recovering. If the structured path had its
        own copy, this would keep passing."""
        fake = NullEmittingIncludeRawFake()
        with mock.patch.object(_null_defaults, "_apply_null_defaults", lambda data, model: None):
            with pytest.raises(ExecutionError, match="failed validation after"):
                _call_structured(fake, [{"role": "user", "content": "go"}], Claim, "structured", {}, max_retries=1)
