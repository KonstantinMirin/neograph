"""Hypothesis property-based tests for _parse_json_response robustness.

Generates random Pydantic models and malformed JSON responses to verify
that the parsing layer either returns a valid model or raises ExecutionError
— never crashes with an unhandled exception.

TASK neograph-f0vp: zero Hypothesis coverage for LLM output parsing.
"""

from __future__ import annotations

import json

import hypothesis.strategies as st
from hypothesis import given, settings
from pydantic import BaseModel, Field

from neograph._llm_retry import _apply_null_defaults, _build_retry_msg, _parse_json_response
from neograph.errors import ExecutionError

# ── Fixed test models (varying complexity) ────────────────────────────────


class Simple(BaseModel):
    name: str
    score: float


class WithDefaults(BaseModel):
    name: str
    note: str = Field(default="")
    count: int = 0
    active: bool = True


class Nested(BaseModel):
    label: str
    inner: Simple


class WithList(BaseModel):
    items: list[Simple]


class DeepNested(BaseModel):
    title: str = ""
    sections: list[Nested] = Field(default_factory=list)


class MixedDefaults(BaseModel):
    required_str: str
    required_int: int
    optional_note: str = ""
    optional_score: float = 0.0
    optional_flag: bool = False


ALL_MODELS = [Simple, WithDefaults, Nested, WithList, DeepNested, MixedDefaults]


# ── Strategies ────────────────────────────────────────────────────────────


def st_simple_instance() -> st.SearchStrategy[dict]:
    return st.fixed_dictionaries(
        {
            "name": st.text(min_size=1, max_size=20),
            "score": st.floats(min_value=-100, max_value=100, allow_nan=False),
        }
    )


def st_with_defaults_instance() -> st.SearchStrategy[dict]:
    return st.fixed_dictionaries(
        {
            "name": st.text(min_size=1, max_size=20),
            "note": st.one_of(st.text(max_size=50), st.just(None)),
            "count": st.one_of(st.integers(min_value=0, max_value=1000), st.just(None)),
            "active": st.one_of(st.booleans(), st.just(None)),
        }
    )


def st_nested_instance() -> st.SearchStrategy[dict]:
    return st.fixed_dictionaries(
        {
            "label": st.text(min_size=1, max_size=20),
            "inner": st_simple_instance(),
        }
    )


def st_with_list_instance() -> st.SearchStrategy[dict]:
    return st.fixed_dictionaries(
        {
            "items": st.lists(st_simple_instance(), min_size=0, max_size=5),
        }
    )


@st.composite
def st_corrupt_json(draw, base_strategy, corruption_type=None):
    """Take a valid dict and apply a random corruption."""
    data = draw(base_strategy)
    json_str = json.dumps(data)

    corruption = corruption_type or draw(
        st.sampled_from(
            [
                "wrap_markdown",
                "add_trailing_comma",
                "truncate",
                "nullify_random_field",
                "add_preamble",
                "stringify_number",
            ]
        )
    )

    if corruption == "wrap_markdown":
        return f"```json\n{json_str}\n```"
    elif corruption == "add_trailing_comma":
        # Add trailing comma before last }
        idx = json_str.rfind("}")
        if idx > 0:
            return json_str[:idx] + "," + json_str[idx:]
        return json_str
    elif corruption == "truncate":
        # Cut off last 1-10 chars
        cut = draw(st.integers(min_value=1, max_value=min(10, len(json_str) // 2)))
        return json_str[:-cut]
    elif corruption == "nullify_random_field":
        if isinstance(data, dict) and data:
            key = draw(st.sampled_from(list(data.keys())))
            data[key] = None
            return json.dumps(data)
        return json_str
    elif corruption == "add_preamble":
        preamble = draw(
            st.sampled_from(
                [
                    "Here is the JSON:\n",
                    "Sure! Let me help:\n\n",
                    "Based on my analysis:\n",
                ]
            )
        )
        return preamble + json_str
    elif corruption == "stringify_number":
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    data[k] = str(v)
                    break
            return json.dumps(data)
        return json_str
    return json_str


# ── Property tests ────────────────────────────────────────────────────────


class TestParseNeverCrashes:
    """_parse_json_response must either return a valid model or raise ExecutionError."""

    @given(data=st_simple_instance())
    @settings(max_examples=50)
    def test_valid_simple_always_parses(self, data):
        """Valid Simple JSON always produces a Simple instance."""
        text = json.dumps(data)
        result = _parse_json_response(text, Simple)
        assert isinstance(result, Simple)
        assert result.name == data["name"]

    @given(data=st_with_defaults_instance())
    @settings(max_examples=50)
    def test_nulls_coerced_to_defaults(self, data):
        """Null values for defaulted fields become the default, not failures."""
        text = json.dumps(data)
        result = _parse_json_response(text, WithDefaults)
        assert isinstance(result, WithDefaults)
        # Null fields should have their defaults
        if data["note"] is None:
            assert result.note == ""
        if data["count"] is None:
            assert result.count == 0
        if data["active"] is None:
            assert result.active is True

    @given(text=st_corrupt_json(st_simple_instance()))
    @settings(max_examples=100)
    def test_corrupted_simple_never_crashes(self, text):
        """Corrupted JSON either parses or raises ExecutionError — never crashes."""
        try:
            result = _parse_json_response(text, Simple)
            assert isinstance(result, Simple)
        except ExecutionError:
            pass  # expected for truly broken input

    @given(text=st_corrupt_json(st_nested_instance()))
    @settings(max_examples=50)
    def test_corrupted_nested_never_crashes(self, text):
        """Corrupted nested JSON either parses or raises ExecutionError."""
        try:
            result = _parse_json_response(text, Nested)
            assert isinstance(result, Nested)
        except ExecutionError:
            pass

    @given(text=st_corrupt_json(st_with_list_instance()))
    @settings(max_examples=50)
    def test_corrupted_list_never_crashes(self, text):
        """Corrupted list JSON either parses or raises ExecutionError."""
        try:
            result = _parse_json_response(text, WithList)
            assert isinstance(result, WithList)
        except ExecutionError:
            pass


class TestApplyNullDefaultsIdempotent:
    """_apply_null_defaults applied twice gives same result as once."""

    @given(data=st_with_defaults_instance())
    @settings(max_examples=50)
    def test_idempotent(self, data):
        """Applying null defaults twice is the same as once."""
        import copy

        d1 = copy.deepcopy(data)
        _apply_null_defaults(d1, WithDefaults)
        d2 = copy.deepcopy(d1)
        _apply_null_defaults(d2, WithDefaults)
        assert d1 == d2


# ── Stringly-null chaos across Optional/nested/list topologies ─────────────
#
# neograph-zhwgh: the stringly-"null" -> None coercion must reach scalar leaves
# at ANY depth, including interiors reached only THROUGH an Optional-wrapped
# nested model (``Mid | None``) or Optional-wrapped list-of-models
# (``list[Leaf] | None``). These property tests randomize the topology (list
# lengths, presence/absence of the optional members) and assert the invariant
# holds across the whole shape, not just the two hand-picked regression cases.


class ChaosLeaf(BaseModel):
    label: str
    n: int | None = None  # Optional scalar -- the sentinel target


class ChaosMid(BaseModel):
    name: str
    solo: ChaosLeaf | None = None  # Optional NESTED MODEL
    kids: list[ChaosLeaf] | None = None  # Optional LIST-OF-MODELS
    maybe_kids: list[ChaosLeaf | None] | None = None  # LIST-OF-OPTIONAL-MODELS
    registry: dict[str, ChaosLeaf] | None = None  # Optional DICT-OF-MODELS


class ChaosRoot(BaseModel):
    title: str
    lead: ChaosMid | None = None  # Optional nested model (2 levels deep)
    mids: list[ChaosMid] | None = None  # Optional list-of-models (2 levels deep)


_SENTINELS = ["null", "none", "nil", "N/A", "NA", "None", "NULL"]


@st.composite
def st_leaf_json(draw, *, sentinel: bool):
    """A ChaosLeaf dict whose Optional int ``n`` is either a legit int or a
    stringly-null sentinel string (when ``sentinel``)."""
    n = draw(st.sampled_from(_SENTINELS)) if sentinel else draw(st.integers(-1000, 1000))
    return {"label": draw(st.text(min_size=1, max_size=8)), "n": n}


@st.composite
def st_mid_json(draw, *, sentinel: bool):
    d = {"name": draw(st.text(min_size=1, max_size=8))}
    if draw(st.booleans()):
        d["solo"] = draw(st_leaf_json(sentinel=sentinel))
    if draw(st.booleans()):
        d["kids"] = [draw(st_leaf_json(sentinel=sentinel)) for _ in range(draw(st.integers(0, 3)))]
    if draw(st.booleans()):
        # list[ChaosLeaf | None]: mix real leaves with genuine null elements
        d["maybe_kids"] = [
            None if draw(st.booleans()) else draw(st_leaf_json(sentinel=sentinel))
            for _ in range(draw(st.integers(0, 3)))
        ]
    if draw(st.booleans()):
        d["registry"] = {f"k{i}": draw(st_leaf_json(sentinel=sentinel)) for i in range(draw(st.integers(0, 3)))}
    return d


@st.composite
def st_root_json(draw, *, sentinel: bool):
    d = {"title": draw(st.text(min_size=1, max_size=8))}
    if draw(st.booleans()):
        d["lead"] = draw(st_mid_json(sentinel=sentinel))
    if draw(st.booleans()):
        d["mids"] = [draw(st_mid_json(sentinel=sentinel)) for _ in range(draw(st.integers(0, 3)))]
    return d


def _all_leaf_ns(root: ChaosRoot) -> list[int | None]:
    """Every ``ChaosLeaf.n`` reachable in a parsed ChaosRoot, at any depth."""
    ns: list[int | None] = []

    def visit_leaf(leaf: ChaosLeaf | None) -> None:
        if leaf is not None:
            ns.append(leaf.n)

    def visit_mid(mid: ChaosMid | None) -> None:
        if mid is None:
            return
        visit_leaf(mid.solo)
        for k in mid.kids or []:
            visit_leaf(k)
        for k in mid.maybe_kids or []:
            visit_leaf(k)  # None elements are skipped inside visit_leaf
        for k in (mid.registry or {}).values():
            visit_leaf(k)

    visit_mid(root.lead)
    for m in root.mids or []:
        visit_mid(m)
    return ns


class TestStringlyNullChaosAcrossTopologies:
    """Stringly-null coercion reaches every Optional scalar leaf at any depth."""

    @given(data=st_root_json(sentinel=True))
    @settings(max_examples=200)
    def test_all_sentinel_leaves_coerce_to_none_never_crash(self, data):
        """Every Optional int leaf carrying a stringly-null sentinel -- at any
        nesting depth, including through Optional-wrapped models/lists -- coerces
        to None; the parse never raises (all sentinel positions are nullable)."""
        text = json.dumps(data)
        result = _parse_json_response(text, ChaosRoot)
        assert isinstance(result, ChaosRoot)
        assert all(n is None for n in _all_leaf_ns(result))

    @given(data=st_root_json(sentinel=False))
    @settings(max_examples=200)
    def test_legit_interiors_preserved_through_optional_wrappers(self, data):
        """Legit interior int values survive the Optional-unwrapping descent --
        the coercion never destroys real data."""
        text = json.dumps(data)
        result = _parse_json_response(text, ChaosRoot)
        assert isinstance(result, ChaosRoot)
        assert all(isinstance(n, int) for n in _all_leaf_ns(result))


class TestRetryMsgAlwaysProducesOutput:
    """_build_retry_msg never returns empty string."""

    @given(model=st.sampled_from(ALL_MODELS))
    @settings(max_examples=20)
    def test_retry_msg_with_model_nonempty(self, model):
        """Retry message with any model produces non-empty output."""
        err = ExecutionError("test")
        err.validation_errors = "some.field: Input should be a valid string"
        msg = _build_retry_msg(err, output_model=model)
        assert len(msg) > 50
        assert "schema" in msg.lower() or "fix" in msg.lower()

    def test_retry_msg_without_model_nonempty(self):
        """Retry message without model still produces useful output."""
        err = ExecutionError("test")
        msg = _build_retry_msg(err)
        assert len(msg) > 20
