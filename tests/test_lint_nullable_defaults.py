"""lint(): a MODEL-AUTHORED field that carries a default AND rejects null.

GH #20 / neograph-bek2f. A Pydantic default applies only when the key is
ABSENT. A present-and-null value overrides it and fails validation -- and
``describe_type`` tells the model it MAY send null for exactly these fields
(``describe_type.py:367`` renders `` or null`` when
``not field_info.is_required() and not _admits_none(field_info.annotation)``).
So the schema neograph ships instructs the model to emit a value the declared
type cannot hold.

THE POINT OF THIS FILE, in the reporter's own words: their first attempt at
this check walked TOP-LEVEL fields only and reported clean on a tree that
already contained the defect -- "an instrument passing because it did not
look". Every test below exists to make that shape unshippable:

* the nested case is FIRST and is the file's reason to exist
  (``RootCause.changes: tuple[Change, ...] = ()``, one level down);
* the report must NAME the dotted path, so a top-level-only walker cannot
  fake a pass;
* ``visited`` dedup must be per-node, so a second node sharing a nested model
  is still reported;
* the walker's ENTRY must be an annotation, so ``outputs=list[Claim]`` is not
  a silent hole.

The complementary failure -- an instrument that cries wolf -- is pinned by the
negative tests: nullable defaults, undefaulted fields, scripted nodes,
framework-collected secondary outputs, excluded fields, and bare ``Any``.

Scope (orchestrator triage on neograph-bek2f): the declared PRIMARY output of
think/agent/act nodes, recursed into nested models. NOT scripted nodes, NOT
node inputs, NOT secondary dict-form output keys.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from neograph import (
    Construct,
    ExcludeFromOutput,
    Node,
    Tool,
    ToolInteraction,
    lint,
)
from neograph.lint import LINT_KIND_META, LintIssue
from tests.schemas import RawText, _producer

# The kind under test. A single string literal, emitted at ONE site with a
# literal ``required=False`` (severity WARN is re-derived from that by
# scripts/gen_api_manifest.py).
KIND = "model_authored_null_rejecting_default"


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMAS -- the reporter's two shapes, plus the negative controls
# ═══════════════════════════════════════════════════════════════════════════

ClaimStatus = Literal["proposed", "accepted", "rejected"]


class Claim(BaseModel):
    """Run-1 shape: the defect is on the TOP-LEVEL declared output."""

    text: str
    status: ClaimStatus = "proposed"


class Verdict(StrEnum):
    OK = "ok"
    BAD = "bad"


class EnumClaim(BaseModel):
    """Run-1 shape again, with an Enum instead of a Literal."""

    text: str
    verdict: Verdict = Verdict.OK


class Change(BaseModel):
    path: str


class RootCause(BaseModel):
    """Run-2 shape: the defect lives inside a homogeneous tuple."""

    summary: str
    changes: tuple[Change, ...] = ()


class Diagnosis(BaseModel):
    """CLEAN at the top level. The only defect is one level down.

    A walker that inspects ``Diagnosis.model_fields`` and stops reports this
    tree as healthy -- the exact instrument the reporter is complaining about.
    """

    title: str
    root: RootCause


class SecondDiagnosis(BaseModel):
    """A DIFFERENT top-level output that reaches the SAME nested defect."""

    headline: str
    root: RootCause


class InteriorChange(BaseModel):
    path: str
    kind: str = "edit"


class TupleWrapper(BaseModel):
    """The only defect is on the tuple's ELEMENT type, one descent down."""

    items: tuple[InteriorChange, ...]


class NullableDefaults(BaseModel):
    """Defaults whose annotations DO admit null -- must stay silent."""

    label: str
    score: int | None = None
    note: str | None = "unset"


class NoDefaults(BaseModel):
    """Every field required -- nothing to warn about."""

    label: str
    count: int
    tags: list[str]


class AnyDefault(BaseModel):
    """A bare ``Any`` genuinely holds null -- explicit skip per triage."""

    label: str
    payload: Any = None


class ExcludedFields(BaseModel):
    """Fields describe_type never shows the model -- must stay silent.

    Live instances: examples/18_typed_projections.py:54 (``Field(exclude=True)``)
    and :69 (``Annotated[str, ExcludeFromOutput]``).
    """

    text: str
    internal_rank: int = Field(default=0, exclude=True)
    source_url: Annotated[str, ExcludeFromOutput] = ""


class Leaf(BaseModel):
    kind: str = "leaf"


class Diamond(BaseModel):
    """Reaches ``Leaf.kind`` by two distinct paths."""

    left: Leaf
    right: Leaf


class TreeNode(BaseModel):
    """Self-referential. ``children`` is REQUIRED so ``weight`` is the only defect."""

    weight: int = 0
    children: list[TreeNode]


class Pong(BaseModel):
    label: str = "pong"
    pings: list[Ping]


class Ping(BaseModel):
    """Mutual recursion Ping -> Pong -> Ping."""

    tag: str = "ping"
    pongs: list[Pong]


TreeNode.model_rebuild()
Ping.model_rebuild()
Pong.model_rebuild()


# ═══════════════════════════════════════════════════════════════════════════
# BUILDERS
# ═══════════════════════════════════════════════════════════════════════════


def _think(name: str, outputs: Any) -> Node:
    """An LLM-mode node whose declared output the MODEL authors."""
    return Node(name, mode="think", outputs=outputs, model="fast", prompt="test/analyze")


def _agent(name: str, outputs: Any) -> Node:
    return Node(
        name,
        mode="agent",
        outputs=outputs,
        model="fast",
        prompt="test/analyze",
        tools=[Tool("read_a", idempotent=True)],
    )


def _pipeline(*nodes: Node) -> Construct:
    return Construct("nullable-defaults", nodes=list(nodes))


def _hits(construct: Construct) -> list[LintIssue]:
    """Every issue of the kind under test, in report order."""
    return [i for i in lint(construct) if i.kind == KIND]


def _blob(issue: LintIssue) -> str:
    """Everything the issue tells a human, as one searchable string."""
    return f"{issue.param} {issue.message}"


# ═══════════════════════════════════════════════════════════════════════════
# THE NESTED CASE -- first, because it is why this check exists
# ═══════════════════════════════════════════════════════════════════════════


class TestNestedReachability:
    """The check must RECURSE. Both reported production failures were nested."""

    def test_reports_the_field_when_the_defect_is_one_level_below_the_declared_output(self):
        """Run-2: ``RootCause.changes: tuple[Change, ...] = ()``, reached via ``Diagnosis.root``.

        ``Diagnosis`` itself is clean. A top-level-only walk returns [] here --
        that walk is the userland instrument that "passed because it did not
        look", and this assertion is what forbids shipping it.
        """
        hits = _hits(_pipeline(_think("diagnose", Diagnosis)))

        assert len(hits) == 1, f"expected the nested defect, got {[_blob(i) for i in hits]}"
        assert hits[0].node_name == "diagnose"
        assert hits[0].required is False, "WARN, not ERROR -- neograph check must stay non-blocking"

    def test_names_the_dotted_path_to_the_nested_field_when_reporting(self):
        """The report is only actionable if it says WHERE.

        Contract: the dotted path from the declared output down to the field,
        e.g. ``Diagnosis.root.changes``. A walker that lost the path cannot
        produce ``root.changes``.
        """
        hits = _hits(_pipeline(_think("diagnose", Diagnosis)))

        assert len(hits) == 1, [_blob(i) for i in hits]
        blob = _blob(hits[0])
        assert "root.changes" in blob, f"issue must name the dotted path, got: {blob!r}"

    def test_descends_into_a_homogeneous_tuple_when_the_defect_is_on_the_element_type(self):
        """``tuple[X, ...]`` must be descended, not just flagged.

        The runtime descent excluded tuples until neograph-sjwny widened it in
        this same release; the reporter's run-2 failure WAS a tuple, and this
        walker must reach the same interiors. ``InteriorChange.kind`` is only
        reachable THROUGH the tuple.
        """
        hits = _hits(_pipeline(_think("diagnose", TupleWrapper)))

        assert len(hits) == 1, [_blob(i) for i in hits]
        assert "kind" in _blob(hits[0])


# ═══════════════════════════════════════════════════════════════════════════
# THE FLAT CASE
# ═══════════════════════════════════════════════════════════════════════════


class TestTopLevelDefect:
    """Run-1: the defect is on the declared output itself."""

    def test_reports_a_literal_typed_field_with_a_default_when_the_model_authors_it(self):
        """``Claim.status: ClaimStatus = 'proposed'`` -- 32.7M tokens, 1 error."""
        hits = _hits(_pipeline(_think("classify", Claim)))

        assert len(hits) == 1, [_blob(i) for i in hits]
        assert hits[0].node_name == "classify"
        assert "status" in _blob(hits[0])
        assert hits[0].required is False

    def test_reports_an_enum_typed_field_with_a_default_when_the_model_authors_it(self):
        """An Enum rejects null exactly as a Literal does."""
        hits = _hits(_pipeline(_think("classify", EnumClaim)))

        assert len(hits) == 1, [_blob(i) for i in hits]
        assert "verdict" in _blob(hits[0])

    def test_reports_the_defect_inside_a_container_when_the_declared_output_is_a_list(self):
        """``outputs=list[Claim]`` -- the walker's ENTRY is an annotation, not a model.

        A walker signatured ``(model: type[BaseModel])`` silently yields nothing
        here. That is the same false negative as the top-level-only walk, one
        layer up.
        """
        hits = _hits(_pipeline(_think("classify", list[Claim])))

        assert len(hits) == 1, [_blob(i) for i in hits]
        assert "status" in _blob(hits[0])


# ═══════════════════════════════════════════════════════════════════════════
# NEGATIVE CONTROLS -- an instrument that cries wolf fails the same requirement
# ═══════════════════════════════════════════════════════════════════════════


class TestSilentCases:
    """Shapes that must NOT be reported."""

    def test_no_issue_when_the_defaulted_field_type_admits_null(self):
        """``x: int | None = None`` is exactly what the fix looks like."""
        assert _hits(_pipeline(_think("summarize", NullableDefaults))) == []

    def test_no_issue_when_the_field_carries_no_default(self):
        """A required field cannot fall back to a default, so there is nothing to warn about."""
        assert _hits(_pipeline(_think("summarize", NoDefaults))) == []

    def test_no_issue_for_a_bare_any_annotation_when_it_carries_a_default(self):
        """``Any`` genuinely holds null (triage: the one explicit skip)."""
        assert _hits(_pipeline(_think("summarize", AnyDefault))) == []

    def test_no_issue_for_a_scripted_node_when_its_output_carries_the_same_defect(self):
        """A scripted node's output is built in PYTHON. No model authors it, so the
        default is legitimate -- the same ``Claim`` that fires on a think node."""
        assert _hits(_pipeline(_producer("summarize", Claim))) == []

    def test_no_issue_for_an_excluded_field_when_describe_type_never_shows_it(self):
        """``Field(exclude=True)`` and ``Annotated[str, ExcludeFromOutput]`` are
        skipped by describe_type at :258/:351, so the model never sees them and
        never emits a null for them. Reporting them is a false positive on
        SHIPPED example models (examples/18_typed_projections.py:54,:69)."""
        assert _hits(_pipeline(_think("project", ExcludedFields))) == []

    def test_no_issue_for_a_scripted_node_when_the_defect_is_nested(self):
        """The mode gate applies to the whole reachable tree, not just its root."""
        assert _hits(_pipeline(_producer("summarize", Diagnosis))) == []


# ═══════════════════════════════════════════════════════════════════════════
# DICT-FORM OUTPUTS -- primary is model-authored, the rest is framework-collected
# ═══════════════════════════════════════════════════════════════════════════


class TestDictFormOutputs:
    """Only the FIRST key of a dict-form output is authored by the model."""

    def test_walks_the_primary_key_when_the_output_is_dict_form(self):
        hits = _hits(_pipeline(_agent("explore", {"result": Claim, "tool_log": list[ToolInteraction]})))

        assert len(hits) == 1, [_blob(i) for i in hits]
        assert hits[0].node_name == "explore"
        assert "status" in _blob(hits[0])

    def test_no_issue_for_a_framework_collected_secondary_key_when_the_output_is_dict_form(self):
        """``ToolInteraction`` itself carries defaulted non-nullable fields
        (``result: str = ''``, ``duration_ms: int = 0``). The framework writes
        them, not the model, so walking the secondary key is a pure false
        positive on every agent/act node in the repo."""
        hits = _hits(_pipeline(_agent("explore", {"result": Claim, "tool_log": list[ToolInteraction]})))

        blobs = [_blob(i) for i in hits]
        assert not any("duration_ms" in b for b in blobs), blobs
        assert not any("ToolInteraction" in b for b in blobs), blobs
        assert not any("tool_log" in b for b in blobs), blobs


# ═══════════════════════════════════════════════════════════════════════════
# CYCLES AND SHARING -- termination, once-only, and the visited-hoisting trap
# ═══════════════════════════════════════════════════════════════════════════


class TestCycleProtection:
    """The walk must terminate, and must report each (model, field) once per node."""

    def test_terminates_and_reports_once_when_the_output_model_is_self_referential(self):
        hits = _hits(_pipeline(_think("plan", TreeNode)))

        assert len(hits) == 1, [_blob(i) for i in hits]
        assert "weight" in _blob(hits[0])

    def test_terminates_and_reports_each_field_once_when_two_models_are_mutually_recursive(self):
        """Ping -> Pong -> Ping. One issue per offending field, not per path."""
        hits = _hits(_pipeline(_think("plan", Ping)))

        blobs = sorted(_blob(i) for i in hits)
        assert len(hits) == 2, blobs
        assert any("tag" in b for b in blobs), blobs
        assert any("label" in b for b in blobs), blobs

    def test_reports_once_when_the_same_field_is_reachable_by_two_paths(self):
        """``Diamond.left`` and ``Diamond.right`` are both ``Leaf``."""
        hits = _hits(_pipeline(_think("plan", Diamond)))

        assert len(hits) == 1, [_blob(i) for i in hits]
        assert "kind" in _blob(hits[0])


class TestPerNodeVisitedScope:
    """``visited`` is FRESH per node walk -- never hoisted across nodes.

    Hoisting it makes the SECOND node sharing a nested model report nothing:
    the dedup requirement quietly reintroduces "an instrument passing because
    it did not look", one node over.
    """

    def test_reports_both_nodes_when_two_llm_nodes_share_one_nested_defective_model(self):
        construct = _pipeline(
            _think("diagnose", Diagnosis),
            _think("review", SecondDiagnosis),
        )
        hits = _hits(construct)

        assert {i.node_name for i in hits} == {"diagnose", "review"}, [(i.node_name, _blob(i)) for i in hits]
        assert len(hits) == 2, [(i.node_name, _blob(i)) for i in hits]

    def test_reports_the_second_node_when_both_declare_the_same_output_model(self):
        construct = _pipeline(_think("first", Claim), _think("second", Claim))
        hits = _hits(construct)

        assert {i.node_name for i in hits} == {"first", "second"}, [(i.node_name, _blob(i)) for i in hits]


# ═══════════════════════════════════════════════════════════════════════════
# KIND REGISTRATION -- the kind is only real once the registry knows it
# ═══════════════════════════════════════════════════════════════════════════


class TestKindRegistration:
    """A kind absent from LINT_KIND_META breaks the manifest generator and the
    website lint table (scripts/gen_api_manifest.py re-derives severity from the
    literal ``required=`` at the single emission site and FAILS LOUD on drift)."""

    def test_kind_is_registered_with_warn_severity_when_it_is_emitted_with_required_false(self):
        assert KIND in LINT_KIND_META, sorted(LINT_KIND_META)
        assert LINT_KIND_META[KIND].severity == "WARN"
        assert "\n" not in LINT_KIND_META[KIND].meaning, "meaning must be single-line"


# ═══════════════════════════════════════════════════════════════════════════
# lint() NEVER RAISES -- its documented contract (lint.py:7)
# ═══════════════════════════════════════════════════════════════════════════


class TestLintStillNeverRaises:
    """Whatever the new walk does, lint reports problems -- it does not throw."""

    def test_returns_a_list_when_the_construct_mixes_defective_and_clean_outputs(self):
        construct = _pipeline(
            _producer("seed", RawText),
            _think("diagnose", Diagnosis),
            _think("summarize", NullableDefaults),
        )

        issues = lint(construct)

        assert isinstance(issues, list)
        assert [i for i in issues if i.kind == KIND] != [], (
            "the defective node must still be reported when clean nodes surround it"
        )
