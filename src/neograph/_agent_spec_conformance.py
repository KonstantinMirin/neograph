"""Agent Spec export conformance predicates -- IR + Flow level (neograph-ftnxl.1).

"Does not raise" has been used as a stand-in for "is portable to a third-party
Agent Spec runtime" -- provably false in known ways (neograph-qtfof.6/.7/.8/.9).
This module owns the PREDICATES that detect those gaps. It imports NOTHING from
``_agent_spec*`` (that would create an import cycle: ``neograph/__init__.py``
already imports ``_agent_spec`` eagerly, and ``_agent_spec`` would need to call
into a conformance report built on top of this module -- see
``_agent_spec_conformance_report.py``, which composes this module with
``_agent_spec`` one-way, exporter -> classifier).

Two walkers, each the SOLE owner of its level (guarded against a second,
re-derived walker by ``tests/test_guards_agent_spec_conformance.py``):

- ``_structural_findings(construct)`` -- IR-level, pre-export. Walks
  ``iter_with_arms`` + this module's own ``Construct`` recursion (never
  ``iter_nodes``, which is leaf-only and would miss a Construct-level
  ``| Each(...)``/``| Loop(...)``) and asks only "which modifiers are
  present", via ``classify_modifiers``/``modifier_names_for_combo`` -- never a
  hand-typed combo list.
- ``_flow_findings(flow)`` -- Flow-level, post-export. Reads the LOWERED
  artifact ``to_agent_spec()`` actually produced, recursing
  ``MapNode.subflow``/``FlowNode.subflow`` (an LlmNode inside an Each-wrapped
  node lives one level down -- a flat scan would silently stop finding it the
  moment the sibling Each predicate retires). Detects the Portal-mesh Swarm
  shape too, for which the empirical calibration authority
  (``tests/test_agent_spec_matrix.py``'s ``UNSUPPORTED_COMBOS``) has no GREEN
  cells at all -- so that shape is NEVER certified PORTABLE.

``export_conformance()`` -- the composer that ATTEMPTS ``to_agent_spec()`` and
decides NOT_EXPORTABLE by catching the exporter's own raise -- lives in
``_agent_spec_conformance_report.py``, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from neograph._ir_branch import iter_with_arms
from neograph.modifiers import classify_modifiers, modifier_names_for_combo
from neograph.node import Node

if TYPE_CHECKING:
    from neograph.construct import Construct


class ConformanceTier(Enum):
    """Three-tier verdict for how portable a Construct's Agent Spec export is."""

    PORTABLE = "portable"
    NEOGRAPH_ROUND_TRIP_ONLY = "neograph_round_trip_only"
    NOT_EXPORTABLE = "not_exportable"


_TIER_RANK: dict[ConformanceTier, int] = {
    ConformanceTier.PORTABLE: 0,
    ConformanceTier.NEOGRAPH_ROUND_TRIP_ONLY: 1,
    ConformanceTier.NOT_EXPORTABLE: 2,
}


@dataclass(frozen=True)
class ConformancePredicateMeta:
    """Tier + tracked bead + one-line meaning for a single predicate id.

    Mirrors ``_lint_kind_registry.LintKindMeta``'s registry ROLE (single
    source of truth, keyed by emitted id) -- not its shape: a conformance
    tier is an IR-taxonomy ``Enum`` (the ``ModifierCombo``/``PrimaryShape``
    convention), not a plain ``str`` like ``LintKindMeta.severity``.
    """

    tier: ConformanceTier
    bead: str
    meaning: str


#: SINGLE SOURCE OF TRUTH for predicate id -> tier + tracked bead + meaning.
#: Every id `_structural_findings`/`_flow_findings` can emit has an entry here;
#: `tests/test_guards_agent_spec_conformance.py` asserts the reverse too (no
#: stale entry for a retired predicate) -- a closed bead with a live predicate
#: fails that guard.
CONFORMANCE_PREDICATE_META: dict[str, ConformancePredicateMeta] = {
    "modifier_metadata_only_fanout": ConformancePredicateMeta(
        ConformanceTier.NEOGRAPH_ROUND_TRIP_ONLY,
        "neograph-qtfof.7",
        "An Each-modified node's fan-out source (MapNode.iterated_item) has no real "
        "DataFlowEdge -- the collection source lives only in neograph/each_spec "
        "metadata, which a foreign runtime ignores.",
    ),
    "modifier_metadata_only_branch": ConformancePredicateMeta(
        ConformanceTier.NEOGRAPH_ROUND_TRIP_ONLY,
        "neograph-qtfof.6",
        "A Loop/Operator-modified node's branch decision (BranchingNode."
        "branching_mapping_key) has no real DataFlowEdge -- the predicate lives "
        "only in neograph/loop_spec or neograph/operator_spec metadata.",
    ),
    "llm_config_missing_api_provider": ConformancePredicateMeta(
        ConformanceTier.NEOGRAPH_ROUND_TRIP_ONLY,
        "neograph-qtfof.8",
        "An exported LlmConfig has api_provider=None, which pyagentspec's "
        "AgentSpecLoader cannot load.",
    ),
    "outermost_end_node_unwired": ConformancePredicateMeta(
        ConformanceTier.NEOGRAPH_ROUND_TRIP_ONLY,
        "neograph-qtfof.9",
        "The outermost Construct's exported EndNode declares no outputs "
        "(construct.output is None for a top-level Construct by design), so a "
        "third-party runtime's invoke() never surfaces a result.",
    ),
    "portal_mesh_unverified": ConformancePredicateMeta(
        ConformanceTier.NEOGRAPH_ROUND_TRIP_ONLY,
        "neograph-qtfof",
        "A Portal peer mesh exports as a Swarm, not a Flow. The empirical "
        "calibration authority (tests/test_agent_spec_matrix.py's "
        "UNSUPPORTED_COMBOS) has zero GREEN cells of this shape, so it is never "
        "certified PORTABLE -- absence of a failure signal is not evidence here.",
    ),
    "exporter_rejected": ConformancePredicateMeta(
        ConformanceTier.NOT_EXPORTABLE,
        "N/A",
        "to_agent_spec() raised ConfigurationError -- the exporter itself decided "
        "this construct cannot be represented at all. See the finding's own "
        "detail for the exporter's message; this predicate never re-derives the "
        "exporter's raise-list.",
    ),
}


@dataclass(frozen=True)
class ConformanceFinding:
    """One predicate firing: which rule, on which node (if applicable), and why."""

    predicate: str
    node_name: str | None
    detail: str


def _is_construct_like(item: Any) -> bool:
    """True for a sub-Construct item (has both .input and .nodes), duck-typed
    to avoid importing Construct at runtime (no cycle risk either way, but
    this mirrors the neutral-module pattern used elsewhere in the IR)."""
    return not isinstance(item, Node) and hasattr(item, "nodes") and hasattr(item, "input")


def _structural_findings(construct: Construct) -> list[ConformanceFinding]:
    """The ONE structural (pre-export, IR-level) walker. See module docstring."""
    findings: list[ConformanceFinding] = []
    _walk_structural(construct, findings)
    return findings


def _walk_structural(construct: Construct, findings: list[ConformanceFinding]) -> None:
    for item in iter_with_arms(construct):
        combo, _mods = classify_modifiers(item)
        names = modifier_names_for_combo(combo)
        node_name = getattr(item, "name", None)
        if "each" in names:
            findings.append(
                ConformanceFinding("modifier_metadata_only_fanout", node_name, "Each modifier present")
            )
        if "loop" in names or "operator" in names:
            findings.append(
                ConformanceFinding("modifier_metadata_only_branch", node_name, "Loop/Operator modifier present")
            )
        if _is_construct_like(item):
            _walk_structural(item, findings)  # type: ignore[arg-type]


def _is_swarm(flow: Any) -> bool:
    try:
        from pyagentspec.swarm import Swarm
    except ImportError:  # pragma: no cover -- pyagentspec is a dev dependency
        return False
    return isinstance(flow, Swarm)


def _flow_findings(flow: Any) -> list[ConformanceFinding]:
    """The ONE Flow-level (post-export) walker. See module docstring."""
    findings: list[ConformanceFinding] = []
    if _is_swarm(flow):
        findings.append(
            ConformanceFinding("portal_mesh_unverified", None, "exports as a Swarm, not a Flow")
        )
        return findings
    _check_outermost_end_node(flow, findings)
    _walk_flow_for_llm_configs(flow, findings)
    return findings


def _check_outermost_end_node(flow: Any, findings: list[ConformanceFinding]) -> None:
    """Only the TOP-LEVEL Flow's EndNode is checked -- a nested subflow's
    EndNode is always wired to its parent MapNode/FlowNode and is not the
    qtfof.9 gap (which is specifically about the outermost Construct)."""
    from pyagentspec.flows.nodes.endnode import EndNode

    for n in getattr(flow, "nodes", None) or []:
        if isinstance(n, EndNode) and not (n.outputs or n.inputs):
            findings.append(
                ConformanceFinding(
                    "outermost_end_node_unwired",
                    getattr(n, "name", None),
                    "EndNode declares no outputs",
                )
            )


def _walk_flow_for_llm_configs(flow: Any, findings: list[ConformanceFinding]) -> None:
    from pyagentspec.flows.nodes.llmnode import LlmNode

    for n in getattr(flow, "nodes", None) or []:
        if isinstance(n, LlmNode):
            api_provider = getattr(getattr(n, "llm_config", None), "api_provider", None)
            if api_provider is None:
                findings.append(
                    ConformanceFinding(
                        "llm_config_missing_api_provider",
                        getattr(n, "name", None),
                        "LlmConfig.api_provider is None",
                    )
                )
        subflow = getattr(n, "subflow", None)
        if subflow is not None:
            _walk_flow_for_llm_configs(subflow, findings)


def _worst_tier(findings: list[ConformanceFinding]) -> ConformanceTier:
    worst = ConformanceTier.PORTABLE
    for f in findings:
        tier = CONFORMANCE_PREDICATE_META[f.predicate].tier
        if _TIER_RANK[tier] > _TIER_RANK[worst]:
            worst = tier
    return worst


@dataclass(frozen=True)
class ConformanceReport:
    """The full conformance verdict for one construct: the worst tier across
    every finding, plus every finding itself (for a human or a caller to
    inspect why)."""

    verdict: ConformanceTier
    findings: tuple[ConformanceFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "findings": [
                {"predicate": f.predicate, "node_name": f.node_name, "detail": f.detail} for f in self.findings
            ],
        }
