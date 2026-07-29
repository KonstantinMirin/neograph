"""Shared pyagentspec node capability registry (neograph-00447).

This module is the SINGLE source of truth for how neograph's Agent Spec export
tooling classifies pyagentspec's concrete ``flows.nodes`` primitives by their
``_get_inferred_inputs`` family. It is a SHARED artifact: both the exhaustive
export matrix (``tests/test_agent_spec_matrix.py``) and the AST structural guard
(``tests/test_guards_agent_spec_lowering.py`` / neograph-tjupj) consume it --
neither re-derives its own copy.

Two tiers, deliberately split (neograph-00447 architect-review finding M2):

* **Tier A -- ``NODE_FAMILIES``**: a pure ``dict[str, str]`` of class-name ->
  family. It imports NOTHING from pyagentspec, so a consumer that only needs the
  name-string classification (the pure-AST always-on structural guard) can
  ``from tests.agent_spec_capabilities import NODE_FAMILIES`` and stay always-on
  in the core suite. Importing this module must NEVER trigger a
  ``pytest.importorskip('pyagentspec')`` skip -- a structural guard silently not
  running is exactly the silent seam the north star forbids.

* **Tier B -- ``all_concrete_flow_node_classes`` / ``assert_registry_complete``**:
  the pyagentspec-dependent completeness walk. It imports pyagentspec *inside the
  function body* (lazy), so callers invoke it under their OWN
  ``pytest.importorskip('pyagentspec')``. The completeness assertion compares the
  Tier-A keyset against the live-walked concrete subclass set, so a new primitive
  pyagentspec ships later fails LOUD instead of silently going unclassified.

Families (verified from each class's ``_get_inferred_inputs`` source):

* ``text_scan``   -- inputs must appear as ``{{ }}`` placeholders in text/config
                     (scanned via ``get_placeholders_from_json_object``).
* ``structural``  -- inputs are read from a nested flow's StartNode / subflow /
                     agent (structural, not text).
* ``echo``        -- inputs merely echo ``self.inputs`` / ``tool.inputs`` /
                     outputs; no coupling.
"""

from __future__ import annotations

import pathlib

# -- Tier A: pure name-string classification (NO pyagentspec import) ----------

NODE_FAMILIES: dict[str, str] = {
    # text-scan / placeholder-coupled
    "LlmNode": "text_scan",
    "ApiNode": "text_scan",
    "InputMessageNode": "text_scan",
    "OutputMessageNode": "text_scan",
    # structural / subflow-based
    "AgentNode": "structural",
    "FlowNode": "structural",
    "CatchExceptionNode": "structural",
    "MapNode": "structural",
    "ParallelMapNode": "structural",
    "ParallelFlowNode": "structural",
    # unconstrained-echo
    "ToolNode": "echo",
    "BranchingNode": "echo",
    "StartNode": "echo",
    "EndNode": "echo",
}

FAMILIES = frozenset({"text_scan", "structural", "echo"})


# -- Derived: placeholder-prompt-emitting export constructors (neograph-tjupj) -

PLACEHOLDER_PROMPT_EMITTING_CONSTRUCTORS: frozenset[str] = frozenset(
    {n for n, f in NODE_FAMILIES.items() if f == "text_scan"}
) | {"AgentNode"}
"""pyagentspec constructor names whose neograph-EXPORT emission carries
neograph-authored ``${var}`` prompt text and therefore MUST be preceded by a
``_translate_placeholders(`` call in ``src/neograph/_agent_spec.py``.

DERIVED, never hand-copied -- two layered facts, single-sourced here so a new
coupled primitive is a LOUD add, not a silent miss:

* ``{n for n, f in NODE_FAMILIES.items() if f == "text_scan"}`` -- the
  pyagentspec placeholder-coupled family straight off ``NODE_FAMILIES`` (Tier A).
  ``LlmNode`` is the flow-node fed ``prompt_template=<translated>``; the OTHER
  text_scan members it collects (currently ``InputMessageNode``; ``ApiNode`` /
  ``OutputMessageNode`` are never constructed by the exporter) are NOT hidden --
  they are narrowed at the guard by a NAMED exemption (``InputMessageNode`` is a
  Portal routing-CHECK node, not prompt-bearing), not by dropping them here.
* ``| {"AgentNode"}`` -- an exporter-specific augmentation. ``AgentNode`` is
  ``structural`` to pyagentspec (``NODE_FAMILIES`` faithfully mirrors its
  ``_get_inferred_inputs``; do NOT reclassify it), but neograph populates the
  nested ``Agent.system_prompt`` with TRANSLATED text via ``_make_agent(...)``,
  so the export site is placeholder-coupled.

Bare ``Agent(...)`` (the sub-constructor inside ``_make_agent``) is deliberately
NOT in this set: it receives an already-translated ``system_prompt`` PARAM, so
demanding a same-body translate would be a false positive -- its pairing is
enforced at the ``_make_agent`` CALL sites instead.
"""


# -- Tier B: pyagentspec-dependent completeness walk (lazy import) ------------


def all_concrete_flow_node_classes() -> set[str]:
    """Names of every concrete ``pyagentspec.flows`` Node subclass, walked live.

    Imports pyagentspec lazily inside the body so importing this module (Tier A)
    never requires the ``agent-spec`` extra. Callers must guard invocation with
    their own ``pytest.importorskip('pyagentspec')``.
    """
    import inspect

    from pyagentspec.flows.node import Node as _PNode

    def _walk(cls: type) -> set[type]:
        found: set[type] = set()
        for sub in cls.__subclasses__():
            found.add(sub)
            found |= _walk(sub)
        return found

    return {c.__name__ for c in _walk(_PNode) if not inspect.isabstract(c)}


def assert_registry_complete() -> None:
    """Fail LOUD if ``NODE_FAMILIES`` and the live pyagentspec node set diverge.

    A new pyagentspec primitive (added upstream) -> ``missing`` non-empty ->
    someone must classify it here (and the AST guard picks it up for free). A
    class we classify that pyagentspec dropped -> ``stale`` non-empty. Also pins
    that every value is a known family.
    """
    walked = all_concrete_flow_node_classes()
    registered = set(NODE_FAMILIES)

    missing = walked - registered
    stale = registered - walked
    assert not missing, (
        "pyagentspec ships concrete flows.nodes Node subclass(es) NOT classified "
        f"in NODE_FAMILIES: {sorted(missing)}. Add each to NODE_FAMILIES with its "
        "_get_inferred_inputs family (text_scan / structural / echo)."
    )
    assert not stale, f"NODE_FAMILIES classifies name(s) pyagentspec no longer ships: {sorted(stale)}."
    bad_family = {k: v for k, v in NODE_FAMILIES.items() if v not in FAMILIES}
    assert not bad_family, f"NODE_FAMILIES has unknown family value(s): {bad_family}"


# ── Agent-Spec export source cluster (neograph-3ffdg.3) ──────────────────────
# The exporter used to be one file, and the AST guards over it read
# ``_agent_spec.py`` directly. neograph-3ffdg.3 split it into four modules as a
# pure file move.
#
# Those guards' CLAIMS are about the export surface as a whole -- "how many
# AgentNode construction sites exist", "does _lower_node itself construct an
# LlmNode" -- not about a filename. Scanning the concatenated cluster keeps each
# claim exactly as strong as before: a pure move leaves every count and every
# function body identical, so a real regression still fails.
#
# SINGLE source of truth: every such guard imports from here. A future split of
# the exporter adds its module to this list and nothing else changes.
_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

AGENT_SPEC_SOURCE_FILES = [
    _SRC_DIR / "_agent_spec.py",
    _SRC_DIR / "_agent_spec_markers.py",
    _SRC_DIR / "_agent_spec_placeholders.py",
    _SRC_DIR / "_agent_spec_node_lowering.py",
    _SRC_DIR / "_agent_spec_portal.py",
]


def agent_spec_source() -> str:
    """Concatenated source of every module in the Agent Spec export cluster."""
    missing = [p.name for p in AGENT_SPEC_SOURCE_FILES if not p.exists()]
    assert not missing, (
        f"Agent Spec export module(s) missing: {missing}. If a module was renamed "
        "or merged, update AGENT_SPEC_SOURCE_FILES -- never let these guards scan "
        "a silently shrinking surface."
    )
    return "\n".join(p.read_text() for p in AGENT_SPEC_SOURCE_FILES)
