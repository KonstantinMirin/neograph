"""Agent-Spec metadata marker keys — the neutral home both sides import from.

Extracted from ``_agent_spec.py`` (neograph-3ffdg.3) as a pure file split.

Why a separate module rather than living in the exporter: ``loader.py`` (the
IMPORT side) used to import these constants from ``_agent_spec.py`` (the EXPORT
side), which is a backwards dependency — the importer does not otherwise need
the exporter. Both sides now import from this neutral module, so neither the
export nor the import path depends on the other.

``_import_agent_spec_flow_classes`` lives here for the same reason: it is the
shared import-guard both sides use to reach the optional ``[agent-spec]``
extra, and keeping ``src/neograph`` Agent-Spec-free at import time is pinned by
``tests/test_guards_agent_spec_core_purity.py``. Do NOT add a module-level
``pyagentspec`` import here.
"""

from __future__ import annotations

from typing import Any

from neograph.errors import ConfigurationError

# a typo cannot silently split the export<->import contract and downgrade a
# marker-bearing primitive to the fail-loud/foreign path. Pinned (no re-inlined
# literals + exact wire-value asserts) by tests/test_guards_agent_spec_markers.py.
_MARK_MODE = "neograph/mode"
_MARK_AGENT_SPEC = "neograph/agent_spec"
_MARK_TOOL_SPEC = "neograph/tool_spec"
_MARK_MODIFIER = "neograph/modifier"
_MARK_GROUP_ID = "neograph/group_id"
_MARK_VARIANT = "neograph/variant"
_MARK_ORACLE_SPEC = "neograph/oracle_spec"
_MARK_EACH_SPEC = "neograph/each_spec"
_MARK_LOOP_SPEC = "neograph/loop_spec"
_MARK_OPERATOR_SPEC = "neograph/operator_spec"
_MARK_BRANCH = "neograph/branch"
_MARK_PORTAL_SPEC = "neograph/portal_spec"
_MARK_PORTAL_OPERATOR_SPEC = "neograph/portal_operator_spec"
_MARK_PROMPT_SPEC = "neograph/prompt_spec"

# --- Branch labels -----------------------------------------------------------
# The ``from_branch`` arm names the export side writes onto ``ControlFlowEdge``
# and the import side matches on. Same single-home reasoning as the marker keys
# above, and the same failure mode: a label that drifts between the two sides is
# a SILENT no-match -- the importer simply fails to recognize the composite and
# the nodes come back as bare primitives, with nothing raised.
#
# They lived as private copies in BOTH ``_agent_spec_portal.py`` and
# ``_agent_spec_modifier_lowering.py``, defined independently with neither
# importing the other, while ``loader.py`` spelled "pause" as a raw literal
# matching neither constant.
#
# Grouped into ONE container rather than six loose module constants, mirroring
# ``StateKeys``: a consumer needs a single import to reach every label, and
# ``Branch.CONTINUE`` at the use site says which vocabulary the string belongs
# to in a way a bare ``_BRANCH_CONTINUE`` does not.
class Branch:
    """The ``from_branch`` arm labels.

    NOTE on the two roles "true"/"false" play. On a ``BranchingNode`` the
    ``mapping`` is {condition OUTCOME -> branch LABEL}. For Operator the outcome
    maps to a differently-named label (``{Branch.TRUE: Branch.PAUSE}``); for a
    plain branch the two coincide (``{Branch.TRUE: Branch.TRUE}``). These name
    the LABEL. Where a mapping KEY is spelled with one, it is because that
    shape's outcome-to-label map is the identity -- if the two roles ever need to
    diverge, add a separate outcome vocabulary rather than widening this one.
    """

    DEFAULT = "default"
    PAUSE = "pause"
    CONTINUE = "continue"
    DONE = "done"
    TRUE = "true"
    FALSE = "false"


def import_pyagentspec(*module_paths: str, found: str | None = None) -> Any:
    """The one shared guarded-import helper for reaching the optional
    ``pyagentspec`` dependency from anywhere in ``src/neograph``.

    Every call site that used to hand-roll its own ``try: import
    pyagentspec.x ... except ImportError: raise ConfigurationError(...)``
    block calls this instead, so the message/hint shape can never drift
    per-site. Imports each dotted module path in ``module_paths`` (via
    ``importlib``, so the import stays function-local to THIS call and never
    reaches module level); returns the single imported module when only one
    path is given, or a tuple of modules in the same order otherwise.
    """
    import importlib

    try:
        modules = tuple(importlib.import_module(path) for path in module_paths)
    except ImportError as exc:
        raise ConfigurationError.build(
            "pyagentspec is not installed",
            expected="the [agent-spec] optional extra",
            found=found or f"ImportError on {'/'.join(module_paths)}",
            hint="install with: uv sync --extra agent-spec (or pip install neograph[agent-spec])",
        ) from exc
    return modules[0] if len(modules) == 1 else modules


def _import_agent_spec_flow_classes() -> Any:
    """Function-local import of pyagentspec's Flow/node/edge classes.

    Thin wrapper over ``import_pyagentspec`` -- only calling ``to_agent_spec()``
    pulls in the optional ``[agent-spec]`` extra.
    """
    return import_pyagentspec(
        "pyagentspec.flows.nodes",
        "pyagentspec.flows.flow",
        "pyagentspec.flows.edges",
        "pyagentspec.property",
        "pyagentspec.tools",
        found="ImportError on pyagentspec.flows/property/tools",
    )
