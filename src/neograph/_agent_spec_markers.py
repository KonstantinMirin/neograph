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


def _import_agent_spec_flow_classes() -> Any:
    """Function-local import of pyagentspec's Flow/node/edge classes.

    Copies ``spec_types._import_agent_spec_property_classes()``'s exact
    import-guard shape so ``src/neograph`` core stays Agent-Spec-free by
    default — only calling ``to_agent_spec()`` pulls in the optional
    ``[agent-spec]`` extra.
    """
    try:
        import pyagentspec.flows.edges as edges_mod
        import pyagentspec.flows.flow as flow_mod
        import pyagentspec.flows.nodes as nodes_mod
        import pyagentspec.property as property_mod
        import pyagentspec.tools as tools_mod
    except ImportError as exc:
        raise ConfigurationError.build(
            "pyagentspec is not installed",
            expected="the [agent-spec] optional extra",
            found="ImportError on pyagentspec.flows/property/tools",
            hint="install with: uv sync --extra agent-spec (or pip install neograph[agent-spec])",
        ) from exc
    return nodes_mod, flow_mod, edges_mod, property_mod, tools_mod
