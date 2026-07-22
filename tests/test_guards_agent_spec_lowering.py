"""Structural guard: agent/act mode export must never regress to a
lossy placeholder (neograph-i3zsh.1's disease scan -- codebase-scan:complete).

Disease pattern: a modifier-lowering function in _agent_spec.py silently
drops a Node field instead of either lowering it to a real Agent Spec
primitive + a neograph/*_spec round-trip marker, or failing loud. The
motivating instance was _lower_node's agent/act branch constructing a
ToolNode placeholder (or, pre-i3zsh.1, failing loud with no real lowering
at all) that would have silently dropped prompt/model/tools.
"""

from __future__ import annotations

import ast
import pathlib

AGENT_SPEC_FILE = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph" / "_agent_spec.py"


def _lower_node_source() -> str:
    tree = ast.parse(AGENT_SPEC_FILE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_lower_node":
            return ast.get_source_segment(AGENT_SPEC_FILE.read_text(), node) or ""
    raise AssertionError("_lower_node not found in _agent_spec.py")


class TestAgentActModeLowersToAgentNode:
    """_lower_node's agent/act branch must construct a real AgentNode, never
    a ToolNode placeholder or a bare fail-loud with no lowering at all."""

    def test_agent_act_branch_constructs_agent_node(self):
        """Positive: the current source builds an AgentNode for agent/act mode."""
        source = _lower_node_source()
        assert "AgentNode(" in source, (
            "_lower_node's agent/act branch must construct a pyagentspec AgentNode "
            "-- the ToolNode placeholder silently dropped prompt/model/tools"
        )

    def test_agent_act_branch_does_not_construct_bare_tool_node_only(self):
        """Negative (regex-slip guard): a ToolNode-only agent/act lowering --
        i.e. a version of this function that constructs ToolNode but NOT
        AgentNode inside the agent/act branch -- must be caught. Simulates
        the exact pre-i3zsh.1 regression by checking the two constructors
        are not conflated: AgentNode must appear BEFORE the final bare
        ToolNode return (which is reached only by scripted/raw modes)."""
        source = _lower_node_source()
        agent_idx = source.find("mode in (\"agent\", \"act\")")
        agent_node_idx = source.find("AgentNode(")
        tool_node_idx = source.rfind("nodes_mod.ToolNode(")
        assert agent_idx != -1, "agent/act mode dispatch branch not found"
        assert agent_node_idx != -1 and agent_node_idx > agent_idx, (
            "AgentNode construction must appear inside (after) the agent/act mode branch"
        )
        # The scripted/raw ToolNode fallback must come AFTER the agent/act
        # branch's own AgentNode construction -- i.e. agent/act mode returns
        # before ever reaching the bare ToolNode fallback.
        assert tool_node_idx > agent_node_idx, (
            "the scripted/raw ToolNode fallback must be structurally reachable only "
            "after the agent/act branch already returned its own AgentNode"
        )

    def test_agent_act_branch_stamps_reconstruction_marker(self):
        """Every irreversible flattening must carry a neograph/-prefixed marker
        (per the exporter's Core Invariant) -- agent/act is no exception."""
        source = _lower_node_source()
        assert "neograph/agent_spec" in source, (
            "agent/act lowering must stamp a neograph/agent_spec marker so a future "
            "from_agent_spec() importer can reconstruct the node losslessly"
        )
