"""Type contracts shared across the Agent Spec LOWERING modules.

``_ExportFlow`` and ``_LoweredItem`` each used to live in whichever module
defined them first (``_agent_spec_modifier_lowering`` and ``_agent_spec``).
Once ``_agent_spec_item_dispatch`` was split out (neograph-qtfof.13) both became
genuine CROSS-module contracts — three modules annotate against them — so they
live in one leaf every direction can import without closing a cycle.

Distinct from ``_agent_spec_types``, which is about the pyagentspec Property /
type-registry surface. This module is only the lowering pipeline's own shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Callable  # noqa: F401
    from typing import Any  # noqa: F401

    from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge  # noqa: F401
    from pyagentspec.flows.node import Node as SpecNode  # noqa: F401

    from neograph._agent_spec import _Exit  # noqa: F401
    from neograph.construct import Construct  # noqa: F401

__all__ = ["_ExportFlow", "_LoweredItem"]

_ExportFlow: TypeAlias = "Callable[[Construct], Any]"
"""``to_agent_spec``, injected. The recursive sub-export is threaded as a
parameter rather than imported, so a lowering module never depends on the
assembler that calls it."""

_LoweredItem: TypeAlias = (
    "tuple[list[SpecNode], list[ControlFlowEdge], list[DataFlowEdge], "
    "SpecNode, list[_Exit], SpecNode, list[tuple[SpecNode, bool]]]"
)
"""What one lowered construct item is: (all_spec_nodes, extra_control_edges,
extra_data_edges, entry_node, exits, data_node, input_targets). Named so the
per-shape arms of ``_lower_construct_item`` can BIND this shape and let the
shared Operator postlude rewrite it, instead of each arm returning its own."""
