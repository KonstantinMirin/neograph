"""Branch sentinel IR — the conditional-branch node the compiler lowers.

A neutral low-level IR module (reachable from every layer, including the
compiler/state/wiring lowering layers) that OWNS the branch sentinel and its
metadata. The DX layer (``forward.py``) produces these during ``forward()``
tracing; the compiler/state/wiring layers consume them. Keeping them here
makes the dependency point DX -> IR, never IR -> DX (the same neutral-home
pattern as ``_normalize.py`` and ``conditions.py``).

``_BranchNode`` is the sibling of ``Node``/``Construct`` in a Construct's node
list (see ``_ir_protocols.ConstructItem``); it is a non-Pydantic sentinel the
compiler recognizes to wire ``add_conditional_edges``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from neograph.modifiers import Modifiable, ModifierSet

if TYPE_CHECKING:
    from neograph.construct import Construct
    from neograph.node import Node


@dataclasses.dataclass(frozen=True)
class _ConditionSpec:
    """Parsed condition specification for compiler lowering."""
    source_node: Node | Construct | None
    attr_chain: list[str]
    op_fn: Any  # operator callable
    op_str: str  # e.g., "<", ">"
    threshold: Any  # right-hand side constant


@dataclasses.dataclass
class _BranchMeta:
    """Branch metadata attached to the node list for compiler consumption.

    This is stored on a sentinel _BranchNode that the compiler recognizes
    and lowers to add_conditional_edges.
    """
    condition_spec: _ConditionSpec
    # Nodes that only appear in the true arm
    true_arm_nodes: list[Node | Construct]
    # Nodes that only appear in the false arm
    false_arm_nodes: list[Node | Construct]


class _BranchNode(Modifiable):
    """Sentinel that carries _BranchMeta in the node list.

    The compiler checks for this type and wires conditional edges instead
    of adding a regular node. It carries a synthetic name for graph wiring.

    Inherits has_modifier, get_modifier, and modifiers from Modifiable.
    """

    def __init__(self, branch_meta: _BranchMeta, branch_id: int) -> None:
        self._neo_branch_meta = branch_meta
        self.name = f"__branch_{branch_id}"
        self.modifier_set = ModifierSet()
        self.output = None
