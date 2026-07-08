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
from collections.abc import Iterator, MutableSequence
from typing import TYPE_CHECKING, Any

from neograph.modifiers import Modifiable, ModifierSet

if TYPE_CHECKING:
    from neograph._ir_protocols import ConstructItem, ConstructLike
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


def iter_with_arms(construct: ConstructLike) -> Iterator[ConstructItem]:
    """Yield every top-level item in ``construct.nodes``, expanding each
    ``_BranchNode`` sentinel into its arm contents in place.

    The single arm-descent primitive for the hand-rolled ``construct.nodes``
    walks that must observe conditionally-executed nodes but do NOT want the
    full leaf-flattening of :func:`neograph.construct.iter_nodes`. Semantics:

    - Non-branch items (``Node`` / ``Construct``) are yielded unchanged, in
      order. For a construct with no ``_BranchNode`` this is identity over
      ``construct.nodes`` — the zero-regression property that makes migrating a
      walk to this helper safe.
    - A ``_BranchNode`` is dropped and replaced by its ``true_arm_nodes`` then
      ``false_arm_nodes`` (each a ``list[Node | Construct]``). Arm items are
      arm-exclusive (``forward.py`` builds arms from ``true_only`` /
      ``false_only``), so each is yielded exactly once.

    ONE-LEVEL expansion only: arm ``Construct``s are yielded opaque, NOT
    recursed into. The compiler's arm lowering (``_wiring.py``) already
    recursively ``_compile()``s arm sub-constructs (which self-validate), so
    each caller's own ``isinstance(item, Construct)`` handling / recursion is
    the sole owner of sub-construct descent — recursing here would double-process
    them. See neograph-vn5f.

    Lives in ``_ir_branch.py`` (not ``construct.py``) so the low-level IR
    consumers ``_construct_validation`` and ``_ir_normalize`` — which sit below
    ``construct.py`` and only ``TYPE_CHECKING``-reference ``Construct`` — can
    import it without a cycle.
    """
    for item in construct.nodes:
        if isinstance(item, _BranchNode):
            meta = item._neo_branch_meta
            yield from meta.true_arm_nodes
            yield from meta.false_arm_nodes
        else:
            yield item


def iter_item_slots(
    construct: Construct,
) -> Iterator[tuple[MutableSequence[Any], int]]:
    """Yield a mutable ``(container, index)`` slot for every item that
    :func:`iter_with_arms` yields, so a walk that rewrites items via
    ``model_copy`` can target the correct storage slot uniformly.

    For a top-level item the slot is ``(construct.nodes, i)``; for an arm item
    the slot is ``(meta.true_arm_nodes, j)`` or ``(meta.false_arm_nodes, j)``.
    The ``_BranchNode`` sentinel's own top-level slot is NOT yielded (it is not
    a real IR node any rewriting walk touches) — it is replaced by its arm
    slots, mirroring :func:`iter_with_arms`.

    The write-back counterpart of :func:`iter_with_arms`, for the two walks
    that mutate nodes in place (``normalize_ir`` and the ``Construct.__init__``
    llm_config/renderer inheritance pass). Without this, an arm node's rewrite
    would land in a detached copy that never reaches the compiled arm. See
    neograph-vn5f.
    """
    for i, item in enumerate(construct.nodes):
        if isinstance(item, _BranchNode):
            meta = item._neo_branch_meta
            for j in range(len(meta.true_arm_nodes)):
                yield (meta.true_arm_nodes, j)
            for j in range(len(meta.false_arm_nodes)):
                yield (meta.false_arm_nodes, j)
        else:
            yield (construct.nodes, i)
