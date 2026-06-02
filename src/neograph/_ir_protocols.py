"""Structural protocols for IR-item polymorphism.

The IR has three sibling item types that participate in a Construct's node
list: ``Node`` (BaseModel), ``Construct`` (BaseModel, also nests itself), and
``_BranchNode`` (non-Pydantic sentinel from ``forward.py``). All three carry
``name: str`` and ``modifier_set: ModifierSet``, but ``_BranchNode`` is NOT a
Pydantic model and cannot participate in a Pydantic field union.

``ConstructItem`` names the shared shape so static annotations replace bare
``Any`` at the boundaries that walk this polymorphic list. The runtime
``BeforeValidator`` on ``Construct.nodes`` still enforces the actual
acceptance rules; this Protocol is the structural typing layer.

``ConstructLike`` names the richer shape a *container* construct exposes to the
recursive validator (``name``/``input``/``output``/``nodes``). It lets
``_validate_node_chain`` recurse into sub-constructs with full static type
safety — no untyped escape-hatch cast and no runtime ``Construct`` import.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from neograph.modifiers import ModifierSet


@runtime_checkable
class ConstructItem(Protocol):
    """Anything that can sit in ``Construct.nodes``: Node, Construct, _BranchNode."""

    name: str
    modifier_set: ModifierSet


class ConstructLike(Protocol):
    """A container construct as the recursive validator sees it.

    Declares only the attributes ``_validate_node_chain`` reads off a
    construct: its ``name`` (error messages), its ``input``/``output``
    boundary ports (``type[BaseModel] | None``, typed ``Any`` here to avoid
    coupling), and its ``nodes`` list. ``Construct`` satisfies this
    structurally; the ``_is_construct_like`` TypeGuard narrows a
    ``ConstructItem`` to this shape for the recursive descent. NOT
    ``runtime_checkable`` — narrowing is done by the TypeGuard's explicit
    attribute checks, never ``isinstance``.
    """

    name: str
    input: Any
    output: Any

    # Read-only property (not a plain attribute) so the element type is
    # covariant: a concrete ``Construct`` whose ``nodes`` is a
    # ``list[Node | Construct | _BranchNode]`` satisfies this without mypy
    # flagging list-vs-Sequence invariance.
    @property
    def nodes(self) -> Sequence[ConstructItem]: ...


__all__ = ["ConstructItem", "ConstructLike"]
