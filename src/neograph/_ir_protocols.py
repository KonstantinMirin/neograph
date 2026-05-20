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
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from neograph.modifiers import ModifierSet


@runtime_checkable
class ConstructItem(Protocol):
    """Anything that can sit in ``Construct.nodes``: Node, Construct, _BranchNode."""

    name: str
    modifier_set: ModifierSet


__all__ = ["ConstructItem"]
