"""``PortalMemberClass`` — the ONE authority for "what kind of Portal mesh
participant is this item" (neograph-dgbqv.3, P7).

Before this module, that question was independently re-derived across the
package by hand-written ``portal.is_dispatch`` / ``portal.is_tool_triggered``
reads and ``member.mode in ("agent", "act")`` compares -- the same
duplicated-source-of-truth disease AGENTS.md records for the Portal rollout.
Every consumer now calls ``portal_member_class(item)`` and matches on
``PortalMemberClass`` instead (pinned by ``tests/test_guards_portal_member_class_consumers.py``).

STRUCTURAL DISCRIMINATION, NO Node/Construct IMPORT (real constraint, not
avoidable by module placement): ``_construct_validation.py`` -- a module
``construct.py`` itself imports at module level to build a ``Construct`` --
is one of this classifier's consumers. A module-level ``from neograph.construct
import Construct`` here would therefore cycle back through
``_construct_validation.py`` regardless of where this file sits in the DAG
otherwise. So node-vs-construct discrimination is duck-typed: a ``Construct``
has a ``.nodes`` field, a ``Node`` does not (the same structural test
``classify_modifiers`` already uses for its own duck-typed fallback path).

PRECEDENCE (load-bearing, architect review yf2ar.28's MOST IMPORTANT finding):
``PortalMemberClass`` is a FLAT enum over a PRODUCT space
(node|construct x mode x operator-present x trigger). The reduction is
therefore LOSSY -- an item can satisfy more than one class's criteria (e.g.
an Operator-guarded agent member), and exactly ONE is returned, by:

    SUB_CONSTRUCT > AGENT_CYCLE_* > ATOMIC_OPERATOR > ATOMIC

DISPATCH is a FIFTH outcome, not a member class at all -- a ``route="decide"``
Portal is a standalone linear node, never a mesh member. ``None`` means "this
item carries no Portal at all" (deliberately not a ``NOT_A_MEMBER`` enum
member, so every consumer's ``match`` stays exhaustive under ``assert_never``
over real participants).

The classifier is called on PRE-VALIDATION IR (before ``_validation_portal.py``
/ ``_construct_validation.py`` reject illegal products), so it must answer
SOMETHING for a product validation later forbids -- "validation forbids it"
is not a reason to leave the classifier's behavior on that product unpinned.

Deliberately NOT here: the mesh GROUPING (``_group_portal_members`` in
``_portal.py`` stays the single grouping authority -- this answers per-ITEM
class only), the combo decomposition table, and Portal pair-LEGALITY
(``modifiers.py``'s ``_DYNAMIC_RULES``, a different question -- "is this
product legal", not "what class is it").
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any


class PortalMemberClass(Enum):
    """What kind of Portal mesh participant an item is.

    Precedence when an item satisfies more than one (LOSSY on the
    operator/trigger axis): SUB_CONSTRUCT > AGENT_CYCLE_* > ATOMIC_OPERATOR >
    ATOMIC. DISPATCH is a fifth outcome, not a member class -- a dispatch-mode
    Portal is never a mesh member.
    """

    ATOMIC = auto()
    ATOMIC_OPERATOR = auto()
    AGENT_CYCLE_OUTPUT = auto()
    AGENT_CYCLE_TOOL = auto()
    SUB_CONSTRUCT = auto()
    DISPATCH = auto()


_AGENT_CYCLE_MODES = frozenset({"agent", "act"})


def portal_member_class(item: Any) -> PortalMemberClass | None:
    """Classify ``item``'s Portal mesh-participant kind, or ``None`` if it
    carries no Portal at all.

    Precedence (lossy -- see module docstring): SUB_CONSTRUCT > AGENT_CYCLE_*
    > ATOMIC_OPERATOR > ATOMIC, with DISPATCH checked first since it is not a
    mesh member at all. A foreign object with no ``.modifier_set`` (e.g. a
    third-party pyagentspec ``Flow``) returns ``None`` -- it is not this
    classifier's job to reconstruct a spec-shape mapping (neograph-dgbqv.5).
    """
    modifier_set = getattr(item, "modifier_set", None)
    if modifier_set is None:
        return None
    portal = modifier_set.portal
    if portal is None:
        return None

    if portal.is_dispatch:
        return PortalMemberClass.DISPATCH
    if getattr(item, "nodes", None) is not None:
        return PortalMemberClass.SUB_CONSTRUCT
    if getattr(item, "mode", None) in _AGENT_CYCLE_MODES:
        return PortalMemberClass.AGENT_CYCLE_TOOL if portal.is_tool_triggered else PortalMemberClass.AGENT_CYCLE_OUTPUT
    return PortalMemberClass.ATOMIC_OPERATOR if modifier_set.operator is not None else PortalMemberClass.ATOMIC
