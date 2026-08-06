"""Agent-Spec Swarm <-> PortalMemberClass encoding -- the ONE table (neograph-dgbqv.5, P10).

``PortalMemberClass`` (``_portal_member.py``) is the runtime taxonomy and sole
authority; this module is the ONE table that reads it in both directions, so no
consumer re-derives "Agent vs Flow", "trigger", or "HandoffMode" locally.

A true leaf: imports ONLY ``neograph._portal_member`` and ``neograph.errors``
(plus stdlib). It must NEVER import ``pyagentspec`` at module level (core-purity
guard) -- ``HandoffMode`` is carried by its STRING VALUE
(``"never"``/``"optional"``/``"always"``), and the export side converts it at its
existing function-local ``from pyagentspec.swarm import HandoffMode``.

TWO tables, because the two axes have DIFFERENT key spaces:

  * ``HANDOFF_MODE_TRIGGER`` -- the MODE axis, TOTAL over ``HandoffMode``'s three
    string values. Absorbs BOTH the forward map (mode -> Portal trigger) and its
    inverse (trigger -> canonical mode).
  * ``SWARM_ENCODING`` -- the MEMBER axis, TOTAL over ``PortalMemberClass``
    (``DISPATCH`` included as a real, non-exportable row).

**The inverse is asymmetric, by design, not by omission.** The table is TOTAL
export-ward (every ``PortalMemberClass`` has a row and every ``HandoffMode`` value
has a row), but only ONE cell per axis is invertible import-ward:

  * MODE axis: ``"tool"`` -> ``"optional"`` (the canonical choice -- ``"always"``
    is byte-identical to ``"optional"`` in the reference LangGraph adapter, per
    the swarm-langgraph-compilation-spike). ``"output"`` -> ``"never"`` (the sole
    row, trivially canonical).
  * MEMBER axis: ``"Flow"`` -> ``SUB_CONSTRUCT`` (the sole row). ``"Agent"`` is
    produced by FOUR classes (ATOMIC/ATOMIC_OPERATOR/AGENT_CYCLE_OUTPUT/
    AGENT_CYCLE_TOOL); the import path never recovers a ``PortalMemberClass``
    from a foreign ``"Agent"`` (it builds a plain ``Node`` and applies the mesh
    trigger instead), so asking that inverse is a loud error, not a guess.

Asking the inverse for a non-invertible key (``spec_class_to_member_class("Agent")``,
or a hypothetical ambiguous mode) raises ``ConfigurationError`` naming the
colliding members -- never a silent last-wins pick.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, NamedTuple, cast

from neograph._portal_member import PortalMemberClass
from neograph.errors import ConfigurationError


class HandoffModeRow(NamedTuple):
    mode_value: str
    trigger: Literal["output", "tool"]
    canonical: bool


#: The MODE axis, TOTAL over HandoffMode's three string values.
HANDOFF_MODE_TRIGGER: tuple[HandoffModeRow, ...] = (
    HandoffModeRow("never", "output", True),
    HandoffModeRow("optional", "tool", True),
    HandoffModeRow("always", "tool", False),
)


class SwarmEncoding(NamedTuple):
    spec_class: Literal["Flow", "Agent"] | None
    export_trigger: Literal["output", "tool"] | None
    import_forced_trigger: Literal["output", "tool"] | None
    gated: bool
    exportable: bool
    reason: str | None
    canonical: bool


#: The MEMBER axis, TOTAL over PortalMemberClass (all six values, DISPATCH
#: included as a real not-exportable row). ``export_trigger`` and
#: ``import_forced_trigger`` are NOT two spellings of one relation -- they
#: answer different questions and neither is derivable from the other:
#: AGENT_CYCLE_TOOL exports 'tool' but on IMPORT takes the mesh trigger (None),
#: while SUB_CONSTRUCT is 'output' on both.
SWARM_ENCODING: dict[PortalMemberClass, SwarmEncoding] = {
    PortalMemberClass.ATOMIC: SwarmEncoding(
        spec_class="Agent",
        export_trigger="output",
        import_forced_trigger=None,
        gated=False,
        exportable=True,
        reason=None,
        canonical=False,
    ),
    PortalMemberClass.ATOMIC_OPERATOR: SwarmEncoding(
        spec_class="Agent",
        export_trigger="output",
        import_forced_trigger=None,
        gated=True,
        exportable=True,
        reason=None,
        canonical=False,
    ),
    PortalMemberClass.AGENT_CYCLE_OUTPUT: SwarmEncoding(
        spec_class="Agent",
        export_trigger="output",
        import_forced_trigger=None,
        gated=False,
        exportable=True,
        reason=None,
        canonical=False,
    ),
    PortalMemberClass.AGENT_CYCLE_TOOL: SwarmEncoding(
        spec_class="Agent",
        export_trigger="tool",
        import_forced_trigger=None,
        gated=False,
        exportable=True,
        reason=None,
        canonical=False,
    ),
    PortalMemberClass.SUB_CONSTRUCT: SwarmEncoding(
        spec_class="Flow",
        export_trigger="output",
        import_forced_trigger="output",
        gated=False,
        exportable=True,
        reason=None,
        canonical=True,
    ),
    PortalMemberClass.DISPATCH: SwarmEncoding(
        spec_class=None,
        export_trigger=None,
        import_forced_trigger=None,
        gated=False,
        exportable=False,
        reason="a dispatch-mode Portal is not a mesh member -- it has no runtime-flow-synthesis "
        "primitive in Agent Spec (neograph-s7zt3.12 / C2)",
        canonical=False,
    ),
}


def _derive_inverse(rows: Iterable[tuple[object, object, bool]]) -> dict[object, object]:
    """Group ``(key, target, canonical)`` triples by ``target`` and pick the
    inverse image per the fail-loud rule (neograph-dgbqv.5 finding 2 refinement):

      * exactly one row in the group -> that row's key is the inverse image;
      * >1 rows, exactly one canonical=True -> that row's key;
      * >1 rows, >1 canonical=True -> ConfigurationError (a contradiction inside
        the table itself -- fails at IMPORT TIME, not at lookup time);
      * >1 rows, ZERO canonical=True -> the target is DELIBERATELY ABSENT from
        the returned mapping. A lookup for it raises ConfigurationError naming
        the colliding keys -- the table states honestly that the direction is
        non-invertible instead of fabricating a canonical answer.
    """
    groups: dict[object, list[tuple[object, bool]]] = {}
    for key, target, canonical in rows:
        groups.setdefault(target, []).append((key, canonical))

    inverse: dict[object, object] = {}
    for target, members in groups.items():
        if len(members) == 1:
            inverse[target] = members[0][0]
            continue
        canonical_members = [key for key, canonical in members if canonical]
        if len(canonical_members) > 1:
            raise ConfigurationError.build(
                f"multiple canonical rows map to the same target {target!r}",
                expected="at most one canonical=True row per inverse target",
                found=f"{len(canonical_members)} canonical rows: {canonical_members}",
                location="_agent_spec_swarm_encoding.py table definition",
            )
        if len(canonical_members) == 1:
            inverse[target] = canonical_members[0]
        # else: zero canonical -- target deliberately absent from `inverse`.
    return inverse


_TRIGGER_TO_MODE: dict[str, str] = cast(
    "dict[str, str]",
    _derive_inverse((row.mode_value, row.trigger, row.canonical) for row in HANDOFF_MODE_TRIGGER),
)

_SPEC_CLASS_TO_MEMBER_CLASS: dict[str, PortalMemberClass] = cast(
    "dict[str, PortalMemberClass]",
    _derive_inverse(
        (cls, row.spec_class, row.canonical) for cls, row in SWARM_ENCODING.items() if row.spec_class is not None
    ),
)

#: The FORWARD mode -> trigger direction. TOTAL and unambiguous by construction
#: (each mode_value has exactly one row), unlike the inverse -- no fail-loud
#: machinery needed here.
_MODE_TO_TRIGGER: dict[str, Literal["output", "tool"]] = {row.mode_value: row.trigger for row in HANDOFF_MODE_TRIGGER}


def mode_to_trigger(mode_value: str, *, default: Literal["output", "tool"] = "output") -> Literal["output", "tool"]:
    """The Portal trigger a ``HandoffMode`` string value maps to.

    Fail-SOFT on an unrecognized value (returns ``default``), preserving
    ``_swarm_trigger``'s pre-migration behavior byte-for-byte -- this ticket
    does not change that function's fail-soft-on-unknown contract.
    """
    return _MODE_TO_TRIGGER.get(mode_value, default)


def trigger_to_canonical_mode(trigger: Literal["output", "tool"]) -> str:
    """The canonical ``HandoffMode`` string value for a Portal trigger."""
    if trigger not in _TRIGGER_TO_MODE:
        colliding = sorted(row.mode_value for row in HANDOFF_MODE_TRIGGER if row.trigger == trigger)
        raise ConfigurationError.build(
            f"trigger {trigger!r} has no canonical HandoffMode",
            expected="a trigger with exactly one canonical=True row",
            found=f"colliding modes with no canonical winner: {colliding}",
        )
    return _TRIGGER_TO_MODE[trigger]


def spec_class_to_member_class(spec_class: Literal["Flow", "Agent"]) -> PortalMemberClass:
    """The canonical ``PortalMemberClass`` for a foreign spec_class string.

    Raises loud for ``"Agent"`` -- four member classes produce it and none is
    canonical, by design (see module docstring).
    """
    if spec_class not in _SPEC_CLASS_TO_MEMBER_CLASS:
        colliding = sorted(
            (cls.name for cls, row in SWARM_ENCODING.items() if row.spec_class == spec_class),
            key=str,
        )
        raise ConfigurationError.build(
            f"spec_class {spec_class!r} has no canonical PortalMemberClass",
            expected="a spec_class with exactly one canonical=True row",
            found=f"colliding member classes with no canonical winner: {colliding}",
            hint="the import path recovers this case by building a Node and applying the mesh "
            "trigger directly, never by inverting this table",
        )
    return _SPEC_CLASS_TO_MEMBER_CLASS[spec_class]


def handoff_mode_for_class(cls: PortalMemberClass) -> str:
    """The canonical HandoffMode string value a member class's export_trigger
    implies. Never a second hand-typed dict -- routed through the ONE mode
    inverse."""
    row = SWARM_ENCODING[cls]
    if row.export_trigger is None:
        raise ConfigurationError.build(
            f"PortalMemberClass.{cls.name} has no export_trigger",
            expected="a class with export_trigger set (i.e. exportable)",
            found=f"exportable={row.exportable}, reason={row.reason!r}",
        )
    return trigger_to_canonical_mode(row.export_trigger)


def mesh_handoff_mode(classes: Iterable[PortalMemberClass]) -> str:
    """The MESH-level ``Swarm.handoff`` aggregation over its members' classes.

    ``Swarm.handoff`` is per-MESH while ``SWARM_ENCODING`` is per-CLASS, so the
    combination rule must be NAMED and STATED rather than implicit. Precedence,
    verified against the pre-migration ``any_tool`` check this replaces:
    **'optional' wins over 'never'** -- if ANY member's export_trigger is
    'tool', the mesh handoff is the canonical mode for 'tool'; otherwise it is
    the canonical mode for 'output'.

    DISPATCH must never appear in ``classes`` -- a dispatch-mode Portal is not a
    mesh member (``_portal_member.py``'s own docstring) -- and raises rather
    than silently defaulting.
    """
    classes = list(classes)
    if PortalMemberClass.DISPATCH in classes:
        raise ConfigurationError.build(
            "DISPATCH is not a mesh member class",
            expected="mesh member classes only (ATOMIC/ATOMIC_OPERATOR/AGENT_CYCLE_OUTPUT/"
            "AGENT_CYCLE_TOOL/SUB_CONSTRUCT)",
            found="PortalMemberClass.DISPATCH in the input classes",
        )
    any_tool = any(SWARM_ENCODING[cls].export_trigger == "tool" for cls in classes)
    return trigger_to_canonical_mode("tool" if any_tool else "output")
