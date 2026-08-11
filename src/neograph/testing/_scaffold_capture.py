"""The modifier-capture registry + Portal mesh assertion emission for the scaffold.

Split out of ``scaffold.py`` rather than added to it: that module sits at its exact
file-size ceiling, and the ratchet blocks growth. The seam is clean -- nothing here
is called except by the generators, and nothing here reaches back into them.

WHY PORTAL NEEDS ITS OWN MODULE AT ALL, when the other four modifiers are one dict
literal each: a mesh is a property of a contiguous RUN of sibling members, not of a
single node. Entry identity is POSITIONAL (``members[0]`` carries the entry-only
knobs and is the only legal jump target), membership is a SET property, and a member
may be a ``Construct`` rather than a ``Node``. None of that fits the per-node capture
dict, so the per-node half here records only what a node knows about itself and
:func:`collect_meshes` records the rest.

Both grouping and classification route through the declared single authorities --
``_group_portal_members`` and ``portal_member_class`` -- never a re-derived walk. A
scaffold that re-implemented grouping would replant the duplicated-source-of-truth
disease this ticket exists to remove.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from neograph._portal import _group_portal_members
from neograph._portal_member import PortalMemberClass, portal_member_class
from neograph.construct import Construct
from neograph.modifiers import _SLOT_RULES
from neograph.naming import field_name_for


def portal_capture(item: Any) -> dict[str, Any]:
    """What ONE node knows about its own Portal.

    Takes the ITEM, not the Portal, so the peer/dispatch split is read from
    ``portal_member_class`` -- the declared authority -- rather than hand-derived
    from ``is_dispatch`` here. Two derivations of "what kind of participant is
    this" is the duplication guard G-PMC exists to prevent.

    The entry-only knobs (``max_hops``/``on_exhaust``) are captured RAW -- ``None``
    means the author never set them. Since neograph-dgbqv.6 those fields are value
    sentinels, so an emitter that captured ``effective_*`` could not tell "unset"
    from "explicitly 10" and would bake ``assert portal.max_hops == 10`` into every
    generated suite for meshes that never chose a budget. A fabricated assertion is
    the "reports covering without covering" failure this whole ticket refuses.
    """
    portal = item.modifier_set.portal
    member_class = _member_class_name(item)
    return {
        "member_class": member_class,
        "is_dispatch": member_class == PortalMemberClass.DISPATCH.name,
        "to": list(portal.to) if portal.to else None,
        "trigger": portal.trigger,
        "max_hops": portal.max_hops,
        "on_exhaust": portal.on_exhaust,
        "name": portal.name,
        # dispatch-mode knobs; all None in peer mode
        "spec_field": portal.spec_field,
        "input_field": portal.input_field,
        "max_depth": portal.max_depth,
        "on_invalid": portal.on_invalid,
        "error_handler": portal.error_handler,
    }


def _member_class_name(item: Any) -> str | None:
    cls = portal_member_class(item)
    return None if cls is None else cls.name


def _member_row(item: Any) -> dict[str, Any]:
    """One mesh member's own routing facts (raw, per portal_capture's docstring)."""
    portal = item.modifier_set.portal
    assert portal is not None  # grouped as Portal-modified
    return {
        "name": item.name,
        "to": list(portal.to or []),
        "trigger": portal.trigger,
        "member_class": _member_class_name(item),
    }


def collect_meshes(construct: Construct) -> list[dict[str, Any]]:
    """One dict per Portal peer mesh in ``construct``, in declaration order.

    Dispatch-mode Portals are deliberately excluded: a ``route="decide"`` Portal is a
    standalone linear node, never a mesh member, so it is covered by the ordinary
    per-node assertions instead.
    """
    members = [
        item
        for item in construct.nodes
        if (cls := portal_member_class(item)) is not None and cls is not PortalMemberClass.DISPATCH
    ]
    if not members:
        return []

    meshes: list[dict[str, Any]] = []
    for group_name, run in _group_portal_members(members).items():
        entry = run[0]
        entry_portal = entry.modifier_set.portal
        assert entry_portal is not None  # grouped as Portal-modified
        meshes.append(
            {
                "name": group_name,
                "field": field_name_for(getattr(entry, "name", "mesh")),
                "members": [m.name for m in run],
                "entry": entry.name,
                # RAW, per portal_capture's docstring: None means "never set".
                "max_hops": entry_portal.max_hops,
                "on_exhaust": entry_portal.on_exhaust,
                "per_member": [
                    _member_row(m)
                    for m in run
                ],
            }
        )
    return meshes


def gen_portal_node_asserts(fname: str, portal: dict[str, Any]) -> list[str]:
    """Per-node assertions for a DISPATCH-mode Portal.

    Peer-mode members get nothing here -- everything true of them is a mesh property
    and is asserted by :func:`gen_mesh_tests`, so duplicating it per node would give
    two places to drift.
    """
    if not portal["is_dispatch"]:
        return []
    lines = [
        f"        portal = {fname}.modifier_set.portal",
        "        assert portal is not None",
        "        assert portal.is_dispatch",
    ]
    # Conditional emission for optional knobs, matching the oracle/merge_prompt
    # precedent already in scaffold.py: assert what the author actually chose,
    # never a default nobody wrote.
    for key in ("spec_field", "input_field", "max_depth", "on_invalid", "error_handler"):
        if portal[key] is not None:
            lines.append(f"        assert portal.{key} == {portal[key]!r}")
    return lines


def gen_mesh_tests(meshes: list[dict[str, Any]], construct_var: str) -> list[str]:
    """A ``TestPortalMesh`` class per mesh.

    Ordered membership is asserted against ``_group_portal_members`` on the LIVE
    construct, which pins membership, order, and therefore entry identity and
    contiguity in one assertion -- reordering two siblings silently changes which
    member owns the budget and which is the legal jump target, and that reordering
    is invisible to any per-node check.
    """
    if not meshes:
        return []
    lines: list[str] = [
        "",
        "",
        "class TestPortalMesh:",
        '    """Portal peer-mesh assertions (auto-generated).',
        "",
        "    Membership is read through neograph's own grouping authority rather than",
        "    re-derived here, so this test cannot disagree with the compiler about what",
        "    the mesh IS.",
        '    """',
    ]
    for i, mesh in enumerate(meshes):
        suffix = f"_{i}" if len(meshes) > 1 else ""
        lines += [
            "",
            f"    def test_mesh{suffix}_membership_and_order(self):",
            "        from neograph._portal import _group_portal_members",
            "        from neograph._portal_member import portal_member_class",
            "",
            f"        members = [n for n in {construct_var}.nodes if portal_member_class(n) is not None]",
            "        groups = _group_portal_members(members)",
            f"        run = groups[{mesh['name']!r}]",
            f"        assert [m.name for m in run] == {mesh['members']!r}, (",
            '            "mesh membership or ORDER changed -- members[0] is the entry, so a "',
            '            "reorder silently moves the budget and the legal jump target"',
            "        )",
            f"        assert run[0].name == {mesh['entry']!r}",
        ]
        entry_asserts = [
            f"        assert entry_portal.{k} == {mesh[k]!r}"
            for k in ("max_hops", "on_exhaust")
            if mesh[k] is not None
        ]
        if entry_asserts:
            lines += [
                "",
                f"    def test_mesh{suffix}_entry_knobs(self):",
                "        from neograph._portal import _group_portal_members",
                "        from neograph._portal_member import portal_member_class",
                "",
                f"        members = [n for n in {construct_var}.nodes if portal_member_class(n) is not None]",
                f"        entry_portal = _group_portal_members(members)[{mesh['name']!r}][0].modifier_set.portal",
                *entry_asserts,
            ]
        lines += [
            "",
            f"    def test_mesh{suffix}_per_member_routing(self):",
            "        from neograph._portal_member import portal_member_class",
            "",
            f"        by_name = {{n.name: n for n in {construct_var}.nodes}}",
            f"        for expected in {mesh['per_member']!r}:",
            '            item = by_name[expected["name"]]',
            "            portal = item.modifier_set.portal",
            '            assert list(portal.to or []) == expected["to"]',
            '            assert portal.trigger == expected["trigger"]',
            '            assert portal_member_class(item).name == expected["member_class"], (',
            '                f"{expected[\'name\']} changed PortalMemberClass -- its lowering changed"',
            "            )",
        ]
    return lines


# --- the capture registry ---------------------------------------------------
#
# One entry per modifier SLOT. The five keys are NOT hand-listed: MODIFIER_CAPTURE
# is derived from ``modifiers._SLOT_RULES``, the declared roster whose own comment
# says "adding a new modifier means adding ONE row here". Deriving rather than
# re-listing is what makes a sixth modifier impossible to miss -- it becomes a loud
# KeyError when ``neograph.testing.scaffold`` is imported, instead of a silent
# fall-through that emits no assertions (the neograph-wvp7j defect).
#
# NOTE the guard consequence: a `set(MODIFIER_CAPTURE) == {r.slot for r in
# _SLOT_RULES}` check would now be VACUOUSLY true. The real guard asserts each
# capture returns a dict for a node carrying that modifier, and that _node_info
# emits a key per slot -- see tests/test_guards_scaffold_modifier_totality.py.

# Every capture takes the NODE, not the modifier, so the table is uniform and needs
# no per-slot special case -- Portal's capture must see the item anyway, to read its
# member class from the classifier rather than hand-deriving it.
_CAPTURE_FNS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "oracle": lambda n: {
        "n": n.modifier_set.oracle.n,
        "merge_fn": n.modifier_set.oracle.merge_fn,
        "merge_prompt": n.modifier_set.oracle.merge_prompt,
    },
    "each": lambda n: {"over": n.modifier_set.each.over, "key": n.modifier_set.each.key},
    "loop": lambda n: {
        "max_iterations": n.modifier_set.loop.max_iterations,
        "on_exhaust": n.modifier_set.loop.on_exhaust,
    },
    "operator": lambda n: {"when": repr(n.modifier_set.operator.when)},
    "portal": portal_capture,
}

MODIFIER_CAPTURE = {r.slot: _CAPTURE_FNS[r.slot] for r in _SLOT_RULES}


def capture_modifiers(node: Any) -> dict[str, Any]:
    """``{slot: captured-dict-or-None}`` for every modifier in the roster."""
    return {
        slot: (None if getattr(node.modifier_set, slot) is None else fn(node))
        for slot, fn in MODIFIER_CAPTURE.items()
    }
