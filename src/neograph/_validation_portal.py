"""Portal mesh assembly validation (design §5) — a validation-cluster peer.

``_check_portal_mesh`` enforces every design-§5 mesh rule at ONE construct
level, raising a ``ConstructError`` that names the offender. Extracted from
``_validation_modifiers`` (which stays under its line cap) as a focused peer
module, mirroring the neograph-e8jg split-by-concern discipline. Package-private:
imported only from within the validation cluster (``_construct_validation``
re-exports the public surface; nothing outside imports a ``_validation_*``
module directly).

ANTI-BAND-AID: the mesh rules live here in the assembly walk, never inlined in
decorators or the compiler. ``effective_producer_type`` is deliberately NOT
consulted — mesh members produce their declared output type unchanged.
"""

from __future__ import annotations

from typing import Any, Literal, cast, get_args, get_origin

from neograph._ir_protocols import ConstructLike
from neograph._normalize import _declared_output, normalize_outputs
from neograph._validation_types import _MISSING, _fmt_type, _resolve_field_annotation, _source_location
from neograph.errors import ConstructError
from neograph.modifiers import HANDOFF_END, Portal, _group_portal_members
from neograph.node import Node


def _member_portal(node: Node) -> Portal:
    """The Portal on a mesh member — non-None by construction.

    Members are collected by ``node.modifier_set.portal is not None``; this
    narrows the ``Portal | None`` slot back to ``Portal`` for the rules that
    read ``to`` / ``route`` / ``model_fields_set``.
    """
    km = node.modifier_set.portal
    assert km is not None  # collected as Portal-modified
    return km


def _check_portal_mesh(construct: ConstructLike) -> None:
    """Validate every Portal mesh at ONE construct level (design §5, extended
    by neograph-fefar to >1 NAMED mesh per level).

    A **mesh** is one NAMED group (``Portal.name``, ``None`` = the implicit
    default group) of Portal-modified sibling Nodes at this level. Every
    design-§5 assembly rule is enforced PER GROUP -- each group is checked
    completely independently (own contiguity, own uniform payload, own
    connected-component, own entry-only knobs, own route field) via
    :func:`_group_portal_members`, the SAME shared grouping helper the IR
    normalizer and the compiler's mesh collector use -- never a re-derived
    inline grouping (ANTI-BAND-AID: a construct that validates must lower
    exactly as validated).

    Called once per construct level from ``_validate_node_chain`` (which recurses
    into sub-constructs), so a mesh at any depth is checked.
    """
    nodes = list(construct.nodes)
    # PEER-mode members only. A dispatch-mode Portal (route="decide") is NOT a
    # mesh member (review M1): it is a standalone linear node whose payload is the
    # emitted spec, not a routed handoff, so the route-field/payload-uniformity
    # checks below do not apply. Including it here would look for a field literally
    # named "decide" on its payload and reject a valid dispatch node.
    member_positions = {
        id(item): i
        for i, item in enumerate(nodes)
        if getattr(item, "modifier_set", None) is not None
        and item.modifier_set.portal is not None
        and not item.modifier_set.portal.is_dispatch
    }
    if not member_positions:
        return

    all_members = [nodes[i] for i in member_positions.values()]
    sibling_names = {getattr(n, "name", None) for n in nodes}

    # A sibling producer literally named "handoff" collides with the reserved
    # mesh-channel input key (design §3.3 / D-RESERVED-KEY). Construct-wide,
    # not per-group.
    for item in nodes:
        if getattr(item, "name", None) == "handoff":
            raise ConstructError.build(
                "a node named 'handoff' collides with the reserved Portal mesh input key",
                found="rename the node; 'handoff' is reserved for the mesh channel",
                construct=construct.name,
                location=_source_location(),
            )

    groups = _group_portal_members(all_members)
    for group_name, members in groups.items():
        _check_one_mesh_group(construct, members, member_positions, nodes, sibling_names, group_name)


def _check_one_mesh_group(
    construct: ConstructLike,
    members: list[Any],
    member_positions: dict[int, int],
    nodes: list[Any],
    sibling_names: set[str | None],
    group_name: str | None,
) -> None:
    """Every design-§5 assembly rule for ONE named mesh group.

    Extracted from the single-mesh-per-level ``_check_portal_mesh`` so each
    group gets the identical rule set applied independently -- unchanged
    logic, just scoped to ``members`` (one group) instead of every Portal
    member at the level.
    """
    member_names = {getattr(m, "name", None) for m in members}
    group_positions = [member_positions[id(m)] for m in members]

    # Member SHAPE checks first: an agent/dict member has no single payload
    # type, so the payload/route checks below would be meaningless.
    #
    # do0d9 (§4 Q2): a Construct member is ADMITTED as a first-class mesh member
    # — its declared boundary output (``_declared_output``) must be the uniform
    # mesh payload, checked by the uniform-payload rule below exactly as for a
    # Node member. The former blanket ``isinstance(member, Node)`` rejection is
    # relaxed. The dict-form check stays Node-only: a Construct's boundary is a
    # single ``.output`` type (no dict-form analog).
    for member in members:
        name = getattr(member, "name", "?")
        # D-MEMBER-MODES (dynamic-handoff-2026-07-13.md): v1 rejected agent/act
        # members here (their multi-node ReAct cycle needed separate
        # terminal-hop Command plumbing). Resolved by neograph-nnds9 — the
        # rejection is narrowed, not deleted: agent/act members still satisfy
        # every OTHER check in this function (uniform payload, route field,
        # dict-outputs-forbidden, contiguity, peer existence, single-mesh,
        # entry-only max_hops/on_exhaust, reserved handoff key) mode-
        # independently via Node.outputs/_declared_output, unchanged below.
        if isinstance(member, Node) and normalize_outputs(member.outputs).is_dict_form:
            raise ConstructError.build(
                f"Portal mesh member '{name}' declares dict-form outputs",
                expected="a single payload output type (D-DICT-OUTPUTS)",
                found="outputs={...}",
                node=name,
                construct=construct.name,
                location=_source_location(),
            )

        # neograph-kdr1u (D4 lift): the approval-node splice is implemented for
        # ATOMIC members only (scripted/think/raw). An agent/act member or a
        # Construct member carrying an Operator gate is NARROWED-REJECTED —
        # never silently accepted with a ban lift wider than the splice
        # actually covers (D4's own silent-never-fires failure mode, just one
        # member class later).
        member_operator = getattr(getattr(member, "modifier_set", None), "operator", None)
        if member_operator is not None and (not isinstance(member, Node) or member.mode in ("agent", "act")):
            raise ConstructError.build(
                f"Portal mesh member '{name}' combines Operator with a "
                f"{'sub-construct' if not isinstance(member, Node) else member.mode + '-mode'} member",
                expected="Operator+Portal approval gate on an ATOMIC (scripted/think/raw) member only",
                found=f"Operator on a {'sub-construct' if not isinstance(member, Node) else member.mode} member",
                node=name,
                construct=construct.name,
                location=_source_location(),
                hint="agent/act and sub-construct mesh members do not yet support the approval-node splice",
            )

    # Past the shape checks every member is a Portal-modified Node OR a
    # Portal-modified Construct. The rules below read only the Portal-agnostic
    # surface (``.name`` / ``_declared_output`` / ``modifier_set.portal``); the
    # one Node-only rule (reserved handoff-key typing) guards on isinstance.
    # ``cast`` keeps the typed Node surface for the Portal-modifier accessors.
    node_members = cast("list[Node]", members)

    # Contiguity WITHIN this group: this group's own members occupy consecutive
    # positions among THEMSELVES (another group's members interleaved would
    # split this group, same as a plain non-Portal node would).
    lo, hi = min(group_positions), max(group_positions)
    if hi - lo + 1 != len(group_positions):
        gap_names = [getattr(nodes[i], "name", "?") for i in range(lo, hi + 1) if i not in group_positions]
        raise ConstructError.build(
            f"Portal mesh members must be contiguous in the construct{f' (mesh {group_name!r})' if group_name else ''}",
            expected="all mesh members adjacent, entry first",
            found=f"non-member nodes split the mesh: {gap_names}",
            construct=construct.name,
            location=_source_location(),
        )

    entry = node_members[0]
    payload = _declared_output(entry)  # single type (dict-form rejected above)

    # Uniform payload: every member declares the SAME single output type.
    for member in node_members[1:]:
        if _declared_output(member) is not payload:
            raise ConstructError.build(
                f"Portal mesh member '{member.name}' has a payload type that differs from the entry's",
                expected=f"all members produce {_fmt_type(payload)} (uniform payload)",
                found=_fmt_type(_declared_output(member)),
                node=member.name,
                construct=construct.name,
                location=_source_location(),
            )

    # Peers: every peer names a Portal-modified sibling IN THIS SAME GROUP.
    for member in node_members:
        for peer in _member_portal(member).to or []:
            if peer not in sibling_names:
                raise ConstructError.build(
                    f"Portal member '{member.name}' names peer '{peer}' which does not exist",
                    expected=f"a sibling node in: {sorted(n for n in sibling_names if n)}",
                    found=f"unknown peer '{peer}'",
                    node=member.name,
                    construct=construct.name,
                    location=_source_location(),
                )
            if peer not in member_names:
                raise ConstructError.build(
                    f"Portal member '{member.name}' names peer '{peer}' which is not in the same mesh",
                    expected=f"every peer must be a member of this mesh ({sorted(n for n in member_names if n)})",
                    found=f"'{peer}' is not a member of this mesh",
                    node=member.name,
                    construct=construct.name,
                    location=_source_location(),
                )

    # Single mesh per GROUP: members form ONE connected component under the peer
    # relation (treated undirected). Two disjoint closures within one group = two
    # meshes sharing a name (still illegal -- D-SINGLE-MESH, now per-group).
    adjacency: dict[str, set[str]] = {m.name: set() for m in node_members}
    for member in node_members:
        for peer in _member_portal(member).to or []:
            adjacency[member.name].add(peer)
            adjacency[peer].add(member.name)
    seen: set[str] = set()
    stack: list[str] = [entry.name]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adjacency[cur] - seen)
    if seen != set(adjacency):
        unreached = sorted(set(adjacency) - seen)
        raise ConstructError.build(
            "two disjoint Portal meshes at one construct level"
            + (f" (mesh {group_name!r})" if group_name else ""),
            expected="one connected mesh per group per level (D-SINGLE-MESH)",
            found=f"members not reachable from entry '{entry.name}': {unreached}",
            construct=construct.name,
            location=_source_location(),
        )

    # max_hops / on_exhaust are entry-only knobs (per group).
    for member in node_members[1:]:
        for knob in ("max_hops", "on_exhaust"):
            if knob in _member_portal(member).model_fields_set:
                raise ConstructError.build(
                    f"Portal member '{member.name}' sets {knob} but it is entry-only",
                    expected=f"set {knob} only on the mesh entry '{entry.name}'",
                    found=f"{knob} set on a non-entry member",
                    node=member.name,
                    construct=construct.name,
                    location=_source_location(),
                )

    # Route field: on the payload model, annotated str or Literal[...]; Literal
    # targets must be ⊆ member names ∪ {HANDOFF_END} (typed-swarm, design §3.2).
    route = _member_portal(entry).route
    ann = _resolve_field_annotation(payload, route)
    if ann is _MISSING:
        raise ConstructError.build(
            f"Portal route field '{route}' is missing from the payload model {_fmt_type(payload)}",
            expected=f"a '{route}' field annotated str or Literal[...]",
            found="no such field on the payload model",
            construct=construct.name,
            location=_source_location(),
        )
    if ann is str:
        pass
    elif get_origin(ann) is Literal:
        valid = member_names | {HANDOFF_END}
        for value in get_args(ann):
            if value not in valid:
                raise ConstructError.build(
                    f"Portal route Literal names '{value}' which is not a mesh member or HANDOFF_END",
                    expected=f"each Literal target in {sorted(str(v) for v in valid)}",
                    found=f"stray target '{value}'",
                    construct=construct.name,
                    location=_source_location(),
                )
    else:
        raise ConstructError.build(
            f"Portal route field '{route}' must be annotated str or Literal[...]",
            expected="str or Literal[...]",
            found=_fmt_type(ann),
            construct=construct.name,
            location=_source_location(),
        )

    # Reserved handoff input: a member declaring inputs={'handoff': T} must type
    # T as the payload model (the mesh channel type, design §3.3). This rule is
    # Node-ONLY by design (§4 Q2): a Construct member's boundary port is its
    # singular ``.input`` (typed + validated by _add_subgraph's own boundary
    # check), not a fan-in ``inputs`` dict — there is no Construct analog to a
    # reserved ``handoff`` inputs key, so a Construct member is skipped here.
    for member in node_members:
        if not isinstance(member, Node):
            continue
        inputs = member.inputs
        if isinstance(inputs, dict) and "handoff" in inputs and inputs["handoff"] is not payload:
            raise ConstructError.build(
                f"Portal member '{member.name}' types its 'handoff' input as "
                f"{_fmt_type(inputs['handoff'])}, not the payload model {_fmt_type(payload)}",
                expected=f"handoff: {_fmt_type(payload)}",
                found=_fmt_type(inputs["handoff"]),
                node=member.name,
                construct=construct.name,
                location=_source_location(),
            )


def _check_portal_dispatch_error_handler(construct: ConstructLike) -> None:
    """Validate a dispatch-mode Portal's ``error_handler`` names a real sibling. A SEPARATE check from ``_check_portal_mesh``: that
    function deliberately excludes dispatch-mode Portals (they are not mesh
    members), so a dispatch-only construct is never visited by it. Mirrors
    the mesh peer-existence check's fail-loud-at-assembly discipline.
    """
    nodes = list(construct.nodes)
    sibling_names = {getattr(item, "name", None) for item in nodes}
    for item in nodes:
        if not isinstance(item, Node):
            continue
        portal = item.modifier_set.portal
        if portal is None or not portal.is_dispatch:
            continue
        if portal.on_invalid != "route_to_error":
            continue
        if portal.error_handler not in sibling_names:
            raise ConstructError.build(
                f"node {item.name!r}'s Portal.error_handler {portal.error_handler!r} "
                "does not name a sibling Node in this construct",
                expected=f"one of {sorted(n for n in sibling_names if n is not None)}",
                found=repr(portal.error_handler),
                node=item.name,
                construct=construct.name,
                location=_source_location(),
            )
