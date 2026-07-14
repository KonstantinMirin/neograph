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

from typing import Literal, cast, get_args, get_origin

from neograph._ir_protocols import ConstructLike
from neograph._normalize import _declared_output, normalize_outputs
from neograph._validation_types import _MISSING, _fmt_type, _resolve_field_annotation, _source_location
from neograph.errors import ConstructError
from neograph.modifiers import HANDOFF_END, Portal
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
    """Validate the Portal mesh at ONE construct level (design §5).

    A **mesh** is the set of Portal-modified sibling Nodes at this level. This
    single helper enforces every design-§5 assembly rule, raising a
    ``ConstructError`` that names the offender (ANTI-BAND-AID: the mesh rules
    live here in the validation walk, never inlined in decorators/compiler).
    ``effective_producer_type`` is deliberately NOT consulted — mesh members
    produce their declared output type unchanged.

    Called once per construct level from ``_validate_node_chain`` (which recurses
    into sub-constructs), so a mesh at any depth is checked.
    """
    nodes = list(construct.nodes)
    # PEER-mode members only. A dispatch-mode Portal (route="decide") is NOT a
    # mesh member (review M1): it is a standalone linear node whose payload is the
    # emitted spec, not a routed handoff, so the route-field/payload-uniformity
    # checks below do not apply. Including it here would look for a field literally
    # named "decide" on its payload and reject a valid dispatch node.
    member_positions = [
        i
        for i, item in enumerate(nodes)
        if getattr(item, "modifier_set", None) is not None
        and item.modifier_set.portal is not None
        and not item.modifier_set.portal.is_dispatch
    ]
    if not member_positions:
        return

    members = [nodes[i] for i in member_positions]
    member_names = {getattr(m, "name", None) for m in members}
    sibling_names = {getattr(n, "name", None) for n in nodes}

    # A sibling producer literally named "handoff" collides with the reserved
    # mesh-channel input key (design §3.3 / D-RESERVED-KEY).
    for item in nodes:
        if getattr(item, "name", None) == "handoff":
            raise ConstructError.build(
                "a node named 'handoff' collides with the reserved Portal mesh input key",
                found="rename the node; 'handoff' is reserved for the mesh channel",
                construct=construct.name,
                location=_source_location(),
            )

    # Member SHAPE checks first: a Construct/agent/dict member has no single
    # payload type, so the payload/route checks below would be meaningless.
    for member in members:
        name = getattr(member, "name", "?")
        if not isinstance(member, Node):
            raise ConstructError.build(
                f"Portal mesh member '{name}' is a Construct",
                expected="mesh members must be sibling Nodes (D-MESH-LEVEL)",
                found="a sub-construct carries a Portal modifier",
                construct=construct.name,
                location=_source_location(),
            )
        if member.mode in ("agent", "act"):
            raise ConstructError.build(
                f"Portal mesh member '{name}' is an {member.mode}-mode node",
                expected="mesh members must be scripted/think/raw (D-MEMBER-MODES)",
                found=f"mode={member.mode}",
                node=name,
                construct=construct.name,
                location=_source_location(),
            )
        if normalize_outputs(member.outputs).is_dict_form:
            raise ConstructError.build(
                f"Portal mesh member '{name}' declares dict-form outputs",
                expected="a single payload output type (D-DICT-OUTPUTS)",
                found="outputs={...}",
                node=name,
                construct=construct.name,
                location=_source_location(),
            )

    # Past the shape checks every member is a Portal-modified Node (else we
    # raised). Narrow the list so the rules below read the typed surface.
    node_members = cast("list[Node]", members)

    # Contiguity: members occupy consecutive positions; the first is the entry.
    lo, hi = member_positions[0], member_positions[-1]
    if hi - lo + 1 != len(member_positions):
        gap_names = [getattr(nodes[i], "name", "?") for i in range(lo, hi + 1) if i not in member_positions]
        raise ConstructError.build(
            "Portal mesh members must be contiguous in the construct",
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

    # Peers: every peer names a Portal-modified sibling.
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
                    f"Portal member '{member.name}' names peer '{peer}' which is not Portal-modified",
                    expected="every peer must be a Portal mesh member",
                    found=f"'{peer}' is a plain (non-Portal) sibling",
                    node=member.name,
                    construct=construct.name,
                    location=_source_location(),
                )

    # Single mesh per level: members form ONE connected component under the peer
    # relation (treated undirected). Two disjoint closures = two meshes.
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
            "two disjoint Portal meshes at one construct level",
            expected="one connected mesh per level (D-SINGLE-MESH)",
            found=f"members not reachable from entry '{entry.name}': {unreached}",
            construct=construct.name,
            location=_source_location(),
        )

    # max_hops / on_exhaust are entry-only knobs.
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
    # T as the payload model (the mesh channel type, design §3.3).
    for member in node_members:
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
