"""Portal mesh routing plumbing — MeshContext and PortalRouteSpec.

neograph-dgbqv.4 (P9 of the Agent Spec dispatch-vocabulary epic).

Every fact a Portal mesh member needs to reach ``Command(goto=...)`` — the
shared channel key, the hop counter, the entry/exit names, the hop budget,
the exhaust policy, the entry-label map, the resolved destination tuple, the
proposed-target field — used to be hand re-derived at each call site in
``_wiring.py``/``factory.py`` instead of built ONCE into a frozen bundle.
``MeshContext`` is built once per mesh (in ``_add_portal_mesh``);
``PortalRouteSpec`` is built once per member.

A neutral leaf module both ``_wiring.py`` and ``factory.py`` can reach —
placement copies ``_agent_cycle_names.py``; shape copies
``_llm_runtime.LlmRuntime`` (``@dataclass(frozen=True)`` + a ``build()``
classmethod). NO ``Command`` import and no back-import of ``factory`` — that
would widen guard G1 (``TestCommandConstructionMonopoly``,
``tests/test_guards_assembly.py``) or force a function-local import, both
refused by AGENTS.md's file-split ladder. This module is PURE DATA: no
``to_command`` method — the two decision functions keep their names and
their home in ``factory.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from neograph._agent_cycle_names import cycle_names
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._normalize import primary_output_field
from neograph._portal import HANDOFF_END
from neograph._portal_member import PortalMemberClass, portal_member_class
from neograph._state_keys import StateKeys
from neograph.naming import field_name_for

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.checkpoint.base import BaseCheckpointSaver
    from pydantic import BaseModel

    from neograph._ir_protocols import ConstructItem
    from neograph._portal import Portal
    from neograph.construct import Construct
    from neograph.node import Node


@dataclass(frozen=True)
class MeshContext:
    """Mesh-wide routing facts, built ONCE per mesh in ``_add_portal_mesh``.

    ``entry_label_map`` is the mesh's DX-visible-peer-name -> real
    LangGraph-node-name map: an agent/act member's real entry is
    ``{member.name}__agent`` (its Portal-visible boundary port, per the Core
    Invariant: routing resolves to an entry, never a region's interior
    ``__tools``/loopback nodes); every other member class maps to itself.
    Built via ``portal_member_class`` — replacing the ``getattr(member,
    "mode", None) in ("agent", "act")`` re-derivation that used to live at
    ``_wiring.py``.
    """

    entry_field: str
    channel_key: str
    count_field: str
    exit_name: str
    entry_name: str
    max_hops: int
    on_exhaust: str
    entry_label_map: dict[str, str]

    @classmethod
    def build(cls, members: list[ConstructItem]) -> MeshContext:
        entry = members[0]
        entry_field = field_name_for(entry.name)
        exit_name = f"__handoff_exit_{entry.name}"
        entry_label_map = {
            member.name: (
                cycle_names(member.name).agent
                if portal_member_class(member)
                in (PortalMemberClass.AGENT_CYCLE_OUTPUT, PortalMemberClass.AGENT_CYCLE_TOOL)
                else member.name
            )
            for member in members
        }
        # max_hops/on_exhaust are ENTRY-only knobs (T1 validation); the wrapper
        # runs per member, so the budget is sourced once from the entry here.
        entry_portal = entry.modifier_set.portal
        assert entry_portal is not None
        return cls(
            entry_field=entry_field,
            channel_key=StateKeys.handoff_payload(entry_field),
            count_field=StateKeys.handoff_hops(entry_field),
            exit_name=exit_name,
            entry_name=entry.name,
            max_hops=entry_portal.max_hops,
            on_exhaust=entry_portal.on_exhaust,
            entry_label_map=entry_label_map,
        )

    def resolved_peers(self, portal: Portal) -> tuple[str, ...]:
        """Peers ONLY, resolved through the entry-label map. No ``exit_name``.

        The ``_wiring.py:745`` ``peer_targets`` shape — feeds an
        AGENT_CYCLE_TOOL member's ``tools_destinations``.
        """
        return tuple(self.entry_label_map.get(t, t) for t in (portal.to or ()))

    def destinations_for(self, portal: Portal) -> tuple[str, ...]:
        """Peers + ``exit_name``. The ``_wiring.py`` ``:358``/``:415``/``:424``/
        ``:746`` shape — every ``graph.add_node(..., destinations=...)`` call
        except an AGENT_CYCLE_TOOL member's ``tools_destinations``.

        Deliberately a DIFFERENT method from :meth:`resolved_peers`, not a
        shared one with an optional flag: collapsing them would silently
        widen a tool-triggered member's declared goto target set, and
        ``destinations=`` is validation/rendering metadata no behavioral test
        asserts on.
        """
        return self.resolved_peers(portal) + (self.exit_name,)


@dataclass(frozen=True)
class PortalRouteSpec:
    """One member's frozen routing bundle, read by the decision functions
    instead of threaded as N separate kwargs."""

    payload_field: str
    route_field: str
    valid_targets: frozenset[str]
    node_name: str
    approve_name: str | None = None
    proposed_field: str | None = None
    handoff_target_key: str | None = None
    loopback_target: str | None = None

    @classmethod
    def for_node(
        cls,
        node: Node,
        portal: Portal,
        ctx: MeshContext,
        *,
        approve_name: str | None = None,
    ) -> PortalRouteSpec:
        field_name = field_name_for(node.name)
        return cls(
            payload_field=primary_output_field(field_name, node.outputs),
            route_field=portal.route,
            valid_targets=frozenset(portal.to or ()) | {HANDOFF_END},
            node_name=node.name,
            approve_name=approve_name,
            proposed_field=StateKeys.portal_proposed_target(field_name) if approve_name else None,
        )

    @classmethod
    def for_sub_construct(cls, sub: Construct, portal: Portal, ctx: MeshContext) -> PortalRouteSpec:
        return cls(
            payload_field=field_name_for(sub.name),
            route_field=portal.route,
            valid_targets=frozenset(portal.to or ()) | {HANDOFF_END},
            node_name=sub.name,
        )

    @classmethod
    def for_tool_member(cls, node: Node, portal: Portal, ctx: MeshContext) -> PortalRouteSpec:
        """The tool-triggered-handoff variant: ``handoff_target_key`` (=
        ``StateKeys.handoff_tool_target(field_name)``) and ``loopback_target``
        (= ``cycle_names(node.name).agent``, the existing primitive — no
        hand-rolled ``'{name}__agent'`` f-string) travel ON the spec, so
        ``_tool_handoff_to_command`` reads them off ``spec`` instead of
        re-deriving them at the ``_wiring.py``/``factory.py`` call site
        (architect review finding 2, neograph-jn555.20)."""
        field_name = field_name_for(node.name)
        return cls(
            payload_field=primary_output_field(field_name, node.outputs),
            route_field=portal.route,
            valid_targets=frozenset(portal.to or ()) | {HANDOFF_END},
            node_name=node.name,
            handoff_target_key=StateKeys.handoff_tool_target(field_name),
            loopback_target=cycle_names(node.name).agent,
        )


@dataclass(frozen=True)
class MeshDeps:
    """Compile-time dependency bundle threaded into the member adapters,
    replacing six individually-threaded kwargs on
    ``_make_portal_subgraph_member_fn``/``_add_portal_agent_cycle_member``.

    ``checkpointer`` is typed under ``TYPE_CHECKING`` only (mirrors
    ``_hot_swap.py``'s ``BaseCheckpointSaver`` import) so this leaf module
    adds no runtime ``langgraph-checkpoint`` edge.
    """

    checkpointer: BaseCheckpointSaver | None = None
    parent_state_model: type[BaseModel] | None = None
    runtime: LlmRuntime = EMPTY_RUNTIME
    scripted_lookup: dict[str, Callable] | None = None
    condition_lookup: dict[str, Callable] | None = None
    tool_factory_lookup: dict[str, Callable] | None = None
