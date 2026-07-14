"""PORTAL T4 — three-surface parity + @node sugar (neograph-kk262).

TDD RED artifact for the @node Portal sugar (``portal=`` / ``route=`` /
``max_hops=`` / ``on_exhaust=`` kwargs that build a ``Portal`` modifier, the
exact mirror of ``loop_when=`` -> ``Loop``). The sugar DOES NOT EXIST YET, so
every builder that assembles a mesh via ``@node(portal=...)`` — and every parity
/ conflict test that calls it — FAILS RED now (``portal=`` is an unexpected
kwarg) and turns GREEN once the next atom lands the sugar in ``decorators.py``.

Core Invariant (design §2.2, D5/H2): the SAME 2-member mesh built via all three
in-scope surfaces produces IDENTICAL IR and IDENTICAL runtime results —
``handoff_param`` AND ``handoff_channel`` converge in the normalizer
(``_ir_normalize.py``) for every surface, with NO writes of either field in
``decorators.py`` / ``_construct_builder.py`` (the neograph-ts7 / fan_out_param
single-writer lesson, pinned by guard G3 in test_guards_llm_runtime.py).

Canonical mesh (all three surfaces): entry ``triage`` with ``portal=["billing"]``
and ``max_hops=6`` (the entry-only knob), terminal ``billing`` with ``portal=[]``
(routes only to HANDOFF_END). Payload is ``RouteHop`` with a plain-str ``goto``
routing field and a carried ``hops`` count; every member consumes the shared
mesh channel via the reserved ``handoff`` inputs key. Flow: START -> triage
(channel empty -> route to billing) -> billing (-> HANDOFF_END) -> exit.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import (
    HANDOFF_END,
    Construct,
    ConstructError,
    Node,
    Portal,
    compile,
    construct_from_functions,
    node,
    run,
)
from neograph._state_keys import StateKeys
from neograph.naming import field_name_for
from tests.fakes import build_test_compile_kwargs, register_scripted


class RouteHop(BaseModel, frozen=True):
    """Uniform mesh payload: plain-str route field + carried hop count."""

    goto: str
    hops: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Shared mesh logic — one behavior, three surfaces. Each surface's node bodies
# call these so the RUNTIME routing result is identical by construction.
# ─────────────────────────────────────────────────────────────────────────────


def _triage_logic(handoff: RouteHop | None) -> RouteHop:
    """Entry: first activation (channel empty) hands off to billing; a re-entry
    (populated channel) would leave the mesh. This mesh routes triage->billing
    exactly once."""
    if handoff is None:
        return RouteHop(goto="billing", hops=1)
    return RouteHop(goto=HANDOFF_END, hops=handoff.hops + 1)


def _billing_logic(handoff: RouteHop) -> RouteHop:
    """Terminal specialist (portal=[]): routes only to HANDOFF_END."""
    return RouteHop(goto=HANDOFF_END, hops=handoff.hops + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Three canonical builders for ONE identical mesh.
# ─────────────────────────────────────────────────────────────────────────────


def _km_mesh_declarative(visits: list[str] | None = None) -> Construct:
    """Declarative surface: Node(...) constructor + | Portal."""
    sink: list[str] = visits if visits is not None else []

    def triage_fn(input_data, config):
        sink.append("triage")
        incoming = input_data.get("handoff") if isinstance(input_data, dict) else None
        return _triage_logic(incoming)

    def billing_fn(input_data, config):
        sink.append("billing")
        return _billing_logic(input_data["handoff"])

    register_scripted("km_decl_triage", triage_fn)
    register_scripted("km_decl_billing", billing_fn)

    triage = Node(
        name="triage",
        mode="scripted",
        inputs={"handoff": RouteHop},
        outputs=RouteHop,
        scripted_fn="km_decl_triage",
    ) | Portal(to=["billing"], max_hops=6)
    billing = Node(
        name="billing",
        mode="scripted",
        inputs={"handoff": RouteHop},
        outputs=RouteHop,
        scripted_fn="km_decl_billing",
    ) | Portal(to=[])
    return Construct("km-decl", nodes=[triage, billing])


def _km_mesh_programmatic(visits: list[str] | None = None) -> Construct:
    """Programmatic surface: Node.scripted(...) | Portal."""
    sink: list[str] = visits if visits is not None else []

    def triage_fn(input_data, config):
        sink.append("triage")
        incoming = input_data.get("handoff") if isinstance(input_data, dict) else None
        return _triage_logic(incoming)

    def billing_fn(input_data, config):
        sink.append("billing")
        return _billing_logic(input_data["handoff"])

    register_scripted("km_prog_triage", triage_fn)
    register_scripted("km_prog_billing", billing_fn)

    triage = (
        Node.scripted("triage", fn="km_prog_triage", inputs={"handoff": RouteHop}, outputs=RouteHop)
        | Portal(to=["billing"], max_hops=6)
    )
    billing = (
        Node.scripted("billing", fn="km_prog_billing", inputs={"handoff": RouteHop}, outputs=RouteHop)
        | Portal(to=[])
    )
    return Construct("km-prog", nodes=[triage, billing])


def _km_mesh_decorator(visits: list[str] | None = None) -> Construct:
    """@node decorator surface: portal=/max_hops= sugar building the Portal.

    RED NOW: ``portal=`` is not yet a recognized ``@node`` kwarg, so decorating
    ``triage`` raises before the mesh assembles. The next atom (sugar in
    decorators.py) turns this green with IR identical to the pipe forms above.
    """
    sink: list[str] = visits if visits is not None else []

    @node(outputs=RouteHop, portal=["billing"], max_hops=6)
    def triage(handoff: RouteHop) -> RouteHop:
        sink.append("triage")
        return _triage_logic(handoff)

    @node(outputs=RouteHop, portal=[])
    def billing(handoff: RouteHop) -> RouteHop:
        sink.append("billing")
        return _billing_logic(handoff)

    return construct_from_functions("km-deco", [triage, billing])


_ALL_BUILDERS = [_km_mesh_declarative, _km_mesh_programmatic, _km_mesh_decorator]
_BUILDER_IDS = ["declarative", "programmatic", "decorator"]

# The mesh channel is keyed off the ENTRY member's producer field.
_HANDOFF_CHANNEL = StateKeys.handoff_payload(field_name_for("triage"))


def _node_by_name(construct: Construct, name: str) -> Node:
    for n in construct.nodes:
        if getattr(n, "name", None) == name:
            assert isinstance(n, Node)
            return n
    raise AssertionError(f"node {name!r} not found in {construct.name}")


def _portal_of(construct: Construct, name: str) -> Portal:
    km = _node_by_name(construct, name).get_modifier(Portal)
    assert isinstance(km, Portal), f"{name} carries no Portal"
    return km


class TestThreeSurfacePortalParity:
    """The same Portal mesh built via declarative ``Node(...)``, programmatic
    ``Node.scripted(...) | Portal``, and ``@node(portal=...)`` sugar produces
    IDENTICAL IR (Portal fields + model_fields_set, inferred handoff_param /
    handoff_channel) and IDENTICAL runtime routing.

    ForwardConstruct is EXEMPT in v1 (D-FORWARD-EXEMPT, independent reviewer
    AGREED — 'stronger than the di_inputs analogy'): a runtime mesh has no
    static dataflow for the tracer to thread, so there is nothing for
    ForwardConstruct.forward() to trace into a Portal. The OTHER three
    surfaces are all in-scope and asserted identical here.
    """

    @pytest.mark.parametrize("build", _ALL_BUILDERS, ids=_BUILDER_IDS)
    def test_mesh_assembles_across_surfaces(self, build):
        """Each surface assembles the 2-member mesh without error."""
        mesh = build()
        assert len(mesh.nodes) == 2
        assert {n.name for n in mesh.nodes} == {"triage", "billing"}

    @pytest.mark.parametrize("build", _ALL_BUILDERS, ids=_BUILDER_IDS)
    def test_handoff_param_and_channel_inferred_per_surface(self, build):
        """Every member carries handoff_param=='handoff' and the entry-keyed
        handoff_channel — inferred by the normalizer regardless of surface."""
        mesh = build()
        for member in ("triage", "billing"):
            n = _node_by_name(mesh, member)
            assert n.handoff_param == "handoff", (member, n.handoff_param)
            assert n.handoff_channel == _HANDOFF_CHANNEL, (member, n.handoff_channel)

    def test_handoff_param_identical_across_surfaces(self):
        """handoff_param is IDENTICAL across all three surfaces (per member)."""
        meshes = {bid: build() for bid, build in zip(_BUILDER_IDS, _ALL_BUILDERS, strict=True)}
        for member in ("triage", "billing"):
            values = {bid: _node_by_name(m, member).handoff_param for bid, m in meshes.items()}
            assert len(set(values.values())) == 1, (member, values)
            assert set(values.values()) == {"handoff"}, (member, values)

    def test_handoff_channel_identical_across_surfaces(self):
        """handoff_channel is IDENTICAL across all three surfaces (per member)."""
        meshes = {bid: build() for bid, build in zip(_BUILDER_IDS, _ALL_BUILDERS, strict=True)}
        for member in ("triage", "billing"):
            values = {bid: _node_by_name(m, member).handoff_channel for bid, m in meshes.items()}
            assert len(set(values.values())) == 1, (member, values)
            assert set(values.values()) == {_HANDOFF_CHANNEL}, (member, values)

    def test_portal_ir_identical_across_surfaces(self):
        """The Portal modifier itself — peers, route, max_hops, on_exhaust AND
        model_fields_set — is IDENTICAL across surfaces (per member).

        model_fields_set equality is LOAD-BEARING, not cosmetic:
        _validation_portal.py:189 reads it for the entry-only knob check, so a
        botched conditional-include (defaults passed explicitly) would make the
        terminal member fail assembly under the pipe form but not the sugar.
        """
        meshes = {bid: build() for bid, build in zip(_BUILDER_IDS, _ALL_BUILDERS, strict=True)}
        for member in ("triage", "billing"):
            kms = {bid: _portal_of(m, member) for bid, m in meshes.items()}
            peers = {bid: km.to for bid, km in kms.items()}
            routes = {bid: km.route for bid, km in kms.items()}
            hops = {bid: km.max_hops for bid, km in kms.items()}
            exhausts = {bid: km.on_exhaust for bid, km in kms.items()}
            field_sets = {bid: km.model_fields_set for bid, km in kms.items()}

            assert len({tuple(p) for p in peers.values()}) == 1, (member, peers)
            assert len(set(routes.values())) == 1, (member, routes)
            assert len(set(hops.values())) == 1, (member, hops)
            assert len(set(exhausts.values())) == 1, (member, exhausts)
            assert len({frozenset(fs) for fs in field_sets.values()}) == 1, (member, field_sets)

    def test_runtime_routing_identical_across_surfaces(self):
        """compile + run each surface; the observed routing order is IDENTICAL."""
        results: dict[str, list[str]] = {}
        for bid, build in zip(_BUILDER_IDS, _ALL_BUILDERS, strict=True):
            visits: list[str] = []
            mesh = build(visits)
            graph = compile(mesh, **build_test_compile_kwargs())
            run(graph, input={})
            results[bid] = visits

        assert results["declarative"] == ["triage", "billing"], results
        assert len({tuple(v) for v in results.values()}) == 1, results


class TestPortalDecoratorConflicts:
    """Decoration-time legality of the @node Portal sugar. These mirror the
    map_over/loop_when conflict raises (decorators.py:378) and fail RED now
    because ``portal=`` does not yet exist on ``@node``.
    """

    def test_peers_with_map_over_raises(self):
        """portal= and map_over= on one @node is a decoration-time conflict."""
        with pytest.raises(ConstructError):

            @node(outputs=RouteHop, portal=["billing"], map_over="upstream")
            def bad(handoff: RouteHop) -> RouteHop: ...

    def test_peers_with_loop_when_raises(self):
        """portal= and loop_when= on one @node is a decoration-time conflict."""
        with pytest.raises(ConstructError):

            @node(outputs=RouteHop, portal=["billing"], loop_when=lambda d: d is not None)
            def bad(handoff: RouteHop) -> RouteHop: ...

    def test_max_hops_without_peers_raises(self):
        """max_hops= without portal= is a decoration-time error (a peer-mode knob
        with no mesh to attach to)."""
        with pytest.raises(ConstructError):

            @node(outputs=RouteHop, max_hops=6)
            def bad(handoff: RouteHop) -> RouteHop: ...

    def test_route_without_peers_raises(self):
        """route= without portal= is a decoration-time error."""
        with pytest.raises(ConstructError):

            @node(outputs=RouteHop, route="goto")
            def bad(handoff: RouteHop) -> RouteHop: ...

    def test_on_exhaust_exit_routes_to_portal(self):
        """on_exhaust='exit' with portal= builds a Portal carrying on_exhaust=='exit'."""

        @node(outputs=RouteHop, portal=["billing"], on_exhaust="exit")
        def triage(handoff: RouteHop) -> RouteHop: ...

        @node(outputs=RouteHop, portal=[])
        def billing(handoff: RouteHop) -> RouteHop: ...

        mesh = construct_from_functions("km-exhaust", [triage, billing])
        km = _portal_of(mesh, "triage")
        assert km.on_exhaust == "exit"

    def test_on_exhaust_last_routes_to_loop(self):
        """on_exhaust='last' with loop_when= builds a Loop (not a Portal) —
        the shared on_exhaust kwarg is routed by trigger."""
        from neograph import Loop

        @node(outputs=RouteHop, loop_when=lambda d: d is None, on_exhaust="last")
        def repeat(handoff: RouteHop) -> RouteHop: ...

        n = repeat  # the decorated Node
        assert n.get_modifier(Loop) is not None
        assert n.get_modifier(Portal) is None
