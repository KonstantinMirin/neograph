"""Phase-5 reuse-and-delete Agent Spec fixes (neograph-s7zt3.9).

Five independently-scoped fixes, each a failing-first TDD pin (master doc
docs/design/agent-spec-portal-master-architecture-2026-07-27.md §5 + Build
Plan Phase 5):

  A1 -- ``to_agent_spec``'s Portal mesh-detection filter was ``isinstance(Node)``
        gated and misclassified a mesh with a Construct member as
        "mixed mesh/non-mesh". Now reuses the modifier-agnostic
        ``classify_modifiers`` (via ``_is_peer_mesh_member``).
  A5 -- the loader routes the reconstructed Swarm mesh through the same
        ``_check_portal_mesh`` assembly gate every hand-written mesh gets
        (already true via ``Construct(...)`` -- pinned here so it cannot regress).
  B3 -- ``_MARK_PORTAL_SPEC``'s max_hops/on_exhaust/route are read back in
        ``_reconstruct_swarm_mesh`` and applied to the ENTRY member's Portal,
        instead of being silently dropped on re-import.
  B2 -- ``_lower_oracle``'s per-variant loop now calls
        ``_reject_unrepresentable_fields`` (the guard the single-node path
        already ran), so an Oracle-modified node with a callable-valued field
        fails loud instead of exporting silently.
  B4 -- a dict-form-OUTPUT producer consumed by a dict-form (or single-type)
        input now wires DataFlowEdges via ``_properties_for``'s ``{key}.{field}``
        prefix, instead of raising / silently dropping the edge.

Gated on ``pyagentspec`` via ``pytest.importorskip`` (the ``[agent-spec]``
optional extra keeps ``src/neograph`` core dependency-light)::

    uv run --extra agent-spec pytest tests/test_agent_spec_phase5_reuse.py
"""

from __future__ import annotations

import warnings

import pytest

pytest.importorskip("pyagentspec")

from pydantic import BaseModel  # noqa: E402

from neograph import HANDOFF_END, Construct, Node, Portal  # noqa: E402
from neograph.errors import ConfigurationError  # noqa: E402
from neograph.modifiers import Oracle  # noqa: E402
from tests.fakes import register_scripted  # noqa: E402


def _wired_edges(flow: object) -> list[tuple[str, str, str, str]]:
    """Every DataFlowEdge as ``(source_node, source_output, dest_node, dest_input)``.

    ``data_flow_connections`` is None-able -- the exporter writes ``<edges> or None``,
    so a control-only shape genuinely exports None and a raw dereference raises
    TypeError instead of failing an assertion (neograph-p0dn8). The ``or []`` here is
    the single guarded read for this file, pinned by
    tests/test_guards_agent_spec_data_flow_reads.py.
    """
    return [
        (e.source_node.name, e.source_output, e.destination_node.name, e.destination_input)
        for e in (flow.data_flow_connections or [])
    ]



class _Payload(BaseModel, frozen=True):
    goto: str = ""


class _Claims(BaseModel, frozen=True):
    text: str = ""


class _Log(BaseModel, frozen=True):
    n: int = 0


# ── A1: mesh-member detection is modifier-agnostic (Node AND Construct) ───────


class TestA1MeshMemberDetection:
    """``to_agent_spec``'s mesh filter must detect a PEER Portal member on a
    Construct as well as a Node -- the old ``isinstance(Node)`` gate dropped a
    Construct member and false-rejected the whole mesh as 'mixed'."""

    def test_is_peer_mesh_member_true_for_construct_member(self):
        from neograph._agent_spec import _is_peer_mesh_member

        sub = Construct("sub", input=_Payload, output=_Payload, nodes=[Node.scripted("r", fn="f", outputs=_Payload)])
        assert _is_peer_mesh_member(sub | Portal(to=["x"])) is True

    def test_is_peer_mesh_member_true_for_node_member(self):
        from neograph._agent_spec import _is_peer_mesh_member

        assert _is_peer_mesh_member(Node.scripted("n", fn="f", outputs=_Payload) | Portal(to=["x"])) is True

    def test_is_peer_mesh_member_false_for_bare_and_dispatch(self):
        from neograph._agent_spec import _is_peer_mesh_member

        bare = Node.scripted("b", fn="f", outputs=_Payload)
        dispatch = Node.scripted("d", fn="f", outputs=_Payload) | Portal(
            route="decide", spec_field="s", input_field="i", output=_Payload, max_depth=3
        )
        assert _is_peer_mesh_member(bare) is False
        assert _is_peer_mesh_member(dispatch) is False

    def test_mesh_with_construct_member_not_false_rejected_as_mixed(self):
        """The integration symptom: a mesh whose non-entry member is a Construct
        must NOT raise the 'mixes a Portal peer mesh with non-mesh nodes' error.
        (Actually lowering the Construct member is C1, out of Phase-5 scope, so
        a different downstream failure is acceptable -- only the MISCLASSIFICATION
        must be gone.)"""
        from neograph._agent_spec import to_agent_spec

        register_scripted("_p5_entry", lambda i, c: _Payload(goto="resolver_sub"))
        register_scripted("_p5_resolve", lambda i, c: _Payload(goto="closer"))
        register_scripted("_p5_close", lambda i, c: _Payload(goto=HANDOFF_END))

        resolver_sub = Construct(
            "resolver_sub",
            input=_Payload,
            output=_Payload,
            nodes=[Node.scripted("resolve", fn="_p5_resolve", outputs=_Payload)],
        )
        pipeline = Construct(
            "parent_mesh",
            nodes=[
                Node.scripted("entry", fn="_p5_entry", inputs={"handoff": _Payload}, outputs=_Payload)
                | Portal(to=["resolver_sub"], max_hops=6),
                resolver_sub | Portal(to=["closer"]),
                Node.scripted("closer", fn="_p5_close", inputs={"handoff": _Payload}, outputs=_Payload) | Portal(to=[]),
            ],
        )

        try:
            to_agent_spec(pipeline)
        except ConfigurationError as exc:  # pragma: no cover - defensive
            assert "mixes a Portal peer mesh" not in str(exc), (
                "A1 regression: a Construct mesh member is still misclassified as a non-mesh node"
            )
        except Exception:
            # A downstream (non-ConfigurationError) failure means detection
            # succeeded and lowering proceeded -- the A1 misclassification is gone.
            pass


# ── B2: Oracle-variant lowering runs the unrepresentable-field guard ──────────


class TestB2OracleGuardsUnrepresentableFields:
    """``_lower_oracle`` must run ``_reject_unrepresentable_fields`` like the
    single-node path does -- an Oracle-modified node carrying a callable-valued
    field must fail loud, not export silently."""

    def test_oracle_node_with_skip_when_is_rejected(self):
        from neograph._agent_spec import to_agent_spec

        node = Node.scripted("seed", fn="f", outputs=_Claims).model_copy(update={"skip_when": lambda d: False})
        node = node | Oracle(n=2, merge_fn="m")
        pipeline = Construct("oracle-skip-pipeline", nodes=[node])

        with pytest.raises(ConfigurationError, match="skip_when"):
            to_agent_spec(pipeline)

    def test_oracle_node_with_renderer_is_rejected(self):
        from neograph._agent_spec import to_agent_spec

        node = Node.scripted("seed", fn="f", outputs=_Claims).model_copy(update={"renderer": lambda x: "x"})
        node = node | Oracle(n=2, merge_fn="m")
        pipeline = Construct("oracle-renderer-pipeline", nodes=[node])

        with pytest.raises(ConfigurationError, match="renderer"):
            to_agent_spec(pipeline)


# ── B4: dict-form-output producers wire edges (no reject / no silent drop) ─────


class TestB4DictFormOutputEdges:
    """A dict-form ``Node.outputs`` producer must wire DataFlowEdges to its
    downstream consumers via ``_properties_for``'s ``compose_property_title``
    qualification -- the input side already does this; the producer side used to
    raise (dict-form input consumer) or silently drop the edge (single-type
    input consumer)."""

    def test_dict_form_input_consumer_of_dict_form_producer_wires_edge(self):
        from neograph._agent_spec import to_agent_spec

        prod = Node.scripted("prod", fn="f", outputs={"result": _Claims, "log": _Log})
        cons = Node.scripted("cons", fn="g", inputs={"prod_result": _Claims}, outputs=_Claims)
        flow = to_agent_spec(Construct("b4-dictin", nodes=[prod, cons]))

        wired = _wired_edges(flow)
        assert ("prod", "result:text", "cons", "prod_result:text") in wired, wired

    def test_single_type_consumer_of_dict_form_producer_wires_edge(self):
        from neograph._agent_spec import to_agent_spec

        prod = Node.scripted("prod", fn="f", outputs={"result": _Claims, "log": _Log})
        cons = Node.scripted("cons", fn="g", inputs=_Claims, outputs=_Log)
        flow = to_agent_spec(Construct("b4-singlein", nodes=[prod, cons]))

        wired = _wired_edges(flow)
        assert ("prod", "result:text", "cons", "text") in wired, wired


# ── B3 + A5: Swarm round-trip + loader re-validation ──────────────────────────


def _register_mesh_fns() -> None:
    register_scripted("_p5_mesh_a", lambda i, c: _Payload(goto="b"))
    register_scripted("_p5_mesh_b", lambda i, c: _Payload(goto=HANDOFF_END))


def _build_mesh(*, max_hops: int = 10, on_exhaust: str = "error") -> Construct:
    _register_mesh_fns()
    return Construct(
        "mesh1",
        nodes=[
            Node.scripted("a", fn="_p5_mesh_a", inputs={"handoff": _Payload}, outputs=_Payload)
            | Portal(to=["b"], max_hops=max_hops, on_exhaust=on_exhaust),
            Node.scripted("b", fn="_p5_mesh_b", inputs={"handoff": _Payload}, outputs=_Payload) | Portal(to=[]),
        ],
    )


class TestB3PortalSpecRoundTrip:
    """The entry-only Portal knobs (max_hops/on_exhaust/route) ride the
    ``neograph/portal_spec`` marker on export and must be read back on import
    onto the ENTRY member -- they were previously dropped even for the working
    all-Node mesh case."""

    def test_max_hops_and_on_exhaust_round_trip_onto_entry(self):
        from neograph._agent_spec import to_agent_spec
        from neograph.loader import from_agent_spec

        swarm = to_agent_spec(_build_mesh(max_hops=6, on_exhaust="exit"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = from_agent_spec(swarm)

        entry_portal = back.nodes[0].modifier_set.portal
        assert entry_portal is not None
        assert entry_portal.max_hops == 6
        assert entry_portal.on_exhaust == "exit"

    def test_non_entry_member_does_not_carry_entry_only_knobs(self):
        from neograph._agent_spec import to_agent_spec
        from neograph.loader import from_agent_spec

        swarm = to_agent_spec(_build_mesh(max_hops=6, on_exhaust="exit"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = from_agent_spec(swarm)

        # Entry-only knobs must not leak onto non-entry members (else
        # _check_portal_mesh's entry-only rule would reject the reconstructed mesh).
        non_entry_portal = back.nodes[1].modifier_set.portal
        assert non_entry_portal is not None
        assert "max_hops" not in non_entry_portal.model_fields_set
        assert "on_exhaust" not in non_entry_portal.model_fields_set


class TestA5LoaderRevalidatesReconstructedMesh:
    """The loader must route the reconstructed Swarm mesh through the same
    ``_check_portal_mesh`` assembly gate every other construction path uses --
    guaranteed today because ``_reconstruct_swarm_mesh`` returns a
    ``Construct(...)`` (whose ``__init__`` runs the gate). Pinned here so a
    future refactor that bypasses assembly (e.g. ``model_construct``) fails loud."""

    def test_from_agent_spec_swarm_routes_through_check_portal_mesh(self, monkeypatch):
        import neograph._construct_validation as cv
        from neograph._agent_spec import to_agent_spec
        from neograph.loader import from_agent_spec

        swarm = to_agent_spec(_build_mesh())

        seen: list[str] = []
        orig = cv._check_portal_mesh

        def _spy(construct):
            seen.append(getattr(construct, "name", "?"))
            return orig(construct)

        monkeypatch.setattr(cv, "_check_portal_mesh", _spy)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from_agent_spec(swarm)

        assert "mesh1" in seen, (
            "the reconstructed Swarm mesh was not routed through _check_portal_mesh "
            "-- the loader is trusting the artifact blindly (A5 regression)"
        )
