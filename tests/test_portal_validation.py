"""Portal assembly-validation tests — every design §5 ConstructError.

Pins T1 (neograph-rwion) IR-layer behavior: a Portal mesh that violates a
design §5 assembly rule raises ``ConstructError`` at ``Construct(...)`` assembly
time (or ``Portal(...)`` construction for mode discrimination), naming the
offender; a legal mesh assembles + compiles cleanly.

These are integration-level tests through the REAL ``Construct(...)`` / ``Node``
/ ``Portal`` surface — assembly validation is pure in-process, no mocks.

Three-surface parity: the mesh is exercised through BOTH the declarative
``Node(...) | Portal(...)`` surface and the programmatic pipe (they are
identical once the slot exists). ``@node`` sugar is T4 (not tested here);
ForwardConstruct is exempt in v1 (D-FORWARD-EXEMPT) — no static dataflow to
trace in a runtime mesh.

Design refs: docs/design/dynamic-handoff-2026-07-13.md §2.1, §3.1-3.3, §5.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from neograph import (
    Construct,
    ConstructError,
    Node,
    Portal,
    compile,
    run,
)
from neograph._construct_validation import effective_producer_type
from tests.fakes import build_test_compile_kwargs, register_scripted

# ═══════════════════════════════════════════════════════════════════════════
# PAYLOAD MODELS — one uniform payload type per mesh (design §3.1 rule 4)
# ═══════════════════════════════════════════════════════════════════════════


class Handoff(BaseModel, frozen=True):
    """Payload with a plain-str route field named `goto`."""

    goto: str
    note: str = ""


class LiteralHandoff(BaseModel, frozen=True):
    """Payload with a Literal-typed route field (the typed-swarm story)."""

    goto: Literal["billing", "technical", "__end__"]


class OtherPayload(BaseModel, frozen=True):
    """A DIFFERENT payload type — used for the non-uniform-payload error."""

    goto: str


class NoRoutePayload(BaseModel, frozen=True):
    """Missing the `goto` route field entirely."""

    text: str


class BadRoutePayload(BaseModel, frozen=True):
    """`goto` annotated as int — not str/Literal."""

    goto: int


register_scripted("f", lambda i, c: Handoff(goto="__end__"))


# ═══════════════════════════════════════════════════════════════════════════
# MESH BUILDERS — declarative + programmatic (identical IR)
# ═══════════════════════════════════════════════════════════════════════════


def _member(name: str, peers: list[str], *, payload=Handoff, handoff=True, **kw) -> Node:
    """A single Portal-modified mesh member.

    Non-entry members declare the reserved `handoff` inputs key (mesh channel);
    the entry omits it (its first activation comes from the outer pipeline).
    """
    inputs = {"handoff": payload} if handoff else None
    return Node.scripted(name, fn="f", inputs=inputs, outputs=payload) | Portal(to=peers, **kw)


def _legal_mesh() -> Construct:
    """A legal three-member mesh, plain-str route, max_hops on entry only."""
    entry = _member("triage", ["billing", "technical"], handoff=False, max_hops=6)
    billing = _member("billing", ["triage", "technical"])
    technical = _member("technical", [])  # terminal specialist: exits only
    return Construct("swarm", nodes=[entry, billing, technical])


# ═══════════════════════════════════════════════════════════════════════════
# HAPPY PATH — a legal mesh assembles + compiles cleanly
# ═══════════════════════════════════════════════════════════════════════════


class TestLegalMeshAssemblesAndCompiles:
    """A legal mesh ASSEMBLES and COMPILES cleanly (T2 — neograph-on6jt).

    A legal mesh passes every §5 rule so ``Construct(...)`` succeeds, and — now
    that T2 lowering has landed — ``compile()`` builds the Command(goto) mesh
    without error (T1 fail-loud-staged this as ``CompileError("lands in T2")``;
    D7 assigns the flip to T2). Three-surface parity is exercised at the
    assembly+compile level (declarative + programmatic); the full runtime routing
    with a genuine cycle lives in ``tests/modifiers/test_portal.py``. See
    ``test_legal_mesh_routes_to_end_and_completes``.
    """

    def test_plain_str_route_mesh_compiles(self):
        """A plain-str `goto` mesh assembles and compiles cleanly."""
        register_scripted("f", lambda i, c: Handoff(goto="__end__"))
        mesh = _legal_mesh()
        assert mesh.name == "swarm"  # Construct(...) assembled — no ConstructError
        compiled = compile(mesh, **build_test_compile_kwargs())
        assert compiled is not None  # compile() lowered the mesh — no CompileError

    def test_literal_route_mesh_compiles(self):
        """A Literal-typed route with all members in peers∪{HANDOFF_END} compiles."""
        register_scripted("f", lambda i, c: Handoff(goto="__end__"))
        entry = (
            Node.scripted("triage", fn="f", outputs=LiteralHandoff)
            | Portal(to=["billing", "technical"], max_hops=4)
        )
        billing = _member("billing", ["triage", "technical"], payload=LiteralHandoff)
        technical = _member("technical", [], payload=LiteralHandoff)
        mesh = Construct("typed-swarm", nodes=[entry, billing, technical])
        assert mesh.name == "typed-swarm"  # assembled cleanly
        compiled = compile(mesh, **build_test_compile_kwargs())
        assert compiled is not None

    def test_programmatic_pipe_surface_compiles(self):
        """The programmatic pipe surface produces the same legal (compiling) mesh."""
        register_scripted("f", lambda i, c: Handoff(goto="__end__"))
        entry = Node.scripted("triage", fn="f", outputs=Handoff) | Portal(to=["billing"], max_hops=6)
        billing = (
            Node.scripted("billing", fn="f", inputs={"handoff": Handoff}, outputs=Handoff)
            | Portal(to=["triage"])
        )
        mesh = Construct("swarm2", nodes=[entry, billing])
        assert mesh.name == "swarm2"  # assembled cleanly
        compiled = compile(mesh, **build_test_compile_kwargs())
        assert compiled is not None

    def test_legal_mesh_routes_to_end_and_completes(self):
        """Runtime routing pin (replaces the T1 staging pin, D7): the legal mesh
        compiles AND runs — the entry routes to HANDOFF_END and the run completes
        through the pass-through exit node without a silent drop."""
        register_scripted("f", lambda i, c: Handoff(goto="__end__"))
        mesh = _legal_mesh()  # entry fn returns goto="__end__" -> exits immediately
        graph = compile(mesh, **build_test_compile_kwargs())
        result = run(graph, input={})
        # The entry routed to HANDOFF_END; its payload is on the bus with the
        # route value the wrapper checked (not silently dropped).
        assert isinstance(result["triage"], Handoff)
        assert result["triage"].goto == "__end__"


# ═══════════════════════════════════════════════════════════════════════════
# effective_producer_type — UNTOUCHED for a mesh member (INVARIANT pin)
# ═══════════════════════════════════════════════════════════════════════════


class TestEffectiveProducerTypeUntouched:
    """A Portal member produces its DECLARED output — the INVARIANT's pin."""

    def test_mesh_member_produces_declared_output(self):
        """effective_producer_type(member) == the member's declared payload type."""
        member = _member("billing", ["triage"])
        assert effective_producer_type(member) is Handoff


# ═══════════════════════════════════════════════════════════════════════════
# §5 ASSEMBLY ERRORS — each names the offender
# ═══════════════════════════════════════════════════════════════════════════


class TestMeshStructureErrors:
    """Design §5.1-5.2 mesh structure rules."""

    def test_unknown_peer_lists_siblings(self):
        """A peer name that isn't a sibling raises, naming it and the siblings."""
        entry = _member("triage", ["nonexistent"], handoff=False)
        billing = _member("billing", ["triage"])
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, billing])
        msg = str(exc.value)
        assert "nonexistent" in msg
        assert "billing" in msg  # lists available siblings

    def test_peer_not_portal_modified_rejected(self):
        """A peer that is a plain (non-Portal) sibling raises, naming it."""
        entry = _member("triage", ["billing"], handoff=False)
        billing = Node.scripted("billing", fn="f", outputs=Handoff)  # NOT Portal-modified
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, billing])
        assert "billing" in str(exc.value)

    def test_non_contiguous_mesh_rejected(self):
        """Mesh members split by a non-member node raise (contiguity rule)."""
        entry = _member("triage", ["billing"], handoff=False)
        gap = Node.scripted("gap", fn="f", outputs=Handoff)
        billing = _member("billing", ["triage"])
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, gap, billing])
        assert "contiguous" in str(exc.value).lower() or "triage" in str(exc.value)

    def test_two_meshes_at_one_level_rejected(self):
        """Two disjoint meshes at the same construct level raise (one-mesh rule)."""
        a1 = _member("a1", ["a2"], handoff=False)
        a2 = _member("a2", ["a1"])
        b1 = _member("b1", ["b2"], handoff=False)
        b2 = _member("b2", ["b1"])
        with pytest.raises(ConstructError) as exc:
            Construct("two-mesh", nodes=[a1, a2, b1, b2])
        assert "mesh" in str(exc.value).lower()

    def test_max_hops_on_non_entry_member_rejected(self):
        """max_hops explicitly set on a non-entry member raises, naming it."""
        entry = _member("triage", ["billing"], handoff=False)
        billing = _member("billing", ["triage"], max_hops=5)  # non-entry — illegal
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, billing])
        msg = str(exc.value)
        assert "billing" in msg and "max_hops" in msg

    def test_mesh_member_is_construct_rejected(self):
        """A sub-Construct as a mesh member raises (D-MESH-LEVEL: siblings only)."""
        register_scripted("sub_f", lambda i, c: Handoff(goto="__end__"))
        sub = Construct(
            "sub",
            input=Handoff,
            output=Handoff,
            nodes=[Node.scripted("inner", fn="sub_f", outputs=Handoff)],
        )
        keyed_sub = sub | Portal(to=["triage"])
        entry = _member("triage", ["sub"], handoff=False)
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, keyed_sub])
        assert "sub" in str(exc.value).lower() or "construct" in str(exc.value).lower()

    def test_agent_member_accepted_as_mesh_member(self):
        """An agent-mode mesh member is ACCEPTED at assembly (neograph-nnds9).

        D-MEMBER-MODES (docs/design/dynamic-handoff-2026-07-13.md) was the v1
        boundary cut rejecting agent/act members; portal-addressability-2026-07-15.md
        names nnds9 as the task that relaxes it (`_check_portal_mesh`,
        src/neograph/_validation_portal.py:96-104). This is the TDD-red pin for
        that relaxation: assembling a mesh with an agent-mode member must NOT
        raise ConstructError -- it currently does, because the blanket
        `member.mode in ('agent', 'act')` rejection has not yet been narrowed.
        Was `test_agent_member_rejected` (retired here per neograph-vihe7.29
        step 7 -- the pin inverts rather than duplicates, since the old and new
        behavior are mutually exclusive for the same input).
        """
        entry = _member("triage", ["researcher"], handoff=False)
        researcher = Node(
            "researcher",
            mode="agent",
            prompt="x",
            model="reason",
            tools=[],
            inputs={"handoff": Handoff},
            outputs=Handoff,
        ) | Portal(to=["triage"])
        mesh = Construct("swarm", nodes=[entry, researcher])  # must NOT raise
        assert mesh.name == "swarm"

    def test_dict_form_outputs_on_member_rejected(self):
        """dict-form outputs on a Portal member raise (D-DICT-OUTPUTS)."""
        entry = (
            Node("triage", mode="scripted", scripted_fn="f", outputs={"result": Handoff, "log": str})
            | Portal(to=["billing"])
        )
        billing = _member("billing", ["triage"])
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, billing])
        assert "triage" in str(exc.value) or "dict" in str(exc.value).lower()

    def test_producer_named_handoff_at_mesh_level_rejected(self):
        """A sibling producer literally named `handoff` collides with the reserved key."""
        entry = _member("triage", ["billing"], handoff=False)
        billing = _member("billing", ["triage"])
        collide = Node.scripted("handoff", fn="f", outputs=Handoff)
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, billing, collide])
        assert "handoff" in str(exc.value).lower()


class TestPayloadUniformity:
    """Design §5.3 — all members share one payload output type."""

    def test_nonuniform_payload_rejected(self):
        """A member whose output type differs from the entry's raises, naming it."""
        entry = _member("triage", ["billing"], handoff=False)
        billing = Node.scripted("billing", fn="f", inputs={"handoff": Handoff}, outputs=OtherPayload) | Portal(
            to=["triage"]
        )
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, billing])
        assert "billing" in str(exc.value) or "payload" in str(exc.value).lower()


class TestRouteFieldErrors:
    """Design §5.4 — route field presence, annotation, Literal membership."""

    def test_route_field_missing_rejected(self):
        """The route field (`goto`) absent from the payload model raises."""
        entry = Node.scripted("triage", fn="f", outputs=NoRoutePayload) | Portal(to=["billing"])
        billing = Node.scripted("billing", fn="f", inputs={"handoff": NoRoutePayload}, outputs=NoRoutePayload) | Portal(
            to=["triage"]
        )
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, billing])
        assert "goto" in str(exc.value)

    def test_route_annotation_not_str_or_literal_rejected(self):
        """A route field annotated int (not str/Literal) raises."""
        entry = Node.scripted("triage", fn="f", outputs=BadRoutePayload) | Portal(to=["billing"])
        billing = Node.scripted("billing", fn="f", inputs={"handoff": BadRoutePayload}, outputs=BadRoutePayload) | Portal(
            to=["triage"]
        )
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, billing])
        assert "goto" in str(exc.value) or "str" in str(exc.value).lower()

    def test_literal_route_with_stray_target_rejected(self):
        """A Literal member not in peers∪{HANDOFF_END} raises, naming the stray."""

        class StrayLiteral(BaseModel, frozen=True):
            goto: Literal["billing", "stray", "__end__"]

        entry = Node.scripted("triage", fn="f", outputs=StrayLiteral) | Portal(to=["billing"])
        billing = Node.scripted("billing", fn="f", inputs={"handoff": StrayLiteral}, outputs=StrayLiteral) | Portal(
            to=["triage"]
        )
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, billing])
        assert "stray" in str(exc.value)


class TestReservedHandoffKey:
    """Design §5.5 — the reserved `handoff` input key."""

    def test_handoff_input_on_non_mesh_node_rejected(self):
        """A plain (non-Portal) node with a `handoff` input key raises."""
        producer = Node.scripted("src", fn="f", outputs=Handoff)
        consumer = Node.scripted("plain", fn="f", inputs={"handoff": Handoff}, outputs=Handoff)
        with pytest.raises(ConstructError) as exc:
            Construct("no-mesh", nodes=[producer, consumer])
        assert "handoff" in str(exc.value).lower()

    def test_handoff_input_typed_not_payload_model_rejected(self):
        """A mesh member whose `handoff` input type != payload model raises."""
        entry = _member("triage", ["billing"], handoff=False)
        billing = Node.scripted("billing", fn="f", inputs={"handoff": OtherPayload}, outputs=Handoff) | Portal(
            to=["triage"]
        )
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, billing])
        assert "handoff" in str(exc.value).lower() or "billing" in str(exc.value)
