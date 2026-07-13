"""Keymaker assembly-validation tests — every design §5 ConstructError.

Pins T1 (neograph-rwion) IR-layer behavior: a Keymaker mesh that violates a
design §5 assembly rule raises ``ConstructError`` at ``Construct(...)`` assembly
time (or ``Keymaker(...)`` construction for mode discrimination), naming the
offender; a legal mesh assembles + compiles cleanly.

These are integration-level tests through the REAL ``Construct(...)`` / ``Node``
/ ``Keymaker`` surface — assembly validation is pure in-process, no mocks.

Three-surface parity: the mesh is exercised through BOTH the declarative
``Node(...) | Keymaker(...)`` surface and the programmatic pipe (they are
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
    CompileError,
    Construct,
    ConstructError,
    Keymaker,
    Node,
    compile,
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
    """A single Keymaker-modified mesh member.

    Non-entry members declare the reserved `handoff` inputs key (mesh channel);
    the entry omits it (its first activation comes from the outer pipeline).
    """
    inputs = {"handoff": payload} if handoff else None
    return Node.scripted(name, fn="f", inputs=inputs, outputs=payload) | Keymaker(peers=peers, **kw)


def _legal_mesh() -> Construct:
    """A legal three-member mesh, plain-str route, max_hops on entry only."""
    entry = _member("triage", ["billing", "technical"], handoff=False, max_hops=6)
    billing = _member("billing", ["triage", "technical"])
    technical = _member("technical", [])  # terminal specialist: exits only
    return Construct("swarm", nodes=[entry, billing, technical])


# ═══════════════════════════════════════════════════════════════════════════
# HAPPY PATH — a legal mesh assembles + compiles cleanly
# ═══════════════════════════════════════════════════════════════════════════


class TestLegalMeshAssembles:
    """A legal mesh ASSEMBLES cleanly at T1; lowering is staged to T2.

    T1 delivers IR + assembly validation only — a legal mesh passes every §5
    rule so ``Construct(...)`` succeeds, but ``compile()`` fail-loud-stages the
    mesh lowering as ``CompileError("Keymaker lowering lands in T2")`` (decision
    D6/D7). These tests therefore assert BOTH: the Construct assembles (no
    exception) AND compile raises the staged error. Three-surface parity is
    exercised at the assembly level (declarative + programmatic); the runtime
    routing tests land in T2. See ``test_keymaker_compile_is_staged_to_t2``.
    """

    def test_plain_str_route_mesh_assembles(self):
        """A plain-str `goto` mesh assembles cleanly (compile staged to T2)."""
        mesh = _legal_mesh()
        assert mesh.name == "swarm"  # Construct(...) assembled — no ConstructError
        with pytest.raises(CompileError, match="lands in T2"):
            compile(mesh, **build_test_compile_kwargs())

    def test_literal_route_mesh_assembles(self):
        """A Literal-typed route with all members in peers∪{HANDOFF_END} passes."""
        entry = (
            Node.scripted("triage", fn="f", outputs=LiteralHandoff)
            | Keymaker(peers=["billing", "technical"], max_hops=4)
        )
        billing = _member("billing", ["triage", "technical"], payload=LiteralHandoff)
        technical = _member("technical", [], payload=LiteralHandoff)
        mesh = Construct("typed-swarm", nodes=[entry, billing, technical])
        assert mesh.name == "typed-swarm"  # assembled cleanly
        with pytest.raises(CompileError, match="lands in T2"):
            compile(mesh, **build_test_compile_kwargs())

    def test_programmatic_pipe_surface_assembles(self):
        """The programmatic pipe surface produces the same legal (assembling) mesh."""
        entry = Node.scripted("triage", fn="f", outputs=Handoff) | Keymaker(peers=["billing"], max_hops=6)
        billing = (
            Node.scripted("billing", fn="f", inputs={"handoff": Handoff}, outputs=Handoff)
            | Keymaker(peers=["triage"])
        )
        mesh = Construct("swarm2", nodes=[entry, billing])
        assert mesh.name == "swarm2"  # assembled cleanly
        with pytest.raises(CompileError, match="lands in T2"):
            compile(mesh, **build_test_compile_kwargs())

    def test_keymaker_compile_is_staged_to_t2(self):
        """T1 fail-loud staging pin: compiling any Keymaker mesh raises CompileError.

        T2 (neograph-on6jt) replaces the compiler placeholder arms with the real
        mesh lowering and REPLACES this test with a runtime routing assertion.
        """
        mesh = _legal_mesh()
        with pytest.raises(CompileError, match="Keymaker lowering lands in T2"):
            compile(mesh, **build_test_compile_kwargs())


# ═══════════════════════════════════════════════════════════════════════════
# effective_producer_type — UNTOUCHED for a mesh member (INVARIANT pin)
# ═══════════════════════════════════════════════════════════════════════════


class TestEffectiveProducerTypeUntouched:
    """A Keymaker member produces its DECLARED output — the INVARIANT's pin."""

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

    def test_peer_not_keymaker_modified_rejected(self):
        """A peer that is a plain (non-Keymaker) sibling raises, naming it."""
        entry = _member("triage", ["billing"], handoff=False)
        billing = Node.scripted("billing", fn="f", outputs=Handoff)  # NOT Keymaker-modified
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
        keyed_sub = sub | Keymaker(peers=["triage"])
        entry = _member("triage", ["sub"], handoff=False)
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, keyed_sub])
        assert "sub" in str(exc.value).lower() or "construct" in str(exc.value).lower()

    def test_agent_member_rejected(self):
        """An agent-mode mesh member raises (D-MEMBER-MODES: scripted/think/raw only)."""
        entry = _member("triage", ["researcher"], handoff=False)
        researcher = Node(
            "researcher",
            mode="agent",
            prompt="x",
            model="reason",
            tools=[],
            inputs={"handoff": Handoff},
            outputs=Handoff,
        ) | Keymaker(peers=["triage"])
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, researcher])
        assert "agent" in str(exc.value).lower() or "researcher" in str(exc.value)

    def test_dict_form_outputs_on_member_rejected(self):
        """dict-form outputs on a Keymaker member raise (D-DICT-OUTPUTS)."""
        entry = (
            Node("triage", mode="scripted", scripted_fn="f", outputs={"result": Handoff, "log": str})
            | Keymaker(peers=["billing"])
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
        billing = Node.scripted("billing", fn="f", inputs={"handoff": Handoff}, outputs=OtherPayload) | Keymaker(
            peers=["triage"]
        )
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, billing])
        assert "billing" in str(exc.value) or "payload" in str(exc.value).lower()


class TestRouteFieldErrors:
    """Design §5.4 — route field presence, annotation, Literal membership."""

    def test_route_field_missing_rejected(self):
        """The route field (`goto`) absent from the payload model raises."""
        entry = Node.scripted("triage", fn="f", outputs=NoRoutePayload) | Keymaker(peers=["billing"])
        billing = Node.scripted("billing", fn="f", inputs={"handoff": NoRoutePayload}, outputs=NoRoutePayload) | Keymaker(
            peers=["triage"]
        )
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, billing])
        assert "goto" in str(exc.value)

    def test_route_annotation_not_str_or_literal_rejected(self):
        """A route field annotated int (not str/Literal) raises."""
        entry = Node.scripted("triage", fn="f", outputs=BadRoutePayload) | Keymaker(peers=["billing"])
        billing = Node.scripted("billing", fn="f", inputs={"handoff": BadRoutePayload}, outputs=BadRoutePayload) | Keymaker(
            peers=["triage"]
        )
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, billing])
        assert "goto" in str(exc.value) or "str" in str(exc.value).lower()

    def test_literal_route_with_stray_target_rejected(self):
        """A Literal member not in peers∪{HANDOFF_END} raises, naming the stray."""

        class StrayLiteral(BaseModel, frozen=True):
            goto: Literal["billing", "stray", "__end__"]

        entry = Node.scripted("triage", fn="f", outputs=StrayLiteral) | Keymaker(peers=["billing"])
        billing = Node.scripted("billing", fn="f", inputs={"handoff": StrayLiteral}, outputs=StrayLiteral) | Keymaker(
            peers=["triage"]
        )
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, billing])
        assert "stray" in str(exc.value)


class TestReservedHandoffKey:
    """Design §5.5 — the reserved `handoff` input key."""

    def test_handoff_input_on_non_mesh_node_rejected(self):
        """A plain (non-Keymaker) node with a `handoff` input key raises."""
        producer = Node.scripted("src", fn="f", outputs=Handoff)
        consumer = Node.scripted("plain", fn="f", inputs={"handoff": Handoff}, outputs=Handoff)
        with pytest.raises(ConstructError) as exc:
            Construct("no-mesh", nodes=[producer, consumer])
        assert "handoff" in str(exc.value).lower()

    def test_handoff_input_typed_not_payload_model_rejected(self):
        """A mesh member whose `handoff` input type != payload model raises."""
        entry = _member("triage", ["billing"], handoff=False)
        billing = Node.scripted("billing", fn="f", inputs={"handoff": OtherPayload}, outputs=Handoff) | Keymaker(
            peers=["triage"]
        )
        with pytest.raises(ConstructError) as exc:
            Construct("swarm", nodes=[entry, billing])
        assert "handoff" in str(exc.value).lower() or "billing" in str(exc.value)
