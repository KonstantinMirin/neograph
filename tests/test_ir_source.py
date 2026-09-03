"""The address/source vocabulary: closed-set, sealing, and frozen-value contract.

Step 0 of ``neograph-9axw6``. These tests pin the properties ``_ir_source``'s module
docstring CLAIMS, including the one it concedes is defeatable -- a docstring that
overstates a guarantee is worse than no docstring, so the defeat is pinned too.

This is the ONE module allowed to construct ``Source``/``PortRef`` values outside
``_ir_normalize.py`` (``TestSourceConstructionMonopoly.TESTS_CONSTRUCTION_ALLOWED``).
"""

from __future__ import annotations

import dataclasses

import pytest

from neograph._ir_source import (
    Accumulated,
    Candidate,
    EachItem,
    HandoffChannel,
    LastPresent,
    LoopCarry,
    Peer,
    Port,
    PortRef,
    Resolved,
    Unresolved,
    _SealedSource,
    source_channel_kind,
)
from neograph.errors import ConstructError

ALL_VARIANTS = (
    Peer(ref=PortRef("settle", "result")),
    Port(),
    EachItem(),
    LoopCarry(),
    HandoffChannel(channel="neo_handoff_entry"),
    LastPresent(refs=(PortRef("carry"), PortRef("seed"))),
    Accumulated(channel="neo_accum_findings"),
)


class TestAPortNamesAnOutputNotAMember:
    """A member with dict-form outputs writes one field per key, so a member NAME
    does not identify a value -- the ``neograph-kgndo`` defect."""

    def test_a_port_ref_carries_the_output_key(self) -> None:
        ref = PortRef("settle", "tool_log")
        assert (ref.member, ref.output) == ("settle", "tool_log")

    def test_output_defaults_to_none_meaning_the_sole_output(self) -> None:
        assert PortRef("verify").output is None

    def test_two_ports_on_one_member_are_distinct_addresses(self) -> None:
        assert PortRef("settle", "result") != PortRef("settle", "tool_log")


class TestAddressesAreFrozenValues:
    """Frozen dataclasses, never Pydantic: a Pydantic model would drag
    ``arbitrary_types_allowed`` for the type payloads into every consumer."""

    def test_a_port_ref_cannot_be_mutated(self) -> None:
        ref = PortRef("a")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.member = "b"  # type: ignore[misc]

    @pytest.mark.parametrize("variant", ALL_VARIANTS, ids=lambda v: type(v).__name__)
    def test_every_variant_is_frozen(self, variant: object) -> None:
        assert dataclasses.is_dataclass(variant)
        assert type(variant).__dataclass_params__.frozen  # type: ignore[attr-defined]

    def test_equal_addresses_compare_and_hash_equal(self) -> None:
        assert Peer(PortRef("a", "x")) == Peer(PortRef("a", "x"))
        assert len({Peer(PortRef("a")), Peer(PortRef("a"))}) == 1

    def test_no_variant_is_a_pydantic_model(self) -> None:
        for variant in ALL_VARIANTS:
            assert not hasattr(type(variant), "model_fields"), (
                f"{type(variant).__name__} is a Pydantic model; the module docstring "
                "says these are frozen dataclasses precisely to keep "
                "arbitrary_types_allowed out of every consumer"
            )


class TestTheSourceSetIsClosed:
    """'Closed is the point: a new arrival mechanism must extend this type, which
    makes it visible, rather than adding a field somewhere.'"""

    def test_an_honest_subclass_outside_the_module_is_refused(self) -> None:
        """ConstructError, not a bare TypeError: every neograph error goes through
        NeographError.build (pinned by test_guards_any_audit). ConstructError is
        the assembly-time class, and extending the IR vocabulary is assembly-time."""
        with pytest.raises(ConstructError) as exc:

            class Sneaky(_SealedSource):
                pass

        msg = str(exc.value)
        assert "CLOSED set" in msg
        assert "_ir_source" in msg
        assert "Sneaky" in msg, "the message must name the offending class"

    def test_subclassing_a_variant_is_refused_too(self) -> None:
        """Not just the base: a variant is the likelier reach."""
        with pytest.raises(ConstructError):

            class SneakyPeer(Peer):
                pass

    def test_the_module_spoof_DEFEATS_the_runtime_tripwire(self) -> None:
        """The concession, pinned.

        A class body's ``__module__`` lands in the namespace ``__init_subclass__``
        reads, so one line defeats the runtime check. This test exists so the
        docstring's honesty is ENFORCED: if someone later hardens
        ``__init_subclass__`` and this starts failing, the tiering claim in the
        module docstring (and in ``TestSourceConstructionMonopoly``) must be
        upgraded rather than left understating the guarantee.

        The authoritative sealing check is the AST subclass-site ban, which reads
        source text and cannot be spoofed -- see
        ``TestSourceConstructionMonopoly.test_detector_flags_a_subclass_site``.
        """
        spoofed = type("Spoof", (_SealedSource,), {"__module__": "neograph._ir_source"})
        assert spoofed.__mro__[1] is _SealedSource, (
            "the one-line __module__ spoof no longer constructs. The runtime "
            "tripwire is now stronger than _ir_source's docstring claims -- update "
            "the docstring's tiering instead of leaving it pessimistic."
        )


class TestEveryVariantHasExactlyOneNamedReadCase:
    """'The interpreter gains exactly one named-read case per variant.'

    ``source_channel_kind`` lives in ``src/neograph/`` on purpose: ``make typecheck``
    is ``mypy src/neograph/``, so the same ``assert_never`` match placed in this test
    file would be graded by NOTHING. Verified when this landed: deleting one case
    yields ``error: Argument 1 to "assert_never" has incompatible type
    "Accumulated"; expected "Never"  [arg-type]`` -- a default-on check under this
    repo's non-strict mypy config, not a strictness-gated one.
    """

    @pytest.mark.parametrize("variant", ALL_VARIANTS, ids=lambda v: type(v).__name__)
    def test_every_variant_names_a_physical_channel(self, variant: object) -> None:
        kind = source_channel_kind(variant)  # type: ignore[arg-type]
        assert kind and isinstance(kind, str)

    def test_the_variant_list_here_is_total_over_the_union(self) -> None:
        """Guards this file: a new variant added to the union but not to
        ALL_VARIANTS would leave the parametrized tests above silently narrower."""
        import typing

        import neograph._ir_source as mod

        union_members = {t.__name__ for t in typing.get_args(mod.Source)}
        covered = {type(v).__name__ for v in ALL_VARIANTS}
        assert covered == union_members, (
            f"ALL_VARIANTS has drifted from the Source union.\n"
            f"  in union, untested: {sorted(union_members - covered)}\n"
            f"  tested, not in union: {sorted(covered - union_members)}"
        )

    def test_channels_are_physical_not_per_feature(self) -> None:
        """Variants map to state channels the runtime HAS, never to features. Two
        variants legitimately share the peer-field channel (``LastPresent`` is an
        ordered set of peer addresses), which is why this asserts a small closed
        vocabulary rather than injectivity."""
        kinds = {source_channel_kind(v) for v in ALL_VARIANTS}  # type: ignore[arg-type]
        assert kinds == {
            "peer-field",
            "port-channel",
            "each-item-channel",
            "carry-list",
            "mesh-channel",
            "accumulator-channel",
        }


class TestResolutionCarriesTheResolversOwnReasons:
    """Design 7.4: diagnostics RENDER the resolver's object; they do not re-probe.
    'Being diagnostic-only is not a licence to re-derive.'"""

    def test_a_resolved_answer_carries_one_source(self) -> None:
        assert Resolved(source=Port()).source == Port()

    def test_an_unresolved_answer_carries_candidates_with_reasons(self) -> None:
        un = Unresolved(
            candidates=(
                Candidate(ref=PortRef("settle", "result"), reason="type mismatch: Seed vs Case"),
                Candidate(ref=PortRef("other"), reason="not reachable on every arm"),
            )
        )
        assert [c.reason for c in un.candidates] == [
            "type mismatch: Seed vs Case",
            "not reachable on every arm",
        ]
        assert un.candidates[0].ref.output == "result"

    def test_unresolved_can_carry_zero_candidates(self) -> None:
        """'Unresolved' covers found-nothing as well as found-several; a resolver
        with no candidates must still be representable without a second type."""
        assert Unresolved(candidates=()).candidates == ()


class TestAccumulatedIsPreRegisteredNotDead:
    """Design 4.1 pre-registers the one variant the roadmap implies, so the
    normalizer mints it and nothing else does."""

    def test_accumulated_exists_and_names_its_own_channel(self) -> None:
        assert source_channel_kind(Accumulated(channel="neo_accum_x")) == "accumulator-channel"

    def test_accumulated_is_not_yet_referenced_by_any_src_module(self) -> None:
        """Pins the 'used by nothing YET' claim. When the accumulator channel lands
        (neograph-iq4a3) this test must be DELETED, not weakened -- it is a
        statement about step 0, and it is the thing that makes 'deliberate, not dead
        code' checkable rather than asserted."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"
        referencing = [
            p.name for p in sorted(root.rglob("*.py")) if p.name != "_ir_source.py" and "Accumulated" in p.read_text()
        ]
        assert referencing == [], (
            f"Accumulated is now referenced by {referencing}. If the accumulator "
            "channel has landed, delete this test; do not relax it."
        )


class TestTheMemberToFieldHopIsComputedHere:
    """``PortRef.field`` -- the member->field hop, in one place.

    Four sites computed it independently and one had it WRONG: after step 1 taught
    assembly to accept the dotted ``"settle.result"``, ``_subconstruct`` still ran
    ``field_name_for`` on the whole string and looked for a field of that literal
    name. Nothing writes it -- the fields are ``settle_result``/``settle_extra`` --
    so a correctly-named port assembled clean and died at run time
    (``neograph-fx3j7``). This is the assertion that must stay right.
    """

    def test_a_dotted_address_resolves_to_the_per_key_state_field(self) -> None:
        assert PortRef("settle", "result").field == "settle_result"

    def test_a_bare_address_resolves_to_the_member_field(self) -> None:
        assert PortRef("settle").field == "settle"

    def test_the_hop_never_yields_a_dotted_field_name(self) -> None:
        """The specific failure mode: a field name containing a dot is one no node
        writes, so it fails a presence check rather than a type check -- which is how
        it reached runtime as "no internal node produced a compatible output value"
        instead of anything naming the real cause."""
        assert "." not in PortRef("settle", "result").field
