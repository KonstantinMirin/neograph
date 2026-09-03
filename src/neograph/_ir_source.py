"""Addresses and sources: the vocabulary for "which upstream value is meant".

Every value that moves between nodes has an ADDRESS -- the producing member and
the output it came from. ``_ir_normalize`` resolves that address once, at
assembly, and stores it on the IR; every other layer READS the stored answer.
This module holds the types that answer travels in.

Design: ``docs/design/port-addressed-dataflow-2026-09-03.md`` sections 3, 4, 7, 9.
Step 0 of ``neograph-9axw6``. This module is referenced by no runtime path yet --
step 1 onward wires it in.

Why a LEAF (zero ``neograph`` imports), and why that is not stylistic
--------------------------------------------------------------------
Defining these types anywhere in the assembly cluster forms three simultaneous
import cycles back to ``node.py`` -- via ``_construct_validation``, via
``_normalize``, and via ``construct``. ``TYPE_CHECKING`` cannot dodge it: ``Node``
is a Pydantic model, so a ``Source | None`` field annotation needs the REAL symbol
at class-build time (that field lands in step 9). The one import this module takes is ``neograph.errors``, which is itself a pure
leaf (zero ``neograph`` imports), so it adds no edge toward ``node.py``. It is
taken rather than raising a bare ``TypeError`` because every neograph error goes
through ``NeographError.build`` -- a rule with its own guard
(``test_guards_any_audit``). Satisfying that guard was preferred over growing its
allowlist: an allowlist entry would have bought this module an exemption it does
not need. ``naming.py`` is the only other leaf on this path; if ``PortRef`` ever
needs a field-name helper, importing ``naming`` -- and only ``naming`` -- stays
acyclic.

Frozen dataclasses, never Pydantic: a Pydantic model would drag
``arbitrary_types_allowed`` for the type payloads into every consumer. House style
follows ``_validation_types.Producer``.

What is genuinely unrepresentable here, and what is merely banned
----------------------------------------------------------------
Stated honestly, because overclaiming it is how the ban stops being believed.
Python cannot make CONSTRUCTION unrepresentable: anything importable is
constructible. The tiers, most binding to least:

1. RUNTIME-SUBTRACTIVE (step 3, NOT this step) -- deleting ``StateBus.keys()``
   makes a bag scan stop compiling rather than merely being forbidden.
2. AST BANS on construction AND subclassing (this step) -- they read source text,
   which is not spoofable. See ``TestSourceConstructionMonopoly``.
3. The ``__init_subclass__`` tripwire below -- a cheap RUNTIME refusal that is
   DEFEATED by a one-line dodge (``__module__ = "neograph._ir_source"`` in the
   class body lands in the namespace ``__init_subclass__`` reads). It is labelled
   "banned, detectable", never "unrepresentable": it sits at exactly the tier of
   the construction-token alternative that was rejected for being a one-line dodge,
   so it does not earn a stronger word than that alternative got.

``assert_never`` does NOT cover the sealing property: a subclass of a variant is a
SUBTYPE, so a type checker accepts it where ``Source`` is expected and ``match``
takes the parent's arm. Exhaustiveness and sealing are separate; only the AST
subclass-site check covers the second.

``_portal_member.py`` is cited elsewhere as the closed-classifier exemplar. It is
a flat ENUM with no payloads, so it is NOT the structural model for these types.
What IS worth copying from it is its CONSUMER-ENUMERATION guard
(``tests/test_guards_portal_member_class_consumers.py``): since ``make typecheck``
runs ``mypy src/neograph/`` only, that guard -- not ``assert_never`` alone -- is
what will deliver exhaustiveness once real consumers exist. Scheduled for step 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from neograph.errors import ConstructError
from neograph.naming import field_name_for, output_field_name

__all__ = [
    "Accumulated",
    "Candidate",
    "EachItem",
    "HandoffChannel",
    "LastPresent",
    "LoopCarry",
    "Peer",
    "Port",
    "PortRef",
    "Resolution",
    "Resolved",
    "Source",
    "Unresolved",
    "source_channel_kind",
]


@dataclass(frozen=True)
class PortRef:
    """An address naming a member's SPECIFIC output, not the member.

    A member with dict-form ``outputs`` writes one state field per key, so a member
    NAME does not identify a value -- which is why ``Construct.output_from``,
    which names a member, is ambiguous for such a member (``neograph-kgndo``).

    ``output=None`` means "the sole output". It is accepted at declaration and
    RESOLVED during normalisation, so nothing downstream sees a member-shaped
    address. A member with more than one output and no key is an error that names
    the available ports.
    """

    member: str
    output: str | None = None

    @classmethod
    def parse(cls, spelling: str) -> PortRef:
        """``"member.output"`` -> a port address; ``"member"`` -> its sole output.

        ONE splitter for both directions, so ``input_from`` and ``output_from``
        cannot disagree about what a dotted address means. The spelling is the
        surface both refusals tell an author to write, so it parses in one place.
        """
        member, _, output_key = spelling.partition(".")
        return cls(member, output_key or None)

    @property
    def field(self) -> str:
        """The STATE FIELD this address names.

        The member->field hop, in one place. Four sites used to make it
        independently, and one of them made it WRONG: after step 1 taught assembly
        to accept the dotted ``"settle.result"``, ``_subconstruct`` still ran
        ``field_name_for(port)`` on the whole string and looked for a field called
        ``settle.result``. No node writes that -- the fields are ``settle_result``
        and ``settle_extra`` -- so a correctly-named port assembled clean and died at
        run time with "no internal node produced a compatible output value". The fix
        for a hop computed in four places is not a fifth careful copy.

        Uses ``naming``'s existing helpers rather than inventing a rule.
        ``_ir_source``'s module docstring already sanctions importing ``naming``, and
        only ``naming``, as acyclic.
        """
        base = field_name_for(self.member)
        return output_field_name(base, self.output) if self.output else base


class _SealedSource:
    """Base for the closed ``Source`` set.

    Refuses a subclass declared outside this module. This is a TRIPWIRE, not a
    guarantee: ``__module__`` is spoofable from a class body in one line, so the
    authoritative check is the AST subclass-site ban in
    ``TestSourceConstructionMonopoly``. See the module docstring's tiering.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__module__ != __name__:
            raise ConstructError.build(
                f"{cls.__qualname__} defines a Source variant outside {__name__}",
                expected=f"every Source variant declared in {__name__}",
                found=f"a variant declared in {cls.__module__}",
                hint=(
                    "Source is a CLOSED set: a new arrival mechanism extends "
                    "_ir_source.py -- which makes it visible and forces every "
                    "source_channel_kind consumer to be taught -- rather than "
                    "being added elsewhere. See "
                    "docs/design/port-addressed-dataflow-2026-09-03.md section 4.1."
                ),
            )


@dataclass(frozen=True)
class Peer(_SealedSource):
    """A named member's output -- the ordinary peer-field read."""

    ref: PortRef


@dataclass(frozen=True)
class Port(_SealedSource):
    """The enclosing construct's input port."""


@dataclass(frozen=True)
class EachItem(_SealedSource):
    """The fanned-out item on the ``Each`` channel."""


@dataclass(frozen=True)
class LoopCarry(_SealedSource):
    """This member's own append-list, latest element."""


@dataclass(frozen=True)
class HandoffChannel(_SealedSource):
    """A Portal mesh channel, keyed by the mesh entry.

    Which mesh member handed you a value is a RUNTIME fact, so it is emergent
    ambiguity, not authored: it funnels into one named channel written by the
    mechanism that creates the fact, rather than demanding a name nobody can
    supply (design section 5).
    """

    channel: str


@dataclass(frozen=True)
class LastPresent(_SealedSource):
    """Ordered candidates; take the last one present.

    ADMISSION CRITERIA (design section 4.2). ``LastPresent`` is the variant every
    hard case reaches for, so stamp it ONLY when one of these holds:

    - At most one candidate can be present at run time BY CONSTRUCTION, which
      branch-arm exclusivity guarantees.
    - The ordering encodes a named, documented precedence rule, such as
      carry-before-seed or inner-producer-before-outer-port.

    It is NEVER the answer to a read that unions several present values; that is
    ``Accumulated``. It is NEVER an escape from refusing authored ambiguity.

    Type filtering happens at assembly, so the runtime asks only whether a field is
    PRESENT. Presence cannot select the wrong one of two values. This is the move
    SSA makes with a phi node: name the alternatives instead of searching for them.
    """

    refs: tuple[PortRef, ...]


@dataclass(frozen=True)
class Accumulated(_SealedSource):
    """Many producers append, one downstream reader takes the union.

    PRE-REGISTERED, used by nothing yet (design section 4.1): the accumulator
    channel (``neograph-iq4a3``) is the one new variant the roadmap implies, so the
    normalizer mints it and nothing else does. This is deliberate, not dead code --
    and note that the accumulator BREAKS the one-producer-per-field premise rather
    than adding a case, so the closed set survives it but the producer model does
    not survive unchanged.
    """

    channel: str


Source = Peer | Port | EachItem | LoopCarry | HandoffChannel | LastPresent | Accumulated
"""The CLOSED set of places a value can arrive from.

Variants correspond to KINDS OF STATE CHANNEL THE RUNTIME PHYSICALLY HAS -- peer
field, port channel, each-item channel, carry list, mesh channel, accumulator
channel. They never correspond to features. A new variant is admissible only
alongside a sanctioned new-IR-capability event, the bar ``AGENTS.md`` already sets
for ``_BranchNode`` and Portal. The interpreter gains exactly one named-read case
per variant.
"""


@dataclass(frozen=True)
class Candidate:
    """One near-miss the resolver considered, with the reason it did not win.

    The reason is computed BY THE RESOLVER and carried here so diagnostics RENDER
    the resolver's object instead of re-probing types themselves (design 7.4).
    Being diagnostic-only is not a licence to re-derive.
    """

    ref: PortRef
    reason: str


@dataclass(frozen=True)
class Resolved:
    """The resolver found exactly one answer."""

    source: Source


@dataclass(frozen=True)
class Unresolved:
    """The resolver found zero or several, and says which.

    Error builders render this; they do not re-run the search that produced it.
    """

    candidates: tuple[Candidate, ...]


Resolution = Resolved | Unresolved
"""What the resolver returns for ONE demand."""


def source_channel_kind(source: Source) -> str:
    """Name the PHYSICAL state channel a ``Source`` reads from.

    This function exists for two reasons, and neither is decoration.

    1. It makes the closed-set property GRADED rather than merely asserted. Adding
       an eighth variant without teaching a consumer fails ``make typecheck`` here,
       because the ``assert_never`` below stops being reachable-with-``Never``. That
       grading only happens because this lives under ``src/neograph/`` -- ``make
       typecheck`` is ``uv run mypy src/neograph/`` (``Makefile``), so ``tests/`` is
       never type-checked and the same match placed in a test would be graded by
       NOTHING. The mechanism works under this repo's non-strict config because
       ``arg-type`` is a default-on check, not a strictness-gated one.
    2. It is the seam ``read_source(bus, source)`` grows from in step 3 -- the single
       runtime interpreter and the only place a ``Source`` meets state. Naming the
       channel is the part that does not need the bus.

    At step 0 there are ZERO other consumers, so this is the deliberate non-vacuity
    treatment the closed set gets -- the same treatment the construction ban gets
    from its slip tests. An untested closed set that nothing matches over is a
    passing ratchet.
    """
    match source:
        case Peer():
            return "peer-field"
        case Port():
            return "port-channel"
        case EachItem():
            return "each-item-channel"
        case LoopCarry():
            return "carry-list"
        case HandoffChannel():
            return "mesh-channel"
        case LastPresent():
            return "peer-field"
        case Accumulated():
            return "accumulator-channel"
    assert_never(source)
