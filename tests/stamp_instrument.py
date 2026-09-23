"""The declared-read stamp instrument (neograph-4cvx8.1, step 0).

What it measures
----------------
``neograph-4cvx8``'s Core Invariant: *every declared read leaves ``Construct()``
either holding exactly one stamped ``Source`` or refused with a reason*. A read
that leaves assembly with NEITHER is the eight-times-recurring defect the ticket
exists to retire -- the node declares an input, the resolver finds nothing, stamps
nothing, and the body is handed ``None`` on a GREEN run.

A DECLARED READ, exactly as the ticket enumerates it, is one of three things:

===========================  ==============================================
read                         stamp that must hold it
===========================  ==============================================
single-type ``inputs=X``     ``Node.input_sources[StateKeys.SINGLE_INPUT]``
each dict-form ``inputs``    ``Node.input_sources[<that key>]``
a placed sub-construct port  ``Construct.port_source``
===========================  ==============================================

Why a build-time instrument and not a hypothesis property
---------------------------------------------------------
The superseded plan made step 0 a ``hypothesis`` property under
``xfail(strict=True)``. The architect review (finding I5) refused it on three
counts, all of which this module answers instead of restating:

1. A strict xfail over a randomized generator XPASSes on any run that draws no
   counterexample, so the suite goes red NONDETERMINISTICALLY. This instrument
   draws nothing: it observes the constructs the corpus actually builds.
2. ``tests/hypothesis/topology.py`` generates Each, Oracle, Loop and sub-construct
   shapes and NEVER branch, Portal, ``run_isolated`` or fan-agent shapes -- so the
   property would have gone true at step 3 and never seen N1, N2, N5, N6 or sdqsv.
   The corpus here is the whole suite plus ``tests/check_fixtures`` plus the
   keyless ``examples/``, which is where those shapes actually live.
3. A stamp is not proof of correctness -- N2 is STAMPED and WRONG (it names the
   false arm's producer, and the true arm then delivers ``None``). This instrument
   therefore claims only PRESENCE. Correctness stays with the run-level tests and
   the ``EXPECT`` check-fixtures (neograph-36302). Read a green instrument as
   "nothing is unaddressed", never as "every address is right".

What it deliberately cannot see
-------------------------------
Stated so a green run is not read as wider cover than it has.

- ``Node.run_isolated`` builds no ``Construct`` at all -- it invokes a node fn
  against a hand-built state dict -- so its 4 measured unstamped reads are
  invisible here. Step 7 pins them with a run-level test.
- A STALE stamp (the 28 fan-agent wrapper reads, N5) is a stamp: present, wrong.
  Only the wrapper's ``neo_subgraph_input`` dict key surfaces here, and it
  surfaces because it is unstamped, not because it is stale.
- ``context=`` back-references are declared reads in the ordinary English sense
  but are not in the ticket's enumeration, and are not audited here.

Determinism, precisely
----------------------
The VERDICT is deterministic because it ranges over a CLOSED shape vocabulary
(:data:`SHAPES`): every read classifies into one of finitely many shapes, and the
verdict asks only which shapes were observed unstamped. Counts and sample sites
are diagnostic only and are NOT asserted -- hypothesis draws a different number of
constructs per run, so a count assertion would be exactly the flakiness finding I5
refused.
"""

from __future__ import annotations

import functools
from collections import Counter
from dataclasses import dataclass
from typing import Any

from neograph._ir_branch import iter_item_slots
from neograph._normalize import normalize_inputs
from neograph._portal_member import PortalMemberClass, portal_member_class
from neograph._state_keys import StateKeys
from neograph.node import Node

__all__ = [
    "SHAPES",
    "DeclaredRead",
    "audit_construct",
    "constructs_seen",
    "install",
    "observed",
    "outside_allowlist",
    "record",
    "shape_report",
]


# The CLOSED vocabulary of unstamped-read shapes. Closed on purpose: it is what
# makes the verdict deterministic over a corpus whose SIZE is not (see the module
# docstring). A new shape is a new row here AND a new row in the guard's
# allowlist-or-fix decision -- never a silent extra string.
SHAPES: frozenset[str] = frozenset(
    {
        # A single-type `inputs=X` on an Each-modified node. The value arrives on
        # the fan-out channel; `EachItem` exists as vocabulary and is not stamped
        # for it. ~196 measured sites. Retired by step 3 (neograph-4cvx8.5).
        "single-type/each-item",
        # A single-type `inputs=X` on a Portal MESH MEMBER. The value arrives by
        # hop on the mesh channel. Retired by step 5 (neograph-4cvx8.7 /
        # neograph-sdqsv) -- by REFUSAL, per the recorded decision, not by a
        # stamp. Of the 4 measured mesh sites a PRESENCE check sees one: the
        # others carry a SIBLING member's Peer stamp, which is sdqsv's
        # wrong-producer half and is exactly the "a stamp is not correctness"
        # limit in the module docstring.
        "single-type/mesh-member",
        # A single-type `inputs=X` on a Loop-modified node with no stamp at all.
        # Distinct from the Loop CARRY, which is a second arrival on a node whose
        # SEED is stamped and so is invisible to a presence check. Step 6.
        "single-type/loop",
        # A single-type `inputs=X` with no modifier to explain it: the resolver
        # simply found nothing. This is the shape that must reach ZERO -- it is
        # the one the runtime hands `None`.
        "single-type/unfed",
        # A dict-form key that is a FRAMEWORK port key (`neo_...`), not a peer
        # name -- the fan-agent wrapper's synthesized `neo_subgraph_input` read.
        # Step 7.
        "dict-key/framework-port",
        # An ordinary named dict-form key. Addressed BY NAME today rather than by
        # a stamped `Source`; the disease scan dispositioned this as the target
        # form (row 47), so it is the row the implement atom is expected to
        # allowlist WITH that reason rather than to fix.
        "dict-key/peer",
        # A placed sub-construct declaring `input=` whose `port_source` is None:
        # the parent resolved no feeder for the port. Step 8 (N3/N7, xejyn).
        "port/sub-construct",
    }
)

_MESH_MEMBER_CLASSES = frozenset(
    {
        PortalMemberClass.ATOMIC,
        PortalMemberClass.ATOMIC_OPERATOR,
        PortalMemberClass.AGENT_CYCLE_OUTPUT,
        PortalMemberClass.AGENT_CYCLE_TOOL,
        PortalMemberClass.SUB_CONSTRUCT,
    }
)

# How many sample sites to retain per shape for the failure message. Sites are a
# DIAGNOSTIC, not part of the verdict; the cap keeps a red run readable when a
# shape has thousands of instances.
_SITE_SAMPLE = 8


@dataclass(frozen=True)
class DeclaredRead:
    """One declared read of one member, and whether assembly stamped it."""

    construct: str
    member: str
    key: str
    shape: str
    stamped: bool

    @property
    def site(self) -> str:
        return f"{self.construct}.{self.member}[{self.key}]"


def _single_type_shape(node: Node) -> str:
    modifiers = node.modifier_set
    if modifiers.each is not None:
        return "single-type/each-item"
    if portal_member_class(node) in _MESH_MEMBER_CLASSES:
        return "single-type/mesh-member"
    if modifiers.loop is not None:
        return "single-type/loop"
    return "single-type/unfed"


def _dict_key_shape(key: str) -> str:
    # `neo_`-prefixed (and the leading-underscore `_neo_isolated_input`) keys are
    # framework channels a synthesizer wrote, never a peer the author named.
    return "dict-key/framework-port" if key.lstrip("_").startswith("neo_") else "dict-key/peer"


def _node_reads(construct_name: str, node: Node) -> list[DeclaredRead]:
    inputs = normalize_inputs(node.inputs)
    if inputs.is_none:
        return []
    table = node.input_sources or {}
    if inputs.is_dict_form:
        return [
            DeclaredRead(
                construct=construct_name,
                member=node.name,
                key=key,
                shape=_dict_key_shape(key),
                stamped=key in table,
            )
            for key in inputs.by_name
        ]
    if inputs.single_type is None:
        return []
    return [
        DeclaredRead(
            construct=construct_name,
            member=node.name,
            key=StateKeys.SINGLE_INPUT,
            shape=_single_type_shape(node),
            stamped=StateKeys.SINGLE_INPUT in table,
        )
    ]


def audit_construct(construct: Any) -> list[DeclaredRead]:
    """Every declared read of ``construct``'s OWN members, stamped or not.

    The rule, in one place, so the session collector and the guard's simulated
    read cannot diverge about what "a declared read" is.

    Per-LEVEL, never recursive into a sub-construct's interior: a sub-construct
    audits its own members when IT is built, and its PORT is a read of the parent,
    answerable only where the parent is. Branch arms ARE descended into (they are
    this construct's members) via the shared ``iter_item_slots`` walk, so an arm
    node's read is never skipped.
    """
    name = getattr(construct, "name", "<unnamed>")
    reads: list[DeclaredRead] = []
    for container, idx in iter_item_slots(construct):
        item = container[idx]
        if isinstance(item, Node):
            reads.extend(_node_reads(name, item))
        elif getattr(item, "nodes", None) is not None and getattr(item, "input", None) is not None:
            reads.append(
                DeclaredRead(
                    construct=name,
                    member=getattr(item, "name", "<unnamed>"),
                    key="port",
                    shape="port/sub-construct",
                    stamped=getattr(item, "port_source", None) is not None,
                )
            )
    return reads


# ---------------------------------------------------------------------------
# Session collection
# ---------------------------------------------------------------------------

_COUNTS: Counter[str] = Counter()
_SITES: dict[str, set[str]] = {}
_CONSTRUCTS_SEEN = Counter({"total": 0})


def record(construct: Any) -> None:
    """Audit one successfully built construct into the session totals."""
    _CONSTRUCTS_SEEN["total"] += 1
    for read in audit_construct(construct):
        if read.stamped:
            continue
        _COUNTS[read.shape] += 1
        sites = _SITES.setdefault(read.shape, set())
        if len(sites) < _SITE_SAMPLE:
            sites.add(read.site)


def observed() -> dict[str, int]:
    """Shape -> how many unstamped reads of it the corpus produced."""
    return dict(_COUNTS)


def constructs_seen() -> int:
    """How many successfully built constructs the collector has audited."""
    return _CONSTRUCTS_SEEN["total"]


def outside_allowlist(counts: dict[str, int], allowlist: dict[str, str]) -> dict[str, int]:
    """The shapes in ``counts`` that ``allowlist`` does NOT excuse.

    The verdict, as a pure function of two dicts, so the guard can exercise it
    against a simulated unstamped read while the real allowlist is empty -- an
    invariant stated as an assertion, never as an empty parametrize
    (``AGENTS.md``, neograph-e8wiv).
    """
    return {shape: n for shape, n in sorted(counts.items()) if shape not in allowlist}


def shape_report(counts: dict[str, int]) -> str:
    """A deterministic, readable rendering of ``counts`` with sample sites."""
    lines = []
    for shape, n in sorted(counts.items()):
        sites = sorted(_SITES.get(shape, set()))
        shown = ", ".join(sites[:_SITE_SAMPLE])
        lines.append(
            f"  {shape}: {n} unstamped read(s); e.g. {shown}" if shown else f"  {shape}: {n} unstamped read(s)"
        )
    return "\n".join(lines)


# The wrapper is compiled with a SYNTHETIC filename, and that is load-bearing,
# not cosmetic. `_validation_types._source_location` walks outward from a raise
# to the first frame that is neither neograph nor pydantic and reports it as the
# author's `Construct(...)` call site; an ordinary wrapper defined in this module
# becomes that frame, and every assembly error in the suite starts saying "at
# stamp_instrument.py:NNN" instead of the user's file (caught by
# tests/test_validation.py::...::test_error_includes_source_location_when_mismatch).
# A frame whose filename starts with "<" is already skipped by that walk, so the
# instrument stays transparent to it -- an instrument that changes a user-visible
# error message is not "no behaviour change".
_SHIM_FILENAME = "<neograph stamp instrument>"
_SHIM_SOURCE = """
def instrumented_init(self, *args, **kwargs):
    _original(self, *args, **kwargs)
    _record(self)
"""


def install() -> None:
    """Start auditing every SUCCESSFULLY built ``Construct`` in this process.

    Wraps ``Construct.__init__`` rather than ``normalize_ir``: a construct that
    fails validation must not be measured (it was refused, which is the
    invariant's other legal outcome), and ``port_source`` is stamped by the
    PARENT's normalization, so the audit has to happen after the whole
    ``__init__`` ran.

    Idempotent -- a second call is a no-op, so importing this module from more
    than one place cannot double-count.
    """
    from neograph.construct import Construct

    original = Construct.__init__
    if getattr(original, "_neo_stamp_instrument", False):
        return

    namespace: dict[str, Any] = {"_original": original, "_record": record}
    exec(compile(_SHIM_SOURCE, _SHIM_FILENAME, "exec"), namespace)  # noqa: S102
    shim = functools.wraps(original)(namespace["instrumented_init"])
    shim._neo_stamp_instrument = True
    Construct.__init__ = shim  # type: ignore[method-assign]
