"""Behavioral pin for the ``PortalMemberClass`` taxonomy + classifier (neograph-dgbqv.3, P7).

TDD RED. ``src/neograph/_portal_member.py`` does not exist yet, so this module
fails at COLLECTION (ImportError). That is the intended red artifact for the
write-test atom neograph-yf2ar.30 — it goes green when the classifier lands.

What this file pins (the classifier's OBSERVABLE contract), and why each part is
here rather than left to the structural guard
``tests/test_guards_portal_member_class_consumers.py``:

* **One class per item.** ``portal_member_class(item)`` answers "what kind of
  Portal mesh participant is this item" for a single IR item, returning
  ``PortalMemberClass | None`` where ``None`` means "carries no Portal at all".
  ``None`` rather than a ``NOT_A_MEMBER`` enum member, so every consumer's
  ``match`` stays exhaustive under ``assert_never`` over real participants
  (dgbqv.3 Risks: "a NOT_A_MEMBER enum member would poison every exhaustive
  match").

* **The precedence, stated and enforced.** This is the architect review's MOST
  IMPORTANT finding (yf2ar.28, resolved by the Refined Plan item 3). The raw
  attribute space is a PRODUCT — ``(node|construct) x mode x operator-present x
  trigger`` — while ``PortalMemberClass`` is a FLAT enum over only the legal
  products. The reduction is therefore LOSSY, and which class an illegal or
  multi-axis product collapses to is a real decision, not an implementation
  detail: the classifier is called on PRE-VALIDATION IR (``state.py`` filters
  all of ``construct.nodes``; ``_construct_validation.py`` and
  ``_validation_portal.py`` run at assembly), so it must return SOMETHING for an
  Operator-guarded agent member and for other products validation later rejects.

  The precedence is ``SUB_CONSTRUCT > AGENT_CYCLE_* > ATOMIC_OPERATOR >
  ATOMIC`` — the order ``_wiring.py``'s ``_add_portal_mesh`` member chain and
  ``_recursion_budget.py``'s ``_member_hop_cost`` already imply. Pinning it here
  means a future reordering is a red test, not a silent re-classification.

* **DISPATCH is a FIFTH outcome, not a member class** (Refined Plan item 8). A
  ``route="decide"`` Portal is a standalone linear node, never a mesh member,
  yet it still reaches ``_member_hop_cost`` today and costs 1 superstep. The
  no-change hop-cost pin below exists so the 6-arm match the migration
  introduces cannot silently re-price it.

Deliberately NOT pinned here: the mesh GROUPING (``_group_portal_members`` stays
the single grouping authority — the classifier answers per-ITEM class only), the
combo decomposition table, and the Portal pair-legality rules at
``modifiers.py``'s ``_DYNAMIC_RULES``. Those are other tickets' single sources
and the classifier must not absorb them.
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import Construct, Node, Operator, Portal

# RED: this module is the deliverable of neograph-dgbqv.3 and does not exist yet.
from neograph._portal_member import PortalMemberClass, portal_member_class


class Handoff(BaseModel, frozen=True):
    goto: str


# ── IR builders (pure IR — nothing here compiles or runs a graph) ─────────────


def _atomic(name: str = "atomic") -> Node:
    return Node(name=name, mode="scripted", fn="_pmc_noop", inputs={"handoff": Handoff}, outputs=Handoff)


def _agent(name: str = "agentic") -> Node:
    return Node(
        name=name,
        mode="agent",
        model="router",
        prompt="test/prompt",
        inputs={"handoff": Handoff},
        outputs=Handoff,
        tools=[],
    )


def _sub(name: str = "sub") -> Construct:
    # Single-type inputs=Handoff (not dict-form {"handoff": Handoff}) so the
    # inner node reads the boundary port by TYPE match -- a sub-construct has
    # no upstream literally named "handoff" to satisfy dict-form's named lookup.
    inner = Node(name=f"{name}_inner", mode="scripted", fn="_pmc_noop", inputs=Handoff, outputs=Handoff)
    return Construct(name, nodes=[inner], input=Handoff, output=Handoff)


def _dispatch_portal() -> Portal:
    return Portal(
        route="decide",
        spec_field="spec",
        input_field="dispatch_input",
        output=Handoff,
        max_depth=3,
    )


# ── the six classes, one item each ───────────────────────────────────────────


class TestEachClassIsReachable:
    """Every ``PortalMemberClass`` member is produced by a real IR item.

    An enum arm no construct can reach is dead vocabulary; this is the
    anti-vacuity half of the behavioral coverage (dgbqv.3 plan step 7).
    """

    def test_atomic_peer_member_classifies_as_atomic(self):
        item = _atomic() | Portal(to=["peer"])
        assert portal_member_class(item) is PortalMemberClass.ATOMIC

    def test_operator_gated_atomic_member_classifies_as_atomic_operator(self):
        item = _atomic() | Portal(to=["peer"]) | Operator(when="pmc_gate")
        assert portal_member_class(item) is PortalMemberClass.ATOMIC_OPERATOR

    def test_agent_mode_output_triggered_member_classifies_as_agent_cycle_output(self):
        item = _agent() | Portal(to=["peer"])  # trigger defaults to "output"
        assert portal_member_class(item) is PortalMemberClass.AGENT_CYCLE_OUTPUT

    def test_act_mode_member_classifies_as_agent_cycle_output(self):
        """``act`` is the mutation-capable twin of ``agent`` — same ReAct cycle,
        same member class. The axis is "does this member have a ReAct turn",
        not the specific mode literal."""
        item = _agent().model_copy(update={"mode": "act"}) | Portal(to=["peer"])
        assert portal_member_class(item) is PortalMemberClass.AGENT_CYCLE_OUTPUT

    def test_tool_triggered_agent_member_classifies_as_agent_cycle_tool(self):
        item = _agent() | Portal(to=["peer"], trigger="tool")
        assert portal_member_class(item) is PortalMemberClass.AGENT_CYCLE_TOOL

    def test_sub_construct_member_classifies_as_sub_construct(self):
        item = _sub() | Portal(to=["peer"])
        assert portal_member_class(item) is PortalMemberClass.SUB_CONSTRUCT

    def test_dispatch_node_classifies_as_dispatch(self):
        item = _atomic("emitter") | _dispatch_portal()
        assert portal_member_class(item) is PortalMemberClass.DISPATCH

    def test_every_enum_member_is_covered_by_this_suite(self):
        """Anti-vacuity tripwire: a member added to the enum without a
        reachability case above fails HERE, not silently."""
        covered = {
            PortalMemberClass.ATOMIC,
            PortalMemberClass.ATOMIC_OPERATOR,
            PortalMemberClass.AGENT_CYCLE_OUTPUT,
            PortalMemberClass.AGENT_CYCLE_TOOL,
            PortalMemberClass.SUB_CONSTRUCT,
            PortalMemberClass.DISPATCH,
        }
        assert set(PortalMemberClass) == covered, (
            "PortalMemberClass gained or lost a member without a reachability "
            "case in TestEachClassIsReachable. Add the case; do not relax this."
        )


class TestNonParticipantsReturnNone:
    """``None`` means "carries no Portal at all" — the contract that keeps the
    enum closed over participants (dgbqv.3 Risks)."""

    def test_plain_node_without_portal_returns_none(self):
        assert portal_member_class(_atomic()) is None

    def test_operator_only_node_returns_none(self):
        """Operator WITHOUT Portal is not a mesh participant — the operator bit
        only refines a class, it never creates one."""
        assert portal_member_class(_atomic() | Operator(when="pmc_gate")) is None

    def test_plain_sub_construct_without_portal_returns_none(self):
        assert portal_member_class(_sub()) is None

    def test_agent_node_without_portal_returns_none(self):
        """The agent/act mode axis alone must not classify — otherwise every
        agent node in the tree becomes a phantom mesh member."""
        assert portal_member_class(_agent()) is None


class TestLossyPrecedence:
    """The MOST IMPORTANT contract (architect review yf2ar.28 / Refined Plan 3).

    ``PortalMemberClass`` is a LOSSY reduction over the product space
    ``(node|construct) x mode x operator-present x trigger``. When an item
    satisfies more than one class, exactly ONE is returned, by the stated
    precedence ``SUB_CONSTRUCT > AGENT_CYCLE_* > ATOMIC_OPERATOR > ATOMIC``.

    These products are reachable on PRE-VALIDATION IR by construction: the
    classifier is called from ``state.py``/``_construct_validation.py``/
    ``_validation_portal.py`` BEFORE the mesh rules that reject them run, so
    "validation forbids it" is not a reason to leave the behavior unpinned.
    """

    def test_agent_beats_operator(self):
        """An Operator-guarded agent member is ``AGENT_CYCLE_OUTPUT``, NOT
        ``ATOMIC_OPERATOR`` — the operator bit is DROPPED, and that loss is the
        documented cost of the flat enum. (``_validation_portal.py`` rejects
        this product later with its own message; the classifier does not.)"""
        item = _agent() | Portal(to=["peer"]) | Operator(when="pmc_gate")
        assert portal_member_class(item) is PortalMemberClass.AGENT_CYCLE_OUTPUT

    def test_tool_trigger_beats_operator(self):
        item = _agent() | Portal(to=["peer"], trigger="tool") | Operator(when="pmc_gate")
        assert portal_member_class(item) is PortalMemberClass.AGENT_CYCLE_TOOL

    def test_sub_construct_beats_operator(self):
        """A Construct has no ReAct turn, so this is the SUB_CONSTRUCT vs
        ATOMIC_OPERATOR edge of the precedence chain."""
        item = _sub() | Portal(to=["peer"]) | Operator(when="pmc_gate")
        assert portal_member_class(item) is PortalMemberClass.SUB_CONSTRUCT

    def test_tool_trigger_on_atomic_member_does_not_yield_agent_cycle_tool(self):
        """``trigger="tool"`` on an ATOMIC member is an ILLEGAL product
        (``_validation_portal.py`` rejects it: a member with no ReAct turn
        cannot emit a transfer tool call). The classifier sees pre-validation
        IR, so it must still answer — and the answer keys off the member KIND
        (atomic), never off ``trigger`` alone. Otherwise a scripted node would
        be classified into the agent-cycle wiring arm."""
        item = _atomic() | Portal(to=["peer"], trigger="tool")
        assert portal_member_class(item) is PortalMemberClass.ATOMIC

    def test_precedence_is_documented_on_the_classifier(self):
        """The precedence is a design contract, not folklore: it must be stated
        where a reader of the classifier will find it (Refined Plan item 3)."""
        doc = (portal_member_class.__doc__ or "") + (PortalMemberClass.__doc__ or "")
        assert "SUB_CONSTRUCT" in doc and "ATOMIC_OPERATOR" in doc, (
            "The classifier must state its precedence chain "
            "(SUB_CONSTRUCT > AGENT_CYCLE_* > ATOMIC_OPERATOR > ATOMIC) and that "
            "the reduction is lossy on the operator/trigger axis."
        )


class TestDispatchIsNotAMeshMember:
    """DISPATCH is the fifth outcome — a participant that carries a Portal but is
    never a mesh MEMBER (Refined Plan item 8)."""

    def test_dispatch_is_distinct_from_every_member_class(self):
        item = _atomic("emitter") | _dispatch_portal()
        cls = portal_member_class(item)
        assert cls is PortalMemberClass.DISPATCH
        assert cls not in {
            PortalMemberClass.ATOMIC,
            PortalMemberClass.ATOMIC_OPERATOR,
            PortalMemberClass.AGENT_CYCLE_OUTPUT,
            PortalMemberClass.AGENT_CYCLE_TOOL,
            PortalMemberClass.SUB_CONSTRUCT,
        }

    def test_dispatch_member_hop_cost_stays_one(self):
        """NO-CHANGE regression pin (architect review LOW finding, Refined Plan
        item 8). ``_mesh_hop_cost`` appends ANY ``PrimaryShape.PORTAL`` item to
        the run, so a dispatch item does reach ``_member_hop_cost``, which
        returns 1 for it today. When step 5 rewrites that function as a match
        over ``PortalMemberClass``, a missing DISPATCH arm would silently
        re-price the recursion budget.

        Green before AND after the migration — it is here to stay green.
        """
        from neograph._recursion_budget import _member_hop_cost

        assert _member_hop_cost(_atomic("emitter") | _dispatch_portal()) == 1

    def test_agent_member_hop_cost_exceeds_atomic(self):
        """Companion pin: the agent-cycle arm must stay MORE expensive than the
        atomic arm, so a match-arm mix-up in step 5 is loud."""
        from neograph._recursion_budget import _member_hop_cost

        atomic_cost = _member_hop_cost(_atomic() | Portal(to=["peer"]))
        agent_cost = _member_hop_cost(_agent() | Portal(to=["peer"]))
        operator_cost = _member_hop_cost(_atomic() | Portal(to=["peer"]) | Operator(when="pmc_gate"))
        assert atomic_cost == 1
        assert operator_cost == 2
        assert agent_cost > operator_cost


class TestClassifierDoesNotAbsorbOtherAuthorities:
    """The classifier COMPOSES existing single sources; it must not become a
    second one (dgbqv.3 Core Invariant)."""

    def test_classifier_module_does_not_redefine_grouping(self):
        """``_group_portal_members`` stays the single mesh-grouping authority.
        A ``def``/``class`` in the classifier module whose name mentions
        grouping means the per-ITEM classifier grew a per-MESH concern."""
        import ast
        import inspect

        import neograph._portal_member as mod

        tree = ast.parse(inspect.getsource(mod))
        defined = {
            n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        offenders = {n for n in defined if "group" in n.lower() or "mesh" in n.lower()}
        assert not offenders, (
            "The per-item classifier must not define grouping/mesh helpers — "
            f"_group_portal_members is the single grouping authority. Found: {sorted(offenders)}"
        )

    def test_classifier_reads_the_portal_discriminators_rather_than_reimplementing_them(self):
        """``Portal.is_dispatch`` / ``.is_tool_triggered`` stay the single source
        for their own narrower axis (``_portal.py``); the classifier READS them
        and must never re-derive the ``route``/``trigger`` literals inline."""
        import ast
        import inspect

        import neograph._portal_member as mod

        src = inspect.getsource(mod)
        tree = ast.parse(src)
        literals = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Compare)
            and any(isinstance(o, ast.Constant) and o.value in ("decide", "tool") for o in [n.left, *n.comparators])
        ]
        assert not literals, (
            "The classifier re-derives the peer/dispatch or output/tool axis from a "
            f"string literal at line(s) {literals}. Read Portal.is_dispatch / "
            ".is_tool_triggered instead (neograph-f27xo's single-source rule)."
        )

    def test_classifier_accepts_only_items_carrying_a_modifier_set(self):
        """Foreign objects (e.g. the pyagentspec ``Flow`` at ``loader.py``'s
        Swarm reconstruction) have no ``.modifier_set``. They must NOT be
        silently classified — ``loader.py``'s spec-shape mapping is the separate
        deliverable of neograph-dgbqv.5, and a foreign object that merely has a
        ``.mode`` attribute must not walk through the mode axis."""

        class ForeignFlow:
            name = "foreign"
            mode = "agent"

        try:
            result: PortalMemberClass | None = portal_member_class(ForeignFlow())  # type: ignore[arg-type]
        except Exception:
            return  # rejecting outright is the stronger, equally acceptable contract
        assert result is None, (
            "A foreign object with no .modifier_set must raise or classify as None, "
            f"never as a member class. Got {result!r}. This is the loader.py:363 "
            "shape that neograph-dgbqv.5 owns."
        )
