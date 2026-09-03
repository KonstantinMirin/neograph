"""neograph-35mur / GH #17: a sub-construct's output boundary must resolve by
DECLARED PORT, not by a reverse type-scan over the child's final state.

``_scan_subgraph_output`` (``_subconstruct.py``) walks the child's result dict in
reverse and returns the first value that ``isinstance``-matches ``output=``. That
is a heuristic over a dict whose contents depend on unrelated declarations, and
when two values share the type it silently returns the WRONG one -- no error, the
run completes green, and the caller gets a value the branch never computed.

The shape that found it in the field: a branch declared ``context=['read']``,
which forwards a parent value into the CHILD's state
(``state.py`` appends context fields AFTER the node output fields), and
``output=Case`` because ``Case`` was genuinely the right domain type. Reverse
iteration therefore reaches the INJECTED CONTEXT first. Every branch came back
carrying the opening case -- five readings and zero claims -- and nothing failed.

Note the action at a distance these tests pin: adding a ``context=`` input can
retroactively break an unrelated ``output=``, with no error anywhere.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import Construct, ConstructError, Node, compile, construct_from_functions, node, run, to_agent_spec
from neograph._subconstruct import item_field_names
from neograph.loader import from_agent_spec
from neograph.state import compile_state_model
from tests.fakes import build_test_compile_kwargs, register_scripted


class Seed(BaseModel, frozen=True):
    tag: str


class Case(BaseModel, frozen=True):
    """The domain type. Deliberately NOT a type invented to be distinct -- the
    cost this bug imposed on its reporter was exactly such a type, whose
    docstring had to say 'the boundary type must be one that appears exactly once
    in a branch's state; distinctness is the whole of its job'."""

    label: str
    readings: int


def _register_bodies() -> None:
    register_scripted("mo_seed", lambda _i, _c: Seed(tag="t"))
    register_scripted("mo_opening", lambda _i, _c: Case(label="OPENING", readings=0))
    register_scripted("mo_settle", lambda _i, _c: Case(label="SETTLED", readings=5))


def _parent_with(sub: Construct) -> Construct:
    """A parent that produces an opening ``Case`` BEFORE the sub-construct, so the
    child's ``context=`` forwards a second ``Case`` into the child's state."""
    return Construct(
        "parent",
        nodes=[
            Node.scripted("read", fn="mo_opening", outputs=Case),
            Node.scripted("seed", fn="mo_seed", inputs=Case, outputs=Seed),
            sub,
        ],
    )


def _settle_node(*, with_context: bool = False) -> Node:
    node = Node.scripted("settle", fn="mo_settle", inputs=Seed, outputs=Case)
    return node.model_copy(update={"context": ["read"]}) if with_context else node


class TestContextMustNotShadowTheOutputBoundary:
    """The reported shape, reproduced end to end through a real compile+run."""

    def test_the_branchs_own_value_crosses_the_boundary_when_context_is_declared(self) -> None:
        """neograph-35mur: THE regression test.

        ``settle`` is the node whose output IS the boundary. ``context=['read']``
        forwards the parent's opening ``Case`` into the same child state. The
        boundary must carry what the branch COMPUTED, never the value it was
        merely handed.
        """
        _register_bodies()
        sub = Construct("branch", input=Seed, output=Case, nodes=[_settle_node(with_context=True)])
        result = run(compile(_parent_with(sub), **build_test_compile_kwargs()), input={"node_id": "x"})

        assert result.get("branch") == Case(label="SETTLED", readings=5), (
            "neograph-35mur: the sub-construct boundary returned the INJECTED CONTEXT value instead of "
            f"the one the branch computed. Got {result.get('branch')!r}. This is the silent-wrong-answer "
            "bug: the reverse type-scan matched the context Case because state.py appends context fields "
            "after node output fields, so reverse iteration reaches them first."
        )


    def test_declaring_context_does_not_change_what_crosses_the_boundary(self) -> None:
        """Action at a distance, stated as an equality.

        The two runs differ ONLY by a ``context=`` declaration, which is an INPUT
        concern and must have no bearing on which value is the OUTPUT. A reader
        adding ``context=`` to reach a run-scoped value has no reason to suspect
        they are re-pointing an unrelated boundary.
        """
        _register_bodies()
        outputs = {}
        for label, with_context in (("without", False), ("with", True)):
            sub = Construct("branch", input=Seed, output=Case, nodes=[_settle_node(with_context=with_context)])
            result = run(compile(_parent_with(sub), **build_test_compile_kwargs()), input={"node_id": "x"})
            outputs[label] = result.get("branch")

        assert outputs["with"] == outputs["without"], (
            "neograph-35mur: adding context= to a child node silently changed which value crossed the "
            f"sub-construct's output boundary: without={outputs['without']!r} with={outputs['with']!r}"
        )


class TestOutputResolvesByDeclaredPortName:
    """The wanted API (neograph-35mur acceptance): ``output='settle'`` -- the NAME
    of the node whose output IS the boundary. Exact, not heuristic."""

    def test_output_from_names_the_producer_and_resolves_exactly(self) -> None:
        _register_bodies()
        sub = Construct(
            "branch", input=Seed, output=Case, output_from="settle", nodes=[_settle_node(with_context=True)]
        )
        result = run(compile(_parent_with(sub), **build_test_compile_kwargs()), input={"node_id": "x"})

        assert result.get("branch") == Case(label="SETTLED", readings=5)

    def test_a_name_that_does_not_exist_refuses_at_assembly(self) -> None:
        """Missing refuses like every other WIRING mistake.

        Asserts neograph's own ``ConstructError``, not bare ``Exception``: a looser
        matcher passes on a pydantic ``ValidationError`` whose text merely CONTAINS
        the name, which would keep passing if the feature were never built.

        Pins the EARLIER of the two gates (review T3): ``Construct(...)`` itself
        refuses, not a later ``compile()``. That is the stronger promise -- the typo
        surfaces on the line where it was written.
        """
        _register_bodies()
        with pytest.raises(ConstructError, match="nosuchnode"):
            Construct(
                "branch", input=Seed, output=Case, output_from="nosuchnode", nodes=[_settle_node(with_context=False)]
            )

    def test_a_name_matching_a_context_field_rather_than_an_item_refuses(self) -> None:
        """review T4: the AMBIGUOUS/ineligible half of the criterion.

        ``read`` names a real field in the child's state -- the forwarded context --
        but it is NOT an item of this construct, so it cannot be the boundary. This
        is the exact value the bug used to return; naming it must refuse rather than
        hand back the old wrong answer through a new spelling."""
        _register_bodies()
        with pytest.raises(ConstructError, match="read"):
            Construct("branch", input=Seed, output=Case, output_from="read", nodes=[_settle_node(with_context=True)])


class TestTwoItemProducersResolveByDeclarationOrder:
    """Decision A (maintainer, 2026-08-24): two ITEM-produced values of the boundary
    type are ORDERED, not ambiguous -- the LAST DECLARED item wins.

    Measured before deciding: 270 output-boundary resolutions across the suite, 26
    with more than one type match, of which 24 are ordinary same-typed chains
    (``review``->``revise``, ``write``->``improve``). Refusing on them would break
    the canonical refine sub-construct and push authors toward a type invented
    purely to be type-distinct -- the exact cost this ticket exists to abolish.

    The bug was never that two ITEMS share a type. It was that a value the child
    was HANDED could win at all."""

    def test_the_last_declared_item_wins(self) -> None:
        _register_bodies()
        register_scripted("mo_second", lambda _i, _c: Case(label="SECOND", readings=1))
        sub = Construct(
            "branch",
            input=Seed,
            output=Case,
            nodes=[
                _settle_node(with_context=False),
                Node.scripted("also_case", fn="mo_second", inputs=Case, outputs=Case),
            ],
        )
        result = run(compile(_parent_with(sub), **build_test_compile_kwargs()), input={"node_id": "x"})

        assert result.get("branch") == Case(label="SECOND", readings=1)

    def test_output_from_overrides_declaration_order(self) -> None:
        """The override exists for exactly the case position gets wrong."""
        _register_bodies()
        register_scripted("mo_second", lambda _i, _c: Case(label="SECOND", readings=1))
        sub = Construct(
            "branch",
            input=Seed,
            output=Case,
            output_from="settle",
            nodes=[
                _settle_node(with_context=False),
                Node.scripted("also_case", fn="mo_second", inputs=Case, outputs=Case),
            ],
        )
        result = run(compile(_parent_with(sub), **build_test_compile_kwargs()), input={"node_id": "x"})

        assert result.get("branch") == Case(label="SETTLED", readings=5)


class TestTheNamedPortMustProduceTheBoundaryType:
    """neograph-x8i3s: ``output_from`` is stored on the IR and then not honoured
    by the boundary check.

    ``check_output_from`` verifies the NAME resolves to exactly one item and stops
    there; the boundary-satisfaction check downstream scans EVERY internal
    producer with no idea a port was named. So when the named member's output type
    does not match ``output=`` and a DIFFERENT member happens to match, assembly
    accepts and the run silently falls back to the type scan -- the disambiguator
    is ignored in exactly the situation it exists for.

    The two-producer shape is load-bearing. With a single producer the construct
    IS refused, but for the wrong reason ('no internal node produces a compatible
    type'), so a one-producer test passes without the fix.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "neograph-x8i3s, closed by neograph-9axw6.2 (Step 1 of the port-addressed "
            "data flow epic): the boundary-satisfaction check does not read output_from, "
            "so a named port whose type mismatches is accepted when a peer happens to "
            "match. Remove this marker when Step 1 lands -- strict=True turns this RED "
            "the moment it does."
        ),
    )
    def test_a_named_member_whose_type_mismatches_is_refused_despite_a_matching_peer(self) -> None:
        _register_bodies()
        register_scripted("mo_seed_out", lambda _i, _c: Seed(tag="s"))

        with pytest.raises(ConstructError) as exc:
            Construct(
                "branch",
                input=Seed,
                output=Case,
                output_from="settle",
                nodes=[
                    Node.scripted("other", fn="mo_settle", inputs=Seed, outputs=Case),
                    Node.scripted("settle", fn="mo_seed_out", inputs=Case, outputs=Seed),
                ],
            )

        msg = str(exc.value)
        assert "settle" in msg, f"the message must name the member that was NAMED. Got: {msg}"
        assert "Seed" in msg, f"the message must state the named member's type. Got: {msg}"
        assert "Case" in msg, f"the message must state the declared boundary type. Got: {msg}"
        assert "no internal node produces" not in msg, (
            "neograph-x8i3s: refusing with the type-scan's message describes the wrong defect. "
            f"The defect is that the member you NAMED produces something else. Got: {msg}"
        )

    def test_a_named_member_whose_type_matches_is_still_accepted(self) -> None:
        """The guard against fixing this by refusing every ``output_from``."""
        _register_bodies()
        register_scripted("mo_second", lambda _i, _c: Case(label="SECOND", readings=1))
        sub = Construct(
            "branch",
            input=Seed,
            output=Case,
            output_from="settle",
            nodes=[
                _settle_node(with_context=False),
                Node.scripted("also_case", fn="mo_second", inputs=Case, outputs=Case),
            ],
        )

        assert sub.output_from == "settle"


class TestContextStaysForwardedAndMerelyBecomesIneligible:
    """review T1: every boundary assertion above can pass for the WRONG reason.

    The tempting minimal fix is "stop forwarding context into the child" or "drop
    context fields from the child's state". Either turns this whole module green
    while BREAKING ``context=`` outright -- the capability the reporter was using
    when they found the bug.

    So the fix is stated as TWO facts that must hold together: the forwarded value
    is STILL THERE, and it is merely no longer ELIGIBLE to be the boundary. A fix
    that removes the field satisfies the first half of this module and fails here.
    """

    def test_the_context_field_is_still_in_the_childs_state(self) -> None:
        sub = Construct("branch", input=Seed, output=Case, nodes=[_settle_node(with_context=True)])
        fields = set(compile_state_model(sub).model_fields)

        assert "read" in fields, (
            "neograph-35mur (review T1): the forwarded context field vanished from the child's state. "
            "The fix narrows what may BE the boundary; it must not stop context= from delivering."
        )

    def test_but_it_is_not_an_eligible_boundary_producer(self) -> None:
        sub = Construct("branch", input=Seed, output=Case, nodes=[_settle_node(with_context=True)])

        assert item_field_names(sub) == ["settle"], (
            "neograph-35mur: only fields the construct's OWN declared items write may be the boundary. "
            "A forwarded context field is a value the child was HANDED, and letting it compete is the bug."
        )


class TestThreeSurfaceParity:
    """CLAUDE.md: an IR-level behavioural change must be exercised through all
    three API surfaces, or the exemption argued.

    The canonical failure this rule exists to catch is a feature that works via
    ``@node`` (which goes through ``_build_construct_from_decorated``) and breaks
    via the programmatic path (which reaches ``Construct(nodes=[...])`` directly).
    ``output_from`` is threaded in the builder AND validated in ``Construct``, so
    both halves need pinning.

    ForwardConstruct is EXEMPT, argued rather than assumed: ``forward.py``
    synthesizes ``Construct(output=<inferred type>)`` from the traced return type
    and has no surface on which an author could name a member, so there is no
    binding for it to carry. Its boundary still benefits from the default
    eligibility rule, which needs no declaration.
    """

    def test_declarative_surface(self) -> None:
        _register_bodies()
        sub = Construct("branch", input=Seed, output=Case, output_from="settle", nodes=[_settle_node()])
        result = run(compile(_parent_with(sub), **build_test_compile_kwargs()), input={"node_id": "x"})
        assert result.get("branch") == Case(label="SETTLED", readings=5)

    def test_programmatic_surface(self) -> None:
        """Built by assignment rather than a literal — the runtime-assembly path."""
        _register_bodies()
        sub = Construct(name="branch", input=Seed, output=Case, nodes=[_settle_node()])
        sub = sub.model_copy(update={"output_from": "settle"})
        result = run(compile(_parent_with(sub), **build_test_compile_kwargs()), input={"node_id": "x"})
        assert result.get("branch") == Case(label="SETTLED", readings=5)

    def test_node_decorator_surface(self) -> None:
        """``construct_from_functions(output_from=...)`` must thread through."""

        @node(outputs=Case)
        def settle(port: Seed) -> Case:
            return Case(label="SETTLED", readings=5)

        sub = construct_from_functions("branch", [settle], input=Seed, output=Case, output_from="settle")
        assert sub.output_from == "settle"

        _register_bodies()
        result = run(compile(_parent_with(sub), **build_test_compile_kwargs()), input={"node_id": "x"})
        assert result.get("branch") == Case(label="SETTLED", readings=5)

    def test_the_decorator_surface_refuses_a_bad_name_too(self) -> None:
        """The refusal must not be reachable only from the declarative surface."""

        @node(outputs=Case)
        def settle(port: Seed) -> Case:
            return Case(label="SETTLED", readings=5)

        with pytest.raises(ConstructError, match="nosuchnode"):
            construct_from_functions("branch", [settle], input=Seed, output=Case, output_from="nosuchnode")


class TestTheDeclaredPortSurvivesAgentSpecRoundTrip:
    """review F6: a boundary PORT is a member NAME, and ``Flow.inputs``/``outputs``
    carry Properties -- i.e. TYPES. So export -> import has no natural place to put
    it, and without an explicit marker the port drops and the reimported construct
    silently reverts to the positional rule.

    That exact trap is already recorded in ``_agent_spec_group_import``'s own
    docstring for ``input``/``output``: they "came back with BOTH set to None ...
    while still passing an is-a-Construct plus combo-matches check, so nothing
    noticed." This pins the third field against the same failure.
    """

    def test_output_from_is_restored_by_from_agent_spec(self) -> None:
        _register_bodies()
        sub = Construct("branch", input=Seed, output=Case, output_from="settle", nodes=[_settle_node()])
        reimported = from_agent_spec(to_agent_spec(_parent_with(sub)))

        subs = [n for n in reimported.nodes if isinstance(n, Construct)]
        assert subs, "the sub-construct did not survive the round trip at all"
        assert subs[0].output_from == "settle", (
            "GH #17: the declared boundary port was dropped by the Agent Spec round trip, so the "
            f"reimported construct is back on the positional rule. Got {subs[0].output_from!r}."
        )
