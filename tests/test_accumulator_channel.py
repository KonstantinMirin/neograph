"""neograph-iq4a3 / GH #16: an ACCUMULATOR CHANNEL -- many nodes append, one
downstream node reads the union.

A value could enter state only as a NODE'S OUTPUT, so accumulating across an Each
fan cost three artifacts none of which was domain logic: a wrapper type carrying
two things through one port, a scripted node inside the branch whose only job was
to push the branch's readings out, and a hand-written merge at the parent. The
failure that motivated it was SILENT: a branch returning a bare ``Claim`` dropped
every reading it took by construction, and the terminal judge re-decided over the
opening reads while each claim cited evidence that was gone. Green run.

The channel is declared where it is WRITTEN, by name, as a dict-form output key
whose value is ``Accumulate[T]``::

    Node(..., outputs={"result": Verdict, "readings": Accumulate[Reading]})   # appends
    Node(..., inputs={"readings": list[Reading]})                            # reads the union

The field on the bus is ``readings`` -- UNPREFIXED, shared by every node that
declares it, merged by the existing ``_concat_reducer``. Nothing else about the
fan changes: the branch's ordinary output stays a per-branch dict, and a branch
still sees only the snapshot it was sent.

ORDERING, defined: the union is ordered WITHIN a branch and UNORDERED ACROSS
branches (Each collects ``Send()`` results in arrival order -- the same rule the
``list[X]``-consumer-of-Each caveat already documents). A consumer wanting
determinism sorts on a stable key it put in the element; these tests do exactly
that, so they cannot pass by accident of scheduling.

Same-snapshot isolation inside the fan is a LangGraph precondition (a superstep
runs from one snapshot and applies writes after it), not something these tests
can make load-bearing -- so they do not pretend to. A node cannot read the
channel it appends to: nothing preceding it produces the field, and the existing
no-producer check refuses it with its usual message.

Three surfaces (AGENTS.md three-surface parity rule): declarative ``Node(...)``,
the programmatic pipe (``Node.scripted(...) | Each(...)``, which is the same
construction), and ``@node`` through ``construct_from_module``.
"""

from __future__ import annotations

import types

import pytest
from pydantic import BaseModel

from neograph import Construct, Each, Node, compile, construct_from_module, node, run
from neograph.errors import ConstructError
from tests.fakes import build_test_compile_kwargs, register_scripted


class Case(BaseModel, frozen=True):
    claims: list[str]


class Reading(BaseModel, frozen=True):
    """One piece of evidence a branch took. ``claim`` is the stable key the
    consumer sorts on, so the assertions are order-independent by construction."""

    claim: str
    locus: str


class Verdict(BaseModel, frozen=True):
    claim: str
    status: str


class Decision(BaseModel, frozen=True):
    readings_seen: int
    loci: list[str]
    verdicts: list[str]


def _register_bodies() -> None:
    register_scripted("acc_seed", lambda _i, _c: Case(claims=["c1", "c2", "c3"]))

    def _verify(input_data, _config):
        claim = input_data["item"]
        return {
            "result": Verdict(claim=claim, status="supported"),
            "readings": [Reading(claim=claim, locus=f"{claim}:a"), Reading(claim=claim, locus=f"{claim}:b")],
        }

    register_scripted("acc_verify", _verify)

    def _decide(input_data, _config):
        readings = sorted(input_data["readings"], key=lambda r: r.locus)
        verdicts = input_data["verify_result"]
        return Decision(
            readings_seen=len(readings),
            loci=[r.locus for r in readings],
            verdicts=sorted(v.claim for v in verdicts.values()),
        )

    register_scripted("acc_decide", _decide)


def _declarative() -> Construct:
    from neograph import Accumulate

    _register_bodies()
    seed = Node(name="seed", mode="scripted", scripted_fn="acc_seed", outputs=Case)
    verify = Node(
        name="verify",
        mode="scripted",
        scripted_fn="acc_verify",
        inputs={"item": str},
        outputs={"result": Verdict, "readings": Accumulate[Reading]},
    ) | Each(over="seed.claims", key="text")
    decide = Node(
        name="decide",
        mode="scripted",
        scripted_fn="acc_decide",
        inputs={"verify_result": dict[str, Verdict], "readings": list[Reading]},
        outputs=Decision,
    )
    return Construct("acc-declarative", nodes=[seed, verify, decide])


def _programmatic() -> Construct:
    from neograph import Accumulate

    _register_bodies()
    seed = Node.scripted("seed", fn="acc_seed", outputs=Case)
    verify = Node.scripted(
        "verify",
        fn="acc_verify",
        inputs={"item": str},
        outputs={"result": Verdict, "readings": Accumulate[Reading]},
    ) | Each(over="seed.claims", key="text")
    decide = Node.scripted(
        "decide",
        fn="acc_decide",
        inputs={"verify_result": dict[str, Verdict], "readings": list[Reading]},
        outputs=Decision,
    )
    return Construct("acc-programmatic", nodes=[seed, verify, decide])


def _decorated() -> Construct:
    from neograph import Accumulate

    mod = types.ModuleType("test_accumulator_channel_mod")

    @node(outputs=Case)
    def seed() -> Case:
        return Case(claims=["c1", "c2", "c3"])

    @node(outputs={"result": Verdict, "readings": Accumulate[Reading]}, map_over="seed.claims", map_key="text")
    def verify(item: str):
        return {
            "result": Verdict(claim=item, status="supported"),
            "readings": [Reading(claim=item, locus=f"{item}:a"), Reading(claim=item, locus=f"{item}:b")],
        }

    @node(outputs=Decision)
    def decide(verify_result: dict[str, Verdict], readings: list[Reading]) -> Decision:
        rs = sorted(readings, key=lambda r: r.locus)
        return Decision(
            readings_seen=len(rs), loci=[r.locus for r in rs], verdicts=sorted(v.claim for v in verify_result.values())
        )

    mod.seed, mod.verify, mod.decide = seed, verify, decide
    return construct_from_module(mod, name="acc-decorated")


SURFACES = {"declarative": _declarative, "programmatic": _programmatic, "decorated": _decorated}


@pytest.mark.parametrize("surface", sorted(SURFACES))
class TestAccumulatorChannelUnionsAcrossAnEachFan:
    """The acceptance criteria, proven by a RUN and not by an assembly."""

    def test_every_branch_appends_and_the_downstream_reader_gets_the_union(self, surface: str) -> None:
        graph = compile(SURFACES[surface](), **build_test_compile_kwargs())

        result = run(graph, input={"node_id": "t"})

        decision = result["decide"]
        # (a) the UNION: 3 branches x 2 readings, exact membership, order-independent.
        assert decision.readings_seen == 6
        assert decision.loci == sorted(f"{c}:{s}" for c in ("c1", "c2", "c3") for s in ("a", "b")), (
            "GH #16: the downstream reader did not receive the union of every branch's appends. "
            f"Got loci={decision.loci}"
        )
        # (c) the channel on the final result is a FLAT list -- NOT Each-wrapped {key: [...]}
        # and NOT node-prefixed. This is the assertion a wrong _state_write path fails.
        assert isinstance(result["readings"], list) and len(result["readings"]) == 6, (
            f"the channel must be a flat, unprefixed list on the bus; got {result.get('readings')!r}"
        )
        assert "verify_readings" not in result, "a channel key must not be written as {node}_{key}"

    def test_the_ordinary_per_branch_output_is_untouched(self, surface: str) -> None:
        """(b) Isolation of what is NOT the channel: the branch's `result` stays a
        per-branch dict under `_merge_dicts`, keyed by the each-key. The channel
        is the ONLY thing that unions."""
        graph = compile(SURFACES[surface](), **build_test_compile_kwargs())

        result = run(graph, input={"node_id": "t"})

        per_branch = result["verify_result"]
        assert set(per_branch) == {"c1", "c2", "c3"}
        assert all(per_branch[c].claim == c for c in per_branch)
        assert result["decide"].verdicts == ["c1", "c2", "c3"]


class TestAccumulatorChannelRefusals:
    """The north star: the wrong program is unrepresentable or refused at
    assembly -- never a green run that quietly drops or mistypes evidence."""

    def test_two_declarations_of_one_channel_must_agree_on_element_type(self) -> None:
        from neograph import Accumulate

        register_scripted("acc_r1", lambda _i, _c: {"result": Verdict(claim="x", status="s"), "readings": []})
        register_scripted("acc_r2", lambda _i, _c: {"result": Verdict(claim="y", status="s"), "readings": []})
        a = Node.scripted("a", fn="acc_r1", outputs={"result": Verdict, "readings": Accumulate[Reading]})
        b = Node.scripted("b", fn="acc_r2", outputs={"result": Verdict, "readings": Accumulate[str]})

        with pytest.raises(ConstructError, match="readings"):
            Construct("acc-mismatch", nodes=[a, b])

    def test_a_channel_may_not_share_a_name_with_a_node_field(self) -> None:
        from neograph import Accumulate

        register_scripted("acc_c1", lambda _i, _c: Verdict(claim="x", status="s"))
        register_scripted("acc_c2", lambda _i, _c: {"result": Verdict(claim="y", status="s"), "readings": []})
        readings = Node.scripted("readings", fn="acc_c1", outputs=Verdict)
        b = Node.scripted("b", fn="acc_c2", outputs={"result": Verdict, "readings": Accumulate[Reading]})

        with pytest.raises(ConstructError, match="readings"):
            Construct("acc-collision", nodes=[readings, b])

    def test_a_channel_cannot_be_the_primary_output_of_an_llm_node(self) -> None:
        from neograph import Accumulate

        with pytest.raises(ConstructError, match="primary|first"):
            Construct(
                "acc-llm-primary",
                nodes=[
                    Node(
                        name="think",
                        mode="think",
                        model="fast",
                        prompt="p",
                        outputs={"readings": Accumulate[Reading], "result": Verdict},
                    )
                ],
            )

    def test_a_channel_read_with_no_appender_is_refused(self) -> None:
        """A silent EMPTY union is the same defect as the one this feature exists
        to remove. The existing dict-form no-producer check covers it -- for any
        node that is not first in the chain. A FIRST node's dict-form inputs are
        not checked at all today (``_check_item_input`` returns when there are no
        producers), which is the first-of-chain tolerance filed as
        neograph-rfp5s; this feature does not widen that, so the pin is the
        realistic shape: a reader placed after a producer."""
        register_scripted("acc_seed_only", lambda _i, _c: Case(claims=["c1"]))
        register_scripted("acc_reader", lambda i, _c: Decision(readings_seen=0, loci=[], verdicts=[]))
        seed = Node.scripted("seed", fn="acc_seed_only", outputs=Case)
        reader = Node.scripted("reader", fn="acc_reader", inputs={"readings": list[Reading]}, outputs=Decision)

        with pytest.raises(ConstructError, match="readings"):
            Construct("acc-no-appender", nodes=[seed, reader])

    def test_the_bare_single_type_form_is_refused_because_a_channel_needs_a_name(self) -> None:
        from neograph import Accumulate

        register_scripted("acc_bare", lambda _i, _c: [])
        with pytest.raises(ConstructError, match="name"):
            Construct("acc-bare", nodes=[Node.scripted("n", fn="acc_bare", outputs=Accumulate[Reading])])


class TestAccumulatorChannelAtTheSpecBoundaries:
    """A channel has no representation in either serialised form. Both
    boundaries say so instead of flattening it to a per-node ``list[T]`` that
    would load or export as an ordinary key -- the silent-wrong-artifact shape
    ``neograph-t1nbp`` already paid for once."""

    def test_agent_spec_export_refuses_a_channel_loudly(self) -> None:
        from neograph import to_agent_spec
        from neograph.errors import ConfigurationError

        with pytest.raises(ConfigurationError, match="channel"):
            to_agent_spec(_programmatic())

    def test_dump_spec_records_the_channel_as_a_loss(self) -> None:
        from neograph import dump_spec

        doc = dump_spec(_programmatic())
        losses = doc.get("neograph/losses") or []
        assert any("accumulator_channel" in str(entry) for entry in losses), (
            f"the dump must record the channel as a loss, not flatten it to list[T]: losses={losses!r}"
        )
