"""neograph-t1nbp: a single-type ``inputs=X`` binding must not be resolved by an
unnamed isinstance scan over the whole state bag -- and the two scans that exist
must not disagree with each other.

``inputs=X`` (the single type, not the fan-in dict) is the DOCUMENTED convenience
shorthand, so it is the road a reader is most likely to take. Two independent
implementations resolve it, with OPPOSITE precedence:

  ``_extract_single_type``  (``_input_shape.py``)  forward over ``state.keys()``
                                                   -> the EARLIEST match wins
  the Agent Spec exporter    (``_agent_spec.py``)  ``reversed(ordered_items[:idx])``
                                                   -> the LATEST match wins

Neither resolves to a producer the author NAMED, so with two same-typed values in
scope each picks a different one and neither reports anything. The run is green
and the exported artifact wires an edge the runtime does not take -- a portability
defect on top of a correctness one.

Sibling of ``tests/test_subconstruct_output_boundary.py`` (neograph-35mur), which
is the same disease on the OUTPUT boundary. Sites three and four of four.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import Construct, NeographError, Node, compile, run, to_agent_spec
from tests.fakes import build_test_compile_kwargs, register_scripted


class Case(BaseModel, frozen=True):
    """The domain type, deliberately shared by both producers.

    Two nodes producing one domain type is ordinary pipeline shape -- a stage and
    its refinement -- not a contrivance. The bug needs no exotic construct.
    """

    label: str


def _register_bodies() -> None:
    register_scripted("t1_a", lambda _i, _c: Case(label="FIRST"))
    register_scripted("t1_b", lambda _i, _c: Case(label="SECOND"))
    register_scripted("t1_sink", lambda i, _c: Case(label=i.label))


def _ambiguous_construct() -> Construct:
    """``sink`` declares ``inputs=Case``; both ``a`` and ``b`` produce a ``Case``.

    ``b`` is ``sink``'s immediately preceding producer and itself consumes ``a``,
    so the pipeline reads top to bottom as a -> b -> sink.
    """
    return Construct(
        "t1nbp",
        nodes=[
            Node.scripted("a", fn="t1_a", outputs=Case),
            Node.scripted("b", fn="t1_b", inputs=Case, outputs=Case),
            Node.scripted("sink", fn="t1_sink", inputs=Case, outputs=Case),
        ],
    )


def _exported_producer_of_sink(construct: Construct) -> str | None:
    """The node the EXPORTED artifact wires into ``sink``, by name."""
    flow = to_agent_spec(construct)
    sources = [
        d.source_node.name for d in (flow.data_flow_connections or []) if d.destination_node.name == "sink"
    ]
    return sources[0] if len(sources) == 1 else None


def _runtime_producer_of_sink(construct: Construct) -> str:
    """The node the RUNTIME actually fed ``sink``, recovered by label.

    ``sink`` echoes the label of whatever it was handed, so the value it returns
    names its own producer -- no instrumentation, and it works identically for
    both surfaces.
    """
    result = run(compile(construct, **build_test_compile_kwargs()), input={"node_id": "x"})
    return {"FIRST": "a", "SECOND": "b"}[result.get("sink").label]


class TestSingleTypeInputResolvesToADeclaredProducer:
    def test_the_run_does_not_silently_consume_a_producer_it_never_declared(self) -> None:
        """The correctness half.

        Two outcomes are acceptable and the acceptance criteria name both: resolve
        to the producer the author would name (``b``, ``sink``'s immediate
        upstream), or REFUSE because two values in scope match and nothing
        distinguishes them. The test is disjunctive because the fix is genuinely
        free to choose -- what it may not do is the third thing, which is to hand
        ``sink`` the value from ``a`` and report nothing.
        """
        _register_bodies()

        try:
            producer = _runtime_producer_of_sink(_ambiguous_construct())
        except NeographError as exc:
            # Refusing is a sanctioned outcome -- but a refusal that does not say
            # WHICH producers collided is not one. Assert the diagnostic, so this
            # branch cannot pass by raising something vague.
            assert "a" in str(exc) and "b" in str(exc), (
                f"refused without naming both colliding producers: {exc}"
            )
            return

        assert producer == "b", (
            "neograph-t1nbp: the runtime's forward isinstance scan over state.keys() handed 'sink' the "
            "value from 'a', skipping 'b' -- its immediately preceding producer, and the one an author "
            "reading this pipeline top to bottom would name. No error was raised and the run completed "
            "green: this is the silent-wrong-answer shape, not a crash."
        )

    def test_the_exported_artifact_wires_the_edge_the_runtime_takes(self) -> None:
        """The portability half, and the one that makes this worse than either
        scan alone.

        An exported Flow that wires a DIFFERENT edge than neograph executes hands a
        consumer a different answer from the pipeline it was exported from. The two
        surfaces must agree by construction -- so this asserts agreement rather
        than either particular winner, and stays true whichever resolution the fix
        adopts.
        """
        _register_bodies()

        try:
            runtime_producer = _runtime_producer_of_sink(_ambiguous_construct())
        except NeographError:
            # Symmetry still has to hold: if the runtime refuses to resolve this
            # shape, the export must not quietly wire an edge for it either.
            with pytest.raises(NeographError):
                to_agent_spec(_ambiguous_construct())
            return

        construct = _ambiguous_construct()
        assert _exported_producer_of_sink(construct) == runtime_producer, (
            "neograph-t1nbp: the Agent Spec export wired 'sink' to a different producer than the runtime "
            f"consumed (runtime={runtime_producer!r}). The exporter walks preceding items in REVERSE "
            "(latest match wins) while _extract_single_type walks state.keys() FORWARD (earliest match "
            "wins). Two implementations of one heuristic, two contradictory precedence rules, no shared "
            "authority -- so the exported artifact does not describe the graph it came from."
        )
