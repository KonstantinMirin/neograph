"""neograph-s7zt3.16 -- ``Flow.from_dict()`` erases Property SUBCLASSES, and the
spec_types bridge dispatches on the CLASS, so every list / dict / nested-object
producer loses its type identity on the JSON path.

THE DEFECT, precisely. pyagentspec's ``AgentSpecSerializer`` emits no
``component_type`` discriminator for a ``Property``, so ``Flow.from_dict()`` hands
back a BASE ``Property`` where the in-memory Flow had a ``ListProperty`` /
``DictProperty`` / ``ObjectProperty``. ``json_schema`` survives verbatim -- only the
class is lost. Probed side by side on a ``list[Tagged]`` producer::

    in-memory        cls=ListProperty  type='array'  json_schema={...unchanged...}
    after from_dict  cls=Property      type='array'  json_schema={...unchanged...}

``spec_types`` branches on ``isinstance(prop, pas.ListProperty / DictProperty /
ObjectProperty / UnionProperty)`` in TWO places, and the erasure defeats both --
differently, which is why the symptom is a confusing type MISMATCH rather than a
clean failure:

  * ``_property_to_field_type`` has a base-``Property`` fallback that routes the
    surviving ``json_schema`` through ``_resolve_field_type``. That function's
    object branch builds an ad-hoc ``create_model(schema['title'])`` class -- it
    does NOT go through the canonical structural-dedup path
    (``_structural_type_name`` + ``register_type``) that the typed ObjectProperty
    branch uses. So the two routes yield structurally-identical but NON-IDENTICAL
    classes.
  * ``_property_type_signature`` has no fallback at all: a base ``Property``
    signatures to the bare top-level keyword ``'array'``, discarding the element
    schema entirely.

Net effect: the producer reconstructs as ``list[<ad-hoc Tagged>]`` while the
consumer side reconstructs as ``AgentSpecType_<hash>``, and Each's element-type
identity check (neograph-3lk2l) rejects the pair.

THE ACCEPTANCE LOCUS is the public bridge, so every test here drives the real
serialization path end to end -- ``to_agent_spec -> to_dict -> Flow.from_dict ->
from_agent_spec`` -- and then compiles and RUNS. A unit test of the normalization
helper would prove a sub-claim, never the round-trip.

The in-memory import is asserted alongside as the CONTROL: it passes today, which
is what makes this a serialization defect rather than a bridge defect.

DELIBERATE NARROWING, recorded rather than silently dropped: the consumers below
declare SINGLE-TYPE inputs (``inputs=DictBag``), not the dict-form
(``inputs={'seed': DictBag}``) these fixtures were first written with. Dict-form
is blocked on the JSON path by **neograph-8zvd1**, a separate P1: dict-form
export mutates each Property title to ``'{upstream}.{field}'`` after
construction, and pyagentspec's validator rejects a dotted title on read-back, so
``Flow.from_dict`` fails before any neograph import code runs. That is a title
defect, not an erasure defect, and folding it in here would make these tests
unable to fail for the reason they exist. Switch them to dict-form when 8zvd1
lands -- it strictly widens what they prove.

When this lands, widen
``tests/test_agent_spec_each_operator.py::TestEachOperatorRoundTrip
::test_flow_from_dict_to_dict_round_trip_preserves_the_operator_composite``
back to the full ``_import(rebuilt)`` assertion it was narrowed from.
"""

from __future__ import annotations

import warnings

from pydantic import BaseModel

from neograph import Construct, Each, Node, compile, run
from neograph._agent_spec import to_agent_spec
from neograph.loader import from_agent_spec
from tests.fakes import build_test_compile_kwargs, register_scripted


class Tagged(BaseModel, frozen=True):
    label: str
    value: int


class ListBag(BaseModel, frozen=True):
    items: list[Tagged]


class DictBag(BaseModel, frozen=True):
    entries: dict[str, Tagged]


class NestedBag(BaseModel, frozen=True):
    inner: Tagged


class OptionalBag(BaseModel, frozen=True):
    maybe: Tagged | None


class Result(BaseModel, frozen=True):
    value: str


def _import(flow):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return from_agent_spec(flow)


def _json_round_trip(flow):
    """The exact path that erases Property subclasses."""
    return type(flow).from_dict(flow.to_dict())


def _sole_upstream(input_data):
    """Read the single upstream value regardless of the input SHAPE.

    An imported pipeline's consumer does not receive the shape the original
    declared: the loader derives inputs from the Flow's DataFlowEdges, so a
    ``inputs=DictBag`` (single-type) consumer comes back dict-form and its
    scripted fn is handed ``{'seed': DictBag(...)}`` rather than the bare model.
    That asymmetry is pre-existing and orthogonal to this ticket -- these
    fixtures must not fail over it, or they stop being able to fail for the
    reason they exist. Both shapes carry the same single value.
    """
    if isinstance(input_data, dict):
        return next(iter(input_data.values()))
    return input_data


def _seed_output_model(construct: Construct):
    """The Pydantic model the imported pipeline reconstructed for node 'seed'."""
    seed = next(n for n in construct.nodes if n.name == "seed")
    outputs = seed.outputs
    return outputs if isinstance(outputs, type) else next(iter(outputs.values()))


def _run(construct: Construct, node_id: str):
    return run(compile(construct, **build_test_compile_kwargs()), input={"node_id": node_id})


class TestListProducerSurvivesJsonRoundTrip:
    """The ticket's headline repro, on the PLAIN Each cell -- an already-GREEN
    combo, which is the proof this is pre-existing and not a fusion-combo bug."""

    @staticmethod
    def _pipeline() -> Construct:
        register_scripted(
            "pe_list_seed",
            lambda input_data, config: ListBag(items=[Tagged(label="a", value=1), Tagged(label="b", value=2)]),
        )
        register_scripted("pe_list_step", lambda input_data, config: Result(value=f"tagged-{input_data.label}"))
        return Construct(
            "pe-list",
            nodes=[
                Node.scripted("seed", fn="pe_list_seed", outputs=ListBag),
                Node.scripted("each_step", fn="pe_list_step", inputs=Tagged, outputs=Result)
                | Each(over="seed.items", key="label"),
            ],
        )

    def test_in_memory_path_runs_as_the_control(self):
        """CONTROL: identical pipeline, Flow never serialized. It fans out
        correctly today -- so any difference below is caused by serialization,
        not by the bridge."""
        result = _run(_import(to_agent_spec(self._pipeline())), "pe-list-mem")

        fanned = result["each_step"]
        assert set(fanned) == {"a", "b"}, fanned
        assert {v.value for v in fanned.values()} == {"tagged-a", "tagged-b"}

    def test_list_element_identity_survives_from_dict(self):
        flow = _json_round_trip(to_agent_spec(self._pipeline()))
        imported = _import(flow)

        graph = compile(imported, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "pe-list-rt"})

        fanned = result["each_step"]
        assert set(fanned) == {"a", "b"}, fanned
        assert {v.value for v in fanned.values()} == {"tagged-a", "tagged-b"}


class TestDictProducerSurvivesJsonRoundTrip:
    """``DictProperty`` erases to the same base ``Property``; a ``dict[str, X]``
    producer consumed downstream must keep its value-type identity."""

    @staticmethod
    def _pipeline() -> Construct:
        register_scripted(
            "pe_dict_seed",
            lambda input_data, config: DictBag(entries={"k": Tagged(label="a", value=1)}),
        )
        register_scripted(
            "pe_dict_consume",
            lambda input_data, config: Result(value=f"got-{_sole_upstream(input_data).entries['k'].label}"),
        )
        return Construct(
            "pe-dict",
            nodes=[
                Node.scripted("seed", fn="pe_dict_seed", outputs=DictBag),
                Node.scripted("consumer", fn="pe_dict_consume", inputs=DictBag, outputs=Result),
            ],
        )

    def test_in_memory_path_runs_as_the_control(self):
        """CONTROL: identical pipeline, Flow never serialized. It runs today --
        so any difference below is caused by serialization, not by the bridge."""
        result = _run(_import(to_agent_spec(self._pipeline())), "pe-dict-mem")
        assert result["consumer"].value == "got-a"

    def test_dict_value_identity_survives_from_dict(self):
        imported = _import(_json_round_trip(to_agent_spec(self._pipeline())))

        result = _run(imported, "pe-dict-rt")

        assert result["consumer"].value == "got-a"


class TestNestedObjectProducerSurvivesJsonRoundTrip:
    """``ObjectProperty`` erases to the same base ``Property``.

    NARROWED to a TYPE-IDENTITY claim, not a run, and the reason is recorded
    rather than the coverage silently dropped. A DIRECT nested-model field
    (``inner: Tagged``) cannot be run end to end today because of
    **neograph-p7dyq**, a separate P1 on the EXPORT side: Pydantic hoists any
    nested BaseModel into ``$defs``, and ``_annotation_to_property``'s ``$ref``
    branch then titles the property after the MODEL (``Tagged``) instead of the
    FIELD (``inner``). The imported pipeline consequently waits on a state field
    ``seed.Tagged`` that nothing ever writes. That fires on the in-memory path
    too, so it is not an erasure defect and must not be fixed from here.

    The claim below is the one this ticket actually owns and is unaffected by
    the title bug: the JSON path and the in-memory path must reconstruct the
    nested object to the SAME class. Widen this to a run when p7dyq lands.

    Note the ObjectProperty erasure path is NOT resting on this class alone --
    the list and dict fixtures both carry an ObjectProperty as their element /
    value type and both RUN, so the branch is behaviourally covered there.
    """

    @staticmethod
    def _pipeline() -> Construct:
        register_scripted(
            "pe_obj_seed",
            lambda input_data, config: NestedBag(inner=Tagged(label="a", value=1)),
        )
        register_scripted(
            "pe_obj_consume",
            lambda input_data, config: Result(value=f"got-{_sole_upstream(input_data).inner.label}"),
        )
        return Construct(
            "pe-obj",
            nodes=[
                Node.scripted("seed", fn="pe_obj_seed", outputs=NestedBag),
                Node.scripted("consumer", fn="pe_obj_consume", inputs=NestedBag, outputs=Result),
            ],
        )

    def test_nested_object_reconstructs_to_the_same_class_on_both_paths(self):
        flow = to_agent_spec(self._pipeline())

        from_memory = _seed_output_model(_import(flow))
        from_json = _seed_output_model(_import(_json_round_trip(flow)))

        # One field either way, and the SAME reconstructed class for it --
        # structural equality is not enough, identity is the invariant.
        (mem_field,) = from_memory.model_fields.values()
        (json_field,) = from_json.model_fields.values()
        assert json_field.annotation is mem_field.annotation


class TestUnionProducerSurvivesJsonRoundTrip:
    """``UnionProperty`` is erased by the same mechanism and was the one shape
    with no coverage. An Optional field is the common form of it."""

    @staticmethod
    def _pipeline() -> Construct:
        register_scripted("pe_union_seed", lambda input_data, config: OptionalBag(maybe=Tagged(label="a", value=1)))
        register_scripted(
            "pe_union_consume",
            lambda input_data, config: Result(value=f"got-{_sole_upstream(input_data).maybe.label}"),
        )
        return Construct(
            "pe-union",
            nodes=[
                Node.scripted("seed", fn="pe_union_seed", outputs=OptionalBag),
                Node.scripted("consumer", fn="pe_union_consume", inputs=OptionalBag, outputs=Result),
            ],
        )

    def test_in_memory_path_runs_as_the_control(self):
        """CONTROL: identical pipeline, Flow never serialized. It runs today --
        so any difference below is caused by serialization, not by the bridge."""
        result = _run(_import(to_agent_spec(self._pipeline())), "pe-union-mem")
        assert result["consumer"].value == "got-a"

    def test_union_member_identity_survives_from_dict(self):
        imported = _import(_json_round_trip(to_agent_spec(self._pipeline())))

        result = _run(imported, "pe-union-rt")

        assert result["consumer"].value == "got-a"


class TestErasedPropertyReconstructsToTheCanonicalClass:
    """The divergence itself, asserted directly rather than only through its
    downstream symptom: the JSON path and the in-memory path must reconstruct a
    producer's output field to the SAME registered class, not merely to two
    structurally-equal ones. Without this, a future change could re-fix the Each
    symptom while leaving the two walkers divergent.
    """

    def test_json_path_and_memory_path_agree_on_the_element_class(self):
        pipeline = TestListProducerSurvivesJsonRoundTrip._pipeline()
        flow = to_agent_spec(pipeline)

        from_memory = _seed_output_model(_import(flow))
        from_json = _seed_output_model(_import(_json_round_trip(flow)))

        assert from_json.model_fields["items"].annotation is from_memory.model_fields["items"].annotation
