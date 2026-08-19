"""dump_spec -- Construct -> machine-readable spec dict (GH issue #9).

The reported defect is a SERIALIZATION failure: ``Construct.model_dump_json()``
raises ``PydanticSerializationError`` because the IR holds live Pydantic classes
in ``Node.inputs``/``Node.outputs``/``Construct.input``/``Construct.output``, and
``model_dump(mode="python")`` "succeeds" only by leaving raw ``<class ...>``
objects in place -- not transportable. ``load_spec`` existed with no inverse, so
the YAML round trip was one-way.

So the acceptance test here is ``json.dumps()``, not "returns a dict": a dict full
of live classes looks correct right up until something tries to serialize it.

The other half of the contract is honesty. A Construct contains values that are
not data and never can be -- ``Loop(when=lambda d: ...)``, ``skip_when``,
``raw_fn``, the ``@node`` function itself. Those are emitted as an IN-BAND
sentinel at their own site, so a consumer cannot mistake an unrepresentable value
for an absent one (a differ comparing two pipelines whose only difference is the
``when`` lambda must not report "identical").
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from neograph import Construct, Loop, Node, node
from neograph.decorators import construct_from_functions
from tests.fakes import register_scripted


class Claim(BaseModel):
    text: str


class Finding(BaseModel):
    note: str
    score: float = 0.0


def _pipeline() -> Construct:
    """seed -> analyze -> refine, where refine carries a CALLABLE Loop.when.

    A self-loop requires the node's output type to match its own reentry input
    type, so ``refine`` is Finding -> Finding; ``analyze`` supplies the
    Claim -> Finding hop that exercises a non-trivial ``inputs``.
    """
    register_scripted("dump_seed", lambda _in, _cfg: Claim(text="c"))
    register_scripted("dump_analyze", lambda _in, _cfg: Finding(note="n"))
    register_scripted("dump_refine", lambda _in, _cfg: Finding(note="n"))

    seed = Node.scripted("seed", fn="dump_seed", outputs=Claim)
    analyze = Node.scripted("analyze", fn="dump_analyze", inputs=Claim, outputs=Finding)
    refine = Node.scripted("refine", fn="dump_refine", inputs=Finding, outputs=Finding) | Loop(
        when=lambda d: d is None or d.score < 0.8, max_iterations=3
    )
    return Construct("cascade", nodes=[seed, analyze, refine])


class TestDumpSpecSerializes:
    """The reported bug: a Construct cannot be turned into transportable data."""

    def test_construct_model_dump_json_still_fails(self):
        """Control. If this ever stops raising, the motivation for dump_spec
        changed and this whole module should be re-read, not silently kept."""
        from pydantic_core import PydanticSerializationError

        with pytest.raises(PydanticSerializationError):
            _pipeline().model_dump_json()

    def test_dump_spec_output_survives_json_dumps(self):
        """The acceptance criterion, at the locus the bug was reported on."""
        from neograph import dump_spec

        payload = dump_spec(_pipeline())

        text = json.dumps(payload)  # must not raise
        assert json.loads(text) == payload

    def test_declared_types_become_names_not_class_objects(self):
        from neograph import dump_spec

        payload = dump_spec(_pipeline())
        by_name = {n["name"]: n for n in payload["nodes"]}

        assert by_name["seed"]["outputs"] == "Claim"
        assert by_name["analyze"]["inputs"] == "Claim"
        assert by_name["analyze"]["outputs"] == "Finding"

    def test_type_schemas_are_emitted_so_the_dump_is_self_contained(self):
        """A bare type NAME is only meaningful to a process that already
        imported the project; the stated consumer is an external viewer."""
        from neograph import dump_spec

        payload = dump_spec(_pipeline())

        assert "Finding" in payload["types"]
        assert payload["types"]["Finding"]["properties"]["note"]["type"] == "string"


class TestUnrepresentableValuesAreMarkedInBand:
    """A loss is visible AT ITS OWN SITE, not only in a sidecar index."""

    def test_callable_loop_when_becomes_a_sentinel_at_its_own_site(self):
        from neograph import dump_spec

        payload = dump_spec(_pipeline())
        refine = next(n for n in payload["nodes"] if n["name"] == "refine")

        when = refine["loop"]["when"]
        assert isinstance(when, dict), (
            f"a callable Loop.when must be a sentinel object, not {when!r} -- "
            "omitting it lets a differ report two different pipelines as identical"
        )
        assert when["neograph/unrepresentable"] == "callable_loop_when"
        assert "ref" in when
        # The sibling field is untouched.
        assert refine["loop"]["max_iterations"] == 3

    def test_losses_index_enumerates_the_same_loss_with_a_path(self):
        from neograph import dump_spec

        payload = dump_spec(_pipeline())
        losses = payload["neograph/losses"]

        assert any(entry["id"] == "callable_loop_when" for entry in losses)
        entry = next(e for e in losses if e["id"] == "callable_loop_when")
        assert entry["path"].endswith(".loop.when"), entry["path"]

    def test_source_paths_are_repo_relative_not_absolute(self):
        """Absolute paths make two dumps of the same pipeline differ across
        checkouts, defeating the diffing and CI use cases that motivate GH #9."""
        from neograph import dump_spec

        for entry in dump_spec(_pipeline())["neograph/losses"]:
            assert not entry.get("source", "").startswith("/"), entry

    def test_strict_refuses_rather_than_returning_a_lossy_dump(self):
        from neograph import ConfigurationError, dump_spec

        with pytest.raises(ConfigurationError) as excinfo:
            dump_spec(_pipeline(), strict=True)

        assert "callable_loop_when" in str(excinfo.value)

    def test_a_lossless_pipeline_has_an_empty_losses_index(self):
        """Anti-vacuity: the manifest is not just always-non-empty noise."""
        from neograph import dump_spec

        register_scripted("dump_plain", lambda _in, _cfg: Claim(text="c"))
        plain = Construct("plain", nodes=[Node.scripted("only", fn="dump_plain", outputs=Claim)])

        payload = dump_spec(plain)

        assert payload["neograph/losses"] == []
        assert isinstance(payload["nodes"][0]["outputs"], str)


class TestDumpIsDeterministic:
    def test_two_dumps_of_the_same_construct_are_byte_identical(self):
        """The diffing tool and the CI arm-comparison check are the cited
        motivation; non-determinism produces spurious diffs and defeats both."""
        from neograph import dump_spec

        a = json.dumps(dump_spec(_pipeline()), sort_keys=False)
        b = json.dumps(dump_spec(_pipeline()), sort_keys=False)

        assert a == b


    def test_losses_index_is_sorted_by_path_not_by_discovery_order(self):
        """The manifest's order must be a function of CONTENT, not traversal.

        Discovery here runs top-level-node first, sub-construct second, so the
        two losses are FOUND as ``nodes[0]...`` then ``constructs[1]...`` while
        their sorted order is the reverse. Without the sort the index order is
        an artifact of the walk, and two structurally equal pipelines assembled
        in different orders would diff.
        """
        from neograph import dump_spec

        register_scripted("dump_head", lambda _in, _cfg: Finding(note="n"))
        register_scripted("dump_inner", lambda _in, _cfg: Finding(note="n"))

        head = Node.scripted("head", fn="dump_head", inputs=Finding, outputs=Finding) | Loop(
            when=lambda d: d is None, max_iterations=2
        )
        inner = Node.scripted(
            "inner", fn="dump_inner", inputs=Finding, outputs=Finding
        ).model_copy(update={"skip_when": lambda d: False})
        sub = Construct("sub", input=Finding, output=Finding, nodes=[inner])

        losses = dump_spec(Construct("ordered", nodes=[head, sub]))["neograph/losses"]
        paths = [entry["path"] for entry in losses]

        assert len(paths) >= 2, paths
        assert paths == sorted(paths), (
            f"manifest is in discovery order, not sorted order: {paths}"
        )
        assert paths[0].startswith("constructs["), paths

class TestThreeSurfaceParity:
    """Any IR-level behavior must hold across all three API surfaces."""

    def test_decorator_surface_dumps(self):
        from neograph import dump_spec

        @node(outputs=Claim)
        def seed_d() -> Claim:
            return Claim(text="c")

        @node(outputs=Finding)
        def refine_d(seed_d: Claim) -> Finding:
            return Finding(note="n")

        payload = dump_spec(construct_from_functions("dec", [seed_d, refine_d]))

        json.dumps(payload)
        # @node normalizes function names to kebab-case; that is existing
        # neograph naming, not a dump concern.
        assert {n["name"] for n in payload["nodes"]} == {"seed-d", "refine-d"}

    def test_declarative_surface_dumps(self):
        from neograph import dump_spec

        register_scripted("dump_decl", lambda _in, _cfg: Claim(text="c"))
        payload = dump_spec(
            Construct("decl", nodes=[Node.scripted("only", fn="dump_decl", outputs=Claim)])
        )

        json.dumps(payload)
        assert payload["nodes"][0]["name"] == "only"

    def test_programmatic_surface_dumps(self):
        from neograph import dump_spec

        register_scripted("dump_prog", lambda _in, _cfg: Claim(text="c"))
        piped = Node.scripted("only", fn="dump_prog", outputs=Claim) | Loop(
            when="always_false", max_iterations=2
        )
        payload = dump_spec(Construct("prog", nodes=[piped]))

        json.dumps(payload)
        # A REGISTERED-NAME loop condition is data and must round-trip as a string.
        assert payload["nodes"][0]["loop"]["when"] == "always_false"
