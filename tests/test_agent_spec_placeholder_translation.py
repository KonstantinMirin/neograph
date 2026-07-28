"""Option F regression tests (neograph-cbpyx): translating neograph ``${var}``
prompts to pyagentspec ``{{ var }}`` placeholders on the Agent Spec EXPORT path.

These pin the behavior the Option F implementation (``_translate_placeholders``,
amending m57mn's Option-B fail-loud guard) creates -- driven through the REAL
``to_agent_spec`` / ``from_agent_spec`` surface over REAL neograph ``Construct``s
and REAL pyagentspec ``Flow``/``LlmNode``/``StartNode`` objects (no mocks at the
translation or serialization boundary).

TDD RED: today, a think/agent/act node with real ``Node.inputs`` and a native
``${var}`` prompt (which pyagentspec's ``{{ }}``-only scanner never sees) raises
``ConfigurationError`` at the ``to_agent_spec`` call (the Option-B guard,
``_check_placeholder_inputs``). So the export-side tests below currently fail at
that call; the collision test currently raises the WRONG error (missing-placeholder,
not a flatten collision). Confirmed by running pytest, not by inspection.

Scope: this file does NOT touch the 8 Oracle+agent/act cells (sibling task
neograph-i7k7j). It pins:
  1. Unreferenced-input round-trip (design doc s3 + addendum item 1).
  2. Flatten collision -> ConfigurationError naming both (design doc s2.1 rule 6).
  3. MEDIUM-2: marker wire-portability (JSON-native marker survives a JSON round trip).
  4. MEDIUM-1: think-each fillability (sub-flow StartNode.inputs == inner flat name).

Run with::

    uv run --extra agent-spec pytest tests/test_agent_spec_placeholder_translation.py
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

pytest.importorskip("pyagentspec")

from neograph import Construct, Node  # noqa: E402
from neograph._agent_spec import to_agent_spec  # noqa: E402
from neograph.errors import ConfigurationError  # noqa: E402
from neograph.loader import from_agent_spec  # noqa: E402
from neograph.modifiers import Each  # noqa: E402

from .schemas import Claims, RawText, _producer  # noqa: E402


class TestUnreferencedInputRoundTrip:
    """Design doc s3 + addendum item 1: a think-mode node declaring
    ``inputs={'prod': T, 'unused': U}`` whose prompt references ONLY
    ``${prod.text}`` must export an ``LlmNode`` carrying ONLY the ``prod``-derived
    Property (an ``LlmNode`` whose sole data channel is its prompt genuinely has
    no data path to ``unused``) with NO ``DataFlowEdge`` targeting ``unused`` --
    yet ``from_agent_spec`` must STILL reconstruct ``unused`` in the original
    ``Node.inputs`` via the round-trip marker (round-trip fidelity and "what the
    exported LlmNode consumes" are two different questions).

    FAILS NOW at the ``to_agent_spec`` call: the Option-B guard rejects the
    ``${prod.text}`` prompt because it contains no ``{{ }}`` placeholder.
    """

    def _pipeline(self) -> Construct:
        prod = _producer("prod", RawText)  # RawText.text
        unused = _producer("unused", Claims)  # Claims.items -- never referenced
        summarize = Node(
            name="summarize",
            mode="think",
            model="fast",
            prompt="process ${prod.text}",
            inputs={"prod": RawText, "unused": Claims},
            outputs=Claims,
        )
        return Construct("unreferenced-input", nodes=[prod, unused, summarize])

    def test_exported_llmnode_carries_only_referenced_property_and_no_unused_edge(self):
        from pyagentspec.flows.edges import DataFlowEdge
        from pyagentspec.flows.nodes import LlmNode

        flow = to_agent_spec(self._pipeline())

        spec_node = next(n for n in flow.nodes if n.name == "summarize")
        assert isinstance(spec_node, LlmNode)

        titles = {p.title for p in (spec_node.inputs or [])}
        assert titles == {"prod_text"}, (
            f"exported LlmNode.inputs must carry ONLY the referenced, translated "
            f"'prod_text' Property (not the unreferenced 'unused' input), got {titles!r}"
        )

        data_edges = [e for e in flow.data_flow_connections if isinstance(e, DataFlowEdge)]
        assert not any(
            e.source_node.name == "unused" and e.destination_node.name == "summarize"
            for e in data_edges
        ), "no DataFlowEdge may target the unreferenced 'unused' input -- the LlmNode has no data path to it"

    def test_roundtrip_marker_reconstructs_unreferenced_input(self):
        flow = to_agent_spec(self._pipeline())
        imported = from_agent_spec(flow)

        reconstructed = next(n for n in imported.nodes if getattr(n, "name", None) == "summarize")
        assert isinstance(reconstructed.inputs, dict), (
            "reconstructed inputs must be dict-form so the original upstream keys survive"
        )
        assert set(reconstructed.inputs.keys()) == {"prod", "unused"}, (
            "from_agent_spec must reconstruct BOTH original inputs (incl. the "
            f"unreferenced 'unused') via the marker, got {set(reconstructed.inputs.keys())!r}"
        )


class TestFlattenCollisionFailsLoud:
    """Design doc s2.1 rule 6: two DISTINCT ``${path}`` references that flatten
    (``.`` -> ``_``) to the SAME flat name must raise ``ConfigurationError``
    naming both original paths AND the collided flat name -- never a silent
    disambiguation. ``a.b_c`` and ``a_b.c`` both flatten to ``a_b_c``.

    FAILS NOW: the Option-B guard raises for a DIFFERENT reason (missing
    ``{{ }}`` placeholders), so its message names the raw dotted titles but
    NOT the collided flat name ``a_b_c`` -- the assertion distinguishing a real
    collision message from the incumbent fail-loud is what currently fails.
    """

    def _pipeline(self) -> Construct:
        class M1(BaseModel, frozen=True):
            b_c: str

        class M2(BaseModel, frozen=True):
            c: str

        a = _producer("a", M1)  # -> title "a.b_c" -> flat "a_b_c"
        a_b = _producer("a_b", M2)  # -> title "a_b.c" -> flat "a_b_c" (COLLISION)
        think = Node(
            name="think1",
            mode="think",
            model="fast",
            prompt="use ${a.b_c} and ${a_b.c}",
            inputs={"a": M1, "a_b": M2},
            outputs=Claims,
        )
        return Construct("flatten-collision", nodes=[a, a_b, think])

    def test_colliding_flat_names_raise_configuration_error_naming_both(self):
        with pytest.raises(ConfigurationError) as exc_info:
            to_agent_spec(self._pipeline())

        msg = str(exc_info.value)
        assert "a_b_c" in msg, (
            "the collision error must name the collided FLAT name 'a_b_c' -- the "
            "incumbent missing-placeholder guard never mentions it (this assertion "
            f"distinguishes a real flatten-collision message from fail-loud), got:\n{msg}"
        )
        assert "a.b_c" in msg and "a_b.c" in msg, (
            f"the collision error must name BOTH original paths 'a.b_c' and 'a_b.c', got:\n{msg}"
        )


class TestMarkerWirePortability:
    """MEDIUM-2 (refinement j8bch.12): the ``neograph/prompt_spec`` round-trip
    marker payload must be strictly JSON-native. If it stores live pyagentspec
    ``Property`` objects, they degrade to plain dicts across a JSON wire round
    trip and the loader's un-flatten (``_dict_form_inputs_from_props``, which
    needs ``.title``/``.model_copy``) crashes -- silently breaking the
    "portable, foreign-consumable + lossless round trip" Core Invariant on any
    JSON/YAML boundary. neograph's only in-memory round-trip test would pass
    even then, so this closes the latent seam.

    FAILS NOW at the ``to_agent_spec`` call (Option-B guard); once translation
    lands, it re-fails on the JSON boundary if the marker is not JSON-native.
    """

    def _pipeline(self) -> Construct:
        prod = _producer("prod", RawText)
        summarize = Node(
            name="summarize",
            mode="think",
            model="fast",
            prompt="process ${prod.text}",
            inputs={"prod": RawText},
            outputs=Claims,
        )
        return Construct("wire-portable", nodes=[prod, summarize])

    def test_prompt_spec_marker_payload_is_strictly_json_native(self):
        # NOTE (neograph-cbpyx impl): the original form of this test called
        # ``flow.model_dump(mode="json")``, which raises "Missing proper
        # serialization context" for EVERY pyagentspec Flow (Components require a
        # serialization context; the JSON-native serializer is ``to_dict()``), so
        # it tested pyagentspec's Component machinery, not this marker. And
        # ``json.dumps(flow.to_dict())`` does NOT discriminate either -- ``to_dict``
        # coerces a stray live Property to a dict, so it would pass even on the
        # regression. The DIRECT, discriminating check for "the marker is strictly
        # JSON-native" is to json.dumps the marker payload itself: a live
        # pyagentspec ``Property`` object there raises ``TypeError`` (not JSON
        # serializable). The end-to-end JSON WIRE round trip is covered by the
        # sibling test below.
        from neograph._agent_spec import _MARK_PROMPT_SPEC

        flow = to_agent_spec(self._pipeline())
        spec_node = next(n for n in flow.nodes if n.name == "summarize")
        marker = (spec_node.metadata or {}).get(_MARK_PROMPT_SPEC)
        assert marker is not None, "think LlmNode must carry the neograph/prompt_spec round-trip marker"
        # Strictly JSON-native: a live Property in original_inputs would make this
        # json.dumps raise TypeError; a JSON-native marker round-trips unchanged.
        assert json.loads(json.dumps(marker)) == marker, (
            "the neograph/prompt_spec marker must be strictly JSON-native (str/dict/list only) so "
            "it survives a JSON/YAML wire round trip -- a stored live pyagentspec Property is not "
            f"JSON serializable, got {marker!r}"
        )

    def test_marker_survives_json_wire_round_trip_and_reconstructs_inputs(self):
        from pyagentspec.flows.flow import Flow

        flow = to_agent_spec(self._pipeline())

        # Cross a real JSON boundary: to_dict is pyagentspec's JSON-native wire
        # form; from_dict rebuilds Component objects but leaves metadata markers
        # as PLAIN dicts (a stored Property would NOT be re-instantiated).
        wire = json.loads(json.dumps(flow.to_dict()))
        reloaded = Flow.from_dict(wire)

        imported = from_agent_spec(reloaded)
        reconstructed = next(n for n in imported.nodes if getattr(n, "name", None) == "summarize")
        assert isinstance(reconstructed.inputs, dict)
        assert "prod" in reconstructed.inputs, (
            "after a JSON wire round trip, from_agent_spec must STILL reconstruct the "
            "original 'prod' input from the JSON-native marker (fails if the marker "
            f"stored live Property objects), got {reconstructed.inputs!r}"
        )


class TestThinkEachFillability:
    """MEDIUM-1 (refinement j8bch.12): a think-mode Each node whose inner prompt
    references ``${item.v}`` -- the sub-flow's ``StartNode.inputs`` is a
    NON-DataFlowEdge consumer of ``_properties_for(node.inputs)`` (dotted
    ``item.v``), while the inner ``LlmNode`` is translated to the flat
    ``item_v``. Both must use the SAME flat name so the ``{{ item_v }}``
    placeholder is actually FILLABLE by a foreign consumer -- a name mismatch
    ships an unfillable placeholder (a silent seam), which is worse than a red
    cell. Asserts fillability, not merely that export succeeds.

    FAILS NOW at the ``to_agent_spec`` call: the inner think node's
    ``${item.v}`` prompt trips the Option-B guard.
    """

    def _pipeline(self) -> Construct:
        class Item(BaseModel, frozen=True):
            v: str

        class Bag(BaseModel, frozen=True):
            items: list[Item]

        class Out(BaseModel, frozen=True):
            r: str

        bag = _producer("bagmaker", Bag)
        check = Node(
            name="check",
            mode="think",
            model="fast",
            prompt="evaluate ${item.v}",
            inputs={"item": Item},
            outputs=Out,
        ) | Each(over="bagmaker.items", key="v")
        return Construct("think-each-fillable", nodes=[bag, check])

    def test_subflow_start_and_inner_share_the_same_flat_input_name(self):
        from pyagentspec.flows.nodes import LlmNode, MapNode, StartNode

        flow = to_agent_spec(self._pipeline())

        map_node = next(n for n in flow.nodes if isinstance(n, MapNode))
        sub = map_node.subflow

        start = next(n for n in sub.nodes if isinstance(n, StartNode))
        inner = next(n for n in sub.nodes if isinstance(n, LlmNode))

        start_titles = {p.title for p in (start.inputs or [])}
        inner_titles = {p.title for p in (inner.inputs or [])}

        assert inner_titles == {"item_v"}, (
            f"inner think LlmNode must translate ${{item.v}} -> flat 'item_v', got {inner_titles!r}"
        )
        assert start_titles == inner_titles, (
            "the sub-flow StartNode.inputs must use the SAME flat name as the inner "
            f"node so {{ item_v }} is fillable, got start={start_titles!r} inner={inner_titles!r}"
        )
