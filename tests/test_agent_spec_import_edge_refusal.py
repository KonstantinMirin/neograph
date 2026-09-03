"""Import refuses AUTHORED ambiguity in a foreign Flow, and only that.

Step 6 of neograph-9axw6, design section 8. Agent Spec resolves two data edges into
one input as last-write-wins; neograph refuses it, because importing the last edge
silently gives a graph that means something different from the one exported.

The discriminator matters as much as the refusal. A modifier group's own members are
named after their owner, and N of them feeding one merge input is STRUCTURAL fan-in
the format legitimately expresses. Refusing that broke the documented stale-marker
fallback, so the rule is narrowed on that naming fact rather than on whatever made
the suite pass.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from neograph._agent_spec_node_import import _inputs_from_data_edges
from neograph.errors import ConstructError


def _edge(src: str, dest: str, dest_input: str) -> NS:
    return NS(
        source_node=NS(name=src),
        destination_node=NS(name=dest),
        destination_input=dest_input,
        source_output=None,
    )


class TestTwoEdgesIntoOneInput:
    def test_two_unrelated_peers_into_one_input_are_refused(self) -> None:
        flow = NS(data_flow_connections=[_edge("alpha", "sink", "payload"), _edge("beta", "sink", "payload")])
        with pytest.raises(ConstructError) as exc:
            _inputs_from_data_edges("sink", flow, {"alpha": int, "beta": int})
        msg = str(exc.value)
        assert "alpha" in msg and "beta" in msg, f"the message must name BOTH edges. Got: {msg}"
        assert "payload" in msg, "the message must name the contested input"

    def test_a_modifier_groups_own_fan_in_is_accepted(self) -> None:
        """The non-vacuity half: without this the refusal above could be a blanket
        ban on every multi-edge input, which breaks a legitimate Oracle fan-in."""
        flow = NS(
            data_flow_connections=[
                _edge("sink__variant_1", "sink", "items"),
                _edge("sink__variant_2", "sink", "items"),
            ]
        )
        result = _inputs_from_data_edges("sink", flow, {"sink__variant_1": int, "sink__variant_2": int})
        assert result is not None
        assert sorted(result) == ["sink__variant_1", "sink__variant_2"]

    def test_one_edge_per_input_is_unaffected(self) -> None:
        flow = NS(data_flow_connections=[_edge("alpha", "sink", "a"), _edge("beta", "sink", "b")])
        result = _inputs_from_data_edges("sink", flow, {"alpha": int, "beta": int})
        assert result is not None
        assert sorted(result) == ["alpha", "beta"]
