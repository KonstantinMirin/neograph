"""Portal knob intent is carried by VALUE, not by ``model_fields_set`` (neograph-dgbqv.6).

``model_fields_set`` records how an instance was CONSTRUCTED, so it does not survive any
boundary that rebuilds the object -- serialization, ``model_validate``, YAML load, Agent-Spec
import. Four hand-rolled encodings existed only to smuggle that intent across those
boundaries, and a legal ``Portal`` could not survive its own ``model_dump()``.
"""

from __future__ import annotations

import pytest

from neograph.modifiers import Portal


class TestPortalSurvivesItsOwnSerialization:
    """The sharpest acceptance test: it is behavioural, and it fails today."""

    def test_a_default_portal_round_trips_through_model_dump(self):
        original = Portal(to=["b"])
        rebuilt = Portal.model_validate(original.model_dump())
        assert rebuilt == original, "a Portal must survive its own default serialization"

    def test_the_round_trip_does_not_need_exclude_unset(self):
        """The unwritten invariant this removes: every dump had to pass
        exclude_unset=True or the payload could not be loaded back."""
        payload = Portal(to=["b"]).model_dump()
        assert "max_depth" in payload and payload["max_depth"] is None
        Portal.model_validate(payload)  # must not raise


class TestUnsetKnobsStayUnset:
    """Intent survives as a value: unset reads as None, explicit reads as itself."""

    def test_unset_knobs_are_none_not_materialized_defaults(self):
        p = Portal(to=["b"])
        assert p.max_hops is None
        assert p.on_exhaust is None
        assert p.trigger is None

    def test_explicit_values_are_distinguishable_from_unset(self):
        assert Portal(to=["b"], max_hops=10).max_hops == 10
        assert Portal(to=["b"]) != Portal(to=["b"], max_hops=10)

    def test_effective_properties_apply_the_defaults(self):
        p = Portal(to=["b"])
        assert p.effective_max_hops == 10
        assert p.effective_on_exhaust == "error"
        assert p.effective_trigger == "output"

    def test_effective_properties_defer_to_an_explicit_value(self):
        p = Portal(to=["b"], max_hops=3, on_exhaust="exit", trigger="tool")
        assert (p.effective_max_hops, p.effective_on_exhaust, p.effective_trigger) == (3, "exit", "tool")

    def test_is_tool_triggered_reads_the_effective_trigger(self):
        """Not `self.trigger == "tool"`: None == "tool" is False, which is the right
        answer BY ACCIDENT -- the read-time seam exists to remove that luck."""
        assert Portal(to=["b"]).is_tool_triggered is False
        assert Portal(to=["b"], trigger="tool").is_tool_triggered is True


class TestModeExclusionsStillHold:
    """The `is not None` rewrite must not weaken any mode rule."""

    def test_peer_mode_still_forbids_max_depth(self):
        with pytest.raises(Exception, match="max_depth"):
            Portal(to=["b"], max_depth=3)

    def test_dispatch_mode_still_forbids_peer_knobs(self):
        with pytest.raises(Exception, match="peer-mode knobs|max_hops"):
            Portal(
                route="decide", spec_field="s", input_field="i", output="o", max_depth=2, max_hops=5
            )
