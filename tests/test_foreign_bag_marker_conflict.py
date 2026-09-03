"""A foreign bag's conflicting markers refuse, instead of first-marker-wins.

Step 8 of neograph-9axw6, design section 10. ``_is_carried`` returned the first
``Carried`` in ``FieldInfo.metadata`` and discarded the rest, so an author who wrote
two paths on one field got whichever pydantic listed first and no indication a
choice had been made -- an answer arriving by POSITION rather than by contract,
which is the shape this epic removes.

A foreign bag is the one place neograph cannot close by construction: user code
populates ``FieldInfo.metadata`` before neograph sees it, so the set cannot be made
unrepresentable the way ``Source`` is. Every ambiguity there is AUTHORED, which is
the kind section 5 says to refuse -- so the bag crosses the boundary once, through a
parser that refuses rather than guesses.
"""

from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import BaseModel

from neograph import Carried
from neograph._output_classify import _is_carried, output_markers
from neograph.errors import ConstructError


class _TwoMarkers(BaseModel):
    field: Annotated[str, Carried("alpha.one"), Carried("beta.two")] = "x"


class _OneMarker(BaseModel):
    field: Annotated[str, Carried("alpha.one")] = "x"


class _NoMarker(BaseModel):
    field: str = "x"


class TestConflictingCarriedMarkers:
    def test_two_markers_on_one_field_refuse_and_name_both(self) -> None:
        with pytest.raises(ConstructError) as exc:
            _is_carried(_TwoMarkers.model_fields["field"], field_label="Two.field")
        msg = str(exc.value)
        assert "alpha.one" in msg and "beta.two" in msg, f"the message must name BOTH paths. Got: {msg}"
        assert "Two.field" in msg, "the message must name the field carrying them"

    def test_the_refusal_reaches_the_shared_predicate(self) -> None:
        """``output_markers`` is what the renderer and the schema projector both call,
        so the refusal has to fire through it -- not only through the private helper."""
        with pytest.raises(ConstructError):
            output_markers(_TwoMarkers.model_fields["field"], field_label="Two.field")


class TestTheRefusalIsNotABlanketBan:
    """Non-vacuity: a rule that refused every marker would pass the test above while
    breaking every working pipeline that uses one."""

    def test_a_single_marker_still_resolves(self) -> None:
        marker = _is_carried(_OneMarker.model_fields["field"])
        assert marker is not None
        assert marker.path == "alpha.one"

    def test_a_field_with_no_marker_is_unaffected(self) -> None:
        assert _is_carried(_NoMarker.model_fields["field"]) is None

    def test_output_markers_still_reports_strip_for_one_marker(self) -> None:
        strip, carried = output_markers(_OneMarker.model_fields["field"])
        assert strip is True
        assert carried is not None and carried.path == "alpha.one"
