"""Structural guard for the Portal addressability design doc (neograph-ozrct).

neograph-ozrct's deliverable is `docs/design/portal-addressability-2026-07-15.md`
-- a design doc, not production code. It produces no runtime-observable
behavior, so per the test-author protocol's structural-disease exception this
guard pins the doc's required STRUCTURE instead of a behavioral contract: the
file must exist at the expected path and contain the specific structural
elements the implementation plan (see `bd show neograph-ozrct`) mandates --
the three-class taxonomy names, the Core Invariant statement, both named
mechanisms, the D-LOWERING-DISSENT acknowledgement added in the refine step
(neograph-u7gy8.28, addressing the architect review's MEDIUM finding), and
citations to the decision log this doc must not re-derive.

This is TDD red: the doc does not exist yet (neograph-u7gy8.7 -- Implement --
is still OPEN), so every test in this module currently FAILS.
"""

from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "docs" / "design" / "portal-addressability-2026-07-15.md"


@pytest.fixture()
def doc_text() -> str:
    if not DOC_PATH.exists():
        pytest.fail(f"design doc not found at {DOC_PATH.relative_to(REPO_ROOT)}")
    return DOC_PATH.read_text(encoding="utf-8")


class TestPortalAddressabilityDocExists:
    """The doc must exist at the exact path the atom prescribes."""

    def test_doc_file_exists_at_expected_path(self):
        assert DOC_PATH.exists(), (
            f"expected design doc at {DOC_PATH.relative_to(REPO_ROOT)} -- "
            "not yet written (neograph-ozrct implementation still open)"
        )


class TestPortalAddressabilityDocTaxonomy:
    """The three-class addressability taxonomy must be named explicitly."""

    def test_doc_names_atomic_class(self, doc_text: str):
        assert "ATOMIC" in doc_text

    def test_doc_names_port_bearing_region_class(self, doc_text: str):
        assert "PORT-BEARING REGION" in doc_text

    def test_doc_names_portless_region_class(self, doc_text: str):
        assert "PORTLESS REGION" in doc_text


class TestPortalAddressabilityDocCoreInvariant:
    """The Core Invariant statement (routing operates on boundary ports) must be present."""

    def test_doc_contains_core_invariant_heading(self, doc_text: str):
        assert "Core Invariant" in doc_text

    def test_doc_states_boundary_ports_invariant(self, doc_text: str):
        assert "BOUNDARY PORTS" in doc_text or "boundary port" in doc_text.lower()


class TestPortalAddressabilityDocMechanisms:
    """Both named mechanisms unlocking port-bearing regions must appear."""

    def test_doc_names_entry_label_map_mechanism(self, doc_text: str):
        assert "entry-label map" in doc_text

    def test_doc_names_mesh_transparent_exit_mechanism(self, doc_text: str):
        assert "mesh-transparent exit" in doc_text


class TestPortalAddressabilityDocLoweringDissent:
    """The D-LOWERING-DISSENT acknowledgement added in the refine step (u7gy8.28)
    must be present -- the architect review's MEDIUM finding required a paragraph
    stating the taxonomy is lowering-substrate-agnostic."""

    def test_doc_cites_lowering_dissent_by_name(self, doc_text: str):
        assert "D-LOWERING-DISSENT" in doc_text

    def test_doc_states_lowering_substrate_agnostic(self, doc_text: str):
        assert "lowering-substrate-agnostic" in doc_text.lower().replace(
            "lowering substrate agnostic", "lowering-substrate-agnostic"
        )


class TestPortalAddressabilityDocDecisionLogCitations:
    """The doc must cite (not re-derive) the decision log's shipped entries."""

    def test_doc_cites_decision_log_file(self, doc_text: str):
        assert "keymaker-decision-log-2026-07-13.md" in doc_text

    def test_doc_cites_d_mesh_level(self, doc_text: str):
        assert "D-MESH-LEVEL" in doc_text

    def test_doc_cites_d_member_modes(self, doc_text: str):
        assert "D-MEMBER-MODES" in doc_text

    def test_doc_cites_d_forward_exempt(self, doc_text: str):
        assert "D-FORWARD-EXEMPT" in doc_text
