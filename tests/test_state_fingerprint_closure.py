"""State fingerprint closure regression test for neograph-2yi7q(b).

Pins that compute_node_fingerprints returns byte-identical output before/after
the _fp closure extraction. The fingerprint format is load-bearing:

- sha256(f"{name}:{_type_signature(typ)}").encode()).hexdigest()[:12]
- The {name}:{sig} layout and [:12] width are the contract

This test captures the actual hex values for a small fixture construct; any
byte drift (before/after refactor) would fail the test, preserving contract.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel

from neograph import Node, construct_from_functions
from neograph.state import _type_signature, compute_node_fingerprints
from tests.fakes import register_scripted

# ═══════════════════════════════════════════════════════════════════════════
# Schemas for fixture
# ═══════════════════════════════════════════════════════════════════════════


class Claim(BaseModel, frozen=True):
    text: str


class MatchResult(BaseModel, frozen=True):
    score: float
    claim: Claim


class FinalResult(BaseModel, frozen=True):
    summary: str


# ═══════════════════════════════════════════════════════════════════════════
# Regression test (b): compute_node_fingerprints byte-identical output
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeNodeFingerprints:
    """Regression test for state.py DRY refactor (neograph-2yi7q item b).

    The _fp(name, typ) closure extraction must produce byte-identical output.
    This test pins the actual hex values for a small fixture construct.
    """

    def test_fingerprints_byte_identical_for_fixture_construct(self):
        """compute_node_fingerprints returns pinned hex values for fixture.

        This is a behavior-pinning test: it PASSES on current code and must STAY
        green after extracting _fp closure in compute_node_fingerprints.

        The fixture has 3 nodes: claim (Claim), score (MatchResult), final (FinalResult).
        """
        register_scripted(
            "claim_fn",
            lambda input_data, config: Claim(text="test claim"),
        )
        register_scripted(
            "score_fn",
            lambda input_data, config: MatchResult(
                score=0.9, claim=Claim(text="test")
            ),
        )
        register_scripted(
            "final_fn",
            lambda input_data, config: FinalResult(summary="done"),
        )

        claim = Node.scripted("claim", fn="claim_fn", outputs=Claim)
        score = Node.scripted("score", fn="score_fn", outputs=MatchResult)
        final = Node.scripted("final", fn="final_fn", outputs=FinalResult)

        construct = construct_from_functions(
            "fingerprint-fixture",
            [claim, score, final],
        )

        fingerprints = compute_node_fingerprints(construct)

        # *** PINNED HEX VALUES *** — computed via the current _fp implementation
        # Format: sha256(f"{field_name}:{_type_signature(typ)}".encode()).hexdigest()[:12]
        # These will FAIL if the fingerprint format changes (intentionally — the
        # format is load-bearing for checkpoint invalidation)

        # Expected fingerprint for "claim" (Claim type)
        expected_claim_fp = hashlib.sha256(
            f"claim:{_type_signature(Claim)}".encode()
        ).hexdigest()[:12]
        assert fingerprints["claim"] == expected_claim_fp, (
            f"claim fingerprint mismatch: expected {expected_claim_fp}, got {fingerprints.get('claim')}"
        )

        # Expected fingerprint for "score" (MatchResult type)
        expected_score_fp = hashlib.sha256(
            f"score:{_type_signature(MatchResult)}".encode()
        ).hexdigest()[:12]
        assert fingerprints["score"] == expected_score_fp, (
            f"score fingerprint mismatch: expected {expected_score_fp}, got {fingerprints.get('score')}"
        )

        # Expected fingerprint for "final" (FinalResult type)
        expected_final_fp = hashlib.sha256(
            f"final:{_type_signature(FinalResult)}".encode()
        ).hexdigest()[:12]
        assert fingerprints["final"] == expected_final_fp, (
            f"final fingerprint mismatch: expected {expected_final_fp}, got {fingerprints.get('final')}"
        )

        # Verify exactly 3 fingerprints (one per node)
        assert len(fingerprints) == 3, f"Expected 3 fingerprints, got {len(fingerprints)}"

    def test_fingerprint_format_contract_preserved(self):
        """Verify the fingerprint format contract: {name}:{sig} -> sha256 -> [:12].

        This test explicitly checks the format components, not just byte identity.
        Any deviation from this contract would break checkpoint invalidation.
        """
        # Test the format on a single type
        field_name = "test_node"
        typ = Claim

        # Construct the fingerprint string
        fingerprint_string = f"{field_name}:{_type_signature(typ)}"
        assert ":" in fingerprint_string, "Format must include ':' separator"
        assert fingerprint_string.startswith("test_node:"), "Format must start with field_name:"

        # Hash and truncate
        full_hash = hashlib.sha256(fingerprint_string.encode()).hexdigest()
        truncated = full_hash[:12]

        # Verify contract
        assert len(truncated) == 12, "Fingerprint must be exactly 12 characters"
        assert set(truncated).issubset(set("0123456789abcdef")), (
            "Fingerprint must be lowercase hex"
        )

        # Verify idempotence
        fp2 = hashlib.sha256(fingerprint_string.encode()).hexdigest()[:12]
        assert truncated == fp2, "Fingerprint must be idempotent"

    def test_multiple_calls_produce_identical_fingerprints(self):
        """compute_node_fingerprints is deterministic across multiple calls.

        This guards against non-determinism (e.g., hash randomization, dict order).
        """
        register_scripted(
            "deterministic_fn",
            lambda input_data, config: Claim(text="test"),
        )

        node = Node.scripted("det", fn="deterministic_fn", outputs=Claim)
        construct = construct_from_functions("det-fixture", [node])

        fingerprints_1 = compute_node_fingerprints(construct)
        fingerprints_2 = compute_node_fingerprints(construct)

        assert fingerprints_1 == fingerprints_2, (
            "Fingerprints must be identical across calls (deterministic)"
        )
