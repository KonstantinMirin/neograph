"""Tests for the shared ModifierCombo decomposition table (Phase 0).

`COMBO_DECOMPOSITION` / `PrimaryShape` / `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` live
in `modifiers.py` next to `_COMBO_MAP` as the single source of truth for combo
*meaning* (decomposition), complementing `_COMBO_MAP`'s single source of truth
for combo *classification*. These tests pin the table's totality, internal
consistency, and agreement with `_COMBO_MAP`'s modifier semantics. At this phase
the table is consumed by nothing yet -- these are the contract for the consumers
(compiler.py, _agent_spec.py, ...) that migrate onto it in later phases.
"""

from __future__ import annotations

from neograph.modifiers import (
    _COMBO_MAP,
    COMBO_DECOMPOSITION,
    SUB_CONSTRUCT_UNSUPPORTED_COMBOS,
    ComboDecomposition,
    ModifierCombo,
    PrimaryShape,
)


def _expected_primary(modifier_names: frozenset[str]) -> PrimaryShape:
    """Derive the primary body-shape a modifier set decomposes to, from the raw
    modifier names -- the same semantics `_COMBO_MAP` classifies by. Precedence
    (portal > each > oracle > loop > bare) matches the compiler's fusion rules:
    Each x Oracle fuses under an EACH-shaped M x N topology, so `each` wins over
    `oracle`; `operator` is orthogonal and never a primary shape.
    """
    if "portal" in modifier_names:
        return PrimaryShape.PORTAL
    if "each" in modifier_names:
        return PrimaryShape.EACH
    if "oracle" in modifier_names:
        return PrimaryShape.ORACLE
    if "loop" in modifier_names:
        return PrimaryShape.LOOP
    return PrimaryShape.BARE


class TestComboDecompositionTable:
    """COMBO_DECOMPOSITION is a total, consistent function over ModifierCombo."""

    def test_is_a_total_function_when_every_combo_has_an_entry(self) -> None:
        assert frozenset(COMBO_DECOMPOSITION) == frozenset(ModifierCombo)

    def test_every_value_is_a_combo_decomposition_when_read(self) -> None:
        for decomp in COMBO_DECOMPOSITION.values():
            assert isinstance(decomp, ComboDecomposition)
            assert isinstance(decomp.primary, PrimaryShape)
            assert isinstance(decomp.has_operator, bool)

    def test_has_operator_is_true_exactly_for_operator_combos(self) -> None:
        for combo, decomp in COMBO_DECOMPOSITION.items():
            names_the_operator = "OPERATOR" in combo.name.split("_")
            assert decomp.has_operator == names_the_operator, combo

    def test_primary_agrees_with_combo_map_modifier_semantics(self) -> None:
        # Invert _COMBO_MAP (combo -> raw modifier-name frozenset) and confirm
        # each decomposition's primary/has_operator matches what those raw
        # modifiers imply -- the two tables must never disagree.
        combo_to_names = {combo: names for names, combo in _COMBO_MAP.items()}
        for combo, decomp in COMBO_DECOMPOSITION.items():
            names = combo_to_names[combo]
            assert decomp.primary == _expected_primary(names), combo
            assert decomp.has_operator == ("operator" in names), combo


class TestPrimaryShape:
    """PrimaryShape enumerates exactly the five body-shapes."""

    def test_members_are_the_five_body_shapes(self) -> None:
        assert {s.name for s in PrimaryShape} == {
            "BARE",
            "EACH",
            "ORACLE",
            "LOOP",
            "PORTAL",
        }

    def test_every_primary_shape_is_reachable_from_the_table(self) -> None:
        reached = {decomp.primary for decomp in COMBO_DECOMPOSITION.values()}
        assert reached == set(PrimaryShape)


class TestSubConstructUnsupportedCombos:
    """The Construct-level restriction set is a small, explicit subset."""

    def test_is_a_subset_of_modifier_combo(self) -> None:
        assert SUB_CONSTRUCT_UNSUPPORTED_COMBOS <= frozenset(ModifierCombo)

    def test_contains_exactly_the_each_oracle_fusion_combos(self) -> None:
        assert SUB_CONSTRUCT_UNSUPPORTED_COMBOS == frozenset(
            {ModifierCombo.EACH_ORACLE, ModifierCombo.EACH_ORACLE_OPERATOR}
        )

    def test_portal_combos_are_excluded(self) -> None:
        # Portal exclusion is deliberate: a Construct Portal mesh member routes
        # through the dedicated mesh path, not the generic Construct-item
        # modifier-check path this frozenset governs -- so it is NOT rejected here.
        assert ModifierCombo.PORTAL not in SUB_CONSTRUCT_UNSUPPORTED_COMBOS
        assert ModifierCombo.PORTAL_OPERATOR not in SUB_CONSTRUCT_UNSUPPORTED_COMBOS

    def test_unsupported_combos_are_each_shaped_fusions_in_the_table(self) -> None:
        # Every unsupported combo is EACH-primary (the fusion shape that has no
        # Construct-level meaning) -- an internal-consistency cross-check.
        for combo in SUB_CONSTRUCT_UNSUPPORTED_COMBOS:
            assert COMBO_DECOMPOSITION[combo].primary == PrimaryShape.EACH
