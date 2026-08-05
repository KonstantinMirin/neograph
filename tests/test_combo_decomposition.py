"""Tests for the shared ModifierCombo decomposition table (Phase 0).

`COMBO_DECOMPOSITION` / `PrimaryShape` / `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` live
in `modifiers.py` next to `_COMBO_MAP` as the single source of truth for combo
*meaning* (decomposition), complementing `_COMBO_MAP`'s single source of truth
for combo *classification*. These tests pin the table's totality, internal
consistency, and agreement with `_COMBO_MAP`'s modifier semantics. They are the
contract its consumers (compiler.py, state.py, _agent_spec.py, loader.py, ...)
read the table under -- including the `fused` column, which is the sole authority
on Each x Oracle fusion (neograph-jtawq.2).
"""

from __future__ import annotations

from neograph.modifiers import (
    _COMBO_MAP,
    COMBO_DECOMPOSITION,
    SUB_CONSTRUCT_UNSUPPORTED_COMBOS,
    ComboDecomposition,
    Each,
    ModifierCombo,
    Oracle,
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
            assert isinstance(decomp.fused, bool)

    def test_has_operator_is_true_exactly_for_operator_combos(self) -> None:
        for combo, decomp in COMBO_DECOMPOSITION.items():
            names_the_operator = "OPERATOR" in combo.name.split("_")
            assert decomp.has_operator == names_the_operator, combo

    def test_fused_is_true_exactly_for_the_each_oracle_combos(self) -> None:
        """G-FUSE: the `fused` column partitions ModifierCombo exactly on
        Each x Oracle co-membership.

        The expectation is derived from the enum NAME -- the same independent
        oracle `test_has_operator_is_true_exactly_for_operator_combos` uses --
        and deliberately NOT from `modifier_names_for_combo` / `_COMBO_MAP`,
        which is what production derives the column from. Deriving from the
        production source would be X == X and could never catch a drift.
        """
        for combo, decomp in COMBO_DECOMPOSITION.items():
            names_both = {"EACH", "ORACLE"} <= set(combo.name.split("_"))
            assert decomp.fused == names_both, combo

    def test_the_fused_partition_is_a_real_split(self) -> None:
        # Non-vacuity for the derivation above: neither side of the partition
        # may be empty, or a broken column would satisfy it trivially.
        fused = {combo for combo, decomp in COMBO_DECOMPOSITION.items() if decomp.fused}
        assert fused
        assert fused != set(ModifierCombo)

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

    def test_equals_the_fused_combos_as_an_intentional_coincidence(self) -> None:
        """The unsupported set and the `fused` column are two DIFFERENT concepts
        that happen to share an extension today.

        "no defined lowering when this combo is attached to a Construct item" is
        not "carries both Each and Oracle" -- neograph-rh5fb (Loop on a Construct
        with differing boundary types) is a candidate member that is not a fusion.
        So `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` stays HAND-WRITTEN and is never
        derived from `fused`; this pin exists so the day the two concepts diverge
        fails LOUD here instead of silently redefining one as the other.
        """
        assert SUB_CONSTRUCT_UNSUPPORTED_COMBOS == frozenset(
            combo for combo, decomp in COMBO_DECOMPOSITION.items() if decomp.fused
        )


class TestFusedColumn:
    """`COMBO_DECOMPOSITION[combo].fused` is the ONE answer to "is this the fused
    Each x Oracle node?" (neograph-jtawq.2, superseding the free-floating
    `is_each_oracle_fused(mods)` predicate of neograph-c265k).

    Fusion is a fact about the COMBO, not about which Each/Oracle INSTANCES are
    attached: EACH_ORACLE is fused whichever Each and Oracle it carries. It stays
    a SECOND, orthogonal question because the table folds EACH_ORACLE /
    EACH_ORACLE_OPERATOR to `primary=EACH` (the fusion is a Node-level topology
    concern), so a consumer standing in a `PrimaryShape.EACH` arm still needs it
    to split a fused node from a plain Each one.

    The exhaustive true/false partition lives in `TestComboDecompositionTable`
    (G-FUSE), derived from the enum name. The tests here are the readable spot
    checks plus the end-to-end proof through a real classified Node.

    Gone with the predicate: its `mods.get(name) is not None` slot-polarity test
    (R-RL2). A column keyed by ModifierCombo has no `mods` dict to be handed a
    None-valued key, so the failure mode it guarded cannot occur.
    """

    def test_is_true_for_the_two_fused_combos(self) -> None:
        for combo in (ModifierCombo.EACH_ORACLE, ModifierCombo.EACH_ORACLE_OPERATOR):
            assert COMBO_DECOMPOSITION[combo].fused is True, combo

    def test_is_false_for_every_non_fused_combo(self) -> None:
        for combo in (
            ModifierCombo.BARE,
            ModifierCombo.EACH,
            ModifierCombo.EACH_OPERATOR,
            ModifierCombo.ORACLE,
            ModifierCombo.ORACLE_OPERATOR,
            ModifierCombo.LOOP,
            ModifierCombo.LOOP_OPERATOR,
            ModifierCombo.OPERATOR,
            ModifierCombo.PORTAL,
            ModifierCombo.PORTAL_OPERATOR,
        ):
            assert COMBO_DECOMPOSITION[combo].fused is False, combo

    def test_splits_a_real_fused_node_from_a_real_plain_each_node(self) -> None:
        """End-to-end through the PROGRAMMATIC surface: a genuinely fused Node
        built with `Node.scripted(...) | Oracle(...) | Each(...)`, classified by
        the real `classify_modifiers`, then asked through the table -- not a
        hand-rolled dict. Pins that the column answers for the shape the compiler
        actually meets, and that it still splits two nodes the `primary=EACH`
        fold makes indistinguishable.
        """
        from neograph.modifiers import classify_modifiers
        from neograph.node import Node
        from tests.fakes import register_scripted
        from tests.schemas import Claims

        register_scripted("jtawq2_gen", lambda v, c: Claims(items=["a"]))
        register_scripted("jtawq2_merge", lambda v, c: v[0])

        fused = (
            Node.scripted("jtawq2_fused", fn="jtawq2_gen", outputs=Claims)
            | Oracle(n=2, merge_fn="jtawq2_merge")
            | Each(over="x.y", key="k")
        )
        combo, _mods = classify_modifiers(fused)
        assert combo is ModifierCombo.EACH_ORACLE
        assert COMBO_DECOMPOSITION[combo].primary is PrimaryShape.EACH  # the fold that forces the 2nd question
        assert COMBO_DECOMPOSITION[combo].fused is True

        plain_each = Node.scripted("jtawq2_plain", fn="jtawq2_gen", outputs=Claims) | Each(over="x.y", key="k")
        plain_combo, _plain_mods = classify_modifiers(plain_each)
        assert COMBO_DECOMPOSITION[plain_combo].primary is PrimaryShape.EACH  # SAME primary shape...
        assert COMBO_DECOMPOSITION[plain_combo].fused is False  # ...and the column still splits them
