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


class TestIsEachOracleFused:
    """`is_each_oracle_fused(mods)` is the ONE named answer to "is this item the
    fused Each x Oracle node?" (neograph-c265k).

    It is deliberately a modifier-PRESENCE read over the dict `classify_modifiers`
    returns, NOT a decomposition question and NOT a ModifierCombo enumeration:
    `COMBO_DECOMPOSITION` folds EACH_ORACLE / EACH_ORACLE_OPERATOR to
    `primary=EACH` (the fusion is a Node concern), so a consumer standing in a
    `PrimaryShape.EACH` arm needs this second, orthogonal test.

    The import is FUNCTION-LOCAL on purpose. Before the predicate lands these
    tests fail with `ImportError` (the prescribed TDD-red polarity), and keeping
    the import out of the module header confines that redness to this class
    instead of taking the whole file's collection down with it.
    """

    #: combo -> the raw modifier-name frozenset that classifies to it. Inverted
    #: from `_COMBO_MAP` so the true/false partition below is DERIVED, never
    #: re-typed -- the same anti-tautology move `_expected_primary` makes.
    _COMBO_TO_NAMES = {combo: names for names, combo in _COMBO_MAP.items()}

    @staticmethod
    def _mods_for(names: frozenset[str]) -> dict[str, object]:
        """A `classify_modifiers`-shaped dict: a key ONLY when the slot is set,
        each mapping to a non-None modifier stand-in."""
        return {name: object() for name in names}

    def test_is_true_for_the_two_fused_combos(self) -> None:
        from neograph.modifiers import is_each_oracle_fused

        for combo in (ModifierCombo.EACH_ORACLE, ModifierCombo.EACH_ORACLE_OPERATOR):
            assert is_each_oracle_fused(self._mods_for(self._COMBO_TO_NAMES[combo])) is True, combo

    def test_is_false_for_every_non_fused_combo(self) -> None:
        from neograph.modifiers import is_each_oracle_fused

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
            assert is_each_oracle_fused(self._mods_for(self._COMBO_TO_NAMES[combo])) is False, combo

    def test_is_true_for_exactly_the_combos_carrying_both_modifiers(self) -> None:
        """Totality, derived from `_COMBO_MAP`: over EVERY combo the predicate
        agrees with "the raw modifier names contain both each and oracle".

        This is the load-bearing test -- the two above are readable spot checks
        of the same partition. Written as a derivation so a new combo added to
        `_COMBO_MAP` is covered automatically and the pair is never re-typed.
        """
        from neograph.modifiers import is_each_oracle_fused

        fused = {combo for combo, names in self._COMBO_TO_NAMES.items() if is_each_oracle_fused(self._mods_for(names))}
        expected = {combo for combo, names in self._COMBO_TO_NAMES.items() if {"each", "oracle"} <= names}
        assert fused == expected
        # ...and the derivation is not vacuous: the partition is a real split.
        assert fused and fused != set(ModifierCombo)

    def test_is_false_when_a_slot_key_carries_none(self) -> None:
        """R-RL2: the body is spelled `mods.get(name) is not None`, not
        `name in mods`. Under `classify_modifiers`'s key-only-when-set contract
        the two are equivalent; for any OTHER dict (a serialized node spec, a
        hand-built one) the `.get` form stays correct where membership silently
        reports a fusion that is not there.
        """
        from neograph.modifiers import is_each_oracle_fused

        assert is_each_oracle_fused({"each": None, "oracle": object()}) is False
        assert is_each_oracle_fused({"each": object(), "oracle": None}) is False
        assert is_each_oracle_fused({}) is False

    def test_agrees_with_classify_modifiers_on_a_real_fused_node(self) -> None:
        """End-to-end through the PROGRAMMATIC surface: a genuinely fused Node
        built with `Node.scripted(...) | Oracle(...) | Each(...)`, classified by
        the real `classify_modifiers`, and handed to the predicate -- not a
        hand-rolled dict. Pins that the predicate reads the shape the compiler
        actually meets, and that it agrees with the table's `primary=EACH` fold.
        """
        from neograph.modifiers import classify_modifiers, is_each_oracle_fused
        from neograph.node import Node
        from tests.fakes import register_scripted
        from tests.schemas import Claims

        register_scripted("c265k_gen", lambda v, c: Claims(items=["a"]))
        register_scripted("c265k_merge", lambda v, c: v[0])

        fused = (
            Node.scripted("c265k_fused", fn="c265k_gen", outputs=Claims)
            | Oracle(n=2, merge_fn="c265k_merge")
            | Each(over="x.y", key="k")
        )
        combo, mods = classify_modifiers(fused)
        assert combo is ModifierCombo.EACH_ORACLE
        assert COMBO_DECOMPOSITION[combo].primary is PrimaryShape.EACH  # the fold that forces the 2nd test
        assert is_each_oracle_fused(mods) is True

        plain_each = Node.scripted("c265k_plain", fn="c265k_gen", outputs=Claims) | Each(over="x.y", key="k")
        plain_combo, plain_mods = classify_modifiers(plain_each)
        assert COMBO_DECOMPOSITION[plain_combo].primary is PrimaryShape.EACH  # SAME primary shape...
        assert is_each_oracle_fused(plain_mods) is False  # ...and the predicate still splits them
