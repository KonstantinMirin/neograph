"""Modifier tests — degenerate patterns, warnings, Loop validation, ModifierSet"""

from __future__ import annotations

import pytest

from neograph import (
    ConfigurationError,
    Construct,
    ConstructError,
    Each,
    Loop,
    Node,
    Operator,
    Oracle,
    compile,
    node,
    run,
)
from neograph.factory import register_scripted
from tests.schemas import (
    Claims,
    _producer,
)

# ═══════════════════════════════════════════════════════════════════════════
# DEV-MODE WARNINGS
#
# NEOGRAPH_DEV=1 emits warnings for ambiguous-but-valid patterns.
# ═══════════════════════════════════════════════════════════════════════════


class TestDegeneratePatterns:
    """Degenerate patterns: Oracle(n=1), Each with 1 item (neograph-gg3h)."""

    def test_oracle_n1_programmatic_compiles_and_runs(self):
        """Oracle(n=1) via programmatic API — degenerate ensemble, 1 gen + merge."""

        register_scripted("deg_gen", lambda i, c: Claims(items=["v1"]))
        register_scripted("deg_merge", lambda v, c: v[0])

        pipeline = Construct("deg-oracle", nodes=[
            Node.scripted("gen", fn="deg_gen", outputs=Claims)
            | Oracle(n=1, merge_fn="deg_merge"),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "deg-o1"})
        assert result["gen"].items == ["v1"]

    def test_each_single_item_collection(self):
        """Each with exactly 1 item — degenerate fan-out, still produces dict."""
        from pydantic import BaseModel


        class Item(BaseModel, frozen=True):
            key: str

        class Batch(BaseModel, frozen=True):
            items: list[Item]

        class Result(BaseModel, frozen=True):
            label: str

        register_scripted("deg_batch", lambda i, c: Batch(items=[Item(key="only")]))
        register_scripted("deg_proc", lambda i, c: Result(label="processed"))

        pipeline = Construct("deg-each", nodes=[
            Node.scripted("batch", fn="deg_batch", outputs=Batch),
            Node.scripted("proc", fn="deg_proc", inputs=Item, outputs=Result)
            | Each(over="batch.items", key="key"),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "deg-e1"})
        assert isinstance(result["proc"], dict)
        assert set(result["proc"].keys()) == {"only"}

    def test_each_empty_collection(self):
        """Each with 0 items — degenerate fan-out, produces empty dict."""
        from pydantic import BaseModel


        class Item(BaseModel, frozen=True):
            key: str

        class Batch(BaseModel, frozen=True):
            items: list[Item]

        class Result(BaseModel, frozen=True):
            label: str

        register_scripted("deg0_batch", lambda i, c: Batch(items=[]))
        register_scripted("deg0_proc", lambda i, c: Result(label="x"))

        pipeline = Construct("deg-each0", nodes=[
            Node.scripted("batch", fn="deg0_batch", outputs=Batch),
            Node.scripted("proc", fn="deg0_proc", inputs=Item, outputs=Result)
            | Each(over="batch.items", key="key"),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "deg-e0"})
        # Empty collection → empty dict (or None depending on reducer)
        proc = result.get("proc")
        assert proc is None or proc == {}

    def test_oracle_n1_decorator_raises(self):
        """@node(ensemble_n=1) raises ConstructError — must be >= 2."""
        with pytest.raises(ConstructError, match="ensemble_n must be >= 2"):
            @node(outputs=Claims, prompt="test", model="fast",
                  ensemble_n=1, merge_fn="dummy")
            def bad() -> Claims: ...

    def test_each_item_failure_crashes_entire_batch(self):
        """neograph-spz1: one failing item in Each kills all other items.
        This test DOCUMENTS the current (broken) behavior. When fixed,
        this test should be updated to expect partial results."""
        from pydantic import BaseModel


        class Item(BaseModel, frozen=True):
            key: str

        class Batch(BaseModel, frozen=True):
            items: list[Item]

        class Result(BaseModel, frozen=True):
            label: str

        register_scripted("spz1_batch", lambda i, c: Batch(items=[
            Item(key="good1"), Item(key="bad"), Item(key="good2"),
        ]))

        def failing_proc(input_data, config):
            if hasattr(input_data, "key") and input_data.key == "bad":
                raise RuntimeError("Simulated API error for item 'bad'")
            return Result(label=f"ok:{input_data.key}")

        register_scripted("spz1_proc", failing_proc)

        pipeline = Construct("spz1-test", nodes=[
            Node.scripted("batch", fn="spz1_batch", outputs=Batch),
            Node.scripted("proc", fn="spz1_proc", inputs=Item, outputs=Result)
            | Each(over="batch.items", key="key"),
        ])
        graph = compile(pipeline)

        # Current behavior: entire graph crashes due to one bad item.
        # When neograph-spz1 is fixed, this should return partial results
        # with good1 and good2 succeeded, bad marked as failed.
        with pytest.raises(RuntimeError, match="Simulated API error"):
            run(graph, input={"node_id": "spz1"})





class TestDevWarnings:
    """Dev-mode warnings for ambiguous modifier patterns."""

    def test_oracle_n1_warns_in_dev_mode(self, monkeypatch):
        """Oracle(n=1) emits a dev warning about single-element ensemble."""
        import warnings

        import neograph._dev_warnings as dw

        monkeypatch.setattr(dw, "DEV_MODE", True)

        gen = _producer("gen", Claims)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gen | Oracle(n=1, merge_fn="dummy_merge")

        dev_msgs = [w for w in caught if "[neograph-dev]" in str(w.message)]
        assert len(dev_msgs) == 1
        assert "ensemble of 1" in str(dev_msgs[0].message)

    def test_oracle_n1_silent_without_dev_mode(self, monkeypatch):
        """Oracle(n=1) does NOT warn when dev mode is off."""
        import warnings

        import neograph._dev_warnings as dw

        monkeypatch.setattr(dw, "DEV_MODE", False)

        gen = _producer("gen", Claims)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gen | Oracle(n=1, merge_fn="dummy_merge")

        dev_msgs = [w for w in caught if "[neograph-dev]" in str(w.message)]
        assert len(dev_msgs) == 0

    def test_oracle_uneven_models_warns_in_dev_mode(self, monkeypatch):
        """Oracle with n not divisible by len(models) warns about uneven distribution."""
        import warnings

        import neograph._dev_warnings as dw

        monkeypatch.setattr(dw, "DEV_MODE", True)

        gen = _producer("gen", Claims)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gen | Oracle(n=5, models=["a", "b"], merge_fn="dummy_merge")

        dev_msgs = [w for w in caught if "[neograph-dev]" in str(w.message)]
        assert any("uneven distribution" in str(w.message) for w in dev_msgs)

    def test_loop_max_iterations_1_warns_in_dev_mode(self, monkeypatch):
        """Loop(max_iterations=1) warns about effectively non-looping config."""
        import warnings

        import neograph._dev_warnings as dw

        monkeypatch.setattr(dw, "DEV_MODE", True)

        n = Node.scripted("looper", fn="noop_loop", inputs=Claims, outputs=Claims)
        register_scripted("noop_loop", lambda data, config: data)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            n | Loop(when=lambda d: False, max_iterations=1)

        dev_msgs = [w for w in caught if "[neograph-dev]" in str(w.message)]
        assert len(dev_msgs) == 1
        assert "max_iterations=1" in str(dev_msgs[0].message)


# =============================================================================
# Coverage gap tests for modifiers.py
# =============================================================================





# =============================================================================
# Coverage gap tests for modifiers.py
# =============================================================================


class TestEachLoopReverseOrder:
    """Lines 86-90: Each applied after Loop is also rejected."""

    def test_each_after_loop_raises(self):
        """Applying Each to a node that already has Loop raises ConstructError."""

        register_scripted("el_fn", lambda data, config: data)
        n = Node.scripted("el", fn="el_fn", inputs=Claims, outputs=Claims)
        n_with_loop = n | Loop(when=lambda d: False, max_iterations=3)

        with pytest.raises(ConstructError, match="Cannot combine Each and Loop"):
            n_with_loop | Each(over="upstream.items", key="label")





class TestLoopMaxIterationsValidation:
    """Lines 366-367: Loop(max_iterations<1) raises ConfigurationError."""

    def test_max_iterations_zero_raises(self):
        """Loop with max_iterations=0 raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="max_iterations must be >= 1"):
            Loop(when=lambda d: True, max_iterations=0)

    def test_max_iterations_negative_raises(self):
        """Loop with max_iterations=-1 raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="max_iterations must be >= 1"):
            Loop(when=lambda d: True, max_iterations=-1)





class TestModifierSet:
    """ModifierSet typed slots — illegal combos are structurally unrepresentable (neograph-v5c1)."""

    def test_bare_modifier_set_has_no_modifiers(self):
        """Empty ModifierSet has BARE combo and empty list."""
        from neograph.modifiers import ModifierCombo, ModifierSet
        ms = ModifierSet()
        assert ms.combo == ModifierCombo.BARE
        assert ms.to_list() == []

    def test_each_only(self):
        """ModifierSet with Each only has EACH combo."""
        from neograph.modifiers import ModifierCombo, ModifierSet
        each = Each(over="x.y", key="k")
        ms = ModifierSet(each=each)
        assert ms.combo == ModifierCombo.EACH
        assert ms.each is each
        assert ms.oracle is None

    def test_oracle_only(self):
        """ModifierSet with Oracle only has ORACLE combo."""
        from neograph.modifiers import ModifierCombo, ModifierSet
        oracle = Oracle(n=3, merge_fn="m")
        ms = ModifierSet(oracle=oracle)
        assert ms.combo == ModifierCombo.ORACLE

    def test_each_oracle_fusion(self):
        """Each + Oracle is valid (M x N fusion)."""
        from neograph.modifiers import ModifierCombo, ModifierSet
        ms = ModifierSet(
            each=Each(over="x.y", key="k"),
            oracle=Oracle(n=3, merge_fn="m"),
        )
        assert ms.combo == ModifierCombo.EACH_ORACLE

    def test_each_loop_rejected_at_construction(self):
        """ModifierSet(each=..., loop=...) is rejected by model_post_init."""
        from neograph.modifiers import ModifierSet
        with pytest.raises(Exception, match="Cannot combine Each and Loop"):
            ModifierSet(
                each=Each(over="x.y", key="k"),
                loop=Loop(when=lambda d: False, max_iterations=1),
            )

    def test_oracle_loop_rejected_at_construction(self):
        """ModifierSet(oracle=..., loop=...) is rejected by model_post_init."""
        from neograph.modifiers import ModifierSet
        with pytest.raises(Exception, match="Cannot combine Oracle and Loop"):
            ModifierSet(
                oracle=Oracle(n=3, merge_fn="m"),
                loop=Loop(when=lambda d: False, max_iterations=1),
            )

    def test_with_modifier_each(self):
        """with_modifier adds Each to an empty set."""
        from neograph.modifiers import ModifierCombo, ModifierSet
        ms = ModifierSet()
        ms2 = ms.with_modifier(Each(over="x.y", key="k"))
        assert ms2.combo == ModifierCombo.EACH
        assert ms.combo == ModifierCombo.BARE  # original unchanged (frozen)

    def test_with_modifier_duplicate_each_rejected(self):
        """with_modifier rejects duplicate Each."""
        from neograph.modifiers import ModifierSet
        ms = ModifierSet(each=Each(over="x.y", key="k"))
        with pytest.raises(ConstructError, match="Duplicate Each"):
            ms.with_modifier(Each(over="a.b", key="c"))

    def test_with_modifier_duplicate_oracle_rejected(self):
        """with_modifier rejects duplicate Oracle."""
        from neograph.modifiers import ModifierSet
        ms = ModifierSet(oracle=Oracle(n=3, merge_fn="m"))
        with pytest.raises(ConstructError, match="Duplicate Oracle"):
            ms.with_modifier(Oracle(n=2, merge_fn="n"))

    def test_with_modifier_duplicate_loop_rejected(self):
        """with_modifier rejects duplicate Loop."""
        from neograph.modifiers import ModifierSet
        ms = ModifierSet(loop=Loop(when=lambda d: False, max_iterations=1))
        with pytest.raises(ConstructError, match="Duplicate Loop"):
            ms.with_modifier(Loop(when=lambda d: True, max_iterations=2))

    def test_with_modifier_duplicate_operator_rejected(self):
        """with_modifier rejects duplicate Operator."""
        from neograph.modifiers import ModifierSet
        ms = ModifierSet(operator=Operator(when="check"))
        with pytest.raises(ConstructError, match="Duplicate Operator"):
            ms.with_modifier(Operator(when="other"))

    def test_with_modifier_each_loop_rejected(self):
        """with_modifier rejects Each when Loop is present."""
        from neograph.modifiers import ModifierSet
        ms = ModifierSet(loop=Loop(when=lambda d: False, max_iterations=1))
        with pytest.raises(ConstructError, match="Cannot combine Each and Loop"):
            ms.with_modifier(Each(over="x.y", key="k"))

    def test_with_modifier_loop_each_rejected(self):
        """with_modifier rejects Loop when Each is present."""
        from neograph.modifiers import ModifierSet
        ms = ModifierSet(each=Each(over="x.y", key="k"))
        with pytest.raises(ConstructError, match="Cannot combine Each and Loop"):
            ms.with_modifier(Loop(when=lambda d: False, max_iterations=1))

    def test_with_modifier_loop_oracle_rejected(self):
        """with_modifier rejects Loop when Oracle is present."""
        from neograph.modifiers import ModifierSet
        ms = ModifierSet(oracle=Oracle(n=3, merge_fn="m"))
        with pytest.raises(ConstructError, match="Cannot combine Oracle and Loop"):
            ms.with_modifier(Loop(when=lambda d: False, max_iterations=1))

    def test_with_modifier_oracle_loop_rejected(self):
        """with_modifier rejects Oracle when Loop is present."""
        from neograph.modifiers import ModifierSet
        ms = ModifierSet(loop=Loop(when=lambda d: False, max_iterations=1))
        with pytest.raises(ConstructError, match="Cannot combine Oracle and Loop"):
            ms.with_modifier(Oracle(n=3, merge_fn="m"))

    def test_pipe_syntax_produces_correct_modifier_set(self):
        """node | Oracle() | Each() produces a ModifierSet with both slots filled."""
        from neograph.modifiers import ModifierCombo
        n = Node.scripted("proc", fn="noop", inputs=Claims, outputs=Claims)
        n2 = n | Oracle(n=3, merge_fn="m") | Each(over="x.y", key="k")
        assert n2.modifier_set.combo == ModifierCombo.EACH_ORACLE
        assert isinstance(n2.modifier_set.oracle, Oracle)
        assert isinstance(n2.modifier_set.each, Each)

    def test_map_produces_correct_modifier_set(self):
        """.map() sets modifier_set.each."""
        from neograph.modifiers import ModifierCombo
        n = Node.scripted("proc", fn="noop", inputs=Claims, outputs=Claims)
        n2 = n.map("upstream.items", key="label")
        assert n2.modifier_set.combo == ModifierCombo.EACH
        assert isinstance(n2.modifier_set.each, Each)
        assert n2.modifier_set.each.over == "upstream.items"
        assert n2.modifier_set.each.key == "label"

    def test_node_modifiers_list_rejects_with_clear_error(self):
        """Node(modifiers=[...]) raises ConstructError (legacy API removed)."""
        with pytest.raises(ConstructError, match="no longer supported"):
            Node("bad", modifiers=[Each(over="x", key="k")])

    def test_to_list_preserves_modifier_instances(self):
        """to_list returns the exact modifier instances from the slots."""
        from neograph.modifiers import ModifierSet
        each = Each(over="x.y", key="k")
        oracle = Oracle(n=3, merge_fn="m")
        ms = ModifierSet(each=each, oracle=oracle)
        lst = ms.to_list()
        assert each in lst
        assert oracle in lst
        assert len(lst) == 2

    def test_modifiers_property_backward_compat(self):
        """Node.modifiers property returns modifier_set.to_list()."""
        n = Node.scripted("proc", fn="noop", inputs=Claims, outputs=Claims)
        n2 = n | Oracle(n=3, merge_fn="m")
        assert len(n2.modifiers) == 1
        assert isinstance(n2.modifiers[0], Oracle)
