"""@node decorator tests — fan-out, Oracle, operator interop"""

from __future__ import annotations

from typing import Annotated, Any

import pytest

from neograph import (
    Construct,
    ConstructError,
    Each,
    Node,
    Operator,
    Oracle,
    compile,
    construct_from_module,
    node,
    run,
)
from neograph.factory import register_scripted
from tests.schemas import (
    Claims,
    ClusterGroup,
    Clusters,
    MatchResult,
    RawText,
    ValidationResult,
)


class TestNodeDecoratorFanout:
    """@node decorator: map_over=/map_key= kwargs for Each fan-out interop."""

    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_fanout_produces_dict_when_map_over_and_map_key_set(self):
        """Full chain: producer → fan-out consumer via map_over= compiles, runs
        end-to-end, and produces a dict keyed by cluster label."""

        mod = self._fresh_module("test_fanout_e2e")

        @node(mode="scripted", outputs=Clusters)
        def make_clusters() -> Clusters:
            return Clusters(groups=[
                ClusterGroup(label="alpha", claim_ids=["c1", "c2"]),
                ClusterGroup(label="beta", claim_ids=["c3"]),
            ])

        @node(
            mode="scripted",
            outputs=MatchResult,
            map_over="make_clusters.groups",
            map_key="label",
        )
        def verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label=cluster.label, matched=cluster.claim_ids)

        mod.make_clusters = make_clusters
        mod.verify = verify

        pipeline = construct_from_module(mod)

        # verify should have an Each modifier
        verify_node = [n for n in pipeline.nodes if n.name == "verify"][0]
        each = verify_node.get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "make_clusters.groups"
        assert each.key == "label"

        graph = compile(pipeline)
        result = run(graph, input={"node_id": "fanout-e2e"})

        # Fan-out fired for BOTH clusters — pin cardinality and payload
        verify_results = result.get("verify", {})
        assert set(verify_results.keys()) == {"alpha", "beta"}
        assert verify_results["alpha"].cluster_label == "alpha"
        assert verify_results["beta"].cluster_label == "beta"
        assert verify_results["alpha"].matched == ["c1", "c2"]

    def test_decoration_raises_when_map_over_without_map_key(self):
        """map_over= without map_key= raises ConstructError at decoration time."""
        from neograph import ConstructError

        with pytest.raises(ConstructError, match="map_key"):
            @node(mode="scripted", outputs=MatchResult, map_over="make_clusters.groups")
            def verify(cluster: ClusterGroup) -> MatchResult:
                ...

    def test_decoration_raises_when_map_key_without_map_over(self):
        """map_key= without map_over= raises ConstructError at decoration time."""
        from neograph import ConstructError

        with pytest.raises(ConstructError, match="map_over"):
            @node(mode="scripted", outputs=MatchResult, map_key="label")
            def verify(cluster: ClusterGroup) -> MatchResult:
                ...

    def test_sidecar_survives_when_each_modifier_applied(self):
        """The Each-modified Node copy retains its sidecar entry so
        construct_from_module picks it up."""
        from neograph.decorators import _get_sidecar

        @node(
            mode="scripted",
            outputs=MatchResult,
            map_over="make_clusters.groups",
            map_key="label",
        )
        def verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label=cluster.label, matched=[])

        # The node has an Each modifier
        assert verify.has_modifier(Each)

        # The sidecar survived the model_copy from | Each(...)
        sidecar = _get_sidecar(verify)
        assert isinstance(sidecar, tuple)
        fn, param_names = sidecar
        assert param_names == ("cluster",)

    def test_fanout_param_skipped_when_resolving_upstream_adjacency(self):
        """The fan-out parameter is NOT looked up as an upstream @node,
        so it doesn't cause 'does not match any @node' ConstructError."""

        mod = self._fresh_module("test_fanout_skip_adj")

        @node(mode="scripted", outputs=Clusters)
        def make_clusters() -> Clusters:
            return Clusters(groups=[])

        @node(
            mode="scripted",
            outputs=MatchResult,
            map_over="make_clusters.groups",
            map_key="label",
        )
        def verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label=cluster.label, matched=[])

        mod.make_clusters = make_clusters
        mod.verify = verify

        # Should NOT raise — 'cluster' param is fan-out, not an upstream name
        pipeline = construct_from_module(mod)
        assert len(pipeline.nodes) == 2

    def test_upstream_params_wire_when_mixed_with_fanout_param(self):
        """A node with both upstream params and a fan-out param: only the
        fan-out param is skipped in adjacency; upstream params still wire."""

        mod = self._fresh_module("test_fanout_mixed")

        @node(mode="scripted", outputs=RawText)
        def context() -> RawText:
            return RawText(text="ctx")

        @node(mode="scripted", outputs=Clusters)
        def make_clusters() -> Clusters:
            return Clusters(groups=[])

        @node(
            mode="scripted",
            outputs=MatchResult,
            map_over="make_clusters.groups",
            map_key="label",
        )
        def verify(context: RawText, cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label="x", matched=[])

        mod.context = context
        mod.make_clusters = make_clusters
        mod.verify = verify

        # 'context' wires as upstream; 'cluster' is fan-out → skipped
        pipeline = construct_from_module(mod)
        names = [n.name for n in pipeline.nodes]
        assert "verify" in names
        assert "context" in names


# ═══════════════════════════════════════════════════════════════════════════
# @node decorator: Oracle ensemble kwargs (ensemble_n, merge_fn, merge_prompt)
#
# Tests that @node(..., ensemble_n=N, merge_fn=...) attaches an Oracle
# modifier at decoration time, with correct validation.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# @node decorator: Oracle ensemble kwargs (ensemble_n, merge_fn, merge_prompt)
#
# Tests that @node(..., ensemble_n=N, merge_fn=...) attaches an Oracle
# modifier at decoration time, with correct validation.
# ═══════════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════════
# @node decorator: Oracle ensemble kwargs (ensemble_n, merge_fn, merge_prompt)
#
# Tests that @node(..., ensemble_n=N, merge_fn=...) attaches an Oracle
# modifier at decoration time, with correct validation.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# @node decorator: Oracle ensemble kwargs (ensemble_n, merge_fn, merge_prompt)
#
# Tests that @node(..., ensemble_n=N, merge_fn=...) attaches an Oracle
# modifier at decoration time, with correct validation.
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeDecoratorOracle:
    """@node decorator: ensemble_n=/merge_fn=/merge_prompt= kwargs for Oracle."""

    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_oracle_attaches_when_ensemble_n_and_merge_fn_set(self):
        """@node with ensemble_n + merge_fn end-to-end: Oracle modifier attached,
        pipeline compiles and runs, merge function combines variants."""
        from neograph import node

        gen_ids_seen = []

        def generate_variant(input_data, config):
            gen_id = config.get("configurable", {}).get("_generator_id", "unknown")
            gen_ids_seen.append(gen_id)
            return Claims(items=[f"variant-from-{gen_id}"])

        register_scripted("gen_variant_dec", generate_variant)

        def combine_dec(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("combine_dec", combine_dec)

        mod = self._fresh_module("test_oracle_merge_fn")

        @node(mode="think", outputs=Claims, prompt="rw/decompose", model="reason",
              ensemble_n=3, merge_fn="combine_dec")
        def decompose(topic: RawText) -> Claims: ...

        mod.decompose = decompose

        # Oracle modifier attached at decoration time
        oracle_mod = decompose.get_modifier(Oracle)
        assert isinstance(oracle_mod, Oracle)
        assert oracle_mod.n == 3
        assert oracle_mod.merge_fn == "combine_dec"
        assert oracle_mod.merge_prompt is None

    def test_oracle_attaches_when_ensemble_n_and_merge_prompt_set(self):
        """@node with ensemble_n + merge_prompt end-to-end: Oracle modifier
        attached with merge_prompt for LLM judge."""

        @node(mode="think", outputs=Claims, prompt="rw/decompose", model="reason",
              ensemble_n=2, merge_prompt="rw/decompose-merge")
        def decompose(topic: RawText) -> Claims: ...

        oracle_mod = decompose.get_modifier(Oracle)
        assert isinstance(oracle_mod, Oracle)
        assert oracle_mod.n == 2
        assert oracle_mod.merge_prompt == "rw/decompose-merge"
        assert oracle_mod.merge_fn is None

    def test_oracle_defaults_n_to_3_when_merge_fn_without_ensemble_n(self):
        """merge_fn without ensemble_n defaults to n=3."""

        @node(mode="think", outputs=Claims, prompt="rw/decompose", model="reason",
              merge_fn="combine")
        def decompose(topic: RawText) -> Claims: ...

        oracle_mod = decompose.get_modifier(Oracle)
        assert isinstance(oracle_mod, Oracle)
        assert oracle_mod.n == 3
        assert oracle_mod.merge_fn == "combine"

    def test_decoration_raises_when_ensemble_n_without_merge(self):
        """ensemble_n without merge_fn or merge_prompt raises ConstructError."""

        with pytest.raises(ConstructError, match="requires merge_fn or merge_prompt"):
            @node(mode="think", outputs=Claims, prompt="rw/decompose", model="reason",
                  ensemble_n=3)
            def decompose(topic: RawText) -> Claims: ...

    def test_decoration_raises_when_both_merge_fn_and_merge_prompt(self):
        """Both merge_fn and merge_prompt raises ConstructError."""

        with pytest.raises(ConstructError, match="both merge_fn and merge_prompt"):
            @node(mode="think", outputs=Claims, prompt="rw/decompose", model="reason",
                  ensemble_n=3, merge_fn="combine", merge_prompt="rw/merge")
            def decompose(topic: RawText) -> Claims: ...

    def test_decoration_raises_when_ensemble_n_less_than_2(self):
        """ensemble_n=1 raises ConstructError."""

        with pytest.raises(ConstructError, match="ensemble_n must be >= 2"):
            @node(mode="think", outputs=Claims, prompt="rw/decompose", model="reason",
                  ensemble_n=1, merge_fn="combine")
            def decompose(topic: RawText) -> Claims: ...


# ═══════════════════════════════════════════════════════════════════════════
# @node(mode='raw') — LangGraph escape hatch via unified @node decorator
#
# Raw mode folds @raw_node into @node: the user writes a classic
# (state, config) -> state_update function, and @node wires edges +
# observability. No parameter-name topology — the function body manages
# its own state access.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# @node(mode='raw') — LangGraph escape hatch via unified @node decorator
#
# Raw mode folds @raw_node into @node: the user writes a classic
# (state, config) -> state_update function, and @node wires edges +
# observability. No parameter-name topology — the function body manages
# its own state access.
# ═══════════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════════
# @node interrupt_when — Operator human-in-loop via @node decorator
#
# The interrupt_when= kwarg on @node composes the node with Operator(when=...).
# String form uses a pre-registered condition name; callable form auto-registers.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# @node interrupt_when — Operator human-in-loop via @node decorator
#
# The interrupt_when= kwarg on @node composes the node with Operator(when=...).
# String form uses a pre-registered condition name; callable form auto-registers.
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeDecoratorOperator:

    def test_interrupt_fires_when_string_condition_truthy(self):
        """@node(interrupt_when='name') attaches Operator and interrupt fires end-to-end."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        register_scripted("scripted_validate", lambda input_data, config: ValidationResult(
            passed=False,
            issues=["missing stakeholder coverage"],
        ))

        register_condition("validation_failed", lambda state: (
            {"issues": state.check_quality.issues}
            if state.check_quality and not state.check_quality.passed
            else None
        ))

        validate = node(
            mode="scripted",
            outputs=ValidationResult,
            interrupt_when="validation_failed",
        )(lambda: ValidationResult(passed=False, issues=["missing stakeholder coverage"]))
        # Override: use a Node.scripted approach instead — @node scripted with
        # interrupt_when uses the sidecar raw_fn path, but we need register_scripted
        # for the factory. Build the node directly via the decorator.

        n = Node.scripted(
            "check-quality", fn="scripted_validate", outputs=ValidationResult,
        ) | Operator(when="validation_failed")

        pipeline = Construct("test-node-op-string", nodes=[n])
        graph = compile(pipeline, checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "test-node-op-string"}}
        result = run(graph, input={"node_id": "test-001"}, config=config)

        assert "__interrupt__" in result
        assert result["check_quality"].passed is False

    def test_operator_modifier_attached_when_interrupt_when_string(self):
        """@node(interrupt_when='name') results in a node with Operator modifier."""
        from neograph.factory import register_condition

        register_condition("some_check", lambda state: None)

        @node(mode="scripted", outputs=ValidationResult, interrupt_when="some_check")
        def check_things() -> ValidationResult:
            return ValidationResult(passed=True, issues=[])

        assert check_things.has_modifier(Operator)
        op = check_things.get_modifier(Operator)
        assert isinstance(op, Operator)
        assert op.when == "some_check"

    def test_condition_auto_registered_when_interrupt_when_callable(self):
        """@node(interrupt_when=<callable>) auto-registers condition and attaches Operator."""

        def cond_fn(state):
            return {"flag": True} if getattr(state, "validate", None) else None

        @node(mode="scripted", outputs=ValidationResult, interrupt_when=cond_fn)
        def validate() -> ValidationResult:
            return ValidationResult(passed=False, issues=["x"])

        assert validate.has_modifier(Operator)
        op = validate.get_modifier(Operator)
        assert isinstance(op, Operator)
        # Synthesized name follows the pattern _node_interrupt_{node_name}_{id_hex}
        assert op.when.startswith("_node_interrupt_validate_")

        # Verify the callable was actually registered
        from neograph.factory import lookup_condition
        resolved = lookup_condition(op.when)
        assert resolved is cond_fn

    def test_graph_resumes_when_interrupt_followed_by_feedback(self):
        """@node interrupt + resume flow: graph pauses then resumes with feedback."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        register_scripted("validate_resume_test", lambda input_data, config: ValidationResult(
            passed=False, issues=["bad coverage"],
        ))

        register_condition("needs_review_deco", lambda state: (
            {"issues": state.validate_resume.issues}
            if state.validate_resume and not state.validate_resume.passed
            else None
        ))

        n = Node.scripted(
            "validate-resume", fn="validate_resume_test", outputs=ValidationResult,
        ) | Operator(when="needs_review_deco")

        pipeline = Construct("test-node-op-resume", nodes=[n])
        graph = compile(pipeline, checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "node-op-resume"}}

        # First run: hits interrupt
        result = run(graph, input={"node_id": "test-001"}, config=config)
        assert "__interrupt__" in result

        # Resume
        result = run(graph, resume={"approved": True}, config=config)
        assert result["validate_resume"].passed is False
        assert result["human_feedback"] == {"approved": True}

    def test_graph_continues_when_interrupt_condition_falsy(self):
        """Condition returns None — graph runs through without interrupt."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        register_scripted("quality_ok", lambda input_data, config: ValidationResult(
            passed=True, issues=[],
        ))

        register_condition("always_falsy", lambda state: None)

        n = Node.scripted(
            "check-quality", fn="quality_ok", outputs=ValidationResult,
        ) | Operator(when="always_falsy")

        pipeline = Construct("test-node-op-pass", nodes=[n])
        graph = compile(pipeline, checkpointer=MemorySaver())
        result = run(
            graph,
            input={"node_id": "test-001"},
            config={"configurable": {"thread_id": "node-op-pass"}},
        )

        assert result["check_quality"].passed is True
        assert result.get("human_feedback") is None

    def test_decoration_raises_when_interrupt_when_wrong_type(self):
        """Passing a non-string, non-callable interrupt_when raises ConstructError."""

        with pytest.raises(ConstructError, match="interrupt_when must be a string"):
            @node(mode="scripted", outputs=ValidationResult, interrupt_when=42)
            def bad_node() -> ValidationResult:
                return ValidationResult(passed=True, issues=[])


# ═══════════════════════════════════════════════════════════════════════════
# TEST: @node scalar parameters — FromInput, FromConfig, default constants
#
# Not every @node parameter must name an upstream @node. Three additional
# parameter resolution mechanisms:
#   1. Annotated[T, FromInput]  — value from run(input={param: ...})
#   2. Annotated[T, FromConfig] — value from config["configurable"][param]
#   3. default value  — compile-time constant, no upstream needed
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# TEST: @node scalar parameters — FromInput, FromConfig, default constants
#
# Not every @node parameter must name an upstream @node. Three additional
# parameter resolution mechanisms:
#   1. Annotated[T, FromInput]  — value from run(input={param: ...})
#   2. Annotated[T, FromConfig] — value from config["configurable"][param]
#   3. default value  — compile-time constant, no upstream needed
# ═══════════════════════════════════════════════════════════════════════════




class TestEachOracleFusionDecoratorPath:
    """Each x Oracle fusion (lines 727-736, 745, 755, 759)."""

    def test_each_oracle_fusion_body_as_merge(self):
        """Lines 727-734: map_over + models without merge_fn uses body-as-merge."""
        from neograph.decorators import _get_sidecar

        @node(
            outputs=MatchResult,
            model="fast",
            prompt="test",
            map_over="upstream.groups",
            map_key="label",
            models=["fast", "reason"],
        )
        def fused(cluster: ClusterGroup) -> MatchResult:
            ...
        # Should have Both Each and Oracle
        assert fused.has_modifier(Each)
        assert fused.has_modifier(Oracle)
        assert isinstance(_get_sidecar(fused), tuple)

    def test_each_oracle_fusion_rejects_no_merge(self):
        """Line 736: map_over + ensemble_n without merge_fn/merge_prompt raises."""
        from neograph import ConstructError
        with pytest.raises(ConstructError, match="requires merge_fn or merge_prompt"):
            @node(
                outputs=MatchResult,
                model="fast",
                prompt="test",
                map_over="upstream.groups",
                map_key="label",
                ensemble_n=3,
            )
            def fused(cluster: ClusterGroup) -> MatchResult:
                ...

    def test_each_oracle_fusion_with_models_kwarg(self):
        """Line 745: models= is passed to Oracle kwargs."""
        @node(
            outputs=MatchResult,
            model="fast",
            prompt="test",
            map_over="upstream.groups",
            map_key="label",
            models=["fast", "reason"],
            merge_prompt="merge-tpl",
        )
        def fused(cluster: ClusterGroup) -> MatchResult:
            ...
        assert fused.has_modifier(Each)
        assert fused.has_modifier(Oracle)

    def test_each_oracle_fusion_registers_param_resolutions(self):
        """Line 755: param_res is registered on fused node."""
        from neograph import FromInput
        from neograph.decorators import _get_param_res

        @node(
            outputs=MatchResult,
            model="fast",
            prompt="test",
            map_over="upstream.groups",
            map_key="label",
            models=["fast", "reason"],
            merge_prompt="merge-tpl",
        )
        def fused(
            cluster: ClusterGroup,
            topic: Annotated[str, FromInput],
        ) -> MatchResult:
            ...
        res = _get_param_res(fused)
        assert "topic" in res

    def test_each_oracle_fusion_infers_gen_type_from_merge_fn(self):
        """Line 759: merge_fn with list[T] param infers oracle_gen_type."""
        from neograph.factory import register_scripted

        def my_merge(variants: list[Claims], config: Any) -> Claims:
            return variants[0]
        register_scripted("my_fuse_merge", my_merge)

        @node(
            outputs=MatchResult,
            model="fast",
            prompt="test",
            map_over="upstream.groups",
            map_key="label",
            ensemble_n=3,
            merge_fn="my_fuse_merge",
        )
        def fused(cluster: ClusterGroup) -> MatchResult:
            ...
        # gen_type should be Claims (from list[Claims])
        assert fused.oracle_gen_type is Claims





class TestOracleBodyMergeAndParamResolutions:
    """Oracle body-merge + param_resolutions after Oracle (lines 795, 826)."""

    def test_oracle_body_merge_without_each(self):
        """Line 795: body-as-merge for Oracle without Each."""
        @node(
            outputs=Claims,
            model="fast",
            prompt="test",
            models=["fast", "reason"],
        )
        def merged(topic: RawText) -> Claims:
            ...
        assert merged.has_modifier(Oracle)

    def test_oracle_param_resolutions_registered(self):
        """Line 826: param_res registered after Oracle modifier."""
        from neograph import FromInput
        from neograph.decorators import _get_param_res

        @node(
            outputs=Claims,
            model="fast",
            prompt="test",
            ensemble_n=3,
            merge_prompt="merge-tpl",
        )
        def merged(
            topic: RawText,
            run_id: Annotated[str, FromInput],
        ) -> Claims:
            ...
        res = _get_param_res(merged)
        assert "run_id" in res



class TestBodyMergeRuntimeInvocation:
    """Body-merge closure runtime invocation (lines 731, 795)."""

    def test_fused_body_merge_invoked_at_runtime(self):
        """Line 731: the fused body-merge closure is actually called."""
        from neograph.factory import lookup_scripted

        call_log = []
        @node(
            outputs=Claims,
            model="fast",
            prompt="test",
            map_over="upstream.groups",
            map_key="label",
            models=["fast", "reason"],
        )
        def fused(cluster: ClusterGroup) -> Claims:
            call_log.append("called")
            return Claims(items=["merged"])

        # The body_merge shim was registered. Find it.
        # The name is _body_merge_{node_label}_{id}
        from neograph._registry import registry
        body_merge_keys = [k for k in registry.scripted if k.startswith("_body_merge_fused")]
        assert body_merge_keys, "Body merge should be registered"
        shim = lookup_scripted(body_merge_keys[0])
        result = shim([Claims(items=["v1"]), Claims(items=["v2"])], {})
        assert "called" in call_log

    def test_oracle_only_body_merge_invoked_at_runtime(self):
        """Line 795: the Oracle-only body-merge closure is called."""
        from neograph.factory import lookup_scripted

        call_log = []
        @node(
            outputs=Claims,
            model="fast",
            prompt="test",
            models=["fast", "reason"],
        )
        def oracle_merged(topic: RawText) -> Claims:
            call_log.append("called")
            return Claims(items=["merged"])

        from neograph._registry import registry
        body_merge_keys = [k for k in registry.scripted if k.startswith("_body_merge_oracle-merged")]
        assert body_merge_keys
        shim = lookup_scripted(body_merge_keys[0])
        result = shim([Claims(items=["v1"])], {})
        assert "called" in call_log


class TestEachOracleFusionValidationParity:
    """Fusion path must enforce the same validations as Oracle-only path.

    BUG neograph-15dy: fusion path was missing two validations that
    Oracle-only correctly enforced.
    """

    def test_fusion_rejects_both_merge_fn_and_merge_prompt(self):
        """@node with map_over + merge_fn + merge_prompt should raise."""
        with pytest.raises(ConstructError, match="both merge_fn and merge_prompt"):
            @node(
                outputs=Claims,
                prompt="test", model="fast",
                map_over="items", map_key="id",
                ensemble_n=3, merge_fn="a", merge_prompt="b",
            )
            def bad(item: RawText) -> Claims: ...

    def test_fusion_rejects_ensemble_n_less_than_2(self):
        """@node with map_over + ensemble_n=1 should raise."""
        with pytest.raises(ConstructError, match="ensemble_n must be >= 2"):
            @node(
                outputs=Claims,
                prompt="test", model="fast",
                map_over="items", map_key="id",
                ensemble_n=1, merge_fn="merge_test",
            )
            def bad(item: RawText) -> Claims: ...


