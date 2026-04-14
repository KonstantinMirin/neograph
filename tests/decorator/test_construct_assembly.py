"""@node decorator tests — construct_from_functions, construct_from_module, dict inputs/outputs"""

from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import BaseModel

from neograph import (
    Construct,
    ConstructError,
    Node,
    compile,
    construct_from_functions,
    construct_from_module,
    node,
    run,
)
from neograph.factory import register_scripted
from tests.fakes import StructuredFake, configure_fake_llm
from tests.schemas import (
    Claims,
    ClassifiedClaims,
    ClusterGroup,
    Clusters,
    MatchResult,
    MergedResult,
    RawText,
)


class TestNodeDecoratorCrossModule:
    """Cross-module composition and name-collision detection."""

    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_pipeline_assembles_when_node_imported_from_another_module(self):
        """@node from module A imported into module B: construct_from_module(B)
        finds both, wires topology correctly, compile+run end-to-end."""

        # Module A: defines an upstream @node
        mod_a = self._fresh_module("cross_mod_a")

        @node(mode="scripted", outputs=RawText)
        def fetch() -> RawText:
            return RawText(text="fetched data")

        mod_a.fetch = fetch

        # Module B: imports fetch from A, defines a downstream @node
        mod_b = self._fresh_module("cross_mod_b")
        mod_b.fetch = fetch  # simulates `from cross_mod_a import fetch`

        @node(mode="scripted", outputs=Claims)
        def process(fetch: RawText) -> Claims:
            return Claims(items=[fetch.text.upper()])

        mod_b.process = process

        pipeline = construct_from_module(mod_b, name="cross-module")
        assert [n.name for n in pipeline.nodes] == ["fetch", "process"]

        graph = compile(pipeline)
        result = run(graph, input={"node_id": "cross-mod-001"})

        assert result["process"] == Claims(items=["FETCHED DATA"])

    def test_construct_raises_when_node_names_collide(self):
        """Two @node functions with the same fn.__name__ in one module
        raises ConstructError listing both colliding names."""
        from neograph import ConstructError

        mod = self._fresh_module("collision_mod")

        @node(mode="scripted", outputs=RawText)
        def compute() -> RawText:
            return RawText(text="first")

        # Second node: different lambda but explicit name='compute' → same field_name
        second_compute = node(mode="scripted", outputs=Claims, name="compute")(
            lambda: Claims(items=["second"])
        )

        mod.metrics_compute = compute
        mod.stats_compute = second_compute

        with pytest.raises(ConstructError, match="name collision"):
            construct_from_module(mod)

    def test_assembly_succeeds_when_collision_resolved_by_explicit_name(self):
        """Same setup as collision test but one has @node(name='unique') —
        no error, assembly succeeds."""

        mod = self._fresh_module("collision_resolved_mod")

        @node(mode="scripted", outputs=RawText)
        def compute() -> RawText:
            return RawText(text="first")

        # Second node: explicit name= avoids collision
        resolved = node(mode="scripted", outputs=Claims, name="stats_compute")(
            lambda: Claims(items=["second"])
        )

        mod.metrics_compute = compute
        mod.stats_compute = resolved

        pipeline = construct_from_module(mod)
        names = {n.name for n in pipeline.nodes}
        assert "compute" in names
        assert "stats-compute" in names


# ═══════════════════════════════════════════════════════════════════════════
# ForwardConstruct — Task 1: base class + node discovery
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# ForwardConstruct — Task 1: base class + node discovery
# ═══════════════════════════════════════════════════════════════════════════





# ═══════════════════════════════════════════════════════════════════════════
# ForwardConstruct — Task 1: base class + node discovery
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# ForwardConstruct — Task 1: base class + node discovery
# ═══════════════════════════════════════════════════════════════════════════


class TestConstructFromFunctions:
    """construct_from_functions() — explicit function list for multi-pipeline files."""

    def test_chain_compiles_when_two_functions_wired_by_param_name(self):
        """Two @node functions wired by parameter name via explicit list."""

        @node(outputs=RawText)
        def cff_seed() -> RawText:
            return RawText(text="hello world")

        @node(outputs=Claims)
        def cff_split(cff_seed: RawText) -> Claims:
            return Claims(items=[w for w in cff_seed.text.split() if w])

        pipeline = construct_from_functions("explicit", [cff_seed, cff_split])
        assert isinstance(pipeline, Construct)
        assert pipeline.name == "explicit"
        assert [n.name for n in pipeline.nodes] == ["cff-seed", "cff-split"]

        graph = compile(pipeline)
        result = run(graph, input={"node_id": "cff-001"})
        assert result["cff_split"].items == ["hello", "world"]

    def test_topo_sort_works_when_list_order_differs_from_dag(self):
        """Explicit list in non-topological order still sorts correctly."""

        @node(outputs=RawText)
        def cff_topo_seed() -> RawText:
            return RawText(text="a b c")

        @node(outputs=Claims)
        def cff_topo_split(cff_topo_seed: RawText) -> Claims:
            return Claims(items=cff_topo_seed.text.split())

        @node(outputs=ClassifiedClaims)
        def cff_topo_report(cff_topo_split: Claims) -> ClassifiedClaims:
            return ClassifiedClaims(
                classified=[{"claim": c, "category": "x"} for c in cff_topo_split.items]
            )

        # Pass in SHUFFLED order — report first, then seed, then split
        pipeline = construct_from_functions(
            "topo", [cff_topo_report, cff_topo_seed, cff_topo_split]
        )
        names = [n.name for n in pipeline.nodes]
        assert names == ["cff-topo-seed", "cff-topo-split", "cff-topo-report"]

    def test_two_pipelines_coexist_when_built_from_separate_lists(self):
        """Two independent pipelines in the same module — the killer use case."""

        # Pipeline A
        @node(outputs=RawText)
        def pipeA_start() -> RawText:
            return RawText(text="pipeline A")

        @node(outputs=RawText)
        def pipeA_end(pipeA_start: RawText) -> RawText:
            return RawText(text=f"A: {pipeA_start.text}")

        # Pipeline B (same file, different nodes)
        @node(outputs=Claims)
        def pipeB_start() -> Claims:
            return Claims(items=["pipeline", "B"])

        @node(outputs=Claims)
        def pipeB_end(pipeB_start: Claims) -> Claims:
            return Claims(items=[f"B:{s}" for s in pipeB_start.items])

        pipeA = construct_from_functions("A", [pipeA_start, pipeA_end])
        pipeB = construct_from_functions("B", [pipeB_start, pipeB_end])

        gA = compile(pipeA)
        gB = compile(pipeB)
        rA = run(gA, input={"node_id": "A-001"})
        rB = run(gB, input={"node_id": "B-001"})

        assert rA["pipeA_end"].text == "A: pipeline A"
        assert rB["pipeB_end"].items == ["B:pipeline", "B:B"]

    def test_construct_raises_when_function_not_decorated(self):
        """A plain function without @node raises a clear error."""
        from neograph import ConstructError

        @node(outputs=RawText)
        def cff_ok() -> RawText:
            return RawText(text="ok")

        def not_a_node(x: RawText) -> Claims:  # missing @node
            return Claims(items=[x.text])

        with pytest.raises(ConstructError, match="not decorated with @node"):
            construct_from_functions("bad", [cff_ok, not_a_node])

    def test_construct_raises_when_non_callable_passed(self):
        """Passing a non-callable raises."""
        from neograph import ConstructError

        @node(outputs=RawText)
        def cff_ok2() -> RawText:
            return RawText(text="ok")

        with pytest.raises(ConstructError, match="not decorated with @node"):
            construct_from_functions("bad", [cff_ok2, "not a function"])

    def test_construct_raises_when_function_names_collide(self):
        """Two functions whose node names collide raise ConstructError."""
        from neograph import ConstructError

        @node(outputs=RawText, name="shared")
        def first() -> RawText:
            return RawText(text="first")

        @node(outputs=RawText, name="shared")
        def second() -> RawText:
            return RawText(text="second")

        with pytest.raises(ConstructError, match="name collision"):
            construct_from_functions("collision", [first, second])





class TestConstructLlmConfigDefault:
    """Construct-level default llm_config inherited by produce/gather/execute nodes."""

    def test_nodes_inherit_config_when_construct_has_default(self):
        """Produce nodes without explicit llm_config inherit the Construct default."""
        from neograph import Construct, Node

        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["x"])))

        # Build via declarative API — Construct carries the default
        a = Node("a", mode="think", outputs=Claims, model="fast", prompt="p")
        b = Node("b", mode="think", inputs=Claims, outputs=Claims, model="fast", prompt="p")

        pipeline = Construct(
            "with-default",
            llm_config={"output_strategy": "json_mode", "temperature": 0.5},
            nodes=[a, b],
        )

        # Both nodes should have inherited the Construct default
        assert pipeline.nodes[0].llm_config == {
            "output_strategy": "json_mode",
            "temperature": 0.5,
        }
        assert pipeline.nodes[1].llm_config == {
            "output_strategy": "json_mode",
            "temperature": 0.5,
        }

    def test_node_config_wins_when_merging_with_construct_default(self):
        """Per-node llm_config merges with Construct default; node wins on conflicts."""
        from neograph import Construct, Node

        a = Node("a", mode="think", outputs=Claims, model="fast", prompt="p",
                 llm_config={"temperature": 0.9, "max_tokens": 1000})

        pipeline = Construct(
            "merged",
            llm_config={"output_strategy": "json_mode", "temperature": 0.2},
            nodes=[a],
        )

        # Construct default provides output_strategy.
        # Node explicit temperature (0.9) overrides construct default (0.2).
        # Node max_tokens passes through.
        assert pipeline.nodes[0].llm_config == {
            "output_strategy": "json_mode",
            "temperature": 0.9,
            "max_tokens": 1000,
        }

    def test_scripted_nodes_get_config_when_construct_has_default(self):
        """Scripted nodes don't get llm_config inheritance (they don't use it)."""
        from neograph import Construct, Node

        register_scripted("noop_k7k", lambda input_data, config: Claims(items=["x"]))
        a = Node.scripted("a-k7k", fn="noop_k7k", outputs=Claims)

        pipeline = Construct(
            "scripted-default",
            llm_config={"output_strategy": "json_mode"},
            nodes=[a],
        )

        # Scripted nodes get the default applied (harmless — they don't use it)
        # but the propagation is uniform to keep the merge logic simple.
        assert pipeline.nodes[0].llm_config == {"output_strategy": "json_mode"}

    def test_node_config_unchanged_when_no_construct_default(self):
        """When Construct has no llm_config, nodes keep their original config unchanged."""
        from neograph import Construct, Node

        a = Node("a", mode="think", outputs=Claims, model="fast", prompt="p",
                 llm_config={"temperature": 0.7})

        pipeline = Construct("no-default", nodes=[a])

        assert pipeline.nodes[0].llm_config == {"temperature": 0.7}

    def test_decorator_inherits_config_when_using_construct_from_functions(self):
        """@node functions inherit the Construct default via construct_from_functions."""

        @node(outputs=Claims, prompt="p", model="fast")
        def cff_default_a() -> Claims: ...

        @node(outputs=Claims, prompt="p", model="fast",
              llm_config={"temperature": 0.9})
        def cff_default_b(cff_default_a: Claims) -> Claims: ...

        pipeline = construct_from_functions(
            "default-cff",
            [cff_default_a, cff_default_b],
            llm_config={"output_strategy": "json_mode", "temperature": 0.2},
        )

        # cff_default_a inherits both fields
        a_node = pipeline.nodes[0]
        assert a_node.llm_config == {"output_strategy": "json_mode", "temperature": 0.2}

        # cff_default_b inherits output_strategy, overrides temperature
        b_node = pipeline.nodes[1]
        assert b_node.llm_config == {"output_strategy": "json_mode", "temperature": 0.9}





# ═══════════════════════════════════════════════════════════════════════════
# TestNodeInputsFieldRename (neograph-kqd.1)
#
# Step 1 of the Node.inputs refactor is a pure field rename:
# Node.input → Node.inputs. Field type stays Any and keeps the same shape
# acceptance (None | type | dict). Runtime behavior is unchanged. These
# tests fail before the rename (Node has no `inputs` field) and pass after.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# TestNodeDecoratorDictInputs (neograph-kqd.4)
#
# @node decoration now emits dict-form inputs={param_name: annotation, ...}
# for all typed upstream params. This is the metadata shift that lets
# step-2's validator catch fan-in mismatches via _check_fan_in_inputs.
# Fan-out params (Each) are stripped from inputs at construct-assembly time.
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeDecoratorDictInputs:
    def test_dict_inputs_emitted_when_single_upstream_typed(self):
        """@node with one typed upstream param emits dict form."""
        from neograph.decorators import construct_from_functions

        @node(outputs=Claims)
        def produce() -> Claims:
            return Claims(items=["a"])

        @node(outputs=MergedResult)
        def consume(produce: Claims) -> MergedResult:
            return MergedResult(final_text=",".join(produce.items))

        construct_from_functions("p", [produce, consume])
        assert isinstance(consume.inputs, dict)
        assert consume.inputs == {"produce": Claims}

    def test_dict_inputs_emitted_when_three_upstreams_typed(self):
        """@node with three typed upstreams emits a 3-key dict."""
        from neograph.decorators import construct_from_functions

        @node(outputs=Claims)
        def produce_a() -> Claims:
            return Claims(items=["a"])

        @node(outputs=RawText)
        def produce_b() -> RawText:
            return RawText(text="b")

        @node(outputs=ClusterGroup)
        def produce_c() -> ClusterGroup:
            return ClusterGroup(label="c", claim_ids=[])

        @node(outputs=MergedResult)
        def consume(
            produce_a: Claims,
            produce_b: RawText,
            produce_c: ClusterGroup,
        ) -> MergedResult:
            return MergedResult(final_text="x")

        construct_from_functions("p", [produce_a, produce_b, produce_c, consume])
        assert isinstance(consume.inputs, dict)
        assert consume.inputs == {
            "produce_a": Claims,
            "produce_b": RawText,
            "produce_c": ClusterGroup,
        }

    def test_fan_out_param_marked_when_map_over_set(self):
        """Each fan-out param stays in inputs dict and node.fan_out_param
        marks it so factory._extract_input routes it to neo_each_item."""
        from neograph.decorators import construct_from_functions

        @node(outputs=Clusters)
        def make_clusters() -> Clusters:
            return Clusters(groups=[])

        @node(
            outputs=MatchResult,
            map_over="make_clusters.groups",
            map_key="label",
        )
        def verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label=cluster.label, matched=[])

        pipeline = construct_from_functions("p", [make_clusters, verify])
        # After assembly, the Construct's node (not the original) has
        # fan_out_param set via model_copy.
        assembled = {n.name: n for n in pipeline.nodes if isinstance(n, Node)}
        assembled_verify = assembled["verify"]
        assert isinstance(assembled_verify.inputs, dict)
        assert "cluster" in assembled_verify.inputs
        assert assembled_verify.inputs["cluster"] is ClusterGroup
        assert assembled_verify.fan_out_param == "cluster"

    def test_validator_catches_mismatch_when_fan_in_types_wrong(self):
        """Step-2's validator catches @node fan-in type mismatches via
        dict-form inputs (no more two-walker setup)."""
        from neograph.decorators import construct_from_functions

        @node(outputs=Claims)
        def upstream() -> Claims:
            return Claims(items=["x"])

        @node(outputs=MergedResult)
        def consume(upstream: RawText) -> MergedResult:  # WRONG TYPE
            return MergedResult(final_text="x")

        with pytest.raises(ConstructError) as exc_info:
            construct_from_functions("p", [upstream, consume])
        msg = str(exc_info.value)
        assert "'upstream'" in msg
        assert "Claims" in msg or "RawText" in msg

    def test_log_shows_scripted_mode_when_fan_in_executes(self):
        """@node fan-in execution logs mode='scripted', not 'raw'
        (neograph-kqd.4 criterion 9)."""
        from neograph.decorators import construct_from_functions

        @node(outputs=Claims)
        def produce_claims() -> Claims:
            return Claims(items=["a", "b"])

        @node(outputs=RawText)
        def produce_text() -> RawText:
            return RawText(text="hello")

        @node(outputs=MergedResult)
        def combine(produce_claims: Claims, produce_text: RawText) -> MergedResult:
            return MergedResult(final_text=produce_text.text + ":" + ",".join(produce_claims.items))

        pipeline = construct_from_functions("p", [produce_claims, produce_text, combine])
        graph = compile(pipeline)

        import structlog
        captured: list[dict] = []

        def capture_processor(logger, method_name, event_dict):
            captured.append(dict(event_dict))
            return event_dict

        structlog.configure(processors=[capture_processor, structlog.processors.KeyValueRenderer()])
        try:
            run(graph, input={"node_id": "test"})
        finally:
            structlog.reset_defaults()

        # Find the node_start event for 'combine' and assert mode='scripted'
        combine_starts = [e for e in captured if e.get("node") == "combine" and e.get("event") == "node_start"]
        assert combine_starts, f"no node_start event for combine; captured={captured}"
        assert all(e.get("mode") == "scripted" for e in combine_starts), (
            f"combine fan-in should log mode='scripted', got: {combine_starts}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TestListOverEachEndToEnd (neograph-kqd.5)
#
# Merge-after-fan-out pattern across all three API surfaces: Each producer
# + list[X] consumer. Validator rule (kqd.2) + factory unwrap (kqd.3) +
# decorator dict-form inputs (kqd.4) + raw_adapter unwrap (kqd.5) wire
# together into a complete end-to-end feature.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# TestConditionalProduce (neograph-s14)
#
# skip_when= predicate bypasses LLM call. skip_value= provides the output.
# Zero LLM tokens consumed when skip fires.
# ═══════════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════════
# NODE.OUTPUTS RENAME (neograph-1bp.1)
# ═══════════════════════════════════════════════════════════════════════════


class TestDecoratorDictOutputs:
    """@node decorator with dict-form outputs (neograph-1bp.5)."""

    def test_per_key_fields_written_when_dict_outputs_declared(self):
        """@node(outputs={'a': X, 'b': Y}) scripted → writes per-key fields."""
        import types


        mod = types.ModuleType("test_dec_dict_out_mod")

        @node(mode="scripted", outputs={"summary": RawText, "tags": Claims})
        def analyze() -> dict:
            return {"summary": RawText(text="hello"), "tags": Claims(items=["a"])}

        @node(mode="scripted", outputs=ClassifiedClaims)
        def classify(analyze_summary: RawText, analyze_tags: Claims) -> ClassifiedClaims:
            return ClassifiedClaims(classified=[{"claim": analyze_summary.text, "category": "ok"}])

        mod.analyze = analyze
        mod.classify = classify
        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        result = run(graph, input={})
        assert result["classify"] == ClassifiedClaims(classified=[{"claim": "hello", "category": "ok"}])

    def test_single_type_works_when_outputs_is_type(self):
        """@node(outputs=X) still works with single type."""

        @node(mode="scripted", outputs=RawText)
        def extract() -> RawText:
            return RawText(text="hi")

        assert extract.outputs is RawText

    def test_outputs_inferred_when_return_annotation_present(self):
        """Return annotation → outputs= when explicit kwarg not set."""

        @node(mode="scripted")
        def extract() -> RawText:
            return RawText(text="hi")

        assert extract.outputs is RawText


# ═══════════════════════════════════════════════════════════════════════════
# INTEROP: @node decorator integration with Operator and Each+DI
# ═══════════════════════════════════════════════════════════════════════════





# ═══════════════════════════════════════════════════════════════════════════
# INTEROP: @node decorator integration with Operator and Each+DI
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeDecoratorInterop:
    """Cross-feature integration: @node with Operator interrupt/resume and Each+DI."""

    def test_output_present_after_resume_when_operator_interrupt_always(self):
        """@node(interrupt_when=<callable>) pauses graph, resume delivers final result."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph import FromInput

        @node(mode="scripted", outputs=Claims)
        def produce(node_id: Annotated[str, FromInput]) -> Claims:
            return Claims(items=["claim-a", "claim-b"])

        @node(
            mode="scripted",
            outputs=Claims,
            interrupt_when=lambda state: {"needs_review": True},
        )
        def review(produce: Claims) -> Claims:
            return Claims(items=[f"reviewed:{c}" for c in produce.items])

        pipeline = construct_from_functions("op_interop", [produce, review])
        checkpointer = MemorySaver()
        graph = compile(pipeline, checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "test-op-interop"}}

        # First run: hits interrupt after review executes
        result = run(graph, input={"node_id": "op-test"}, config=config)
        assert "__interrupt__" in result

        # Resume with human feedback
        result = run(graph, resume={"approved": True}, config=config)
        assert result["review"] == Claims(items=["reviewed:claim-a", "reviewed:claim-b"])
        assert result["human_feedback"] == {"approved": True}

    def test_di_param_resolves_when_node_inside_each_fanout(self):
        """@node with map_over (Each) + Annotated[str, FromInput] resolves both fan-out
        item and DI param correctly."""
        from neograph import FromInput

        @node(mode="scripted", outputs=Clusters)
        def producer() -> Clusters:
            return Clusters(groups=[
                ClusterGroup(label="alpha", claim_ids=["c1"]),
                ClusterGroup(label="beta", claim_ids=["c2", "c3"]),
            ])

        @node(
            mode="scripted",
            outputs=MatchResult,
            map_over="producer.groups",
            map_key="label",
        )
        def consumer(
            cluster: ClusterGroup,
            node_id: Annotated[str, FromInput],
        ) -> MatchResult:
            return MatchResult(cluster_label=f"{node_id}:{cluster.label}", matched=[])

        pipeline = construct_from_functions("each_di", [producer, consumer])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-42"})

        consumer_results = result["consumer"]
        assert set(consumer_results.keys()) == {"alpha", "beta"}
        assert consumer_results["alpha"].cluster_label == "test-42:alpha"
        assert consumer_results["beta"].cluster_label == "test-42:beta"

    def test_di_param_resolves_downstream_of_each_barrier(self):
        """FromInput param resolves correctly in a node AFTER the Each
        barrier/assemble step (neograph-iio2)."""
        from neograph import FromInput

        @node(mode="scripted", outputs=Clusters)
        def producer() -> Clusters:
            return Clusters(groups=[
                ClusterGroup(label="alpha", claim_ids=["c1"]),
            ])

        @node(
            mode="scripted",
            outputs=MatchResult,
            map_over="producer.groups",
            map_key="label",
        )
        def verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label=cluster.label, matched=["ok"])

        @node(mode="scripted", outputs=RawText)
        def summarize(
            verify: dict[str, MatchResult],
            pipeline_id: Annotated[str, FromInput],
        ) -> RawText:
            return RawText(text=f"{pipeline_id}:{len(verify)}")

        pipeline = construct_from_functions("after_each", [producer, verify, summarize])
        graph = compile(pipeline)
        result = run(graph, input={"pipeline_id": "test-99"})

        assert result["summarize"].text == "test-99:1"

    def test_di_bundled_model_resolves_downstream_of_each_barrier(self):
        """Bundled BaseModel FromInput resolves after Each barrier (neograph-iio2)."""
        from neograph import FromInput

        class PipeCtx(BaseModel, frozen=True):
            node_id: str
            project_root: str = "/tmp"

        @node(mode="scripted", outputs=Clusters)
        def producer() -> Clusters:
            return Clusters(groups=[
                ClusterGroup(label="alpha", claim_ids=["c1"]),
            ])

        @node(
            mode="scripted",
            outputs=MatchResult,
            map_over="producer.groups",
            map_key="label",
        )
        def verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label=cluster.label, matched=["ok"])

        @node(mode="scripted", outputs=RawText)
        def summarize(
            verify: dict[str, MatchResult],
            ctx: Annotated[PipeCtx, FromInput],
        ) -> RawText:
            return RawText(text=f"{ctx.node_id}:{len(verify)}")

        pipeline = construct_from_functions("bundled_after_each", [producer, verify, summarize])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "bundled-test", "project_root": "/src"})

        assert result["summarize"].text == "bundled-test:1"


# ═══════════════════════════════════════════════════════════════════════════
# Coverage gap tests — lines missed in decorators.py
# ═══════════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════════
# TestListOverEachEndToEnd (neograph-kqd.5)
#
# Merge-after-fan-out pattern across all three API surfaces: Each producer
# + list[X] consumer. Validator rule (kqd.2) + factory unwrap (kqd.3) +
# decorator dict-form inputs (kqd.4) + raw_adapter unwrap (kqd.5) wire
# together into a complete end-to-end feature.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# TestConditionalProduce (neograph-s14)
#
# skip_when= predicate bypasses LLM call. skip_value= provides the output.
# Zero LLM tokens consumed when skip fires.
# ═══════════════════════════════════════════════════════════════════════════

class TestConditionalProduce:
    def test_skip_value_returned_when_skip_when_true(self):
        """When skip_when returns True, the node returns skip_value
        without any LLM call."""
        from neograph.decorators import construct_from_functions
        from tests.fakes import StructuredFake, configure_fake_llm

        configure_fake_llm(lambda tier: StructuredFake(lambda m: m()))

        @node(outputs=Claims)
        def seed() -> Claims:
            return Claims(items=["single"])

        @node(
            outputs=MergedResult,
            mode="think",
            model="fast",
            prompt="p",
            skip_when=lambda inp: len(inp.items) == 1,
            skip_value=lambda inp: MergedResult(final_text=inp.items[0]),
        )
        def maybe_merge(seed: Claims) -> MergedResult: ...

        pipeline = construct_from_functions("skip-test", [seed, maybe_merge])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "t"})
        assert result["maybe_merge"].final_text == "single"

    def test_llm_called_when_skip_when_false(self):
        """When skip_when returns False, the normal LLM path runs."""
        from neograph.decorators import construct_from_functions
        from tests.fakes import StructuredFake, configure_fake_llm

        configure_fake_llm(
            lambda tier: StructuredFake(lambda m: MergedResult(final_text="llm-result")),
        )

        @node(outputs=Claims)
        def seed() -> Claims:
            return Claims(items=["a", "b"])

        @node(
            outputs=MergedResult,
            mode="think",
            model="fast",
            prompt="p",
            skip_when=lambda inp: len(inp.items) == 1,
            skip_value=lambda inp: MergedResult(final_text=inp.items[0]),
        )
        def maybe_merge(seed: Claims) -> MergedResult: ...

        pipeline = construct_from_functions("no-skip", [seed, maybe_merge])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "t"})
        assert result["maybe_merge"].final_text == "llm-result"

    def test_skip_fields_stored_when_set_on_node(self):
        """skip_when and skip_value are proper Node fields."""
        def pred(x):
            return True

        def val(x):
            return x
        n = Node(
            "t", mode="think", inputs=Claims, outputs=MergedResult,
            model="fast", prompt="p",
            skip_when=pred, skip_value=val,
        )
        assert n.skip_when is pred
        assert n.skip_value is val

    def test_skip_fields_none_when_not_set(self):
        """Nodes without skip_when have it as None (backward compat)."""
        n = Node("t", mode="think", inputs=Claims, outputs=MergedResult,
                 model="fast", prompt="p")
        assert n.skip_when is None
        assert n.skip_value is None

    def test_skip_fields_passed_when_set_via_decorator(self):
        """@node(skip_when=...) passes through to Node."""

        @node(
            outputs=MergedResult, mode="think", model="fast", prompt="p",
            skip_when=lambda x: True,
            skip_value=lambda x: MergedResult(final_text="skipped"),
        )
        def my_node(seed: Claims) -> MergedResult: ...

        assert callable(my_node.skip_when)
        assert callable(my_node.skip_value)


# ═══════════════════════════════════════════════════════════════════════════
# NODE.OUTPUTS RENAME (neograph-1bp.1)
# ═══════════════════════════════════════════════════════════════════════════


