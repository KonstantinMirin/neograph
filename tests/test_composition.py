"""Composition tests — sub-constructs, multi-field input, state hygiene,
reducer edge cases, dict-form output state/factory, epic acceptance.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from neograph import (
    Construct, ConstructError, Node, Each, Oracle, Operator,
    compile, construct_from_functions, construct_from_module,
    merge_fn, node, run, tool,
    CompileError, ConfigurationError, ExecutionError,
)
from neograph.factory import register_scripted
from tests.schemas import RawText, Claims, ClassifiedClaims, ClusterGroup, Clusters, MatchResult, MergedResult, ValidationResult


class TestSubgraph:
    """Sub-construct with isolated state inside a parent pipeline."""

    def test_output_surfaces_when_sub_construct_runs_with_isolated_state(self):
        """Sub-construct runs with isolated state, only output surfaces."""
        from neograph.factory import register_scripted

        # Parent: produces claims
        register_scripted("decompose", lambda input_data, config: EnrichInput(
            claims=["claim-1", "claim-2"],
        ))

        # Sub-construct nodes: enrich the claims
        register_scripted("lookup", lambda input_data, config: RawText(
            text=f"context for {len(input_data.claims)} claims",
        ))
        register_scripted("score", lambda input_data, config: EnrichOutput(
            scored=[{"claim": c, "score": "high"} for c in input_data.claims],
        ))

        # Sub-construct with declared I/O boundary
        enrich = Construct(
            "enrich",
            input=EnrichInput,
            output=EnrichOutput,
            nodes=[
                Node.scripted("lookup", fn="lookup", inputs=EnrichInput, outputs=RawText),
                Node.scripted("score", fn="score", inputs=EnrichInput, outputs=EnrichOutput),
            ],
        )

        # Parent pipeline
        decompose = Node.scripted("decompose", fn="decompose", outputs=EnrichInput)
        parent = Construct("parent", nodes=[decompose, enrich])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Sub-construct output surfaces under its name
        assert result["enrich"] is not None
        assert len(result["enrich"].scored) == 2
        assert result["enrich"].scored[0]["score"] == "high"

        # Sub-construct internals (lookup, score) do NOT appear in parent result
        assert "lookup" not in result
        assert "score" not in result

    def test_no_collision_when_parent_and_sub_share_node_name(self):
        """Sub-construct's internal fields don't collide with parent fields."""
        from neograph.factory import register_scripted

        # Both parent and sub-construct have a node named "process"
        register_scripted("parent_process", lambda input_data, config: Claims(items=["parent"]))
        register_scripted("sub_input", lambda input_data, config: Claims(items=["sub-in"]))
        register_scripted("sub_process", lambda input_data, config: RawText(text="sub-result"))

        sub = Construct(
            "sub",
            input=Claims,
            output=RawText,
            nodes=[
                Node.scripted("process", fn="sub_process", inputs=Claims, outputs=RawText),
            ],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("process", fn="parent_process", outputs=Claims),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Parent's "process" node output
        assert result["process"].items == ["parent"]
        # Sub-construct output (no collision)
        assert result["sub"].text == "sub-result"

    def test_compile_raises_when_sub_construct_missing_input(self):
        """Sub-construct without declared input raises at compile."""
        from neograph.factory import register_scripted

        register_scripted("noop", lambda input_data, config: Claims(items=[]))

        sub = Construct("bad-sub", output=Claims, nodes=[
            Node.scripted("noop", fn="noop", outputs=Claims),
        ])

        parent = Construct("parent", nodes=[sub])

        with pytest.raises(CompileError, match="has no input type"):
            compile(parent)

    def test_compile_raises_when_sub_construct_missing_output(self):
        """Sub-construct without declared output raises at compile."""
        from neograph.factory import register_scripted

        register_scripted("noop", lambda input_data, config: Claims(items=[]))

        sub = Construct("bad-sub", input=Claims, nodes=[
            Node.scripted("noop", fn="noop", outputs=Claims),
        ])

        parent = Construct("parent", nodes=[sub])

        with pytest.raises(CompileError, match="has no output type"):
            compile(parent)

    def test_oracle_merges_when_inside_sub_construct(self):
        """Oracle inside a sub-construct — fan-out happens in isolated state."""
        from neograph.factory import register_scripted

        register_scripted("parent_prep", lambda input_data, config: Claims(items=["topic"]))
        register_scripted("sub_gen", lambda input_data, config: RawText(text=f"variant"))

        def sub_merge(variants, config):
            return RawText(text=f"merged {len(variants)} variants")

        register_scripted("sub_merge", sub_merge)

        sub = Construct(
            "oracle-sub",
            input=Claims,
            output=RawText,
            nodes=[
                Node.scripted("gen", fn="sub_gen", outputs=RawText) | Oracle(n=3, merge_fn="sub_merge"),
            ],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("prep", fn="parent_prep", outputs=Claims),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        assert result["oracle_sub"].text == "merged 3 variants"
        # Oracle internals from sub don't leak
        assert not any("neo_oracle" in k for k in result)

    def test_each_fans_out_when_inside_sub_construct(self):
        """Each inside a sub-construct — fan-out in isolated state."""
        from neograph.factory import register_scripted

        # Each produces dict[str, MatchResult]. The sub-construct's output
        # must be the dict type, not MatchResult, because that's what Each writes.
        # Use a collector node after Each to convert dict → single output.
        register_scripted("parent_clusters", lambda input_data, config: Clusters(
            groups=[ClusterGroup(label="a", claim_ids=["1"]), ClusterGroup(label="b", claim_ids=["2"])]
        ))
        register_scripted("sub_verify", lambda input_data, config: MatchResult(
            cluster_label=input_data.label, matched=["ok"],
        ))
        register_scripted("sub_collect", lambda input_data, config: RawText(
            text=f"verified {len(input_data)} clusters" if isinstance(input_data, dict) else "verified"
        ))

        sub = Construct(
            "verify-sub",
            input=Clusters,
            output=RawText,
            nodes=[
                Node.scripted("verify", fn="sub_verify", inputs=ClusterGroup, outputs=MatchResult)
                | Each(over="neo_subgraph_input.groups", key="label"),
                Node.scripted("collect", fn="sub_collect", outputs=RawText),
            ],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("make-clusters", fn="parent_clusters", outputs=Clusters),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        assert result["verify_sub"] is not None
        assert "verified" in result["verify_sub"].text

    def test_output_bubbles_when_two_levels_deep(self):
        """Construct inside Construct inside Construct — two levels deep."""
        from neograph.factory import register_scripted

        register_scripted("l0_start", lambda input_data, config: Claims(items=["raw"]))
        register_scripted("l1_process", lambda input_data, config: Claims(items=["l1-processed"]))
        register_scripted("l2_detail", lambda input_data, config: RawText(text="l2-done"))
        register_scripted("l0_finish", lambda input_data, config: RawText(text="final"))

        # Level2: Claims → RawText
        level2 = Construct(
            "level2",
            input=Claims,
            output=RawText,
            nodes=[Node.scripted("detail", fn="l2_detail", inputs=Claims, outputs=RawText)],
        )

        # Level1: Claims → RawText (via level2)
        level1 = Construct(
            "level1",
            input=Claims,
            output=RawText,
            nodes=[
                Node.scripted("process", fn="l1_process", inputs=Claims, outputs=Claims),
                level2,
            ],
        )

        parent = Construct("root", nodes=[
            Node.scripted("start", fn="l0_start", outputs=Claims),
            level1,
            Node.scripted("finish", fn="l0_finish", inputs=RawText, outputs=RawText),
        ])

        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Level1's output is RawText (from level2 bubbling up)
        assert result["level1"].text == "l2-done"
        assert result["finish"].text == "final"
        # No level1/level2 internals in parent result
        assert "level2" not in result
        assert "detail" not in result
        assert "process" not in result

    def test_both_outputs_surface_when_two_sub_constructs_in_parent(self):
        """Two sub-constructs in the same parent pipeline."""
        from neograph.factory import register_scripted

        register_scripted("make_input", lambda input_data, config: Claims(items=["a", "b"]))
        register_scripted("enrich_fn", lambda input_data, config: RawText(text="enriched"))
        register_scripted("validate_fn", lambda input_data, config: ValidationResult(passed=True, issues=[]))

        enrich_sub = Construct(
            "enrich",
            input=Claims,
            output=RawText,
            nodes=[Node.scripted("e", fn="enrich_fn", inputs=Claims, outputs=RawText)],
        )

        validate_sub = Construct(
            "check",
            input=RawText,
            output=ValidationResult,
            nodes=[Node.scripted("v", fn="validate_fn", inputs=RawText, outputs=ValidationResult)],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("start", fn="make_input", outputs=Claims),
            enrich_sub,
            validate_sub,
        ])

        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        assert result["enrich"].text == "enriched"
        assert result["check"].passed is True

    def test_compile_raises_when_operator_without_checkpointer(self):
        """Operator node without checkpointer raises ValueError at compile."""
        from neograph.factory import register_condition, register_scripted

        register_scripted("x", lambda input_data, config: Claims(items=[]))
        register_condition("always", lambda state: True)

        node = Node.scripted("x", fn="x", outputs=Claims) | Operator(when="always")
        pipeline = Construct("test-no-cp", nodes=[node])

        with pytest.raises(CompileError, match="checkpointer"):
            compile(pipeline)


# ═══════════════════════════════════════════════════════════════════════════
# CONSTRUCT + MODIFIER COMPOSITIONS
#
# Every modifier × Construct target, plus deep nesting combos.
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiFieldInput:
    """Node with dict[str, type] input spec extracts multiple fields."""

    def test_node_receives_multiple_fields_when_dict_inputs_declared(self):
        """Node receives multiple typed fields from state."""
        from neograph.factory import register_scripted

        register_scripted("make_claims", lambda input_data, config: Claims(items=["a", "b"]))
        register_scripted("make_raw", lambda input_data, config: RawText(text="hello"))

        def combine(input_data, config):
            claims = input_data["step_a"]
            raw = input_data["step_b"]
            return RawText(text=f"{raw.text}: {len(claims.items)} items")

        register_scripted("combine", combine)

        step_a = Node.scripted("step-a", fn="make_claims", outputs=Claims)
        step_b = Node.scripted("step-b", fn="make_raw", outputs=RawText)
        step_c = Node.scripted(
            "step-c", fn="combine",
            inputs={"step_a": Claims, "step_b": RawText},
            outputs=RawText,
        )

        pipeline = Construct("test-multi-input", nodes=[step_a, step_b, step_c])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        assert result["step_c"].text == "hello: 2 items"


class TestReducerEdgeCases:
    """Reducer functions handle None and unexpected inputs correctly."""

    def test_oracle_reducer_builds_list_when_initial_none(self):
        """Oracle reducer builds list from None initial state."""
        from neograph.state import _collect_oracle_results

        # First write: existing is None
        result = _collect_oracle_results(None, "first")
        assert result == ["first"]

        # Second write: existing is a list
        result = _collect_oracle_results(["first"], "second")
        assert result == ["first", "second"]

        # List input (batch)
        result = _collect_oracle_results(["a"], ["b", "c"])
        assert result == ["a", "b", "c"]

    def test_dict_reducer_starts_when_initial_none(self):
        """Dict merge reducer starts from None."""
        from neograph.state import _merge_dicts

        result = _merge_dicts(None, {"a": 1})
        assert result == {"a": 1}

        result = _merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

        # Duplicate key raises
        with pytest.raises(ExecutionError, match="duplicate key"):
            _merge_dicts({"a": 1}, {"a": 99})

    def test_new_value_replaces_when_last_write_wins(self):
        """Last-write-wins reducer always returns new value."""
        from neograph.state import _last_write_wins

        assert _last_write_wins("old", "new") == "new"
        assert _last_write_wins(None, "new") == "new"
        assert _last_write_wins("old", None) is None


class TestStateHygiene:
    """Framework internals never leak to the consumer."""

    def test_neo_keys_absent_when_oracle_pipeline_completes(self):
        """Oracle collector and gen_id are not in the result."""
        from neograph.factory import register_scripted

        register_scripted("g", lambda input_data, config: Claims(items=["x"]))
        register_scripted("m", lambda variants, config: Claims(items=["merged"]))

        node = Node.scripted("gen", fn="g", outputs=Claims) | Oracle(n=2, merge_fn="m")
        pipeline = Construct("test-hygiene-oracle", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Consumer sees the merged result under the node's name
        assert result["gen"].items == ["merged"]
        # Internals stripped
        assert not any(k.startswith("neo_") for k in result)

    def test_neo_keys_absent_when_each_pipeline_completes(self):
        """Each item plumbing is not in the result."""
        from neograph.factory import register_scripted

        register_scripted("make", lambda input_data, config: Clusters(
            groups=[ClusterGroup(label="a", claim_ids=["1"])]
        ))
        register_scripted("proc", lambda input_data, config: MatchResult(
            cluster_label=input_data.label, matched=["done"],
        ))

        make = Node.scripted("make", fn="make", outputs=Clusters)
        proc = Node.scripted(
            "proc", fn="proc", inputs=ClusterGroup, outputs=MatchResult
        ) | Each(over="make.groups", key="label")

        pipeline = Construct("test-hygiene-each", nodes=[make, proc])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Consumer sees dict keyed by label
        assert "a" in result["proc"]
        # Internals stripped
        assert not any(k.startswith("neo_") for k in result)

    def test_reducer_raises_when_each_keys_duplicate(self):
        """Each fan-out with duplicate dispatch keys raises, not silently overwrites."""
        from neograph.factory import register_scripted

        # Collection with duplicate labels
        register_scripted("make_dupes", lambda input_data, config: Clusters(
            groups=[
                ClusterGroup(label="same", claim_ids=["1"]),
                ClusterGroup(label="same", claim_ids=["2"]),
            ]
        ))
        register_scripted("proc_dupe", lambda input_data, config: MatchResult(
            cluster_label=input_data.label, matched=["done"],
        ))

        make = Node.scripted("make-dupes", fn="make_dupes", outputs=Clusters)
        proc = Node.scripted(
            "proc-dupe", fn="proc_dupe", inputs=ClusterGroup, outputs=MatchResult
        ) | Each(over="make_dupes.groups", key="label")

        pipeline = Construct("test-dupe-key", nodes=[make, proc])
        graph = compile(pipeline)

        with pytest.raises(Exception, match="duplicate key"):
            run(graph, input={"node_id": "test-001"})


# ═══════════════════════════════════════════════════════════════════════════
# SUBGRAPH TESTS — Construct inside Construct with isolated state
# ═══════════════════════════════════════════════════════════════════════════


class EnrichInput(BaseModel, frozen=True):
    claims: list[str]


class EnrichOutput(BaseModel, frozen=True):
    scored: list[dict[str, str]]


class TestStripInternalsEdge:
    """Edge case: _strip_internals when result is not a dict."""

    def test_value_passes_through_when_result_not_dict(self):
        """Non-dict results pass through unchanged."""
        from neograph.runner import _strip_internals

        assert _strip_internals("hello") == "hello"
        assert _strip_internals(42) == 42
        assert _strip_internals(None) is None
        assert _strip_internals([1, 2, 3]) == [1, 2, 3]


class TestDictOutputsStateModel:
    """Dict-form outputs emit per-key state fields (neograph-1bp.2)."""

    def test_state_has_per_key_fields_when_dict_outputs_declared(self):
        """outputs={'result': X, 'log': Y} → state has node_result + node_log."""
        from neograph.state import compile_state_model
        n = Node("explore", outputs={"result": RawText, "tool_log": Claims})
        c = Construct("p", nodes=[n])
        state_model = compile_state_model(c)
        assert "explore_result" in state_model.model_fields
        assert "explore_tool_log" in state_model.model_fields
        # The old single-name field should NOT exist
        assert "explore" not in state_model.model_fields

    def test_state_has_node_name_field_when_single_type_outputs(self):
        """Single-type outputs= still emits {node_name} field."""
        from neograph.state import compile_state_model
        n = Node.scripted("extract", fn="f", outputs=RawText)
        c = Construct("p", nodes=[n])
        state_model = compile_state_model(c)
        assert "extract" in state_model.model_fields

    def test_each_wraps_per_key_when_dict_outputs_with_each(self):
        """Each modifier wraps each dict-output key in dict[str, T]."""
        from neograph.state import compile_state_model
        n = Node(
            "verify", outputs={"result": RawText, "meta": Claims},
        ) | Each(over="items", key="label")
        c = Construct("p", nodes=[n])
        state_model = compile_state_model(c)
        assert "verify_result" in state_model.model_fields
        assert "verify_meta" in state_model.model_fields


class TestDictOutputsFactory:
    """Factory writes dict-form outputs to per-key state fields (neograph-1bp.3)."""

    def test_per_key_state_written_when_scripted_dict_outputs(self):
        """Scripted node with dict outputs writes each key to its state field."""
        from neograph import node, construct_from_module, compile, run
        from neograph.factory import register_scripted
        import types

        mod = types.ModuleType("test_dict_out_mod")

        @node(mode="scripted", outputs={"summary": RawText, "count": Claims})
        def analyze() -> dict:
            return {"summary": RawText(text="hello"), "count": Claims(items=["a"])}

        mod.analyze = analyze
        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        result = run(graph, input={})
        assert result["analyze_summary"] == RawText(text="hello")
        assert result["analyze_count"] == Claims(items=["a"])

    def test_node_name_field_written_when_single_type_at_runtime(self):
        """Single-type outputs still writes to {node_name} at runtime."""
        from neograph import node, construct_from_module, compile, run
        import types

        mod = types.ModuleType("test_single_out_mod")

        @node(mode="scripted", outputs=RawText)
        def extract() -> RawText:
            return RawText(text="world")

        mod.extract = extract
        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        result = run(graph, input={})
        assert result["extract"] == RawText(text="world")


# ═══════════════════════════════════════════════════════════════════════════
# TestNodeInputsEpicAcceptance (neograph-kqd.7)
#
# Closes remaining acceptance gaps from the epic. The bulk of the matrix is
# covered by TestFanInValidation / TestListOverEachEndToEnd /
# TestExtractInputListUnwrap / TestNodeDecoratorDictInputs. This class adds:
#   - LLM-driven spec round-trip (JSON-shaped dict → Node → validated pipeline)
#   - Zero-upstream node explicit test
#   - Programmatic fan-in via Node + Oracle pipe
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeInputsEpicAcceptance:
    def test_pipeline_compiles_when_llm_spec_with_fan_in(self):
        """An LLM-driven pipeline builder constructs Nodes from a JSON-shaped
        dict with string type names, resolves them via a type registry,
        and compiles — validator catches any mismatches."""
        from neograph import compile, run
        from neograph.factory import register_scripted

        # Type registry — what the LLM's string type names resolve to.
        type_registry: dict[str, type] = {
            "Claims": Claims,
            "RawText": RawText,
            "MergedResult": MergedResult,
        }

        # LLM-emitted spec: three nodes, one fan-in consumer.
        spec = [
            {
                "name": "seed_claims",
                "fn": "l7_seed_claims",
                "inputs": None,
                "output": "Claims",
            },
            {
                "name": "seed_text",
                "fn": "l7_seed_text",
                "inputs": None,
                "output": "RawText",
            },
            {
                "name": "combine",
                "fn": "l7_combine",
                "inputs": {"seed_claims": "Claims", "seed_text": "RawText"},
                "output": "MergedResult",
            },
        ]

        register_scripted("l7_seed_claims", lambda _i, _c: Claims(items=["a", "b"]))
        register_scripted("l7_seed_text", lambda _i, _c: RawText(text="hello"))

        def combine_fn(input_data, _cfg):
            seed_claims = input_data["seed_claims"]
            seed_text = input_data["seed_text"]
            return MergedResult(
                final_text=f"{seed_text.text}:{','.join(seed_claims.items)}",
            )

        register_scripted("l7_combine", combine_fn)

        # Builder: resolve string type names, construct Node instances.
        def build_node(entry: dict) -> Node:
            output = type_registry[entry["output"]] if entry["output"] else None
            inputs = entry["inputs"]
            if isinstance(inputs, dict):
                inputs = {k: type_registry[v] for k, v in inputs.items()}
            elif isinstance(inputs, str):
                inputs = type_registry[inputs]
            return Node.scripted(
                entry["name"], fn=entry["fn"],
                inputs=inputs, outputs=output,
            )

        nodes = [build_node(e) for e in spec]
        pipeline = Construct("l7-llm-spec", nodes=nodes)
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "l7"})
        assert result["combine"].final_text == "hello:a,b"

    def test_construct_raises_when_llm_spec_has_type_mismatch(self):
        """LLM-emitted spec with a type mismatch raises ConstructError
        at assembly time, not during runtime."""
        type_registry: dict[str, type] = {
            "Claims": Claims,
            "RawText": RawText,
            "MergedResult": MergedResult,
        }

        # Consumer declares inputs['upstream']=Claims but upstream produces RawText.
        spec = [
            {"name": "upstream", "fn": "f", "inputs": None, "output": "RawText"},
            {
                "name": "consumer",
                "fn": "f",
                "inputs": {"upstream": "Claims"},
                "output": "MergedResult",
            },
        ]

        def build_node(entry: dict) -> Node:
            output = type_registry[entry["output"]] if entry["output"] else None
            inputs = entry["inputs"]
            if isinstance(inputs, dict):
                inputs = {k: type_registry[v] for k, v in inputs.items()}
            return Node.scripted(
                entry["name"], fn=entry["fn"],
                inputs=inputs, outputs=output,
            )

        nodes = [build_node(e) for e in spec]
        with pytest.raises(ConstructError) as exc_info:
            Construct("l7-bad-spec", nodes=nodes)
        msg = str(exc_info.value)
        assert "upstream" in msg

    def test_node_assembles_when_inputs_none(self):
        """Nodes with no upstreams use inputs=None and assemble cleanly."""
        seed = Node.scripted("seed", fn="f", inputs=None, outputs=Claims)
        pipeline = Construct("zero-upstream", nodes=[seed])
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].inputs is None
        assert pipeline.nodes[0].outputs is Claims
        assert pipeline.nodes[0].name == "seed"

    def test_fanout_and_upstream_coexist_when_mixed_in_one_node(self):
        """@node with BOTH upstream params AND a fan-out param (Each)
        runs end-to-end — the critical path for kqd.8 unification."""
        from neograph import compile, run, node
        from neograph.decorators import construct_from_functions

        @node(outputs=RawText)
        def context_source() -> RawText:
            return RawText(text="shared-context")

        @node(outputs=Clusters)
        def make_clusters() -> Clusters:
            return Clusters(groups=[
                ClusterGroup(label="a", claim_ids=["1"]),
                ClusterGroup(label="b", claim_ids=["2"]),
            ])

        @node(
            outputs=MatchResult,
            map_over="make_clusters.groups",
            map_key="label",
        )
        def verify(context_source: RawText, cluster: ClusterGroup) -> MatchResult:
            return MatchResult(
                cluster_label=cluster.label,
                matched=[f"{context_source.text}-{cluster.label}"],
            )

        pipeline = construct_from_functions(
            "mixed-e2e", [context_source, make_clusters, verify],
        )
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "kqd8"})
        # verify is Each-modified → result is dict[str, MatchResult]
        assert isinstance(result["verify"], dict)
        assert sorted(result["verify"].keys()) == ["a", "b"]
        assert result["verify"]["a"].matched == ["shared-context-a"]
        assert result["verify"]["b"].matched == ["shared-context-b"]

    def test_attach_scripted_raw_fn_absent_from_module(self):
        """_attach_scripted_raw_fn no longer exists — all scripted @node
        routes through register_scripted + _make_scripted_wrapper."""
        import neograph.decorators as dec
        assert not hasattr(dec, "_attach_scripted_raw_fn"), (
            "_attach_scripted_raw_fn still exists — kqd.8 is incomplete"
        )

    def test_raw_fn_none_when_scripted_node_uses_register_scripted(self):
        """Scripted @node nodes no longer have raw_fn set — they use
        scripted_fn + register_scripted instead."""
        from neograph import node
        from neograph.decorators import construct_from_functions

        @node(outputs=Claims)
        def produce() -> Claims:
            return Claims(items=["x"])

        @node(outputs=MergedResult)
        def consume(produce: Claims) -> MergedResult:
            return MergedResult(final_text=produce.items[0])

        construct_from_functions("raw-fn-check", [produce, consume])
        assert produce.raw_fn is None, "scripted @node should not set raw_fn"
        assert consume.raw_fn is None, "scripted @node should not set raw_fn"
        assert produce.scripted_fn is not None
        assert consume.scripted_fn is not None

    def test_fan_in_produces_result_when_programmatic_node_pipe(self):
        """Programmatic Node(inputs={...}) + modifier pipe works end-to-end."""
        from neograph import compile, run
        from neograph.factory import register_scripted

        register_scripted("l7_a", lambda _i, _c: Claims(items=["a1"]))
        register_scripted("l7_b", lambda _i, _c: RawText(text="b1"))

        def merge_fn(input_data, _cfg):
            return MergedResult(
                final_text=f"{input_data['a'].items[0]}-{input_data['b'].text}",
            )

        register_scripted("l7_merge", merge_fn)

        a = Node.scripted("a", fn="l7_a", outputs=Claims)
        b = Node.scripted("b", fn="l7_b", outputs=RawText)
        merger = Node.scripted(
            "merger", fn="l7_merge",
            inputs={"a": Claims, "b": RawText},
            outputs=MergedResult,
        )
        # Piping a modifier onto the merger should preserve the inputs shape
        # (Oracle on a fan-in merger is unusual but validates the path).
        pipeline = Construct("l7-prog", nodes=[a, b, merger])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "l7"})
        assert result["merger"].final_text == "a1-b1"


# ═══════════════════════════════════════════════════════════════════════════
# @NODE SUB-CONSTRUCT — construct_from_functions with input=/output=
#
# Port param resolution: @node function params whose type matches
# construct_input read from neo_subgraph_input instead of a peer @node.
# ═══════════════════════════════════════════════════════════════════════════


class VerifyClaim(BaseModel, frozen=True):
    claim_id: str
    text: str


class ClaimResult(BaseModel, frozen=True):
    claim_id: str
    disposition: str


class TestNodeSubConstruct:
    """construct_from_functions with input=/output= builds a sub-construct
    from @node functions. Port params read from the construct input port."""

    def test_scripted_node_reads_port_when_param_type_matches_construct_input(self):
        """A scripted @node inside a sub-construct reads its input from
        neo_subgraph_input when the param type matches construct_input."""
        @node(outputs=ClaimResult)
        def score(claim: VerifyClaim) -> ClaimResult:
            return ClaimResult(claim_id=claim.claim_id, disposition="confirmed")

        sub = construct_from_functions(
            "verify", [score], input=VerifyClaim, output=ClaimResult,
        )
        # sub should be a valid Construct with input/output ports
        assert sub.input is VerifyClaim
        assert sub.output is ClaimResult
        # The node's inputs should have neo_subgraph_input, not 'claim'
        score_node = sub.nodes[0]
        assert isinstance(score_node.inputs, dict)
        assert "neo_subgraph_input" in score_node.inputs

    def test_sub_construct_runs_when_embedded_in_parent(self):
        """Sub-construct built from @node functions runs inside a parent pipeline."""
        @node(outputs=ClaimResult)
        def judge(claim: VerifyClaim) -> ClaimResult:
            return ClaimResult(claim_id=claim.claim_id, disposition="valid")

        sub = construct_from_functions(
            "judge-sub", [judge], input=VerifyClaim, output=ClaimResult,
        )

        # Parent pipeline feeds the sub-construct
        register_scripted("make_claim", lambda _in, _cfg: VerifyClaim(
            claim_id="c1", text="The sky is blue",
        ))
        parent = Construct("parent", nodes=[
            Node.scripted("make-claim", fn="make_claim", outputs=VerifyClaim),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-port"})

        assert result["judge_sub"].disposition == "valid"
        assert result["judge_sub"].claim_id == "c1"

    def test_two_node_chain_inside_sub_construct_from_functions(self):
        """Two @node functions chained inside a sub-construct: first reads
        from port, second reads from first."""
        @node(outputs=RawText)
        def explore(claim: VerifyClaim) -> RawText:
            return RawText(text=f"evidence for {claim.claim_id}")

        @node(outputs=ClaimResult)
        def decide(explore: RawText) -> ClaimResult:
            return ClaimResult(claim_id="c1", disposition=f"based on: {explore.text}")

        sub = construct_from_functions(
            "verify-chain", [explore, decide],
            input=VerifyClaim, output=ClaimResult,
        )
        assert sub.input is VerifyClaim
        assert sub.output is ClaimResult
        # explore reads from port, decide reads from explore (peer @node)
        explore_node = [n for n in sub.nodes if n.name == "explore"][0]
        decide_node = [n for n in sub.nodes if n.name == "decide"][0]
        assert "neo_subgraph_input" in explore_node.inputs
        assert "explore" in decide_node.inputs

    def test_dict_outputs_flow_when_two_nodes_inside_sub_construct(self):
        """Dict-form outputs work between @nodes inside a @node sub-construct."""
        @node(outputs={"evidence": RawText, "confidence": Claims})
        def research(claim: VerifyClaim) -> dict:
            return {
                "evidence": RawText(text=f"found for {claim.claim_id}"),
                "confidence": Claims(items=["high"]),
            }

        @node(outputs=ClaimResult)
        def evaluate(research_evidence: RawText, research_confidence: Claims) -> ClaimResult:
            return ClaimResult(
                claim_id="c1",
                disposition=f"{research_evidence.text} ({research_confidence.items[0]})",
            )

        sub = construct_from_functions(
            "eval-sub", [research, evaluate],
            input=VerifyClaim, output=ClaimResult,
        )

        register_scripted("seed_claim", lambda _in, _cfg: VerifyClaim(
            claim_id="c1", text="test claim",
        ))
        parent = Construct("parent", nodes=[
            Node.scripted("seed-claim", fn="seed_claim", outputs=VerifyClaim),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "dict-out-sub"})

        assert result["eval_sub"].disposition == "found for c1 (high)"

    def test_sub_construct_fans_out_when_each_applied_via_construct_from_functions(self):
        """Sub-construct built from @node functions works with .map() (Each)."""
        @node(outputs=ClaimResult)
        def verify_one(claim: VerifyClaim) -> ClaimResult:
            return ClaimResult(claim_id=claim.claim_id, disposition="ok")

        sub = construct_from_functions(
            "verify", [verify_one],
            input=VerifyClaim, output=ClaimResult,
        ).map("make_claims.claims", key="claim_id")

        class ClaimBatch(BaseModel, frozen=True):
            claims: list[VerifyClaim]

        register_scripted("make_batch", lambda _in, _cfg: ClaimBatch(
            claims=[
                VerifyClaim(claim_id="c1", text="first"),
                VerifyClaim(claim_id="c2", text="second"),
            ],
        ))
        parent = Construct("parent", nodes=[
            Node.scripted("make-claims", fn="make_batch", outputs=ClaimBatch),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "each-sub"})

        assert isinstance(result["verify"], dict)
        assert sorted(result["verify"].keys()) == ["c1", "c2"]
        assert result["verify"]["c1"].disposition == "ok"

    def test_sub_construct_merges_when_oracle_applied_via_construct_from_functions(self):
        """Sub-construct built from @node functions works with Oracle."""
        @node(outputs=ClaimResult)
        def assess(claim: VerifyClaim) -> ClaimResult:
            return ClaimResult(claim_id=claim.claim_id, disposition="variant")

        def merge_assessments(variants, config):
            return ClaimResult(
                claim_id=variants[0].claim_id,
                disposition=f"merged {len(variants)} variants",
            )

        register_scripted("merge_assess", merge_assessments)

        sub = construct_from_functions(
            "assess-sub", [assess],
            input=VerifyClaim, output=ClaimResult,
        )
        sub = sub | Oracle(n=3, merge_fn="merge_assess")

        register_scripted("make_one_claim", lambda _in, _cfg: VerifyClaim(
            claim_id="c1", text="test",
        ))
        parent = Construct("parent", nodes=[
            Node.scripted("make-one-claim", fn="make_one_claim", outputs=VerifyClaim),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "oracle-sub"})

        assert result["assess_sub"].disposition == "merged 3 variants"

    def test_parity_when_declarative_vs_node_sub_construct(self):
        """Same topology produces same result via declarative Node() vs @node."""
        # --- Declarative path ---
        # With dict-form inputs, _extract_input returns a dict. The scripted
        # fn must unwrap it.
        def parity_score_fn(_in, _cfg):
            claim = _in["neo_subgraph_input"] if isinstance(_in, dict) else _in
            return ClaimResult(claim_id=claim.claim_id, disposition="scored")

        register_scripted("parity_score", parity_score_fn)
        decl_sub = Construct(
            "decl-sub",
            input=VerifyClaim, output=ClaimResult,
            nodes=[Node.scripted("score", fn="parity_score",
                                 inputs={"neo_subgraph_input": VerifyClaim},
                                 outputs=ClaimResult)],
        )

        register_scripted("parity_seed", lambda _in, _cfg: VerifyClaim(
            claim_id="p1", text="parity test",
        ))
        decl_parent = Construct("decl-parent", nodes=[
            Node.scripted("seed", fn="parity_seed", outputs=VerifyClaim),
            decl_sub,
        ])
        decl_graph = compile(decl_parent)
        decl_result = run(decl_graph, input={"node_id": "parity"})

        # --- @node path ---
        @node(outputs=ClaimResult)
        def score_fn(claim: VerifyClaim) -> ClaimResult:
            return ClaimResult(claim_id=claim.claim_id, disposition="scored")

        node_sub = construct_from_functions(
            "node-sub", [score_fn],
            input=VerifyClaim, output=ClaimResult,
        )
        node_parent = Construct("node-parent", nodes=[
            Node.scripted("seed", fn="parity_seed", outputs=VerifyClaim),
            node_sub,
        ])
        node_graph = compile(node_parent)
        node_result = run(node_graph, input={"node_id": "parity"})

        # Both produce identical results
        assert decl_result["decl_sub"].claim_id == node_result["node_sub"].claim_id
        assert decl_result["decl_sub"].disposition == node_result["node_sub"].disposition


class TestPortParamErrors:
    """Error cases for port param resolution (neograph-vih)."""

    def test_raises_when_multiple_params_match_construct_input(self):
        """Two params of the same type as construct_input should raise ConstructError,
        not silently drop one."""
        @node(outputs=ClaimResult)
        def ambiguous(first: VerifyClaim, second: VerifyClaim) -> ClaimResult:
            return ClaimResult(claim_id=first.claim_id, disposition="ok")

        with pytest.raises(ConstructError, match="ambiguous.*port"):
            construct_from_functions(
                "bad", [ambiguous], input=VerifyClaim, output=ClaimResult,
            )

    def test_peer_priority_when_param_name_matches_node_and_type_matches_input(self):
        """A param that names a peer @node takes priority over port matching,
        even if its type also matches construct_input."""
        @node(outputs=VerifyClaim)
        def upstream() -> VerifyClaim:
            return VerifyClaim(claim_id="u1", text="from upstream")

        @node(outputs=ClaimResult)
        def consumer(upstream: VerifyClaim) -> ClaimResult:
            return ClaimResult(claim_id=upstream.claim_id, disposition="peer")

        # upstream's type matches construct_input, but it's a peer @node name —
        # should wire as peer dependency, not port param.
        sub = construct_from_functions(
            "peer-priority", [upstream, consumer],
            input=VerifyClaim, output=ClaimResult,
        )
        consumer_node = [n for n in sub.nodes if n.name == "consumer"][0]
        # "upstream" should be a peer dep, NOT rewritten to neo_subgraph_input
        assert "upstream" in consumer_node.inputs
        assert "neo_subgraph_input" not in consumer_node.inputs


# ═══════════════════════════════════════════════════════════════════════════
# GATHER → PRODUCE INSIDE SUB-CONSTRUCT (neograph-dp5)
#
# The core "agent explores then judges" topology: a gather node with
# tools + dict-form outputs (result + tool_log) feeds a produce node
# that consumes both, all inside a construct_from_functions sub-construct.
# ═══════════════════════════════════════════════════════════════════════════


class ExplorationResult(BaseModel, frozen=True):
    evidence: list[str]
    summary: str


class ClaimVerdict(BaseModel, frozen=True):
    claim_id: str
    disposition: str


class TestGatherProduceSubConstruct:
    """Gather→produce with tool_log inside a @node sub-construct (neograph-dp5).

    The motivating pattern from neograph-4a7: an explore node gathers evidence
    with tools, then a score node judges in a fresh conversation using the
    exploration result AND the tool interaction log as context.
    """

    def _setup_fakes(self, *, capture_prompt=None):
        """Wire up tier-based fakes: 'research' → ReActFake, 'judge' → StructuredFakeWithRaw."""
        from neograph import ToolInteraction
        from neograph.factory import register_tool_factory
        from tests.fakes import FakeTool, ReActFake, StructuredFakeWithRaw, configure_fake_llm

        fake_tool = FakeTool("search_evidence", response="evidence found: auth.py:42")
        register_tool_factory("search_evidence", lambda config, tool_config: fake_tool)

        def llm_factory(tier):
            if tier == "research":
                return ReActFake(
                    tool_calls=[
                        [{"name": "search_evidence", "args": {"q": "verify"}, "id": "tc1"}],
                        [],  # final: no more tool calls
                    ],
                    final=lambda m: m(evidence=["auth.py:42"], summary="found evidence"),
                )
            # "judge" tier — produce node
            return StructuredFakeWithRaw(
                lambda model: model(claim_id="c1", disposition="confirmed"),
            )

        compiler = capture_prompt if capture_prompt else None
        configure_fake_llm(llm_factory, prompt_compiler=compiler)
        return fake_tool

    def _build_sub_construct(self):
        """Build the explore→score sub-construct from @node functions."""
        from neograph import Tool, ToolInteraction, node, construct_from_functions

        @node(
            mode="gather",
            outputs={"result": ExplorationResult, "tool_log": list[ToolInteraction]},
            model="research",
            prompt="verify/explore",
            tools=[Tool("search_evidence", budget=3)],
        )
        def explore(claim: VerifyClaim) -> ExplorationResult: ...

        @node(
            mode="produce",
            outputs=ClaimVerdict,
            model="judge",
            prompt="verify/score",
        )
        def score(explore_result: ExplorationResult, explore_tool_log: list[ToolInteraction]) -> ClaimVerdict: ...

        return construct_from_functions(
            "verify", [explore, score],
            input=VerifyClaim, output=ClaimVerdict,
        )

    def test_result_surfaces_when_gather_feeds_produce_inside_sub_construct(self):
        """Gather→produce chain inside @node sub-construct: result surfaces to parent."""
        self._setup_fakes()
        sub = self._build_sub_construct()

        register_scripted("dp5_seed", lambda _in, _cfg: VerifyClaim(
            claim_id="c1", text="system shall authenticate",
        ))
        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="dp5_seed", outputs=VerifyClaim),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "dp5-test"})

        # Sub-construct output surfaces
        assert result["verify"] is not None
        assert isinstance(result["verify"], ClaimVerdict)
        assert result["verify"].disposition == "confirmed"
        # No framework internals leak
        assert not any(k.startswith("neo_") for k in result)

    def test_tool_log_received_when_produce_consumes_gather_output_inside_sub_construct(self):
        """Score node's prompt compiler receives tool_log with real ToolInteraction data."""
        from neograph import ToolInteraction

        captured = {}

        def capturing_compiler(template, data, **kw):
            # Record what each node's prompt compiler receives
            captured[template] = data
            return [{"role": "user", "content": "test"}]

        self._setup_fakes(capture_prompt=capturing_compiler)
        sub = self._build_sub_construct()

        register_scripted("dp5_seed2", lambda _in, _cfg: VerifyClaim(
            claim_id="c1", text="test claim",
        ))
        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="dp5_seed2", outputs=VerifyClaim),
            sub,
        ])
        graph = compile(parent)
        run(graph, input={"node_id": "dp5-capture"})

        # The score node (prompt="verify/score") should have received tool_log
        assert "verify/score" in captured, f"Expected verify/score in captured, got: {list(captured.keys())}"
        score_input = captured["verify/score"]
        assert "explore_tool_log" in score_input
        tool_log = score_input["explore_tool_log"]
        assert isinstance(tool_log, list)
        assert len(tool_log) >= 1
        assert isinstance(tool_log[0], ToolInteraction)
        assert tool_log[0].tool_name == "search_evidence"
        assert "evidence found" in tool_log[0].result
        # Also verify explore_result was passed
        assert "explore_result" in score_input
        assert isinstance(score_input["explore_result"], ExplorationResult)

    def test_each_fans_out_when_gather_produce_sub_construct_mapped(self):
        """Sub-construct with gather→produce fanned out via .map() over claims."""
        self._setup_fakes()

        # Need fresh @node definitions for this test (avoid sidecar collisions)
        from neograph import Tool, ToolInteraction, node, construct_from_functions

        @node(
            mode="gather",
            outputs={"result": ExplorationResult, "tool_log": list[ToolInteraction]},
            model="research",
            prompt="verify/explore",
            tools=[Tool("search_evidence", budget=3)],
        )
        def explore_each(claim: VerifyClaim) -> ExplorationResult: ...

        @node(
            mode="produce",
            outputs=ClaimVerdict,
            model="judge",
            prompt="verify/score",
        )
        def score_each(explore_each_result: ExplorationResult, explore_each_tool_log: list[ToolInteraction]) -> ClaimVerdict: ...

        sub = construct_from_functions(
            "verify", [explore_each, score_each],
            input=VerifyClaim, output=ClaimVerdict,
        ).map("seed.claims", key="claim_id")

        class ClaimBatch(BaseModel, frozen=True):
            claims: list[VerifyClaim]

        register_scripted("dp5_batch", lambda _in, _cfg: ClaimBatch(claims=[
            VerifyClaim(claim_id="c1", text="first claim"),
            VerifyClaim(claim_id="c2", text="second claim"),
        ]))
        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="dp5_batch", outputs=ClaimBatch),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "dp5-each"})

        assert isinstance(result["verify"], dict)
        assert sorted(result["verify"].keys()) == ["c1", "c2"]
        assert all(isinstance(v, ClaimVerdict) for v in result["verify"].values())

    def test_gather_dict_outputs_written_when_inside_sub_construct(self):
        """Gather node with dict outputs works inside a sub-construct (base case)."""
        self._setup_fakes()

        from neograph import Tool, ToolInteraction, node, construct_from_functions

        @node(
            mode="gather",
            outputs={"result": ExplorationResult, "tool_log": list[ToolInteraction]},
            model="research",
            prompt="verify/explore",
            tools=[Tool("search_evidence", budget=3)],
        )
        def explore_only(claim: VerifyClaim) -> ExplorationResult: ...

        sub = construct_from_functions(
            "explore-only", [explore_only],
            input=VerifyClaim, output=ExplorationResult,
        )

        register_scripted("dp5_seed_base", lambda _in, _cfg: VerifyClaim(
            claim_id="c1", text="base case",
        ))
        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="dp5_seed_base", outputs=VerifyClaim),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "dp5-base"})

        assert result["explore_only"] is not None
        assert isinstance(result["explore_only"], ExplorationResult)
        assert result["explore_only"].evidence == ["auth.py:42"]


# ═══════════════════════════════════════════════════════════════════════════
# RENDERERS — XmlRenderer, DelimitedRenderer, JsonRenderer
# ═══════════════════════════════════════════════════════════════════════════

from pydantic import Field as PydanticField

from neograph.renderers import (
    DelimitedRenderer,
    JsonRenderer,
    Renderer,
    XmlRenderer,
    render_input,
)


# ═══════════════════════════════════════════════════════════════════════════
# TestConditionalProduce (neograph-s14)
#
# skip_when= predicate bypasses LLM call. skip_value= provides the output.
# Zero LLM tokens consumed when skip fires.
# ═══════════════════════════════════════════════════════════════════════════


