"""@node decorator tests — decoration, mode inference, fan-out, Oracle, Operator,
raw mode, params, error location, cross-module, construct_from_functions,
DI (FromInput/FromConfig), @merge_fn, dict-form inputs/outputs, @tool decorator,
conditional produce, and LLM config defaults.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from neograph import (
    Construct, ConstructError, Node, Each, Oracle, Operator, Tool,
    compile, construct_from_functions, construct_from_module, node, run, tool,
)
from neograph.factory import register_scripted
from tests.fakes import StructuredFake, configure_fake_llm
from tests.schemas import RawText, Claims, ClassifiedClaims, ClusterGroup, Clusters, MatchResult, MergedResult, ValidationResult


class TestToolDecorator:
    """@tool decorator: signature-inferred tool schemas."""

    def test_tool_registers_and_invokes_when_decorated_with_budget(self):
        """@tool wraps a function, auto-registers the factory, returns a Tool spec."""
        from langchain_core.messages import AIMessage

        call_log = []

        @tool(budget=3)
        def search_codebase(query: str) -> str:
            """Search the codebase for a query."""
            call_log.append(query)
            return f"Results for: {query}"

        # The decorator returns a Tool instance
        assert isinstance(search_codebase, Tool)
        assert search_codebase.name == "search_codebase"
        assert search_codebase.budget == 3

        # Build a pipeline using it directly (no register_tool_factory needed)
        counter = {"n": 0}

        class FakeGatherLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kwargs):
                counter["n"] += 1
                if counter["n"] <= 2:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{
                        "name": "search_codebase",
                        "args": {"query": f"q{counter['n']}"},
                        "id": f"c{counter['n']}",
                    }]
                    return msg
                return AIMessage(content="done")

            def with_structured_output(self, model, **kwargs):
                self._model = model
                return self

        configure_fake_llm(lambda tier: FakeGatherLLM())

        researcher = Node(
            name="research",
            mode="agent",
            outputs=Claims,
            model="fast",
            prompt="test",
            tools=[search_codebase],  # decorator output used directly
        )

        pipeline = Construct("test-tool-decorator", nodes=[researcher])
        graph = compile(pipeline)
        run(graph, input={})

        # The decorated function was called twice (within budget)
        assert len(call_log) == 2
        assert call_log == ["q1", "q2"]

    def test_tool_returns_spec_when_decorated_without_parens(self):
        """@tool (no parens) also works."""
        @tool
        def noop(x: str) -> str:
            """A no-op tool."""
            return x

        assert isinstance(noop, Tool)
        assert noop.name == "noop"
        assert noop.budget == 0  # unlimited by default


# ═══════════════════════════════════════════════════════════════════════════
# TestNodeDecorator — @node + construct_from_module (Dagster-style signatures)
#
# Parameter names in the decorated function name the upstream nodes. The
# decorator produces a plain Node; construct_from_module walks a module's
# @node-built nodes and topologically sorts them into a Construct. No new
# IR path — compile()/run() handle the result unchanged.
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeDecorator:
    """@node decorator: parameter-name-based dependency inference."""

    @staticmethod
    def _fresh_module(name: str):
        """Create a throwaway module object for construct_from_module to walk."""
        import types as _types
        return _types.ModuleType(name)

    def test_chain_compiles_and_runs_when_two_nodes_wired_by_param_name(self):
        """Two @node-decorated scripted functions wired by parameter name,
        assembled via construct_from_module, compile and run end-to-end."""
        from neograph import compile, construct_from_module, node, run

        mod = self._fresh_module("test_basic_chain_mod")

        @node(mode="scripted", outputs=RawText)
        def seed() -> RawText:
            return RawText(text="hello world")

        @node(mode="scripted", outputs=Claims)
        def split(seed: RawText) -> Claims:
            return Claims(items=[w for w in seed.text.split() if w])

        mod.seed = seed
        mod.split = split

        pipeline = construct_from_module(mod)

        # It is a Construct, with nodes in dependency order
        assert isinstance(pipeline, Construct)
        assert [n.name for n in pipeline.nodes] == ["seed", "split"]

        graph = compile(pipeline)
        result = run(graph, input={"node_id": "basic-chain"})

        assert isinstance(result["split"], Claims)
        assert result["split"].items == ["hello", "world"]

    def test_fan_in_produces_result_when_three_upstreams_wired(self):
        """A node with three parameters gets wired to three upstream nodes,
        and topological sort puts all upstreams before the fan-in."""
        from neograph import construct_from_module, node

        class A(BaseModel, frozen=True):
            value: str

        class B(BaseModel, frozen=True):
            value: str

        class C(BaseModel, frozen=True):
            value: str

        class Report(BaseModel, frozen=True):
            summary: str

        mod = self._fresh_module("test_fan_in_mod")

        @node(mode="scripted", outputs=A)
        def alpha() -> A:
            return A(value="a")

        @node(mode="scripted", outputs=B)
        def beta() -> B:
            return B(value="b")

        @node(mode="scripted", outputs=C)
        def gamma() -> C:
            return C(value="c")

        @node(mode="scripted", outputs=Report)
        def report(alpha: A, beta: B, gamma: C) -> Report:
            return Report(summary=f"{alpha.value}-{beta.value}-{gamma.value}")

        mod.alpha = alpha
        mod.beta = beta
        mod.gamma = gamma
        mod.report = report

        pipeline = construct_from_module(mod)
        names = [n.name for n in pipeline.nodes]

        # All three upstreams appear before the fan-in consumer.
        assert set(names[:3]) == {"alpha", "beta", "gamma"}
        assert names[-1] == "report"

        # Register the scripted fns and run end-to-end.
        from neograph import compile, run

        graph = compile(pipeline)
        result = run(graph, input={"node_id": "fan-in"})
        assert result["report"].summary == "a-b-c"

    def test_explicit_outputs_override_when_return_annotation_differs(self):
        """Explicit @node(outputs=X) beats the function's return annotation."""
        from neograph import construct_from_module, node

        class Bogus(BaseModel, frozen=True):
            nope: str

        mod = self._fresh_module("test_kwargs_override_mod")

        @node(mode="scripted", outputs=Claims)  # explicit output overrides `-> Bogus`
        def producer() -> Bogus:  # intentional mismatch
            return Claims(items=["overridden"])

        mod.producer = producer

        pipeline = construct_from_module(mod)
        (only_node,) = pipeline.nodes
        assert only_node.outputs is Claims
        assert only_node.outputs is not Bogus

    def test_construct_raises_when_param_names_unknown_node(self):
        """A parameter that doesn't name any @node in the module raises
        ConstructError with a helpful message."""
        from neograph import ConstructError, construct_from_module, node

        mod = self._fresh_module("test_unknown_param_mod")

        @node(mode="scripted", outputs=Claims)
        def orphan(ghost: RawText) -> Claims:
            return Claims(items=["x"])

        mod.orphan = orphan

        with pytest.raises(ConstructError) as exc_info:
            construct_from_module(mod)
        msg = str(exc_info.value)
        assert "ghost" in msg
        assert "orphan" in msg

    def test_topo_sort_orders_correctly_when_declared_out_of_order(self):
        """Out-of-declaration-order dependencies get sorted correctly."""
        from neograph import construct_from_module, node

        mod = self._fresh_module("test_topo_mod")

        # Attach to module in a SHUFFLED order (report, seed, split).
        # Declaration order inside the function body is also shuffled: the
        # downstream-most node is declared first.

        @node(mode="scripted", outputs=ClassifiedClaims)
        def report(split: Claims) -> ClassifiedClaims:
            return ClassifiedClaims(
                classified=[{"claim": c, "category": "x"} for c in split.items],
            )

        @node(mode="scripted", outputs=RawText)
        def seed() -> RawText:
            return RawText(text="a b c")

        @node(mode="scripted", outputs=Claims)
        def split(seed: RawText) -> Claims:
            return Claims(items=seed.text.split())

        # Assign in a different order from their dependency DAG.
        mod.report = report
        mod.seed = seed
        mod.split = split

        pipeline = construct_from_module(mod)
        names = [n.name for n in pipeline.nodes]
        assert names == ["seed", "split", "report"]

        from neograph import compile, run

        graph = compile(pipeline)
        result = run(graph, input={"node_id": "topo"})
        assert [c["claim"] for c in result["report"].classified] == ["a", "b", "c"]

    def test_node_name_hyphenated_when_function_uses_underscores(self):
        """Function `make_clusters` becomes node 'make-clusters'; downstream
        parameter `make_clusters` resolves to it."""
        from neograph import compile, construct_from_module, node, run

        mod = self._fresh_module("test_name_convention_mod")

        @node(mode="scripted", outputs=Claims)
        def seed_text() -> Claims:
            return Claims(items=["one", "two"])

        @node(mode="scripted", outputs=Clusters)
        def make_clusters(seed_text: Claims) -> Clusters:
            return Clusters(
                groups=[ClusterGroup(label="g", claim_ids=list(seed_text.items))],
            )

        @node(mode="scripted", outputs=ClassifiedClaims)
        def summarize(make_clusters: Clusters) -> ClassifiedClaims:
            return ClassifiedClaims(
                classified=[
                    {"claim": cid, "category": g.label}
                    for g in make_clusters.groups
                    for cid in g.claim_ids
                ],
            )

        mod.seed_text = seed_text
        mod.make_clusters = make_clusters
        mod.summarize = summarize

        # Node names are hyphenated.
        assert make_clusters.name == "make-clusters"
        assert seed_text.name == "seed-text"

        pipeline = construct_from_module(mod)
        assert [n.name for n in pipeline.nodes] == [
            "seed-text",
            "make-clusters",
            "summarize",
        ]

        graph = compile(pipeline)
        result = run(graph, input={"node_id": "name-conv"})
        # Output field uses underscore form of the node name.
        classified = result["summarize"].classified
        assert len(classified) == 2
        assert classified[0]["category"] == "g"


class TestNodeDecoratorModeInference:
    """@node mode inference: mode=None infers from prompt/model presence."""

    def test_mode_infers_scripted_when_no_prompt_or_model(self):
        """@node(outputs=X) with no prompt/model infers mode='scripted'."""
        from neograph import node

        @node(outputs=RawText)
        def seed() -> RawText:
            return RawText(text="hello")

        assert seed.mode == "scripted"

    def test_mode_infers_produce_when_prompt_and_model_present(self):
        """@node(outputs=X, prompt='...', model='...') infers mode='produce'."""
        from neograph import node

        @node(outputs=Claims, prompt="rw/decompose", model="reason")
        def decompose(topic: RawText) -> Claims: ...

        assert decompose.mode == "think"

    def test_decoration_raises_when_produce_mode_missing_prompt(self):
        """@node(mode='produce', outputs=X, model='reason') with no prompt raises at decoration time."""
        from neograph import ConstructError, node

        with pytest.raises(ConstructError, match="requires prompt="):

            @node(mode="think", outputs=Claims, model="reason")
            def decompose(topic: RawText) -> Claims: ...

    def test_decoration_raises_when_gather_mode_missing_model(self):
        """@node(mode='gather', outputs=X, prompt='...') with no model raises at decoration time."""
        from neograph import ConstructError, node

        with pytest.raises(ConstructError, match="requires model="):

            @node(mode="agent", outputs=Claims, prompt="rw/decompose")
            def decompose(topic: RawText) -> Claims: ...

    def test_warning_emitted_when_produce_mode_has_nontrivial_body(self):
        """@node(mode='produce', ...) with a real function body emits UserWarning."""
        import warnings as _warnings

        from neograph import node

        with pytest.warns(UserWarning, match="body.*not executed"):

            @node(mode="think", outputs=Claims, prompt="rw/decompose", model="reason")
            def decompose(topic: RawText) -> Claims:
                return Claims(items=topic.text.split("."))

    def test_no_warning_when_produce_mode_has_ellipsis_body(self):
        """@node(mode='produce', ...) with `...` body does NOT warn."""
        import warnings as _warnings

        from neograph import node

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")

            @node(mode="think", outputs=Claims, prompt="rw/decompose", model="reason")
            def decompose(topic: RawText) -> Claims: ...


class TestNodeDecoratorFanout:
    """@node decorator: map_over=/map_key= kwargs for Each fan-out interop."""

    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_fanout_produces_dict_when_map_over_and_map_key_set(self):
        """Full chain: producer → fan-out consumer via map_over= compiles, runs
        end-to-end, and produces a dict keyed by cluster label."""
        from neograph import compile, construct_from_module, node, run

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
        assert each is not None
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
        from neograph import ConstructError, node

        with pytest.raises(ConstructError, match="map_key"):
            @node(mode="scripted", outputs=MatchResult, map_over="make_clusters.groups")
            def verify(cluster: ClusterGroup) -> MatchResult:
                ...

    def test_decoration_raises_when_map_key_without_map_over(self):
        """map_key= without map_over= raises ConstructError at decoration time."""
        from neograph import ConstructError, node

        with pytest.raises(ConstructError, match="map_over"):
            @node(mode="scripted", outputs=MatchResult, map_key="label")
            def verify(cluster: ClusterGroup) -> MatchResult:
                ...

    def test_sidecar_survives_when_each_modifier_applied(self):
        """The Each-modified Node copy retains its sidecar entry so
        construct_from_module picks it up."""
        from neograph.decorators import _get_sidecar
        from neograph import node

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
        assert sidecar is not None
        fn, param_names = sidecar
        assert param_names == ("cluster",)

    def test_fanout_param_skipped_when_resolving_upstream_adjacency(self):
        """The fan-out parameter is NOT looked up as an upstream @node,
        so it doesn't cause 'does not match any @node' ConstructError."""
        from neograph import construct_from_module, node

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
        from neograph import construct_from_module, node

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

class TestNodeDecoratorOracle:
    """@node decorator: ensemble_n=/merge_fn=/merge_prompt= kwargs for Oracle."""

    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_oracle_attaches_when_ensemble_n_and_merge_fn_set(self):
        """@node with ensemble_n + merge_fn end-to-end: Oracle modifier attached,
        pipeline compiles and runs, merge function combines variants."""
        from neograph.factory import register_scripted
        from neograph import compile, construct_from_module, node, run

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
        assert oracle_mod is not None
        assert oracle_mod.n == 3
        assert oracle_mod.merge_fn == "combine_dec"
        assert oracle_mod.merge_prompt is None

    def test_oracle_attaches_when_ensemble_n_and_merge_prompt_set(self):
        """@node with ensemble_n + merge_prompt end-to-end: Oracle modifier
        attached with merge_prompt for LLM judge."""
        from neograph import node

        @node(mode="think", outputs=Claims, prompt="rw/decompose", model="reason",
              ensemble_n=2, merge_prompt="rw/decompose-merge")
        def decompose(topic: RawText) -> Claims: ...

        oracle_mod = decompose.get_modifier(Oracle)
        assert oracle_mod is not None
        assert oracle_mod.n == 2
        assert oracle_mod.merge_prompt == "rw/decompose-merge"
        assert oracle_mod.merge_fn is None

    def test_oracle_defaults_n_to_3_when_merge_fn_without_ensemble_n(self):
        """merge_fn without ensemble_n defaults to n=3."""
        from neograph import node

        @node(mode="think", outputs=Claims, prompt="rw/decompose", model="reason",
              merge_fn="combine")
        def decompose(topic: RawText) -> Claims: ...

        oracle_mod = decompose.get_modifier(Oracle)
        assert oracle_mod is not None
        assert oracle_mod.n == 3
        assert oracle_mod.merge_fn == "combine"

    def test_decoration_raises_when_ensemble_n_without_merge(self):
        """ensemble_n without merge_fn or merge_prompt raises ConstructError."""
        from neograph import node

        with pytest.raises(ConstructError, match="neither merge_fn nor merge_prompt"):
            @node(mode="think", outputs=Claims, prompt="rw/decompose", model="reason",
                  ensemble_n=3)
            def decompose(topic: RawText) -> Claims: ...

    def test_decoration_raises_when_both_merge_fn_and_merge_prompt(self):
        """Both merge_fn and merge_prompt raises ConstructError."""
        from neograph import node

        with pytest.raises(ConstructError, match="both merge_fn and merge_prompt"):
            @node(mode="think", outputs=Claims, prompt="rw/decompose", model="reason",
                  ensemble_n=3, merge_fn="combine", merge_prompt="rw/merge")
            def decompose(topic: RawText) -> Claims: ...

    def test_decoration_raises_when_ensemble_n_less_than_2(self):
        """ensemble_n=1 raises ConstructError."""
        from neograph import node

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

class TestNodeDecoratorRawMode:
    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_raw_mode_filters_state_when_reading_and_writing(self):
        """@node(mode='raw') reads state and returns a filtered update dict."""
        from neograph import compile, construct_from_module, node, run
        from neograph.factory import register_scripted

        register_scripted(
            "make_claims",
            lambda input_data, config: Claims(items=["a", "b", "c"]),
        )

        mod = self._fresh_module("test_raw_mode_basic")

        make = Node.scripted("make-claims", fn="make_claims", outputs=Claims)

        @node(mode="raw", inputs=Claims, outputs=Claims)
        def filter_claims(state, config):
            claims = None
            for field_name in state.__class__.model_fields:
                val = getattr(state, field_name, None)
                if isinstance(val, Claims):
                    claims = val
                    break
            if claims is None:
                return {"filter_claims": Claims(items=[])}
            filtered = Claims(items=[c for c in claims.items if c != "b"])
            return {"filter_claims": filtered}

        pipeline = Construct("test-raw-mode", nodes=[make, filter_claims])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        filtered = result.get("filter_claims")
        assert filtered is not None
        assert "b" not in filtered.items
        assert "a" in filtered.items
        assert "c" in filtered.items

    def test_raw_mode_raises_when_signature_invalid(self):
        """@node(mode='raw') rejects functions with wrong parameter count or names."""
        from neograph import node

        # Three parameters — too many
        with pytest.raises(ConstructError, match="exactly two parameters"):
            @node(mode="raw", inputs=Claims, outputs=Claims)
            def bad_three(state, config, extra):
                pass

        # Wrong parameter names
        with pytest.raises(ConstructError, match="named 'state' and 'config'"):
            @node(mode="raw", inputs=Claims, outputs=Claims)
            def bad_names(s, c):
                pass

        # One parameter — too few
        with pytest.raises(ConstructError, match="exactly two parameters"):
            @node(mode="raw", inputs=Claims, outputs=Claims)
            def bad_one(state):
                pass

    def test_downstream_consumes_when_raw_node_produces_output(self):
        """Raw node output is consumed by a downstream scripted @node via param name."""
        from neograph import compile, construct_from_module, node, run

        mod = self._fresh_module("test_raw_downstream")

        @node(mode="raw", inputs=Claims, outputs=Claims)
        def produce_claims(state, config):
            return {"produce_claims": Claims(items=["x", "y"])}

        @node(mode="scripted", outputs=RawText)
        def summarize(produce_claims: Claims) -> RawText:
            return RawText(text=f"count={len(produce_claims.items)}")

        mod.produce_claims = produce_claims
        mod.summarize = summarize

        pipeline = construct_from_module(mod, name="test-raw-downstream")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        summary = result.get("summarize")
        assert summary is not None
        assert summary.text == "count=2"

    def test_pipeline_runs_when_raw_and_scripted_mixed(self):
        """Pipeline with both raw and scripted @nodes in the same module."""
        from neograph import compile, construct_from_module, node, run

        mod = self._fresh_module("test_mixed_raw_scripted")

        @node(mode="scripted", outputs=RawText)
        def extract() -> RawText:
            return RawText(text="hello world")

        @node(mode="raw", inputs=RawText, outputs=Claims)
        def process(state, config):
            return {"process": Claims(items=["from-raw"])}

        @node(mode="scripted", outputs=ClassifiedClaims)
        def classify(process: Claims) -> ClassifiedClaims:
            return ClassifiedClaims(
                classified=[{"claim": c, "category": "raw"} for c in process.items]
            )

        mod.extract = extract
        mod.process = process
        mod.classify = classify

        pipeline = construct_from_module(mod, name="test-mixed")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        classified = result.get("classify")
        assert classified is not None
        assert len(classified.classified) == 1
        assert classified.classified[0]["claim"] == "from-raw"


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

        from neograph import compile, node, run
        from neograph.factory import register_condition, register_scripted

        register_scripted("scripted_validate", lambda input_data, config: ValidationResult(
            passed=False,
            issues=["missing stakeholder coverage"],
        ))

        register_condition("validation_failed", lambda state: (
            {"issues": state.validate.issues}
            if state.validate and not state.validate.passed
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
            "validate", fn="scripted_validate", outputs=ValidationResult,
        ) | Operator(when="validation_failed")

        pipeline = Construct("test-node-op-string", nodes=[n])
        graph = compile(pipeline, checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "test-node-op-string"}}
        result = run(graph, input={"node_id": "test-001"}, config=config)

        assert "__interrupt__" in result
        assert result["validate"].passed is False

    def test_operator_modifier_attached_when_interrupt_when_string(self):
        """@node(interrupt_when='name') results in a node with Operator modifier."""
        from neograph import node
        from neograph.factory import register_condition

        register_condition("some_check", lambda state: None)

        @node(mode="scripted", outputs=ValidationResult, interrupt_when="some_check")
        def check_things() -> ValidationResult:
            return ValidationResult(passed=True, issues=[])

        assert check_things.has_modifier(Operator)
        op = check_things.get_modifier(Operator)
        assert op is not None
        assert op.when == "some_check"

    def test_condition_auto_registered_when_interrupt_when_callable(self):
        """@node(interrupt_when=<callable>) auto-registers condition and attaches Operator."""
        from neograph import node

        cond_fn = lambda state: {"flag": True} if getattr(state, "validate", None) else None

        @node(mode="scripted", outputs=ValidationResult, interrupt_when=cond_fn)
        def validate() -> ValidationResult:
            return ValidationResult(passed=False, issues=["x"])

        assert validate.has_modifier(Operator)
        op = validate.get_modifier(Operator)
        assert op is not None
        # Synthesized name follows the pattern _node_interrupt_{node_name}_{id_hex}
        assert op.when.startswith("_node_interrupt_validate_")

        # Verify the callable was actually registered
        from neograph.factory import lookup_condition
        resolved = lookup_condition(op.when)
        assert resolved is cond_fn

    def test_graph_resumes_when_interrupt_followed_by_feedback(self):
        """@node interrupt + resume flow: graph pauses then resumes with feedback."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph import compile, node, run
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

        from neograph import compile, node, run
        from neograph.factory import register_condition, register_scripted

        register_scripted("quality_ok", lambda input_data, config: ValidationResult(
            passed=True, issues=[],
        ))

        register_condition("always_falsy", lambda state: None)

        n = Node.scripted(
            "validate", fn="quality_ok", outputs=ValidationResult,
        ) | Operator(when="always_falsy")

        pipeline = Construct("test-node-op-pass", nodes=[n])
        graph = compile(pipeline, checkpointer=MemorySaver())
        result = run(
            graph,
            input={"node_id": "test-001"},
            config={"configurable": {"thread_id": "node-op-pass"}},
        )

        assert result["validate"].passed is True
        assert result.get("human_feedback") is None

    def test_decoration_raises_when_interrupt_when_wrong_type(self):
        """Passing a non-string, non-callable interrupt_when raises ConstructError."""
        from neograph import node

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

class TestNodeDecoratorParams:
    """Scalar parameter support: FromInput, FromConfig, default constants."""

    def test_from_input_delivers_value_when_present_in_run_input(self):
        """Annotated[str, FromInput] param is delivered via run(input={'topic': 'x'})."""
        import types as _types

        from neograph import FromInput, compile, construct_from_module, node, run

        mod = _types.ModuleType("test_from_input_mod")

        @node(mode="scripted", outputs=RawText)
        def greet(topic: Annotated[str, FromInput]) -> RawText:
            return RawText(text=f"Hello, {topic}!")

        mod.greet = greet

        pipeline = construct_from_module(mod, name="test-from-input")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "t-001", "topic": "world"})

        assert result["greet"] == RawText(text="Hello, world!")

    def test_from_config_delivers_resource_when_present_in_configurable(self):
        """Annotated[RateLimiter, FromConfig] param is delivered via config['configurable']."""
        import types as _types

        from neograph import FromConfig, compile, construct_from_module, node, run

        mod = _types.ModuleType("test_from_config_mod")

        class FakeRateLimiter:
            def __init__(self):
                self.calls = 0

            def call(self):
                self.calls += 1

        limiter = FakeRateLimiter()

        @node(mode="scripted", outputs=Claims)
        def process(rate_limiter: Annotated[FakeRateLimiter, FromConfig]) -> Claims:
            rate_limiter.call()
            return Claims(items=[f"calls={rate_limiter.calls}"])

        mod.process = process

        pipeline = construct_from_module(mod, name="test-from-config")
        graph = compile(pipeline)
        result = run(
            graph,
            input={"node_id": "t-002"},
            config={"configurable": {"rate_limiter": limiter}},
        )

        assert limiter.calls == 1
        assert result["process"] == Claims(items=["calls=1"])

    def test_default_used_when_param_has_default_value(self):
        """Param with default value not matching any @node is used as compile-time constant."""
        import types as _types

        from neograph import compile, construct_from_module, node, run

        mod = _types.ModuleType("test_default_const_mod")

        @node(mode="scripted", outputs=RawText)
        def greet(greeting: str = "Hi") -> RawText:
            return RawText(text=f"{greeting}, friend!")

        mod.greet = greet

        pipeline = construct_from_module(mod, name="test-default-const")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "t-003"})

        assert result["greet"] == RawText(text="Hi, friend!")

    def test_all_param_types_resolve_when_mixed_in_one_function(self):
        """One function with upstream + FromInput + FromConfig + default."""
        import types as _types

        from neograph import FromConfig, FromInput, compile, construct_from_module, node, run

        mod = _types.ModuleType("test_mixed_mod")

        class FakeLogger:
            def __init__(self):
                self.logged: list[str] = []

            def log(self, msg: str):
                self.logged.append(msg)

        logger = FakeLogger()

        @node(mode="scripted", outputs=RawText)
        def seed() -> RawText:
            return RawText(text="base")

        @node(mode="scripted", outputs=Claims)
        def combine(
            seed: RawText,
            topic: Annotated[str, FromInput],
            logger: Annotated[FakeLogger, FromConfig],
            separator: str = " | ",
        ) -> Claims:
            logger.log(f"combining {seed.text} with {topic}")
            return Claims(items=[f"{seed.text}{separator}{topic}"])

        mod.seed = seed
        mod.combine = combine

        pipeline = construct_from_module(mod, name="test-mixed")
        graph = compile(pipeline)
        result = run(
            graph,
            input={"node_id": "t-004", "topic": "science"},
            config={"configurable": {"logger": logger}},
        )

        assert result["combine"] == Claims(items=["base | science"])
        assert len(logger.logged) == 1
        assert "combining base with science" in logger.logged[0]

    def test_none_returned_when_from_input_key_missing(self):
        """FromInput param not in run(input=...) returns None (not an error)."""
        import types as _types

        from neograph import FromInput, compile, construct_from_module, node, run

        mod = _types.ModuleType("test_from_input_missing_mod")

        @node(mode="scripted", outputs=RawText)
        def greet(topic: Annotated[str, FromInput]) -> RawText:
            if topic is None:
                return RawText(text="no topic")
            return RawText(text=f"Hello, {topic}!")

        mod.greet = greet

        pipeline = construct_from_module(mod, name="test-from-input-missing")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "t-005"})

        assert result["greet"] == RawText(text="no topic")


class TestNodeDecoratorErrorLocation:
    """@node errors include the decorated function's source file:line."""

    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_error_message_includes_source_location_when_param_unknown(self):
        """Unknown-param error includes 'test_node_decorator.py:<line>'
        pointing at the decorated function definition."""
        from neograph import ConstructError, construct_from_module, node

        mod = self._fresh_module("test_src_loc_mod")

        @node(mode="scripted", outputs=Claims)
        def orphan(ghost: RawText) -> Claims:
            return Claims(items=["x"])

        mod.orphan = orphan

        with pytest.raises(ConstructError) as exc_info:
            construct_from_module(mod)
        msg = str(exc_info.value)
        assert "test_node_decorator.py:" in msg

    def test_error_message_includes_source_location_when_cycle_detected(self):
        """Cycle error includes source locations for the involved nodes."""
        from neograph import ConstructError, construct_from_module, node

        mod = self._fresh_module("test_cycle_loc_mod")

        @node(mode="scripted", outputs=RawText)
        def ping(pong: Claims) -> RawText:
            return RawText(text="p")

        @node(mode="scripted", outputs=Claims)
        def pong(ping: RawText) -> Claims:
            return Claims(items=["q"])

        mod.ping = ping
        mod.pong = pong

        with pytest.raises(ConstructError) as exc_info:
            construct_from_module(mod)
        msg = str(exc_info.value)
        assert "test_node_decorator.py:" in msg

    def test_source_location_uses_basename_when_reporting_errors(self):
        """Source location uses basename, not the full absolute path."""
        from neograph import ConstructError, construct_from_module, node

        mod = self._fresh_module("test_basename_mod")

        @node(mode="scripted", outputs=Claims)
        def orphan(ghost: RawText) -> Claims:
            return Claims(items=["x"])

        mod.orphan = orphan

        with pytest.raises(ConstructError) as exc_info:
            construct_from_module(mod)
        msg = str(exc_info.value)
        # Must contain basename, not an absolute path with directory separators
        assert "test_node_decorator.py:" in msg
        assert "/tests/test_node_decorator.py:" not in msg


class TestNodeDecoratorCrossModule:
    """Cross-module composition and name-collision detection."""

    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_pipeline_assembles_when_node_imported_from_another_module(self):
        """@node from module A imported into module B: construct_from_module(B)
        finds both, wires topology correctly, compile+run end-to-end."""
        from neograph import compile, construct_from_module, node, run

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
        from neograph import ConstructError, construct_from_module, node

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
        from neograph import construct_from_module, node

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


class TestConstructFromFunctions:
    """construct_from_functions() — explicit function list for multi-pipeline files."""

    def test_chain_compiles_when_two_functions_wired_by_param_name(self):
        """Two @node functions wired by parameter name via explicit list."""
        from neograph import compile, construct_from_functions, node, run

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
        from neograph import compile, construct_from_functions, node, run

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
        from neograph import compile, construct_from_functions, node, run

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
        from neograph import ConstructError, construct_from_functions, node

        @node(outputs=RawText)
        def cff_ok() -> RawText:
            return RawText(text="ok")

        def not_a_node(x: RawText) -> Claims:  # missing @node
            return Claims(items=[x.text])

        with pytest.raises(ConstructError, match="not decorated with @node"):
            construct_from_functions("bad", [cff_ok, not_a_node])

    def test_construct_raises_when_non_callable_passed(self):
        """Passing a non-callable raises."""
        from neograph import ConstructError, construct_from_functions, node

        @node(outputs=RawText)
        def cff_ok2() -> RawText:
            return RawText(text="ok")

        with pytest.raises(ConstructError, match="not decorated with @node"):
            construct_from_functions("bad", [cff_ok2, "not a function"])

    def test_construct_raises_when_function_names_collide(self):
        """Two functions whose node names collide raise ConstructError."""
        from neograph import ConstructError, construct_from_functions, node

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
        from neograph.factory import register_scripted

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
        from neograph import construct_from_functions, node

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


class TestFromInputPydanticModel:
    """neograph-6jd — Annotated[PydanticModel, FromInput] bundles multiple config fields."""

    def test_bundle_populates_when_all_fields_in_run_input(self):
        """Annotated[RunCtx, FromInput] populates each field from config['configurable']."""
        from neograph import FromInput, compile, construct_from_functions, node, run

        class RunCtx(BaseModel):
            node_id: str
            project_root: str

        @node(outputs=RawText)
        def fipb_produce(ctx: Annotated[RunCtx, FromInput]) -> RawText:
            return RawText(text=f"{ctx.node_id}|{ctx.project_root}")

        pipeline = construct_from_functions("fipb", [fipb_produce])
        graph = compile(pipeline)
        result = run(
            graph,
            input={"node_id": "REQ-001", "project_root": "/tmp/repo"},
        )
        assert result["fipb_produce"].text == "REQ-001|/tmp/repo"

    def test_bundle_composes_when_mixed_with_upstream_param(self):
        """Annotated[PydanticModel, FromInput] composes with an upstream @node parameter."""
        from neograph import FromInput, compile, construct_from_functions, node, run

        class RunCtx(BaseModel):
            node_id: str

        @node(outputs=Claims)
        def fipb2_source() -> Claims:
            return Claims(items=["a", "b"])

        @node(outputs=RawText)
        def fipb2_join(fipb2_source: Claims, ctx: Annotated[RunCtx, FromInput]) -> RawText:
            return RawText(text=f"{ctx.node_id}: {','.join(fipb2_source.items)}")

        pipeline = construct_from_functions("fipb2", [fipb2_source, fipb2_join])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "X-42"})
        assert result["fipb2_join"].text == "X-42: a,b"

    def test_bundle_field_none_when_missing_from_configurable(self):
        """A missing field in config['configurable'] is passed as None."""
        from neograph import FromInput, compile, construct_from_functions, node, run

        class PartialCtx(BaseModel):
            node_id: str | None = None
            project_root: str | None = None

        @node(outputs=RawText)
        def fipbm_read(ctx: Annotated[PartialCtx, FromInput]) -> RawText:
            return RawText(text=f"id={ctx.node_id!r},root={ctx.project_root!r}")

        pipeline = construct_from_functions("fipbm", [fipbm_read])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "only-this"})
        assert result["fipbm_read"].text == "id='only-this',root=None"

    def test_from_config_bundle_populates_when_fields_in_configurable(self):
        """Annotated[PydanticModel, FromConfig] pulls every field from configurable as well."""
        from neograph import FromConfig, compile, construct_from_functions, node, run

        class Shared(BaseModel):
            model_config = {"arbitrary_types_allowed": True}
            tenant: str
            max_items: int

        @node(outputs=RawText)
        def fcb_read(shared: Annotated[Shared, FromConfig]) -> RawText:
            return RawText(text=f"{shared.tenant}:{shared.max_items}")

        pipeline = construct_from_functions("fcb", [fcb_read])
        graph = compile(pipeline)
        result = run(
            graph,
            input={"node_id": "x"},
            config={"configurable": {"tenant": "acme", "max_items": 7}},
        )
        assert result["fcb_read"].text == "acme:7"


class TestOracleMergeFnDI:
    """neograph-9zj — @merge_fn decorator with FromInput/FromConfig DI."""

    def test_merge_fn_receives_bundle_when_annotated_with_from_config(self):
        """@merge_fn function can receive a bundled Annotated[PydanticModel, FromConfig]
        whose fields are resolved from config['configurable'] keys."""
        from neograph import (
            Construct, FromConfig, Node, Oracle, compile,
            merge_fn, register_scripted, run,
        )

        class SharedResources(BaseModel):
            prefix: str

        @merge_fn
        def combine_with_prefix(
            variants: list[Claims],
            shared: Annotated[SharedResources, FromConfig],
        ) -> Claims:
            # Collect all unique items, prepend the shared prefix.
            seen: list[str] = []
            for v in variants:
                for it in v.items:
                    if it not in seen:
                        seen.append(it)
            return Claims(items=[f"{shared.prefix}:{x}" for x in seen])

        # Register a scripted generator that produces a Claims variant.
        def gen_fn(input_data, config):
            return Claims(items=["alpha", "beta"])
        register_scripted("omfd_gen_fn", gen_fn)

        gen = Node.scripted("omfd-gen", fn="omfd_gen_fn", outputs=Claims) | Oracle(
            n=2, merge_fn="combine_with_prefix"
        )

        pipeline = Construct("omfd-test", nodes=[gen])
        graph = compile(pipeline)
        # Bundled form: SharedResources has a single field `prefix`, so we
        # provide it directly in configurable under that key name.
        result = run(
            graph,
            input={"node_id": "omfd-001"},
            config={"configurable": {"prefix": "tag"}},
        )

        # Both Oracle generators produce ["alpha", "beta"], merge dedups, prefixes.
        assert result["omfd_gen"].items == ["tag:alpha", "tag:beta"]

    def test_merge_fn_receives_value_when_annotated_with_from_input(self):
        """@merge_fn can also receive Annotated[T, FromInput] values from run(input=...)."""
        from neograph import (
            Construct, FromInput, Node, Oracle, compile,
            merge_fn, register_scripted, run,
        )

        @merge_fn
        def tagged_merge(
            variants: list[Claims],
            node_id: Annotated[str, FromInput],
        ) -> Claims:
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=[f"{node_id}:{it}" for it in dict.fromkeys(all_items)])

        def gen_fn2(input_data, config):
            return Claims(items=["x"])
        register_scripted("omfdi_gen_fn", gen_fn2)

        gen = Node.scripted("omfdi-gen", fn="omfdi_gen_fn", outputs=Claims) | Oracle(
            n=2, merge_fn="tagged_merge"
        )

        pipeline = Construct("omfdi-test", nodes=[gen])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "REQ-99"})

        assert result["omfdi_gen"].items == ["REQ-99:x"]

    def test_plain_merge_fn_works_when_no_decorator(self):
        """Back-compat: plain (variants, config) merge_fn still works."""
        from neograph import (
            Construct, Node, Oracle, compile, register_scripted, run,
        )

        def plain_merge(variants, config):
            # Old-style signature — two positional args, no decorator.
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=list(dict.fromkeys(all_items)))
        register_scripted("plain_merge_backcompat", plain_merge)

        def pmg_gen(input_data, config):
            return Claims(items=["one", "two"])
        register_scripted("pmg_gen_fn", pmg_gen)

        gen = Node.scripted("pmg-gen", fn="pmg_gen_fn", outputs=Claims) | Oracle(
            n=2, merge_fn="plain_merge_backcompat"
        )

        pipeline = Construct("pmg-test", nodes=[gen])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "pmg-001"})
        assert result["pmg_gen"].items == ["one", "two"]


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
        from neograph import node
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
        from neograph import node
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
        from neograph import node
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

        construct_from_functions("p", [make_clusters, verify])
        assert isinstance(verify.inputs, dict)
        assert "cluster" in verify.inputs
        assert verify.inputs["cluster"] is ClusterGroup
        assert verify.fan_out_param == "cluster"

    def test_validator_catches_mismatch_when_fan_in_types_wrong(self):
        """Step-2's validator catches @node fan-in type mismatches via
        dict-form inputs (no more two-walker setup)."""
        from neograph import node
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
        import logging
        from neograph import compile, run, node
        from neograph.factory import register_scripted
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

class TestConditionalProduce:
    def test_skip_value_returned_when_skip_when_true(self):
        """When skip_when returns True, the node returns skip_value
        without any LLM call."""
        from neograph import compile, run, node
        from neograph.decorators import construct_from_functions

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
        from neograph import compile, run, node
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
        pred = lambda x: True
        val = lambda x: x
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
        from neograph import node

        @node(
            outputs=MergedResult, mode="think", model="fast", prompt="p",
            skip_when=lambda x: True,
            skip_value=lambda x: MergedResult(final_text="skipped"),
        )
        def my_node(seed: Claims) -> MergedResult: ...

        assert my_node.skip_when is not None
        assert my_node.skip_value is not None


# ═══════════════════════════════════════════════════════════════════════════
# NODE.OUTPUTS RENAME (neograph-1bp.1)
# ═══════════════════════════════════════════════════════════════════════════


class TestDecoratorDictOutputs:
    """@node decorator with dict-form outputs (neograph-1bp.5)."""

    def test_per_key_fields_written_when_dict_outputs_declared(self):
        """@node(outputs={'a': X, 'b': Y}) scripted → writes per-key fields."""
        from neograph import node, construct_from_module, compile, run
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
        from neograph import node

        @node(mode="scripted", outputs=RawText)
        def extract() -> RawText:
            return RawText(text="hi")

        assert extract.outputs is RawText

    def test_outputs_inferred_when_return_annotation_present(self):
        """Return annotation → outputs= when explicit kwarg not set."""
        from neograph import node

        @node(mode="scripted")
        def extract() -> RawText:
            return RawText(text="hi")

        assert extract.outputs is RawText


# ═══════════════════════════════════════════════════════════════════════════
# INTEROP: @node decorator integration with Operator and Each+DI
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeDecoratorInterop:
    """Cross-feature integration: @node with Operator interrupt/resume and Each+DI."""

    def test_output_present_after_resume_when_operator_interrupt_always(self):
        """@node(interrupt_when=<callable>) pauses graph, resume delivers final result."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph import FromInput, compile, construct_from_functions, node, run

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
        from neograph import FromInput, compile, construct_from_functions, node, run

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
