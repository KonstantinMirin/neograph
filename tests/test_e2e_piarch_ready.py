"""End-to-end acceptance tests: what must pass for NeoGraph to be useful for piarch.

Each test proves a real capability, not just data structure correctness.
All LLM calls use fakes — no API keys needed.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from neograph import Construct, ConstructError, Node, Operator, Oracle, Each, Tool, compile, raw_node, run, tool
from tests.fakes import FakeTool, ReActFake, StructuredFake, TextFake, configure_fake_llm


# ═══════════════════════════════════════════════════════════════════════════
# TEST SCHEMAS — shared across tests
# ═══════════════════════════════════════════════════════════════════════════

class RawText(BaseModel, frozen=True):
    text: str

class Claims(BaseModel, frozen=True):
    items: list[str]

class ClassifiedClaims(BaseModel, frozen=True):
    classified: list[dict[str, str]]  # [{claim: str, category: str}]

class ClusterGroup(BaseModel, frozen=True):
    label: str
    claim_ids: list[str]

class Clusters(BaseModel, frozen=True):
    groups: list[ClusterGroup]

class MatchResult(BaseModel, frozen=True):
    cluster_label: str
    matched: list[str]

class MergedResult(BaseModel, frozen=True):
    final_text: str

class ValidationResult(BaseModel, frozen=True):
    passed: bool
    issues: list[str]


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Scripted pipeline end-to-end
#
# Three deterministic nodes. No LLM. Data flows A→B→C through typed state.
# This proves: compile() builds a working graph, run() executes it,
# state bus passes data between nodes.
# ═══════════════════════════════════════════════════════════════════════════

class TestScriptedPipeline:
    def test_three_node_pipeline(self):
        """Data flows through 3 scripted nodes via typed state bus."""
        import types as _types

        from neograph import construct_from_module, node

        mod = _types.ModuleType("test_scripted_pipeline_mod")

        @node(mode="scripted", output=RawText)
        def extract() -> RawText:
            return RawText(text="hello world")

        @node(mode="scripted", output=Claims)
        def split(extract: RawText) -> Claims:
            return Claims(items=["claim-1", "claim-2"])

        @node(mode="scripted", output=ClassifiedClaims)
        def classify(split: Claims) -> ClassifiedClaims:
            return ClassifiedClaims(
                classified=[{"claim": c, "category": "fact"} for c in split.items]
            )

        mod.extract = extract
        mod.split = split
        mod.classify = classify

        pipeline = construct_from_module(mod, name="test-scripted")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Verify data flowed through all three nodes
        assert isinstance(result["classify"], ClassifiedClaims)
        classified = result["classify"]
        assert len(classified.classified) == 2
        assert classified.classified[0]["claim"] == "claim-1"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Produce mode — structured LLM output
#
# A node calls a (fake) LLM, gets structured Pydantic output.
# This proves: produce dispatch works, LLM integration wired,
# output validated and written to state.
# ═══════════════════════════════════════════════════════════════════════════

class TestProduceMode:
    def test_produce_node_with_fake_llm(self):
        """Produce node calls LLM and gets structured output."""
        import types as _types

        from neograph import construct_from_module, node

        configure_fake_llm(
            lambda tier: StructuredFake(
                lambda m: m(items=["extracted-1", "extracted-2", "extracted-3"])
            )
        )

        mod = _types.ModuleType("test_produce_mode_mod")

        @node(mode="produce", output=Claims, model="fast", prompt="test/extract")
        def extract() -> Claims: ...

        mod.extract = extract

        pipeline = construct_from_module(mod, name="test-produce")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        assert isinstance(result["extract"], Claims)
        assert len(result["extract"].items) == 3
        assert result["extract"].items[0] == "extracted-1"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Gather mode — ReAct loop with per-tool budgets
#
# A node with tools, budget enforced. Tool gets called, budget exhausts,
# LLM forced to respond.
# This proves: gather dispatch works, tool budget tracker enforces limits,
# exhausted tools get unbound.
# ═══════════════════════════════════════════════════════════════════════════

class TestGatherMode:
    def test_gather_with_tool_budgets(self):
        """Gather node uses tools within budget, then forced to respond."""
        import types as _types

        from neograph import construct_from_module, node
        from neograph.factory import register_tool_factory

        search_tool = FakeTool("search_nodes", response="found")
        register_tool_factory("search_nodes", lambda config, tool_config: search_tool)

        # Fake LLM: calls tool twice, then responds (budget will cap at 2 anyway)
        fake = ReActFake(
            tool_calls=[
                [{"name": "search_nodes", "args": {"query": "test"}, "id": "call-1"}],
                [{"name": "search_nodes", "args": {"query": "test"}, "id": "call-2"}],
                [],  # stop — forced response
            ],
            final=lambda m: m(items=["done researching"]),
        )
        configure_fake_llm(lambda tier: fake)

        mod = _types.ModuleType("test_gather_budget_mod")

        @node(
            mode="gather",
            output=Claims,
            model="reason",
            prompt="test/explore",
            tools=[Tool(name="search_nodes", budget=2)],
        )
        def explore() -> Claims: ...

        mod.explore = explore

        pipeline = construct_from_module(mod, name="test-gather")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Tool was called exactly twice (budget=2)
        assert len(search_tool.calls) == 2

    def test_gather_unlimited_budget(self):
        """Tool with budget=0 is never exhausted — LLM decides when to stop."""
        import types as _types

        from neograph import construct_from_module, node
        from neograph.factory import register_tool_factory

        lookup_tool = FakeTool("lookup", response="result")
        register_tool_factory("lookup", lambda config, tool_config: lookup_tool)

        fake = ReActFake(
            tool_calls=[
                [{"name": "lookup", "args": {}, "id": "c1"}],
                [{"name": "lookup", "args": {}, "id": "c2"}],
                [{"name": "lookup", "args": {}, "id": "c3"}],
                [{"name": "lookup", "args": {}, "id": "c4"}],
                [{"name": "lookup", "args": {}, "id": "c5"}],
                [],  # LLM stops on its own
            ],
            final=lambda m: m(items=["done"]),
        )
        configure_fake_llm(lambda tier: fake)

        mod = _types.ModuleType("test_gather_unlimited_mod")

        @node(
            mode="gather",
            output=Claims,
            model="fast",
            prompt="test/scan",
            tools=[Tool(name="lookup", budget=0)],  # unlimited
        )
        def scan() -> Claims: ...

        mod.scan = scan

        pipeline = construct_from_module(mod, name="test-unlimited")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Tool was called 5 times — budget=0 never blocked it
        assert len(lookup_tool.calls) == 5


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Oracle — 3-way ensemble + merge
#
# Three parallel generators produce variants, barrier merges them.
# This proves: Oracle modifier expands to fan-out Send() + merge barrier,
# all three generators run, results converge.
# ═══════════════════════════════════════════════════════════════════════════

class TestOracle:
    def test_three_way_ensemble(self):
        """Oracle dispatches 3 generators and merges results."""
        gen_ids_seen = []

        from neograph.factory import register_scripted

        # Each generator records its ID and produces a variant
        def generate_variant(input_data, config):
            gen_id = config.get("configurable", {}).get("_generator_id", "unknown")
            gen_ids_seen.append(gen_id)
            return Claims(items=[f"variant-from-{gen_id}"])

        register_scripted("generate_variant", generate_variant)

        # Merge: combine all variant items into one Claims
        def combine_variants(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("combine_variants", combine_variants)

        gen_node = Node.scripted(
            "generate", fn="generate_variant", output=Claims
        ) | Oracle(n=3, merge_fn="combine_variants")

        pipeline = Construct("test-oracle", nodes=[gen_node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # All 3 generators ran
        assert len(gen_ids_seen) == 3
        # Merge combined all 3 variants
        merged = result.get("generate")
        assert merged is not None
        assert len(merged.items) == 3

    def test_llm_merge(self):
        """Oracle with merge_prompt calls LLM to judge-merge variants."""
        from neograph.factory import register_scripted

        register_scripted("gen", lambda input_data, config: Claims(items=["v1"]))

        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["merged-consensus"])))

        gen_node = Node.scripted(
            "generate", fn="gen", output=Claims
        ) | Oracle(n=2, merge_prompt="test/merge")

        pipeline = Construct("test-oracle-llm", nodes=[gen_node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        merged = result.get("generate")
        assert merged is not None
        assert merged.items == ["merged-consensus"]

    def test_rejects_bare_oracle(self):
        """Oracle without merge_prompt or merge_fn is a ValueError."""
        with pytest.raises(ValueError, match="merge_prompt.*merge_fn"):
            Oracle(n=3)

    def test_rejects_both_merge_options(self):
        """Oracle with both merge_prompt and merge_fn is a ValueError."""
        with pytest.raises(ValueError, match="not both"):
            Oracle(n=3, merge_prompt="x", merge_fn="y")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: Each — dynamic fan-out over collection
#
# Node fans out over a list of clusters, processes each in parallel,
# results collected as dict[key, result].
# This proves: Each modifier expands to Send() per item,
# barrier collects, dict reducer merges results.
# ═══════════════════════════════════════════════════════════════════════════

class TestEach:
    def test_fanout_over_collection(self):
        """Each dispatches per-item and collects results."""
        from neograph.factory import register_scripted

        # First node produces clusters
        register_scripted("make_clusters", lambda input_data, config: Clusters(
            groups=[
                ClusterGroup(label="alpha", claim_ids=["c1", "c2"]),
                ClusterGroup(label="beta", claim_ids=["c3"]),
            ]
        ))

        # Fan-out node processes each cluster
        register_scripted("verify_cluster", lambda input_data, config: MatchResult(
            cluster_label=input_data.label if hasattr(input_data, 'label') else "unknown",
            matched=["match-1"],
        ))

        make = Node.scripted("make-clusters", fn="make_clusters", output=Clusters)
        verify = Node.scripted(
            "verify", fn="verify_cluster", input=ClusterGroup, output=MatchResult
        ) | Each(over="make_clusters.groups", key="label")

        pipeline = Construct("test-replicate", nodes=[make, verify])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Both clusters were processed — pin cardinality AND per-key payload
        verify_results = result.get("verify", {})
        assert set(verify_results.keys()) == {"alpha", "beta"}
        assert verify_results["alpha"].cluster_label == "alpha"
        assert verify_results["beta"].cluster_label == "beta"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5b: .map() — sugar over Each(over=..., key=...)
#
# Node.map() accepts a lambda introspected at definition time, or a string
# path as an escape hatch. Both compile to the same Each modifier.
# This proves: .map() is pure sugar, lambda introspection resolves attribute
# chains to dotted paths, the resulting graph runs identically to | Each(...).
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeMap:
    """Node.map() — lambda- and string-path fan-out sugar over `| Each(...)`."""

    def test_map_with_lambda_resolves_path(self):
        """A lambda `s.foo.bar` resolves to the same Each(over='foo.bar', ...)."""
        node = Node.scripted("verify", fn="noop", input=ClusterGroup, output=MatchResult)
        mapped = node.map(lambda s: s.make_clusters.groups, key="label")

        each = mapped.get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "make_clusters.groups"
        assert each.key == "label"

    def test_map_with_string_path(self):
        """A string source is passed straight through to Each.over."""
        node = Node.scripted("verify", fn="noop", input=ClusterGroup, output=MatchResult)
        mapped = node.map("make_clusters.groups", key="label")

        each = mapped.get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "make_clusters.groups"
        assert each.key == "label"

    def test_map_equivalent_to_pipe_each(self):
        """node.map(...) and node | Each(...) produce structurally identical nodes."""
        base = Node.scripted("verify", fn="noop", input=ClusterGroup, output=MatchResult)

        via_map = base.map(lambda s: s.make_clusters.groups, key="label")
        via_pipe = base | Each(over="make_clusters.groups", key="label")

        assert via_map.modifiers == via_pipe.modifiers

    def test_map_end_to_end_fanout(self):
        """.map() drives the same fan-out/collect behavior as | Each(...)."""
        from neograph.factory import register_scripted

        register_scripted("make_clusters", lambda input_data, config: Clusters(
            groups=[
                ClusterGroup(label="alpha", claim_ids=["c1", "c2"]),
                ClusterGroup(label="beta", claim_ids=["c3"]),
            ]
        ))
        register_scripted("verify_cluster", lambda input_data, config: MatchResult(
            cluster_label=input_data.label if hasattr(input_data, "label") else "unknown",
            matched=["match-1"],
        ))

        make = Node.scripted("make-clusters", fn="make_clusters", output=Clusters)
        verify = Node.scripted(
            "verify", fn="verify_cluster", input=ClusterGroup, output=MatchResult
        ).map(lambda s: s.make_clusters.groups, key="label")

        pipeline = Construct("test-map", nodes=[make, verify])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Fan-out fired for BOTH clusters — pin cardinality and payload
        verify_results = result.get("verify", {})
        assert set(verify_results.keys()) == {"alpha", "beta"}
        assert verify_results["alpha"].cluster_label == "alpha"
        assert verify_results["beta"].cluster_label == "beta"

    def test_map_lambda_with_no_attrs_raises(self):
        """`lambda s: s` has no path — clear error."""
        node = Node.scripted("verify", fn="noop", output=MatchResult)
        with pytest.raises(TypeError, match="at least one attribute"):
            node.map(lambda s: s, key="label")

    def test_map_lambda_returning_scalar_raises(self):
        """`lambda s: 42` — clear error, not a silent Each."""
        node = Node.scripted("verify", fn="noop", output=MatchResult)
        with pytest.raises(TypeError, match="attribute-access chain"):
            node.map(lambda s: 42, key="label")

    def test_map_lambda_that_errors_raises_typeerror(self):
        """A lambda that does something illegal (e.g. indexing) reports cleanly."""
        node = Node.scripted("verify", fn="noop", output=MatchResult)
        with pytest.raises(TypeError, match="pure attribute-access chain"):
            node.map(lambda s: s.items[0], key="label")  # __getitem__ on recorder

    def test_map_rejects_dunder_attribute_access(self):
        """`lambda s: s.__dict__.foo` must not silently produce Each(over='__dict__.foo')."""
        node = Node.scripted("verify", fn="noop", input=ClusterGroup, output=MatchResult)
        with pytest.raises(TypeError, match="pure attribute-access chain"):
            node.map(lambda s: s.__dict__.foo, key="label")

    def test_map_rejects_leading_underscore_attribute(self):
        """Reject `lambda s: s._private.field` — underscores are a footgun trapdoor."""
        node = Node.scripted("verify", fn="noop", input=ClusterGroup, output=MatchResult)
        with pytest.raises(TypeError, match="pure attribute-access chain"):
            node.map(lambda s: s._private.x, key="label")

    def test_map_user_exception_propagates_unchanged(self):
        """Non-attribute errors (e.g. ZeroDivisionError) propagate with their own type."""
        node = Node.scripted("verify", fn="noop", input=ClusterGroup, output=MatchResult)
        with pytest.raises(ZeroDivisionError):
            node.map(lambda s: 1 / 0 and s.x, key="label")

    def test_map_rejects_non_string_non_callable(self):
        """Passing an int or other non-source type raises immediately."""
        node = Node.scripted("verify", fn="noop", output=MatchResult)
        with pytest.raises(TypeError, match="string path or a lambda"):
            node.map(42, key="label")  # type: ignore[arg-type]

    def test_map_on_construct(self):
        """Construct also gets .map() via Modifiable — sub-construct fan-out."""
        inner = Node.scripted("inner", fn="noop", input=ClusterGroup, output=MatchResult)
        sub = Construct("sub", input=ClusterGroup, output=MatchResult, nodes=[inner])
        mapped = sub.map(lambda s: s.upstream.items, key="label")

        each = mapped.get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "upstream.items"
        assert each.key == "label"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: Operator — human-in-the-loop interrupt
#
# A node produces a validation result. If validation fails,
# the graph pauses via interrupt(). Resume with human input.
# This proves: Operator modifier wires interrupt() correctly,
# graph pauses and resumes.
# ═══════════════════════════════════════════════════════════════════════════

class TestOperator:
    def test_interrupt_on_failure(self):
        """Graph pauses when Operator condition is met."""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import interrupt

        from neograph.factory import register_condition, register_scripted

        register_scripted("validate", lambda input_data, config: ValidationResult(
            passed=False,
            issues=["missing stakeholder coverage"],
        ))

        register_condition("validation_failed", lambda state: (
            {"issues": state.validate.issues} if state.validate and not state.validate.passed else None
        ))

        validate = Node.scripted(
            "validate", fn="validate", output=ValidationResult
        ) | Operator(when="validation_failed")

        pipeline = Construct("test-operator", nodes=[validate])
        graph = compile(pipeline, checkpointer=MemorySaver())

        # Run — with checkpointer, interrupt returns result with __interrupt__
        config = {"configurable": {"thread_id": "test-interrupt"}}
        result = run(graph, input={"node_id": "test-001"}, config=config)

        # Verify the graph paused
        assert "__interrupt__" in result
        assert result["validate"].passed is False


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: Raw node alongside declarative nodes
#
# A @raw_node function mixed with Node() declarations in the same Construct.
# This proves: raw escape hatch works, framework wires edges around it,
# data flows through raw node like any other.
# ═══════════════════════════════════════════════════════════════════════════

class TestRawNode:
    def test_raw_node_in_pipeline(self):
        """@raw_node works alongside declarative nodes."""
        from neograph.factory import register_scripted

        register_scripted("make_claims", lambda input_data, config: Claims(items=["a", "b", "c"]))

        @raw_node(input=Claims, output=Claims)
        def filter_claims(state, config):
            """Raw node: custom filtering logic."""
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

        make = Node.scripted("make-claims", fn="make_claims", output=Claims)

        pipeline = Construct("test-raw", nodes=[make, filter_claims])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Raw node filtered out "b"
        filtered = result.get("filter_claims")
        assert filtered is not None
        assert "b" not in filtered.items
        assert "a" in filtered.items
        assert "c" in filtered.items


# ═══════════════════════════════════════════════════════════════════════════
# TEST 8: Realistic mini-pipeline (simplified RW)
#
# read → decompose (produce) → classify (scripted) → verify (gather+fanout)
# Mixes all modes in a single pipeline.
# This proves: heterogeneous pipelines work end-to-end.
# ═══════════════════════════════════════════════════════════════════════════

class TestMiniRWPipeline:
    def test_mixed_mode_pipeline(self):
        """Realistic pipeline mixing produce + scripted modes."""
        import types as _types

        from neograph import construct_from_module, node

        def respond(model):
            if model is Claims:
                return Claims(items=["r1: system shall validate", "r2: system shall log"])
            if model is ClassifiedClaims:
                return ClassifiedClaims(classified=[
                    {"claim": "r1", "category": "requirement"},
                    {"claim": "r2", "category": "requirement"},
                ])
            return model()

        configure_fake_llm(lambda tier: StructuredFake(respond))

        mod = _types.ModuleType("test_mini_rw_mod")

        @node(mode="produce", output=Claims, model="reason", prompt="rw/decompose")
        def decompose() -> Claims: ...

        @node(mode="produce", output=ClassifiedClaims, model="fast", prompt="rw/classify")
        def classify(decompose: Claims) -> ClassifiedClaims: ...

        @node(mode="scripted", output=RawText)
        def catalog() -> RawText:
            return RawText(text="node catalog: 42 nodes")

        mod.decompose = decompose
        mod.classify = classify
        mod.catalog = catalog

        pipeline = construct_from_module(mod, name="mini-rw")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "BR-RW-042"})

        # All three nodes produced output
        assert isinstance(result["decompose"], Claims)
        assert isinstance(result["classify"], ClassifiedClaims)
        assert isinstance(result["catalog"], RawText)
        assert len(result["decompose"].items) == 2
        assert len(result["classify"].classified) == 2
        assert "42 nodes" in result["catalog"].text


# ═══════════════════════════════════════════════════════════════════════════
# COVERAGE GAP TESTS — every remaining code path
# ═══════════════════════════════════════════════════════════════════════════


class TestExecuteMode:
    """Execute mode: ReAct tool loop with mutation tools."""

    def test_execute_with_tools(self):
        """Execute node calls tools and produces structured output."""
        import types as _types

        from neograph import construct_from_module, node
        from neograph.factory import register_tool_factory

        write_tool = FakeTool("write_file", response="written")
        register_tool_factory("write_file", lambda config, tool_config: write_tool)

        fake = ReActFake(
            tool_calls=[
                [{"name": "write_file", "args": {"path": "out.txt"}, "id": "call-1"}],
                [],  # stop
            ],
            final=lambda m: m(text="done"),
        )
        configure_fake_llm(lambda tier: fake)

        mod = _types.ModuleType("test_execute_mode_mod")

        @node(
            mode="execute",
            output=RawText,
            model="fast",
            prompt="test/write",
            tools=[Tool(name="write_file", budget=1)],
        )
        def writer() -> RawText: ...

        mod.writer = writer

        pipeline = construct_from_module(mod, name="test-execute")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        assert len(write_tool.calls) == 1
        assert write_tool.calls[0]["path"] == "out.txt"


class TestOperatorContinues:
    """Operator condition is falsy — graph continues without interrupt."""

    def test_operator_passes_when_condition_falsy(self):
        """Graph runs through Operator without pausing when condition returns None."""
        from neograph.factory import register_condition, register_scripted

        register_scripted("check_quality", lambda input_data, config: ValidationResult(
            passed=True, issues=[],
        ))

        register_condition("validation_failed", lambda state: None)  # always falsy

        validate = Node.scripted(
            "validate", fn="check_quality", output=ValidationResult
        ) | Operator(when="validation_failed")

        from langgraph.checkpoint.memory import MemorySaver

        pipeline = Construct("test-operator-pass", nodes=[validate])
        graph = compile(pipeline, checkpointer=MemorySaver())
        result = run(graph, input={"node_id": "test-001"}, config={"configurable": {"thread_id": "pass-test"}})

        assert result["validate"].passed is True
        assert result.get("human_feedback") is None


class TestOperatorResume:
    """Operator interrupt + resume flow."""

    def test_interrupt_then_resume(self):
        """Graph pauses at interrupt, resumes with human feedback via run()."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        register_scripted("validate_thing", lambda input_data, config: ValidationResult(
            passed=False, issues=["bad coverage"],
        ))

        register_condition("needs_review", lambda state: (
            {"issues": state.validate_thing.issues}
            if state.validate_thing and not state.validate_thing.passed
            else None
        ))

        validate = Node.scripted(
            "validate-thing", fn="validate_thing", output=ValidationResult
        ) | Operator(when="needs_review")

        pipeline = Construct("test-resume", nodes=[validate])
        graph = compile(pipeline, checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "resume-test"}}

        # First run: hits interrupt — returns with __interrupt__
        result = run(graph, input={"node_id": "test-001"}, config=config)
        assert "__interrupt__" in result

        # Resume via run()
        result = run(graph, resume={"approved": True}, config=config)

        assert result["validate_thing"].passed is False
        assert result["human_feedback"] == {"approved": True}


class TestErrorPaths:
    """Every error path the framework raises."""

    def test_produce_without_configure_llm(self):
        """Produce node without configure_llm() raises RuntimeError."""
        node = Node(name="fail", mode="produce", output=Claims, model="fast", prompt="x")
        pipeline = Construct("test-no-llm", nodes=[node])
        graph = compile(pipeline)

        with pytest.raises(ValueError, match="LLM not configured"):
            run(graph, input={"node_id": "test-001"})

    def test_unregistered_scripted_fn(self):
        """Referencing unregistered scripted function raises ValueError."""
        node = Node.scripted("bad", fn="nonexistent_fn", output=Claims)
        pipeline = Construct("test-bad-fn", nodes=[node])

        with pytest.raises(ValueError, match="not registered"):
            compile(pipeline)

    def test_unregistered_oracle_merge_fn(self):
        """Oracle with unregistered merge_fn raises ValueError at compile."""
        from neograph.factory import register_scripted

        register_scripted("gen", lambda input_data, config: Claims(items=["x"]))

        node = Node.scripted(
            "gen", fn="gen", output=Claims
        ) | Oracle(n=2, merge_fn="nonexistent_merge")

        pipeline = Construct("test-bad-merge", nodes=[node])

        with pytest.raises(ValueError, match="not registered"):
            compile(pipeline)

    def test_unregistered_operator_condition(self):
        """Operator with unregistered condition raises ValueError at compile."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_scripted

        register_scripted("something", lambda input_data, config: Claims(items=[]))

        node = Node.scripted(
            "something", fn="something", output=Claims
        ) | Operator(when="nonexistent_condition")

        pipeline = Construct("test-bad-condition", nodes=[node])

        with pytest.raises(ValueError, match="not registered"):
            compile(pipeline, checkpointer=MemorySaver())

    def test_unregistered_tool_factory(self):
        """Gather node with unregistered tool factory raises ValueError."""
        configure_fake_llm(lambda tier: ReActFake(tool_calls=[[]]))

        node = Node(
            name="explore",
            mode="gather",
            output=Claims,
            model="fast",
            prompt="x",
            tools=[Tool(name="ghost_tool", budget=1)],
        )

        pipeline = Construct("test-bad-tool", nodes=[node])
        graph = compile(pipeline)

        with pytest.raises(ValueError, match="not registered"):
            run(graph, input={"node_id": "test-001"})

    def test_run_with_no_input_or_resume(self):
        """run() with neither input nor resume raises ValueError."""
        from neograph.factory import register_scripted

        register_scripted("noop", lambda input_data, config: Claims(items=[]))
        node = Node.scripted("noop", fn="noop", output=Claims)
        pipeline = Construct("test-no-args", nodes=[node])
        graph = compile(pipeline)

        with pytest.raises(ValueError, match="input or resume"):
            run(graph)

    def test_node_without_output_type(self):
        """Node with no output type raises ValueError at compile."""
        node = Node(name="bad-node", mode="produce", model="fast", prompt="x")
        pipeline = Construct("test-no-output", nodes=[node])

        with pytest.raises(ValueError, match="no output type"):
            compile(pipeline)


class TestLLMUnknownToolCall:
    """LLM hallucinates a tool name the framework doesn't have."""

    def test_unknown_tool_call_handled(self):
        """Framework responds with error message, doesn't crash."""
        from langchain_core.messages import AIMessage

        from neograph._llm import configure_llm
        from neograph.factory import register_tool_factory

        search_tool = FakeTool("search", response="found it")
        register_tool_factory("search", lambda config, tool_config: search_tool)

        call_counter = {"n": 0}

        class FakeLLMHallucinator:
            def __init__(self):
                self._structured = False

            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kwargs):
                if self._structured:
                    return self._model(items=["recovered"])
                call_counter["n"] += 1
                if call_counter["n"] == 1:
                    # LLM hallucinates a tool that doesn't exist
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "delete_everything", "args": {}, "id": "bad-1"}]
                    return msg
                return AIMessage(content="ok done")

            def with_structured_output(self, model, **kwargs):
                clone = FakeLLMHallucinator()
                clone._structured = True
                clone._model = model
                return clone

        configure_llm(
            llm_factory=lambda tier: FakeLLMHallucinator(),
            prompt_compiler=lambda template, data: [{"role": "user", "content": "test"}],
        )

        node = Node(
            name="explore",
            mode="gather",
            output=Claims,
            model="fast",
            prompt="test/explore",
            tools=[Tool(name="search", budget=5)],
        )

        pipeline = Construct("test-hallucinated-tool", nodes=[node])
        graph = compile(pipeline)
        # Should complete without crashing — unknown tool gets error message
        result = run(graph, input={"node_id": "test-001"})
        assert isinstance(result["explore"], Claims)
        assert result["explore"].items == ["recovered"]


class TestModifierAsFirstNode:
    """Modifiers on the first node wire from START, not from a previous node."""

    def test_oracle_at_start(self):
        """Oracle as the first (and only) node — router wired from START."""
        from neograph.factory import register_scripted

        register_scripted("gen_start", lambda input_data, config: Claims(items=["from-start"]))

        def merge_start(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("merge_start", merge_start)

        node = Node.scripted(
            "gen", fn="gen_start", output=Claims
        ) | Oracle(n=2, merge_fn="merge_start")

        pipeline = Construct("test-oracle-start", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        merged = result.get("gen")
        assert merged is not None
        assert len(merged.items) == 2

    def test_each_at_start(self):
        """Each as the first node — router wired from START."""
        from langgraph.graph import END, StateGraph

        from neograph.compiler import _add_node_to_graph
        from neograph.factory import register_scripted
        from neograph.state import compile_state_model

        register_scripted("process_item", lambda input_data, config: MatchResult(
            cluster_label=input_data.label, matched=["done"],
        ))

        process = Node.scripted(
            "process", fn="process_item", input=ClusterGroup, output=MatchResult
        ) | Each(over="make_items.groups", key="label")

        pipeline = Construct("test-each-start", nodes=[process])
        state_model = compile_state_model(pipeline)
        graph = StateGraph(state_model)
        prev = _add_node_to_graph(graph, process, None)
        graph.add_edge(prev, END)
        compiled = graph.compile()
        # Compilation succeeded — Each wired from START without crash


class TestMultiFieldInput:
    """Node with dict[str, type] input spec extracts multiple fields."""

    def test_dict_input_extraction(self):
        """Node receives multiple typed fields from state."""
        from neograph.factory import register_scripted

        register_scripted("make_claims", lambda input_data, config: Claims(items=["a", "b"]))
        register_scripted("make_raw", lambda input_data, config: RawText(text="hello"))

        def combine(input_data, config):
            claims = input_data["step_a"]
            raw = input_data["step_b"]
            return RawText(text=f"{raw.text}: {len(claims.items)} items")

        register_scripted("combine", combine)

        step_a = Node.scripted("step-a", fn="make_claims", output=Claims)
        step_b = Node.scripted("step-b", fn="make_raw", output=RawText)
        step_c = Node.scripted(
            "step-c", fn="combine",
            input={"step_a": Claims, "step_b": RawText},
            output=RawText,
        )

        pipeline = Construct("test-multi-input", nodes=[step_a, step_b, step_c])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        assert result["step_c"].text == "hello: 2 items"



class TestReducerEdgeCases:
    """Reducer functions handle None and unexpected inputs correctly."""

    def test_oracle_results_collected_from_empty(self):
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

    def test_merge_dicts_from_none(self):
        """Dict merge reducer starts from None."""
        from neograph.state import _merge_dicts

        result = _merge_dicts(None, {"a": 1})
        assert result == {"a": 1}

        result = _merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

        # Duplicate key raises
        with pytest.raises(ValueError, match="duplicate key"):
            _merge_dicts({"a": 1}, {"a": 99})

    def test_last_write_wins(self):
        """Last-write-wins reducer always returns new value."""
        from neograph.state import _last_write_wins

        assert _last_write_wins("old", "new") == "new"
        assert _last_write_wins(None, "new") == "new"
        assert _last_write_wins("old", None) is None


class TestStateHygiene:
    """Framework internals never leak to the consumer."""

    def test_oracle_internals_stripped(self):
        """Oracle collector and gen_id are not in the result."""
        from neograph.factory import register_scripted

        register_scripted("g", lambda input_data, config: Claims(items=["x"]))
        register_scripted("m", lambda variants, config: Claims(items=["merged"]))

        node = Node.scripted("gen", fn="g", output=Claims) | Oracle(n=2, merge_fn="m")
        pipeline = Construct("test-hygiene-oracle", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Consumer sees the merged result under the node's name
        assert result["gen"].items == ["merged"]
        # Internals stripped
        assert not any(k.startswith("neo_") for k in result)

    def test_each_internals_stripped(self):
        """Each item plumbing is not in the result."""
        from neograph.factory import register_scripted

        register_scripted("make", lambda input_data, config: Clusters(
            groups=[ClusterGroup(label="a", claim_ids=["1"])]
        ))
        register_scripted("proc", lambda input_data, config: MatchResult(
            cluster_label=input_data.label, matched=["done"],
        ))

        make = Node.scripted("make", fn="make", output=Clusters)
        proc = Node.scripted(
            "proc", fn="proc", input=ClusterGroup, output=MatchResult
        ) | Each(over="make.groups", key="label")

        pipeline = Construct("test-hygiene-each", nodes=[make, proc])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Consumer sees dict keyed by label
        assert "a" in result["proc"]
        # Internals stripped
        assert not any(k.startswith("neo_") for k in result)

    def test_each_duplicate_key_raises(self):
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

        make = Node.scripted("make-dupes", fn="make_dupes", output=Clusters)
        proc = Node.scripted(
            "proc-dupe", fn="proc_dupe", input=ClusterGroup, output=MatchResult
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


class TestSubgraph:
    """Sub-construct with isolated state inside a parent pipeline."""

    def test_subgraph_basic(self):
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
                Node.scripted("lookup", fn="lookup", input=EnrichInput, output=RawText),
                Node.scripted("score", fn="score", input=EnrichInput, output=EnrichOutput),
            ],
        )

        # Parent pipeline
        decompose = Node.scripted("decompose", fn="decompose", output=EnrichInput)
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

    def test_subgraph_state_isolation(self):
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
                Node.scripted("process", fn="sub_process", input=Claims, output=RawText),
            ],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("process", fn="parent_process", output=Claims),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Parent's "process" node output
        assert result["process"].items == ["parent"]
        # Sub-construct output (no collision)
        assert result["sub"].text == "sub-result"

    def test_subgraph_without_input_raises(self):
        """Sub-construct without declared input raises at compile."""
        from neograph.factory import register_scripted

        register_scripted("noop", lambda input_data, config: Claims(items=[]))

        sub = Construct("bad-sub", output=Claims, nodes=[
            Node.scripted("noop", fn="noop", output=Claims),
        ])

        parent = Construct("parent", nodes=[sub])

        with pytest.raises(ValueError, match="has no input type"):
            compile(parent)

    def test_subgraph_without_output_raises(self):
        """Sub-construct without declared output raises at compile."""
        from neograph.factory import register_scripted

        register_scripted("noop", lambda input_data, config: Claims(items=[]))

        sub = Construct("bad-sub", input=Claims, nodes=[
            Node.scripted("noop", fn="noop", output=Claims),
        ])

        parent = Construct("parent", nodes=[sub])

        with pytest.raises(ValueError, match="has no output type"):
            compile(parent)

    def test_subgraph_with_oracle_inside(self):
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
                Node.scripted("gen", fn="sub_gen", output=RawText) | Oracle(n=3, merge_fn="sub_merge"),
            ],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("prep", fn="parent_prep", output=Claims),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        assert result["oracle_sub"].text == "merged 3 variants"
        # Oracle internals from sub don't leak
        assert not any("neo_oracle" in k for k in result)

    def test_subgraph_with_each_inside(self):
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
                Node.scripted("verify", fn="sub_verify", input=ClusterGroup, output=MatchResult)
                | Each(over="neo_subgraph_input.groups", key="label"),
                Node.scripted("collect", fn="sub_collect", output=RawText),
            ],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("make-clusters", fn="parent_clusters", output=Clusters),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        assert result["verify_sub"] is not None
        assert "verified" in result["verify_sub"].text

    def test_nested_subgraphs(self):
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
            nodes=[Node.scripted("detail", fn="l2_detail", input=Claims, output=RawText)],
        )

        # Level1: Claims → RawText (via level2)
        level1 = Construct(
            "level1",
            input=Claims,
            output=RawText,
            nodes=[
                Node.scripted("process", fn="l1_process", input=Claims, output=Claims),
                level2,
            ],
        )

        parent = Construct("root", nodes=[
            Node.scripted("start", fn="l0_start", output=Claims),
            level1,
            Node.scripted("finish", fn="l0_finish", input=RawText, output=RawText),
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

    def test_multiple_subgraphs_in_parent(self):
        """Two sub-constructs in the same parent pipeline."""
        from neograph.factory import register_scripted

        register_scripted("make_input", lambda input_data, config: Claims(items=["a", "b"]))
        register_scripted("enrich_fn", lambda input_data, config: RawText(text="enriched"))
        register_scripted("validate_fn", lambda input_data, config: ValidationResult(passed=True, issues=[]))

        enrich_sub = Construct(
            "enrich",
            input=Claims,
            output=RawText,
            nodes=[Node.scripted("e", fn="enrich_fn", input=Claims, output=RawText)],
        )

        validate_sub = Construct(
            "check",
            input=RawText,
            output=ValidationResult,
            nodes=[Node.scripted("v", fn="validate_fn", input=RawText, output=ValidationResult)],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("start", fn="make_input", output=Claims),
            enrich_sub,
            validate_sub,
        ])

        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        assert result["enrich"].text == "enriched"
        assert result["check"].passed is True

    def test_operator_without_checkpointer_raises(self):
        """Operator node without checkpointer raises ValueError at compile."""
        from neograph.factory import register_condition, register_scripted

        register_scripted("x", lambda input_data, config: Claims(items=[]))
        register_condition("always", lambda state: True)

        node = Node.scripted("x", fn="x", output=Claims) | Operator(when="always")
        pipeline = Construct("test-no-cp", nodes=[node])

        with pytest.raises(ValueError, match="checkpointer"):
            compile(pipeline)


# ═══════════════════════════════════════════════════════════════════════════
# CONSTRUCT + MODIFIER COMPOSITIONS
#
# Every modifier × Construct target, plus deep nesting combos.
# ═══════════════════════════════════════════════════════════════════════════


class TestConstructOracle:
    """Construct | Oracle — run entire sub-pipeline N times, merge outputs."""

    def test_construct_oracle_scripted_merge(self):
        """Sub-pipeline runs 3 times, scripted merge combines outputs."""
        from neograph.factory import register_scripted

        register_scripted("sub_step_a", lambda input_data, config: Claims(items=["step-a"]))
        register_scripted("sub_step_b", lambda input_data, config: RawText(
            text=f"processed: {input_data.items[0]}" if input_data else "processed: none"
        ))

        def merge_sub_outputs(variants, config):
            all_texts = [v.text for v in variants]
            return RawText(text=" | ".join(all_texts))

        register_scripted("merge_sub", merge_sub_outputs)

        sub = Construct(
            "enrich",
            input=Claims,
            output=RawText,
            nodes=[
                Node.scripted("step-a", fn="sub_step_a", output=Claims),
                Node.scripted("step-b", fn="sub_step_b", input=Claims, output=RawText),
            ],
        ) | Oracle(n=3, merge_fn="merge_sub")

        register_scripted("make_input", lambda input_data, config: Claims(items=["raw"]))

        parent = Construct("parent", nodes=[
            Node.scripted("start", fn="make_input", output=Claims),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # 3 variants merged into one
        assert result["enrich"] is not None
        assert result["enrich"].text.count("processed") == 3

    def test_construct_oracle_llm_merge(self):
        """Sub-pipeline runs 2 times, LLM merge combines outputs."""
        from neograph.factory import register_scripted

        register_scripted("gen_claim", lambda input_data, config: Claims(items=["variant"]))

        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["llm-merged"])))

        sub = Construct(
            "gen-pipeline",
            input=Claims,
            output=Claims,
            nodes=[Node.scripted("gen", fn="gen_claim", output=Claims)],
        ) | Oracle(n=2, merge_prompt="test/merge")

        register_scripted("seed", lambda input_data, config: Claims(items=["seed"]))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed", output=Claims),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        assert result["gen_pipeline"].items == ["llm-merged"]


class TestConstructEach:
    """Construct | Each — run entire sub-pipeline per collection item."""

    def test_construct_each_over_collection(self):
        """Sub-pipeline runs once per cluster, results collected as dict."""
        from neograph.factory import register_scripted

        register_scripted("make_clusters", lambda input_data, config: Clusters(
            groups=[
                ClusterGroup(label="alpha", claim_ids=["c1"]),
                ClusterGroup(label="beta", claim_ids=["c2", "c3"]),
            ]
        ))

        register_scripted("sub_analyze", lambda input_data, config: RawText(
            text=f"analyzed: {input_data.label}"
        ))
        register_scripted("sub_score", lambda input_data, config: MatchResult(
            cluster_label=input_data.label if hasattr(input_data, 'label') else "unknown",
            matched=[f"scored-{input_data.text}" if hasattr(input_data, 'text') else "scored"],
        ))

        sub = Construct(
            "verify-cluster",
            input=ClusterGroup,
            output=MatchResult,
            nodes=[
                Node.scripted("analyze", fn="sub_analyze", input=ClusterGroup, output=RawText),
                Node.scripted("score", fn="sub_score", input=RawText, output=MatchResult),
            ],
        ) | Each(over="make_clusters.groups", key="label")

        parent = Construct("parent", nodes=[
            Node.scripted("make-clusters", fn="make_clusters", output=Clusters),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Both clusters processed
        verify_results = result["verify_cluster"]
        assert "alpha" in verify_results
        assert "beta" in verify_results


class TestConstructOperator:
    """Construct | Operator — check condition after sub-pipeline completes."""

    def test_construct_operator_interrupts(self):
        """Sub-pipeline runs, then Operator checks and interrupts."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        register_scripted("sub_validate", lambda input_data, config: ValidationResult(
            passed=False, issues=["coverage gap"],
        ))

        register_condition("sub_failed", lambda state: (
            {"issues": state.enrich.issues}
            if hasattr(state, 'enrich') and state.enrich and not state.enrich.passed
            else None
        ))

        sub = Construct(
            "enrich",
            input=Claims,
            output=ValidationResult,
            nodes=[Node.scripted("val", fn="sub_validate", output=ValidationResult)],
        ) | Operator(when="sub_failed")

        register_scripted("seed", lambda input_data, config: Claims(items=["data"]))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed", output=Claims),
            sub,
        ])
        graph = compile(parent, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "construct-op-test"}}

        result = run(graph, input={"node_id": "test-001"}, config=config)

        # Sub-pipeline ran and produced output
        assert result["enrich"] is not None
        assert result["enrich"].passed is False
        # Interrupted
        assert "__interrupt__" in result

    def test_construct_operator_passes(self):
        """Sub-pipeline runs, condition is falsy, graph continues."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        register_scripted("sub_ok", lambda input_data, config: ValidationResult(
            passed=True, issues=[],
        ))

        register_condition("sub_check", lambda state: None)  # always passes

        sub = Construct(
            "check",
            input=Claims,
            output=ValidationResult,
            nodes=[Node.scripted("ok", fn="sub_ok", output=ValidationResult)],
        ) | Operator(when="sub_check")

        register_scripted("seed2", lambda input_data, config: Claims(items=["ok"]))
        register_scripted("done", lambda input_data, config: RawText(text="complete"))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed2", output=Claims),
            sub,
            Node.scripted("done", fn="done", output=RawText),
        ])
        graph = compile(parent, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "construct-op-pass"}}
        result = run(graph, input={"node_id": "test-001"}, config=config)

        assert result["done"].text == "complete"


class TestDeepCompositions:
    """Complex nesting: modifiers inside modifiers, tool exhaustion, etc."""

    def test_oracle_inside_each(self):
        """Each item gets Oracle ensemble — fan-out inside fan-out."""
        from neograph.factory import register_scripted

        register_scripted("make_items", lambda input_data, config: Clusters(
            groups=[ClusterGroup(label="x", claim_ids=["1"]), ClusterGroup(label="y", claim_ids=["2"])]
        ))

        gen_count = {"n": 0}

        def gen_variant(input_data, config):
            gen_count["n"] += 1
            return RawText(text=f"variant-{gen_count['n']}")

        register_scripted("gen_v", gen_variant)

        def merge_v(variants, config):
            return RawText(text=f"merged-{len(variants)}")

        register_scripted("merge_v", merge_v)

        # Inner sub-construct: Oracle (2 variants, merge)
        inner = Construct(
            "oracle-inner",
            input=ClusterGroup,
            output=RawText,
            nodes=[
                Node.scripted("gen", fn="gen_v", output=RawText)
                | Oracle(n=2, merge_fn="merge_v"),
            ],
        )

        # Outer: Each over clusters, each runs the Oracle sub-pipeline
        # This means: 2 clusters × 2 Oracle variants = 4 generator calls + 2 merges
        parent = Construct("parent", nodes=[
            Node.scripted("make", fn="make_items", output=Clusters),
            inner | Each(over="make.groups", key="label"),
        ])

        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Each cluster got Oracle'd — 2 clusters × 2 generators = 4 total
        assert gen_count["n"] == 4
        assert "x" in result["oracle_inner"]
        assert "y" in result["oracle_inner"]

    def test_tool_budget_exhaustion_in_subgraph(self):
        """Gather node inside subgraph exhausts tool budget, forced to respond."""
        from neograph.factory import register_scripted, register_tool_factory

        deep_search_tool = FakeTool("deep_search", response="found")
        register_tool_factory("deep_search", lambda config, tool_config: deep_search_tool)

        fake = ReActFake(
            tool_calls=[
                [{"name": "deep_search", "args": {}, "id": "c1"}],
                [{"name": "deep_search", "args": {}, "id": "c2"}],
                [{"name": "deep_search", "args": {}, "id": "c3"}],
                [{"name": "deep_search", "args": {}, "id": "c4"}],
                [{"name": "deep_search", "args": {}, "id": "c5"}],
                [],  # stop
            ],
            final=lambda m: m(text="search complete"),
        )
        configure_fake_llm(lambda tier: fake)

        register_scripted("prep_search", lambda input_data, config: Claims(items=["query"]))

        sub = Construct(
            "deep-search",
            input=Claims,
            output=RawText,
            nodes=[
                Node(
                    name="search",
                    mode="gather",
                    input=Claims,
                    output=RawText,
                    model="fast",
                    prompt="test/search",
                    tools=[Tool(name="deep_search", budget=3)],
                ),
            ],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("prep", fn="prep_search", output=Claims),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Tool budget was 3, so exactly 3 calls made despite LLM wanting 5
        assert len(deep_search_tool.calls) == 3
        # Subgraph still produced output
        assert result["deep_search"] is not None

    def test_operator_inside_subgraph_bubbles_interrupt(self):
        """Operator inside a sub-construct pauses the entire parent pipeline."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        register_scripted("sub_check", lambda input_data, config: ValidationResult(
            passed=False, issues=["needs human"],
        ))

        register_condition("inner_failed", lambda state: (
            {"reason": "inner check failed"}
            if hasattr(state, 'check') and state.check and not state.check.passed
            else None
        ))

        sub = Construct(
            "inner",
            input=Claims,
            output=ValidationResult,
            nodes=[
                Node.scripted("check", fn="sub_check", output=ValidationResult)
                | Operator(when="inner_failed"),
            ],
        )

        register_scripted("start_fn", lambda input_data, config: Claims(items=["go"]))
        register_scripted("after_fn", lambda input_data, config: RawText(text="should not reach"))

        parent = Construct("parent", nodes=[
            Node.scripted("start", fn="start_fn", output=Claims),
            sub,
            Node.scripted("after", fn="after_fn", output=RawText),
        ])

        # Operator lives inside sub-construct — parent needs checkpointer
        # so the recursive compile of the sub-construct gets it
        from langgraph.checkpoint.memory import MemorySaver

        graph = compile(parent, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "inner-op-test"}}
        result = run(graph, input={"node_id": "test-001"}, config=config)

        # The interrupt inside the subgraph should surface
        # "after" node should not have run
        assert result.get("after") is None or "__interrupt__" in result


# ═══════════════════════════════════════════════════════════════════════════
# REMAINING COVERAGE GAPS — paths not yet exercised
# ═══════════════════════════════════════════════════════════════════════════


class TestFirstNodeEdgeCases:
    """Subgraphs and modified constructs as the first node (wired from START)."""

    def test_plain_subgraph_as_first_node(self):
        """Sub-construct as the very first node — no upstream data."""
        from neograph.factory import register_scripted

        register_scripted("self_seed", lambda input_data, config: Claims(items=["self-seeded"]))

        sub = Construct(
            "first-sub",
            input=Claims,
            output=Claims,
            nodes=[Node.scripted("seed", fn="self_seed", output=Claims)],
        )

        # Sub-construct is the ONLY node — wired from START
        parent = Construct("parent", nodes=[sub])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        assert result["first_sub"].items == ["self-seeded"]

    def test_construct_oracle_as_first_node(self):
        """Construct | Oracle as the first node — router wired from START."""
        from neograph.factory import register_scripted

        register_scripted("gen_first", lambda input_data, config: Claims(items=["v"]))

        def merge_first(variants, config):
            return Claims(items=[f"merged-{len(variants)}"])

        register_scripted("merge_first", merge_first)

        sub = Construct(
            "oracle-first",
            input=Claims,
            output=Claims,
            nodes=[Node.scripted("g", fn="gen_first", output=Claims)],
        ) | Oracle(n=2, merge_fn="merge_first")

        parent = Construct("parent", nodes=[sub])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        assert result["oracle_first"].items == ["merged-2"]

    def test_construct_each_as_first_node(self):
        """Construct | Each as the first node — needs collection in state."""
        from neograph.factory import register_scripted

        # For Each to work as first node, the collection must come from
        # somewhere. In practice this means the state is pre-seeded.
        # This tests that the router wires from START without crashing.
        # We can't run it (no collection in initial state), but compile must succeed.
        register_scripted("proc_first", lambda input_data, config: RawText(text="done"))

        sub = Construct(
            "each-first",
            input=ClusterGroup,
            output=RawText,
            nodes=[Node.scripted("p", fn="proc_first", output=RawText)],
        ) | Each(over="data.items", key="label")

        parent = Construct("parent", nodes=[sub])
        # Compile succeeds — Each wired from START
        graph = compile(parent)


# ═══════════════════════════════════════════════════════════════════════════
# Assembly-time type validation — compile errors surface at Construct(...)
# ═══════════════════════════════════════════════════════════════════════════

# Tiny factory helpers for the validation tests below — each test cares
# about input/output type flow, not scripted function names, so the `fn="f"`
# placeholder is noise that the helpers strip.
def _producer(name: str, out: type) -> Node:
    return Node.scripted(name, fn="f", output=out)


def _consumer(name: str, in_: type, out: type) -> Node:
    return Node.scripted(name, fn="f", input=in_, output=out)


class TestConstructValidation:
    """Input/output compatibility is checked at Construct assembly time."""

    def test_valid_chain_passes(self):
        """A correctly typed chain assembles without error."""
        a = _producer("a", RawText)
        b = _consumer("b", RawText, Claims)
        c = _consumer("c", Claims, ClassifiedClaims)
        pipeline = Construct("good", nodes=[a, b, c])
        # Pin the resulting construct shape so a no-op validator regression
        # (e.g. silently dropping nodes) is caught.
        assert len(pipeline.nodes) == 3
        assert [n.name for n in pipeline.nodes] == ["a", "b", "c"]

    def test_plain_input_mismatch_raises(self):
        """Downstream input with no compatible upstream raises ConstructError
        AND the error message lists the upstream producers."""
        a = _producer("a", RawText)
        b = _consumer("b", Claims, ClassifiedClaims)
        with pytest.raises(ConstructError) as exc_info:
            Construct("bad", nodes=[a, b])
        msg = str(exc_info.value)
        assert "declares input=Claims" in msg
        # Pin the producer listing, not just the header
        assert "node 'a': RawText" in msg

    def test_mismatch_hint_suggests_map(self):
        """When upstream has list[input_type] field, hint names the correct path."""
        # Clusters has `groups: list[ClusterGroup]` — the hint should point
        # at `s.a.groups`, not some other field.
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult)
        with pytest.raises(ConstructError) as exc_info:
            Construct("bad-fanout", nodes=[a, b])
        msg = str(exc_info.value)
        assert "did you forget to fan out" in msg
        # Pin the concrete path the hint renders — a bug that emits the
        # wrong field (e.g. s.b.other) would pass a mere phrase match.
        assert "s.a.groups" in msg

    def test_mismatch_error_includes_source_location(self):
        """Error message includes a file:line pointer to the user call site."""
        a = _producer("a", RawText)
        b = _consumer("b", Claims, ClassifiedClaims)
        with pytest.raises(ConstructError, match=r"at test_e2e_piarch_ready\.py:\d+"):
            Construct("bad-loc", nodes=[a, b])

    def test_each_correct_path_passes(self):
        """Each whose path resolves to list[input_type] assembles AND attaches the modifier."""
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult).map(
            lambda s: s.a.groups, key="label"
        )
        pipeline = Construct("good-each", nodes=[a, b])
        assert len(pipeline.nodes) == 2
        each = pipeline.nodes[1].get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "a.groups"
        assert each.key == "label"

    def test_each_missing_field_raises(self):
        """Each path that walks to a non-existent field raises."""
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.nonexistent", key="label"
        )
        with pytest.raises(ConstructError, match="has no field 'nonexistent'"):
            Construct("bad-each-field", nodes=[a, b])

    def test_each_terminal_not_list_raises(self):
        """Each whose terminal field isn't a list is flagged."""
        # RawText.text is a str — not a list; Each can't fan out over it.
        a = _producer("a", RawText)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.text", key="label"
        )
        with pytest.raises(ConstructError, match="not a list"):
            Construct("bad-each-terminal", nodes=[a, b])

    def test_each_list_wrong_element_raises(self):
        """Each whose list element type doesn't match input raises."""
        # Claims.items is list[str]; downstream wants ClusterGroup, not str.
        a = _producer("a", Claims)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.items", key="label"
        )
        with pytest.raises(ConstructError, match=r"list\[str\]"):
            Construct("bad-each-element", nodes=[a, b])

    def test_first_item_with_input_deferred_to_runtime(self):
        """First-of-chain with declared input is NOT flagged — runtime-seeded."""
        # No producers exist yet when checking `b` — defer to runtime.
        b = _consumer("b", Claims, ClassifiedClaims)
        pipeline = Construct("top-level", nodes=[b])
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].input is Claims

    def test_top_level_each_deferred_to_runtime(self):
        """Each at position 0 whose root isn't a known producer defers cleanly.

        Exercises construct.py:189 — Each root-not-in-producers branch.
        """
        process = _consumer("process", ClusterGroup, MatchResult) | Each(
            over="seeded_from_runtime.groups", key="label"
        )
        pipeline = Construct("top-each", nodes=[process])
        assert len(pipeline.nodes) == 1
        each = pipeline.nodes[0].get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "seeded_from_runtime.groups"

    def test_sub_construct_input_port_satisfies_inner_node(self):
        """Inner node reading from the sub-construct's input port validates."""
        inner = _consumer("inner", Claims, Claims)
        # sub.input=Claims is injected as `neo_subgraph_input` producer for inner
        sub = Construct("sub", input=Claims, output=Claims, nodes=[inner])
        assert sub.input is Claims
        assert sub.output is Claims
        assert len(sub.nodes) == 1

    def test_sub_construct_chained_in_parent(self):
        """Parent producing sub.input satisfies the sub-construct's input check."""
        upstream = _producer("upstream", Claims)
        sub = Construct(
            "sub", input=Claims, output=ClassifiedClaims,
            nodes=[_consumer("inner", Claims, ClassifiedClaims)],
        )
        parent = Construct("parent", nodes=[upstream, sub])
        assert len(parent.nodes) == 2
        assert parent.nodes[1].input is Claims

    def test_sub_construct_input_mismatch_in_parent(self):
        """Parent's upstream output incompatible with sub.input raises with
        a tight error pinning BOTH the construct name and the clause."""
        upstream = _producer("upstream", RawText)
        sub = Construct(
            "sub", input=Claims, output=ClassifiedClaims,
            nodes=[_consumer("inner", Claims, ClassifiedClaims)],
        )
        with pytest.raises(ConstructError) as exc_info:
            Construct("parent", nodes=[upstream, sub])
        msg = str(exc_info.value)
        # Anchor on the specific sub-construct name (not a permissive `sub.*`)
        # and the declaration clause — bug that raised the wrong inner error
        # first would not match both anchors.
        assert "'sub' in construct 'parent'" in msg
        assert "declares input=Claims" in msg

    def test_construct_error_is_valueerror(self):
        """ConstructError subclasses ValueError for existing except clauses."""
        a = _producer("a", RawText)
        b = _consumer("b", Claims, ClassifiedClaims)
        with pytest.raises(ValueError):
            Construct("bad", nodes=[a, b])

    def test_dict_input_skipped(self):
        """Nodes with dict[str, type] input spec aren't statically validated."""
        # step-c has multi-field input; static validation punts to runtime.
        step_a = _producer("step-a", Claims)
        step_b = _producer("step-b", RawText)
        step_c = Node.scripted(
            "step-c", fn="f",
            input={"step_a": Claims, "step_b": RawText},
            output=RawText,
        )
        pipeline = Construct("multi-input", nodes=[step_a, step_b, step_c])
        assert len(pipeline.nodes) == 3
        assert isinstance(pipeline.nodes[2].input, dict)

    def test_dict_class_input_deferred(self):
        """input=dict (raw class) defers to runtime isinstance scan.

        factory._extract_input handles this shape via isinstance(val, dict)
        over state fields; the validator must not reject it even when no
        upstream producer's output is a dict subclass.
        """
        a = _producer("a", RawText)
        b = Node.scripted("b", fn="f", input=dict, output=Claims)
        pipeline = Construct("dict-class", nodes=[a, b])
        assert len(pipeline.nodes) == 2
        assert pipeline.nodes[1].input is dict

    def test_dict_generic_input_deferred(self):
        """input=dict[str, X] (parameterized generic) defers to runtime.

        factory._is_instance_safe uses get_origin(dict[str, X]) → dict and
        then isinstance(val, dict); the validator must accept this shape too.
        """
        a = _producer("a", RawText)
        b = Node.scripted("b", fn="f", input=dict[str, Claims], output=Claims)
        pipeline = Construct("dict-generic", nodes=[a, b])
        assert len(pipeline.nodes) == 2
        assert pipeline.nodes[1].input == dict[str, Claims]


class TestConstructOracleErrors:
    """Error paths for Construct | Oracle."""

    def test_construct_oracle_unregistered_merge_fn(self):
        """Construct | Oracle with unregistered merge_fn raises at compile."""
        from neograph.factory import register_scripted

        register_scripted("gen_err", lambda input_data, config: Claims(items=[]))

        sub = Construct(
            "bad-oracle-sub",
            input=Claims,
            output=Claims,
            nodes=[Node.scripted("g", fn="gen_err", output=Claims)],
        ) | Oracle(n=2, merge_fn="nonexistent_merge_fn")

        parent = Construct("parent", nodes=[sub])

        with pytest.raises(ValueError, match="not registered"):
            compile(parent)


class TestStripInternalsEdge:
    """Edge case: _strip_internals when result is not a dict."""

    def test_strip_non_dict(self):
        """Non-dict results pass through unchanged."""
        from neograph.runner import _strip_internals

        assert _strip_internals("hello") == "hello"
        assert _strip_internals(42) == 42
        assert _strip_internals(None) is None
        assert _strip_internals([1, 2, 3]) == [1, 2, 3]


class TestLLMConfig:
    """Per-node LLM configuration flows through to the factory."""

    def test_llm_config_passed_to_factory(self):
        """Node's llm_config dict reaches the llm_factory."""
        factory_calls = []

        def tracking_factory(tier, node_name=None, llm_config=None):
            factory_calls.append({
                "tier": tier,
                "node_name": node_name,
                "llm_config": llm_config,
            })
            return StructuredFake(lambda m: m(items=["result"]))

        configure_fake_llm(tracking_factory)

        node = Node(
            name="custom-llm",
            mode="produce",
            output=Claims,
            model="reason",
            prompt="test",
            llm_config={"temperature": 0.9, "max_tokens": 2000},
        )

        pipeline = Construct("test-llm-config", nodes=[node])
        graph = compile(pipeline)
        run(graph, input={"node_id": "test-001"})

        # Factory was called with the node's llm_config
        assert len(factory_calls) == 1
        assert factory_calls[0]["tier"] == "reason"
        assert factory_calls[0]["node_name"] == "custom-llm"
        assert factory_calls[0]["llm_config"]["temperature"] == 0.9
        assert factory_calls[0]["llm_config"]["max_tokens"] == 2000

    def test_llm_config_backward_compatible(self):
        """Old-style factory(tier) still works without llm_config."""
        # Old-style factory: only accepts tier
        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["ok"])))

        node = Node(
            name="old-style",
            mode="produce",
            output=Claims,
            model="fast",
            prompt="test",
            llm_config={"temperature": 0.5},  # this won't crash old factory
        )

        pipeline = Construct("test-compat", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        assert result["old_style"].items == ["ok"]

    def test_input_injected_into_config(self):
        """Pipeline input fields (node_id, project_root) accessible via config["configurable"]."""
        from neograph.factory import register_scripted

        config_seen = {}

        def capture_config(input_data, config):
            configurable = config.get("configurable", {})
            config_seen["node_id"] = configurable.get("node_id")
            config_seen["project_root"] = configurable.get("project_root")
            config_seen["custom_field"] = configurable.get("custom_field")
            return Claims(items=["done"])

        register_scripted("capture", capture_config)

        node = Node.scripted("capture", fn="capture", output=Claims)
        pipeline = Construct("test-config-inject", nodes=[node])
        graph = compile(pipeline)

        result = run(graph,
                     input={"node_id": "BR-RW-042", "project_root": "/my/project"},
                     config={"configurable": {"custom_field": "extra-data"}})

        # All input fields + custom config accessible
        assert config_seen["node_id"] == "BR-RW-042"
        assert config_seen["project_root"] == "/my/project"
        assert config_seen["custom_field"] == "extra-data"

    def test_prompt_compiler_receives_config(self):
        """Prompt compiler gets node_name and full config with pipeline metadata."""
        compiler_calls = []

        def tracking_compiler(template, data, *, node_name=None, config=None):
            compiler_calls.append({
                "template": template,
                "node_name": node_name,
                "node_id": config.get("configurable", {}).get("node_id") if config else None,
                "project_root": config.get("configurable", {}).get("project_root") if config else None,
            })
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            lambda tier: StructuredFake(lambda m: m(items=["result"])),
            prompt_compiler=tracking_compiler,
        )

        node = Node(name="analyze", mode="produce", output=Claims, model="fast", prompt="rw/analyze")
        pipeline = Construct("test-prompt-ctx", nodes=[node])
        graph = compile(pipeline)
        run(graph, input={"node_id": "BR-001", "project_root": "/proj"})

        assert len(compiler_calls) == 1
        assert compiler_calls[0]["template"] == "rw/analyze"
        assert compiler_calls[0]["node_name"] == "analyze"
        assert compiler_calls[0]["node_id"] == "BR-001"
        assert compiler_calls[0]["project_root"] == "/proj"


class TestToolDecorator:
    """@tool decorator: signature-inferred tool schemas."""

    def test_tool_decorator_auto_registers(self):
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
            mode="gather",
            output=Claims,
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

    def test_tool_decorator_without_args(self):
        """@tool (no parens) also works."""
        @tool
        def noop(x: str) -> str:
            """A no-op tool."""
            return x

        assert isinstance(noop, Tool)
        assert noop.name == "noop"
        assert noop.budget == 0  # unlimited by default


class TestRunIsolated:
    """Node.run_isolated() — direct invocation for unit testing."""

    def test_scripted_node_run_isolated(self):
        """Scripted nodes can be tested without compile()/run()."""
        from neograph import register_scripted

        register_scripted("upper", lambda input_data, config: RawText(
            text=input_data.text.upper() if input_data else "NONE"
        ))

        upper_node = Node.scripted("upper", fn="upper", input=RawText, output=RawText)

        # Direct invocation — no pipeline, no compile, no run
        result = upper_node.run_isolated(input=RawText(text="hello"))

        assert isinstance(result, RawText)
        assert result.text == "HELLO"

    def test_produce_node_run_isolated(self):
        """Produce nodes can be tested with a fake LLM."""
        configure_fake_llm(lambda tier: StructuredFake(
            lambda model: model(items=["isolated-result"])
        ))

        decompose = Node(
            "decompose", mode="produce", output=Claims, model="fast", prompt="test"
        )

        result = decompose.run_isolated()

        assert isinstance(result, Claims)
        assert result.items == ["isolated-result"]

    def test_run_isolated_with_config(self):
        """run_isolated passes config through to the node function."""
        from neograph import register_scripted

        seen_config = {}
        def fn(input_data, config):
            seen_config.update(config.get("configurable", {}))
            return Claims(items=["ok"])

        register_scripted("cfg_test", fn)
        node = Node.scripted("cfg-test", fn="cfg_test", output=Claims)

        result = node.run_isolated(
            config={"configurable": {"node_id": "TEST-001", "env": "staging"}}
        )

        assert result.items == ["ok"]
        assert seen_config["node_id"] == "TEST-001"
        assert seen_config["env"] == "staging"


class TestConfigInjectionPatterns:
    """Real-world config injection patterns from piarch consumer."""

    def test_shared_resources_in_config(self):
        """Consumer passes shared infrastructure (rate limiter, tracer) via config."""
        from neograph.factory import register_scripted

        class FakeRateLimiter:
            def __init__(self):
                self.calls = 0

            def call(self):
                self.calls += 1

        limiter = FakeRateLimiter()

        def node_with_resources(input_data, config):
            rl = config["configurable"]["rate_limiter"]
            rl.call()
            return Claims(items=[f"calls={rl.calls}"])

        register_scripted("resourced", node_with_resources)

        pipeline = Construct("test-resources", nodes=[
            Node.scripted("step", fn="resourced", output=Claims),
        ])
        graph = compile(pipeline)
        result = run(
            graph,
            input={"node_id": "test"},
            config={"configurable": {"rate_limiter": limiter}},
        )

        assert limiter.calls == 1
        assert result["step"].items == ["calls=1"]

    def test_config_available_in_subgraph(self):
        """Config flows through to sub-construct node functions."""
        from neograph.factory import register_scripted

        seen_in_sub = {}

        def sub_node(input_data, config):
            seen_in_sub["node_id"] = config.get("configurable", {}).get("node_id")
            seen_in_sub["custom"] = config.get("configurable", {}).get("custom")
            return RawText(text="sub-done")

        register_scripted("sub_node", sub_node)
        register_scripted("parent_seed", lambda input_data, config: Claims(items=["x"]))

        sub = Construct(
            "sub",
            input=Claims,
            output=RawText,
            nodes=[Node.scripted("inner", fn="sub_node", output=RawText)],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="parent_seed", output=Claims),
            sub,
        ])
        graph = compile(parent)
        run(graph,
            input={"node_id": "BR-042"},
            config={"configurable": {"custom": "my-value"}})

        # Sub-construct node function received both pipeline input and custom config
        assert seen_in_sub["node_id"] == "BR-042"
        assert seen_in_sub["custom"] == "my-value"

    def test_config_available_in_oracle_generators(self):
        """Each Oracle generator sees config with node_id and custom fields."""
        from neograph.factory import register_scripted

        gen_configs = []

        def gen_fn(input_data, config):
            gen_configs.append({
                "node_id": config.get("configurable", {}).get("node_id"),
                "project_root": config.get("configurable", {}).get("project_root"),
                "gen_id": config.get("configurable", {}).get("_generator_id"),
            })
            return Claims(items=["v"])

        def merge_fn(variants, config):
            return Claims(items=[f"merged-{len(variants)}"])

        register_scripted("cfg_gen", gen_fn)
        register_scripted("cfg_merge", merge_fn)

        node = Node.scripted(
            "gen", fn="cfg_gen", output=Claims
        ) | Oracle(n=3, merge_fn="cfg_merge")

        pipeline = Construct("test-oracle-config", nodes=[node])
        graph = compile(pipeline)
        run(graph,
            input={"node_id": "BR-099", "project_root": "/proj"},
            config={})

        # All 3 generators saw the pipeline metadata
        assert len(gen_configs) == 3
        for gc in gen_configs:
            assert gc["node_id"] == "BR-099"
            assert gc["project_root"] == "/proj"
            assert gc["gen_id"] is not None  # generator ID also present

    def test_config_available_in_each_fanout(self):
        """Each fan-out node sees config with node_id."""
        from neograph.factory import register_scripted

        each_configs = []

        register_scripted("make_groups", lambda input_data, config: Clusters(
            groups=[ClusterGroup(label="a", claim_ids=["1"]),
                    ClusterGroup(label="b", claim_ids=["2"])]
        ))

        def verify_with_config(input_data, config):
            each_configs.append({
                "node_id": config.get("configurable", {}).get("node_id"),
                "label": input_data.label if hasattr(input_data, "label") else "?",
            })
            return MatchResult(cluster_label=input_data.label, matched=["ok"])

        register_scripted("verify_cfg", verify_with_config)

        make = Node.scripted("make", fn="make_groups", output=Clusters)
        verify = Node.scripted(
            "verify", fn="verify_cfg", input=ClusterGroup, output=MatchResult
        ) | Each(over="make.groups", key="label")

        pipeline = Construct("test-each-config", nodes=[make, verify])
        graph = compile(pipeline)
        run(graph, input={"node_id": "BR-050"})

        # Both fan-out invocations saw node_id
        assert len(each_configs) == 2
        assert all(c["node_id"] == "BR-050" for c in each_configs)
        labels = {c["label"] for c in each_configs}
        assert labels == {"a", "b"}

    def test_multi_node_pipeline_all_see_config(self):
        """Every node in a multi-step pipeline sees the same config metadata."""
        from neograph.factory import register_scripted

        nodes_seen = []

        def tracking_fn(input_data, config):
            nodes_seen.append({
                "node_id": config.get("configurable", {}).get("node_id"),
                "env": config.get("configurable", {}).get("env"),
            })
            return Claims(items=["done"])

        register_scripted("track_a", tracking_fn)
        register_scripted("track_b", tracking_fn)
        register_scripted("track_c", tracking_fn)

        pipeline = Construct("test-all-config", nodes=[
            Node.scripted("a", fn="track_a", output=Claims),
            Node.scripted("b", fn="track_b", output=Claims),
            Node.scripted("c", fn="track_c", output=Claims),
        ])
        graph = compile(pipeline)
        run(graph,
            input={"node_id": "REQ-001"},
            config={"configurable": {"env": "staging"}})

        assert len(nodes_seen) == 3
        for seen in nodes_seen:
            assert seen["node_id"] == "REQ-001"
            assert seen["env"] == "staging"


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT STRATEGIES — structured, json_mode, text
# ═══════════════════════════════════════════════════════════════════════════


class TestOutputStrategyStructured:
    """Default strategy: llm.with_structured_output(model). Current behavior."""

    def test_structured_output_default(self):
        """Produce node uses with_structured_output by default."""
        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["via-structured"])))

        node = Node(name="extract", mode="produce", output=Claims, model="fast", prompt="test")
        pipeline = Construct("test-structured", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert result["extract"].items == ["via-structured"]


class TestOutputStrategyJsonMode:
    """json_mode strategy: inject schema into prompt, parse response as JSON."""

    def test_json_mode_injects_schema_and_parses(self):
        """json_mode: schema injected into prompt, LLM returns raw JSON, framework parses."""
        compiler_calls = []

        def tracking_compiler(template, data, **kw):
            compiler_calls.append(kw)
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            lambda tier: TextFake('{"items": ["via-json-mode"]}'),
            prompt_compiler=tracking_compiler,
        )

        node = Node(
            name="extract",
            mode="produce",
            output=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-json-mode", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        # Parsed correctly from raw JSON
        assert result["extract"].items == ["via-json-mode"]

    def test_json_mode_handles_markdown_fences(self):
        """json_mode: strips markdown code fences before parsing."""
        configure_fake_llm(lambda tier: TextFake('```json\n{"items": ["fenced"]}\n```'))

        node = Node(
            name="extract",
            mode="produce",
            output=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-json-fence", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert result["extract"].items == ["fenced"]


class TestOutputStrategyText:
    """text strategy: LLM returns plain text, consumer's prompt_compiler handles schema."""

    def test_text_mode_parses_json_from_response(self):
        """text mode: LLM returns text containing JSON, framework extracts and parses."""
        configure_fake_llm(
            lambda tier: TextFake('Here is my analysis:\n{"items": ["from-text"]}\nDone.')
        )

        node = Node(
            name="extract",
            mode="produce",
            output=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "text"},
        )
        pipeline = Construct("test-text", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert result["extract"].items == ["from-text"]


class TestOutputStrategyOnGather:
    """Output strategy also applies to the final parse in gather/execute modes."""

    def test_gather_json_mode_parses_final_response(self):
        """Gather node with json_mode parses the final structured output from raw JSON."""
        from langchain_core.messages import AIMessage

        from neograph._llm import configure_llm
        from neograph.factory import register_tool_factory

        lookup_tool = FakeTool("lookup", response="found")
        register_tool_factory("lookup", lambda config, tool_config: lookup_tool)

        # Custom fake: json_mode gather requires specific JSON text as final response,
        # which ReActFake's hardcoded "done" text can't provide.
        call_n = {"n": 0}

        class FakeGatherLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kwargs):
                call_n["n"] += 1
                if call_n["n"] == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "lookup", "args": {}, "id": "c1"}]
                    return msg
                return AIMessage(content='{"items": ["gathered-json"]}')

        configure_llm(
            llm_factory=lambda tier: FakeGatherLLM(),
            prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": "go"}],
        )

        node = Node(
            name="research",
            mode="gather",
            output=Claims,
            model="fast",
            prompt="test",
            tools=[Tool(name="lookup", budget=1)],
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-gather-json", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert result["research"].items == ["gathered-json"]


class TestPromptCompilerReceivesOutputModel:
    """prompt_compiler must receive output_model and llm_config for json_mode prompts."""

    def test_compiler_receives_output_model_in_produce(self):
        """Produce node's prompt compiler sees the output Pydantic model."""
        compiler_calls = []

        def tracking_compiler(template, data, **kw):
            compiler_calls.append(kw)
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            lambda tier: StructuredFake(lambda m: m(items=["ok"])),
            prompt_compiler=tracking_compiler,
        )

        node = Node(name="x", mode="produce", output=Claims, model="fast", prompt="test")
        pipeline = Construct("test", nodes=[node])
        graph = compile(pipeline)
        run(graph, input={"node_id": "test"})

        assert len(compiler_calls) == 1
        assert compiler_calls[0].get("output_model") is Claims

    def test_compiler_receives_llm_config(self):
        """Prompt compiler sees the node's llm_config including output_strategy."""
        compiler_calls = []

        def tracking_compiler(template, data, **kw):
            compiler_calls.append(kw)
            return [{"role": "user", "content": "test"}]

        configure_fake_llm(
            lambda tier: TextFake('{"items": ["ok"]}'),
            prompt_compiler=tracking_compiler,
        )

        node = Node(
            name="x", mode="produce", output=Claims, model="fast", prompt="test",
            llm_config={"output_strategy": "json_mode", "temperature": 0.5},
        )
        pipeline = Construct("test", nodes=[node])
        graph = compile(pipeline)
        run(graph, input={"node_id": "test"})

        assert len(compiler_calls) == 1
        assert compiler_calls[0].get("llm_config") == {"output_strategy": "json_mode", "temperature": 0.5}
        assert compiler_calls[0].get("output_model") is Claims

    def test_json_mode_compiler_can_inject_schema(self):
        """End-to-end: compiler injects JSON schema into prompt for json_mode."""
        import json

        injected_prompts = []

        def schema_injecting_compiler(template, data, **kw):
            output_model = kw.get("output_model")
            llm_config = kw.get("llm_config", {})
            strategy = llm_config.get("output_strategy", "structured")

            prompt = f"Analyze: {template}"
            if strategy in ("json_mode", "text") and output_model:
                schema = json.dumps(output_model.model_json_schema(), indent=2)
                prompt += f"\n\nReturn a JSON object matching this schema:\n{schema}"

            injected_prompts.append(prompt)
            return [{"role": "user", "content": prompt}]

        configure_fake_llm(
            lambda tier: TextFake('{"items": ["schema-injected"]}'),
            prompt_compiler=schema_injecting_compiler,
        )

        node = Node(
            name="x", mode="produce", output=Claims, model="fast", prompt="decompose",
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert result["x"].items == ["schema-injected"]
        # Verify schema was injected into the prompt
        assert "json_schema" in injected_prompts[0] or "items" in injected_prompts[0]
        assert "Return a JSON object" in injected_prompts[0]


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

    def test_node_decorator_basic_chain(self):
        """Two @node-decorated scripted functions wired by parameter name,
        assembled via construct_from_module, compile and run end-to-end."""
        from neograph import compile, construct_from_module, node, run

        mod = self._fresh_module("test_basic_chain_mod")

        @node(mode="scripted", output=RawText)
        def seed() -> RawText:
            return RawText(text="hello world")

        @node(mode="scripted", output=Claims)
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

    def test_node_decorator_fan_in_three_upstreams(self):
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

        @node(mode="scripted", output=A)
        def alpha() -> A:
            return A(value="a")

        @node(mode="scripted", output=B)
        def beta() -> B:
            return B(value="b")

        @node(mode="scripted", output=C)
        def gamma() -> C:
            return C(value="c")

        @node(mode="scripted", output=Report)
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

    def test_node_decorator_explicit_kwargs_override_annotations(self):
        """Explicit @node(output=X) beats the function's return annotation."""
        from neograph import construct_from_module, node

        class Bogus(BaseModel, frozen=True):
            nope: str

        mod = self._fresh_module("test_kwargs_override_mod")

        @node(mode="scripted", output=Claims)  # explicit output overrides `-> Bogus`
        def producer() -> Bogus:  # intentional mismatch
            return Claims(items=["overridden"])

        mod.producer = producer

        pipeline = construct_from_module(mod)
        (only_node,) = pipeline.nodes
        assert only_node.output is Claims
        assert only_node.output is not Bogus

    def test_node_decorator_unknown_param_raises(self):
        """A parameter that doesn't name any @node in the module raises
        ConstructError with a helpful message."""
        from neograph import ConstructError, construct_from_module, node

        mod = self._fresh_module("test_unknown_param_mod")

        @node(mode="scripted", output=Claims)
        def orphan(ghost: RawText) -> Claims:
            return Claims(items=["x"])

        mod.orphan = orphan

        with pytest.raises(ConstructError) as exc_info:
            construct_from_module(mod)
        msg = str(exc_info.value)
        assert "ghost" in msg
        assert "orphan" in msg

    def test_node_decorator_topological_ordering(self):
        """Out-of-declaration-order dependencies get sorted correctly."""
        from neograph import construct_from_module, node

        mod = self._fresh_module("test_topo_mod")

        # Attach to module in a SHUFFLED order (report, seed, split).
        # Declaration order inside the function body is also shuffled: the
        # downstream-most node is declared first.

        @node(mode="scripted", output=ClassifiedClaims)
        def report(split: Claims) -> ClassifiedClaims:
            return ClassifiedClaims(
                classified=[{"claim": c, "category": "x"} for c in split.items],
            )

        @node(mode="scripted", output=RawText)
        def seed() -> RawText:
            return RawText(text="a b c")

        @node(mode="scripted", output=Claims)
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

    def test_node_decorator_name_underscore_to_hyphen(self):
        """Function `make_clusters` becomes node 'make-clusters'; downstream
        parameter `make_clusters` resolves to it."""
        from neograph import compile, construct_from_module, node, run

        mod = self._fresh_module("test_name_convention_mod")

        @node(mode="scripted", output=Claims)
        def seed_text() -> Claims:
            return Claims(items=["one", "two"])

        @node(mode="scripted", output=Clusters)
        def make_clusters(seed_text: Claims) -> Clusters:
            return Clusters(
                groups=[ClusterGroup(label="g", claim_ids=list(seed_text.items))],
            )

        @node(mode="scripted", output=ClassifiedClaims)
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
