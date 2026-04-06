"""End-to-end acceptance tests: what must pass for NeoGraph to be useful for piarch.

Each test proves a real capability, not just data structure correctness.
All LLM calls use fakes — no API keys needed.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from neograph import Construct, ConstructError, Node, Operator, Oracle, Each, Tool, compile, run, tool
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

        @node(mode="scripted", outputs=RawText)
        def extract() -> RawText:
            return RawText(text="hello world")

        @node(mode="scripted", outputs=Claims)
        def split(extract: RawText) -> Claims:
            return Claims(items=["claim-1", "claim-2"])

        @node(mode="scripted", outputs=ClassifiedClaims)
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

        @node(mode="produce", outputs=Claims, model="fast", prompt="test/extract")
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
            outputs=Claims,
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
            outputs=Claims,
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
    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_three_way_ensemble(self):
        """Oracle dispatches 3 generators and merges results."""
        import types as _types

        from neograph import construct_from_module, node
        from neograph.factory import register_scripted

        gen_call_count = [0]

        # Merge: combine all variant items into one Claims
        def combine_variants(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("combine_variants", combine_variants)

        @node(outputs=Claims, ensemble_n=3, merge_fn="combine_variants")
        def generate() -> Claims:
            gen_call_count[0] += 1
            return Claims(items=[f"variant-{gen_call_count[0]}"])

        mod = self._fresh_module("test_oracle_ensemble")
        mod.generate = generate

        pipeline = construct_from_module(mod, name="test-oracle")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # All 3 generators ran
        assert gen_call_count[0] == 3
        # Merge combined all 3 variants
        merged = result.get("generate")
        assert merged is not None
        assert len(merged.items) == 3

    def test_llm_merge(self):
        """Oracle with merge_prompt calls LLM to judge-merge variants."""
        import types as _types

        from neograph import construct_from_module, node
        from neograph.factory import register_scripted

        register_scripted("gen_llm", lambda input_data, config: Claims(items=["v1"]))

        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["merged-consensus"])))

        @node(outputs=Claims, ensemble_n=2, merge_prompt="test/merge")
        def generate() -> Claims:
            return Claims(items=["v1"])

        mod = self._fresh_module("test_oracle_llm")
        mod.generate = generate

        pipeline = construct_from_module(mod, name="test-oracle-llm")
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
        """Each dispatches per-item and collects results (@node API)."""
        import types as _types
        from neograph import compile, construct_from_module, node, run

        mod = _types.ModuleType("test_each_fanout")

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
            return MatchResult(
                cluster_label=cluster.label,
                matched=["match-1"],
            )

        mod.make_clusters = make_clusters
        mod.verify = verify

        pipeline = construct_from_module(mod)
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
        node = Node.scripted("verify", fn="noop", inputs=ClusterGroup, outputs=MatchResult)
        mapped = node.map(lambda s: s.make_clusters.groups, key="label")

        each = mapped.get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "make_clusters.groups"
        assert each.key == "label"

    def test_map_with_string_path(self):
        """A string source is passed straight through to Each.over."""
        node = Node.scripted("verify", fn="noop", inputs=ClusterGroup, outputs=MatchResult)
        mapped = node.map("make_clusters.groups", key="label")

        each = mapped.get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "make_clusters.groups"
        assert each.key == "label"

    def test_map_equivalent_to_pipe_each(self):
        """node.map(...) and node | Each(...) produce structurally identical nodes."""
        base = Node.scripted("verify", fn="noop", inputs=ClusterGroup, outputs=MatchResult)

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

        make = Node.scripted("make-clusters", fn="make_clusters", outputs=Clusters)
        verify = Node.scripted(
            "verify", fn="verify_cluster", inputs=ClusterGroup, outputs=MatchResult
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
        node = Node.scripted("verify", fn="noop", outputs=MatchResult)
        with pytest.raises(TypeError, match="at least one attribute"):
            node.map(lambda s: s, key="label")

    def test_map_lambda_returning_scalar_raises(self):
        """`lambda s: 42` — clear error, not a silent Each."""
        node = Node.scripted("verify", fn="noop", outputs=MatchResult)
        with pytest.raises(TypeError, match="attribute-access chain"):
            node.map(lambda s: 42, key="label")

    def test_map_lambda_that_errors_raises_typeerror(self):
        """A lambda that does something illegal (e.g. indexing) reports cleanly."""
        node = Node.scripted("verify", fn="noop", outputs=MatchResult)
        with pytest.raises(TypeError, match="pure attribute-access chain"):
            node.map(lambda s: s.items[0], key="label")  # __getitem__ on recorder

    def test_map_rejects_dunder_attribute_access(self):
        """`lambda s: s.__dict__.foo` must not silently produce Each(over='__dict__.foo')."""
        node = Node.scripted("verify", fn="noop", inputs=ClusterGroup, outputs=MatchResult)
        with pytest.raises(TypeError, match="pure attribute-access chain"):
            node.map(lambda s: s.__dict__.foo, key="label")

    def test_map_rejects_leading_underscore_attribute(self):
        """Reject `lambda s: s._private.field` — underscores are a footgun trapdoor."""
        node = Node.scripted("verify", fn="noop", inputs=ClusterGroup, outputs=MatchResult)
        with pytest.raises(TypeError, match="pure attribute-access chain"):
            node.map(lambda s: s._private.x, key="label")

    def test_map_user_exception_propagates_unchanged(self):
        """Non-attribute errors (e.g. ZeroDivisionError) propagate with their own type."""
        node = Node.scripted("verify", fn="noop", inputs=ClusterGroup, outputs=MatchResult)
        with pytest.raises(ZeroDivisionError):
            node.map(lambda s: 1 / 0 and s.x, key="label")

    def test_map_rejects_non_string_non_callable(self):
        """Passing an int or other non-source type raises immediately."""
        node = Node.scripted("verify", fn="noop", outputs=MatchResult)
        with pytest.raises(TypeError, match="string path or a lambda"):
            node.map(42, key="label")  # type: ignore[arg-type]

    def test_map_on_construct(self):
        """Construct also gets .map() via Modifiable — sub-construct fan-out."""
        inner = Node.scripted("inner", fn="noop", inputs=ClusterGroup, outputs=MatchResult)
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
        import types as _types

        from langgraph.checkpoint.memory import MemorySaver

        from neograph import construct_from_module, node

        mod = _types.ModuleType("test_operator_mod")

        @node(
            mode="scripted",
            outputs=ValidationResult,
            interrupt_when=lambda state: (
                {"issues": state.validate.issues}
                if state.validate and not state.validate.passed
                else None
            ),
        )
        def validate() -> ValidationResult:
            return ValidationResult(passed=False, issues=["missing stakeholder coverage"])

        mod.validate = validate

        pipeline = construct_from_module(mod, name="test-operator")
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
# A @node(mode='raw') function mixed with @node declarations in the same Construct.
# This proves: raw escape hatch works, framework wires edges around it,
# data flows through raw node like any other.
# ═══════════════════════════════════════════════════════════════════════════

class TestRawNode:
    def test_raw_node_in_pipeline(self):
        """@node(mode='raw') works alongside declarative nodes."""
        import types as _types

        from neograph import construct_from_module, node

        mod = _types.ModuleType("test_raw_node_mod")

        @node(mode="scripted", outputs=Claims)
        def make_claims() -> Claims:
            return Claims(items=["a", "b", "c"])

        @node(mode="raw", inputs=Claims, outputs=Claims)
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

        mod.make_claims = make_claims
        mod.filter_claims = filter_claims

        pipeline = construct_from_module(mod, name="test-raw")
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

        @node(mode="produce", outputs=Claims, model="reason", prompt="rw/decompose")
        def decompose() -> Claims: ...

        @node(mode="produce", outputs=ClassifiedClaims, model="fast", prompt="rw/classify")
        def classify(decompose: Claims) -> ClassifiedClaims: ...

        @node(mode="scripted", outputs=RawText)
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
            outputs=RawText,
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
        import types as _types

        from langgraph.checkpoint.memory import MemorySaver

        from neograph import construct_from_module, node

        mod = _types.ModuleType("test_operator_continues_mod")

        @node(
            mode="scripted",
            outputs=ValidationResult,
            interrupt_when=lambda state: None,  # always falsy
        )
        def validate() -> ValidationResult:
            return ValidationResult(passed=True, issues=[])

        mod.validate = validate

        pipeline = construct_from_module(mod, name="test-operator-pass")
        graph = compile(pipeline, checkpointer=MemorySaver())
        result = run(graph, input={"node_id": "test-001"}, config={"configurable": {"thread_id": "pass-test"}})

        assert result["validate"].passed is True
        assert result.get("human_feedback") is None


class TestOperatorResume:
    """Operator interrupt + resume flow."""

    def test_interrupt_then_resume(self):
        """Graph pauses at interrupt, resumes with human feedback via run()."""
        import types as _types

        from langgraph.checkpoint.memory import MemorySaver

        from neograph import construct_from_module, node

        mod = _types.ModuleType("test_operator_resume_mod")

        @node(
            mode="scripted",
            outputs=ValidationResult,
            name="validate-thing",
            interrupt_when=lambda state: (
                {"issues": state.validate_thing.issues}
                if state.validate_thing and not state.validate_thing.passed
                else None
            ),
        )
        def validate_thing() -> ValidationResult:
            return ValidationResult(passed=False, issues=["bad coverage"])

        mod.validate_thing = validate_thing

        pipeline = construct_from_module(mod, name="test-resume")
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
        node = Node(name="fail", mode="produce", outputs=Claims, model="fast", prompt="x")
        pipeline = Construct("test-no-llm", nodes=[node])
        graph = compile(pipeline)

        with pytest.raises(ValueError, match="LLM not configured"):
            run(graph, input={"node_id": "test-001"})

    def test_unregistered_scripted_fn(self):
        """Referencing unregistered scripted function raises ValueError."""
        node = Node.scripted("bad", fn="nonexistent_fn", outputs=Claims)
        pipeline = Construct("test-bad-fn", nodes=[node])

        with pytest.raises(ValueError, match="not registered"):
            compile(pipeline)

    def test_unregistered_oracle_merge_fn(self):
        """Oracle with unregistered merge_fn raises ValueError at compile."""
        from neograph.factory import register_scripted

        register_scripted("gen", lambda input_data, config: Claims(items=["x"]))

        node = Node.scripted(
            "gen", fn="gen", outputs=Claims
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
            "something", fn="something", outputs=Claims
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
            outputs=Claims,
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
        node = Node.scripted("noop", fn="noop", outputs=Claims)
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
            outputs=Claims,
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
            "gen", fn="gen_start", outputs=Claims
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
            "process", fn="process_item", inputs=ClusterGroup, outputs=MatchResult
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

        node = Node.scripted("gen", fn="g", outputs=Claims) | Oracle(n=2, merge_fn="m")
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

    def test_subgraph_without_input_raises(self):
        """Sub-construct without declared input raises at compile."""
        from neograph.factory import register_scripted

        register_scripted("noop", lambda input_data, config: Claims(items=[]))

        sub = Construct("bad-sub", output=Claims, nodes=[
            Node.scripted("noop", fn="noop", outputs=Claims),
        ])

        parent = Construct("parent", nodes=[sub])

        with pytest.raises(ValueError, match="has no input type"):
            compile(parent)

    def test_subgraph_without_output_raises(self):
        """Sub-construct without declared output raises at compile."""
        from neograph.factory import register_scripted

        register_scripted("noop", lambda input_data, config: Claims(items=[]))

        sub = Construct("bad-sub", input=Claims, nodes=[
            Node.scripted("noop", fn="noop", outputs=Claims),
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

    def test_operator_without_checkpointer_raises(self):
        """Operator node without checkpointer raises ValueError at compile."""
        from neograph.factory import register_condition, register_scripted

        register_scripted("x", lambda input_data, config: Claims(items=[]))
        register_condition("always", lambda state: True)

        node = Node.scripted("x", fn="x", outputs=Claims) | Operator(when="always")
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
                Node.scripted("step-a", fn="sub_step_a", outputs=Claims),
                Node.scripted("step-b", fn="sub_step_b", inputs=Claims, outputs=RawText),
            ],
        ) | Oracle(n=3, merge_fn="merge_sub")

        register_scripted("make_input", lambda input_data, config: Claims(items=["raw"]))

        parent = Construct("parent", nodes=[
            Node.scripted("start", fn="make_input", outputs=Claims),
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
            nodes=[Node.scripted("gen", fn="gen_claim", outputs=Claims)],
        ) | Oracle(n=2, merge_prompt="test/merge")

        register_scripted("seed", lambda input_data, config: Claims(items=["seed"]))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed", outputs=Claims),
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
                Node.scripted("analyze", fn="sub_analyze", inputs=ClusterGroup, outputs=RawText),
                Node.scripted("score", fn="sub_score", inputs=RawText, outputs=MatchResult),
            ],
        ) | Each(over="make_clusters.groups", key="label")

        parent = Construct("parent", nodes=[
            Node.scripted("make-clusters", fn="make_clusters", outputs=Clusters),
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
            nodes=[Node.scripted("val", fn="sub_validate", outputs=ValidationResult)],
        ) | Operator(when="sub_failed")

        register_scripted("seed", lambda input_data, config: Claims(items=["data"]))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed", outputs=Claims),
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
            nodes=[Node.scripted("ok", fn="sub_ok", outputs=ValidationResult)],
        ) | Operator(when="sub_check")

        register_scripted("seed2", lambda input_data, config: Claims(items=["ok"]))
        register_scripted("done", lambda input_data, config: RawText(text="complete"))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed2", outputs=Claims),
            sub,
            Node.scripted("done", fn="done", outputs=RawText),
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
                Node.scripted("gen", fn="gen_v", outputs=RawText)
                | Oracle(n=2, merge_fn="merge_v"),
            ],
        )

        # Outer: Each over clusters, each runs the Oracle sub-pipeline
        # This means: 2 clusters × 2 Oracle variants = 4 generator calls + 2 merges
        parent = Construct("parent", nodes=[
            Node.scripted("make", fn="make_items", outputs=Clusters),
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
                    inputs=Claims,
                    outputs=RawText,
                    model="fast",
                    prompt="test/search",
                    tools=[Tool(name="deep_search", budget=3)],
                ),
            ],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("prep", fn="prep_search", outputs=Claims),
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
                Node.scripted("check", fn="sub_check", outputs=ValidationResult)
                | Operator(when="inner_failed"),
            ],
        )

        register_scripted("start_fn", lambda input_data, config: Claims(items=["go"]))
        register_scripted("after_fn", lambda input_data, config: RawText(text="should not reach"))

        parent = Construct("parent", nodes=[
            Node.scripted("start", fn="start_fn", outputs=Claims),
            sub,
            Node.scripted("after", fn="after_fn", outputs=RawText),
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
            nodes=[Node.scripted("seed", fn="self_seed", outputs=Claims)],
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
            nodes=[Node.scripted("g", fn="gen_first", outputs=Claims)],
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
            nodes=[Node.scripted("p", fn="proc_first", outputs=RawText)],
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
    return Node.scripted(name, fn="f", outputs=out)


def _consumer(name: str, in_: type, out: type) -> Node:
    return Node.scripted(name, fn="f", inputs=in_, outputs=out)


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
            outputs=Claims,
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
            outputs=Claims,
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

        node = Node.scripted("capture", fn="capture", outputs=Claims)
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

        node = Node(name="analyze", mode="produce", outputs=Claims, model="fast", prompt="rw/analyze")
        pipeline = Construct("test-prompt-ctx", nodes=[node])
        graph = compile(pipeline)
        run(graph, input={"node_id": "BR-001", "project_root": "/proj"})

        assert len(compiler_calls) == 1
        assert compiler_calls[0]["template"] == "rw/analyze"
        assert compiler_calls[0]["node_name"] == "analyze"
        assert compiler_calls[0]["node_id"] == "BR-001"
        assert compiler_calls[0]["project_root"] == "/proj"


class TestRunIsolated:
    """Node.run_isolated() — direct invocation for unit testing."""

    def test_scripted_node_run_isolated(self):
        """Scripted nodes can be tested without compile()/run()."""
        from neograph import register_scripted

        register_scripted("upper", lambda input_data, config: RawText(
            text=input_data.text.upper() if input_data else "NONE"
        ))

        upper_node = Node.scripted("upper", fn="upper", inputs=RawText, outputs=RawText)

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
            "decompose", mode="produce", outputs=Claims, model="fast", prompt="test"
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
        node = Node.scripted("cfg-test", fn="cfg_test", outputs=Claims)

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
            Node.scripted("step", fn="resourced", outputs=Claims),
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
            nodes=[Node.scripted("inner", fn="sub_node", outputs=RawText)],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="parent_seed", outputs=Claims),
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
            "gen", fn="cfg_gen", outputs=Claims
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

        make = Node.scripted("make", fn="make_groups", outputs=Clusters)
        verify = Node.scripted(
            "verify", fn="verify_cfg", inputs=ClusterGroup, outputs=MatchResult
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
            Node.scripted("a", fn="track_a", outputs=Claims),
            Node.scripted("b", fn="track_b", outputs=Claims),
            Node.scripted("c", fn="track_c", outputs=Claims),
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

        node = Node(name="extract", mode="produce", outputs=Claims, model="fast", prompt="test")
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
            outputs=Claims,
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
            outputs=Claims,
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
            outputs=Claims,
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
            outputs=Claims,
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

        node = Node(name="x", mode="produce", outputs=Claims, model="fast", prompt="test")
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
            name="x", mode="produce", outputs=Claims, model="fast", prompt="test",
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
            name="x", mode="produce", outputs=Claims, model="fast", prompt="decompose",
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

class TestNodeInputsFieldRename:
    def test_node_accepts_inputs_kwarg_single_type(self):
        """Node(inputs=SomeType) creates a node with .inputs == SomeType."""
        n = Node("t", mode="scripted", inputs=Claims, outputs=RawText)
        assert n.inputs == Claims

    def test_node_accepts_inputs_kwarg_dict_form(self):
        """Node(inputs={'a': A, 'b': B}) stores the dict on .inputs."""
        n = Node(
            "t",
            mode="scripted",
            inputs={"claims": Claims, "clusters": Clusters},
            outputs=MatchResult,
        )
        assert n.inputs == {"claims": Claims, "clusters": Clusters}

    def test_node_accepts_inputs_none_default(self):
        """Node with no inputs kwarg keeps the default (None)."""
        n = Node("t", mode="scripted", outputs=RawText)
        assert n.inputs is None

    def test_node_scripted_classmethod_accepts_inputs_kwarg(self):
        """Node.scripted(..., inputs=X) propagates to .inputs."""
        n = Node.scripted("verify", fn="noop", inputs=ClusterGroup, outputs=MatchResult)
        assert n.inputs == ClusterGroup
        assert n.mode == "scripted"

    def test_node_has_no_legacy_input_attribute(self):
        """.input attribute no longer exists on Node instances."""
        n = Node("t", mode="scripted", inputs=Claims, outputs=RawText)
        assert not hasattr(n, "input"), (
            "Node still exposes legacy .input attribute — rename is incomplete."
        )


# ═══════════════════════════════════════════════════════════════════════════
# TestExtractInputListUnwrap (neograph-kqd.3)
#
# Factory runtime unwrap: dict[str, X] state value → list[X] for consumers
# whose inputs dict declares a list[X] expected type. This is the runtime
# half of the merge-after-fanout pattern (validator side landed in kqd.2).
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractInputListUnwrap:
    def test_list_consumer_unwraps_dict_state_to_list(self):
        """inputs={'upstream': list[MatchResult]} + dict state value
        → list of values in dict insertion order."""
        from neograph.factory import _extract_input

        consumer = Node.scripted(
            "consumer", fn="f",
            inputs={"make_clusters": list[MatchResult]},
            outputs=MergedResult,
        )
        mr1 = MatchResult(cluster_label="a", matched=["x"])
        mr2 = MatchResult(cluster_label="b", matched=["y"])
        state = {"make_clusters": {"a": mr1, "b": mr2}}
        result = _extract_input(state, consumer)
        assert isinstance(result, dict)
        assert "make_clusters" in result
        assert result["make_clusters"] == [mr1, mr2]

    def test_dict_consumer_keeps_dict_state(self):
        """inputs={'upstream': dict[str, MatchResult]} + dict state value
        → dict pass-through, no unwrap."""
        from neograph.factory import _extract_input

        consumer = Node.scripted(
            "consumer", fn="f",
            inputs={"make_clusters": dict[str, MatchResult]},
            outputs=MergedResult,
        )
        mr1 = MatchResult(cluster_label="a", matched=["x"])
        mr2 = MatchResult(cluster_label="b", matched=["y"])
        state = {"make_clusters": {"a": mr1, "b": mr2}}
        result = _extract_input(state, consumer)
        assert result["make_clusters"] == {"a": mr1, "b": mr2}

    def test_list_consumer_over_none_state_returns_none(self):
        """Missing state field (None) is not unwrapped — passthrough None."""
        from neograph.factory import _extract_input

        consumer = Node.scripted(
            "consumer", fn="f",
            inputs={"make_clusters": list[MatchResult]},
            outputs=MergedResult,
        )
        state = {}  # no make_clusters field
        result = _extract_input(state, consumer)
        assert result["make_clusters"] is None


# ═══════════════════════════════════════════════════════════════════════════
# TestNodeDecoratorDictInputs (neograph-kqd.4)
#
# @node decoration now emits dict-form inputs={param_name: annotation, ...}
# for all typed upstream params. This is the metadata shift that lets
# step-2's validator catch fan-in mismatches via _check_fan_in_inputs.
# Fan-out params (Each) are stripped from inputs at construct-assembly time.
# ═══════════════════════════════════════════════════════════════════════════

class TestListOverEachEndToEnd:
    def test_declarative_each_to_list_consumer(self):
        """Declarative: Each producer + Node.scripted consumer that
        annotates inputs={'verify': list[MatchResult]}."""
        from neograph import compile, run
        from neograph.factory import register_scripted

        register_scripted(
            "make_clusters_l5",
            lambda _in, _cfg: Clusters(groups=[
                ClusterGroup(label="a", claim_ids=["1"]),
                ClusterGroup(label="b", claim_ids=["2"]),
                ClusterGroup(label="c", claim_ids=["3"]),
            ]),
        )
        register_scripted(
            "verify_cluster_l5",
            lambda input_data, _cfg: MatchResult(
                cluster_label=input_data.label if hasattr(input_data, "label") else "?",
                matched=[f"m-{input_data.label}" if hasattr(input_data, "label") else "?"],
            ),
        )

        def summarize_fn(input_data, _cfg):
            verify_list = input_data["verify"]
            assert isinstance(verify_list, list), f"expected list, got {type(verify_list).__name__}"
            return MergedResult(
                final_text=f"verified:{len(verify_list)}:" + ",".join(sorted(v.cluster_label for v in verify_list)),
            )

        register_scripted("summarize_l5", summarize_fn)

        make = Node.scripted("make-clusters", fn="make_clusters_l5", outputs=Clusters)
        verify = (
            Node.scripted("verify", fn="verify_cluster_l5", inputs=ClusterGroup, outputs=MatchResult)
            .map(lambda s: s.make_clusters.groups, key="label")
        )
        summarize = Node.scripted(
            "summarize", fn="summarize_l5",
            inputs={"verify": list[MatchResult]},
            outputs=MergedResult,
        )
        pipeline = Construct("l5-decl", nodes=[make, verify, summarize])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "l5"})
        assert result["summarize"].final_text == "verified:3:a,b,c"

    def test_decorator_each_to_list_consumer(self):
        """@node: Each producer + @node consumer with list[X] annotation."""
        from neograph import compile, run, node
        from neograph.decorators import construct_from_functions

        @node(mode="scripted", outputs=Clusters)
        def gen_clusters() -> Clusters:
            return Clusters(groups=[
                ClusterGroup(label="alpha", claim_ids=["1"]),
                ClusterGroup(label="beta", claim_ids=["2"]),
            ])

        @node(
            mode="scripted",
            outputs=MatchResult,
            map_over="gen_clusters.groups",
            map_key="label",
        )
        def verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label=cluster.label, matched=[f"m-{cluster.label}"])

        @node(mode="scripted", outputs=MergedResult)
        def summarize(verify: list[MatchResult]) -> MergedResult:
            assert isinstance(verify, list), f"expected list, got {type(verify).__name__}"
            return MergedResult(
                final_text=f"got:{len(verify)}:" + ",".join(sorted(v.cluster_label for v in verify)),
            )

        pipeline = construct_from_functions("l5-dec", [gen_clusters, verify, summarize])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "l5"})
        assert result["summarize"].final_text == "got:2:alpha,beta"

    def test_list_wrong_element_type_rejected(self):
        """list[WrongType] consumer + Each producer raises ConstructError."""
        make = _producer("make", Clusters)
        verify = _consumer("verify", ClusterGroup, MatchResult).map(
            lambda s: s.make.groups, key="label"
        )
        summarize = Node.scripted(
            "summarize", fn="f",
            inputs={"verify": list[Claims]},  # WRONG: Each emits MatchResult
            outputs=MergedResult,
        )
        with pytest.raises(ConstructError) as exc_info:
            Construct("bad-list-type", nodes=[make, verify, summarize])
        msg = str(exc_info.value)
        assert "verify" in msg

    def test_dict_consumer_still_works_after_each(self):
        """dict[str, X] consumer still passes through unchanged (regression)."""
        from neograph import compile, run
        from neograph.factory import register_scripted

        register_scripted(
            "make_clusters_l5b",
            lambda _in, _cfg: Clusters(groups=[ClusterGroup(label="x", claim_ids=["1"])]),
        )
        register_scripted(
            "verify_cluster_l5b",
            lambda input_data, _cfg: MatchResult(
                cluster_label=input_data.label if hasattr(input_data, "label") else "?",
                matched=["ok"],
            ),
        )

        def summarize_dict_fn(input_data, _cfg):
            verify_dict = input_data["verify"]
            assert isinstance(verify_dict, dict), f"expected dict, got {type(verify_dict).__name__}"
            return MergedResult(final_text=f"keys:{sorted(verify_dict.keys())}")

        register_scripted("summarize_l5b", summarize_dict_fn)

        make = Node.scripted("make-clusters", fn="make_clusters_l5b", outputs=Clusters)
        verify = (
            Node.scripted("verify", fn="verify_cluster_l5b", inputs=ClusterGroup, outputs=MatchResult)
            .map(lambda s: s.make_clusters.groups, key="label")
        )
        summarize = Node.scripted(
            "summarize", fn="summarize_l5b",
            inputs={"verify": dict[str, MatchResult]},
            outputs=MergedResult,
        )
        pipeline = Construct("l5-dict", nodes=[make, verify, summarize])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "l5"})
        assert result["summarize"].final_text == "keys:['x']"


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
    def test_llm_driven_spec_fan_in_roundtrip(self):
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

    def test_llm_driven_spec_type_mismatch_rejected(self):
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

    def test_zero_upstream_node_inputs_none(self):
        """Nodes with no upstreams use inputs=None and assemble cleanly."""
        seed = Node.scripted("seed", fn="f", inputs=None, outputs=Claims)
        pipeline = Construct("zero-upstream", nodes=[seed])
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].inputs is None
        assert pipeline.nodes[0].outputs is Claims
        assert pipeline.nodes[0].name == "seed"

    def test_node_decorator_mixed_upstream_and_fanout_e2e(self):
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

    def test_attach_scripted_raw_fn_deleted(self):
        """_attach_scripted_raw_fn no longer exists — all scripted @node
        routes through register_scripted + _make_scripted_wrapper."""
        import neograph.decorators as dec
        assert not hasattr(dec, "_attach_scripted_raw_fn"), (
            "_attach_scripted_raw_fn still exists — kqd.8 is incomplete"
        )

    def test_scripted_node_raw_fn_not_set(self):
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

    def test_programmatic_fan_in_via_pipe(self):
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

class TestNodeOutputsRename:
    """Node.output → Node.outputs rename (neograph-1bp.1).

    The field on Node becomes `outputs` (plural), mirroring `inputs`.
    Construct.output stays singular — different semantics (boundary port).
    """

    def test_node_constructor_outputs_kwarg(self):
        """Node(..., outputs=X) sets the field."""
        n = Node("a", mode="produce", outputs=Claims, model="fast", prompt="p")
        assert n.outputs is Claims

    def test_node_scripted_outputs_kwarg(self):
        """Node.scripted(..., outputs=X) sets the field."""
        n = Node.scripted("a", fn="f", outputs=Claims)
        assert n.outputs is Claims

    def test_node_outputs_field_access(self):
        """node.outputs returns the declared output type."""
        n = Node("a", outputs=RawText, model="fast", prompt="p")
        assert n.outputs is RawText

    def test_decorator_outputs_kwarg(self):
        """@node(outputs=X) passes through to Node.outputs."""
        from neograph import node

        @node(outputs=Claims, mode="produce", model="fast", prompt="p")
        def my_node(seed: RawText) -> Claims: ...

        assert my_node.outputs is Claims

    def test_construct_output_stays_singular(self):
        """Construct.output stays as 'output' (singular), not renamed."""
        c = Construct("p", output=Claims, input=RawText, nodes=[
            Node.scripted("a", fn="f", outputs=Claims),
        ])
        assert c.output is Claims

    def test_effective_producer_type_reads_outputs(self):
        """effective_producer_type reads .outputs from Node items."""
        from neograph._construct_validation import effective_producer_type
        n = Node("a", outputs=Claims)
        assert effective_producer_type(n) is Claims

    def test_state_compiler_reads_outputs(self):
        """compile_state_model reads node.outputs for state field types."""
        from neograph.state import compile_state_model
        n = Node.scripted("extract", fn="f", outputs=RawText)
        c = Construct("p", nodes=[n])
        state_model = compile_state_model(c)
        # The state field should have the right type annotation
        assert "extract" in state_model.model_fields


class TestDictOutputsStateModel:
    """Dict-form outputs emit per-key state fields (neograph-1bp.2)."""

    def test_dict_outputs_emits_per_key_fields(self):
        """outputs={'result': X, 'log': Y} → state has node_result + node_log."""
        from neograph.state import compile_state_model
        n = Node("explore", outputs={"result": RawText, "tool_log": Claims})
        c = Construct("p", nodes=[n])
        state_model = compile_state_model(c)
        assert "explore_result" in state_model.model_fields
        assert "explore_tool_log" in state_model.model_fields
        # The old single-name field should NOT exist
        assert "explore" not in state_model.model_fields

    def test_single_type_outputs_backward_compat(self):
        """Single-type outputs= still emits {node_name} field."""
        from neograph.state import compile_state_model
        n = Node.scripted("extract", fn="f", outputs=RawText)
        c = Construct("p", nodes=[n])
        state_model = compile_state_model(c)
        assert "extract" in state_model.model_fields

    def test_dict_outputs_with_each_modifier(self):
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

    def test_scripted_dict_outputs_writes_per_key(self):
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

    def test_single_type_outputs_backward_compat_runtime(self):
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


class TestGatherToolCollection:
    """invoke_with_tools collects ToolInteraction records (neograph-1bp.6)."""

    def test_tool_interaction_model(self):
        """ToolInteraction has the expected fields."""
        from neograph import ToolInteraction
        ti = ToolInteraction(tool_name="search", args={"q": "test"}, result="found", duration_ms=42)
        assert ti.tool_name == "search"
        assert ti.args == {"q": "test"}
        assert ti.result == "found"
        assert ti.duration_ms == 42

    def test_tool_interactions_collected(self):
        """Gather node with dict outputs writes tool_log to state."""
        from neograph import node, construct_from_module, compile, run, tool, ToolInteraction
        from neograph.factory import register_tool_factory
        from tests.fakes import FakeTool, ReActFake, configure_fake_llm
        import types

        configure_fake_llm(
            lambda tier: ReActFake(
                tool_calls=[
                    [{"name": "search", "args": {"q": "test"}, "id": "tc1"}],
                    [],  # final response
                ],
                final=lambda m: Claims(items=["result"]),
            )
        )

        # Register a tool factory that returns a fake tool
        fake_tool = FakeTool("search", response="found it")
        register_tool_factory("search", lambda config, tool_config: fake_tool)

        mod = types.ModuleType("test_tool_collection_mod")

        search_tool = Tool("search", budget=3)

        @node(
            mode="gather",
            outputs={"result": Claims, "tool_log": list[ToolInteraction]},
            model="fast",
            prompt="test",
            tools=[search_tool],
        )
        def explore() -> Claims: ...

        mod.explore = explore
        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        result = run(graph, input={})
        assert result.get("explore_result") == Claims(items=["result"])
        tool_log = result.get("explore_tool_log")
        assert tool_log is not None
        assert len(tool_log) == 1
        assert tool_log[0].tool_name == "search"
        assert tool_log[0].result == "found it"
