"""Validation tests — assembly-time type checking, fan-in validation,
Each path resolution, effective_producer_type, list/dict compatibility,
dict-form outputs validation, Oracle error paths, and lint() DI validation.
"""


from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import BaseModel

from neograph import (
    CompileError,
    ConfigurationError,
    Construct,
    ConstructError,
    Each,
    ExecutionError,
    FromConfig,
    FromInput,
    Node,
    Oracle,
    Tool,
    compile,
    construct_from_functions,
    node,
    run,
)
from tests.fakes import build_test_compile_kwargs
from tests.schemas import (
    Claims,
    ClassifiedClaims,
    ClusterGroup,
    Clusters,
    MatchResult,
    MergedResult,
    RawText,
    ValidationResult,
    _consumer,
    _producer,
)

# ═══════════════════════════════════════════════════════════════════════════
# Construct assembly-time validation
# ═══════════════════════════════════════════════════════════════════════════

class TestConstructValidation:
    """Input/output compatibility is checked at Construct assembly time."""

    def test_valid_chain_assembles_when_types_match(self):
        """A correctly typed chain assembles without error."""
        a = _producer("a", RawText)
        b = _consumer("b", RawText, Claims)
        c = _consumer("c", Claims, ClassifiedClaims)
        pipeline = Construct("good", nodes=[a, b, c])
        assert len(pipeline.nodes) == 3
        assert [n.name for n in pipeline.nodes] == ["a", "b", "c"]

    def test_mismatch_raises_when_no_compatible_upstream(self):
        """Downstream input with no compatible upstream raises ConstructError
        AND the error message lists the upstream producers."""
        a = _producer("a", RawText)
        b = _consumer("b", Claims, ClassifiedClaims)
        with pytest.raises(ConstructError) as exc_info:
            Construct("bad", nodes=[a, b])
        msg = str(exc_info.value)
        assert "declares inputs=Claims" in msg
        assert "node 'a': RawText" in msg

    def test_hint_suggests_map_when_upstream_has_list_field(self):
        """When upstream has list[input_type] field, hint names the correct path."""
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult)
        with pytest.raises(ConstructError) as exc_info:
            Construct("bad-fanout", nodes=[a, b])
        msg = str(exc_info.value)
        assert "did you forget to fan out" in msg
        assert "s.a.groups" in msg

    def test_error_includes_source_location_when_mismatch(self):
        """Error message includes a file:line pointer to the user call site."""
        a = _producer("a", RawText)
        b = _consumer("b", Claims, ClassifiedClaims)
        with pytest.raises(ConstructError, match=r"at test_validation\.py:\d+"):
            Construct("bad-loc", nodes=[a, b])

    def test_each_assembles_when_path_resolves_to_list(self):
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

    def test_each_raises_when_field_missing(self):
        """Each path that walks to a non-existent field raises."""
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.nonexistent", key="label"
        )
        with pytest.raises(ConstructError, match="has no field 'nonexistent'"):
            Construct("bad-each-field", nodes=[a, b])

    def test_each_raises_when_terminal_not_list(self):
        """Each whose terminal field isn't a list is flagged."""
        a = _producer("a", RawText)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.text", key="label"
        )
        with pytest.raises(ConstructError, match="not a list"):
            Construct("bad-each-terminal", nodes=[a, b])

    def test_each_raises_when_list_element_type_wrong(self):
        """Each whose list element type doesn't match input raises."""
        a = _producer("a", Claims)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.items", key="label"
        )
        with pytest.raises(ConstructError, match=r"list\[str\]"):
            Construct("bad-each-element", nodes=[a, b])

    def test_first_item_deferred_when_has_input(self):
        """First-of-chain with declared input is NOT flagged -- runtime-seeded."""
        b = _consumer("b", Claims, ClassifiedClaims)
        pipeline = Construct("top-level", nodes=[b])
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].inputs is Claims

    def test_top_level_each_deferred_when_root_unknown(self):
        """Each at position 0 whose root isn't a known producer defers cleanly."""
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
        sub = Construct("sub", input=Claims, output=Claims, nodes=[inner])
        assert sub.input is Claims
        assert sub.output is Claims
        assert len(sub.nodes) == 1

    def test_sub_construct_validates_when_chained_in_parent(self):
        """Parent producing sub.input satisfies the sub-construct's input check."""
        upstream = _producer("upstream", Claims)
        sub = Construct(
            "sub", input=Claims, output=ClassifiedClaims,
            nodes=[_consumer("inner", Claims, ClassifiedClaims)],
        )
        parent = Construct("parent", nodes=[upstream, sub])
        assert len(parent.nodes) == 2
        assert parent.nodes[1].input is Claims

    def test_sub_construct_raises_when_parent_type_incompatible(self):
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
        assert "'sub' in construct 'parent'" in msg
        assert "declares input=Claims" in msg

    def test_construct_error_is_valueerror(self):
        """ConstructError subclasses ValueError for existing except clauses."""
        a = _producer("a", RawText)
        b = _consumer("b", Claims, ClassifiedClaims)
        with pytest.raises(ValueError):
            Construct("bad", nodes=[a, b])

    def test_dict_input_skipped_when_multi_field(self):
        """Nodes with dict[str, type] input spec aren't statically validated."""
        step_a = _producer("step-a", Claims)
        step_b = _producer("step-b", RawText)
        step_c = Node.scripted(
            "step-c", fn="f",
            inputs={"step_a": Claims, "step_b": RawText},
            outputs=RawText,
        )
        pipeline = Construct("multi-input", nodes=[step_a, step_b, step_c])
        assert len(pipeline.nodes) == 3
        assert isinstance(pipeline.nodes[2].inputs, dict)

    def test_dict_class_input_deferred_when_raw_dict(self):
        """input=dict (raw class) defers to runtime isinstance scan."""
        a = _producer("a", RawText)
        b = Node.scripted("b", fn="f", inputs=dict, outputs=Claims)
        pipeline = Construct("dict-class", nodes=[a, b])
        assert len(pipeline.nodes) == 2
        assert pipeline.nodes[1].inputs is dict

    def test_dict_generic_input_deferred_when_parameterized(self):
        """input=dict[str, X] (parameterized generic) defers to runtime."""
        a = _producer("a", RawText)
        b = Node.scripted("b", fn="f", inputs=dict[str, Claims], outputs=Claims)
        pipeline = Construct("dict-generic", nodes=[a, b])
        assert len(pipeline.nodes) == 2
        assert pipeline.nodes[1].inputs == dict[str, Claims]

    def test_each_downstream_rejected_when_raw_input(self):
        """Consumer declaring raw input=X after an Each-modified producer
        that emits dict[str, X] must be rejected at assembly time."""
        make = _producer("make", Clusters)
        verify = _consumer("verify", ClusterGroup, MatchResult).map(
            lambda s: s.make.groups, key="label"
        )
        summarize = _consumer("summarize", MatchResult, MergedResult)
        with pytest.raises(ConstructError, match=r"dict\[str, MatchResult\]"):
            Construct("bad", nodes=[make, verify, summarize])

    def test_each_downstream_accepted_when_dict_input(self):
        """Consumer with input=dict (raw class) after Each-modified producer passes."""
        make = _producer("make", Clusters)
        verify = _consumer("verify", ClusterGroup, MatchResult).map(
            lambda s: s.make.groups, key="label"
        )
        summarize = Node.scripted("summarize", fn="f", inputs=dict, outputs=MergedResult)
        pipeline = Construct("good-dict", nodes=[make, verify, summarize])
        assert len(pipeline.nodes) == 3

    def test_each_downstream_accepted_when_typed_dict_input(self):
        """Consumer with input=dict[str, X] matching Each output passes."""
        make = _producer("make", Clusters)
        verify = _consumer("verify", ClusterGroup, MatchResult).map(
            lambda s: s.make.groups, key="label"
        )
        summarize = Node.scripted(
            "summarize", fn="f",
            inputs=dict[str, MatchResult], outputs=MergedResult,
        )
        pipeline = Construct("good-typed-dict", nodes=[make, verify, summarize])
        assert len(pipeline.nodes) == 3

    def test_each_downstream_rejected_when_wrong_element_type(self):
        """Consumer with input=dict[str, WrongType] after Each is rejected."""
        make = _producer("make", Clusters)
        verify = _consumer("verify", ClusterGroup, MatchResult).map(
            lambda s: s.make.groups, key="label"
        )
        summarize = Node.scripted(
            "summarize", fn="f",
            inputs=dict[str, ValidationResult], outputs=MergedResult,
        )
        with pytest.raises(ConstructError):
            Construct("bad-element", nodes=[make, verify, summarize])

    def test_each_hint_mentions_dict_when_raw_consumer(self):
        """Error for raw-type consumer after Each mentions 'via Each'
        and suggests using dict input."""
        make = _producer("make", Clusters)
        verify = _consumer("verify", ClusterGroup, MatchResult).map(
            lambda s: s.make.groups, key="label"
        )
        summarize = _consumer("summarize", MatchResult, MergedResult)
        with pytest.raises(ConstructError) as exc_info:
            Construct("bad-hint", nodes=[make, verify, summarize])
        msg = str(exc_info.value)
        assert "via Each" in msg
        assert "dict" in msg


# ═══════════════════════════════════════════════════════════════════════════
# Construct | Oracle error paths
# ═══════════════════════════════════════════════════════════════════════════

class TestConstructOracleErrors:
    """Error paths for Construct | Oracle."""

    def test_unregistered_merge_fn_raises_when_compiled(self):
        """Construct | Oracle with unregistered merge_fn raises at compile."""
        from tests.fakes import build_test_compile_kwargs, register_scripted

        register_scripted("gen_err", lambda input_data, config: Claims(items=[]))

        sub = Construct(
            "bad-oracle-sub",
            input=Claims,
            output=Claims,
            nodes=[Node.scripted("g", fn="gen_err", outputs=Claims)],
        ) | Oracle(n=2, merge_fn="nonexistent_merge_fn")

        parent = Construct("parent", nodes=[sub])

        with pytest.raises(ConfigurationError, match="not registered"):
            compile(parent, **build_test_compile_kwargs())


# ═══════════════════════════════════════════════════════════════════════════
# @node fan-in validation
# ═══════════════════════════════════════════════════════════════════════════

class TestModifiableMapErrors:
    """Error paths for Modifiable.map() — string/lambda introspection."""

    def test_map_resolves_when_lambda_path_valid(self):
        """Happy path: lambda with valid attribute chain produces Each modifier."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        mapped = n.map(lambda s: s.make.groups, key="label")
        each = mapped.get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "make.groups"
        assert each.key == "label"

    def test_map_resolves_when_string_path_given(self):
        """String path is used directly without introspection."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        mapped = n.map("make.groups", key="label")
        each = mapped.get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "make.groups"

    def test_map_raises_when_source_not_string_or_callable(self):
        """Non-string, non-callable source raises ConstructError."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        with pytest.raises(ConstructError, match="must be a string path or a lambda"):
            n.map(42, key="label")

    def test_map_raises_when_lambda_uses_indexing(self):
        """Lambda with subscript/indexing raises ConstructError (not a pure attr chain)."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        with pytest.raises(ConstructError, match="pure attribute-access chain"):
            n.map(lambda s: s.make.groups[0], key="label")

    def test_map_raises_when_lambda_accesses_underscore_attr(self):
        """Lambda accessing underscore-prefixed attribute raises ConstructError."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        with pytest.raises(ConstructError, match="pure attribute-access chain"):
            n.map(lambda s: s.make._private, key="label")

    def test_map_raises_when_lambda_returns_non_recorder(self):
        """Lambda that returns a non-recorder value raises ConstructError."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        with pytest.raises(ConstructError, match="must return an attribute-access chain"):
            n.map(lambda s: "literal_string", key="label")

    def test_map_raises_when_lambda_is_identity(self):
        """Lambda that returns the recorder without any attribute access raises ConstructError."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        with pytest.raises(ConstructError, match="must access at least one attribute"):
            n.map(lambda s: s, key="label")

    def test_map_raises_when_called_twice(self):
        """Calling .map() twice raises ConstructError — duplicate Each is invalid."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        mapped_once = n.map("a.groups", key="label")
        with pytest.raises(ConstructError, match="Duplicate Each"):
            mapped_once.map("b.items", key="id")


# ═══════════════════════════════════════════════════════════════════════════
# _check_each_path edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckEachPathErrors:
    """Edge cases for _check_each_path beyond the standard 3 error paths."""

    def test_single_segment_path_defers_when_no_dot(self):
        """Each(over="a") with no dot — root matches upstream but no field to walk.
        split_each_path returns root='a', segments=(). The path resolves to the
        raw upstream type, which must be a list for validation to pass. Since
        Clusters is NOT a list, this should raise 'not a list'."""
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a", key="label"
        )
        with pytest.raises(ConstructError, match="not a list"):
            Construct("single-seg", nodes=[a, b])

    def test_single_segment_path_raises_when_root_unknown(self):
        """Each(over="unknown") with no matching upstream raises ConstructError."""
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="unknown", key="label"
        )
        with pytest.raises(ConstructError, match="root 'unknown' does not match"):
            Construct("single-seg-reject", nodes=[a, b])

    def test_empty_path_string_rejected_at_construction(self):
        """Each(over='') — rejected by field_validator at construction time."""
        with pytest.raises((ValueError, Exception), match="must not be empty"):
            Each(over="", key="label")

    def test_deeply_nested_path_resolves_when_fields_exist(self):
        """Multi-level dotted path that walks through nested models."""

        class Inner(BaseModel, frozen=True):
            claim_ids: list[str]

        class Middle(BaseModel, frozen=True):
            inner: Inner

        class Outer(BaseModel, frozen=True):
            middle: Middle

        a = _producer("a", Outer)
        # Path: a.middle.inner.claim_ids → list[str], element str
        b = Node.scripted(
            "b", fn="f", inputs=str, outputs=MatchResult,
        ) | Each(over="a.middle.inner.claim_ids", key="id")
        pipeline = Construct("deep-path", nodes=[a, b])
        assert len(pipeline.nodes) == 2

    def test_deeply_nested_path_raises_when_intermediate_missing(self):
        """Multi-level path where an intermediate segment doesn't exist."""

        class Shallow(BaseModel, frozen=True):
            name: str

        a = _producer("a", Shallow)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.name.nonexistent.deep", key="label"
        )
        with pytest.raises(ConstructError, match="has no field 'nonexistent'"):
            Construct("deep-missing", nodes=[a, b])

    def test_path_raises_when_terminal_is_non_list_primitive(self):
        """Path resolving to a primitive (int) raises 'not a list'."""

        class WithInt(BaseModel, frozen=True):
            count: int

        a = _producer("a", WithInt)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.count", key="label"
        )
        with pytest.raises(ConstructError, match="not a list"):
            Construct("prim-terminal", nodes=[a, b])

    def test_each_key_raises_when_field_missing_on_item_type(self):
        """Each.key must name a valid field on the list element type.
        each.key='nonexistent' on list[ClusterGroup] should raise (neograph-mn41)."""
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.groups", key="nonexistent"
        )
        with pytest.raises(ConstructError, match="has no field 'nonexistent'"):
            Construct("bad-each-key", nodes=[a, b])

    def test_each_key_passes_when_field_exists_on_item_type(self):
        """Each.key='label' on list[ClusterGroup] (which has a 'label' field)
        should assemble without error."""
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.groups", key="label"
        )
        pipeline = Construct("ok-each-key", nodes=[a, b])
        assert len(pipeline.nodes) == 2

    def test_each_key_skipped_when_element_type_is_primitive(self):
        """Each.key on list[str] (no model_fields) defers to runtime."""

        class HasStrings(BaseModel, frozen=True):
            tags: list[str]

        a = _producer("a", HasStrings)
        b = Node.scripted(
            "b", fn="f", inputs=str, outputs=MatchResult,
        ) | Each(over="a.tags", key="value")
        # str has no model_fields — should defer to runtime, not raise.
        pipeline = Construct("prim-key", nodes=[a, b])
        assert len(pipeline.nodes) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Node name collision detection (neograph-x820)
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeNameCollision:
    """Nodes whose names differ only by hyphens vs underscores must be
    rejected at compile time — they map to the same state field and would
    silently share loop counters, reducers, etc."""

    def test_collision_raises_when_hyphen_and_underscore_names_collide(self):
        """'my-node' and 'my_node' both map to field 'my_node' — must raise."""
        a = _producer("my-node", RawText)
        b = _producer("my_node", Claims)
        with pytest.raises(CompileError, match="name collision"):
            compile(Construct("collision", nodes=[a, b]), **build_test_compile_kwargs())

    def test_no_collision_when_names_differ(self):
        """Two nodes with genuinely different names compile fine."""
        from tests.fakes import register_scripted
        register_scripted("f_node_a", lambda input_data, config: RawText(text="a"))
        register_scripted("f_node_b", lambda input_data, config: Claims(items=["b"]))

        a = Node.scripted("node-a", fn="f_node_a", outputs=RawText)
        b = Node.scripted("node-b", fn="f_node_b", inputs=RawText, outputs=Claims)
        result = run(compile(Construct("no-collision", nodes=[a, b]), **build_test_compile_kwargs()), input={})
        assert isinstance(result["node_b"], Claims)

    def test_sub_construct_names_do_not_collide_with_parent(self):
        """Sub-construct node names live in separate state scopes — no error
        even if a parent node and sub-construct-internal node share a name."""
        from tests.fakes import register_scripted

        register_scripted("inner_fn", lambda input_data, config: Claims(items=["ok"]))
        register_scripted("parent_fn", lambda input_data, config: RawText(text="raw"))

        inner_node = Node.scripted("my_node", fn="inner_fn", inputs=RawText, outputs=Claims)
        sub = Construct(
            "sub",
            input=RawText,
            output=Claims,
            nodes=[inner_node],
        )
        parent_node = Node.scripted("my-parent", fn="parent_fn", outputs=RawText)
        # 'my_node' inside sub and 'my-parent' in parent — different scopes, no collision
        parent = Construct("parent", nodes=[parent_node, sub])
        result = run(compile(parent, **build_test_compile_kwargs()), input={})
        assert isinstance(result["sub"], Claims)


# ═══════════════════════════════════════════════════════════════════════════
# Compile-time: tool factory registration check (neograph-9513)
# ═══════════════════════════════════════════════════════════════════════════

class TestToolFactoryRegistrationCheck:
    """compile() must verify that every tool referenced by agent/act nodes
    is registered in _tool_factory_registry."""

    def test_unregistered_tool_raises_at_compile_when_agent_mode(self):
        """Agent node with unregistered tool raises CompileError at compile()."""
        from tests.fakes import StructuredFake, configure_fake_llm
        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m()))
        n = Node(
            "research",
            mode="agent",
            inputs=RawText,
            outputs=Claims,
            model="fast",
            prompt="test",
            tools=[Tool("nonexistent_tool_9513", budget=3)],
        )
        pipeline = Construct("bad-tool", nodes=[n])
        with pytest.raises(CompileError, match="nonexistent_tool_9513"):
            compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)

    def test_unregistered_tool_raises_at_compile_when_act_mode(self):
        """Act node with unregistered tool raises CompileError at compile()."""
        from tests.fakes import StructuredFake, configure_fake_llm
        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m()))
        n = Node(
            "actor",
            mode="act",
            inputs=RawText,
            outputs=Claims,
            model="fast",
            prompt="test",
            tools=[Tool("missing_tool_9513", budget=1)],
        )
        pipeline = Construct("bad-act-tool", nodes=[n])
        with pytest.raises(CompileError, match="missing_tool_9513"):
            compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)

    def test_registered_tool_passes_compile_when_agent_mode(self):
        """Agent node with registered tool compiles without error."""
        from tests.fakes import StructuredFake, configure_fake_llm, register_tool_factory

        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m()))
        register_tool_factory("registered_tool_9513", lambda config, tool_config: None)

        n = Node(
            "research-ok",
            mode="agent",
            inputs=RawText,
            outputs=Claims,
            model="fast",
            prompt="test",
            tools=[Tool("registered_tool_9513", budget=3)],
        )
        pipeline = Construct("good-tool", nodes=[n])
        compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)  # no raise = tool factory check passed


# ═══════════════════════════════════════════════════════════════════════════
# Compile-time: LLM + prompt compiler configured (neograph-fn5x)
# ═══════════════════════════════════════════════════════════════════════════

class TestLlmConfiguredCheck:
    """compile() must verify _llm_factory and _prompt_compiler are set
    when any node has mode in (think, agent, act)."""

    def test_unconfigured_llm_raises_at_compile_when_think_node(self):
        """Think node compiled without llm_factory= kwarg raises CompileError."""
        n = Node(
            "think-node",
            mode="think",
            inputs=RawText,
            outputs=Claims,
            model="fast",
            prompt="test",
        )
        pipeline = Construct("bad-llm", nodes=[n])
        with pytest.raises(CompileError, match="llm_factory"):
            compile(pipeline, **build_test_compile_kwargs())

    def test_unconfigured_prompt_compiler_raises_at_compile(self):
        """llm_factory= passed but prompt_compiler= missing raises CompileError."""
        n = Node(
            "think-node-pc",
            mode="think",
            inputs=RawText,
            outputs=Claims,
            model="fast",
            prompt="test",
        )
        pipeline = Construct("bad-pc", nodes=[n])
        with pytest.raises(CompileError, match="prompt_compiler"):
            compile(
                pipeline,
                llm_factory=lambda tier: None,
                **build_test_compile_kwargs(),
            )

    def test_scripted_only_compiles_without_llm_configured(self):
        """Pipeline with only scripted nodes compiles even without LLM kwargs."""
        from tests.fakes import register_scripted
        register_scripted("fn_no_llm_test", lambda input_data, config: RawText(text="ok"))
        n = Node.scripted("scripted-only", fn="fn_no_llm_test", outputs=RawText)
        pipeline = Construct("scripted-ok", nodes=[n])
        compile(pipeline, **build_test_compile_kwargs())  # no raise = scripted-only pipeline doesn't need LLM


# ═══════════════════════════════════════════════════════════════════════════
# Compile-time: output_strategy validation (neograph-0b2m)
# ═══════════════════════════════════════════════════════════════════════════

class TestOutputStrategyValidation:
    """output_strategy must be one of the allowed literals.

    Post-pej0 the LlmConfig Literal field enforces this at Node construction
    -- strictly earlier than the previous compile-time check.
    """

    def test_invalid_output_strategy_raises_at_node_construction(self):
        """Node with bogus output_strategy raises ValidationError at construction."""
        from pydantic import ValidationError

        from tests.fakes import StructuredFake, configure_fake_llm
        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m()))
        with pytest.raises(ValidationError, match="output_strategy"):
            Node(
                "bad-strat",
                mode="think",
                inputs=RawText,
                outputs=Claims,
                model="fast",
                prompt="test",
                llm_config={"output_strategy": "banana"},
            )

    def test_valid_output_strategies_pass_compile(self, **__llm_kw):
        """Nodes with valid output_strategy values compile without error."""
        from tests.fakes import StructuredFake, configure_fake_llm
        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m()))
        for strategy in ("structured", "json_mode", "text"):
            n = Node(
                f"strat-{strategy}",
                mode="think",
                inputs=RawText,
                outputs=Claims,
                model="fast",
                prompt="test",
                llm_config={"output_strategy": strategy},
            )
            pipeline = Construct(f"strat-{strategy}-pipe", nodes=[n])
            compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)  # no raise = strategy accepted

    def test_no_output_strategy_defaults_without_error(self):
        """Node with no output_strategy (default) compiles fine."""
        from tests.fakes import StructuredFake, configure_fake_llm
        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m()))
        n = Node(
            "no-strat",
            mode="think",
            inputs=RawText,
            outputs=Claims,
            model="fast",
            prompt="test",
        )
        pipeline = Construct("no-strat-pipe", nodes=[n])
        compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)  # no raise = default strategy accepted


# ═══════════════════════════════════════════════════════════════════════════
# Sub-construct output boundary contract (neograph-c4se)
# ═══════════════════════════════════════════════════════════════════════════

class TestFromInputRequired:
    """FromInput(required=True) raises ExecutionError at runtime when missing."""

    def test_required_from_input_raises_when_missing(self):
        """Runtime: required=True param not in config raises ExecutionError."""
        @node(outputs=RawText)
        def my_node(
            topic: Annotated[str, FromInput(required=True)],
        ) -> RawText:
            return RawText(text=topic)

        pipeline = construct_from_functions("req", [my_node])
        graph = compile(pipeline, **build_test_compile_kwargs())
        with pytest.raises(ExecutionError, match="topic"):
            run(graph, input={})

    def test_required_from_input_works_when_present(self):
        """Runtime: required=True param that IS in config works normally."""
        @node(outputs=RawText)
        def my_node(
            topic: Annotated[str, FromInput(required=True)],
        ) -> RawText:
            return RawText(text=topic)

        pipeline = construct_from_functions("req-ok", [my_node])
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"topic": "hello"})
        assert result["my_node"].text == "hello"

    def test_required_from_config_raises_when_missing(self):
        """Runtime: required=True FromConfig param not in config raises."""
        @node(outputs=RawText)
        def my_node(
            key: Annotated[str, FromConfig(required=True)],
        ) -> RawText:
            return RawText(text=key)

        pipeline = construct_from_functions("req-cfg", [my_node])
        graph = compile(pipeline, **build_test_compile_kwargs())
        with pytest.raises(ExecutionError, match="key"):
            run(graph, input={})


# ═══════════════════════════════════════════════════════════════════════════
# NeographError.build() error builder pattern
# ═══════════════════════════════════════════════════════════════════════════

class TestErrorBuilder:
    """NeographError.build() classmethod produces consistently structured
    error messages with what/expected/found/hint/location/node/construct."""

    def test_build_minimal_message_when_only_what(self):
        """build() with just `what` produces a plain message."""
        from neograph.errors import NeographError
        err = NeographError.build("something broke")
        assert isinstance(err, NeographError)
        assert str(err) == "something broke"

    def test_build_full_message_when_all_fields(self):
        """build() with all fields produces the structured format."""
        from neograph.errors import NeographError
        err = NeographError.build(
            "type mismatch",
            expected="Claims",
            found="RawText",
            hint="check your upstream",
            location="test.py:42",
            node="verify",
            construct="pipeline",
        )
        msg = str(err)
        assert msg.startswith("[Node 'verify' in construct 'pipeline']")
        assert "type mismatch" in msg
        assert "\n  expected: Claims" in msg
        assert "\n  found: RawText" in msg
        assert "\n  hint: check your upstream" in msg
        assert "\n  at test.py:42" in msg

    def test_build_node_only_prefix_when_no_construct(self):
        """build() with node= but no construct= uses [Node 'X'] prefix."""
        from neograph.errors import NeographError
        err = NeographError.build("failed", node="verify")
        assert str(err).startswith("[Node 'verify'] failed")

    def test_build_construct_only_prefix_when_no_node(self):
        """build() with construct= but no node= uses [Construct 'X'] prefix."""
        from neograph.errors import NeographError
        err = NeographError.build("failed", construct="pipeline")
        assert str(err).startswith("[Construct 'pipeline'] failed")

    def test_build_returns_subclass_when_called_on_subclass(self):
        """ConstructError.build() returns a ConstructError, not NeographError."""
        err = ConstructError.build("type mismatch", node="x")
        assert isinstance(err, ConstructError)
        assert isinstance(err, ValueError)  # dual inheritance preserved

    def test_build_returns_compile_error_when_called_on_compile_error(self):
        """CompileError.build() returns a CompileError."""
        from neograph.errors import CompileError
        err = CompileError.build("missing checkpointer")
        assert isinstance(err, CompileError)

    def test_build_returns_configuration_error_when_called_on_config_error(self):
        """ConfigurationError.build() returns a ConfigurationError."""
        err = ConfigurationError.build(
            "function not registered",
            hint="use register_scripted()",
        )
        assert isinstance(err, ConfigurationError)
        assert "function not registered" in str(err)
        assert "register_scripted()" in str(err)

    def test_execution_error_build_passes_validation_errors(self):
        """ExecutionError.build() accepts validation_errors kwarg."""
        err = ExecutionError.build(
            "DI resolution failed",
            node="my_node",
            found="field X missing from config",
            validation_errors="field X missing",
        )
        assert isinstance(err, ExecutionError)
        assert err.validation_errors == "field X missing"
        assert "DI resolution failed" in str(err)

    def test_execution_error_build_without_validation_errors(self):
        """ExecutionError.build() without validation_errors defaults to None."""
        err = ExecutionError.build("runtime failure", node="n")
        assert isinstance(err, ExecutionError)
        assert err.validation_errors is None

    def test_build_omits_absent_fields_when_partial(self):
        """build() with only expected= and hint= omits found= and location=."""
        from neograph.errors import NeographError
        err = NeographError.build(
            "wrong type",
            expected="int",
            hint="check annotation",
        )
        msg = str(err)
        assert "\n  expected: int" in msg
        assert "\n  hint: check annotation" in msg
        assert "\n  found:" not in msg
        assert "\n  at " not in msg


class TestTypeSpecValidation:
    """Node.inputs/outputs TypeSpec validator must reject non-type garbage.

    BUG neograph-m91y: _validate_type_spec is a no-op — accepts everything.
    """

    def test_string_rejected_as_inputs(self):
        """inputs='SomeType' (string, not a type) must raise."""
        with pytest.raises((TypeError, ValueError)):
            Node("bad", mode="scripted", inputs="SomeType", outputs=Claims)

    def test_int_rejected_as_outputs(self):
        """outputs=42 (int, not a type) must raise."""
        with pytest.raises((TypeError, ValueError)):
            Node("bad", mode="scripted", inputs=Claims, outputs=42)

    def test_list_of_strings_rejected(self):
        """inputs=['a', 'b'] must raise."""
        with pytest.raises((TypeError, ValueError)):
            Node("bad", mode="scripted", inputs=["a", "b"], outputs=Claims)

    def test_valid_types_accepted(self):
        """Smoke: valid type, dict, None, and generic alias all pass."""
        # These must NOT raise
        Node("ok1", mode="scripted", outputs=Claims)  # inputs=None default
        Node("ok2", mode="scripted", inputs=Claims, outputs=MatchResult)
        Node("ok3", mode="scripted", inputs={"a": Claims}, outputs=MatchResult)
        Node("ok4", mode="scripted", inputs=list[Claims], outputs=MatchResult)

    def test_generic_alias_accepted_as_inputs(self):
        """Generic aliases (list[X], dict[str,X], X|None) must pass validation.

        BUG neograph-vs6w: static annotation was type|dict|None which
        doesn't include generic aliases. PlainValidator is the real gate.
        """
        # These are NOT `type` instances — they're GenericAlias/UnionType
        Node("ga1", mode="scripted", inputs=list[Claims], outputs=Claims)
        Node("ga2", mode="scripted", inputs=dict[str, Claims], outputs=Claims)
        Node("ga3", mode="scripted", inputs=Claims | None, outputs=Claims)

    def test_dict_with_string_values_accepted(self):
        """Dict values can be strings (loader path before type resolution).

        BUG neograph-vs6w: string dict values pass validation but the
        static annotation says dict[str, type].
        """
        Node("sv1", mode="scripted", inputs={"a": "Claims"}, outputs=Claims)


class TestSingleTypeInputsDeprecation:
    """Single-type inputs= should emit DeprecationWarning at assembly time.

    TASK neograph-np0y: _extract_single_type does O(N) isinstance scan.
    Phase 1 adds a warning to signal migration to dict-form.
    """

    def test_single_type_inputs_warns_at_assembly(self):
        """Construct assembly with single-type inputs on non-first node warns."""
        a = _producer("a", RawText)
        b = _consumer("b", RawText, Claims)  # _consumer uses single-type inputs
        with pytest.warns(DeprecationWarning, match="single-type.*inputs"):
            Construct(name="test", nodes=[a, b])


# ═══════════════════════════════════════════════════════════════════════════
# Loop condition lint checks
# ═══════════════════════════════════════════════════════════════════════════


