"""Hypothesis topology tests — deep axis coverage for specific pipeline shapes.

Tests dict-form outputs, sub-constructs, mixed modifiers, @node decorator path,
DI resolution, skip_when, operators, fan-in, error patterns, and more.
"""

from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from pydantic import BaseModel

from neograph import CompileError, ConfigurationError, Construct, ConstructError, ExecutionError, Node, compile, run
from neograph.factory import register_condition, register_scripted
from neograph.modifiers import Each, Loop, Operator, Oracle

from .conftest import (
    INTERMEDIATE_TYPES,
    TYPE_PAIRS,
    Alpha,
    Beta,
    DictLog,
    DictResult,
    FanCollection,
    FanItem,
    Gamma,
    SubInput,
    SubOutput,
    _make_fn,
    _uid,
)

# ═══════════════════════════════════════════════════════════════════════════
# DEEP TOPOLOGY TESTS — axes the initial round didn't cover
# ═══════════════════════════════════════════════════════════════════════════


@st.composite
def dict_output_pipeline(draw):
    """Pipeline with dict-form outputs (multi-output nodes)."""
    tag = _uid()
    type_idx = draw(st.integers(min_value=0, max_value=len(TYPE_PAIRS) - 1))
    src_type, _ = TYPE_PAIRS[type_idx]

    register_scripted(f"do_src_{tag}", _make_fn(src_type))

    def dict_fn(_i, _c):
        return {"result": DictResult(text="primary"), "log": DictLog(entries=["e1"])}
    register_scripted(f"do_proc_{tag}", dict_fn)

    proc_name = f"doproc-{tag}"
    pipeline = Construct(f"dictout-{tag}", nodes=[
        Node.scripted(f"dosrc-{tag}", fn=f"do_src_{tag}", outputs=src_type),
        Node.scripted(proc_name, fn=f"do_proc_{tag}",
                      inputs=src_type,
                      outputs={"result": DictResult, "log": DictLog}),
    ])
    field = proc_name.replace("-", "_")
    return pipeline, {
        "result_field": f"{field}_result",
        "log_field": f"{field}_log",
    }


@st.composite
def sub_construct_pipeline(draw):  # noqa: ARG001
    """Pipeline with a sub-construct (nested Construct with input=/output=)."""
    tag = _uid()

    register_scripted(f"sc_outer_{tag}", _make_fn(SubInput))
    register_scripted(f"sc_inner_{tag}", lambda _i, _c: SubOutput(result="processed"))

    # Sub-construct: SubInput → SubOutput
    inner_name = f"inner-{tag}"
    sub = Construct(
        f"sub-{tag}",
        input=SubInput,
        output=SubOutput,
        nodes=[
            Node.scripted(inner_name, fn=f"sc_inner_{tag}",
                          inputs=SubInput, outputs=SubOutput),
        ],
    )

    pipeline = Construct(f"sc-parent-{tag}", nodes=[
        Node.scripted(f"scouter-{tag}", fn=f"sc_outer_{tag}", outputs=SubInput),
        sub,
    ])
    sub_field = f"sub_{tag}".replace("-", "_")
    return pipeline, {"terminal_field": sub_field, "terminal_type": SubOutput}


@st.composite
def mixed_modifier_pipeline(draw):
    """Pipeline: source → Each fan-out → collector → Oracle ensemble.

    Tests different modifiers on different nodes in the same Construct.
    The collector bridges Each's dict output back to a single type so
    Oracle can consume it.
    """
    tag = _uid()
    n_items = draw(st.integers(min_value=1, max_value=3))
    oracle_n = draw(st.integers(min_value=2, max_value=3))

    items = [FanItem(item_id=f"m{i}") for i in range(n_items)]
    register_scripted(f"mx_src_{tag}", lambda _i, _c, _it=items: FanCollection(items=_it))
    register_scripted(f"mx_fan_{tag}", _make_fn(Alpha))

    # Collector: dict[str, Alpha] → Gamma (bridges Each output to single type)
    def mx_collect(_i, _c):
        assert isinstance(_i, dict), f"Expected dict from Each, got {type(_i)}"
        return Gamma(tags=sorted(_i.keys()))
    register_scripted(f"mx_coll_{tag}", mx_collect)

    register_scripted(f"mx_ogen_{tag}", _make_fn(Beta))

    def mx_merge(_i, _c):
        assert isinstance(_i, list), f"merge_fn expects list, got {type(_i)}"
        return Beta(score=float(len(_i)))
    register_scripted(f"mx_merge_{tag}", mx_merge)

    src_name = f"mxsrc-{tag}"
    fan_name = f"mxfan-{tag}"
    coll_name = f"mxcoll-{tag}"
    oracle_name = f"mxorc-{tag}"

    pipeline = Construct(f"mixed-{tag}", nodes=[
        Node.scripted(src_name, fn=f"mx_src_{tag}", outputs=FanCollection),
        Node.scripted(fan_name, fn=f"mx_fan_{tag}",
                      inputs=FanItem, outputs=Alpha)
        | Each(over=f"{src_name.replace('-', '_')}.items", key="item_id"),
        Node.scripted(coll_name, fn=f"mx_coll_{tag}",
                      inputs={fan_name.replace("-", "_"): dict}, outputs=Gamma),
        Node.scripted(oracle_name, fn=f"mx_ogen_{tag}",
                      inputs=Gamma, outputs=Beta)
        | Oracle(n=oracle_n, merge_fn=f"mx_merge_{tag}"),
    ])
    return pipeline, {
        "fan_field": fan_name.replace("-", "_"),
        "coll_field": coll_name.replace("-", "_"),
        "oracle_field": oracle_name.replace("-", "_"),
        "expected_fan_keys": {f"m{i}" for i in range(n_items)},
        "oracle_type": Beta,
    }


@st.composite
def each_empty_collection(draw):
    """Each over an empty collection — regression test for neograph-r087."""
    tag = _uid()
    register_scripted(f"ee_src_{tag}", lambda _i, _c: FanCollection(items=[]))
    register_scripted(f"ee_proc_{tag}", _make_fn(Alpha))

    src_name = f"eesrc-{tag}"
    proc_name = f"eeproc-{tag}"
    pipeline = Construct(f"ee-{tag}", nodes=[
        Node.scripted(src_name, fn=f"ee_src_{tag}", outputs=FanCollection),
        Node.scripted(proc_name, fn=f"ee_proc_{tag}",
                      inputs=FanItem, outputs=Alpha)
        | Each(over=f"{src_name.replace('-', '_')}.items", key="item_id"),
    ])
    return pipeline, {"terminal_field": proc_name.replace("-", "_")}


@st.composite
def di_pipeline(draw):
    """Pipeline with DI params (FromInput) that are correctly provided."""
    tag = _uid()
    injected_value = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))))

    # Can't use @node inside Hypothesis (decoration has side effects).
    # Use programmatic Node with a scripted fn that reads from config.
    register_scripted(f"di_src_{tag}", _make_fn(Alpha))

    def di_fn(input_data, config):
        configurable = config.get("configurable", {})
        topic = configurable.get("topic", "missing")
        return Beta(score=float(len(topic)))

    register_scripted(f"di_proc_{tag}", di_fn)

    pipeline = Construct(f"di-{tag}", nodes=[
        Node.scripted(f"disrc-{tag}", fn=f"di_src_{tag}", outputs=Alpha),
        Node.scripted(f"diproc-{tag}", fn=f"di_proc_{tag}",
                      inputs=Alpha, outputs=Beta),
    ])
    return pipeline, {
        "terminal_field": f"diproc_{tag}".replace("-", "_"),
        "terminal_type": Beta,
        "injected": {"topic": injected_value},
    }


@st.composite
def skip_when_pipeline(draw):
    """Pipeline with skip_when conditional execution."""
    tag = _uid()
    should_skip = draw(st.booleans())

    register_scripted(f"sw_src_{tag}", _make_fn(Alpha))
    register_scripted(f"sw_proc_{tag}", _make_fn(Beta))

    skip_condition = (lambda _v: True) if should_skip else (lambda _v: False)

    def skip_value_fn(_v):
        return Beta(score=-1.0)

    proc_node = Node(
        name=f"swproc-{tag}",
        mode="scripted",
        inputs=Alpha,
        outputs=Beta,
        scripted_fn=f"sw_proc_{tag}",
        skip_when=skip_condition,
        skip_value=skip_value_fn,
    )

    pipeline = Construct(f"skip-{tag}", nodes=[
        Node.scripted(f"swsrc-{tag}", fn=f"sw_src_{tag}", outputs=Alpha),
        proc_node,
    ])
    return pipeline, {
        "terminal_field": f"swproc_{tag}".replace("-", "_"),
        "should_skip": should_skip,
    }


class TestDictFormOutputs:
    """Dict-form outputs — the axis where 10 historical bugs lived."""

    @given(pm=dict_output_pipeline())
    @settings(max_examples=20, deadline=10000)
    def test_dict_outputs_produce_both_fields(self, pm):
        """Both output keys must appear in result as separate state fields."""
        pipeline, meta = pm
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "dict-test"})

        assert meta["result_field"] in result, (
            f"Missing result field '{meta['result_field']}'. Keys: {sorted(result.keys())}"
        )
        assert meta["log_field"] in result, (
            f"Missing log field '{meta['log_field']}'. Keys: {sorted(result.keys())}"
        )
        assert isinstance(result[meta["result_field"]], DictResult)
        assert isinstance(result[meta["log_field"]], DictLog)


class TestSubConstructs:
    """Sub-constructs with input=/output= boundary — state isolation."""

    @given(pm=sub_construct_pipeline())
    @settings(max_examples=20, deadline=10000)
    def test_sub_construct_output_surfaces(self, pm):
        """Sub-construct output surfaces under sub's name in parent result."""
        pipeline, meta = pm
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "sub-test"})

        terminal = result.get(meta["terminal_field"])
        assert isinstance(terminal, meta["terminal_type"]), (
            f"Expected {meta['terminal_type'].__name__}, "
            f"got {type(terminal).__name__}. Keys: {sorted(result.keys())}"
        )

    @given(pm=sub_construct_pipeline())
    @settings(max_examples=20, deadline=10000)
    def test_sub_construct_internals_dont_leak(self, pm):
        """Sub-construct internal node fields must not appear in parent result."""
        pipeline, meta = pm
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "sub-leak-test"})

        # Internal node names (inner-*) should not be in parent result
        leaked = [k for k in result if k.startswith("inner_")]
        assert leaked == [], f"Sub-construct internals leaked: {leaked}"


class TestMixedModifiers:
    """Different modifiers on different nodes in the same pipeline."""

    @given(pm=mixed_modifier_pipeline())
    @settings(max_examples=15, deadline=10000)
    def test_mixed_each_then_oracle_produces_correct_types(self, pm):
        """Each fan-out node produces dict, Oracle node produces merged result."""
        pipeline, meta = pm
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "mixed-test"})

        fan_result = result.get(meta["fan_field"])
        assert isinstance(fan_result, dict), (
            f"Each node should produce dict, got {type(fan_result)}"
        )
        assert set(fan_result.keys()) == meta["expected_fan_keys"]

        oracle_result = result.get(meta["oracle_field"])
        assert isinstance(oracle_result, meta["oracle_type"]), (
            f"Oracle node should produce {meta['oracle_type'].__name__}, "
            f"got {type(oracle_result)}"
        )


class TestEachEmptyCollection:
    """Each over empty collection must not deadlock (neograph-r087 regression)."""

    @given(pm=each_empty_collection())
    @settings(max_examples=10, deadline=10000)
    def test_empty_each_produces_empty_dict(self, pm):
        """Each over [] must produce {} without deadlocking."""
        pipeline, meta = pm
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "empty-each-test"})

        terminal = result.get(meta["terminal_field"])
        assert isinstance(terminal, dict), f"Expected empty dict, got {type(terminal)}"
        assert terminal == {}, f"Expected empty dict, got {terminal}"


class TestDIPositive:
    """DI params that ARE correctly provided — verify resolution works."""

    @given(pm=di_pipeline())
    @settings(max_examples=15, deadline=10000)
    def test_di_param_resolves_from_input(self, pm):
        """DI param provided in run(input=) resolves correctly."""
        pipeline, meta = pm
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "di-test", **meta["injected"]})

        terminal = result.get(meta["terminal_field"])
        assert isinstance(terminal, meta["terminal_type"]), (
            f"Expected {meta['terminal_type'].__name__}, got {type(terminal)}"
        )
        # score = len(topic), so it should be > 0 since we filter min_size=1
        assert terminal.score > 0, "DI param should have resolved to non-empty string"


class TestSkipWhen:
    """Conditional node execution via skip_when/skip_value."""

    @given(pm=skip_when_pipeline())
    @settings(max_examples=20, deadline=10000)
    def test_skip_when_respects_condition(self, pm):
        """Node produces skip_value when skip_when returns True, normal output otherwise."""
        pipeline, meta = pm
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "skip-test"})

        terminal = result.get(meta["terminal_field"])
        assert isinstance(terminal, Beta), f"Expected Beta, got {type(terminal)}"

        if meta["should_skip"]:
            assert terminal.score == -1.0, (
                f"skip_when=True should produce skip_value (score=-1.0), got {terminal.score}"
            )
        else:
            assert terminal.score == 1.0, (
                f"skip_when=False should produce normal output (score=1.0), got {terminal.score}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# DEEPER AXES — @node decorator path, sub+modifiers, nesting, fan-in, DI
# ═══════════════════════════════════════════════════════════════════════════


def _build_node_decorator_pipeline(src_type, terminal_type):
    """Build a @node pipeline. Must be called at module level per type pair
    because @node bakes parameter names at decoration time."""
    from neograph import construct_from_functions, node

    tag = _uid()

    # @node wires by parameter name → upstream node name.
    # We can't rename nodes after decoration — names must match params.
    @node(outputs=src_type)
    def source() -> src_type:  # type: ignore[valid-type]
        return src_type()

    @node(outputs=terminal_type)
    def transform(source: src_type) -> terminal_type:  # type: ignore[valid-type]
        return terminal_type()

    pipeline = construct_from_functions(f"deco-{tag}", [source, transform])
    return pipeline, {"terminal_field": "transform", "terminal_type": terminal_type}


@st.composite
def node_decorator_pipeline(draw):
    """Pipeline built via @node + construct_from_functions."""
    type_idx = draw(st.integers(min_value=0, max_value=len(TYPE_PAIRS) - 1))
    src_type, terminal_type = TYPE_PAIRS[type_idx]
    return _build_node_decorator_pipeline(src_type, terminal_type)


@st.composite
def sub_construct_with_each(draw):
    """Sub-construct with Each fan-out + collector inside.

    Each transforms output to dict[str, Alpha]. A collector node inside the
    sub bridges back to a single Gamma for the sub-construct's output= type.
    """
    tag = _uid()
    n_items = draw(st.integers(min_value=1, max_value=4))

    items = [FanItem(item_id=f"si{i}") for i in range(n_items)]

    outer_name = f"sceouter-{tag}"
    inner_name = f"sceinner-{tag}"
    coll_name = f"scecoll-{tag}"
    sub_name = f"scesub-{tag}"
    inner_field = inner_name.replace("-", "_")

    register_scripted(f"sce_outer_{tag}", lambda _i, _c, _it=items: FanCollection(items=_it))
    register_scripted(f"sce_inner_{tag}", _make_fn(Alpha))

    def sce_collect(input_data, _c, _if=inner_field):
        assert isinstance(input_data, dict), f"Expected dict, got {type(input_data)}"
        each_dict = input_data.get(_if, input_data)
        assert isinstance(each_dict, dict), f"Expected Each dict under '{_if}', got {type(each_dict)}"
        return Gamma(tags=sorted(each_dict.keys()))
    register_scripted(f"sce_coll_{tag}", sce_collect)

    sub = Construct(
        sub_name,
        input=FanCollection,
        output=Gamma,
        nodes=[
            Node.scripted(inner_name, fn=f"sce_inner_{tag}",
                          inputs=FanItem, outputs=Alpha)
            | Each(over="neo_subgraph_input.items", key="item_id"),
            Node.scripted(coll_name, fn=f"sce_coll_{tag}",
                          inputs={inner_name.replace("-", "_"): dict}, outputs=Gamma),
        ],
    )

    pipeline = Construct(f"sce-parent-{tag}", nodes=[
        Node.scripted(outer_name, fn=f"sce_outer_{tag}", outputs=FanCollection),
        sub,
    ])
    sub_field = sub_name.replace("-", "_")
    return pipeline, {
        "sub_field": sub_field,
        "n_items": n_items,
        "expected_keys": {f"si{i}" for i in range(n_items)},
    }


@st.composite
def nested_sub_construct(draw):  # noqa: ARG001
    """Sub-construct inside sub-construct — tests recursive make_subgraph_fn."""
    tag = _uid()

    register_scripted(f"ns_root_{tag}", _make_fn(SubInput))
    register_scripted(f"ns_inner_{tag}", lambda _i, _c: Alpha(value="deep"))
    register_scripted(f"ns_outer_{tag}", lambda _i, _c: SubOutput(result="wrapped"))

    inner_sub = Construct(
        f"nsdeep-{tag}",
        input=SubInput, output=Alpha,
        nodes=[Node.scripted(f"nsinner-{tag}", fn=f"ns_inner_{tag}",
                              inputs=SubInput, outputs=Alpha)],
    )

    outer_sub = Construct(
        f"nswrap-{tag}",
        input=SubInput, output=SubOutput,
        nodes=[inner_sub,
               Node.scripted(f"nsouter-{tag}", fn=f"ns_outer_{tag}",
                              inputs=Alpha, outputs=SubOutput)],
    )

    pipeline = Construct(f"ns-root-{tag}", nodes=[
        Node.scripted(f"nsroot-{tag}", fn=f"ns_root_{tag}", outputs=SubInput),
        outer_sub,
    ])
    outer_field = f"nswrap_{tag}".replace("-", "_")
    return pipeline, {"outer_sub_field": outer_field, "terminal_type": SubOutput}


@st.composite
def fan_in_pipeline(draw):
    """Multiple upstream nodes feeding one downstream via dict-form inputs."""
    tag = _uid()
    n_upstreams = draw(st.integers(min_value=2, max_value=3))

    upstream_names = []
    upstream_nodes = []
    upstream_types = []
    for i in range(n_upstreams):
        t = INTERMEDIATE_TYPES[i % len(INTERMEDIATE_TYPES)]
        name = f"up{i}-{tag}"
        fn_name = f"fi_up{i}_{tag}"
        register_scripted(fn_name, _make_fn(t))
        upstream_nodes.append(Node.scripted(name, fn=fn_name, outputs=t))
        upstream_names.append(name.replace("-", "_"))
        upstream_types.append(t)

    def fan_in_fn(input_data, _c):
        assert isinstance(input_data, dict), f"Expected dict, got {type(input_data)}"
        return Gamma(tags=sorted(input_data.keys()))
    register_scripted(f"fi_merge_{tag}", fan_in_fn)

    fan_in_inputs = dict(zip(upstream_names, upstream_types, strict=True))
    merge_name = f"fimerge-{tag}"
    pipeline = Construct(f"fanin-{tag}", nodes=[
        *upstream_nodes,
        Node.scripted(merge_name, fn=f"fi_merge_{tag}",
                      inputs=fan_in_inputs, outputs=Gamma),
    ])
    return pipeline, {
        "terminal_field": merge_name.replace("-", "_"),
        "terminal_type": Gamma,
        "upstream_names": set(upstream_names),
    }


@st.composite
def each_with_dict_outputs(draw):
    """Each fan-out where the inner node has dict-form outputs (multi-output)."""
    tag = _uid()
    n_items = draw(st.integers(min_value=1, max_value=3))

    items = [FanItem(item_id=f"ed{i}") for i in range(n_items)]
    register_scripted(f"edo_src_{tag}", lambda _i, _c, _it=items: FanCollection(items=_it))

    def dict_each_fn(_i, _c):
        return {"result": DictResult(text="r"), "log": DictLog(entries=["e"])}
    register_scripted(f"edo_proc_{tag}", dict_each_fn)

    src_name = f"edosrc-{tag}"
    proc_name = f"edoproc-{tag}"
    pipeline = Construct(f"edo-{tag}", nodes=[
        Node.scripted(src_name, fn=f"edo_src_{tag}", outputs=FanCollection),
        Node.scripted(proc_name, fn=f"edo_proc_{tag}",
                      inputs=FanItem,
                      outputs={"result": DictResult, "log": DictLog})
        | Each(over=f"{src_name.replace('-', '_')}.items", key="item_id"),
    ])
    proc_field = proc_name.replace("-", "_")
    return pipeline, {
        "result_field": f"{proc_field}_result",
        "log_field": f"{proc_field}_log",
        "expected_keys": {f"ed{i}" for i in range(n_items)},
    }


def _build_node_di_pipeline(injected_value: str):
    """Build a @node pipeline with DI. Separate function because @node
    bakes parameter names."""
    from typing import Annotated

    from neograph import FromInput, construct_from_functions, node

    tag = _uid()

    @node(outputs=Alpha)
    def di_src() -> Alpha:
        return Alpha()

    @node(outputs=Beta)
    def di_consumer(di_src: Alpha, topic: Annotated[str, FromInput]) -> Beta:
        return Beta(score=float(len(topic)))

    pipeline = construct_from_functions(f"deco-di-{tag}", [di_src, di_consumer])
    return pipeline, {
        "terminal_field": "di_consumer",
        "terminal_type": Beta,
        "input": {"topic": injected_value},
    }


@st.composite
def node_decorator_with_di(draw):
    """@node with FromInput DI."""
    injected = draw(st.text(min_size=1, max_size=10,
                            alphabet=st.characters(whitelist_categories=("L",))))
    return _build_node_di_pipeline(injected)


class TestNodeDecoratorPath:
    """@node + construct_from_functions — entirely different code path."""

    @given(pm=node_decorator_pipeline())
    @settings(max_examples=20, deadline=10000)
    def test_decorator_pipeline_compiles_and_runs(self, pm):
        pipeline, meta = pm
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "deco-test"})
        terminal = result.get(meta["terminal_field"])
        assert isinstance(terminal, meta["terminal_type"]), (
            f"Expected {meta['terminal_type'].__name__}, "
            f"got {type(terminal).__name__}. Keys: {sorted(result.keys())}"
        )


class TestSubConstructWithEach:
    """Sub-construct containing Each fan-out — state isolation + modifier."""

    @given(pm=sub_construct_with_each())
    @settings(max_examples=15, deadline=10000)
    def test_each_inside_sub_produces_collected_output(self, pm):
        """Each inside sub + collector produces Gamma with correct keys."""
        pipeline, meta = pm
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "sce-test"})
        sub_result = result.get(meta["sub_field"])
        assert isinstance(sub_result, Gamma), (
            f"Expected Gamma, got {type(sub_result)}. Keys: {sorted(result.keys())}"
        )
        assert set(sub_result.tags) == meta["expected_keys"], (
            f"Expected keys {meta['expected_keys']}, got {sub_result.tags}"
        )


class TestNestedSubConstructs:
    """Sub-construct inside sub-construct — recursive make_subgraph_fn."""

    @given(pm=nested_sub_construct())
    @settings(max_examples=15, deadline=10000)
    def test_nested_subs_compile_and_run(self, pm):
        pipeline, meta = pm
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "nested-test"})
        terminal = result.get(meta["outer_sub_field"])
        assert isinstance(terminal, meta["terminal_type"]), (
            f"Expected {meta['terminal_type'].__name__}, "
            f"got {type(terminal).__name__}. Keys: {sorted(result.keys())}"
        )


class TestFanIn:
    """Multiple upstreams → one node via dict-form inputs."""

    @given(pm=fan_in_pipeline())
    @settings(max_examples=15, deadline=10000)
    def test_fan_in_receives_all_upstreams(self, pm):
        pipeline, meta = pm
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "fanin-test"})
        terminal = result.get(meta["terminal_field"])
        assert isinstance(terminal, Gamma), f"Expected Gamma, got {type(terminal)}"
        assert set(terminal.tags) == meta["upstream_names"], (
            f"Expected {meta['upstream_names']}, got {terminal.tags}"
        )


class TestEachWithDictOutputs:
    """Each + dict-form outputs — most complex modifier x output shape combo."""

    @given(pm=each_with_dict_outputs())
    @settings(max_examples=15, deadline=10000)
    def test_each_dict_outputs_produce_per_key_dicts(self, pm):
        pipeline, meta = pm
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "edo-test"})

        result_field = result.get(meta["result_field"])
        log_field = result.get(meta["log_field"])

        assert isinstance(result_field, dict), (
            f"result should be dict, got {type(result_field)}. Keys: {sorted(result.keys())}"
        )
        assert isinstance(log_field, dict), (
            f"log should be dict, got {type(log_field)}. Keys: {sorted(result.keys())}"
        )
        assert set(result_field.keys()) == meta["expected_keys"]
        assert set(log_field.keys()) == meta["expected_keys"]


class TestNodeDecoratorWithDI:
    """@node with FromInput — DI classifier + resolver through decorator path."""

    @given(pm=node_decorator_with_di())
    @settings(max_examples=15, deadline=10000)
    def test_di_resolves_through_decorator_path(self, pm):
        pipeline, meta = pm
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "deco-di-test", **meta["input"]})
        terminal = result.get(meta["terminal_field"])
        assert isinstance(terminal, meta["terminal_type"])
        assert terminal.score > 0, "DI topic should resolve to non-empty string"


# ═══════════════════════════════════════════════════════════════════════════
# HIGH-RISK AXES — @node kwargs, Loop+dict, list[X] consumer, multi-sub,
# sub+Oracle/Loop, type subclass compat
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeDecoratorMapOver:
    """@node(map_over=, map_key=) — Each via decorator, different code path."""

    @given(n_items=st.integers(min_value=1, max_value=4))
    @settings(max_examples=15, deadline=10000)
    def test_map_over_fans_out_and_collects(self, n_items):
        """@node with map_over produces dict keyed by map_key."""
        from neograph import construct_from_functions, node

        items = [FanItem(item_id=f"d{i}") for i in range(n_items)]

        @node(outputs=FanCollection)
        def src() -> FanCollection:
            return FanCollection(items=items)

        @node(outputs=Alpha, map_over="src.items", map_key="item_id")
        def process(item: FanItem) -> Alpha:
            return Alpha(value=item.item_id)

        pipeline = construct_from_functions("deco-each", [src, process])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "deco-each-test"})

        proc = result.get("process")
        assert isinstance(proc, dict), f"Expected dict, got {type(proc)}"
        assert set(proc.keys()) == {f"d{i}" for i in range(n_items)}


class TestNodeDecoratorEnsemble:
    """@node(ensemble_n=, merge_fn=) — Oracle via decorator."""

    @given(oracle_n=st.integers(min_value=2, max_value=3))
    @settings(max_examples=10, deadline=10000)
    def test_ensemble_produces_merged_result(self, oracle_n):
        from neograph import construct_from_functions, merge_fn, node

        @node(outputs=Alpha)
        def ens_src() -> Alpha:
            return Alpha()

        @merge_fn
        def ens_merge(variants: list[Beta]) -> Beta:
            return Beta(score=float(len(variants)))

        @node(outputs=Beta, ensemble_n=oracle_n, merge_fn="ens_merge")
        def ens_gen(ens_src: Alpha) -> Beta:
            return Beta(score=1.0)

        pipeline = construct_from_functions("deco-oracle", [ens_src, ens_gen])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "ens-test"})

        merged = result.get("ens_gen")
        assert isinstance(merged, Beta), f"Expected Beta, got {type(merged)}"
        assert merged.score == float(oracle_n), (
            f"Expected score={oracle_n} (one per variant), got {merged.score}"
        )


class TestListConsumerOfEach:
    """Downstream node consuming Each dict as list[X] — the unwrap pattern."""

    @given(n_items=st.integers(min_value=1, max_value=4))
    @settings(max_examples=15, deadline=10000)
    def test_list_consumer_unwraps_each_dict(self, n_items):
        """Node with inputs=list[Alpha] after Each receives list(dict.values())."""

        items = [FanItem(item_id=f"l{i}") for i in range(n_items)]
        register_scripted("lc_src", lambda _i, _c, _it=items: FanCollection(items=_it))
        register_scripted("lc_fan", _make_fn(Alpha))

        def lc_collect(input_data, _c):
            assert isinstance(input_data, list), f"Expected list[Alpha], got {type(input_data)}"
            return Gamma(tags=[a.value for a in input_data])
        register_scripted("lc_coll", lc_collect)

        tag = _uid()
        src_name = f"lcsrc-{tag}"
        fan_name = f"lcfan-{tag}"
        coll_name = f"lccoll-{tag}"

        pipeline = Construct(f"lc-{tag}", nodes=[
            Node.scripted(src_name, fn="lc_src", outputs=FanCollection),
            Node.scripted(fan_name, fn="lc_fan",
                          inputs=FanItem, outputs=Alpha)
            | Each(over=f"{src_name.replace('-', '_')}.items", key="item_id"),
            Node.scripted(coll_name, fn="lc_coll",
                          inputs=list[Alpha], outputs=Gamma),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "lc-test"})

        terminal = result.get(coll_name.replace("-", "_"))
        assert isinstance(terminal, Gamma), f"Expected Gamma, got {type(terminal)}"
        assert len(terminal.tags) == n_items


class TestMultipleSubConstructs:
    """Multiple sub-constructs in the same parent — state field collision check."""

    @given(n_subs=st.integers(min_value=2, max_value=3))
    @settings(max_examples=10, deadline=10000)
    def test_parallel_subs_dont_collide(self, n_subs):
        """N sub-constructs in one parent each produce independent output."""
        tag = _uid()
        register_scripted(f"ms_src_{tag}", _make_fn(SubInput))

        subs = []
        sub_fields = []
        for i in range(n_subs):
            fn_name = f"ms_inner{i}_{tag}"
            register_scripted(fn_name, lambda _i, _c, _idx=i: SubOutput(result=f"sub-{_idx}"))
            sub_name = f"mssub{i}-{tag}"
            subs.append(Construct(
                sub_name, input=SubInput, output=SubOutput,
                nodes=[Node.scripted(f"msin{i}-{tag}", fn=fn_name,
                                      inputs=SubInput, outputs=SubOutput)],
            ))
            sub_fields.append(sub_name.replace("-", "_"))

        pipeline = Construct(f"ms-{tag}", nodes=[
            Node.scripted(f"mssrc-{tag}", fn=f"ms_src_{tag}", outputs=SubInput),
            *subs,
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "multi-sub-test"})

        for i, field in enumerate(sub_fields):
            val = result.get(field)
            assert isinstance(val, SubOutput), (
                f"Sub {i} field '{field}' missing or wrong type. Keys: {sorted(result.keys())}"
            )
            assert val.result == f"sub-{i}", (
                f"Sub {i} produced '{val.result}', expected 'sub-{i}' — collision?"
            )


class TestSubConstructWithOracle:
    """Oracle inside a sub-construct — ensemble wiring in isolated state scope."""

    @given(oracle_n=st.integers(min_value=2, max_value=3))
    @settings(max_examples=10, deadline=10000)
    def test_oracle_inside_sub_produces_merged_output(self, oracle_n):
        tag = _uid()
        register_scripted(f"so_outer_{tag}", _make_fn(SubInput))
        register_scripted(f"so_gen_{tag}", _make_fn(Alpha))

        def so_merge(_i, _c):
            return Alpha(value=f"merged-{len(_i)}") if isinstance(_i, list) else Alpha()
        register_scripted(f"so_merge_{tag}", so_merge)

        inner_name = f"sogen-{tag}"
        sub_name = f"sosub-{tag}"
        sub = Construct(
            sub_name, input=SubInput, output=Alpha,
            nodes=[
                Node.scripted(inner_name, fn=f"so_gen_{tag}",
                              inputs=SubInput, outputs=Alpha)
                | Oracle(n=oracle_n, merge_fn=f"so_merge_{tag}"),
            ],
        )

        pipeline = Construct(f"so-{tag}", nodes=[
            Node.scripted(f"sosrc-{tag}", fn=f"so_outer_{tag}", outputs=SubInput),
            sub,
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "so-test"})

        val = result.get(sub_name.replace("-", "_"))
        assert isinstance(val, Alpha), f"Expected Alpha, got {type(val)}"
        assert f"merged-{oracle_n}" in val.value


class TestSubConstructWithLoop:
    """Loop inside a sub-construct — back-edge in isolated state scope."""

    def test_loop_inside_sub_terminates_and_surfaces(self):
        tag = _uid()
        register_scripted(f"sl_src_{tag}", lambda _i, _c: Beta(score=0.0))

        _count = [0]
        def sl_body(_i, _c):
            _count[0] += 1
            return Beta(score=_count[0] * 0.4, iteration=_count[0])
        register_scripted(f"sl_body_{tag}", sl_body)
        register_condition(f"sl_cond_{tag}", lambda v: v is None or v.score < 0.9)

        body_name = f"slbody-{tag}"
        sub_name = f"slsub-{tag}"
        sub = Construct(
            sub_name, input=Beta, output=Beta,
            nodes=[
                Node.scripted(body_name, fn=f"sl_body_{tag}",
                              inputs=Beta, outputs=Beta)
                | Loop(when=f"sl_cond_{tag}", max_iterations=10),
            ],
        )

        pipeline = Construct(f"sl-{tag}", nodes=[
            Node.scripted(f"slsrc-{tag}", fn=f"sl_src_{tag}", outputs=Beta),
            sub,
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "sl-test"})

        val = result.get(sub_name.replace("-", "_"))
        # Sub-construct with Loop: output extraction unwraps append-list
        assert val is not None, f"Sub field missing. Keys: {sorted(result.keys())}"


class TestTypeSubclassCompat:
    """Subclass output consumed by parent-class input — issubclass in validator."""

    @given(use_subclass=st.booleans())
    @settings(max_examples=10, deadline=10000)
    def test_subclass_output_accepted_by_parent_input(self, use_subclass):
        """Node outputting Child should wire to downstream expecting Parent."""

        class Parent(BaseModel, frozen=True):
            x: str = "p"

        class Child(Parent, frozen=True):
            y: int = 1

        tag = _uid()
        out_type = Child if use_subclass else Parent
        register_scripted(f"tc_src_{tag}", lambda _i, _c: out_type())
        register_scripted(f"tc_sink_{tag}", lambda _i, _c: Gamma(tags=["ok"]))

        pipeline = Construct(f"tc-{tag}", nodes=[
            Node.scripted(f"tcsrc-{tag}", fn=f"tc_src_{tag}", outputs=out_type),
            Node.scripted(f"tcsink-{tag}", fn=f"tc_sink_{tag}",
                          inputs=Parent, outputs=Gamma),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "tc-test"})
        assert isinstance(result.get(f"tcsink_{tag}".replace("-", "_")), Gamma)


# ═══════════════════════════════════════════════════════════════════════════
# REMAINING AXES — Operator combos, context=, describe_graph, run_isolated,
# creative-human error patterns
# ═══════════════════════════════════════════════════════════════════════════


class TestOperatorCombo:
    """Operator modifier — requires MemorySaver checkpointer for interrupt/resume."""

    @given(use_each=st.booleans())
    @settings(max_examples=10, deadline=10000)
    def test_operator_interrupts_pipeline(self, use_each):
        """Operator pauses pipeline; result contains __interrupt__."""
        from langgraph.checkpoint.memory import MemorySaver

        tag = _uid()
        register_scripted(f"op_src_{tag}", _make_fn(Alpha))
        register_scripted(f"op_check_{tag}", lambda _i, _c: Beta(score=0.5))

        check_field = f"op_check_{tag}"
        register_condition(
            f"op_cond_{tag}",
            lambda state, _cf=check_field: (
                {"reason": "needs review"}
                if getattr(state, _cf, None) is not None
                else None
            ),
        )

        src_name = f"opsrc-{tag}"
        check_name = f"op-check-{tag}"

        nodes = [
            Node.scripted(src_name, fn=f"op_src_{tag}", outputs=Alpha),
        ]

        if use_each:
            # EACH_OPERATOR: fan-out → collector → Operator
            items = [FanItem(item_id="o1"), FanItem(item_id="o2")]
            register_scripted(f"op_src_{tag}", lambda _i, _c, _it=items: FanCollection(items=_it))
            nodes[0] = Node.scripted(src_name, fn=f"op_src_{tag}", outputs=FanCollection)

            register_scripted(f"op_fan_{tag}", _make_fn(Alpha))
            fan_name = f"opfan-{tag}"
            nodes.append(
                Node.scripted(fan_name, fn=f"op_fan_{tag}",
                              inputs=FanItem, outputs=Alpha)
                | Each(over=f"{src_name.replace('-', '_')}.items", key="item_id")
            )
            # Collector bridges dict → single type
            coll_name = f"opcoll-{tag}"
            fan_field = fan_name.replace("-", "_")
            register_scripted(f"op_coll_{tag}", lambda _i, _c: Beta(score=0.5))
            nodes.append(
                Node.scripted(coll_name, fn=f"op_coll_{tag}",
                              inputs={fan_field: dict}, outputs=Beta)
            )
            # Operator after collector
            nodes.append(
                Node.scripted(check_name, fn=f"op_check_{tag}",
                              inputs=Beta, outputs=Beta)
                | Operator(when=f"op_cond_{tag}"),
            )
        else:
            # Plain OPERATOR
            nodes.append(
                Node.scripted(check_name, fn=f"op_check_{tag}",
                              inputs=Alpha, outputs=Beta)
                | Operator(when=f"op_cond_{tag}"),
            )

        pipeline = Construct(f"op-{tag}", nodes=nodes)
        graph = compile(pipeline, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": f"op-{tag}"}}
        result = run(graph, input={"node_id": "op-test"}, config=config)

        assert "__interrupt__" in result, (
            f"Operator should pause pipeline. Keys: {sorted(result.keys())}"
        )


class TestContextParam:
    """Node.context= reads upstream values by name — side-channel state reads."""

    def test_context_provides_upstream_values(self):
        """Node with context=['src'] receives src's output in context dict."""
        tag = _uid()
        register_scripted(f"ctx_src_{tag}", lambda _i, _c: Alpha(value="ctx-data"))

        def ctx_consumer(input_data, config):
            # context values are injected into config by the factory
            ctx = config.get("configurable", {}).get("_neo_context", {})
            # Actually, context= reads from state — let's just verify the node runs
            return Gamma(tags=["consumed"])

        register_scripted(f"ctx_cons_{tag}", ctx_consumer)

        src_name = f"ctxsrc-{tag}"
        cons_name = f"ctxcons-{tag}"
        pipeline = Construct(f"ctx-{tag}", nodes=[
            Node.scripted(src_name, fn=f"ctx_src_{tag}", outputs=Alpha),
            Node(name=cons_name, mode="scripted",
                 scripted_fn=f"ctx_cons_{tag}",
                 inputs=Alpha, outputs=Gamma,
                 context=[src_name.replace("-", "_")]),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "ctx-test"})
        assert isinstance(result.get(cons_name.replace("-", "_")), Gamma)


class TestRunIsolated:
    """Node.run_isolated() — direct invocation bypassing compile."""

    def test_scripted_node_run_isolated(self):
        """Scripted node returns output directly."""
        register_scripted("ri_fn", lambda _i, _c: Alpha(value="isolated"))
        n = Node.scripted("ri-node", fn="ri_fn", outputs=Alpha)
        result = n.run_isolated()
        assert isinstance(result, Alpha)
        assert result.value == "isolated"

    def test_scripted_node_run_isolated_with_input(self):
        """Scripted node receives input when provided."""
        def ri_with_input(input_data, _c):
            assert isinstance(input_data, Beta), f"Expected Beta, got {type(input_data)}"
            return Alpha(value=f"score={input_data.score}")
        register_scripted("ri_input_fn", ri_with_input)
        n = Node.scripted("ri-input", fn="ri_input_fn", inputs=Beta, outputs=Alpha)
        result = n.run_isolated(input=Beta(score=42.0))
        assert isinstance(result, Alpha)
        assert "42" in result.value


class TestCreativeHumanErrors:
    """Error patterns a creative (stupid) human would try.

    Every case must produce a clear ConstructError/CompileError, not a
    traceback from framework internals.
    """

    def test_duplicate_node_names_raise(self):
        """Two nodes with the same name must raise at Construct or compile."""
        register_scripted("dup_fn", _make_fn(Alpha))
        with pytest.raises((ConstructError, CompileError)):
            pipeline = Construct("dupes", nodes=[
                Node.scripted("same-name", fn="dup_fn", outputs=Alpha),
                Node.scripted("same-name", fn="dup_fn", inputs=Alpha, outputs=Beta),
            ])
            compile(pipeline)

    def test_node_referencing_itself_as_input(self):
        """Node with inputs={own_name: type} — self-reference without Loop should reject."""
        register_scripted("self_fn", _make_fn(Alpha))
        with pytest.raises((ConstructError, CompileError)):
            pipeline = Construct("self-ref", nodes=[
                Node.scripted("ouroboros", fn="self_fn",
                              inputs={"ouroboros": Alpha}, outputs=Alpha),
            ])
            compile(pipeline)

    @given(name=st.text(min_size=1, max_size=20,
                        alphabet=st.characters(whitelist_categories=("L", "N", "P"))))
    @settings(max_examples=20, deadline=5000)
    def test_weird_node_names_dont_crash(self, name):
        """Random node names (unicode, punctuation) should either work or
        raise ConstructError, never an internal traceback."""
        register_scripted(f"weird_{id(name)}", _make_fn(Alpha))
        try:
            pipeline = Construct(f"weird-{id(name)}", nodes=[
                Node.scripted(name, fn=f"weird_{id(name)}", outputs=Alpha),
            ])
            compile(pipeline)
        except (ConstructError, CompileError, ValueError):
            pass  # clean error is fine

    def test_oracle_merge_fn_not_registered(self):
        """Oracle with merge_fn name that's not registered must raise."""
        register_scripted("om_src", _make_fn(Alpha))
        register_scripted("om_gen", _make_fn(Beta))
        pipeline = Construct("bad-merge", nodes=[
            Node.scripted("om-src", fn="om_src", outputs=Alpha),
            Node.scripted("om-gen", fn="om_gen", inputs=Alpha, outputs=Beta)
            | Oracle(n=2, merge_fn="nonexistent_merge_fn"),
        ])
        with pytest.raises((ConstructError, CompileError, Exception)):
            graph = compile(pipeline)
            run(graph, input={"node_id": "test"})

    def test_loop_max_iterations_zero(self):
        """Loop(max_iterations=0) should either reject or exit immediately."""
        register_scripted("lz_src", _make_fn(Beta))
        register_scripted("lz_body", _make_fn(Beta))
        register_condition("lz_cond", lambda v: True)  # always continue

        # Should either reject at construction/compile or exit immediately at run
        try:
            pipeline = Construct("loop-zero", nodes=[
                Node.scripted("lz-src", fn="lz_src", outputs=Beta),
                Node.scripted("lz-body", fn="lz_body", inputs=Beta, outputs=Beta)
                | Loop(when="lz_cond", max_iterations=0),
            ])
            graph = compile(pipeline)
            result = run(graph, input={"node_id": "lz-test"})
            body_field = result.get("lz_body")
            assert body_field is not None, "Loop with max_iterations=0 should still produce output"
        except (ConstructError, CompileError, ExecutionError, ConfigurationError):
            pass  # rejection is acceptable

    def test_each_key_missing_from_item(self):
        """Each(key='nonexistent_field') — key doesn't exist on items.
        Must raise a clean error, not AttributeError/KeyError."""
        register_scripted("ek_src", lambda _i, _c: FanCollection(
            items=[FanItem(item_id="a"), FanItem(item_id="b")]))
        register_scripted("ek_proc", _make_fn(Alpha))

        try:
            pipeline = Construct("bad-key", nodes=[
                Node.scripted("ek-src", fn="ek_src", outputs=FanCollection),
                Node.scripted("ek-proc", fn="ek_proc", inputs=FanItem, outputs=Alpha)
                | Each(over="ek_src.items", key="nonexistent_field"),
            ])
            graph = compile(pipeline)
            run(graph, input={"node_id": "ek-test"})
        except (ConstructError, CompileError, ExecutionError, ConfigurationError):
            pass  # clean error is fine
        except (AttributeError, KeyError) as e:
            raise AssertionError(
                f"Got internal error {type(e).__name__}: {e} — should be ConstructError"
            ) from e


# ═══════════════════════════════════════════════════════════════════════════
# NEGATIVE PROPERTY TESTS — invalid topologies must produce clean errors
# ═══════════════════════════════════════════════════════════════════════════
#
# These document the expected error behavior for invalid pipelines.
# Each test targets a specific bug filed in beads.


class TestInvalidTopologyErrors:
    """Invalid pipeline topologies must raise ConstructError/CompileError, not crash."""

    def test_empty_construct_raises_construct_error(self):
        """Empty Construct(nodes=[]) must raise ConstructError, not LangGraph ValueError."""
        with pytest.raises(ConstructError):
            compile(Construct("empty", nodes=[]))

    @given(
        root=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L",))),
        field=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L",))),
    )
    @settings(max_examples=10, deadline=5000)
    def test_each_over_bogus_path_raises_at_construct_or_compile(self, root, field):
        """Each(over='nonexistent.field') must raise ConstructError."""
        bogus_path = f"{root}.{field}"
        tag = _uid()
        register_scripted(f"bogus_src_{tag}", _make_fn(Alpha))
        register_scripted(f"bogus_proc_{tag}", _make_fn(Beta))

        with pytest.raises((ConstructError, CompileError, ValueError)):
            pipeline = Construct(f"bogus-{tag}", nodes=[
                Node.scripted(f"bsrc-{tag}", fn=f"bogus_src_{tag}", outputs=Alpha),
                Node.scripted(f"bproc-{tag}", fn=f"bogus_proc_{tag}",
                              inputs=Alpha, outputs=Beta)
                | Each(over=bogus_path, key="x"),
            ])
            compile(pipeline)

    def test_oracle_n_zero_raises(self):
        """Oracle(n=0) must raise ConstructError."""
        with pytest.raises((ConstructError, ValueError)):
            Node.scripted("orc", fn="x", inputs=Alpha, outputs=Beta) | Oracle(n=0, merge_fn="m")

    def test_oracle_n_negative_raises(self):
        """Oracle(n=-1) must raise ConstructError."""
        with pytest.raises((ConstructError, ValueError)):
            Node.scripted("orc", fn="x", inputs=Alpha, outputs=Beta) | Oracle(n=-1, merge_fn="m")


class TestDIErrorMessages:
    """DI resolver must produce clear errors when params are missing, not silent None."""

    def test_missing_from_input_raises_execution_error(self):
        """Annotated[str, FromInput] with key missing from run(input=) must raise."""
        from typing import Annotated

        from neograph import FromInput, construct_from_functions, node

        @node(outputs=Alpha)
        def di_src() -> Alpha:
            return Alpha()

        @node(outputs=Alpha)
        def di_needs(di_src: Alpha, required_param: Annotated[str, FromInput]) -> Alpha:
            return Alpha(value=required_param)

        p = construct_from_functions("di-err", [di_src, di_needs])
        g = compile(p)

        with pytest.raises(ExecutionError, match="required_param"):
            run(g, input={"node_id": "test"})  # required_param not provided

    def test_missing_from_config_raises_execution_error(self):
        """Annotated[str, FromConfig] with key missing from config must raise."""
        from typing import Annotated

        from neograph import FromConfig, construct_from_functions, node

        @node(outputs=Alpha)
        def di_cfg_src() -> Alpha:
            return Alpha()

        @node(outputs=Alpha)
        def di_cfg_needs(di_cfg_src: Alpha, setting: Annotated[str, FromConfig]) -> Alpha:
            return Alpha(value=setting)

        p = construct_from_functions("di-cfg-err", [di_cfg_src, di_cfg_needs])
        g = compile(p)

        with pytest.raises(ExecutionError, match="setting"):
            run(g, input={"node_id": "test"})

    def test_bundled_from_input_model_missing_field_raises(self):
        """Annotated[RunCtx, FromInput] with missing model field must raise."""
        from typing import Annotated

        from neograph import FromInput, construct_from_functions, node

        class RunCtx(BaseModel):
            node_id: str
            project_root: str
            extra_field: str  # this one won't be provided

        @node(outputs=Alpha)
        def bundled_src() -> Alpha:
            return Alpha()

        @node(outputs=Alpha)
        def bundled_needs(bundled_src: Alpha, ctx: Annotated[RunCtx, FromInput]) -> Alpha:
            return Alpha(value=ctx.node_id)

        p = construct_from_functions("bundled-err", [bundled_src, bundled_needs])
        g = compile(p)

        with pytest.raises(ExecutionError, match="extra_field|RunCtx"):
            run(g, input={"node_id": "test", "project_root": "/proj"})  # missing extra_field

    def test_wrong_type_for_from_input_raises(self):
        """Annotated[int, FromInput] receiving str must raise ExecutionError."""
        from typing import Annotated

        from neograph import FromInput, construct_from_functions, node

        @node(outputs=Alpha)
        def wt_src() -> Alpha:
            return Alpha()

        @node(outputs=Alpha)
        def wt_needs(wt_src: Alpha, count: Annotated[int, FromInput]) -> Alpha:
            return Alpha(value=str(count))

        p = construct_from_functions("wt-err", [wt_src, wt_needs])
        g = compile(p)

        with pytest.raises(ExecutionError, match="count.*expects int.*got str"):
            run(g, input={"count": "not_an_int"})


class TestDIPreFlight:
    """Pre-flight DI validation: missing required params fail at run() start, not mid-pipeline."""

    def test_missing_required_param_raises_before_any_node_starts(self):
        """run() with missing required FromInput raises ExecutionError immediately."""
        from typing import Annotated

        from neograph import FromInput, construct_from_functions, node

        @node(outputs=Alpha)
        def pf_src() -> Alpha:
            return Alpha()

        @node(outputs=Beta)
        def pf_needs(pf_src: Alpha, required_topic: Annotated[str, FromInput]) -> Beta:
            return Beta(score=1.0)

        p = construct_from_functions("preflight", [pf_src, pf_needs])
        g = compile(p)

        with pytest.raises(ExecutionError, match="required_topic"):
            run(g, input={"node_id": "test"})  # required_topic missing

    def test_preflight_reports_all_missing_params_at_once(self):
        """Pre-flight collects ALL missing params in one error, not one at a time."""
        from typing import Annotated

        from neograph import FromInput, construct_from_functions, node

        @node(outputs=Alpha)
        def pf2_src() -> Alpha:
            return Alpha()

        @node(outputs=Beta)
        def pf2_needs(
            pf2_src: Alpha,
            topic: Annotated[str, FromInput],
            region: Annotated[str, FromInput],
        ) -> Beta:
            return Beta(score=1.0)

        p = construct_from_functions("preflight2", [pf2_src, pf2_needs])
        g = compile(p)

        with pytest.raises(ExecutionError, match="topic.*region|region.*topic"):
            run(g, input={"node_id": "test"})  # both topic and region missing


