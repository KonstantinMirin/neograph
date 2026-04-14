"""TopologySpec infrastructure: dataclasses, strategies, surface adapters.

TopologySpec is a surface-neutral pipeline description. Strategies produce
TopologySpecs; surface adapters build real Constructs from them.
"""

from __future__ import annotations

from collections.abc import Callable as CallableType
from dataclasses import dataclass
from dataclasses import field as dc_field

import hypothesis.strategies as st

from neograph import Construct, Node, compile
from neograph.factory import register_condition, register_scripted
from neograph.modifiers import Each, Loop, Oracle

from .conftest import (
    INTERMEDIATE_TYPES,
    TYPE_PAIRS,
    Alpha,
    Beta,
    FanCollection,
    FanItem,
    Gamma,
    _make_fn,
    _make_transform_fn,
    _uid,
)


@dataclass(frozen=True)
class NodeSpec:
    """Surface-neutral description of one pipeline node."""
    name: str
    output_type: type
    fn: CallableType                                   # scripted behavior
    input_type: type | dict[str, type] | None = None  # None = source node
    modifier: str | None = None                       # 'each', 'oracle', 'each_oracle', 'loop'
    each_over: str | None = None
    each_key: str | None = None
    oracle_n: int | None = None
    merge_fn: CallableType | None = None


@dataclass(frozen=True)
class TopologySpec:
    """Surface-neutral description of a complete pipeline."""
    name: str
    nodes: tuple[NodeSpec, ...]
    terminal_field: str
    terminal_type: type
    meta: dict = dc_field(default_factory=dict)


# -- Topology strategies (produce TopologySpecs, not Constructs) -----------

@st.composite
def bare_topology(draw):
    """Random bare chain: source -> 0-2 intermediates -> terminal."""
    tag = _uid()
    depth = draw(st.integers(min_value=0, max_value=2))
    type_idx = draw(st.integers(min_value=0, max_value=len(TYPE_PAIRS) - 1))
    src_type, terminal_type = TYPE_PAIRS[type_idx]

    chain_types = [src_type]
    for _i in range(depth):
        idx = draw(st.integers(min_value=0, max_value=len(INTERMEDIATE_TYPES) - 1))
        chain_types.append(INTERMEDIATE_TYPES[idx])
    chain_types.append(terminal_type)

    node_specs = []
    for i in range(len(chain_types) - 1):
        in_t, out_t = chain_types[i], chain_types[i + 1]
        name = f"n{i}-{tag}"
        if i == 0:
            node_specs.append(NodeSpec(
                name=name, output_type=out_t, fn=_make_fn(out_t),
            ))
        else:
            prev_field = node_specs[-1].name.replace("-", "_")
            use_dict = draw(st.booleans())
            if use_dict:
                node_specs.append(NodeSpec(
                    name=name, output_type=out_t,
                    input_type={prev_field: in_t},
                    fn=lambda _i, _c, _ot=out_t: _ot(),
                ))
            else:
                node_specs.append(NodeSpec(
                    name=name, output_type=out_t, input_type=in_t,
                    fn=_make_transform_fn(in_t, out_t),
                ))

    terminal_field = node_specs[-1].name.replace("-", "_")
    return TopologySpec(
        name=f"bare-{tag}", nodes=tuple(node_specs),
        terminal_field=terminal_field, terminal_type=terminal_type,
    )


@st.composite
def each_topology(draw):
    """Random Each fan-out: source(collection) -> process | Each."""
    tag = _uid()
    n_items = draw(st.integers(min_value=1, max_value=5))
    items = [FanItem(item_id=f"id-{i}", data=f"d{i}") for i in range(n_items)]

    src_name = f"esrc-{tag}"
    proc_name = f"eproc-{tag}"
    return TopologySpec(
        name=f"each-{tag}",
        nodes=(
            NodeSpec(name=src_name, output_type=FanCollection,
                     fn=lambda _i, _c, _it=items: FanCollection(items=_it)),
            NodeSpec(name=proc_name, output_type=Beta, input_type=FanItem,
                     fn=lambda _i, _c: Beta(score=0.5),
                     modifier="each",
                     each_over=f"{src_name.replace('-', '_')}.items",
                     each_key="item_id"),
        ),
        terminal_field=proc_name.replace("-", "_"),
        terminal_type=dict,
        meta={"expected_keys": {f"id-{i}" for i in range(n_items)}},
    )


@st.composite
def oracle_topology(draw):
    """Random Oracle ensemble with varying N."""
    tag = _uid()
    oracle_n = draw(st.integers(min_value=2, max_value=3))
    type_idx = draw(st.integers(min_value=0, max_value=len(TYPE_PAIRS) - 1))
    src_type, out_type = TYPE_PAIRS[type_idx]

    def merge(input_data, _c, _ot=out_type):
        return _ot()

    return TopologySpec(
        name=f"oracle-{tag}",
        nodes=(
            NodeSpec(name=f"osrc-{tag}", output_type=src_type,
                     fn=_make_fn(src_type)),
            NodeSpec(name=f"ogen-{tag}", output_type=out_type,
                     input_type=src_type, fn=_make_fn(out_type),
                     modifier="oracle", oracle_n=oracle_n, merge_fn=merge),
        ),
        terminal_field=f"ogen_{tag}".replace("-", "_"),
        terminal_type=out_type,
        meta={"oracle_n": oracle_n},
    )


@st.composite
def each_oracle_topology(draw):
    """Fused Each+Oracle with random N and collection size."""
    tag = _uid()
    oracle_n = draw(st.integers(min_value=2, max_value=3))
    n_items = draw(st.integers(min_value=1, max_value=4))
    items = [FanItem(item_id=f"k{i}") for i in range(n_items)]

    src_name = f"eosrc-{tag}"
    proc_name = f"eoproc-{tag}"

    def merge(input_data, _c):
        if isinstance(input_data, list):
            return Gamma(tags=[f"v{j}" for j in range(len(input_data))])
        return Gamma()

    return TopologySpec(
        name=f"eo-{tag}",
        nodes=(
            NodeSpec(name=src_name, output_type=FanCollection,
                     fn=lambda _i, _c, _it=items: FanCollection(items=_it)),
            NodeSpec(name=proc_name, output_type=Gamma, input_type=FanItem,
                     fn=_make_fn(Gamma),
                     modifier="each_oracle",
                     each_over=f"{src_name.replace('-', '_')}.items",
                     each_key="item_id",
                     oracle_n=oracle_n, merge_fn=merge),
        ),
        terminal_field=proc_name.replace("-", "_"),
        terminal_type=dict,
        meta={"expected_keys": {f"k{i}" for i in range(n_items)},
              "oracle_n": oracle_n},
    )


@st.composite
def fan_in_topology(draw):
    """Two source nodes feeding one downstream via dict-form inputs."""
    tag = _uid()
    type_idx = draw(st.integers(min_value=0, max_value=len(TYPE_PAIRS) - 1))
    src_a_type, src_b_type = TYPE_PAIRS[type_idx]
    terminal_type = Gamma

    src_a = f"fa-{tag}"
    src_b = f"fb-{tag}"
    consumer = f"fc-{tag}"
    src_a_field = src_a.replace("-", "_")
    src_b_field = src_b.replace("-", "_")

    def consumer_fn(input_data, _c):
        assert isinstance(input_data, dict), f"Fan-in expected dict, got {type(input_data)}"
        return Gamma(tags=[src_a_type.__name__, src_b_type.__name__])

    return TopologySpec(
        name=f"fanin-{tag}",
        nodes=(
            NodeSpec(name=src_a, output_type=src_a_type, fn=_make_fn(src_a_type)),
            NodeSpec(name=src_b, output_type=src_b_type, fn=_make_fn(src_b_type)),
            NodeSpec(name=consumer, output_type=terminal_type,
                     input_type={src_a_field: src_a_type, src_b_field: src_b_type},
                     fn=consumer_fn),
        ),
        terminal_field=consumer.replace("-", "_"),
        terminal_type=terminal_type,
        meta={"fan_in_keys": {src_a_field, src_b_field}},
    )


@st.composite
def sub_construct_topology(draw):
    """Source -> sub-construct(inner_node) -> result."""
    tag = _uid()
    type_idx = draw(st.integers(min_value=0, max_value=len(TYPE_PAIRS) - 1))
    src_type, inner_out_type = TYPE_PAIRS[type_idx]

    src_name = f"ssrc-{tag}"
    inner_name = f"sinner-{tag}"
    sub_name = f"sub-{tag}"

    return TopologySpec(
        name=f"subcon-{tag}",
        nodes=(
            NodeSpec(name=src_name, output_type=src_type, fn=_make_fn(src_type)),
            NodeSpec(name=inner_name, output_type=inner_out_type,
                     input_type=src_type,
                     fn=_make_transform_fn(src_type, inner_out_type)),
        ),
        terminal_field=sub_name.replace("-", "_"),
        terminal_type=inner_out_type,
        meta={"is_sub_construct": True, "sub_name": sub_name,
              "sub_input": src_type, "sub_output": inner_out_type,
              "inner_names": [inner_name]},
    )


@st.composite
def deep_chain_topology(draw):
    """Chain of 4-7 nodes. More hops = more chances for wiring bugs."""
    tag = _uid()
    depth = draw(st.integers(min_value=4, max_value=7))

    chain_types = []
    for _i in range(depth + 1):
        idx = draw(st.integers(min_value=0, max_value=len(INTERMEDIATE_TYPES) - 1))
        chain_types.append(INTERMEDIATE_TYPES[idx])

    node_specs = []
    for i in range(depth):
        in_t, out_t = chain_types[i], chain_types[i + 1]
        name = f"d{i}-{tag}"
        if i == 0:
            node_specs.append(NodeSpec(name=name, output_type=out_t, fn=_make_fn(out_t)))
        else:
            use_dict = draw(st.booleans())
            if use_dict:
                prev_field = node_specs[-1].name.replace("-", "_")
                node_specs.append(NodeSpec(
                    name=name, output_type=out_t,
                    input_type={prev_field: in_t},
                    fn=lambda _i, _c, _ot=out_t: _ot(),
                ))
            else:
                node_specs.append(NodeSpec(
                    name=name, output_type=out_t, input_type=in_t,
                    fn=_make_transform_fn(in_t, out_t),
                ))

    return TopologySpec(
        name=f"deep-{tag}", nodes=tuple(node_specs),
        terminal_field=node_specs[-1].name.replace("-", "_"),
        terminal_type=chain_types[-1],
        meta={"depth": depth},
    )


@st.composite
def loop_topology(draw):
    """Self-loop with random exit threshold, guaranteed to converge."""
    tag = _uid()
    step_size = draw(st.floats(min_value=0.2, max_value=0.5))
    max_iters = draw(st.integers(min_value=3, max_value=8))
    safe_max = (max_iters - 1) * step_size
    threshold = draw(st.floats(min_value=0.3, max_value=max(0.31, safe_max - 0.01)))

    src_name = f"lsrc-{tag}"
    body_name = f"lbody-{tag}"

    def loop_fn(input_data, _c, _ss=step_size):
        prev_score = 0.0
        prev_iter = 0
        if isinstance(input_data, Beta):
            prev_score = input_data.score
            prev_iter = input_data.iteration
        elif isinstance(input_data, list) and input_data:
            prev_score = input_data[-1].score
            prev_iter = input_data[-1].iteration
        return Beta(score=prev_score + _ss, iteration=prev_iter + 1)

    cond_name = f"lcond_{tag}"
    register_condition(cond_name, lambda val, _t=threshold: val is None or val.score < _t)

    return TopologySpec(
        name=f"loop-{tag}",
        nodes=(
            NodeSpec(name=src_name, output_type=Beta,
                     fn=lambda _i, _c: Beta(score=0.0)),
            NodeSpec(name=body_name, output_type=Beta, input_type=Beta,
                     fn=loop_fn,
                     modifier="loop"),
        ),
        terminal_field=body_name.replace("-", "_"),
        terminal_type=list,
        meta={"threshold": threshold, "step_size": step_size,
              "max_iters": max_iters, "loop_cond": cond_name},
    )


@st.composite
def loop_exhaustion_topology(draw):
    """Self-loop that CANNOT converge -- must raise ExecutionError."""
    tag = _uid()
    max_iters = draw(st.integers(min_value=2, max_value=5))

    src_name = f"lexsrc-{tag}"
    body_name = f"lexbody-{tag}"

    cond_name = f"lexcond_{tag}"
    register_condition(cond_name, lambda val: True)

    return TopologySpec(
        name=f"loopex-{tag}",
        nodes=(
            NodeSpec(name=src_name, output_type=Beta,
                     fn=lambda _i, _c: Beta(score=0.0)),
            NodeSpec(name=body_name, output_type=Beta, input_type=Beta,
                     fn=lambda _i, _c: Beta(score=0.0),
                     modifier="loop"),
        ),
        terminal_field=body_name.replace("-", "_"),
        terminal_type=list,
        meta={"max_iters": max_iters, "loop_cond": cond_name,
              "must_exhaust": True},
    )


@st.composite
def skip_when_topology(draw):
    """Pipeline where intermediate node has skip_when predicate."""
    tag = _uid()
    threshold = draw(st.floats(min_value=0.0, max_value=1.0))
    input_score = draw(st.floats(min_value=0.0, max_value=1.0))

    src_name = f"sksrc-{tag}"
    skip_name = f"sknode-{tag}"

    should_skip = input_score > threshold

    return TopologySpec(
        name=f"skip-{tag}",
        nodes=(
            NodeSpec(name=src_name, output_type=Beta,
                     fn=lambda _i, _c, _s=input_score: Beta(score=_s)),
            NodeSpec(name=skip_name, output_type=Alpha, input_type=Beta,
                     fn=lambda _i, _c: Alpha(value="executed")),
        ),
        terminal_field=skip_name.replace("-", "_"),
        terminal_type=Alpha,
        meta={"skip_threshold": threshold, "input_score": input_score,
              "should_skip": should_skip},
    )


# Topologies that always succeed and are stateless (safe for determinism, reuse)
any_topology_spec = st.one_of(
    bare_topology(),
    each_topology(),
    oracle_topology(),
    each_oracle_topology(),
    fan_in_topology(),
    sub_construct_topology(),
    deep_chain_topology(),
)


# -- Surface adapters (TopologySpec -> compiled graph) ---------------------

def _build_scripted_surface(spec: TopologySpec):
    """Build via Node.scripted + Construct (programmatic API)."""
    if spec.meta.get("is_sub_construct"):
        outer_nodes = []
        inner_nodes = []
        src_ns = spec.nodes[0]
        t = _uid()
        fn_name = f"scr_{src_ns.name}_{t}".replace("-", "_")
        register_scripted(fn_name, src_ns.fn)
        outer_nodes.append(Node.scripted(src_ns.name, fn=fn_name,
                                         outputs=src_ns.output_type))

        for ns in spec.nodes[1:]:
            t2 = _uid()
            fn_name2 = f"scr_{ns.name}_{t2}".replace("-", "_")
            register_scripted(fn_name2, ns.fn)
            inner_nodes.append(Node.scripted(ns.name, fn=fn_name2,
                                             inputs=ns.input_type,
                                             outputs=ns.output_type))

        sub = Construct(
            spec.meta["sub_name"], nodes=inner_nodes,
            input=spec.meta["sub_input"], output=spec.meta["sub_output"],
        )
        outer_nodes.append(sub)
        return compile(Construct(spec.name, nodes=outer_nodes))

    if spec.meta.get("skip_threshold") is not None:
        threshold = spec.meta["skip_threshold"]
        nodes = []
        for ns in spec.nodes:
            t = _uid()
            fn_name = f"scr_{ns.name}_{t}".replace("-", "_")
            register_scripted(fn_name, ns.fn)
            node = Node.scripted(ns.name, fn=fn_name,
                                 inputs=ns.input_type, outputs=ns.output_type)
            if ns.input_type is not None:
                def skip_pred(val, _t=threshold):
                    return isinstance(val, Beta) and val.score > _t

                def skip_val(_val):
                    return Alpha(value="skipped")

                node = node.model_copy(update={
                    "skip_when": skip_pred,
                    "skip_value": skip_val,
                })
            nodes.append(node)
        return compile(Construct(spec.name, nodes=nodes))

    nodes = []
    for ns in spec.nodes:
        t = _uid()
        fn_name = f"scr_{ns.name}_{t}".replace("-", "_")
        register_scripted(fn_name, ns.fn)
        node = Node.scripted(ns.name, fn=fn_name,
                             inputs=ns.input_type, outputs=ns.output_type)
        node = _apply_spec_modifiers(node, ns, f"scr_{t}", spec)
        nodes.append(node)
    return compile(Construct(spec.name, nodes=nodes))


def _apply_spec_modifiers(node, ns: NodeSpec, prefix: str, spec: TopologySpec | None = None):
    """Apply modifiers from a NodeSpec to a Node."""
    if ns.modifier == "each":
        node = node | Each(over=ns.each_over, key=ns.each_key)
    elif ns.modifier == "oracle":
        merge_name = f"{prefix}_m_{ns.name}".replace("-", "_")
        register_scripted(merge_name, ns.merge_fn)
        node = node | Oracle(n=ns.oracle_n, merge_fn=merge_name)
    elif ns.modifier == "each_oracle":
        merge_name = f"{prefix}_m_{ns.name}".replace("-", "_")
        register_scripted(merge_name, ns.merge_fn)
        node = node | Oracle(n=ns.oracle_n, merge_fn=merge_name)
        node = node | Each(over=ns.each_over, key=ns.each_key)
    elif ns.modifier == "loop":
        cond_name = spec.meta["loop_cond"] if spec else "default_cond"
        max_iters = spec.meta.get("max_iters", 10) if spec else 10
        node = node | Loop(when=cond_name, max_iterations=max_iters)
    elif ns.modifier is not None:
        raise ValueError(f"Unknown modifier '{ns.modifier}' on node '{ns.name}'")
    return node


def _register_type_safe(t: type):
    """Register a type for the loader, ignoring duplicates."""
    from neograph.spec_types import register_type
    try:
        register_type(t.__name__, t)
    except Exception:
        pass
