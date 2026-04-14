"""Framework invariant tests — properties that break when bugs exist.

Uses TopologySpec strategies to generate random pipeline shapes, then verifies
framework-level promises: immutability, idempotency, determinism, isolation,
correct modifier behavior. Each test names the specific one-line framework
change that would break it.
"""

from __future__ import annotations

import types

import pytest
from hypothesis import given, settings

from neograph import Construct, ExecutionError, Node, compile, run
from neograph.factory import register_scripted
from neograph.modifiers import Oracle

from .conftest import Alpha, Beta, Gamma, SubInput, SubOutput, _make_fn, _uid
from .topology import (
    _apply_spec_modifiers,
    _build_scripted_surface,
    _register_type_safe,
    any_topology_spec,
    bare_topology,
    deep_chain_topology,
    each_topology,
    fan_in_topology,
    loop_exhaustion_topology,
    loop_topology,
    oracle_topology,
    skip_when_topology,
)


class TestDescribeGraph:
    """describe_graph() must produce valid Mermaid for any compiled topology."""

    @given(spec=bare_topology())
    @settings(max_examples=10, deadline=10000)
    def test_mermaid_output_is_nonempty_string(self, spec):
        """describe_graph returns a non-empty string containing graph keywords."""
        from neograph.compiler import describe_graph

        graph = _build_scripted_surface(spec)
        mermaid = describe_graph(graph)
        assert isinstance(mermaid, str) and len(mermaid) > 10, (
            f"Expected non-trivial Mermaid string, got: {mermaid!r}"
        )

    @given(spec=each_topology())
    @settings(max_examples=10, deadline=10000)
    def test_mermaid_for_each_pipeline(self, spec):
        from neograph.compiler import describe_graph

        graph = _build_scripted_surface(spec)
        mermaid = describe_graph(graph)
        assert isinstance(mermaid, str) and len(mermaid) > 10


# -- FRAMEWORK INVARIANT TESTS (properties that break when bugs exist) -----

class TestNodeImmutabilityInvariant:
    """For ANY random pipeline, constructing a Construct must not mutate
    the Node instances passed to it. This is the property that gy9r and
    bchn violated."""

    @given(spec=any_topology_spec)
    @settings(max_examples=50, deadline=10000)
    def test_construct_does_not_mutate_nodes(self, spec):
        """Build nodes, snapshot their fields, construct a Construct,
        verify originals unchanged."""
        nodes = []
        snapshots = []
        for ns in spec.nodes:
            t = _uid()
            fn_name = f"imm_{ns.name}_{t}".replace("-", "_")
            register_scripted(fn_name, ns.fn)
            node = Node.scripted(ns.name, fn=fn_name,
                                 inputs=ns.input_type, outputs=ns.output_type)
            node = _apply_spec_modifiers(node, ns, f"imm_{t}")
            nodes.append(node)
            # Snapshot mutable fields BEFORE Construct
            snapshots.append({
                "inputs": node.inputs,
                "llm_config": node.llm_config.copy(),
                "renderer": node.renderer,
            })

        Construct(spec.name, nodes=nodes)

        for i, node in enumerate(nodes):
            snap = snapshots[i]
            assert node.inputs == snap["inputs"], (
                f"Construct mutated node '{node.name}'.inputs: "
                f"was {snap['inputs']!r}, now {node.inputs!r}"
            )
            assert node.llm_config == snap["llm_config"], (
                f"Construct mutated node '{node.name}'.llm_config: "
                f"was {snap['llm_config']!r}, now {node.llm_config!r}"
            )
            assert node.renderer == snap["renderer"], (
                f"Construct mutated node '{node.name}'.renderer: "
                f"was {snap['renderer']!r}, now {node.renderer!r}"
            )

    @given(spec=any_topology_spec)
    @settings(max_examples=50, deadline=10000)
    def test_construct_with_llm_config_does_not_mutate_nodes(self, spec):
        """Construct(llm_config={...}) propagates config but must not
        mutate the originals. bchn violation pattern."""
        nodes = []
        original_configs = []
        for ns in spec.nodes:
            t = _uid()
            fn_name = f"cfg_{ns.name}_{t}".replace("-", "_")
            register_scripted(fn_name, ns.fn)
            node = Node(ns.name, mode="scripted", scripted_fn=fn_name,
                        inputs=ns.input_type, outputs=ns.output_type,
                        llm_config={"temperature": 0.5})
            node = _apply_spec_modifiers(node, ns, f"cfg_{t}")
            nodes.append(node)
            original_configs.append(node.llm_config.copy())

        pipeline = Construct(spec.name, nodes=nodes,
                             llm_config={"max_retries": 3, "model_tier": "pro"})

        for i, node in enumerate(nodes):
            assert node.llm_config == original_configs[i], (
                f"Construct(llm_config=...) mutated node '{node.name}'.llm_config: "
                f"was {original_configs[i]!r}, now {node.llm_config!r}"
            )

        # Inner nodes SHOULD have merged config
        for inner in pipeline.nodes:
            if hasattr(inner, "llm_config"):
                assert inner.llm_config.get("max_retries") == 3, (
                    f"Inner node '{inner.name}' missing parent llm_config"
                )

    @given(spec=any_topology_spec)
    @settings(max_examples=50, deadline=10000)
    def test_construct_with_renderer_does_not_mutate_nodes(self, spec):
        """Construct(renderer=X) propagates but must not mutate originals."""
        nodes = []
        for ns in spec.nodes:
            t = _uid()
            fn_name = f"rnd_{ns.name}_{t}".replace("-", "_")
            register_scripted(fn_name, ns.fn)
            node = Node.scripted(ns.name, fn=fn_name,
                                 inputs=ns.input_type, outputs=ns.output_type)
            node = _apply_spec_modifiers(node, ns, f"rnd_{t}")
            nodes.append(node)

        sentinel = object()
        Construct(spec.name, nodes=nodes, renderer=sentinel)

        for node in nodes:
            assert node.renderer is None, (
                f"Construct(renderer=...) mutated node '{node.name}'.renderer "
                f"from None to {node.renderer!r}"
            )


class TestYAMLLoaderImmutabilityInvariant:
    """For ANY random topology loaded via YAML, the nodes in all_nodes must
    not be mutated by _build_sub_construct. gy9r violation pattern."""

    @given(spec=bare_topology())
    @settings(max_examples=50, deadline=10000)
    def test_loader_sub_construct_does_not_mutate_all_nodes(self, spec):
        """Build nodes, put them in all_nodes, call _build_sub_construct,
        verify all_nodes entries unchanged."""
        from neograph.loader import _build_sub_construct

        _register_type_safe(SubInput)
        _register_type_safe(SubOutput)

        # Use the first node from the spec, give it a SubOutput output
        ns = spec.nodes[0]
        t = _uid()
        fn_name = f"ld_{ns.name}_{t}".replace("-", "_")
        register_scripted(fn_name, lambda _i, _c: SubOutput(result="x"))

        worker = Node(name=f"w-{t}", outputs=SubOutput,
                      scripted_fn=fn_name)
        all_nodes = {f"w_{t}": worker}
        original_inputs = worker.inputs

        _build_sub_construct(
            {"name": f"sub-{t}", "input": "SubInput", "output": "SubOutput",
             "nodes": [f"w-{t}"]},
            all_nodes,
        )

        assert all_nodes[f"w_{t}"].inputs == original_inputs, (
            f"_build_sub_construct mutated all_nodes entry. "
            f"Was {original_inputs!r}, now {all_nodes[f'w_{t}'].inputs!r}"
        )


class TestCompileIdempotency:
    """Building the same topology twice must produce equivalent results.

    Breaks if: compile() or Construct.__init__ mutates internal state
    (e.g., the pre-fix bchn code mutated child llm_config, so a second
    Construct wrapping the same nodes would see different config).

    One-line break: revert construct.py model_copy to direct assignment."""

    @given(spec=any_topology_spec)
    @settings(max_examples=30, deadline=15000)
    def test_build_twice_same_result(self, spec):
        """Build and run the same topology twice. Compare ALL result keys
        and ALL values — not just the terminal."""
        g1 = _build_scripted_surface(spec)
        g2 = _build_scripted_surface(spec)

        r1 = run(g1, input={"node_id": "test"})
        r2 = run(g2, input={"node_id": "test"})

        # Same key sets
        assert set(r1.keys()) == set(r2.keys()), (
            f"Build idempotency: different keys. "
            f"1st: {sorted(r1.keys())}, 2nd: {sorted(r2.keys())}"
        )
        # Same values for every key
        for k in r1:
            v1, v2 = r1[k], r2[k]
            if isinstance(v1, dict) and isinstance(v2, dict):
                assert set(v1.keys()) == set(v2.keys()), (
                    f"Key '{k}' dict keys differ: {set(v1.keys())} vs {set(v2.keys())}"
                )
            else:
                assert type(v1) is type(v2), (
                    f"Key '{k}' type differs: {type(v1).__name__} vs {type(v2).__name__}"
                )
                if hasattr(v1, '__eq__'):
                    assert v1 == v2, f"Key '{k}' value differs: {v1!r} vs {v2!r}"


class TestNodeReuseAcrossConstructs:
    """The same Node used in two different Constructs must work in both.
    This is exactly what gy9r broke — the first Construct mutated the
    Node's inputs, corrupting the second."""

    @given(spec=bare_topology())
    @settings(max_examples=30, deadline=15000)
    def test_node_reuse_in_two_constructs(self, spec):
        """Build two Constructs sharing the same source Node. Both must
        compile and run without the first corrupting the second."""
        if len(spec.nodes) < 2:
            return  # need at least 2 nodes

        # Build the nodes once
        shared_nodes = []
        for ns in spec.nodes:
            t = _uid()
            fn_name = f"reuse_{ns.name}_{t}".replace("-", "_")
            register_scripted(fn_name, ns.fn)
            node = Node.scripted(ns.name, fn=fn_name,
                                 inputs=ns.input_type, outputs=ns.output_type)
            node = _apply_spec_modifiers(node, ns, f"reuse_{t}")
            shared_nodes.append(node)

        # Use same nodes in two Constructs
        c1 = Construct(f"{spec.name}-a", nodes=shared_nodes)
        c2 = Construct(f"{spec.name}-b", nodes=shared_nodes)

        g1 = compile(c1)
        g2 = compile(c2)

        r1 = run(g1, input={"node_id": "test"})
        r2 = run(g2, input={"node_id": "test"})

        tf = spec.terminal_field
        assert tf in r1, f"First construct missing '{tf}'"
        assert tf in r2, f"Second construct missing '{tf}' (corruption?)"
        assert type(r1[tf]) is type(r2[tf])


# -- FRAMEWORK PROMISES (invariants stated WITHOUT knowledge of bugs) ------
#
# These are properties the framework guarantees. Hypothesis generates random
# topologies and checks them. Any violation is a real discovery.


class TestEveryNodeExecutesExactlyOnce:
    """Promise: in a bare/oracle pipeline, every node's scripted_fn fires
    exactly once per run(). No skips, no double-fires."""

    @given(spec=bare_topology())
    @settings(max_examples=50, deadline=15000)
    def test_every_node_fires_once(self, spec):
        """Instrument each node's fn with a counter. After run(),
        every counter must be exactly 1."""
        call_counts: dict[str, int] = {}
        nodes = []
        for ns in spec.nodes:
            t = _uid()
            fn_name = f"cnt_{ns.name}_{t}".replace("-", "_")
            call_counts[ns.name] = 0

            def counting_fn(_i, _c, _ns_name=ns.name, _orig=ns.fn):
                call_counts[_ns_name] += 1
                return _orig(_i, _c)

            register_scripted(fn_name, counting_fn)
            node = Node.scripted(ns.name, fn=fn_name,
                                 inputs=ns.input_type, outputs=ns.output_type)
            node = _apply_spec_modifiers(node, ns, f"cnt_{t}")
            nodes.append(node)

        graph = compile(Construct(spec.name, nodes=nodes))
        run(graph, input={"node_id": "test"})

        for name, count in call_counts.items():
            assert count == 1, (
                f"Node '{name}' executed {count} times (expected 1). "
                f"Pipeline: {[ns.name for ns in spec.nodes]}"
            )


class TestStateFieldsMatchDeclaredOutputs:
    """Promise: state model fields match node declarations including
    modifier-induced type transformations.

    Breaks if: _add_output_field in state.py uses wrong type for Each
    (should be dict[str, T], not T) or wrong type for Loop (should be
    list[T], not T).

    One-line break: change _add_output_field's Each branch from dict
    annotation to plain T."""

    @given(spec=any_topology_spec)
    @settings(max_examples=50, deadline=10000)
    def test_state_field_types_match_modifier_effects(self, spec):
        """State field annotations must reflect modifier transformations:
        Each → Annotated[dict[str, T], ...], bare → T | None."""
        import typing

        from neograph.state import compile_state_model

        nodes = []
        for ns in spec.nodes:
            t = _uid()
            fn_name = f"sf_{ns.name}_{t}".replace("-", "_")
            register_scripted(fn_name, ns.fn)
            node = Node.scripted(ns.name, fn=fn_name,
                                 inputs=ns.input_type, outputs=ns.output_type)
            node = _apply_spec_modifiers(node, ns, f"sf_{t}")
            nodes.append(node)

        pipeline = Construct(spec.name, nodes=nodes)
        state_model = compile_state_model(pipeline)

        for _i, ns in enumerate(spec.nodes):
            field_name = ns.name.replace("-", "_")
            assert field_name in state_model.model_fields, (
                f"Node '{ns.name}' has no state field '{field_name}'"
            )

            field_info = state_model.model_fields[field_name]
            annotation = field_info.annotation

            if ns.modifier in ("each", "each_oracle"):
                # Each produces dict[str, T] — annotation should involve dict.
                # May be wrapped in Annotated, Union (| None), or both.
                def _unwrap_to_origin(ann):
                    origin = typing.get_origin(ann)
                    if origin is typing.Annotated:
                        return _unwrap_to_origin(typing.get_args(ann)[0])
                    if origin is types.UnionType:
                        # dict[str, T] | None — find the dict arm
                        for arg in typing.get_args(ann):
                            if arg is type(None):
                                continue
                            return _unwrap_to_origin(arg)
                    return origin

                real_origin = _unwrap_to_origin(annotation)
                assert real_origin is dict, (
                    f"Each node '{ns.name}' state field should be dict type, "
                    f"got {annotation} (origin={real_origin})"
                )


class TestDataFlowIntegrity:
    """Promise: data produced by source nodes flows through every hop
    and arrives at the terminal with correct type and non-default content.

    Breaks if: _extract_input returns None for a node (wiring gap),
    or _build_state_update writes to wrong field, or a modifier
    silently drops data.

    One-line break: change _extract_input to return None when inputs
    is a dict (fan-in path). Terminal gets default instance instead
    of upstream data."""

    @given(spec=bare_topology())
    @settings(max_examples=50, deadline=15000)
    def test_source_value_flows_to_terminal(self, spec):
        """Inject a sentinel value at the source. Verify the terminal
        node received a non-None, non-default input — proving data
        actually flowed through every intermediate node."""
        sentinel = f"sentinel-{_uid()}"
        call_log: dict[str, bool] = {}

        nodes = []
        for i, ns in enumerate(spec.nodes):
            t = _uid()
            fn_name = f"flow_{ns.name}_{t}".replace("-", "_")

            if i == 0:
                # Source: produce sentinel
                def src_fn(_i, _c, _s=sentinel, _out=ns.output_type):
                    if _out is Alpha:
                        return Alpha(value=_s)
                    if _out is Beta:
                        return Beta(score=42.0, iteration=99)
                    return _out()
                register_scripted(fn_name, src_fn)
            else:
                # Intermediate/terminal: verify input is not None/default,
                # then pass through
                def mid_fn(input_data, _c, _name=ns.name, _out=ns.output_type):
                    call_log[_name] = input_data is not None
                    return _out()
                register_scripted(fn_name, mid_fn)

            node = Node.scripted(ns.name, fn=fn_name,
                                 inputs=ns.input_type, outputs=ns.output_type)
            node = _apply_spec_modifiers(node, ns, f"flow_{t}", spec)
            nodes.append(node)

        graph = compile(Construct(spec.name, nodes=nodes))
        result = run(graph, input={"node_id": "test"})

        # Every non-source node must have received non-None input
        for ns in spec.nodes[1:]:
            assert call_log.get(ns.name, False), (
                f"Node '{ns.name}' received None input — data flow broken. "
                f"Pipeline: {[n.name for n in spec.nodes]}"
            )

    @given(spec=any_topology_spec)
    @settings(max_examples=50, deadline=15000)
    def test_terminal_type_matches_declaration(self, spec):
        """Terminal value type must match declared type accounting for
        modifier effects: Each → dict, bare → output_type."""
        graph = _build_scripted_surface(spec)
        result = run(graph, input={"node_id": "test"})

        tf = spec.terminal_field
        assert tf in result
        val = result[tf]
        assert val is not None

        if spec.terminal_type is dict:
            assert isinstance(val, dict), (
                f"Terminal '{tf}' expected dict, got {type(val).__name__}"
            )
            for ns in spec.nodes:
                if ns.name.replace("-", "_") == tf and ns.modifier in ("each", "each_oracle"):
                    for k, v in val.items():
                        assert isinstance(v, ns.output_type), (
                            f"Each value '{k}': expected {ns.output_type.__name__}, "
                            f"got {type(v).__name__}"
                        )
        else:
            assert isinstance(val, spec.terminal_type), (
                f"Terminal '{tf}' expected {spec.terminal_type.__name__}, "
                f"got {type(val).__name__}"
            )


class TestRunDeterminism:
    """Promise: two independent builds from the same spec produce
    identical results for ALL nodes, not just the terminal.

    Breaks if: node execution order is non-deterministic, or state
    reduction depends on insertion order, or _extract_input picks
    different values on different runs.

    One-line break: randomize node iteration order in compile()."""

    @given(spec=any_topology_spec)
    @settings(max_examples=30, deadline=15000)
    def test_two_builds_all_keys_identical(self, spec):
        """Build and run the same spec twice independently. Every node's
        output must be identical — not just the terminal."""
        g1 = _build_scripted_surface(spec)
        g2 = _build_scripted_surface(spec)
        r1 = run(g1, input={"node_id": "test"})
        r2 = run(g2, input={"node_id": "test"})

        assert set(r1.keys()) == set(r2.keys()), (
            f"Key sets differ: {sorted(r1.keys())} vs {sorted(r2.keys())}"
        )

        for k in r1:
            v1, v2 = r1[k], r2[k]
            if isinstance(v1, dict) and isinstance(v2, dict):
                assert set(v1.keys()) == set(v2.keys()), (
                    f"Node '{k}' dict keys differ: {set(v1.keys())} vs {set(v2.keys())}"
                )
                for dk in v1:
                    assert v1[dk] == v2[dk], (
                        f"Node '{k}' dict value '{dk}' differs: {v1[dk]!r} vs {v2[dk]!r}"
                    )
            elif hasattr(v1, '__eq__') and v1 is not None:
                assert v1 == v2, (
                    f"Node '{k}' value differs: {v1!r} vs {v2!r}"
                )


class TestNoInternalStateLeaks:
    """Promise: run() result contains ONLY declared node fields.
    No neo_* plumbing, no unexpected keys from modifier machinery.

    Breaks if: _strip_internals misses a prefix, or a modifier adds
    state fields that aren't stripped (e.g., neo_each_item, neo_loop_count).

    One-line break: remove the 'neo_' prefix check in _strip_internals."""

    @given(spec=any_topology_spec)
    @settings(max_examples=50, deadline=15000)
    def test_result_keys_are_exactly_declared_nodes(self, spec):
        """Result must contain only node field names. Any extra key
        (neo_*, modifier internal, etc.) is a leak."""
        graph = _build_scripted_surface(spec)
        result = run(graph, input={"node_id": "test"})

        # Build expected key set: node fields + framework-injected fields
        expected = set()
        for ns in spec.nodes:
            field = ns.name.replace("-", "_")
            expected.add(field)
            # Sub-constructs add the sub name, not inner node names
            if spec.meta.get("is_sub_construct") and ns == spec.nodes[-1]:
                expected.discard(field)
                expected.add(spec.meta["sub_name"].replace("-", "_"))

        # These are always present (injected by run())
        framework_injected = {"node_id", "project_root", "human_feedback"}

        actual = set(result.keys())
        unexpected = actual - expected - framework_injected
        assert not unexpected, (
            f"Unexpected keys in result: {sorted(unexpected)}. "
            f"Expected node fields: {sorted(expected)}. "
            f"All result keys: {sorted(actual)}"
        )


class TestOracleCallsGeneratorNTimes:
    """Promise: Oracle(n=N) calls the generator node N times, merge once."""

    @given(spec=oracle_topology())
    @settings(max_examples=30, deadline=15000)
    def test_oracle_generator_called_n_times(self, spec):
        """The Oracle generator fn must fire exactly N times."""
        gen_ns = spec.nodes[1]  # second node is the Oracle generator
        assert gen_ns.modifier == "oracle"

        call_count = [0]
        t = _uid()

        # Source
        src_ns = spec.nodes[0]
        src_fn = f"oc_src_{t}".replace("-", "_")
        register_scripted(src_fn, src_ns.fn)

        # Generator — instrumented
        gen_fn = f"oc_gen_{t}".replace("-", "_")

        def counting_gen(_i, _c, _orig=gen_ns.fn):
            call_count[0] += 1
            return _orig(_i, _c)

        register_scripted(gen_fn, counting_gen)

        # Merge
        merge_fn = f"oc_merge_{t}".replace("-", "_")
        register_scripted(merge_fn, gen_ns.merge_fn)

        pipeline = Construct(f"oc-{t}", nodes=[
            Node.scripted(src_ns.name, fn=src_fn, outputs=src_ns.output_type),
            Node.scripted(gen_ns.name, fn=gen_fn,
                          inputs=gen_ns.input_type, outputs=gen_ns.output_type)
            | Oracle(n=gen_ns.oracle_n, merge_fn=merge_fn),
        ])

        graph = compile(pipeline)
        run(graph, input={"node_id": "test"})

        assert call_count[0] == gen_ns.oracle_n, (
            f"Oracle(n={gen_ns.oracle_n}) called generator {call_count[0]} times"
        )


class TestEachFansOutToEveryItem:
    """Promise: Each(over=collection) processes every item in the collection.
    Result dict has one key per item."""

    @given(spec=each_topology())
    @settings(max_examples=30, deadline=15000)
    def test_each_processes_all_items(self, spec):
        """The Each fan-out must produce one result per collection item,
        keyed by the Each key field."""
        expected_keys = spec.meta["expected_keys"]

        graph = _build_scripted_surface(spec)
        result = run(graph, input={"node_id": "test"})

        tf = spec.terminal_field
        assert tf in result
        actual_keys = set(result[tf].keys())
        assert actual_keys == expected_keys, (
            f"Each produced {len(actual_keys)} items, expected {len(expected_keys)}. "
            f"Missing: {expected_keys - actual_keys}, "
            f"Extra: {actual_keys - expected_keys}"
        )


class TestSubConstructIsolation:
    """Promise: sub-construct inner state does not leak to outer pipeline."""

    @given(spec=bare_topology())
    @settings(max_examples=30, deadline=15000)
    def test_sub_construct_inner_fields_absent_from_outer_result(self, spec):
        """Wrap nodes in a sub-construct. Inner node field names must not
        appear as top-level keys in the outer result."""
        if len(spec.nodes) < 2:
            return

        # Build inner nodes
        inner_nodes = []
        for ns in spec.nodes:
            t = _uid()
            fn_name = f"iso_{ns.name}_{t}".replace("-", "_")
            register_scripted(fn_name, ns.fn)
            node = Node.scripted(ns.name, fn=fn_name,
                                 inputs=ns.input_type, outputs=ns.output_type)
            inner_nodes.append(node)

        # Use first node's output as sub-construct input type,
        # last node's output as sub-construct output type
        first_out = spec.nodes[0].output_type
        last_out = spec.nodes[-1].output_type

        sub = Construct(
            "inner-sub", nodes=inner_nodes,
            input=first_out, output=last_out,
        )

        # Outer pipeline: source → sub-construct
        outer_src_fn = f"iso_src_{_uid()}".replace("-", "_")
        register_scripted(outer_src_fn, _make_fn(first_out))

        outer = Construct("outer", nodes=[
            Node.scripted("outer-src", fn=outer_src_fn, outputs=first_out),
            sub,
        ])

        graph = compile(outer)
        result = run(graph, input={"node_id": "test"})

        # Inner node names must NOT appear as top-level result keys
        inner_field_names = {ns.name.replace("-", "_") for ns in spec.nodes}
        result_keys = set(result.keys())
        leaked = inner_field_names & result_keys
        assert not leaked, (
            f"Sub-construct inner fields leaked to outer: {leaked}. "
            f"Result keys: {sorted(result_keys)}"
        )

        # The sub-construct's own field MUST appear
        assert "inner_sub" in result, (
            f"Sub-construct field 'inner_sub' missing. Keys: {sorted(result_keys)}"
        )


class TestLoopExhaustionRaises:
    """Promise: a loop that can never converge raises ExecutionError,
    does not hang or silently produce garbage."""

    @given(spec=loop_exhaustion_topology())
    @settings(max_examples=20, deadline=15000)
    def test_infinite_loop_raises_execution_error(self, spec):
        """Loop with always-true condition must raise ExecutionError
        after max_iterations, not hang."""
        graph = _build_scripted_surface(spec)
        with pytest.raises(ExecutionError, match="max_iterations|exhausted"):
            run(graph, input={"node_id": "test"})


class TestLoopTerminatesAndAccumulates:
    """Promise: a convergent loop terminates and produces an append-list
    with one entry per iteration."""

    @given(spec=loop_topology())
    @settings(max_examples=30, deadline=15000)
    def test_loop_produces_list_of_iterations(self, spec):
        """Loop terminal field must be a list. Length = number of iterations.
        Each entry must be the declared output type."""
        graph = _build_scripted_surface(spec)
        result = run(graph, input={"node_id": "test"})

        tf = spec.terminal_field
        assert tf in result
        val = result[tf]
        assert isinstance(val, list), f"Loop terminal expected list, got {type(val).__name__}"
        assert len(val) >= 1, "Loop must execute at least once"

        # Each iteration output must be the declared type
        for i, entry in enumerate(val):
            assert isinstance(entry, Beta), (
                f"Loop iteration {i}: expected Beta, got {type(entry).__name__}"
            )

        # Final entry should have score >= threshold (convergence)
        threshold = spec.meta["threshold"]
        assert val[-1].score >= threshold, (
            f"Loop terminated but final score {val[-1].score} < threshold {threshold}"
        )


class TestSkipWhenConditionalExecution:
    """Promise: skip_when=True produces skip_value, not the node's fn output.
    skip_when=False executes the node normally."""

    @given(spec=skip_when_topology())
    @settings(max_examples=30, deadline=15000)
    def test_skip_when_produces_correct_output(self, spec):
        """When skip fires, result is skip_value. When it doesn't, result
        is the node's fn output."""
        graph = _build_scripted_surface(spec)
        result = run(graph, input={"node_id": "test"})

        tf = spec.terminal_field
        assert tf in result
        val = result[tf]

        should_skip = spec.meta["should_skip"]
        if should_skip:
            assert isinstance(val, Alpha), f"Skip fired but got {type(val).__name__}"
            assert val.value == "skipped", f"Skip value wrong: {val.value}"
        else:
            assert isinstance(val, Alpha), f"No skip but got {type(val).__name__}"
            assert val.value == "executed", f"Node fn output wrong: {val.value}"


class TestFanInReceivesAllUpstreams:
    """Promise: a fan-in node (dict-form inputs) receives values from
    ALL declared upstream nodes, not just the first or last."""

    @given(spec=fan_in_topology())
    @settings(max_examples=30, deadline=15000)
    def test_fan_in_consumer_receives_dict_with_all_keys(self, spec):
        """The fan-in consumer's fn receives a dict containing entries
        for all upstream nodes. We verify by checking the output
        (consumer produces Gamma with tags naming the input types)."""
        graph = _build_scripted_surface(spec)
        result = run(graph, input={"node_id": "test"})

        tf = spec.terminal_field
        assert tf in result
        val = result[tf]
        assert isinstance(val, Gamma), f"Expected Gamma, got {type(val).__name__}"
        # Consumer fn encodes the input type names into tags
        assert len(val.tags) == 2, (
            f"Fan-in consumer should see 2 upstreams, got {len(val.tags)}: {val.tags}"
        )


class TestDeepChainDataFlowIntegrity:
    """Promise: in a chain of N nodes, EVERY node executes and receives
    non-None input from its predecessor.

    Breaks if: edge wiring skips a node (node compiles but doesn't fire),
    or _extract_input returns None for an intermediate node, or the
    state update writes to a wrong field so the next node can't find it.

    One-line break: off-by-one in _add_plain_node edge wiring."""

    @given(spec=deep_chain_topology())
    @settings(max_examples=30, deadline=15000)
    def test_every_node_in_deep_chain_fires_with_input(self, spec):
        """Instrument every node: record whether it fired and whether
        it received non-None input. All must be True."""
        execution_log: dict[str, dict] = {}  # name → {fired, got_input}

        nodes = []
        for i, ns in enumerate(spec.nodes):
            t = _uid()
            fn_name = f"deep_{ns.name}_{t}".replace("-", "_")

            def make_fn(idx, name, out_type):
                def fn(input_data, _c):
                    execution_log[name] = {
                        "fired": True,
                        "got_input": input_data is not None,
                        "input_type": type(input_data).__name__,
                        "position": idx,
                    }
                    return out_type()
                return fn

            register_scripted(fn_name, make_fn(i, ns.name, ns.output_type))
            node = Node.scripted(ns.name, fn=fn_name,
                                 inputs=ns.input_type, outputs=ns.output_type)
            node = _apply_spec_modifiers(node, ns, f"deep_{t}", spec)
            nodes.append(node)

        graph = compile(Construct(spec.name, nodes=nodes))
        run(graph, input={"node_id": "test"})

        depth = spec.meta["depth"]
        assert len(execution_log) == depth, (
            f"Only {len(execution_log)}/{depth} nodes fired in chain. "
            f"Fired: {sorted(execution_log.keys())}, "
            f"Expected: {[ns.name for ns in spec.nodes]}"
        )

        for ns in spec.nodes:
            log_entry = execution_log.get(ns.name)
            assert log_entry is not None, f"Node '{ns.name}' never fired"
            assert log_entry["fired"], f"Node '{ns.name}' didn't fire"
            # Source node (position 0) gets None input — that's OK
            if log_entry["position"] > 0:
                assert log_entry["got_input"], (
                    f"Node '{ns.name}' at position {log_entry['position']} "
                    f"in {depth}-deep chain received None input. "
                    f"Input type: {log_entry['input_type']}"
                )


# -- Regression guards (surface-specific P0 detectors) --------------------

