"""Cross-surface regression guard tests.

Each test exercises the same pipeline through multiple API surfaces (scripted,
YAML loader, LLM fake) and verifies behavioral equivalence. These are
deterministic, not Hypothesis-driven — they catch surface-specific bugs like
gy9r (YAML mutation) and bchn (Construct propagation).
"""

from __future__ import annotations

from neograph import Construct, Node, compile, run
from neograph.factory import register_condition, register_scripted
from neograph.modifiers import Each, Loop, Oracle

from .conftest import Alpha, Beta, FanCollection, FanItem, Gamma, SubInput, SubOutput


class TestYAMLSurfaceSharedNodeImmutability:
    """YAML surface: shared nodes across sub-constructs must not corrupt.
    Regression guard for neograph-gy9r."""

    def test_shared_node_in_two_yaml_sub_constructs(self):
        """A node used in a YAML sub-construct must not have its inputs
        mutated in the all_nodes dict. Before gy9r fix, _build_sub_construct
        mutated the shared Node.inputs directly."""
        from neograph.loader import _build_sub_construct
        from neograph.spec_types import register_type

        register_type("SubInput", SubInput)
        register_type("SubOutput", SubOutput)

        register_scripted("yaml_shared_worker",
                          lambda _i, _c: SubOutput(result="from-worker"))

        worker_node = Node(
            name="worker", outputs=SubOutput,
            scripted_fn="yaml_shared_worker",
        )

        all_nodes = {"worker": worker_node}
        original_inputs = worker_node.inputs

        # First sub-construct wires the node
        sub_a = _build_sub_construct(
            {"name": "sub-a", "input": "SubInput", "output": "SubOutput",
             "nodes": ["worker"]},
            all_nodes,
        )

        # Original in all_nodes must be unchanged (immutable IR)
        assert all_nodes["worker"].inputs == original_inputs, (
            f"_build_sub_construct mutated shared node! "
            f"Was {original_inputs!r}, now {all_nodes['worker'].inputs!r}"
        )

        # The sub-construct's copy must have inputs wired
        assert sub_a.nodes[0].inputs is not None

    def test_yaml_sub_construct_node_gets_correct_inputs(self):
        """YAML sub-construct with 2 nodes: second node gets dict-form
        inputs wired to input port + first node's output."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("SubInput", SubInput)
        register_type("SubOutput", SubOutput)

        register_scripted("yaml_step1",
                          lambda _i, _c: SubInput(payload="step1-out"))
        register_scripted("yaml_step2",
                          lambda _i, _c: SubOutput(result="step2-out"))

        spec = {
            "name": "yaml-2node-sub",
            "nodes": [
                {"name": "source", "mode": "scripted",
                 "scripted_fn": "yaml_step1", "outputs": "SubInput"},
                {"name": "step1", "mode": "scripted",
                 "scripted_fn": "yaml_step1", "outputs": "SubInput"},
                {"name": "step2", "mode": "scripted",
                 "scripted_fn": "yaml_step2", "outputs": "SubOutput"},
            ],
            "constructs": [
                {"name": "process", "input": "SubInput", "output": "SubOutput",
                 "nodes": ["step1", "step2"]},
            ],
            "pipeline": {"nodes": ["source", "process"]},
        }

        construct = load_spec(spec)
        graph = compile(construct)
        result = run(graph, input={"node_id": "test"})

        assert "process" in result
        assert isinstance(result["process"], SubOutput)


class TestLLMSurfaceConfigPropagation:
    """LLM surface: llm_config must reach the factory with merged values.
    Regression guard for neograph-bchn."""

    def test_merged_llm_config_reaches_factory(self):
        """Construct(llm_config=parent) + Node(llm_config=child) → factory
        receives merged dict with child winning on conflicts."""
        from tests.fakes import StructuredFake, configure_fake_llm

        received_configs = []

        def tracking_factory(tier, **kw):
            received_configs.append(kw.get("llm_config", {}))
            return StructuredFake(respond=lambda model: model())

        configure_fake_llm(tracking_factory)

        child = Node(
            "cfg-child", mode="think", model="fast",
            prompt="test", outputs=Alpha,
            llm_config={"temperature": 0.7},
        )
        original_config = child.llm_config.copy()

        pipeline = Construct("cfg-test", nodes=[child],
                             llm_config={"temperature": 0.3, "max_retries": 5})
        graph = compile(pipeline)
        run(graph, input={"node_id": "test"})

        # Original node unchanged (immutable IR)
        assert child.llm_config == original_config, (
            f"Original mutated: {child.llm_config} != {original_config}"
        )

        # Inner node has merged config (child wins on temperature)
        inner = pipeline.nodes[0]
        assert inner.llm_config["temperature"] == 0.7, "Child must win on conflict"
        assert inner.llm_config["max_retries"] == 5, "Parent field must propagate"

        # Factory received the merged config
        assert len(received_configs) >= 1, "Factory was never called"
        factory_config = received_configs[0]
        assert factory_config.get("temperature") == 0.7
        assert factory_config.get("max_retries") == 5

    def test_renderer_propagation_immutable(self):
        """Construct(renderer=X) propagates to children via model_copy,
        not mutation."""
        child = Node.scripted("rend-child", fn="noop", outputs=Alpha)
        assert child.renderer is None

        sentinel = object()
        pipeline = Construct("rend-test", nodes=[child], renderer=sentinel)

        # Original unchanged
        assert child.renderer is None, "Original node mutated by Construct"
        # Copy in construct has it
        assert pipeline.nodes[0].renderer is sentinel


class TestLLMSurfaceOracleCallCount:
    """LLM surface: Oracle(n=N) must call the LLM exactly N times."""

    def test_oracle_calls_llm_n_times(self):
        """Oracle with N=3 on an LLM node must invoke StructuredFake 3 times
        for the generator, then once for the merge (scripted)."""
        from tests.fakes import StructuredFake, configure_fake_llm

        call_count = [0]

        def counting_respond(model):
            call_count[0] += 1
            return model()

        fake = StructuredFake(respond=counting_respond)
        configure_fake_llm(lambda _tier, **kw: fake)

        register_scripted("oracle_merge_count", lambda _i, _c: Beta())

        pipeline = Construct("llm-oracle-count", nodes=[
            Node("o-src", mode="think", model="fake",
                 prompt="gen", outputs=Alpha),
            Node("o-gen", mode="think", model="fake",
                 prompt="produce", inputs=Alpha, outputs=Beta)
            | Oracle(n=3, merge_fn="oracle_merge_count"),
        ])
        graph = compile(pipeline)
        call_count[0] = 0
        run(graph, input={"node_id": "test"})

        # 1 call for src + 3 calls for Oracle variants = 4 LLM calls
        assert call_count[0] == 4, (
            f"Expected 4 LLM calls (1 src + 3 oracle), got {call_count[0]}"
        )


class TestCrossSurfaceValueEquivalence:
    """Same deterministic function → same output values across surfaces."""

    def test_scripted_vs_decorator_produce_same_values(self):
        """A deterministic transform through scripted API and @node must
        produce identical output values."""
        from neograph._construct_builder import construct_from_functions
        from neograph.decorators import node as node_dec

        # Scripted surface
        register_scripted("equiv_src", lambda _i, _c: Alpha(value="seed-42"))
        register_scripted("equiv_xform", lambda _i, _c: Beta(score=99.5, iteration=7))

        scripted_pipe = Construct("equiv-scripted", nodes=[
            Node.scripted("equiv-src", fn="equiv_src", outputs=Alpha),
            Node.scripted("equiv-xform", fn="equiv_xform",
                          inputs=Alpha, outputs=Beta),
        ])
        g1 = compile(scripted_pipe)
        r1 = run(g1, input={"node_id": "test"})

        # Decorator surface — same functions
        @node_dec(outputs=Alpha)
        def equiv_dec_src() -> Alpha:
            return Alpha(value="seed-42")

        @node_dec(outputs=Beta)
        def equiv_dec_xform(equiv_dec_src: Alpha) -> Beta:
            return Beta(score=99.5, iteration=7)

        dec_pipe = construct_from_functions("equiv-dec", [equiv_dec_src, equiv_dec_xform])
        g2 = compile(dec_pipe)
        r2 = run(g2, input={"node_id": "test"})

        # VALUES must match, not just types
        assert r1["equiv_xform"] == r2["equiv_dec_xform"], (
            f"Scripted: {r1['equiv_xform']} != Decorator: {r2['equiv_dec_xform']}"
        )


class TestEachKeyParityAcrossSurfaces:
    """Each fan-out must produce the same keys regardless of surface."""

    def test_each_keys_match_scripted_vs_yaml(self):
        """Same 3 items fanned out via programmatic and YAML surfaces
        must produce identical dict keys."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("FanCollection", FanCollection)
        register_type("FanItem", FanItem)
        register_type("Beta", Beta)

        items = [FanItem(item_id=f"k{i}", data=f"val{i}") for i in range(3)]
        expected_keys = {f"k{i}" for i in range(3)}

        # --- Scripted surface ---
        register_scripted("ek_src", lambda _i, _c: FanCollection(items=items))
        register_scripted("ek_proc", lambda _i, _c: Beta(score=0.5))

        scripted_pipe = Construct("ek-scripted", nodes=[
            Node.scripted("ek-src", fn="ek_src", outputs=FanCollection),
            Node.scripted("ek-proc", fn="ek_proc",
                          inputs=FanItem, outputs=Beta)
            | Each(over="ek_src.items", key="item_id"),
        ])
        g1 = compile(scripted_pipe)
        r1 = run(g1, input={"node_id": "test"})
        scripted_keys = set(r1["ek_proc"].keys())

        # --- YAML surface ---
        register_scripted("ek_src_y", lambda _i, _c: FanCollection(items=items))
        register_scripted("ek_proc_y", lambda _i, _c: Beta(score=0.5))

        spec = {
            "name": "ek-yaml",
            "nodes": [
                {"name": "ek-src-y", "mode": "scripted",
                 "scripted_fn": "ek_src_y", "outputs": "FanCollection"},
                {"name": "ek-proc-y", "mode": "scripted",
                 "scripted_fn": "ek_proc_y",
                 "inputs": {"item": "FanItem"},
                 "outputs": "Beta",
                 "each": {"over": "ek_src_y.items", "key": "item_id"}},
            ],
            "pipeline": {"nodes": ["ek-src-y", "ek-proc-y"]},
        }
        construct = load_spec(spec)
        g2 = compile(construct)
        r2 = run(g2, input={"node_id": "test"})
        yaml_keys = set(r2["ek_proc_y"].keys())

        assert scripted_keys == expected_keys, f"Scripted keys: {scripted_keys}"
        assert yaml_keys == expected_keys, f"YAML keys: {yaml_keys}"
        assert scripted_keys == yaml_keys, "Key sets must match across surfaces"


class TestLoopIterationParityAcrossSurfaces:
    """Same loop condition → same iteration count across surfaces."""

    def test_loop_iters_match_programmatic_vs_forward(self):
        """Same loop (exit at score >= 0.8, step 0.4) through programmatic
        and ForwardConstruct surfaces must execute the same number of iters."""
        from neograph.forward import ForwardConstruct

        prog_count = [0]
        fwd_count = [0]

        def make_loop_fn(counter):
            def fn(_i, _c):
                counter[0] += 1
                return Beta(score=counter[0] * 0.4, iteration=counter[0])
            return fn

        # --- Programmatic surface ---
        register_scripted("lip_seed", lambda _i, _c: Beta(score=0.0))
        register_scripted("lip_body", make_loop_fn(prog_count))
        register_condition("lip_cond", lambda v: v is None or v.score < 0.8)

        prog_pipe = Construct("lip-prog", nodes=[
            Node.scripted("lip-seed", fn="lip_seed", outputs=Beta),
            Node.scripted("lip-body", fn="lip_body",
                          inputs=Beta, outputs=Beta)
            | Loop(when="lip_cond", max_iterations=10),
        ])
        g1 = compile(prog_pipe)
        r1 = run(g1, input={"node_id": "test"})

        # --- ForwardConstruct surface ---
        register_scripted("lif_seed", lambda _i, _c: Beta(score=0.0))
        register_scripted("lif_body", make_loop_fn(fwd_count))

        class LoopFwd(ForwardConstruct):
            seed = Node.scripted("lif-seed", fn="lif_seed", outputs=Beta)
            body = Node.scripted("lif-body", fn="lif_body",
                                 inputs=Beta, outputs=Beta)

            def forward(self, topic):
                s = self.seed(topic)
                return self.loop(
                    body=[self.body],
                    when=lambda v: v is None or v.score < 0.8,
                    max_iterations=10,
                )(s)

        g2 = compile(LoopFwd("lip-fwd"))
        r2 = run(g2, input={"node_id": "test"})

        assert prog_count[0] == fwd_count[0], (
            f"Programmatic: {prog_count[0]} iters, Forward: {fwd_count[0]} iters"
        )
        assert prog_count[0] == 2, f"Expected 2 iterations (0.4, 0.8), got {prog_count[0]}"


class TestLLMSurfaceEachOracleCallCount:
    """LLM Each+Oracle: items × N variant calls."""

    def test_each_oracle_calls_llm_items_times_n(self):
        """Each(3 items) + Oracle(n=2): expect 6 generator LLM calls +
        1 source call = 7 total. This exercises the hardest dispatch combo
        through the LLM surface."""
        from tests.fakes import StructuredFake, configure_fake_llm

        call_count = [0]

        def counting_respond(model):
            call_count[0] += 1
            return model()

        fake = StructuredFake(respond=counting_respond)
        configure_fake_llm(lambda _tier, **kw: fake)

        items = [FanItem(item_id=f"eo{i}") for i in range(3)]
        register_scripted("eo_count_merge", lambda _i, _c: Gamma(tags=[str(len(_i))]))

        pipeline = Construct("llm-each-oracle", nodes=[
            Node("eo-src", mode="think", model="fake",
                 prompt="gen", outputs=FanCollection,
                 llm_config={"output_strategy": "structured"}),
            Node("eo-proc", mode="think", model="fake",
                 prompt="produce", inputs=FanItem, outputs=Gamma)
            | Oracle(n=2, merge_fn="eo_count_merge")
            | Each(over="eo_src.items", key="item_id"),
        ])

        # Pre-register a scripted fn that returns the collection
        # (the LLM fake returns FanCollection() which has items=[])
        # We need the source to actually produce items for Each
        register_scripted("eo_src_scripted",
                          lambda _i, _c: FanCollection(items=items))
        # Replace the first node with scripted to control the collection
        pipeline = Construct("llm-each-oracle", nodes=[
            Node.scripted("eo-src", fn="eo_src_scripted", outputs=FanCollection),
            Node("eo-proc", mode="think", model="fake",
                 prompt="produce", inputs=FanItem, outputs=Gamma)
            | Oracle(n=2, merge_fn="eo_count_merge")
            | Each(over="eo_src.items", key="item_id"),
        ])

        graph = compile(pipeline)
        call_count[0] = 0
        result = run(graph, input={"node_id": "test"})

        # 3 items × 2 oracle variants = 6 LLM calls
        assert call_count[0] == 6, (
            f"Expected 6 LLM calls (3 items × 2 oracle), got {call_count[0]}"
        )
        assert "eo_proc" in result
        assert isinstance(result["eo_proc"], dict)
        assert len(result["eo_proc"]) == 3
