"""Parity test matrix + ratchet (neograph-e9zse.4).

THE INVARIANT (epic neograph-e9zse): every topology class the declarative
``Construct`` form can express, ``ForwardConstruct.forward()`` must ALSO
express — tracing to IDENTICAL IR. This suite is the permanent guard:

- ``PARITY_CORPUS`` pairs each capability with a declarative builder and a
  forward() twin builder; a parametrized test asserts the traced node list
  is structurally identical to the declarative one (``assert_ir_identical``).
- ``REQUIRED_CAPABILITIES`` is the ratchet: a new declarative capability is
  added to the required set FIRST; the coverage test then fails loud until
  its forward() twin row lands. Phase-2 capabilities (Oracle ensemble,
  Operator/interrupt HITL) are tracked in ``PHASE2_PENDING`` and move into
  the required set when neograph-e9zse.5/.6 ship their tracing surfaces.

Do NOT special-case any single topology: the cascade reference shape
(intake -> Loop(get_claims -> Each(verify) -> collect) -> explain) is one
row of the corpus, not the mechanism.

Known non-twin by design: the bare ``for x in proxy`` sugar emits PER-NODE
``node | Each(key="label")`` while ``self.each()`` emits a wrapped
``Construct | Each`` sub-construct — they are DISTINCT corpus rows with
distinct declarative twins, never asserted equivalent to each other.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest
from pydantic import BaseModel

from neograph import Construct, Each, ForwardConstruct, Node, compile, run
from neograph.modifiers import Loop
from tests.fakes import build_test_compile_kwargs, register_scripted

# ═══════════════════════════════════════════════════════════════════════════
# Shared schemas + condition callables (module-level so declarative and
# forward twins share the SAME objects — Loop.when is compared by identity)
# ═══════════════════════════════════════════════════════════════════════════


class PItem(BaseModel, frozen=True):
    pid: str


class PBatch(BaseModel, frozen=True):
    items: list[PItem]


class PText(BaseModel, frozen=True):
    text: str


class PVerdict(BaseModel, frozen=True):
    summary: str
    score: float = 1.0


def _run_once_when(d):
    """Loop condition: run the body once (loop only while nothing produced)."""
    return d is None


def _inner_when(d):
    return d is None


def _skip_never(data):
    return False


def _skip_fallback(data):
    return PText(text="skipped")


def _register_corpus_fns():
    """Idempotently register every scripted fn the corpus rows use."""
    register_scripted(
        "par_seed_batch",
        lambda _i, _c: PBatch(items=[PItem(pid="p1"), PItem(pid="p2")]),
    )
    register_scripted("par_text", lambda _i, _c: PText(text="T"))
    register_scripted("par_passthrough", lambda batch, _c: batch)
    register_scripted("par_verify", lambda item, _c: PText(text=f"ok-{item.pid}"))
    register_scripted(
        "par_fuse",
        lambda data, _c: PVerdict(summary=f"{data['left'].text}+{len(data['right'].items)}"),
    )
    register_scripted(
        "par_collect",
        lambda data, _c: PVerdict(summary=",".join(sorted(v.text for v in data["each_verify"].values()))),
    )
    register_scripted(
        "par_multi",
        lambda _i, _c: {"summary": PText(text="S"), "batch": PBatch(items=[PItem(pid="m")])},
    )
    register_scripted(
        "par_report",
        lambda data, _c: PVerdict(summary=data["analyze_summary"].text),
    )
    register_scripted("par_refine", lambda t, _c: PText(text="refined"))
    register_scripted("par_grade", lambda data, _c: PVerdict(summary="graded"))


# ═══════════════════════════════════════════════════════════════════════════
# The single IR-equality helper — every corpus row goes through this
# ═══════════════════════════════════════════════════════════════════════════


def _modifier_each_params(item):
    """(over, key, on_error) of an Each modifier on a Node or Construct."""
    if item.has_modifier(Each):
        each = item.get_modifier(Each)
        return (each.over, each.key, each.on_error)
    return None


def assert_ir_identical(traced, decl, path="root"):
    """Recursive structural IR equality between a traced item and its
    declarative twin: names, boundary ports, modifier params (Each verbatim,
    Loop with when-identity), node fields, and member recursion."""
    if isinstance(decl, Construct):
        assert isinstance(traced, Construct), (
            f"{path}: expected Construct {decl.name!r}, got {type(traced).__name__}"
        )
        assert traced.name == decl.name, f"{path}: name {traced.name!r} != {decl.name!r}"

        assert _modifier_each_params(traced) == _modifier_each_params(decl), (
            f"{path}: Each modifier params differ on {decl.name!r}"
        )
        assert traced.has_modifier(Loop) == decl.has_modifier(Loop), (
            f"{path}: Loop modifier presence differs on {decl.name!r}"
        )
        if decl.has_modifier(Loop):
            t_loop, d_loop = traced.get_modifier(Loop), decl.get_modifier(Loop)
            assert t_loop.when is d_loop.when, f"{path}: Loop.when identity differs"
            assert t_loop.max_iterations == d_loop.max_iterations
            assert t_loop.on_exhaust == d_loop.on_exhaust

        assert traced.input is decl.input, f"{path}: input port differs on {decl.name!r}"
        assert traced.output is decl.output, f"{path}: output port differs on {decl.name!r}"

        assert len(traced.nodes) == len(decl.nodes), (
            f"{path}: member count {len(traced.nodes)} != {len(decl.nodes)} on {decl.name!r}"
        )
        for i, (t, d) in enumerate(zip(traced.nodes, decl.nodes, strict=True)):
            assert_ir_identical(t, d, path=f"{path}.{decl.name}[{i}]")
    else:
        assert isinstance(traced, Node), (
            f"{path}: expected Node {decl.name!r}, got {type(traced).__name__}"
        )
        assert not isinstance(traced, Construct)
        assert traced.name == decl.name, f"{path}: name {traced.name!r} != {decl.name!r}"
        assert traced.inputs == decl.inputs, f"{path}: inputs differ on {decl.name!r}"
        assert traced.outputs == decl.outputs, f"{path}: outputs differ on {decl.name!r}"
        assert traced.scripted_fn == decl.scripted_fn, (
            f"{path}: scripted_fn differs on {decl.name!r}"
        )
        assert traced.skip_when is decl.skip_when, f"{path}: skip_when differs"
        assert traced.skip_value is decl.skip_value, f"{path}: skip_value differs"
        assert _modifier_each_params(traced) == _modifier_each_params(decl), (
            f"{path}: per-node Each params differ on {decl.name!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Corpus rows — one (capability, declarative builder, forward builder) each.
# Builders return the assembled node list (both post-Construct.__init__).
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass(frozen=True)
class ParityRow:
    capability: str
    declarative_nodes: Callable[[], list]
    forward_nodes: Callable[[], list]


def _decl_straight_line():
    return Construct(
        "twin",
        nodes=[
            Node.scripted("seed", fn="par_seed_batch", outputs=PBatch),
            Node.scripted("shape", fn="par_passthrough", inputs=PBatch, outputs=PBatch),
        ],
    ).nodes


def _fwd_straight_line():
    class Twin(ForwardConstruct):
        seed = Node.scripted("seed", fn="par_seed_batch", outputs=PBatch)
        shape = Node.scripted("shape", fn="par_passthrough", inputs=PBatch, outputs=PBatch)

        def forward(self, topic):
            batch = self.seed(topic)
            return self.shape(batch)

    return Twin().nodes


_FAN_IN_INPUTS = {"left": PText, "right": PBatch}


def _decl_fan_in():
    return Construct(
        "twin",
        nodes=[
            Node.scripted("left", fn="par_text", outputs=PText),
            Node.scripted("right", fn="par_seed_batch", outputs=PBatch),
            Node.scripted("fuse", fn="par_fuse", inputs=_FAN_IN_INPUTS, outputs=PVerdict),
        ],
    ).nodes


def _fwd_fan_in():
    class Twin(ForwardConstruct):
        left = Node.scripted("left", fn="par_text", outputs=PText)
        right = Node.scripted("right", fn="par_seed_batch", outputs=PBatch)
        fuse = Node.scripted("fuse", fn="par_fuse", inputs=_FAN_IN_INPUTS, outputs=PVerdict)

        def forward(self, topic):
            left_out = self.left(topic)
            right_out = self.right(topic)
            return self.fuse(left_out, right_out)

    return Twin().nodes


def _decl_per_node_fan_out():
    return Construct(
        "twin",
        nodes=[
            Node.scripted("seed", fn="par_seed_batch", outputs=PBatch),
            Node.scripted("verify", fn="par_verify", outputs=PText)
            | Each(over="seed.items", key="label"),
        ],
    ).nodes


def _fwd_per_node_fan_out():
    class Twin(ForwardConstruct):
        seed = Node.scripted("seed", fn="par_seed_batch", outputs=PBatch)
        verify = Node.scripted("verify", fn="par_verify", outputs=PText)

        def forward(self, topic):
            batch = self.seed(topic)
            for item in batch.items:
                self.verify(item)
            return batch

    return Twin().nodes


def _decl_each_sub_construct():
    return Construct(
        "twin",
        nodes=[
            Node.scripted("seed", fn="par_seed_batch", outputs=PBatch),
            Construct(
                "each-verify",
                input=PItem,
                output=PText,
                nodes=[
                    Node.scripted("verify", fn="par_verify", inputs=PItem, outputs=PText),
                ],
            )
            | Each(over="seed.items", key="pid", on_error="raise"),
        ],
    ).nodes


def _fwd_each_sub_construct():
    class Twin(ForwardConstruct):
        seed = Node.scripted("seed", fn="par_seed_batch", outputs=PBatch)
        verify = Node.scripted("verify", fn="par_verify", outputs=PText)

        def forward(self, topic):
            batch = self.seed(topic)
            return self.each(body=[self.verify], key="pid")(batch.items)

    return Twin().nodes


def _decl_each_on_error_collect():
    return Construct(
        "twin",
        nodes=[
            Node.scripted("seed", fn="par_seed_batch", outputs=PBatch),
            Construct(
                "each-verify",
                input=PItem,
                output=PText,
                nodes=[
                    Node.scripted("verify", fn="par_verify", inputs=PItem, outputs=PText),
                ],
            )
            | Each(over="seed.items", key="pid", on_error="collect"),
        ],
    ).nodes


def _fwd_each_on_error_collect():
    class Twin(ForwardConstruct):
        seed = Node.scripted("seed", fn="par_seed_batch", outputs=PBatch)
        verify = Node.scripted("verify", fn="par_verify", outputs=PText)

        def forward(self, topic):
            batch = self.seed(topic)
            return self.each(body=[self.verify], key="pid", on_error="collect")(batch.items)

    return Twin().nodes


def _decl_loop_over_nodes():
    return Construct(
        "twin",
        nodes=[
            Node.scripted("seed", fn="par_text", outputs=PText),
            Construct(
                "loop-refine",
                input=PText,
                output=PText,
                nodes=[
                    Node.scripted("refine", fn="par_refine", inputs=PText, outputs=PText),
                ],
            )
            | Loop(when=_run_once_when, max_iterations=3, on_exhaust="last"),
        ],
    ).nodes


def _fwd_loop_over_nodes():
    class Twin(ForwardConstruct):
        seed = Node.scripted("seed", fn="par_text", outputs=PText)
        refine = Node.scripted("refine", fn="par_refine", outputs=PText)

        def forward(self, topic):
            t = self.seed(topic)
            return self.loop(
                body=[self.refine],
                when=_run_once_when,
                max_iterations=3,
                on_exhaust="last",
            )(t)

    return Twin().nodes


_COLLECT_INPUTS = {"each_verify": dict[str, PText]}


def _decl_cascade():
    """The reference cascade: Loop(get_claims -> Each'd verify -> collect)."""
    each_sub = Construct(
        "each-verify",
        input=PItem,
        output=PText,
        nodes=[
            Node.scripted("verify", fn="par_verify", inputs=PItem, outputs=PText),
        ],
    ) | Each(over="get_claims.items", key="pid", on_error="raise")
    return Construct(
        "twin",
        nodes=[
            Node.scripted("intake", fn="par_seed_batch", outputs=PBatch),
            Construct(
                "loop-get_claims-each-verify-collect",
                input=PBatch,
                output=PVerdict,
                nodes=[
                    Node.scripted(
                        "get_claims", fn="par_passthrough", inputs=PBatch, outputs=PBatch
                    ),
                    each_sub,
                    Node.scripted(
                        "collect", fn="par_collect", inputs=_COLLECT_INPUTS, outputs=PVerdict
                    ),
                ],
            )
            | Loop(when=_run_once_when, max_iterations=3, on_exhaust="last"),
        ],
    ).nodes


def _fwd_cascade():
    return _fwd_cascade_pipeline().nodes


def _fwd_cascade_pipeline():
    class Twin(ForwardConstruct):
        intake = Node.scripted("intake", fn="par_seed_batch", outputs=PBatch)
        get_claims = Node.scripted("get_claims", fn="par_passthrough", outputs=PBatch)
        verify = Node.scripted("verify", fn="par_verify", outputs=PText)
        collect = Node.scripted(
            "collect", fn="par_collect", inputs=_COLLECT_INPUTS, outputs=PVerdict
        )

        def forward(self, topic):
            batch = self.intake(topic)
            return self.loop(
                body=[
                    self.get_claims,
                    self.each(body=[self.verify], key="pid", over="get_claims.items"),
                    self.collect,
                ],
                when=_run_once_when,
                max_iterations=3,
                on_exhaust="last",
            )(batch)

    return Twin()


_GRADE_INPUTS = {"loop_refine": PText}


def _decl_loop_in_loop():
    inner = Construct(
        "loop-refine",
        input=PText,
        output=PText,
        nodes=[Node.scripted("refine", fn="par_refine", inputs=PText, outputs=PText)],
    ) | Loop(when=_inner_when, max_iterations=2, on_exhaust="last")
    return Construct(
        "twin",
        nodes=[
            Node.scripted("seed", fn="par_text", outputs=PText),
            Construct(
                "loop-shape-loop-refine-grade",
                input=PText,
                output=PVerdict,
                nodes=[
                    Node.scripted("shape", fn="par_refine", inputs=PText, outputs=PText),
                    inner,
                    Node.scripted(
                        "grade", fn="par_grade", inputs=_GRADE_INPUTS, outputs=PVerdict
                    ),
                ],
            )
            | Loop(when=_run_once_when, max_iterations=3, on_exhaust="last"),
        ],
    ).nodes


def _fwd_loop_in_loop():
    class Twin(ForwardConstruct):
        seed = Node.scripted("seed", fn="par_text", outputs=PText)
        shape = Node.scripted("shape", fn="par_refine", outputs=PText)
        refine = Node.scripted("refine", fn="par_refine", outputs=PText)
        grade = Node.scripted("grade", fn="par_grade", inputs=_GRADE_INPUTS, outputs=PVerdict)

        def forward(self, topic):
            t = self.seed(topic)
            return self.loop(
                body=[
                    self.shape,
                    self.loop(
                        body=[self.refine],
                        when=_inner_when,
                        max_iterations=2,
                        on_exhaust="last",
                    ),
                    self.grade,
                ],
                when=_run_once_when,
                max_iterations=3,
                on_exhaust="last",
            )(t)

    return Twin().nodes


_MULTI_OUTPUTS = {"summary": PText, "batch": PBatch}
_REPORT_INPUTS = {"analyze_summary": PText}


def _decl_multi_output():
    return Construct(
        "twin",
        nodes=[
            Node.scripted("analyze", fn="par_multi", outputs=_MULTI_OUTPUTS),
            Node.scripted("report", fn="par_report", inputs=_REPORT_INPUTS, outputs=PVerdict),
        ],
    ).nodes


def _fwd_multi_output():
    class Twin(ForwardConstruct):
        analyze = Node.scripted("analyze", fn="par_multi", outputs=_MULTI_OUTPUTS)
        report = Node.scripted(
            "report", fn="par_report", inputs=_REPORT_INPUTS, outputs=PVerdict
        )

        def forward(self, topic):
            analyzed = self.analyze(topic)
            return self.report(analyzed.summary)

    return Twin().nodes


def _decl_skip_marks():
    return Construct(
        "twin",
        nodes=[
            Node.scripted("seed", fn="par_text", outputs=PText),
            Node(
                "guarded",
                mode="scripted",
                scripted_fn="par_refine",
                inputs=PText,
                outputs=PText,
                skip_when=_skip_never,
                skip_value=_skip_fallback,
            ),
        ],
    ).nodes


def _fwd_skip_marks():
    class Twin(ForwardConstruct):
        seed = Node.scripted("seed", fn="par_text", outputs=PText)
        guarded = Node(
            "guarded",
            mode="scripted",
            scripted_fn="par_refine",
            inputs=PText,
            outputs=PText,
            skip_when=_skip_never,
            skip_value=_skip_fallback,
        )

        def forward(self, topic):
            t = self.seed(topic)
            return self.guarded(t)

    return Twin().nodes


PARITY_CORPUS = [
    ParityRow("straight_line", _decl_straight_line, _fwd_straight_line),
    ParityRow("fan_in", _decl_fan_in, _fwd_fan_in),
    ParityRow("per_node_fan_out", _decl_per_node_fan_out, _fwd_per_node_fan_out),
    ParityRow("each_sub_construct_custom_key", _decl_each_sub_construct, _fwd_each_sub_construct),
    ParityRow("each_on_error_collect", _decl_each_on_error_collect, _fwd_each_on_error_collect),
    ParityRow("loop_over_nodes", _decl_loop_over_nodes, _fwd_loop_over_nodes),
    ParityRow("loop_over_sub_construct_body", _decl_cascade, _fwd_cascade),
    ParityRow("loop_in_loop", _decl_loop_in_loop, _fwd_loop_in_loop),
    ParityRow("multi_output_dict", _decl_multi_output, _fwd_multi_output),
    ParityRow("skip_marks", _decl_skip_marks, _fwd_skip_marks),
]

# The ratchet: a NEW declarative capability is added here FIRST; the coverage
# test below then fails loud until its forward() twin row lands in the corpus.
REQUIRED_CAPABILITIES = frozenset(row.capability for row in PARITY_CORPUS)

# Phase-2 tracing surfaces (epic neograph-e9zse): move each key from here
# into REQUIRED_CAPABILITIES (with its corpus row) when the surface ships.
#   - "oracle_ensemble"    -> neograph-e9zse.5 (self.ensemble())
#   - "operator_interrupt" -> neograph-e9zse.6 (HITL exposure)
PHASE2_PENDING = frozenset({"oracle_ensemble", "operator_interrupt"})


class TestParityMatrix:
    """Every corpus row's forward() twin traces to IR identical to its
    declarative builder — the epic's acceptance principle, one row per
    topology class."""

    @pytest.mark.parametrize("row", PARITY_CORPUS, ids=lambda r: r.capability)
    def test_traced_ir_identical_to_declarative_when_twin_built(self, row):
        _register_corpus_fns()
        decl_nodes = row.declarative_nodes()
        traced_nodes = row.forward_nodes()
        assert len(traced_nodes) == len(decl_nodes), (
            f"{row.capability}: node count {len(traced_nodes)} != {len(decl_nodes)}"
        )
        for i, (traced, decl) in enumerate(zip(traced_nodes, decl_nodes, strict=True)):
            assert_ir_identical(traced, decl, path=f"{row.capability}[{i}]")


class TestParityRatchet:
    """The fail-loud guard: required capabilities must all have corpus rows,
    and pending Phase-2 keys must not silently rot."""

    def test_every_required_capability_has_a_corpus_row(self):
        covered = {row.capability for row in PARITY_CORPUS}
        missing = REQUIRED_CAPABILITIES - covered
        assert not missing, (
            f"declarative capabilities without a forward() parity row: {sorted(missing)}. "
            "Add the forward() twin (and its corpus row) before shipping the capability."
        )

    def test_corpus_has_no_duplicate_capability_keys(self):
        keys = [row.capability for row in PARITY_CORPUS]
        assert len(keys) == len(set(keys)), "duplicate capability keys in PARITY_CORPUS"

    def test_phase2_pending_keys_do_not_overlap_required(self):
        overlap = PHASE2_PENDING & REQUIRED_CAPABILITIES
        assert not overlap, (
            f"{sorted(overlap)} shipped: remove from PHASE2_PENDING (its corpus row "
            "now carries the coverage)"
        )


class TestCascadeReferenceIntegration:
    """The cascade reference row also runs end-to-end: the traced pipeline
    and the declarative twin produce the same outputs on the same input."""

    def test_traced_cascade_runs_and_matches_declarative_run(self):
        _register_corpus_fns()

        traced_pipeline = _fwd_cascade_pipeline()
        traced_graph = compile(traced_pipeline, **build_test_compile_kwargs())
        traced_result = run(traced_graph, input={"node_id": "parity-cascade-fwd"})

        declarative_parent = Construct("Twin", nodes=_decl_cascade())
        decl_graph = compile(declarative_parent, **build_test_compile_kwargs())
        decl_result = run(decl_graph, input={"node_id": "parity-cascade-decl"})

        traced_out = traced_result["loop_get_claims_each_verify_collect"]
        decl_out = decl_result["loop_get_claims_each_verify_collect"]
        assert traced_out == decl_out
        assert PVerdict(summary="ok-p1,ok-p2") in (
            traced_out if isinstance(traced_out, list) else [traced_out]
        )
