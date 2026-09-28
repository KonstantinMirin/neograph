"""A declared read must resolve to the value its author named -- run-level.

``neograph-4cvx8`` step 1 (child ``neograph-4cvx8.2``). The epic's Core Invariant:
*every declared read leaves ``Construct()`` either holding exactly one stamped
``Source`` or refused with a reason*. Five separately-filed bugs say that today it
can do neither -- assembly reports nothing, the runtime has an absence to
interpret, and the body is handed ``None`` on a green run.

This file is the RUN-LEVEL half of the evidence.
``tests/test_guards_declared_read_stamped.py`` (step 0) measures whether a read
holds a stamp at all; a stamp is presence, never correctness -- N2 below is
stamped and WRONG. So these tests build the graph, RUN it, and assert the VALUE
the consuming body received, through ``run()``'s own result. Where the expected
new behaviour is an assembly REFUSAL, they assert the ``ConstructError`` and what
its message has to say.

The five, with the behaviour measured on ``develop`` @ 812454f:

=========================  ===================================================
``neograph-la3a4``         declarative/programmatic ``inputs=list[T]`` after a
                           ``Loop`` producer of ``T``: validation passes
                           (``_loop_aware_compatible``), the resolver uses plain
                           ``_types_compatible`` over triples that discard
                           ``Producer.is_loop``, nothing is stamped, the body
                           receives ``None``. Measured: ``Summary(seen=[])``.
``neograph-z2ayo`` (N2)    ``pre(R) -> branch(R, R) -> after(inputs=R)``: the
                           validator's ``ArmScopedProducers`` hides arm producers
                           after the join and passes on ``pre``, while
                           ``_stamp_single_type_sources`` extends ``visible`` with
                           BOTH arms and takes ``matches[-1]``. Measured stamp:
                           ``Peer(false_arm)``; the TRUE arm ran; ``after``
                           received ``None``.
``neograph-yi9t5`` (N3)    a ported sub-construct at position 0 of a ported
                           parent: ``port_source_field`` sees no preceding
                           producer and has no ``Port()`` fallback, so the inner
                           body receives ``None`` while the parent's own port
                           holds the value.
``neograph-chunx`` (N8)    ``stamp_sub_construct_ports`` walks both arms as one
                           flat list (``iter_item_slots``), so a FALSE-arm
                           sub-construct is stamped with a TRUE-arm producer.
                           Measured stamp: ``Peer(true_arm)``; on the false path
                           that field is unbound and the inner body receives
                           ``None``. (Filed as "reported, not yet run-verified" --
                           it is run-verified now.)
``neograph-x24iw`` (N9)    one ``Construct`` object placed in two parents keeps
                           the FIRST parent's ``port_source`` (assigned in place,
                           never overwritten), so the second parent's child reads
                           a field that does not exist there. Measured: parent A's
                           child saw ``FROM-ALPHA``, parent B's child saw ``None``.
=========================  ===================================================

Surfaces, and the exemptions
----------------------------
The three-surface parity rule applies to ``neograph-la3a4``, which is a plain
consumer read: it is exercised on the DECLARATIVE and PROGRAMMATIC surfaces, and
the ``@node`` decorator is the PARITY REFERENCE -- a ``list[T]`` parameter is
dict-form, which is exactly why the documented ``all_in_scope`` projection works
there today and nowhere else (``tests/test_loop.py::TestLoopScopeProjections``).
That asymmetry IS the bug, so the decorator case is asserted here as the
reference behaviour rather than as a third red case.

The other four have a narrower surface set, for structural reasons:

* A branch arm exists only through ``ForwardConstruct`` tracing or a programmatic
  ``_BranchNode`` sentinel -- ``construct_from_functions`` has no branch spelling.
  N2 is therefore covered on BOTH of the surfaces that can express it.
* ``ForwardConstruct._discover_node_attrs`` collects ``Node`` class attributes
  only ("sub-pipelines enter a trace via self.each() / self.loop() /
  self.ensemble(), never as class attrs"), and ``ignored_types=(Node,)`` makes a
  ``Construct`` class attribute a Pydantic error. N3/N8/N9 are sub-construct PORT
  questions, so the decorator surface cannot express them at all.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from neograph import (
    Construct,
    ConstructError,
    Each,
    ForwardConstruct,
    Loop,
    Node,
    compile,
    construct_from_functions,
    node,
    run,
)
from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
from tests.fakes import build_test_compile_kwargs, register_scripted

# ═══════════════════════════════════════════════════════════════════════════
# Schemas and body builders
#
# Every body RECORDS what it was handed and ECHOES an identifying label, so the
# value under test is observable through run()'s own result -- the same public
# surface that produced it -- and the recording is only a diagnostic for the
# failure message.
# ═══════════════════════════════════════════════════════════════════════════


class Draft(BaseModel, frozen=True):
    content: str
    iteration: int = 0
    score: float = 0.0


class Summary(BaseModel, frozen=True):
    iterations_seen: list[int]


class Token(BaseModel, frozen=True):
    """The carrier type. ``take_true`` and ``score`` drive branch conditions.

    Two spellings because the two branch surfaces need different ones: the
    programmatic ``_ConditionSpec`` carries its own ``op_fn`` and reads
    ``take_true`` directly, while a traced ``ForwardConstruct`` condition must be
    a COMPARISON (``score > 0.5``). A bare truthiness condition traces to
    ``_ConditionSpec(op_fn=operator.truth, attr_chain=[])`` and
    ``_wiring_branch`` then calls it as ``op_fn(value, threshold)`` -- two
    arguments to a one-argument callable, so every such branch dies at run time
    with ``ExecutionError: branch condition raised TypeError``. Unrelated to this
    epic; avoided here rather than worked around, and reported on the bead.
    """

    label: str
    take_true: bool = False
    score: float = 0.0


class Echo(BaseModel, frozen=True):
    saw: str


class Bag(BaseModel, frozen=True):
    """A container to fan over, for the one Each case in this file."""

    items: list[Token]


def _absent(value: Any) -> str:
    """The label an echo body emits when it was handed no usable value.

    Spelled out rather than crashing, so the assertion reports WHAT arrived
    instead of an ``AttributeError`` from inside a node body.
    """
    return f"NO-VALUE:{value!r}"


class _Recorder:
    """A scripted body that records its input and echoes an identifying label.

    The builder for every consumer in this file. Instantiated per test, so no
    module-level state leaks between cases.
    """

    def __init__(self) -> None:
        self.received: list[Any] = []

    def echo(self, input_data: Any, _config: Any) -> Echo:
        self.received.append(input_data)
        label = input_data.label if isinstance(input_data, Token) else _absent(input_data)
        return Echo(saw=label)

    def summarize_history(self, input_data: Any, _config: Any) -> Summary:
        self.received.append(input_data)
        history = input_data if isinstance(input_data, list) else []
        return Summary(iterations_seen=[d.iteration for d in history])


def _emit_token(label: str, *, take_true: bool = False, score: float = 0.0) -> Any:
    """A constant-``Token`` scripted body."""

    def body(_input_data: Any, _config: Any) -> Token:
        return Token(label=label, take_true=take_true, score=score)

    return body


def _keep_refining(draft: Draft | None) -> bool:
    """The Loop condition: a None-safe score threshold."""
    return draft is None or draft.score < 0.9


def _refine(input_data: Any, _config: Any) -> Draft:
    """One refinement step: reads the latest Draft, emits the next iteration."""
    previous = input_data if isinstance(input_data, Draft) else None
    nth = (previous.iteration if previous else 0) + 1
    return Draft(content=f"v{nth}", iteration=nth, score=0.4 * nth)


# The history a 5-iteration cap with a 0.9 threshold and a 0.4 step produces:
# scores 0.4, 0.8, 1.2 -- three iterations, then the condition goes false.
EXPECTED_HISTORY = [1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════════
# Part 1 -- neograph-la3a4
#
# `all_in_scope`: a Loop-modified node's state field IS the per-iteration
# history (state.py's PrimaryShape.LOOP -- Annotated[list[T], append]), so a
# downstream consumer declaring list[T] receives all of it. AGENTS.md documents
# this as a free projection over storage that already exists. It works on the
# @node surface because a `refine: list[Draft]` parameter is DICT-FORM; the
# single-type declarative/programmatic spelling of the same read is accepted by
# the validator and then resolved by nothing.
# ═══════════════════════════════════════════════════════════════════════════


def _register_loop_bodies(recorder: _Recorder) -> None:
    register_scripted("dr_seed", lambda _i, _c: Draft(content="v0"))
    register_scripted("dr_refine", _refine)
    register_scripted("dr_summarize", recorder.summarize_history)


def _loop_history_declarative(recorder: _Recorder) -> Construct:
    """Declarative surface: ``Node.scripted(inputs=list[Draft])`` after a Loop."""
    _register_loop_bodies(recorder)
    return Construct(
        "la3a4-declarative",
        nodes=[
            Node.scripted("seed", fn="dr_seed", outputs=Draft),
            Node.scripted("refine", fn="dr_refine", inputs=Draft, outputs=Draft)
            | Loop(when=_keep_refining, max_iterations=5),
            Node.scripted("summarize", fn="dr_summarize", inputs=list[Draft], outputs=Summary),
        ],
    )


def _loop_history_programmatic(recorder: _Recorder) -> Construct:
    """Programmatic surface: the nodes are built and piped one at a time, then
    assembled into a list at "runtime" -- the path an LLM-driven or config-driven
    builder takes. Same IR, a different route to it."""
    _register_loop_bodies(recorder)
    seed = Node(name="seed", mode="scripted", scripted_fn="dr_seed", outputs=Draft)
    refine = Node(name="refine", mode="scripted", scripted_fn="dr_refine", inputs=Draft, outputs=Draft)
    refine = refine | Loop(when=_keep_refining, max_iterations=5)
    summarize = Node(
        name="summarize",
        mode="scripted",
        scripted_fn="dr_summarize",
        inputs=list[Draft],
        outputs=Summary,
    )
    assembled: list[Any] = []
    for item in (seed, refine, summarize):
        assembled.append(item)
    return Construct("la3a4-programmatic", nodes=assembled)


class TestLoopHistoryReachesASingleTypeConsumer:
    """neograph-la3a4: ``inputs=list[T]`` after a ``Loop`` producer of ``T``."""

    @pytest.mark.parametrize(
        "build",
        [_loop_history_declarative, _loop_history_programmatic],
        ids=["declarative", "programmatic"],
    )
    def test_all_in_scope_delivers_the_full_history_when_declared_as_a_single_type(self, build):
        """The consumer receives every iteration's Draft, in iteration order.

        This is the documented ``all_in_scope`` projection -- no new storage, no
        new mechanism, just a read-time view of the append-list the Loop already
        writes. The consumer declares the SINGLE-TYPE form (``inputs=list[Draft]``,
        not ``inputs={"refine": list[Draft]}``), which the validator accepts via
        ``_loop_aware_compatible`` and the resolver then fails to address.
        """
        recorder = _Recorder()
        pipeline = build(recorder)

        result = run(compile(pipeline, **build_test_compile_kwargs()), input={"node_id": "la3a4"})

        assert [d.iteration for d in result["refine"]] == EXPECTED_HISTORY, (
            "precondition: the Loop must have produced a 3-element history for this test to mean anything"
        )
        assert result["summarize"].iterations_seen == EXPECTED_HISTORY, (
            "neograph-la3a4: the consumer declared inputs=list[Draft] after a Loop producer of Draft. "
            "Construct() accepted it (_validation_inputs uses _loop_aware_compatible) but "
            "resolve_single_type_source compares with plain _types_compatible over "
            "(field, effective_type, item) triples that DISCARD Producer.is_loop -- so nothing was "
            f"stamped and the body was handed {recorder.received[0]!r} on a green run. "
            "One predicate, or the validator must refuse what the resolver cannot address."
        )

    def test_the_decorator_dict_form_is_the_parity_reference(self):
        """PARITY REFERENCE, green today -- and the asymmetry is the bug.

        A ``refine: list[Draft]`` parameter decorates into
        ``inputs={"refine": list[Draft]}``, which is dict-form: the read is
        addressed BY NAME and never goes near the single-type resolver. Same
        pipeline, same projection, same expected value -- so the two red cases
        above are a three-surface parity gap, not a missing feature.

        Pinned independently by ``test_loop.py::TestLoopScopeProjections``; kept
        here so the comparison this file rests on is visible beside it.
        """

        @node(outputs=Draft)
        def seed() -> Draft:
            return Draft(content="v0")

        @node(outputs=Draft, loop_when=_keep_refining, max_iterations=5)
        def refine(seed: Draft) -> Draft:
            return _refine(seed, None)

        @node(outputs=Summary)
        def summarize(refine: list[Draft]) -> Summary:
            return Summary(iterations_seen=[d.iteration for d in refine])

        pipeline = construct_from_functions("la3a4-decorator", [seed, refine, summarize])

        result = run(compile(pipeline, **build_test_compile_kwargs()), input={"node_id": "la3a4-dec"})

        assert result["summarize"].iterations_seen == EXPECTED_HISTORY


# ═══════════════════════════════════════════════════════════════════════════
# Part 2 -- neograph-z2ayo (N2), and the neograph-q63q9 idiom
#
# The expected behaviour after step 1 is a REFUSAL, decided on evidence: stamping
# the pre-branch producer instead would silently deliver a STALE value on BOTH
# arms of the ForwardConstruct if/else idiom. The author's escape hatch is
# `input_from`, so the refusal has to point there.
# ═══════════════════════════════════════════════════════════════════════════


def _shadowed_post_join_programmatic(recorder: _Recorder) -> Construct:
    """Programmatic surface: ``pre(Token) -> branch(Token, Token) -> after(Token)``.

    The branch sentinel is built directly because a branch arm has no declarative
    spelling (``tests/test_branch_arm_noniter_walks.py`` sets the precedent). The
    condition routes to the TRUE arm, so the false arm never runs.
    """
    register_scripted("dr_pre", _emit_token("PRE-BRANCH", take_true=True))
    register_scripted("dr_true", _emit_token("TRUE-ARM"))
    register_scripted("dr_false", _emit_token("FALSE-ARM"))
    register_scripted("dr_after", recorder.echo)

    pre = Node.scripted("pre", fn="dr_pre", outputs=Token)
    true_arm = Node.scripted("true_arm", fn="dr_true", inputs=Token, outputs=Token)
    false_arm = Node.scripted("false_arm", fn="dr_false", inputs=Token, outputs=Token)
    after = Node.scripted("after", fn="dr_after", inputs=Token, outputs=Echo)
    condition = _ConditionSpec(
        source_node=pre,
        attr_chain=["take_true"],
        op_fn=lambda value, _threshold: bool(value),
        op_str="route",
        threshold=None,
    )
    meta = _BranchMeta(
        condition_spec=condition,
        true_arm_nodes=[true_arm],
        false_arm_nodes=[false_arm],
    )
    return Construct("z2ayo-programmatic", nodes=[pre, _BranchNode(meta, 0), after])


def _shadowed_post_join_forward(recorder: _Recorder) -> Construct:
    """ForwardConstruct surface: the same shape through ``forward()`` tracing.

    The condition is a COMPARISON, not a truthiness test -- see ``Token``.
    """
    register_scripted("dr_pre", _emit_token("PRE-BRANCH", score=1.0))
    register_scripted("dr_true", _emit_token("TRUE-ARM"))
    register_scripted("dr_false", _emit_token("FALSE-ARM"))
    register_scripted("dr_after", recorder.echo)

    class ShadowedPostJoin(ForwardConstruct):
        pre = Node.scripted("pre", fn="dr_pre", outputs=Token)
        true_arm = Node.scripted("true_arm", fn="dr_true", inputs=Token, outputs=Token)
        false_arm = Node.scripted("false_arm", fn="dr_false", inputs=Token, outputs=Token)
        after = Node.scripted("after", fn="dr_after", inputs=Token, outputs=Echo)

        def forward(self, topic):
            produced = self.pre(topic)
            chosen = self.true_arm(produced) if produced.score > 0.5 else self.false_arm(produced)
            return self.after(chosen)

    return ShadowedPostJoin()


class TestPostJoinReadShadowedByABranchArm:
    """neograph-z2ayo (N2): a post-join single-type read that a branch arm shadows."""

    @pytest.mark.parametrize(
        "build",
        [_shadowed_post_join_programmatic, _shadowed_post_join_forward],
        ids=["programmatic", "forward"],
    )
    def test_construct_refuses_and_points_the_author_at_input_from(self, build):
        """Refused at ``Construct()``, naming ``input_from`` as the way to say
        which producer is meant.

        The validator accepts this today (``pre`` is visible after the join and
        satisfies ``Token``) while ``_stamp_single_type_sources`` extends its
        visible set with BOTH arms and takes ``matches[-1]`` -- the FALSE arm.
        Two candidate sets, two answers, no report.

        Stamping ``pre`` instead is NOT the fix and this test must not be
        weakened into accepting it: the same walk serves the if/else idiom below,
        where ``pre``'s value is stale on BOTH arms. Refusal is the step-1
        decision; ``neograph-q63q9`` is what later turns the legitimate
        every-arm-produces-it case into a ``LastPresent`` phi instead.

        When no refusal comes, the test RUNS the pipeline so the failure reports
        the silent wrong answer rather than a bare "DID NOT RAISE".
        """
        recorder = _Recorder()
        try:
            pipeline = build(recorder)
        except ConstructError as exc:
            message = str(exc)
            assert "input_from" in message, (
                "the refusal must name the escape hatch that lets the author SAY which producer is "
                f"meant -- input_from. Got: {message}"
            )
            assert "after" in message, f"the refusal must name the read it refused. Got: {message}"
            return

        result = run(compile(pipeline, **build_test_compile_kwargs()), input={"node_id": "z2ayo"})

        pytest.fail(
            "neograph-z2ayo: Construct() accepted a post-join read that a branch arm shadows, and "
            f"stamped it {recorder.received and type(recorder.received[0]).__name__ or 'nothing'}-shaped. "
            f"The TRUE arm ran (state has {sorted(result)!r}) but 'after' was handed "
            f"{recorder.received[0]!r} and returned {result['after']!r} -- a green run with a missing "
            "value. Expected: ConstructError at Construct() pointing at input_from."
        )


def _if_else_idiom(recorder: _Recorder) -> Construct:
    """The ``neograph-q63q9`` idiom: every arm produces the joined type, and
    NOTHING before the branch does."""
    register_scripted("dr_check", _emit_token("CHECK", score=1.0))
    register_scripted("dr_true", lambda _i, _c: Echo(saw="TRUE-ARM"))
    register_scripted("dr_false", lambda _i, _c: Echo(saw="FALSE-ARM"))
    register_scripted("dr_after", lambda i, c: Token(label=recorder.echo(i, c).saw))

    class IfElseIdiom(ForwardConstruct):
        check = Node.scripted("check", fn="dr_check", outputs=Token)
        true_arm = Node.scripted("true_arm", fn="dr_true", inputs=Token, outputs=Echo)
        false_arm = Node.scripted("false_arm", fn="dr_false", inputs=Token, outputs=Echo)
        after = Node.scripted("after", fn="dr_after", inputs=Echo, outputs=Token)

        def forward(self, topic):
            checked = self.check(topic)
            chosen = self.true_arm(checked) if checked.score > 0.5 else self.false_arm(checked)
            return self.after(chosen)

    return IfElseIdiom()


class TestTheIfElseIdiomIsRefusedWithTrueAdvice:
    """The reviewer's idiom: ``out = self.a(c) if cond else self.b(c); self.after(out)``.

    TODAY it is refused -- correctly, in the sense that no single producer is
    reachable on every path -- but with advice the author has already followed.

    THE CORRECT LONG-TERM BEHAVIOUR IS ``neograph-q63q9``: when EVERY arm produces
    a compatible value, the read is stamped ``LastPresent`` over the arms' last
    producers and the runtime takes the first present rung, so the idiom delivers
    the TAKEN arm's value on both arms. That ticket -- not this step -- is what
    unrefuses this shape. Step 1 owns only the message: a refusal that tells the
    author to do the thing they did is not a usable refusal.
    """

    def test_the_refusal_does_not_advise_what_the_author_already_did(self):
        """Every arm DOES produce a compatible value here, so the hint is false.

        ``_build_no_producer_error``/``_suggest_hint`` re-probe the producers with
        their own predicate instead of rendering the resolver's
        ``Unresolved.candidates``; step 1 migrates them, which is where both
        halves of this assertion come from.
        """
        recorder = _Recorder()

        with pytest.raises(ConstructError) as exc_info:
            _if_else_idiom(recorder)

        message = str(exc_info.value)
        assert "have every arm produce a compatible value" not in message, (
            "the refusal advises producing a compatible value on every arm -- which is exactly the "
            "program that was written. Both arms produce Echo. Advice that restates what the author "
            f"did tells them nothing about what to change. Got: {message}"
        )
        assert "true_arm" in message and "false_arm" in message, (
            "a refusal must say WHERE the compatible values are, so the author can name one with "
            f"input_from. Neither arm producer is mentioned. Got: {message}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Part 3 -- neograph-yi9t5 (N3): a ported sub-construct at position 0 of a
# ported parent.
#
# resolve_single_type_source's rule 3 already says the enclosing construct's own
# port is a candidate when no declared producer matches. port_source_field has no
# such rung, so the child's port is answered by "nothing" instead of "the parent's
# port" -- and the parent's port is holding the value the child was placed to read.
# ═══════════════════════════════════════════════════════════════════════════


def _child_first_in_a_ported_parent(recorder: _Recorder) -> Construct:
    """``root[seed -> parent[child[inner]]]``, ``parent`` and ``child`` both ported."""
    register_scripted("dr_seed", _emit_token("FROM-PARENT-PORT"))
    register_scripted("dr_inner", recorder.echo)

    child = Construct(
        "child",
        input=Token,
        output=Echo,
        nodes=[Node.scripted("inner", fn="dr_inner", inputs=Token, outputs=Echo)],
    )
    parent = Construct("parent", input=Token, output=Echo, nodes=[child])
    return Construct(
        "yi9t5-root",
        nodes=[Node.scripted("seed", fn="dr_seed", outputs=Token), parent],
    )


class TestPortedChildAtPositionZeroReadsTheParentPort:
    """neograph-yi9t5 (N3)."""

    def test_the_inner_body_receives_the_parents_port_value(self):
        """``child`` is first in ``parent``, so no peer precedes it -- the only
        value that can feed its port is the parent's OWN port, which the parent
        already resolved to ``seed``.
        """
        recorder = _Recorder()
        root = _child_first_in_a_ported_parent(recorder)

        result = run(compile(root, **build_test_compile_kwargs()), input={"node_id": "yi9t5"})

        assert result["parent"].saw == "FROM-PARENT-PORT", (
            "neograph-yi9t5: a ported sub-construct at position 0 of a ported parent. "
            "stamp_sub_construct_ports -> port_source_field sees only PRECEDING producers and has no "
            "Port() fallback (resolve_single_type_source's rule 3 has one), so port_source stayed None "
            f"and the inner body was handed {recorder.received[0]!r} while the parent's own port held "
            "the value. Nothing was reported."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Part 4 -- neograph-chunx (N8): the flat both-arm walk.
#
# Filed as "reported by the elegance review; not yet run-verified -- the step must
# reproduce it first". It reproduces: the stamp below is Peer(true_arm) and the
# false-arm run delivers None.
# ═══════════════════════════════════════════════════════════════════════════


def _sub_construct_in_the_false_arm(recorder: _Recorder) -> Construct:
    """``seed -> branch(true: true_arm produces Token | false: sub[inner])``.

    ``seed`` routes to the FALSE arm. Both ``seed`` and ``true_arm`` produce a
    ``Token``, so an arm-SCOPED walk resolves the sub-construct's port to ``seed``
    and a flat both-arm walk resolves it to ``true_arm`` -- a field that is
    unbound on the path actually taken.
    """
    register_scripted("dr_seed", _emit_token("PRE-BRANCH", take_true=False))
    register_scripted("dr_true", _emit_token("TRUE-ARM"))
    register_scripted("dr_inner", recorder.echo)

    seed = Node.scripted("seed", fn="dr_seed", outputs=Token)
    true_arm = Node.scripted("true_arm", fn="dr_true", inputs=Token, outputs=Token)
    sub = Construct(
        "sub",
        input=Token,
        output=Echo,
        nodes=[Node.scripted("inner", fn="dr_inner", inputs=Token, outputs=Echo)],
    )
    condition = _ConditionSpec(
        source_node=seed,
        attr_chain=["take_true"],
        op_fn=lambda value, _threshold: bool(value),
        op_str="route",
        threshold=None,
    )
    meta = _BranchMeta(condition_spec=condition, true_arm_nodes=[true_arm], false_arm_nodes=[sub])
    return Construct("chunx-parent", nodes=[seed, _BranchNode(meta, 0)])


class TestFalseArmSubConstructPortIsArmScoped:
    """neograph-chunx (N8)."""

    def test_the_false_arm_child_reads_the_pre_branch_producer_not_the_true_arm(self):
        """A sub-construct in the FALSE arm must never be fed from the TRUE arm.

        This is the cross-arm read the validator already refuses for a Node; the
        port question answers it with a different walk, so the sub-construct gets
        what the Node could not.
        """
        recorder = _Recorder()
        parent = _sub_construct_in_the_false_arm(recorder)

        result = run(compile(parent, **build_test_compile_kwargs()), input={"node_id": "chunx"})

        assert "true_arm" not in result, (
            "precondition: the FALSE arm must be the path taken for this test to mean anything"
        )
        assert result["sub"].saw == "PRE-BRANCH", (
            "neograph-chunx: stamp_sub_construct_ports builds its `preceding` list with "
            "iter_item_slots, which yields true-arm items and then false-arm items into ONE flat "
            "list -- so a false-arm sub-construct is stamped with the LAST compatible producer "
            "across BOTH arms. The stamp was the true-arm producer; on the path actually taken that "
            f"field is unbound, so the inner body was handed {recorder.received[0]!r}. "
            "The port question must use the same arm-SCOPED candidate set the validator uses."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Part 5 -- neograph-x24iw (N9): a Construct reused in two parents.
#
# Also filed as "reported, not yet run-verified". It reproduces: the in-place
# assignment plus the never-overwrite skip means parent B's child reads parent A's
# field name, which does not exist in B.
# ═══════════════════════════════════════════════════════════════════════════


def _two_parents_sharing_one_child(recorder: _Recorder) -> tuple[Construct, Construct]:
    """One ``Construct`` OBJECT placed in two parents, each with its own feeder.

    Reuse is the point: a sub-pipeline defined once and placed in two flows is
    ordinary composition, and the two parents are built in order so A's answer is
    the one that could be retained.
    """
    register_scripted("dr_alpha", _emit_token("FROM-ALPHA"))
    register_scripted("dr_beta", _emit_token("FROM-BETA"))
    register_scripted("dr_inner", recorder.echo)

    child = Construct(
        "child",
        input=Token,
        output=Echo,
        nodes=[Node.scripted("inner", fn="dr_inner", inputs=Token, outputs=Echo)],
    )
    parent_a = Construct(
        "parent-a",
        nodes=[Node.scripted("alpha", fn="dr_alpha", outputs=Token), child],
    )
    parent_b = Construct(
        "parent-b",
        nodes=[Node.scripted("beta", fn="dr_beta", outputs=Token), child],
    )
    return parent_a, parent_b


class TestReusedConstructResolvesItsOwnPortPerParent:
    """neograph-x24iw (N9)."""

    def test_each_parent_feeds_the_shared_child_from_its_own_producer(self):
        """Both parents run; each child instance receives ITS parent's value.

        ``stamp_sub_construct_ports`` assigns ``item.port_source`` IN PLACE and
        skips an item that already has one, so the second parent's placement is a
        no-op and its child reads a field name that exists only in the first
        parent. Copy-not-mutate (``model_copy``) is the discipline every other
        stamp in ``_ir_normalize`` follows.
        """
        recorder = _Recorder()
        parent_a, parent_b = _two_parents_sharing_one_child(recorder)

        result_a = run(compile(parent_a, **build_test_compile_kwargs()), input={"node_id": "x24iw-a"})
        result_b = run(compile(parent_b, **build_test_compile_kwargs()), input={"node_id": "x24iw-b"})

        assert result_a["child"].saw == "FROM-ALPHA", (
            f"the FIRST parent's child must read its own producer. Got {result_a['child']!r}"
        )
        assert result_b["child"].saw == "FROM-BETA", (
            "neograph-x24iw: the same Construct object placed in two parents keeps the FIRST "
            "parent's port_source (assigned in place at stamp_sub_construct_ports, skipped when "
            "already set). Parent B's child read parent A's field name, which does not exist in B, "
            f"so the inner body was handed {recorder.received[-1]!r} and the port was silently "
            "omitted. Each parent must resolve its own port."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Part 6 -- neograph-3mqw4: ``input_from`` is an AUTHORED name, and assembly
# used to take it on trust.
#
# The resolver short-circuited on it and stamped Peer(PortRef.parse(spelling))
# whatever the spelling said, citing a type check in _validation_inputs that did
# not exist. So a misspelling silently DISPLACED a compatible producer: the field
# it named is written by nobody, and the body was handed None on a green run.
#
# The scan for this step found it was the last authored reference in the IR that
# nothing verified -- output_from, Each.over, bound_args, Portal route=, carried=,
# context= and string Loop conditions all have a named checker.
# ═══════════════════════════════════════════════════════════════════════════


def _input_from_names_nothing(recorder: _Recorder) -> Construct:
    """``producer(Token) -> consumer(inputs=Token, input_from='nosuch')``."""
    register_scripted("dr_if_producer", _emit_token("REAL-PRODUCER"))
    register_scripted("dr_if_consumer", recorder.echo)

    return Construct(
        "mqw4-nonexistent",
        nodes=[
            Node.scripted("producer", fn="dr_if_producer", outputs=Token),
            Node(
                name="consumer",
                mode="scripted",
                scripted_fn="dr_if_consumer",
                inputs=Token,
                outputs=Echo,
                input_from="nosuch",
            ),
        ],
    )


def _input_from_names_an_incompatible_producer(recorder: _Recorder) -> Construct:
    """``teller(Echo) -> token_maker(Token) -> consumer(inputs=Token, input_from='teller')``."""
    register_scripted("dr_if_echo", lambda _i, _c: Echo(saw="WRONG-TYPE"))
    register_scripted("dr_if_token", _emit_token("RIGHT-TYPE"))
    register_scripted("dr_if_consumer2", recorder.echo)

    return Construct(
        "mqw4-incompatible",
        nodes=[
            Node.scripted("teller", fn="dr_if_echo", outputs=Echo),
            Node.scripted("token_maker", fn="dr_if_token", outputs=Token),
            Node(
                name="consumer",
                mode="scripted",
                scripted_fn="dr_if_consumer2",
                inputs=Token,
                outputs=Echo,
                input_from="teller",
            ),
        ],
    )


class TestInputFromIsResolvedNotTrusted:
    """neograph-3mqw4."""

    def test_a_name_that_matches_no_producer_is_refused(self):
        """The repro, as a refusal.

        Before: assembles, compiles, runs green, ``input_sources`` holds
        ``Peer(PortRef(member='nosuch'))``, and the body receives ``None`` while
        ``producer`` sits there producing exactly the declared type.
        """
        recorder = _Recorder()
        try:
            pipeline = _input_from_names_nothing(recorder)
        except ConstructError as exc:
            message = str(exc)
            assert "input_from" in message and "nosuch" in message, (
                f"the refusal must quote the name it could not resolve. Got: {message}"
            )
            assert "producer" in message, (
                f"the refusal must show what IS available, so the author can see the misspelling. Got: {message}"
            )
            return

        result = run(compile(pipeline, **build_test_compile_kwargs()), input={"node_id": "mqw4"})
        pytest.fail(
            "neograph-3mqw4: input_from='nosuch' assembled. The consumer was handed "
            f"{recorder.received[0]!r} and returned {result['consumer']!r} while 'producer' produced a "
            "compatible Token. Expected: ConstructError at Construct()."
        )

    def test_a_name_whose_producer_has_the_wrong_type_is_refused(self):
        """Naming a REAL member is not enough; it must produce the declared type.

        ``teller`` exists and is visible, so an existence-only check would accept
        this and hand the body an ``Echo`` where it declared ``Token`` -- or, since
        the field holds the wrong shape, whatever the runtime made of it.
        """
        recorder = _Recorder()
        with pytest.raises(ConstructError) as exc_info:
            _input_from_names_an_incompatible_producer(recorder)

        message = str(exc_info.value)
        assert "teller" in message, f"the refusal must name the producer it rejected. Got: {message}"
        assert "Echo" in message or "Token" in message, (
            f"the refusal must show the type mismatch it found. Got: {message}"
        )

    def test_input_from_can_name_a_branch_arm_producer(self):
        """The escape hatch step 1's refusal ADVISES must actually exist.

        Step 1 refuses a post-join read that a branch arm shadows and tells the
        author to name the producer with ``input_from``. If ``input_from`` resolved
        only against the VISIBLE set, an arm producer would be unnameable and that
        advice would be false -- the same defect class as the hint it replaced.

        Naming an arm is the author overriding an inference the resolver correctly
        declines to make. The TRUE arm runs here, so the value arrives; when a named
        arm does NOT run the field is absent, which step 9 turns into a loud failure
        rather than a silent None.
        """
        recorder = _Recorder()
        register_scripted("dr_arm_pre", _emit_token("PRE-BRANCH", take_true=True))
        register_scripted("dr_arm_true", _emit_token("TRUE-ARM"))
        register_scripted("dr_arm_false", _emit_token("FALSE-ARM"))
        register_scripted("dr_arm_after", recorder.echo)

        pre = Node.scripted("pre", fn="dr_arm_pre", outputs=Token)
        true_arm = Node.scripted("true_arm", fn="dr_arm_true", inputs=Token, outputs=Token)
        false_arm = Node.scripted("false_arm", fn="dr_arm_false", inputs=Token, outputs=Token)
        after = Node(
            name="after",
            mode="scripted",
            scripted_fn="dr_arm_after",
            inputs=Token,
            outputs=Echo,
            input_from="true_arm",
        )
        condition = _ConditionSpec(
            source_node=pre,
            attr_chain=["take_true"],
            op_fn=lambda value, _threshold: bool(value),
            op_str="route",
            threshold=None,
        )
        meta = _BranchMeta(condition_spec=condition, true_arm_nodes=[true_arm], false_arm_nodes=[false_arm])
        pipeline = Construct("mqw4-arm-named", nodes=[pre, _BranchNode(meta, 0), after])

        result = run(compile(pipeline, **build_test_compile_kwargs()), input={"node_id": "mqw4-arm"})

        assert result["after"] == Echo(saw="TRUE-ARM"), (
            "input_from must be able to name a branch-arm producer -- it is the only spelling that "
            f"can, and step 1's refusal advertises it. Got {result['after']!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Part 7 -- neograph-mkeul: three input shapes the validator waved through
# "deferring to runtime", where the runtime then had nothing to defer TO.
#
# `inputs=dict`, `inputs=dict[str, X]` with no dict-typed producer, and a
# non-class non-generic annotation each returned early from _check_item_input.
# The resolver stamped nothing, the runtime isinstance filter matched nothing,
# and the body received None on a green run.
#
# The legitimate cases share the spelling and must keep working: a producer that
# writes `dict[str, X]` (an Each-modified node) IS satisfiable by both `dict` and
# `dict[str, X]`, which is the documented way to consume a whole fan-out.
# ═══════════════════════════════════════════════════════════════════════════


class TestUnresolvableGenericShapesAreRefused:
    """neograph-mkeul. Each shape is refused, or resolves where it truly can."""

    @staticmethod
    def _pipeline_with(input_spec: Any, recorder: _Recorder) -> Construct:
        """``maker(Token) -> reader(inputs=<input_spec>)``: nothing writes a dict."""
        register_scripted("dr_mk_maker", _emit_token("A-MODEL-NOT-A-DICT"))
        register_scripted("dr_mk_reader", recorder.echo)
        return Construct(
            f"mkeul-{getattr(input_spec, '__name__', str(input_spec))}",
            nodes=[
                Node.scripted("maker", fn="dr_mk_maker", outputs=Token),
                Node.scripted("reader", fn="dr_mk_reader", inputs=input_spec, outputs=Echo),
            ],
        )

    @pytest.mark.parametrize(
        ("input_spec", "label"),
        [
            (dict, "bare-dict"),
            (dict[str, Token], "parameterized-dict"),
            (Any, "non-class-annotation"),
        ],
        ids=["bare_dict", "dict_str_X", "Any"],
    )
    def test_a_shape_no_producer_can_satisfy_is_refused(self, input_spec, label):
        """Refused at ``Construct()``, not deferred to a runtime that cannot answer.

        ``maker`` writes a ``Token``. Nothing in the construct writes a mapping, so
        there is no reading of these declarations under which the body gets a value
        -- which is why "defers to runtime" was never a deferral: it was a decision
        to hand the body ``None``.
        """
        recorder = _Recorder()
        try:
            pipeline = self._pipeline_with(input_spec, recorder)
        except ConstructError as exc:
            assert "reader" in str(exc), f"the refusal must name the read it refused. Got: {exc}"
            return

        result = run(compile(pipeline, **build_test_compile_kwargs()), input={"node_id": "mkeul"})
        pytest.fail(
            f"neograph-mkeul ({label}): inputs={input_spec!r} assembled and ran green. The body was "
            f"handed {recorder.received[0]!r} and returned {result['reader']!r}. Expected a "
            "ConstructError at Construct()."
        )

    def test_a_bare_dict_read_of_a_fanned_producer_still_resolves(self):
        """The legitimate twin, which must NOT be caught by the refusal.

        An ``Each``-modified producer writes ``dict[str, X]``, and consuming the whole
        fan with ``inputs=dict`` is the documented spelling for it -- so the shape is
        only unresolvable when no producer writes a mapping. Refusing the shape
        itself, rather than the absence of a producer for it, would break this.
        """
        recorder = _Recorder()
        register_scripted("dr_mk_seed", lambda _i, _c: Bag(items=[Token(label="x"), Token(label="y")]))
        register_scripted("dr_mk_fan", lambda item, _c: Echo(saw=f"fanned-{item.label}"))
        register_scripted("dr_mk_whole", recorder.echo)

        seed = Node.scripted("seed", fn="dr_mk_seed", outputs=Bag)
        fan = Node.scripted("fan", fn="dr_mk_fan", inputs=Token, outputs=Echo) | Each(over="seed.items", key="label")
        whole = Node.scripted("whole", fn="dr_mk_whole", inputs=dict, outputs=Echo)
        pipeline = Construct("mkeul-legit", nodes=[seed, fan, whole])

        run(compile(pipeline, **build_test_compile_kwargs()), input={"node_id": "mkeul-ok"})

        received = recorder.received[0]
        assert isinstance(received, dict) and set(received) == {"x", "y"}, (
            f"a bare dict read of an Each producer must receive the whole fan; got {received!r}"
        )
        assert {k: v.saw for k, v in received.items()} == {"x": "fanned-x", "y": "fanned-y"}, (
            f"the fanned results must arrive keyed by each.key; got {received!r}"
        )
