"""Tests for Loop modifier — cyclical graphs for iterative refinement.

These tests define the desired API and behavior for the Loop modifier.
They are written BEFORE implementation (TDD red). Each test exercises
a specific loop pattern through compile() + run() end-to-end.

Patterns covered:
  1. Self-loop: single node refines its own output
  2. Multi-node loop: review → revise cycle with back-edge
  3. Max iterations: hard cap with on_exhaust behavior
  4. Loop inside sub-construct: isolated iteration
  5. Each + Loop: per-item independent iteration
  6. ForwardConstruct while loop: Python control flow compiles to cycle
  7. Loop exit condition: predicate on state determines exit
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from neograph import (
    Construct,
    ConstructError,
    Node,
    Each,
    compile,
    construct_from_functions,
    construct_from_module,
    node,
    run,
)
from neograph.factory import register_scripted


# -- Schemas for loop tests ---------------------------------------------------

class Draft(BaseModel, frozen=True):
    content: str
    iteration: int = 0
    score: float = 0.0


class ReviewResult(BaseModel, frozen=True):
    score: float
    feedback: str


class Essay(BaseModel, frozen=True):
    content: str
    final_score: float
    iterations: int


class ClaimItem(BaseModel, frozen=True):
    claim_id: str
    text: str
    confidence: float = 0.0


class ClaimBatch(BaseModel, frozen=True):
    items: list[ClaimItem]


# =============================================================================
# Pattern 1: Self-loop — single node refines its own output
# =============================================================================


class TestSelfLoop:
    """A single node takes Draft, improves it, returns Draft. Loops until
    the score threshold is met or max iterations reached."""

    def test_self_loop_exits_when_condition_met(self):
        """Node loops 3 times, score reaches threshold, exits."""
        call_count = [0]

        @node(
            outputs=Draft,
            loop_when=lambda draft: draft is None or draft.score < 0.8,
            max_iterations=10,
        )
        def refine(seed: Draft) -> Draft:
            call_count[0] += 1
            new_score = seed.score + 0.3
            return Draft(
                content=f"v{call_count[0]}",
                iteration=seed.iteration + 1,
                score=new_score,
            )

        @node(outputs=Draft)
        def seed() -> Draft:
            return Draft(content="v0", iteration=0, score=0.0)

        pipeline = construct_from_functions("self-loop", [seed, refine])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "loop-1"})

        # Should have looped 3 times (0.0 → 0.3 → 0.6 → 0.9 exits)
        assert call_count[0] == 3
        # Loop result is a list (append reducer). Last element is final.
        assert isinstance(result["refine"], list)
        assert result["refine"][-1].score >= 0.8
        assert result["refine"][-1].iteration == 3
        # History is preserved: all iterations available
        assert len(result["refine"]) == 3

    def test_self_loop_respects_max_iterations(self):
        """Loop exits after max_iterations even if condition still true."""
        @node(outputs=Draft)
        def seed() -> Draft:
            return Draft(content="v0", score=0.0)

        @node(
            outputs=Draft,
            loop_when=lambda d: d.score < 0.8,
            max_iterations=3,
            on_exhaust="last",
        )
        def never_good(draft: Draft) -> Draft:
            return Draft(content="still bad", iteration=draft.iteration + 1, score=0.1)

        pipeline = construct_from_functions("capped", [seed, never_good])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "loop-cap"})

        assert result["never_good"][-1].iteration == 3
        assert result["never_good"][-1].score == 0.1  # never improved

    def test_self_loop_raises_on_exhaust_error(self):
        """When on_exhaust='error' (default), exceeding max_iterations raises."""
        from neograph import ExecutionError

        @node(outputs=Draft)
        def seed() -> Draft:
            return Draft(content="v0", score=0.0)

        @node(
            outputs=Draft,
            loop_when=lambda d: d.score < 0.8,
            max_iterations=2,
            on_exhaust="error",
        )
        def always_bad(draft: Draft) -> Draft:
            return Draft(content="bad", iteration=draft.iteration + 1, score=0.0)

        pipeline = construct_from_functions("error-loop", [seed, always_bad])
        graph = compile(pipeline)

        with pytest.raises(ExecutionError, match="max_iterations"):
            run(graph, input={"node_id": "loop-err"})


# =============================================================================
# Pattern 2: Multi-node loop — review → revise cycle
# =============================================================================


class TestMultiNodeLoop:
    """Multi-node loop body expressed as a sub-construct with Loop modifier.
    The review+revise cycle is a sub-construct that loops until approved."""

    def test_review_revise_cycle_as_looping_sub_construct(self):
        """review+revise sub-construct loops until score >= 0.8."""
        from neograph.modifiers import Loop

        review_count = [0]

        @node(outputs=Draft)
        def draft() -> Draft:
            return Draft(content="initial", score=0.0)

        @node(outputs=ReviewResult)
        def review(draft: Draft) -> ReviewResult:
            review_count[0] += 1
            score = 0.3 * review_count[0]
            return ReviewResult(score=min(score, 1.0), feedback=f"iteration {review_count[0]}")

        @node(outputs=Draft)
        def revise(draft: Draft, review: ReviewResult) -> Draft:
            return Draft(
                content=f"revised: {review.feedback}",
                iteration=draft.iteration + 1,
                score=review.score,
            )

        # Loop body as sub-construct: Draft in, Draft out
        refine = construct_from_functions(
            "refine", [review, revise], input=Draft, output=Draft,
        ) | Loop(when=lambda d: d is None or d.score < 0.8, max_iterations=10)

        pipeline = construct_from_functions("multi-loop", [draft, refine])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "multi-loop"})

        assert review_count[0] == 3  # 0.3, 0.6, 0.9
        final = result["refine"][-1]
        assert final.score >= 0.8
        assert final.iteration >= 2


# =============================================================================
# Pattern 3: Loop inside a sub-construct
# =============================================================================


class TestLoopInSubConstruct:
    """The refinement loop lives inside a sub-construct. The parent sees
    only Topic → Essay, hiding the iteration."""

    def test_sub_construct_with_internal_loop(self):
        """Sub-construct loops internally, parent sees clean I/O."""
        @node(outputs=Draft)
        def write(topic: Draft) -> Draft:
            return Draft(content="draft", score=0.5)

        @node(
            outputs=Draft,
            loop_when=lambda d: d is None or d.score < 0.8,
            max_iterations=5,
        )
        def improve(write: Draft) -> Draft:
            return Draft(
                content="improved",
                iteration=write.iteration + 1,
                score=write.score + 0.2,
            )

        refine_sub = construct_from_functions(
            "refine", [write, improve],
            input=Draft, output=Draft,
        )

        # Parent pipeline
        @node(outputs=Draft)
        def seed() -> Draft:
            return Draft(content="topic", score=0.0)

        @node(outputs=Essay)
        def finalize(refine: Draft) -> Essay:
            return Essay(content=refine.content, final_score=refine.score, iterations=refine.iteration)

        pipeline = construct_from_functions("writer", [seed, refine_sub, finalize])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "sub-loop"})

        final = result["finalize"][-1] if isinstance(result["finalize"], list) else result["finalize"]
        assert final.final_score >= 0.8
        assert final.iterations >= 2
        # Sub-construct internals don't leak
        assert "write" not in result
        assert "improve" not in result


# =============================================================================
# Pattern 4: Each + Loop — per-item independent iteration
# =============================================================================


class TestEachPlusLoop:
    """Each item in a collection is refined via a looping sub-construct
    inside an Each fan-out."""

    def test_per_item_loop_with_each_over_looping_sub_construct(self):
        """Each claim gets its own sub-construct that loops internally."""

        @node(outputs=ClaimBatch)
        def make_claims() -> ClaimBatch:
            return ClaimBatch(items=[
                ClaimItem(claim_id="c1", text="easy claim", confidence=0.7),
                ClaimItem(claim_id="c2", text="hard claim", confidence=0.3),
            ])

        # Loop is on the NODE inside the sub-construct, not on the Construct
        @node(
            outputs=ClaimItem,
            loop_when=lambda c: c is None or c.confidence < 0.9,
            max_iterations=5,
        )
        def verify(claim: ClaimItem) -> ClaimItem:
            return ClaimItem(
                claim_id=claim.claim_id,
                text=claim.text,
                confidence=min(claim.confidence + 0.3, 1.0),
            )

        # Sub-construct with internal loop, wrapped by Each
        verify_sub = construct_from_functions(
            "verify", [verify], input=ClaimItem, output=ClaimItem,
        )
        verify_each = verify_sub | Each(over="make_claims.items", key="claim_id")

        pipeline = construct_from_functions("each-loop", [make_claims, verify_each])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "each-loop"})

        verified = result["verify"]
        assert isinstance(verified, dict)
        assert all(v.confidence >= 0.9 for v in verified.values())
        # c1 needed 1 iteration (0.7→1.0), c2 needed 2 (0.3→0.6→0.9)


# =============================================================================
# Pattern 5: Validator catches type mismatch on back-edge
# =============================================================================


class TestLoopValidation:
    """Assembly-time validation for Loop modifier."""

    def test_raises_when_loop_output_incompatible_with_input(self):
        """Self-loop: outputs must be compatible with inputs for back-edge."""
        from neograph import Loop

        class InputType(BaseModel, frozen=True):
            x: str

        class OutputType(BaseModel, frozen=True):
            y: int  # different type — can't loop back

        @node(outputs=OutputType)
        def incompatible(data: InputType) -> OutputType:
            return OutputType(y=1)

        with pytest.raises(ConstructError, match="loop.*compatible"):
            incompatible | Loop(when=lambda x: True, max_iterations=3)

    def test_raises_when_loop_param_ambiguous_type_match(self):
        """When multiple upstreams produce the same type, error on ambiguity."""

        @node(outputs=Draft)
        def seed_a() -> Draft:
            return Draft(content="a")

        @node(outputs=Draft)
        def seed_b() -> Draft:
            return Draft(content="b")

        @node(
            outputs=Draft,
            loop_when=lambda d: d.score < 0.8,
            max_iterations=3,
        )
        def ambiguous(draft: Draft) -> Draft:
            return Draft(content="x", score=1.0)

        with pytest.raises(ConstructError, match="matches multiple upstreams"):
            construct_from_functions("ambiguous", [seed_a, seed_b, ambiguous])

    def test_raises_when_each_and_loop_combined(self):
        """Each + Loop on the same node is restricted."""
        with pytest.raises(ConstructError, match="Each.*Loop.*cannot be combined"):
            @node(
                outputs=ClaimItem,
                map_over="claims.items",
                map_key="claim_id",
                loop_when=lambda c: c.confidence < 0.9,
                max_iterations=5,
            )
            def bad_combo(claim: ClaimItem) -> ClaimItem:
                return claim


# =============================================================================
# Pattern 6: ForwardConstruct with while loop
# =============================================================================


class TestForwardConstructLoop:
    """ForwardConstruct: Python while/for loop compiles to cycle."""

    def test_for_loop_traces_nodes_in_forward(self):
        """ForwardConstruct forward() with for loop traces the nodes.
        NOTE: The tracer does NOT compile for/while to graph cycles (neograph-mwx3).
        It traces the loop body once. Cycle support requires Loop modifier."""
        from neograph import ForwardConstruct

        class Writer(ForwardConstruct):
            draft = Node.scripted("draft", fn="fc_draft", outputs=Draft)
            review = Node.scripted("review", fn="fc_review", outputs=ReviewResult)
            revise = Node.scripted("revise", fn="fc_revise", outputs=Draft)

            def forward(self, topic):
                d = self.draft(topic)
                for _ in range(5):
                    r = self.review(d)
                    if r.score >= 0.8:
                        break
                    d = self.revise(d, r)
                return d

        register_scripted("fc_draft", lambda _in, _cfg: Draft(content="v0", score=0.0))

        _review_count = [0]

        def fc_review(_in, _cfg):
            _review_count[0] += 1
            return ReviewResult(score=0.3 * _review_count[0], feedback=f"feedback-{_review_count[0]}")

        register_scripted("fc_review", fc_review)

        def fc_revise(_in, _cfg):
            return Draft(content="revised", score=0.0, iteration=1)

        register_scripted("fc_revise", fc_revise)

        writer = Writer()
        graph = compile(writer)
        result = run(graph, input={"node_id": "fc-loop"})

        # Known limitation (neograph-mwx3): the tracer does NOT compile
        # for/while to graph cycles. It traces the loop body nodes once.
        # This test verifies the nodes were at least traced and ran.
        assert _review_count[0] >= 1, "review did not run at all"
        assert result.get("draft") is not None, "draft node did not produce output"

    def test_self_loop_cycles_via_explicit_loop_primitive(self):
        """self.loop() compiles to a sub-construct with Loop modifier.
        The review+revise body loops until score >= 0.8."""
        from neograph import ForwardConstruct

        class Writer(ForwardConstruct):
            draft = Node.scripted("draft", fn="fc_draft_loop", outputs=Draft)
            review = Node.scripted("review", fn="fc_review_loop", outputs=ReviewResult)
            revise = Node.scripted("revise", fn="fc_revise_loop", outputs=Draft)

            def forward(self, topic):
                d = self.draft(topic)
                d = self.loop(
                    body=[self.review, self.revise],
                    when=lambda r: r is None or r.score < 0.8,
                    max_iterations=10,
                )(d)
                return d

        register_scripted(
            "fc_draft_loop",
            lambda _in, _cfg: Draft(content="v0", score=0.0),
        )

        _review_count = [0]

        def fc_review_loop(_in, _cfg):
            _review_count[0] += 1
            return ReviewResult(
                score=0.3 * _review_count[0],
                feedback=f"feedback-{_review_count[0]}",
            )

        register_scripted("fc_review_loop", fc_review_loop)

        def fc_revise_loop(_in, _cfg):
            return Draft(content="revised", score=0.3 * _review_count[0], iteration=_review_count[0])

        register_scripted("fc_revise_loop", fc_revise_loop)

        writer = Writer()
        graph = compile(writer)
        result = run(graph, input={"node_id": "fc-loop-prim"})

        # Loop cycled: review ran 3 times (0.3, 0.6, 0.9)
        assert _review_count[0] == 3, (
            f"Expected review to run 3 times, ran {_review_count[0]}"
        )

    def test_single_node_loop_body_via_explicit_loop_primitive(self):
        """self.loop(body=[self.refine], ...) compiles a single-node self-loop
        via ForwardConstruct. The refine node loops until score >= 0.8."""
        from neograph import ForwardConstruct

        class Refiner(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_seed_single", outputs=Draft)
            refine = Node.scripted("refine", fn="fc_refine_single", outputs=Draft)

            def forward(self, topic):
                d = self.seed(topic)
                d = self.loop(
                    body=[self.refine],
                    when=lambda r: r is None or r.score < 0.8,
                    max_iterations=5,
                )(d)
                return d

        register_scripted(
            "fc_seed_single",
            lambda _in, _cfg: Draft(content="seed-v0", score=0.1),
        )

        _refine_count = [0]

        def fc_refine_single(_in, _cfg):
            _refine_count[0] += 1
            new_score = 0.1 + 0.25 * _refine_count[0]
            return Draft(
                content=f"refined-v{_refine_count[0]}",
                iteration=_refine_count[0],
                score=new_score,
            )

        register_scripted("fc_refine_single", fc_refine_single)

        pipeline = Refiner()
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "fc-single-loop"})

        # 0.1 seed → refine1=0.35, refine2=0.60, refine3=0.85 → exits
        assert _refine_count[0] == 3, (
            f"Expected refine to run 3 times, ran {_refine_count[0]}"
        )

    def test_on_exhaust_last_exits_with_last_result(self):
        """self.loop() with on_exhaust='last': loop never meets condition,
        exits at max_iterations with last result (no error)."""
        from neograph import ForwardConstruct

        class NeverDone(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_seed_exhaust", outputs=Draft)
            tweak = Node.scripted("tweak", fn="fc_tweak_exhaust", outputs=Draft)

            def forward(self, topic):
                d = self.seed(topic)
                d = self.loop(
                    body=[self.tweak],
                    when=lambda r: r is None or r.score < 0.99,
                    max_iterations=3,
                    on_exhaust="last",
                )(d)
                return d

        register_scripted(
            "fc_seed_exhaust",
            lambda _in, _cfg: Draft(content="seed-exhaust", score=0.0),
        )

        _tweak_count = [0]

        def fc_tweak_exhaust(_in, _cfg):
            _tweak_count[0] += 1
            return Draft(
                content=f"tweak-v{_tweak_count[0]}",
                iteration=_tweak_count[0],
                score=0.1 * _tweak_count[0],
            )

        register_scripted("fc_tweak_exhaust", fc_tweak_exhaust)

        pipeline = NeverDone()
        graph = compile(pipeline)
        # Should NOT raise — on_exhaust='last' means exit silently
        result = run(graph, input={"node_id": "fc-exhaust-last"})

        # Loop ran exactly max_iterations=3 times (never reached score 0.99)
        assert _tweak_count[0] == 3, (
            f"Expected tweak to run 3 times, ran {_tweak_count[0]}"
        )

    def test_on_exhaust_error_raises_execution_error(self):
        """self.loop() with on_exhaust='error': loop exceeds max_iterations
        and raises ExecutionError."""
        from neograph import ExecutionError, ForwardConstruct

        class AlwaysBad(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_seed_err", outputs=Draft)
            polish = Node.scripted("polish", fn="fc_polish_err", outputs=Draft)

            def forward(self, topic):
                d = self.seed(topic)
                d = self.loop(
                    body=[self.polish],
                    when=lambda r: r is None or r.score < 0.99,
                    max_iterations=2,
                    on_exhaust="error",
                )(d)
                return d

        register_scripted(
            "fc_seed_err",
            lambda _in, _cfg: Draft(content="seed-err", score=0.0),
        )

        register_scripted(
            "fc_polish_err",
            lambda _in, _cfg: Draft(content="polished", score=0.1),
        )

        pipeline = AlwaysBad()
        graph = compile(pipeline)

        with pytest.raises(ExecutionError, match="max_iterations"):
            run(graph, input={"node_id": "fc-exhaust-err"})

    def test_loop_followed_by_branch_composes(self):
        """self.loop() followed by an if/else branch — both features compose.
        The loop refines a draft, then a branch selects the output path.

        Currently fails because the branch tracer puts the loop Construct
        into the arm-node lists and compile_state_model expects Node.outputs
        (plural) but Construct has .output (singular).
        """
        from neograph import ForwardConstruct

        class Confidence(BaseModel, frozen=True):
            score: float

        class HighResult(BaseModel, frozen=True):
            label: str

        class LowResult(BaseModel, frozen=True):
            label: str

        class LoopThenBranch(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_seed_lb", outputs=Draft)
            refine = Node.scripted("refine", fn="fc_refine_lb", outputs=Draft)
            check = Node.scripted("check", fn="fc_check_lb", outputs=Confidence)
            high = Node.scripted("high", fn="fc_high_lb", outputs=HighResult)
            low = Node.scripted("low", fn="fc_low_lb", outputs=LowResult)

            def forward(self, topic):
                d = self.seed(topic)
                d = self.loop(
                    body=[self.refine],
                    when=lambda r: r is None or r.score < 0.5,
                    max_iterations=5,
                )(d)
                c = self.check(d)
                if c.score > 0.7:
                    return self.high(c)
                else:
                    return self.low(c)

        register_scripted(
            "fc_seed_lb",
            lambda _in, _cfg: Draft(content="lb-seed", score=0.0),
        )

        _refine_lb_count = [0]

        def fc_refine_lb(_in, _cfg):
            _refine_lb_count[0] += 1
            return Draft(
                content=f"lb-refined-v{_refine_lb_count[0]}",
                iteration=_refine_lb_count[0],
                score=0.3 * _refine_lb_count[0],
            )

        register_scripted("fc_refine_lb", fc_refine_lb)
        register_scripted(
            "fc_check_lb",
            lambda _in, _cfg: Confidence(score=0.9),
        )
        register_scripted(
            "fc_high_lb",
            lambda _in, _cfg: HighResult(label="high-after-loop"),
        )
        register_scripted(
            "fc_low_lb",
            lambda _in, _cfg: LowResult(label="low-after-loop"),
        )

        pipeline = LoopThenBranch()
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "fc-loop-branch"})

        # refine: 0.0 → 0.3 → 0.6 → exits (score >= 0.5)
        assert _refine_lb_count[0] == 2, (
            f"Expected refine to run 2 times, ran {_refine_lb_count[0]}"
        )
        # check returns 0.9 > 0.7, so high path runs
        assert "high" in result
        assert result["high"].label == "high-after-loop"

    def test_branch_followed_by_loop(self):
        """if/else branch followed by self.loop() — branch first, then loop."""
        from neograph import ForwardConstruct

        class BranchThenLoop(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_seed_bl", outputs=Draft)
            check = Node.scripted("check", fn="fc_check_bl", outputs=Draft)
            boost = Node.scripted("boost", fn="fc_boost_bl", outputs=Draft)
            refine = Node.scripted("refine", fn="fc_refine_bl", outputs=Draft)

            def forward(self, topic):
                d = self.seed(topic)
                c = self.check(d)
                if c.score > 0.5:
                    d = self.boost(c)
                d = self.loop(
                    body=[self.refine],
                    when=lambda r: r is None or r.score < 0.8,
                    max_iterations=5,
                )(d)
                return d

        register_scripted("fc_seed_bl", lambda _in, _cfg: Draft(content="bl-seed", score=0.6))
        register_scripted("fc_check_bl", lambda _in, _cfg: Draft(content="checked", score=0.6))
        register_scripted("fc_boost_bl", lambda _in, _cfg: Draft(content="boosted", score=0.6))

        _refine_bl_count = [0]

        def fc_refine_bl(_in, _cfg):
            _refine_bl_count[0] += 1
            d = _in if isinstance(_in, Draft) else Draft(content="", score=0.0)
            return Draft(content=f"refined-{_refine_bl_count[0]}", score=d.score + 0.15, iteration=_refine_bl_count[0])

        register_scripted("fc_refine_bl", fc_refine_bl)

        pipeline = BranchThenLoop()
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "fc-branch-loop"})

        # Branch should have taken the boost path (0.6 > 0.5)
        # Loop: 0.6 → 0.75 → 0.90 → exits (2 iterations)
        assert _refine_bl_count[0] >= 2, (
            f"Expected refine to run >= 2 times, ran {_refine_bl_count[0]}"
        )

    def test_two_sequential_loops(self):
        """Two self.loop() calls in sequence — both should cycle independently."""
        from neograph import ForwardConstruct

        class TwoLoops(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_seed_2l", outputs=Draft)
            rough = Node.scripted("rough", fn="fc_rough_2l", outputs=Draft)
            polish = Node.scripted("polish", fn="fc_polish_2l", outputs=Draft)

            def forward(self, topic):
                d = self.seed(topic)
                d = self.loop(
                    body=[self.rough],
                    when=lambda r: r is None or r.score < 0.5,
                    max_iterations=5,
                )(d)
                d = self.loop(
                    body=[self.polish],
                    when=lambda r: r is None or r.score < 0.9,
                    max_iterations=5,
                )(d)
                return d

        register_scripted("fc_seed_2l", lambda _in, _cfg: Draft(content="2l-seed", score=0.0))

        _rough_count = [0]
        def fc_rough_2l(_in, _cfg):
            _rough_count[0] += 1
            d = _in if isinstance(_in, Draft) else Draft(content="", score=0.0)
            return Draft(content=f"rough-{_rough_count[0]}", score=d.score + 0.2, iteration=_rough_count[0])

        _polish_count = [0]
        def fc_polish_2l(_in, _cfg):
            _polish_count[0] += 1
            d = _in if isinstance(_in, Draft) else Draft(content="", score=0.0)
            return Draft(content=f"polish-{_polish_count[0]}", score=d.score + 0.15, iteration=_polish_count[0])

        register_scripted("fc_rough_2l", fc_rough_2l)
        register_scripted("fc_polish_2l", fc_polish_2l)

        pipeline = TwoLoops()
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "fc-two-loops"})

        # Rough: 0.0 → 0.2 → 0.4 → 0.6 (3 iterations, exits at 0.6 >= 0.5)
        assert _rough_count[0] == 3, f"rough ran {_rough_count[0]}, expected 3"
        # Polish: 0.6 → 0.75 → 0.90 (2 iterations, exits at 0.90 >= 0.9)
        assert _polish_count[0] == 2, f"polish ran {_polish_count[0]}, expected 2"
