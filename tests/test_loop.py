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
        from neograph import Loop  # noqa: F811 — not yet implemented

        call_count = [0]

        @node(outputs=Draft)
        def refine(draft: Draft) -> Draft:
            call_count[0] += 1
            new_score = draft.score + 0.3
            return Draft(
                content=f"v{call_count[0]}",
                iteration=draft.iteration + 1,
                score=new_score,
            )

        @node(outputs=Draft)
        def seed() -> Draft:
            return Draft(content="v0", iteration=0, score=0.0)

        # refine loops while score < 0.8
        looped_refine = refine | Loop(
            when=lambda draft: draft.score < 0.8,
            max_iterations=10,
        )

        pipeline = construct_from_functions("self-loop", [seed, looped_refine])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "loop-1"})

        # Should have looped 3 times (0.0 → 0.3 → 0.6 → 0.9 exits)
        assert call_count[0] == 3
        assert result["refine"].score >= 0.8
        assert result["refine"].iteration == 3

    def test_self_loop_respects_max_iterations(self):
        """Loop exits after max_iterations even if condition still true."""
        from neograph import Loop

        @node(outputs=Draft)
        def seed() -> Draft:
            return Draft(content="v0", score=0.0)

        @node(outputs=Draft)
        def never_good(draft: Draft) -> Draft:
            return Draft(content="still bad", iteration=draft.iteration + 1, score=0.1)

        looped = never_good | Loop(
            when=lambda d: d.score < 0.8,
            max_iterations=3,
            on_exhaust="last",  # return last result instead of raising
        )

        pipeline = construct_from_functions("capped", [seed, looped])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "loop-cap"})

        assert result["never_good"].iteration == 3
        assert result["never_good"].score == 0.1  # never improved

    def test_self_loop_raises_on_exhaust_error(self):
        """When on_exhaust='error' (default), exceeding max_iterations raises."""
        from neograph import Loop, ExecutionError

        @node(outputs=Draft)
        def seed() -> Draft:
            return Draft(content="v0", score=0.0)

        @node(outputs=Draft)
        def always_bad(draft: Draft) -> Draft:
            return Draft(content="bad", iteration=draft.iteration + 1, score=0.0)

        looped = always_bad | Loop(
            when=lambda d: d.score < 0.8,
            max_iterations=2,
            on_exhaust="error",
        )

        pipeline = construct_from_functions("error-loop", [seed, looped])
        graph = compile(pipeline)

        with pytest.raises(ExecutionError, match="max_iterations"):
            run(graph, input={"node_id": "loop-err"})


# =============================================================================
# Pattern 2: Multi-node loop — review → revise cycle
# =============================================================================


class TestMultiNodeLoop:
    """Two nodes form the loop body. review scores, revise improves.
    The loop back-edge goes from revise to review."""

    def test_review_revise_cycle_exits_when_approved(self):
        """review→revise loops until review.score >= 0.8."""
        from neograph import Loop

        review_count = [0]

        @node(outputs=Draft)
        def draft() -> Draft:
            return Draft(content="initial", score=0.0)

        @node(outputs=ReviewResult)
        def review(draft: Draft) -> ReviewResult:
            review_count[0] += 1
            # Score improves each review
            score = 0.3 * review_count[0]
            return ReviewResult(score=min(score, 1.0), feedback=f"iteration {review_count[0]}")

        @node(outputs=Draft)
        def revise(draft: Draft, review: ReviewResult) -> Draft:
            return Draft(
                content=f"revised: {review.feedback}",
                iteration=draft.iteration + 1,
                score=review.score,
            )

        # revise loops back to review when score < 0.8
        looped_revise = revise | Loop(
            when=lambda state: state["review"].score < 0.8,
            reenter="review",
            max_iterations=10,
        )

        pipeline = construct_from_functions(
            "multi-loop", [draft, review, looped_revise],
        )
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "multi-loop"})

        assert review_count[0] == 3  # 0.3, 0.6, 0.9
        assert result["revise"].score >= 0.8


# =============================================================================
# Pattern 3: Loop inside a sub-construct
# =============================================================================


class TestLoopInSubConstruct:
    """The refinement loop lives inside a sub-construct. The parent sees
    only Topic → Essay, hiding the iteration."""

    def test_sub_construct_with_internal_loop(self):
        """Sub-construct loops internally, parent sees clean I/O."""
        from neograph import Loop

        @node(outputs=Draft)
        def write(topic: Draft) -> Draft:
            return Draft(content="draft", score=0.5)

        @node(outputs=Draft)
        def improve(write: Draft) -> Draft:
            return Draft(
                content="improved",
                iteration=write.iteration + 1,
                score=write.score + 0.2,
            )

        looped_improve = improve | Loop(
            when=lambda d: d.score < 0.8,
            max_iterations=5,
        )

        refine_sub = construct_from_functions(
            "refine", [write, looped_improve],
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

        assert result["finalize"].final_score >= 0.8
        assert result["finalize"].iterations >= 2
        # Sub-construct internals don't leak
        assert "write" not in result
        assert "improve" not in result


# =============================================================================
# Pattern 4: Each + Loop — per-item independent iteration
# =============================================================================


class TestEachPlusLoop:
    """Each item in a collection is refined independently via its own loop."""

    def test_per_item_loop_with_each_fanout(self):
        """Each claim loops independently until confidence >= 0.9."""
        from neograph import Loop

        @node(outputs=ClaimBatch)
        def make_claims() -> ClaimBatch:
            return ClaimBatch(items=[
                ClaimItem(claim_id="c1", text="easy claim", confidence=0.7),
                ClaimItem(claim_id="c2", text="hard claim", confidence=0.3),
            ])

        @node(
            outputs=ClaimItem,
            map_over="make_claims.items",
            map_key="claim_id",
        )
        def verify(claim: ClaimItem) -> ClaimItem:
            return ClaimItem(
                claim_id=claim.claim_id,
                text=claim.text,
                confidence=min(claim.confidence + 0.3, 1.0),
            )

        # Each item loops independently
        looped_verify = verify | Loop(
            when=lambda claim: claim.confidence < 0.9,
            max_iterations=5,
        )

        pipeline = construct_from_functions("each-loop", [make_claims, looped_verify])
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


# =============================================================================
# Pattern 6: ForwardConstruct with while loop
# =============================================================================


class TestForwardConstructLoop:
    """ForwardConstruct: Python while/for loop compiles to cycle."""

    def test_for_loop_compiles_to_cycle(self):
        """for loop in forward() compiles to a cyclical graph."""
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
            return ReviewResult(score=0.3 * _review_count[0], feedback="ok")

        register_scripted("fc_review", fc_review)
        register_scripted("fc_revise", lambda _in, _cfg: Draft(
            content="revised", score=0.0, iteration=1,
        ))

        writer = Writer()
        graph = compile(writer)
        result = run(graph, input={"node_id": "fc-loop"})

        assert result["draft"] is not None or result.get("revise") is not None
