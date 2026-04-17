"""Example 15: Loop Refinement — iterative improvement until quality met.

Scenario: An essay writing pipeline where:
  1. draft: writes an initial draft
  2. review: scores the draft (0-1 quality score)
  3. revise: improves the draft using review feedback
  4. Steps 2-3 repeat until quality >= 0.8 or max 5 iterations

This demonstrates the Loop modifier — neograph's support for cyclical
graphs. The loop compiles to a LangGraph conditional back-edge.

Patterns demonstrated:
  1. Self-loop: single node refines its own output
  2. Multi-node loop body as sub-construct with Loop
  3. Loop inside a sub-construct (parent sees clean I/O)
  4. ForwardConstruct with self.loop() — explicit cycle primitive
  5. Produce + validate with Loop (input != output)

Run:
    python examples/15_loop_refinement.py
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import (
    compile,
    construct_from_functions,
    node,
    run,
)
from neograph import Loop


# -- Schemas ------------------------------------------------------------------

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


# =============================================================================
# Pattern 1: Self-loop — single node refines its own output
# =============================================================================
#
# Graph: start -> seed -> refine -(loop)-> end
#
# refine takes a Draft, improves it, returns a Draft. Loops until
# score >= 0.8 or max 5 iterations.

def demo_self_loop():
    """Self-loop: single node refines its own output."""
    call_count = [0]

    @node(outputs=Draft)
    def seed() -> Draft:
        return Draft(content="Initial draft about microservices", iteration=0, score=0.0)

    @node(
        outputs=Draft,
        loop_when=lambda d: d is None or d.score < 0.8,
        max_iterations=5,
    )
    def refine(draft: Draft) -> Draft:
        """Each iteration improves the score by 0.3."""
        call_count[0] += 1
        new_score = draft.score + 0.3
        return Draft(
            content=f"v{call_count[0]}: refined draft",
            iteration=draft.iteration + 1,
            score=new_score,
        )

    pipeline = construct_from_functions("self-loop", [seed, refine])
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "self-loop-demo"})

    # Loop result is an append-list: all iterations preserved
    history = result["refine"]
    final = history[-1]

    print("=== Pattern 1: Self-loop ===")
    print(f"Iterations: {len(history)}")
    for i, draft in enumerate(history):
        print(f"  [{i+1}] score={draft.score:.1f} content={draft.content!r}")
    print(f"Final: score={final.score:.1f}, iteration={final.iteration}")
    print()


# =============================================================================
# Pattern 2: Multi-node loop body as sub-construct
# =============================================================================
#
# Graph: start -> draft -> [refine: review -> revise] -(loop)-> finalize -> end
#
# The review+revise cycle is a sub-construct that takes Draft in, produces
# Draft out. The Loop modifier on the sub-construct makes it repeat.

def demo_multi_node_loop():
    """Multi-node loop: review + revise cycle as a looping sub-construct."""
    review_count = [0]

    @node(outputs=Draft)
    def draft() -> Draft:
        return Draft(content="Initial essay about authentication", score=0.0)

    @node(outputs=ReviewResult)
    def review(draft: Draft) -> ReviewResult:
        review_count[0] += 1
        score = 0.3 * review_count[0]
        return ReviewResult(
            score=min(score, 1.0),
            feedback=f"Iteration {review_count[0]}: {'needs work' if score < 0.8 else 'approved'}",
        )

    @node(outputs=Draft)
    def revise(draft: Draft, review: ReviewResult) -> Draft:
        return Draft(
            content=f"Revised: {review.feedback}",
            iteration=draft.iteration + 1,
            score=review.score,
        )

    @node(outputs=Essay)
    def finalize(refine: Draft) -> Essay:
        return Essay(content=refine.content, final_score=refine.score, iterations=refine.iteration)

    # Loop body = sub-construct with Draft in, Draft out
    refine = construct_from_functions(
        "refine", [review, revise], input=Draft, output=Draft,
    ) | Loop(when=lambda d: d is None or d.score < 0.8, max_iterations=10)

    pipeline = construct_from_functions("writer", [draft, refine, finalize])
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "multi-loop-demo"})

    # refine field is an append-list from the loop
    history = result["refine"]
    essay = result["finalize"]

    print("=== Pattern 2: Multi-node loop (sub-construct) ===")
    print(f"Review iterations: {review_count[0]}")
    for i, d in enumerate(history):
        print(f"  [{i+1}] score={d.score:.1f} content={d.content!r}")
    print(f"Final essay: score={essay.final_score}, iterations={essay.iterations}")
    print()


# =============================================================================
# Pattern 3: Loop inside a sub-construct (hidden from parent)
# =============================================================================
#
# Graph: start -> seed -> [refine: write -> improve -(loop)->] -> finalize -> end
#
# The refinement loop is internal to the sub-construct. The parent pipeline
# sees only Draft -> Essay.

def demo_loop_in_sub_construct():
    """Loop inside a sub-construct — parent sees clean I/O."""
    improve_count = [0]

    @node(outputs=Draft)
    def write(topic: Draft) -> Draft:
        return Draft(content="First draft", score=0.5)

    @node(
        outputs=Draft,
        loop_when=lambda d: d is None or d.score < 0.8,
        max_iterations=5,
    )
    def improve(write: Draft) -> Draft:
        improve_count[0] += 1
        return Draft(
            content=f"Improved v{improve_count[0]}",
            iteration=write.iteration + 1,
            score=write.score + 0.2,
        )

    refine_sub = construct_from_functions(
        "refine", [write, improve],
        input=Draft, output=Draft,
    )

    @node(outputs=Draft)
    def seed() -> Draft:
        return Draft(content="topic: distributed systems", score=0.0)

    @node(outputs=Essay)
    def finalize(refine: Draft) -> Essay:
        return Essay(content=refine.content, final_score=refine.score, iterations=refine.iteration)

    pipeline = construct_from_functions("writer", [seed, refine_sub, finalize])
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "sub-loop-demo"})

    essay = result["finalize"]

    print("=== Pattern 3: Loop inside sub-construct ===")
    print(f"Internal improve iterations: {improve_count[0]}")
    print(f"Final essay: score={essay.final_score}, iterations={essay.iterations}")
    # Sub-construct internals don't leak to parent result
    print(f"Parent sees 'refine' output: {'refine' in result}")
    print(f"Parent does NOT see 'write': {'write' not in result}")
    print(f"Parent does NOT see 'improve': {'improve' not in result}")
    print()


# =============================================================================
# Pattern 4: ForwardConstruct with self.loop()
# =============================================================================
#
# Graph: start -> draft -> [loop-body: review -> revise] -(loop)-> end
#
# self.loop() is the explicit cycle primitive for ForwardConstruct. Python
# for/while loops in forward() trace the body once but don't produce graph
# cycles (same limitation as torch.jit.trace). self.loop() compiles to a
# sub-construct with Loop modifier — a real back-edge in the graph.

def demo_forward_loop():
    """ForwardConstruct with self.loop() — explicit cycle primitive."""
    from neograph import ForwardConstruct, Node, register_scripted

    review_count = [0]

    register_scripted(
        "fc_draft_ex15",
        lambda _in, _cfg: Draft(content="Initial essay about distributed systems", score=0.0),
    )

    def fc_review(_in, _cfg):
        review_count[0] += 1
        score = 0.3 * review_count[0]
        return ReviewResult(
            score=min(score, 1.0),
            feedback=f"Iteration {review_count[0]}: {'needs work' if score < 0.8 else 'approved'}",
        )

    register_scripted("fc_review_ex15", fc_review)

    def fc_revise(_in, _cfg):
        return Draft(
            content=f"Revised v{review_count[0]}",
            iteration=review_count[0],
            score=0.3 * review_count[0],
        )

    register_scripted("fc_revise_ex15", fc_revise)

    class Writer(ForwardConstruct):
        draft  = Node.scripted("draft", fn="fc_draft_ex15", outputs=Draft)
        review = Node.scripted("review", fn="fc_review_ex15", outputs=ReviewResult)
        revise = Node.scripted("revise", fn="fc_revise_ex15", outputs=Draft)

        def forward(self, topic):
            d = self.draft(topic)
            d = self.loop(
                body=[self.review, self.revise],
                when=lambda r: r is None or r.score < 0.8,
                max_iterations=10,
            )(d)
            return d

    graph = compile(Writer())
    result = run(graph, input={"node_id": "forward-loop-demo"})

    print("=== Pattern 4: ForwardConstruct with self.loop() ===")
    print(f"Review iterations: {review_count[0]}")
    print(f"Draft output: {result.get('draft')}")
    # The loop body is a sub-construct; its output appears under the
    # sub-construct name, not individual node names.
    for key, val in result.items():
        if isinstance(val, list) and val and hasattr(val[0], "score"):
            for i, d in enumerate(val):
                print(f"  [{i+1}] score={d.score:.1f} content={d.content!r}")
    print()


# =============================================================================
# Pattern 5: Produce + Validate with Loop (input != output)
# =============================================================================
#
# Graph: start -> seed -> [produce_validate: produce -> validate] -(loop)-> report -> end
#
# The sub-construct takes ProduceInput and returns Validated — different types.
# The producer fails on the first call (simulating a transient error) and
# succeeds on the second. The validator checks for errors. The Loop condition
# retries until validation passes or max iterations are exhausted.

def demo_produce_validate():
    """Produce + validate with Loop — input != output, retry on failure."""

    class ProduceInput(BaseModel, frozen=True):
        request_id: str
        payload: str

    class ProduceOutput(BaseModel, frozen=True):
        request_id: str
        data: str
        error: str | None = None

    class Validated(BaseModel, frozen=True):
        request_id: str
        data: str
        errors: list[str]

    class Report(BaseModel, frozen=True):
        request_id: str
        data: str
        attempts: int

    produce_calls = [0]

    @node(outputs=ProduceOutput)
    def produce(topic: ProduceInput) -> ProduceOutput:
        """Fails on first call, succeeds on second."""
        produce_calls[0] += 1
        if produce_calls[0] == 1:
            return ProduceOutput(
                request_id=topic.request_id,
                data="",
                error="transient failure: service unavailable",
            )
        return ProduceOutput(
            request_id=topic.request_id,
            data=f"generated data for '{topic.payload}'",
            error=None,
        )

    @node(outputs=Validated)
    def check(produce: ProduceOutput) -> Validated:
        """Check the produced output for errors."""
        errors: list[str] = []
        if produce.error:
            errors.append(produce.error)
        if not produce.data:
            errors.append("empty data")
        return Validated(
            request_id=produce.request_id,
            data=produce.data,
            errors=errors,
        )

    # Sub-construct: ProduceInput -> Validated (different types)
    produce_validate = construct_from_functions(
        "produce_validate", [produce, check],
        input=ProduceInput, output=Validated,
    ) | Loop(
        when=lambda v: v is None or bool(v.errors),
        max_iterations=5,
    )

    @node(outputs=ProduceInput)
    def seed() -> ProduceInput:
        return ProduceInput(request_id="req-42", payload="quarterly report")

    @node(outputs=Report)
    def report(produce_validate: Validated) -> Report:
        return Report(
            request_id=produce_validate.request_id,
            data=produce_validate.data,
            attempts=produce_calls[0],
        )

    pipeline = construct_from_functions(
        "produce-pipeline", [seed, produce_validate, report],
    )
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "produce-validate-demo"})

    history = result["produce_validate"]
    final_report = result["report"]

    print("\n" + "=" * 60)
    print("PRODUCE+VALIDATE LOOP (input != output)")
    print("=" * 60)
    print(f"Producer calls: {produce_calls[0]}")
    for i, v in enumerate(history):
        status = "PASS" if not v.errors else f"FAIL ({', '.join(v.errors)})"
        print(f"  [{i+1}] {status} data={v.data!r}")
    print(f"Final report: request_id={final_report.request_id}, "
          f"data={final_report.data!r}, attempts={final_report.attempts}")
    print()


# =============================================================================

if __name__ == "__main__":
    demo_self_loop()
    demo_multi_node_loop()
    demo_loop_in_sub_construct()
    demo_forward_loop()
    demo_produce_validate()
