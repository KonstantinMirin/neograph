"""Example 15: Loop Refinement -- iterative improvement until quality met.

Scenario: An essay writing pipeline where:
  1. draft: LLM writes an initial draft
  2. review: LLM scores the draft (0-1 quality score)
  3. revise: LLM improves the draft using review feedback
  4. Steps 2-3 repeat until quality >= 0.8 or max 5 iterations

This demonstrates the Loop modifier -- neograph's support for cyclical
graphs. The loop compiles to a LangGraph conditional back-edge.

Demonstrates:
  - Loop modifier on @node: loop_when= + max_iterations=
  - Self-loop (single node refines its own output)
  - Multi-node loop body (review → revise cycle)
  - Loop inside a sub-construct
  - Each + Loop composition (per-item refinement)

Run:
    python examples/15_loop_refinement.py
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import (
    Construct,
    Node,
    Tool,
    compile,
    configure_llm,
    construct_from_functions,
    node,
    register_scripted,
    run,
)
# from neograph import Loop  # 0.3.0 — not yet implemented


# -- Schemas ------------------------------------------------------------------

class Topic(BaseModel, frozen=True):
    text: str


class Draft(BaseModel, frozen=True):
    content: str
    iteration: int = 0


class ReviewResult(BaseModel, frozen=True):
    score: float
    feedback: str


class Essay(BaseModel, frozen=True):
    content: str
    final_score: float
    iterations: int


# -- Pattern 1: Self-loop (single node refines its own output) ----------------
#
# The simplest loop: a node takes a Draft, improves it, returns a Draft.
# Loops until the review score >= 0.8 or max 5 iterations.
#
# Graph:
#   start → draft → refine ⟲ (loop_when: score < 0.8) → end
#
# @node(
#     outputs=Draft,
#     mode="think",
#     model="reason",
#     prompt="refine",
#     loop_when=lambda draft: draft.iteration < 3,  # simplified condition
#     max_iterations=5,
# )
# def refine(draft: Draft) -> Draft: ...


# -- Pattern 2: Multi-node loop (review → revise cycle) ----------------------
#
# Two nodes form the loop body. review scores the draft, revise improves it.
# The loop exits when review.score >= 0.8.
#
# Graph:
#   start → draft → review → revise ⟲ (back to review) → finalize → end
#
# @node(outputs=Draft, mode="think", model="fast", prompt="draft")
# def draft(topic: Topic) -> Draft: ...
#
# @node(outputs=ReviewResult, mode="think", model="reason", prompt="review")
# def review(draft: Draft) -> ReviewResult: ...
#
# @node(
#     outputs=Draft,
#     mode="think",
#     model="fast",
#     prompt="revise",
#     loop_when=lambda state: state["review"].score < 0.8,
#     loop_to="review",       # back-edge target
#     max_iterations=5,
# )
# def revise(draft: Draft, review: ReviewResult) -> Draft: ...
#
# @node(outputs=Essay)
# def finalize(draft: Draft, review: ReviewResult) -> Essay:
#     return Essay(content=draft.content, final_score=review.score, iterations=draft.iteration)


# -- Pattern 3: Loop inside a sub-construct -----------------------------------
#
# The review→revise loop lives inside a sub-construct. The parent pipeline
# doesn't know about the internal iteration — it just sees Topic → Essay.
#
# Graph:
#   start → draft → [refine-sub: review ⟲ revise] → publish → end
#
# refine_sub = construct_from_functions(
#     "refine", [review, revise],
#     input=Draft, output=Draft,
# )
# # The Loop is on the sub-construct itself:
# # refine_sub = refine_sub | Loop(when="needs_refinement", max_iterations=5)
#
# pipeline = construct_from_functions("writer", [draft, refine_sub, finalize])


# -- Pattern 4: Each + Loop (per-item refinement) ----------------------------
#
# A collection of claims, each refined independently. Each item loops until
# its own quality score passes.
#
# Graph:
#   start → make_claims → [Each: verify_claim ⟲ (loop per item)] → merge → end
#
# @node(
#     outputs=VerifiedClaim,
#     mode="think",
#     model="reason",
#     prompt="verify",
#     map_over="make_claims.items",
#     map_key="claim_id",
#     loop_when=lambda claim: claim.confidence < 0.9,
#     max_iterations=3,
# )
# def verify_claim(claim: RawClaim) -> VerifiedClaim: ...


# -- Pattern 5: ForwardConstruct with while loop -----------------------------
#
# class WriterPipeline(ForwardConstruct):
#     draft = Node(outputs=Draft, mode="think", prompt="draft", model="fast")
#     review = Node(outputs=ReviewResult, mode="think", prompt="review", model="reason")
#     revise = Node(outputs=Draft, mode="think", prompt="revise", model="fast")
#
#     def forward(self, topic):
#         d = self.draft(topic)
#         for _ in range(5):
#             r = self.review(d)
#             if r.score >= 0.8:
#                 break
#             d = self.revise(d, r)
#         return d


# -- Runnable demo (scripted, no Loop yet) ------------------------------------
# Simulates what the Loop modifier would do, using scripted nodes.

iteration_count = [0]


def scripted_refine_loop(input_data, config):
    """Simulates the review→revise loop with a Python while loop."""
    topic = "microservice authentication"
    draft = f"Draft about {topic}"
    for i in range(5):
        iteration_count[0] += 1
        score = 0.5 + (i * 0.15)  # improves each iteration
        if score >= 0.8:
            return Essay(content=f"Final: {draft}", final_score=score, iterations=i + 1)
    return Essay(content=f"Max iterations: {draft}", final_score=0.7, iterations=5)


register_scripted("refine_loop", scripted_refine_loop)


pipeline = Construct("writer", nodes=[
    Node.scripted("draft", fn="refine_loop", outputs=Essay),
])


if __name__ == "__main__":
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "loop-demo"})

    essay = result["draft"]
    print(f"Essay: {essay.content}")
    print(f"Final score: {essay.final_score}")
    print(f"Iterations: {essay.iterations}")
    print(f"\nThis is a simulation. With the Loop modifier (0.3.0),")
    print(f"the iteration would be expressed declaratively:")
    print(f"  @node(outputs=Draft, loop_when=lambda d: d.score < 0.8, max_iterations=5)")
    print(f"  def refine(draft: Draft) -> Draft: ...")
