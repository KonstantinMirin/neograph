"""Example 27: ForwardConstruct — imperative agent wiring (branch, loop, fan-out, ensemble, HITL).

`ForwardConstruct` lets you write an agent pipeline as an ordinary Python
`forward()` method: control flow *is* graph topology. An `if` becomes a
conditional edge; `self.loop()` a real back-edge; `self.each()` a fan-out;
`self.ensemble()` an Oracle; `self.interrupt()` a human-in-the-loop gate. The
tracer discovers the graph from one symbolic pass (torch.fx style) and compiles
to the SAME IR the declarative form produces — so `forward()` is pure DX sugar,
never a second runtime.

Six self-contained demos, each runnable end-to-end. The nodes here are
`Node.scripted` fakes so the example is deterministic and keyless — but each one
stands in for a real agent stage. In production you would write, e.g.::

    triage = Node(outputs=Triage, mode='think', prompt='triage', model='fast')
    verify = Node(outputs=Verdict, mode='agent', prompt='verify', model='reason',
                  tools=[Tool('search_web', budget=3)])

and the `forward()` wiring below is byte-identical.

Run (keyless, no network):
    uv run --extra dev python examples/27_forward_agent_wiring.py
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import (
    ForwardConstruct,
    Node,
    compile,
    run,
)

try:  # a real checkpointer for the HITL demo (ships with langgraph)
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:  # pragma: no cover
    MemorySaver = None  # type: ignore[assignment,misc]


# ── shared typed channels ─────────────────────────────────────────────────────


class Triage(BaseModel, frozen=True):
    confidence: float
    topic: str


class Analysis(BaseModel, frozen=True):
    summary: str
    depth: str


class Draft(BaseModel, frozen=True):
    content: str
    score: float = 0.0
    revision: int = 0


class Claim(BaseModel, frozen=True):
    id: str
    text: str


class ClaimBatch(BaseModel, frozen=True):
    claims: list[Claim]


class Verdict(BaseModel, frozen=True):
    id: str
    supported: bool


class Report(BaseModel, frozen=True):
    supported: int
    total: int


# =============================================================================
# Demo 1 — BRANCH: `if` compiles to a conditional edge
# =============================================================================
# A triage stage routes to deep vs. shallow analysis. The tracer re-runs
# forward() with each branch flipped to discover both arms, then lowers the
# `if` to add_conditional_edges. Only proxy-vs-CONSTANT comparisons are legal.


def demo_branch() -> None:
    print("=" * 68)
    print("DEMO 1 — branch: `if` -> conditional edge (deep vs shallow analysis)")
    print("=" * 68)

    def fn_triage(inp, _cfg):
        # In production: a `think` node scoring how hard the topic is.
        return Triage(confidence=0.55, topic="distributed consensus")

    def fn_deep(inp, _cfg):
        return Analysis(summary=f"deep dive on {inp.topic}", depth="deep")

    def fn_shallow(inp, _cfg):
        return Analysis(summary=f"quick scan of {inp.topic}", depth="shallow")

    class Router(ForwardConstruct):
        triage = Node.scripted("triage", fn="fn_triage", outputs=Triage)
        deep = Node.scripted("deep", fn="fn_deep", inputs=Triage, outputs=Analysis)
        shallow = Node.scripted("shallow", fn="fn_shallow", inputs=Triage, outputs=Analysis)

        def forward(self, topic):
            checked = self.triage(topic)
            if checked.confidence > 0.8:  # proxy attr vs constant — the only legal branch shape
                return self.shallow(checked)
            return self.deep(checked)

    graph = compile(Router(), scripted={"fn_triage": fn_triage, "fn_deep": fn_deep, "fn_shallow": fn_shallow})
    result = run(graph, input={"node_id": "d1"})
    picked = result.get("deep") or result.get("shallow")
    print(f"triage confidence 0.55 (< 0.8) -> routed to: {picked.depth} ({picked.summary})\n")
    assert picked.depth == "deep"  # low confidence took the deep arm


# =============================================================================
# Demo 2 — SELF.LOOP: an iterative refine cycle (real graph back-edge)
# =============================================================================
# Python for/while trace once (no cycle). self.loop() builds a sub-construct
# with a Loop modifier — a genuine back-edge. The body re-runs until `when`
# returns False or max_iterations is hit.


def demo_loop() -> None:
    print("=" * 68)
    print("DEMO 2 — self.loop(): draft -> [review -> revise] cycle until score >= 0.8")
    print("=" * 68)

    rounds = {"n": 0}

    def fn_draft(inp, _cfg):
        return Draft(content="v0", score=0.0)

    def fn_review(inp, _cfg):
        rounds["n"] += 1
        return Draft(content=inp.content, score=min(0.3 * rounds["n"], 1.0), revision=rounds["n"])

    def fn_revise(inp, _cfg):
        return Draft(content=f"v{rounds['n']}", score=inp.score, revision=rounds["n"])

    class Writer(ForwardConstruct):
        draft = Node.scripted("draft", fn="fn_draft", outputs=Draft)
        review = Node.scripted("review", fn="fn_review", inputs=Draft, outputs=Draft)
        revise = Node.scripted("revise", fn="fn_revise", inputs=Draft, outputs=Draft)

        def forward(self, topic):
            d = self.draft(topic)
            d = self.loop(
                body=[self.review, self.revise],
                when=lambda d: d is None or d.score < 0.8,  # None-safe: first iteration
                max_iterations=10,
            )(d)
            return d

    graph = compile(Writer(), scripted={"fn_draft": fn_draft, "fn_review": fn_review, "fn_revise": fn_revise})
    result = run(graph, input={"node_id": "d2"})
    # A self.loop body is a sub-construct; its output accumulates as a list.
    history = next((v for v in result.values() if isinstance(v, list) and v and hasattr(v[0], "score")), [])
    print(f"looped {rounds['n']} rounds; scores: {[round(d.score, 1) for d in history]}\n")
    assert history and history[-1].score >= 0.8


# =============================================================================
# Demo 3 — SELF.EACH: custom-key fan-out over a batch
# =============================================================================
# self.each(body=[...], key=...) fans a sub-construct out over a collection,
# keyed by your field. Downstream sees the barrier as dict[str, T]. A collector
# node reduces it (a loop body can't END on a fanned-out member; standalone
# each can return the dict directly, but here we reduce to a Report).


def demo_each_fanout() -> None:
    print("=" * 68)
    print("DEMO 3 — self.each(): fan verify out over claims, keyed by id, then reduce")
    print("=" * 68)

    def fn_extract(inp, _cfg):
        return ClaimBatch(claims=[Claim(id=f"c{i}", text=f"claim {i}") for i in range(4)])

    def fn_verify(item, _cfg):
        # Runs once PER claim (fan-out). In production: an `agent` node with tools.
        return Verdict(id=item.id, supported=(int(item.id[1:]) % 2 == 0))

    def fn_report(inp, _cfg):
        # `inp` is the dict[str, Verdict] barrier, keyed by the each key ("id").
        verdicts = list(inp["each_verify"].values())
        return Report(supported=sum(v.supported for v in verdicts), total=len(verdicts))

    class Auditor(ForwardConstruct):
        extract = Node.scripted("extract", fn="fn_extract", outputs=ClaimBatch)
        verify = Node.scripted("verify", fn="fn_verify", inputs=Claim, outputs=Verdict)
        report = Node.scripted("report", fn="fn_report", inputs={"each_verify": dict[str, Verdict]}, outputs=Report)

        def forward(self, topic):
            batch = self.extract(topic)
            self.each(body=[self.verify], key="id")(batch.claims)  # fan-out, keyed by Verdict.id
            return self.report(batch)

    graph = compile(Auditor(), scripted={"fn_extract": fn_extract, "fn_verify": fn_verify, "fn_report": fn_report})
    result = run(graph, input={"node_id": "d3"})
    rep = result["report"]
    print(f"verified {rep.total} claims in parallel; {rep.supported} supported\n")
    assert rep.total == 4


# =============================================================================
# Demo 4 — CASCADE: fan-out INSIDE a loop (the e9zse flagship shape)
# =============================================================================
# The topology ForwardConstruct could NOT express before e9zse: a loop whose
# body contains a fan-out. Deferred builders (self.each) nest into a
# self.loop body. A nested each never gets called directly, so its `over=` is
# supplied at construction. A collector after the each closes the loop iteration.


def demo_cascade() -> None:
    print("=" * 68)
    print("DEMO 4 — cascade: self.loop(body=[ get_claims, self.each(verify), collect ])")
    print("=" * 68)

    def fn_intake(inp, _cfg):
        return ClaimBatch(claims=[Claim(id=f"c{i}", text=f"claim {i}") for i in range(3)])

    def fn_get_claims(inp, _cfg):
        return inp  # pass the batch through (in production: re-fetch / expand)

    def fn_verify(item, _cfg):
        return Verdict(id=item.id, supported=True)

    def fn_collect(inp, _cfg):
        verdicts = list(inp["each_verify"].values())
        return Report(supported=sum(v.supported for v in verdicts), total=len(verdicts))

    class Cascade(ForwardConstruct):
        intake = Node.scripted("intake", fn="fn_intake", outputs=ClaimBatch)
        get_claims = Node.scripted("get_claims", fn="fn_get_claims", inputs=ClaimBatch, outputs=ClaimBatch)
        verify = Node.scripted("verify", fn="fn_verify", inputs=Claim, outputs=Verdict)
        collect = Node.scripted("collect", fn="fn_collect", inputs={"each_verify": dict[str, Verdict]}, outputs=Report)

        def forward(self, topic):
            batch = self.intake(topic)
            return self.loop(
                body=[
                    self.get_claims,
                    self.each(body=[self.verify], key="id", over="get_claims.claims"),  # nested: over= at build
                    self.collect,  # collector reduces the fan-out barrier
                ],
                when=lambda r: r is None or r.total == 0,  # run the investigate round once
                max_iterations=2,
            )(batch)

    graph = compile(
        Cascade(),
        scripted={
            "fn_intake": fn_intake,
            "fn_get_claims": fn_get_claims,
            "fn_verify": fn_verify,
            "fn_collect": fn_collect,
        },
    )
    result = run(graph, input={"node_id": "d4"})
    reports = next((v for v in result.values() if isinstance(v, list) and v and hasattr(v[0], "total")), [])
    print(f"cascade ran; final report: {reports[-1].supported}/{reports[-1].total} supported\n")
    assert reports and reports[-1].total == 3


# =============================================================================
# Demo 5 — SELF.ENSEMBLE: N parallel generators + judge-merge (Oracle)
# =============================================================================
# self.ensemble(node, n=N, merge_fn=...) builds an Oracle: N parallel runs of
# the generator, then a merge. Form-aware: a node ref emits Node | Oracle; a
# list emits Construct | Oracle. Byte-identical to the declarative twin.


def demo_ensemble() -> None:
    print("=" * 68)
    print("DEMO 5 — self.ensemble(): 3 parallel drafts, merged to the best")
    print("=" * 68)

    def fn_seed(inp, _cfg):
        return Draft(content="seed", score=0.0)

    def fn_generate(inp, _cfg):
        return Draft(content="candidate", score=0.7)

    def fn_merge(variants, _cfg):
        # `variants` is the list of the N generator outputs; pick the best.
        best = max(variants, key=lambda d: d.score)
        return Draft(content=f"merged from {len(variants)} candidates", score=best.score)

    class Ensembled(ForwardConstruct):
        seed = Node.scripted("seed", fn="fn_seed", outputs=Draft)
        generate = Node.scripted("generate", fn="fn_generate", inputs=Draft, outputs=Draft)

        def forward(self, topic):
            s = self.seed(topic)
            return self.ensemble(self.generate, n=3, merge_fn="fn_merge")(s)

    graph = compile(
        Ensembled(),
        # merge_fn= names a scripted fn(variants, config) -> merged; it rides
        # the same scripted= registry as the nodes.
        scripted={"fn_seed": fn_seed, "fn_generate": fn_generate, "fn_merge": fn_merge},
    )
    result = run(graph, input={"node_id": "d5"})
    merged = result["generate"]
    print(f"ensemble merged 3 candidates -> {merged.content} (score {merged.score})\n")
    assert "merged from 3" in merged.content


# =============================================================================
# Demo 6 — SELF.INTERRUPT: human-in-the-loop gate before a mutation
# =============================================================================
# self.interrupt(node, when='condition') attaches an Operator: the graph
# checkpoints and STOPS before the node when the registered condition fires,
# and resumes with run(graph, resume={...}). Needs a checkpointer.


def demo_interrupt() -> None:
    print("=" * 68)
    print("DEMO 6 — self.interrupt(): pause before `apply` for human approval")
    print("=" * 68)
    if MemorySaver is None:
        print("(skipped — langgraph checkpoint saver unavailable)\n")
        return

    def fn_precheck(inp, _cfg):
        return Analysis(summary="2 issues found", depth="review-needed")

    def fn_apply(inp, _cfg):
        return Report(supported=1, total=1)  # the mutating action, gated behind approval

    def needs_review(state) -> bool:
        # The condition receives the compiled Pydantic state model; read the
        # upstream node's output field by name.
        val = getattr(state, "precheck", None)
        return bool(val and val.depth == "review-needed")

    class Gated(ForwardConstruct):
        precheck = Node.scripted("precheck", fn="fn_precheck", outputs=Analysis)
        apply = Node.scripted("apply", fn="fn_apply", inputs=Analysis, outputs=Report)

        def forward(self, topic):
            checked = self.precheck(topic)
            # self.interrupt attaches the Operator gate; a plain `| Operator(...)`
            # on the class attr would be the equivalent declarative form.
            return self.interrupt(self.apply, when="needs_review")(checked)

    graph = compile(
        Gated(),
        scripted={"fn_precheck": fn_precheck, "fn_apply": fn_apply},
        conditions={"needs_review": needs_review},
        checkpointer=MemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "d6"}}
    paused = run(graph, input={"node_id": "d6"}, config=cfg)
    print(f"paused before `apply` (condition fired): __interrupt__ present = {'__interrupt__' in paused}")
    resumed = run(graph, resume={"approved": True}, config=cfg)
    print(f"resumed with approval -> apply ran: {resumed.get('apply')}\n")
    assert "__interrupt__" in paused


def main() -> None:
    demo_branch()
    demo_loop()
    demo_each_fanout()
    demo_cascade()
    demo_ensemble()
    demo_interrupt()
    print("=" * 68)
    print("All ForwardConstruct surfaces ran: branch, loop, each fan-out,")
    print("fan-out-in-loop cascade, ensemble, and human-in-the-loop interrupt.")
    print("Same imperative forward() -> same IR the declarative form compiles.")
    print("=" * 68)


if __name__ == "__main__":
    main()
