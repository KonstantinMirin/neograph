"""Example 13: Agent + Think Sub-Construct -- agent explores, thinker judges.

Scenario: A claim verification pipeline. For each claim:
  1. Explore (agent mode): uses a search_evidence tool to find supporting
     evidence. The tool returns a TYPED Pydantic model (EvidenceHit), not
     a string. The framework preserves the typed result in ToolInteraction
     and renders it as JSON for the LLM.
  2. Score (think mode): receives the exploration result AND the typed tool
     log as context, judges the claim in a fresh LLM conversation.

The two-phase pattern is critical for quality: the explore phase does noisy
tool-calling work, while the score phase gets a clean, distilled research
packet -- no hedging from tool context noise.

Both nodes are @node-decorated functions. The sub-construct is built via
construct_from_functions(input=, output=) with port param resolution -- the
explore node's `claim: VerifyClaim` param is automatically wired to the
sub-construct's input port. Then .map() fans out over a collection.

Demonstrates:
  - Typed tool results: tools return Pydantic models, not strings
  - ToolInteraction.typed_result preserves the original model
  - @node agent mode with dict-form outputs (result + tool_log)
  - @node think mode consuming dict-output references (explore_result, explore_tool_log)
  - construct_from_functions with input=/output= for sub-construct boundary
  - Port param resolution (claim param -> neo_subgraph_input)
  - .map() fan-out on a sub-construct
  - context= for verbatim state injection (pre-formatted catalogs)

Run:
    python examples/13_gather_produce_subconstruct.py
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from neograph import (
    Construct,
    Node,
    Tool,
    ToolInteraction,
    compile,
    configure_llm,
    construct_from_functions,
    node,
    register_scripted,
    register_tool_factory,
    run,
)


# -- Schemas ------------------------------------------------------------------

class VerifyClaim(BaseModel, frozen=True):
    claim_id: str
    text: str


class ClaimBatch(BaseModel, frozen=True):
    claims: list[VerifyClaim]


class ExplorationResult(BaseModel, frozen=True):
    evidence: list[str]
    summary: str


class ClaimVerdict(BaseModel, frozen=True):
    claim_id: str
    disposition: str
    reasoning: str


class EvidenceHit(BaseModel, frozen=True):
    """Typed tool result -- what search_evidence returns.
    The framework preserves this in ToolInteraction.typed_result and
    renders it as JSON for the LLM (not Python repr)."""
    source_file: str
    line: int
    snippet: str
    relevance: float


# -- Fake LLMs (replace with real OpenRouter/OpenAI in production) ------------

class FakeExploreLLM:
    """Simulates a gather-mode LLM: 1 tool call, then structured result.

    The ReAct loop runs: LLM calls tool → tool returns → LLM responds "done" →
    framework does final structured parse to get ExplorationResult.
    """

    def __init__(self):
        self._call_count = 0
        self._structured = False

    def bind_tools(self, tools):
        clone = FakeExploreLLM()
        clone._call_count = self._call_count
        return clone

    def invoke(self, messages, **kwargs):
        if self._structured:
            return self._model(
                evidence=["auth.py:42", "crypto.py:18"],
                summary="found 2 references supporting the claim",
            )
        self._call_count += 1
        if self._call_count == 1:
            msg = AIMessage(content="")
            msg.tool_calls = [{
                "name": "search_evidence",
                "args": {"query": "verify claim"},
                "id": "call-1",
            }]
            return msg
        return AIMessage(content="exploration complete")

    def with_structured_output(self, model, **kwargs):
        clone = FakeExploreLLM()
        clone._call_count = self._call_count
        clone._model = model
        clone._structured = True
        return clone


class FakeScoreLLM:
    """Simulates a think-mode LLM that scores a claim."""

    def with_structured_output(self, model, **kwargs):
        self._model = model
        return self

    def invoke(self, messages, **kwargs):
        return self._model(
            claim_id="scored",
            disposition="confirmed",
            reasoning="evidence supports the claim based on 2 source references",
        )


# -- Fake tool ----------------------------------------------------------------

class FakeEvidenceSearch:
    """Fake tool that returns a TYPED Pydantic model, not a string.
    In production, this would query a code search index or knowledge graph."""
    name = "search_evidence"

    def invoke(self, args):
        return EvidenceHit(
            source_file="auth.py",
            line=42,
            snippet="def authenticate(user, password): ...",
            relevance=0.95,
        )


register_tool_factory("search_evidence", lambda config, tool_config: FakeEvidenceSearch())


# -- Configure LLM layer -----------------------------------------------------

def llm_factory(tier):
    if tier == "research":
        return FakeExploreLLM()
    return FakeScoreLLM()  # "judge" tier


configure_llm(
    llm_factory=llm_factory,
    prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": "verify"}],
)


# -- Pipeline nodes -----------------------------------------------------------
# Phase 1: Explore with tools (agent mode, dict outputs: result + tool_log)
# Phase 2: Score in fresh conversation (think mode, consumes result + tool_log)

@node(
    mode="agent",
    outputs={"result": ExplorationResult, "tool_log": list[ToolInteraction]},
    model="research",
    prompt="verify/explore",
    tools=[Tool("search_evidence", budget=3)],
)
def explore(claim: VerifyClaim) -> ExplorationResult:
    # Body unused -- LLM handles execution via prompt= + tools=
    ...


@node(
    mode="think",
    outputs=ClaimVerdict,
    model="judge",
    prompt="verify/score",
)
def score(explore_result: ExplorationResult, explore_tool_log: list[ToolInteraction]) -> ClaimVerdict:
    # Body unused -- LLM handles execution via prompt=
    # explore_result and explore_tool_log are dict-output references:
    # they resolve to the upstream explore node's per-key state fields.
    ...


# -- Sub-construct: explore -> score ------------------------------------------
# Built from @node functions with port param resolution.
# explore's `claim: VerifyClaim` param matches input=VerifyClaim,
# so it reads from neo_subgraph_input (the sub-construct's input port).

verify_claim = construct_from_functions(
    "verify-claim", [explore, score],
    input=VerifyClaim, output=ClaimVerdict,
)


# -- Parent pipeline ----------------------------------------------------------
# ALL @node functions + the sub-construct assembled via construct_from_functions.
# No Construct(nodes=[...]) workaround, no Node.scripted -- pure @node + sub-construct.

class FinalReport(BaseModel, frozen=True):
    summary: str
    claim_count: int


@node(outputs=ClaimBatch)
def flatten_claims() -> ClaimBatch:
    return ClaimBatch(claims=[
        VerifyClaim(claim_id="REQ-1", text="system shall authenticate users via SSO"),
        VerifyClaim(claim_id="REQ-2", text="system shall encrypt data at rest"),
        VerifyClaim(claim_id="REQ-3", text="system shall log all access attempts"),
    ])


@node(outputs=FinalReport)
def deterministic_merge(verify_claim: dict[str, ClaimVerdict]) -> FinalReport:
    """Consume the Each-fanned-out sub-construct results as dict[str, ClaimVerdict]."""
    verdicts = [f"{k}: {v.disposition}" for k, v in sorted(verify_claim.items())]
    return FinalReport(
        summary="\n".join(verdicts),
        claim_count=len(verify_claim),
    )


# construct_from_functions: @node functions + sub-construct.map() in one call.
# flatten_claims → verify_claim.map(Each fan-out) → deterministic_merge
pipeline = construct_from_functions(
    "verification-pipeline",
    [flatten_claims, verify_claim.map("flatten_claims.claims", key="claim_id"), deterministic_merge],
)


# -- Run ----------------------------------------------------------------------

if __name__ == "__main__":
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "VERIFY-001"})

    verdicts = result["verify_claim"]
    print(f"Verified {len(verdicts)} claims:\n")
    for claim_id, verdict in sorted(verdicts.items()):
        print(f"  {claim_id}: {verdict.disposition}")
        print(f"    reasoning: {verdict.reasoning}")

    print(f"\n{result['deterministic_merge'].summary}")
    print(f"\nTotal claims: {result['deterministic_merge'].claim_count}")
    print(f"Result keys: {sorted(result.keys())}")
    print("Note: explore/score internals are NOT in the result")

    # -- Typed tool results demo ------------------------------------------------
    # The tool_log lives inside the sub-construct (doesn't surface to parent).
    # To show typed_result, run a standalone agent node:
    import types
    from neograph import construct_from_module

    mod = types.ModuleType("typed_tool_demo")

    @node(
        mode="agent",
        outputs={"result": ExplorationResult, "tool_log": list[ToolInteraction]},
        model="research", prompt="verify/explore",
        tools=[Tool("search_evidence", budget=1)],
    )
    def demo_explore() -> ExplorationResult: ...

    mod.demo_explore = demo_explore
    demo_graph = compile(construct_from_module(mod))
    demo_result = run(demo_graph, input={"node_id": "demo"})

    tool_log = demo_result["demo_explore_tool_log"]
    print("\n-- Typed tool results --")
    print(f"  tool_log[0].tool_name: {tool_log[0].tool_name}")
    print(f"  tool_log[0].result (rendered BAML): {tool_log[0].result[:60]}...")
    print(f"  tool_log[0].typed_result: {tool_log[0].typed_result}")
    print(f"  typed_result.source_file: {tool_log[0].typed_result.source_file}")
    print(f"  typed_result.relevance: {tool_log[0].typed_result.relevance}")

    # -- Context injection demo -------------------------------------------------
    # context= injects verbatim state fields alongside typed input.
    # The catalog is pre-formatted for LLM consumption — not BAML-rendered.

    class GraphCatalog(BaseModel, frozen=True):
        """Pre-computed graph catalog for LLM context."""
        content: str

    ctx_mod = types.ModuleType("context_demo")

    @node(outputs=GraphCatalog)
    def build_catalog() -> GraphCatalog:
        return GraphCatalog(content=(
            "=== Graph Catalog (BFS order) ===\n"
            "UC-001: User Authentication [impl: auth.py]\n"
            "UC-002: Data Encryption [impl: crypto.py]\n"
            "BR-001: Password min 12 chars [traces: UC-001]\n"
        ))

    @node(
        mode="agent",
        outputs={"result": ExplorationResult, "tool_log": list[ToolInteraction]},
        model="research", prompt="verify/explore",
        tools=[Tool("search_evidence", budget=1)],
        context=["build_catalog"],  # <-- verbatim state injection
    )
    def ctx_explore(build_catalog: GraphCatalog) -> ExplorationResult: ...

    ctx_mod.build_catalog = build_catalog
    ctx_mod.ctx_explore = ctx_explore
    ctx_graph = compile(construct_from_module(ctx_mod))
    ctx_result = run(ctx_graph, input={"node_id": "ctx-demo"})

    print("\n-- Context injection --")
    print("  context=['build_catalog'] passes the catalog verbatim to the prompt compiler.")
    print("  The prompt compiler receives: context={'build_catalog': GraphCatalog(content=...)}")
    print(f"  explore result: {ctx_result['ctx_explore_result'].summary}")
