"""Example 13: Gather + Produce Sub-Construct -- agent explores then judges.

Scenario: A claim verification pipeline. For each claim:
  1. Explore (gather mode): uses a search_evidence tool to find supporting
     evidence, produces an ExplorationResult + tool interaction log
  2. Score (produce mode): receives the exploration result AND the tool log
     as context, judges the claim in a fresh LLM conversation

The two-phase pattern is critical for quality: the explore phase does noisy
tool-calling work, while the score phase gets a clean, distilled research
packet -- no hedging from tool context noise.

Both nodes are @node-decorated functions. The sub-construct is built via
construct_from_functions(input=, output=) with port param resolution -- the
explore node's `claim: VerifyClaim` param is automatically wired to the
sub-construct's input port. Then .map() fans out over a collection.

Demonstrates:
  - @node gather mode with dict-form outputs (result + tool_log)
  - @node produce mode consuming dict-output references (explore_result, explore_tool_log)
  - construct_from_functions with input=/output= for sub-construct boundary
  - Port param resolution (claim param -> neo_subgraph_input)
  - .map() fan-out on a sub-construct

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
    """Simulates a produce-mode LLM that scores a claim."""

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
    name = "search_evidence"

    def invoke(self, args):
        query = args.get("query", "?")
        return f"Found reference for: {query} -> auth.py:42"


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
# Phase 1: Explore with tools (gather mode, dict outputs: result + tool_log)
# Phase 2: Score in fresh conversation (produce mode, consumes result + tool_log)

@node(
    mode="gather",
    outputs={"result": ExplorationResult, "tool_log": list[ToolInteraction]},
    model="research",
    prompt="verify/explore",
    tools=[Tool("search_evidence", budget=3)],
)
def explore(claim: VerifyClaim) -> ExplorationResult:
    # Body unused -- LLM handles execution via prompt= + tools=
    ...


@node(
    mode="produce",
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
# seed is declarative (Node.scripted) because the parent mixes it with
# the sub-Construct. @node + Construct in one Construct() call requires
# the programmatic form for the non-sub-construct nodes.

register_scripted("seed_claims", lambda _in, _cfg: ClaimBatch(claims=[
    VerifyClaim(claim_id="REQ-1", text="system shall authenticate users via SSO"),
    VerifyClaim(claim_id="REQ-2", text="system shall encrypt data at rest"),
    VerifyClaim(claim_id="REQ-3", text="system shall log all access attempts"),
]))

parent = Construct("verification-pipeline", nodes=[
    Node.scripted("seed", fn="seed_claims", outputs=ClaimBatch),
    verify_claim.map("seed.claims", key="claim_id"),
])


# -- Run ----------------------------------------------------------------------

if __name__ == "__main__":
    graph = compile(parent)
    result = run(graph, input={"node_id": "VERIFY-001"})

    verdicts = result["verify_claim"]
    print(f"Verified {len(verdicts)} claims:\n")
    for claim_id, verdict in sorted(verdicts.items()):
        print(f"  {claim_id}: {verdict.disposition}")
        print(f"    reasoning: {verdict.reasoning}")
    print(f"\nResult keys: {sorted(result.keys())}")
    print("Note: explore/score internals are NOT in the result -- only verify_claim")
