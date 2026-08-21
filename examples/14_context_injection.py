"""Example 14: context= -- reading run state a node's input port never carries.

`context=` lets a node read a value produced EARLIER IN THE RUN without it
being threaded through every intervening shape, and the reference is DECLARED
so a missing binding fails at assembly rather than at run time. It works inside
a fan-out, where the port carries which ITEM and context carries which RUN.

This example happens to show the verbatim-rendering property -- the catalog
must reach the model exactly as written -- but that is a property of the
channel, not its purpose. See AGENTS.md, "Run-scoped state".

Originally titled "Context Injection -- verbatim state in sub-constructs".

Scenario: A requirements verification pipeline where:
  1. build_catalog: scripted node produces a pre-formatted graph catalog
  2. make_claims: scripted node produces claims to verify
  3. verify sub-construct: for each claim, an agent explores with tools
     while seeing the catalog as verbatim context

The catalog is crafted for LLM consumption (BFS-ordered, incident-encoded), so
it must reach the model as written. Two things make that true, and the example
needs both: context= keeps a configured renderer off the value, and
GraphCatalog.render_for_prompt() says what the type looks like to an LLM. Skip
the presenter and the model sees a wrapped repr of the catalog rather than the
catalog -- which is what this example used to do while claiming otherwise.

This demonstrates:
  - context= on @node for state injection no renderer may wrap
  - Context forwarded from parent state into sub-constructs
  - Three rendering strategies unified in one pipeline:
      * Typed input (VerifyClaim) -- BAML via describe_value
      * Tool results (EvidenceHit) -- BAML via describe_value
      * Context (catalog) -- verbatim, no rendering

Run:
    python examples/14_context_injection.py
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from neograph import (
    Tool,
    ToolInteraction,
    compile,
    construct_from_functions,
    node,
    run,
)

# -- Schemas ------------------------------------------------------------------

class GraphCatalog(BaseModel, frozen=True):
    """Pre-computed graph catalog for LLM context."""
    content: str

    def render_for_prompt(self) -> str:
        """Hand the model the catalog text and nothing else.

        This is what makes "verbatim" true rather than merely intended. A model
        has no verbatim form of its own: before neograph-ufqr7 the framework
        stringified it and the prompt received the Pydantic repr
        ``content='=== Graph Catalog ...'`` -- wrapped, with the newlines
        escaped. Declaring a presenter is how a type says what it looks like to
        an LLM, and the context channel now honours it like every other channel.
        """
        return self.content


class VerifyClaim(BaseModel, frozen=True):
    claim_id: str
    text: str


class ClaimBatch(BaseModel, frozen=True):
    claims: list[VerifyClaim]


class EvidenceHit(BaseModel, frozen=True):
    """Typed tool result -- what search_evidence returns."""
    source: str = Field(description="Source artifact")
    line: int = Field(description="Line number")
    relevance: float = Field(description="Relevance score 0-1")


class Verdict(BaseModel, frozen=True):
    claim_id: str
    disposition: str
    evidence_count: int


# -- Fake LLM ----------------------------------------------------------------

class FakeAgentLLM:
    """1 tool call, then structured result."""

    def __init__(self):
        self._call_count = 0
        self._structured = False

    def bind_tools(self, tools):
        clone = FakeAgentLLM()
        clone._call_count = self._call_count
        return clone

    def invoke(self, messages, **kwargs):
        if self._structured:
            return self._model(
                claim_id="scored", disposition="confirmed", evidence_count=1,
            )
        self._call_count += 1
        if self._call_count == 1:
            msg = AIMessage(content="")
            msg.tool_calls = [{
                "name": "search_evidence",
                "args": {"query": "verify"},
                "id": "call-1",
            }]
            return msg
        # Agent mode parses the final ReAct turn as JSON (schema injected up
        # front), so emit a valid Verdict.
        return AIMessage(
            content=Verdict(
                claim_id="scored", disposition="confirmed", evidence_count=1,
            ).model_dump_json()
        )

    def with_structured_output(self, model, **kwargs):
        clone = FakeAgentLLM()
        clone._call_count = self._call_count
        clone._model = model
        clone._structured = True
        return clone


# -- Fake tool ----------------------------------------------------------------

class FakeSearch:
    name = "search_evidence"

    def invoke(self, args, config=None, **kwargs):
        return EvidenceHit(source="auth.py", line=42, relevance=0.95)


# -- Tool factories + LLM layer (passed to compile() below) -------------------

TOOL_FACTORIES = {
    "search_evidence": lambda cfg, tc: FakeSearch(),
}


def llm_factory(tier):
    return FakeAgentLLM()


SEEN_CONTEXT: dict[str, str] = {}


def prompt_compiler(template, data, **kw):
    # Record what the context channel actually delivered, so the closing lines
    # can PROVE the claim instead of asserting it.
    SEEN_CONTEXT.update(kw.get("context") or {})
    return [
        {"role": "user", "content": f"template={template} context={kw.get('context', 'none')}"},
    ]


# -- Pipeline -----------------------------------------------------------------

# Step 1: Build a graph catalog (pre-formatted for LLM consumption)
@node(outputs=GraphCatalog)
def build_catalog() -> GraphCatalog:
    return GraphCatalog(content=(
        "=== Graph Catalog (BFS from UC-001) ===\n"
        "UC-001: User Authentication [auth.py:10-85]\n"
        "  BR-001: Password min 12 chars [traces: UC-001]\n"
        "  BR-002: Session timeout 30min [traces: UC-001]\n"
        "UC-002: Data Encryption [crypto.py:1-120]\n"
        "  BR-003: AES-256 at rest [traces: UC-002]\n"
    ))


# Step 2: Produce claims to verify
@node(outputs=ClaimBatch)
def make_claims() -> ClaimBatch:
    return ClaimBatch(claims=[
        VerifyClaim(claim_id="C1", text="system authenticates via SSO"),
        VerifyClaim(claim_id="C2", text="data encrypted at rest with AES-256"),
    ])


# Step 3: Sub-construct -- for each claim, explore with catalog as context
@node(
    mode="agent",
    outputs={"result": Verdict, "tool_log": list[ToolInteraction]},
    model="research",
    prompt="verify/explore",
    tools=[Tool("search_evidence", budget=2)],
    context=["build_catalog"],  # <-- catalog forwarded from parent state
)
def verify(claim: VerifyClaim) -> Verdict: ...


verify_claim = construct_from_functions(
    "verify-claim", [verify],
    input=VerifyClaim, output=Verdict,
).map("make_claims.claims", key="claim_id")


# Step 4: Assemble -- @node functions + sub-construct in one call
pipeline = construct_from_functions(
    "verification",
    [build_catalog, make_claims, verify_claim],
)


# -- Run ----------------------------------------------------------------------

if __name__ == "__main__":
    graph = compile(
        pipeline,
        llm_factory=llm_factory,
        prompt_compiler=prompt_compiler,
        tool_factories=TOOL_FACTORIES,
    )
    result = run(graph, input={"node_id": "VERIFY-001"})

    print("=== Verification Results ===\n")

    verdicts = result["verify_claim"]
    for cid, v in sorted(verdicts.items()):
        print(f"  {cid}: {v.disposition} (evidence: {v.evidence_count})")

    print(f"\nResult keys: {sorted(result.keys())}")
    print("\nThe catalog reached the agent's prompt compiler as written --")
    print("context=['build_catalog'] keeps any configured renderer off it, and")
    print("GraphCatalog.render_for_prompt() gives it its verbatim text form.")
    print("The agent saw the catalog alongside its typed VerifyClaim input.")

    # Proof, not narration: what the compiler received IS the catalog string.
    delivered = SEEN_CONTEXT["build_catalog"]
    assert delivered == result["build_catalog"].content, delivered
    print(f"\n  first line delivered: {delivered.splitlines()[0]}")
