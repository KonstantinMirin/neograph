"""Example 23: neograph as an MCP client — selective binding, per-run identity, gated mutation.

The scenario is a CRM deal-review pipeline talking to a REAL Model Context
Protocol server (``examples/_mcp_demo_server.py``, a plain FastMCP server) spawned
as a stdio subprocess. The MCP layer is genuine end-to-end — tool discovery,
per-operator auth echo, structured results all round-trip over real stdio JSON-RPC.
The ONLY fakes are the LLMs (a real model needs an API key; the tool calls it
makes are scripted so the example is deterministic and keyless).

The featured wiring is the shipped ``neograph[mcp]`` battery: ONE call to
``mcp_tool_factories(...)`` turns a typed server config into the
``{name: async factory}`` dict that ``compile(tool_factories=...)`` already
accepts. A consumer who wants house rules hand-rolls the same factory — the
APPENDIX at the bottom shows that escape hatch, plus the streamable-http auth path.

Five beats, in narrative order:

  1. SELECTIVE BINDING — the point vs create_react_agent's one-agent-holds-all-tools.
     The battery returns a DICT; each node binds only ITS slice via per-node
     ``tools=``. The research node gets the read-only readers (crm_search,
     kb_lookup); the action node gets the mutating update_deal ONLY. Least
     privilege per node, one client.

  2. RAW BaseTool passthrough + budget — a battery tool handed straight to a node
     in ``tools=[raw_tool]`` with zero ceremony, and ``lint()`` catching the
     async-only MCP tool that a sync ``run()`` cannot drive.

  3. PER-RUN IDENTITY — a ``token_provider`` reads ``config['configurable']['mcp_auth']``
     and mints each run's identity. Over stdio there are no HTTP headers, so
     identity rides as a tool ARGUMENT the server echoes back (``acting_as``). We
     run the pipeline as operator-A then operator-B and read which token each
     run's tools carried straight out of the tool log.

  4. GATED MUTATION — ``gate_tools_when`` on the action node pauses the run BEFORE
     update_deal fires. Resume with ``{"approved": True}`` and it runs exactly
     once; resume with ``{"approved": False}`` and the tool never runs — the
     agent is told why and finalizes anyway. MCP mutations meet the HITL story.

  5. TYPED RESULTS — each reader factory declares ``output_model=``, so neograph
     rehydrates the server's ``structuredContent`` into a client model and
     ``ToolInteraction.typed_result`` IS that model (``search.typed_result.hits[0]``
     is a ``Deal``, no JSON unpacking). Read-only readers are marked ``idempotent=True``.

Run (needs the mcp-examples extra; no API keys, no network beyond the local
subprocess):
    uv run --extra dev --extra mcp-examples python examples/23_mcp_client_selective_binding.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from neograph import (
    Tool,
    ToolInteraction,
    arun,
    compile,
    construct_from_functions,
    lint,
    node,
)
from neograph_mcp import StdioServer, mcp_tool_factories

# ── The real MCP server: spawned as a stdio subprocess ───────────────────────
# No ports, no network — the client (this file) launches the server as a child
# process and speaks real MCP over its stdio pipes.

DEMO_SERVER = str((Path(__file__).parent / "_mcp_demo_server.py").resolve())

CRM = StdioServer(command=sys.executable, args=[DEMO_SERVER])


# ── Per-run identity: the token_provider (beat 3) ────────────────────────────
# The battery calls this once per superstep with config['configurable']. Over
# stdio the returned token rides as the `token` tool argument (the server echoes
# it under `acting_as`); over streamable-http it would ride as a bearer header
# instead (see the APPENDIX). neograph never inspects the token — it only carries
# it from config to the tool.


def token_provider(configurable: dict[str, Any]) -> str:
    return configurable.get("mcp_auth", "anon")


# ── Schemas ──────────────────────────────────────────────────────────────────


class Deal(BaseModel, frozen=True):
    """A CRM deal — the typed shape a structured MCP result rehydrates into."""

    id: str
    name: str
    stage: str


# ── Client-side result models (beat 5) ───────────────────────────────────────
# The typed channel: declare output_model= on a reader factory and neograph
# rehydrates the server's structuredContent into OUR model (fields we care about;
# Pydantic ignores the rest). ToolInteraction.typed_result then IS the model, and
# the next ReAct turn's ToolMessage is its BAML rendering — no hand-parsing.


class CrmSearchResult(BaseModel, frozen=True):
    """What crm_search returns — hits rehydrate straight into Deal models."""

    hits: list[Deal]
    acting_as: str  # the per-run identity the server echoed


class KbResult(BaseModel, frozen=True):
    """What kb_lookup returns — we only need the echoed identity here."""

    acting_as: str


class ResearchBrief(BaseModel, frozen=True):
    deal_id: str
    summary: str
    playbook: str


class ActionOutcome(BaseModel, frozen=True):
    deal_id: str
    status: str  # "applied" | "blocked"
    note: str


# ── Fake LLMs (the ONLY fakes — the MCP layer is real) ───────────────────────
# History-driven, like a real stateless model: each turn decides its next move
# from the ToolMessages already in the conversation, so it replays identically
# across an interrupt/resume.


class _ResearchFake:
    """Agent turns: crm_search -> kb_lookup -> final ResearchBrief."""

    _BRIEF = ResearchBrief(
        deal_id="D1",
        summary="Acme renewal is in negotiation; anchor on realized value.",
        playbook="renewal-playbook",
    )

    def __init__(self) -> None:
        self._model: type[BaseModel] | None = None
        self._structured = False

    def bind_tools(self, tools: list) -> _ResearchFake:
        return self

    def abind_tools(self, *a: Any, **k: Any) -> _ResearchFake:
        return self

    def with_structured_output(self, model: type[BaseModel], **k: Any) -> _ResearchFake:
        clone = _ResearchFake()
        clone._model = model
        clone._structured = True
        return clone

    def invoke(self, messages: list, **k: Any) -> Any:
        if self._structured:
            return self._BRIEF
        n = sum(isinstance(m, ToolMessage) for m in messages)
        if n == 0:
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": "crm_search", "args": {"query": "Acme"}, "id": "s1"}]
            return msg
        if n == 1:
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": "kb_lookup", "args": {"topic": "renewal-playbook"}, "id": "k1"}]
            return msg
        return AIMessage(content=self._BRIEF.model_dump_json())

    async def ainvoke(self, *a: Any, **k: Any) -> Any:
        return self.invoke(*a, **k)


class _ActionFake:
    """Act turns: request update_deal once, then finalize. The final answer
    reflects whether the mutation was denied (a denial ToolMessage was fed back)."""

    def __init__(self) -> None:
        self._model: type[BaseModel] | None = None
        self._structured = False

    def bind_tools(self, tools: list) -> _ActionFake:
        return self

    def abind_tools(self, *a: Any, **k: Any) -> _ActionFake:
        return self

    def with_structured_output(self, model: type[BaseModel], **k: Any) -> _ActionFake:
        clone = _ActionFake()
        clone._model = model
        clone._structured = True
        return clone

    @staticmethod
    def _denied(messages: list) -> bool:
        return any(isinstance(m, ToolMessage) and "denied" in str(m.content).lower() for m in messages)

    def invoke(self, messages: list, **k: Any) -> Any:
        if self._structured:
            if self._denied(messages):
                return self._model(  # type: ignore[misc]
                    deal_id="D1",
                    status="blocked",
                    note="update_deal was denied by the reviewer; deal left unchanged.",
                )
            return self._model(deal_id="D1", status="applied", note="Advanced D1 to closed-won.")  # type: ignore[misc]
        n = sum(isinstance(m, ToolMessage) for m in messages)
        if n == 0:
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": "update_deal", "args": {"deal_id": "D1", "stage": "closed-won"}, "id": "u1"}]
            return msg
        # A tool result (or a denial) is now in history — finalize.
        if self._denied(messages):
            return AIMessage(content='{"deal_id":"D1","status":"blocked","note":"denied by reviewer"}')
        return AIMessage(content='{"deal_id":"D1","status":"applied","note":"advanced D1"}')

    async def ainvoke(self, *a: Any, **k: Any) -> Any:
        return self.invoke(*a, **k)


def llm_factory(tier: str) -> Any:
    return _ActionFake() if tier == "action" else _ResearchFake()


def prompt_compiler(template: str, data: Any, **kw: Any) -> list[dict[str, str]]:
    # The fakes ignore prompt content; a trivial compiler keeps the example keyless.
    return [{"role": "user", "content": template}]


# ── Beat 1: the battery, sliced per node for least-privilege binding ─────────
# ONE call builds a factory per tool the server exposes. The return is a DICT so
# we can hand each node just its slice — the research readers vs the action mutator.

FACTORIES = mcp_tool_factories(
    {"crm": CRM},
    token_provider=token_provider,
    namespace=False,  # single server, bare tool names (no "crm::" prefix)
    # TYPED RESULTS (beat 5): each reader's structuredContent rehydrates into a
    # client model, so typed_result carries the model and the ToolMessage is its
    # BAML render — read attributes, never hand-parse content blocks.
    output_models={"crm_search": CrmSearchResult, "kb_lookup": KbResult},
)

READER_TOOLS = ["crm_search", "kb_lookup"]  # read-only slice for the research node
MUTATOR_TOOLS = ["update_deal"]  # the mutating slice for the action node


# ── Pipelines ─────────────────────────────────────────────────────────────────


@node(
    mode="agent",
    outputs={"result": ResearchBrief, "tool_log": list[ToolInteraction]},
    model="research",
    prompt="research/brief",
    # SELECTIVE BINDING: only the read-only readers. This agent CANNOT call
    # update_deal — it is not in its tools=. idempotent=True marks them replay-safe
    # (a read may be re-invoked to re-derive an expired resource; a mutation may not).
    tools=[
        Tool("crm_search", budget=2, idempotent=True),
        Tool("kb_lookup", budget=1, idempotent=True),
    ],
    # Tell the model its per-tool budget up front (a framework system preamble).
    llm_config={"announce_tool_budget": True},
)
def research() -> ResearchBrief:  # body unused — the LLM drives via prompt= + tools=
    ...


research_pipeline = construct_from_functions("crm-research", [research])


@node(
    mode="act",  # act mode = mutating tools
    outputs=ActionOutcome,
    model="action",
    prompt="action/apply",
    # SELECTIVE BINDING: the mutating tool ONLY, with a hard budget of one call.
    tools=[Tool("update_deal", budget=1)],
    # GATED MUTATION: pause BEFORE the update_deal superstep and ask a human.
    gate_tools_when=lambda state: {"pending_tool": "update_deal", "reason": "Approve CRM mutation?"},
)
def apply_action() -> ActionOutcome:  # body unused — the LLM drives
    ...


action_pipeline = construct_from_functions("crm-action", [apply_action])


# ── Demos ─────────────────────────────────────────────────────────────────────


async def demo_research_identity_and_typed_results() -> None:
    """Beats 1, 3, 5: selective read-only binding, per-run identity, typed results."""
    print("=" * 66)
    print("BEAT 1/3/5: read-only binding, per-run identity, typed MCP results")
    print("=" * 66)

    graph = compile(
        research_pipeline,
        llm_factory=llm_factory,
        prompt_compiler=prompt_compiler,
        # Least-privilege: this graph only knows the reader factories.
        tool_factories={k: FACTORIES[k] for k in READER_TOOLS},
    )

    for operator in ("operator-A", "operator-B"):
        # Per-run identity flows through config['configurable']['mcp_auth'].
        result = await arun(
            graph,
            input={"node_id": "REVIEW-1"},
            config={"configurable": {"mcp_auth": operator}},
        )
        brief: ResearchBrief = result["research_result"]
        tool_log: list[ToolInteraction] = result["research_tool_log"]

        # Read which identity each run's tools carried, straight from the tool log.
        # typed_result IS a client model (CrmSearchResult / KbResult), so .acting_as
        # is a plain attribute — no JSON unpacking.
        who = {i.tool_name: i.typed_result.acting_as for i in tool_log}
        print(f"\nRun as {operator}:")
        print(f"  brief.summary : {brief.summary}")
        print(f"  tools carried : {who}")
        assert all(v == operator for v in who.values()), f"identity mismatch: {who}"

        # TYPED RESULT: crm_search's hits are already Deal models on the typed channel.
        search = next(i for i in tool_log if i.tool_name == "crm_search")
        deal = search.typed_result.hits[0]
        print(f"  typed Deal    : {deal!r}")
        assert deal.id == "D1"

    print("\nNote: the research agent has NO update_deal in tools= — it structurally")
    print("cannot mutate. announce_tool_budget=True told the model its call budgets.")


async def demo_raw_passthrough_and_lint() -> None:
    """Beat 2: raw BaseTool passthrough + the lint async-driver guard."""
    print("\n" + "=" * 66)
    print("BEAT 2: raw BaseTool passthrough + lint tool_requires_async_driver")
    print("=" * 66)

    # A battery factory yields a real MCP BaseTool; hand it straight to a node.
    # (Fetched once with a static service token — identity is not the point here.)
    raw_search = await FACTORIES["crm_search"]({"configurable": {"mcp_auth": "service-account"}}, None)
    print(f"\nraw tool           : {raw_search.name} (async-only: coroutine set, func={raw_search.func})")

    @node(mode="agent", outputs=ResearchBrief, model="research", prompt="research/brief", tools=[raw_search])
    def raw_research() -> ResearchBrief:  # body unused
        ...

    raw_pipeline = construct_from_functions("raw-passthrough", [raw_research])

    # lint() flags the async-only MCP tool up front: a sync run() cannot drive it.
    issues = [i for i in lint(raw_pipeline) if i.kind == "tool_requires_async_driver"]
    for issue in issues:
        print(f"lint flags         : [{issue.kind}] {issue.message}")
    assert issues, "expected lint to flag the async-only MCP tool"

    # Sync run() would raise ConfigurationError('use arun()'); arun() drives it fine:
    #     run(graph, input=...)          # -> ConfigurationError: async-only tool
    graph = compile(raw_pipeline, llm_factory=llm_factory, prompt_compiler=prompt_compiler)
    result = await arun(graph, input={"node_id": "REVIEW-1"}, config={"configurable": {"mcp_auth": "service-account"}})
    print(f"arun() result      : {result['raw_research'].summary}")


async def demo_gated_mutation() -> None:
    """Beat 4: gate_tools_when pauses before the mutation — approve vs deny legs."""
    print("\n" + "=" * 66)
    print("BEAT 4: gated mutation — pause before update_deal, then approve / deny")
    print("=" * 66)

    graph = compile(
        action_pipeline,
        llm_factory=llm_factory,
        prompt_compiler=prompt_compiler,
        tool_factories={k: FACTORIES[k] for k in MUTATOR_TOOLS},
        checkpointer=MemorySaver(),  # required for interrupt/resume
    )

    for leg, decision in (("APPROVE", True), ("DENY", False)):
        config = {"configurable": {"thread_id": f"deal-{leg}", "mcp_auth": "operator-A"}}

        # First pass: the gate pauses BEFORE update_deal fires.
        paused = await arun(graph, input={"node_id": "REVIEW-1"}, config=config)
        assert "__interrupt__" in paused, "gate did not pause before the mutation"
        payload = paused["__interrupt__"][0].value
        print(f"\n{leg} leg — paused at gate: {payload}")

        # Resume with the human decision.
        result = await arun(graph, resume={"approved": decision}, config=config)
        outcome: ActionOutcome = result["apply_action"]
        print(f"  resumed with approved={decision} -> status={outcome.status!r}: {outcome.note}")
        if decision:
            assert outcome.status == "applied"
        else:
            # DENY: update_deal never ran; the agent was told why and finalized.
            assert outcome.status == "blocked"


async def main() -> None:
    await demo_research_identity_and_typed_results()
    await demo_raw_passthrough_and_lint()
    await demo_gated_mutation()
    print("\n" + "=" * 66)
    print("All beats passed. One MCP client, per-node least privilege, per-run")
    print("identity, and a human gate in front of every mutation.")
    print("=" * 66)


if __name__ == "__main__":
    asyncio.run(main())


# ─────────────────────────────────────────────────────────────────────────────
# APPENDIX A — the hand-rolled factory (the override / escape hatch)
# ─────────────────────────────────────────────────────────────────────────────
# mcp_tool_factories() is an OVERRIDABLE default (the DefaultPromptCompiler
# precedent): the seam is compile(tool_factories=...), and the battery is one
# convenient way to fill it. When you want house rules, hand-roll the factory —
# it is a plain async callable (config, tool_config) -> tool that OWNS its client:
#
#     from langchain_mcp_adapters.client import MultiServerMCPClient
#
#     def my_crm_search_factory(config, tool_config):
#         async def _factory(config, tool_config):
#             token = config["configurable"].get("mcp_auth", "anon")
#             client = MultiServerMCPClient({"crm": {
#                 "transport": "stdio", "command": sys.executable, "args": [DEMO_SERVER]}})
#             tools = await client.get_tools(server_name="crm")
#             tool = next(t for t in tools if t.name == "crm_search")
#             # stdio has no headers, so bind identity as the echoed `token` arg:
#             original = tool.coroutine
#             async def with_identity(**kw): return await original(**{**kw, "token": token})
#             return tool.model_copy(update={"coroutine": with_identity})
#         return _factory
#
#     compile(pipeline, tool_factories={"crm_search": my_crm_search_factory(None, None)})
#
# The client is created and OWNED inside the factory — neograph core never holds
# an MCP session (the nmb2 session-ownership invariant).
#
# ─────────────────────────────────────────────────────────────────────────────
# APPENDIX B — production auth over streamable-http (httpx.Auth, NOT stdio)
# ─────────────────────────────────────────────────────────────────────────────
# Over stdio, per-run identity rides as a tool ARGUMENT (above). In production
# against a streamable-http MCP server, identity rides as a bearer header minted
# PER REQUEST by an httpx.Auth — the adapter's Connection.auth is typed httpx.Auth
# and its auth_flow() runs once per tools/call. Swap StdioServer for HttpServer
# and supply the auth; nothing else changes:
#
#     import httpx
#     from langchain_mcp_adapters.client import MultiServerMCPClient
#
#     class PerRunBearer(httpx.Auth):
#         def __init__(self, provider, run_ctx): self._p, self._rc = provider, run_ctx
#         async def async_auth_flow(self, request):
#             request.headers["Authorization"] = f"Bearer {await self._p(self._rc)}"  # per request
#             yield request
#
#     async def crm_tools_factory(config, tool_config):
#         cfg = config["configurable"]
#         auth = PerRunBearer(cfg["mcp_auth"]["crm"], cfg["run_ctx"])
#         client = MultiServerMCPClient({"crm": {
#             "transport": "streamable_http", "url": CRM_URL, "auth": auth}})
#         tools = await client.get_tools()   # stateless: fresh session per call, auth per request
#         return next(t for t in tools if t.name == "crm_search")
#
# neograph never parses the token — it lives in the adapter's httpx client, never
# in neograph state, the checkpoint, or the schema fingerprint. The MCP SDK ships
# OAuthClientProvider / ClientCredentialsOAuthProvider / PrivateKeyJWTOAuthProvider
# (all httpx.Auth) for the OAuth 2.1 / on-behalf-of cases.
#
# mcp 1.28.x nuance: the old streamablehttp_client(url, headers=, auth=) is
# deprecated; auth now configures the httpx.AsyncClient the transport uses. The
# adapter absorbs this, so Connection.auth: httpx.Auth stays stable across the
# SDK 1.x line — which is exactly why session lifecycle and per-request auth are
# the adapter's job, not neograph's.
