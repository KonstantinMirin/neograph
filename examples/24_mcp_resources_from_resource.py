"""Example 24: FromResource mechanics — the Typed Resource Manifest with Ephemeral Hydration.

The pattern this example NAMES: an agent acquires large corpora in one step (its
tool results carry MCP ``resource_link`` blocks), and later nodes consume those
corpora SELECTIVELY — a checkpoint and a human pause later — without the bytes
ever touching the bus. Typed references travel; blobs are fetched on demand, by
the one node that needs them, and re-derived if the link died during the pause.

The story is a CRM deal-research pipeline:

    research (agent) --> gate (pauses for a human) --> parse_activity --> consume (think)

Everything runs OFFLINE and with ZERO API keys: the LLMs are fakes, but the MCP
layer is REAL — the demo server (``examples/_mcp_demo_server.py``) is spawned as a
stdio subprocess and answers the actual Model Context Protocol. The pause/resume
rides a real file-backed SqliteSaver so the manifest genuinely survives a
checkpoint.

The five beats, in narrative order:

  1. TYPED DOMAIN READERS — ``resource_reader`` turns a URI template + output model
     into a typed tool. The agent queries a FRACTION of the email corpus and gets
     back an ``EmailPage``, not an untyped ``read_resource(uri) -> bytes`` blob.
  2. THE MANIFEST — the agent's ``get_deal`` result carries ``resource_link`` blocks;
     neograph lifts them into a checkpointed ``ResourceRef`` manifest. We read it
     off the bus via ``aget_state`` — refs on the bus, blobs never.
  3. FromResource ACROSS NODE SHAPES — a deterministic node hydrates by templated
     URI; a think node hydrates the manifest ref straight into its prompt.
  4. SELF-HEALING HYDRATION — we arm a resource expiry DURING the human pause. On
     resume the email read fails; neograph replays the producing call (the idempotency
     gate is satisfied), re-reads, and the pipeline completes.
  5. PER-RUN FETCH CACHE — a fetch is cached on the framework-minted run id, so a
     ref read twice in one run hits the server once; a resume mints a fresh run id,
     so the same ref is re-fetched (invalidate-on-resume, by construction).

Run (needs the mcp-examples extra; no keys, no network beyond the local subprocess):

    uv run --extra dev --extra mcp-examples python examples/24_mcp_resources_from_resource.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from neograph import (
    FromInput,
    FromResource,
    arun,
    compile,
    node,
    resource_reader,
)
from neograph.naming import field_name_for

# The real FastMCP demo server (spawned as a stdio subprocess). Both this example
# and example 23 share it; only its LLMs are ever faked, never the protocol.
DEMO_SERVER = Path(__file__).resolve().parent / "_mcp_demo_server.py"
DEAL_ID = "D1"

# One stable expiry-marker path, shared by every demo-server subprocess spawn (the
# read and the later arming call live in different subprocesses, so an in-memory
# flag would not survive). Cleared at the start of main().
_STATE_MARKER = Path(tempfile.gettempdir()) / "neograph_ex24_expiry.marker"


# ── Schemas ──────────────────────────────────────────────────────────────────


class DealSummary(BaseModel, frozen=True):
    """What the research agent produces — a tiny structured summary. The heavy
    corpora it touched stay behind as ResourceRefs, not inlined here."""

    deal_id: str
    headline: str
    recent_email_subjects: list[str]


class EmailPage(BaseModel, frozen=True):
    """Typed result of the ``read_emails`` reader — a date-range FRACTION of the
    email corpus. Contrast a generic ``read_resource(uri) -> bytes`` blob: the
    agent gets fields it can reason over, and ``ToolInteraction.typed_result``
    carries this model, not a repr string."""

    deal_id: str
    start: str
    end: str
    emails: list[dict]


class ActivityHistory(BaseModel, frozen=True):
    """Activity-history corpus, hydrated by a TEMPLATED FromResource(uri=...)."""

    deal_id: str
    events: list[dict]


class EmailHistory(BaseModel, frozen=True):
    """Email-history corpus, hydrated from the MANIFEST by FromResource(ref=...)."""

    deal_id: str
    start: str = ""
    end: str = ""
    emails: list[dict] = []


class GateAck(BaseModel, frozen=True):
    """Trivial output of the human-gate node — its only job is to pause."""

    reviewed_deal: str


class Brief(BaseModel, frozen=True):
    """The final consume-step output (a think node)."""

    text: str


# ── Beat 5 plumbing: an instrumented fetcher that counts reads per run ─────────
#
# The shipped battery's fetcher does the real MCP read. We wrap it to COUNT reads,
# keyed by the framework-minted run id, so beat 5 can show the per-run cache: a ref
# read within one run hits the server once; a resume (fresh run id) re-reads.

_FETCH_LOG: list[tuple[str, str]] = []  # (run_id, uri)


def _counting_fetcher(inner):
    async def fetch(uri: str):
        _FETCH_LOG.append((_CURRENT_RUN_ID[0], str(uri)))
        return await inner(uri)

    return fetch


# The run id is a config-only key the runner mints; we mirror it into a module box
# purely so the demo's fetch log can attribute a read to its run (the fetch seam
# itself carries no config). Real code would not need this — it is beat-5 telemetry.
_CURRENT_RUN_ID = ["run-1"]


# ── The manifest-emitting tool ────────────────────────────────────────────────
#
# get_deal returns resource_link blocks — the manifest neograph lifts. We call the
# REAL server's get_deal over a raw MCP session and return the blocks as dicts with
# STRING uris. Why not bind it via mcp_tool_factories (the tool battery)? Because
# langchain-mcp-adapters rewrites resource_link blocks to type:'file', which the
# lift does not recognize; and the raw block's uri is a pydantic AnyUrl the lift
# cannot scan. Preserving raw resource_link blocks client-side is exactly what the
# shipped resource REPLAYER does for the same reason — see neograph_mcp._client.


def _stdio_connection(state_marker: Path) -> dict:
    """The StdioConnection dict that spawns the demo server as a subprocess. The
    expiry marker path is pinned via env so every spawn (the read and the later
    arming call live in different subprocesses) agrees on it."""
    return {
        "command": sys.executable,
        "args": [str(DEMO_SERVER)],
        "transport": "stdio",
        "env": {"NEOGRAPH_MCP_DEMO_STATE": str(state_marker)},
    }


def _make_get_deal_tool():
    """A typed, async ``get_deal`` tool that preserves raw resource_link blocks.

    Marked idempotent=True (read-only) so a lifted ref's producing call is
    replay-eligible — the hard gate the self-heal beat depends on."""
    from langchain_core.tools import StructuredTool
    from langchain_mcp_adapters.client import MultiServerMCPClient

    async def _get_deal(deal_id: str, token: str = "anon") -> list:
        client = MultiServerMCPClient({"crm": _stdio_connection(_STATE_MARKER)})
        async with client.session("crm") as session:
            result = await session.call_tool("get_deal", {"deal_id": deal_id, "token": token})
        # Preserve resource_link blocks with STRING uris (the shape the lift scans).
        blocks: list = []
        for block in result.content:
            btype = getattr(block, "type", None)
            if btype == "resource_link":
                blocks.append(
                    {
                        "type": "resource_link",
                        "uri": str(block.uri),
                        "name": block.name,
                        "mimeType": getattr(block, "mimeType", None),
                    }
                )
            elif btype == "text":
                blocks.append({"type": "text", "text": block.text})
        return blocks

    return StructuredTool.from_function(
        coroutine=_get_deal,
        name="get_deal",
        description="Fetch a CRM deal; returns a summary plus resource_link refs to its corpora.",
        metadata={"ng_idempotent": True},
    )


# ── Beat 1: a typed domain reader (a fraction query, not a blob read) ──────────

read_emails = resource_reader(
    "read_emails",
    uri_template="mcp://crm/deals/{deal_id}/emails/{start}/{end}",
    output_model=EmailPage,
    description="Read a date-range fraction of a deal's email history.",
    idempotent=True,  # read-only -> replay-safe; the default, spelled out here
)

# The manifest-emitting tool, built once at module scope so it wires through the
# @node(tools=[...]) decorator exactly like any other raw BaseTool (auto-registered
# at compile()).
get_deal_tool = _make_get_deal_tool()


# ── Fake LLMs (the only fakes; replace with a real BaseChatModel in production) ─


class _ResearchAgentLLM:
    """Agent-mode fake: call get_deal (emits the manifest), then read a fraction
    of the emails, then answer. The async ReAct loop drives ainvoke turn by turn."""

    _ANSWER = DealSummary(
        deal_id=DEAL_ID,
        headline="Acme renewal in negotiation; recent pricing + renewal threads.",
        recent_email_subjects=["Pricing questions", "Renewal terms"],
    )

    def __init__(self) -> None:
        self._turn = 0
        self._structured = False

    def bind_tools(self, tools: list) -> _ResearchAgentLLM:
        return self

    def with_structured_output(self, model: type, **kw) -> _ResearchAgentLLM:
        clone = _ResearchAgentLLM()
        clone._turn = self._turn
        clone._structured = True
        return clone

    async def ainvoke(self, messages: list, **kw):
        if self._structured:
            return self._ANSWER
        self._turn += 1
        if self._turn == 1:
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": "get_deal", "args": {"deal_id": DEAL_ID, "token": "op-A"}, "id": "t1"}]
            return msg
        if self._turn == 2:
            msg = AIMessage(content="")
            msg.tool_calls = [
                {
                    "name": "read_emails",
                    "args": {"deal_id": DEAL_ID, "start": "2024-04-01", "end": "2024-12-31"},
                    "id": "t2",
                }
            ]
            return msg
        return AIMessage(content=self._ANSWER.model_dump_json())


class _ConsumeThinkLLM:
    """Think-mode fake: turns the prompt (which already carries the manifest-
    hydrated email history) into a one-line Brief. Reflecting the prompt back lets
    the demo SHOW that the healed corpus actually reached the model."""

    def __init__(self) -> None:
        self._structured = False

    def with_structured_output(self, model: type, **kw) -> _ConsumeThinkLLM:
        clone = _ConsumeThinkLLM()
        clone._structured = True
        return clone

    async def ainvoke(self, messages: list, **kw):
        last = messages[-1] if messages else {}
        prompt = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
        brief = Brief(text=f"Deal brief drafted from hydrated corpus. {prompt}")
        if self._structured:
            return brief
        return AIMessage(content=brief.model_dump_json())


def _llm_factory(tier: str):
    if tier == "consume":
        return _ConsumeThinkLLM()
    return _ResearchAgentLLM()


def _prompt_compiler(template, data, *, di_inputs=None, **kw):
    """Template-ref compiler. For the consume node it folds the manifest-hydrated
    email history (a FromResource di_input) straight into the prompt text — the
    async di_inputs path awaits the fetch before the prompt is built."""
    di_inputs = di_inputs or {}
    history = di_inputs.get("history")
    dossier = di_inputs.get("dossier")
    if history is not None:
        subjects = [e.get("subject") for e in history.emails]
        content = f"Write a one-line brief. Recent email subjects: {subjects}"
    elif dossier is not None:
        content = f"Research the deal. Activity so far: {len(dossier.events)} events."
    else:
        content = "Write a one-line brief."
    return [{"role": "user", "content": content}]


# ── The pipeline nodes ────────────────────────────────────────────────────────


@node(
    mode="agent",
    outputs=DealSummary,
    model="research",
    prompt="research/scan",
    tools=[get_deal_tool, read_emails],
)
def research(
    deal_id: Annotated[str, FromInput],
    dossier: Annotated[ActivityHistory, FromResource(uri="mcp://crm/deals/{deal_id}/activity")],
) -> DealSummary:
    # Body unused: the fake LLM drives the ReAct loop. get_deal emits the manifest
    # resource_links; read_emails queries a typed fraction of the email corpus.
    # ``dossier`` is a FromResource on an AGENT node: it is fetched once and cached
    # across every ReAct superstep of this run (beat 5), then folded into the prompt.
    ...


@node(
    outputs=GateAck,
    interrupt_when=lambda state: (
        {"deal": state.research.deal_id, "message": "Approve deal research before consuming corpora?"}
        if state.research is not None
        else None
    ),
)
def gate(research: DealSummary) -> GateAck:
    """A human-in-the-loop pause between acquiring the manifest and consuming it —
    the real checkpoint boundary the refs must survive (and where expiry strikes)."""
    return GateAck(reviewed_deal=research.deal_id)


@node(outputs=ActivityHistory)
async def parse_activity(
    gate: GateAck,
    deal_id: Annotated[str, FromInput],
    activity: Annotated[ActivityHistory, FromResource(uri="mcp://crm/deals/{deal_id}/activity")],
) -> ActivityHistory:
    """Beat 3(a): a DETERMINISTIC node hydrates a templated URI. JSON -> model via
    model_validate_json, before the body runs. No manifest needed — the URI shape
    is known at author time."""
    return activity


@node(
    mode="think",
    outputs=Brief,
    model="consume",
    prompt="consume/brief",
)
async def consume(
    parse_activity: ActivityHistory,
    history: Annotated[EmailHistory, FromResource(ref="email-history", max_bytes=200_000)],
) -> Brief:
    """Beat 3(b): a THINK node hydrates the manifest ref straight into its prompt.
    The body never runs (LLM mode); ``history`` reaches the prompt via the async
    di_inputs path. This is the node the self-heal beat protects."""
    ...


def _build_pipeline():
    import types

    from neograph import construct_from_module

    mod = types.ModuleType("ex24_mod")
    mod.research = research
    mod.gate = gate
    mod.parse_activity = parse_activity
    mod.consume = consume
    return construct_from_module(mod, name="deal-research")


# ── Driver ────────────────────────────────────────────────────────────────────


async def _arm_expiry() -> None:
    """Arm the demo server's one-shot email-history expiry — simulating a link that
    died during the human pause."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({"crm": _stdio_connection(_STATE_MARKER)})
    async with client.session("crm") as session:
        await session.call_tool("arm_email_expiry", {})


def _quiet_logs() -> None:
    """Keep the demo output readable — drop neograph's INFO node-lifecycle logs."""
    import logging

    import structlog

    logging.getLogger().setLevel(logging.WARNING)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


async def main() -> None:
    from neograph_mcp import StdioServer, mcp_resource_fetcher

    _quiet_logs()
    _STATE_MARKER.unlink(missing_ok=True)  # start un-armed
    db_path = Path(tempfile.mktemp(suffix="_ex24_checkpoints.sqlite"))

    server = StdioServer(
        command=sys.executable,
        args=[str(DEMO_SERVER)],
        env={"NEOGRAPH_MCP_DEMO_STATE": str(_STATE_MARKER)},
    )
    # FEATURED WIRING: the shipped [mcp] battery hands us (fetcher, replayer) to drop
    # straight into config. Self-healing hydration then works without hand-rolling a
    # session — the replayer re-invokes the producing call through the same client.
    fetcher, replayer = mcp_resource_fetcher({"crm": server})
    fetcher = _counting_fetcher(fetcher)  # beat-5 telemetry only

    pipeline = _build_pipeline()

    config_base = {
        "mcp_resource_fetcher": fetcher,
        "mcp_resource_replayer": replayer,
    }

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        graph = compile(
            pipeline,
            checkpointer=saver,
            llm_factory=_llm_factory,
            prompt_compiler=_prompt_compiler,
        )
        thread = {"configurable": {"thread_id": "deal-D1", "deal_id": DEAL_ID, **config_base}}

        # ── Run 1: research + manifest lift, then pause at the human gate ──────
        _CURRENT_RUN_ID[0] = "run-1"
        print("=" * 70)
        print("RUN 1 — research the deal, then pause for human review")
        print("=" * 70)
        result = await arun(graph, input={"node_id": "n1", "deal_id": DEAL_ID}, config=thread)

        summary = result.get("research")
        print("\n[beat 1] research agent produced a typed summary:")
        print(f"         headline: {summary.headline}")
        print(f"         recent email subjects (from a typed EmailPage fraction): {summary.recent_email_subjects}")

        assert "__interrupt__" in result, "expected the human gate to pause the run"
        print(f"\n[gate]   paused for review: {result['__interrupt__'][0].value['message']}")

        # ── Beat 2: the manifest on the bus (refs, never blobs) ───────────────
        state = await graph.aget_state(thread)
        manifest_field = f"neo_resource_manifest_{field_name_for('research')}"
        manifest = state.values.get(manifest_field, [])
        print(f"\n[beat 2] resource manifest lifted onto the '{manifest_field}' channel:")
        for ref in manifest:
            print(f"         - kind={ref.kind!r} uri={ref.uri}")
            print(
                f"           producing_call={ref.producing_call.tool_name}({ref.producing_call.args})"
                f" idempotent={ref.producing_call.producer_idempotent}"
            )
        assert any(r.kind == "email-history" for r in manifest), "email-history ref missing from manifest"

        # ── Beat 4 setup: the link dies during the pause ──────────────────────
        print("\n[beat 4] arming email-history expiry DURING the pause (the link the resume will need dies)...")
        await _arm_expiry()

        # ── Resume: parse_activity + consume; consume self-heals the expiry ───
        _CURRENT_RUN_ID[0] = "run-2"
        print("\n" + "=" * 70)
        print("RUN 2 — resume: hydrate corpora, self-heal the expired link, consume")
        print("=" * 70)
        result = await arun(graph, resume={"approved": True}, config=thread)

        activity = result.get("parse_activity")
        print("\n[beat 3a] parse_activity hydrated a templated FromResource(uri=...):")
        print(f"          activity events: {len(activity.events)}")

        brief = result.get("consume")
        print("\n[beat 3b/4] consume hydrated FromResource(ref='email-history') into its prompt,")
        print("            self-healed the armed expiry (replay -> re-read), and answered:")
        print(f"            brief: {brief.text}")

        # ── Beat 5: the per-run fetch cache ───────────────────────────────────
        # research is an AGENT — it ran several ReAct supersteps (get_deal, then
        # read_emails, then answer), each re-preparing its prompt. Yet its dossier
        # FromResource was fetched ONCE: the awaited fetch is cached on the
        # framework-minted run id, stable across supersteps. The resume mints a
        # FRESH run id, so the same activity resource is re-fetched — the cache is
        # invalidate-on-resume by construction (it must never serve a pre-pause
        # value into a post-resume run).
        r1_activity = sum(1 for r, u in _FETCH_LOG if r == "run-1" and u.endswith("/activity"))
        r2_activity = sum(1 for r, u in _FETCH_LOG if r == "run-2" and u.endswith("/activity"))
        print("\n[beat 5] the research agent ran several ReAct supersteps, yet its dossier")
        print(f"         FromResource was fetched {r1_activity}x in run-1 (cached on the run id).")
        print(f"         The resume mints a fresh run id -> re-fetch (run-2 activity reads: {r2_activity}).")
        assert r1_activity == 1, f"expected the dossier fetch to be cached across supersteps, got {r1_activity}"

    print("\nDone. Refs traveled on the bus; blobs were fetched on demand and re-derived on expiry.")


if __name__ == "__main__":
    asyncio.run(main())
