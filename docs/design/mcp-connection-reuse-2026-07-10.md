# MCP connection reuse across supersteps — deep research + spike (2026-07-10)

Context: neograph-cnxlx (cross-superstep MCP connection reuse with transparent
reconnect). The agent/act ReAct loop runs each superstep as a distinct asyncio
task; today each superstep re-invokes the tool factory, which rebuilds a
`MultiServerMCPClient`, and the adapter's stateless per-call session model
reconnects on every tool invocation.

Two evidence sources: (1) a local spike against `examples/_mcp_demo_server.py`,
(2) a 100-agent deep-research run (5 search angles, 15 sources fetched, every
claim 3-vote adversarially verified against primary sources — GitHub API,
framework source code, official docs).

## Local spike result

Open an adapter persistent session (`client.session()`) in task A, then:

| Operation | From a different task | Result |
|---|---|---|
| `session.call_tool(...)` | task B | **works** (returned normally) |
| `cm.__aexit__(...)` (close) | task C | **raises** `RuntimeError: Attempted to exit cancel scope in a different task than it was entered in` |

So the task-affinity constraint binds **disposal only**, not calls. Any reuse
design must guarantee open and close happen in the same task; calls may come
from any task.

## Research findings (all adversarially verified; confidence in brackets)

1. **[high] Root cause confirmed and unresolved upstream.** The MCP Python
   SDK's `ClientSession`/transports use anyio task groups whose cancel scopes
   bind to the creating task; exiting from a different task (or closing
   sessions out of LIFO order) deterministically raises the RuntimeError. SDK
   maintainer (felixweinberger) attributed it to anyio internals, said the SDK
   is unlikely to move off anyio, and closed the issues with no recommended
   cross-task lifecycle pattern.
   Sources: modelcontextprotocol/python-sdk#922, #79; agronholm/anyio
   discussion #937.

2. **[high] The only working single-task pattern in SDK threads**: hold one
   async-with (AsyncExitStack or session context) open in a single long-lived
   task around the entire loop — open, list_tools, N calls, close, all in the
   same task. Splitting enter/exit across tasks reproduces the error;
   `exit_stack.pop_all()` does not help. Source: python-sdk#79 (tested code,
   verified verbatim via GitHub API).

3. **[high] langchain-mcp-adapters is stateless by default — per-call
   reconnect is the documented, intentional default**, positioned as
   acceptable for stateless tools ("By default, MultiServerMCPClient is
   stateless: each tool invocation creates a fresh MCP session, executes the
   tool, and then cleans up"). Persistence is explicit opt-in: caller holds
   `async with client.session(server_name)` open and binds tools to that
   session; no background owner-task mode is offered. Persistent mode is
   documented as needed specifically for STATEFUL servers.
   Sources: docs.langchain.com/oss/python/langchain/mcp; adapters README.

4. **[high] OpenAI Agents SDK = run-lived owner/worker task.**
   `MCPServerManager` is an async CM (connect_all on enter, cleanup_all in
   reverse order on exit, same-task guaranteed), and parallel connects use a
   dedicated worker task per server with a command queue — source comment:
   "breaks libraries that require connect/cleanup in the same task (e.g. AnyIO
   cancel scopes)". Per-run `list_tools` latency is mitigated with an opt-in
   `cache_tools_list`, not by reconnecting.
   Sources: openai.github.io agents docs (mcp/manager); src/agents/mcp/manager.py.

5. **[high] Pydantic AI = caller-held with auto-reconnect fallback.**
   `async with agent` opens connections to all MCP toolsets around the run;
   skipping the CM falls back to per-need open/close (implicit reconnects
   accepted). Its refcounted enter/exit broke under parallel usage exactly on
   cancel-scope task affinity (#2818); lead maintainer DouweM's workaround:
   "make sure the server (or agent) is entered explicitly around the entire
   context where it's used ... entered first and exited last in the same
   place." General fix later via PR #4514.
   Sources: pydantic-ai docs mcp/client.md; pydantic/pydantic-ai#2818.

6. **[medium] The actor pattern exists in production** where calls originate
   from arbitrary tasks: Hermes Agent (NousResearch) runs each MCP server as a
   long-lived asyncio.Task on a background loop, holding the transport's
   async-with open for the whole session; shutdown is signalled so the exit
   happens in the opening task — source docstring cites the anyio requirement
   explicitly. (Blog entry-point, but code-verified.)
   Sources: NousResearch/hermes-agent tools/mcp_tool.py.

7. **[low, blog-quality] Cost anchor**: a fresh npx-launched Node stdio server
   costs ~2.9s per call (spawn + init + handshake) vs ~119ms median on a
   persistent session (~25x). Specific to npx/Node; pre-installed binaries and
   Python stdio servers spawn much faster; the initialize round-trip itself is
   a small fraction. n=10, single machine — do not generalize the constant.

8. **[high] Synthesis for neograph**: per-superstep reconnection is defensible
   mainstream practice (it IS the adapter default) — the current behavior is
   not wrong; it forfeits stateful servers and pays spawn cost per call on
   stdio. The caller-held async-with works iff the session context is held
   around the entire graph invocation in one task. Where calls must cross task
   boundaries AND the hold-open cannot be arranged, the run-lived owner task is
   the only proven pattern. No maintainer-blessed third option exists.

## Implication for neograph (the design insight)

The spike shows calls cross tasks fine — only disposal is task-affine. The
pydantic-ai maintainer's workaround ("entered first and exited last in the same
place") maps directly onto neograph's shape: the CONSUMER's task that calls
`arun()`/`astream()` is a single long-lived task spanning every superstep.

So the natural neograph pattern is **consumer-held, run-scoped session** — a
battery async CM the consumer opens AROUND the run and threads via config:

```python
async with mcp_run_context(servers, token_provider=...) as mcp_ctx:
    result = await neograph.arun(graph, input=..., config=mcp_ctx.config(base_config))
```

- Open + close both happen in the consumer's task (safe disposal; matches
  finding 2/5's only-working pattern).
- Supersteps' tool calls ride the held session from their own tasks (proven
  safe by the spike).
- nmb2 preserved verbatim: the client/session is CONSUMER-owned (stronger than
  today — the consumer literally holds the CM); neograph core untouched.
- Transparent reconnect: when the config carries no held session (consumer
  didn't opt in) the factory falls back to today's per-call stateless mode —
  which is the adapter's own documented default. A resumed run (fresh process)
  simply runs without the CM or re-enters it — cache-miss = reconnect for free.
- No actor/queue machinery needed (that's the fallback if a framework cannot
  arrange a hold-open task; neograph CAN — the runner's caller task).

Trade-off vs the OpenAI-style owner task: the consumer must remember the CM to
get reuse (opt-in, like langchain/pydantic-ai); an owner task would make reuse
automatic but adds a background-task subsystem, queue marshaling, and lifecycle
edge cases the ecosystem only reaches for when a hold-open task cannot exist.

## Recommendation

Implement cnxlx as the consumer-held `mcp_run_context` battery CM (opt-in
persistent sessions bound into the tool factories via config), keeping the
stateless per-call path as the default and the transparent-reconnect fallback.
Defer any owner-task/actor design unless a concrete consumer surfaces that
cannot hold a CM around its run.
