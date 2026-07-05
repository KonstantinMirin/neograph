# Adversarial architect review — epic neograph-nmb2 ("neograph-owned MCP session + mcp_servers= typed config")

## VERDICT: (c) — kill nmb2. It reinvents the ecosystem's abstraction.

nmb2 proposes neograph OWN the MCP session lifecycle (typed `mcp_servers=` config, a two-lifetime session factory, disposal on cancellation) so it has "something to attach" the per-run token to. The premise is false on two independent counts:

1. **There is no long-lived MCP session for neograph to own.** In current `langchain-mcp-adapters`, `client.get_tools()` without an explicit `client.session(...)` context opens a **fresh session per tool call** (stateless mode, via `create_session`). The adapter already owns session lifecycle — open-per-call and dispose-per-call — so there is nothing to hold, cache, or dispose at the neograph layer. nmb2's "disposal on cancellation" obligation is a solution to a problem the adapter doesn't create.

2. **The typed multi-server config nmb2 wants already exists** as the adapter's `Connection` TypedDict (`transport` + `url`/`headers`/`auth` for `streamable_http`/`sse`, `command`/`args`/`env` for `stdio`), and the per-run auth hook attaches **natively** via `Connection.auth: httpx.Auth`, whose `auth_flow()` runs **per-request** — exactly the "mint a fresh, audience-bound token per call" cadence w74k.3.1 specifies. The MCP SDK even ships the providers (`OAuthClientProvider`, `ClientCredentialsOAuthProvider`, `PrivateKeyJWTOAuthProvider` — all `httpx.Auth`) for the on-behalf-of / machine-to-machine case.

w74k.3.1 is satisfied by **one small neograph change** — allow `async def` tool factories on the `arun()` path (dual-path factory invocation, the same H2 "driver selects" pattern used everywhere else) — **plus the ecosystem's native auth**. No `mcp_servers=` config, no owned session, no disposal machinery. The "sync-factory wall" the executor hit is a single un-twinned line, not a missing foundation.

This is the exact "reinventing runtime" disease the three-layer-principle doc was written to prevent: **when the ecosystem already offers an abstraction, USE it, never rebuild it.** Session lifecycle and per-request auth are Layer-3 (engine/adapter) concerns. nmb2 pulls them into neograph.

---

## 1. What the ecosystem already owns (findings, version-pinned)

Repo env (`uv pip list`): `langchain-mcp-adapters` and the `mcp` SDK are **not installed** — neograph has zero MCP runtime dependency today, which is itself evidence that the happy path ("load tools ahead of compile, register a factory") needs no neograph MCP code. Findings below are pinned to **`langchain-mcp-adapters 0.3.0`** and **`mcp 1.28.1`** (latest on PyPI as of this review) and verified against the adapter 0.3.0 wheel source directly (`langchain_mcp_adapters/{sessions,tools,client}.py`), cross-checked via deepwiki against `langchain-ai/langchain-mcp-adapters` and `modelcontextprotocol/python-sdk`.

### 1.1 MultiServerMCPClient = nmb2's proposed config surface, already shipped

The `Connection` TypedDict (union of `StdioConnection` / `SSEConnection` / `StreamableHttpConnection` / `WebsocketConnection`) accepts per server:

| transport | fields |
|---|---|
| `stdio` | `command`, `args`, `env`, `cwd`, `encoding`, `session_kwargs` |
| `streamable_http` | `url`, `headers`, `timeout`, `sse_read_timeout`, `terminate_on_close`, `httpx_client_factory`, **`auth: httpx.Auth`**, `session_kwargs` |
| `sse` | `url`, `headers`, `timeout`, `sse_read_timeout`, `httpx_client_factory`, **`auth: httpx.Auth`**, `session_kwargs` |
| `websocket` | `url`, `session_kwargs` |

This is nmb2's "mcp_servers= typed config (transport: HTTP->bearer, stdio->env)" — the transport split (`headers`/`auth` for HTTP, `env` for stdio) is exactly the adapter's shape. neograph would be re-typing a config schema the adapter already ships. **Source (adapter 0.3.0 `sessions.py`):** `StdioConnection` L81, `SSEConnection` L128 (`auth: NotRequired[httpx.Auth]` L159), `StreamableHttpConnection` L163 (`auth: NotRequired[httpx.Auth]` L190), `WebsocketConnection` L194, `Connection` union L206.

### 1.2 THE LOAD-BEARING FACT — stateless session-per-call is the default

`client.get_tools()` (no explicit `client.session(server)` context) → `load_mcp_tools` → `convert_mcp_tool_to_langchain_tool` with **no session** → each tool execution calls `create_session` to open a **fresh session, run one `tools/call`, and close it**. Confirmed against adapter 0.3.0 source: `client.get_tools` docstring L170 states verbatim *"A new session will be created for each tool call"*; `convert_mcp_tool_to_langchain_tool` L459-465 `if session is None: async with create_session(...)`. The persistent alternative (`async with client.session(name) as s: load_mcp_tools(s)`) is opt-in.

**Consequence for nmb2:** there is **no long-lived session** in the default path. neograph "owning session creation" would mean adopting the *persistent* mode (a deliberate downgrade in isolation) purely to have a handle to stamp — but stamping is already handled per-request by `auth` (§1.3). The owned-lifetime disposal obligations (nmb2's cancellation/dispose scope; the two-lifetime "cached live session invalidated on resume" rule) apply to a resource **the adapter, not neograph, manages**.

### 1.3 Per-request auth is NATIVE — `httpx.Auth` on the connection

`Connection.auth` is typed `httpx.Auth` and flows to the underlying `httpx.AsyncClient` that the streamable-http / sse transport uses. Because `httpx.Auth.auth_flow()` (or `async_auth_flow`) is invoked **once per HTTP request**, a custom `httpx.Auth` subclass mints/refreshes the bearer token per `tools/call` — the precise "provider invoked at call time, re-mintable, audience-bound" contract of w74k.3.1 §6.3.1. The MCP SDK (`mcp/client/auth.py`) ships:

- `OAuthClientProvider` — full OAuth 2.1 + PKCE, auto-refresh, `httpx.Auth`.
- `ClientCredentialsOAuthProvider`, `PrivateKeyJWTOAuthProvider` — machine-to-machine (the on-behalf-of / service-identity case), both `httpx.Auth`.

So "carry identity, never decide; audience-bound (RFC 8707); no verbatim passthrough; re-mint on resume" is delivered by an `httpx.Auth` the consumer supplies — server-side enforcement unchanged. neograph never parses the token because it never touches it; the token lives in the adapter's httpx client. The MCP SDK (mcp 1.28.1) ships:

- `OAuthClientProvider` (`mcp/client/auth/oauth2.py:217`, subclass of `httpx.Auth`, `async_auth_flow` L491) — full OAuth 2.1 + PKCE, auto-refresh.
- `ClientCredentialsOAuthProvider` / `PrivateKeyJWTOAuthProvider` / `RFC7523OAuthClientProvider` (`mcp/client/auth/extensions/client_credentials.py` L24/L194/L390) — machine-to-machine / on-behalf-of; all subclass `OAuthClientProvider`, hence all `httpx.Auth`. (These live in the `extensions` subpackage, not the top-level `mcp.client.auth` exports.)

**Deprecation nuance (important, verified — does NOT change the verdict):** in mcp 1.28.1 the old `streamablehttp_client(url, headers=..., auth=...)` is `@deprecated`; the replacement `streamable_http_client(url, *, http_client: httpx.AsyncClient | None = None, ...)` takes NO `auth`/`headers` params — you configure them on the `httpx.AsyncClient` you pass in (and `StreamableHTTPTransport.__init__` runtime-warns that direct `headers`/`auth` are ignored). **The injection point moved from the transport function to the `AsyncClient`.** The adapter absorbs this: `langchain-mcp-adapters 0.3.0` `sessions._create_streamable_http_session` builds the client via `create_mcp_http_client(headers=headers, timeout=..., auth=auth)` and passes it as `http_client=`. So `Connection.auth: httpx.Auth` still works end-to-end from the consumer's perspective; the consumer-facing wiring is stable across the SDK 1.x line. This actually *reinforces* verdict (c): the auth-injection plumbing is churning inside the ecosystem — precisely the layer neograph must NOT own, or it inherits that churn.

### 1.4 Latency

Stateless (fresh session per call): the MCP lifecycle spec (2025-06-18) mandates `initialize` request→response then a `notifications/initialized`, before normal ops — so a fresh-session `tools/call` is ~3 POSTs (initialize, initialized notification, tools/call) + connection/TLS setup if no pooling, vs 1 POST on a reused session. The spec doesn't quantify ms; the shape is ~3x round-trips per call. This is the ONLY axis on which an owned/cached session helps — and it is a **pure performance optimization that belongs to the adapter's persistent mode**, reachable by the consumer via `async with client.session(...)` inside their factory, not a correctness gap and not neograph's to build. For an I/O-bound agent doing a handful of tool calls per run, the handshake overhead is second-order.

---

## 2. The actual FR (w74k.3.1), re-derived — satisfiable with ZERO owned lifecycle

**w74k.3.1 wants:** per-run identity (on-behalf-of), token minted per run via a provider in `config['configurable']`, audience-bound, never passthrough, cached once per run, re-mintable on resume.

**Where each requirement lands under the async-factory verdict:**

| Requirement | Mechanism | Owner |
|---|---|---|
| provider in `config['configurable']['mcp_auth'][server]` | already the documented DI rail (`config={"configurable": {...}}`, same as `rate_limiter`) | neograph (config passthrough — exists) |
| minted per run, at call time | async tool factory reads provider, `await`s it, builds the per-run tool | neograph (the 1 change) + user |
| audience-bound, never passthrough, opaque | `httpx.Auth` subclass (or SDK `*OAuthProvider`) mints attenuated aud-bound token | ecosystem (`mcp/client/auth.py`) |
| re-mint on resume | factory re-invoked per superstep (already true — `_agent_cycle._build_turn_prep`), fresh process → provider re-called | neograph (already correct — the two-lifetime rule is already satisfied by per-superstep re-instantiation) |
| token never in checkpoint/fingerprint/log | token lives in the httpx client inside the adapter tool closure, never in neograph state | ecosystem + trivial guard |

### 2.1 Consumer sketch — on-behalf-of, async-factory verdict (NO neograph session)

```python
# USER writes: a per-run token provider (the w74k.3.1 hook, unchanged shape)
class PerRunBearer(httpx.Auth):
    def __init__(self, provider, run_ctx): self._p, self._rc = provider, run_ctx
    async def async_auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {await self._p(self._rc)}"  # minted per request
        yield request

async def axiom_tools_factory(config, tool_config):          # async tool factory
    cfg = config["configurable"]
    auth = PerRunBearer(cfg["mcp_auth"]["axiom"], cfg["run_ctx"])
    client = MultiServerMCPClient({"axiom": {
        "transport": "streamable_http", "url": AXIOM_URL, "auth": auth}})
    tools = await client.get_tools()          # stateless: fresh session per call, auth per request
    return tools[0]

research = Node("research", mode="agent", tools=[Tool("axiom.search")])
compile(pipeline, tool_factories={"axiom.search": axiom_tools_factory})
await arun(graph, input=..., config={"configurable": {
    "run_ctx": {"operator": "alice"}, "mcp_auth": {"axiom": mint_axiom_token}}})
```

Two operators → two `run_ctx` → two tokens, verified in scratch (`consumer_sketch.py`, `async_factory_probe.py`). **No neograph-owned session, no disposal, no `mcp_servers=` config.** A thin optional convenience helper (verdict (b) fallback) could wrap the `MultiServerMCPClient(...) + get_tools()` boilerplate, but even that owns no session — it returns adapter tools.

### 2.2 Comparison vs nmb2's owned-lifetime proposal

| Axis | Async-factory (verdict c) | nmb2 owned lifecycle |
|---|---|---|
| Surface added to neograph | 1 change: async factory support on `arun()` | `mcp_servers=` config schema, session factory, cache, disposal API |
| Cancellation/disposal | adapter closes its own per-call session; nothing to dispose | neograph must dispose live session on `CancelledError` (new obligation, new tests) |
| Checkpoint/resume (two-lifetime) | already correct — factory re-invoked per superstep, fresh process re-mints | neograph must implement "cache keyed to (process,loop), invalidate on resume" — reimplementing what per-superstep re-instantiation gives free |
| Latency | stateless per-call handshake (adapter default); persistent opt-in via user factory | neograph-cached session (faster) — but this is the ONLY win, and it's adapter-reachable |
| Failure modes | adapter/httpx surface, well-trodden | neograph owns a stateful network resource across checkpoints — the highest-risk thing the two-lifetime doc warns about |

The only thing nmb2 buys is within-run session caching (latency). That is (a) optional, (b) the adapter's persistent-session mode, (c) reachable by a consumer today. It does not justify neograph owning a stateful, checkpoint-surviving network resource.

---

## 3. The sync-factory wall — self-inflicted, one line

**Is "factories are sync" a real constraint?** No — it's a single un-twinned call.

- Factory contract: `Callable[[config, tool_config], BaseTool]` (`tool.py`, `register_tool_factory`).
- Invocation site: `_tool_loop._prepare_tool_loop` line ~275: `tool_instances[name] = factory(config, tool_spec.config)` — synchronous.
- `_prepare_tool_loop` is shared verbatim by the sync body (`_agent_cycle.agent_body`/`tools_body`) and the async body (`aagent_body`/`atools_body`). The async bodies already `await` `caller.ainvoke` and `tool_fn.ainvoke` — but the factory that BUILDS `tool_fn` is called synchronously inside the shared prep.

**Post-m6d3 the path is already twinned at the node level.** `_wiring._add_agent_cycle` registers `RunnableLambda(agent_sync, afunc=agent_async)` — LangGraph selects the async twin under `arun()`. `_build_turn_prep` (which calls `_prepare_tool_loop`) is invoked INSIDE each body, per superstep, with live `config`. So the async body is the natural place to `await` an async factory. The "factories are sync" wall is purely that `_prepare_tool_loop` has no async twin for the ONE line that instantiates tools.

**Minimal change** (mirrors the khff/H2 "driver selects, fail-loud on the wrong driver" pattern):

1. Split tool-instantiation out of the shared `_prepare_tool_loop` (or add `_aprepare_tool_loop`): the sync path calls `factory(config, tc)`; if the factory is `iscoroutinefunction` (or returns an awaitable), **raise `ConfigurationError`** ("async tool factory requires arun()") — same fail-loud shape already used at `_agent_cycle.py:352` for async-only tools under sync `run()`.
2. The async path (`aparepare` / inside `aagent_body`+`atools_body`) does `inst = factory(config, tc); inst = await inst if isawaitable(inst) else inst`.
3. Both paths keep passing `config` — the token provider is already in `config['configurable']`.

Scratch-verified in `async_factory_probe.py`: async path awaits async factory AND handles sync factories; sync path fails loud on an async-only factory.

**Does this alone dissolve "nothing to attach to"? YES.** An async factory reads `config['configurable']['mcp_auth'][server]`, awaits the provider (or builds an `httpx.Auth` that awaits it per request), and returns a per-run tool — with **zero new config surface**. The executor's "nothing to attach to" was true only because the factory couldn't be async and thus couldn't await a token broker; it was never a missing session foundation.

---

## 4. Verdict + plan

### Chosen: (c) — kill nmb2; re-scope w74k.3.1 to async-factory + adapter-native auth.

**Purity test applied:** minting a token / building a tool is side-effect-free-ish cognition that happens at the start of a superstep with `config` in hand → Layer 2 (node-internal). It does NOT need a checkpointer-visible boundary. Owning a live session across checkpoints is exactly the side-effectful, resume-fragile thing the purity test pushes OUT of neograph. **"Use the engine's abstraction" invariant:** session lifecycle + per-request auth are the adapter's/`httpx`'s abstractions; neograph must use them.

**Where the two-lifetime rule genuinely requires neograph code:** essentially nowhere new. Lifetime (1) across-checkpoint (re-mintable provider keyed on `config`) is satisfied because the factory is already re-invoked per superstep and reads `config` fresh — a resumed run in a fresh process re-calls the provider automatically. Lifetime (2) within-run live-handle caching is **optional performance owned by the adapter** (persistent-session mode); neograph deliberately does NOT cache (confirmed: no memo across the 6 `_build_turn_prep` sites), which is the *conservative-correct* choice. The resume re-mint that the doc flags as needing framework code is delivered by "don't cache + read config per superstep," which already ships.

### Implementation plan (the re-scoped w74k.3.1)

**Files / publics:**
- `src/neograph/_tool_loop.py` — add `_aprepare_tool_loop` (or factor tool-instantiation into a helper with sync+async twins); sync path fail-loud on coroutine factory.
- `src/neograph/_agent_cycle.py` — `aagent_body` / `atools_body` call the async prep (await factory); `agent_body` / `tools_body` unchanged (sync prep).
- `src/neograph/lint.py` — `_resolve_tool_object` / `_check_async_only_tools`: if `iscoroutinefunction(factory)`, classify as `tool_requires_async_driver` WITHOUT calling `factory({}, cfg)` (calling it returns a coroutine, not a tool — current lint would misintrospect). This is the `tool_requires_async_driver` lint implication.
- Docs: `website` MCP page + example 13 variant — the async-factory + `httpx.Auth` recipe; explicitly document that neograph does NOT own MCP sessions (the adapter does) and that per-run auth is an `httpx.Auth` the consumer supplies.

**Publics added:** none required. (Optional, verdict-(b)-lite convenience — a documented recipe or a tiny `neograph.mcp_tool_factory(server_config, auth=provider)` that wraps `MultiServerMCPClient(...).get_tools()`. It owns NO session; it's sugar. File as a separate optional ticket, not a blocker.)

**Tests (6-cell where IR-level):** async-factory support is an execution-surface change, so the mandatory cells are the two execution surfaces × the tool-bearing agent path. Full 6-cell (3 API surfaces × 2 drivers) applies because factory registration is reachable from `@node(tools=)`, declarative `Node(tools=)`, and programmatic. Concretely:
- Integration: register an **async** factory that reads `config['configurable']['mcp_auth'][server]`; `arun()` under two run-context identities; assert each `tools/call` carried the correct per-identity token at a **stub httpx.Auth / stub server**; assert the no-auth default path unaffected. (This is w74k.3.1's existing verification, now runnable.)
- Fail-loud: async factory under sync `run()` → `ConfigurationError` ("async tool factory requires arun()").
- Lint: async factory → `tool_requires_async_driver` emitted; lint does not crash calling the coroutine factory.
- Guard: token never enters checkpoint state / schema fingerprint / `ToolInteraction` (the §6.3.1 guard — still needed, cheap); structural guard that neograph never reads token contents (still needed).
- E2E: one tool call resolves different server-side authz under two operators (unchanged from w74k.3.1).

**w74k.3.1 acceptance becomes:** "Async tool factories are supported on `arun()` (fail-loud on `run()`); an async factory reading `config['configurable']['mcp_auth'][server]` mints a per-run token that the consumer's `httpx.Auth`/SDK provider stamps per `tools/call`; static `headers=` (a sync factory) stays the default; neograph makes no authz decision, owns no MCP session, and never parses the token; token never enters checkpoint/fingerprint/log." The "depends on nmb2" link is removed; the blocker dissolves.

**Guard/lint for the invariant:** the three-layer structural guard (already planned §3.5) plus a new assertion that neograph has **no MCP session lifecycle code** (no `client.session(...)` / persistent-session ownership in `src/neograph/`) — locks the verdict so a future session doesn't creep back in.

---

## 5. What nmb2's author got right / missed

**Got right:**
- The concrete blocker was real: an async token broker CANNOT be awaited by a sync factory, so w74k.3.1 genuinely could not ship as-is. Filing a blocker instead of silently assuming a foundation was the correct instinct.
- The two-lifetime rule IS the right lens for per-run credentials surviving checkpoint→resume.
- Correctly identified that static `headers=` should stay the default and the provider is the escape hatch.

**Missed:**
- **The foundation already exists in the ecosystem.** Diagnosed "neograph doesn't own session creation" as a gap to fill, when the adapter's stateless-session-per-call + `Connection.auth: httpx.Auth` is the foundation — and per-request auth is exactly where a per-run token attaches. Owning a session is the wrong fix; it's the disease the three-layer doc names.
- **The blocker is one un-twinned line, not a missing subsystem.** "Nothing to attach to" conflated "the factory can't be async" (a 1-line H2 twin gap) with "neograph doesn't own sessions" (true but irrelevant — it shouldn't). The attach point is `config['configurable']` + an async factory, both of which nearly exist.
- **The per-superstep factory re-instantiation already satisfies the across-checkpoint lifetime.** nmb2 proposes to BUILD the "invalidate cache on resume" machinery that the current no-cache + read-config-per-superstep design already delivers for free.
- **Owned lifecycle only buys latency,** and that win is the adapter's persistent-session mode — reachable by a consumer without any neograph code.

**Net:** nmb2's instinct (unblock w74k.3.1) is right; its mechanism (own the session) is the anti-pattern. Kill the epic, land the async-factory twin + the adapter-native auth recipe, and w74k.3.1 ships smaller and more correct than its original design.
