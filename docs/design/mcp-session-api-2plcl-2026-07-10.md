# Design: public `neograph_mcp` session API for calling federated MCP tools (neograph-2plcl)

Date: 2026-07-10. Status: **REVIEWED — approved with changes, all incorporated below.**
Ticket: neograph-2plcl (P3, feature — downgraded 2026-07-09 to ergonomic DX, not a capability gap).
Primary consumer: agent-stark stark-8ok (`hubspot_comms` composite over the federated HubSpot CRM MCP).
Adversarial review: fresh-context reviewer, 2026-07-10, verdict APPROVE-WITH-CHANGES; its two
BLOCKERs (ExceptionGroup at `__aenter__`, timeout deferral) and two MAJORs (demo-server
`structuredContent`, content-conversion contradiction) are folded into §4/§6/§9. Failure-mode
claims below marked *(probed)* were verified empirically against the pinned stack
(`langchain-mcp-adapters 0.3.0`, `mcp 1.28.1`, `httpx 0.28.1`), not just read from source.

---

## 1. Problem

A composite tool — one that ISSUES N calls to federated MCP primitives and assembles
the result — is buildable on today's public surface, but only awkwardly:

```python
tool = await mcp_tool_factory(server, spec, tool_name=..., token_provider=...)(config, None)
result = await tool.ainvoke(args)
```

Two costs, per primitive call:

1. **Factory materialization connects and runs full discovery** — `_make_tool_factory`'s
   body does `client.get_tools(server_name=...)`, which opens a session, runs
   `initialize` + `tools/list`, and closes it.
2. **`tool.ainvoke` opens ANOTHER fresh session** — the adapter's stateless mode
   (no bound session) creates a session per tool execution (adapter 0.3.0
   `tools.py:460-487`; `client.get_tools` docstring: "A new session will be
   created for each tool call").

So agent-stark's `hubspot_comms(domain)` — 1 `search_crm_objects` + 4 association
reads + up to 4 batch reads = **5–9 primitive calls** — would open **10–18 MCP
sessions** per invocation, each costing the MCP lifecycle handshake (`initialize`
request/response + `initialized` notification + the call itself, ~3 round-trips)
plus TCP/TLS setup against the gateway. And the consumer must hand-juggle N
one-tool factories to do it.

What's missing is only a public **session**: connect once, call N tools by name,
get structured results, close. All the machinery (transport dicts, token
minting, stdio arg injection, the persistent-session mode itself) already exists
inside `src/neograph_mcp/_client.py` — the fetcher/replayer already use
`client.session(...)` internally. This is a public-surface gap, exactly like
neograph-g2jg was.

## 2. Consumer requirements (agent-stark, verified 2026-07-10)

From `~/projects/agent-stark` (ADR-0014 amended 2026-07-08; stark-8ok; `tools/_gateway.py`):

- **Transport**: streamable-HTTP against IBM ContextForge v1.0.5 (`HttpServer(url=gateway+"/mcp")`).
  stdio matters only for neograph's own demo server / tests.
- **Naming**: ContextForge re-exposes federated tools as `<peer>-<tool-with-hyphens>`
  (it slugifies underscores: `perplexity_research` → `pplxpeer-perplexity-research`).
  A composite calls primitives by their **namespaced** name directly — no rename
  needed on the call path (rename exists to satisfy a fixed LLM-facing `Tool(name)`
  binding; a direct call has no such binding). NOTE: this is a deliberate
  deviation from the ticket's "same as mcp_tool_factory's rename path" wording —
  what carries over is the transport + token machinery, not rename, which has no
  meaning without a name binding.
- **Token**: minted per run by a `token_provider(configurable)` reading
  `config['configurable']` at AWAIT time, never at build (same contract as
  `mcp_tool_factory`). Gateway injects the upstream HubSpot credential itself.
- **Offline-at-build**: hard requirement — agent-stark's `build_tool_factories`
  runs at compile in an offline test suite, guarded by
  `tests/test_guards_gateway_offline_build.py` on their side.
- **Async-only is fine**: agent-stark is fully async (`arun` path; guards pin that
  registered lc_tools are `async def`).
- **Result shape**: MCP `structuredContent` must be a JSON object; the consumer
  unwraps its own envelope (`Engagements{kind:'engagements', items:[...]}`).
  The session API must hand back `structuredContent` faithfully, no post-processing.
- **Volume**: 5–9 sequential calls per composite invocation; composite is invoked
  from an agent node's tool (built by a consumer factory closure that has `config`).

## 3. Constraints (non-negotiable)

- **nmb2 (docs/design/mcp-session-ownership-review-2026-07-05.md)**: neograph
  *core* never creates, holds, or disposes MCP sessions. The battery package
  `src/neograph_mcp` is the CONSUMER-owned side (it already owns sessions in
  `mcp_resource_fetcher`), and `tests/test_guards_mcp_session_ownership.py`
  scans `src/neograph` only. A session API here is exactly the "adapter's
  persistent mode, reachable by the consumer" path the nmb2 review sanctioned
  (§1.4: a pure performance optimization, opt-in via `client.session(...)`).
  The adversarial review confirmed: sanctioned, not a stretch.
- **Offline-at-build**: constructing the session object performs ZERO network I/O
  (the g2jg precedent); the connect fires at `async with` entry.
- **Session lifetime ≤ one node/tool invocation**: the session must never enter
  neograph state, a checkpoint, or outlive the consumer frame that opened it.
  This keeps the two-lifetime/resume analysis of nmb2 trivially satisfied: a
  resumed run re-opens a session and re-mints identity because the consumer's
  body re-executes. Partially self-enforcing: an `McpSession` cannot be
  checkpointed, and anyio cancel scopes require enter and exit in the SAME task
  (crossing tasks fails loudly with a `RuntimeError`) — documented, not merely
  hoped for.
- **Layering**: `neograph_mcp` imports the adapter + mcp SDK + `langchain_core`
  only (the adapter already depends on `langchain_core`; the battery-import
  guard bans only langgraph/run-layer imports); nothing from the langgraph
  engine or neograph run-layer internals.
- **Versions pinned**: `langchain-mcp-adapters 0.3.0`, `mcp >=1.28,<2` (existing pins).

## 4. Design

### 4.1 Public API

New module `src/neograph_mcp/_session.py`, re-exported through
`neograph_mcp/__init__.py` `__all__` (the public contract):

```python
def mcp_session(
    server_key: str,
    spec: StdioServer | HttpServer,
    *,
    token_provider: TokenProvider | None = None,
    config: Any | None = None,
    stdio_token_arg: str = "token",
    timeout: float | None = 30.0,
) -> McpSession: ...


class McpSession:
    """One MCP connection, N tool calls. Async context manager; consumer-owned.

    Construction is ZERO network I/O; the connect (initialize handshake) fires
    at `async with` entry. Identity is minted ONCE at entry: from
    config['configurable'] via token_provider when config is given (the
    mcp_tool_factory contract), else via the no-config provider shape (the
    mcp_resource_fetcher contract). Open it INSIDE the node/tool body that
    uses it — enter and exit in the same task (anyio cancel-scope rule) — and
    never store it in state or across supersteps.
    """

    async def __aenter__(self) -> McpSession: ...
    async def __aexit__(self, *exc) -> None: ...

    async def call(self, tool_name: str, args: dict[str, Any] | None = None) -> McpCallResult: ...
    async def tool_names(self) -> list[str]: ...   # paginated tools/list over the same session (cached)


@dataclass(frozen=True)
class McpCallResult:
    """The result of one tools/call, converted once, no lossy post-processing."""
    content: list[dict[str, Any]]     # langchain-style content blocks, SAME table as
                                      # the adapter's bound-tool path (see §4.2) — so
                                      # this matches what ToolInteraction.typed_result
                                      # carries today. Always a list.
    structured: dict[str, Any] | None # CallToolResult.structuredContent, verbatim

    @property
    def text(self) -> str | None: ... # first text block's text, else None (convenience)


class McpToolCallError(Exception):
    """A tools/call returned isError=True. Carries server_key, tool_name, the
    converted error content, and structuredContent when the server included one
    (a consumer's Degraded-mapping path may want it)."""
```

`timeout` (seconds) bounds BOTH the connect+initialize at `__aenter__` (via
`asyncio.timeout`) and each `call()` (via the SDK-native
`session.call_tool(..., read_timeout_seconds=timedelta(seconds=timeout))`).
Default **30.0**, not `None` — see §4.2 failure modes. `None` opts out.

Consumer sketch (the stark-8ok composite, abridged):

```python
async def hubspot_comms_factory(config: Any, tool_config: Any) -> Any:
    @lc_tool
    async def hubspot_comms(domain: str) -> list | Empty | Degraded:
        async with mcp_session("stark-gateway", HttpServer(url=f"{base}/mcp"),
                               token_provider=provider, config=config) as s:
            found = await s.call("hubspotpeer-search-crm-objects",
                                 {"query": domain, "object_type": "companies"})
            company = ...  # unwrap found.structured
            engagements = []
            for kind in ("notes", "emails", "calls", "meetings"):
                r = await s.call("hubspotpeer-get-crm-objects", {...})
                engagements.extend(...)  # unwrap r.structured
        return engagements or Empty()
    return hubspot_comms
```

One connection for all 5–9 calls; token minted once from the run's config; the
session dies with the tool invocation.

### 4.2 Semantics

**Connect (deferred).** `__aenter__` resolves the token
(`_resolve_token(token_provider, config)` when `config is not None`, else
`_resolve_token_no_config(token_provider)` — both existing helpers), builds the
client via the existing `_client_for(server_key, spec, token)` (bearer header
for `HttpServer`, env/argv for `StdioServer`), and enters
`client.session(server_key)` (adapter persistent mode: one `create_session` +
`initialize`), all inside `asyncio.timeout(timeout)`. `mcp_session(...)` itself
does nothing but store arguments.

**Call.** `call(name, args)` issues
`session.call_tool(name, args or {}, read_timeout_seconds=...)` on the shared
`ClientSession` and converts the raw `CallToolResult` locally, replicating the
adapter's content-block table exactly (adapter `tools.py:175-283`), using the
PUBLIC `langchain_core.messages.content` block constructors — NOT the adapter's
private `_convert_call_tool_result`:

| MCP block | converted to |
|---|---|
| `TextContent` | text block |
| `ImageContent` | image block (base64) |
| `ResourceLink` | file block (uri) — exercised by the demo server's `get_deal` on day one |
| `EmbeddedResource` (text/blob) | text / file block |
| `AudioContent` | `NotImplementedError` (adapter parity) |

`content` is ALWAYS a list (the adapter never yields a bare `str` for a
`CallToolResult` in 0.3.0). `structuredContent` is surfaced verbatim on
`.structured`. `isError=True` raises `McpToolCallError` (typed, carries the
converted error content + any structuredContent). Replicating the table rather
than importing the private symbol keeps the parity claim honest AND avoids
private-import fragility; ~30 lines, pinned by tests.

**stdio identity parity.** Over stdio, identity rides as a tool argument. On the
first `call()` (lazily, and only when `spec` is `StdioServer` and a token was
minted), the session runs a **paginated** `tools/list` (cursor loop — the
adapter paginates too; a single-page read would silently truncate on a large
federated gateway) once over the same connection, caches each tool's declared
argument names, and injects `stdio_token_arg` into `args` when the tool declares
it — the same declares-check semantics as `_inject_stdio_token`, applied to an
args dict instead of a BaseTool. As on the factory path, the framework-minted
token OVERRIDES a caller-supplied value for that arg (identity is
framework-carried, never call-site-chosen) — documented, since on the direct-call
path the caller is the trusted consumer. Over `HttpServer` identity is already
on the bearer header and no listing happens on the call path; `tool_names()`
triggers the same cached (paginated) listing lazily on either transport.
A server mutating its tool list mid-session (`listChanged`) is NOT handled —
acceptable under the lifetime rule (a session lives for one invocation) and
stated here deliberately.

**Errors escape typed, not wrapped.** *(probed)* The anyio task groups inside
the transports wrap failures in `ExceptionGroup`s — including at CONNECT time:
a refused port surfaces from `client.session().__aenter__` as
`ExceptionGroup(ConnectError)`, and a dead stdio server as a doubly-nested
`ExceptionGroup(ExceptionGroup(McpError("Connection closed")))`. So a
single-level unwrap at exit is insufficient. `McpSession` applies a shared
`_unwrap_single(exc)` helper that RECURSIVELY descends
`BaseExceptionGroup`s while `len(exceptions) == 1`, at BOTH `__aenter__` and
`__aexit__`. Multi-leaf groups (a real double failure) propagate as-is — no
information is discarded to force a single type. Net effect: a transport
failure or `McpError` reaches the consumer as its own type, promptly.

**Failure modes and the timeout default.** *(probed)* Two hang classes exist on
the pinned stack and are why `timeout` ships in v1 with a non-None default
rather than deferred:
- a stdio subprocess that never answers `initialize` hangs `__aenter__`
  **indefinitely** (`ClientSession` has no default request timeout);
- an HTTP server that accepts TCP but never responds stalls `__aenter__` on the
  adapter's default read timeout — up to **300 s**
  (`httpx.Timeout(30, read=300)`, `sessions.py:56-57`).
The ticket's verification bullet ("a transport failure surfaces as a typed
error, not a hang") names transport failures generally; only the refused-port
case is naturally prompt. `asyncio.timeout` on `__aenter__` + SDK-native
`read_timeout_seconds` per call close both classes with zero new machinery.

**Concurrency.** Documented for sequential use within one consumer frame (the
stark loop is sequential), enter/exit in the same task. The SDK multiplexes
JSON-RPC ids so interleaved awaits happen to work, but v1 makes no concurrency
guarantee and does not test it — a consumer wanting 4-way fan-out opens 4
sessions or awaits sequentially.

**No rename parameter.** Rename exists on the factory path to reconcile a
gateway-namespaced discovered name with a fixed LLM-facing `Tool(name)` binding.
A direct `call()` has no name binding to satisfy — the caller passes the
namespaced name the server actually exposes (discovery via `tool_names()`).
Adding rename here would be a second naming layer with no consumer.

### 4.3 What is reused vs added

| Piece | Status |
|---|---|
| `StdioServer` / `HttpServer` specs | reused as-is |
| `_connection` / `_client_for` (transport + bearer) | reused as-is |
| `_resolve_token` / `_resolve_token_no_config` | reused as-is |
| stdio declares-check semantics | shared logic, factored so `_inject_stdio_token` and the session use one helper |
| persistent session (`client.session(...)`) | same adapter mode the fetcher/replayer already use |
| content-block conversion | replicated table (public `langchain_core` constructors), ~30 lines, test-pinned |
| `McpSession`, `McpCallResult`, `McpToolCallError`, `mcp_session` | new, ~150 lines in `_session.py` |

No changes to `src/neograph` (core). No changes to compile()/factory seams.

## 5. Alternatives considered

- **(a) Status quo (do nothing).** Capability exists (materialize factory +
  `ainvoke`). Rejected as the *resolution* of the ticket because the composite
  pays 2 sessions per primitive call (10–18 per `hubspot_comms` invocation) and
  the pattern is undocumented hand-juggling. The ticket's downgrade note itself
  names the two things worth shipping: the one-connection session and a blessed
  composite example.
- **(b) Session-bound BaseTools** (`session.tool(name)` returning an adapter
  tool bound to the shared session via `convert_mcp_tool_to_langchain_tool(session, tool)`).
  Maximum adapter reuse, but the consumer then talks langchain tool-interface
  (`ainvoke`, artifact tuples) when all it wants is "call name with args, get
  structured result". Rejected for v1 surface; trivially addable later if a
  consumer wants to hand session-bound tools to something langchain-shaped.
- **(c) Multi-server session** (dict of servers + routing, fetcher-style).
  YAGNI: the consumer composes over ONE gateway; the singular shape mirrors
  `mcp_tool_factory`. A two-server composite opens two sessions.
- **(d) Sync facade** (`_run_sync` bridge). A persistent session is pinned to an
  event loop; a sync facade needs a dedicated loop thread and would contradict
  the existing "MCP tools are async-only, lint flags sync run()" story. Async-only.
- **(e) neograph-core session ownership.** Killed by nmb2; not reopened. This
  design deliberately lives in the battery package the consumer owns.
- **(f) Per-request re-mint via `Connection.auth: httpx.Auth`** (nmb2 §1.3's
  native path). Would re-mint identity per call instead of once at entry. The
  battery already chose static headers on the factory path; a session lives for
  one invocation (5–9 sequential calls), so short-TTL exposure is negligible.
  Consistency wins; the httpx.Auth escape hatch remains available to a
  hand-rolling consumer.

## 6. Prerequisite: demo server must emit `structuredContent`

*(probed)* FastMCP on mcp 1.28.1 emits `structuredContent` only for
schema-serializable return annotations: `-> dict` and `-> list` yield **None**;
`-> dict[str, Any]` and `-> SomeModel` yield the payload. Every current demo
tool is annotated `-> dict` or `-> list`, so structured-fidelity tests are
unwritable today. Prerequisite step: change the demo-server annotations to
`dict[str, Any]` (and/or add one BaseModel-returning tool). This is additive
for examples 23/24/25 but MUST be verified by re-running
`uv run --extra dev --extra mcp-examples pytest tests/test_mcp_examples_e2e.py`
(the AGENTS.md examples rule).

## 7. Verification plan (integration & E2E only, per ticket)

In `tests/test_mcp_battery.py` (`@requires_mcp`, run
`uv run --extra dev --extra mcp pytest tests/test_mcp_battery.py`):

1. **Offline-at-build**: `mcp_session(...)` construction performs zero network
   I/O (no subprocess spawn / no connect; same assertion style as
   `TestLazySingleToolFactory.test_construction_is_zero_network...`).
2. **Multi-primitive over ONE session** (stdio demo server): inside one
   `async with`, `call("crm_search", ...)` then `call("get_deal", ...)`;
   assert both results; assert the demo server echoes the SAME `acting_as`
   token on both calls (stdio arg injection through the session). NOTE:
   `get_deal` returns `ResourceLink` blocks — this test also pins the
   content-conversion table's file-block arm.
3. **HttpServer leg** (existing `TestHttpServerSmoke` harness): bearer identity
   rides on `call()`; `stdio_token_arg` NOT injected over http.
4. **Typed errors, all three classes**:
   (i) refused port → the natural typed exception (e.g. `ConnectError`) from
   `__aenter__` — recursively unwrapped, not an `ExceptionGroup`;
   (ii) hung server (stdio subprocess that sleeps instead of answering
   `initialize` — a 1-line fixture) → `TimeoutError` at ~`timeout`, not a hang;
   (iii) a tools/call with `isError=True` → `McpToolCallError` with the error
   content. Mechanism (no server changes needed, verified): a missing required
   argument or an unknown tool name both return `isError=True` from FastMCP.
5. **structuredContent fidelity**: after §6, a demo tool annotated
   `-> dict[str, Any]` surfaces its payload verbatim on `.structured`.
6. **E2E composite**: a scripted composite (consumer factory building an lc_tool,
   or a raw-mode node) calls 2 primitives through one session and assembles a
   typed output — driven through `compile()` + `arun()`, asserting the assembled
   output, not internals.
7. **Guards**: `test_guards_mcp_session_ownership.py` stays green unchanged
   (scans `src/neograph` only — this design adds nothing there); the battery
   import guard stays green (`langchain_core` is already adapter-implied).
8. **Examples harness**: `tests/test_mcp_examples_e2e.py` green after the §6
   demo-server change and after adding example 26.

## 8. Documentation plan

- **Example 26** (`examples/26_mcp_composite_session.py`): a composite tool over
  the demo server's `crm_search` + `get_deal` — the blessed "scripted composite
  over federated primitives" pattern stark-8ok asked for. Keyless, needs
  `--extra mcp-examples`; auto-discovered by `tests/test_mcp_examples_e2e.py`
  (`examples/2?_mcp_*.py` glob).
- **Website** `walkthrough/mcp-client.mdx` (or a short section on
  `concepts/mcp-integration.mdx`): "Composing over federated primitives" — when
  to bind (agent decides, use `mcp_tool_factory`) vs when to compose (scripted
  determinism, use `mcp_session`), the one-connection efficiency argument, and
  the lifetime rule (open per invocation, same task, never store).
- `_session.py` module docstring: restate nmb2 (consumer-owned, never core),
  the lifetime + same-task rule, and the timeout default.

## 9. Decisions taken

- No rename on the call path (deliberate deviation from the ticket's
  "rename path" phrasing — transport + token machinery is what carries over).
- Single-server session; async-only; no sync facade.
- `McpCallResult` = frozen dataclass: `content: list[dict]` (adapter-parity
  block table, always a list), `structured` verbatim, `.text` convenience.
  No auto-JSON-parsing magic.
- Content conversion replicated locally against PUBLIC types (mcp
  `CallToolResult` + `langchain_core` block constructors); the adapter's
  private `_convert_call_tool_result` is not imported.
- Token minted once at `__aenter__` (session lifetime ≤ one invocation ⇒ one
  config ⇒ one identity); over stdio the minted token overrides a
  caller-supplied token arg (framework-carried identity), matching the factory
  path.
- `timeout: float | None = 30.0` in v1 (NOT deferred): `asyncio.timeout` on
  connect + `read_timeout_seconds` per call — both hang classes are real on the
  pinned stack (§4.2).
- Recursive single-leaf `ExceptionGroup` unwrap at both `__aenter__` and
  `__aexit__`; multi-leaf groups propagate as-is.
- Paginated `tools/list` (cursor loop) for the lazy listing; stale-listing
  (`listChanged`) deliberately unhandled under the lifetime rule.
- `McpToolCallError` carries structuredContent when present (Degraded-mapping
  consumers may want it).
