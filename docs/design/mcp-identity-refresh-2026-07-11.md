# MCP per-run token freeze — Option D design spike (neograph-qslrx)

Date: 2026-07-11
Status: DESIGN (read-only spike; no production code changed)
Ticket: neograph-qslrx
Author: architect spike

---

## 1. Problem

The MCP tool instances built for an agent/act ReAct node are cached under
`StateKeys.RUN_ID` for the whole run. On the **static `token_provider` bearer**
path the bearer is resolved ONCE at the first ReAct superstep and baked into the
HTTP connection's `Authorization` header at connect. Every later superstep of the
same run reuses that frozen header. A run whose tool phase outlives the IdP
access-token lifespan (ox-troubleshooting-demo: Keycloak 300s default, an `Each`
verify fan-out spanning >5 min) therefore sends a **stale** bearer and the server
denies — a spurious `ACCESS_DENIED` for a fully-authorized principal.

The freeze is specific to the **static `token_provider` bearer**. The explicit
OAuth path (`HttpServer(auth=client_credentials_auth(...))`, neograph-27h3s)
already refreshes per request and is immune to the cache, because httpx invokes
the attached `httpx.Auth`'s `auth_flow` on every request against the persistent
transport client.

stdio has the analogous freeze: identity rides as a tool ARGUMENT baked into the
wrapped coroutine at build time (`_inject_stdio_token`), so it is frozen for the
run the same way. Lower practical urgency (stdio = local subprocess, rarely a
real expiring IdP token) but the same root cause.

## 2. Verified mechanism (cited against current source)

Files: `src/neograph/_tool_loop.py`, `src/neograph/_run_cache.py`,
`src/neograph_mcp/_client.py`, `src/neograph_mcp/_session.py`,
`src/neograph_mcp/_run_context.py`, `src/neograph_mcp/_prompt.py`.

- **RUN_ID cache wrap.** `_tool_loop.py:484` `tool_instances = get_or_build(config,
  f"tools:{node_name}", lambda: _instantiate_tools(...))` (sync driver) and
  `_tool_loop.py:530` `await aget_or_build(config, f"tools:{node_name}",
  lambda: _ainstantiate_tools(...))` (async driver). `_run_cache.py:114
  get_or_build` / `:138 aget_or_build` key on `(run_id, subkey)`; `run_id` is the
  framework-minted, config-only `StateKeys.RUN_ID`, stable across all supersteps
  of one run (docstring `_run_cache.py:1-38` states outright: "There is
  deliberately no mid-run expiry recheck"). So `_ainstantiate_tools` — and thus
  the tool factory — runs ONCE per run; subsequent supersteps get the cached tool.
- **Factory resolves + bakes identity once.** `_client.py:452 _factory` →
  `_client.py:455 token = await _resolve_token(token_provider, config)`; the http
  client is built at `_client.py:467 client = _client_for(server_key, spec, token,
  callbacks=callbacks)`.
- **`_connection`'s two http branches + stdio branch** (`_client.py:247-269`):
  stdio branch (`:256-260`) carries no identity in the connection. http branch:
  `:262-263 if token is not None and spec.auth is None: headers["Authorization"] =
  f"Bearer {token}"` (the FROZEN static bearer) vs `:265-266 if spec.auth is not
  None: conn["auth"] = spec.auth` (the OAuth `httpx.Auth`, per-request). Docstring
  (`:247-255`) names it "the SINGLE choke point for transport identity — a change
  here lights all three surfaces (tool factories, resource fetcher,
  `mcp_session`)".
- **stdio freeze.** `_client.py:475-476` `if isinstance(spec, StdioServer) and
  token is not None: tool = _inject_stdio_token(tool, token, stdio_token_arg)`.
  `_inject_stdio_token` (`:355-377`) closes over the **build-time** `token` string
  in `_with_identity` (`:373-375 kwargs[arg_name] = token`). Frozen for the run.
- **`_resolve_token`** (`:294-305`): reads `config['configurable']`, calls
  `token_provider(configurable)`, awaits if awaitable (handles sync OR async
  providers). `_resolve_token_no_config` (`:308-320`) is the config-less resource
  variant.
- **All surfaces route through `_client_for` → `_connection`.** Confirmed callers
  of `_client_for`: tool factory (`_client.py:467`), discovery
  (`_client.py:382`), resource fetcher/replayer (`_client.py:702, :719`),
  `mcp_session` (`_session.py:218`), `mcp_run_context` (`_run_context.py:92`),
  prompt fetch (`_prompt.py:88`). So the choke-point claim holds for the
  header-vs-auth WIRING; but each caller pre-resolves the token itself, so the
  identity SOURCE change touches each http caller.

### Load-bearing assumption — CONFIRMED

Option D rests on: *httpx invokes `auth_flow` per request against the adapter's
persistent streamable-http client.* This is httpx's documented contract
(`Client._send_handling_auth` drives `async_auth_flow(request)` for **every**
request, not only on 401). It is confirmed end-to-end for this stack by
`tests/test_mcp_oauth.py:208
test_token_refresh_on_expiry_succeeds_on_same_session_without_rebuild`: inside a
single `mcp_session` connection, call 1 uses token A, the token is revoked
server-side, call 2 on the SAME session succeeds with a freshly-minted token B —
proving the adapter's persistent httpx client re-runs `auth_flow` per request
with no reconnect (the fixture is a real FastMCP streamable-http + real SDK OAuth
provider, no protocol mocking). A framework `_TokenProviderAuth` is just another
`httpx.Auth` on the exact same seam, so it inherits the same per-request driving.

Residual (small): (a) test 2 exercises `mcp_session`, not the agent loop under
the RUN_ID cache — the "refresh across supersteps under the RUN_ID tool cache"
scenario is untested for BOTH the OAuth and the new path (the new test suite
closes this). (b) our `_TokenProviderAuth` must be a correct `httpx.Auth`
(`async_auth_flow` mutates headers then yields once). Low risk.

Note: `website/src/content/docs/concepts/mcp-integration.mdx:77` already claims
"over streamable-http it rides as a bearer header minted **per request** by an
`httpx.Auth`" — which is currently TRUE only for the explicit `auth=` path and
FALSE for the static `token_provider`. Option D makes the code match the already
published promise.

## 3. `_TokenProviderAuth` design

An internal `httpx.Auth` subclass that re-invokes `token_provider` per request and
stamps the bearer, replacing the connect-time static header.

```python
# _client.py, near _connection (the transport-identity choke point).
# httpx imported FUNCTION-LOCALLY (module keeps import light; matches the existing
# local httpx imports at _client.py:124, :139). The class is DEFINED inside a
# builder so the httpx base class is only referenced when the mcp extra is present.

def _http_identity(spec, token_provider, config, *, use_config=True):
    """The httpx.Auth for one http server's per-run identity, choke-point rule:
    explicit HttpServer.auth (real OAuth) WINS; else wrap token_provider so it is
    re-invoked PER REQUEST; else None (anonymous)."""
    if spec.auth is not None:
        return spec.auth
    if token_provider is None:
        return None
    import httpx

    class _TokenProviderAuth(httpx.Auth):
        requires_request_body = False
        requires_response_body = False

        async def async_auth_flow(self, request):
            token = (await _resolve_token(token_provider, config)) if use_config \
                else (await _resolve_token_no_config(token_provider))
            if token is not None:
                request.headers["Authorization"] = f"Bearer {token}"
            yield request

        def sync_auth_flow(self, request):
            # streamable-http is async-only, so this is defensive. A sync
            # transport cannot await an async provider -> fail loud.
            configurable = (config or {}).get("configurable", {}) if use_config else {}
            result = token_provider(configurable) if use_config else token_provider()
            if inspect.isawaitable(result):
                raise TypeError("async token_provider on a sync auth flow; use arun()")
            if result is not None:
                request.headers["Authorization"] = f"Bearer {result}"
            yield request

    return _TokenProviderAuth()
```

Key points:
- **Closes over `config` at factory-build time** (the run's config). Within a run
  `config['configurable']` is stable across supersteps, and `_factory` runs once
  per run under the cache — so closing over the first superstep's config is
  correct; a resume mints a new RUN_ID → new factory call → new Auth with fresh
  config.
- **Async provider bridging.** `async_auth_flow` awaits `_resolve_token` which
  already handles sync-or-async providers (`inspect.isawaitable`). The MCP
  streamable-http transport is httpx.AsyncClient, so `async_auth_flow` is the path
  actually exercised. `sync_auth_flow` is defensive and fails loud on an async
  provider.
- **No 401 retry.** Each request just re-mints; we do not inspect the response
  (`requires_response_body = False`). Turnkey 401-triggered refresh remains the
  job of the SDK OAuth provider on the `auth=` path.
- **Precedence when both `HttpServer.auth` and `token_provider` are set:**
  explicit `spec.auth` WINS (matches today's "auth wins over bearer",
  `_connection` docstring). token_provider is ignored for http when real OAuth is
  configured — never two `Authorization` writers.
- **Home:** `_client.py`, the transport-identity choke point. Respects the nmb2
  session-ownership invariant (the Auth is created inside the consumer-held
  factory/session, never by neograph core) and the light-import rule (httpx local).

## 4. Change manifest

| # | Site | Change | Why |
|---|------|--------|-----|
| 1 | `_client.py:247-269` `_connection` | Delete the static-bearer branch (`:262-263`). Signature `(spec, token)` → `(spec, *, auth=None)`; http sets `conn["auth"] = auth if auth is not None else spec.auth`; stdio unchanged. `token` param dropped (now unused on every branch). | Removes the FROZEN connect-time header; http identity is always an `httpx.Auth`. |
| 2 | `_client.py` (new, near `_connection`) | Add `_TokenProviderAuth(httpx.Auth)` + `_http_identity(spec, token_provider, config, *, use_config)` builder (section 3). | Single construction site for the per-request wrap + the explicit-auth-wins precedence. |
| 3 | `_client.py:272-291` `_client_for` | `token: str \| None` → `auth: Any \| None`; forward to `_connection(spec, auth=auth)`. | The client seam now carries an Auth, not a resolved string. |
| 4 | `_client.py:452-489` `_make_tool_factory._factory` | Remove build-time `token = await _resolve_token(...)` for the http client. http: `auth = _http_identity(spec, token_provider, config); client = _client_for(..., auth=auth)`. stdio: build client with no header identity; inject identity per call (item 5). Held-session path (`:457-465`) unchanged for binding but see item 8/9 for how the held session got its identity. | The per-request re-mint replaces the once-per-run bake. |
| 5 | `_client.py:355-377` `_inject_stdio_token` | Signature takes `(tool, token_provider, config, arg_name)` instead of a baked `token`; `_with_identity` re-resolves `token = await _resolve_token(token_provider, config)` PER CALL before stamping `kwargs[arg_name]`. Call site `:476` updated. | The stdio analogue of the freeze fix (re-resolve per call, don't close over a build-time value). |
| 6 | `_client.py:380-384` `_discover_tool_names` | Pass `auth=spec.auth` for http (OAuth discovery must authenticate). Decision point: also pass the wrapped token_provider Auth (`_http_identity(spec, token_provider, {})`) to authenticate token_provider-guarded discovery — RECOMMENDED (strictly more correct; today discovery is anonymous). Requires threading `token_provider` into `_discover_tool_names`. | Keep OAuth discovery working after `_connection` stops reading `spec.auth` implicitly; optionally close the anonymous-discovery gap. |
| 7 | `_client.py:699-725` `mcp_resource_fetcher` (fetcher + replayer) | http: build `auth = _http_identity(spec, token_provider, None, use_config=False)`; `_client_for(..., auth=auth)`. Drop the pre-resolved `token`. | Same freeze on the resource path (one-shot read, LOWER urgency — no long-lived multi-request loop, but a late fetch under a resumed/cached run could send a stale token). |
| 8 | `_session.py:213-218` `McpSessionHandle.__aenter__` | http: `auth = _http_identity(spec, token_provider, config, use_config=<config is not None>)`; `_client_for(..., auth=auth)`. stdio: unchanged (sessions rarely stdio; per-call arg injection is the factory concern). **[SUPERSEDED by neograph-hs3mr, 2026-07-11: the stdio scope-out was a frequency argument, not a correctness one — `McpSession.call()` now re-resolves `token_provider` per call on stdio too, so identity is per-call fresh on EVERY surface/transport; a constant provider pins.]** | `mcp_session` holds ONE long-lived connection across N `.call()`s — the exact long-lived-freeze case the OAuth path already fixes. Inherits the fix via the choke point once it passes an Auth. |
| 9 | `_run_context.py:86-92` `__aenter__` | Same as item 8: http build `auth` and pass to `_client_for` instead of the resolved token; stdio unchanged. **[Verified under neograph-hs3mr: `mcp_run_context` has NO stdio minting site at all — stdio identity rides the factory's per-call `_inject_stdio_token`, so "stdio unchanged" here was already per-call, not a freeze.]** | Held run-scoped sessions are long-lived across the run's supersteps — the http held path must refresh per request too. |
| 10 | `_prompt.py:88` | http: build `auth`, pass to `_client_for`. | Choke-point parity; LOWEST urgency (one-shot prompt fetch). |
| 11 | `website/src/content/docs/concepts/mcp-integration.mdx` (line 77) + `walkthrough/mcp-client.mdx` | Document: static `token_provider` is now invoked PER REQUEST (freeze removed); a static-string provider returns the same value (no actual refresh) — use a re-fetching provider or `HttpServer.auth` + `client_credentials_auth` for turnkey exchange/refresh. Line 77's "per request by an httpx.Auth" claim becomes TRUE for both paths. | Acceptance criterion requires the docs update; and the current doc claim is retroactively made honest. |
| 12 | `tests/test_mcp_identity_refresh.py` (new) OR extend `test_mcp_battery.py`/`test_mcp_oauth.py` | The freeze-gone regression (section 6). | Pins the fix; closes the across-supersteps-under-RUN_ID-cache coverage gap for both the new path and the existing OAuth path. |

Edit-site count: **12** (10 production sites across `_client.py`/`_session.py`/
`_run_context.py`/`_prompt.py`, 1 docs, 1 tests). Production `_client.py` carries
the bulk (items 1-7); the other three modules (items 8-10) each inherit the fix
with a one-line "build auth, pass auth" swap at their `_client_for` call.

### Why the choke point isn't a literally-free inheritance

`_connection` centralizes the header-vs-auth WIRING, but every surface
pre-resolves its own token and passes it down (because the tool path has `config`
and the resource path does not — that resolve-variant knowledge lives at the
caller). So each http caller flips from "resolve token → pass token" to "build
`_http_identity` → pass auth". The `_http_identity` helper keeps that a single
one-line call per site; the freeze deletion itself is centralized at `_connection`
(item 1).

## 5. Backward-compatibility analysis

- **Invocation cadence.** `token_provider` moves from ONCE per (run, tool) to
  PER REQUEST. Within a single httpx request `auth_flow` runs exactly once (no
  double-call), so **no within-request memo is needed**. Across requests, re-mint
  is the entire point.
- **Static-string providers:** identity semantics UNCHANGED (same string every
  call); only the call count rises. Safe.
- **Side-effecting / expensive providers:** now run N times (e.g. a provider that
  does its own token exchange per call → N exchanges). This is the desired refresh
  behavior, but consumers who cannot afford per-request minting should cache with
  expiry inside their provider, or (preferred) move to `HttpServer.auth +
  client_credentials_auth` for turnkey SDK-owned exchange. Call this out for ox
  (Keycloak) explicitly — the durable answer for them is the OAuth `auth=` path;
  Option D unfreezes the static path as the secondary, DIY-refresh option.
- **Wire format unchanged:** the header is still `Authorization: Bearer <token>`,
  just stamped by `auth_flow` per request instead of a static connect header. No
  server-visible shape change.
- **Existing tests scanned.** `tests/test_mcp_battery.py` providers are pure
  (`lambda configurable: configurable.get("op", "anon")`) with NO call-count
  assertions; the OAuth tests count server-minted tokens, not provider calls. No
  known breakage. `test_per_run_identity_rides_the_held_session_path`
  (`test_mcp_battery.py:1202`, stdio, operator-A across 2 turns) still passes:
  per-call re-resolution of a stable `op` yields the same identity.
- **0.x posture:** sole downstream = piarch/ox/agent-stark; no deprecation
  burden. Breaking the internal `_inject_stdio_token` / `_client_for` / `_connection`
  signatures is fine (all private).

## 6. Test plan (behavioral, integration/e2e — not construction)

Reuse the real demo servers + fakes: `examples/_mcp_demo_server.py`,
`tests/_mcp_oauth_demo_server.py`, `tests/test_mcp_battery.py` patterns,
`tests/fakes.py::ReActFake`.

1. **http freeze-gone (primary).** A fake `token_provider` that returns a
   DIFFERENT token on each successive call (e.g. a counter: `t-1`, `t-2`, ...).
   Drive an `agent`/`act` ReAct loop across ≥2 supersteps (`ReActFake` with two
   tool-call turns then a final) under the RUN_ID tool cache (`arun`, so the tool
   is built once and cached). Server echoes the received bearer (`whoami` /
   `acting_as`). Assert turn-1's call carried `t-1` and turn-2's carried a FRESH
   `t-2` — proving the freeze is gone WITHOUT a reconnect. This is the exact
   scenario the RUN_ID cache froze.
2. **stdio twin.** Same shape with `StdioServer` + a counter provider; assert
   `acting_as` echoes a fresh token on turn 2 (per-call re-resolution inside the
   stdio coroutine wrapper). Mirrors `test_stdio_token_provider_injects_...`
   (`test_mcp_battery.py:346`).
3. **static-string regression.** A `token_provider` returning a constant string
   still authenticates every turn with the SAME token — proving Option D does not
   break the common static case (identity unchanged, just per-request-invoked).
4. **OAuth-across-supersteps pin (gap-closing).** Extend the 27h3s refresh test to
   the agent loop under the RUN_ID cache (not just `mcp_session`): revoke mid-run
   between supersteps, assert turn-2 refreshes without reconnect. Closes the
   existing coverage hole that `test_token_refresh_on_expiry...` (mcp_session
   only) leaves open.
5. **held-session parity** (optional): a long-lived `mcp_run_context` http session
   with a counter provider — assert per-request re-mint on the held path
   (item 9).

Acceptance criterion (from the bead): a run whose tool phase outlives the IdP
token lifespan refreshes mid-run (per-call re-resolution) rather than sending a
stale token; a test simulates expired-then-refreshed across supersteps without
reconnect; documented in the MCP auth/token_provider docs. Tests 1-4 + item 11
satisfy it.

## 7. Risks / edge cases / open questions

- **[LOAD-BEARING] auth_flow-per-request premise** — CONFIRMED (section 2) via
  httpx's contract + `test_mcp_oauth.py:208`. The one thing to re-verify FIRST in
  implementation: that our `_TokenProviderAuth.async_auth_flow` is actually driven
  by the langchain-mcp-adapters streamable-http client for every `tools/call`
  (write test 1 RED first and watch it re-mint). If for any reason the adapter
  short-circuits auth on a pooled connection, this premise — and the whole fix —
  needs the fallback of rebuilding the client per call (defeats the cache), so
  validate it before building out items 2-10.
- **Async `token_provider` bridging.** `async_auth_flow` awaits `_resolve_token`
  (already sync/async-safe). `sync_auth_flow` fails loud on an async provider —
  acceptable because the MCP transport is async-only.
- **Cache layering.** RUN_ID cache still caches the CLIENT/tool (built once per
  run); the Auth runs per request underneath. Confirmed sound — mirrors the OAuth
  path. The Auth instance closes over the run's config, stable across supersteps.
- **401 vs proactive.** `_TokenProviderAuth` re-mints proactively each request; it
  does NOT need 401-retry logic (that stays the SDK OAuth provider's job on the
  `auth=` path). If a re-minted token is itself already expired at send time, the
  server 401s and the model sees an error result — same as any other call failure;
  a `token_provider` that mints fresh each call avoids it.
- **Discovery identity (item 6).** Open decision: keep discovery anonymous
  (today's behavior, minimal change) vs authenticate discovery with the wrapped
  Auth (strictly more correct, tiny risk). Recommend the latter.
- **Resource/prompt urgency (items 7, 10).** One-shot reads — include for choke-
  point parity but flag LOWER priority than the tool-loop (item 4) and long-lived
  session (items 8, 9) paths.
- **Docs honesty (item 11).** `mcp-integration.mdx:77` currently over-claims;
  land the code + doc together so the claim is true.

## 8. Recommendation

Adopt **Option D** as scoped above: collapse the static `token_provider` bearer
onto the same `httpx.Auth` mechanism the OAuth path uses, via an internal
`_TokenProviderAuth` whose `async_auth_flow` re-invokes `token_provider` per
request, and DELETE `_connection`'s static-bearer header branch so the choke point
has exactly one http-identity mechanism.

- **Priority order:** tool-loop http (items 1-4) + long-lived sessions (items 8, 9)
  are the real fix; stdio (item 5) is the same principle at lower practical
  urgency (local subprocess); resource/prompt (items 7, 10) are choke-point parity.
- **Validate FIRST:** the auth_flow-per-`tools/call` premise (write test 1 RED and
  confirm the adapter re-mints on turn 2 before building the rest).
- **Framework vs consumer boundary (must be documented):** unifying makes refresh
  POSSIBLE; whether the token actually CHANGES is the consumer's `token_provider`'s
  job. The framework's contract becomes "call the provider per request (stop
  freezing it)"; a static-string provider still returns the same value. For
  turnkey framework-owned exchange/refresh, `HttpServer.auth +
  client_credentials_auth` stays the recommended path (and the durable answer for
  ox/Keycloak). The `token_provider` contract itself (`configurable -> str`,
  sync-or-async) is UNCHANGED.
