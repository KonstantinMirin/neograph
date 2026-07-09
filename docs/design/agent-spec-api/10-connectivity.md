# Agent Spec → neograph mapping: Connectivity & Resilience

## Source

Oracle Open Agent Spec v26.1.2, API Reference → Connectivity & Resilience section. Classes:
- `pyagentspec.auth.AuthConfig`
- `pyagentspec.auth.OAuthEndpoints`
- `pyagentspec.auth.PKCEMethod`
- `pyagentspec.auth.PKCEPolicy`
- `pyagentspec.auth.OAuthClientConfig`
- `pyagentspec.auth.ScopePolicy`
- `pyagentspec.auth.OAuthConfig`
- `pyagentspec.retrypolicy.RetryPolicy`

Documentation: https://oracle.github.io/agent-spec/26.1.2/api/index.html (Connectivity & Resilience section)

## Class-by-class mapping

| Agent Spec class | What it does | LangGraph primitive | neograph concept | Ecosystem | Status |
|------------------|--------------|-------------------|-----------------|-----------|--------|
| `RetryPolicy` | Retry configuration for LLM/tool calls: max_attempts, backoff strategy, retry_on (error predicate) | No LangGraph primitive; adapters implement their own retry | `LlmConfig.max_retries` (int, default=1) in `_llm_config.py`. Retry implemented in `_llm_retry.py` for parse-failure recovery only. No backoff configuration; retries are immediate with error-feedback. | langchain (provider-level retries) | PARTIAL |
| `AuthConfig` | Base class for authentication configurations | No LangGraph primitive; auth is provider-level | No equivalent. Authentication is provider-level via LangChain integrations (API keys, OAuth handled by provider clients like `ChatAnthropic`, `ChatOpenAI`) | langchain (provider-level) | GAP-AS |
| `OAuthEndpoints` | Explicit OAuth endpoint configuration (auth_url, token_url, refresh_url) | No LangGraph primitive | No equivalent. OAuth endpoints are configured on provider clients, not exposed as first-class neograph config | langchain (provider-level) | GAP-AS |
| `PKCEMethod` | PKCE method enum (S256 / plain) | No LangGraph primitive | No equivalent. PKCE is provider-level for OAuth flows | langchain (provider-level) | GAP-AS |
| `PKCEPolicy` | Policy configuration for Proof Key for Code Exchange | No LangGraph primitive | No equivalent. PKCE policies are provider-level | langchain (provider-level) | GAP-AS |
| `OAuthClientConfig` | OAuth client identity / registration (client_id, client_secret) | No LangGraph primitive | No equivalent. Client credentials are passed to provider clients, not neograph IR | langchain (provider-level) | GAP-AS |
| `ScopePolicy` | OAuth scope policy configuration | No LangGraph primitive | No equivalent. Scopes are provider-level | langchain (provider-level) | GAP-AS |
| `OAuthConfig` | Configure OAuth-based authentication for a tool or transport | No LangGraph primitive | No equivalent. OAuth config is provider-level, not a first-class neograph type | langchain (provider-level) | GAP-AS |

## Status legend used

- **PARTIAL**: Feature exists but with reduced capability or different shape. Import/export requires mapping/conversion.
- **GAP-AS**: Feature exists in Agent Spec but absent in neograph. Auth/Security is provider-level (LangChain), not a first-class neograph concern. Marked "AS" for "at supplier" — delegated to the ecosystem.

## Serialization notes

**RetryPolicy**: Agent Spec serializes as a rich object with fields like:
- `max_attempts: int` (default 3)
- `backoff: BackoffStrategy` (fixed / exponential / custom)
- `retry_on: list[str]` (error types to retry on)
- `max_wait_seconds: int` (cap total retry time)

neograph's `LlmConfig` serializes only:
- `max_retries: int` (default 1)
- No backoff fields
- No error-type filtering

**OAuth family**: Agent Spec serializes full OAuth configuration trees (endpoints, PKCE, client credentials, scopes). neograph has no serialization for these — they live on provider clients and are opaque to neograph's IR.

## Export lowering

**RetryPolicy**: When exporting a neograph `Construct` to Agent Spec:
1. Read `LlmConfig.max_retries` from each LLM-mode node
2. Map to `RetryPolicy(max_attempts=max_retries)` with fixed defaults for missing fields:
   - `backoff`: assume fixed (no exponential backoff in neograph)
   - `retry_on`: empty list (neograph retries all parse failures)
   - `max_wait_seconds`: None (no cap)
3. This is a **lossy lowering** — Agent Spec consumers see a reduced retry policy.

**OAuth family**: When exporting:
1. neograph has no OAuth config to export
2. **NO-REPR** — these fields are omitted from the Agent Spec output
3. Consumers must configure auth at the provider level (outside Agent Spec)

## Import reconstruction

**RetryPolicy**: When importing Agent Spec with `RetryPolicy`:
1. Extract `max_attempts` (or default to Agent Spec's default 3)
2. Set `LlmConfig.max_retries = max_attempts`
3. Drop `backoff`, `retry_on`, `max_wait_seconds` — neograph cannot represent them
4. **Lossy import** — richer retry policies are degraded to "retry N times immediately"

**OAuth family**: When importing Agent Spec with `OAuthConfig` or subclasses:
1. **IGNORE** — these classes are not represented in neograph's IR
2. Auth config becomes the consumer's responsibility (provider-level LangChain setup)
3. A warning/emitter pattern could surface "this Agent Spec requires OAuth; ensure your llm_factory provides it"

## Verdict for interop

**RetryPolicy**: Partial fidelity. The single integer `max_retries` round-trips, but richer retry semantics (backoff, error filtering) are lost on both import and export. For neograph's target use case (LLM-driven graph construction), parse-failure retry is sufficient — but Agent Spec consumers expecting configurable backoff will see degraded behavior.

**OAuth family**: GAP-AS — no fidelity. OAuth/PKCE/Scope configuration is entirely outside neograph's IR. This is the **single biggest interop risk for this section**: Agent Specs that rely on first-class OAuth configuration (e.g., for `RemoteTool` or MCP transports with auth) cannot be fully represented in neograph. Import drops the auth config; export cannot emit it. Consumers must handle auth at the provider layer (LangChain client initialization), which is orthogonal to the Agent Spec wire format.
