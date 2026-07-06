# MCP Resources vs Tools in Multi-Step Pipelines: Field Survey (2026-07)

Date: 2026-07-06
Method: deep-research fan-out; 19 claims survived 3-vote adversarial verification (unanimous 3-0 each); 6 claims refuted and excluded. Sources cited inline; refuted claims listed at the end for transparency.

Research question: for multi-step LLM pipelines that acquire resources in one step (e.g. a ReAct node whose tool results carry MCP `resource_link`s) and consume them selectively in later steps — possibly hours later, across a checkpoint/HITL boundary — does the field have named, established patterns, or is everyone improvising?

**Bottom line: the protocol formally supports the acquire-now/hydrate-later shape, but deliberately refuses to specify everything that makes it hard (lifetime, discoverability, partial reads), and the ecosystem has no named consensus pattern. The closest precedents (Google ADK artifacts, one arXiv ResourceLink-patterns paper, LangGraph's checkpointer/store split) independently converge on the same design you are considering: typed references in checkpointed state, ephemeral on-demand hydration, re-fetch rather than materialize. Everyone is improvising, but they are improvising in the same direction.**

---

## 1. Protocol reality (2025-06-18 spec, unchanged through 2025-11-25)

### 1.1 `resource_link` is a first-class tool-result content type — and it is exactly the deferred-consumption primitive

The 2025-06-18 spec formally defines `resource_link` as a tool-result content block: "A tool **MAY** return links to Resources, to provide additional context or data. In this case, the tool will return a URI that can be subscribed to or fetched by the client." The normative keyword is MAY — servers are permitted, not encouraged, to emit them. The intended consumption model is client-side deferred fetch-or-subscribe, i.e. the acquire-now/hydrate-later pattern.

- https://modelcontextprotocol.io/specification/2025-06-18/server/tools (Tool Result → Resource Links)
- Schema: `ResourceLink extends Resource { type: "resource_link" }`, member of the `ContentBlock` union used by `CallToolResult.content` — https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/schema/2025-06-18/schema.ts (line ~753)

### 1.2 Tool-emitted links are decoupled from the resource listing

Spec, verbatim (Info callout, retained unchanged in 2025-11-25): "Resource links returned by tools are not guaranteed to appear in the results of a `resources/list` request."

Consequence: a later pipeline step **cannot** re-discover or validate a tool-emitted link by enumerating the server's resources. Tool-emitted links may reference resources that are effectively ephemeral or unlisted. The only way to know whether a link is still good is to try `resources/read` and handle `-32002 Resource not found`.

### 1.3 The spec is silent on URI stability, TTL, and expiry — by omission, not oversight

The `Resource` / `ResourceLink` data types carry only `uri/name/title/description/mimeType/size/annotations` (plus `_meta`). The only temporal metadata anywhere is the optional `annotations.lastModified` hint (ISO 8601 modification time — not an expiry). There is **no** lifetime, validity, or stability contract of any kind. Cross-checkpoint durability of a resource URI is entirely server-defined.

The 2025-11-25 revision changed nothing here: the only resource-related changelog entry is cosmetic icon metadata (SEP-973). The 2026-07-28 release candidate (SEP-2549) adds `ttlMs`/`cacheScope` freshness metadata to list/read results — independent confirmation that no freshness contract existed before, and still no expiry semantics for `resource_link` in tool results.

- https://modelcontextprotocol.io/specification/2025-06-18/server/resources
- https://modelcontextprotocol.io/specification/2025-11-25/changelog
- https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate/

### 1.4 Resources are application-driven; templates exist but no partial-read semantics

The spec: "Resources in MCP are designed to be **application-driven**, with host applications determining how to incorporate context based on their needs... implementations are free to expose resources through any interface pattern that suits their needs — the protocol itself does not mandate any specific user interaction model." Model-driven selection is listed as only one of three example patterns.

RFC 6570 URI templates (`resources/templates/list`, with completion-API argument autocomplete) are the only parameterization mechanism. `resources/read` takes a `uri` and returns full contents — **no byte ranges, offsets, or partial reads**. The protocol's pagination cursors apply to *listing* operations only, not to resource content. Any "emails 1–5 of 40" semantics exists only if a server encodes it into its own template parameters (e.g. `crm://deals/{id}/emails?offset={o}&limit={n}`) — a per-server convention, not a protocol feature.

- https://modelcontextprotocol.io/specification/2025-06-18/server/resources

### 1.5 The community explicitly did NOT converge on resources as the cross-step handoff primitive

When the protocol community standardized deferred result retrieval, it chose **task tokens, not resource URIs**. SEP-1391 (long-running operations) proposed client-generated `operation.token` polled via `tools/async/status` / fetched via `tools/async/result`. It was superseded by SEP-1686 "Tasks" (accepted; shipped in the 2025-11-25 spec as the tasks utility), which uses receiver-generated task IDs polled via `tasks/get` / `tasks/result` — and whose design record **explicitly considered and rejected a resource-based tracking approach**: "a resource-based tracking system would be convention-based rather than standardized... lack of consistent ways to distinguish status-tracking resources from ordinary resources."

Both proposals make result lifetime **explicitly negotiated and finite**: the client requests a keep-alive/`ttl`, the server may override it, and after expiry the server may discard the result. Deferred-result expiry across pauses is a recognized protocol-level problem with an official answer of "no guarantee of indefinite retention."

- https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1391 (SEP-1391, superseded)
- SEP-1686 / PR #1732; https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks

## 2. Client support and the model-controlled vs app-controlled split

MCP's `initialize` capability negotiation does not let a server determine which features a client actually consumes — `ClientCapabilities` declares only `roots/sampling/elicitation/experimental`; there is no field for "I render resources" or "I subscribe to list_changed". Apify (an MCP server vendor, maintaining an empirical 46+ client capability matrix) argues this drives servers to a lowest-common-denominator, tools-only feature set: "Ultimately this leads to the stagnation of the MCP protocol, where neither servers nor clients have motivation to adopt latest protocol features." Their matrix shows tools support as near-universal while resources support is a minority feature (e.g. ChatGPT tools-only vs Claude Code supporting resources/prompts/roots). The 2026-07-28 RC removing the initialize handshake corroborates that the core maintainers also found the negotiation model deficient.

Caveat: the stagnation mechanism is vendor opinion, not core-maintainer consensus, and a precise per-client matrix snapshot from the official docs repo could **not** be verified in this research pass (all four claims sourced to `docs/clients.mdx` were refuted 0-3 — treat any specific "client X supports resources: yes/no" table as unverified as of this report).

- https://github.com/apify/mcp-client-capabilities
- https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle

Practical consequence for a custom host like neograph: you cannot rely on the resources feature being exercised anywhere in the wild as a template; but as a *host you control*, none of the client-support weakness applies to you — you implement `resources/read` yourself via the SDK. The spec's application-driven stance (§1.4) explicitly licenses either choice: host-curated injection or wrapping `read_resource` as a model-visible tool. The protocol refuses to pick.

## 3. The closest published pattern: dual-response + ephemeral links + pinning

The one focused publication on this exact problem is Frees, "Extending ResourceLink: Patterns for Large Dataset Processing in MCP Applications" (arXiv 2510.05968, Oct 2025 — single-author preprint, not peer-reviewed). It proposes:

- **Dual-response tool results**: "(1) preview data suitable for LLM analysis, and (2) a ResourceLink for out-of-band retrieval of complete datasets... Complete dataset retrieval occurs through RESTful HTTP endpoints that bypass the LLM context entirely." (Closest published analog to acquire-now/consume-later; note its later consumption goes to the client app out-of-band and never re-enters LLM context, whereas the neograph design hydrates fractions into later prompts.)
- **Ephemeral-by-default links with explicit lifecycle**: "Resource links are ephemeral in nature, they do not represent (necessarily) persistent entities on the server, rather artifacts of tools invocations." Expiry via an `expires_at` field; a "pinning" operation (HTTP PUT, out-of-band — not an MCP operation) "removes automatic expiration, converting ephemeral queries to persistent artifacts."
- **An explicit ecosystem-gap admission**: "Practical patterns for implementing scalable reporting architectures remain undocumented," calling for community standardization of discovery mechanisms, REST endpoint contracts, and auth patterns to "reduce implementation fragmentation."

That last point is the direct answer to "is there consensus?": as of late 2025, a practitioner writing in the area states plainly that the patterns are undocumented and fragmented. Nothing found in this research pass contradicts that; the 2026 MCP roadmap acknowledges the need for "streamed and reference-based result types" without a shipped standard.

- https://arxiv.org/html/2510.05968v1

## 4. Cross-step persistence patterns in the major frameworks

### 4.1 LangGraph: checkpointer for HITL-surviving thread state; store for cross-thread memory

Official persistence docs draw a hard line: checkpointers "persist a thread's graph state as checkpoints," with sanctioned use cases of "conversation continuity, human-in-the-loop workflows, time travel, and fault tolerance" — i.e. **checkpointed state channels are the designated home for anything that must survive a HITL pause within a thread**. The `Store` abstraction is positioned for long-term, cross-thread, application-defined data ("user preferences, facts, and shared knowledge") — memory-like, **not** per-run pipeline artifacts. Nothing in the docs prescribes Store for artifact manifests between pipeline steps.

For neograph this means: a ResourceRef manifest belongs in a state channel (checkpointed, thread-scoped, HITL-surviving), not in BaseStore. This matches LangGraph's own deepagents backend split (StateBackend = "thread-scoped scratch pads; intermediate results within a single conversation").

- https://docs.langchain.com/oss/python/langgraph/persistence
- https://docs.langchain.com/oss/python/langgraph/interrupts
- https://docs.langchain.com/oss/python/deepagents/backends

### 4.2 Google ADK: references-in-state, content-in-artifact-store, always-re-fetch hydration

ADK is the framework with the most explicit official doctrine here, and it matches the design under consideration point for point:

- "Session state is generally not optimized for storing large amounts of data. Artifacts provide a dedicated mechanism for persisting larger blobs without cluttering the session state." Artifacts are addressed by filename + auto-versioned integer; state carries simple identifiers ("Store simple identifiers if needed, and retrieve the complex object elsewhere").
- `LoadArtifactsTool` implements on-demand hydration where loaded content is **deliberately ephemeral**: "The loaded artifact content is not permanently saved back into the session history, so the model should call the tool again when it needs the same artifact in a later turn." The tool's own hardcoded response says "artifact contents temporarily inserted and removed. to access these artifacts, call load_artifacts tool again." Content is spliced into the outgoing LLM request per-call and never written to session events.

That is an explicit **always re-fetch, never materialize into conversation state** stance from a major framework. Important guarantee difference: ADK artifacts are framework-managed and versioned — no URI expiry problem — whereas MCP resource_links carry no such guarantee (§1.3), so ADK gets to skip the expiry question that neograph must answer.

- https://google.github.io/adk-docs/artifacts/ (now https://adk.dev/artifacts/)
- https://google.github.io/adk-docs/sessions/state/
- https://github.com/google/adk-python — `src/google/adk/tools/load_artifacts_tool.py`

## 5. Partial/selective consumption economics

No protocol-level answer exists (§1.4): `resources/read` is all-or-nothing, and slicing exists only via server-defined RFC 6570 template parameters. The observable practice split:

- **Server-side slicing via templates/tool parameters** when the server author controls both ends (the Frees paper's REST endpoints support `?fields=`, pagination, aggregation — but outside MCP).
- **A search/query tool instead of resources** when selection is content-dependent (the tools-only lowest-common-denominator pressure of §2 pushes this way regardless).

For "emails 1–5 of the activity history": if the CRM server exposes a template like `crm://deals/{id}/emails?range={r}`, hydrating a fraction is a single parameterized `resources/read`. If it only emits an opaque `resource_link` to the whole history, the consumer must read all and slice host-side, or fall back to a search tool. Which one you get is entirely per-server — plan for both.

## 6. Recommendation for neograph (typed channels + checkpointing + HITL)

**Yes — typed ResourceRef manifest in state + on-demand hydration is the right default**, and it is the pattern the field is independently converging on (ADK doctrine, Frees dual-response, LangGraph state-channel positioning). Specifics:

1. **Manifest in a checkpointed state channel.** A typed `ResourceRef` (uri, name, mimeType, size?, lastModified?, source server, and — critically — the *tool call that produced it*: tool name + args). LangGraph's own docs designate checkpointed state as the HITL-survival mechanism; Store is the wrong tier for per-run artifacts.
2. **Hydration as a DI-injected fetch, not raw model-controlled `read_resource`, by default.** The spec is explicitly application-driven; a typed graph compiler is the "host application" and app-curated hydration (a later node declares which refs/fractions it needs; the framework fetches and renders into the prompt) preserves neograph's compile-time-validation value proposition. Offering a `read_resource` tool to agent/act-mode nodes is a legitimate opt-in for exploratory nodes, not the default.
3. **Re-fetch on hydration; cache into state only under a size threshold, as an optimization with provenance.** ADK's stance (ephemeral injection, re-call per turn) is the safest replay/checkpoint-compatible default: fetched content never silently becomes stale state. A small-content inline cache (content + fetched_at + the ref) is fine as a keyed optimization, but the ref stays the source of truth. Never store *only* materialized content without the ref — that forecloses re-derivation.
4. **Expiry after a checkpoint pause: layered fallback, fail loud at the end.** Because the spec gives zero lifetime contract (§1.3) and tool-emitted links aren't even guaranteed listable (§1.2):
   - (a) attempt `resources/read`; on success, proceed;
   - (b) on `-32002`/failure, **re-derive by replaying the recorded producing tool call** (this is why the manifest must carry tool name + args — it is the only re-derivation path the protocol reliably gives you, and it mirrors the Frees paper's "links are artifacts of tool invocations" framing);
   - (c) if replay is impossible or non-idempotent, **fail loud** with a typed error naming the ref and the pause duration — silent staleness is worse. Do not build TTL heuristics into the framework; the protocol has none to key off (until SEP-2549 `ttlMs` lands, which is worth tracking — if/when servers emit it, surface it on ResourceRef and use it to warn at interrupt time that refs may not survive the pause).
5. **Partial consumption: support both shapes.** If the server exposes RFC 6570 templates, let the ref carry template params so hydration can request a slice; otherwise fetch-and-slice host-side. Don't assume server-side ranges exist — they usually won't (§5).

Name it what it is: there is no established industry name to borrow. "Resource manifest + on-demand hydration" is descriptive and matches the three convergent precedents; you would be *naming* the pattern, not adopting one.

## Caveats

- **Client support matrix unverified.** All four claims sourced to the official `docs/clients.mdx` capability matrix were refuted in verification (0-3). The tools-dominant picture rests on the Apify matrix (a commercial server vendor) plus the structural capability-negotiation argument. Do not cite specific per-client resources-support facts from this report.
- **The Frees paper is a single-author, non-peer-reviewed preprint** — the strongest available signal on ResourceLink lifecycle patterns, but one practitioner's proposal, not adopted practice.
- **Time sensitivity is high.** The 2026-07-28 MCP release candidate is described as the largest revision since launch (removes the initialize handshake; adds `ttlMs`/`cacheScope` cache metadata via SEP-2549). Protocol-silence findings are pinned to 2025-06-18/2025-11-25 and may soften within months. Re-check after the RC ratifies.
- Absence-of-pattern findings (no named cross-checkpoint hydration pattern, no partial-read practice) are negative claims: well-supported by explicit gap admissions (Frees; SEP-1686 design record; MCP roadmap) but unfalsifiable in the strong sense.
- Two ecosystem-gap claims about LangGraph docs and MCP async were refuted in verification and are excluded; their refutation does not affect the surviving findings.

## Open questions

1. Will SEP-2549's `ttlMs`/`cacheScope` (2026-07-28 RC) extend to `resource_link` blocks in tool results, or remain list/read-only — i.e., will the protocol ever give tool-emitted links a lifetime contract?
2. Do real-world CRM-class MCP servers (HubSpot, Salesforce, Attio connectors) actually emit `resource_link`s with RFC 6570 templates for history slicing, or do they flatten everything into tools? (No verified per-server evidence surfaced; needs hands-on inspection of specific servers.)
3. What do production hosts that DO implement resources (Claude Code, Claude Desktop per Apify's matrix) do with tool-emitted resource_links specifically — auto-hydrate, surface to the user, or drop them?
4. Is tool-call replay for re-derivation safe enough in practice for act-mode (mutating) producers, or does neograph need a purity/idempotency annotation on tools before recommending replay as the expiry fallback?

## Refuted claims (excluded, listed for transparency)

- Official docs clients.mdx matrix claims (4 claims: near-universal tools vs ~8 resources clients; Claude Desktop/Code/VS Code/Cursor/LibreChat specifics) — vote 0-3 each.
- "LangGraph persistence docs provide no reference-vs-content guidance at all" — vote 0-3.
- "MCP has no standardized async/long-running mechanism as of 2025-06-18-era" — vote 0-3 (SEP-1686 Tasks shipped in 2025-11-25).

## Primary sources

- https://modelcontextprotocol.io/specification/2025-06-18/server/resources
- https://modelcontextprotocol.io/specification/2025-06-18/server/tools
- https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/schema/2025-06-18/schema.ts
- https://modelcontextprotocol.io/specification/2025-11-25/changelog
- https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1391 (SEP-1391; superseded by SEP-1686 / PR #1732)
- https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate/
- https://arxiv.org/html/2510.05968v1 (Frees, ResourceLink patterns)
- https://docs.langchain.com/oss/python/langgraph/persistence
- https://google.github.io/adk-docs/artifacts/ ; https://google.github.io/adk-docs/sessions/state/
- https://github.com/google/adk-python (load_artifacts_tool.py)
- https://github.com/apify/mcp-client-capabilities
