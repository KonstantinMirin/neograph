# FromResource + ResourceRef manifest: comparative mechanics + refined design (neograph-hgpt)

Date: 2026-07-06
Scope: folds the maintainer's 2026-07-06 refinements (R1 typed resource-readers, R2 injection through all node shapes, R3 concrete competitor mechanics) into the neograph-hgpt design, grounded against the actual code (`_agent_cycle.py`, `_di_classify.py`, `di.py`, `_scripted_registry.py`, `_dispatch.py`, `_tool_loop.py`, `lint.py`).
Companions: `mcp-resources-patterns-2026-07-06.md` (field survey), `three-layer-principle-2026-07-03.md` (governing architecture), `mcp-session-ownership-review-2026-07-05.md` (no-session-ownership verdict).

---

## Part 1 — Comparative mechanics (the CRM scenario, mechanics-level)

**Scenario under test for every framework:** a research node acquires references to `activity-history` and `email-history` for a CRM deal; later nodes selectively hydrate fractions (e.g. emails 1-5 of 40); a 5-hour human-in-the-loop pause intervenes between acquire and hydrate.

The question that matters is not "do they have artifacts" but the five mechanics: **(1) the exact ref/artifact API, (2) what is literally in the ref and where content lives, (3) how the ref survives the pause, (4) the expiry / re-derivation story, (5) typed vs stringly.**

### 1.1 Google ADK (google/adk-python) — the closest structural precedent

- **API.** `ArtifactService` abstract base with three impls (`InMemoryArtifactService`, `GcsArtifactService`, `FileArtifactService`). Signatures (verified against source):
  ```python
  async def save_artifact(*, app_name, user_id, filename, artifact: types.Part | dict,
                          session_id=None, custom_metadata=None) -> int          # returns revision int
  async def load_artifact(*, app_name, user_id, filename, session_id=None,
                          version: int | None = None) -> types.Part | None       # None version -> latest
  ```
  Versions are **auto-incremented integers** starting at 0 (GCS lists existing versions and increments the max; InMemory uses list length). `save_artifact` returns the new revision id.
- **What is in the ref / where content lives.** Session **state carries only the filename string** (`list_artifact_keys` returns filenames). The version is resolved lazily to "latest" at `load_artifact` time unless explicitly pinned. Internally an `ArtifactVersion` object carries `(version, canonical_uri)` where the `canonical_uri` includes the version — so the *durable* identity is `filename@version` even though state holds only the bare name. **Content lives in the `ArtifactService`, a store entirely separate from the `SessionService`.**
- **Pause survival.** This is the load-bearing split: `SessionService` (state/events) and `ArtifactService` (blob store) are **independent durable services**. A 5-hour pause is just time between two service calls; the filename in state survives because state is persisted, and the blob survives because the artifact store is durable and versioned. Nothing about the pause is special.
- **Expiry / re-derivation.** ADK artifacts are **framework-managed and versioned — there is no URI-expiry problem to solve.** GC is the app's responsibility (delete via the service); the framework never expires them. So ADK *sidesteps* the hard question neograph faces (a tool-emitted MCP `resource_link` has no lifetime contract). `LoadArtifactsTool` hydration is **deliberately ephemeral**: `_append_artifacts_to_llm_request` splices content into the *outgoing* LLM request per call and its docstring says "artifact contents temporarily inserted and removed... call load_artifacts tool again" — content is **never written back to session events**. That is the always-re-fetch-never-materialize stance, verbatim from a major framework.
- **Typed vs stringly.** Ref is a **string filename** (stringly). Loaded content is a `types.Part` (a typed multimodal container: inline bytes / text / function-response) — structured-ish but not a domain Pydantic model. The `user:` filename prefix is a **namespace marker**: `user:foo` is user-scoped (visible across sessions for that user); a bare name is session-scoped. `LoadArtifactsTool` retries with a `user:` prefix if a bare-name load misses.

### 1.2 OpenAI (Agents SDK + Assistants v2 files/containers)

- **API.** `client.files.create(file=..., purpose="assistants"|"user_data"|...)` returns a `File` object whose durable handle is a **`file_id` string** (`file-abc123`). Code-interpreter runs inside a **container** (`client.containers.create()` -> `container_id`); files produced by the tool are **container files** addressed by `(container_id, file_id)`, fetched via `client.containers.files.content(container_id, file_id)`.
- **What is in the ref / where content lives.** The ref is a **`file_id` string** (or `container_id`+`file_id` for tool artifacts). Content lives in OpenAI's files/containers storage service, not in thread state; a thread message references a `file_id`.
- **Pause survival.** Uploaded files via `/v1/files` are durable (see expiry below), so a `file_id` in a persisted workflow survives a 5-hour pause. **Containers do NOT survive comfortably:** a container **expires 20 minutes after its last activity** (idle-expiry). A 5-hour HITL pause **guarantees** the container is gone; its `container_id` and all container-file `file_id`s become dead (`container_expired` error on access).
- **Expiry / re-derivation.** Two-tier: `/v1/files` uploads **persist until explicitly deleted** (`DELETE /v1/files/{id}`; GC is manual) — one agent reported a 30-day retention window in some products, but the base Files API is delete-to-GC. **Code-interpreter containers expire (~20 min idle); on expiry the data is discarded and NOT recoverable** — you must re-create the container and re-run the code to re-derive the artifact. There is **no automatic re-derivation**; the application owns replay.
- **Typed vs stringly.** Entirely **stringly** — `file_id` / `container_id` are opaque strings; retrieved content is bytes/text with a MIME type. No typed contract at the ref layer.

### 1.3 Claude Agent SDK + Anthropic Files API

- **API.** Files API: upload returns a **`file_id` string**; referenced in content blocks / tool results by that id. The code-execution tool runs in a **container**.
- **What is in the ref / where content lives.** Ref = `file_id` string; content lives in Anthropic's Files store. The Agent SDK persists conversation transcripts via a `SessionStore` (`.jsonl` by default; S3/Redis/Postgres adapters), so the `file_id` strings embedded in the transcript survive a resume.
- **Pause survival.** **Files API has no TTL** — "Files persist until you delete them"; `DELETE /v1/files/{file_id}` is the only GC. So a `file_id` survives a 5-hour (or 5-day) pause as long as nobody deleted it. The **code-execution container has a hard 30-day expiry** ("Containers expire 30 days after creation"), and — usefully — the `container_id` is **reusable** via a top-level `container` param, so you can re-attach to the same container within the window. A 5-hour pause is well inside 30 days, so unlike OpenAI's 20-minute idle window, a Claude container survives the pause.
- **Expiry / re-derivation.** Files: no expiry, manual delete. Container: 30-day hard expiry, `container_expired` error past it; re-derivation = re-create container + re-run. GC is caller-driven.
- **Typed vs stringly.** **Stringly** (`file_id`, `container_id`); content is bytes/text.

### 1.4 LlamaIndex (picked over Semantic Kernel — more concrete persisted-artifact story)

- **API.** Workflow `Context` is serializable: `Context.to_dict(serializer=JsonSerializer()|PickleSerializer())` / `Context.from_dict(...)`. Artifacts/refs held between steps live inside `Context` state. Document/ingestion refs are `doc_id` / `node_id` strings resolved against a **docstore**.
- **What is in the ref / where content lives.** A `doc_id`/`node_id` string in `Context` state; content (the `Document` / `BaseNode`) lives in the docstore. `Context.to_dict` is the durable snapshot.
- **Pause survival.** Whatever is in `Context` survives if you serialize it and restore with `from_dict`. **No TTL anywhere** — persistence is entirely caller-driven.
- **Expiry / re-derivation.** No expiry. The one hash-keyed cross-pause re-derivation mechanism in the entire survey is the **`IngestionPipeline` `DocstoreStrategy`** (`UPSERTS` / `DUPLICATES_ONLY` / `UPSERTS_AND_DELETE`): documents are keyed by a content hash so re-ingesting is idempotent and skips unchanged content. That is a *dedup/re-derivation-by-hash* mechanism, closest in spirit to "replay the producing step, don't re-store what's unchanged."
- **Typed vs stringly.** Ref is a **string id**; hydrated result is a typed `Document`/`BaseNode`. So typed *content*, stringly *ref* — same shape as ADK.

### 1.5 Synthesis

| Axis | ADK | OpenAI | Claude SDK | LlamaIndex |
|---|---|---|---|---|
| Ref content | filename string (internal `filename@version` canonical_uri) | `file_id` (+`container_id`) | `file_id` (+`container_id`) | `doc_id`/`node_id` |
| Content location | ArtifactService (separate from SessionService) | Files/Containers service | Files store / container | docstore |
| Pause survival | durable versioned store + state filename | files persist; **container dies at 20 min idle** | files no-TTL; container 30-day, `container_id` reusable | `Context.to_dict` snapshot, caller-driven |
| Expiry + re-derivation | no expiry; manual GC; **always re-fetch, never materialize** | files: delete-to-GC; container: **expire, no auto re-derive** | files: no TTL; container 30-day; manual re-run | no TTL; **hash-keyed `DocstoreStrategy`** idempotent re-ingest |
| Typed? | ref stringly, content `types.Part` | fully stringly | fully stringly | ref stringly, content typed node |

**What the field agrees on (steal this):**
1. **Ref-in-state + content-in-separate-durable-store.** Universal. Nobody inlines large content into conversation/thread state. ADK's SessionService/ArtifactService split is the cleanest statement; LangGraph's checkpointer/Store split is the same idea. neograph's equivalent: a typed ref in a **checkpointed state channel**, content re-fetched from the consumer-owned MCP server.
2. **Ephemeral hydration — re-fetch, never materialize.** ADK's `LoadArtifactsTool` is the strongest precedent: content is spliced per call and never persisted back. This is exactly the replay-safe stance the purity test wants.
3. **The ref is the source of truth, never only the materialized content.** Everyone keeps the id/handle; content is derivable.

**What nobody solves (neograph must own, or deliberately punt):**
1. **Cross-pause expiry re-derivation of a *reference whose backing store has no lifetime contract*.** ADK/LlamaIndex sidestep it (framework-managed stores never expire). OpenAI/Claude *have* expiry but offer **no automatic re-derivation** — the app re-runs the producing step by hand. **MCP `resource_link` is the worst case:** no TTL contract at all (spec-silent, §1.3 of the survey), and not even guaranteed listable (§1.2). So neograph is the only one that both (a) can't manage the store and (b) wants automatic re-derivation. This is precisely why the manifest must carry **the producing tool call** — replaying it is the *only* re-derivation path the protocol reliably gives.
2. **Typed refs.** Everyone is stringly at the ref layer (best case a typed *hydrated* object, as in ADK/LlamaIndex). **Nobody carries a typed, domain-named contract on the ref itself.** This is the gap R1 targets and where neograph's typed-channel paradigm is genuinely differentiated: the ref knows its `kind`, and the reader that hydrates it declares a Pydantic output model.

**Steal:** the SessionService/ArtifactService split (checkpointed ref + re-fetch), the LoadArtifactsTool ephemerality, the "ref carries producing call" idea (implicit in ADK versioning, explicit need for us). **Skip:** framework-owned artifact stores (violates no-session-ownership + three-layer), auto-versioning integers (server owns URIs, not us), 20-minute idle containers (not our lifecycle to manage).

---

## Part 2 — The refined neograph design

### 2.0 Layer map (stated up front, per the three-layer principle)

| Piece | Layer | Justification |
|---|---|---|
| `FromResource` marker + DIBinding kind | Layer 2 (node-runtime, DI) | a resource read is side-effect-free replayable I/O -> legal at node-entry resolution |
| async DI resolution twin | Layer 2 | same driver-selects shape as async tool factories; no engine touch |
| typed resource-reader helper (`resource_reader(...)`) | Layer 2 | emits a plain LangChain `BaseTool`; the engine never knows |
| `ResourceRef` model + manifest channel | Layer 1 declares the state channel; Layer 2 writes it | channel emission is compile-time topology; the write is node-internal |
| lifting refs from `resource_link` tool results | Layer 2 (inside `tools_body`/`atools_body`) | it is node-internal cognition over a tool result; not scheduling |
| layered expiry fallback (read -> replay -> fail loud) | Layer 2 | replaying a *read* is pure; replaying a *producing tool call* is the boundary caveat (see risks) |
| lint checks | tooling | reuses `tool_requires_async_driver` |

Nothing here owns an MCP session (nmb2 verdict holds): the client / fetcher is consumer-supplied via `config['configurable']`, exactly like tool factories and the per-run token provider.

### 2.1 R1 — the typed resource-reader helper (replaces the generic `read_resource` tool)

**The trap being avoided:** a generic `read_resource(uri) -> bytes` tool is untyped — neither the developer nor the LLM knows the output shape, and `ToolInteraction.typed_result` degrades to opaque str (it currently falls through `_render_tool_result_for_llm` to `str(result)`). That breaks the typed-channel paradigm the same way every untyped tool does.

**The helper** — a factory that turns *(uri template + output model + name)* into a properly typed `BaseTool`, so declaring a domain-named resource-reader is trivial:

```python
def resource_reader(
    name: str,                         # domain-named: "read_deal", "read_emails"
    *,
    uri_template: str,                 # RFC 6570: "crm://deals/{deal_id}/emails?range={range}"
    output_model: type[BaseModelT],    # the declared Pydantic contract the read yields
    description: str,                  # LLM-facing tool doc (what this reader is for)
    parse: Callable[[bytes, str], BaseModelT] | None = None,  # bytes+mime -> model; default: model_validate_json
    budget: int = 0,
) -> BaseTool:
    """Emit a typed, async LangChain BaseTool that reads ONE known resource KIND.

    The tool's args schema is derived from the uri_template's RFC 6570 vars
    (deal_id, range, ...) so the LLM (agent mode) or the caller (scripted/DI)
    supplies typed parameters. At call time it resolves the fetcher from
    config['configurable']['mcp_resource_fetcher'] (consumer-owned, async),
    reads the interpolated URI, and parses the blob into output_model.
    Returns the typed model instance -> ToolInteraction.typed_result carries
    the Pydantic model, NOT a repr string."""
```

- **Composition with existing machinery.** It emits a `StructuredTool` whose `coroutine` is set and `func` is None — i.e. it is **async-only**, so `is_async_only_tool()` already returns True and the existing `tool_requires_async_driver` lint fires for free. It plugs into `Node(tools=[read_emails])` unchanged; `register_bound_tool_factories` auto-registers it via `Tool._bound_tool`; the agent cycle's `tools_body` executes it via the existing budget/tracker path. **No new tool infrastructure.**
- **`ToolInteraction.typed_result` flow.** In `atools_body` line ~443, `result = await tool_fn.ainvoke(...)` becomes the `output_model` instance; `ToolInteraction(..., typed_result=result)` already carries it; `_render_tool_result_for_llm` already renders a BaseModel via the node renderer (not `str()`). So a downstream node consuming `tool_log` gets structured data. **Zero change to `tools_body`.**
- **The `read_blob` escape hatch.** For genuinely opaque content, ship one `read_blob(uri) -> BlobResult` where `BlobResult` is a tiny declared model (`bytes_b64`/`text`, `mime`, `size`). It is the *exception* — documented as "use a typed `resource_reader` unless the content has no schema." It keeps typed_result honest (a `BlobResult`, not raw bytes) while admitting the untyped case.

### 2.2 R2 — `FromResource` injection through all three node shapes (parse/consume split)

The load-bearing DX is that `FromResource` **splits resource PARSING from resource CONSUMPTION** — the parse is debugged separately from the logic that uses the parsed model, and (with async across all primitives) the composition stays robust. It must inject cleanly through all three node shapes:

```python
# 1. Deterministic node (JSON -> model): resolver fetches + model_validate_json at node entry
@node(outputs=Assessment)
async def assess(doc: Annotated[ContractDoc, FromResource("crm://deals/42/contract")],
                 claims: Claims) -> Assessment: ...

# 2. Think node (LLM parses text): FromResource yields raw text; a schema-aligned LLM parse
#    is the DOCUMENTED recipe for unstructured content (NOT silent LLM parse inside DI).
#    v1: FromResource requires an explicit parser for text/*; the think-node pattern is
#    "fetch text via FromResource(text) -> feed as prompt input -> LLM emits the model".
@node(outputs=Summary, prompt="Summarize: ${history}", model="...")
async def summarize(history: Annotated[str, FromResource("crm://deals/42/emails", mime="text")]): ...

# 3. Sub-construct (multi-step translation): the port param is FromResource-fed; the
#    sub-pipeline does multi-step parse -> the boundary output surfaces the typed model.
construct_from_functions("hydrate", [fetch, normalize, validate],
                         input=RawHistory, output=CleanHistory)
```

**Why this works against the code — the async resolution twin.** The scripted shim (`_scripted_registry._register_node_scripted`) resolves each param via `binding.resolve(config)` **synchronously**. `FromResource` cannot resolve synchronously (the fetch is `await`ed). The clean mechanism reuses the existing driver-selects pattern (`ScriptedDispatch.execute` fails loud on an awaitable; `aexecute` awaits it — `_dispatch.py:119-158`):

- Add `DIKind.FROM_RESOURCE` to `di.py` and an async resolver `DIBinding.aresolve(config)` that `await`s the fetcher. `resolve()` (sync) for a `FROM_RESOURCE` binding **raises `ConfigurationError`** ("FromResource requires the async driver; use arun()") — the khff fail-loud shape already used at `_agent_cycle.py:390` and `_dispatch.py:134`.
- The scripted shim gains an async branch: **if any binding is `FROM_RESOURCE`, the shim is a coroutine** (it must `await` the fetch). `ScriptedDispatch.aexecute` already awaits an awaitable shim result; `execute` already fails loud on one. **So `FromResource` threads through the existing sync/async dispatch with no new dispatch class** — exactly the property that made async tool factories a one-line twin.
- Concretely: `_register_node_scripted` builds both `scripted_shim` (sync, resolves non-resource bindings; raises via `binding.resolve` if a resource binding is hit under sync) and, when a resource binding is present, makes the shim `async def` so it `await`s `binding.aresolve(config)`. The compiler already routes `aexecute` under `arun()`.

**Three-surface parity.** `FROM_RESOURCE` is classified in `_classify_di_params` (the ONE classifier for `@node` + `@merge_fn`), so declarative `Node.scripted()` and programmatic `Node() | ...` inherit it. The 6-cell matrix (3 surfaces x {run/arun}) applies; the `run()` cells assert the `ConfigurationError`.

### 2.3 FromResource v1/v2 boundary (re-cut given the typed reader)

The typed-reader helper (R1) changes what `FromResource` itself must do. **`FromResource` and `resource_reader` are two faces of the same read**, split by who drives the parameters:

- **`FromResource` = app-curated, static/templated-from-FromInput hydration.** The *consumer's field* declares which resource KIND it consumes; the framework fetches at node entry. This is the default for deterministic/think/sub-construct nodes (R2).
- **`resource_reader` tool = model-curated hydration.** The *LLM* decides which slice to pull (emails 1-5 vs 30-35) inside an agent node. Opt-in for exploratory nodes.

**v1 (this molecule):**
- `FromResource(uri)` — **static URI string**. mime `application/json` -> `model_validate_json` into the declared model. mime `text/*` -> requires an explicit `parse=` callable OR is typed `str` (raw text passthrough for the think-node recipe). **No silent LLM parse in DI** (hidden cognition + cost — explicitly banned).
- Async resolver twin + fail-loud on `run()`.
- `resource_reader(...)` helper emitting a typed async `BaseTool` (agent-mode hydration), + `read_blob` escape hatch.
- Missing/fetch-fail/validation-fail -> typed error at node entry naming param + uri + node. **No retry in v1** (consumer wraps with Loop/skip_when).

**v2 (follow-on, rides on the manifest):**
- **Templated URIs** — `FromResource("crm://deals/{deal_id}/emails?range={range}")` interpolated from `FromInput` values / manifest ref params. (The `resource_reader` helper already does RFC 6570 interpolation, so v2 `FromResource` borrows that machinery.)
- **Manifest-driven hydration** — `FromResource(ref=...)` reads a `ResourceRef` from the manifest channel instead of a literal URI, enabling the acquire-now/hydrate-later flow with layered expiry (2.4).
- `ttlMs`/`cacheScope` surfacing when SEP-2549 servers ship it (warn at interrupt time that a ref may not survive the pause).

### 2.4 The `ResourceRef` manifest

**Model shape** (a typed Pydantic model, the exact opposite of the field's stringly refs):

```python
class ProducingCall(BaseModel, frozen=True):
    tool_name: str                    # the tool that emitted this link
    args: dict[str, Any]              # its args -> replay is (tool_name, args)

class ResourceRef(BaseModel, frozen=True):
    uri: str                          # the resource_link uri (server-defined stability)
    kind: str                         # domain KIND: "email-history", "activity-history"
    server: str                       # which MCP server (for the consumer's fetcher routing)
    producing_call: ProducingCall     # THE re-derivation path (survey rec 4b)
    mime: str | None = None           # hint from the resource_link block
    size: int | None = None           # hint
    fetched_at: str | None = None     # ISO ts when last hydrated (provenance)
    ttl_ms: int | None = None         # v2: SEP-2549 when servers emit it
```

**Which state channel.** A new `neo_`-prefixed **checkpointed** channel, `StateKeys.resource_manifest(field)` (builder alongside `agent_tool_log`), holding `list[ResourceRef]`. Checkpointed = HITL-surviving (LangGraph designates checkpointed state, not Store, as the pause-survival tier — survey §4.1). It is `neo_`-prefixed so `_strip_internals` keeps it out of user-facing output but the checkpoint retains it.

**Where refs are LIFTED from `resource_link` content.** In the agent cycle's tools bodies (`_agent_cycle.tools_body` / `atools_body`), the loop `for tc in tool_calls:` already holds the raw tool `result` (line ~404/443). A `resource_link` content block arrives inside that result (MCP tools return LangChain content lists). The lift is a **new helper `_lift_resource_refs(result, tc) -> list[ResourceRef]`** called right where `ToolInteraction` is built:
- scan `result` content for `type == "resource_link"` blocks;
- for each, build a `ResourceRef(uri=block.uri, kind=<from tool/link>, server=<tool's server>, producing_call=ProducingCall(tool_name=name, args=tc["args"]), mime=block.mimeType, size=block.size)`;
- return them; the body writes them to `manifest_key` alongside `tlog_key`.

This is **a `tools_body` concern, not `_finish_tool_loop`** — the producing call (`tc["name"]`, `tc["args"]`) is only in scope during the tool-execution loop, and lifting per-call keeps the ref co-located with the `ToolInteraction` it corresponds to. It is Layer-2 node-internal cognition over a tool result (no engine verb). One new helper + one extra return key in both twins; the router/parse are untouched.

**Layered expiry handling (concrete node-level behavior)** — when a later node hydrates a ref (v2 `FromResource(ref=...)` or a manifest-driven reader):
1. **read**: `await fetcher.read(ref.uri)`. On success -> parse into the declared model, proceed. Update `fetched_at`.
2. **replay**: on `-32002` / fetch failure -> **replay the producing call**: re-invoke `ref.producing_call.tool_name` with `ref.producing_call.args` (the consumer's fetcher/tool registry resolves it), take the fresh `resource_link`, read that. This is the ONLY protocol-reliable re-derivation path (survey rec 4b; MCP links are "artifacts of tool invocations").
3. **fail loud**: if replay is impossible (producing tool absent) or non-idempotent (see risks) -> raise a typed `ResourceExpiredError(ref=..., node=..., pause_context=...)`. **Silent staleness is worse than a loud failure.**

Step 2 is the one place a "read" can smuggle in a "call". If the producing tool was an `act`-mode (mutating) tool, replay is unsafe — see Risk 1; v1 does NOT do replay (no manifest), so this lands with v2 and requires the idempotency annotation.

### 2.5 Sequencing — the ticket cut

Honest v1/v2 boundary: **v1 rides entirely on existing machinery** (DI classifier, sync/async dispatch driver-select, tool factories, lint `tool_requires_async_driver`, tool_log rendering). **v2 needs the manifest channel + the resource-link lift + expiry replay**, which is genuinely new state topology and depends on the async DI resolver twin from v1.

| Ticket | Scope | Layer | Depends on | v1/v2 |
|---|---|---|---|---|
| **hgpt.1 — `resource_reader` typed helper + `read_blob`** | The R1 factory: uri-template -> args schema, async fetch via `config['configurable']['mcp_resource_fetcher']`, parse into `output_model`, emit async `BaseTool`; `read_blob` escape hatch. Lint already covers it (async-only). Tests: agent-mode hydration, `typed_result` carries the model, `tool_requires_async_driver` fires. | L2 | nothing open (fetcher is consumer-supplied, like tool factories) | **v1** |
| **hgpt.2 — `FromResource` marker + async DI resolver twin** | `DIKind.FROM_RESOURCE`, `DIBinding.aresolve`, `resolve()` fail-loud on sync; `_classify_di_params` classifies it; scripted shim async branch (coroutine when a resource binding present). Static URI, json->model / text->parser-or-str. Lint reuses `tool_requires_async_driver` for FromResource+run(). Tests: 6-cell (3 surfaces x run/arun); run() cells assert `ConfigurationError`. | L2 | hgpt.1 (shared fetcher config key + parse convention) — soft dep; can parallelize | **v1** |
| **hgpt.3 — `ResourceRef` manifest + resource-link lift** | The typed model, `StateKeys.resource_manifest`, `_lift_resource_refs` in `tools_body`/`atools_body`, checkpointed channel wiring in compile, `_strip_internals` exclusion. Tests: a stub MCP tool returns a `resource_link`; assert a `ResourceRef` with the producing call lands in the manifest and survives an interrupt/resume. | L1 channel + L2 lift | hgpt.2 (async resolver established) | **v2** |
| **hgpt.4 — manifest-driven hydration + layered expiry + templated URIs** | `FromResource(ref=...)`/templated URIs; the read->replay->fail-loud fallback; the idempotency gate on replay (only replay `think`/read-only producers; refuse `act` producers unless annotated). Tests: expired ref -> replay -> success; non-idempotent producer -> fail loud. | L2 | hgpt.3 (manifest exists) + the tool idempotency annotation (may spawn its own ticket) | **v2** |

hgpt.1 + hgpt.2 are the shippable v1 (interim patterns in the survey become first-class). hgpt.3 + hgpt.4 are v2 and depend on each other. None owns an MCP session; the no-session-ownership structural guard extends to assert `resource_reader`/`FromResource` never call `client.session(...)`.

### 2.6 The pattern name

The survey says neograph would be *naming* this, not adopting it. Proposed name for the docs:

> **Typed Resource Manifest with Ephemeral Hydration** — short form **"the manifest/hydrate pattern"**. A typed `ResourceRef` (carrying its producing call) lives in checkpointed state; consumption re-fetches on demand and never materializes content into durable state; expired refs are re-derived by replaying the producing call, or fail loud.

The differentiator vs ADK/LangGraph/LlamaIndex, stated in one line for the docs: **"the ref is typed and self-healing"** — typed because `kind` + a declared reader model replace the stringly `file_id`; self-healing because the producing call travels with the ref, which is the only re-derivation the protocol's lifetime-free `resource_link` allows.

---

## Part 3 — Adversarial self-check (top 3 risks)

**Risk 1 — Replay-safety of the producing call (the purity-test violation vector).** The expiry fallback (2.4 step 2) re-invokes `ref.producing_call`. A *read* is replay-safe, but the tool that *produced the link* may not be — an `act`-mode (mutating) CRM tool that emits a `resource_link` as a side effect of creating something is NOT idempotent; replaying it double-writes. This is exactly the purity test's boundary: a read may live in a node body, a mutation may not. **Mitigation:** v1 ships no replay (no manifest, so the risk doesn't exist yet). v4/hgpt.4 gates replay on a **tool idempotency annotation** — only `think`/read-only producers are replay-eligible; an `act` producer with no `idempotent=True` annotation makes replay refuse and **fail loud** rather than double-mutate. This is also survey open-question #4, so it is a known unknown, not a blind spot. The annotation likely deserves its own small ticket.

**Risk 2 — `FromResource` text into a think node exceeds context.** R2 case 2 fetches `text/*` and feeds it as prompt input; a 40-email history can blow the model's context window at hydration time, and because DI resolution is at node entry, the failure surfaces as an opaque provider 400 *inside* the LLM call, not as a clean neograph error. **Mitigation:** (a) the typed `resource_reader` with a **templated slice** (`?range=1-5`) is the recommended path precisely because it bounds size at the *fetch*, not after; (b) `FromResource(text)` for whole-history hydration should carry a documented size guard — a `max_bytes` on the marker that fails loud at node entry ("resource X is N bytes, exceeds max_bytes M; use a templated `resource_reader` to slice") **before** the blob reaches the prompt. This turns a confusing downstream 400 into a fail-loud-at-entry naming the ref. The parse/consume split (R2) actually helps here: because parsing is a separate node, an oversized fetch fails in the parse node with a clear locus, not tangled into consumption.

**Risk 3 — The lift point silently drops refs / servers that don't emit `resource_link`.** `_lift_resource_refs` only captures refs when the server chooses to emit `resource_link` blocks (spec MAY, not SHOULD — survey §1.1). A CRM server that flattens everything into tool results with inline data produces an **empty manifest**, and a later `FromResource(ref=...)` finds nothing — a confusing "ref not found" far from the cause. Worse, tool-emitted links aren't guaranteed listable (§1.2), so there's no enumeration cross-check. **Mitigation:** (a) make the empty-manifest case explicit — if a node declares manifest-driven hydration but the upstream agent node produced zero refs of the needed `kind`, **fail loud at assembly-adjacent lint** ("node hydrates kind='email-history' but no upstream producer emits resource_links of that kind"), not at runtime; (b) document that manifest hydration requires a server that emits `resource_link`s, and the fallback for flat servers is the typed `resource_reader` tool (which fetches by constructed URI, not by lifted ref) — so the two R1/R2 mechanisms cover each other's gaps. This keeps the design honest about the protocol's "servers are permitted, not encouraged" reality.

---

## Appendix — code touchpoints (for the implementer)

- `di.py`: `DIKind.FROM_RESOURCE`; `DIBinding.aresolve`; `resolve()` fail-loud branch.
- `_di_classify.py`: `_classify_di_params` recognizes `FromResource` marker (new class alongside `FromInput`/`FromConfig`).
- `_scripted_registry.py`: `_register_node_scripted` builds an async shim when a resource binding is present (coroutine so `aexecute` awaits, `execute` fails loud — no new dispatch class).
- `_dispatch.py`: unchanged — `ScriptedDispatch.execute`/`aexecute` already implement the driver-select (119-158).
- `tool.py`: `resource_reader()` factory + `BlobResult` model + `read_blob`; emits async-only `StructuredTool` so `is_async_only_tool` + lint already cover it.
- `_agent_cycle.py`: `tools_body`/`atools_body` gain `_lift_resource_refs(result, tc)` + a `manifest_key` return (v2 hgpt.3).
- `_state_keys.py`: `StateKeys.resource_manifest(field)` builder.
- `lint.py`: `_check_async_only_tools`-adjacent — flag `FromResource` param on a node as `tool_requires_async_driver` intent (needs arun()).
- structural guard: extend the no-MCP-session-ownership guard to `resource_reader`/`FromResource` (never `client.session(...)`).
