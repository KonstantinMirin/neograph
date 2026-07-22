# Changelog

All notable changes to NeoGraph will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.3] - 2026-07-22

A hotfix release completing the 0.7.2 stringly-`"null"` repair, cut directly from `main`.

### Fixed

- **Stringly-`"null"` coercion now reaches interiors of `Optional`-wrapped nested models and lists.** 0.7.2 coerced a stringly-`"null"` on a *direct* Optional field (including `list[X] | None` and `Model | None`), but its recursive descent into nested models/lists only fired when the field annotation was a *bare* `BaseModel` or `list[...]` — an `Optional` wrapper (`parent: Company | None`, `products: list[Product] | None`) is a `Union`, so both descent branches were skipped and a stringly-`"null"` on an *interior* field (e.g. `parent.langs: list[str] | None`, `products[i].price: int | None`) still reached Pydantic raw and aborted the node with `list_type`/`int_parsing`. `_apply_null_defaults` now peels a single `Optional` wrapper (via the new single-site `_unwrap_optional` seam) before the nested-model and list-item descent, so the coercion reaches every Optional scalar leaf at any depth. Legit interior values are preserved; a required (non-Optional) field receiving `"null"` still fails loud. Covered by deterministic regressions plus a Hypothesis property sweep over randomized Optional/nested/list topologies.

## [0.7.2] - 2026-07-16

A hotfix release: one bug fix, cut directly from `main`/v0.7.1 (not from `develop`, which carries unreleased 0.8.0-track work).

### Fixed

- **Structured parse no longer aborts on a stringly-`"null"` LLM emission for Optional fields.** Some models (observed: GLM 5.2) intermittently emit the JSON *string* `"null"` (or `"none"`/`""`) for an `Optional` numeric/enum field instead of a real JSON `null` — `json_repair` leaves the string intact and Pydantic then raised `int_parsing`/`enum`, aborting the whole node mid-run. `_apply_null_defaults` now recognizes the stringly-null sentinel on Optional fields only (verified via the field's annotation, never the value) and coerces it to `None` before the existing null/default disposition runs, recursing into nested models and `list[BaseModel]` items exactly as it already did for real `None`. A required (non-Optional) field receiving `"null"` still fails loud.

## [0.7.1] - 2026-07-15

A maintenance release: two bug fixes surfaced by a downstream consumer, plus documentation and example additions that accrued after the 0.7.0 tag. No new public API — the `Portal` dynamic-handoff surface remains on `develop` for 0.8.0.

### Fixed

- **json_mode: a `repair_json` blowup is retryable, and truncation gets a continuation re-prompt, not a blind re-issue** (`neograph-8uoot`). A max_tokens-truncated response sent `json_repair`'s recursive-descent parser over the stack limit; the call sat outside the parse guard, so the error escaped the retry loop and killed the run. `repair_json` failures now become `ExecutionError` and enter the same error-feedback retry as every other malformation. And a `finish_reason == 'length'` / `stop_reason == 'max_tokens'` response with no parseable payload is re-prompted with a continuation directive — the truncated reasoning is fed back and the model is told to emit ONLY the JSON payload — instead of the generic repair message (a blind re-issue at temperature=0 would likely reproduce the same runaway). Truncation logs a `llm_response_truncated` warning for observability. `TestRepairJsonGuarded` pins the guard structurally.
- **dict-output reference params take peer priority over port classification** (`neograph-f45ad3b`). `_identify_port_params` classified any sub-construct param whose type subclasses the construct input as a port param, even when its name was a `{upstream}_{output_key}` reference to a dict-output producer — so a downstream node consuming e.g. an enriched subclass of the construct input failed assembly with a spurious `neo_subgraph_input` type mismatch. Dict-output references now get the same priority as peer `@node` names; true port params are unchanged.

### Added

- **Example 27 — ForwardConstruct imperative agent-wiring showcase.** A runnable walkthrough of `branch` / `self.loop` / `self.each` / `self.ensemble` / `self.interrupt` in the imperative `forward()` surface (a 0.7.0 feature), pinned by `tests/test_example_forward_wiring.py`.

### Documentation

- README refreshed to the current 0.7 surface (MCP client, async-native four-verb execution, durable resume, BAML-style rendering, one-line observability).
- AGENTS.md de-staled: reference sections, guardrails, and north-star positioning brought current.
- Design notes added under `docs/design/` (Agent Spec ratification, TypeScript feature-parity study).

## [0.7.0] - 2026-07-13

0.7.0 finishes the imperative `ForwardConstruct` surface, removes one born-redundant Loop feature, and hardens MCP identity + error surfacing to "fail loud and precise" at every boundary.

### Breaking

- **`Loop.history` / `@node(loop_history=)` / the `neo_loop_history_{node}` state field removed** (`neograph-eef83`). It duplicated data the self-loop already surfaces: a self-loop node's output field is itself an append-list of every iteration (`result[node]`), so no separate history flag is needed. `history` was legal only on the Node self-loop — exactly where the main field already collects — making it fully redundant. It was a schema-first speculative field, superseded on its birth-day by the append reducer, and `TestLoopHistoryRemoved` guards against its return.

### Added

#### ForwardConstruct expressiveness parity (`neograph-e9zse`)

The imperative `forward()` form can now express every topology the declarative form can, tracing to identical IR:

- **`self.each(body, over=, key=, on_error=)`** — fan-out over a sub-construct with a custom key (not just per-node, not just `key="label"`).
- **Loop bodies accept nested deferred builders** — `self.loop(body=[... self.each(...) ...])`, the fan-out-inside-loop shape (the cascade topology) that was previously infeasible.
- **`self.ensemble(...)`** — Oracle ensemble tracing; **`self.interrupt(...)`** — HITL Operator tracing. Both form-aware (node → `Node | Modifier`, list → `Construct | Modifier`), sharing one wrap builder, emitting only existing IR.
- Fan-in, multi-output dict, and `skip_when` verified through tracing.
- Branch richness and `try/except` are capped as documented v1 limits with a loud escape to the declarative form (proxy-vs-proxy comparisons now raise at trace time instead of mis-tracing).
- A parity test matrix + ratchet enforces "traced IR == declarative IR" per topology.

#### Other

- **`@node(merge_model=)` / `@node(map_on_error=)` decorator parity** (`neograph-d5pvl`) — forward to `Oracle.merge_model` / `Each.on_error`.
- **`construct_from_module` collects module-level sub-constructs** (`neograph-xv9ay`) — one member-selection predicate shared with `construct_from_functions`; a well-formed sub-construct at module level is wired, not silently dropped (an output-less stored-pipeline artifact is skipped with a `ConstructArtifactSkipped` warning).
- **`FakeMcpSession` per-tool tri-modal values** (`neograph-4o7yu`) — script a composite's N same-tool calls by args or as an ordered sequence.
- **Idempotent repeat-call guard** in the agent cycle — an identical repeated tool call is served from the cycle's history.

### Fixed

#### MCP identity — per-call fresh on every surface and transport

- **No mid-run token freeze** (`neograph-qslrx`). The static `token_provider` bearer was baked into the connection at build and reused for the whole run via the RUN_ID tool cache; a run whose tool phase outlived the IdP token lifespan sent a stale token. `token_provider` is now wrapped in a per-request `httpx.Auth` (`_TokenProviderAuth`), unifying it onto the same mechanism the OAuth `HttpServer.auth` path uses — identity is re-resolved per request; a static-string provider still pins.
- **stdio session identity re-resolved per call** (`neograph-hs3mr`) — `McpSession.call()` over stdio no longer mints once at `__aenter__`. Identity is now per-call fresh on every surface and transport; a constant provider pins. Guarded by `TestNoMintOnceTokenOnInstanceState`.

#### MCP error surfacing — fail loud AND precise

- **Bare leaf at every transport exit boundary** (`neograph-2itlh`, `neograph-lcrwd`). anyio wraps exceptions from the streamable-http/stdio transports in `ExceptionGroup`s; a consumer catching a specific exception type around an MCP call now gets that bare type, not a wrapper. Fixed at the build/discovery path (`get_tools`, the factory, `_resilient`) and the mid-session boundaries (`McpSession.call`/`list`, resource fetcher/replayer), with a ratcheting AST guard (empty allowlist) so no boundary can re-wrap again. CancelledError is exempt (cooperative cancellation); multi-leaf groups are preserved.

#### Other

- Design-doc `@node(output=)` → `outputs=` drift + a permanent guard over `docs/design/` (`neograph-1h02l`).
- De-tautologized the ForwardConstruct parity ratchet — `REQUIRED_CAPABILITIES` is now an independent source of truth (`neograph-zrcln`).
- Resource-fetcher fail-loud monopolized in `_require_fetcher`; bare module-level logger binding enforced; assorted assertion-strength and guard hardening.

## [0.6.0] - 2026-07-10

0.6.0 is a large, backward-compatible release over 0.5.0. It adds a full MCP client battery, first-class async execution, typed resource hydration, an agent/act subgraph rework, and a compile-time-verified documentation pipeline. No public 0.5.0 API was removed or changed incompatibly.

### Added

#### MCP client battery — new optional `neograph[mcp]` package

A second top-level import package, `neograph_mcp`, ships in the same distribution. `neograph` core stays MCP-free (importing `neograph_mcp` without the extra fails loud with an install hint). neograph never owns an MCP session — the adapters own connection lifecycle; neograph owns typing, wiring, per-run identity, and replay-safety.

- **`mcp_tool_factories(servers)`** — connect once, discover a server's tools, and get a `{name: factory}` dict you slice per node for least-privilege binding.
- **`mcp_tool_factory(server, spec, tool_name=...)`** — lazy single-tool factory with **zero network at construction** (offline compile/test paths) and a gateway-federated `<peer>-<tool>` → bare `Tool(name)` rename.
- **`mcp_session`** — call N federated tools over **one** connection from a scripted composite (`async with mcp_session(...) as s: await s.call(tool, args, output_model=...)`).
- **`mcp_run_context`** — run-scoped connection reuse across an agent's ReAct supersteps (1 connect, not N), reconnect-safe across interrupt/resume (the held session is a config-only key that never enters the checkpoint).
- **Typed tool results** via `output_model=` / `output_models=` — rehydrate a tool's `structuredContent` into your Pydantic model; `ToolInteraction.typed_result` *is* the model.
- **Per-run identity** via `token_provider` — rides as a tool argument over stdio, a bearer header over streamable-http; framework-carried (never LLM-chosen), never enters state, the checkpoint, or the schema fingerprint.
- **Production auth** — `HttpServer.auth` + `client_credentials_auth` wrapping the MCP SDK's OAuth 2.1 / client-credentials / JWT `httpx.Auth` providers (token refresh without reconnect).
- **`mcp_prompt_source`** — consume server-provided prompt templates (`prompts/get`), closing the third MCP primitive (tools + resources + prompts).
- **Progress notifications** from long-running tools surfaced into `stream()` / `astream()` as `McpProgress` events (never enter state/checkpoint).
- **Transport resilience** — per-call timeout + bounded retry on transport errors only; an `isError` result is never retried; non-idempotent tools are never replayed after an ambiguous failure; a retry counts against tool budget once.
- **Gated mutations** — `gate_tools_when=` pauses a checkpointed run before a mutating tool fires; approve runs it exactly once, deny never runs it.
- **Keyless test fakes** — `neograph_mcp.testing`: `FakeMcpSession` + fake tool factory / resource fetcher, structurally parity-pinned to the real session's `output_model` contract.

#### Async execution — one pipeline, four verbs

- **`run` / `arun` / `stream` / `astream`** — the same compiled graph runs under any verb; the framework carries the sync/async duality (no async flag at compile time, no second pipeline).
- Async scripted/raw node bodies, async LLM + tool seams, async checkpoint helpers.
- **Driver ↔ checkpointer matching fails loud both directions** — a sync `run()` against an async-only saver (or the reverse) raises a `ConfigurationError` naming the fix, never half-persists.
- Fail-loud on an `async def` body under sync `run()`.
- Async agent turns execute concurrent tool calls **concurrently** while preserving sequential order + per-tool budget semantics.

#### Resource hydration — MCP resources as typed inputs

- **`Annotated[Model, FromResource(uri)]`** — fetch + validate a resource at node entry, before your function runs (async DI twin; fails loud under `run()`).
- **`resource_reader()`** typed domain reader + `read_blob` escape hatch.
- **ResourceRef manifest** — runtime-discovered `resource_link`s lifted into a checkpointed manifest; downstream nodes hydrate by domain kind with layered, self-healing expiry (read → replay idempotent producer → `ResourceExpiredError`), templated URIs, `max_bytes` caps, and a per-run fetch cache.

#### Agent / act subgraph rework

- Agent/act nodes compile to an **inline agent-subgraph** (the ReAct monolith is gone), parsing the final ReAct turn as output to eliminate double-generation, with opportunistic parse-first structured output.
- **Fan-over-agent auto-wrap** — `Oracle` / `Each` / `Loop` over an agent/act node, with input-port synthesis for upstream inputs.
- **`ask_human()`** typed mid-loop HITL sugar + a safety lint; opt-in framework-generated tool-budget preamble.

#### di_inputs — resolved DI values reach prompt templates

- `FromInput` / `FromConfig` params are usable as `{var}` in `think`/`agent`/`act` prompt templates (opt-in via a `di_inputs`-aware compiler); on a name collision the upstream output shadows the di_input. Lint gains a matching third column.

#### Prompt compiler

- **`DefaultPromptCompiler`** + exported fail-loud prompt primitives.
- **Public `compile_prompt()`** for eval harnesses — byte-identical prompts inside and outside the graph.
- Container rendering deltas (fan-in dicts, `ToolInteraction` lists).

#### Verifiable docs (neograph.pro)

- **API manifest generator + pytest freshness guard** — any public-surface delta fails the test suite.
- **remark-api plugin** — validates and autolinks backticked symbol references against the manifest at build time; a dotted `Type.member` ref to a missing member fails the Astro build.
- **Manifest-generated reference sections** with kind-namespaced anchors, dotted `Type.member` refs linking to field-row anchors, and a cross-link **coverage-guard capstone**.
- **Docs-snippet execution testing** — the Python snippets embedded in the docs are executed and drift-checked.

#### Other

- **`json_mode`** sends the provider-native `response_format={"type": "json_object"}`.
- **Per-run id primitive** — `StateKeys.RUN_ID` (`_neo_run_id`), fresh per attempt, stable within a run, config-only.
- **`observe=`** opt-in Langfuse auto-attach + finalize flush.
- **`Each(on_error='collect')`** — partial-failure collection for `.map` fan-outs.
- **Public `neograph.testing` fakes** — `FakeLLM` / `install_fake_llm`.
- Trace span hygiene — named node runnables, node metadata on the engine's own spans.

### Fixed

- **Checkpoint auto-rewind hardened.** Schema fingerprints are now structural (a same-`__qualname__` model with a changed field type invalidates); a pruned history fails loud instead of silently resuming from the tip; a non-coercible field-type change rewinds rather than raising a raw `ValidationError` first. (Detection + targeted re-execution — not arbitrary state migration.)
- Parent checkpointer + conditions threaded into branch-arm sub-constructs; branch-arm descent fixed across the IR tree walks.
- Fail-loud on a scripted node that ran and returned `None`; fail-closed on an absent `Each` over-root.
- Error hierarchy: `ResourceExpiredError` / `NonIdempotentReplayError` re-parented under `ExecutionError`.
- `_run_cache` single-flight hardening; structured-output retry parity with `json_mode`; `DefaultPromptCompiler` all-DI think-node crash; null coerced to `default_factory()` for list/dict fields.
- 5 API-drifted documentation snippets repaired (caught by the new docs-snippet testing).

### Packaging

- **Two top-level packages in one distribution** (`neograph` + `neograph_mcp`); one version, one tag, one Trusted-Publishing release.
- New extras: **`mcp`**, **`mcp-examples`** (alongside `langfuse`, `dev`). `pip install neograph` pulls zero MCP dependencies; `pip install neograph[mcp]` adds the battery.
- **`py.typed` shipped for both packages** — `neograph_mcp`'s public API is typed.

### Examples + Docs

- New runnable examples: MCP client selective binding, MCP resources via `FromResource`, gateway single-tool binding, and a composite `mcp_session` walkthrough — all keyless against a real stdio FastMCP demo server.
- New concept pages: **MCP Integration**, **Sync & Async Execution**, **Resource Hydration**, plus the verifiable-docs API reference and cross-linked symbol pages.

## [0.5.0] - 2026-06-04

### Breaking

- **`configure_llm` removed** — pass `llm_factory=` to `compile()` instead. The module-level singleton is gone; LLM configuration is now bound at compile time and captured by closure, so multiple compiled graphs in one process can use different factories.
- **`register_scripted`, `register_condition`, `register_tool_factory` removed** — pass `conditions=` and `tool_factories=` to `compile()` instead. Scripted nodes are handled automatically by `@node`. Each compile gets its own isolated registry; nothing leaks across compiles.
- **`RetryPolicy` scope narrowed to scripted nodes only.** Transient LLM errors (rate limits, 5xx) are now the `llm_factory`'s responsibility (configure on the returned `BaseChatModel`); output-quality retries (parse failure, validation) move to `LlmConfig.max_retries`. Single-responsibility split — no more overlap.
- **Single-type `inputs=` shorthand removed** (was deprecated in 0.4.0). Use dict-form `inputs={"name": SomeType}`.
- **LangGraph dependency pinned and required.** Positioning updated: neograph is "the fastest way to build production-grade agents on LangGraph," not a backend-neutral abstraction. The private `_serde` shim is gone; we use LangGraph's public API throughout.

### Added

#### Multimodal
- **Vision/image inputs via `${image:field}` in inline prompts** (examples 21, 22). New `configure_image(...)` policy + `resolve_image(...)` helper for size/MIME/URL allowlist enforcement.

#### Verify subsystem
- **`verify_compiled(graph) -> list[VerifyIssue]`** — post-compile structural verification, complementary to `lint()`. Catches issues that only exist on the compiled `StateGraph` (orphan checkpointer wiring, state-bus key drift, etc.).

#### Testing scaffold
- **`neograph.testing` auto-generates test suites from pipeline definitions**: per-node fakes, fan-out resilience cases, sub-construct fixtures. Mode-aware (think/agent/scripted), tier-aware (fast/reason/creative).

#### Checkpoint auto-resume
- **Schema-aware rewind on resume.** Schema fingerprints (state model + per-node) attached to compiled graph and persisted with checkpoints. On `run(graph, config=...)` with a changed schema, neograph walks `get_state_history()` backwards, finds the checkpoint before the earliest changed node, and resumes from there — by default. Opt out with `auto_resume=False` to get `CheckpointSchemaError` instead. (Example 19.)

#### Lint expansion
- **Inline `${var}` placeholder validation** against predicted input keys (raw, no flattened, no framework extras).
- **Template-ref `{var}` placeholder validation** when you pass `template_resolver=`.
- **Loop `when` condition checks** — registered-name resolution, `None`-safety smoke test (catches the common `lambda d: d.score < 0.8` bug that crashes when `d is None`).
- **Oracle merge-hook signature checks** against the variant type.
- **`neograph check --setup` reads `get_known_template_vars()`** from your check-setup module.

#### Oracle merge hooks
- **`merge_prompt` now receives upstream context** alongside the variant list. Use `${variants}` for the list, `${upstream.field}` for upstream data. (Example 20.)
- **`MergePreProcess` / `MergePostProcess` / `MergeFallback` hooks** for variant transformation around the merge.

#### Renderer pipeline
- **`render_for_prompt()` returning a `BaseModel` is auto-rendered** through the active renderer. Fields of the returned model flatten into template variables, so prompts can reference `${nested_field}` directly without manual unpacking.
- **`ExcludeFromOutput` marker** — fields visible in input rendering but stripped from the structured-output schema. Lets you carry context into the prompt without confusing the LLM's response schema.

#### YAML/JSON pipeline specs
- **Typed Pydantic schema for specs** (`Spec`, `NodeSpec`, etc., publicly importable via `loader`). JSON-schema document published at `src/neograph/schemas/neograph-pipeline.schema.json`.
- **`Spec.version: Literal[1]` forward-compat gate** — unknown spec versions fail loudly. (Example 16.)

#### Public API surface
- **Typed callback Protocols** exported for IDE help and downstream typing:
  `LlmFactory`, `PromptCompiler`, `CostCallback`, `MergeFallback`, `MergePostProcess`, `MergePreProcess`, `SkipPredicate`, `SkipValueFactory`, `RawNodeFn`, `TypeSpecStatic`.
- **`type_display_name`, `ExcludeFromOutput`** re-exported.

### Examples + Docs

- **10 new runnable examples** (12–22): input rendering, gather→produce sub-construct, context injection, loop refinement, spec-driven pipeline, fan-out resilience, typed projections, checkpoint auto-resume, Oracle merge hooks, multimodal vision, image-security policy.
- **4 sub-projects rewritten** to the canonical `prompt_compiler` pattern (`code-review`, `lead-outreach`, `spec-builder`, `lead-research`). New shared helper at `examples/_shared.py` covers the simple file-per-prompt case.
- **All examples run end-to-end on real APIs** — verified live this release.
- **New website pages**: Prompt Compiler, Checkpoint Resume, Retry Semantics, Multimodal Vision walkthrough. API reference, quick-start, full-pipeline, oracle-ensemble, produce-and-gather walkthroughs updated for the `compile()` kwargs API.

### Fixed

- **Structured-output schema 400s on open `dict[str, str]` fields** — example output models migrated to named Pydantic types so OpenAI strict-mode `response_format` accepts them across providers.
- **Sub-construct auto-fan-in (YAML loader) now wires dict-form upstream producers correctly** via the monopolized `normalize_outputs` / `primary_output_field` helpers.
- **Oracle merge `prompt_compiler` receives `{"variants": [...]}` consistently** — example merge compilers read `data["variants"]` instead of iterating the bare dict.
- **`observable_pipeline.py` declares its `langfuse` + `langchain` dependencies.**
- **OpenRouter model swap**: retired `google/gemini-2.0-flash-001` replaced with `openai/gpt-4o-mini` across examples.

### Architecture (summary)

Internal cleanup landing in this release is extensive — six architecture-decision epics (Q1–Q6) closed, a multi-wave ARCH-SWEEP epic closed, validation cluster split, helper monopolies enforced via structural guards, DIP inversion of `_BranchNode` corrected. None of it changes user-facing behavior beyond what's listed above; details in commit history if curious.

## [0.4.0] - 2026-04-14

### Breaking

- **`DIBinding.payload` removed** -- replaced with typed `default_value` (CONSTANT) and `model_cls` (MODEL kinds) fields.
- **`parse_condition` and `ModifierSet` removed from `__all__`** -- still importable, not in `from neograph import *`.
- **`_validate_type_spec` rejects non-type inputs** -- `Node(inputs="SomeType")` or `Node(outputs=42)` raise `TypeError` at construction time.
- **`Construct.nodes` validates items** -- rejects non-model types (dicts, strings, ints) at construction time via `BeforeValidator`.
- **Single-type `inputs=` emits `DeprecationWarning`** -- use dict-form `inputs={"name": SomeType}` for explicit named resolution.

### Added

- **`render_input` in public API** -- `from neograph import render_input`.
- **`render_for_prompt()` returning BaseModel auto-rendered** -- typed presentation projections get BAML/XML/JSON rendered through the active renderer.
- **`--version` CLI flag** -- `python -m neograph --version`.
- **Missing inline prompt `${vars}` emit structlog warning** -- logs `prompt_var_missing` with available keys instead of silent empty-string.
- **Loader path heuristic hardened** -- newline pre-filter prevents YAML strings from being misidentified as file paths.
- **Checkpoint crash recovery** -- `run(graph, config=...)` with no input resumes from last checkpoint. Detects existing checkpoints automatically when input is provided.
- **Null-to-default coercion** -- LLM returning `null` for fields with defaults (e.g., `str = ""`) auto-coerces to the default. Recursive for nested models.
- **Structured retry with schema** -- retry prompts include `describe_type(output_model)` so the LLM sees the expected structure on self-correction. Default retries bumped from 1 to 2.

### Fixed

- **`exclude=True` fields omitted from schemas and renderers** -- `describe_type`, `XmlRenderer`, `DelimitedRenderer` now skip `Field(exclude=True)` fields. Prevents LLMs from producing pipeline-internal values.
- **12 bare `except Exception` handlers eliminated** -- narrowed to specific exception types. Structural guard prevents new ones.
- **Subconstruct context field types preserved** -- parent field types propagated into subconstruct state models instead of erasing to `Any`. Fixes msgpack checkpoint allowlist for context fields.
- **Checkpoint resume with input** -- `run(graph, input={...}, config=...)` with existing checkpoint now resumes instead of restarting. Input injected into config for DI, `None` passed to `graph.invoke()` for resume.
- **DI preflight check on crash-recovery path** -- missing FromInput params fail at the gate with a clear error, not deep inside a node.

### Architecture

- **`_sidecar.py` extracted** -- breaks circular import between `decorators.py` and `_construct_builder.py`. Structural guard enforces one-way imports.
- **`_build_oracle_kwargs` extracted** -- deduplicates Oracle composition in `node()` decorator. Fixes latent bug: fusion path now validates `ensemble_n >= 2`.
- **`_is_instance_safe` deduplicated** -- `factory.py` imports from `di.py`.
- **16 deferred imports eliminated** -- leaf-module imports promoted to top-level. Budget guard added (ceiling: 40).
- **`NodeItem` type alias** -- replaces 10 bare `Any` signatures in validation.
- **`model_copy` calls batched** -- `_cleanup_inputs_and_register` does one copy per node instead of three.

### Testing

- **1362 tests** (was 999 at 0.3.0). Test suite restructured into 36 files across 5 packages.
- **Hypothesis property-based testing** -- 95 tests across topology strategies, LLM output parsing, registry interactions, and modifier edge cases.
- **68 check fixtures** (52 should-fail + 16 should-pass).
- **Structural guards** -- bare `except Exception`, deferred import budget, no-payload field, sidecar module boundary.

## [0.3.0] - 2026-04-09

### Added

- **Each x Oracle fusion** (`neograph-tpgi`). `map_over=` + `ensemble_n=` on the
  same `@node` produces a flat M x N Send topology. For M items and N Oracle
  generators, all M x N calls run concurrently. Results are grouped by
  `each.key` and `merge_fn` is called per group. No sub-construct workaround needed.

- **`@merge_fn` state params** (`neograph-jg2g`). Non-DI parameters in `@merge_fn`
  are auto-wired from graph state by name, matching `@node`'s upstream wiring
  pattern. Compile-time validation catches unknown fields, self-references, and
  type mismatches.

- **`describe_graph()`** (`neograph-vxrg`). Returns a Mermaid diagram string for a
  compiled graph. `NEOGRAPH_DEV=1` auto-prints a DAG summary after every `compile()`.

- **`neograph check` CLI** (`neograph-0hzr`). `neograph check my_pipeline.py`
  discovers Constructs, runs `compile()` + `lint()`, reports pass/fail. Supports
  `--config` (JSON) and `--setup` (Python module).

- **`lint()` helper** (`neograph-fn5x`). `lint(construct, config=...)` validates
  DI bindings against a sample config. Returns `list[LintIssue]`. Checks
  FromInput/FromConfig scalar and bundled model params, and merge_fn DI.

- **Dev-mode warnings** (`neograph-o846`). `NEOGRAPH_DEV=1` emits warnings for
  ambiguous-but-valid patterns: `Oracle(n=1)`, uneven model distribution,
  `Loop(max_iterations=1)`.

- **Compiler safety net** — rustc-style fixture suite. 48 `should_fail` + 13
  `should_pass` fixtures with parametrized test harness. Every validation rule
  has a corresponding fixture.

- **Compile-time validation** — 7 new checks for "if it compiles, it runs":
  tool factory registration, LLM+prompt configured, output_strategy values,
  Each.key field existence, sub-construct output boundary, Loop/branch
  condition wrapping, context= reference validation.

- **Error-feedback retry** — on LLM parse failure, sends Pydantic validation
  details back to the LLM for self-correction. Configurable `max_retries`.

- **Brace-counting JSON extraction** — replaces regex-based extraction for
  reliable parsing of LLM responses containing multiple JSON objects.

- **3 mini-project examples**: lead-research (Each fan-out), code-review
  (per-file analysis), spec-builder (NL to pipeline spec).

- **Model compatibility test suite** — 28 parametrized tests verifying schema
  round-trip across output strategies and model tiers.

- **34 documentation pages** (was 27). 7 new concept pages: check-cli, lint,
  visualize, dev-mode, each-oracle-fusion, renderers, merge-fn. API reference
  expanded with 8 new entries.

### Fixed

- **15 validation gaps closed** (all from adversarial fixture suite):
  - P0: FromInput shadows upstream node, duplicate modifiers silently dropped,
    sub-construct output boundary bypassed by input port
  - P1: Optional/Union crashes `_types_compatible`, context= references never
    validated, required=True broken for bundled BaseModel DI
  - P2: Double DI marker silently picks first, merge_fn DI invisible to lint(),
    Oracle+Each unguarded in `__or__`
  - P3: Operator condition masked by checkpointer guard, skip_when bad field
    not caught, Loop history=True on Construct silently ignored, type registry
    not idempotent
  - P4: YAML bomb DoS (1MB size limit), Loop skip_when without skip_value
    ambiguity (now warns)

- **Latent bug**: `factory.py` used `ExecutionError` on 2 lines without importing
  it. Would have raised `NameError` at runtime if triggered.

- **Documentation**: `Construct(outputs=...)` → `output=` (singular) in 6 website
  pages. Copy-pasted code would have silently dropped the output boundary.

### Changed

- **999 tests** (was ~400 at 0.2.0). **99% code coverage** — 21 of 22 modules
  at 100%.

- **0 test warnings** (was 4 Pydantic field-shadowing warnings).

- **0 known_gaps** in the fixture suite (was 15).

---

## [0.2.0] — 2026-04-08

### Changed — BREAKING

**`Node.output` → `Node.outputs: dict[str, type]`** (`neograph-1bp`).

`Node` now carries a plural `outputs` field that supports both single-type
(backward compatible) and dict-form for multiple named outputs:

```python
# Single-type (unchanged DX):
extract = Node("extract", outputs=RawText, ...)

# Dict-form (N named outputs):
explore = Node(
    "explore",
    outputs={"result": Claims, "tool_log": list[ToolInteraction]},
    mode="gather", tools=[search], model="fast", prompt="explore",
)
# → state fields: explore_result, explore_tool_log
```

Gather/execute nodes with a `"tool_log"` output key automatically collect
`ToolInteraction` records from the ReAct loop. Demand-driven: no overhead
if no downstream node references `tool_log`.

`Construct.output` stays singular — sub-construct boundary port, same as
`Construct.input`.

**`Node.input` → `Node.inputs: dict[str, type]`** (`neograph-kqd`).

`Node` now carries a plural `inputs` field keyed by upstream name, matching
the same shape across all three API surfaces (declarative, `@node`, and
programmatic/runtime). First-class fan-in validation lands for every surface,
not just `@node`:

```python
# Before (0.1.x):
report = Node("report", input=Claims, outputs=Report)
# Fan-in was impossible to validate statically with a single type.

# After (0.2.x):
report = Node(
    "report",
    inputs={"claims": Claims, "scores": Scores, "verified": VerifyResult},
    outputs=Report,
)
```

`@node`-decorated functions are unchanged at the user-visible layer —
parameter annotations become the `inputs` dict automatically.
`Node.scripted(...)` renamed the kwarg from `input=` to `inputs=`. Sub-construct
boundaries (`Construct(input=Claims, output=...)`) and runtime state seeds
(`run(graph, input={...})`) stay as-is — they're distinct concepts.

**New: `list[X]` consumers of `Each`-modified upstreams** (merge-after-fanout).

A downstream node can consume a fanned-out result as a `list[X]` instead of
`dict[str, X]`. The validator accepts the compatibility via a new rule in
`_types_compatible`, and the factory/@node raw adapter unwrap via
`list(values())` at runtime:

```python
@node(outputs=Clusters)
def make_clusters() -> Clusters: ...

@node(outputs=MatchResult, map_over="make_clusters.groups", map_key="label")
def verify(cluster: ClusterGroup) -> MatchResult: ...

@node(outputs=Summary)
def summarize(verify: list[MatchResult]) -> Summary:
    return Summary(count=len(verify))
```

Ordering is `dict.values()` insertion order — LangGraph barrier arrival
order, not `each.over` collection order. Use this form for order-
independent reductions; if you need deterministic order, keep the
`dict[str, X]` form and sort on the key.

**Why:** Single source of truth for fan-in type compatibility. Declarative
pipelines and LLM-driven runtime specs now get the same assembly-time type
checking that `@node` has always had. Validator collapses from two walkers
to one (`_validate_fan_in_types` in `decorators.py` is gone). The
`mode=raw` log-line quirk for scripted fan-in `@node` nodes is gone —
`factory._make_raw_wrapper` now logs `mode=node.mode` so scripted @node
dispatch reports `mode='scripted'` correctly.

**Migration (for piarch and other direct consumers):**
- `Node(..., input=X, ...)` → `Node(..., inputs=X, ...)` (single-type form
  still accepted for backward compat with isinstance-scan semantics).
- `Node.scripted(..., input=X, ...)` → `Node.scripted(..., inputs=X, ...)`.
- `@node(..., input=X, ...)` → `@node(..., inputs=X, ...)` (the decorator
  kwarg was renamed too).
- `Construct(input=X, output=Y, ...)` unchanged (sub-construct boundary).
- `run(graph, input={...})` unchanged (runtime state seed).

Follow-up: `_attach_scripted_raw_fn` still dispatches scripted @node via
`raw_fn`; full unification with `_make_scripted_wrapper` is tracked in
`neograph-kqd.8` and deferred — it's a pure structural cleanup with no
user-visible change.

---

**Dependency-injection surface switched from `FromInput[T]` to `Annotated[T, FromInput]`.**
The previous form used `FromInput` / `FromConfig` as `typing.Generic` subscriptions,
which had a hidden rule: `FromInput[str]` meant "pull the parameter by name" but
`FromInput[SomePydanticModel]` silently meant "bundle — pull every field of the
model". Same syntax, two different resolution strategies based on whether the inner
type happened to be a `BaseModel`.

The 0.2 surface uses `typing.Annotated` with `FromInput` / `FromConfig` as
markers — the FastAPI dependency-injection pattern (`Annotated[User, Depends(...)]`).
The primary annotation is the real type; the marker tells neograph where the value
comes from:

```python
from typing import Annotated
from neograph import node, FromInput, FromConfig

# Before (0.1.x):
def my_node(topic: FromInput[str]) -> ...: ...
def my_node(ctx: FromInput[RunCtx]) -> ...: ...         # bundled (silently different)

# After (0.2.x):
def my_node(topic: Annotated[str, FromInput]) -> ...: ...
def my_node(ctx: Annotated[RunCtx, FromInput]) -> ...: ...  # bundled (same syntax)
```

**Why:** one resolution path, no hidden BaseModel rule, primary annotation is the
real type (IDE autocomplete sees `ctx: RunCtx` directly), standard typing semantics,
matches the FastAPI pattern Python developers already know. The internal
classifier is simpler and has fewer failure modes.

**Migration:** mechanical — wrap every existing `FromInput[T]` or `FromConfig[T]`
in `Annotated[T, FromInput]` / `Annotated[T, FromConfig]`. The old Generic-
subscription form is removed entirely (no deprecation shim — we are at 0.2).
Attempting `FromInput[str]` now raises `TypeError: type 'FromInput' is not
subscriptable`.

### Added

- **`@merge_fn` decorator** for Oracle merge functions with `FromInput` /
  `FromConfig` dependency injection. The first parameter receives the list of
  variants; subsequent parameters are resolved the same way as `@node`
  parameters. Legacy `(variants, config) -> output` merge functions still work
  unchanged. See `TestOracleMergeFnDI` for end-to-end examples.
- **`FromInput[PydanticModel]` bundles** (via the new `Annotated` surface) —
  constructs the model by pulling each of its declared fields from
  `config['configurable']` under the field's name. Eliminates per-field
  boilerplate for pipeline metadata (`node_id`, `project_root`, tenant
  context, etc.).
- **Frame-walking classifier** — handles locally-defined Pydantic model classes
  (e.g. `class RunCtx` inside a test method) by walking the caller's frame
  stack at decoration time. Needed because `from __future__ import annotations`
  strips closure references, the same technique Pydantic uses for forward-ref
  resolution.

### Fixed

- **`_validate_fan_in_types` unwraps Each-modified upstream outputs as
  `dict[str, output]`** before the type-compatibility check (`neograph-ayq`).
  Previously, a downstream `@node` parameter annotated `dict[str, MatchResult]`
  against an upstream with `.map()` would raise a false-positive rejection
  because the fan-in walker ignored the modifier. The `_construct_validation`
  walker already had this rule (fixed in 0.1.0 via `neograph-8k3`); this brings
  the `@node` walker in line.

## [0.1.0] - 2026-04-05

Initial public release.

### Added

**Three API surfaces, one compiler.**

- **`@node` decorator** — Dagster-style functions-as-nodes API. Parameter names
  are edges, the framework infers the topology from your function signatures.
  - `construct_from_module()` / `construct_from_functions()` assemble pipelines
    from `@node`-decorated functions
  - Mode inference: `prompt=` + `model=` → `produce`; neither → `scripted`
  - Five modes: `scripted`, `produce`, `gather`, `execute`, `raw`
  - Modifier kwargs: `map_over=` / `map_key=` (fan-out), `ensemble_n=` /
    `merge_fn=` / `merge_prompt=` (Oracle), `interrupt_when=` (human-in-the-loop)
  - Non-node parameters: `FromInput[T]`, `FromConfig[T]`, default-value constants
  - Full fan-in parameter type validation across all upstreams
  - Decoration-time validation with source-location error messages
  - Cross-module composition and name-collision detection

- **`ForwardConstruct`** — DSPy/PyTorch-style class-based API with Python control flow.
  - Subclass `ForwardConstruct`, declare `Node` class attributes, override `forward()`
  - Python `if` compiles to LangGraph conditional edges (symbolic-proxy tracing)
  - Python `for` over proxy attributes compiles to Each fan-out
  - Call `forward()` directly in tests with real values (not the traced graph)

- **Programmatic IR** — `Node` + `Construct` + `|` pipe syntax for runtime construction.
  - Runtime pipeline assembly from LLM output, config files, or routing layers
  - Assembly-time `ConstructError` validation on every IR-level construction
  - Construct-level default `llm_config` inherited by all produce nodes

- **Shared infrastructure**
  - `compile(construct)` → LangGraph `StateGraph`
  - `run(graph, input=..., resume=..., config=...)` → execution
  - `configure_llm(llm_factory, prompt_compiler)` → one-time LLM setup
  - `@tool` decorator for tool registration with per-tool budgets
  - `FromConfig[T]` for observability providers, rate limiters, shared resources
  - `Node.run_isolated()` for unit-testing individual nodes
  - `structlog`-based structured logging on every node execution

### Dependencies

- Python >= 3.11
- pydantic >= 2.0
- langgraph >= 0.2
- langchain-core >= 0.3
- structlog >= 23.0

Optional: `langfuse>=3.0` for observability integration.

[0.4.0]: https://github.com/KonstantinMirin/neograph/releases/tag/v0.4.0
[0.3.0]: https://github.com/KonstantinMirin/neograph/releases/tag/v0.3.0
[0.2.0]: https://github.com/KonstantinMirin/neograph/releases/tag/v0.2.0
[0.1.0]: https://github.com/KonstantinMirin/neograph/releases/tag/v0.1.0
