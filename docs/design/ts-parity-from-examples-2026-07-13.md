# TypeScript Feature-Parity — Grounded in 34 Real Examples

**Date**: 2026-07-13
**Author**: synthesis pass over 34 per-example port reports (`docs/design/ts-parity/*.ts` sketches + written analyses)
**Supersedes (corrects)**: the Feature Parity Matrix and Effort Estimate in `docs/design/typescript-port.md`, which were built from **toy snippets**. This document validates that matrix against every runnable example in `examples/` and re-rates each feature on real-pipeline evidence.

> **Headline correction.** `typescript-port.md` frames the hard, load-bearing risk as the **AD-0 compiler transformer** (8 days, "signature IS the DAG"). The real examples say otherwise: **the transformer is irrelevant to ~half the suite** (the programmatic/declarative surface) and **buys almost nothing** for the other half (dead-body LLM nodes with empty signatures). The actual parity risk lives in three places the matrix under-weights or omits entirely: (1) **runtime-reflection seams** Python has and TS structurally lacks (`prompt_compiler` introspection-gating, DI classification, raw-node `model_fields`/`isinstance`); (2) **whole subsystems with no matrix row at all** (MCP battery, checkpoint schema-fingerprint auto-rewind, `${image:...}` multimodal, model-field visibility metadata); and (3) the **type/value duality** that makes `run()` results and every rendered value untyped in TS, contradicting the "typed end-to-end" pitch.

---

## 1. Updated Parity Matrix (aggregated across examples)

Ratings are re-derived from real usage. `# ex.` = examples that actually exercise the feature. **Bold rows are corrections** to `typescript-port.md` (its rating in parentheses).

### 1a. Surfaces & assembly

| Feature | TS rating | # ex. | Consolidated real-pipeline friction |
|---|---|---|---|
| Programmatic `Node(...) \| Modifier` surface | **Direct** | 15+ | The genuinely clean path. `\|`→`.pipe()` is cosmetic. Every "mostly-direct" verdict in the suite lives here (05, 10, 11, 12, 16, 17, 22, vs_langgraph 01–05). |
| `Construct(input=/output=)` sub-construct boundary ports | Direct | 05, 10, 13, 14, vs_lg 05 | Ports 1:1; preserves the "no hand-written state-mapping wrapper" win vs LangGraph. Nested `Construct`-as-node maps to LangGraph.js subgraphs. |
| **`construct_from_module`** (Not-in-v0.1.0) | **Blocked → explicit list** | ~14 | The single most-used assembly call in the suite (01, 01c, 02, 03, 04, 06, 07, 08, 09, 13, 13b, 20, 21, 24). No TS module-symbol reflection exists. Every one degrades to a hand-enumerated `constructFromFunctions([...])`. **Topo-order survives; list membership becomes a hand-maintained drift point** — forgetting a node silently drops it. This is a real capability loss, not a syntax change; it is the *literal teaching point* of examples 01/01c/08. |
| **`compile()` options bag** (Direct, `compile(construct,{checkpointer})`) | **Redesign (under-specified)** | 10+ | The doc's compile signature is **wrong for the entire config-driven surface**. Real `compile()` must carry `llm_factory`, `prompt_compiler`, `scripted={}`, `conditions={}`, `tool_factories={}` (05, 10, 11, 13, 21, 27, all vs_langgraph). The declarative/programmatic surface is *defined by* these registries; the matrix omits all of them. |
| `@node` decorator → `node({...}, fn)` wrapper | Redesign | ~16 | TS decorators can't decorate standalone functions; mechanical but pervasive. Zero-param seeds need an empty `{}` config where Python writes a bare `@node`. |
| **AD-0 transformer ("signature IS the DAG")** | **Direct-but-often-inert** | — | Load-bearing for **zero** programmatic-surface examples and **low-value** for the @node ones, because they are dominated by *dead-body* LLM nodes (empty signature, explicit `outputs=`). Where it matters (01c fan-in, 07 wiring) it works — *if* it survives minify/bundle order and captures the `const` binding name. Net: the doc's marquee bet is the least-exercised feature in the suite. |
| **Node-name capture from `const x = node(...)`** | Redesign | 01c, 02, 06, 09, 10 | Python gets node name free from `fn.__name__`; TS arrow fns are anonymous, so the transformer must walk to the enclosing `const` binding. Fan-in edge resolution + rename-fragility ride on this. |
| **snake_case state-field ↔ camelCase param/const** | Redesign | 06, 10, 13, 21 | `{node}_{key}` state fields (`explore_tool_log`, `check_results`) and node names with hyphens are inherently snake-flavored; idiomatic camelCase silently breaks string-match wiring. Needs a normalization policy the doc never states. |

### 1b. Modifiers

| Feature | TS rating | # ex. | Consolidated friction |
|---|---|---|---|
| Each fan-out (`map_over`/`map_key` → Send + merge-to-dict) | Direct | 04, 10, 13, 14, 16, 17, vs_lg 03 | Native LangGraph.js Send + reducer. String-path `over=` needs no reflection. **Ordering caveat carries over** (arrival order, not collection order). |
| **Each fan-out receiver param (`fan_out_param`)** | Redesign | 04, 13, 14 | No declared config slot names *which* callback param receives `neo_each_item`; leans wholly on transformer "sole non-DI param" inference. Breaks the moment a fan-out node also reads a peer upstream — Python disambiguates by element-type match (a type→value boundary crossing). |
| Loop modifier (`loopWhen`, `.pipe(Loop)`, `self.loop`) | Direct | 15, 16, 27 | All five surface forms map to LangGraph.js conditional back-edges. Mutable closure counters are *cleaner* in TS (no `[0]` boxing). |
| **Loop `d is None` first-iteration sentinel** | Redesign | 15, 16 | TS has **two** empties (`null`/`undefined`) where Python has one; which one `when()` receives must be pinned or the `loop_condition_none_unsafe` bug class reopens. |
| Oracle ensemble (fan-out N, barrier-merge) | Direct | 03, 10, 20, vs_lg 03 | Semantics port cleanly. But see next two rows. |
| **Oracle `merge_pre/post_process/fallback` (Callable fields)** | **Redesign (New)** | 03, 20, vs_lg 03 | Function-valued fields **break the "Modifier = frozen Zod model" story** — callbacks aren't Zod-schemable. **And**: Python validates hook arity + type-hints at assembly (`_validate_merge_hooks`); the transformer only reaches `node()` call sites, not functions passed as config values, so a wrong-arity hook is caught by *neither* build nor runtime-until-fallback-fires. Compile-time-safety regression at the self-healing seam. Matrix lists only `ensemble_n`. |
| **Oracle `merge_fn` by string-name + `scripted={}` registry** | Redesign | 03, 05, 10 | Oracle references merge by *string*; transformer can't follow a string literal to a value. Needs an invented `mergeFn(name, fn)` self-registration wrapper + a documented `compile(scripted={})` seam. |
| Operator interrupt (string condition) | Direct | 09, 10, 27, vs_lg 04 | `interruptWhen:"cond"` maps to LangGraph.js `interrupt()`. |
| **Operator `interrupt_when` *callable* returning a payload** | **Redesign** | 09 | Matrix documents only the string→bool form. Real use is `(state)=>Payload\|null` reading node outputs off a dynamically-shaped `state`. No AD-0 transformer can type `state` (its shape emerges from *assembly*, not any one signature) → hand-maintained `StateView` kept in sync by hand. |

### 1c. LLM / prompt / rendering seams

| Feature | TS rating | # ex. | Consolidated friction |
|---|---|---|---|
| **`prompt_compiler` introspection-gating (`_accepted_params`/`**kw`)** | **Blocked → declarative opt-in** | ~11 | The **most pervasive HIGH-severity gap**. Python inspects the compiler's declared params to decide whether to pass `di_inputs`/`context`/`config` (the neograph-euyh opt-in that stops literal `{domain}` shipping to the model). TS cannot read a closure's param names at runtime (arrows erase them, minifiers rename, `fn.length` is positional-only). Must be redesigned as an explicit capability flag (`{wants:['diInputs','context']}`). Appears in 03, 07, 08, 11, 14, 20, vs_langgraph 01–05. Matrix marks `prompt_compiler` "Direct" and never mentions the seam. |
| **`llm_factory` signature (`tier, node_name=, llm_config=`)** | **Redesign** | 07, 08, 10, 11 | Matrix documents `llmFactory(tier)` only. Real factory takes `node_name`+`llm_config`; Python filters by introspection (`_accepted_params`) to "pass only what you declare". No TS analogue → collapses to a fixed context-object contract. Plus `llm_factory→None` sentinel (03) infects every call site with nullability Python never had. |
| think/agent/act mode inference | Direct | many | Same logic from config-field presence. |
| **Dead-body LLM node (`...`)** | Redesign | ~10 | Python `...` is a clean unimplemented body that still satisfies the return type. TS wrapper *requires* a callback that returns its annotation → throwaway `throw`/`return undefined as unknown as T` stub. No `node({...})`-without-callback overload for pure-LLM nodes. Dead-body warning is "Skip" so nothing guards an accidentally-live body. (02, 07, 08, 13, 13b, 14, 20, 21, 23, 25) |
| **Dict-form outputs `{result, tool_log:[...]}`** | **Redesign (missing row)** | 13, 14, 23, 25 | A fn returns **one** type → AD-0 return-extraction cannot express two named outputs → forced explicit `outputs` config that collides with the extracted return annotation (needs a precedence rule). Not in the matrix at all. |
| Renderers (XML/Delimited/JSON) core | Direct | 12, 18, 22 | Format strings port; inline `${var.field}` raw-object access is *easier* in TS (native property access). |
| **`render_for_prompt()` model-method dispatch (Level 1)** | **Redesign (New)** | 12, 18 | `hasattr(value,'render_for_prompt')` per-instance dispatch has **no Zod home**: `z.infer<T>` is a methodless plain object, and any prototype dies across the msgpack/JSON checkpointer round-trip. Both escape hatches degrade (class-model diverges from the validated value; schema→renderer side-channel breaks `renderInput(value)`-alone). Matrix's "renderers Direct" assumes renderers only ever see schemas/plain values. |
| **`describe_value` / `render_input` / `build_rendered_input` on a bare value** | **Redesign (signature change)** | 12, 18 | In Python the instance *is* its type (`value.__class__.model_fields`). A plain JS object carries no schema → every one of these must **widen its signature to also take the schema**. Every call site diverges from the Python one-arg form. Matrix implies value-only signatures that wouldn't compile. |
| `describe_type` (schema→notation) | Direct-with-effort | 12, 18 | "Already TS-native notation" — but the port must read `Field(description=)` off Zod `._def.description` and re-implement two-pass nested-class hoisting over `ZodObject` internals, gated by field-visibility metadata. Not the freebie the matrix implies. |
| **`RenderedInput` + inline/template-ref key asymmetry** | New (unbudgeted) | 12, 18 | The load-bearing rendering abstraction (`raw`/`rendered`/`flattened`/`available_keys_inline` vs `_template`) is **absent from the matrix**. Ports as plain data but is a wholly unaccounted design surface. |
| **`context=[...]` verbatim state injection** | Redesign (missing row) | 13, 14 | Verbatim (non-rendered) state into the compiler's `context` arg has no matrix row; its plumbing depends on the same compiler-param introspection gate that TS lacks. |
| **`${image:...}` multimodal grammar + `resolve_image`/`configure_image`** | **New (entirely unmapped)** | 21, 22 | Distinct prompt grammar (`_IMAGE_RE` → text/image content blocks) + an entire subsystem: magic-byte MIME sniff, base64/data-URI, file-path read, size cap, `allowed_dirs` sandbox, and a `prompt_compiler` **bypass**. Matrix covers only plain `${var}`. **Security-relevant**: naive TS path-containment (`startsWith`) is a live traversal CVE (Python gets `Path.resolve`+`is_relative_to` free); `configure_image` global singleton is unsafe under bundler module-duplication. |

### 1d. DI, tools, MCP

| Feature | TS rating | # ex. | Consolidated friction |
|---|---|---|---|
| `FromInput`/`FromConfig` branded types | Direct-with-transformer | 24 (+ implied) | Works *if* the transformer lands. Bundling (BaseModel inner) needs transformer BaseModel-detection. |
| **`FromResource(uri=/ref=/max_bytes=)` DI marker** | **Blocked (New)** | 24 | A branded *type* carries no runtime value; `FromResource` carries **configured runtime args** that must reach the runtime → the binding must move **out of the signature** into a `node()` config map. Directly breaks "signature IS the DAG" for exactly the params the example is about. Not in the DI matrix (only `FromInput`/`FromConfig`). |
| Tool(name, budget, idempotent), `gate_tools_when`, `announce_tool_budget` | Direct | 02, 13, 14, 23 | Plain config objects. `gate_tools_when`/`announce_tool_budget` not enumerated in matrix but likely Direct. |
| **Raw tool arg-schema inference** | Redesign | 13b, 23 | AD-0 is scoped to `@node` fns, **not** standalone tool callbacks → raw tools need hand-written Zod arg schemas. "Signature IS the schema" does not extend to tools. |
| **`ToolInteraction.typed_result`** | **Redesign (New, no generic)** | 02, 13, 13b, 23, 24, 25 | The typed-tool-result *is* the feature these examples advertise. `typed_result: Any` → TS `unknown` with **no generic on ToolInteraction** → every read is a cast. LangChain.js `ToolMessage.content` is string-shaped, so carrying a typed object needs the `content_and_artifact` channel, and even then it's structural (no `instanceof`). The MCP typed-binding seam — neograph's *positioned differentiator* per project memory — is exactly where the TS API is thinnest. Matrix has no `ToolInteraction` row. |
| **MCP battery** (`mcp_session`, `mcp_tool_factory[ies]`, `StdioServer`, `token_provider`, `output_model` rehydration, `mcp_resource_fetcher`→(fetcher,replayer), resource_link manifest lift, self-heal replay) | **New (entirely unmapped)** | 13b, 23, 24, 25, 26 | **Zero MCP rows anywhere in `typescript-port.md`.** `@langchain/mcp-adapters` + `@modelcontextprotocol/sdk` give a *path*, but the deferred-connect factory, gateway rename, per-run token identity, `resource_link` manifest lift onto `neo_resource_manifest_*`, `-32002` expiry self-heal, and `McpCallResult` content-block conversion (pinned to `langchain-mcp-adapters`) are all **net-new package design**, not a syntax port. This is the largest unbudgeted surface in the suite. |
| **async-only tool / `run()` vs `arun()` split + `tool_requires_async_driver` lint** | **Skip (dissolves)** | 13b, 23, 24, 25, 26 | JS/LangGraph.js is Promise-native — **no sync/async tool or driver split exists**. "async-only" is inexpressible, `run()`/`arun()` collapse to one async `run()`, and the headline lint check guards nothing → must be dropped from the taxonomy. Example 13b *exists to teach* this guardrail; a TS port compiles but no longer teaches its point. A rare simplification in TS's favor — but the matrix records neither the collapse nor the lost lint kind. |

### 1e. Checkpoint, state, models

| Feature | TS rating | # ex. | Consolidated friction |
|---|---|---|---|
| compile(checkpointer) + interrupt/resume | Direct | 09, 10, 17, 19, 23 | `resume=` maps to LangGraph.js `Command({resume})`; needs a deliberate wrapper. `__interrupt__` is a neograph normalization LangGraph.js surfaces differently → re-normalize. |
| **Checkpoint schema-fingerprint + auto-rewind** (`compute_schema_fingerprint`, `compute_node_fingerprints`, `_type_signature`, `_compute_invalidated_nodes`, `get_state_history` divergence walk, `auto_resume`, `CheckpointSchemaError`) | **New (unmapped)** | 17, 19 | The *entire raison d'être* of examples 17/19 has **no matrix row** — checkpointer appears only in the `compile()` signature + one "Msgpack → Medium" line. Source of truth must move from Pydantic `model_fields` reflection to **structural hashing of Zod schema objects**, and the transformer must emit a **structural** descriptor, not just `typeName` — else it reintroduces the pre-v63o same-qualname false-negative that stops rewind. `get_state_history()`/`checkpoint_id`-injection resume is unscoped. |
| **`run()` result typing (`result['node'].field`)** | **Redesign (New)** | ~all | Untyped `Record<string, unknown>` → cast on every read. Python is *also* untyped here, but a TS-first library is *expected* to derive a `StateOf<Construct>` type keyed by node name. Absent from the design. This is where "typed end-to-end" dies — at the two human-facing reads (interrupt condition + final result). |
| **Raw-node nominal state reflection (`model_fields` + `isinstance`)** | **Blocked → structural** | 06, 26 | `state.__class__.model_fields` + `isinstance(val, Claims)` has **no TS path**: Zod types erase, LangGraph.js state is a plain object with no field registry. Structural `safeParse` probing is false-positive-prone (any `{items:string[]}` matches). Nominal→structural semantic change. |
| **`frozen=True` Pydantic models** | Skip / degrade | ~all 34 | No Zod runtime freeze; `.readonly()` is compile-time only; hashability lost. Low severity almost everywhere (dedup keys on string fields), but the "18 fields map 1:1" claim silently drops immutability + value-equality. Latent bite if any Oracle dedup ever keys on model instances (JS `Set` = reference equality). |
| **`Field(exclude=True)` / `Annotated[T, ExcludeFromOutput]`** | **New (unmapped)** | 18 | Per-field visibility metadata read by *both* renderer and `describe_type` with opposite gates. Zod has no field-metadata slot beyond string `.describe()`. Needs an invented `neoField(zod, {exclude:none\|output\|all})` tri-state wrapper taught to both consumers. **AD-0 cannot help — it extracts node-function params, not model fields.** |
| Spec-loaded runtime types (`lookup_type` → Zod, untyped bodies) | Redesign (type fidelity) | 16 | Cleanest *structural* port (DAG is YAML data) but `lookup_type` returns `z.ZodType<unknown>` — **no compile-time face**, so every scripted body + result is `unknown`/`any`. The exact opposite of "typed end-to-end". No codegen (`neograph generate types`) path proposed. |
| ForwardConstruct `forward()` tracing + `self.each/ensemble/interrupt/loop` builders | **Redesign** | 15, 27 | `if (proxy)` `__bool__` branch has no JS hook → must rewrite to `.gt().then().else()`, losing early-return/fall-through. AD-5 specs *only* branch + bare `.loop()`; the **deferred-builder methods** (`this.each(body,key)(coll)`, `this.ensemble`, `this.interrupt`, loop-with-body-array, nested each-in-loop) are **entirely unspecified**. Example 15 *dodges* the hard `__bool__` path (uses only `self.loop()`); example 27 hits it head-on. Matrix internally contradicts itself (fluent `.then/.else` **and** "re-trace by flipping each branch" — the fluent API captures both arms eagerly and does *not* re-trace). |

---

## 2. Hardest ports, ranked (true redesign/blocked risk)

Ranked by **(blockedness × frequency × size of unmapped design work)** — not by the doc's stated worry.

1. **MCP battery + typed-result binding** (13b, 23, 24, 25, 26 — 5 examples). *Entirely absent* from the design. A net-new `@neograph/mcp` package: sessions, stdio/http servers, token providers, `output_model` rehydration, `resource_link` manifest lift, self-heal replay, per-run fetch cache, `FromResource` DI. This is neograph's *positioned differentiator* (MCP typed-binding seam) yet the single thinnest spot in the TS design. **Highest risk: unscoped scope, not merely hard.**

2. **`prompt_compiler` / `llm_factory` runtime-introspection seams** (~11 examples, HIGH severity). Python's `_accepted_params` signature-gating (`di_inputs`/`context`/`config` opt-in) is **structurally impossible** in TS. Not one example blocks on it *behaviorally*, but the safety contract it encodes (don't ship literal `{domain}` to the model) evaporates unless redesigned as an explicit declarative opt-in. Ubiquitous → ripples through the whole LLM surface.

3. **ForwardConstruct tracing** (27; 15 dodges it). `_Proxy.__bool__` re-tracing has **no JS equivalent** (no boolean-coercion hook). Fluent `.gt().then().else()` is a path, but the deferred-builder surface (`this.each/ensemble/interrupt`, loop-with-body, each-in-loop cascade) is unspecified. The doc's own AD-5 rows contradict each other on how branch discovery works.

4. **Checkpoint schema-fingerprint auto-rewind** (17, 19). Pure Python runtime type-reflection over `model_fields`; must move to structural Zod-schema hashing **and** the transformer must emit a *structural* descriptor or silently reintroduce the pre-v63o false-negative that disables rewind. Unmapped subsystem; small example count but core to the durability pitch.

5. **Model-layer metadata + `render_for_prompt` dispatch** (12, 18). Zod schemas produce methodless plain objects with no field-metadata slot → no home for `Field(exclude)`, `ExcludeFromOutput`, or per-instance `render_for_prompt`; and `describe_value`/`render_input` need an added schema arg at every call site. AD-0 is useless (model fields, not node params). A parallel `neoField` metadata system must be invented and hand-threaded.

6. **Raw-node nominal reflection** (06, 26). `isinstance`/`model_fields` over state → structural `safeParse` (false-positive-prone). The one mode where AD-0 contributes nothing *and* the type-reflective body pattern has no clean TS path.

7. **`${image:...}` multimodal + image security** (21, 22). Whole unmapped subsystem with **security semantics** (path-traversal, singleton identity) that Python's stdlib hides and a naive TS port gets wrong.

8. **`construct_from_module`** (~14 examples). *Explicitly cut.* High friction / low technical risk — a mechanical workaround exists (`constructFromFunctions([...])`) but it's a pervasive DX regression and it invalidates the literal narration of examples 01/01c/08.

**Not the risk the doc thinks it is:** the AD-0 transformer. It is irrelevant to all 15+ programmatic-surface examples, and for the @node examples it's dominated by dead-body/empty-signature nodes where only the return type (given explicitly anyway) matters. It's an 8–10 day bet on a differentiator the suite barely exercises.

---

## 3. Consolidated API gaps the real examples expose

Things `typescript-port.md` must **add or change** (not in the current matrix):

**Missing subsystems (no row at all):**
- **MCP** — the entire `neograph_mcp` surface (sessions, tool/resource factories, `token_provider`, `output_model` rehydration, gateway rename, manifest lift, self-heal, per-run cache). 5 examples.
- **Checkpoint schema-fingerprint / auto-rewind** — `compute_schema_fingerprint`, `compute_node_fingerprints`, `_type_signature` (structural), `_compute_invalidated_nodes`, `get_state_history` divergence walk, `auto_resume`, `CheckpointSchemaError.invalidated_nodes`.
- **`${image:...}` multimodal** — inline image grammar, `resolve_image`/`configure_image`, MIME/base64/size/`allowed_dirs` sandbox, prompt_compiler bypass, target-runtime (fs vs edge) scoping.
- **`RenderedInput`** abstraction + inline/template-ref `available_keys` asymmetry; `LlmRuntime` + node>runtime>None renderer resolution.

**Contract changes (rated "Direct" but wrong):**
- **`compile()` options bag** — must list `llm_factory`, `prompt_compiler`, `scripted`, `conditions`, `tool_factories`. The current `compile(construct,{checkpointer})` cannot express *any* config-driven example.
- **`prompt_compiler` seam** — replace runtime signature-introspection with an explicit declarative opt-in (`{wants:[...]}`); pin the variant/`data` shape (parsed z.infer objects vs class instances vs raw JSON); specify the user-facing `(template, data, ...) → {role,content}[]` contract (never given).
- **`llm_factory` contract** — `node_name`/`llm_config` params + null-sentinel nullability.
- **`ToolInteraction`** — ship a Zod `ToolInteractionSchema` for output-map declaration **and** a generic `ToolInteraction<T>` so `typed_result` is typable (currently `unknown`).
- **Dict-form outputs** — express `{result, tool_log:[...]}` and the return-annotation-vs-`outputs` precedence rule.
- **`render_for_prompt`/`describe_value`/`render_input`** — value+schema signatures; a `Renderable` interface or schema→projector binding; `{value, schema}` envelope for BaseModel-returning projections.
- **`Field(exclude)` / `ExcludeFromOutput`** — a `neoField(exclude:none|output|all)` tri-state field-metadata channel read by renderer **and** describe_type.
- **Typed `run()` result** — a `StateOf<Construct>` derivation (node-name → output schema), or explicitly concede casts and drop the "typed end-to-end" claim for the result boundary.
- **`interrupt_when` callable** form (`(state)=>Payload|null`) + a typed `state` view; typed `conditions` registry + `__interrupt__` sentinel typing.
- **`context=` verbatim injection** row.
- **`FromResource`** and, generally, **any *configured* DI marker** — branded types can't carry `uri=/ref=/max_bytes=`; these must relocate to `node()` config, breaking "signature IS the DAG".

**Semantic pins TS must decide (Python has one answer, TS has two):**
- Loop first-iteration sentinel: `null` vs `undefined` vs `default()`.
- Nominal vs structural boundary-port routing: `input=Claims` resolves by `isinstance` in Python; Zod structural typing makes same-shape producers collide.
- Node-name → state-field identifier normalization (hyphen→underscore, snake↔camel).
- `frozen=True` value-semantics: `Object.freeze` transform vs `readonly` type vs drop.

**Taxonomy removals:**
- `tool_requires_async_driver` lint + the `run()`/`arun()` split — no referent in async-native JS.
- Dead-body AST warning — "Skip", but note the inverse cost: TS *forces* a body you must neutralize.

---

## 4. Is "8–12 weeks to parity" realistic?

**Optimistic — materially so — for parity with what these 34 examples actually exercise.** The ~65-day / 9–14-week estimate is a defensible estimate *for the surface the toy-snippet matrix scoped*. But the real examples reveal that the matrix **omits entire subsystems from the day-count**:

| Unbudgeted surface (not in the 65-day total) | Evidence | Rough add |
|---|---|---|
| MCP battery (`@neograph/mcp`) | 13b, 23, 24, 25, 26 | +15–25 d |
| Checkpoint schema-fingerprint / auto-rewind | 17, 19 | +5–8 d |
| `${image:...}` multimodal + security | 21, 22 | +3–5 d |
| `render_for_prompt`/`describe_value`/`RenderedInput`/`neoField` (the "Renderers = 2 d" line vastly understates) | 12, 18 | +4–6 d |
| `prompt_compiler`/`llm_factory` seam redesign (declarative opt-in) | ~11 ex. | +2–4 d |
| Typed `run()` result (`StateOf<Construct>`) | ~all | +3–5 d |
| Dict-form outputs + `ToolInteraction<T>` + `context=` | 13, 14, 23, 25 | +2–3 d |

That is **~35–55 additional days** on top of ~65 — i.e., **real parity is closer to 16–22 weeks solo**, or the estimate holds *only if scope is explicitly cut* (defer MCP, multimodal, and checkpoint auto-rewind out of v0.1.0-ts and accept they are Python-only at launch).

**Where the risk concentrates — and it is NOT where the doc says.** The doc bets its risk budget on the AD-0 transformer (8 d). The evidence relocates the risk to:
- **Unmapped subsystems** (MCP above all) — you can't estimate what isn't in the matrix.
- **Reflection-dependent seams** — `prompt_compiler`/DI/raw-node introspection are Python capabilities TS *structurally lacks*; these aren't "2–5 day redesigns", they're **contract redesigns** that ripple across many examples.
- **The transformer is over-invested** — irrelevant to half the suite, low-value for the rest. Its 10-day premium buys DX for the least-exercised path.

Net: **8–12 weeks is a scoped-subset estimate mislabeled as parity.** Honest options: (a) 16–22 weeks for true parity with the example suite, or (b) keep ~10–12 weeks and explicitly ship a *reduced* v0.1.0-ts (programmatic + declarative surface, LLM modes, Each/Loop/Oracle/Operator, checkpoint interrupt/resume — **no MCP, no auto-rewind, no multimodal, no model-field metadata**).

---

## 5. Verdict + 3 load-bearing decisions

**What theoretical TS parity actually looks like.** The *structure* of neograph ports remarkably well: the programmatic/declarative surface (Node/Construct/`.pipe`/compile/run/Each/Loop/Oracle/Operator/subgraphs/checkpoint-resume) is genuinely Direct, and LangGraph.js supplies Send, reducers, subgraphs, and interrupts natively — over half the suite is "mostly-direct" with only syntax deltas. What does **not** port is everything that leans on Python's runtime reflection and the type≡value identity of Pydantic: the `prompt_compiler`/DI introspection gates, raw-node `isinstance`/`model_fields`, `render_for_prompt` model-methods, `Field(exclude)`, checkpoint fingerprinting, and typed `run()` results all degrade or must be re-invented — so the "typed end-to-end" pitch thins exactly at the human- and model-facing boundaries. And three whole subsystems (MCP, multimodal, auto-rewind) are simply **not in the current design**, MCP being the one that carries neograph's stated differentiator. Parity is *achievable* — nothing is truly un-portable — but it is a **larger, differently-shaped effort than the transformer-centric estimate implies**, and it demands an explicit contract layer where Python got reflection for free.

**Three decisions to make before writing a line of TS:**

1. **Scope: MCP + advanced subsystems in v0.1.0, or explicitly deferred?** MCP typed-binding is the positioned differentiator yet is 100% unmapped (5 examples). Decide now to either fund `@neograph/mcp` (+15–25 d) or ship a reduced v0.1.0 that concedes the differentiator is Python-only at launch. Same call for checkpoint auto-rewind and multimodal. This single decision is the difference between "10 weeks" and "20 weeks".

2. **Replace runtime reflection with an explicit declarative contract.** `prompt_compiler`/`llm_factory` param-introspection, DI classification, and configured markers (`FromResource`) must become **declared capabilities** (`{wants:['diInputs','context']}`, config-resident resource specs), because signature reflection is structurally impossible in TS. Design this seam *once*, up front — it ripples through ~11 examples and every LLM node.

3. **Commit to a type/value duality strategy** — the schema-value↔type-name bridge and a `StateOf<Construct>` typed-result derivation — *or* consciously accept `unknown`/casts at every result and rendered-value boundary and stop claiming "typed end-to-end". This also settles the Zod-schema-vs-class question for `render_for_prompt`, `frozen`, model-field metadata, and nominal boundary-port routing. Everything downstream (rendering, checkpoint fingerprint, describe_value, run result) inherits from this choice.

> Corollary: **do not front-load the AD-0 transformer.** It is the doc's biggest single line item and the least-exercised feature in the suite. Build the programmatic surface + the contract layer (decision 2) first; add the transformer as DX sugar once the IR is proven — or skip it for v0.1.0 in favor of explicit declarations, which every example already tolerates.
