# Hybrid Data Flow — Codebase Grounding & Implementation Plan

Companion to `docs/design/hybrid-dataflow-2026-08-14.md` (the original proposal). Produced by a 12-agent workflow (6 sonnet codebase-mapping passes + 5 fable feasibility passes + 1 fable synthesis) run 2026-08-14 against `develop` @ `b5034d4`. Cited by the `neograph-ftnxl` epic and its 10 children (`.1`–`.11`, `.3` unused).

## Executive summary

The proposal splits cleanly into one immediately-shippable safety win, two small parallel-safe P2 features, and a tail of genuinely large or demand-unproven work that must not jump the existing 0.8 queue. Highest leverage is the Agent Spec conformance classifier (Option 1): days of work, zero IR change, and it converts the already-ticketed "export succeeded but foreign runtimes can't run it" seam (qtfof.6/.7/.8/.9) into a loud classified verdict that every future marker will be measured against. The narrow feature slice worth landing now is Carried+Reasoned output markers (dispatch/DX layer, DI-marker precedent, strip-splice lockstep guard mandatory) plus the tool-ledger ordinal and selector view (~70% already built) and documenting the free Loop history projection. Everything else is explicitly gated: Selected behind the ledger ordinal and its own referent-typing design round; provides/requires behind fixing vn5f branch soundness (a contract guarantee atop the current validator would be false); Each indexing behind proven piarch demand with pre-Send stamping only; and scope_path behind a Portal-precedent new-IR decision document rather than code. Only provides/requires aliasing and any eventual scope_path implementation are true new-IR-capability work under the single-writer/guard regime — the marker family and ledger deliberately are not.

## North star verdict

Mostly this bundle is expressiveness and observability, not an extension of the unrepresentable class — and the plan must not market it otherwise. The one genuine (small) extension is Carried: a path into the node's own input/context models is statically resolvable at assembly time, on the same footing as _check_fan_in_inputs. Reasoned is prompt sugar (zero safety delta); Selected is runtime-membership-checked at best (only its key type is static — calling it compile-time-checked would overclaim); provides/requires could join the bounded claim but only AFTER vn5f branch soundness, because a contract guarantee atop cross-arm-leaky reachability is false by construction. The highest silent-seam risks, in order: (1) the describe_type strip and the dispatch splice for Carried are two sites that must move in lockstep — a Carried field with a Pydantic default that gets stripped but not spliced is a silently-defaulted value the model 'never lied about'; this needs a structural guard from day one (the di_inputs lint/runtime-divergence precedent). (2) Any runtime-resolving address that None-soft-fails — especially an Each index derived from barrier arrival order, which would launder a documented nondeterminism into a deterministic-looking API. (3) Contract-name indirection re-admitting dangling references unless the namespace stays closed with assembly-time ConstructError. (4) On the export side, 'to_agent_spec didn't raise' remaining a false proxy for 'portable' — the conformance classifier converts that silent seam into a loud, classified one, which is the most on-North-star item in the whole plan.

## Sequencing vs. the existing 0.8 backlog

The conformance classifier (P1) goes FIRST and interleaves directly with the open qtfof family: it cites qtfof.6/.7/.8/.9 as its initial predicates, and each subsequent edge fix (real DataFlowEdge for Each's iterated_item, Loop decision-input edge, EndNode outputs, api_provider default) becomes ordinary follow-on work that flips a predicate off — a measurable portability ratchet. Per the gatekeeper pass, fixing that concrete broken-for-third-parties output outranks every speculative primitive here, so the qtfof edge fixes themselves should still be scheduled ahead of or alongside the P2 epics, not displaced by them. vn5f (P2) is pulled forward from its current standing because it now blocks the provides/requires track in addition to being standalone correctness debt. The Carried/Reasoned and tool-ledger epics are parallel-safe with each other and with qtfof (disjoint files: describe_type/_dispatch/_output_classify vs _agent_tool_calls/tool.py vs _agent_spec*), suitable for the /team pattern. Coordinate the Carried epic with the file-size DEFER themes in neograph-jtawq: it adds a marker-aware pass to describe_type.py (already 552 lines) and a new seam call in _dispatch.py — if jtawq plans to split either file, land the split first or scope the epic to the post-split layout to avoid double churn; the new _output_classify.py module is itself jtawq-friendly (new concern, new file). The eval loop neograph-rmvl is independent of everything here — no ordering constraint; the only soft synergy is that Reasoned rationale fields and the ordinal-stamped tool ledger would enrich eval traces later, which argues for landing those two P2s before rmvl consumes trace shapes, but does not block rmvl starting. Every new marker/contract also adds a future ROUND_TRIP_ONLY predicate to the classifier, which is another reason the classifier lands first: it is the mechanism that keeps later epics honest about export fidelity instead of deepening the metadata-only-edge family silently.

## Open decisions requiring an explicit call

- Agent Spec interop: confirm Option 1 (conformance classifier now, standard-extension deferred past 0.8) over Option 2 — all passes and the locked positioning memory point to Option 1, but it forecloses near-term lossless export of the new primitives, so it should be an explicit call.
- Carried path syntax and reach: confirm phase 1 restricts Carried(path) to the node's OWN input_data + config (dotted attribute paths). Allowing references to other nodes' outputs adds a validator-invisible dataflow edge and escalates the feature from DX sugar to IR work — the passes unanimously recommend the restriction, but it caps the feature's expressiveness.
- Selected cardinality and referent typing: may a Selected field pick exactly one key, or also a set of keys? And what may a key point at in v1 — ledger entries only, list elements, Each results? The passes want a short design round on this before any Selected code.
- Selected failure semantics / retry: when the model emits a key outside the offered set, phase 1 fails loud with no retry — is a bounded re-ask loop (and its budget) wanted later, and who owns that budget (node config vs run config)?
- Ledger retention + serialization: ToolInteraction.typed_result is a live object not guaranteed checkpoint-serializable — decide whether the ledger contract promises durable results across resume (requires a serialization story, possibly folding ProducingCall-style args-only records) or documents typed_result as resume-volatile.
- Cross-node run-wide ledger: defer (recommended — per-node tool_log covers the doc's examples) or schedule the new state channel now?
- Ledger export: should tool-ledger contents lower into Agent Spec exports at all, or be explicitly classified ROUND_TRIP_ONLY metadata from day one? (Recommended: the latter, via a classifier predicate.)
- Migration: no backwards-compat shims at 0.x is the standing rule — confirm that piarch pipelines consuming raw tool_log lists and unprojected output models just get updated in lockstep, no deprecation cycle.
- Each iteration_index demand gate: is there a concrete piarch use case today that sorted dict-key consumption does not cover? If not, the epic stays parked.
- scope_path go/no-go: approve producing the new-IR-capability decision document (Portal-precedent format) at all, or drop nested scoped addressing from the roadmap until a real nested composition exists in piarch.
- provides/requires ambition ceiling: confirm v1 is strict aliasing — duplicate providers are a hard error, no priority/override or multi-candidate semantics — and that the long-term intent is for context= to resolve through the contract table (one lookup mechanism).

## Proposed epic breakdown (as synthesized; see beads for final filed form)

### 1. Agent Spec export conformance classifier (Option 1)
- **Track**: agent-spec-interop | **Priority**: P1 | **New IR capability**: False

Convert 'export didn't raise' into an honest portability verdict. First slice: new module beside _agent_spec.py exposing export_conformance(construct) -> ConformanceReport with a three-tier verdict (PORTABLE / NEOGRAPH_ROUND_TRIP_ONLY / NOT_EXPORTABLE). Initial predicates are purely structural walks over existing IR: Each/Loop/Operator present -> ROUND_TRIP_ONLY (qtfof.7/.6), top-level Construct EndNode-outputs gap -> ROUND_TRIP_ONLY (qtfof.9), api_provider=None -> flagged (qtfof.8), plus the existing raise-list (_agent_spec.py:26-35) -> NOT_EXPORTABLE. Wire into to_agent_spec(strict=...): default warns/attaches report on ROUND_TRIP_ONLY, strict=True raises. Fixtures per tier + a guard test asserting every predicate cites a bead. No IR changes, no factory/validation edits — days, not weeks. Each later qtfof edge fix flips a predicate off, giving a measurable portability ratchet. Option 2 (extending the Agent Spec standard) is explicitly rejected for 0.8 per the locked not-a-standard-setter positioning; the only Option-2 sliver kept is a docs page for the existing neograph/* metadata-marker convention (separate epic).

### 2. Branch-sound per-arm reachability in the fan-in validator (neograph-vn5f)
- **Track**: provides-requires | **Priority**: P2 | **New IR capability**: False

Fix the known soundness hole where iter_with_arms flattens branch arms into one producer set, so a consumer can validate against a producer that only exists on a divergent arm (cross-arm leakage). This is the load-bearing blocker: both the provides_requires and cross_cutting passes conclude any 'guaranteed present on every path' contract claim is FALSE until this lands — shipping contracts atop it would advertise a guarantee the validator can't honor, an existential defect per the North star. First slice: per-path producer accumulation in _validate_node_chain for Branch arms (a producer registered inside an arm is visible only downstream of the join if ALL arms produce it, else it's arm-local), with should_fail fixtures for cross-arm consumption and a should_pass fixture for all-arms-produce. No new IR fields — this is validator-internal bookkeeping in _construct_validation.py.

### 3. Carried + Reasoned output-field markers (schema projection + assembly, think mode)
- **Track**: schema-projection | **Priority**: P2 | **New IR capability**: False
- **IR note**: All state consumers (state.py, fan-in validator, fingerprints, effective_producer_type) see the declared unprojected type; markers are recovered from the output model at dispatch time, exactly as DI markers are from params. Escalates to IR only if Carried is ever allowed to reference other nodes' outputs — which phase 1 forbids.

Annotated markers on output-model fields, mirroring the FromInput/FromConfig input-side idiom (di.py/_di_classify.py are the pattern donors). First slice, think mode only: (a) new _output_classify.py reading get_type_hints(model, include_extras=True) over output-model fields, cached without touching node.py; (b) projection — a synthesized Pydantic subclass with Carried fields dropped, fed to BOTH describe_type's rendered schema and the with_structured_output model (string surgery alone is insufficient); (c) assembly — one _assemble_output(node, projected_result, input_data, config) call at the ThinkDispatch post-parse seam (sync+async twins), splicing Carried values from the node's OWN input_data + config paths only; (d) fail-loud: unresolvable path is ExecutionError at runtime and a lint rule + should_fail fixture at assembly time — never None-fill, and a Pydantic default on a Carried field must NOT mask a missed splice; (e) a structural guard pinning describe_type-strip <-> dispatch-splice lockstep (the di_inputs lint/runtime-divergence failure mode), including auditing describe_value so few-shot examples never show unprojected instances. Reasoned rides along as the zero-cost default (model-authored, stays in schema, optionally rendered with its rationale description). Explicit scope fence keeping this DX-layer sugar: Carried may NOT reference other nodes' outputs (that would add a validator-invisible dataflow edge and escalate to IR work); agent/act (_shape_tool_output) wiring and Selected are phase 2. Three-surface parity: declarative/programmatic surfaces must be explicitly supported-or-exempt (markers live on the model, so they should work surface-neutrally — test it).

### 4. Tool ledger: ordinal stamping + ToolLedger selector view
- **Track**: tool-ledger | **Priority**: P2 | **New IR capability**: False

~70% exists: ToolInteraction already captures tool_name/args/result/typed_result/duration_ms in call order on a checkpointed list channel. First slice (no IR, no normalizer, no guard): (1) stamp a per-tool ordinal int on each record at _agent_tool_calls.py build time from the already-incrementing ToolBudgetTracker count — 'the 2nd call to search' becomes addressable data instead of consumer filter-then-index; (2) a read-time ToolLedger view class over the existing flat list with first/last/all(tool_name)/by-tool-grouped selectors — zero state-bus change. Explicitly deferred: cross-node run-wide ledger (new state channel; nothing in the doc's examples needs it if the consumer declares the producing node's tool_log), and the typed_result serialization/replayability story (typed_result is a live object not guaranteed checkpoint-safe — a decision, not a structure; see open decisions). This epic is a hard prerequisite for Selected-from-ledger: without ordinals, ledger addressing is positional guesswork.

### 5. Document the free Loop scope projections (all_in_scope / from_enclosing)
- **Track**: scoped-addressing | **Priority**: P3 | **New IR capability**: False

The full Loop iteration history already lives on the state bus (_append_loop_result; list position = iteration index since Loop is sequential) and _unwrap_loop_value already hands list[T] consumers the whole history — all_in_scope is the existing behavior unadvertised, and from_enclosing(n) is list[-n]. First slice: name and document the projection (website loop docs + AGENTS.md), add pinning tests for the list-position==iteration-index invariant, and optionally a tiny read-side helper. No new storage, no schema-fingerprint churn, no checkpoint migration — projection wins over representation for the non-nested case. Deliberately excludes Each indexing and nested scope_path (separate epics).

### 6. Selected output-field marker (key-typed selection over offered sets)
- **Track**: schema-projection | **Priority**: P3 | **New IR capability**: False
- **Depends on**: Carried + Reasoned output-field markers (schema projection + assembly, think mode), Tool ledger: ordinal stamping + ToolLedger selector view

Phase 2 of the marker family, gated on both Carried (shares the projection/assembly machinery) and the tool-ledger ordinal (a Selected referencing ledger entries without ordinals is positional guesswork). First slice: Selected[T] projects to its key type — Literal[...] of actual keys whenever the offered collection is statically known, plain str otherwise; materialization at the assembly seam fails LOUD on a key outside the offered set (any 'accept the string anyway' fallback is the silent seam; bounded-retry is deferred). Honesty constraint from the gatekeeper pass: do NOT market Selected as compile-time-checkable — the offered set is runtime data; only the field type is static. The referent-typing question (what may a key point at: ledger entries, list elements, Each results?) needs its own short design round before implementation, and typed_result serializability must be resolved for Selected-from-ledger to survive checkpoint resume.

### 7. provides/requires phase 1: contract-name aliasing (validation-time only)
- **Track**: provides-requires | **Priority**: P3 | **New IR capability**: True
- **IR note**: Requires a Node.provides field (or equivalent) plus a Construct-level contract table, written solely by _ir_normalize.py under the single-writer discipline, pinned by extending the G3 IR_FIELDS guard, and documented in AGENTS.md alongside handoff_param/handoff_channel. This is the Portal precedent applied deliberately, not defensively.
- **Depends on**: Branch-sound per-arm reachability in the fan-in validator (neograph-vn5f)

NOT a rename of fan-in — today's producer namespace is identity-keyed end-to-end (ProducerMap keyed by field_name_for(producer.name)), and full contracts need multi-candidate resolution + state-bus indirection, which is Portal-class IR/compiler work. Phase 1 deliberately ships the smallest honest slice: provides='name' as pure aliasing. A normalizer-built {contract_name: field_name} table on the Construct (single writer in _ir_normalize.py, G3-style guard); the validator resolves dict-form inputs and context= keys through the table before ProducerMap lookup; _extract_input maps back through the same table at runtime (one shared resolver, both sites). Duplicate provides -> ConstructError; unresolved contract -> ConstructError (the namespace stays closed — indirection must not re-admit dangling references, the exact failure class neograph deletes). Zero state-bus change: the wire key stays producer-derived, so no schema-fingerprint or checkpoint churn. should_fail fixtures: unresolved contract, duplicate providers, cross-arm single-provider. HARD GATE: does not ship before branch-sound reachability (vn5f) — 'guaranteed present on every path' is falsified by the current validator. Deferred: multi-candidate satisfaction, substitutability, priority/override semantics, Agent Spec lowering (aliasing must eventually lower to real DataFlowEdges or it widens the qtfof metadata-only family — the conformance classifier gets a predicate for it). Long-term direction: context= resolves via this table so there is one lookup mechanism, not two.

### 8. Each positional iteration_index (pre-Send stamping) — demand-gated
- **Track**: scoped-addressing | **Priority**: P3 | **New IR capability**: False
- **IR note**: New neo_ state key under StateKeys discipline, but no new Node IR field and no normalizer writer — the Layer-A fragment guard covers it.

Each keys results by business field and the barrier collects in ARRIVAL order, not each.over order — an index derived at the barrier would look deterministic while being nondeterministic, laundering a documented non-guarantee into a safe-looking API (an anti-safety move per the North star). The only acceptable design: stamp the positional index from the each.over collection position BEFORE the Send() fan-out, carried in the Send payload under a StateKeys-built EACH_ITEM-adjacent key. First slice: pre-Send stamping + exposure on the fanned-out item + a test proving the index matches each.over order even under adversarial barrier arrival order. Gate: do not start until piarch demonstrates a concrete need — the passes agree the sorted-dict-key workaround covers order-dependent consumption today. This is a StateKeys/dispatch-payload addition, not a Node IR field, so it stays under the existing StateKeys centralization guard rather than the G3 single-writer regime.

### 9. Document the neograph/* Agent Spec metadata-marker convention
- **Track**: agent-spec-interop | **Priority**: P4 | **New IR capability**: False
- **Depends on**: Agent Spec export conformance classifier (Option 1)

The cheap sliver of Option 2 worth keeping: the metadata markers (neograph/each_spec, neograph/loop_spec, etc.) already ARE a de-facto vendor extension of Agent Spec. First slice: one public docs page on neograph.pro specifying each marker's shape, semantics, and round-trip contract, cross-referenced from the conformance classifier's ROUND_TRIP_ONLY verdict so a foreign-runtime author knows exactly what a metadata-aware loader must implement. Costs a docs page, not standards negotiation. Everything else in Option 2 (getting Oracle/WayFlow to grow fan-out edges, iteration channels, scope paths) is explicitly deferred past 0.8 per the not-a-standard-setter positioning.

### 10. scope_path nested addressing — deferred design gate, not implementation
- **Track**: scoped-addressing | **Priority**: P4 | **New IR capability**: True
- **IR note**: Flagged true because any implementation would require new single-writer IR field(s) and state-bus plumbing per the Portal precedent — but the epic as scoped ships only the decision document, deliberately deferring that cost until demand is proven.

Sub-constructs are per-invocation isolated (neo_subgraph_input, fresh state per outer iteration) and inner structure is discarded at the output boundary — there is no representation to project scope_path FROM, so this is genuinely new IR at Portal scale: threading through subgraph input, stamping barrier writes, a single-writer normalizer field, structural guards. Additionally, nested Loop x Each on a single node is unrepresentable today (ModifierCombo has no dual-primary shape), so the addressing target is a composition that barely exists. First slice is NOT code: a one-page new-IR-capability decision doc (Portal-precedent format: proposed field(s), single writer, guard plan, silent-seam analysis — a runtime-resolving address like verify[3].score that None-resolves re-admits the exact failure class Portal closed) plus a validated piarch use case. Only if the user green-lights that doc does implementation get filed. Backlog until demand exists.

---

## Appendix A: Codebase mapping findings (6 sonnet passes, ground truth as of 2026-08-14)

### ir_outputs
Now composing the report.

## Findings

**1. `Node.outputs` dict-form vs single-type, and where `describe_type` enters**

- `Node.outputs: TypeSpec` (`node.py:95`) is `type | dict[str,type] | None`. Discrimination is monopolized in `_normalize.py:61-97` (`normalize_outputs`) — dict form → `primary`/`primary_key` (first item) + `secondary` (rest, e.g. `tool_log`); single-type → `primary=type, primary_key=None`. Per AGENTS.md this is the sole `isinstance(outputs, dict)` site (guard-enforced).
- The LLM-facing schema: `ThinkDispatch.execute`/`aexecute` (`_dispatch.py:264/298`) call `_resolve_primary_output(node)` (`_dispatch.py:364-382`), which returns `no.primary` (or `node.oracle_gen_type` when Oracle overrides) — always a concrete `BaseModel` subclass, never a dict. That single class is passed as `output_model=` into `_llm.invoke_structured`/`ainvoke_structured` (`_llm.py:239,371`). Inside, `output_schema = describe_type(output_model)` (`_llm.py:315`) renders it to the TS-like notation embedded in the prompt (used by the "schema-in-prompt + parse" fallback path per the module comment at `_llm.py:38,45,88`). So **`describe_type` only ever sees the primary output model** — dict-form secondary keys (like `tool_log`) are framework-collected, never shown to the LLM.

**2. render→call→parse pipeline location; hook point for a post-parse pre-write "assembly" stage**

- Render: `_render_input` (`_dispatch.py:341-361`) builds a `RenderedInput` and picks raw vs `for_template_ref` based on inline-vs-template prompt.
- Call+parse: both live inside `_llm.invoke_structured`/`ainvoke_structured` (`_llm.py:239`/`371`) — LangChain's `with_structured_output` (or the schema-in-prompt+regex/json parse fallback) returns an already-validated `BaseModel` instance of `output_model`. `ThinkDispatch.execute` receives this as `result` and does nothing else but wrap it: `NodeOutput(multi={primary_key: result})` or `NodeOutput(single=result)` (`_dispatch.py:282-284`).
- State write: `_execute_node`/`_aexecute_node` (`_execute.py:130-131`, `185-186`) call `_build_state_update(node, field_name, output.value, bus)` (`_state_write.py:52`), which is the actual write to the LangGraph state dict.
- **Hook point**: `ThinkDispatch.execute`/`aexecute` between the `_llm.invoke_structured(...)` return and the `return NodeOutput(...)` wrap (`_dispatch.py:280-284` sync, `301-315` async) is exactly a post-parse/pre-state-write seam — `result` is already the fully-parsed, validated model instance, and nothing downstream (`_build_state_update`) inspects how it got there. The agent/act ReAct cycle has the equivalent seam in `_shape_tool_output` (`_dispatch.py:318-338`), the shared postamble the cycle's parse node calls — same shape, same insertion point. Both dispatch functions are small, single-purpose, and already contain a "postamble" comment boundary, so inserting an `_apply_output_assembly(node, result)` call right before the `NodeOutput(...)` construction in both would not restructure the pipeline — it's a same-function one-line insertion, duplicated at the two existing dispatch sites (think + tool-output-shape), which is the existing duplication pattern the codebase already tolerates (sync/aexecute twins).

**3. Existing "not model-authored" precedent**

- `Annotated[T, FromInput]` / `Annotated[T, FromConfig]` (`di.py` `DIKind` enum lines 60-94, markers imported in `decorators.py:70-71`) are exactly this idea but for **input params**, not output fields: they mark a node parameter as sourced from `config['configurable']` rather than from an upstream node's model output. Classified once by `_classify_di_params` at decoration time into `Node._param_res` (PrivateAttr, `node.py:196`), resolved uniformly at runtime via `DIBinding.resolve`/`aresolve` (`di.py:239,319`). This is real, load-bearing precedent for the `Annotated`-marker idiom in this codebase — but it is a **classifier over function *parameters*** (walks `inspect.signature`/`get_type_hints` on a callable), not over **fields of a Pydantic output model**. There is currently zero code that inspects `Annotated` metadata on `node.outputs`' model's fields — `describe_type.py`'s two-pass renderer (`_count_classes` then `_render_model_body`/`_render_type`, `describe_type.py:83-130`+) walks `model_fields` but only reads `FieldInfo`/type/description/constraints/default, never `typing.get_type_hints(..., include_extras=True)` on the model to recover per-field `Annotated` markers.

**4. Minimal-diff path for a per-output-field marker (`Annotated[T, Reasoned]`/`Carried(path)`/`Selected(...)`)**

- This is sugar over dict-form outputs' **prompt-schema rendering**, not a new IR capability, IF the marker only changes what the LLM is asked to produce (i.e., strip marked fields from the `describe_type` schema shown to the model and instead splice/compute them post-parse). It does not need new `Node` IR fields, a `Command(goto)` runtime, or `neo_`-prefixed state keys — none of the three criteria in AGENTS.md's "genuinely new IR capability" bar (new field on `Node` with single-writer; new runtime control-flow construct; new mesh state keys) apply. Concretely:
  - `describe_type` needs a marker-aware skip: when rendering a model's fields (`_render_model_body`, in the `_count_classes`/`_render_type` pass), fields annotated with a "not model-authored" marker are omitted from the schema shown to the LLM (new optional parameter or internal check against `typing.get_type_hints(model, include_extras=True)` per field, mirroring the `include_extras=True` pattern already used in `_di_classify.py` for params).
  - `ThinkDispatch`/`_shape_tool_output`'s existing post-parse-pre-write seam (finding #2) is where the marker's runtime resolution (`FromInput`/`FromConfig`-style read from `config`/upstream, or a `Carried(path)` copy-forward from `raw_input`) gets spliced into `result` before it's wrapped into `NodeOutput`. This mirrors the DI resolver pattern (`DIBinding.resolve`) rather than requiring anything new in `factory.py`/`_state_write.py`.
  - `decorators.py` involvement is optional: if the marker classifier is scoped to walk the output **model's own fields** (like `_classify_di_params` walks a function's params) rather than the node function's params, it can live as a new small module (e.g. `_output_classify.py`) called from the same two dispatch sites, without touching `node.py`/`_construct_validation.py`/`factory.py`/`modifiers.py` (the off-limits list) — keeping this squarely in the DX/dispatch layer per AGENTS.md's layering rules, the same footing as `di_inputs`' addition (config side-channel, no new IR field, `_dispatch.py`-scoped).
  - Caveat: if `Carried`/`Selected` markers need to read from *other nodes'* state (not just config or the node's own `raw_input`), that starts to look like a new dataflow edge and would need `Node.inputs` awareness — still expressible as "read from `NodeInput.fan_in`/`.single`", which `ThinkDispatch.execute` already receives as `input_data`, so still no new IR field is required.

**Files referenced**: `src/neograph/node.py:94-96,195-198`; `src/neograph/_normalize.py:61-98`; `src/neograph/_dispatch.py:245-338,364-382`; `src/neograph/_llm.py:239-320,371`; `src/neograph/_execute.py:94-190`; `src/neograph/_state_write.py:52-90`; `src/neograph/di.py:60-94,214-320`; `src/neograph/describe_type.py:83-130`.

---

### tool_ledger
Enough detail gathered. Composing the findings report now.

**tool.py:148-164** — `ToolInteraction` fields: `tool_name: str`, `args: dict[str, Any]`, `result: str` (rendered), `typed_result: Any` (raw object), `duration_ms: int`. Frozen BaseModel. No `ordinal`/index field, no `node_name`/turn field.

**Collection shape — flat list, call order, NOT per-tool-name addressable.** `state.py:372`: `StateKeys.agent_tool_log(field)` is `Annotated[list, _concat_reducer]` — one flat `list[ToolInteraction]` per node, accumulated across ReAct turns via LangGraph's concat reducer (`_state_reducers.py:39`). `_agent_tool_calls.py:118-138,159-170` builds one `ToolInteraction` per tool call, appended in `tool_call` batch order (sync: `_agent_cycle.py:566-613` one-at-a-time; async: `:615-680` gather but reassembled in original `tool_call` order via `plan`). There is no dict/grouping-by-`tool_name` structure anywhere — a consumer wanting "all calls to `search`" must filter the flat list themselves (`[i for i in tool_log if i.tool_name == "search"]`), and nothing in the framework does that filtering or exposes a grouped view.

**No ordinal/index (answer to Q2): no.** Nothing stamps a per-tool call count. The 2nd call to `search` is only reachable positionally via list-filter-then-index by the *consumer* — `[i for i in tool_log if i.tool_name=="search"][1]`. The only counter that exists is `ToolBudgetTracker._counts` (`_agent_cycle.py:245-247`, `_tools_result` `_agent_cycle.py:558`), a `dict[tool_name, int]` used purely for budget enforcement (`can_call`/`record_call`), never stamped onto the `ToolInteraction` record itself and not retained per-call (it's a running total, overwritten each turn, not history).

**Opt-in mechanism (Q3): declared via dict-form outputs, framework-collected always, exposed only on demand.** Per AGENTS.md "Gather tool collection" section and `_dispatch.py:325-336` (`_shape_tool_output`): the ReAct cycle *always* collects `ToolInteraction`s into the `neo_agent_tool_log_{field}` state channel regardless of whether the node declares it (`_tools_result` at `_agent_cycle.py:561` unconditionally writes `tlog_key: interactions`) — collection is not actually demand-gated at the collection site. What IS demand-gated is *exposure to the node's declared output*: `_shape_tool_output` only copies `tool_interactions` into the node's `result_dict["tool_log"]` when `"tool_log" in no.all_keys`, i.e. only when the node declares `outputs={"result": X, "tool_log": list[ToolInteraction]}`. So AGENTS.md's "no collection overhead if no consumer references tool_log" is slightly imprecise: the per-turn accumulation into the internal channel always happens during the ReAct loop; only the *downstream-visible* copy is demand-driven.

**Gap analysis (Q4).** Already present: args + rendered result + typed result + duration, captured per call, in call order, checkpointed. Missing relative to "store calls not results, with args+result+ordinal, per-tool addressable":
- No `ordinal`/sequence field on `ToolInteraction` itself (only inferable from list position, and only within one node's flat list).
- No per-tool-name grouping/index — no `dict[tool_name, list[ToolInteraction]]` view, no "call N of tool X" selector, no "first/last call to X" helper anywhere in `renderers.py`, `_dispatch.py`, or `_agent_cycle.py`.
- No cross-node ledger — `tool_log` is scoped per agent/act node's own field (`StateKeys.agent_tool_log(field)`); there is no run-wide ledger aggregating tool calls across multiple nodes.
- `result` is the rendered string, and `typed_result` is the live raw object (not guaranteed serializable/replayable) — there's no separate persisted "args+result recipe" independent of the rendering pipeline (contrast with `ProducingCall` in `tool.py:167-183`, which IS a minimal `(tool_name, args)` replay record but carries no result and no ordinal either).

Files: `/Users/konst/projects/neograph/src/neograph/tool.py:148-164`, `/Users/konst/projects/neograph/src/neograph/_agent_tool_calls.py`, `/Users/konst/projects/neograph/src/neograph/_agent_cycle.py`, `/Users/konst/projects/neograph/src/neograph/state.py:356-372`, `/Users/konst/projects/neograph/src/neograph/_dispatch.py:325-368`.

---

### scoping
## Findings: lexical scoping of state across loops/fan-out (neograph)

**1. Each's fan-out keying and ordering**

`Each` results are keyed by a *business field on the item*, not identity/index: `key_val = getattr(each_item, each_mod.key, str(each_item))` (`src/neograph/_state_write.py:122,129`; same pattern in `_wiring_oracle_each.py:89`). The barrier field is `Annotated[dict[str, output_type], _merge_dicts]` (`state.py:466-469`), and `_merge_dicts` merges additively, keeping the *first* value on a duplicate key (`_state_reducers.py:56-66`) — no index fallback beyond `str(item)` if the field is missing.

AGENTS.md's note is **confirmed accurate**: `dict.values()` preserves Python insertion order (arrival order into the dict via LangGraph's parallel `Send()` barrier), which is not the same as `each.over`'s collection order. `_unwrap_each_dict` (`di.py:177-190`) does `list(val.values())` for `list[X]` consumers — order-dependent reductions must consume `dict[str, X]` and sort explicitly. There is no positional/index key anywhere in the Each path.

**2. Loop: history exists in state but is normally hidden**

Loop's state field is *already* a full history list with an append-only reducer: `Annotated[list[output_type], _append_loop_result]` (`state.py:482-487`; `_append_loop_result` at `_state_reducers.py:26-30` does `[*existing, new]`). So the full per-iteration history genuinely lives on the state bus, keyed by nothing but list position (arrival order = iteration order, since Loop is sequential not parallel).

Consumption is unwrap-on-read, symmetric to Each: `_unwrap_loop_value` (`di.py:153-176`) returns `val[-1]` (latest) unless the consumer's declared type origin is `list`, in which case the full list passes through unchanged. So "all iterations" **is** obtainable today — declare the downstream param as `list[T]` instead of `T` — but it's an undocumented mirror of the Each list-consumer trick, not a first-class "iteration history" API, and there's no per-iteration metadata (timestamp, index, condition-eval result) attached, just the bare value list.

**3. Nesting: no scope-path, discarded through construct boundaries**

`ModifierCombo` (`modifiers.py:89-100`) has no `LOOP+EACH` (or any dual-primary-shape) combination — a single Node/Construct carries exactly one of Each/Oracle/Loop/Portal (+ optional Operator). Nesting can only happen *structurally*: a `Loop`-wrapped sub-`Construct` can contain nodes that carry their own `Each`, or vice versa.

Sub-constructs are fully isolated: each compiles to its own `StateGraph` with its own field namespace, receiving input only via `neo_subgraph_input` (`_state_keys.py:48`, `_subconstruct.py:31,163-170`, `_construct_builder.py:129,186,204`). When an outer `Each`/`Loop` re-invokes that sub-construct (once per fan-out item or once per iteration), it's a brand-new isolated state instance each time — any inner Each dict-keying or Loop history *inside* the sub-construct is invisible outside it; only the sub-construct's declared `output` boundary value survives into the outer barrier/list. There is no `(scope_path, iteration_index)` tuple, no composite key, nothing resembling a path — outer accumulation is flat (one dict-by-business-key layer for Each, one list-by-arrival layer for Loop) and inner structure is thrown away at the construct boundary unless explicitly re-exposed through `output`.

**4. Gap sizing for the proposed `(name, scope_path, iteration_index)` addressing**

This is a real gap, not a documentation gap — the current model has no representation to project *from*:

- **Each**: keys are business-semantic (`each.key` field value), not positional; no `iteration_index` exists at all. Building `index` selectors would need either (a) synthesizing an index at dispatch time and threading it through `Send()` payload (new IR field, similar to `EACH_ITEM`), or (b) accepting index = arrival order, which is explicitly the non-guarantee AGENTS.md warns about.
- **Loop**: `iteration_index` is recoverable for free — list position is already iteration order (sequential, not parallel) — a read-time projection (`from_enclosing(n)` → `list[-n]`, `all_in_scope` → the list itself, `latest` → `list[-1]`) could be built entirely on top of `_unwrap_loop_value`/`_append_loop_result` with **no state-bus schema change**.
- **`scope_path` (nesting)**: this is the actual missing primitive. Because sub-construct invocations are isolated per fan-out item/iteration and boundary-collapsed, there is currently no channel carrying "which Each item / which Loop iteration produced this inner value" past the sub-construct boundary. Supporting `scope_path` for nested Each-in-Loop or Loop-in-Each would require a genuinely new IR capability (an accumulating path/context value threaded through `neo_subgraph_input` and stamped onto barrier writes) — on the order of the Portal precedent (new `Node` field, single normalizer writer, guard-pinned), not a read-time projection.

**Net**: `latest` and `all_in_scope` for a single non-nested Loop are near-free read-time projections over existing representations. `all_in_scope`/`from_enclosing(n)` for Each need an index to be added to the fan-out payload (small IR addition). Anything crossing a sub-construct boundary (the nested case the design doc is really motivated by) needs new state-bus plumbing — a scope-path-carrying channel — comparable in scope to the Portal mesh addition, not a thin read-time layer.

---

### fan_in_contracts
## Findings: provides/requires vs. today's name-based dataflow

**1. Addressing mechanism — node-identity-based, not contract-based.**
`Node.inputs`/`Node.outputs` are `TypeSpec` fields (`node.py:94-95`). The dict-form fan-in key must equal the **producer's own name**, normalized: `field_name_for(item.name)` (`naming.py:7-9`), optionally suffixed `_​{output_key}` for dict-form outputs via `output_field_name` (`naming.py:12-20`). `_check_fan_in_inputs` (`_validation_inputs.py:107-124`) looks a key up directly in the `ProducerMap` (keyed by `field_name`, `_validation_types.py:59`) — one producer per field name, no indirection table. The one real decoupling that exists: **dict-form `Node.outputs`** lets a node emit several named artifacts whose *key* need not match the node's name (`_construct_validation.py:219-232`, key = `{field_name_for(node.name)}_{output_key}`), and `context=` (`_construct_validation.py:190-201`) lets any node read *any* ambient/upstream producer field by name regardless of graph position/depth — but both still resolve by the literal producer-derived field string, never by an independent capability name. There is no "N nodes can satisfy contract C" resolution anywhere; `ProducerMap` is `OrderedDict[field_name, Producer]` — last writer to a field name wins, not "any compatible satisfier."

**2. Reachability checking — order-accumulation only, not branch-sound.**
`_validate_node_chain` (`_construct_validation.py:111-374`) walks nodes in declaration order, accumulating an `OrderedDict` of producers seen-so-far (`producers[field_name] = Producer(...)`, e.g. `:236-240`). A consumer input is checked against whatever's accumulated at that point — this is linear "is a producer declared earlier" logic, not real per-path reachability analysis. Explicitly documented as unsound for branches: `iter_with_arms` flattens conditional arms into the same producer set without recording arm membership, so **cross-arm leakage is a known, uncaught gap** (`_construct_validation.py:141-150`, "Cross-arm leakage... NOT caught here... documented limitation... neograph-vn5f"). So today's check answers "was this declared upstream in source order," not "is this guaranteed present on every path that reaches this node."

**3. Is provides/requires ~90% expressible via output-key renaming? No — real missing layer.**
Dict-form outputs give you a distinct *key* per artifact, but the field name is always prefixed by the producing node's own identity (`field_name_for(name)` is baked into every producer registration, `:217`, `:223`, `:236`). Swapping which node supplies a contract means renaming that node (and updating every consumer's key string) — there's no separate contract namespace, no "any of these N nodes satisfies requirement X" resolution, and no runtime state-key indirection (the compiled state field literally *is* the node-derived name; `state.py`'s `_add_output_field` mirrors this). True provides/requires substitutability needs: (a) a contract-name registry independent of `node.name`, (b) multi-candidate-producer resolution with ambiguity/exclusivity rules (especially across the unsound branch-arm case from #2), and (c) a state-bus key mapping layer decoupling the wire format from producer identity. This is maybe 30-40% there (key-vs-type decoupling exists), not 90%.

**4. Cross-reference to `context=`.** `context=` is the closest existing analog — it already breaks strict adjacency, letting a node read any reachable upstream/ambient producer by name from anywhere in the tree (position-independent). But it's still identity-keyed (`field_name_for(ctx_name)` must match an actual producer's field, `:194`), not contract-keyed, and shares the same branch-unsoundness gap. `di_inputs`/`FromConfig` are truly ambient (config-sourced, not node-produced), so they're orthogonal — not part of the provides/requires dataflow graph at all.

---

### raw_channels
## Findings: raw_inputs/raw_di_inputs/raw_context vs. proposed Carried/Selected output fields

**1. What the three raw_* channels give a prompt_compiler, and which side they're on**

All three are INPUT-side only — they exist purely so a `prompt_compiler` can receive the *live Python object* behind a prompt variable instead of (or alongside) its rendered string, for template LOGIC (branching, iterating a model's children) rather than `{var}` text substitution:

- `raw_inputs` (`_llm_render.py:197,211`) = `to_raw_inputs(input_data)` — the node's own upstream inputs, unrendered.
- `raw_context` (`_llm_render.py:233`) = `to_raw_inputs(context)` — the node's declared `context=` fields, unrendered.
- `raw_di_inputs` (`_llm_render.py:249`) = `to_raw_inputs(di_inputs)` — resolved `FromInput`/`FromConfig` values, unrendered.

All three are built by the same `to_raw_inputs` (`renderers.py:441-455`), the un-rendered twin of `to_prompt_input` (`renderers.py:418-438`); `to_rendered` (`renderers.py:515`) is the shared rendering ladder each rendered sibling goes through. All three are opt-in via the same `prompt_compiler_params` introspection gate (`_llm_render.py:255-258`). AGENTS.md's own framing confirms this: `raw_di_inputs` exists because "a compiler using a DI scalar for LOGIC... had no way to recover the typed value," and `NO_RAW_SIBLING_ALLOWLIST` is now empty — every rendered channel *that flows into prompt construction* has a raw sibling. There is no OUTPUT-side raw channel; nothing here touches the model's structured-output parse.

**2. Same mechanism or distinct?**

Related-but-distinct, not the same mechanism viewed from two ends — they don't share infrastructure with a Carried/Selected-output proposal, and shouldn't be described as such. raw_inputs/raw_di_inputs/raw_context feed the *compiler that builds the prompt going INTO the model* (`runtime.prompt_compiler(template, prompt_input, **kwargs)`, `_llm_render.py:260`). Carried/Selected fields would operate on the *far side*: after the model returns, filling fields of the node's OWN output type from a tool result or upstream value without model transcription — i.e. post-parse state construction (`_build_state_update` / `invoke_structured`/`invoke_with_tools` territory in `factory.py`), not prompt assembly. `to_raw_inputs`/`to_rendered` have no role there today. The resemblance is thematic (both are "give the runtime the real object instead of a rendered string") but the raw_* channels solve it for template-authoring logic, while Carried/Selected solves it for output materialization — different lifecycle stage, different call sites, no shared function.

**3. Existing "model emits X, runtime materializes Y from X" pattern?**

`factory.py`/`_oracle.py` output-strategy code has nothing resembling "model picks a key, runtime looks up the real object" for output *fields*. `_oracle.py`'s `_merge_variants`/`_run_merge_prompt`/`_run_merge_fn` (`_oracle.py:113-407`) select among *already-materialized* variant objects by index/fn, not by model-emitted key lookup into a fresh object.

The closest real precedent is tool-calling itself, in `_agent_cycle.py`/`_agent_tool_calls.py`: the model emits `tool_calls` (name + args, `_agent_cycle.py:499,535`), and the runtime looks up the actual registered tool function by name and executes it, materializing the real result as a `ToolInteraction` (per the `di_inputs`/tool_log AGENTS.md section). That's "model emits a selector, runtime resolves the real object" — but for tool dispatch, not for populating a node's declared output-model fields. No existing code takes a model-chosen key and looks up/copies a value into an output-model field.

---

### agent_spec_fidelity
Now I have enough to write the report.

## Findings: Agent Spec export fidelity — DataFlowEdge, Each/Loop lowering, and conformance

**1. Each/Loop per-iteration and per-fan-out data — documented gap, not a silent last-write-wins.**

The exporter is deliberately fail-loud-first (`src/neograph/_agent_spec.py:18-28`): anything it cannot round-trip safely raises `ConfigurationError` rather than emitting a "looks-correct" placeholder. Within that discipline it does still have a real, *known and ticketed* hole:

- **Each → MapNode** (`src/neograph/_agent_spec_modifier_lowering.py:264-303`, `_lower_each`): the `over`/`key`/`on_error` fields have no Agent-Spec primitive and ride only in `metadata['neograph/each_spec']`. Critically, per neograph-**qtfof.7**, the MapNode's `iterated_item` input has **no real `DataFlowEdge`** — the fan-out source is metadata-only. A metadata-blind (third-party) Agent Spec runtime has literally no wired data path telling it what to iterate over — confirmed live: e.g. cell `scripted-each-single` exports with `data_flow_connections=None`. This isn't last-write-wins corruption; it's a structurally missing edge that a foreign runtime can't execute at all. Filed 2026-08-05, still OPEN.
- **Loop/Operator → BranchingNode** (`_agent_spec_modifier_lowering.py:376-385`/`:634-641`): same pattern — the `when` predicate lives only in `metadata['neograph/loop_spec'|'operator_spec']['when']`; the `branching_mapping_key` input that should carry the decision has no `DataFlowEdge` either (neograph-**qtfof.6**, same filing date, OPEN).
- There's a third, separately confirmed gap: the **outermost Construct's EndNode declares no outputs** (`construct.output` is always `None` at the top level by design), so a third-party runtime's `invoke()` silently returns `{}` even though the graph computed a real result — and if outputs *are* declared, the third party raises instead because there's still no peer edge feeding the EndNode (neograph-**qtfof.9**, merged from qtfof.10). This is the closest thing to "looks correct but isn't" in the codebase — verified via an actual EXECUTE+COMPARE harness (`neograph-jn555.31`/dgbqv.2), not just static shape inspection.
- The module docstring's Core Invariant framing ("never a silent downgrade or truncation") holds for *neograph-round-trip* fidelity (its own `from_agent_spec()` reads the markers back structurally-validated, `loader.py:226-233`). It does **not** yet hold for a genuinely metadata-blind third-party consumer — that promise is aspirational and currently broken in three specific, ticketed ways.

**2. Open beads issues — yes, under epic `neograph-qtfof` ("Portal × Agent Spec — v2: deferred features + docs"):**
- `qtfof.6` — BranchingNode's decision input has no DataFlowEdge (Loop/Operator).
- `qtfof.7` — MapNode's `iterated_item` has no DataFlowEdge (Each).
- `qtfof.8` — `LlmConfig(api_provider=None)` rejected by third-party loader for think-mode cells.
- `qtfof.9` — outermost EndNode declares no outputs → empty/erroring third-party `invoke()`.
All are P3, OPEN, found during a dedicated `dgbqv.2` research/EXECUTE+COMPARE pass (2026-08-05/06) that specifically probed real third-party interop rather than just wire round-trip.

**3. No conformance classifier exists.** Export is unconditional in the sense that `to_agent_spec()` always attempts the lowering and either succeeds or raises `ConfigurationError` for the cases it *knows* it can't represent (raw_fn, callable `Loop.when`, Oracle merge hooks, dispatch-mode Portal, etc. — enumerated at `_agent_spec.py:26-35`). There is no separate predicate/function that answers "is this Construct losslessly exportable to a metadata-blind Agent Spec reader" — the qtfof.6/.7/.9 gaps above are exactly cases that pass export silently (no raise) but are *not* actually consumable by a foreign runtime. That's the concrete design gap the proposal should target: today "doesn't raise" is being used as a stand-in for "is portable," and those are provably not the same thing.

---

## Appendix B: Feasibility reasoning (5 fable passes)

## reasoned_carried_selected
## Feasibility assessment: Reasoned / Carried / Selected output-field kinds

**Verdict: smaller than a new-IR feature, bigger than pure sugar — a two-seam DX/dispatch-layer change with one genuinely hard sub-problem (Selected's referent resolution). Ship Carried first.**

### Where projection and assembly plug in

Both seams already exist and are narrow:

- **Projection** (derive the LLM-facing schema): `describe_type`'s field walk (`_render_model_body` / `_render_type`, `src/neograph/describe_type.py:83-130`). Today it reads `model_fields` but never `get_type_hints(model, include_extras=True)`, so it is marker-blind. Add a marker-aware pass: drop `Carried` fields, replace `Selected[T]` with its key type (`str` / `Literal[...]`). The call site is single: `output_schema = describe_type(output_model)` inside `invoke_structured`/`ainvoke_structured` (`_llm.py:315` via `:239/371`), fed by `_resolve_primary_output` (`_dispatch.py:364-382`) — always one concrete model, so projection is a pure function `project(model) -> synthetic_model` computed once per node. The `with_structured_output` path must receive the *projected* model too, not just the prompt schema — this needs a synthesized Pydantic subclass, not just string surgery.
- **Assembly** (fill Carried, materialize Selected): the post-parse/pre-write seam in `ThinkDispatch.execute`/`aexecute` (`_dispatch.py:280-284` / `301-315`) and `_shape_tool_output` (`_dispatch.py:318-338`) for agent/act. `result` is fully parsed there and `_build_state_update` downstream is agnostic to provenance. One `_assemble_output(node, projected_result, input_data, config)` call at those three sites (the sync/async twin duplication is the codebase's accepted pattern).

### Sugar or new IR capability?

**DX-layer sugar, precisely because no IR consumer changes.** Test against the project's own Portal bar: (1) no new `Node` field is needed — the markers live on the output *model's* fields, recoverable at dispatch time via `include_extras=True`, exactly as `_di_classify.py` does for params; the classification can be cached on a PrivateAttr or a new `_output_classify.py` module without touching `node.py`; (2) no new runtime control-flow construct; (3) no new state keys — the *declared* (unprojected) type is what `state.py`, the fan-in validator, and fingerprints see, so downstream consumers and `effective_producer_type` are untouched. The off-limits modules (`factory.py`, `_construct_validation.py`, `node.py`) stay closed. The one wrinkle: `describe_type` is public API shared by all surfaces, so marker-awareness there is a general capability, not @node-specific — that's fine, it's the DI-marker precedent (`di.py` is likewise surface-neutral).

**Caveat that could escalate it**: if `Carried(path)` may reference *other nodes'* outputs beyond the node's own `input_data`/config, you're adding a dataflow edge the validator can't see — that would demand IR-level `inputs` awareness. Restrict phase 1 to `input_data` + config and it stays sugar.

### Closest existing mechanism

`FromInput`/`FromConfig` + `DIBinding.resolve` — the same idiom on the input side. Extend the pattern (new classifier over model fields, same `Annotated` markers, same resolve-shape), don't extend `DIBinding` itself: its resolution sources (config) differ from Carried's (`input_data` paths). For Selected, the only precedent is tool-call dispatch ("model emits name, runtime resolves object") — thematic, not reusable code.

### North-star risk

Two silent-failure re-admission points, both fixable:

1. **Carried path resolves to None/missing** at runtime → a silently-null field the model "never lied about." Must be (a) validated at *assembly time* — lint/validator checks the path against the upstream's model fields, a should_fail check-fixture — and (b) `ExecutionError` at runtime on missing, never `None`-fill. Fail-soft here is an existential defect by the project's own rule.
2. **Selected key doesn't exist in the referent collection.** The model *can* emit a bad key; project to `Literal[...]` of actual keys when the collection is statically known, otherwise fail loud on lookup miss (with a bounded-retry story deferred). Also: `describe_type` marker-blindness in any *second* render path (e.g. `describe_value` few-shot examples showing unprojected instances) would silently teach the model to emit Carried fields — audit both renderers, pin with a guard test.

### Phase-1 slice

**Carried only, own-inputs-and-config paths only, think mode only**: marker + `_output_classify.py`, projection in `describe_type` + synthesized projected model in `_dispatch`, assembly at the ThinkDispatch seam, fail-loud on unresolvable path, lint rule + should_fail fixture, three-surface parity test (declarative/programmatic surfaces exempt-or-supported explicitly). Selected and agent/act wiring are phase 2 — Selected's referent-typing question (what may a key point at?) deserves its own design round. Reasoned, if it's just "model-authored, keep in schema," is the default and costs nothing.

Key files: `src/neograph/describe_type.py`, `src/neograph/_dispatch.py`, `src/neograph/_llm.py:239-320`, `src/neograph/di.py` (pattern donor), `src/neograph/_di_classify.py` (pattern donor).

---

## scoped_addressing_ledger
## Feasibility assessment: scoped lexical addressing (3.4) + tool ledger (3.6)

### (b) Tool ledger — ~70% already built; the doc's framing overstates greenfield

`ToolInteraction` (tool.py:148-164) already captures `tool_name`, `args`, rendered `result`, `typed_result`, `duration_ms`, per call, in call order, checkpointed via the `_concat_reducer` list channel (state.py:372). Collection is unconditional (the "demand-driven" AGENTS.md claim covers only *exposure*, not collection — `_tools_result` always writes the internal channel). What is genuinely missing:

1. **Ordinal**: no per-tool call index on the record; "2nd call to `search`" is consumer-side filter-then-index. The only counter (`ToolBudgetTracker._counts`) is a running budget total, never stamped on records. Fix: one `int` field stamped at `_agent_tool_calls.py` build time from the tracker's count — trivially cheap since the counter already increments there.
2. **Selectors**: no `first/last/all(tool_name)` helpers, no grouped `dict[tool_name, list]` view. A read-time projection (a small `ToolLedger` view class wrapping the flat list) needs zero state-bus change.
3. **Cross-node ledger**: `tool_log` is per-node. A run-wide ledger would be a new state channel — defer; nothing in Example 4 requires it if the consumer declares the producing node's `tool_log`.
4. **Replayability**: `typed_result` is a live object; `ProducingCall` (tool.py:167-183) is the existing args-only replay record but carries no result/ordinal. If the ledger contract wants durable results, that's a serialization decision, not a structure.

Phase-1 verdict: **ship it as ordinal field + reader-side selectors over the existing flat list**. No IR change, no normalizer, no guard — dispatch/DX layer only.

### (a) Scoped addressing — split into three tiers with sharply different costs

- **`latest` / `all_in_scope` for non-nested Loop: near-free.** The full iteration history already lives on the bus (`_append_loop_result`, list position = iteration index since Loop is sequential). `_unwrap_loop_value` already returns the full list to `list[T]` consumers. `from_enclosing(n)` is `list[-n]`. Pure read-time projection; document + name it, don't re-store it.
- **Each `iteration_index`: small IR addition, but semantically fraught.** Each keys by business field (`getattr(item, each.key)`), and the barrier's arrival order ≠ `each.over` order — the AGENTS.md caveat is load-bearing. A true positional index must be synthesized at dispatch and threaded through the `Send()` payload (an `EACH_ITEM`-adjacent state key). Doable, but exposing an index that *looks* deterministic over a nondeterministic-arrival dict is exactly the kind of silent-wrongness the North star forbids. If added, the index must come from `each.over` position stamped pre-Send, not barrier order.
- **`scope_path` (the nested case the doc is really about): genuinely new IR, Portal-scale.** Sub-constructs are per-invocation isolated (`neo_subgraph_input`, fresh state each outer iteration/item); inner structure is discarded at the `output` boundary. There is no representation to project *from* — a scope-path channel would need threading through subgraph input, stamping on barrier writes, a single-writer normalizer field, and a structural guard. Note also: nested Loop×Each on a *single* node is unrepresentable today (`ModifierCombo` has no dual-primary shape); nesting exists only structurally via sub-constructs, and their per-iteration data already vanishes at the boundary. So the nested-addressing use case is partly addressing a composition that barely exists yet — validate demand (piarch) before paying Portal-scale cost.

**Verdict**: `(name, scope_path, iteration_index)` should NOT be a new state-bus representation for tiers 1–2 (projection wins: cheaper, no schema-fingerprint churn, no checkpoint migration). Only `scope_path` needs new plumbing, and it should be deferred.

### Biggest North-star risk

A scope-addressing DSL that resolves *at runtime* re-admits the exact failure class Portal closed: an address like `verify[3].score` that silently resolves to `None` (missing key, pruned iteration, arrival-order index) is a silent seam. Any phase must make bad addresses a **compile-time `ConstructError`** (resolvable against declared modifiers) or fail-loud at runtime — never `None`-soft. The Each arrival-order index is the specific trap.

### Recommended phase-1 slice

1. Tool ledger: ordinal field + `ToolLedger` selector view (smallest, self-contained, no IR).
2. Loop `all_in_scope`/`from_enclosing`: documented projection over the existing history list.
3. Each index: only with pre-Send positional stamping; defer otherwise.
4. `scope_path`: defer pending a real piarch nested use case; if built, follow the Portal pattern (single-writer IR field, guard-pinned).

---

## provides_requires
## Feasibility assessment: provides/requires named-contract dataflow

**Distance from current fan-in: real new machinery, ~30-40% there, not a rename.** Today's fan-in is identity-keyed end-to-end: dict-form `Node.inputs` keys must equal `field_name_for(producer.name)` (+`_{output_key}`), `ProducerMap` is `OrderedDict[field_name, Producer]` with exactly one producer per field, and the compiled LangGraph state field *is* the producer-derived name (`state.py` mirrors it). Dict-form outputs decouple artifact-key from type but not from producer identity. Contracts need three genuinely missing pieces: (a) a contract namespace independent of `node.name`, (b) multi-candidate resolution with ambiguity/exclusivity rules ("N nodes can satisfy C" has no representation — last-writer-wins today), (c) a state-bus indirection layer so the wire key isn't the producer's name. (b) and (c) touch `_construct_validation.py`, `state.py`, and `factory.py` — IR/compiler layer, off-limits for sugar. Per the Portal precedent, that means new IR field(s) (`Node.provides` / contract entries in `inputs`), a single normalizer writer, and a structural guard. This is a Portal-class addition, bigger than the doc implies if the doc frames it as aliasing.

**"Requires is statically checkable" is already ~70% true — restated, minus one soundness hole.** `_validate_node_chain` already checks every consumer input against accumulated upstream producers at assembly time, with typed mismatch errors via `effective_producer_type`. But it is *declaration-order accumulation*, not per-path reachability: `iter_with_arms` flattens branch arms into one producer set, so cross-arm leakage is a known uncaught gap (neograph-vn5f). A contract system that promises "guaranteed present on every path" would be **falsified by the existing branch unsoundness** — the doc's claim is only honest if vn5f is fixed first or the guarantee is explicitly scoped to linear flow. Do not ship the promise before the soundness.

**vs `context=`: complement, and eventually the substrate.** `context=` already gives position-independent reads but is identity-keyed and shares the branch gap. Contracts and `context=` should converge: a contract name is exactly what `context=` *wants* to reference (capability, not producer). Phase-1 should not replace `context=`; long-term, `context='summary'` resolving via the contract table instead of `field_name_for` is the right end state — one lookup mechanism, not two. `di_inputs`/`FromConfig` are orthogonal (config-sourced, not dataflow).

**North-star risk: indirection re-admits dangling references — the exact failure class neograph deletes.** Today a wrong `inputs` key fails loud at assembly because the namespace is closed (producers that exist). A contract namespace is open: `requires('summary')` where nothing provides it, or two providers on divergent branch arms where only one runs, is a *new* way to write a broken program. Combined with vn5f, "contract satisfied on some arm" would validate and then be missing at runtime — a silent seam, existential per AGENTS.md. Mitigations must be non-negotiable: unresolved contract = assembly `ConstructError`; duplicate providers = error (no priority/override semantics in v1); contracts registered per-Construct, no global registry. Also note Agent Spec export: contract indirection adds another neograph-only concept that must lower to real `DataFlowEdge`s or it deepens the qtfof.6/.7 metadata-only-edge family.

**Minimal phase-1 slice (no multi-candidate resolution, no branch semantics):**
1. `provides='name'` as pure aliasing: normalizer (`_ir_normalize.py`, single writer, G3-style guard) builds a `{contract_name: field_name}` table on the Construct; validator resolves `inputs`/`context` keys through it before `ProducerMap` lookup. Duplicate `provides` → `ConstructError`. Zero state-bus change — the state key stays producer-derived; only validation-time lookup is indirected.
2. Consumer side: allow contract names in dict-form `inputs` and `context=`, resolved via the table; `_extract_input` maps back through the same table at runtime (one shared resolver, both sites).
3. should_fail fixtures: unresolved contract, duplicate providers, cross-arm single-provider (xfail pending vn5f).
Defer: multi-candidate satisfaction, substitutability sets, Agent Spec lowering. That slice delivers producer-rename-without-consumer-churn (the doc's core ergonomic win) while keeping the broken-state set closed.

---

## agent_spec_interop
## Feasibility assessment: Section 5 (Agent Spec interop, Option 1 vs 2)

Note: the design doc body arrived as `undefined` in my task payload; this assessment maps section 5's described claims against the codebase/beads ground truth from the research findings.

**1. Are the fidelity gaps already tracked?** Yes, substantially. Epic `neograph-qtfof` already holds four P3 open tickets that cover the structural core of any 5.1 matrix: `qtfof.7` (Each→MapNode has no real `DataFlowEdge` for `iterated_item` — fan-out source is metadata-only), `qtfof.6` (Loop/Operator→BranchingNode decision input has no edge; `when` lives only in `metadata['neograph/loop_spec']`), `qtfof.8` (`LlmConfig(api_provider=None)` rejected by third-party loaders), `qtfof.9` (outermost EndNode declares no outputs → foreign `invoke()` returns `{}`). These were found via a real EXECUTE+COMPARE harness (dgbqv.2), not shape inspection. If 5.1 restates these, it adds no new work — it should cite the tickets. What the doc CAN add that the tickets don't: gaps tied to the *other* proposed primitives (tool-ledger export, scoped addressing, provides/requires contracts) have no Agent Spec representation at all and are genuinely untracked — but they're also unbuilt in neograph, so they're speculative rows, not export bugs.

**2. Option 1 — conformance classifier: buildable now, and it's the right move.** The critical finding: `to_agent_spec()` already fail-louds on *known-unrepresentable* features (raw_fn, callable `when`, merge hooks, dispatch-mode Portal — enumerated at `_agent_spec.py:26-35`), but the qtfof.6/.7/.9 cases **pass export without raising while being unexecutable by a metadata-blind runtime**. "Doesn't raise" is currently a false proxy for "portable." A classifier partitioning into `PORTABLE / NEOGRAPH_ROUND_TRIP_ONLY / NOT_EXPORTABLE` is well-defined today with zero dependency on scoped-addressing or tool-ledger work: its predicates are purely structural over the existing IR (does the construct contain Each/Loop/Operator → ROUND_TRIP_ONLY per qtfof.6/.7; top-level Construct → ROUND_TRIP_ONLY per qtfof.9; the existing raise-list → NOT_EXPORTABLE). It's a walk over `Construct.nodes` + modifier inspection — no new IR fields, no normalizer writer, no factory/validation edits, so it clears the layer-discipline bar trivially (a new sibling module in the `_agent_spec*` family). It also serves the north star: it converts a silent seam ("export succeeded but a foreign runtime can't run it") into a loud, classified one — exactly the subtractive posture.

**3. Option 2 — extend the spec: not near-term.** Getting Oracle's Agent Spec (or WayFlow's implementation) to grow real fan-out edges, iteration-history channels, or scope-path addressing is standards work with an external counterparty, multi-quarter at best, and contradicts the locked positioning memory: neograph is *not* a standard-setter; it imports/exports Agent Spec, it doesn't shape it. Option 2 should not block 0.8. The one cheap sliver of Option 2 worth keeping: the metadata-marker convention (`neograph/each_spec` etc.) already IS a de-facto vendor extension; documenting it publicly costs a docs page, not negotiation.

**4. Phase-1 recommendation (minimal diff).** Ship Option 1's classifier:
- New `export_conformance(construct) -> ConformanceReport` (module beside `_agent_spec.py`), three-tier verdict + per-finding ticket refs; predicates initially = the qtfof.6/.7/.8/.9 conditions + the existing raise-list.
- Wire it into `to_agent_spec(strict=...)`: default emits the report (or warns) on ROUND_TRIP_ONLY; `strict=True` raises. No wire-format change, no fixture churn.
- Should_fail/should_pass fixtures per tier; guard test asserting every classifier predicate cites an open or closed bead.
- Fixing the actual edges (qtfof.6/.7/.9) becomes ordinary follow-on work *behind* the classifier — each fix flips a predicate off, giving a measurable portability ratchet.

Estimated size: one module + tests, no IR changes, days not weeks. Option 2 deferred to post-0.8 positioning.

---

## cross_cutting_north_star
**Gatekeeper verdict on the composite proposal** (note: the design doc body arrived as `undefined`; this ruling is grounded in the five codebase-mapping passes, which characterize each primitive concretely).

**1. Does the bundle extend "unrepresentable"?** Mostly no — and be precise, because the marketing temptation here is real:

- **Carried(path)** is the only piece that genuinely extends the fan-in/type class: a path into the node's own input/context models is statically resolvable at assembly time (path exists, type compatible), same footing as `_check_fan_in_inputs`. That's a real, small extension.
- **Selected** is NOT compile-time-checkable in the same sense: the offered set (tool-ledger entries, list contents) is runtime data. Only the field's *type* is checkable; membership is a runtime check at best. Calling it "unrepresentable" would overclaim.
- **Reasoned** is prompt-schema sugar. Zero safety delta.
- **Provides/requires** *could* extend the claim, but the validator it would sit on is branch-unsound today (cross-arm leakage, neograph-vn5f, explicitly documented as uncaught). A contract layer atop unsound reachability advertises a guarantee that is false — an existential defect by our own North-star tax, not a feature. It cannot honestly join the bounded claim until per-path reachability lands first.
- **Scoped addressing / tool ledger** are expressiveness and observability, not safety. `scope_path` across sub-construct boundaries is Portal-scale new IR (findings #3, scoping pass), and index-selectors over Each would *launder the documented arrival-order non-guarantee into a safe-looking API* — an anti-safety move.

**2. Highest silent-seam risks if built carelessly:**
- **Carried → None**: a path that resolves missing must raise, not default. Worse: if the marked field has a Pydantic default, stripping it from the LLM schema while forgetting the mandatory splice yields a silently-defaulted field. The strip (in `describe_type`) and the splice (at the `_dispatch.py` post-parse seam) are two sites that must move in lockstep — the exact lint/runtime-divergence failure mode AGENTS.md already documents for `di_inputs`. This needs a structural guard from day one.
- **Selected fallback**: model emits a key outside the offered set → any "accept the string anyway" fallback is the silent seam. Also `ToolInteraction.typed_result` is a live object, not guaranteed serializable — Carried/Selected-from-ledger silently breaks on checkpoint resume.
- **Ledger has no ordinal** (tool_ledger pass): "2nd call to search" is consumer list-filtering; a Selected referencing it without an ordinal field is positional guesswork dressed as addressing.
- **Agent Spec export**: every new marker is more metadata-only lowering, widening the already-ticketed "doesn't raise ≠ portable" gap (qtfof.6/.7/.9).

**3. Scope for one maintainer, one consumer:** wholesale, this is a multi-quarter research program. Provides/requires needs a contract-name registry + branch-sound reachability + state-key indirection (findings: ~30-40% exists, not 90%). `scope_path` needs new state-bus plumbing comparable to the Portal mesh. The ledger needs ordinals, a serialization story, and cross-node aggregation. Meanwhile the 0.8 backlog holds *concrete correctness debt*: exported graphs a metadata-blind runtime literally cannot execute (qtfof.7's missing MapNode DataFlowEdge, qtfof.9's empty invoke). Trading known broken-for-third-parties output for speculative new primitives is the wrong ordering.

**4. Verdict: explicitly defer the bundle past 0.8.** Do not fold it into neograph-qtfof — it's orthogonal, and qtfof's real missing piece is a *conformance classifier* ("losslessly exportable?" as a predicate), which every new marker would further burden.

**Narrow slice to land now** (one epic, ~1-2 weeks, dispatch-layer only per the ir_outputs findings — no new IR fields, no off-limits modules):
1. **Carried + Reasoned** as `Annotated` output-field markers at the two existing post-parse seams (`ThinkDispatch` + `_shape_tool_output`), fail-loud on unresolvable path, with a guard test pinning describe_type-strip ↔ splice lockstep.
2. **Document the free Loop history projection** (`list[T]` consumer = full iteration history) — it already works and is unadvertised.

Defer Selected until the ledger grows an ordinal field; defer provides/requires behind vn5f branch soundness; defer scope_path behind an explicit new-IR-capability decision with its own single-writer + guard design, per the Portal precedent.