# Essential vs Incidental Complexity Review — 2026-06-02

A Brooks-lens audit of the inventory called out in the architecture-decisions doc and the cross-cutting findings the maintainer flagged from recent sessions. Single question per item: *how much of this complexity is inherent in the problem of compiling typed Python pipelines to LangGraph graphs, and how much is incidental to choices we made along the way?*

The cure is upstream of the symptom. Where I find an "INCIDENTAL — architectural" pattern, the fix must NOT trade structural complexity for branching complexity, NOR collapse types into dicts. Both moves shift complexity rather than remove it; Brooks calls this swapping accidents.

---

## Summary

Of the 15 items inventoried, roughly **40% is essential** (validator polymorphism, fingerprinting, state bus, normalizers, three closure factories for three topologies, LLM runtime as a bundle, kwarg-driven compile, retry concern split, multiple test fakes for multiple LLM-shape contracts), and **60% carries some incidental complexity** — most of it concentrated in three sites:

1. **The four-branch `_call_structured` recovery ladder + DSML detection** (§4) — INCIDENTAL — accreted; the symptom of a single upstream cause (LangChain's `include_raw=True` polymorphism + provider-specific tool-call markup leaking into structured-output channels).
2. **Three assembly paths producing the same IR but normalizing it differently** (cross-cutting) — INCIDENTAL — architectural; the upstream cause is that `Construct.__init__` was retrofitted with `_normalize_fan_out_params` and `_normalize_oracle_gen_type` to catch up with what `_build_construct_from_decorated` did eagerly. Each new IR field that the @node path infers becomes a new `_normalize_*` method in `construct.py`. Compounds with item 3.
3. **`__name__` doing double duty as LangGraph node identity AND error label** (cross-cutting) — INCIDENTAL — architectural; this is the upstream cause of (a) `node_name=` threading through the three closure factories in `_oracle.py`, (b) the `_ = node_name` placeholder at `_oracle.py:68`, and (c) the structural guard that polices it. One label-and-identity collision is generating defensive code in four files.

The two **root upstream choices** I would change are stated at the end under "The incidental-complexity tree": the assembly-path divergence (single-pass declarative builder shared by YAML + Construct.__init__), and the LangGraph node-identity/label collision (give the Node a separate `display_label` so `__name__` becomes a pure routing key).

The 4268-line `tests/test_structural_guards.py` is INCIDENTAL — local (a budgeted price worth paying), but its size is itself diagnostic: 60 guard classes is a measure of how much of the architecture is currently policed by negative space rather than positive types. Each guard is a place where the type system cannot express the invariant. That is a price we have priced in deliberately, but it puts a ceiling on how much more structural complexity the codebase can accept before the guard layer becomes the dominant maintenance cost.

---

## §1 LangGraph commitment / `llm_factory` indirection

**File**: `src/neograph/_llm_runtime.py:1-140`, decision at `docs/design/architecture-decisions.md:7-15`.

**Essential complexity**: Provider-neutrality lives at the model layer (any `BaseChatModel`) but execution is committed to LangGraph. The `llm_factory` callable is the user's hook to construct provider clients (OpenAI, Anthropic, OpenRouter, Bedrock) without neograph naming any of them. This is genuinely irreducible — neograph cannot pick a provider for the user, and the user's choice must be late-bound. FastAPI's `Depends(...)` resolves the identical problem the same way.

**Incidental complexity**: None of significance in this file specifically. The `_accepted_params` inspect-and-cache pattern at lines 29-43 is essential (avoiding `inspect.signature` per request) and the result is closed over once. The `EMPTY_RUNTIME` sentinel at line 93 is essential — distinguishing "no runtime supplied" from "runtime with all-None fields" is a real semantic distinction (the scripted-only construct fully compiles with neither, but an LLM-only construct must raise).

**Verdict**: ESSENTIAL. The shape is correct: a frozen bundle of user-supplied callables, captured once, threaded as a value. Nothing in this module wants to be different.

**Note**: The `check_llm_kwargs_or_raise` helper at lines 96-140 is essential domain logic (fail-loud at compile time, not at run time, when LLM-mode nodes lack their factories). Its function-local imports of `Construct`, `Node`, `CompileError` (lines 110-112) are deliberate cycle-avoidance, not accreted defense.

---

## §2 `LlmRuntime` frozen dataclass

**File**: `src/neograph/_llm_runtime.py:46-93`.

**Essential complexity**: Two `compile()` calls must produce two isolated graphs that coexist in one process. The runtime kwargs (`llm_factory`, `prompt_compiler`, `renderer`, `cost_callback`) are coupled — they travel together through every factory closure, and freezing them at compile time is what makes the isolation guarantee real. The frozen dataclass is the smallest possible object that meets the contract.

**Incidental complexity**: The dual representation (`llm_factory` AND `llm_factory_params: frozenset[str]`) is borderline. The params set exists to avoid `inspect.signature` on every prompt-compiler call. That is an essential optimization (a 30-node pipeline would call `inspect.signature` 30+ times per run). The cache placement on the bundle is correct — the bundle is the right scope for this caching.

The `EMPTY_RUNTIME` module-global sentinel (line 93) is shared mutable-by-convention state, but since the dataclass is frozen, this is safe and idiomatic. Pydantic's `Field(default_factory=...)` reuses this exact pattern.

**Verdict**: ESSENTIAL. The `LlmRuntime` is the minimum structural object that encodes the §2 isolation contract. It would be wrong to break this into individual args threaded through every helper signature; doing so would multiply the surface area at every closure boundary.

---

## §3 Per-compile registries (compile() signature)

**Files**: `src/neograph/compiler.py:57-71` (signature), `compiler.py:120-140` (registry assembly), `compiler.py:280-320` (`_collect_required_di`).

**Essential complexity**: Per-compile lookups for `scripted`, `conditions`, `tool_factories` are essential to the §2 isolation guarantee. Module-level registries were the upstream cause of test pollution and multi-tenancy bugs; killing them was correct. The kwargs are the registration path; there is no other.

**Incidental complexity present**: Yes, **INCIDENTAL — local**. The signature has accreted two private kwargs — `_runtime` and `_scripted_lookup` (compiler.py:69-70) — used only by `_add_subgraph` to pass parent state to child compile. The leading underscore prefix is the only thing keeping these out of the public API. The same effect could be achieved with a private helper `_compile_subgraph(construct, *, runtime, scripted_lookup, ...)` that bypasses the public-signature ceremony entirely. Cost to fix: low (one helper extraction). Benefit: `compile()` becomes the strictly public surface it claims to be.

Decorator-side registries (`_decorator_scripted`, `_decorator_conditions`, `_decorator_tool_factories` imported at compiler.py:35-39) are module-global mutable dicts populated by decoration. They DO cross test boundaries — `tests/fakes.py:96-104` reads from them. This is INCIDENTAL — architectural: the @node decorator wants to attach merge-fn / interrupt-when / tool shims at decoration time, but the only place to store them is module state. The cleanest fix is to attach these to the function's own `__wrapped__` (or a sidecar dict keyed by the function object), then have `compile()` collect them by walking the construct's node sidecars rather than by reading a global. This eliminates the decorator-side globals entirely and brings DI registration in line with the per-compile contract.

**Verdict**: The public contract is ESSENTIAL. The private compile-kwarg back-channel and the decorator-side globals are INCIDENTAL — fixable without changing the user-facing API.

---

## §4 `_llm_retry.py` + LangChain `include_raw=True` four-branch ladder

**Files**: `src/neograph/_llm_retry.py:1-361`, `src/neograph/_llm_dispatch.py:24-119`.

**Essential complexity**: LLMs return malformed JSON. Repairing it (json_repair + null-default coercion + bare-array unwrap) and retrying with structured feedback is an irreducible domain problem. `_invoke_json_with_retry` (lines 264-307) and `_parse_json_response` (lines 154-227) encode genuine knowledge — how to extract a balanced JSON value from prose-wrapped LLM output, when to coerce nulls, when to wrap a bare array into a single-list-field model.

The four hook points `merge_pre_process` / `merge_post_process` / `merge_fallback` (oracle.py:288-296) are essential — each one solves a concrete user need (custom variant aggregation, post-merge enrichment, graceful degradation on LLM failure).

**Incidental complexity present**: **INCIDENTAL — accreted, and severe**. `_call_structured` at `_llm_dispatch.py:48-109` has a four-branch tree:

1. Normal path: `with_structured_output(model, include_raw=True)` → `{"parsed": ..., "raw": ...}` → extract.
2. Silent variant: `parsed=None` + raw contains DSML → re-parse via `_attempt_dsml_recovery`.
3. TypeError on `include_raw=True` (provider rejected): retry without it.
4. TypeError on retry: peek last message for DSML, else raw-invoke, then `_attempt_dsml_recovery`.

The upstream cause is not in our code. It is two converging realities:
- LangChain's `with_structured_output(include_raw=True)` shape is not uniformly supported across providers — some raise `TypeError`, some return a `dict`, some return the bare parsed model. Our four branches are walking a compat ladder over LangChain's provider polymorphism.
- DeepSeek R1 (and a few others) emit XML/DSML tool-call markup in structured-output channels after budget exhaustion. Detection is content-based (`_DSML_PATTERN` at line 29) because no signal in the response shape identifies it.

**Architectural fix location**: Above `_call_structured`. The right abstraction is a thin compat shim that *normalizes* the LangChain `with_structured_output` return into a single shape (`Result = Parsed(model) | Raw(text) | Failed(error)`) and absorbs the TypeError-on-include_raw retry. Once that shim exists, `_call_structured` becomes a switch on `Result` with one path per case. The DSML detection is then a single post-step that runs whenever the shim returns `Raw(text)` or `Failed`.

This is INCIDENTAL — accreted because each branch was added in response to a specific provider failure (`neograph-0tid` traces visible in the comments). The branches are real bugs we've already fixed; the structural cost is that the fixes accumulated without being unified.

**Cost to fix**: medium. The shim would be one new module (~80 lines) and would *delete* about 60 lines from `_llm_dispatch.py`. Net code: smaller; cyclomatic complexity per function: dramatically lower. Risk: each branch encodes a real provider quirk and merging them needs careful test coverage — fortunately the symptoms are well-documented in tests.

**Verdict**: The retry mechanism is ESSENTIAL. The four-branch dispatch is INCIDENTAL — accreted; deserves a dedicated compat shim above LangChain.

---

## §5 Five test fakes (`tests/fakes.py`)

**File**: `tests/fakes.py:1-604`.

**Essential complexity**: There are genuinely five LLM contracts that exist in the codebase (and must therefore be faked separately):
- **StructuredFake** — `with_structured_output().invoke()` returns a parsed model
- **StructuredFakeWithRaw** — same but honors `include_raw=True`, returns `{"parsed", "raw"}`
- **ReActFake** — `bind_tools().invoke()` returns `AIMessage` with `tool_calls`, then `with_structured_output()` for final parse
- **TextFake** — `invoke()` returns plain `AIMessage(content=...)` for json_mode/text strategies
- **StringArgsFake**, **GuardFake**, **StubbornFake** — narrow fakes for specific edge cases (tool-args coercion, guard fires, runaway loops)

Each fake corresponds to a distinct *production* contract the framework needs to handle. Collapsing them into one branching fake would just relocate the polymorphism from class hierarchy to method bodies; Brooks's principle says that is moving complexity, not removing it.

**Incidental complexity present**: **INCIDENTAL — local**. Three small items:

1. The `register_scripted/condition/tool_factory` test helpers at lines 45-87 are the test-side mirror of the killed module-level src/ registry functions. They exist because nine test files still call `register_scripted("name", fn)` and the autouse fixture clears between tests. This is correct migration debt, not accidental complexity. The shape is right; the helpers should stay until the test bodies migrate to pass `scripted={...}` directly to `compile()`.
2. The deprecated `configure_fake_llm` at line 559 is a clear migration alias; documentation already names it as such.
3. `lookup_scripted/lookup_condition` (lines 90-117) reach into `_decorator_scripted` / `_decorator_conditions` module globals. This is leakage from the same root cause as §3's decorator-side globals — if those become sidecar-attached, this leakage disappears.

**Verdict**: ESSENTIAL — five contracts, five fakes is correct. Cleanup is bounded and proportional to the §3 decorator-side-globals fix.

---

## §6 Test observability contract

**File**: `tests/test_observability_contract.py:1-571`.

**Essential complexity**: Operators monitoring on log event names (`react_max_iterations_exceeded`, `react_token_budget_exceeded`, `trailing_tool_call_markup`, `react_guard_forced_break`, `auto_resume_schema_change`) need a public, breaking-change-safe contract. The contract test exists so any rename is a deliberate edit visible in a single file, not a silent breakage. This is the right shape.

**Incidental complexity present**: Borderline. The test file is structured per-event-name with two tests each (`test_event_name_emitted_when_*` + `test_event_payload_includes_*`). The duplication of setup (lines 47-73, 82-108, 127-153, 162-188 ...) is INCIDENTAL — local: each test builds a `FakeTool`, registers it, builds `GuardFake`, sets up tools/tracker/config nearly identically. A small helper would dedupe ~80 lines without moving the per-event assertions. Cost to fix: trivial.

**Verdict**: ESSENTIAL contract, INCIDENTAL — local duplication. Don't touch unless rewriting other tests; the cost is real but the duplication makes each test individually readable.

---

## §7 StateBus protocol + two adapters + `get_required`

**File**: `src/neograph/_state_bus.py:1-88`.

**Essential complexity**: The compiled graph passes a Pydantic `BaseModel` state to most node wrappers, but subgraph dispatch and `Node.run_isolated` pass a `dict`. The downstream helpers that read from state used to take `state: Any` and branch internally (`isinstance(state, dict)` everywhere); this is replaced by adapt-at-the-edge + uniform `StateBus` Protocol — the textbook fix for parametric polymorphism. The two concrete adapters are essentially identical in shape, which is the desired outcome: the polymorphism is resolved at one place and never returns.

The `get_required(key, *, node_label=None)` distinction (essential vs optional reads, per architecture-decisions §7) catches a real class of bugs: silent-None on a framework-required field used to surface confusingly downstream. The distinction is essential domain logic.

**Incidental complexity present**: Two minor items, both INCIDENTAL — local:

1. The `_MISSING: Any = object()` sentinel on line 21 — used only in `_ModelStateBus.get_required` — exists to distinguish "attribute absent" from "attribute is None." This is essential. The `Any` annotation is the leak; a tighter type (`_MissingType = NewType('_MissingType', object)`) would be more honest, but the cost-benefit is poor.
2. `__slots__` on both adapters (lines 44, 63) is essential micro-optimization given they're created per state-read site. Fine as-is.

**Verdict**: ESSENTIAL. This is the cleanest module in the inventory — exactly the pattern a senior reviewer would propose for adapt-at-the-edge.

---

## §8 Polymorphic `Node.outputs` / `Node.inputs` + `normalize_*`

**Files**: `src/neograph/_normalize.py:1-104`, `src/neograph/node.py:148-149`, `src/neograph/state.py:321-435`, `src/neograph/factory.py` (consumers).

**Essential complexity**: User-facing polymorphism is essential for the §8 ergonomic contract: 90% of users want `outputs=Claims` (single type); the dict form `outputs={"result": Claims, "tool_log": list[ToolInteraction]}` exists for the genuine multi-output case (agent tool log, structured + raw token usage). Forcing every user to write `outputs={"result": Claims}` would be the dict-replacing-types failure mode the maintainer explicitly rejected.

`NormalizedOutputs` / `NormalizedInputs` are the right abstraction: discriminate once at the edge, downstream callers consume a tagged record (`primary`, `primary_key`, `secondary`, `all_keys`, `is_dict_form`, `is_none`). This is the same pattern as §7 — adapt-at-the-edge with a Protocol/dataclass on the inside.

**Incidental complexity present**: Two items.

1. **INCIDENTAL — local**: `_validate_type_spec` at `node.py:77-99` re-runs the polymorphism discrimination as a Pydantic validator. This is necessary because `Any` would let an `int` through. Pydantic v2's `Annotated[Any, PlainValidator(...)]` is the idiomatic shape (line 116). No fix needed; this is the price of polymorphism at a Pydantic field boundary in the absence of PEP 747 `TypeForm`.

2. **INCIDENTAL — architectural** at the producer side: `effective_producer_type` (`_construct_validation.py:58-87`) and `_add_output_field` (`state.py:321-378`) are *two* implementations of "what does this node write to the state bus, accounting for modifiers." The validator path returns a TYPE; the state-builder path REGISTERS A FIELD. The rule encoded in each is identical: Each → `dict[str, X]`, Oracle → `(collector_field, X)`, dict-form-outputs → per-key fields. The two paths must stay in lockstep — the comment at `_construct_validation.py:70-72` explicitly cross-references the state.py side. This is structurally duplicate logic that can drift.

   The architecturally correct fix: introduce `ProducerEffect` — a small value object that captures "what fields to register" AND "what type each is" in one returned record. Both `effective_producer_type` and `_add_output_field` consume it. New modifiers add one rule to one function. Cost to fix: medium. Benefit: eliminates a known drift surface (the prior `neograph-8k3` / `neograph-ayq` bugs were exactly this drift).

**Verdict**: The user-facing polymorphism is ESSENTIAL. The normalizer pattern is correct. The producer-side duplication between validator and state-builder is INCIDENTAL — architectural; fixable by extracting a `ProducerEffect` record.

---

## §9 Two-level fingerprinting + auto-resume rewind

**Files**: `src/neograph/state.py:264-318` (fingerprint compute), `src/neograph/runner.py:77-249` (verify + rewind + transitive closure).

**Essential complexity**: Schema-aware checkpoint resume is a real domain capability and the two-level decomposition is essential:
- **Schema fingerprint** (compute_schema_fingerprint, sha256 of non-framework field annotations) — answers "did anything change?"
- **Per-node fingerprints** (compute_node_fingerprints, sha256 per state-field) — answers "specifically what?"
- **Transitive closure** (runner.py:239-249) — turns "specifically what" into "what to re-execute."

Each piece pulls its weight. Removing any one collapses the capability ("auto-rewind to the right superstep, re-execute only what's invalidated"). This is the most genuinely sophisticated piece of essential complexity in the codebase.

**Incidental complexity present**: One item, **INCIDENTAL — local**:

- `_build_producer_consumer_adjacency` at `runner.py:198-236` walks the construct AT RESUME TIME to build the adjacency. The same adjacency is already known at compile time (the validator walked it). Walking twice is wasteful and creates a place where the two adjacencies could disagree (different rules for `Each.over` parsing, different handling of `context=`). Cost to fix: low — compute the adjacency once at compile time and stash it on `compiled._neo_adjacency`. Then `_compute_invalidated_nodes` reads the pre-built adjacency instead of rebuilding it.

**Verdict**: ESSENTIAL capability. Adjacency duplication is INCIDENTAL — local.

---

## Cross-cutting: three API surfaces, three assembly paths

**Files**: `src/neograph/_construct_builder.py:1-704` (@node path), `src/neograph/loader.py:1-275` (YAML path), `src/neograph/construct.py:114-219` (declarative path).

**Essential complexity**: Three API surfaces (declarative `Node(...)`, `@node` decorator, YAML `load_spec`) are essential. The maintainer documented why: declarative is for runtime construction by LLMs/config systems; @node is the human-facing default; YAML is for project specs. The three coexist by design and produce identical IR.

**Incidental complexity present**: **INCIDENTAL — architectural, and significant**. The three paths populate the IR *differently*. The @node path eagerly sets `fan_out_param` and `oracle_gen_type` in `_cleanup_inputs_and_register` (`_construct_builder.py:540-587`). The YAML and declarative paths defer these to `Construct.__init__` via `_normalize_fan_out_params` and `_normalize_oracle_gen_type` (`construct.py:163-218`). The two normalizer methods on Construct exist *only* to catch up to what `_build_construct_from_decorated` did already.

This is the upstream cause of:
- The `Construct._normalize_*` methods, which will grow proportionally to new @node-inferable IR fields.
- The `TestConstructNormalizesEveryAtNodeOnlyIRField` structural guard (test_structural_guards.py:4168) that enforces parity.
- The `neograph-ts7` bug class (validator works for one path, breaks for another). Three-surface parity is currently a *test invariant*, not a structural one.

**Upstream cause**: `_build_construct_from_decorated` does too much. It does name resolution, port identification, DI classification, fan-out detection, constant classification, collision checks, adjacency, topo sort, AND IR finalization (fan_out_param, oracle_gen_type, scripted shim). Of those, only the first eight are @node-specific (the @node path needs to derive structure from Python signatures). The last three are IR-level normalizations that every surface needs.

**Architecturally correct fix**: extract a `_normalize_ir(construct)` function that runs the @node-only IR finalizations (`fan_out_param`, `oracle_gen_type`, future fields) on ANY Construct, regardless of how it was built. `Construct.__init__` calls it. `_build_construct_from_decorated` calls it. The YAML loader calls it. The two `_normalize_*` methods on Construct disappear, the future ones never get written, and the structural guard becomes "is this function called from __init__" instead of "do all individual normalizations exist."

**Cost to fix**: medium. The normalization functions already exist; the change is moving them to one module and having every assembly path delegate.

**Why this matters**: this is the single biggest source of incidental complexity in the inventory, by leverage. The fix prevents an entire class of future bugs (any new @node-inferable field would otherwise need its own `_normalize_*` method, its own structural guard, its own three-surface test). The cure is upstream of the symptoms — and the symptoms are scattered across `construct.py`, `_construct_builder.py`, `test_structural_guards.py`, and `loader.py`.

---

## Cross-cutting: three closure factories in `_oracle.py` with `node_name` threading

**File**: `src/neograph/_oracle.py:49-122, 331-360`.

**Essential complexity**: Three factories — `make_oracle_redirect_fn`, `make_eachoracle_redirect_fn`, `make_each_redirect_fn` — produce three different closures that LangGraph dispatches into. Each closure has a distinct contract:
- Oracle redirect: write to collector field, not consumer field.
- Each×Oracle redirect: tag results with the Each key.
- Each redirect: dict-key the result by the Each item's key field.

Three closures are essential — they encode three distinct fan-out/merge topologies. Collapsing them into one branching function would substitute control-flow complexity for structural complexity (the rejected anti-pattern).

**Incidental complexity present**: **INCIDENTAL — architectural** in the `node_name=` keyword threading. The `node_name` parameter exists because LangGraph node identity uses `__name__`, and the redirect closures override `__name__` to `raw_fn.__name__` (lines 80, 120, 359), which loses the user's declared node name. The three factories then accept `node_name=` as a keyword-only parameter solely to surface in `StateMissingError` messages. `make_oracle_redirect_fn` doesn't even *use* `node_name` yet — line 68 reads `_ = node_name  # reserved for future get_required calls`, a placeholder argument to enforce future-proof callers.

**Upstream cause**: `__name__` is doing double duty. It is both:
- LangGraph's routing key (how add_node finds the function), and
- The error-message label (what the user sees when something goes wrong).

These two roles have different requirements (routing key needs to be unique per closure to avoid clobbering; label needs to be stable per the user's declared identity). Conflating them forces every site that wraps a node function to also thread the user-facing label as a separate parameter.

**Architecturally correct fix**: give `Node` a separate `_display_label: str` (or rely on the existing `node.name`) and stop overriding `__name__` on closures. The closure's `__name__` becomes a synthesized unique routing key (`_oracle_redirect_{field}_{collector}`), and error messages explicitly carry the user's `node.name`. The `node_name=` kwarg can then disappear from the factory signatures. The `_ = node_name` placeholder at line 68 disappears. The structural guard policing future drift becomes unnecessary.

**Cost to fix**: low-medium. Touches all the `make_*_fn` factories in `_oracle.py` and the call sites in `compiler.py`. About 30 lines of net change.

**Verdict**: Three closures are ESSENTIAL. The `node_name=` threading is INCIDENTAL — architectural, downstream of the `__name__` overload.

---

## Cross-cutting: `_normalize_fan_out_params` + `_normalize_oracle_gen_type` (and the future ones that will follow)

Covered in detail under the assembly-paths item above. This is the symptom; the upstream cause is the assembly-path divergence between `_build_construct_from_decorated` and `Construct.__init__`. Fix the upstream; these two methods (and any future ones) disappear.

---

## Cross-cutting: validator vs runtime fan-out detection (recently deduplicated)

**File**: `src/neograph/_construct_validation.py:611-619` (validator fan-out detection), `test_structural_guards.py:4129` (`TestNoRuntimeFanOutDetection`).

**Essential complexity**: Validator must know which keys are fan-out receivers vs upstream producers; runtime must extract the fan-out item into `input_data`. These are two different uses of the same information.

**Incidental complexity present, now fixed**: The duplication has been removed this session — only the validator detects fan-out, runtime reads `node.fan_out_param`. The structural guard `TestNoRuntimeFanOutDetection` is the new positive contract. This was INCIDENTAL — architectural, and the dedup was the right fix. The guard test will prevent re-divergence.

**Verdict**: Was INCIDENTAL — architectural. Now resolved. The current `fan_out_param` IR field + validator-and-runtime read pattern is the correct shape.

---

## Cross-cutting: recursive `_validate_node_chain(ambient_producers=...)` with deferred context checks

**File**: `src/neograph/_construct_validation.py:253-486`.

**Essential complexity**: Sub-constructs nest. A node inside a sub-construct may reference `context=` for a field produced by the parent. The parent's producer set is unknown when the sub-construct self-validates (because it's validated during its own `__init__`, before the parent embeds it). Two options:
- Validate context at sub-construct creation time → must know parent producers → impossible.
- Defer context checks; validate them when the parent re-walks → essential.

The `ambient_producers=` parameter encodes this; the `context_checkable = (ambient_producers is not None or construct.input is None)` check at line 313 captures the rule precisely.

**Incidental complexity present**: None. This is essential domain logic with no obvious simplification. The deferred-validation pattern matches Pydantic's own approach to forward refs (resolved later when more context is available). The single recursive walker is the right shape.

**Verdict**: ESSENTIAL. This is the kind of code Brooks called inherent to the problem — language-of-the-domain logic for a real domain constraint.

---

## Cross-cutting: `Node._sidecar` PrivateAttr storing user Callable

**File**: `src/neograph/node.py:202-204`, `src/neograph/_sidecar.py:1-143`.

**Essential complexity**: The @node decorator needs to attach IR-level metadata (the original function, parameter names, DI bindings) to a `Node` instance. Pydantic schema validation would reject a `Callable` field without `arbitrary_types_allowed`. `PrivateAttr` is the idiomatic Pydantic v2 pattern for this and is what FastAPI uses for `Depends(...)` storage on routes.

The maintainer's comment ("Why we keep the sidecar rather than eagerly resolving") in CLAUDE.md is exactly right — separating "what the node is" from "how this runtime invokes it" enables the TypeScript port (LangGraphJS via a different runtime backend) without changing the IR.

**Incidental complexity present**: None of significance. The `_sidecar.py` extraction (line 143) exists to break the `decorators.py ↔ _construct_builder.py` import cycle — that's structural hygiene, not accreted defense. The function-local import at `_sidecar.py:101-104` is the only minor smell; it's there because `infer_oracle_gen_type` is called both from `_construct_builder.py` (which already imported decorators) and from `construct.py:_normalize_oracle_gen_type` (which would create a cycle). The cycle is itself a symptom of the assembly-path divergence above; once normalize is consolidated, this function-local import disappears.

**Verdict**: ESSENTIAL. The sidecar pattern is correct. The one function-local import is downstream of the assembly-path issue.

---

## Cross-cutting: structural guards proliferating (60 classes, 4268 lines)

**File**: `tests/test_structural_guards.py` — 60 classes, 201 test methods.

**Essential complexity**: Each guard encodes an invariant the type system cannot express. Examples:
- "No `isinstance(_, dict)` on `.outputs` outside `_normalize.py`" — Python has no concept of "this discrimination should live in exactly one module."
- "No private `langgraph._*` imports" — Python has no concept of "this module's underscore-prefix is meaningful to me."
- "All `NeographError.build()`, never raw `NeographError(...)`" — Python has no concept of "this constructor is private."
- "@node-only IR fields must all be normalized in `Construct.__init__`" — would be unnecessary if the assembly paths converged (see above).

For each invariant, there is no language feature that would let us state the rule once and have the compiler enforce it. The guard test IS the type system. This is a budgeted, deliberate price for invariants the language can't express.

**Incidental complexity present**: **INCIDENTAL — local in aggregate, but each guard is essential**. The guard layer's size is itself diagnostic, not pathological:
- ~15 guards exist *because* of the assembly-path divergence — they would disappear if the assembly paths consolidated.
- ~5 guards exist *because* of the `__name__` double-duty — they would disappear if the routing key and label were split.
- ~5 guards exist *because* of the four-branch `_call_structured` ladder — they would disappear if the LangChain compat shim consolidated.
- ~5 guards exist because of historical decorator-side global registries that haven't been migrated to sidecars yet.
- The remaining ~30 are genuinely essential — they police real architectural invariants (layering DAG, module responsibilities, error discipline, type-boundary purity) that the language cannot express.

So the guard count is itself a leading indicator. If the three root incidental complexities upstream are fixed, the guard count would drop by ~30 without losing any architectural invariant.

**Verdict**: ESSENTIAL in aggregate, but ~50% of guards are *symptomatic* — they exist because the type system cannot express invariants that should be structural. Each fix above eliminates the corresponding guards.

---

## Cross-cutting: `__name__` doing double duty

Covered under "three closure factories" above. The upstream choice was conflating LangGraph routing keys with user-facing labels. Splitting them eliminates the downstream complexity in `_oracle.py`, the kg8l audit comment, and one structural guard.

---

## The incidental-complexity tree

Three root upstream choices generate the bulk of the downstream incidental complexity:

### Root 1 (highest leverage): Assembly-path divergence

`_build_construct_from_decorated` does both @node-specific structural inference (parsing function signatures, resolving DI, building adjacency) AND IR-level normalization (setting `fan_out_param`, `oracle_gen_type`, `scripted_fn`). The other two assembly paths (YAML, declarative) have to retroactively replay the IR-level normalization via `_normalize_*` methods on `Construct`.

**Eliminates**:
- `Construct._normalize_fan_out_params` + `_normalize_oracle_gen_type` (and every future `_normalize_*`)
- Up to 15 structural guards
- The `infer_oracle_gen_type` function-local import in `_sidecar.py`
- One class of three-surface-parity bugs (`neograph-ts7` and similar)

**Cure**: Extract `neograph._ir_normalize.normalize_ir(construct)` containing all IR-level inferences. Have every assembly path (declarative `Construct.__init__`, `_build_construct_from_decorated`, YAML `_build_construct`) delegate to it once before validation.

### Root 2 (medium leverage): `__name__` routing-key vs label collision

LangGraph identifies nodes by callable `__name__`. Wrapper closures override `__name__` to the wrapped function's name; this clobbers the user-declared `node.name` for error messages. Every wrapper factory then needs a `node_name=` keyword to thread the label.

**Eliminates**:
- `node_name=` kwarg from all three factories in `_oracle.py`
- The `_ = node_name` placeholder at `_oracle.py:68`
- A class of structural guards policing the threading
- Some confusion in error messages where the user sees synthesized routing keys instead of their declared name

**Cure**: Stop overriding `__name__` on wrapper closures. Let them have unique synthesized routing keys (`_oracle_redirect_{field}_{generator_idx}`). Use `node.name` (already stored on the Node instance) wherever the user-facing label is needed.

### Root 3 (smaller but accreted): LangChain `with_structured_output` compat ladder

Each branch in `_call_structured` was added in response to a specific provider failure, without unification. The pattern is recoverable through a compat shim that normalizes the LangChain return shape.

**Eliminates**:
- ~60 lines of branching in `_llm_dispatch.py`
- Cognitive load per change: any new provider quirk gets *added at the shim*, not threaded through the four branches
- One source of "DSML detection re-runs in two places" symptom

**Cure**: New module `_llm_structured_compat.py` that normalizes `with_structured_output(model, include_raw=True)` into a single `StructuredResult = Parsed(model) | Raw(text) | Failed(error)` and absorbs the TypeError-on-include_raw retry. `_call_structured` becomes a switch on the result.

---

## Recommendations

Prioritized by ratio of *downstream incidental complexity eliminated* to *upstream change cost*. Each recommendation respects the constraint: no dict-over-types, no branching-over-polymorphism.

### Priority 1 — Extract `neograph._ir_normalize` (Root 1)

**Cost**: medium (one new module, four call-site changes).
**Benefit**: highest leverage. Eliminates the `_normalize_*` methods on `Construct`, prevents future ones, kills ~15 structural guards, breaks one function-local import cycle, and eliminates an entire bug class.
**Risk**: low — the normalizations already exist as pure functions; the change is choosing one site to run them and removing duplicates.

Best-practice reference: this is the same pattern Hypothesis uses for strategy normalization (`hypothesis.internal.conjecture.utils.calc_label_from_name` lives in one place, every strategy consumes it). Mojo's `@parameter` decorator follows the same architecture — IR-level finalization lives in one normalize pass after parsing.

### Priority 2 — Split LangGraph routing key from display label (Root 2)

**Cost**: low-medium (~30 lines net, mostly in `_oracle.py`).
**Benefit**: eliminates a class of "node_name threading" that will otherwise return every time a new wrapper factory is added.
**Risk**: low — the routing key is internal; users never see it. The display label already exists as `node.name`.

Best-practice reference: FastAPI's `APIRoute.endpoint.__name__` is the unique internal identifier; `APIRoute.name` is the user-facing route name. They are explicitly separate fields.

### Priority 3 — `ProducerEffect` record for state-builder + validator parity (§8)

**Cost**: medium (one new dataclass, two call-site refactors).
**Benefit**: eliminates the structural duplication between `effective_producer_type` and `_add_output_field`. New modifiers add one rule to one place.
**Risk**: low — both paths produce the same effects today; capturing them in a value object is conservative.

### Priority 4 — `_llm_structured_compat` shim (Root 3)

**Cost**: medium (~80 LOC new, ~60 LOC deleted, careful test coverage).
**Benefit**: turns the four-branch `_call_structured` into a one-case-per-result-variant switch. Each new provider quirk is added at the shim, not threaded through the dispatch.
**Risk**: medium — each branch encodes a real production fix (neograph-0tid); the unification needs to preserve every case. Mitigation: the existing tests are written against the symptoms (DSML in raw, TypeError on include_raw, etc.) and will catch regressions.

Best-practice reference: LangChain itself does NOT provide this shim, but Pydantic AI does — `pydantic_ai._adapters` wraps multiple model providers behind a single result shape. Same pattern.

### Priority 5 — Migrate decorator-side globals to sidecar storage (§3 partial)

**Cost**: medium (touches `decorators.py`, `compiler.py`, `tests/fakes.py`).
**Benefit**: eliminates the last remaining module-level mutable state. Brings the @node decorator into full compliance with the §2 per-compile isolation guarantee.
**Risk**: low-medium — the migration is mechanical. The test-side `register_*` helpers stay as-is during migration and shrink to nothing once all test sites migrate to passing `scripted={...}` to `compile()`.

### Priority 6 — Adjacency stash on compiled graph (§9)

**Cost**: low (one new field, one walker move).
**Benefit**: avoids walking the construct twice and eliminates one drift surface.
**Risk**: low.

### Not recommended

- **Do not** collapse the five test fakes into one branching fake. They encode five real LLM contracts; collapsing them moves complexity, not removes it.
- **Do not** collapse the three closure factories in `_oracle.py` into one. Each encodes a distinct topology; collapsing them substitutes branching for structure.
- **Do not** remove the `_validate_node_chain` recursive `ambient_producers=` pattern. The deferred validation is essential domain logic; the recursion is the cleanest possible shape.
- **Do not** simplify `Node.outputs` polymorphism away. The single-type form is a genuine ergonomic win and the dict form is a genuine multi-output capability. The normalizer pattern is the correct abstraction.
- **Do not** remove fingerprinting or auto-resume. The two-level decomposition is essential; the slight overlap with adjacency reconstruction is a minor local issue (Priority 6).

---

## External references that informed this view

- **FastAPI dependency-injection internals** — confirms the `Annotated[T, Marker]` + frozen-bundle + closure-capture pattern for runtime configuration; same as `LlmRuntime`.
- **Pydantic v2 validator pipeline** — confirms `PrivateAttr` + Pydantic-bypassed Callable storage is idiomatic for the `_sidecar` pattern.
- **Hypothesis strategy system** — confirms single-normalization-pass architecture for IR finalization; same as the proposed `_ir_normalize`.
- **LangChain `with_structured_output`** (read for §4) — confirms the four-branch ladder is walking real provider polymorphism, not over-engineered defense; confirms the compat-shim fix.
- **Pydantic AI `_adapters`** — confirms the compat-shim pattern works in production for the same multi-provider unification problem.
- **Brooks, "No Silver Bullet"** — the lens itself: essential complexity is inherent to the problem; the goal is to eliminate accidental complexity without conjuring more in its place.
