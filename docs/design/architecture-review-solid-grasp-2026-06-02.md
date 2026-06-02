# SOLID/GRASP Architecture Review — 2026-06-02

Lens: SOLID (SRP, OCP, LSP, ISP, DIP) and GRASP (Information Expert, Creator, Controller, Low Coupling, High Cohesion, Polymorphism, Pure Fabrication, Indirection, Protected Variations). Every item below is rated EXEMPLARY or PROBLEMATIC against those principles only.

Scope: §1–§9 from `docs/design/architecture-decisions.md` and the cross-cutting findings called out in the prompt.

---

## Summary

15 items reviewed. Verdict counts:

- **EXEMPLARY (5)**: §7 StateBus, §8 normalizers, §5 fake refactor, the `effective_producer_type` helper extracted out of `_construct_validation.py`, the `_normalize.py` polymorphism boundary.
- **MOSTLY-EXEMPLARY with a defect (3)**: §1 LangGraph commitment, §2 LlmRuntime, §3 per-compile registries.
- **PROBLEMATIC (7)**: §4 retry stack, §9 fingerprinting, three-API-surface assembly path, three Oracle/Each closure factories, `Construct._normalize_*` runtime drift compensation, `Node._sidecar` PrivateAttr coupling, recursive `_validate_node_chain` with `ambient_producers`.

The dominant cross-cutting violation is **DIP/Protected Variations bypassed by `getattr` and `isinstance` probes** in the dispatch and validation paths. The polymorphic-output normalizer (§8) is the model the rest of the codebase has not yet adopted. The recent fixes ("hydra" symptom the maintainer is suspicious of) are nearly all in modules that still discriminate on shape at the call site rather than behind a Protocol or normalizer — every new combination forces a new branch, and missing the same branch in a sibling site is the actual root cause.

---

## §1 LangGraph commitment / llm_factory

**Principles at stake**: Dependency Inversion, Protected Variations, Information Expert, Pure Fabrication.

**Verdict**: EXEMPLARY with one wart, severity LOW.

**Evidence**:
- `src/neograph/_llm_runtime.py:46-87` — `LlmRuntime` is a frozen dataclass that aggregates the four cross-cutting LLM dependencies (`llm_factory`, `prompt_compiler`, `renderer`, `cost_callback`) and is threaded into every closure via `runtime=` kwargs. This is a textbook Pure Fabrication: there is no domain object called "LlmRuntime," but the framework needs one cohesive carrier to avoid four-argument signatures everywhere.
- `src/neograph/compiler.py:100-113` — explicit, single source of runtime resolution. No ambient/global lookup.
- `docs/design/architecture-decisions.md:9-15` — the framing "LangGraph commitment, provider-neutrality at model layer" correctly separates two concerns that were previously conflated in the now-deleted "backend abstraction" design. The repo follows through: `langgraph.*` private imports are forbidden by structural guard, and `BaseChatModel` is the swap point.

**Why this is good architecturally**: high-level modules (compiler, factory, dispatch) depend on the `LlmRuntime` abstraction; LangChain/LangGraph concretions are only touched where they actually do work (`_llm.invoke_structured`, `_tool_loop`, `compiler.StateGraph`). DIP honored. The decision to *stop* pretending the backend was swappable is itself an instance of Protected Variations applied honestly — it acknowledges the variations the framework actually protects (LLM provider) and refuses to protect ones it doesn't (graph runtime).

**The wart**: `compile()` accepts `_runtime` and `_scripted_lookup` as private positional-style kwargs (compiler.py:69-70) so sub-construct compilation can thread parent state. This is a minor ISP smell — the public `compile()` signature now leaks an implementation argument used only by the recursive `_add_subgraph` call. Cleaner would be a separate internal `_compile_sub(...)` entry point that doesn't pretend to be the public API. Not severe; the convention is documented.

**Web-search informed**: the "provider-neutral abstraction at the model layer, framework commitment everywhere else" pattern is the same shape LangChain itself uses (BaseChatModel as the seam, no abstraction at the runnable/chain layer). The decision is consistent with established LangChain ecosystem norms.

---

## §2 LlmRuntime (frozen dataclass aggregator)

**Principles at stake**: Information Expert, Low Coupling, High Cohesion, Single Responsibility, Pure Fabrication.

**Verdict**: EXEMPLARY, severity none.

**Evidence**:
- `src/neograph/_llm_runtime.py:46-87` — `LlmRuntime` carries exactly four cross-cutting LLM dependencies plus precomputed signature introspection. The `frozen=True` enforces immutability so closures cannot mutate a shared bundle (the multi-tenant guarantee documented at §2).
- `src/neograph/_llm_runtime.py:96-140` — `check_llm_kwargs_or_raise` is a fail-loud Information Expert: it walks the construct (the data it owns the question about) and answers "are LLM kwargs missing?" with the offending node names. Shared by `compile()`, `lint()`, and `Node.run_isolated()` — three callers, one helper, single responsibility honored.
- `src/neograph/_llm_runtime.py:29-43` — `_accepted_params` precomputes once per compile so per-invocation closures avoid the `inspect.signature` cost. Information Expert: the runtime is the only place that knows what its callables accept.
- `src/neograph/_llm_runtime.py:93` — `EMPTY_RUNTIME` is a shared frozen sentinel, not a `None`. Good Liskov citizenship: every consumer can call `.llm_factory_params` on it without a None-check.

**Why this is good**: the alternative (passing 4 separate args through every closure or stuffing them into a global) would either explode signatures or violate DIP via ambient state. The dataclass is the smallest possible Pure Fabrication that solves both. The signature-introspection-once optimization shows the right Information Expert split: `LlmRuntime.build` knows what to inspect; consumers don't.

The only thing I'd watch for in the future: `Any`-typed fields (`llm_factory: LlmFactory | None`) defer typing to TYPE_CHECKING. That's fine for now, but if `LlmFactory` ever grows methods the framework calls directly, promote it from a callable type alias to a Protocol so DIP at the model boundary is structural rather than nominal.

---

## §3 Per-compile registries (`scripted`, `conditions`, `tool_factories`)

**Principles at stake**: Dependency Inversion, Protected Variations, Single Responsibility, Low Coupling.

**Verdict**: EXEMPLARY in intent, MOSTLY-EXEMPLARY in execution, severity LOW.

**Evidence**:
- `src/neograph/compiler.py:120-140` — `compile()` builds three per-compile dicts (`scripted_lookup`, `condition_lookup`, `tool_factory_lookup`) and threads them through every closure factory call. Two `compile()` calls produce two independent dicts, as §2 requires.
- `src/neograph/compiler.py:127-130` — there is a residual module-level `_decorator_scripted` dict (and `_decorator_conditions`, `_decorator_tool_factories`) seeded at decoration time from `decorators.py:86-88`. `compile()` *copies* these into the per-compile dict, so the per-compile dict is the only thing closures see — but the *source of truth* for decoration-time shims is still a module-global.

**Why this is mostly good**: the per-compile dict is the right answer to DIP at the registration boundary — closures depend on a dict-shaped abstraction injected at compile time, not on a process global. The "two `compile()` calls = two independent runtimes" guarantee survives, which is what multi-tenant servers actually need.

**Why the residual decoration-time dict is a LOW-severity violation**: `_decorator_scripted` at `decorators.py:86` is a true process global. The `@node`/`@merge_fn`/`@tool` decorators execute at import time and write into it; tests need cleanup fixtures (`tests/conftest.py`'s registry cleanup) because of this. The strict reading of DIP says decorator-side registration should also produce per-module objects, with `compile()` discovering them via explicit `nodes=[...]` rather than via a shared mutable. The current design accepts the global because Python decorators run at import and there's no clean place to stage the result otherwise — but it's an accepted tradeoff, not an absence of violation. Severity is LOW only because the per-compile copy step makes the global invisible to runtime code.

**Architecturally correct alternative**: Make `@node`/`@merge_fn` attach the shim to the returned object (already happens for `@node` via `_scripted_shim` PrivateAttr per sidecar pattern) and have `construct_from_module` walk module attrs once. That is mostly what already happens — `@merge_fn` is the holdout still using a name-keyed module dict (`_sidecar.py:69-70`). Migrating `@merge_fn` to the sidecar pattern would let the three module-level dicts shrink.

---

## §4 `_llm_retry.py` + DSML recovery + LangChain `include_raw=True` compat ladder

**Principles at stake**: Single Responsibility, Open/Closed, Liskov Substitution, Low Coupling, Polymorphism.

**Verdict**: PROBLEMATIC, severity HIGH.

**Evidence**:
- `src/neograph/_llm_dispatch.py:24-119` — the `_call_structured` function is doing four things at once: dispatch by strategy string, attempt `include_raw=True`, fall back to `include_raw=False` on `TypeError`, fall back to raw `llm.invoke` if both error, run DSML detection on each of the three text sources. Cyclomatic complexity is high; the comments at lines 75-93 are themselves diagnostic that the author knew the dispatch ladder was a workaround for provider-rejected kwargs.
- `src/neograph/_llm_retry.py:310-361` — `_attempt_dsml_recovery` is a content-based recovery path that can itself fail and then escalate into the generic retry path (`_invoke_json_with_retry`, line 358). So there are now *two* retry loops nested: one inside `_attempt_dsml_recovery` (one targeted retry), one inside `_invoke_json_with_retry` (N error-feedback retries), and the dispatch site can also retry via the `include_raw` compat ladder.
- `src/neograph/_llm_retry.py:264-307` — `_invoke_json_with_retry` itself does five things: invoke the LLM, accumulate usage, call `_parse_json_response`, on failure build a retry message with schema, re-invoke. Each retry path duplicates the usage-merging arithmetic at lines 285-289 and 302-306.
- `docs/design/architecture-decisions.md:46-58` — the §3 decision is explicit that retry concerns live "in exactly one layer" with a three-row table. The actual code violates this: DSML recovery is a fourth retry kind (output-format recovery for provider quirks) that wasn't enumerated in the decision, and it lives orthogonally to output-quality retry — but the two interact in `_attempt_dsml_recovery` line 357-361 where targeted retry escalates to generic retry.

**Why this is a SOLID violation**:
- **SRP violated**: `_call_structured` has three change axes (new strategy, new provider compat shim, new content-recovery heuristic) and any of the three forces edits to the same function.
- **OCP violated**: adding a new structured-output strategy means editing the `if strategy == ...` ladder. Adding a new content-recovery heuristic (e.g., for a different provider's malformation pattern) means adding another branch inside the `try/except TypeError` chain.
- **DIP violated**: the dispatch path knows the names of specific provider quirks (DSML, `include_raw=True` rejection). High-level retry logic depends on low-level provider behavior shapes, not on abstractions.
- **LSP violated**: `with_structured_output` is supposed to be a polymorphic LangChain interface — but `_call_structured` knows that some implementations reject `include_raw=True` (line 74) and have to be retried without it. The polymorphic interface isn't actually polymorphic; the framework is paying the cost of a leaky abstraction.

**Architecturally correct alternative**: extract a `StructuredOutputAdapter` Protocol with one method (`invoke_for_model(messages, model, config) -> (parsed, raw, usage)`). Implementations: `StructuredOutputAdapter`, `JsonModeAdapter`, `TextAdapter`. Provider-quirk shims (the `include_raw` ladder, the DSML detector) become decorators applied around adapters at runtime construction time. The retry loop becomes a single generic `with_retry(adapter, max_retries, on_failure: Callable[[ParseError], str])` wrapper. Each of the four retry concerns lives in its own composable layer rather than tangling inside `_call_structured`'s nested try/excepts. The §3 design intent (each concern in one layer) is preserved by *structure*, not by documentation alone.

The maintainer's "hydra" suspicion is most justified here. Every new provider quirk adds another `try/except TypeError` arm, and the recent DSML-recovery fix had to be wired into three code paths (structured-silent, structured-TypeError, tool-loop final) — that's three branches that must stay aligned because the recovery logic is in helper functions but the dispatch is open-coded.

---

## §5 Test fakes

**Principles at stake**: Single Responsibility, Interface Segregation, Liskov Substitution, Dependency Inversion.

**Verdict**: EXEMPLARY, severity none.

**Evidence**:
- `tests/fakes.py:125-147` (StructuredFake), `tests/fakes.py:150-195` (StructuredFakeWithRaw), `tests/fakes.py:203-378` (ReActFake), `tests/fakes.py:379-` (TextFake) — four shape-specific fakes, each implementing only the LangChain method surface its scenario exercises. ISP honored: `StructuredFake` doesn't pretend to support `bind_tools`; `ReActFake` doesn't pretend to support `with_structured_output(include_raw=True)`.
- `tests/fakes.py:525-553` — `build_fake_llm_kwargs` returns a dict for splat into `compile(...)`. The old `configure_fake_llm` form that mutated module state is now a thin alias (line 559-569). This is exactly the right DI migration: ambient state → explicit injection.
- `tests/fakes.py:572-594` — `build_fake_runtime` lets tests construct an `LlmRuntime` directly for helpers that bypass `compile()`. Liskov-clean: the fake runtime is structurally indistinguishable from a real one.

**Why this is good architecturally**: each fake is the Information Expert for one LangChain shape, no more. Tests assemble exactly the fakes their scenario needs. The migration from mutate-and-import to build-and-splat eliminates a class of test pollution bugs and aligns the test surface with the production DI surface — both now use the same `LlmRuntime` carrier.

The §6 discipline ("tests target behavior, not dispatch path") is enforced *by the shape of the fakes*: a `StructuredFake` literally cannot count `invoke` calls in interesting ways because the contract is "return the model"; a test that wanted to assert call sequence would have to write a new fake (and would be reviewed for §6 compliance). Good Protected Variations: the fakes vary in the dimension the framework actually varies in (LangChain method shape), not in spurious dimensions like call counting.

---

## §6 Test discipline triage

**Principles at stake**: Single Responsibility (test layer), Low Coupling between test and impl.

**Verdict**: EXEMPLARY, severity none.

**Evidence**: `tests/test_observability_contract.py` and the 23-test rewrite of `tests/modes/test_llm_internals.py` (counts per `docs/design/test-discipline-audit-2026-05-20.md`). The discipline is now: pure-function tests target inputs/outputs (renderers, normalizers, fingerprints); behavioral tests target user-visible contracts; mock-plumbing assertions on internal call sequences are deleted.

**Why this is good**: the principle violated by the old tests was Low Coupling — tests were depending on internal call structure that wasn't part of the contract. Refactors that didn't change behavior broke tests because the tests had pinned implementation. The triage moves the assertion targets to the actual contract surface, which is what §6 codifies.

This is the same pattern as `Node.outputs` polymorphism (§8): pin the *meaning*, not the *shape*. The test discipline change and the normalizer extraction are different applications of the same principle (Protected Variations: vary the shape, don't break the meaning).

---

## §7 StateBus required-getter

**Principles at stake**: Liskov Substitution, Polymorphism (GRASP), Information Expert, Single Responsibility, Protected Variations.

**Verdict**: EXEMPLARY, severity none.

**Evidence**:
- `src/neograph/_state_bus.py:24-40` — `StateBus` Protocol with three methods (`get`, `get_required`, `keys`). Both `_DictStateBus` (line 43) and `_ModelStateBus` (line 61) implement the same shape. `adapt_state(state)` at line 80 is the single dispatch point.
- `src/neograph/_state_bus.py:52-55` (dict variant), `:70-74` (model variant) — `get_required` raises `StateMissingError` with the `node_label` parameter when the key is absent. Symmetric error path across both implementations.
- §7 design rule "Silent-`None` reads of required fields are a bug" is enforced at the Protocol level: callers must choose `get(...)` (optional) or `get_required(...)` (required) by *method name*, not by a positional default. The asymmetry is a feature.
- Used by `_oracle.py:105` (Each×Oracle redirect required EACH_ITEM), `_oracle.py:347` (Each redirect required EACH_ITEM) — these are exactly the sites the §7 discipline targets, and they now use `get_required` with `node_label=` so error messages surface the user's node name rather than the mangled `raw_fn.__name__`.

**Why this is good architecturally**: this is *textbook* DIP + Protected Variations + Polymorphism. The framework varies state shape (dict vs BaseModel) at runtime; the abstraction is a Protocol with two methods that mean what they say. The frozen `_MISSING` sentinel (line 21) avoids the standard pitfall of `None` colliding with explicit-None state values. The Liskov contract is exact: both implementations raise `StateMissingError` on absence, both return values including `None` when explicitly present.

Compare to the pre-StateBus pattern (`getattr(state, key, default)` scattered across helpers): every site re-implemented its own dict-vs-BaseModel branching, and the error messages were inconsistent. The Protocol is a Pure Fabrication that subsumed dozens of `isinstance(state, dict)` probes into one place. The maintenance cost dropped to zero per new caller.

**The one improvement vector** (not a defect): widening adoption. Some helpers in `_oracle.py` and `factory.py` still take `state: Any` and use `getattr(state, ...)` directly — `_inject_oracle_config` (line 27) takes `StateBus` correctly, but `make_oracle_merge_fn`'s inner closure at `_oracle.py:269` uses `getattr(state, field_name_for(key), None)` for upstream context resolution. Migrating those last sites would close the loop. Currently the StateBus discipline is convention-enforced, not structurally enforced.

---

## §8 Polymorphic outputs (single-type vs dict-form)

**Principles at stake**: Polymorphism (GRASP), Open/Closed, Single Responsibility, Protected Variations, Don't Repeat Yourself.

**Verdict**: EXEMPLARY at the normalizer, PROBLEMATIC at the still-unmigrated sites, severity MEDIUM.

**Evidence**:
- `src/neograph/_normalize.py:22-105` — `NormalizedOutputs` and `NormalizedInputs` are tagged unions (`is_dict_form`, `is_none` discriminators plus `primary`, `primary_key`, `secondary`, `all_keys` fields). Single function (`normalize_outputs`, `normalize_inputs`) does the discrimination. Module docstring explicitly notes the structural guard.
- `tests/test_structural_guards.py` (per §8 design rule) enforces no `isinstance(<expr>.outputs, dict)` in `src/neograph/`.
- `src/neograph/_oracle.py:146,182,200` — `_unwrap_oracle_results` and `_build_oracle_merge_result` still take raw `TypeSpecStatic` and re-discriminate via `isinstance(output_model, dict)`. These sites get the *type* (not the Node), so they can't call `normalize_outputs` directly without refactoring the signature. This is a hole in the abstraction: the normalizer covers the Node-attached case, not the type-only case.
- `src/neograph/state.py:336-381` (`_add_output_field`), `:382-434` (`_add_single_output_field`) — both use the normalizer correctly. Good.
- `src/neograph/_construct_validation.py:354-365` — `isinstance(output_type, dict)` is checked inline in the validator's producer-registration loop. The validator imports `normalize_outputs` indirectly via `_normalize.py` but does not use it for this discrimination. Per the structural guard's letter, this is a violation of §8's intent (one place); per the guard's wording, it may be exempted if it's an `output_type` rather than `node.outputs`.

**Why the normalizer is EXEMPLARY**: it is the canonical Polymorphism (GRASP) solution to a sum-type discrimination problem in a language without proper sum types. The frozen dataclass with discriminator fields is what Rust enums or OCaml variants look like in Python. Adding a new outputs form (e.g., `tuple[str, ...] → multi-output positional`) would require editing `normalize_outputs` once and the callers automatically pick up the new shape via the dataclass fields.

**Why the unmigrated sites are PROBLEMATIC**: the SOLID principle here is OCP. The normalizer was supposed to be the single point where dict-vs-single-type is decided. Every `isinstance(X, dict)` outside the normalizer is a parallel discriminator that must be kept in sync. The recent Oracle/Each fix that the maintainer mentions is exactly this category — three closure factories (`make_oracle_redirect_fn`, `make_eachoracle_redirect_fn`, `make_each_redirect_fn`) all had to learn the dict-form-output prefix-scan rule independently (`_oracle.py:67-78`, `:98-118`, `:343-360`). When a new output shape is added, three sites need editing, not one.

**Architecturally correct alternative**: extend `NormalizedOutputs` with the methods Oracle/Each need (`detect_dict_form_in_result(result_dict) -> bool`, `extract_primary_value(result_dict) -> Any`) so the discrimination logic lives at the normalizer, and the three closure factories receive a `NormalizedOutputs` instead of raw `output_model`. The signatures of `make_oracle_*_fn` should take `outputs: NormalizedOutputs` not `output_model: TypeSpecStatic`. That migration would close the hole.

**Web-search informed**: this is the standard "data-class-as-tagged-union" pattern recommended by Python's own typing community for pre-PEP-747 polymorphic types. The normalizer matches established best practice; the holes are not in the pattern but in the migration completeness.

---

## §9 Schema fingerprinting

**Principles at stake**: Single Responsibility, Information Expert, Low Coupling, Protected Variations.

**Verdict**: PROBLEMATIC, severity MEDIUM.

**Evidence**:
- `src/neograph/state.py:264-299` — `compute_node_fingerprints` walks the construct and computes per-node hashes. The function handles three cases (single-type outputs, dict-form outputs, sub-constructs) inline with `hasattr(item, "outputs")`, `no.is_dict_form`, `hasattr(item, "nodes")`. The discrimination is duplicated across the three cases (note the `typ.__qualname__ if isinstance(typ, type) else str(typ)` repetition at lines 284, 289, 297).
- `src/neograph/state.py:302-318` — `compute_schema_fingerprint` is cleanly a single-responsibility function over state model fields. Excludes framework prefixes by string-list filter (line 310). Cohesive.
- `src/neograph/runner.py:159-195` — `_compute_invalidated_nodes` reads `_neo_node_fingerprints` from the compiled graph as a typed attribute via `getattr(graph, "_neo_node_fingerprints", None)`. This is a Pure Fabrication that smuggles per-compile state through a side-attribute. `compile()` writes it at `compiler.py:240`; `runner.py` reads it back. Liskov-fragile: any test that constructs a "fake compiled graph" without those attributes silently degrades.
- `src/neograph/runner.py:198-236` — `_build_producer_consumer_adjacency` re-implements parts of the validator's producer registration (looks at `inputs`, `Each.over`, `context`) for the transitive-closure walk. The validator already has this information in its `producers: list[Producer]` walk — but it's discarded after validation. The adjacency rebuild is a second walk over the same IR, with subtly different rules.

**Why this is PROBLEMATIC**:
- **DIP violated**: the runner reaches into the compiled graph's private attributes (`_neo_*`) via `getattr` rather than depending on an explicit `CompiledNeograph` Protocol that declares the attrs as public. The LangGraph `CompiledStateGraph` is bolted-onto via dynamic attribute injection, which is a textbook DIP violation (high-level runner depends on low-level langgraph implementation detail by piggybacking on its mutability).
- **Single Responsibility violated**: `_compute_invalidated_nodes` does fingerprint diff + adjacency rebuild + transitive closure. Each is a separate concern; they share one function because the data flows linearly, but a change to any of the three forces edits to one body.
- **DRY violated**: producer→consumer adjacency exists implicitly in the validator's walk (`_construct_validation.py:281-376`) but is rebuilt from scratch in the runner. The two walks diverge: the validator handles `oracle_gen_type`, the runner doesn't. The runner's `add_edge(upstream_name, consumer_field)` at runner.py:221 is a flattened version of the validator's much richer producer tracking. A change to the IR's edge semantics needs synchronized updates in both — and the runner's version is less expressive, so it can silently miss invalidations.
- **Information Expert violated**: the construct is the Information Expert for its own adjacency, but the runner re-extracts it. Either the construct should expose `.adjacency()` (and the validator should consume it too), or the validator should attach the producer graph to the compiled output (parallel to fingerprints) so the runner doesn't recompute.

**Architecturally correct alternative**: introduce a typed `CompiledNeograph` wrapper that owns the LangGraph compiled graph plus the framework metadata (fingerprints, adjacency, required-DI, construct ref). The runner depends on `CompiledNeograph` directly, not `getattr` on LangGraph internals. `Construct` (or a `ConstructIR` view) exposes `.adjacency()` as the Information Expert, used by both the validator and the runner. This removes the duplication and makes the framework's side-channel state visible in the type system.

**Web-search informed**: the "wrap a third-party class with your own typed facade rather than monkey-patching its attributes" pattern is GRASP Protected Variations applied to FFI/library boundaries — it's the same lesson as the StateBus pattern (§7) but applied to the compiled graph rather than to state.

---

## Cross-cutting finding 1 — Three API surfaces and the assembly-path divergence

**Principles at stake**: Single Responsibility, DRY, Information Expert, Protected Variations.

**Verdict**: PROBLEMATIC, severity HIGH.

**Evidence**:
- `src/neograph/_construct_builder.py:518-593` — `_cleanup_inputs_and_register` is the `@node` decoration path's "final pass": strips DI params from `inputs`, rewrites port/loop params, sets `fan_out_param`, registers scripted shims, infers `oracle_gen_type`. Five distinct responsibilities, applied via accumulated `updates: dict[str, Any]` and one `model_copy` per node.
- `src/neograph/construct.py:163-218` — `Construct._normalize_fan_out_params` and `Construct._normalize_oracle_gen_type` are the YAML/programmatic surface's mirror pass: do the same two normalizations (`fan_out_param`, `oracle_gen_type`) by re-deriving them from scratch in `Construct.__init__`. The doc comments at line 156 explicitly call out: "All @node-only field inferences here, once, so all three surfaces converge before validation."
- Three surfaces → two implementations of the same IR-level normalization. The decorator path does its normalization at `_build_construct_from_decorated` time (during `construct_from_module`); the programmatic and YAML paths rely on `Construct.__init__` to retroactively do the same thing. CLAUDE.md explicitly documents this as a recurring source of bugs (the `neograph-ts7` reference).

**Why this is PROBLEMATIC**:
- **SRP violated**: `_cleanup_inputs_and_register` has five reasons to change (DI rewrite, port rewrite, loop rename, fan_out detection, shim registration, oracle inference). Each is independently triggered by different IR shapes; their combination is accidental.
- **DRY violated, structurally**: `fan_out_param` inference exists in two implementations (`_construct_builder.py:564-565` and `construct.py:163-195`). They are not equivalent — the decorator path knows the function signature (so it can derive fan_out from "param name not matched to upstream"); the Construct path has to re-derive from `inputs` dict keys vs peer node names. The same outcome via two algorithms. Bugs in one don't surface in the other until both surfaces are exercised in tests.
- **Information Expert violated**: `Node` should be the Information Expert for its own `fan_out_param` and `oracle_gen_type`. Instead, the IR-construction layer (`Construct.__init__`) and the @node-construction layer (`_cleanup_inputs_and_register`) both reach in and set these fields via `model_copy`. There is no single owner.
- **Protected Variations failed**: the three API surfaces vary in *how* the IR is assembled, but they should converge on *what* the IR contains. The current design has two convergence paths instead of one, and CLAUDE.md acknowledges the recurring bug pattern.

**Architecturally correct alternative**: introduce a `Node.finalize()` method (or a post-init hook) that is the *one* place fan_out_param, oracle_gen_type, and scripted_fn are derived. Both the decorator path and `Construct.__init__` call it; the inputs are *the Node itself* (Information Expert pattern). The signature-derived fan_out (decorator path) is computed *before* finalize and passed in as a parameter; the dict-key-derived fan_out (Construct path) is computed inside finalize from `node.inputs` against the peer set. Finalize then has one job: take an under-specified Node and produce a fully-specified Node. The duplication collapses to one algorithm with one optional hint parameter.

This is the single largest cohesion problem in the IR layer. The "hydra" symptom the maintainer is suspicious of is most visible here: every IR-shape change (fan_out, oracle_gen_type, dict-form outputs) has to be implemented twice and tested through all three surfaces. The three-surface parity rule in CLAUDE.md is a *test discipline* attempting to compensate for an *architectural defect*.

---

## Cross-cutting finding 2 — Three closure factories in `_oracle.py` with `node_name` threaded through

**Principles at stake**: Single Responsibility, DRY, Open/Closed, Polymorphism (GRASP).

**Verdict**: PROBLEMATIC, severity MEDIUM.

**Evidence**:
- `src/neograph/_oracle.py:49-81` — `make_oracle_redirect_fn`.
- `src/neograph/_oracle.py:84-121` — `make_eachoracle_redirect_fn`.
- `src/neograph/_oracle.py:331-360` — `make_each_redirect_fn`.

All three:
- Wrap `raw_fn` to redirect/tag its output by reading state.
- Take `node_name` as a required kwarg (line 53, 87, 334) only so that future `get_required` calls can surface the user-declared name rather than LangGraph's mangled function identity. The Oracle variant doesn't even use it yet (`_oracle.py:68` — "reserved for future get_required calls — see kg8l audit").
- Implement single-type-vs-dict-form output discrimination via `prefix = f"{field_name}_"` and `any(k.startswith(prefix) for k in result)` (lines 67, 76 and 98, 116).
- Each one is a slight permutation of the same closure pattern (Oracle: collect to list; EachOracle: tag with each_key, collect to tagged list; Each: key by each_key, produce dict).

**Why this is PROBLEMATIC**:
- **SRP smell, not violation**: each factory does build one closure, which is one responsibility. But the *family of factories* is the smell — three near-identical bodies, three places to update on each new output shape or each new modifier combo.
- **OCP violated**: adding a new modifier combination (e.g., `Loop × Each` if it were ever supported) means a fourth factory in the same shape.
- **DRY violated structurally**: the dict-form-detection logic (`any(k.startswith(prefix) for k in result)`) is duplicated three times. When a new output form is added — say, list-form outputs — all three factories need synchronized updates.
- **Polymorphism (GRASP) opportunity missed**: this is the canonical case for a Strategy or Decorator pattern. A `RedirectStrategy` Protocol with `before_call(state) -> ExtractedContext` and `transform_result(result, ctx) -> dict[str, Any]` methods would let each modifier provide its own strategy, and `make_redirect_fn(raw_fn, strategy, field_name)` becomes one function.

**Architecturally correct alternative**: extract a `ResultTransform` Protocol with two methods (`extract_context(state) -> Context`, `wrap_result(result_dict, field_name, context) -> dict[str, Any]`). Implementations: `OracleCollect`, `EachOracleTag`, `EachKey`. The redirect factory becomes one function: `make_redirect_fn(raw_fn, transform, field_name, node_name)`. Dict-form discrimination moves to the transform implementations (or, better, to a shared helper that takes a `NormalizedOutputs` per §8). When a fourth modifier combo arises, you write one new transform implementation; when a new output shape arises, you edit one helper.

This is the second hydra. The maintainer is right to be suspicious: the recent `node_name` threading change touched three factories, and the dict-form prefix-scan logic was duplicated three times before that. The closures are not the problem; the *absence of a Strategy abstraction* over them is.

---

## Cross-cutting finding 3 — `Construct._normalize_fan_out_params` / `_normalize_oracle_gen_type`

**Principles at stake**: Information Expert, Single Responsibility, Open/Closed.

**Verdict**: PROBLEMATIC, severity MEDIUM (subset of cross-cutting finding 1).

**Evidence**:
- `src/neograph/construct.py:163-218` — both methods run inside `Construct.__init__` after `super().__init__`. They mutate `self.nodes[i]` via `model_copy`. The docstring at line 197-205 explicitly cross-references the @node decoration path: "Mirrors the inference at `_construct_builder.py:583`".
- Both methods exist *because* the IR-parity rule needs enforcement — but the rule is enforced by duplicating the inference logic, not by sharing it.
- `Construct.__init__` itself is doing five things: validate-non-empty, propagate llm_config, propagate renderer, normalize fan_out_param, normalize oracle_gen_type, run validation. Four of those are IR-completion concerns; one (`super().__init__`) is Pydantic plumbing. This violates SRP at the `__init__` level.

**Why this is PROBLEMATIC**: the existence of these two methods is an admission that `Construct.__init__` is doing the wrong layer's job. The Information Expert for fan_out_param is `Node`; the Information Expert for oracle_gen_type is `Node` + its `Oracle` modifier's `merge_fn`. Construct should not be deriving fields on Nodes it contains. This is a Feature Envy code smell at the architectural level: Construct envies Node's internal state and reaches in to fix it.

**Architecturally correct alternative**: per finding 1, move the inference to `Node.finalize(peer_field_names)` or similar. `Construct.__init__` then loops `for node in self.nodes: node.finalize(peers)` — one line, no inference logic in the Construct. The decorator path's pre-inference becomes an optional argument or a separate `Node.finalize_with_signature(fn, params)`. The two normalize methods on Construct delete entirely.

---

## Cross-cutting finding 4 — `Node._sidecar` PrivateAttr storing user Callable + DI bindings

**Principles at stake**: Single Responsibility, Liskov Substitution, Dependency Inversion, Cohesion.

**Verdict**: PROBLEMATIC, severity LOW-MEDIUM.

**Evidence**:
- `src/neograph/node.py:202-204` — three `PrivateAttr` fields on Node: `_sidecar`, `_param_res`, `_scripted_shim`. Comment block at 196-201 explains the design rationale.
- `src/neograph/_sidecar.py:31-46` — accessor functions wrap the PrivateAttr access. The `_register_sidecar`, `_set_param_res`, `_get_*` family is six functions for what should be field access.
- `src/neograph/node.py:209` — `model_config = {"arbitrary_types_allowed": True}` is required because `raw_fn`, `renderer`, `skip_when`, `skip_value`, and `tools` are not Pydantic-validatable. The sidecar callable adds to this pile.
- CLAUDE.md "Why PrivateAttr, not proper fields" section acknowledges this: "the sidecar carries a `Callable` ... which can't go through Pydantic schema validation without `arbitrary_types_allowed` on every downstream consumer."

**Why this is PROBLEMATIC**:
- **Cohesion lowered**: Node is the IR-shape declaration (`name`, `mode`, `inputs`, `outputs`, modifiers). The sidecar is implementation metadata (the original callable, DI bindings, the synthesized shim). Putting both on Node merges two concerns: "what the IR is" and "how this runtime invokes it". CLAUDE.md itself separates these conceptually ("the sidecar carries the IR-level metadata that the compiler needs ... separates 'what the node is' from 'how this runtime invokes it'") but then implements them on the same object.
- **Liskov fragility**: a Node assembled programmatically (no `@node`) has `_sidecar = None`. A Node assembled via `@node` has it set. Downstream code (`compiler._collect_scripted_shims:272` calls `getattr(item, "_scripted_shim", None)`) must always defensive-check. The Node type isn't substitutable: two Nodes with the same `name`/`inputs`/`outputs` behave differently depending on hidden private state.
- **DIP partial**: the sidecar is the runtime's representation, but it's stored on the IR object. The IR depends on the runtime's needs.
- **`arbitrary_types_allowed` cost**: every consumer of Node now sees a model that might contain arbitrary types, which weakens Pydantic's normal guarantees. This is partially a Pydantic limitation, partially a design choice.

**Why severity is LOW-MEDIUM, not HIGH**: the alternative (a parallel `NodeRuntime` map keyed by Node identity) introduces lifetime-management problems and was explicitly considered and rejected (CLAUDE.md "No global dicts, no `weakref.finalize`, no re-registration needed after `|`"). The PrivateAttr+model_copy approach is the *least bad* of the available implementations given Pydantic v2's constraints. But "least bad" is still not "good": the right architectural answer is a separate IR-vs-runtime split where Node is pure data and a `NodeAssembly` wrapper carries the runtime metadata.

**Architecturally correct alternative**: keep `Node` as a pure-IR Pydantic model. The `@node` decorator returns a `DecoratedNode` dataclass with two fields: `node: Node` and `runtime_meta: SidecarMetadata`. `construct_from_module` collects `DecoratedNode` instances (not `Node` instances), splits them into a `Construct(nodes=[...])` and a `runtime_map: dict[str, SidecarMetadata]`. `compile()` takes both. No PrivateAttr, no `arbitrary_types_allowed` for the sidecar, no `getattr` defensive checks downstream. The migration is mechanical but invasive — currently the framework treats `Node` as the universal currency, and splitting it would touch every test.

This is the kind of refactor that's worth doing once at a major version bump (0.x → 1.0 territory) and not before.

---

## Cross-cutting finding 5 — Recursive `_validate_node_chain` with `ambient_producers`

**Principles at stake**: Single Responsibility, High Cohesion, Open/Closed.

**Verdict**: PROBLEMATIC, severity MEDIUM.

**Evidence**:
- `src/neograph/_construct_validation.py:253-486` — `_validate_node_chain` is 233 lines. It does: validates input type compatibility, runs the deprecation warning for single-type inputs, validates context references, recurses into sub-constructs, validates loop+skip_when interaction, validates merge_fn state params, validates merge hook signatures, and checks the output boundary contract. Nine distinct responsibilities.
- `src/neograph/_construct_validation.py:256` — `ambient_producers` parameter is the recursion-state argument used by sub-construct re-validation. Its presence is the design admitting that the validator is doing context-resolution work that the single-pass design originally avoided.
- The function is the single source of truth for the validator (good), but it shoulders every validation rule the framework has ever added (bad).

**Why this is PROBLEMATIC**:
- **SRP violated catastrophically**: nine responsibilities under one function head. Adding a tenth means a tenth `if isinstance(item, Node): ...` block inside the same loop. Each block reads from the same `producers` list and writes errors with similar but not identical structure.
- **OCP violated**: every new validation rule is a `git diff` inside `_validate_node_chain`. The rules don't compose; they accumulate.
- **Cohesion lowered**: the function's nine rules don't share an algorithmic structure — they all happen to need the producer list, so they were piled into one walk. A Strategy pattern (`Validator` Protocol with `apply(item, producers, ambient) -> Iterable[ConstructError]`) would let each rule live in its own class with a stable signature.

**Architecturally correct alternative**: extract each rule into a `Validator` (Protocol with one `apply` method) and run `for v in VALIDATORS: errors.extend(v.apply(...))`. The walker becomes a 20-line dispatcher; each rule becomes a 20-line class with a clear name (`InputCompatibilityValidator`, `ContextReferenceValidator`, `MergeHookSignatureValidator`, etc.). New rules don't edit the walker. Tests target individual validators.

This is also the place where the `ambient_producers` complexity becomes manageable: instead of being a hidden recursion-state parameter, it's a normal field on a `ValidationContext` dataclass passed to every validator.

---

## Cross-cutting principles violated repeatedly

Three patterns recur across the PROBLEMATIC items:

### 1. `getattr`/`isinstance` discrimination instead of Polymorphism

Sites: `_construct_validation.py:79` (`item.outputs if isinstance(item, Node) else getattr(item, "output", None)`), `_oracle.py:67,98,343` (dict-form output detection), `runner.py:198-236` (re-extracting adjacency by `getattr`), `state.py:264-299` (`hasattr(item, "outputs") ... hasattr(item, "nodes")`).

The §8 normalizer pattern is the model. The fix is to push the discrimination into a Protocol method or a normalizer function called once at the boundary. Every additional `isinstance`/`getattr` check at a call site is a future bug waiting for the maintainer to forget the same check at a sibling site.

### 2. The "shared rule, two implementations" cohesion break

Sites: `_cleanup_inputs_and_register` vs `Construct._normalize_*` (fan_out_param, oracle_gen_type), validator's producer registration vs runner's adjacency rebuild, the three `make_*_redirect_fn` factories.

The fix is to identify the rule's Information Expert and put the rule there once. Every duplicated rule is one more "did I update both copies?" check at code review time.

### 3. SRP failure via accumulation

Sites: `_validate_node_chain` (9 rules in one function), `_call_structured` (4 concerns in one function), `_invoke_json_with_retry` (5 concerns in one function), `Construct.__init__` (5 jobs in one constructor).

The fix is the Strategy pattern: a Protocol with one method, an implementation per concern, a thin loop that runs them. The cost is more files; the win is each new rule lands as a new file rather than a new branch.

---

## Recommendations

Ranked by SOLID/GRASP severity. Each addresses one of the three cross-cutting patterns above.

### High priority (HIGH severity, hydra symptoms)

**R1. Refactor `_call_structured` and the retry stack (§4 fix).**
Extract a `StructuredOutputAdapter` Protocol with three implementations (`StructuredAdapter`, `JsonModeAdapter`, `TextAdapter`). Provider-quirk shims (`include_raw` ladder, DSML recovery) become decorators applied at runtime construction. Replace the nested try/except retry ladder with a generic `with_output_quality_retry(adapter, max_retries, build_feedback)` helper. This converts every future provider quirk from "another `except TypeError` arm" into "another decorator." DSML recovery moves from a content-detection helper called from three sites into a single decorator applied to the appropriate adapters.

**R2. Unify the three-API-surface assembly path (cross-cutting finding 1 + 3).**
Introduce `Node.finalize(peer_field_names)` as the single place fan_out_param, oracle_gen_type, and scripted_fn are derived. Both `_build_construct_from_decorated` and `Construct.__init__` call it. The decorator path passes an optional `signature_hint` for parameters; the YAML/programmatic path passes None. Delete `Construct._normalize_fan_out_params` and `_normalize_oracle_gen_type`. The three-surface parity test discipline becomes structurally enforced rather than test-enforced.

### Medium priority (MEDIUM severity)

**R3. Replace the three `make_*_redirect_fn` factories with a Strategy pattern (cross-cutting finding 2).**
Extract `ResultTransform` Protocol. One generic `make_redirect_fn(raw_fn, transform, field_name, node_name)`. Dict-form detection moves to the transform implementations or to a shared `NormalizedOutputs` helper per §8.

**R4. Extend the §8 normalizer to cover Oracle/Each call sites.**
Migrate `_unwrap_oracle_results`, `_build_oracle_merge_result`, and the three redirect factories to take `NormalizedOutputs` instead of raw `output_model: TypeSpecStatic`. Close the §8 structural-guard hole.

**R5. Split `_validate_node_chain` into a Validator Strategy chain (cross-cutting finding 5).**
9 rules → 9 `Validator` classes with one `apply` method each. Walker becomes a 20-line dispatcher. New rules don't edit the walker; tests target individual validators.

**R6. Wrap the compiled graph in a typed `CompiledNeograph` facade (§9 fix).**
Replace `getattr(graph, "_neo_*", None)` calls with attribute access on a typed wrapper. Move the producer→consumer adjacency from a runner re-derivation to a Construct method (`adjacency()`), consumed by both the validator and the runner.

### Lower priority (LOW severity, accepted tradeoffs but worth tracking)

**R7. Migrate `@merge_fn` to the sidecar pattern (§3 cleanup).**
Eliminate `_merge_fn_registry` as a module-global; attach metadata to the merge function (or to a returned descriptor) the same way `@node` attaches `_scripted_shim` to Node.

**R8. Long-term: split `Node` into `Node` (pure IR) + `DecoratedNode` (IR + sidecar) (cross-cutting finding 4).**
Worth doing at a 0.x → 1.0 boundary. Removes `arbitrary_types_allowed` for the sidecar surface, restores Liskov substitutability of Node, eliminates the PrivateAttr defensive-getattr pattern across the framework.

---

## Closing observation

The codebase has *one* genuinely excellent architectural pattern — the §7 StateBus / §8 normalizer / §5 fakes triad — and the codebase is internally aware that this is the model. The structural guards enforce the pattern's letter; CLAUDE.md preaches the pattern's spirit. The PROBLEMATIC items are uniformly cases where the same pattern would have applied but wasn't reached for, usually because the abstraction would have meant slightly more upfront refactoring than the immediate fix warranted.

The maintainer's "hydra" intuition is correct: every fix that's framed as "update three sites" rather than "extend one abstraction" creates the next bug, because the three sites drift. The §4 retry stack and the three closure factories in `_oracle.py` are the two clearest hydra factories in the codebase. R1 and R3 are the two interventions most likely to eliminate the symptom at the root.
