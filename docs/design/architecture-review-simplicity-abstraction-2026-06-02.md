# Simplicity & Abstraction-Value Review — 2026-06-02

Lens: does each abstraction earn its keep without killing the value it claims to provide? Verdicts respect the maintainer's pushback — polymorphism is good when substitutability is real, and replacing classes with branching functions is never the recommendation.

---

## Summary

Items reviewed: 14 (9 from architecture-decisions.md + 5 cross-cutting).

| Verdict | Count | Items |
|---|---|---|
| EARNS-KEEP | 9 | §1 LangGraph commitment, §2 LlmRuntime, §3 retry layering, §5 fakes, §7 StateBus, §8 normalize_outputs/inputs, §9 two-level fingerprinting, three-API-surface IR, Node._sidecar |
| OVERWROUGHT | 3 | §3 `_attempt_dsml_recovery` content-based recovery path, §6 observability contract file (test-as-spec), three closure factories in `_oracle.py` (cosmetic polymorphism in dict-form/single-type branches) |
| UNDER-ABSTRACTED | 2 | Per-compile registries (`scripted` / `conditions` / `tool_factories` are three parallel dicts with the same shape — should be one `CompileRegistry`), `Construct._normalize_*` sibling methods (loose ad-hoc inference that should be a `NodeNormalizer` strategy registry) |

Headline: the IR-level abstractions are largely correct — the polymorphic outputs/inputs, the StateBus protocol, the LlmRuntime bundle, and the sidecar pattern all carry real semantic content with three or more callers. The drift is concentrated in two places: (a) the LLM-recovery path has grown a content-based DSML detector that branches on response shape inside a function that's already branching on strategy (this is the genuine `if/elif`-grows-features anti-pattern), and (b) the compile() kwargs (`scripted=`, `conditions=`, `tool_factories=`) are three structurally identical registries handled by three parallel code paths — primitive-obsession on `dict[str, Callable]` where a thin `CompileRegistry` type would consolidate the merge / lookup / validation logic.

---

## §1 LangGraph commitment / `llm_factory` indirection

**Files:** `src/neograph/_llm_runtime.py:55`, `src/neograph/runner.py:1-360`, `src/neograph/compiler.py:24-30` (imports from `langgraph.graph`).

**Verdict: EARNS-KEEP.**

**Rule of three:** `llm_factory` has > 3 consumers — `invoke_structured`, `invoke_with_tools`, the Oracle merge LLM-judge path, and per-tier model resolution. The factory shape (`Callable[[str], BaseChatModel]`) is exactly the FastAPI-style "Provider" pattern: a callable parameterized by a tier name. Removing it forces every node to either bind a concrete model class (ties IR to LangChain class names) or accept a `BaseChatModel` instance directly (loses per-tier dispatch).

**Concrete-first:** the LangGraph commitment is recent (§1 records the decision to stop pretending the IR is backend-neutral). Before that, there was a "backend abstraction" layer that was deleted. The `llm_factory` callable survived because it's not about backends — it's about per-tier model selection inside one backend (fast / reason / large). That's a genuinely different axis.

**Cognitive cost vs payoff:** zero indirection beyond "user closure returns a model." The cost is one extra function call per LLM dispatch; the payoff is per-tier routing without leaking provider class names into IR.

**Reversibility:** removing `llm_factory` would force every Node to carry a `model_cls=` field with a hardcoded LangChain class. That's worse: it couples the IR to specific provider class names and makes test injection ugly (every test would have to monkey-patch class references).

**Recommendation:** leave alone. The structural guard against private `langgraph.*` imports is the right enforcement mechanism — keeps the commitment honest without overengineering. One tightening: the docstring at `_llm_runtime.py:7-12` is more about §2 than §1; consider moving §1's commitment statement into `compiler.py`'s module docstring so anyone reading the compile entrypoint sees the dependency posture immediately.

---

## §2 `LlmRuntime` frozen dataclass

**Files:** `src/neograph/_llm_runtime.py:46-93`.

**Verdict: EARNS-KEEP.**

**Rule of three:** `LlmRuntime` is threaded through `make_node_fn`, `make_oracle_merge_fn`, `make_each_redirect_fn`, `make_oracle_redirect_fn`, `_add_subgraph`, `_add_each_oracle_fused`, and the test path via `build_fake_runtime`. Six+ real consumers, all reading the same five fields (`llm_factory`, `prompt_compiler`, `renderer`, `cost_callback`, plus the cached param sets).

**Concrete-first:** the bundle was extracted AFTER the four kwargs proliferated through every factory closure. The grep'd evidence: 69 references to `LlmRuntime`/`EMPTY_RUNTIME` across 11 modules. Without the bundle, each closure would either carry four separate keyword arguments (parameter-list bloat that breaks on every new kwarg) or read a global (forbidden by §2's multi-tenant isolation rule). The frozen dataclass is the smallest correct abstraction: it lets the kwargs flow as one parameter without losing typing.

**Cognitive cost:** low. The dataclass has six fields, none of which require explanation beyond their type name. The `_accepted_params` introspection cache is a one-line optimization with a clear name.

**Reversibility:** if you delete `LlmRuntime`, every closure signature in `factory.py` / `_oracle.py` / `_wiring.py` grows by four params. The current Pyright-greppable change set is bounded; without the bundle, adding a sixth runtime knob (e.g. `tracer=`) is a fan-out edit across the dispatch layer.

**The `EMPTY_RUNTIME` sentinel** (line 93) is the one place to scrutinize. It's a frozen instance with all `None` fields, distinct from `None` itself. The distinction is meaningful: closures fall back to `EMPTY_RUNTIME` instead of `None` so they can always call `runtime.llm_factory` safely (it just returns `None`). This is the textbook Null Object pattern and is exactly the right structural choice — better than 30 `if runtime is None:` guards downstream.

**Recommendation:** leave alone. If anything, document the Null Object reasoning at line 93 — future readers will otherwise be tempted to "simplify" the sentinel away.

---

## §3 Retry concerns + `_llm_retry.py` + `_llm_dispatch.py`

**Files:** `src/neograph/_llm_dispatch.py:1-120`, `src/neograph/_llm_retry.py:1-362`.

**Verdict: EARNS-KEEP for the layered split; OVERWROUGHT for `_attempt_dsml_recovery`'s content-based detection inside two already-branching dispatch sites.**

**Rule of three:** the THREE-LAYER retry split (transient failures in user's `llm_factory` via `.with_retry`, output-quality retry in `_invoke_json_with_retry`, in-node retry inside scripted functions) is the textbook separation of concerns from the LangChain community discussions on retry strategies (multiple OutputFixingParser threads). Each layer has a different failure model and a different reasonable response. Collapsing them into one `retry_policy=` kwarg loses that distinction.

**Concrete-first:** the table at architecture-decisions.md §3 codifies an existing split. The split came from real bug reports — token-budget bugs where output-quality retry replayed expensive tool calls. The decision to forbid `LangChain.OutputFixingParser` in favor of BAML-rendered feedback came after a concrete failure where Schema-based format instructions wedged the model.

**Cognitive cost:** `_invoke_json_with_retry` is one clear loop. `_parse_json_response` is one clear sequence (extract balanced JSON → repair via `json_repair` → apply null defaults → validate). These earn their lines.

**Where the abstraction starts to wobble:** `_attempt_dsml_recovery` (lines 310-361) and the TypeError-with-DSML fallback in `_call_structured` (lines 79-108). The recovery now has THREE code paths that all do "detect DSML markup, then retry with a budget-exhausted message":
1. `_tool_loop` post-ReAct final response (not shown but referenced)
2. `_call_structured` silent variant (parsed=None, line 65)
3. `_call_structured` TypeError variant (line 101)

The cross-strategy detection (`_DSML_PATTERN`) is correctly a module constant. But the recovery FLOW (collect text → check pattern → invoke targeted retry → fall back to generic retry) is implemented inline at three sites with slightly different control flow each time. This is the classic case of "the abstraction is right (`_attempt_dsml_recovery` is the recovery primitive), but the call-site polymorphism (detection in three places with subtly different exception ladders) is overwrought."

**Reversibility:** the layered retry split is irreversible without losing real behavior (act-mode safety, BAML feedback fidelity). The DSML recovery flow can be tightened: a `DsmlGuard` Strategy object that takes (raw_text, exc) and returns either a recovered model or re-raises would centralize the three call sites into one `guard = DsmlGuard(cfg, output_model, llm, messages, config); result = guard.try_recover(raw_text)` invocation.

**Recommendation:** keep §3's three-layer policy; introduce a `DsmlGuard` class (not a function) in `_llm_retry.py` to absorb the three branching call sites. The class form is correct here because the guard holds state across attempts (whether the generic-retry path has already been consulted) and exposes a stable entry point. Do NOT collapse the three retry concerns into one knob — that would be the "single retry_policy" anti-pattern §3 explicitly rejected.

---

## §4 Module / function single-responsibility discipline

**Files:** policy across `src/neograph/*.py`; enforcement in `tests/test_structural_guards.py`.

**Verdict: EARNS-KEEP.**

The discipline itself is enforced by structural guards (function-local `from neograph...` imports forbidden in new code, import DAG cycles broken via Protocol modules). The `_ir_protocols.py` + `_llm_protocols.py` extraction pattern is the textbook "Acyclic Dependencies Principle" remedy. The `_construct_builder.py → _sidecar.py ← decorators.py` split was specifically forced by a real circular import, and the guard prevents it from regressing.

**Rule of three:** the Protocol-extraction pattern has been applied at least three times (`ConstructItem` in `_ir_protocols.py`, `StateBus` in `_state_bus.py`, the `LlmFactory`/`PromptCompiler` aliases under TYPE_CHECKING in `_llm_runtime.py`). Each extraction had a concrete circular-import or test-double substitution driver. Not preemptive.

**Recommendation:** leave alone.

---

## §5 Test fakes (5 of them)

**Files:** `tests/fakes.py:125-512` (`StructuredFake`, `StructuredFakeWithRaw`, `ReActFake`, `StringArgsFake`, `TextFake`, `GuardFake`, `StubbornFake`, `FakeTool`).

**Verdict: EARNS-KEEP.**

**Rule of three:** there are actually SEVEN fakes, each tied to a distinct test surface:
- `StructuredFake` — `.with_structured_output()` for produce mode
- `StructuredFakeWithRaw` — `include_raw=True` path
- `ReActFake` — `bind_tools()` + scripted call sequence
- `StringArgsFake` — provider quirk where tool_calls.args arrives as JSON string
- `TextFake` — plain text invoke for `json_mode`/`text` strategies
- `GuardFake` / `StubbornFake` — ReAct loop guard tests

Each fake encodes a specific provider/LangChain behavior. They're not substitutable: a test that needs to script tool calls cannot use `StructuredFake`. Substitutability is exactly what `langgraph-dev`-style mocks lack, which is why test infrastructure for LangGraph tends to grow per-scenario doubles. Seven is not "too many" — it's "one per provider behavior surface."

**Cognitive cost:** the fakes are isolated in one file with clear `═══` section breaks. Each has a docstring stating its scope. Onboarding cost is low because tests using them follow a pattern: `fake = SomeFake(...)` then `compile(c, **configure_fake_llm(lambda tier: fake))`.

**Recommendation:** leave alone. The single thing worth adding is a tabular "when to use which fake" comment at the top of `fakes.py` — currently scattered across each class's docstring. (Not a refactor, a comment.)

---

## §6 Test observability contract file

**File:** `tests/test_observability_contract.py:1-80+`.

**Verdict: OVERWROUGHT (mildly).**

**Rule of three:** the file is a single-source-of-truth for five user-observable log event names. The PRINCIPLE (don't pin event names in scattered behavioral tests) is correct — those tests should pin observable results, not log strings. But the IMPLEMENTATION (a parallel test file that pins each event by full integration setup) replicates substantial fake plumbing that the behavioral tests already exercise.

**Concrete-first:** Five events being checked are: `react_max_iterations_exceeded`, `react_token_budget_exceeded`, `trailing_tool_call_markup`, `react_guard_forced_break`, `auto_resume_schema_change`. Each contract test re-runs the same `invoke_with_tools` / `compile` / `run` machinery just to capture one log line.

**Cognitive cost vs payoff:** the contract file works, but it doubles the test surface for these five events. A reader investigating "why does react_max_iterations_exceeded fire" reads two tests: the behavioral one and the contract one.

**Substitutability:** a thinner alternative is a single constant module (`src/neograph/_log_contract.py`) declaring `EVENT_NAMES: Final[set[str]]` with structured guard tests verifying every `log.warning(...)` / `log.info(...)` event string in `src/neograph/` matches the registry. That's a guard, not a runtime test, and the cost is one structural-guard scan.

**Reversibility:** keeping the file does no harm. The criticism is that the abstraction (test-as-contract) is heavier than the alternative (constant + AST guard). For five events the heavyweight form is borderline; if it grows to 15 it should be migrated.

**Recommendation:** keep for now; if the event list crosses ~10, migrate to a constant + AST guard. Document the threshold in the file's module docstring so the migration trigger is obvious.

---

## §7 StateBus protocol + adapters

**Files:** `src/neograph/_state_bus.py:1-89`.

**Verdict: EARNS-KEEP.**

**Rule of three:** `adapt_state` / `StateBus` is used across `_execute.py`, `_oracle.py`, `_subconstruct.py`, `_wiring.py`, `_input_shape.py`, `_state_write.py` — at least 6 callers, all of which previously branched internally on `isinstance(state, dict)`. The duplication was real; the consolidation is real.

**Concrete-first:** the protocol was extracted AFTER the dict-vs-BaseModel branching appeared in 6+ places (`get_required` was the trigger — different error messaging per shape). The architecture decision §7 explicitly distinguishes "required read" from "optional read" with the latter only for fan-out collector fields. That distinction would be lost in a flat `state.get(key)` API.

**Substitutability:** the two implementations (`_DictStateBus`, `_ModelStateBus`) are genuinely substitutable — they implement the same three methods with the same semantics. This is the textbook case where Protocol-based polymorphism is correct: two implementations behind one interface, both with non-trivial logic (the model variant uses `getattr` + sentinel; the dict variant uses `__contains__`).

**Recommendation:** leave alone. The `@runtime_checkable` decoration is justified — `adapt_state` returns a `StateBus`, and consumers can `isinstance(x, StateBus)` if they need to (in practice they don't, but the decorator costs nothing).

---

## §8 Polymorphic `Node.outputs` / `Node.inputs` + `normalize_outputs`/`normalize_inputs`

**Files:** `src/neograph/_normalize.py:1-105`, `src/neograph/node.py:148-149`.

**Verdict: EARNS-KEEP.**

**Rule of three:** 11 modules import `normalize_outputs` or `normalize_inputs` (per grep). The single-type-vs-dict-form discrimination was repeated 18+ times before the normalizer existed (per the docstring at `_normalize.py:6`). The structural guard against `isinstance(<expr>.outputs, dict)` outside the normalizer enforces the contract.

**Substitutability:** `NormalizedOutputs` is a tagged union via flags (`is_dict_form`, `is_none`) rather than ADT subclasses. This is a deliberate choice — Python's `match` on classes works but loses the `primary`/`primary_key`/`secondary` field stability. The frozen dataclass form is easier to consume than a sealed-class hierarchy in Python (no `match` overhead, no sealed enforcement). Reasonable choice for the language.

**Cognitive cost:** low. Consumers do `no = normalize_outputs(node.outputs); if no.is_dict_form: ...`. The discriminator is local to one expression.

**Reversibility:** removing the normalizer would re-introduce the 18+ scattered `isinstance` checks, EACH of which would have to derive `primary_key` / `secondary_keys` / `all_keys` independently. The normalizer is genuine deduplication.

**Recommendation:** leave alone. The one nit: `NormalizedOutputs.primary: Any` could be `TypeSpecStatic`-typed if `node.py:126` is imported. The `Any` here is loose against §5's discipline (IR public API). Tighten to `TypeSpecStatic` in a future cleanup.

---

## §9 Two-level fingerprinting

**Files:** `src/neograph/state.py:264-318`, `src/neograph/runner.py:77-249`.

**Verdict: EARNS-KEEP.**

**Rule of three:** two fingerprints with two clear purposes. `compute_schema_fingerprint` is the "did the state-model shape change at all?" gate that triggers per-node analysis. `compute_node_fingerprints` is the per-node "which specific node changed?" map that drives transitive invalidation. They serve different consumers (`_verify_checkpoint_schema` vs `_compute_invalidated_nodes`).

**Concrete-first:** the two-level design was driven by the use case "I edited one node's output type — re-run only that node and its descendants." A single fingerprint would force whole-pipeline re-execution on any change. The split is exactly the "Merkle-tree" pattern that Prefect's cache-miss model and Hamilton's incremental DAG re-execution both use. Standard practice.

**Cognitive cost:** the fingerprints are 12-16 char SHA-256 prefixes; the transitive closure walk (`_transitive_closure`, runner.py:239) is a 12-line BFS. Very low.

**Reversibility:** dropping per-node fingerprints means dropping `auto_resume`. That's a user-visible feature regression.

**Recommendation:** leave alone. The one structural concern: `compute_node_fingerprints` at `state.py:264` is in the state module but `_compute_invalidated_nodes` is in runner.py. The fingerprint-comparison logic should arguably live next to its computation. Cross-reference comment at minimum.

---

## Cross-cutting 1: Three API surfaces producing same IR

**Files:** `src/neograph/_construct_builder.py:1-704`, `src/neograph/loader.py:1-275`, `src/neograph/decorators.py:1-780`.

**Verdict: EARNS-KEEP.**

**Rule of three:** the three surfaces (declarative `Node(...)`, `@node` decorator, YAML loader) all produce `Construct(nodes=[Node, ...])`. The IR convergence is enforced by `Construct.__init__` running `_normalize_fan_out_params`, `_normalize_oracle_gen_type`, and `_validate_node_chain` regardless of which surface built the construct. The "three-surface parity rule" in CLAUDE.md is the test discipline that backs this.

**Concrete-first:** the YAML loader was added AFTER the declarative form was stable; the `@node` decorator was added AFTER both. Each addition required convergence work (the `_normalize_*` methods in `construct.py:163-218` exist specifically because the YAML and programmatic paths don't have decorator-time hooks the way `@node` does). This is a converged design, not a preemptive abstraction.

**Reversibility:** dropping any surface loses a real use case. `@node` is the human-ergonomic default; declarative is what LLM-driven runtime construction (per MEMORY.md `project_llm_graph_construction.md`) actually emits; YAML loader is the config-driven path that piarch uses.

**Recommendation:** leave alone. The CLAUDE.md three-surface parity rule is the right enforcement and should be cited at the top of `_construct_builder.py` (currently only in CLAUDE.md). Adding it as a module docstring at `decorators.py:1` would help next-session agents not regress this property.

---

## Cross-cutting 2: Three closure factories in `_oracle.py`

**Files:** `src/neograph/_oracle.py:49-361`.

**Verdict: OVERWROUGHT (mildly — the closures are correct but the dispatch INSIDE each closure isn't substitutable).**

The three factories: `make_oracle_redirect_fn` (line 49), `make_eachoracle_redirect_fn` (line 84), `make_each_redirect_fn` (line 331), plus `make_oracle_merge_fn` (line 212). Each returns a closure with a different shape (Oracle redirect writes to `collector_field`; EachxOracle redirect writes tagged tuples; Each redirect writes a keyed dict).

**Substitutability:** the THREE factories produce closures with DIFFERENT signatures and DIFFERENT result shapes. They're NOT substitutable. They're three separate primitives. The maintainer is correct that collapsing them into one function with mode-switch would worsen complexity.

**Where the wobble is:** inside `make_oracle_redirect_fn` (line 70) and `make_eachoracle_redirect_fn` (line 101), the closure body branches on result shape:
```python
val = result.get(field_name)
if val is not None:
    return {collector_field: val}
# Dict-form outputs: per-key fields like {field_name}_{key}
if any(k.startswith(prefix) for k in result):
    return {collector_field: result}
return result
```
This is unwrap logic that mirrors what `_normalize.py:normalize_outputs` already does at IR level. The closures branch on RUNTIME shape; the IR has already decided the shape. The branching should be lifted: pass `is_dict_form: bool` (or `NormalizedOutputs`) into the factory, build the right closure once. Today the closure pays the `any(k.startswith(prefix))` scan on every dispatch.

**Recommendation:** keep the three factories. Refactor the closure body to use a pre-resolved unwrap strategy (computed once at factory time from `node.outputs`'s normalized form). The strategy can be a small `_UnwrapStrategy` enum or just a closed-over function: `_unwrap = _make_unwrap(no)`. This is the right CORRECT abstraction — not collapsing factories, but eliminating the runtime branch inside each closure.

---

## Cross-cutting 3: `Construct._normalize_fan_out_params` + `_normalize_oracle_gen_type`

**Files:** `src/neograph/construct.py:163-218`.

**Verdict: UNDER-ABSTRACTED.**

The two methods are sibling inference passes that run in `Construct.__init__` to make YAML/programmatic surfaces produce the same IR as the `@node` surface. They're labelled "IR-parity normalization" (line 152). Each method walks `self.nodes`, finds nodes matching a predicate (Each + dict-form inputs; Oracle + merge_fn), infers a field value, and `model_copy`s the node with the update.

**Rule of three:** today there are TWO normalizers. If a third inference rule appears (e.g. inferring `context=` from a model field that matches a peer node's output name), it WILL be a third sibling method, and the pattern continues drifting toward ad hoc.

**Concrete-first:** these methods exist BECAUSE the decorator surface has decoration-time hooks the other surfaces don't. The pattern is "IR convergence pass." That's a real concept worth naming.

**Substitutability:** the two methods today are NOT substitutable as written (different predicates, different inference rules), but they share an EXACT shape: walk nodes, filter, derive value, `model_copy`. This is the textbook "registered strategy" pattern.

**Recommendation:** introduce `_IrParityPass` as a small Protocol or callable, with two registered passes (`fan_out_param_pass`, `oracle_gen_type_pass`). `Construct.__init__` iterates the registry. Adding a third pass is a one-line registration. The two methods today are tightly bound to `Construct`'s internals (`model_copy`, `peer_field_names`), so the pass interface needs `(self, peer_field_names)` access. A Protocol with `apply(construct) -> None` works.

This is the canonical case of "the abstraction is missing." Today's code couples the two passes to the constructor; tomorrow's third pass will either join them there (worse) or live elsewhere with reduced visibility (also worse).

---

## Cross-cutting 4: `Node._sidecar` PrivateAttr

**Files:** `src/neograph/node.py:198-204`, `src/neograph/_sidecar.py:1-144`.

**Verdict: EARNS-KEEP.**

**Rule of three:** the sidecar carries (a) the original function for shim construction, (b) the param-name tuple for adjacency, (c) the DI bindings for resolution, (d) the scripted shim closure for compile() lookup. Four distinct downstream consumers (`_construct_builder` for shim, `_construct_builder` for adjacency, factory for DI, compiler for scripted-fn dispatch).

**Concrete-first:** the sidecar was extracted to `_sidecar.py` AFTER a circular-import bug (decorators ↔ _construct_builder). The extraction is documented in CLAUDE.md as the "Storage lives in `_sidecar.py`" rule.

**Why PrivateAttr is correct:** the alternative is a global `dict[id(node), SidecarData]` keyed by node identity. Global state breaks Pydantic's `model_copy` semantics (the new node loses its sidecar). PrivateAttr is preserved by `model_copy` (Pydantic v2 copies `__pydantic_private__`). This is exactly what the property is for. The cost (Pydantic's schema bypass) is real but bounded: only Node carries it, and `arbitrary_types_allowed` is already enabled there.

**Substitutability:** the sidecar pattern is the inverse of inheritance — you attach metadata to instances rather than subclassing. For this use case (metadata from a decorator that runs at import time, must survive `|` operations that produce new instances), PrivateAttr is the cleanest choice in Python. Alternatives (weakref dict, monkey-patched attr, sealed sentinel class hierarchy) all worsen one of: GC behavior, type-safety, or model_copy compatibility.

**Recommendation:** leave alone. The CLAUDE.md documentation of "why PrivateAttr, not proper fields" is exactly the kind of decision record that prevents regression.

---

## Cross-cutting 5: Validator-as-recursive-function with `ambient_producers` parameter

**Files:** `src/neograph/_construct_validation.py:253-344`.

**Verdict: EARNS-KEEP.**

**Rule of three:** the validator is one function with one parameter (`ambient_producers`) that captures parent context for nested sub-construct validation. The parameter exists because inner-node `context=` references must resolve against the union of (ambient parent producers + locally-collected producers). Without the parameter, sub-construct validation would either (a) reject all `context=` references it can't resolve locally (false positives), or (b) defer all context validation to the top level (less localized error messages).

**Concrete-first:** the parameter was added when a real bug appeared — sub-constructs validating in isolation rejected `context=` references that the parent had. The recursive form with ambient-passing was the surgical fix.

**Substitutability question — should this be split?** Two callables would be `validate_root_construct(c)` and `validate_nested_construct(c, ambient)`. Both would internally walk the same node list with the same producer-registration logic. That's 90% code duplication. The single function with an optional parameter is correct here — the variation is in ONE detail (which producers count as ambient), not in the algorithm shape.

The maintainer's pushback is directly relevant: "collapsing N polymorphic classes into one function with branching makes complexity WORSE." But that's not the situation here. This is one ALGORITHM with one OPTIONAL parameter. Splitting would be the inverse anti-pattern: duplicating an algorithm to avoid an optional param.

**Recommendation:** leave alone. If the function grows past ~300 lines or accumulates a third optional parameter, consider extracting a `_ChainValidator` class that holds `(construct, ambient_producers, producers)` as state and exposes `validate()`. Today's function form is fine.

---

## Cross-cutting 6: Per-compile registries (`scripted`, `conditions`, `tool_factories`)

**Files:** `src/neograph/compiler.py:66-140`, `src/neograph/decorators.py:86-103`.

**Verdict: UNDER-ABSTRACTED.**

`compile()` accepts three structurally identical `dict[str, Callable]` kwargs and does the same merge for each:
1. Seed from `_decorator_scripted` / `_decorator_conditions` / `_decorator_tool_factories`
2. Merge in caller's explicit kwarg (caller wins)
3. Validate (different rule per registry — e.g. tool factories validate against `Node.tools`)
4. Thread through the compile path as a `dict[str, Callable]` parameter

Every factory closure signature carries three `_lookup` parameters that flow together. The decorator-side registries are three parallel module-level dicts with three parallel `register_*` helpers (`decorators.py:91-103`).

**Rule of three:** three identical-shape registries IS the rule of three. The pattern is fully concrete.

**Concrete-first:** the three were added incrementally (scripted first, then conditions for Loop/Operator, then tool_factories for the @tool decorator). Each addition copy-pasted the previous registry's pattern.

**Substitutability:** the three registries are GENUINELY substitutable in their core operation (merge, lookup, validate-key-present). The only variation is the validation rule, which is per-registry. A `CompileRegistry` class with `name: str`, `decorator_dict: dict`, `kwarg_dict: dict | None`, and a `validate(construct) -> None` method would consolidate the merge logic and let the compiler iterate them.

```python
@dataclass
class CompileRegistry:
    name: str  # "scripted" | "conditions" | "tool_factories"
    decorator_dict: dict[str, Callable]
    user_dict: dict[str, Callable] | None = None
    def merged(self) -> dict[str, Callable]: ...
    def validate(self, construct: Construct) -> None: ...  # registry-specific
```

The compiler would build `[ScriptedRegistry(...), ConditionsRegistry(...), ToolFactoryRegistry(...)]` and thread them as one parameter (`registries: dict[str, CompileRegistry]`).

**Cognitive cost vs payoff TODAY:** every new closure factory has to grow three `_lookup` parameters; every new top-level helper that needs them has to plumb three. The fanout is real. Three (going to four if telemetry-callbacks ever become registry-based) is the inflection point.

**Reversibility:** the current form WORKS. Refactoring is a quality improvement, not a bug fix. But the under-abstraction will keep duplicating the merge logic at every new addition.

**Recommendation:** introduce `CompileRegistry`. Replace the three kwargs with a `registries=` dict (or keep the kwargs and build the dict internally). Closure signatures shrink from three lookups to one `registries` parameter. The decorator-side dicts become `CompileRegistry.decorator_dict` fields on three module-scoped instances. The structural-guard verifying decorator-dict usage stays valid (renamed).

This is the textbook case of UNDER-abstraction: the pattern is fully repeated three times, yet there's no type for it. The maintainer's pushback against "replace classes with dict-branching" doesn't apply here — the recommendation is the OPPOSITE direction: replace three parallel dicts with three instances of one type.

---

## Patterns across the codebase

**Consistently EARNS-KEEP:**
- Frozen dataclass bundles for "this set of args flows together as one parameter" (`LlmRuntime`, `Producer`, `NormalizedOutputs`, `NormalizedInputs`, `Spec`). All have ≥3 consumers and zero hidden state.
- Protocol-based polymorphism where implementations have non-trivial logic (`StateBus`, `ConstructItem`, the various `SkipPredicate`/`RawNodeFn` shapes). Each protocol has multiple genuine implementations or covers a structurally diverse set.
- Per-compile-time variation captured in closures (factory layer). Two `compile()` calls produce independent graphs because nothing escapes to module scope. The `EMPTY_RUNTIME` sentinel is the right Null-Object companion.
- Inference passes that bridge the three API surfaces. The IR convergence is enforced at one point (`Construct.__init__`) regardless of surface.

**Consistently OVERWROUGHT:**
- Content-based recovery flows scattered across dispatch sites (the DSML recovery in `_llm_dispatch.py` + `_llm_retry.py` + `_tool_loop.py`). Detection is correctly centralized (the `_DSML_PATTERN` constant); the recovery flow is not.
- Runtime branching on shape-info that's already known at IR-build time (the closure bodies in `_oracle.py` deciding single-vs-dict form on every dispatch).

**Consistently UNDER-ABSTRACTED:**
- Parallel-shaped registries with one-off merge/validate logic (the three compile() registries).
- Sibling inference methods on `Construct.__init__` (today two; will grow).
- Both cases share a signature: "we have N instances of the same structural pattern, and N is ≥2 today, and there's no named type."

---

## Recommendations

### STRENGTHEN abstractions (introduce new types)

1. **`CompileRegistry`** for the `scripted` / `conditions` / `tool_factories` triple. Consolidates merge + validate + thread-through logic. Three instances replace three parallel dicts. Closure signatures shrink.
2. **`_IrParityPass` Protocol** for the two `Construct._normalize_*` methods. Registry-based iteration in `Construct.__init__`. Third pass is a one-line addition.
3. **`DsmlGuard` class** in `_llm_retry.py`. Absorbs the three call sites in `_call_structured` and `_tool_loop` that currently inline the "detect → targeted retry → fall back to generic retry" sequence.

### LIGHTEN abstractions (eliminate runtime branching that IR already knows)

4. **Precompute unwrap strategy in `_oracle.py` factories.** Pass `NormalizedOutputs` into `make_oracle_redirect_fn` / `make_eachoracle_redirect_fn` and select the right unwrap closure at factory time rather than on every dispatch. Same factory count; thinner closure body.

### LEAVE ALONE (earning their keep)

5. `LlmRuntime`, `EMPTY_RUNTIME` sentinel, the `llm_factory` indirection.
6. `StateBus` Protocol + two adapters.
7. `normalize_outputs` / `normalize_inputs` + their structural guard.
8. Two-level fingerprinting + `_compute_invalidated_nodes`.
9. The three API surfaces + IR convergence in `Construct.__init__`.
10. `Node._sidecar` PrivateAttr pattern.
11. The retry concern split (transient / output-quality / in-node).
12. The validator-as-recursive-function with `ambient_producers` parameter.
13. The seven test fakes in `tests/fakes.py`.
14. (Mildly) the observability contract file — keep until the event count crosses ~10, then migrate to AST guard.

### Tighten without restructuring

15. `NormalizedOutputs.primary: Any` → `TypeSpecStatic`. Matches §5 discipline.
16. Document the `EMPTY_RUNTIME` Null Object pattern at `_llm_runtime.py:93` so future readers don't simplify it away.
17. Cross-reference `compute_node_fingerprints` (state.py) ↔ `_compute_invalidated_nodes` (runner.py). The pair conceptually belongs together; the split is acceptable but should be linked in docstrings.

### What we did NOT recommend

- Replacing any polymorphic class hierarchy with a function-with-branching.
- Removing type-safe wrappers that have only one variant TODAY (the wrappers carry semantic content — e.g. `EMPTY_RUNTIME` vs `None`).
- Collapsing the three API surfaces into one. They serve genuinely different users (humans / runtime construction / config-driven).
- Collapsing the three retry concerns into one knob. The layering reflects real failure-model differences.
