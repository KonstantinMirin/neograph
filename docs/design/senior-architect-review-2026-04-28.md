# Senior Architect Review

**Date**: 2026-04-28
**Reviewer**: External-style review by an opus-class agent, briefed to act as a senior Python architect with no political stake. Maintainer requested blunt, direct feedback over agreement.
**Codebase state at review**: develop @ post-ianf epic close (1705 tests passing, lint+mypy clean, 37 source modules)
**Recent context**: epic `neograph-ianf` (typed `LlmConfig` IR, `Renderer` Protocol, callback Protocols, Pydantic spec schema, RunnableConfig consistency) just landed.

This document is preserved verbatim from the reviewer's output. Treat it as the source of truth for architectural-debt prioritization decisions until the next review supersedes it.

---

## Executive verdict

This is a serious 0.x library that has done one thing genuinely well — turning typed Python signatures into a validated DAG with assembly-time type checking — but it is built on top of LangGraph in a way that smears the host framework everywhere, carries a non-trivial amount of module-level mutable state, and has a layered architecture whose cleanness is enforced by structural test guards rather than by the import graph itself. It is production-shaped for a single consumer (piarch) under a single maintainer; it would not pass a hiring-bar architecture review at a serious infra shop today, and the gap is mostly fixable.

---

## Strengths (cited and specific)

1. **Assembly-time validator with three-surface parity.** `effective_producer_type` in `src/neograph/_construct_validation.py:41` as the single producer-side rule, plus `_check_fan_in_inputs` at `:520`, is the right shape. It's the rare case where the abstraction actually pays rent — fixing a modifier rule in one place makes all three API surfaces correct. The `# CHECK_ERROR:` rustc-style fixtures in `tests/check_fixtures/` driven by `tests/test_check_fixtures.py` are unusual for a Python project and the right way to test a validator.

2. **Typed error model with a builder.** `errors.py:NeographError.build` and the structural guard at `tests/test_structural_guards.py:31` (`TestErrorBuilderEnforcement`) that AST-scans every `raise` statement to make sure nobody bypasses `.build()` is exactly the kind of cheap permanent guardrail I rarely see enforced. Good signal that the maintainer has internalized the "guard-first" lesson.

3. **The `ModifierCombo` enum + `assert_never` exhaustiveness pattern** at `modifiers.py:49` and `factory.py:225` is principled. Adding a new modifier combination forces a `match` exhaustion failure at every dispatch site. This is a pattern most Python codebases skip; the type narrowing is real.

4. **Schema fingerprinting with auto-resume** (`runner.py:75`) is clever and the right shape for the problem. Per-node fingerprints in `state.py` mean "I changed prompt text" doesn't invalidate the whole pipeline. Most LangGraph users hit this exact wall and have no answer.

5. **`Node.inputs`/`Node.outputs` plural vs `Construct.input`/`output` singular** is a thoughtful naming choice — the structural difference (fan-in to a node vs single boundary port for a sub-construct) is real and the names reflect it. I would have named them differently, but it's principled.

That's the real list. Most "strengths" you'd find in a glowing review (test counts, lint cleanliness, mypy clean) are table stakes, not strengths.

---

## Weaknesses (the meat — ordered by severity)

### 1. Module-level mutable state for LLM configuration is the single biggest design wart

`_llm.py:98-109` declares six module-level mutables:

```python
_llm_factory: LlmFactory | None = None
_llm_factory_params: set[str] | frozenset[str] = set()
_prompt_compiler: PromptCompiler | None = None
_prompt_compiler_params: set[str] | frozenset[str] = set()
_global_renderer: Renderer | None = None
_cost_callback: CostCallback | None = None
```

`configure_llm` at `:226` uses a six-name `global` statement to mutate them. Every test that touches an LLM has to either use a fixture to reset state or share state with every other test. This is the reason `tests/conftest.py` exists, the reason `_get_global_renderer` returns module state from inside `factory._dispatch.py:226`, and the reason `compiler.py:130` defers an import to `from neograph._llm import _llm_factory, _prompt_compiler` to peek at module state for graph fingerprinting.

What a senior engineer would do: thread an `LlmConfig`-shaped runtime context object through `compile()` (which is where most consumers configure once anyway), or attach it to `RunnableConfig["configurable"]` (consistent with the rest of the framework). Making this a singleton was expedient; making it not a singleton would let two pipelines coexist in one process with different LLM configs, would let tests not need a "reset" fixture, and would enable a backend-neutral story (today, swapping the LangGraph backend means deciding what to do with these globals). The `_registry.py` module already shows that the maintainer knows how to do this — the registry was migrated out of three module-level dicts in `factory.py`. The same migration has not happened for `_llm.py` and it's the more important one.

The runtime introspection of `_accepted_params` (`_llm.py:161`) is the consequence of this: the framework has to inspect the user's callable signature at registration time to decide what kwargs to pass, because there's no typed handshake. The `Protocol`s added in the recent ianf cleanup don't actually constrain the call shape — they declare `def __call__(self, tier: str, *args, **kwargs)` to keep both old and new shapes working. The "typed" Protocol is structural-typing theater; the runtime still introspects.

### 2. Deferred imports inside function bodies are a smell that point to circular dependencies you haven't actually removed

53 function-local `from neograph...` imports across `src/neograph/`. A sample:

- `_oracle.py:24,29` — `from neograph.factory import _state_get`, `lookup_scripted` — at every call. Reason: `_oracle.py` is imported by `factory.py`, so `_oracle.py` cannot import from `factory.py` at module scope.
- `_dispatch.py:129,167,168,177,226,269` — six function-local imports, including imports inside `ThinkDispatch.execute` and `ToolDispatch.execute` — i.e. on the hot path. `_get_global_renderer` is wrapped in a `try/except ImportError` at `_dispatch.py:179`, defending against a circular import that "shouldn't happen" — that's a clear "I don't trust the import graph" tell.
- `decorators.py:152,587,731` — three deferred imports of `factory.register_scripted`.
- `compiler.py:130` — `from neograph._llm import _llm_factory, _prompt_compiler` to read module-level state for graph fingerprinting.

The CLAUDE.md mentions a "deferred import budget of ~45." It is now 49+. A budget for a workaround is not the same as fixing the workaround. What a senior engineer would do: invert the dependency. The two real cycles are `factory ↔ _oracle` and `decorators ↔ factory ↔ _construct_builder`. Both can be broken by extracting a small `_runtime_protocols.py` (or just `_dispatch_types.py`) module that all three import from. The maintainer already did this for `_sidecar.py` to break `decorators ↔ _construct_builder` — same recipe applies. The fact that this isn't done suggests the cycles aren't fully understood, or that the cost of the deferred-import idiom is being underestimated.

`_dispatch.py:177-181`'s `try/except ImportError` to import a sibling module is the canonical anti-pattern. There is no scenario where `_llm` legitimately fails to import here; the catch is defensive boilerplate against a phantom cycle. Either the import should always succeed (move it to module top), or the dependency is wrong (and should be inverted). Not both.

### 3. `factory.py` and `_llm.py` are god-modules with cross-cutting responsibilities

`factory.py` at 631 lines is the central dispatch: `make_node_fn`, `make_subgraph_fn`, `make_oracle_redirect_fn` (re-exported from `_oracle`), `_extract_input`, `_build_state_update`, `_apply_skip_when`, `_inject_oracle_config`, `_state_get`, `_extract_context`. That's six axes: factory construction, state I/O, modifier I/O, oracle plumbing, observability, skip logic. The recent file-split work (`_dispatch.py`, `_oracle.py`) extracted the dispatch protocol and oracle functions, which helped — but the structural-guard rules at `tests/test_structural_guards.py:84-107` exist precisely because this stuff drifts back. If you need an AST guard saying "don't put `make_oracle_redirect_fn` in `factory.py`", the right answer is that the boundary is too leaky: the function names should not be ambiguous about which file they belong to. They are because the responsibilities aren't crisp.

`_llm.py` at 901 lines mixes (a) the global-state singleton, (b) `Protocol` definitions, (c) `_call_structured` strategy dispatch, (d) JSON parsing + retry, (e) `render_prompt` (a public introspection function), (f) `invoke_structured`. The recent `_tool_loop.py` extraction took the agent loop out of `_llm.py`, which was right. A second pass should split it again: `_llm_protocols.py` (the Protocols + global state), `_llm_dispatch.py` (`_call_structured`, retry), `_llm_render.py` (`render_prompt` + `_compile_prompt`).

### 4. Type-system usage is principled in places but cargo-culted in others

`Any` count in the load-bearing modules: `_wiring.py` 27, `_oracle.py` 11, `factory.py` 17, `_construct_validation.py` 11. Some are unavoidable (state is `dict | BaseModel`, the user can declare any type as input/output). But many are not: the `state: Any` pattern at `factory.py:54,61,79` exists because `_state_get` was written before the state model was typed. `compile_state_model` produces a typed Pydantic model (`state.py:81`) — passing it through as `Any` everywhere downstream throws away the type information.

The `TypeSpec = Annotated[Any, PlainValidator(_validate_type_spec)]` at `node.py:105` is honest about the limitation (no PEP 747 TypeForm yet) but the entire `inputs: TypeSpec` field is statically `Any`, so mypy can't help anywhere `Node.inputs` is consumed. Combined with the dict-form / single-type / None polymorphism (`isinstance(node.outputs, dict)` checks scattered across 8 files), this is one of the more error-prone parts of the codebase. A `Inputs = type | DictInputs | None` discriminated union with `DictInputs = NewType("DictInputs", dict)` would let mypy at least check the polymorphism arm at use sites.

`arbitrary_types_allowed=True` appears on `Node`, `Construct`, `ForwardConstruct`, and `Oracle`. Three of those are because Pydantic doesn't know how to validate `Renderer | None`, `Callable | None`, and the `_BranchNode` sentinel — fair. But `Oracle` (`modifiers.py:376`) needs it for `Callable` fields, which is the kind of thing that, taken together with the function-as-a-string dispatch in `register_scripted`, suggests the IR is unsure about whether functions are first-class IR values or names that resolve via a registry. (The maintainer is aware — `_sidecar.py` exists exactly because of this tension.)

### 5. LangGraph coupling is not quarantined despite the stated goal

CLAUDE.md says the long-term direction is "abstract away from LangGraph, support multiple backends + TypeScript companion" and `factory.py` says raw mode is "the classic LangGraph escape hatch." The actual coupling, however, is everywhere:

- `langchain_core.runnables.RunnableConfig` is in 8 modules (`factory`, `runner`, `_llm`, `_dispatch`, `_oracle`, `_wiring`, `node`, `_tool_loop`). Two of these (`node.py:28`, `_oracle.py:14`) are in the IR layer — meaning the IR itself imports from langchain_core.
- `langgraph.graph.{END,START,StateGraph}` is in `compiler.py` and `_wiring.py` (expected) but `langgraph.types.{Send, interrupt}` is referenced inside `_wiring.py` and `langgraph.types.Command` inside `runner.py`. The runner is supposed to be the user-facing entry point — and it imports `Command` to handle resume.
- `compiler.py:51` reaches into `langgraph._internal._serde` (note the underscore — private API) for `build_serde_allowlist`. There's a defensive `try/except ImportError` at `:62` but the comment "LangGraph version doesn't have this API" is doing a lot of work. When LangGraph 1.0 ships and renames or removes `_serde`, this silently degrades.

If the goal is multi-backend, the IR layer (`node.py`, `construct.py`, `modifiers.py`) needs to not import `RunnableConfig`. Today, `Node.raw_fn: RawNodeFn | None` (`node.py:60`) defines `RawNodeFn` as a Protocol typed against `RunnableConfig` — the IR knows about LangGraph. That's a load-bearing assumption that breaks the multi-backend story before it starts.

What a senior engineer would do: introduce a `NeoConfig` (already half-done — `_llm_config.py:LlmConfig` is the right shape) that the IR sees, and translate to `RunnableConfig` at the LangGraph boundary in `compiler.py` / `runner.py`. The runtime user-supplied callback Protocols (`LlmFactory`, `PromptCompiler`) should also see `NeoConfig`, not `RunnableConfig`.

### 6. Tests are mostly behavior-focused, but `test_llm_internals.py` and the parametrize-everything pattern hide some implementation-coupling

`test_validation.py` reads well — it tests "this Construct should fail to assemble with this message" which is the right abstraction. Hypothesis tests in `tests/hypothesis/` are substantive (1348 lines of topology generation in `test_topologies.py`) — that's not checkbox usage.

But `test_llm_internals.py` at 4416 lines is doing a lot of mock plumbing. Sample at `:836`: a fake R1 model defined inline, with `bind_tools` returning `self`, with content carefully crafted to look like XML — testing one specific bug. Many of these tests are pinned to the exact internal invocation path: `call_n["n"]` counting, asserting `assemble_verify -> summary` edge presence (`:821`). That's testing the exact LangGraph topology produced — fine until LangGraph changes its internal naming conventions. The tool-loop test count (`tests/modes/`) ballooned during the recent tool_loop coverage epic; some of that coverage is "did the mock get called" rather than "is the user-observable behavior right."

The structural guards at `test_structural_guards.py` are good (the AST-scan-for-bare-`raise` pattern is the right shape) but `TestFileSplitEnforcement` at `:76` is suspicious — the rule "don't put `make_oracle_redirect_fn` in `factory.py`" is enforced because, presumably, someone keeps adding it back. If the guard is permanent, the underlying boundary is wrong, not just the placement.

### 7. Smaller things worth noting

- `runner.py:265` — `input["neo_schema_fingerprint"] = fp` mutates the caller's input dict. That's surprising; a defensive copy would be standard.
- `factory.py:54` — `_state_get(state, key)` does `getattr(state, key, None)` — silently swallowing missing fields is exactly the kind of thing that causes "ghost state" bugs. The schema fingerprinting protects against schema changes between runs, but not against a typo in `_state_get` calls inside a single run.
- `_construct_validation.py:323` — `import structlog` inside a function body. If structlog is in `pyproject.toml`, just import at module top. If it isn't, don't depend on it.
- `node.py:300-304` — `Node.run_isolated` returns `result.get(field_name)` without raising if the field is missing. Silent `None` returns make for mysterious test failures.
- `_construct_builder.py:30` and several other modules import from `neograph.naming` for `field_name_for` — this single helper threads through the whole codebase as a string-based naming convention. Magic strings (`"neo_oracle_gen_id"`, `"neo_each_item"`, `"neo_loop_count_*"`, `"neo_subgraph_input"`) appear in `factory.py`, `_oracle.py`, `_wiring.py`, `state.py`, `runner.py`. There's no central enum or constants module. A typo in any of these is a runtime bug that no test guards.

---

## Risk assessment

1. **LangGraph 1.0 will break the codebase.** `compiler.py:51` uses `langgraph._internal._serde` (underscore-prefixed = private). `_wiring.py` uses `Send`, `interrupt`, the `defer=True` flag on `add_node` (`:78`), and `path_map=` — all of these have changed in LangGraph minor versions. The defensive `try/except ImportError` at `compiler.py:62` swallows the future failure silently rather than warning. When the underlying API changes shape, the symptom will be "msgpack warnings on resume" — not a clear error.

2. **Schema fingerprinting is correct in spirit but the implementation is fragile.** `_compute_invalidated_nodes` (`runner.py:158`) returns the directly-changed nodes but the comment at `:191` says "DAG walking requires construct access which isn't available on the compiled graph. The changed set is sufficient." That's not sufficient — if node B reads from node A's output and A's fingerprint changes, B's fingerprint may not change but B's behavior does. The transitive-descendants comment acknowledges this gap and punts. In production, this is a cache-invalidation bug class waiting to happen.

3. **Module-level state means concurrent pipelines can't coexist.** Two `compile()` calls in the same process with different `configure_llm` configurations will silently share the last config. For a library positioned as "production-grade agents on LangGraph," this is a real limitation — multi-tenant servers that compile pipelines per request will either need per-request `configure_llm` calls (race condition) or per-process pools.

4. **The `@node` body-as-merge fallback path in `decorators.py:151` is genuinely scary.** The decorator detects `models=` without `merge_fn`/`merge_prompt`, warns, and synthesizes a registered scripted function whose name includes `id(f):x` — i.e., depends on Python object identity. If the function is ever copied (via `model_copy`, decorator chaining, etc.), the name diverges from the registered shim. PrivateAttr survives `model_copy` (per CLAUDE.md), but the registered name is keyed by `id(f)` at decoration time, not at copy time. There's no test for this boundary that I saw.

5. **The LLM call retry path at `_llm.py:706-715` mixes two retry strategies** — `max_retries` from `llm_config` and the framework's own JSON-parse retry. If both fire on a transient failure, the user gets `max_retries × json_retry` total calls. This is documented nowhere.

---

## Recommendations (prioritized for a solo maintainer with one downstream consumer)

1. **Eliminate `_llm.py` module-level mutable state in the next minor.** Migrate to a `LlmRuntime` object passed through `compile()` and stashed on the compiled graph. This is the single highest-leverage refactor. It unblocks (a) backend abstraction, (b) test isolation, (c) multi-tenant. Cost: ~2 days. Risk: medium (touches every LLM-calling test). The `_registry` migration is the template.

2. **Inventory the deferred imports and break the actual cycles.** Pick the top three (`_oracle ↔ factory`, `decorators ↔ factory`, `_dispatch ↔ _llm`) and extract a small `_runtime_protocols.py` that all three import from. Delete the function-local imports. Ban new function-local imports via a structural guard. Cost: ~1 day. The deferred-import budget should go down to single digits.

3. **Define a `NeoConfig` at the IR layer and translate to `RunnableConfig` at the boundary.** Stop importing `RunnableConfig` from `node.py`, `_oracle.py`, `_construct_validation.py`. The IR should not know LangGraph exists. Cost: ~3 days but most of it is mechanical. This is a precondition for the multi-backend / TS-companion story.

4. **Constants module for `neo_*` magic strings.** `state.py` is the natural home — define `class StateKeys` with `EACH_ITEM = "neo_each_item"` etc. and import the constants. A typo guard test (grep-based) checks no `"neo_*"` string literals exist outside `state.py`. Cost: half a day, eliminates a real bug class.

5. **Tighten the `Node.outputs` polymorphism.** `outputs: type | dict[str, type] | None` is duck-typed across 8+ files via `isinstance(outputs, dict)`. Either introduce a sealed wrapper class (`SingleOutput(t)` vs `MultiOutput({...})`) or live with the polymorphism but centralize the discrimination in one helper that returns a tagged result. The existing `_resolve_primary_output` (`_dispatch.py:239`) is the start of this — keep going.

6. **Don't bother with backwards-compat shims.** The maintainer has stated this; honor it. The `LlmRuntime` migration above can ship as a hard 0.5.0 break. There is one consumer.

7. **Decide whether you're a LangGraph DX layer or a competing graph engine.** Right now neograph reads as the former in the IR but advertises the latter in CLAUDE.md ("backend-neutral"). The two require different architectures. If multi-backend is real, the dependency-quarantine work is unavoidable. If it isn't, drop the goal from CLAUDE.md and own being a LangGraph DX layer — that's a perfectly defensible product.

---

## Where this sits relative to peers

Versus **Hamilton** (Stitch's typed DAG): Hamilton is more mature on the typed-DAG-from-functions axis — it's been doing function-signature-as-dependency for years and its config injection (via `@config.when`) is more refined than `FromInput`/`FromConfig`. Neograph is materially better on the LLM-orchestration axis (Hamilton has no built-in agent loop, no Oracle ensembling, no tool budgets) and on the fan-out/fan-in story (Hamilton's parallel execution is coarser). For pure data pipelines, Hamilton; for LLM pipelines that need typed structure, neograph.

Versus **BAML**: BAML is a different beast — a separate language for prompt+output-schema definition, transpiles to multiple host languages. Neograph's `describe_type` is a (much smaller) subset of what BAML does for schema-rendering; the rest of neograph (graph topology, modifiers, fan-out) is out of BAML's scope. They're complementary and the BAML-as-renderer integration would be more interesting than the current XmlRenderer/JsonRenderer/DelimitedRenderer trio.

Versus **DSPy**: DSPy optimizes prompts; neograph compiles graphs. Different layer of the stack. Neograph's typed contracts give you something DSPy doesn't — assembly-time validation. DSPy gives you something neograph doesn't — automatic prompt search. They could coexist (neograph as the topology layer, DSPy as the per-node prompt optimizer).

Versus **Burr**: Burr (DAGWorks) is the closest direct competitor — typed state machine on top of an arbitrary backend. Burr is more disciplined about backend independence (it has a real backend abstraction with multiple implementations); neograph is more LangGraph-coupled but has a richer modifier vocabulary (Each×Oracle fusion, Loop-on-Construct). Burr's docs are better; neograph's compile-time validation is stronger.

Versus **LangChain Expression Language (LCEL) + LangGraph raw**: this is the actual baseline and the one the positioning is staked on. Neograph wins handily on type safety and the "pipeline reads like Python signatures" front. It loses on community size, debugger tooling, and the inertia of being native LangGraph.

The honest verdict: neograph is doing one thing — assembly-time typed graph compilation for LLM pipelines on top of LangGraph — that nothing else does as well. That's a real moat. The architecture below the IR layer needs another pass before the moat is defensible against a single LangGraph 1.0 release.
