# Changelog

All notable changes to NeoGraph will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
