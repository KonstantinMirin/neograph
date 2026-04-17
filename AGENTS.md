# AGENTS.md

Project-specific context and operational workflow for agents working on neograph. This file is the source of truth; `CLAUDE.md` is a symlink to it so every coding agent that supports either convention picks up the same content.

---

## What neograph is

A declarative LLM graph compiler on top of LangGraph. You declare a pipeline as typed Python functions; neograph infers the DAG from parameter names, validates types at assembly time, and compiles to a LangGraph `StateGraph` with checkpointing, observability, and tool orchestration.

**Positioning**: "the fastest way to build production-grade agents on LangGraph." Typed end-to-end, durable, observable, focused on the logic — not the wiring.

**Website**: [neograph.pro](https://neograph.pro) (Astro + Starlight at `website/`, deployed via AWS Amplify on every push). Don't forget to update website content when API surfaces change.

---

## Operational: beads workflow

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

### Quick reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

### Landing the plane (session completion)

**When ending a work session, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.**

1. **File issues for remaining work** — create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) — tests, linters, builds
3. **Update issue status** — close finished work, update in-progress items
4. **PUSH TO REMOTE** — this is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** — clear stashes, prune remote branches
6. **Verify** — all changes committed AND pushed
7. **Hand off** — provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing — that leaves work stranded locally
- NEVER say "ready to push when you are" — YOU must push
- If push fails, resolve and retry until it succeeds

---

## Three API surfaces, one compiler

This is the most important architectural fact. All three produce the same internal IR (`Construct` with `.nodes: list[Node | Construct]`) and compile through the same `compile()` path.

| Surface | When to use | Where it lives |
|---|---|---|
| `@node` decorator | Default for humans writing source code. Functions are nodes, parameter names are edges. | `src/neograph/decorators.py` |
| `ForwardConstruct` | Pipelines with Python control flow (`if`/`loop`/`try`). Class-based, `forward()` traced via symbolic proxies. | `src/neograph/forward.py` |
| `Node` + `Construct` + `\|` pipe | Runtime construction by LLMs, config systems, routing layers. Programmatic composition. | `src/neograph/node.py`, `construct.py`, `modifiers.py` |

**All three coexist. The programmatic form is not legacy.** It is the primary path for runtime-assembled pipelines (e.g., LLM tool-calling builds a spec, the runtime constructs `Node`s and pipes modifiers, `compile()` + `run()`). See `website/src/content/docs/runtime/llm-driven.mdx` for the documented use case.

---

## `Node.inputs`: fan-in shape for all three API surfaces

`Node.inputs` is a `dict[str, type]` mapping upstream-name → expected-type. Every API surface — declarative, `@node`, and programmatic/runtime — produces the same dict shape, so a single validator walks all three:

| Surface | How it produces `inputs` |
|---|---|
| Declarative `Node(...)` | Author writes `inputs={'claims': Claims, 'scores': Scores}` (or a single type for backward-compat; the single-type form skips fan-in validation and defers to runtime isinstance scan). |
| `@node` decorator | Decoration walks the function signature and emits `inputs={param_name: annotation}` for every typed upstream param. Fan-out (`map_over`) receivers, DI params (`FromInput`/`FromConfig`), and default-value constants are stripped at construct-assembly time. |
| Programmatic / runtime | Same dict shape as declarative. LLM-driven pipelines serialize to JSON with string type names and resolve them via a type registry. |

**One validator walker, not two.** `_validate_node_chain` in `_construct_validation.py` handles every surface. When `item.inputs` is a dict instance, `_check_fan_in_inputs` walks each `(upstream_name, expected_type)` pair and looks up the producer by `field_name`. Mismatches raise `ConstructError` with the specific key that failed and the type it saw vs expected.

**The producer side is shared.** `effective_producer_type(item)` in `_construct_validation.py` computes "what type does this node write to the state bus, accounting for modifiers". It's the single source of truth for modifier-aware type effects.

**Rule for new modifiers that reshape state**: teach `effective_producer_type` about the new rule. The validator picks it up automatically. Do NOT re-inline modifier checks elsewhere — prior drift caused `neograph-8k3` and `neograph-ayq` before this helper existed.

Current rules encoded in `effective_producer_type`:
- `Each` modifier → `dict[str, output]` (see `state.py:_add_output_field` for the state builder side of the same rule)
- Anything else → raw `output` unchanged

### `list[X]` consumers of `Each` producers (merge-after-fan-out)

A downstream node can consume an Each-modified upstream's fanned-out results as a `list[X]`:

```python
@node(output=Summary)
def summarize(verify: list[MatchResult]) -> Summary: ...
```

The validator (`_types_compatible`) accepts `list[X]` against a `dict[str, X]` producer when element types are compatible. At runtime, `factory._extract_input` (and the `@node` raw adapter for scripted @nodes) unwraps via `list(values())` before passing the list to the consumer.

**Ordering caveat**: `dict.values()` preserves insertion order, but Each's barrier collects `Send()` results in arrival order, not `each.over` collection order. Use `list[X]` for order-independent reductions (counts, aggregates, summaries). If you need deterministic ordering, consume as `dict[str, X]` and sort explicitly on the key.

### `Node.inputs` (plural) vs `Construct.input` (singular)

These are different fields with different roles — the naming is intentional, not a typo:

- **`Node.inputs`** (plural, `dict[str, type] | type | None`) — declares what a node *consumes* from the state bus. Dict form enables fan-in (multiple upstream producers); single-type form is a convenience shorthand.
- **`Construct.input`** (singular, `type[BaseModel] | None`) — declares the *boundary port* when a Construct is used as a sub-construct inside another Construct. It defines the isolated state the sub-pipeline receives, not a fan-in mapping.

The plural/singular split reflects the structural difference: a Node can consume from many upstream producers (hence a dict of inputs), while a sub-construct has exactly one typed entry point (hence a single input type).

---

## `Node.outputs`: N named outputs + agent tool context

`Node.outputs` mirrors `Node.inputs`. It is `dict[str, type] | type | None`:

| Form | Example | State fields |
|------|---------|-------------|
| Single type (backward compat) | `outputs=Claims` | `{node_name}` |
| Dict form (multi-output) | `outputs={"result": Claims, "tool_log": list[ToolInteraction]}` | `{node_name}_result`, `{node_name}_tool_log` |

**State model**: `compile_state_model` creates one field per output key. Each/Oracle modifiers apply independently per key.

**Validator**: `_validate_node_chain` registers one producer per output key. Downstream nodes reference upstream output keys via `{upstream}_{key}` naming in their `inputs` dict.

**Factory**: `_build_state_update` writes dict-form outputs to per-key state fields. For LLM modes, the first dict key is the "primary" output type passed to `invoke_structured`/`invoke_with_tools`. Secondary keys (like `tool_log`) are framework-collected.

**Gather tool collection**: `invoke_with_tools` collects `ToolInteraction(tool_name, args, result, duration_ms)` during the ReAct loop. When the node declares `"tool_log"` as an output key, the factory writes the interactions to the tool_log state field. Demand-driven: no collection overhead if no consumer references tool_log.

**@node decorator**: `@node(outputs={"result": X, "tool_log": list[ToolInteraction]})` passes through. Return annotation inference: `def f() -> X` infers `outputs=X` (single type). Parameters named `{upstream}_{output_key}` are resolved via `_resolve_dict_output_param` in `construct_from_module`.

### `Node.outputs` (plural) vs `Construct.output` (singular)

Same pattern as inputs/input:

- **`Node.outputs`** (plural, `dict[str, type] | type | None`) — declares what a node *produces* to the state bus. Dict form enables multi-output; single-type form is a convenience shorthand.
- **`Construct.output`** (singular, `type[BaseModel] | None`) — declares the *boundary port* when a Construct is used as a sub-construct. It defines what surfaces from the isolated sub-pipeline, not a multi-output mapping.

---

## Layer discipline

Do NOT add `@node`-specific logic to the low-level modules. The layering is:

```
User code
   │
   ▼
@node / ForwardConstruct / runtime Node | Modifier  ← DX layer (decorators.py, forward.py)
   │
   ▼
Construct(nodes=[...])                              ← IR layer (construct.py, _construct_validation.py)
   │
   ▼
compile()                                           ← Compiler layer (compiler.py, state.py)
   │
   ▼
factory._make_*_wrapper                             ← Runtime dispatch layer (factory.py)
   │
   ▼
LangGraph StateGraph
```

Concrete rules derived from this:

- **`node.py`, `construct.py`, `_construct_validation.py`, `factory.py`, `modifiers.py` are off-limits for @node-layer features.** The @node decorator is sugar over the IR; it must produce instances those modules already accept. This was a hard rule during the @node production-readiness epic and it paid off — every gap got fixed in `decorators.py` without touching the IR.
- **The only exception**: when a genuinely new IR capability is needed (e.g., `ForwardConstruct` needed `_BranchNode` sentinel support in `compiler.py` + `state.py`). Adding those was deliberate and documented.
- **Sub-constructs can be @node or declarative.** `construct_from_functions("verify", [explore, score], input=VerifyClaim, output=ClaimResult)` builds a sub-construct from `@node` functions. Params whose type matches `input=` are port params — they read from `neo_subgraph_input` instead of a peer `@node`. The declarative form `Construct(input=X, output=Y, nodes=[...])` also works. Both produce the same IR.

---

## DI surface (post-0.2): `Annotated[T, FromInput/FromConfig]`

**Breaking change from 0.1.x → 0.2.0.dev**. The old `FromInput[T]` Generic subscription form is gone. The new form uses `typing.Annotated` with `FromInput` / `FromConfig` as markers — the FastAPI `Annotated[User, Depends(...)]` pattern.

```python
from typing import Annotated
from neograph import node, FromInput, FromConfig

@node(output=Result)
def my_node(
    upstream:   Claims,                              # upstream @node
    topic:      Annotated[str, FromInput],           # from run(input={...})
    ctx:        Annotated[RunCtx, FromInput],        # BUNDLED (inner is a BaseModel)
    limiter:    Annotated[RateLimiter, FromConfig],  # shared resource
    max_items:  int = 10,                            # constant
) -> Result: ...
```

**Key rule**: if the inner type is a Pydantic `BaseModel` subclass, the resolver **bundles** — it constructs an instance by pulling each model field from `config['configurable']` under that field's name. Otherwise it does per-parameter lookup by the parameter's name. This is the only place the inner type affects resolution semantics.

**Why the bundle rule exists**: piarch had 60+ lines of boilerplate repeating `node_id: Annotated[str, FromInput], project_root: Annotated[str, FromInput], ...` across 20 nodes. Bundling a `RunCtx(BaseModel)` eliminates the repetition. See `neograph-6jd`.

**Classifier implementation notes** (`_classify_di_params` in `decorators.py`):
- Uses `typing.get_type_hints(f, localns=..., include_extras=True)` to preserve `Annotated` metadata.
- Walks the caller's frame stack (8 hops max) to capture locally-defined classes. Needed because `from __future__ import annotations` strips closure references, so `class RunCtx` defined inside a test method isn't findable via `f.__globals__` or `f.__closure__`. Pydantic uses the same technique for forward-ref resolution.
- `frame_depth=2` means: from inside `_classify_di_params`, frame 0 is the helper, frame 1 is `decorator(f)` inside `node()`, frame 2 is the user's call site. Anything deeper misses the decorated function's enclosing scope.

**Runtime resolution** (`_resolve_di_value` in `decorators.py`):
- `from_input` / `from_config` → read `config['configurable'][param_name]`
- `from_input_model` / `from_config_model` → construct the model by pulling each field from `config['configurable'][field_name]`
- `constant` → use the captured default value
- Unmatched → `None` passed (user code handles missing data)

Shared between `@node` raw adapters and `@merge_fn` wrappers. One resolver, one classifier, both decorators.

---

## `@node` sidecar pattern

`@node` stores the original function and its metadata on the Node via Pydantic `PrivateAttr` fields:

- `Node._sidecar: tuple[Callable, tuple[str, ...]] | None` — the original function and its parameter name tuple. Used at assembly time by `_construct_builder.py` to wire the DAG and build scripted shims.
- `Node._param_res: dict[str, DIBinding] | None` — DI bindings from `_classify_di_params`. Consumed at assembly time for shim construction and at lint time.

Both are `PrivateAttr(default=None)`, preserved by `model_copy` (Pydantic v2 copies `__pydantic_private__`). No global dicts, no `weakref.finalize`, no re-registration needed after `|` — `model_copy` handles it.

**Storage lives in `_sidecar.py`** (extracted from `decorators.py` to break the circular import). Import graph: `decorators.py → _sidecar.py ← _construct_builder.py` (one-way, no cycles). A structural guard test enforces that `_construct_builder.py` never imports from `decorators.py`.

**Why PrivateAttr, not proper fields**: the sidecar carries a `Callable` (the user's function), which can't go through Pydantic schema validation without `arbitrary_types_allowed` on every downstream consumer. PrivateAttr bypasses schema while staying on the Node instance.

**Why we keep the sidecar rather than eagerly resolving**: the sidecar carries **backend-neutral metadata** (the original function, param names, DI bindings). The LangGraph compiler consumes this to build scripted shims registered by string name. A future backend (direct execution, TypeScript transpilation) would consume the same metadata differently. Eagerly resolving to `scripted_fn` names bakes in LangGraph assumptions.

---

## `describe_type` / `describe_value` — LLM-facing schema rendering

`src/neograph/describe_type.py` (433 lines, 13 functions) renders Pydantic models into a TypeScript-style notation that LLMs parse more reliably than JSON Schema. Used by the factory layer to build structured output instructions.

**Two public functions** (both re-exported from `neograph`):
- `describe_type(model, prefix=..., hoist_classes=...)` — renders a model class into a schema string with auto-hoisted nested classes
- `describe_value(instance, prefix=...)` — renders a model *instance* as a typed value literal (for few-shot examples)

**Two-pass architecture**: pass 1 (`_count_classes`) counts how many times each nested class appears across the model tree. Pass 2 (`_render_model_body` / `_render_type`) emits the notation, hoisting classes that appear more than once (or all, per `hoist_classes=`).

**Handles**: primitives, `list[T]`, `dict[K,V]`, `Optional[T]`, `Union[A,B]`, `Literal[...]`, `Enum`, nested `BaseModel`, `tuple[...]`, forward refs, field descriptions, constraints, and defaults.

**Tests**: `test_renderers.py` — 88 tests covering all type combinations, edge cases, and round-trip parsing.

---

## RenderedInput — single rendering abstraction

`src/neograph/renderers.py` (dataclass at line 32). The single object that bundles all rendering artifacts for prompt construction. Produced by `build_rendered_input(input_data, renderer=None)`.

**Five fields**:
- `raw: dict[str, Any] | Any` — original Pydantic models, used by inline `${var}` prompts for dotted attribute access
- `rendered: dict[str, Any] | Any` — BAML-rendered strings, used by template-ref prompts via `prompt_compiler`
- `flattened: dict[str, Any]` — extra fields from `render_for_prompt()` BaseModel returns, available only in template-ref prompts
- `available_keys_inline: set[str]` — keys valid for inline `${var}` (raw dict keys only, no flattened, no framework extras)
- `available_keys_template: set[str]` — keys valid for template-ref `{var}` (raw + flattened + framework extras)

**`for_template_ref` property** — merges `rendered` and `flattened` dicts, with `rendered` keys taking precedence. This is what the `prompt_compiler` receives.

**Consumers**: `_dispatch.py:_render_input()` (mode dispatch layer) and `_llm.py:render_prompt()` (prompt inspection).

**The inline/template-ref split**: inline prompts (`${var}`) get `ri.raw` — raw Pydantic objects for `getattr` chains. Template-ref prompts get `ri.for_template_ref` — pre-rendered strings + flattened fields. Flattened fields and framework extras (`node_id`, `project_root`) are NOT available in inline prompts.

---

## Checkpoint resume — schema-aware auto-rewind

When a pipeline runs with a checkpointer and the same `thread_id`, neograph detects schema changes and automatically rewinds to re-execute only the affected nodes.

**Schema fingerprinting** (`state.py`):
- `compute_schema_fingerprint(state_model)` — SHA-256 prefix of sorted `(field_name, annotation_string)` pairs, excluding framework fields (`neo_*`). Attached to compiled graph as `_neo_schema_fingerprint`.
- `compute_node_fingerprints(construct)` — `dict[str, str]` mapping each node's state field to a SHA-256 prefix of `"{field_name}:{type.__qualname__}"`. Attached as `_neo_node_fingerprints`.

**At compile time** (`compiler.py:204-205`): both fingerprints are stashed on the compiled graph.

**At run time** (`runner.py:267-272`): fingerprints are injected into the initial state dict so they persist in the checkpoint.

**On resume** (`runner.py:_verify_checkpoint_schema`): stored fingerprints are compared against current. If they differ:
- `_compute_invalidated_nodes()` diffs per-node fingerprints to find which nodes changed
- `auto_resume=True` (default): `_auto_resume_from_divergence()` walks `get_state_history()` backwards, finds the checkpoint where the earliest changed node was in `.next`, injects that `checkpoint_id` into config, and `invoke(None)` resumes from there
- `auto_resume=False`: raises `CheckpointSchemaError(invalidated_nodes=...)` for explicit handling

**What triggers invalidation**: output class renamed, field added/removed/type-changed. Prompt text changes do NOT trigger invalidation (fingerprints are type-based, not content-based).

---

## Lint: template placeholder validation

`lint()` (`src/neograph/lint.py`) now validates template placeholders in addition to DI bindings. Full signature:

```python
lint(construct, *, config=None, known_template_vars=None, template_resolver=None)
```

**Three check categories**:
1. DI binding checks (original) — `FromInput`/`FromConfig` params vs config dict
2. Inline prompt placeholder checks — `${var}` against predicted input dict keys (no flattened, no framework extras)
3. Template-ref placeholder checks — `{var}` against predicted input keys + flattened fields + known extras (requires `template_resolver`)

**The inline/template-ref key asymmetry** is the most common lint confusion. Inline prompts see fewer keys because they resolve via raw attribute access (no rendering pipeline). Template-ref prompts see more keys because the rendering pipeline produces flattened fields and framework extras.

**`_predict_input_keys(node, include_flattened=True)`** — internal helper that computes what keys a node will see at runtime. `include_flattened=False` for inline, `True` for template-ref.

**Setup module exports** for `neograph check --setup`:
- `get_check_config()` — config dict (required)
- `get_template_resolver()` — `Callable[[str], str | None]` (optional)
- `get_known_template_vars()` — iterable of extra var names (optional)

**Loop condition checks** (neograph-sfj8): lint also validates Loop `when` conditions:
- `loop_condition_unregistered` (ERROR): string condition not in registry
- `loop_condition_none_unsafe` (WARN for callables, ERROR for string conditions): smoke-tests `when(None)` to catch the most common Loop bug -- `lambda d: d.score < 0.8` without a `d is None or` guard

**Oracle merge_prompt upstream context**: `merge_prompt` now passes upstream context alongside the variant list. `make_oracle_merge_fn` accepts a `node_inputs` parameter and builds `{"variants": primary, **upstream_from_state}` as input_data. Templates use `${variants}` for the variant list and `${upstream.field}` for upstream data.

---

## Modes and mode inference

`@node` supports five execution modes:

| Mode | When | Body runs? | Dispatch |
|---|---|---|---|
| `scripted` | No `prompt=`/`model=` | ✓ | `_execute_node` via `ScriptedDispatch` |
| `think` | `prompt=` + `model=` present | ✗ (dead code) | `_execute_node` via `ThinkDispatch` |
| `agent` | Same + `tools=` (read-only) | ✗ | `_execute_node` via `ToolDispatch` |
| `act` | Same + `tools=` (mutations) | ✗ | `_execute_node` via `ToolDispatch` |
| `raw` | Explicit `mode='raw'` | ✓ | `factory._make_raw_wrapper` via `raw_fn` |

**Mode inference**: if `mode=` is not passed, the decorator looks at other kwargs — `prompt=` + `model=` → `think`; neither → `scripted`. Mode `raw` always requires explicit opt-in (enforces the `(state, config)` signature).

**Dead-body warning**: LLM modes emit a `UserWarning` at decoration time if the function body is non-trivial (not `...`, `pass`, or a bare return). AST-based check — handles common false positives.

**Scripted `@node` dispatches via `register_scripted`.** At construct-assembly time, `_register_node_scripted` in `decorators.py` builds a shim closure that resolves `FromInput`/`FromConfig`/constant params from `config`, reads upstream values from `input_data` (the dict returned by `factory._extract_input`), and calls the user function with positional args. The shim is registered via `register_scripted` under a synthesized name, and `node.scripted_fn` points to it. The factory's `_execute_node` picks it up via `ScriptedDispatch` — **one dispatch path for all node modes** (neograph-y8ww).

`Node.fan_out_param` tells `_extract_input` which `inputs` key should read from `state["neo_each_item"]` instead of from a named upstream field. This is the only IR-level concession to the `@node` layer — it applies equally to programmatic `Each` nodes with dict-form inputs.

---

## Git workflow

- **`main`** — stable. Only tagged releases and critical hotfix PRs.
- **`develop`** — active development. All new work lands here. Currently at 0.3.0. Piarch and other downstream consumers pull from this branch via `uv add "neograph @ git+https://github.com/KonstantinMirin/neograph.git@develop"`.
- **Release path**: when `develop` is ready, merge to `main`, tag `vX.Y.Z`, push the tag. `.github/workflows/publish.yml` triggers on `v*` tags and publishes to PyPI via Trusted Publishing (no tokens, OIDC-scoped).
- **Version bumps**: on `develop` we increment normally. On `main` at the release tag we tag `vX.Y.Z`.

**Never publish directly.** The GitHub Actions workflow is the only publish path. This gives us a pypi.org Trusted Publisher gate + an optional manual-approval environment reviewer.

---

## Test conventions

### Test file layout (36 files across 5 packages)

**Root tests** (18 files):

| File | Scope | Tests |
|------|-------|-------|
| `test_validation.py` | Assembly-time type checking; fan-in; effective_producer_type; lint(); TypeSpec | ~128 |
| `test_renderers.py` | XmlRenderer, DelimitedRenderer, JsonRenderer, describe_type, render_prompt | ~88 |
| `test_forward.py` | ForwardConstruct base class, tracer, compilation, branching, loops | ~67 |
| `test_composition.py` | Sub-constructs, @node sub-constructs, state hygiene, reducers, dict-form | ~63 |
| `test_coverage_gaps.py` | Coverage gap tests for uncovered code paths | ~60 |
| `test_conditions.py` | parse_condition, condition registry | ~45 |
| `test_loop.py` | Loop modifier: self-loop, Loop-on-Construct, ForwardConstruct, skip_when | ~41 |
| `test_structural_guards.py` | AST-scanning guards against regressions | ~37 |
| `test_inline_prompts.py` | Inline prompt compilation, template rendering | ~29 |
| `test_di.py` | DI bindings, resolution, typed fields | ~27 |
| `test_spec_loader.py` | YAML/spec loader, type resolution | ~26 |
| `test_obligation_r1r2.py` | Behavioral obligation tests | ~23 |
| `test_cli.py` | CLI entry points | ~22 |
| `test_spec_types.py` | Type registry | ~20 |
| `test_spec_schema.py` | Spec schema validation | ~14 |
| `test_model_compat.py` | Pydantic model compatibility | ~14 |
| `test_fakes.py` | LLM fake infrastructure tests | ~7 |
| `test_check_fixtures.py` | Compiler safety net (parametrized fixtures) | ~2 |

**Package tests** (18 files in 4 packages):

| Package | Files | Scope | Tests |
|---------|-------|-------|-------|
| `decorator/` | 5 files | @node, @tool, @merge_fn decorators; mode inference; DI; construct assembly; edge cases | ~165 |
| `modes/` | 5 files | Scripted/think/agent/act/raw modes; execution; output strategies; LLM internals; I/O | ~156 |
| `modifiers/` | 5 files | Oracle, Each, Operator, compositions, modifier edge cases | ~119 |
| `hypothesis/` | 3 files | Property-based testing: topologies, invariants, regression | ~71 |

Supporting files: `conftest.py` (registry cleanup fixture), `schemas.py` (shared Pydantic models + `_producer`/`_consumer` helpers), `fakes.py` (LLM fakes).

### Compiler safety net (fixture-based validation testing)

`tests/check_fixtures/` — rustc-style fixture suite that tests the validator itself, not just pipelines. Each fixture is a self-contained `.py` file with a top-level `Construct`. A parametrized test in `test_check_fixtures.py` discovers them automatically.

| Directory | Purpose | Convention |
|-----------|---------|------------|
| `should_fail/` | Each file has one known defect. Must raise during import or compile. | `# CHECK_ERROR: <regex>` comment matches the expected error message |
| `should_pass/` | Valid pipelines. Must import and compile cleanly. | No special comment needed |
| `known_gaps/` | Defects the validator SHOULD catch but doesn't yet. Each is a filed bug. | Same `# CHECK_ERROR:` as should_fail; moves to should_fail when the validator is fixed |

**Rules:**
- Every new validation rule gets a corresponding should_fail fixture AND a should_pass fixture.
- Fixtures derived from real consumer code (piarch patterns) are higher quality than hypothetical ones. When adding fixtures, look at actual usage in `piarch/src/derive_ensemble/constructs/`.
- The fixture author should be different from the validation author when possible — neograph-a9n2 was caught by a fixture written AFTER the validation was "done."
- `known_gaps/` IS the backlog for validation improvements. When a fixture should_fail but doesn't, move it there and file a bug.
- Keep fixtures minimal — one Construct, one defect, ~15 lines.

### General test conventions

- **New tests go in the matching file.** If a feature spans multiple files, put the test where the primary behavior lives and add cross-references in docstrings.
- **BDD naming**: `test_{what_should_happen}_when_{condition}`. Class docstrings describe the feature being tested.
- **Throwaway modules for `construct_from_module` tests**: use `types.ModuleType("test_xyz_mod")` and attach `@node` functions as attributes. Don't pollute real modules. Pattern is `TestNodeDecorator._fresh_module`.
- **Fakes live in `tests/fakes.py`**: `FakeTool`, `StructuredFake`, `TextFake`, `ReActFake`, `configure_fake_llm`. Don't invent new fakes unless the existing ones genuinely don't cover the case.
- **TDD the user explicitly expects**: for bug fixes, write the failing repro first, verify it fails, then fix. The user has asked for this multiple times -- honor it on every bug-fix task.
- **Three-surface parity rule**: any IR-level behavioral change (`node.py`, `_construct_validation.py`, `factory.py`, `state.py`) must be tested through all three API surfaces -- `@node` decorator, declarative `Node.scripted()`, and programmatic `Node() | Modifier()`. This is the most common source of bugs: a feature works via `@node` (which runs through `_build_construct_from_decorated`) but breaks via the programmatic API (which goes straight to `Construct(nodes=[...])`). The `neograph-ts7` bug was exactly this pattern -- `fan_out_param` was set only in the decorator path, so programmatic `Each` + dict-form inputs failed validation. Test all three surfaces or explain why a surface is exempt.

---

## Examples

19 runnable examples in `examples/`, each narrated as a walkthrough on neograph.pro. Most use `@node` except two that stay declarative (example 10 mixed, example 11 config injection). Sub-constructs (example 05) can now use either `@node` with `construct_from_functions(input=, output=)` or declarative `Construct(input=, output=, nodes=[...])`.

**Examples must run end-to-end.** Breaking one is a regression. When you change an API surface, run every example that doesn't require real API keys (01, 01c, 02, 03, 04, 05, 06, 08, 09, 10). Examples 07 and 11 hit real OpenRouter/OpenAI — example 07 has a pre-existing known failure that predates anything in this session, document any new failures separately.

---

## Website

Astro + Starlight at `website/`. Deployed on Amplify from the main repo, triggered by any push that touches `website/` (actually just any push — Amplify rebuilds on every commit). The build must succeed or the site breaks.

**Always run `npm run build` in `website/` after content changes.** 26 pages, build takes ~2 seconds. Silent breakages are rare but possible (broken MDX frontmatter, missing `Annotated` import in code examples, etc.).

**Custom components** in `website/src/components/`:
- `SiteTitle.astro` — monospace "neograph" wordmark
- `Banner.astro` — site-wide sponsor banner ("Built by Postindustria...")

Both are Starlight slot overrides configured in `website/astro.config.mjs` under `components:`.

**License split**:
- Code: MIT (`LICENSE` file at repo root)
- Documentation: CC BY-ND 4.0, © Constantine Mirin, mirin.pro
- Every website page and the README has the doc-license footer.

---

## Things explicitly deleted / avoided

- **`@raw_node` decorator**: removed in favor of `@node(mode='raw')`. Grep should return zero hits.
- **`FromInput[T]` / `FromConfig[T]` Generic subscription form**: removed in 0.2.0.dev. Use `Annotated[T, FromInput]`. The old form raises `TypeError: type 'FromInput' is not subscriptable`, which is intentional — clean error beats silent breakage.
- **Emojis in docs**: the user explicitly rejected them ("kill emojis, that's sooo LLM-ish"). Don't reintroduce. If a code comment uses one, replace with plain text.
- **Line counts as a value metric**: the user explicitly rejected framing value around "X lines vs Y lines". The docs talk about what neograph *does* (type safety, durability, observability, focus on logic), not how many lines shorter it is than raw LangGraph. Comparison table "What you don't write" stays on the Why-not-LangGraph page but isn't on the landing.
- **`TestPyPI` in the release flow**: skipped. The real PyPI release went directly from CI. Documented that TestPyPI is optional, not required, for alpha releases.

---

## Known open DX items

These aren't bugs, just things worth considering for future sessions:

- `@merge_fn` uses a function-name-keyed registry (`_merge_fn_registry` in `decorators.py`). Parallel to the `@node` id-keyed sidecar. Both patterns work; the name-keyed form is cleaner for merge_fn because Oracle references them by string name anyway. If you add another decorator that's referenced by string name, copy that pattern.
- The sponsor banner on neograph.pro is hardcoded in a component. If we ever add more sponsors or commercial positioning, it should probably move to config.

---

## User preferences (from the build sessions)

- **Blunt, direct answers preferred over agreement.** If an API has a DX problem, say so. The user will happily refactor at 0.x.
- **No backwards-compat shims at 0.x.** Breaking changes are fine; deprecation cycles are unnecessary at this scale and one known user.
- **TDD for bug fixes, always.** Write the failing test first.
- **Parallel agent teams for multi-file work.** The `/team` slash command invokes a team with scoped file regions. Use it for anything that can be parallelized without file conflicts.
- **User is the sole maintainer and sole downstream user (piarch).** No migration burden for hypothetical users.
