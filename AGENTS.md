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

**The producer side is shared.** `effective_producer_type(item)` — defined in `_validation_types.py` and re-exported through `_construct_validation.py` (its `__all__`) — computes "what type does this node write to the state bus, accounting for modifiers". It's the single source of truth for modifier-aware type effects.

**Rule for new modifiers that reshape state**: teach `effective_producer_type` about the new rule. The validator picks it up automatically. Do NOT re-inline modifier checks elsewhere — prior drift caused `neograph-8k3` and `neograph-ayq` before this helper existed.

Current rules encoded in `effective_producer_type`:
- `Each` modifier → `dict[str, output]` (see `state.py:_add_output_field` for the state builder side of the same rule)
- Anything else → raw `output` unchanged

**The declared-output selector is also monopolized** (`neograph-8cqd`): the `Node.outputs` (plural) vs `Construct.output` (singular) discrimination lives once in `_declared_output` in `_normalize.py` (a neutral low-level module reachable from every layer, including the DX layer `forward.py`). Do NOT hand-roll `getattr(item, 'output', None)` — call `_declared_output(item)`. `TestDeclaredOutputSelectorMonopoly` bans the inline form outside `_normalize.py`. The one sanctioned exception is `compiler.py`'s three-way `isinstance(_BranchNode/Construct/Node)` match, which dispatches to three different graph-builders with different signatures — an irreducible sum-type, not a selector.

### `list[X]` consumers of `Each` producers (merge-after-fan-out)

A downstream node can consume an Each-modified upstream's fanned-out results as a `list[X]`:

```python
@node(outputs=Summary)
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
- **The KEYMAKER dynamic-handoff exception (mode a peer routing) is the second sanctioned new-IR capability, on the same footing as `_BranchNode`.** A runtime mesh needs IR + runtime support that the linear model does not, so it deliberately adds:
  - **Two IR fields on `Node`** — `Node.handoff_param` (the reserved `handoff` input key, the exact sibling of `fan_out_param`) and `Node.handoff_channel` (the entry-keyed shared mesh channel). **Both are written by a single writer, `_ir_normalize.py`, and nowhere else** — the same single-writer invariant `fan_out_param` has (the `neograph-ts7` lesson), pinned by guard **G3** (`IR_FIELDS` frozenset in `test_guards_llm_runtime.py`). Do NOT set either field in `decorators.py` / `_construct_builder.py`; the three surfaces converge in the normalizer.
  - **A `Command(goto)` runtime**: a `Keymaker` member lowers to a wrapper returning LangGraph `Command(goto=..., update=...)`, so control flow is derived at runtime, not from a static edge. **`Command(` may be constructed ONLY in `factory.py` and `runner.py`** — the monopoly that ratchets the new capability, pinned by guard **G1** (`TestCommandConstructionMonopoly` in `test_guards_assembly.py`).
  - **`neo_`-prefixed mesh state keys** built only via `StateKeys.handoff_payload(...)` / `StateKeys.handoff_hops(...)` (no inline f-strings, on top of the Layer-A `neo_`-fragment guard). The mesh-assembly validation rules live in `_validation_keymaker.py` (`_check_keymaker_mesh`).

  This is the pattern to copy for any future genuinely-new IR capability: add the field(s) with a single writer, confine the new runtime construct to the compiler/runtime layer, and pin both with structural guards written failing-first.
- **Sub-constructs can be @node or declarative.** `construct_from_functions("verify", [explore, score], input=VerifyClaim, output=ClaimResult)` builds a sub-construct from `@node` functions. Params whose type matches `input=` are port params — they read from `neo_subgraph_input` instead of a peer `@node`. The declarative form `Construct(input=X, output=Y, nodes=[...])` also works. Both produce the same IR.

### Naming policy: `__all__` is the public contract; the `_` module prefix is advisory only

The leading-underscore on a *module* name (e.g. `_llm.py`, `_dispatch.py`) is a weak, advisory hint — it does NOT reliably signal public vs internal in either direction. Some unprefixed modules are internal-only (`factory.py`, `state.py`, `di.py`, `naming.py`), and some underscore-prefixed modules export public API through `__all__` (`_llm.py`, `_image.py`). **The single source of truth for what is public is the package facade: a symbol is public iff it is re-exported from `neograph/__init__.py` and listed in its `__all__`.** Do NOT infer a module's or symbol's visibility from its underscore prefix, and do NOT mass-rename to "fix" the mismatch pre-release — the churn is not worth it for one downstream consumer. When adding a new public symbol, wire it through `__init__.__all__`; when adding an internal one, no rename ceremony is required.

**Review checklist item**: when reviewing a change that adds or moves a symbol, confirm its public/internal status is expressed through `neograph/__init__.py`'s `__all__` (the contract), not inferred from the module-name `_` prefix (advisory only).

### Logging convention: module-level bare `get_logger()`

Every module that logs binds a module-level logger with the **bare** call: `log = structlog.get_logger()` (no `__name__` or explicit name argument). structlog resolves the calling module for you, so passing `__name__` is redundant and just invites drift across modules. Do NOT write `get_logger(__name__)` or `get_logger("neograph")` for a new module-level `log`; copy the bare form the other ~12 modules use.

---

## DI surface (post-0.2): `Annotated[T, FromInput/FromConfig]`

**Breaking change from 0.1.x → 0.2.0.dev**. The old `FromInput[T]` Generic subscription form is gone. The new form uses `typing.Annotated` with `FromInput` / `FromConfig` as markers — the FastAPI `Annotated[User, Depends(...)]` pattern.

```python
from typing import Annotated
from neograph import node, FromInput, FromConfig

@node(outputs=Result)
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

**Classifier implementation notes** (`_classify_di_params` in `_di_classify.py`, imported by `decorators.py`):
- Uses `typing.get_type_hints(f, localns=..., include_extras=True)` to preserve `Annotated` metadata.
- Captures the caller's local namespace in a single shot: `node()` / `merge_fn()` grab `sys._getframe(1).f_locals` ONCE at decoration time (`decorators.py:324`, `:742`) and pass it explicitly as `caller_ns` down to `_classify_di_params` → `_build_annotation_namespace`. No frame-stack walk and no frame-depth arithmetic — the closure carries the captured namespace into `decorator(f)`, so the one hop from user call site to `node()`'s frame is fixed and correct for both the `@node(...)` and bare-`@node` forms. This matters because `from __future__ import annotations` stringifies annotations and strips closure references, so a `class RunCtx` defined inside a test method isn't findable via `f.__globals__` or `f.__closure__`; `caller_ns` supplies it as `localns`. `_build_annotation_namespace` merges the DI markers, the function's closure vars (`inspect.getclosurevars`), and the caller ns (skipping `_`-prefixed names and never shadowing markers).

**Runtime resolution** — one path, `DIBinding.resolve(config)` (`di.py:329`); `_resolve_di_args` (`_di_classify.py`) maps a node's `ParamResolution` to positional args by calling `resolve()` per binding:
- `FROM_INPUT` / `FROM_CONFIG` → read `config['configurable'][name]`; type-check against `inner_type`; raise `ExecutionError` when `required` and missing
- `FROM_INPUT_MODEL` / `FROM_CONFIG_MODEL` → construct `model_cls` by pulling each field from `config['configurable'][field_name]`
- `FROM_RESOURCE` → hydrate from the MCP resource URI/ref (with `max_bytes` cap)
- `FROM_STATE` (merge_fn only) → read from the passed `state`
- `CONSTANT` → use the captured `default_value`
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

**Why we keep the sidecar rather than eagerly resolving**: the sidecar carries the IR-level metadata that the compiler needs (the original function, param names, DI bindings). The Python compiler consumes this to build scripted shims registered by string name into LangGraph. Eagerly resolving to LangGraph-Python `scripted_fn` registry names at IR construction time would bake the Python runtime's registration mechanics into the IR; keeping the sidecar separates "what the node is" from "how this runtime invokes it".

---

## `describe_type` / `describe_value` — LLM-facing schema rendering

`src/neograph/describe_type.py` (512 lines, 15 functions) renders Pydantic models into a TypeScript-style notation that LLMs parse more reliably than JSON Schema. Used by the factory layer to build structured output instructions.

**Two public functions** (both re-exported from `neograph`):
- `describe_type(model, prefix=..., hoist_classes=...)` — renders a model class into a schema string with auto-hoisted nested classes
- `describe_value(instance, prefix=...)` — renders a model *instance* as a typed value literal (for few-shot examples)

**Two-pass architecture**: pass 1 (`_count_classes`) counts how many times each nested class appears across the model tree. Pass 2 (`_render_model_body` / `_render_type`) emits the notation, hoisting classes that appear more than once (or all, per `hoist_classes=`).

**Handles**: primitives, `list[T]`, `dict[K,V]`, `Optional[T]`, `Union[A,B]`, `Literal[...]`, `Enum`, nested `BaseModel`, `tuple[...]`, forward refs, field descriptions, constraints, and defaults.

**Tests**: `test_renderers.py` — 88 tests covering all type combinations, edge cases, and round-trip parsing.

---

## RenderedInput — single rendering abstraction

`src/neograph/renderers.py` (`RenderedInput` dataclass at line 33). The single object that bundles all rendering artifacts for prompt construction. Produced by `build_rendered_input(input_data, renderer=None)`.

**Five fields**:
- `raw: dict[str, Any] | Any` — original Pydantic models, used by inline `${var}` prompts for dotted attribute access
- `rendered: dict[str, Any] | Any` — BAML-rendered strings, used by template-ref prompts via `prompt_compiler`
- `flattened: dict[str, Any]` — extra fields from `render_for_prompt()` BaseModel returns, available only in template-ref prompts
- `available_keys_inline: set[str]` — keys valid for inline `${var}` (raw dict keys only, no flattened, no framework extras)
- `available_keys_template: set[str]` — keys valid for template-ref `{var}` (raw + flattened + framework extras)

**`for_template_ref` property** — merges `rendered` and `flattened` dicts, with `rendered` keys taking precedence. This is what the `prompt_compiler` receives.

**Consumers**: `_dispatch.py:_render_input()` (mode dispatch layer) and `_llm_render.py:render_prompt()` (prompt inspection).

**The inline/template-ref split**: inline prompts (`${var}`) get `ri.raw` — raw Pydantic objects for `getattr` chains. Template-ref prompts get `ri.for_template_ref` — pre-rendered strings + flattened fields. Flattened fields and framework extras (`node_id`, `project_root`) are NOT available in inline prompts.

---

## Checkpoint resume — schema-aware auto-rewind

When a pipeline runs with a checkpointer and the same `thread_id`, neograph detects schema changes and automatically rewinds to re-execute only the affected nodes.

**Schema fingerprinting** (`state.py`):
- `compute_schema_fingerprint(state_model)` — SHA-256 prefix of sorted `(field_name, _type_signature(annotation))` pairs, excluding framework fields (`neo_*`). Stashed on the compiled graph as `graph.schema_fingerprint`.
- `compute_node_fingerprints(construct)` — `dict[str, str]` mapping each node's state field to a SHA-256 prefix of `"{field_name}:{_type_signature(type)}"`. Dict-form outputs are fingerprinted per key (`{node}_{key}`). Stashed as `graph.node_fingerprints`.
- **`_type_signature(typ)` (structural, not qualname-only — neograph-v63o)** — BOTH fingerprints fold one level of field detail through this shared helper: a Pydantic model hashes `module.Qualname` + sorted `(field, str(annotation))` pairs; generics (`list[X]`, `dict[K,V]`, Each's `dict[str, X]`) are unwrapped so a change on the wrapped model is visible. This replaced qualname-only hashing so a **same-`__qualname__` model with a changed field type now invalidates** — the old coarse hash was a false-negative that stopped the rewind from triggering at all. Both fingerprints had to move in lockstep: `str(annotation)` on a same-qualname changed field is identical, so without folding the signature into `compute_schema_fingerprint` too, the schema-fp GATE (`_decide_checkpoint_schema` returns `None` on a match) never opens and the enriched node fingerprint would be dead code.

**At compile time** (`compiler.py:300-301`): both fingerprints are stashed on the compiled graph.

**At run time** (`runner.py:497-500`): the schema fingerprint is injected into the initial state dict (under `StateKeys.SCHEMA_FINGERPRINT`) so it persists in the checkpoint.

**On resume** (`runner.py:_verify_checkpoint_schema` → `_decide_checkpoint_schema`): the stored schema fingerprint is compared against current. If they differ:
- `_compute_invalidated_nodes()` (`runner.py:267`) diffs per-node fingerprints to find which nodes changed.
- `auto_resume=True` (default): `_auto_resume_from_divergence()` walks `get_state_history()` backwards for the OLDEST checkpoint whose `.next` intersects the invalidated set, injects that `checkpoint_id` into config, and `invoke(None)` resumes from there. **Fail-loud on no rewind point (neograph-v63o):** if `invalidated` is non-empty but NO snapshot has an invalidated node pending in `.next` (history pruned, or every invalidated node already ran), it does NOT silently resume from the tip — it raises `CheckpointSchemaError(invalidated_nodes=...)` via the single-sited `_raise_no_rewind_point`, surfaced BEFORE any node re-executes. Silently resuming would re-hand the caller stale results (the durability pitch's one actively-false spot). Empty `invalidated` stays a genuine no-op (nothing changed).
- `auto_resume=False`: raises `CheckpointSchemaError(invalidated_nodes=...)` for explicit handling.

**Migration (correct-and-desired, pinned)**: the fingerprint FORMAT changed with `_type_signature`, so existing pre-v63o checkpoints show a schema mismatch on first resume after upgrade → every node looks changed → one full re-run. This is deliberate (better than trusting a coarser stale signature) and pinned by `test_old_format_node_fingerprint_invalidates_on_upgrade`.

**What triggers invalidation**: output class renamed, field added/removed/type-changed (including a same-name-class field-type change, post-v63o). Prompt text changes do NOT trigger invalidation (fingerprints are type-based, not content-based).

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

**The THIRD column — `di_inputs` (neograph-euyh)**: a template-ref prompt can ALSO reference a node's `FromInput`/`FromConfig` parameter names (e.g. `{domain}` for `domain: Annotated[str, FromInput]`) — but ONLY when the app's `prompt_compiler` opts in by declaring a `di_inputs` parameter (or `**kwargs`). `lint(construct, ..., prompt_compiler=...)` introspects the compiler's signature with the same `_accepted_params` helper the runtime uses to gate the kwarg; when it accepts `di_inputs`, the node's DI param names become valid template-ref placeholders (`_di_template_var_names`). Without opt-in the placeholder is flagged `template_placeholder_unresolvable` — because the resolved value never reaches the template and the literal `{domain}` would ship to the model (the agent-stark production incident). Inline `${var}` prompts NEVER get this column: they resolve via raw attribute access, not the compiler seam. So the full asymmetry is: inline = raw input keys; template-ref = input keys + flattened + framework extras; template-ref WITH a di_inputs-aware compiler = the above **+ DI param names**.

**Runtime-vs-lint coverage (in lockstep)**: the lint column lights up for ANY LLM-mode node (think/agent/act) whose compiler accepts `di_inputs`, and at RUNTIME all three modes now inject di_inputs — `think`/`raw` via `ThinkDispatch` and `agent`/`act` via the ReAct cycle's shared pre-prep (`_agent_cycle._turn_prep_kwargs`, neograph-jhz4). A `{domain}` placeholder that passes lint resolves at runtime for every LLM mode. If a new LLM-mode dispatch path is ever added, it must call `_inject_di_inputs` before its `_compile_prompt` or the lint rule and runtime coverage will silently diverge again.

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

## `di_inputs` — resolved DI values reaching prompt templates (neograph-euyh)

An LLM-mode node (`think`/`agent`/`act`) never runs its body, so — unlike scripted nodes, whose shim resolves DI — its `FromInput`/`FromConfig` params were historically dropped: a `domain: Annotated[str, FromInput]` referenced as `{domain}` in a template never became a template var, and a fail-soft compiler shipped the literal `'{domain}'` to the model (the agent-stark incident; the only workaround was a scripted seed node copying run-input onto the bus). `di_inputs` removes that need.

**Plumbing (config side-channel, mirrors `_oracle_model`)**: `ThinkDispatch` (`_dispatch.py:_inject_di_inputs`) resolves the node's `_param_res` bindings ONCE via the canonical `DIBinding.resolve(config)` (no second resolver — same path `_resolve_di_args` uses) and stashes the `{param_name: value}` map into `config['configurable']` under `StateKeys.DI_INPUTS` (`_neo_di_inputs`, a config-only key — never enters state, never touches the schema fingerprint). `_compile_prompt` (`_llm_render.py`) reads it back and passes it to the compiler as an **introspection-gated** kwarg via the existing `prompt_compiler_params`/`_ACCEPT_ALL` filter — so only a compiler declaring `di_inputs` (or `**kwargs`) receives it. This avoids threading a new positional through the `_llm`/`_tool_loop` call chain. Only which DI kinds are template-usable is centralized in `di.DI_TEMPLATE_KINDS` (FROM_INPUT/FROM_CONFIG + their MODEL forms; CONSTANT and FROM_STATE excluded).

**Precedence (collision rule)**: in `DefaultPromptCompiler.build_vars`, `di_inputs` is the BASE layer and rendered upstream OUTPUTS are laid on top — on a name collision the **upstream output shadows the di_input**. Rationale: an upstream producer named `domain` is the node-local, dataflow-derived value; `di_inputs` is run-wide ambient context, so the narrower binding wins. This is also the **zero-behavior-change** justification: `di_inputs` only fills names not already produced by an upstream node, so no existing pipeline's `{name}` binding changes meaning when a FromInput param collides. `None` collapses to `{}` (the `render_inputs` total-dict contract), so an all-DI leaf node still gets its `{domain}` var.

**Three-surface parity — decorator-only, by construction**: `di_inputs` is sourced from `node._param_res`, which is populated ONLY by the `@node` decorator's `_classify_di_params`. Declarative `Node(...)` and programmatic `Node() | Modifier()` surfaces carry no `FromInput`/`FromConfig` bindings (DI markers are an `Annotated`-param, decorator-layer concept), so `_param_res` is empty and `_inject_di_inputs` returns config unchanged. The other two surfaces are therefore EXEMPT — there is no DI binding to expose. The lint third-column (`_di_template_var_names`) reflects the same: it reads `_param_res`, so it only lights up for `@node`-built nodes.

**Agent/act (`_agent_cycle.py`) — wired via the same injector (neograph-jhz4)**: agent/act nodes compile to a multi-node ReAct cycle that does NOT go through `_execute_node`/`ThinkDispatch`, so the injection happens at `_turn_prep_kwargs` — the single shared pre-prep both sync/async turn-prep twins call, before the cycle's `_compile_prompt`. It reuses the exact same module-level `_inject_di_inputs`, `StateKeys.DI_INPUTS` key, and canonical resolver as think mode (no second mechanism). Per-superstep re-injection is idempotent (copy-not-mutate). All LLM modes now reach the model with di_inputs.

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
- **`develop`** — active development. All new work lands here. The authoritative version is `__version__` in `src/neograph/__init__.py` (do not hard-code it here — it drifts). Piarch and other downstream consumers pull from this branch via `uv add "neograph @ git+https://github.com/KonstantinMirin/neograph.git@develop"`.
- **Release path**: when `develop` is ready, merge to `main`, tag `vX.Y.Z`, push the tag. `.github/workflows/publish.yml` triggers on `v*` tags and publishes to PyPI via Trusted Publishing (no tokens, OIDC-scoped).
- **Version bumps**: on `develop` we increment normally. On `main` at the release tag we tag `vX.Y.Z`.

**Never publish directly.** The GitHub Actions workflow is the only publish path. This gives us a pypi.org Trusted Publisher gate + an optional manual-approval environment reviewer.

---

## Test conventions

### Test file layout

The suite grows every wave, so the counts below rot fast — recount rather than
trust a frozen number: `ls tests/test_*.py | wc -l` (root),
`ls tests/{decorator,modes,modifiers,hypothesis}/test_*.py | wc -l` (packages).
As of 2026-07-08: **89 root `test_*.py` files + 29 package files = 118 total.**

**Root tests** (~89 files). The table below is a REPRESENTATIVE index of the
primary suites, not an exhaustive enumeration — many focused files (async,
checkpoint, MCP, guards, observability) are not listed row-by-row.

`test_validation.py` and `test_structural_guards.py` were split by concern in
neograph-e8jg (no file exceeds 1200 lines; class names unchanged so guards stay
discoverable). The validation suite is now several files; the structural-guard
suite has since grown to ~17 `test_guards_*.py` files (`ls tests/test_guards_*.py`).

| File | Scope | Tests |
|------|-------|-------|
| `test_validation.py` | Core assembly validation: construct/oracle errors, Each-path, name collision, tool/LLM config, output strategy, error builder, TypeSpec, FromInput-required, single-type deprecation | ~72 |
| `test_fanin_validation.py` | Fan-in: dict-form inputs, Each interop, effective_producer_type, list/dict compat, dict-form outputs, three-surface parity | ~35 |
| `test_lint.py` | lint() DI bindings, obligation gaps, Loop condition checks | ~29 |
| `test_template_lint.py` | lint() inline `${var}` and template-ref `{var}` placeholder checks | ~44 |
| `test_context_validation.py` | Sub-construct context-field + output-boundary validation | ~15 |
| `test_guards_assembly.py` | Guards: error builder, file-split, assembly import DAG, subconstruct boundaries, dead code, no-Any boundaries, no-sidecar-pattern | ~50 |
| `test_guards_ir_compiler.py` | Guards: IR typing, compiler wiring, node mutation, branch nodes, build-construct body size, registry dicts | ~23 |
| `test_guards_sidecar_imports.py` | Guards: sidecar module, function-local import allowlist, tool-loop import graph, langgraph imports, IO polymorphism | ~12 |
| `test_guards_any_audit.py` | Guards: no-Any in public IR APIs, arbitrary-types justification, public functions raise NeographError | ~10 |
| `test_guards_function_local_imports.py` | Guards: function-local factory/llm imports, retry-policy signature, StateKeys centralization, no module-level registration | ~23 |
| `test_guards_llm_runtime.py` | Guards: factory kwargs, LLM responsibility/cohesion, StateBus.get discipline, runtime fan-out, normalize_ir field writer, routing-key invariant | ~30 |
| `test_renderers.py` | XmlRenderer, DelimitedRenderer, JsonRenderer, describe_type, render_prompt | ~88 |
| `test_forward.py` | ForwardConstruct base class, tracer, compilation, branching, loops | ~67 |
| `test_composition.py` | Sub-constructs, @node sub-constructs, state hygiene, reducers, dict-form | ~63 |
| `test_coverage_gaps.py` | Coverage gap tests for uncovered code paths | ~60 |
| `test_conditions.py` | parse_condition, condition registry | ~45 |
| `test_loop.py` | Loop modifier: self-loop, Loop-on-Construct, ForwardConstruct, skip_when | ~41 |
| `test_node_sidecar_contract.py` | Pins PrivateAttr (`_sidecar`/`_param_res`/`_scripted_shim`) preservation across model_copy/pipe/deepcopy | ~8 |
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
| `test_checkpoint_auto_rewind.py` | Schema-aware auto-rewind: fail-loud-on-no-rewind-point contract, sync + async | ~5 |

**Package tests** (29 `test_*.py` files across 4 packages):

| Package | Files | Scope | Tests |
|---------|-------|-------|-------|
| `decorator/` | 5 files | @node, @tool, @merge_fn decorators; mode inference; DI (incl. `TestMergeFnDuplicateRegistration`); construct assembly; edge cases | ~165 |
| `modes/` | 10 files | Scripted/think/agent/act/raw modes; execution; output strategies; LLM internals; I/O | ~156 |
| `modifiers/` | 5 files | Oracle, Each, Operator, compositions, modifier edge cases | ~119 |
| `hypothesis/` | 9 files | Property-based testing: topologies, invariants, regression | ~71 |

Supporting files: `conftest.py` (registry cleanup fixture), `schemas.py` (shared Pydantic models + `_producer`/`_consumer` helpers), `fakes.py` (LLM fakes).

### Compiler safety net (fixture-based validation testing)

`tests/check_fixtures/` — rustc-style fixture suite that tests the validator itself, not just pipelines. Each fixture is a self-contained `.py` file with a top-level `Construct`. A parametrized test in `test_check_fixtures.py` discovers them automatically.

| Directory | Purpose | Convention |
|-----------|---------|------------|
| `should_fail/` | Each file has one known defect. Must raise during import or compile. | `# CHECK_ERROR: <regex>` comment matches the expected error message |
| `should_pass/` | Valid pipelines. Must import and compile cleanly. | No special comment needed |

Only these two directories exist and are scanned by `test_check_fixtures.py`. A
`known_gaps/` tier (validator-SHOULD-catch-but-doesn't-yet fixtures) was
documented previously but was never created — the backlog for validation
improvements lives in beads, not a fixture directory. If you want a fixture tier
for known gaps, create `known_gaps/` AND teach `test_check_fixtures.py` to scan
it (xfail-style) before documenting it here.

**Rules:**
- Every new validation rule gets a corresponding should_fail fixture AND a should_pass fixture.
- Fixtures derived from real consumer code (piarch patterns) are higher quality than hypothetical ones. When adding fixtures, look at actual usage in `piarch/src/derive_ensemble/constructs/`.
- The fixture author should be different from the validation author when possible — neograph-a9n2 was caught by a fixture written AFTER the validation was "done."
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

21 runnable examples in `examples/`, each narrated as a walkthrough on neograph.pro. Most use `@node` except two that stay declarative (example 10 mixed, example 11 config injection). Example 27 is the ForwardConstruct imperative-wiring showcase (branch/self.loop/self.each/self.ensemble/self.interrupt, keyless, pinned by `tests/test_example_forward_wiring.py`). Sub-constructs (example 05) can now use either `@node` with `construct_from_functions(input=, output=)` or declarative `Construct(input=, output=, nodes=[...])`.

**Examples must run end-to-end.** Breaking one is a regression. When you change an API surface, run every example that doesn't require real API keys (01, 01c, 02, 03, 04, 05, 06, 08, 09, 10). The keyed examples are 07 and observable_pipeline.py — both hit real OpenRouter (observable_pipeline additionally pushes to Langfuse; run it with `--extra langfuse`), and both were verified passing end-to-end on 2026-07-09. Example 11 was converted to a FakeLLM and is keyless. Document any new failures separately.

### MCP examples (23/24/25) — no-key but need the `mcp-examples` extra (neograph-g4q9)

The MCP-featuring examples exercise the **real** Model Context Protocol against a
shared stdio demo server (`examples/_mcp_demo_server.py`) — no fakes at the
protocol layer, no network, no API keys. They are **keyless but NOT
dependency-light**: they need `mcp` + `langchain-mcp-adapters`, which live in the
`mcp-examples` optional extra (`[project.optional-dependencies].mcp-examples`),
**not** core deps and **not** the default dev group. This keeps `src/neograph`
MCP-free (the no-session-ownership guard scans `src/` only) and the core
`uv run --extra dev pytest` suite light.

- **Run the MCP E2E harness**: `uv run --extra dev --extra mcp-examples pytest tests/test_mcp_examples_e2e.py`
- The harness (`tests/test_mcp_examples_e2e.py`) is `pytest.importorskip`-gated, so
  the core suite **skips** it cleanly without the extra. It proves the demo server
  end-to-end (tool discovery, `get_deal` resource_link manifest, RFC-6570 email
  fraction read, per-operator auth echo, real `-32002` expiry + self-heal) and
  auto-discovers `examples/2?_mcp_*.py` to run each example as a subprocess (23/24
  plug in via neograph-qb7q / neograph-3m6g; 25 illustrates the singular
  `mcp_tool_factory` — offline build + gateway rename, neograph-sfdz1).
- **The distinction to remember**: "no-key" ≠ "no extra". Examples 23/24/25 are on the
  no-key list but you must pass `--extra mcp-examples` to run them or their tests.
- **Two verified `mcp` 1.28.x SDK gaps the demo server works around** (documented
  in the server's module docstring): FastMCP's `@mcp.resource` can't express
  RFC-6570 query templates, and its `@tool`/`@resource` wrappers swallow JSON-RPC
  error codes (a real `-32002` needs a custom low-level `read_resource` handler on
  `mcp._mcp_server`). Pin is `mcp>=1.28,<2` (mcp 2.0 renames `FastMCP`→`MCPServer`).

---

## Website

Astro + Starlight at `website/`. Deployed on Amplify from the main repo, triggered by any push that touches `website/` (actually just any push — Amplify rebuilds on every commit). The build must succeed or the site breaks.

**Always run `npm run build` in `website/` after content changes.** 26 pages, build takes ~2 seconds. Silent breakages are rare but possible (broken MDX frontmatter, missing `Annotated` import in code examples, etc.).

**Verifiable-docs remark plugin (Stage B, `website/plugins/remark-api.mjs`).** Wired in `astro.config.mjs` under `markdown.remarkPlugins`. It validates + autolinks backticked API-symbol references against the introspection-generated manifest (`website/src/data/api-manifest.json`, regenerated by `scripts/gen_api_manifest.py`). Tiered confidence: a dotted `Type.member` ref to a fielded type with a missing member **fails the Astro build** (HARD); a bare token autolinks on exact match or stays inert (SOFT, never build-failing). Run `npm test` in `website/` (node:test on `plugins/*.test.mjs`) before pushing website/ plugin changes — it is the plugin's regression suite, separate from the pytest gate.

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

- `@merge_fn` uses a function-name-keyed registry (`_merge_fn_registry` in `decorators.py`). This is NOT registry debt and the name-keyed pattern is the RIGHT shape here (neograph-pbya evaluated migrating it to the `@node` PrivateAttr-sidecar / per-compile shape and DECIDED TO KEEP the name-keyed global). Structural reason: `@node` can self-store on a sidecar because it *returns the Node it decorates* — decoration site and storage target are the same object. `@merge_fn` decorates a standalone function, returns the bare function, and that function is referenced from a DIFFERENT object (`Oracle(merge_fn='combine')`) purely by STRING NAME. No Node/Oracle is in scope at decoration time, so a name→metadata map is a structurally-required symbol table, not incidental global state (mirrors conditions/tool_factories, which the 2026-05 per-compile architecture still seeds from a decoration-time global). The prior silent-overwrite defect is closed: a same-name collision between two DIFFERENT definition sites now FAILS LOUD (`ConstructError` naming both `module.qualname` + `file:lineno`). Idempotent for the same definition site — re-importing the same function object, re-running the same `def` in a loop/hypothesis example (new fn object, shared code object), and module reloads (recompiled code object, same source site + qualname) all stay safe via `_same_def_site`. If you add another decorator referenced by string name, copy this pattern (name-keyed registry + fail-loud collision guard).
- The sponsor banner on neograph.pro is hardcoded in a component. If we ever add more sponsors or commercial positioning, it should probably move to config.

---

## User preferences (from the build sessions)

- **Blunt, direct answers preferred over agreement.** If an API has a DX problem, say so. The user will happily refactor at 0.x.
- **No backwards-compat shims at 0.x.** Breaking changes are fine; deprecation cycles are unnecessary at this scale and one known user.
- **TDD for bug fixes, always.** Write the failing test first.
- **Parallel agent teams for multi-file work.** The `/team` slash command invokes a team with scoped file regions. Use it for anything that can be parallelized without file conflicts.
- **User is the sole maintainer and sole downstream user (piarch).** No migration burden for hypothetical users.
