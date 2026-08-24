# AGENTS.md

Project-specific context and operational workflow for agents working on neograph. This file is the source of truth; `CLAUDE.md` is a symlink to it so every coding agent that supports either convention picks up the same content.

---

## What neograph is

A declarative LLM graph compiler on top of LangGraph. Declare a pipeline as typed Python functions (`@node`), `ForwardConstruct` classes, or programmatic `Node | Modifier` chains; neograph infers the DAG from parameter names, validates types at assembly time, and compiles to a LangGraph `StateGraph`. Three surfaces, one compiler.

**Positioning**: all of LangGraph's power, but *safer* and with the wiring handled — write the logic, not the graph. What it has become (`CHANGELOG.md` is the authoritative feature list; keep it and this in sync):

- **Safer control flow** — dynamic routing (`Portal`) targets a region's entry port, never its interior, so a class of LangGraph deadlocks / runaway loops is unrepresentable (see North star).
- **Compile-time validation** — types, fan-in, and reducibility checked at assembly, before you run; rustc-style check-fixture suite + `neograph check`.
- **Async-native** — one graph, four verbs (`run` / `arun` / `stream` / `astream`); driver↔checkpointer mismatch fails loud.
- **Durable** — schema-aware checkpoint auto-rewind; fail-loud on no rewind point.
- **BAML-style prompt rendering + prompt management** — Pydantic models rendered to TS-like schema LLMs parse reliably (`describe_type`); inline `${var}` and template-ref prompts; public `compile_prompt()` for in-/out-of-graph parity.
- **MCP client battery** (`neograph[mcp]`) — typed tool results, per-run identity fresh per request and transport, resource hydration, run-scoped connection reuse (reconnect-safe across resume), gated mutations, progress notifications, transport resilience, keyless fakes.
- **One-line observability** — `observe=` Langfuse auto-attach; structured logs + named spans per node.
- **Agents & HITL** — agent/act ReAct subgraphs, typed tool logs, `ask_human()`, tool-approval gates.

**Website**: [neograph.pro](https://neograph.pro) (Astro + Starlight at `website/`, deployed via AWS Amplify on every push). Don't forget to update website content when API surfaces change.

---

## North star: the restriction is the product

neograph's edge is **subtractive** — a LangGraph in which a class of broken programs *cannot be written*. Raw `Command(goto)` can jump into a loop's interior and silently deadlock; neograph's entry-port routing makes that state **unrepresentable**, not merely caught (the statechart / reducible-control-flow model — a jump reaches a region's entry, never its middle).

- **Decision filter**: measure every change against *does this keep the broken-state set unwriteable?* A change that buys ergonomics by re-admitting a silent failure is a regression **even if all tests pass**. Out-constrain LangGraph; don't just out-feature it.
- **Bounded claim** (don't overclaim): *unrepresentable* — invalid region entry, non-reducible control flow, type/fan-in mismatch; *fail-loud* — `max_hops`, schema drift on resume, missing DI/config; *not covered* — semantically-wrong-but-valid routing, LLM output quality (say so in docs too).
- **The tax**: "safer than LangGraph" is falsified by a *single* silent seam, so guard-first TDD and fail-loud-over-fail-soft keep the claim true — they are not overhead, and a band-aid that leaves a silent seam is an existential defect, never deferrable polish. The moat is the opinionated whole, not any one feature.

---

## Operational: beads workflow

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

### Quick reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd export -o .beads/issues.jsonl               # Sync with git
```

### Landing the plane (session completion)

**When ending a work session, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.**

1. **File issues for remaining work** — create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) — tests, linters, builds
3. **Update issue status** — close finished work, update in-progress items
4. **PUSH TO REMOTE** — this is MANDATORY:
   ```bash
   git pull --rebase
   bd export -o .beads/issues.jsonl
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

**Rule for new modifiers that reshape state**: teach `effective_producer_type` about the new rule. The validator picks it up automatically. Do NOT re-inline modifier checks elsewhere — `effective_producer_type` is the single source of truth for modifier-aware type effects.

Current rules encoded in `effective_producer_type`:
- `Each` modifier → `dict[str, output]` (see `state.py:_add_output_field` for the state builder side of the same rule)
- Anything else → raw `output` unchanged

**The declared-output selector is also monopolized**: the `Node.outputs` (plural) vs `Construct.output` (singular) discrimination lives once in `_declared_output` in `_normalize.py` (a neutral low-level module reachable from every layer, including the DX layer `forward.py`). Do NOT hand-roll `getattr(item, 'output', None)` — call `_declared_output(item)`. `TestDeclaredOutputSelectorMonopoly` bans the inline form outside `_normalize.py`. The one sanctioned exception is `compiler.py`'s three-way `isinstance(_BranchNode/Construct/Node)` match, which dispatches to three different graph-builders with different signatures — an irreducible sum-type, not a selector.

### `list[X]` consumers of `Each` producers (merge-after-fan-out)

A downstream node can consume an Each-modified upstream's fanned-out results as a `list[X]`:

```python
@node(outputs=Summary)
def summarize(verify: list[MatchResult]) -> Summary: ...
```

The validator (`_types_compatible`) accepts `list[X]` against a `dict[str, X]` producer when element types are compatible. At runtime, `factory._extract_input` (and the `@node` raw adapter for scripted @nodes) unwraps via `list(values())` before passing the list to the consumer.

**Ordering caveat**: `dict.values()` preserves insertion order, but Each's barrier collects `Send()` results in arrival order, not `each.over` collection order. Use `list[X]` for order-independent reductions (counts, aggregates, summaries). If you need deterministic ordering, consume as `dict[str, X]` and sort explicitly on the key.

### The free Loop scope projections: `all_in_scope` and `from_enclosing(n)`

Mirrors the Each `list[X]` rule above, but for `Loop`. A `Loop`-modified node's state field is *already* a full per-iteration history — `state.py`'s `PrimaryShape.LOOP` case is `Annotated[list[output_type], _append_loop_result]`, and `_append_loop_result` (`_state_reducers.py`) is a pure append (`[*existing, new]`). Loop is sequential (not parallel like Each), so list position **is** iteration order for a non-nested Loop — no new storage, no per-iteration metadata, just a read-time view over what already exists:

- **`all_in_scope`** — a downstream node declares its input as `list[T]` instead of `T`; it receives the FULL history, in order, instead of the latest-only unwrap. `di.py`'s `_unwrap_loop_value` (the single source of truth for Loop unwrap, shared by `_extract_input`/`_resolve_merge_args`/`loop_router`) already passes `list[T]` through unchanged at runtime — this name documents that existing passthrough, it isn't new mechanism.
- **`from_enclosing(n)`** — `history[-n]`, a plain negative index into the same `list[T]` `all_in_scope` sees. No dedicated helper exists or is needed; it's documented shorthand for the slice.

```python
@node(outputs=Draft, loop_when=lambda d: d is None or d.score < 0.9, max_iterations=5)
def refine(seed: Draft) -> Draft: ...

@node(outputs=Summary)
def summarize(refine: list[Draft]) -> Summary:
    # all_in_scope: refine is the FULL per-iteration history, latest last.
    ...
```

**Validator note**: unlike Each (whose producer type is `dict[str, X]` at the type-annotation level, so the existing shape-only `_types_compatible` dict→list rule covers it for free), a Loop-modified producer's declared/effective type stays the bare element type `T` — Loop doesn't change `Node.outputs`/`effective_producer_type` the way Each does. A `list[T]`-declaring consumer therefore needs a *modifier-aware* compatibility check, not a shape-only one: `Producer.is_loop` (`_validation_types.py`) plus `_loop_aware_compatible` (checked at the `_validation_inputs.py` call sites, not inlined into `_types_compatible`) accept `list[T]` against a Loop producer of `T` — element-type-checked, not a blanket bypass.

**Scope**: covers only a single, non-nested Loop. Each indexing (`iteration_index`) and nested/cross-sub-construct scope addressing are separate, unresolved gaps — do not conflate them with this projection.

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

**Gather tool collection**: `invoke_with_tools` collects `ToolInteraction(tool_name, args, result, duration_ms, ordinal)` during the ReAct loop. Collection is UNCONDITIONAL (`_agent_cycle.py`'s tool-call handling always builds and stamps each `ToolInteraction`, including the per-tool-name `ordinal` from `ToolBudgetTracker.record_call`) — only EXPOSURE to the node's declared output is demand-gated: when the node declares `"tool_log"` as an output key, the factory writes the interactions to the tool_log state field via `_shape_tool_output`; when it doesn't, the interactions are still built and stamped, just never surfaced. "No collection overhead if no consumer references tool_log" (the prior wording here) was inaccurate — see `neograph-ftnxl.5`'s corrective note.

**Tool ledger** (`neograph-ftnxl.5`): `ToolLedger` (`_tool_ledger.py`, exported from `neograph`) is a pure read-time selector view over an existing `list[ToolInteraction]` — `first(name)`/`last(name)`/`all(name)`/`grouped()`/`by_key(key)`. The canonical address is `ToolInteraction.key` (`f"{tool_name}#{ordinal}"`), `None` when `ordinal == 0` (a handoff ack or a pre-0.8 checkpointed record — those are UNADDRESSABLE, not merely ordinal-1). `ordinal` is derived from the single already-checkpointed counter (`ToolBudgetTracker._counts`), never a second counter. **Durability boundary**: `tool_name`/`args`/`result`/`duration_ms`/`ordinal` survive checkpoint resume; `typed_result` is documented and TESTED as RESUME-VOLATILE (langgraph's serializer flattens the `Any`-typed field via `model_dump()` — a `BaseModel` result reads back as a plain `dict`). **Breaking-resume note**: adding `ordinal` to the frozen `ToolInteraction` model changes `compute_schema_fingerprint`/`compute_node_fingerprints` for any node that declares `tool_log` as a dict-form output — a pre-`ftnxl.5` checkpoint invalidates on resume (forced re-execution, or `CheckpointSchemaError` with no rewind point). Defensible at 0.x (no back-compat shims), but the effect must be stated, not silent.

**@node decorator**: `@node(outputs={"result": X, "tool_log": list[ToolInteraction]})` passes through. Return annotation inference: `def f() -> X` infers `outputs=X` (single type). Parameters named `{upstream}_{output_key}` are resolved via `_resolve_dict_output_param` in `construct_from_module`.

### A container output type is a first-class declaration

`outputs=list[Reading]` is supported, and so is `dict[str, Reading]`. A node whose whole output IS a collection says so directly; nothing new needs to enter the domain model to give it something declarable.

This was decided rather than inherited (`neograph-tp8dj`). Until 0.7.9 it was a road with three potholes and no signposts: the decorator compared `outputs=` to the return annotation by IDENTITY, so a matching `list[X]` pair was rejected; the json_mode parse tail called `model_validate_json`, which only a `BaseModel` SUBCLASS has, so the declaration failed at runtime on every row; and `outputs=list[X]` appeared zero times across every example, while `_llm_retry`'s bare-array auto-wrap catered specifically to the container-model idiom. All three are fixed and `examples/32_list_output_type.py` exercises the direct road end to end.

**The container model is still right when the collection travels WITH something else** — a summary, a confidence, a cursor — or when the container is a domain concept you would have written anyway. `Each(over="node.items")` fans over its field, which is the shape examples 04, 10 and 17 use.

**The failure mode to watch is minting one container PER NODE during a bug fix.** A downstream consumer measured seven such classes, four introduced while fixing something else, taking an agreed ten-class domain model to eighteen — and a reviewer asking when `RoundDelta` joined the ontology got the answer that it had not, it was minted to satisfy an output declaration. If the container is not a concept you would name on a whiteboard, declare the list.

Both strategies accept a container type: `structured` through constrained decoding, `json_mode` through `TypeAdapter`. They must keep agreeing — a strategy that cannot parse a type the other accepts is the defect this section exists to prevent recurring.

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

- **`node.py`, `construct.py`, `_construct_validation.py`, `factory.py`, `modifiers.py` are off-limits for @node-layer features.** The @node decorator is sugar over the IR; it must produce instances those modules already accept. Fix every @node-layer gap in `decorators.py` without touching the IR.
- **The only exception**: when a genuinely new IR capability is needed (e.g., `ForwardConstruct` needed `_BranchNode` sentinel support in `compiler.py` + `state.py`). Adding those was deliberate and documented.
- **The PORTAL dynamic-handoff exception (mode a peer routing) is the second sanctioned new-IR capability, on the same footing as `_BranchNode`.** (`Portal` was formerly `Keymaker`; the mechanism names below keep the accurate `handoff` word.) A runtime mesh needs IR + runtime support that the linear model does not, so it deliberately adds:
  - **Two IR fields on `Node`** — `Node.handoff_param` (the reserved `handoff` input key, the exact sibling of `fan_out_param`) and `Node.handoff_channel` (the entry-keyed shared mesh channel). **Both are written by a single writer, `_ir_normalize.py`, and nowhere else** — the same single-writer invariant `fan_out_param` has, pinned by guard **G3** (`IR_FIELDS` frozenset in `test_guards_llm_runtime.py`). Do NOT set either field in `decorators.py` / `_construct_builder.py`; the three surfaces converge in the normalizer.
  - **A `Command(goto)` runtime**: a `Portal` member lowers to a wrapper returning LangGraph `Command(goto=..., update=...)`, so control flow is derived at runtime, not from a static edge. **`Command(` may be constructed ONLY in `factory.py` and `runner.py`** — the monopoly that ratchets the new capability, pinned by guard **G1** (`TestCommandConstructionMonopoly` in `test_guards_assembly.py`).
  - **`neo_`-prefixed mesh state keys** built only via `StateKeys.handoff_payload(...)` / `StateKeys.handoff_hops(...)` (no inline f-strings, on top of the Layer-A `neo_`-fragment guard). The mesh-assembly validation rules live in `_validation_portal.py` (`_check_portal_mesh`).

  This is the pattern to copy for any future genuinely-new IR capability: add the field(s) with a single writer, confine the new runtime construct to the compiler/runtime layer, and pin both with structural guards written failing-first.
  - **Lesson from the Portal rollout (2026-07-27), binding on every future modifier or `ModifierCombo` addition**: the single-writer discipline above was followed for Portal's two IR *fields*, but NOT for its *dispatch* — "what does this `ModifierCombo` decompose into, and which consumers need to know" was never centralized. Instead, `compiler.py`, `state.py`, `_state_write.py`, `_subconstruct.py`, `_input_shape.py`, `runner.py`, and `_wiring.py` each independently grew their own combo-membership check (a `match combo:` arm or a `combo in (X, X_OPERATOR)` test) for `PORTAL`/`PORTAL_OPERATOR`. This is the SAME duplicated-source-of-truth anti-pattern the `agent-spec-oracle-inputs-2026-07-25-architecture-retrospective.md` retrospective diagnosed for `_lower_node`/`_lower_oracle`, just codebase-wide instead of file-local — found only because the Agent Spec rewrite's adversarial review asked "does this really sweep every consumer" instead of trusting a hand-picked list (and even that sweep initially missed one file, `_wiring.py`, found only by a second, independent review pass — treat any such list, including this one, as provisional until re-verified by grep, not as closed). **Going forward**: any new `Modifier`/`ModifierCombo` value, or any change to how `Operator`-style orthogonal wrappers compose with a primary modifier, must update ONE shared `ModifierCombo -> decomposition` table and every consumer must read it — never re-derive the decomposition locally. A structural guard enumerating every real consumer (verified by re-running the sweep, not copied from the last time someone ran it) is mandatory before the change is considered done. See `docs/design/modifier-combo-single-source-of-truth-2026-07-27.md` for the full consumer inventory and the fix in progress.
- **Sub-constructs can be @node or declarative.** `construct_from_functions("verify", [explore, score], input=VerifyClaim, output=ClaimResult)` builds a sub-construct from `@node` functions. Params whose type matches `input=` are port params — they read from `neo_subgraph_input` instead of a peer `@node`. The declarative form `Construct(input=X, output=Y, nodes=[...])` also works. Both produce the same IR.

### Naming policy: `__all__` is the public contract; the `_` module prefix is advisory only

The leading-underscore on a *module* name (e.g. `_llm.py`, `_dispatch.py`) is a weak, advisory hint — it does NOT reliably signal public vs internal in either direction. Some unprefixed modules are internal-only (`factory.py`, `state.py`, `di.py`, `naming.py`), and some underscore-prefixed modules export public API through `__all__` (`_llm.py`, `_image.py`). **The single source of truth for what is public is the package facade: a symbol is public iff it is re-exported from `neograph/__init__.py` and listed in its `__all__`.** Do NOT infer a module's or symbol's visibility from its underscore prefix, and do NOT mass-rename to "fix" the mismatch pre-release — the churn is not worth it for one downstream consumer. When adding a new public symbol, wire it through `__init__.__all__`; when adding an internal one, no rename ceremony is required.

**Review checklist item**: when reviewing a change that adds or moves a symbol, confirm its public/internal status is expressed through `neograph/__init__.py`'s `__all__` (the contract), not inferred from the module-name `_` prefix (advisory only).

### Logging convention: module-level bare `get_logger()`

Every module that logs binds a module-level logger with the **bare** call: `log = structlog.get_logger()` (no `__name__` or explicit name argument). structlog resolves the calling module for you, so passing `__name__` is redundant and just invites drift across modules. Do NOT write `get_logger(__name__)` or `get_logger("neograph")` for a new module-level `log`; copy the bare form the other ~19 modules use.

---

## DI surface: `Annotated[T, FromInput/FromConfig]`

DI params use `typing.Annotated` with `FromInput` / `FromConfig` as markers — the FastAPI `Annotated[User, Depends(...)]` pattern.

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

**Why the bundle rule exists**: piarch had 60+ lines of boilerplate repeating `node_id: Annotated[str, FromInput], project_root: Annotated[str, FromInput], ...` across 20 nodes. Bundling a `RunCtx(BaseModel)` eliminates the repetition.

**Classifier implementation notes** (`_classify_di_params` in `_di_classify.py`, imported by `decorators.py`):
- Uses `typing.get_type_hints(f, localns=..., include_extras=True)` to preserve `Annotated` metadata.
- Captures the caller's local namespace in a single shot: `node()` / `merge_fn()` grab `sys._getframe(1).f_locals` ONCE at decoration time (`decorators.py:387`, `:850`) and pass it explicitly as `caller_ns` down to `_classify_di_params` → `_build_annotation_namespace`. No frame-stack walk and no frame-depth arithmetic — the closure carries the captured namespace into `decorator(f)`, so the one hop from user call site to `node()`'s frame is fixed and correct for both the `@node(...)` and bare-`@node` forms. This matters because `from __future__ import annotations` stringifies annotations and strips closure references, so a `class RunCtx` defined inside a test method isn't findable via `f.__globals__` or `f.__closure__`; `caller_ns` supplies it as `localns`. `_build_annotation_namespace` merges the DI markers, the function's closure vars (`inspect.getclosurevars`), and the caller ns (skipping `_`-prefixed names and never shadowing markers).

**Runtime resolution** — one path, `DIBinding.resolve(config)` (`di.py:355`); `_resolve_di_args` (`_di_classify.py`) maps a node's `ParamResolution` to positional args by calling `resolve()` per binding:
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

`src/neograph/describe_type.py` (552 lines, 15 functions) renders Pydantic models into a TypeScript-style notation that LLMs parse more reliably than JSON Schema. Used by the factory layer to build structured output instructions.

**Two public functions** (both re-exported from `neograph`):
- `describe_type(model, prefix=..., hoist_classes=...)` — renders a model class into a schema string with auto-hoisted nested classes
- `describe_value(instance, prefix=...)` — renders a model *instance* as a typed value literal (for few-shot examples)

**Two-pass architecture**: pass 1 (`_count_classes`) counts how many times each nested class appears across the model tree. Pass 2 (`_render_model_body` / `_render_type`) emits the notation, hoisting classes that appear more than once (or all, per `hoist_classes=`).

**Handles**: primitives, `list[T]`, `dict[K,V]`, `Optional[T]`, `Union[A,B]`, `Literal[...]`, `Enum`, nested `BaseModel`, `tuple[...]`, forward refs, field descriptions, constraints, and defaults.

**Tests**: `test_renderers.py` — 148 tests covering all type combinations, edge cases, and round-trip parsing.

---

## RenderedInput — single rendering abstraction

`src/neograph/renderers.py` (`RenderedInput` dataclass at line 34). The single object that bundles all rendering artifacts for prompt construction. Produced by `build_rendered_input(input_data, renderer=None)`.

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
- **`_type_signature(typ)` (structural, not qualname-only)** — BOTH fingerprints fold one level of field detail through this shared helper: a Pydantic model hashes `module.Qualname` + sorted `(field, str(annotation))` pairs; generics (`list[X]`, `dict[K,V]`, Each's `dict[str, X]`) are unwrapped so a change on the wrapped model is visible. This replaced qualname-only hashing so a **same-`__qualname__` model with a changed field type now invalidates** — the old coarse hash was a false-negative that stopped the rewind from triggering at all. Both fingerprints had to move in lockstep: `str(annotation)` on a same-qualname changed field is identical, so without folding the signature into `compute_schema_fingerprint` too, the schema-fp GATE (`_decide_checkpoint_schema` returns `None` on a match) never opens and the enriched node fingerprint would be dead code.

**At compile time** (`compiler.py:332-333`): both fingerprints are stashed on the compiled graph.

**At run time** (`runner.py:607`): the schema fingerprint is injected into the initial state dict (under `StateKeys.SCHEMA_FINGERPRINT`) so it persists in the checkpoint.

**On resume** (`runner.py:_verify_checkpoint_schema` → `_decide_checkpoint_schema`): the stored schema fingerprint is compared against current. If they differ:
- `_compute_invalidated_nodes()` (`runner.py:341`) diffs per-node fingerprints to find which nodes changed.
- `auto_resume=True` (default): `_auto_resume_from_divergence()` walks `get_state_history()` backwards for the OLDEST checkpoint whose `.next` intersects the invalidated set, injects that `checkpoint_id` into config, and `invoke(None)` resumes from there. **Fail-loud on no rewind point:** if `invalidated` is non-empty but NO snapshot has an invalidated node pending in `.next` (history pruned, or every invalidated node already ran), it does NOT silently resume from the tip — it raises `CheckpointSchemaError(invalidated_nodes=...)` via the single-sited `_raise_no_rewind_point`, surfaced BEFORE any node re-executes. Silently resuming would re-hand the caller stale results. Empty `invalidated` stays a genuine no-op (nothing changed).
- `auto_resume=False`: raises `CheckpointSchemaError(invalidated_nodes=...)` for explicit handling.

**What triggers invalidation**: output class renamed, field added/removed/type-changed (including a same-name-class field-type change). Prompt text changes do NOT trigger invalidation (fingerprints are type-based, not content-based).

---

## Lint: template placeholder validation

`lint()` (`src/neograph/lint.py`) now validates template placeholders in addition to DI bindings. Full signature:

```python
lint(construct, *, config=None, known_template_vars=None, template_resolver=None)
```

**One enumeration, two readers.** `iter_di_bindings(item)` in `_lint_di.py` is the single definition of where DI bindings live — a node's own `_param_res` plus its Oracle `merge_fn`'s. Both `lint()`'s payload check and `input_contract()` walk it, so a new binding site reaches both surfaces or neither. Do NOT re-enumerate bindings in a new caller; that is how the two drift.

**Framework consumers.** `output_field_unconsumed` decides deadness across four axes, and the fourth is the one a reader forgets: the FRAMEWORK reads fields too. `_framework_field_reads` (`_lint_consumers.py`) is the SINGLE derivation of those readers, every one of them a name already sitting in the IR — `Each(over="clusters.groups")`, `Portal(route=...)` in peer mode, `Portal(spec_field=..., input_field=...)` in dispatch mode, a branch condition's `attr_chain`, and `Construct.output` (every arm that satisfies the boundary is a terminal producer, not just `nodes[-1]`). **Rule for a new modifier that names a field: teach that one function.** Deriving a reader at a second site is how four of these were missed one at a time, and the miss is silent — the check reports a live field as dead. A `Loop`/`Operator` `when=` callable is deliberately absent: a lambda's field reads are not derivable.

**Two directions.** The linter historically checked one: every kind reported a reference with no source, or a missing config value. None reported a value that arrives and reaches nothing. The supply-side kinds close that asymmetry — `template_input_unreferenced` (a bound input, DI parameter, or context field the node's own template never names) and `output_field_unconsumed` (a produced field nothing reads). Both are WARN, because demand is read from template TEXT and a custom `prompt_compiler` can consume a name the template never spells.

**`lint()` returns defects, and nothing else.** A `FromInput`/`FromConfig` parameter a caller supplies at run time is the graph's INPUT CONTRACT, which is not a defect — so it does not travel in the issue list at all. Read it from `input_contract(construct)`, which returns `InputBinding` records (`node_name`, `param`, `kind`, `source`, `type_name`, `required`, `model_name`), and which `neograph check` prints as its own section outside the lists that decide the exit code. A correct graph therefore lints to ZERO, which is what makes an all-output-fails gate reachable; before that, the strictest available policy was "fails on `required` only", the same trust-the-classification posture a padded config exploits. A config key that matches NO binding is itself an ERROR (`config_key_unmatched`), because a key accepted for being present rather than for being named is how a padded config silences a real defect. Only a binding no caller can satisfy is an error: `from_input_unsatisfiable` fires when an `Each` item or a `Loop` carry is bound with `FromInput`, and it is derived from construct structure so no config can silence it. Demanding a config is what pushed one consumer to pad a fixture with a key no caller could pass, which silenced a real defect while the pipeline computed every fanned branch from the padded value. Pass `config=` to check a specific payload — a different, optional question.

**Three check categories**:
1. DI binding checks — `FromInput`/`FromConfig` params. Only when `config=` is passed: an unsatisfied key is an ERROR, and a key matching no binding is an ERROR. With no config there is nothing to check, and the contract is reported by `input_contract()`.
2. Inline prompt placeholder checks — `${var}` against predicted input dict keys (no flattened, no framework extras)
3. Template-ref placeholder checks — `{var}` against predicted input keys + flattened fields + known extras (requires `template_resolver`)

**The inline/template-ref key asymmetry** is the most common lint confusion. Inline prompts see fewer keys because they resolve via raw attribute access (no rendering pipeline). Template-ref prompts see more keys because the rendering pipeline produces flattened fields and framework extras.

**The THIRD column — `di_inputs`**: a template-ref prompt can ALSO reference a node's `FromInput`/`FromConfig` parameter names (e.g. `{domain}` for `domain: Annotated[str, FromInput]`) — but ONLY when the app's `prompt_compiler` opts in by declaring a `di_inputs` parameter (or `**kwargs`). `lint(construct, ..., prompt_compiler=...)` introspects the compiler's signature with the same `_accepted_params` helper the runtime uses to gate the kwarg; when it accepts `di_inputs`, the node's DI param names become valid template-ref placeholders (`_di_template_var_names`). Without opt-in the placeholder is flagged `template_placeholder_unresolvable` — because the resolved value never reaches the template and the literal `{domain}` would ship to the model. Inline `${var}` prompts NEVER get this column: they resolve via raw attribute access, not the compiler seam. So the full asymmetry is: inline = raw input keys; template-ref = input keys + flattened + framework extras; template-ref WITH a di_inputs-aware compiler = the above **+ DI param names**; and template-ref WITH a `context`-aware compiler = **+ the node's declared `context=` field names** (neograph-ait72, the same opt-in shape via `_compiler_accepts_context`). That fourth column only became correct once `DefaultPromptCompiler` actually threaded `context` — before that, lint calling the placeholder unresolvable was right.

**Runtime-vs-lint coverage (in lockstep)**: the lint column lights up for ANY LLM-mode node (think/agent/act) whose compiler accepts `di_inputs`, and at RUNTIME all three modes now inject di_inputs — `think`/`raw` via `ThinkDispatch` and `agent`/`act` via the ReAct cycle's shared pre-prep (`_agent_cycle._turn_prep_kwargs`). A `{domain}` placeholder that passes lint resolves at runtime for every LLM mode. If a new LLM-mode dispatch path is ever added, it must call `_inject_di_inputs` before its `_compile_prompt` or the lint rule and runtime coverage will silently diverge again.

**`_predict_input_keys(node, include_flattened=True)`** — internal helper that computes what keys a node will see at runtime. `include_flattened=False` for inline, `True` for template-ref.

**Setup module exports** for `neograph check --setup`:
- `get_check_config()` — config dict (required)
- `get_template_resolver()` — `Callable[[str], str | None]` (optional)
- `get_known_template_vars()` — iterable of extra var names (optional)

**Loop condition checks**: lint also validates Loop `when` conditions:
- `loop_condition_unregistered` (ERROR): string condition not in registry
- `loop_condition_none_unsafe` (WARN for callables, ERROR for string conditions): smoke-tests `when(None)` to catch the most common Loop bug -- `lambda d: d.score < 0.8` without a `d is None or` guard

**Oracle merge_prompt upstream context**: `merge_prompt` now passes upstream context alongside the variant list. `make_oracle_merge_fn` accepts a `node_inputs` parameter and builds `{"variants": primary, **upstream_from_state}` as input_data. Templates use `${variants}` for the variant list and `${upstream.field}` for upstream data.

---

## `di_inputs` — resolved DI values reaching prompt templates

An LLM-mode node (`think`/`agent`/`act`) never runs its body, so — unlike scripted nodes, whose shim resolves DI — its `FromInput`/`FromConfig` params are NOT auto-exposed to prompt templates without `di_inputs`: a `domain: Annotated[str, FromInput]` referenced as `{domain}` never becomes a template var, and a fail-soft compiler ships the literal `'{domain}'` to the model. `di_inputs` closes that gap.

**Plumbing (config side-channel, mirrors `_oracle_model`)**: `ThinkDispatch` (`_dispatch.py:_inject_di_inputs`) resolves the node's `_param_res` bindings ONCE via the canonical `DIBinding.resolve(config)` (no second resolver — same path `_resolve_di_args` uses) and stashes the `{param_name: value}` map into `config['configurable']` under `StateKeys.DI_INPUTS` (`_neo_di_inputs`, a config-only key — never enters state, never touches the schema fingerprint). `_compile_prompt` (`_llm_render.py`) reads it back and passes it to the compiler as an **introspection-gated** kwarg via the existing `prompt_compiler_params`/`_ACCEPT_ALL` filter — so only a compiler declaring `di_inputs` (or `**kwargs`) receives it. This avoids threading a new positional through the `_llm`/`_tool_loop` call chain. Only which DI kinds are template-usable is centralized in `di.DI_TEMPLATE_KINDS` (FROM_INPUT/FROM_CONFIG + their MODEL forms; CONSTANT and FROM_STATE excluded).

**Precedence (collision rule)**: in `DefaultPromptCompiler.build_vars` the namespace is layered in order of increasing specificity — `di_inputs` (BASE), then the node's declared `context` fields, then rendered upstream OUTPUTS on top — on a name collision the **upstream output shadows the di_input**. Rationale: an upstream producer named `domain` is the node-local, dataflow-derived value; `di_inputs` is run-wide ambient context, so the narrower binding wins. This is also the **zero-behavior-change** justification: `di_inputs` only fills names not already produced by an upstream node, so no existing pipeline's `{name}` binding changes meaning when a FromInput param collides. `None` collapses to `{}` (the `render_inputs` total-dict contract), so an all-DI leaf node still gets its `{domain}` var. The `context` layer was added by neograph-cbfd9: `__call__` had been swallowing the seam's `context=` kwarg in `**_kw`, so a channel the node author DECLARED never reached the template at all — a `**kwargs` catch-all defeats the introspection gate by claiming every channel. `tests/test_guards_prompt_channels.py` now derives both sides by AST (what `_compile_prompt` offers vs what the shipped compiler declares) so the next channel cannot be forgotten the same way.

**Three-surface parity — decorator-only, by construction**: `di_inputs` is sourced from `node._param_res`, which is populated ONLY by the `@node` decorator's `_classify_di_params`. Declarative `Node(...)` and programmatic `Node() | Modifier()` surfaces carry no `FromInput`/`FromConfig` bindings (DI markers are an `Annotated`-param, decorator-layer concept), so `_param_res` is empty and `_inject_di_inputs` returns config unchanged. The other two surfaces are therefore EXEMPT — there is no DI binding to expose. The lint third-column (`_di_template_var_names`) reflects the same: it reads `_param_res`, so it only lights up for `@node`-built nodes.

**Agent/act (`_agent_cycle.py`) — wired via the same injector**: agent/act nodes compile to a multi-node ReAct cycle that does NOT go through `_execute_node`/`ThinkDispatch`, so the injection happens at `_turn_prep_kwargs` — the single shared pre-prep both sync/async turn-prep twins call, before the cycle's `_compile_prompt`. It reuses the exact same module-level `_inject_di_inputs`, `StateKeys.DI_INPUTS` key, and canonical resolver as think mode (no second mechanism). Per-superstep re-injection is idempotent (copy-not-mutate). All LLM modes now reach the model with di_inputs.

**`raw_di_inputs` — the typed escape hatch (neograph-fqcm6)**: `di_inputs` is always rendered text, same as `input_data`. `input_data` has a raw counterpart (`raw_inputs=`, built from `to_raw_inputs`) for a compiler that needs the live object rather than its rendered form; until 0.7.6, `di_inputs` had no equivalent, so a compiler using a DI scalar for LOGIC (e.g. `isinstance(deal_id, int)`) rather than literal template substitution had no way to recover the typed value — `di_inputs['deal_id']` was unconditionally the string `'4822'`. Fixed by adding `all_kwargs['raw_di_inputs'] = to_raw_inputs(di_inputs)` next to the existing `raw_inputs` line in `_compile_prompt` — same opt-in shape (introspection-gated, `DefaultPromptCompiler` does not declare it), same underlying `to_raw_inputs` (already generic over any `Mapping`, so no new renderer code was needed). `context` had the identical gap under a different excuse (`render_for_prompt()` only customizes rendered TEXT, is per-MODEL rather than per-NODE, and never returns the typed object); it was tracked separately as neograph-ebxdg and is now **closed the same way** — `all_kwargs['raw_context'] = to_raw_inputs(context)`, same opt-in gate, same `to_raw_inputs` reuse. That was the last rendered channel without a raw sibling, so `NO_RAW_SIBLING_ALLOWLIST` in `tests/test_guards_prompt_channels.py` is now **empty** and may only stay that way: a new rendered channel gets a raw sibling, not an allowlist row. Word the guarantee as *"the object `context[k]` was rendered FROM"*, not "the raw state value" — `_extract_context` reads each field through `read_upstream(expected_type=str)`, so a `Loop`-modified upstream yields the LATEST element rather than the append-list (exact parity with the rendered channel beside it, which is the point).

---

## The `@node` modifier-kwarg registry: `_node_modifier_kwargs.py`

`@node`'s modifier-dispatch sugar (`map_over=`, `loop_when=`, `ensemble_n=`, `interrupt_when=`, `portal=`, and their satellites) is driven by a `ModifierCombo`-keyed registry in `src/neograph/_node_modifier_kwargs.py`, not a flat if/elif chain hand-checking each kwarg:

- **`MODIFIER_KWARGS`** — one `ModifierKwargs` row per modifier (each/oracle/operator/loop/portal), each declaring its `triggers` (any ONE non-`None` means the modifier is requested) and `satellites` (kwargs that configure the modifier once triggered, but never trigger it alone — e.g. `max_iterations`/`on_exhaust` are Loop satellites; `on_exhaust` is ALSO a Portal satellite, since it is shared).
- **`IDENTITY_KWARGS`** — every `@node` kwarg with no modifier-dispatch meaning (`mode`, `inputs`, `outputs`, `prompt`, etc.) — always valid regardless of combo.
- **`derive_combo(kwargs, node_label=...)`** — reads which triggers are non-`None`, then resolves the implied modifier-name set through `combo_for_modifier_names` (the one `_COMBO_MAP` validity authority in `modifiers.py`) into a `ModifierCombo`. Raises `ConstructError` for an unrecognized combination.
- **`valid_kwargs(combo)`** — every kwarg a given `ModifierCombo` legally accepts: `IDENTITY_KWARGS` plus every trigger/satellite of every modifier the combo carries (read via `modifier_names_for_combo`, never a hand-typed combo→kwargs table).
- **`_check_kwargs_against_shape(kwargs, combo, node_label=, defaults=)`** — the Phase 3 strictness gate: compares each passed kwarg's VALUE against its live signature default (never `is not None` — `map_on_error` is `node()`'s only non-`None` default, so an is-not-None test would reject it on every non-Each node in the codebase) and raises `ConstructError` naming the offending kwarg + its owning trigger(s) if any passed-and-non-default kwarg falls outside `valid_kwargs(combo)`. Called in `decorators.py` between `derive_combo`/`modifier_names_for_combo` and the five modifier builders — BEFORE any builder side effect (a rejected node must not leave a scripted-registry shim or a `UserWarning` behind it).

**The rule for a new `@node` kwarg**: declare its owning row (as a trigger or satellite) in `MODIFIER_KWARGS`, or add it to `IDENTITY_KWARGS` if it carries no modifier-dispatch meaning. `tests/test_guards_modifier_composition_completeness.py` enforces this totality against `inspect.signature(node)` directly — an undeclared 32nd kwarg fails CI.

---

## Run-scoped state: `context=`, the declared back-reference

A step often needs a value produced earlier in the run that is NOT what its input port hands it — run identity, a session handle, a briefing. `context=["ctx"]` on a Node reads that value straight from state, without it being threaded through every intervening shape.

```python
verify = Node(name="verify", mode="think", inputs=Claim, outputs=Verdict,
              model="fast", prompt="check", context=["ctx"]) | Each(over="claims.items", key="text")
# every branch: port item = one Claim, context = the run's RunCtx
```

Three properties, and each is the reason it exists rather than a side effect:

- **Declared, never ambient.** `_construct_validation` checks that some upstream produces the named field, so a missing binding is a `ConstructError` at assembly. A reader of the node sees everything it consumes — the pure-function bias is preserved, which an invisible global would destroy.
- **Works under fan-out.** The port carries WHICH ITEM, `context` carries WHICH RUN. These were never competing, and a fanned branch is the shape people most often believe is impossible.
- **Reads STATE, not config.** So a value that changes during the run — a session restored after a HITL gate fires hours later — is expressible. Static config cannot express that; this is the distinction to reach for when someone proposes threading a value through `run(input=)`.

Verbatim (un-rendered) delivery is a PROPERTY of the channel, useful for a pre-formatted catalog. It is not the purpose. Documenting it as the purpose is what made the capability undiscoverable for years — a consumer filed a design proposal for a feature that already shipped (GH #15). When you document a general mechanism through its narrowest use case, the people who need the general form cannot find it.

**Known limits, both tracked**: LLM-mode nodes only (`_execute` gates on `node.mode != "scripted"`), and `context=` makes the model SEE a value — nothing BINDS it into a tool call, so a model can still compose a tool argument of its own invention.

---

## Run-scoped state: how a step reaches a value produced earlier

An LLM-mode node declares `context=["ctx"]` and reads the field straight from state — declared, validated at assembly, and it works inside a fan-out where the port already carries the mapped item.

A SCRIPTED node needs no such mechanism, and deliberately does not get one (`neograph-7e065`). An upstream read is already a normal typed input, and dict-form `inputs` lets a fanned branch declare both at once:

```python
@node(outputs=Out, map_over="claims.items", map_key="text")
def branch(item: Claim, ctx: RunCtx) -> Out: ...   # item = WHICH ITEM, ctx = WHICH RUN
```

That route is **better** than `context=`, not merely equivalent: the validator type-checks a fan-in input and it creates a real dataflow edge, while a `context` field is typed `Any` in `state.py` and declares none. Widening `context=` to scripted nodes would add a second, weaker way to do one thing. Pinned by `tests/test_scripted_run_state.py`.

**What neither closes**: a value the model composes into a TOOL CALL is still the model's until it is bound. Use `Tool(bound_args={"warehouse_id": "audit.warehouse_id"})` when the argument's correctness matters — seeing a value and being unable to override it are different guarantees.

---

## Modes and mode inference

`@node` supports five execution modes:

| Mode | When | Body runs? | Dispatch |
|---|---|---|---|
| `scripted` | No `prompt=`/`model=` | ✓ | `_execute_node` via `ScriptedDispatch` |
| `think` | `prompt=` + `model=` present | ✗ (dead code) | `_execute_node` via `ThinkDispatch` |
| `agent` | Same + `tools=` (read-only) | ✗ | ReAct cycle (`_agent_cycle.py`) |
| `act` | Same + `tools=` (mutations) | ✗ | ReAct cycle (`_agent_cycle.py`) |
| `raw` | Explicit `mode='raw'` | ✓ | `factory._make_raw_wrapper` via `raw_fn` |

**Mode inference**: if `mode=` is not passed, the decorator looks at other kwargs — `prompt=` + `model=` → `think`; neither → `scripted`. Mode `raw` always requires explicit opt-in (enforces the `(state, config)` signature).

**Dead-body warning**: LLM modes emit a `UserWarning` at decoration time if the function body is non-trivial (not `...`, `pass`, or a bare return). AST-based check — handles common false positives.

**Scripted `@node` dispatches via `register_scripted`.** At construct-assembly time, `_register_node_scripted` in `_scripted_registry.py` (re-exported through `decorators.py`) builds a shim closure that resolves `FromInput`/`FromConfig`/constant params from `config`, reads upstream values from `input_data` (the dict returned by `factory._extract_input`), and calls the user function with positional args. The shim is registered via `register_scripted` under a synthesized name, and `node.scripted_fn` points to it. The factory's `_execute_node` picks it up via `ScriptedDispatch` — **one dispatch path for all node modes**.

`Node.fan_out_param` tells `_extract_input` which `inputs` key should read from `state["neo_each_item"]` instead of from a named upstream field. This is the only IR-level concession to the `@node` layer — it applies equally to programmatic `Each` nodes with dict-form inputs.

---

## Git workflow

- **`main`** — stable. Only tagged releases and critical hotfix PRs.
- **`develop`** — active development. All new work lands here. The authoritative version is `__version__` in `src/neograph/__init__.py` (do not hard-code it here — it drifts). Piarch and other downstream consumers pull from this branch via `uv add "neograph @ git+https://github.com/KonstantinMirin/neograph.git@develop"`.
- **Forward-port**: after a release tag, merge `main` back to `develop` and then run `make forward-port-check`. A forward-port merge resolves conflicts file by file, and resolving in favour of `develop`'s side silently discards what the release branch documented — which is how the 0.7.8 port dropped two CHANGELOG sections, the AGENTS.md lint documentation, and a drafted upstream report while the test suite stayed green. The check compares release headings and `docs/` paths across the two branches, because a passing test run cannot distinguish "the docs merged" from "the docs were discarded".
- **Release path**: when `develop` is ready, merge to `main`, **run `make release-gate` on merged `main`**, then tag `vX.Y.Z` and push the tag. `.github/workflows/publish.yml` triggers on `v*` tags and publishes to PyPI via Trusted Publishing (no tokens, OIDC-scoped).
- **Version bumps**: on `develop` we increment normally. On `main` at the release tag we tag `vX.Y.Z`. `__version__` and `pyproject.toml`'s `version` move together — `TestVersionSync` fails if you bump one alone.

### The release gate is MANDATORY on merged `main`, before the tag

```bash
git checkout main && git merge --no-ff release/X.Y.Z -m "release: merge X.Y.Z (...)"
set -a && . .env && set +a          # live credentials
make release-gate                    # quality + live + mcp + examples + website + skipcheck
git tag -a vX.Y.Z -m "..." && git push origin main && git push origin vX.Y.Z
```

**`make quality` is NOT sufficient to tag**, and neither was `quality + live`. Both report success while an arbitrary subset of the suite does not run. `make release-gate` runs six targets:

| target | covers |
|---|---|
| `quality` | the offline suite, ruff, mypy |
| `live` | the real Langfuse checks, with `NEOGRAPH_REQUIRE_LIVE=1` so absent credentials are a hard ERROR rather than a skip |
| `mcp` | the suite with `--extra mcp --extra mcp-examples`, so `neograph[mcp]` — a second shipped top-level package — is exercised instead of importorskipped |
| `examples` | every keyless example end to end, plus the MCP e2e harness |
| `website` | `npm ci && npm run build`, because the api-manifest guard couples page content to the public API |
| `skipcheck` | `scripts/check_skips.py`, which fails on ANY skipped test — there is no allowlist |

**A skip is invisible in a pass count**, so `skipcheck` fails on ANY skip and there is **no allowlist**. One shipped briefly (`tests/skip_allowlist.txt`, "empty by design, may only shrink") and was deleted without a single entry ever being added: a test exists to verify a behaviour, so a test that does not run is a defect with a cause, and writing its reason into a file does not fix the cause — it only stops anyone being told about it. When a behaviour is genuinely known-broken, mark it `xfail(strict=True)`: reported distinctly from a pass, and it turns RED the moment the gap closes, so the exemption cannot outlive its reason. That is the property an allowlist can never have.

**Tag the commit you gated.** Not a later one, and not the branch tip if it moved.

This exists because the same failure happened twice, in the same shape: a success signal compatible with the thing you care about not running.

0.7.4 was tagged after a green `make quality` on merged `main` that had silently skipped the two live tests for want of exported keys; one of them was flaky and the tag went out with it. The build was caught at the manual-approval gate and the tag was moved before anything reached PyPI, but only by luck of the reviewer gate.

0.7.7 was then tagged through a green `quality + live` gate while 74 `mcp` tests skipped for a missing extra, the examples were never run, and the website was never built. Measured afterwards: with every extra installed the suite is 3338 passed and 0 skipped, where that gate saw 3224 passed and 86 skipped. 114 tests contributed nothing. The rule is now mechanical, not remembered, and `skipcheck` is what makes it so.

**Never publish directly.** The GitHub Actions workflow is the only publish path. This gives us a pypi.org Trusted Publisher gate + an optional manual-approval environment reviewer.

**If a tag must be moved** (nothing published yet — check `curl -s https://pypi.org/pypi/neograph/json | jq -r .info.version`): cancel the pending run first (`gh run cancel <id>`) so it cannot be approved by accident, then `git push --delete origin vX.Y.Z && git tag -d vX.Y.Z`, re-gate, re-tag, re-push. Once a version is on PyPI it can never be reused — at that point the only path is a new patch version.

---

## Test conventions

### The gate command, and which extras it does NOT include

**Run the whole suite with `uv run pytest`.** That is it — no `--extra`. `make quality`
(`test` + `lint` + `typecheck`) runs the same thing. Everything the gate needs, including
`pyagentspec` and the toolchain (`pytest`, `pytest-asyncio`, `ruff`, `mypy`), lives in
`[dependency-groups].dev`, which uv installs by default.

There is deliberately **no `dev` extra** (neograph-x75es). Older docs and commit messages say
`uv run --extra dev pytest`; that form is gone. Having both a `dev` extra and a `dev`
dependency-group gave every new dev dependency two plausible homes, only one of which the gate
reads — and that is exactly how 271 Agent Spec tests plus `pytest-asyncio` ended up outside the
gate. **One namespace: `[dependency-groups].dev`.** Optional *consumer-facing* extras
(`agent-spec`, `mcp`, `mcp-examples`, `langfuse`) are unaffected and still exist.

What the bare gate still does NOT run — the MCP suites, and only those:

| Suite | Command | Why it stays extra-gated |
|---|---|---|
| `tests/test_mcp_examples_e2e.py` | `uv run --extra mcp-examples pytest tests/test_mcp_examples_e2e.py` | `importorskip`-gated; real MCP over stdio |
| `tests/test_mcp_battery.py`, `test_mcp_oauth.py`, `test_mcp_transport_resilience.py`, `test_mcp_fakes.py`, MCP half of `test_guards_api_manifest.py` | `uv run --extra mcp pytest <file>` | `skipif(not _HAS_MCP)`; ~82 tests total |

That optionality is **structural, not an oversight**: the no-session-ownership guard's premise is
that the MCP stack stays out of the core install. "No-key" != "no extra" (see the MCP examples
section). Everything else runs by default.

**Adding a new dependency-gated test?** `tests/test_guards_test_gate_deps.py` enforces the rule:
every `pytest.importorskip("x")` and every `find_spec("x")`-fed `skipif` must have its
distribution in `[dependency-groups].dev`, or be listed in `GATED_IMPORT_EXEMPTIONS` with a
structural reason (checked non-vacuous against the extra it names, and checked for staleness).
Prefer the default group — `importorskip` is the SILENT form: the module yields zero tests and
the summary line says nothing, so a green gate proves nothing about it.

**Do NOT "tidy" `importorskip` onto the ~32 Agent Spec tests that hard-fail without
`pyagentspec`.** The asymmetry (13 files guarded, ~32 tests unguarded) is deliberate: a loud
failure beats a silent skip, and adding guards would spread the disease this rule exists to
remove.

**`src/neograph` stays Agent-Spec-free.** Now that `pyagentspec` is always installed, that is no
longer proven by the dependency simply being absent — it is proven explicitly by
`tests/test_guards_agent_spec_core_purity.py` (subprocess + `sys.modules`, plus a static
module-level-import scan), the same shape as the MCP purity guard in `test_mcp_battery.py`.

### Test file layout

The suite grows every wave, so the counts below rot fast — recount rather than
trust a frozen number: `ls tests/test_*.py | wc -l` (root),
`ls tests/{decorator,modes,modifiers,hypothesis}/test_*.py | wc -l` (packages).
As of 2026-07-14: **122 root `test_*.py` files + 30 package files = 152 total.**

**Root tests** (~122 files). The table below is a REPRESENTATIVE index of the
primary suites, not an exhaustive enumeration — many focused files (async,
checkpoint, MCP, guards, observability) are not listed row-by-row.

`test_validation.py` and `test_structural_guards.py` were split by concern (no file exceeds 1200 lines; class names unchanged so guards stay
discoverable). The validation suite is now several files; the structural-guard
suite has since grown to ~27 `test_guards_*.py` files (`ls tests/test_guards_*.py`).

| File | Scope | Tests |
|------|-------|-------|
| `test_validation.py` | Core assembly validation: construct/oracle errors, Each-path, name collision, tool/LLM config, output strategy, error builder, TypeSpec, FromInput-required, single-type deprecation | ~72 |
| `test_fanin_validation.py` | Fan-in: dict-form inputs, Each interop, effective_producer_type, list/dict compat, dict-form outputs, three-surface parity | ~35 |
| `test_lint.py` | lint() DI bindings, obligation gaps, Loop condition checks | ~34 |
| `test_template_lint.py` | lint() inline `${var}` and template-ref `{var}` placeholder checks | ~50 |
| `test_context_validation.py` | Sub-construct context-field + output-boundary validation | ~15 |
| `test_guards_assembly.py` | Guards: error builder, file-split, assembly import DAG, subconstruct boundaries, dead code, no-Any boundaries, no-sidecar-pattern | ~100 |
| `test_guards_ir_compiler.py` | Guards: IR typing, compiler wiring, node mutation, branch nodes, build-construct body size, registry dicts | ~30 |
| `test_guards_sidecar_imports.py` | Guards: sidecar module, function-local import allowlist, tool-loop import graph, langgraph imports, IO polymorphism | ~21 |
| `test_guards_any_audit.py` | Guards: no-Any in public IR APIs, arbitrary-types justification, public functions raise NeographError | ~10 |
| `test_guards_function_local_imports.py` | Guards: function-local factory/llm imports, retry-policy signature, StateKeys centralization, no module-level registration | ~45 |
| `test_guards_llm_runtime.py` | Guards: factory kwargs, LLM responsibility/cohesion, StateBus.get discipline, runtime fan-out, normalize_ir field writer, routing-key invariant | ~81 |
| `test_renderers.py` | XmlRenderer, DelimitedRenderer, JsonRenderer, describe_type, render_prompt | ~148 |
| `test_forward.py` | ForwardConstruct base class, tracer, compilation, branching, loops | ~95 |
| `test_composition.py` | Sub-constructs, @node sub-constructs, state hygiene, reducers, dict-form | ~95 |
| `test_coverage_gaps.py` | Coverage gap tests for uncovered code paths | ~60 |
| `test_conditions.py` | parse_condition, condition registry | ~45 |
| `test_loop.py` | Loop modifier: self-loop, Loop-on-Construct, ForwardConstruct, skip_when | ~41 |
| `test_node_sidecar_contract.py` | Pins PrivateAttr (`_sidecar`/`_param_res`/`_scripted_shim`) preservation across model_copy/pipe/deepcopy | ~8 |
| `test_inline_prompts.py` | Inline prompt compilation, template rendering | ~64 |
| `test_di.py` | DI bindings, resolution, typed fields | ~37 |
| `test_spec_loader.py` | YAML/spec loader, type resolution | ~44 |
| `test_obligation_r1r2.py` | Behavioral obligation tests | ~23 |
| `test_cli.py` | CLI entry points | ~30 |
| `test_spec_types.py` | Type registry | ~20 |
| `test_spec_schema.py` | Spec schema validation | ~14 |
| `test_model_compat.py` | Pydantic model compatibility | ~14 |
| `test_fakes.py` | LLM fake infrastructure tests | ~7 |
| `test_check_fixtures.py` | Compiler safety net (parametrized fixtures) | ~2 |
| `test_checkpoint_auto_rewind.py` | Schema-aware auto-rewind: fail-loud-on-no-rewind-point contract, sync + async | ~10 |

**Package tests** (30 `test_*.py` files across 4 packages):

| Package | Files | Scope | Tests |
|---------|-------|-------|-------|
| `decorator/` | 5 files | @node, @tool, @merge_fn decorators; mode inference; DI (incl. `TestMergeFnDuplicateRegistration`); construct assembly; edge cases | ~183 |
| `modes/` | 10 files | Scripted/think/agent/act/raw modes; execution; output strategies; LLM internals; I/O | ~360 |
| `modifiers/` | 6 files | Oracle, Each, Operator, compositions, modifier edge cases | ~204 |
| `hypothesis/` | 9 files | Property-based testing: topologies, invariants, regression | ~130 |

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
- The fixture author should be different from the validation author when possible — a fixture written AFTER the validation is "done" catches gaps the author's own fixtures miss.
- Keep fixtures minimal — one Construct, one defect, ~15 lines.

### The file-size ratchet (`tests/test_guards_file_size.py`)

**The goal is not 500 lines.** The number is a proxy. What it is a proxy *for*: a module
that has grown long enough to hold parts which are **conceptually separate but
implementationally entangled** — clusters that no longer belong together, yet share
helpers, constants and imports densely enough that nobody can tell where the seam is.
That state is invisible until someone tries to move something, and by then the coupling
is load-bearing.

Forcing the split surgically is what surfaces the seam. Every extraction in the
`neograph-3ffdg` wave found something the surveys had not: a helper with no callers in
its own file, a cluster interleaved with another's target, a "self-contained" region
with a runtime dependency on its parent. **The architecture improved because the
procedure forced the question, not because a file got shorter.**

Read that as the decision rule whenever the two conflict. A split that lands a file at
509 with clean boundaries beats one that reaches 499 by putting a function in the wrong
module. `_oracle.py` sits ten lines over its cap for exactly this reason:
`_inject_oracle_config` would have closed the gap and belonged nowhere near the cluster
it would have joined.

#### What the guard enforces

Every `.py` file under `src/neograph/` (recursively, so `testing/` is included) must be
under 500 lines unless it has an entry in that guard's `ALLOWLIST`. The allowlist is a
per-file **exact ceiling**, not a blanket exemption, and obeys the same shrink-only
discipline as this repo's other ratchets: **growth is blocked and fixed in-PR, never
deferred.**

- **Exact, not merely sufficient.** `ALLOWLIST[f] == len(read_text().splitlines())`. A
  ceiling sitting above its file is silent headroom for future growth. Both hand-rolled
  caps this guard replaced had already drifted stale-loose before anyone noticed
  (`compiler.py` 775 vs 761 real; `_llm_retry.py` 665 vs 658) — which is why there is no
  tolerance band.
- **A shrink lowers the ceiling in the same commit.** **A red `test_guards_file_size`
  after a successful split is the expected, correct signal, not a regression to route
  around.** The failure message prints the paste-ready replacement literal.
- **A file that drops under 500 has its entry DELETED**, never lowered to a sub-500
  number — otherwise the allowlist quietly grants a private ceiling to a file the plain
  rule should govern.

A ruff-format pass legitimately changes line counts and so legitimately requires a
number update. Keep the dict one-entry-per-line and sorted by posix path, so parallel
branches edit disjoint lines.

#### Two refusals that outrank the ceiling

Both were exercised in the 3ffdg wave and both are binding, not precedent-by-accident.
When a split can only be completed by doing one of these, **take the smaller extraction
and file the remainder** — a file left over its ceiling with a written reason is a
better outcome than either.

1. **Never widen a capability monopoly to buy a split.** If moving a cluster would add a
   module to `_ALLOWED` in guard G1 (`Command` construction), to
   `ALLOWED_GRAPH_ONLY_MODULES` (three-layer engine verbs), or to
   `FUNCTION_LOCAL_IMPORT_ALLOWLIST`, the split is wrong as scoped. Those ratchets exist
   to keep a dangerous capability confined; spending one to shorten a file trades a real
   invariant for a cosmetic metric. `factory.py` stays at 949 for this reason.
   Distinguish three cases: **re-key** (the thing moved — point the entry at the new
   module and remove the old), **redistribute** (N justified uses scattered across N
   files, total unchanged — list them all and say the count), **widen** (a capability
   reaches somewhere new — refuse).

2. **Never let a "pure split" become a behaviour change.** A move may add a parameter to
   thread a dependency the cluster genuinely needs (`export_flow`, `shim_factory`,
   `resolve_condition` — all landed this way). It may not restructure mutual recursion,
   change what a function computes, or require new tests to prove it still works. When
   the only way through is three injected callables across six functions, that is a
   design ticket, not a file split. `loader.py` stays at 1181 for this reason.

The corresponding decision ladder for dependencies, in order: annotation-only reference
→ `TYPE_CHECKING`; helper used only by the moving cluster → move it too; otherwise
inject as a parameter; otherwise take a smaller extraction. A deferred/function-local
import is never the answer — it requires growing an allowlist.

Two narrower line caps deliberately survive alongside the ratchet — `LINE_CAP` 330 and
400 in `test_guards_assembly.py`. They govern **disjoint** file sets (every file they
cover is well under 500), so no file is capped twice. They keep loose `>` semantics;
tightening them was out of scope, not endorsed.

The operational how-to for performing a split — the inventory sweeps, AST-with-decorators
slicing, the import-surface proof, the ordering of ceiling vs formatting — lives in
[`docs/file-split-procedure.md`](docs/file-split-procedure.md). It is a living document,
rewritten in place rather than appended to; update it there when a split teaches something
new.

### General test conventions

- **New tests go in the matching file.** If a feature spans multiple files, put the test where the primary behavior lives and add cross-references in docstrings.
- **BDD naming**: `test_{what_should_happen}_when_{condition}`. Class docstrings describe the feature being tested.
- **Throwaway modules for `construct_from_module` tests**: use `types.ModuleType("test_xyz_mod")` and attach `@node` functions as attributes. Don't pollute real modules. Pattern is `TestNodeDecorator._fresh_module`.
- **Fakes live in `tests/fakes.py`**: `FakeTool`, `StructuredFake`, `TextFake`, `ReActFake`, `configure_fake_llm`. Don't invent new fakes unless the existing ones genuinely don't cover the case.
- **TDD the user explicitly expects**: for bug fixes, write the failing repro first, verify it fails, then fix. Honor it on every bug-fix task.
- **Three-surface parity rule**: any IR-level behavioral change (`node.py`, `_construct_validation.py`, `factory.py`, `state.py`) must be tested through all three API surfaces -- `@node` decorator, declarative `Node.scripted()`, and programmatic `Node() | Modifier()`. This is the most common source of bugs: a feature works via `@node` (which runs through `_build_construct_from_decorated`) but breaks via the programmatic API (which goes straight to `Construct(nodes=[...])`). Canonical failure: `fan_out_param` set only in the decorator path, so programmatic `Each` + dict-form inputs fail validation. Test all three surfaces or explain why a surface is exempt.

---

## Examples

30+ runnable examples in `examples/`, most narrated as a walkthrough on neograph.pro. Most use `@node` except two that stay declarative (example 10 mixed, example 11 config injection). Example 27 is the ForwardConstruct imperative-wiring showcase (branch/self.loop/self.each/self.ensemble/self.interrupt, keyless, pinned by `tests/test_example_forward_wiring.py`). Examples 28/29 are the `Portal` dynamic-handoff showcases (peer-routing mesh + runtime flow dispatch), pinned by `tests/test_example_portal.py` / `tests/test_example_portal_dynamic_flow.py`. Sub-constructs (example 05) can now use either `@node` with `construct_from_functions(input=, output=)` or declarative `Construct(input=, output=, nodes=[...])`.

**Examples must run end-to-end.** Breaking one is a regression. When you change an API surface, run every example that doesn't require real API keys (01, 01c, 02, 03, 04, 05, 06, 08, 09, 10). The keyed examples are 07 and observable_pipeline.py — both hit real OpenRouter (observable_pipeline additionally pushes to Langfuse; run it with `--extra langfuse`), and both were verified passing end-to-end on 2026-07-09. Example 11 was converted to a FakeLLM and is keyless. Document any new failures separately.

### Live Langfuse correlation check — needs REAL keys

`tests/test_observe_trace_live.py` is the only test that talks to a live external
service. It proves the half of neograph-s65y2 the offline suite structurally
cannot: that Langfuse actually records the trace under the id neograph derived
from `run_id` (`Langfuse.create_trace_id(seed=run_id)`, handed over as
`trace_context`). Its control asserts the raw `run_id` still 404s — that was the
original bug, and a 200 there would mean the two identity spaces collided.

```bash
set -a && . .env && set +a
uv run --extra langfuse pytest tests/test_observe_trace_live.py
```

Without `LANGFUSE_SECRET_KEY` + `LANGFUSE_PUBLIC_KEY` the module skips (2 skips
in the default gate's count). **That skip is a documented hole, not coverage** —
a green `make quality` says nothing about the live path. Re-run it after any
change to `_merge_observe_callbacks`, `_identity_binds`, or the langfuse pin.
Langfuse ingests asynchronously (~25s observed to first 200), so the test polls;
it is slow by nature and deliberately not in the default gate.

### MCP examples (23/24/25/26) — no-key but need the `mcp-examples` extra

The MCP-featuring examples exercise the **real** Model Context Protocol against a
shared stdio demo server (`examples/_mcp_demo_server.py`) — no fakes at the
protocol layer, no network, no API keys. They are **keyless but NOT
dependency-light**: they need `mcp` + `langchain-mcp-adapters`, which live in the
`mcp-examples` optional extra (`[project.optional-dependencies].mcp-examples`),
**not** core deps and **not** the default dev group. This keeps `src/neograph`
MCP-free (the no-session-ownership guard scans `src/` only) and the core
`uv run pytest` suite light.

- **Run the MCP E2E harness**: `uv run --extra mcp-examples pytest tests/test_mcp_examples_e2e.py`
- The harness (`tests/test_mcp_examples_e2e.py`) is `pytest.importorskip`-gated, so
  the core suite **skips** it cleanly without the extra. It proves the demo server
  end-to-end (tool discovery, `get_deal` resource_link manifest, RFC-6570 email
  fraction read, per-operator auth echo, real `-32002` expiry + self-heal) and
  auto-discovers `examples/2?_mcp_*.py` to run each example as a subprocess (23/24
  are tool-factory examples; 25 illustrates the singular
  `mcp_tool_factory` — offline build + gateway rename; 26 exercises the `mcp_session` composite over one connection).
- **The distinction to remember**: "no-key" ≠ "no extra". Examples 23/24/25/26 are on the
  no-key list but you must pass `--extra mcp-examples` to run them or their tests.
- **Two verified `mcp` 1.28.x SDK gaps the demo server works around** (documented
  in the server's module docstring): FastMCP's `@mcp.resource` can't express
  RFC-6570 query templates, and its `@tool`/`@resource` wrappers swallow JSON-RPC
  error codes (a real `-32002` needs a custom low-level `read_resource` handler on
  `mcp._mcp_server`). Pin is `mcp>=1.28,<2` (mcp 2.0 renames `FastMCP`→`MCPServer`).

---

## Website

Astro + Starlight at `website/`. Deployed on Amplify from the main repo, triggered by any push that touches `website/` (actually just any push — Amplify rebuilds on every commit). The build must succeed or the site breaks.

**Always run `npm run build` in `website/` after content changes.** 46 pages, build takes ~2 seconds. Silent breakages are rare but possible (broken MDX frontmatter, missing `Annotated` import in code examples, etc.).

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
- **`FromInput[T]` / `FromConfig[T]` Generic subscription form**: not supported — use `Annotated[T, FromInput]`. The subscription form raises `TypeError: type 'FromInput' is not subscriptable` (intentional; clean error beats silent breakage).
- **Emojis in docs**: the user explicitly rejected them ("kill emojis, that's sooo LLM-ish"). Don't reintroduce. If a code comment uses one, replace with plain text.
- **Line counts as a value metric**: the user explicitly rejected framing value around "X lines vs Y lines". The docs talk about what neograph *does* (type safety, durability, observability, focus on logic), not how many lines shorter it is than raw LangGraph. Comparison table "What you don't write" stays on the Why-not-LangGraph page but isn't on the landing.
- **`TestPyPI` in the release flow**: not used. TestPyPI is optional, not required, for alpha releases.

---

## Known open DX items

These aren't bugs, just things worth considering for future sessions:

- `@merge_fn` uses a function-name-keyed registry (`_merge_fn_registry` in `decorators.py`). Keep it name-keyed — it is a structurally-required symbol table, not registry debt: `@merge_fn` decorates a standalone function and returns the bare function, which is referenced from a DIFFERENT object (`Oracle(merge_fn='combine')`) purely by STRING NAME, so no Node/Oracle is in scope at decoration time to self-store on (unlike `@node`, which self-stores on a sidecar because it *returns the Node it decorates*). It mirrors conditions/tool_factories, which the per-compile architecture also seeds from a decoration-time global. Same-name collisions between DIFFERENT definition sites FAIL LOUD (`ConstructError` naming both `module.qualname` + `file:lineno`); the same definition site is idempotent (re-import, re-run in a loop/hypothesis, module reload) via `_same_def_site`. If you add another decorator referenced by string name, copy this pattern (name-keyed registry + fail-loud collision guard).
- The sponsor banner on neograph.pro is hardcoded in a component. If we ever add more sponsors or commercial positioning, it should probably move to config.

---

## User preferences

- **Blunt, direct answers preferred over agreement.** If an API has a DX problem, say so. The user will happily refactor at 0.x.
- **No backwards-compat shims at 0.x.** Breaking changes are fine; deprecation cycles are unnecessary at this scale and one known user.
- **TDD for bug fixes, always.** Write the failing test first.
- **Parallel agent teams for multi-file work.** The `/team` slash command invokes a team with scoped file regions. Use it for anything that can be parallelized without file conflicts.
- **User is the sole maintainer and sole downstream user (piarch).** No migration burden for hypothetical users.

## Beads storage — updated 2026-08-18 (supersedes older guidance in this file)

bd talks to a **local dolt sql-server** for this project (`.beads/dolt/<db>`); worktrees share it
through a `.beads/redirect` file rather than keeping their own tracker. A shared Dolt server on
Hetzner, reachable only over Tailscale, holds a replicated copy — it is a **backup target, never
the live database**.

- **Never point bd at the remote.** Measured: ~6.8 s per `bd ready` remote vs ~0.2 s local, because
  bd makes ~200 sequential round trips per command and has no connection reuse.
- **Never run `bd dolt push` / `bd dolt pull`.** They fail with `Access denied for user 'root'` —
  Dolt's server-side push path hardcodes root with an empty password and no config can change it.
  Replicate with `ox-troubleshooting-demo/scripts/beads_sync.sh`.
- **`bd sync` does not exist** in bd 1.1.2 — not `--flush-only`, not `--from-main`. JSONL export is
  automatic; `bd export -o .beads/issues.jsonl` is the explicit form.
- **If bd reports "No issues found" unexpectedly, stop and diagnose — do not create anything.**
  `.beads/metadata.json` is a git-tracked pointer to the database, so a branch checkout can silently
  revert it; bd then creates an EMPTY database rather than failing. This once hid 5279 issues.
  Run `make beads-preflight` (in ox-troubleshooting-demo) to check every project.
- **Where `.beads` is gitignored, `export.git-add` must stay `false`**, or every bd write fails with
  `auto-export: git add failed` while reads keep working.

Full rationale and measurements:
`pi-agentic-coding/plugins/dev-practices/skills/session-completion/references/beads-topology.md`
