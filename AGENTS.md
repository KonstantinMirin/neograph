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
| `ForwardConstruct` | Pipelines with Python control flow (`if`/`for`/`try`). Class-based, `forward()` traced via symbolic proxies. | `src/neograph/forward.py` |
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
- **Sub-constructs stay declarative.** `Construct(input=X, output=Y, nodes=[...])` is the explicit boundary for isolated sub-pipelines. `construct_from_module` produces one `Construct` per module; it does not inline sub-pipelines. Example 05 and example 10 deliberately keep the declarative form for this reason.

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

`@node` can't mutate `Node` (pydantic v2 with schema validation). Instead:

- `_node_sidecar: dict[id(Node), (fn, param_names, fan_out_param)]` — stores the original function, its parameter tuple, and the fan-out param name (for Each).
- `_param_resolutions: dict[id(Node), ParamResolution]` — separate sidecar for DI metadata (kept separate so the 3-tuple contract of `_node_sidecar` stays stable).
- `weakref.finalize(node, pop_id, node_id)` on both — entries evict when the Node is garbage collected.

**When you apply a modifier via `|`** (e.g., `@node(map_over=...)`, `@node(ensemble_n=...)`, `@node(interrupt_when=...)`), the modifier returns a new `Node` via `model_copy`. The sidecars are keyed by `id()`, so the new instance has no entry. **You must re-register the sidecars on the modified copy.** Grep for `_register_sidecar(n, ...)` after `n = n | Oracle(...)` / `n = n | Each(...)` / `n = n | Operator(...)` — every modifier attachment in `@node` does this.

This was the cause of `neograph-jyw` before the fix. Any new modifier kwarg you add must follow the same pattern.

---

## Modes and mode inference

`@node` supports five execution modes:

| Mode | When | Body runs? | Dispatch |
|---|---|---|---|
| `scripted` | No `prompt=`/`model=` | ✓ | `factory._make_scripted_wrapper` via `raw_fn` |
| `produce` | `prompt=` + `model=` present | ✗ (dead code) | `factory._make_produce_wrapper` |
| `gather` | Same + `tools=` (read-only) | ✗ | `factory._make_gather_wrapper` |
| `execute` | Same + `tools=` (mutations) | ✗ | `factory._make_execute_wrapper` |
| `raw` | Explicit `mode='raw'` | ✓ | `factory._make_raw_wrapper` via `raw_fn` |

**Mode inference**: if `mode=` is not passed, the decorator looks at other kwargs — `prompt=` + `model=` → `produce`; neither → `scripted`. Mode `raw` always requires explicit opt-in (enforces the `(state, config)` signature).

**Dead-body warning**: LLM modes emit a `UserWarning` at decoration time if the function body is non-trivial (not `...`, `pass`, or a bare return). AST-based check — handles common false positives.

**Scripted `@node` dispatches via `register_scripted`.** At construct-assembly time, `_register_node_scripted` in `decorators.py` builds a shim closure that resolves `FromInput`/`FromConfig`/constant params from `config`, reads upstream values from `input_data` (the dict returned by `factory._extract_input`), and calls the user function with positional args. The shim is registered via `register_scripted` under a synthesized name, and `node.scripted_fn` points to it. The factory's `_make_scripted_wrapper` picks it up — **one dispatch path for all scripted nodes**, no more `raw_fn` workaround.

`Node.fan_out_param` tells `_extract_input` which `inputs` key should read from `state["neo_each_item"]` instead of from a named upstream field. This is the only IR-level concession to the `@node` layer — it applies equally to programmatic `Each` nodes with dict-form inputs.

---

## Git workflow

- **`main`** — stable. Only tagged releases and critical hotfix PRs. Currently at v0.1.0.
- **`develop`** — active development. All new work lands here. Currently at 0.2.0.dev0. Piarch and other downstream consumers pull from this branch via `uv add "neograph @ git+https://github.com/KonstantinMirin/neograph.git@develop"`.
- **Release path**: when `develop` is ready, merge to `main`, tag `vX.Y.Z`, push the tag. `.github/workflows/publish.yml` triggers on `v*` tags and publishes to PyPI via Trusted Publishing (no tokens, OIDC-scoped).
- **Version bumps**: on `develop` we use PEP 440 pre-release markers (`0.2.0.dev0`, `0.2.0.dev1`, ...). On `main` at the release tag we bump to the final version (`0.2.0`).

**Never publish directly.** The GitHub Actions workflow is the only publish path. This gives us a pypi.org Trusted Publisher gate + an optional manual-approval environment reviewer.

---

## Test conventions

- **One monolithic test file**: `tests/test_e2e_piarch_ready.py`. Currently ~4900 lines, 217 tests. Do not split it — the existing convention is one file, many `Test*` classes, each scoped to a feature.
- **Test class naming**: `TestFeatureName` at the top level. New tests from a feature go in a dedicated class appended at the end of the file.
- **Throwaway modules for `construct_from_module` tests**: use `types.ModuleType("test_xyz_mod")` and attach `@node` functions as attributes. Don't pollute real modules. Pattern is `TestNodeDecorator._fresh_module` (around line 2666).
- **Fakes live in `tests/fakes.py`**: `FakeTool`, `StructuredFake`, `TextFake`, `ReActFake`, `configure_fake_llm`. Don't invent new fakes unless the existing ones genuinely don't cover the case.
- **TDD the user explicitly expects**: for bug fixes, write the failing repro first, verify it fails, then fix. The user has asked for this multiple times — honor it on every bug-fix task.

---

## Examples

13 runnable examples in `examples/`, each narrated as a walkthrough on neograph.pro. Most use `@node` except three that stay declarative (example 05 sub-constructs, example 10 mixed, example 11 config injection).

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

- **Blunt, direct answers preferred over agreement.** If an API has a DX problem, say so. The user will happily refactor at 0.1.0/0.2.0.
- **No backwards-compat shims at 0.x.** Breaking changes are fine; deprecation cycles are unnecessary at this scale and one known user.
- **TDD for bug fixes, always.** Write the failing test first.
- **Parallel agent teams for multi-file work.** The `/team` slash command invokes a team with scoped file regions. Use it for anything that can be parallelized without file conflicts.
- **User is the sole maintainer and sole downstream user (piarch).** No migration burden for hypothetical users.
