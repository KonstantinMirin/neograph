# Architecture Review Synthesis — 2026-06-02

Three independent Opus reviews of the neograph codebase through three lenses (SOLID/GRASP, Simplicity-without-killing-abstraction, Essential-vs-Incidental complexity) converged on the same small set of root causes. This synthesis combines their verdicts into an actionable roadmap.

Source documents:
- `architecture-review-solid-grasp-2026-06-02.md`
- `architecture-review-simplicity-abstraction-2026-06-02.md`
- `architecture-review-incidental-complexity-2026-06-02.md`

## The convergence — three lenses, three root causes

| Root cause | SOLID/GRASP verdict | Simplicity verdict | Incidental verdict |
|---|---|---|---|
| **R1. Assembly-path divergence** (`_construct_builder` vs `Construct._normalize_*`) | HIGH severity; Information Expert + cohesion violations | UNDER-ABSTRACTED (sibling normalizer methods need `_IrParityPass` Protocol) | **Root 1 — highest leverage**; eliminates ~15 structural guards |
| **R2. Retry/DSML compat ladder** (`_call_structured` 4-branch + scattered DSML recovery) | HIGH severity; SRP+OCP+DIP+LSP violated simultaneously | UNDER-ABSTRACTED (needs `DsmlGuard` class); OVERWROUGHT in scatter | **Root 3 — accreted compat layer** |
| **R3. `__name__` doing double duty** (LangGraph routing key vs user-facing label) | MEDIUM severity; missed polymorphism (Strategy pattern) | OVERWROUGHT (three near-identical factories) | **Root 2 — `node_name` threading would not exist if labels were separated from routing** |

The remarkable signal: every reviewer reached these same three conclusions from a different starting principle. That convergence makes them the highest-confidence interventions.

## The model — what's already excellent

All three reviews independently identified the **StateBus / `normalize_outputs` / test-fakes triad** as the architecturally exemplary pattern the codebase has internalized. The PROBLEMATIC items are uniformly cases where the SAME pattern was not reached for, usually because the immediate fix didn't seem to need it.

**Concretely earning their keep** (consensus across all three reviews — DO NOT TOUCH):

| Abstraction | Why it earns |
|---|---|
| `StateBus` Protocol + `_DictStateBus` + `_ModelStateBus` | Two genuinely substitutable shapes; both have non-trivial logic; polymorphism is real |
| `normalize_outputs` / `normalize_inputs` + `NormalizedOutputs`/`NormalizedInputs` | The single-type/dict-form distinction is a genuine ergonomic vs capability trade-off; the boundary normalization keeps downstream code single-form |
| Five test fakes | Each encodes a distinct LLM contract; collapsing them would substitute branching for structure |
| Three API surfaces (declarative / @node / programmatic) | Serve genuinely different users (humans / runtime construction / config-driven); explicitly NOT to be collapsed |
| `Node._sidecar` PrivateAttr | Pragmatic Pydantic-bypass; alternatives are worse |
| Two-level fingerprinting (schema + per-node) | Schema = ABI invalidation; per-node = locate-divergent-step. Two genuine concerns, two genuine fields |
| `LlmRuntime` + `EMPTY_RUNTIME` Null Object | Frozen dataclass with ≥3 consumers; sentinel saves `is None` branching at every call site |
| Recursive `_validate_node_chain(ambient_producers=...)` | Reviewer 1 (SOLID) and Reviewer 3 (incidental) both endorsed; deferred validation is essential domain logic, recursion is the cleanest shape |
| Retry concern split (transient / output-quality / in-node) | Real failure-model differences |

## What to actually change — the three root-cause epics

### EPIC 1: Unify IR finalization — `neograph._ir_normalize` module

**Eliminates:** `Construct._normalize_fan_out_params` + `_normalize_oracle_gen_type` (and every future `_normalize_*`); `infer_oracle_gen_type` function-local import; up to 15 structural guards; one class of three-surface-parity bugs.

**Pattern:** Extract `neograph._ir_normalize.normalize_ir(construct)`. Every assembly path (declarative `Construct.__init__`, `_build_construct_from_decorated`, YAML `_build_construct`) delegates to it once before validation. Each individual normalizer is a function in `_ir_normalize`; the dispatch is a registry-iteration in one place.

**Why polymorphism here (not branching):** Each IR rule has its own predicate (when does it fire?) and its own inference logic. Each is a function with two responsibilities (`should_apply`, `apply`). A `IrNormalizer` Protocol with N implementations is the GRASP Strategy pattern's textbook case. A switch statement would inline N predicates into one function.

**External references** (from incidental review):
- Hypothesis's `hypothesis.internal.conjecture.utils.calc_label_from_name` (single-site normalization)
- Pydantic v2's validator pipeline (post_init normalization registered, not inlined)
- Mojo's `@parameter` decorator (IR finalization in one normalize pass)

**Replaces these tickets:** vgc1 + aqau (already shipped) refactor into this; `fbk5` (aqau hardening) collapses into this; `xs3e` becomes partially obsolete.

### EPIC 2: Provider-quirk compat shim — `neograph._llm_structured_compat`

**Eliminates:** ~60 lines of branching in `_llm_dispatch.py::_call_structured`; the scattered DSML detection across `_llm_dispatch.py` + `_llm_retry.py` + `_tool_loop.py`; the cognitive load where every new provider quirk requires editing 4 branches.

**Pattern:** New module normalizes `BaseChatModel.with_structured_output(model, include_raw=True)` into a `StructuredResult = Parsed(model) | Raw(text) | Failed(error)` sum-type. `_call_structured` becomes a switch on the result variant. Each provider quirk (DSML recovery, TypeError-on-include_raw, parsed=None+raw) is a decorator applied to the appropriate adapter at construction time, not a try/except branch.

**Why polymorphism here:** Three structurally distinct output strategies (`structured`, `json_mode`, `text`) become three `StructuredOutputAdapter` implementations of a Protocol. The 4-branch ladder is replaced by adapter selection at compile time + a thin `with_retry` loop.

**External references:**
- Pydantic AI's `pydantic_ai._adapters` (multi-provider compat shim — exact pattern)
- LangChain itself does NOT provide this layer, which is why we have the ladder

**Replaces this ticket:** `cx18` (Replace LangChain compat ladder) is exactly this epic.

### EPIC 3: Split LangGraph routing key from user-facing label

**Eliminates:** `node_name=` kwarg threading from all three closure factories in `_oracle.py`; the `_ = node_name` placeholder at `_oracle.py:68`; a class of "did we update all three?" structural guards; the YAGNI defensive kwarg the reviewer flagged.

**Pattern:** Stop overriding `__name__` on wrapper closures (factory.py:121 line). Let wrappers have unique synthesized routing keys (`_oracle_redirect_{field}_{generator_idx}`). Use `node.name` (already stored on the Node IR) wherever the user-facing label is needed. The three redirect factories collapse their `node_name` parameter; each closure looks up the label from the captured `Node` instance via its own state.

**Why polymorphism (not collapse):** The three redirect factories (`make_oracle_redirect_fn`, `make_eachoracle_redirect_fn`, `make_each_redirect_fn`) DO encode three distinct topologies — collapsing them would substitute branching for structure (all three reviews explicitly warned against this). What's eliminated is the THREADING of `node_name` through all three, not the three factories themselves. They become smaller (no `node_name` kwarg) but remain three.

**External references:**
- FastAPI's `APIRoute`: `endpoint.__name__` (internal) vs `route.name` (user-facing) are explicitly separate fields

**Replaces this ticket:** `1u6f` (kg8l hardening) becomes "split routing-key from label" instead of "remove YAGNI kwarg" — the root fix subsumes the local one.

## Disposition of every previously-filed P2/P3 ticket

| Ticket | Original framing | After synthesis |
|---|---|---|
| `cx18` LangChain compat | "Replace ladder with provider-agnostic path" | **= EPIC 2** |
| `ylk9` `bus.get_counter` | Extract helper for `or 0` idiom | Standalone — small, defensible, do it |
| `e8jg` Split oversized test files | Split by concern | Standalone — purely housekeeping |
| `fvlj` PrivateAttr contract test | Pin Pydantic preservation | Standalone — small, defensible |
| `gup9` Ghost state-model fields | Eliminate post-init mutation | **Subsumed by EPIC 1** (assembly-path unification freezes IR before state model is built) |
| `xs3e` ta43 hardening | `ConstructLike` Protocol + ValidationMode + OrderedDict | Three parts: Protocol stays standalone (LOW); ValidationMode subsumed by epic-style work or LOW; OrderedDict is a perf-only change, defer |
| `1u6f` kg8l hardening | Remove YAGNI kwarg + `run()` integration test | **= EPIC 3** (real test stays; YAGNI kwarg disappears for the right reason) |
| `fbk5` aqau hardening | Normalizer dispatch list + DRY + relocation-proof guard | **= EPIC 1** |
| `3zai` Split `_construct_builder.py` | Single Responsibility | **Subsumed by EPIC 1** (most of `_cleanup_inputs_and_register` moves to `_ir_normalize`; remaining logic is small enough to stay in `_construct_builder`) |

**Result:** 9 backlog tickets collapse into 3 epics + 3 standalone small items (`ylk9`, `e8jg`, `fvlj`).

## What the reviews explicitly told us NOT to do

Every review independently produced a "do not" list. The combined list:

- **Do not** replace types with dicts. Loses safety, increases incidental complexity downstream.
- **Do not** collapse polymorphic class hierarchies into branching functions. Moves complexity from structure to logic.
- **Do not** collapse the five test fakes. Each encodes a real LLM contract.
- **Do not** collapse the three API surfaces. They serve genuinely different users.
- **Do not** collapse the three redirect factories in `_oracle.py`. Each encodes a distinct topology.
- **Do not** remove `_validate_node_chain`'s recursive `ambient_producers=` pattern.
- **Do not** simplify `Node.outputs` polymorphism away — the dual form is a genuine ergonomic + capability trade-off.
- **Do not** remove fingerprinting, auto-resume, `LlmRuntime`, `StateBus`, or the sidecar pattern.

## The pattern to apply

All three reviews converged on the same architectural prescription, expressed slightly differently:

> The codebase has ONE excellent pattern: **define a Protocol, register implementations, dispatch via polymorphism, normalize at boundary**. The `StateBus` / `normalize_outputs` / fakes triad embodies it. Every PROBLEMATIC site is a place where this pattern was not reached for.

The three epics apply this pattern at the three sites where it would have the most leverage:
- EPIC 1 applies it to IR finalization (instead of N sibling methods on Construct)
- EPIC 2 applies it to provider quirks (instead of a try/except ladder)
- EPIC 3 applies it to closure identity (instead of threading `node_name` everywhere)

## Execution order

Recommended sequence — each epic is roughly self-contained and unblocks the next:

1. **EPIC 1 (IR finalization)** first — highest leverage; eliminates the most downstream complexity; the pattern it establishes (`_ir_normalize` module + Protocol-based dispatch) is the model for Epics 2 and 3.
2. **EPIC 3 (routing key / label split)** second — smaller scope (~30 LOC net); validates the Protocol-based dispatch pattern in a second domain before doing it at scale.
3. **EPIC 2 (compat shim)** third — largest scope; benefits from the disciplined application of the pattern from Epics 1 and 2; addresses the most accreted layer.

Standalone small items (`ylk9`, `e8jg`, `fvlj`) can run in parallel with any epic — they don't intersect.

## Net effect

If all three epics land:
- ~15 structural guards delete
- ~60 lines of branching in `_llm_dispatch.py` delete
- 9 backlog tickets resolve through 3 epics + 3 small items
- One class of "three-surface IR drift" bugs becomes architecturally impossible
- One class of "did we update all three closure factories?" bugs becomes architecturally impossible
- One class of "another provider quirk = another try/except arm" growth pattern is blocked

The codebase shrinks in incidental complexity while keeping every abstraction that's earning its keep. The hydra is killed not by deleting heads but by removing the asymmetry that grows new ones.
