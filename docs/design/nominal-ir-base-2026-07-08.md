# Nominal IR base: making duck-typed IR invariants type-representable

**Date**: 2026-07-08
**Status**: Design (drives implementation molecule neograph-i657)
**Ticket**: neograph-i657 (PAT-01 root-cause close from review `.claude/code-review/080726_2107/synthesis.md`)
**Companion docs**: `three-layer-principle-2026-07-03.md` (layer discipline; the compiler sum-type exception)
**Author role**: independent architect (design phase; read-only on `src/`)

---

## 0. The mandate, in one line

`Node` / `Construct` / `_BranchNode` share **no nominal base**. Every
single-source-of-truth (SSOT) invariant over them — "the declared-output
selector has one home", "don't hand-roll Node-vs-Construct discrimination",
"walks must descend into branch arms" — is a *convention* policed by an
**AST/regex guard**. Guards match a *syntactic form*, so the semantic
equivalent in a different spelling slips past. The review confirmed **four live
bypasses** (two already re-patched by neograph-awor). The mandate: give the IR
a nominal base so the wrong code becomes **unrepresentable or
type-checker-caught**, not test-caught — and retire the guards that thereby
become redundant.

**The verdict up front** (§6): **ship this as the flagship 0.7 structural
item, NOT gated into 0.6.0.** The evidence — including a decisive negative
probe — is that a nominal base is genuinely valuable but does **not** by itself
close PAT-01 (mypy's own `hasattr`-narrowing re-opens the exact bypass), so a
small residual guard survives regardless. The change is a maximum-blast-radius
IR-core edit whose safety net is the fixture/6-cell/e2e suite; doing it under a
release deadline is precisely how the four bypasses were born. It lands cleanly
as **Wave F, after Waves D/E merge**.

---

## 1. What the code actually is today (grounded)

### 1.1 The three sibling types

| Type | File | Base classes | Pydantic? | frozen? | Declared-output field |
|---|---|---|---|---|---|
| `Node` | `node.py:152` | `Modifiable, BaseModel` | yes | **no** | `.outputs` (plural, `TypeSpec`) |
| `Construct` | `construct.py:88` | `Modifiable, BaseModel` | yes | **no** | `.output` (singular, `type[BaseModel] \| None`) |
| `_BranchNode` | `_ir_branch.py:55` | `Modifiable` (plain) | **no** | n/a | `.output` (attr set to `None` in `__init__`) |

Verified: neither `Node` nor `Construct` is `frozen` (grep for `frozen` in
both files returns nothing on the class). The sidecars (`_sidecar`,
`_param_res`, `_scripted_shim`) are `PrivateAttr(default=None)` on `Node` only.

### 1.2 The structural-typing layer that already exists

`_ir_protocols.py` already names two Protocols:

- **`ConstructItem`** (`runtime_checkable`): `name: str`, `modifier_set:
  ModifierSet`. Used as the element type of `Construct.nodes`. This is the
  closest thing to a nominal base today — but it is a **Protocol**
  (structural), so it carries no *behavior* and cannot host the
  declared-output method. It is `runtime_checkable`, which (§5.2) is
  structurally weak.
- **`ConstructLike`** (not runtime_checkable): the richer container shape
  (`name`/`input`/`output`/`nodes`) the recursive validator narrows to via the
  `_is_construct_like` **TypeGuard** (`_validation_types.py:62`).

`Modifiable` (`modifiers.py:191`) is a **plain mixin** (not a Protocol, not an
ABC) that all three inherit, carrying `has_modifier`/`get_modifier`/`__or__`/`map`
and declaring `name: str` + `modifier_set: ModifierSet` as bare annotations.

**The key gap**: there is a *behavioral* mixin (`Modifiable`) and a *structural*
Protocol (`ConstructItem`), but **neither carries the declared-output
polymorphic method**. `_declared_output` lives as a free function in
`_normalize.py:128` that does `item.outputs if isinstance(item, Node) else
getattr(item, "output", None)` — the discrimination the whole ticket is about.

### 1.3 The two live bypasses the base targets

Both are Node-vs-Construct discrimination hand-rolled via `hasattr`, invisible
to the getattr-only matcher (both now *are* caught by the `_hasattr_output_sites`
scanner added in neograph-awor — but as an AST guard, not a type constraint):

- **`state.py:391`** (`compute_node_fingerprints._fingerprint_item`):
  ```python
  if hasattr(item, "outputs") and item.outputs is not None:   # Node
      ...
  elif hasattr(item, "nodes"):                                 # Construct
      if hasattr(item, "output") and item.output is not None:
  ```
- **`modifiers.py:245`** (`Modifiable.__or__`, Loop validation):
  ```python
  if hasattr(result, "outputs"):                               # Node
      validate_loop_self_edge(result)
  elif hasattr(result, "output") and hasattr(result, "input"): # Construct
      validate_loop_construct(result)
  ```

### 1.4 The walker landscape (the migration surface)

The dominant idiom, repeated across ~10 modules, is:
```python
for item in iter_with_arms(construct):     # or construct.nodes / iter_nodes
    if isinstance(item, Construct): ... recurse ...
    if not isinstance(item, Node): return   # skip _BranchNode
    ... item.inputs / item.outputs / item.mode ...
```

Walkers inventoried (grep-verified):

| Module | Site(s) | Reads off the item | Base helps? |
|---|---|---|---|
| `_construct_validation.py` | `_validate_node_chain` (148,152,199,206,219,251,307) | `.inputs`/`.input` (narrowed), `_declared_output(item)` | declared-output → method; inputs/input narrow stays |
| `_validation_types.py` | `effective_producer_type` (78,98), `_is_construct_like` (62) | `_declared_output(item)`; `.input`/`.nodes` | declared-output → method |
| `_ir_normalize.py` | `_peer_field_names` (84,88), `normalize_ir` (239) | `isinstance(item, Node)` + `.outputs` | narrowed access; stays |
| `state.py` | `_fingerprint_item` (391) **[bypass]**, `compile_state_model` (129-131,158,282,412) | `hasattr(.outputs/.output/.nodes)` | **bypass → method (partial, §5.2)** |
| `modifiers.py` | `__or__` (245) **[bypass]**, `classify_modifiers` (95-109) | `hasattr(.outputs/.output/.input)`; `getattr(.modifier_set)` | **bypass → method (partial, §5.2)** |
| `runner.py` | `_collect_edges` (325,330), `_has_agent_node` (72) | `getattr(.inputs)`, `getattr(.name)`, `isinstance Node` | `.inputs` narrow; declared none |
| `verify.py` | `_walk` (97,112,187,190) | `isinstance Construct/Node` | narrowed; stays |
| `lint.py` | `_walk` (223,225,330,350,547) | `isinstance Construct/Node`, `.mode`, `.tools` | narrowed; stays |
| `compiler.py` | 3-way dispatch (243,254,266) | `isinstance _BranchNode/Construct/Node` | **sanctioned sum-type — STAYS** |
| `_fan_agent.py` | (153,166) | `isinstance Node` + `.mode` | narrowed; stays |
| `tool.py` | (139) | `isinstance Node` + `.tools` | narrowed; stays |
| `construct.py` | `iter_nodes` (74-85), `__init__` inheritance (172-183) | `isinstance`, `hasattr(.llm_config/.renderer)` | walk stays; hasattr on llm_config/renderer is a *different* concern (see §2.4) |

**Conclusion from the inventory**: the base collapses exactly the
**declared-output discrimination** into a polymorphic method. Everything else is
either (a) a *legitimate* concrete-type narrowing to reach a genuinely
type-specific field (`Node.inputs` plural vs `Construct.input` singular — these
are different fields, an irreducible discrimination like the compiler's), or
(b) module/import/layer concerns the base does not touch.

---

## 2. SHAPE — decided: ABC mixin carrying the polymorphic surface

### 2.1 The decision

Introduce a new **abstract base class**, `IRItem(abc.ABC)`, in a neutral
low-level module (`_ir_item.py`, sibling of `_ir_branch.py` / `_normalize.py`),
that:

- declares `name: str` and `modifier_set` as the shared shape,
- carries `declared_output()` as an **`@abc.abstractmethod`**,
- is inherited by all three: `class Node(Modifiable, IRItem, BaseModel)`,
  `class Construct(Modifiable, IRItem, BaseModel)`, `class _BranchNode(Modifiable,
  IRItem)`.

`Modifiable` folds INTO `IRItem` OR stays a peer mixin — **decided: keep
`Modifiable` as a separate concrete mixin and add `IRItem` as a peer abstract
base.** Rationale: `Modifiable` carries `__or__`/`map` (pipe composition
behavior) that is orthogonal to "is an IR tree item"; `_BranchNode` needs
`Modifiable` too but its `__or__` is never exercised. Merging them couples two
concerns. The MRO `(Node, Modifiable, IRItem, ABC, BaseModel, object)` is clean
(probed).

### 2.2 Why ABC-mixin, not Protocol, not plain base — with evidence

Three shapes were probed empirically (`scratchpad/probe_base.py`,
`probe_mypy.py`, `probe_bypass2.py`; outputs in §5).

| Criterion | ABC mixin | `Protocol(runtime_checkable)` | plain shared base |
|---|---|---|---|
| Forces every subtype to implement `declared_output` | **YES** (mypy: "Cannot instantiate abstract class … with abstract attribute"; runtime: `TypeError` at instantiation) | no (structural — a class that *happens* to have the attr passes) | **no** (inherits a default; `BadNode` silently uses it) |
| `isinstance` is sound | yes (nominal) | **false-accepts** a non-method attr: `declared_output = 42` passes `isinstance` | yes |
| Hosts behavior (the method) | yes | no (Protocols are shape-only) | yes |
| `_BranchNode` (non-Pydantic) can inherit without becoming a model | **YES** (probed: `is BaseModel? False`, `isinstance IRItem True`) | yes | no (would force it to be a BaseModel) |
| Metaclass conflict with `ModelMetaclass` | **none** — `ModelMetaclass` *is already* a subclass of `ABCMeta` (probed) | n/a | n/a |
| Instantiation overhead | **+1.1%** (1.015 vs 1.004 µs/inst, 50k iters) | ~0 | ~0 |

The plain shared base fails the core mandate: it lets a subtype *silently*
inherit a wrong default (`BadNodeB.declared_output()` returned `None` with no
error). The Protocol fails soundness (false-accepts `declared_output = 42`) —
which is *the exact "syntactic form vs semantic equivalent" failure mode*
PAT-01 is about, reintroduced at the type layer. The ABC mixin is the only
shape that makes "an IR item that forgets its declared-output rule"
**unrepresentable** (a `TypeError`/mypy error).

### 2.3 Pydantic base-insertion: no disturbance (probed)

`scratchpad/probe_base.py` + `probe_branch.py` proved, against the **real**
neograph `Node`:

- `_sidecar` / `_param_res` / `_scripted_shim` all **survive `model_copy(update=)`**
  through a class carrying the ABC base (`__pydantic_private__` copies as
  before).
- `model_fields` keys / ordering **unchanged**.
- Real `Node`: `| Oracle(...)` pipe preserves `.outputs`; `model_dump()` (18
  keys) and `model_validate({...})` roundtrip intact — spec-loader
  serialization path unaffected.
- `_BranchNode` inherits `IRItem` and stays a **plain (non-BaseModel) class**;
  abstract enforcement fires (`BadBranch` → `TypeError`).

### 2.4 What does NOT go on the base

- **`Node.inputs` (plural) vs `Construct.input` (singular)** stay as
  concrete-type fields; walkers that need them narrow via `isinstance`. This is
  a genuine sum-type discrimination (different fields, different roles per
  AGENTS.md), NOT a spelling of the declared-output selector. Do not force a
  `declared_input()` onto the base — it would fabricate a false symmetry
  (`_BranchNode` has no input; `Node.inputs` is a fan-in map while
  `Construct.input` is a boundary port).
- **`.llm_config` / `.renderer` inheritance** (`construct.py:175,177`): these
  `hasattr` checks are propagation-of-defaults concerns over BaseModel
  children, not IR-type discrimination. They stay (and are a different guard
  family, unrelated to PAT-01).
- **`.nodes`** stays on `Construct`/`ConstructLike` only.

---

## 3. THE POLYMORPHIC SURFACE — the base's exact contents

Decided surface for `IRItem`:

| Member | Kind | Node | Construct | _BranchNode | Rationale |
|---|---|---|---|---|---|
| `name: str` | attr (declared) | field | field | `__init__`-set | already shared via Modifiable/ConstructItem; declare here as the canonical home |
| `declared_output(self) -> TypeSpecStatic` | **`@abstractmethod`** | `return normalize_outputs(self.outputs).primary` for single/None; but see note | `return self.output` | `return None` | **the load-bearing member** — subsumes `_declared_output`. Abstract forces each subtype to state its rule. |
| `modifier_set: ModifierSet` | attr (declared) | field | field | `__init__`-set | shared; already on `ConstructItem` |

**Critical semantic note on `declared_output()`**: the current free function
`_declared_output` returns the **raw** field (`Node.outputs` *whole* — which may
be a dict, single type, or None — and `Construct.output`). Its callers
(`effective_producer_type`, `_validate_node_chain`) then apply
`normalize_outputs`/dict-handling themselves. To preserve behavior EXACTLY, the
method must return the **same raw value** the free function does, not a
pre-normalized primary. So:

```python
class Node(Modifiable, IRItem, BaseModel):
    def declared_output(self) -> TypeSpecStatic:
        return self.outputs          # raw — dict | type | None, unchanged

class Construct(Modifiable, IRItem, BaseModel):
    def declared_output(self) -> TypeSpecStatic:
        return self.output

class _BranchNode(Modifiable, IRItem):
    def declared_output(self) -> TypeSpecStatic:
        return self.output           # always None today
```

`_declared_output(item)` becomes a one-line delegating call `item.declared_output()`
OR is deleted and callers call the method directly (decided in §4.4: **delete**,
0.x no-compat).

**Members deliberately NOT added** (evaluated and rejected):
- `declared_input()` — §2.4 (false symmetry).
- a `child_items()` / iteration hook — `iter_nodes`/`iter_with_arms` already
  exist as free functions in the neutral `construct.py`/`_ir_branch.py`; they
  dispatch on `isinstance` to three *different* descent behaviors (leaf-flatten
  vs one-level arm-expand). Making them methods on the base would either force
  `_BranchNode` to know about arm-expansion (it does — it holds the arm lists)
  OR split the walk logic across three classes. **Decided: leave the walk
  helpers as free functions.** They are already monopolized (one home each) and
  their guards are behavioral (do they descend into arms?), which a method
  wouldn't make unrepresentable. This is the honest scope boundary: the base
  fixes *discrimination*, not *traversal*.
- a `modifier_combo()` accessor — `classify_modifiers` + `ModifierSet.combo`
  already monopolize this; `modifier_set` is already shared. No new member
  needed.

**Net surface: ONE abstract method (`declared_output`) + two declared shared
attrs.** Deliberately minimal — the base's value is the *nominal* relationship
and the *one* forced method, not a fat interface.

---

## 4. WHAT RETIRES vs WHAT STAYS

### 4.1 The honest retirement principle

A guard retires **only after** the type surface provably covers its invariant,
demonstrated by a **mypy-failure demo** for the wrong code. Where mypy does NOT
catch the wrong code (the `hasattr`-narrowing hole, §5.2), the guard **stays**
(possibly shrunk). No guard is deleted on faith.

### 4.2 Guard inventory and disposition

Based on first-hand reads of the guard classes (the exhaustive cross-check by a
parallel inventory agent is folded in where it extends this):

| File | Class | Invariant | Disposition | Why |
|---|---|---|---|---|
| `test_guards_helper_monopoly.py` | `TestDeclaredOutputSelectorMonopoly` (`test_no_handrolled_getattr_output_selector`) | no `getattr(_, 'output')` outside `_normalize.py` | **SHRINK → residual keeper** | mypy catches *direct* `.output` on `IRItem`, but NOT `getattr(str-literal)` (§5.2). The getattr-scanner half **stays**. |
| ″ | `TestDeclaredOutputSelectorMonopoly` (`test_no_handrolled_hasattr_output_selector`) | no `hasattr(_, 'output'/'outputs')` discrimination | **STAYS (keeper)** | mypy's `hasattr`-narrowing *legitimizes* `hasattr(item,'outputs'); item.outputs` (§5.2 negative probe). The base does NOT make this unrepresentable. This is the load-bearing honesty of the whole ticket. |
| ″ | `TestIrWalkHelperMonopoly` (`test_declared_output_ternary_appears_once`, whitespace-slip meta) | the `item.outputs if isinstance… else getattr` ternary appears once | **RETIRE** | after the ternary moves *into* `Node.declared_output`/`Construct.declared_output` as `return self.outputs`, the ternary form ceases to exist — nothing to police. The *idiom* is gone, replaced by dispatch. |
| ″ | `TestIrWalkHelperMonopoly` (`test_compiler_collectors_use_iter_nodes`, `iter_nodes` single-definer) | walks go through `iter_nodes` | **STAYS** | traversal monopoly, not type discrimination (§3 scope boundary). |
| `test_guards_ir_compiler.py` | `TestNodeIRTyping` | `Node.inputs`/`.outputs` not bare `Any` | **STAYS** | still valuable — the base doesn't retype the fields; bare-`Any` at the field is a separate defect. |
| ″ | `TestBranchNodeIsNotDuckTyped` | `_BranchNode` has no `has_modifier`/`get_modifier` stubs; "should inherit from a shared base, not duck-type" | **RETIRE (subsumed)** | the base makes this *structural*: `_BranchNode(Modifiable, IRItem)` inherits everything. The guard's stated aspiration becomes the code. Replace with a one-line `isinstance(_BranchNode(), IRItem)` assertion (a positive pin, not a syntactic ban). |
| ″ | `TestIterNodesCoversBranchArms` | `iter_nodes` descends into arms | **STAYS** | behavioral traversal invariant; a method wouldn't make arm-skipping unrepresentable. |
| `test_guards_branch_arm_walks.py` | `TestNoArmBlindNodesWalks` + `TestArmBlindDetectorMetaTests` | raw `construct.nodes` walks route through `iter_with_arms` | **STAYS** | traversal, not discrimination. Untouched. |
| `test_guards_helper_monopoly.py` | `TestNoRawOutputsInFanInWiring` (`_subscript_assigns_raw_outputs`) | no `x[...] = expr.outputs` raw fan-in mis-wire | **STAYS** | about *dict-form primary selection*, not Node/Construct discrimination. |
| `test_guards_assembly.py` | `TestNoSidecarPattern` (3 tests) | no global `_node_sidecar`/`_param_resolutions` dict, no `weakref.finalize` | **STAYS** | module-level storage discipline; base is irrelevant. |
| `test_guards_assembly.py`, `_function_local_imports.py`, `_sidecar_imports.py`, `_three_layer.py` | import-direction / layer / file-split / function-local-budget guards | module topology | **STAYS (all)** | about *modules*, not *types* — the base changes nothing here. |
| `test_guards_any_audit.py` | no-`Any` in public IR APIs, arbitrary-types justification | typing hygiene | **STAYS** (may gain: `IRItem` is a new public-ish name to audit) | |

### 4.3 The honest count

- **RETIRE (fully): 2** guard *tests* — `TestBranchNodeIsNotDuckTyped` (→ one
  positive isinstance pin) and `test_declared_output_ternary_appears_once`
  (+ its whitespace-slip meta). Both become *structural facts*.
- **SHRINK: 1** — `TestDeclaredOutputSelectorMonopoly` loses its rationale for
  *direct-access* forms but keeps the `getattr`-string scanner.
- **KEEPER (must stay despite the base): 1** — the `hasattr`-output scanner,
  because mypy narrowing defeats the type here.
- **STAYS (untouched): the rest** (~13+ guard classes, all traversal /
  module / typing-hygiene).

**This is materially fewer than the review's ~10-guards-evaporate estimate.**
The review over-counted because it assumed a nominal base makes *all*
Node-vs-Construct discrimination unrepresentable. The negative probe (§5.2)
shows mypy's `hasattr`-narrowing keeps the most important bypass spelling
*representable and type-clean* — so its guard is permanent. **This is the single
most important correction this design makes to the review's framing, and it is
the load-bearing input to the 0.6-vs-0.7 verdict.**

---

## 5. RISK REGISTER — executed probes with outputs

All probes run with the project venv (`uv run python`) against real neograph
classes where possible. Scripts persisted in the session scratchpad.

### 5.1 Pydantic private-attr / model_copy / serialization survival — PASS

`probe_base.py` (Shape A) + `probe_branch.py`:
```
sidecar survives model_copy: True    param_res survives: True    scripted_shim survives: True
model_fields keys: ['name', 'outputs']            (ordering unchanged)
real Node: pipe works, outputs preserved: True
model_dump keys count: 18            model_validate roundtrip name: y
_BranchNode: is BaseModel? False     isinstance IRItem: True     (abstract enforced)
```
**Risk: none.** Sidecars, pipe, and the loader path are unaffected.

### 5.2 The decisive negative: mypy does NOT catch the `hasattr` bypass — PARTIAL

`probe_mypy.py` (direct access + missing override):
```
error: Cannot instantiate abstract class "ForgotOverride" with abstract attribute "declared_output"  [abstract]
error: "IRItem" has no attribute "output"  [attr-defined]
```
→ mypy **catches** a forgotten override and a **direct** `item.output` on an
`IRItem`-typed var. Good.

`probe_bypass2.py` (the actual bypass spelling):
```
probe_bypass2.py:11: error: "IRItem" has no attribute "outputs"   # direct access: CAUGHT
# hasattr-guarded access at line ~16: NO ERROR
```
→ mypy **narrows through `hasattr`**, so `if hasattr(item,"outputs"): item.outputs`
type-checks clean. And `getattr(item,"outputs",None)` with a string literal is
invisible to mypy (`bad_walker` produced 0 errors).

**Consequence, spelled out**: the nominal base makes the *naive* rewrite of the
two live bypasses (`item.outputs` directly) a mypy error — a real win — but the
*`hasattr`/`getattr`-string* spellings remain type-clean. neograph's mypy config
(`pyproject.toml:126`) is permissive with no flag to disable `hasattr`
narrowing. **Therefore a residual AST guard (the `hasattr`/`getattr`-output
scanner) is permanent.** The base reduces the guard's *surface area and
importance* (the common accidental form is now caught by the compiler) but does
not eliminate it. Honesty demands stating this in the release notes: "we made
the wrong thing *harder* to express and *usually* caught by the type checker —
not categorically impossible."

### 5.3 Instantiation overhead — negligible

`probe_base` perf run: **+1.1%** (1.015 µs vs 1.004 µs/inst over 50k). Node
instantiation is not on any hot path (it happens at assembly, once per node).
**Risk: none.**

### 5.4 Metaclass conflict — none

`ModelMetaclass is subclass of ABCMeta: True` (probed). Adding `abc.ABC` to a
`BaseModel`'s bases introduces no metaclass clash because pydantic's metaclass
already derives from `ABCMeta`. **Risk: none.**

### 5.5 isinstance narrowing on the sanctioned compiler match — unaffected

The compiler's 3-way `isinstance(_BranchNode/Construct/Node)`
(`compiler.py:243-267`) dispatches to three different builders. Making all three
subtypes of `IRItem` does **not** change these `isinstance` checks (they narrow
to the *concrete* type, and `assert isinstance(item, Node)` still narrows the
`ConstructItem` protocol to `Node`). Probed in `probe_surface.py`: a walker
typed over `Sequence[IRItem]` calls `declared_output()` polymorphically AND
narrows to `NodeM`/`ConstructM` for concrete-only fields, both clean. **Risk:
none — the sum-type exception is orthogonal to the base.**

### 5.6 Public-API surface — additive, non-breaking

`Node`/`Construct` are exported (`__init__.py:31,73`). Adding a base class is
**additive** to their MRO: existing `isinstance(x, Node)` in consumer code is
unaffected; existing attribute access is unaffected; `Node(...)` construction is
unaffected (the abstractmethod IS implemented, so no `TypeError`). `IRItem`
itself need NOT be exported (internal), keeping the public surface identical.
**Risk: none for the sole downstream consumer (piarch).** No 0.x compat concern.

### 5.7 The three-surface parity risk — the real test-design cost

`declared_output` and the base must behave identically whether the item was
built via `@node`, declarative `Node(...)`, or programmatic `Node() | Modifier`.
Since all three converge on the same `Node`/`Construct` instances (per AGENTS.md
and `construct.py:190` `normalize_ir`), the method is defined once on the class
and cannot diverge by surface. **But** the parity RULE still requires the
regression tests to exercise `declared_output()` and the migrated
`state.py`/`modifiers.py` sites through all three surfaces (the `neograph-ts7`
lesson). This is a test-authoring obligation, not a design risk.

### 5.8 Latent-behavior-change risk — the gating input to the verdict

The migration rewrites the two `hasattr` bypasses AND (optionally) routes
`_declared_output` callers to the method. Every rewrite is *supposed* to be
behavior-preserving. What catches a slip?

- **82 compile fixtures** (`tests/check_fixtures/`) — exercise the validator
  (which consumes `_declared_output` via `effective_producer_type`) end-to-end.
  A declared-output regression surfaces as a fixture that stops raising / starts
  raising.
- **The 6-cell parity** (3 surfaces × {validate, compile}) around dict-form
  outputs.
- **e2e/checkpoint** — `state.py:391` feeds `compute_node_fingerprints`; a
  fingerprint regression would break auto-rewind. **Caveat (from the review's
  TQ-01)**: auto-rewind has *no e2e execution-count test* today, so a
  fingerprint change is proven only by three disjoint units. This *raises* the
  escape probability for the `state.py` site specifically. **Mitigation: land
  the TQ-01 e2e (a separate ticket) BEFORE or WITH the `state.py` migration**,
  or accept that the `state.py` rewrite is proven by unit + fingerprint-equality
  assertion only.

---

## 6. THE VERDICT — 0.7, not 0.6-gating

### 6.1 Decision criteria, answered with evidence

**(a) Implementation size (days).** Honest estimate:
- Base introduction (additive, nothing breaks): **~0.5 day** — new `_ir_item.py`,
  three `declared_output` methods, three base-list edits, an `import` fix.
- Migrate the two `hasattr` bypasses to `declared_output()`: **~0.5 day** +
  three-surface tests.
- Route/delete `_declared_output` callers (`effective_producer_type`,
  `_validate_node_chain`): **~0.5 day** — mechanical, fixture-covered.
- Guard retirement (2 retire, 1 shrink) with per-guard mypy-failure demos:
  **~0.5 day**.
- AGENTS.md monopoly-section rewrite + this doc's decisions: **~0.5 day**.
- **Total: ~2.5 days**, assuming the TQ-01 e2e is a *separate* prerequisite
  ticket (add ~1 day if bundled).

**(b) Probability a latent behavioral change escapes the net.**
*Moderate for the `state.py`/fingerprint site* (no e2e execution-count test —
review TQ-01), *low elsewhere* (82 fixtures + 6-cell catch validator/compile
regressions). The escape probability is dominated by the one feature
(auto-rewind) whose seam is under-tested — the same gap the review flagged as
"the novel gem has no e2e test." Gating 0.6 on this change means gating a
release on a rewrite of the auto-rewind fingerprint path *before* it has an e2e
test. **That is the exact risk profile that produced the four bypasses under
deadline.**

**(c) Cost of shipping 0.6 WITHOUT it.** Four patched bypasses
(neograph-awor) + the guard fat. But the guard fat is *smaller than the review
implied* (§4.3): the permanent residual is essentially **one keeper guard** (the
`hasattr`/`getattr`-output scanner) that the base **cannot** retire anyway
(§5.2). So "shipping 0.6 without the base" does NOT leave a large removable-debt
overhang — it leaves ~2 retireable guards deferred and one permanent guard that
would remain either way. The "clean structural story in the release notes" is
real but is a **0.7 headline** ("the IR now has a nominal base; wrong-type
discrimination is a compile error"), not a 0.6 bug-fix.

### 6.2 The verdict

**Ship as the flagship 0.7 structural item. Do NOT gate 0.6.0.** Land it as
**Wave F, after Waves D and E merge** (Wave D is actively touching
`state.py`/`modifiers.py`/`_dispatch.py`/`_oracle.py`/`runner.py`/`_agent_cycle.py`
— the base migration edits `state.py:391` and `modifiers.py:245`, a direct
collision). Rebase Wave F onto post-D `state.py`/`modifiers.py`.

The maintainer *may* still pull it into 0.6 — the change is small and the probes
are green — but the evidence says the value proposition is a **clean structural
release-note story**, not an urgent debt-paydown, because (i) the bypasses are
already patched, (ii) the permanent guard survives regardless, and (iii) the
one site with elevated escape risk (auto-rewind fingerprints) wants its e2e test
first. A 0.7 landing gets all three right; a 0.6 gate re-runs the deadline
gamble for a mostly-cosmetic timing win.

---

## 7. MIGRATION PLAN (ordered, sized, collision-aware)

**Precondition**: Waves D and E merged. Rebase onto their `state.py` /
`modifiers.py`.

1. **[additive, ~0.5d] Introduce `IRItem`.** New `src/neograph/_ir_item.py`:
   `class IRItem(abc.ABC)` with `name`, `modifier_set` declarations and
   `@abstractmethod declared_output(self) -> TypeSpecStatic`. Add to the three
   bases. Implement `declared_output` returning the raw field on each (Node
   `self.outputs`, Construct `self.output`, `_BranchNode` `self.output`).
   *Nothing breaks* — all three already have the field; the method just names
   the rule. Guard-first: add a positive pin `isinstance(_BranchNode(...),
   IRItem)` before migrating.
2. **[~0.5d] Migrate the two bypasses.** Rewrite `state.py:391` and
   `modifiers.py:245` to type their loop/param var as `IRItem` and call
   `item.declared_output()` where they currently `hasattr`-discriminate. For
   `modifiers.py.__or__`, the Node-vs-Construct split still needs `isinstance`
   (it dispatches to `validate_loop_self_edge` vs `validate_loop_construct` —
   genuinely different functions, like the compiler) — so this becomes
   `isinstance(result, Node)` / `isinstance(result, Construct)`, NOT `hasattr`.
   For `state.py`, `_fingerprint_item` uses `declared_output()` + the existing
   `normalize_outputs`. Three-surface tests for each.
3. **[~0.5d] Route/delete `_declared_output`.** Replace the free function's body
   with `return item.declared_output()` OR delete it and update the ~3 callers
   (`effective_producer_type`, `_validate_node_chain`) to call the method.
   **Decided: delete** (0.x no-compat; it is internal). Fixtures cover this.
4. **[~0.5d] Guard retirement, each with a mypy-failure demo.**
   - Delete `TestBranchNodeIsNotDuckTyped`; replace with a one-line positive
     `isinstance` pin. Demo: `class BadBranch(Modifiable): ...` without `IRItem`
     → show it is not an `IRItem`.
   - Delete `test_declared_output_ternary_appears_once` + its whitespace-slip
     meta (the ternary no longer exists).
   - Shrink `TestDeclaredOutputSelectorMonopoly`: keep the `getattr`-string and
     `hasattr` scanners (§5.2 keeper); drop any assertion that assumed direct
     access is the only form. Add a comment citing the mypy `hasattr`-narrowing
     hole as the *reason* the scanner is permanent.
5. **[~0.5d] Docs.** Rewrite the AGENTS.md `_declared_output` monopoly section
   ("the declared-output selector is a polymorphic method on `IRItem`, not a
   free function; the residual `hasattr`/`getattr` guard exists because mypy
   narrows `hasattr`"). Update the compiler-sum-type-exception note to mention
   the base. Persist this doc.
6. **Independent architect review gate** (mandatory per ticket).

**Collision list**: Wave D owns `state.py` + `modifiers.py` NOW → step 2 must
rebase onto D. No other collisions (`_ir_item.py` is new; `_normalize.py`,
`_construct_validation.py`, `_validation_types.py` are not in D's set).

---

## 8. Open decisions handed to the maintainer

1. **Bundle the TQ-01 auto-rewind e2e into this molecule, or keep separate?**
   Recommendation: **separate prerequisite**, but sequence it before step 2's
   `state.py` migration so the fingerprint rewrite has an execution-count safety
   net (§5.8).
2. **Delete `_declared_output` vs keep as a one-line shim?** Recommendation:
   **delete** (§4.4, 0.x no-compat, internal name). If the maintainer prefers a
   smaller diff, a delegating shim is acceptable and costs nothing.
3. **Export `IRItem`?** Recommendation: **no** (keep internal); revisit only if
   piarch needs to write a walker over IR items.
```
