# `node()` shape-registry design: the kwarg contract derived FROM `ModifierCombo`, not guessed from kwargs

Date: 2026-08-04. Scope: `src/neograph/decorators.py` `node()` + `src/neograph/_node_modifier_kwargs.py`.
Ticket: neograph-jtawq.4 (Theme 4, epic neograph-jtawq). Status: DESIGN — no code changed.

**Direction of derivation (the governing idea).** The closed set of valid node shapes
already exists as data: `ModifierCombo` (`modifiers.py:61-81`) and `_COMBO_MAP`
(`modifiers.py:87-100`, which raw modifier-name sets are valid). The decorator's 32
kwargs must stop being an independent flat namespace that gets pattern-matched after
the fact. Instead: the valid-parameter contract per shape is **declared once, keyed
by that same enum**, and the raw kwargs actually passed are **validated against the
declared contract** — expected parameters derived FROM the known shape registry,
never the shape deduced post-hoc from whichever parameters happened to be set and
the rest silently ignored.

Every claim below was re-derived from the working tree on 2026-08-04. `decorators.py`
is 671 lines today (its exact ceiling in `tests/test_guards_file_size.py:79`), not the
~515/~789 in older ticket text.

---

## 1. Verified current state

### 1.1 The shape of the problem

`node()` has **32 parameters**: `fn` plus 31 keyword-only kwargs
(`decorators.py:127-161`). The inner `decorator(f)` closure spans
`decorators.py:242-637` (~396 lines). The kwargs partition cleanly:

| Group | Kwargs (count) |
|---|---|
| Node identity (pass through to `Node(...)` at :473-488, or mode machinery) | `fn`, `mode`, `inputs`, `outputs`, `model`, `prompt`, `llm_config`, `tools`, `name`, `renderer`, `context`, `skip_when`, `skip_value`, `gate_tools_when` (14) |
| Each sugar | `map_over`, `map_key`, `map_on_error` (3) |
| Oracle sugar | `ensemble_n`, `models`, `merge_fn`, `merge_prompt`, `merge_pre_process`, `merge_post_process`, `merge_fallback`, `merge_model` (8) |
| Operator sugar | `interrupt_when` (1) |
| Loop sugar | `loop_when`, `max_iterations` (2) |
| Portal sugar | `portal`, `route`, `max_hops` (3) |
| Shared Loop/Portal | `on_exhaust` (1) |

The modifier-application chain is five ad hoc branches (`decorators.py:490-618`):

1. `if map_over is not None and has_oracle_kwarg:` — fused Each×Oracle (:496-519)
2. `elif map_over is not None:` — Each only (:522-526)
3. `if has_oracle_kwarg and map_over is None:` — Oracle only (:535-554)
4. `if interrupt_when is not None:` — Operator (:557-573)
5. `if loop_when is not None:` — Loop (:580-597)
6. `if portal is not None:` — Portal (:607-618)

with `has_oracle_kwarg` at :491-493 (`ensemble_n`/`models`/`merge_fn`/`merge_prompt`
presence). Each branch pipes its modifier inline (`n = n | Oracle(...)` etc.) and —
for branches 3-6 — re-runs `_register_sidecar` / `_set_param_res` on the fresh copy.
Five sidecar re-registration sites: :528-530, :549-551, :571-573, :595-597, :616-618.
Diagnosis confirmed: not a tangle — 18 raw kwargs standing in for five concepts, each
detected by its own presence-test, no shared classification, no declared per-shape
contract.

### 1.2 (a) Composability of the branches; Oracle+Loop really reaches both

**Confirmed with a correction.** Branches 4-6 are independent `if`s, so
`interrupt_when` composes with every primary, and `@node(ensemble_n=3, merge_fn=...,
loop_when=...)` executes BOTH the Oracle-piping branch (:548) and the Loop-piping
branch (:594). Nothing at the top of the function prevents it; the broken node is
prevented one layer down, at the second pipe: `Modifiable.__or__` →
`ModifierSet.with_modifier` (`modifiers.py:824-866`), whose `_SLOT_RULES` table
(`modifiers.py:727-754`) has Loop excluding an occupied `oracle` slot →
`ConstructError("Cannot combine Oracle and Loop on the same item")`
(`modifiers.py:701-704`). Loud, not silent.

**Correction to the original brief's hypothesis**: Each and Oracle are NOT mutually
exclusive at the kwarg level — `map_over` + oracle-kwargs is the valid combined path
(branch 1), mirroring `ModifierCombo.EACH_ORACLE`. Modifiers COMBINE; that is the
domain's actual structure, and the kwarg-level shape universe is exactly
`_COMBO_MAP`'s grid: **6 primaries** {BARE, EACH, ORACLE, EACH_ORACLE, LOOP, PORTAL}
× **operator flag** = the 12 `ModifierCombo` values. The enum IS the closed
classification type; nothing new needs inventing — which is precisely what makes the
registry direction (§2) possible.

### 1.3 (b) Existing decoration-time validation, and the two-pair gap

Decoration-time rejections today (`decorators.py:243-295`):

| Pair | Where | Error names the kwargs? |
|---|---|---|
| `map_over` without `map_key` (and inverse) | :244-255 | yes |
| `map_over` + `loop_when` | :256-261 | yes |
| `portal` + `map_over` | :269-274 | yes |
| `portal` + `loop_when` | :275-280 | yes |
| `max_hops` without `portal` | :284-289 | yes |
| `route` without `portal` | :290-295 | yes |

Missing at decoration time relative to `_COMBO_MAP`'s invalid sets: **oracle-kwargs +
`loop_when`** and **oracle-kwargs + `portal`** — both currently fall through to the
pipe machinery's `_SLOT_RULES` raise with a modifier-level message ("Cannot combine
Oracle and Loop"), not a kwarg-level one. Loud but inconsistent: four invalid pairs
get kwarg-named errors, two get modifier-named errors from a different layer.
Verified relevant for the redesign: **every test asserting those pipe-layer messages
goes through the programmatic surface** (`Node.scripted(...) | Oracle | Loop` in
`tests/test_loop.py:378-408`, direct `ModifierSet(...)` in
`tests/modes/test_node_io.py:230-244`, `tests/modifiers/test_modifier_edge_cases.py:330-406`,
`tests/modifiers/test_portal.py:226`) — none through `@node(...)` kwargs. So the
decorator layer can take ownership of these two errors without touching any pinned
assertion (§3).

### 1.4 (c) `on_exhaust` — one kwarg, two owners

`on_exhaust: Literal["error", "last", "exit"] | None` (:157) is consumed by Loop
(valid values `error|last`) and Portal (valid values `error|exit`). It is explicitly
exempted from the "satellite requires trigger" pattern (comment :282-283); each
owning branch rejects the OTHER owner's cross-value with a kwarg-named error (Loop
rejects `'exit'` :581-587; Portal rejects `'last'` :608-614). Because `{loop,
portal}` is not in `_COMBO_MAP` (and :275 rejects the pair with a kwarg-named error
first), at most one owner is ever active — ownership resolves unambiguously. In the
registry (§2.3) this is expressed as the SAME kwarg appearing in two modifier rows; a
shape's valid set contains it iff the shape contains either owner, and value-domain
validation stays in the owning builder.

### 1.5 The s7zt3.10 hazard and what already pins it

The silent-drop bug — both `map_over` branches `return`ed before the `interrupt_when`
tail, so `@node(map_over=..., interrupt_when=...)` built a plain EACH node with the
Operator silently dropped — is fixed (rebind-and-fall-through, incident documented
in-code at `decorators.py:511-518`) and pinned twice: behaviorally by
`tests/decorator/test_fanout_oracle.py::TestMapOverDoesNotDropOperator` (asserts
EACH_OPERATOR / EACH_ORACLE_OPERATOR classification through the decorator), and
structurally by `tests/test_guards_modifier_composition_completeness.py` RULE 1 — a
pure-AST scanner banning any `return` inside a top-level `if` of `decorator(f)` whose
test mentions {`map_over`, `has_oracle_kwarg`, `interrupt_when`, `loop_when`,
`portal`}, with synthetic-source meta-tests. That scanner is welded to the current
five-branch shape; the refactor must evolve it in the same commit or it goes silently
vacuous (§6.2). Whether the redesign makes the bug CLASS unrepresentable — not just
re-guarded — is argued in §5.

### 1.6 Existing builders and the sidecar

`_node_modifier_kwargs.py` holds exactly **three** kwarg builders —
`_build_oracle_kwargs` (:68), `_build_each_kwargs` (:153), `_build_portal_kwargs`
(:163) — plus `_apply_eager_oracle_gen_type` (:52; the ONE sanctioned
decoration-time `oracle_gen_type` write, pinned by `ALLOWED_PREPOP` in
`test_guards_llm_runtime.py`, keyed on that module) and `_is_trivial_body`. There is
no fourth or fifth: **Loop inlines** its kwarg-building (`decorators.py:588-593`) and
**Operator inlines** its condition-name resolution + `register_condition`
(:558-568). The design adds `_build_loop_kwargs` and `_build_operator_kwargs` by
verbatim extraction. Note the builders' conditional-include discipline is
load-bearing: `_build_portal_kwargs`'s docstring (:166-174) records that
`model_fields_set` parity with the programmatic `| Portal(...)` form is read by
`_validation_portal` — the registry must NOT take over defaults (§2.6).

Sidecar: `_register_sidecar` / `_set_param_res` set `Node._sidecar` /
`Node._param_res` (`_sidecar.py:33-42`) — Pydantic `PrivateAttr`s **preserved by
`model_copy`** (AGENTS.md; pinned by `tests/test_node_sidecar_contract.py` across
model_copy/pipe/deepcopy). `Modifiable.__or__` is a `model_copy`
(`modifiers.py:392`) and intermediates are discarded, so the five re-registration
sites are defensive redundancy: **one terminal registration after the last pipe is
observationally identical**. The eager scripted-shim block (:632-635) reads
`_get_sidecar(n)` and must stay after registration — it already sits last.

### 1.7 The silent seams the current lenient namespace permits (dangling satellites)

The "extra kwarg on the wrong shape" behavior is applied **inconsistently** today.
Dangling kwargs that RAISE: `map_key` (:251), `max_hops` (:284), `route` (:290).
Dangling kwargs that are **silently ignored**:

- `max_iterations` without `loop_when` — read only at :590
- `on_exhaust` without `loop_when`/`portal` — read only at :581/:592 and :608/:615
- `map_on_error='collect'` without `map_over` — read only inside the Each branches
- `merge_model`, `merge_pre_process`, `merge_post_process`, `merge_fallback` without
  any `has_oracle_kwarg` trigger — read only inside `_build_oracle_kwargs` calls

`@node(outputs=X, max_iterations=5)` with no `loop_when` decorates cleanly and the
knob vanishes. This is exactly the disease the registry direction cures: there is no
declared per-shape contract for a passed kwarg to be validated against. No test in
`tests/decorator/` (or elsewhere — grep 2026-08-04 for silent-ignore assertions)
pins the lenient behavior, but making these raise IS a behavior change for
previously-accepted buggy inputs — the strictness accounting is in §4.

---

## 2. The design: a per-shape kwarg contract keyed by `ModifierCombo`, derived — not enumerated

### 2.1 The registry is two small tables plus a derivation, not twelve hand-written rows

The naive reading of "a `dict[ModifierCombo, ParamSchema]`" — twelve hand-authored
entries — would itself reintroduce the disease this codebase spent the 2026-07-27
audit removing (docs/design/modifier-combo-single-source-of-truth-2026-07-27.md;
docs/design/agent-spec-target-architecture-2026-08-03.md): the twelve rows'
*composition* (which modifiers each combo contains) would be a second,
decorator-local copy of `_COMBO_MAP`, editable independently and guaranteed to
drift. `EACH_ORACLE_OPERATOR`'s valid kwargs are not new information — they are
exactly Each's ∪ Oracle's ∪ Operator's.

So the registry declares the kwarg contract **per modifier** (5 rows — genuinely new,
decorator-layer information: which of the 32 kwargs express each modifier) and
**derives** the per-shape contract through the IR's own combo-membership reader:

```python
class ModifierKwargs(NamedTuple):
    """The @node kwarg surface of ONE modifier — the decorator-layer rename
    table for its IR fields. The only genuinely new information in the design."""
    name: str                        # _COMBO_MAP modifier name: 'each' | 'oracle' | ...
    triggers: tuple[str, ...]        # presence of any of these MEANS this modifier
    satellites: tuple[str, ...]      # configure it; valid only when it is present
    field_map: dict[str, str]        # kwarg -> modifier-class field (documentation + guardable)

MODIFIER_KWARGS: tuple[ModifierKwargs, ...] = (
    ModifierKwargs("oracle", ("ensemble_n", "models", "merge_fn", "merge_prompt"),
                   ("merge_pre_process", "merge_post_process", "merge_fallback", "merge_model"),
                   {"ensemble_n": "n", "models": "models", "merge_fn": "merge_fn", ...}),
    ModifierKwargs("each",     ("map_over",),       ("map_key", "map_on_error"),
                   {"map_over": "over", "map_key": "key", "map_on_error": "on_error"}),
    ModifierKwargs("operator", ("interrupt_when",), (), {"interrupt_when": "when"}),
    ModifierKwargs("loop",     ("loop_when",),      ("max_iterations", "on_exhaust"),
                   {"loop_when": "when", "max_iterations": "max_iterations", "on_exhaust": "on_exhaust"}),
    ModifierKwargs("portal",   ("portal",),         ("route", "max_hops", "on_exhaust"),
                   {"portal": "to", "route": "route", "max_hops": "max_hops", "on_exhaust": "on_exhaust"}),
)

IDENTITY_KWARGS: frozenset[str] = frozenset({...the 14 pass-through names, §1.1...})

def valid_kwargs(combo: ModifierCombo) -> frozenset[str]:
    """The declared valid @node kwarg set for a shape. Total over ModifierCombo
    because modifier_names_for_combo is total; composition read from _COMBO_MAP
    via its sanctioned inverse — never re-stated here."""
    names = modifier_names_for_combo(combo)                # modifiers.py:254
    rows = [r for r in MODIFIER_KWARGS if r.name in names]
    return IDENTITY_KWARGS | frozenset(k for r in rows for k in (*r.triggers, *r.satellites))
```

This satisfies the "declared ONCE, keyed by the SAME enum" requirement in the only
way that doesn't duplicate the enum's structure: `ModifierCombo` is the key of the
derived total function `valid_kwargs`; the per-modifier rows are the single new
declaration; combo composition stays exclusively `_COMBO_MAP`'s. Within-shape
constraints that are *value-level* (merge_fn XOR merge_prompt, `ensemble_n >= 2`,
`on_exhaust` value domains, `map_over` requires `map_key`) remain in the builders and
the modifier classes' own `model_post_init` — see §2.6 for why the registry must not
absorb them.

Note `on_exhaust` appears in two rows — the §1.4 sharing expressed as data:
`valid_kwargs(LOOP*)` and `valid_kwargs(PORTAL*)` both contain it,
`valid_kwargs(ORACLE)` does not, and since `{loop, portal}` has no combo, no shape
ever has two owners for it.

### 2.2 Construction flow: derive the shape, validate against it, then five independent, plainly-named applications

The flow is three named steps — derive, validate, build — and the build step
mirrors the domain's actual structure: **modifiers compose** (that is what
`EACH_ORACLE` is), so applying them is five independent membership checks, not a
selection among mutually exclusive cases.

**Why not a `match` on primary shape — the compiler precedent examined and
rejected for this layer.** `compiler.py:543-619` dispatches
`match COMBO_DECOMPOSITION[combo].primary:` with an `is_each_oracle_fused` split
before it, and `_agent_spec.py:208-320` does the same for export lowering. That
pattern is RIGHT there and WRONG here, because the two layers answer different
questions. The compiler chooses **which graph-topology algorithm** builds the node —
Oracle's fan-out+merge vs Loop's conditional back-edge vs the fused "flat M x N Send
topology" (`_add_each_oracle_fused`, compiler.py:548-560, its own comment naming it
a distinct topology). A node is built by exactly ONE topology algorithm, so mutual
exclusivity is true at that layer and a match (plus one genuine fusion algorithm) is
the honest encoding. The decorator does something different in kind: it **pipes
composable modifier objects** onto a Node — and at this layer the "fused" combo is
nothing special, it is literally two ordinary pipes today
(`n = n | Oracle(...)` then `n = n | Each(...)`, `decorators.py:509-510`). Forcing a
mutually-exclusive match onto a compositional domain would require carving
`EACH_ORACLE` out as a special pre-match case — and every future combinable pair
(say, a new modifier that `_COMBO_MAP` permits alongside Loop) would need ANOTHER
hand-written carve-out, the same failure shape as the original 32-kwarg branches.

The build step is therefore five unconditional membership checks against the same
`members` value the validation step already used — unrolled rather than looped, so
each line reads as "if this shape includes X, apply the X modification":

```python
def apply_node_sugar(n: Node, *, node_label: str, f: Callable, kwargs: Mapping[str, Any]) -> Node:
    # 1. DERIVE the shape from the kwargs. combo_for_modifier_names reads
    #    _COMBO_MAP — the one validity authority — and raises on a set that
    #    is not a shape (kwarg-named pre-checks for known pairs run earlier, §3).
    combo = derive_combo(kwargs, node_label=node_label)
    members = modifier_names_for_combo(combo)              # modifiers.py:254

    # 2. VALIDATE the passed kwargs against the shape's declared contract (§2.5).
    _check_kwargs_against_shape(kwargs, combo, node_label)

    # 3. BUILD: one independent, plainly-named application per member modifier,
    #    all reading the SAME `members` value step 2 validated against.
    if "oracle" in members:
        n = _build_oracle_node(n, node_label, f, kwargs)   # n | Oracle(**_build_oracle_kwargs(...))
    if "each" in members:
        n = _build_each_node(n, node_label, kwargs)        # n | Each(**_build_each_kwargs(...))
    if "operator" in members:
        n = _build_operator_node(n, node_label, kwargs)    # register_condition + n | Operator(when=...)
    if "loop" in members:
        n = _build_loop_node(n, node_label, kwargs)        # n | Loop(**_build_loop_kwargs(...))
    if "portal" in members:
        n = _build_portal_node(n, node_label, kwargs)      # n | Portal(**_build_portal_kwargs(...))

    if "oracle" in members:
        n = _apply_eager_oracle_gen_type(n)                # once, end of chain
    return n


def derive_combo(kwargs: Mapping[str, Any], *, node_label: str) -> ModifierCombo:
    """Which shape do these kwargs ask for? Presence of a row's trigger kwargs
    summons that modifier; the name set classifies via _COMBO_MAP."""
    wanted = frozenset(r.name for r in MODIFIER_KWARGS
                       if any(kwargs.get(t) is not None for t in r.triggers))
    return combo_for_modifier_names(wanted, context=node_label)   # modifiers.py:230
```

A reader answers the two load-bearing questions by inspection: (a) *what shape does
this argument combination produce* — read `derive_combo` (five trigger rows, one
`_COMBO_MAP` lookup); (b) *what does that shape ADD over plain node construction* —
read the checks that fire for it, each a 2-5-line named function wrapping the
corresponding `_build_*_kwargs` helper (§1.6). `EACH_ORACLE` needs no special case:
its `oracle` and `each` checks both fire, reproducing today's two pipes. And the
generalization property is structural: a future modifier that `_COMBO_MAP` permits
alongside Loop needs exactly one new `_COMBO_MAP` entry (an IR-layer decision), one
`ModifierKwargs` row, and one `if "x" in members:` line — **no carve-out, ever**.

**Order.** The check order (oracle, each, operator, loop, portal) is today's exact
pipe sequence for all 12 combos — verified combo-by-combo: Oracle-before-Each
matches the combined branch (:509-510), and Operator-before-Loop/Portal matches the
branch order (:557 → :580 → :607). It is also semantically inert: `ModifierSet`
stores modifiers in typed slots (`modifiers.py:757-771`), not a list, so the final
IR is order-independent; Loop's pipe-time validation (`modifiers.py:394-411`) reads
node I/O types, unaffected by which other modifiers already landed; and the
portal-dispatch+Operator rejection checks the *resulting* set regardless of landing
order (`modifiers.py:857-865`). The fixed order is kept anyway so warning/error
firing order is bit-for-bit today's.

`decorator(f)` then reads: existing guard clauses (§3) → mode/DI/inference/
`Node(...)` construction unchanged (:297-488) → `apply_node_sugar` → **one**
`_register_sidecar` + `_set_param_res` site → the eager scripted-shim block
unchanged and last (:620-635) → `return n`. `_apply_eager_oracle_gen_type` moves
from two call sites (:519, :554) to one end-of-chain site — equivalent, because
`oracle_gen_type_for` reads node fields the other pipes don't touch and every pipe
is a private-preserving `model_copy`.

**`PrimaryShape` and `COMBO_DECOMPOSITION` are deliberately NOT imported.** The
decorator never needs the concept "primary shape" — only "which modifier names are
present". Those two tables answer the compiler's question (which single topology
algorithm), not the decorator's (which composable objects to pipe); consuming them
here would carry an unused concept for false symmetry with a genuinely different
problem.

### 2.3 Why derivation is by triggers (the set-matching alternative, examined)

A purer-sounding reading of "find the ModifierCombo whose declared valid-kwarg-set
matches" — subset-match the full passed set against all twelve `valid_kwargs` values
— is ambiguous by construction: the valid sets are **nested**
(`valid_kwargs(EACH) ⊂ valid_kwargs(EACH_ORACLE) ⊂ valid_kwargs(EACH_ORACLE_OPERATOR)`),
so `{map_over, map_key}` subset-matches three combos and BARE's superset matches
everything. Any disambiguation needs a minimality rule — "the smallest combo whose
triggers are all present" — and that rule *is* trigger-based derivation. So the
design keeps the trigger/satellite distinction as part of the declared contract
(each row says which of its kwargs *summon* the modifier and which merely configure
it), and the direction of derivation is still registry→kwargs: triggers determine
the shape, the shape's derived contract then governs everything passed. The
trigger sets are not new invention either — they are today's de-facto semantics
(`has_oracle_kwarg` :491-493; `map_over`/`interrupt_when`/`loop_when`/`portal`
presence-tests), now declared as data instead of encoded in branch conditions.
`map_on_error` stays a satellite (non-None default `"raise"`, :140 — setting it has
never summoned an Each and must not start to).

### 2.4 The layering question, resolved concretely: decorator layer, importing the enum

The registry must live in the **decorator layer** (`_node_modifier_kwargs.py`),
importing `ModifierCombo` / `combo_for_modifier_names` / `modifier_names_for_combo`
as its vocabulary. Not in `modifiers.py`. Four concrete reasons, none hand-waved:

1. **AGENTS.md's layer discipline is explicit and names the module**: "`node.py`,
   `construct.py`, `_construct_validation.py`, `factory.py`, `modifiers.py` are
   off-limits for @node-layer features. The @node decorator is sugar over the IR; it
   must produce instances those modules already accept. Fix every @node-layer gap in
   `decorators.py` without touching the IR." Kwarg NAMES (`map_over`, `ensemble_n`,
   `interrupt_when`) are the @node layer's vocabulary — they appear nowhere in the IR
   and mean nothing to the programmatic or declarative surfaces. A table of them
   inside `modifiers.py` is a decorator feature in an off-limits module, full stop.
2. **The IR-side "abstract param roles" the alternative imagines ALREADY EXIST — as
   the modifier classes' own Pydantic fields.** `Oracle.n/models/merge_prompt/merge_fn/
   merge_model/merge_pre_process/...` (`modifiers.py:562-571`), `Each.over/key/on_error`,
   `Loop.when/max_iterations/on_exhaust` (:657-659), `Operator.when` (:627),
   `Portal.to/route/max_hops/on_exhaust` — with their value-level validation in
   `model_post_init` (Oracle's merge_fn-XOR-merge_prompt at :580-599, Loop's
   on_exhaust domain at :661-668) and their combination validity in
   `_COMBO_MAP`/`_SLOT_RULES`. A new role table beside the existing ones would
   duplicate those field declarations in a second notation — a second vocabulary for
   the same concept, the precise disease the 2026-07-27 audit removed. The
   decorator-side registry instead RECORDS the kwarg→field rename (`field_map`),
   which is the one fact that genuinely lives at the decorator layer.
3. **The dependency direction already exists and points the right way**:
   `decorators.py:121` imports `Each, Loop, Operator, Oracle, Portal` from
   `modifiers.py`; `_node_modifier_kwargs.py` already imports `Node` and
   `ConstructError` and is the established home of the sugar builders and the one
   sanctioned `oracle_gen_type` decoration-time write (`ALLOWED_PREPOP`). Adding
   `from neograph.modifiers import ModifierCombo, combo_for_modifier_names,
   modifier_names_for_combo` extends an existing one-way edge — `compiler.py:67-71`
   already imports combo vocabulary from the same module for its own dispatch, so
   this is the established consumption pattern. The reverse direction would make
   the IR layer aware of its own sugar.
4. **The file-size ratchet blocks the alternative mechanically**: `modifiers.py` is
   881 lines with an exact ALLOWLIST ceiling; growth is "blocked and fixed in-PR,
   never deferred" (AGENTS.md). `_node_modifier_kwargs.py` at 183 lines absorbs
   ~180 new lines with no entry needed.

What DOES belong at the IR layer stays there: shape validity (`_COMBO_MAP`),
membership (`modifier_names_for_combo`), field domains (`model_post_init`). The
registry consumes them and re-states none.

### 2.5 The contract check — every passed kwarg must belong to the shape being built

```python
def _check_kwargs_against_shape(kwargs, combo, node_label) -> None:
    passed = {k for k, v in kwargs.items() if v is not None}
    invalid = passed - valid_kwargs(combo)
    if invalid:
        raise ConstructError.build(
            f"kwarg(s) not valid for a {combo.name} node",
            node=node_label, found=", ".join(f"{k}=" for k in sorted(invalid)),
            hint=f"a {combo.name} node accepts: {sorted(valid_kwargs(combo) - IDENTITY_KWARGS)}",
        )
```

This is the maintainer's inversion made operational: the shape's declared contract
is computed from the registry and the *passed* kwargs are checked against it —
`merge_pre_process=` on what turns out to be a LOOP node is now a construction-time
error naming exactly the kwarg that doesn't belong and the shape being built.
Today's code has no such check at all; each branch consumes what it recognizes and
the rest evaporates (§1.7).

### 2.6 Mode is an orthogonal axis; required/XOR/defaults stay where they are

**Mode is not part of the shape key.** The original brief's
`dict[ModifierCombo | "scripted", ParamSchema]` sketch folds execution mode into
the key; don't — mode (scripted/think/agent/act/raw) and combo are independent
axes (a scripted node can carry Each, a think node Oracle+Operator), and folding
them yields 5×12 rows re-deriving both taxonomies. Mode-specific requirements stay
where they are validated today: prompt/model for LLM modes (:308-320), raw's
`(state, config)` signature (:360-373), dead-body warning (:322-355). The registry
governs exactly the modifier-sugar namespace (18 kwargs); the 14 identity kwargs
are valid for every shape via `IDENTITY_KWARGS`.

Three kinds of within-shape constraint, three homes, each argued:

- **Pairing/dangling rules expressible as presence** (`map_over` requires `map_key`;
  `max_hops`/`route` require `portal`): the first stays a kept guard clause (§3);
  the rest are subsumed by §2.5 (a satellite passed without its row active is
  outside `valid_kwargs(combo)`).
- **Value-level rules** (merge_fn XOR merge_prompt; `ensemble_n >= 2`; `on_exhaust`
  cross-value domains): stay in the builders (`_build_oracle_kwargs`
  `_node_modifier_kwargs.py:111-128`; the loop/portal cross-value raises moving
  into `_build_loop_kwargs`/`_build_portal_kwargs` call paths) — they are not pure
  presence facts (body-as-merge resolution at :92-109 can SUPPLY the merge strategy
  before the XOR is judged), and the authoritative versions live in the modifier
  classes' `model_post_init` anyway; the builder versions exist to name kwargs in
  the error. Encoding them again as registry data would be a third copy.
- **Defaults**: stay with the modifier classes, reached via the builders'
  conditional-include discipline. Load-bearing, not stylistic:
  `_build_portal_kwargs`'s docstring (:166-174) records that `model_fields_set`
  parity with programmatic `| Portal(...)` is read by `_validation_portal` to
  enforce entry-only knobs. A registry that materialized defaults would set fields
  the user never passed and silently break that validation.

### 2.7 What IS shared with the downstream dispatches — exactly this much, verified

§2.2 rejects sharing the compiler's dispatch SHAPE; that is not the same as sharing
nothing. The audit of what the decorator consumes vs re-invents:

- **Shared, by design**: `_COMBO_MAP` through its two sanctioned readers
  (`combo_for_modifier_names`, `modifier_names_for_combo`) — the validity and
  membership facts — and `ConstructError.build` as the error FORMATTER every
  validity raise already goes through (the decorator's guard clauses, `_SLOT_RULES`'
  raises, `combo_for_modifier_names`' raise, and §2.5's contract error all call it).
- **`is_each_oracle_fused` (modifiers.py:288-305): nothing to share, verified not
  assumed.** Its docstring states its reason for existing: a consumer "standing in
  a `PrimaryShape.EACH` arm" holding modifier INSTANCES needs a second, orthogonal
  test to tell a fused node from a plain Each one — a question that only arises
  under shape-dispatch, where `COMBO_DECOMPOSITION` folded the Oracle out of view.
  The decorator never stands in that position: it holds NAMES, and under §2.2 the
  fusion is not a case at all — the `oracle` and `each` membership checks both
  firing IS the fusion. Importing the predicate would import the question along
  with it.
- **No third combo-validity error path exists to unify.** Grep (2026-08-04) for
  "Cannot combine" / "Invalid modifier combination" across `src/neograph/` hits
  ONLY `modifiers.py` — `_construct_validation.py` and `_validation_portal.py`
  have no combo-validity wording of their own. So the wording census after this
  design is: pipe layer "Cannot combine X and Y on the same item" (modifier-named,
  `_SLOT_RULES`), classifier "Invalid modifier combination" (set-named,
  `combo_for_modifier_names`), decorator "X= (A) and Y= (B) cannot be combined"
  (kwarg-named guard clauses) + §2.5's contract error (kwarg-and-shape-named).
  These differ in *vocabulary level* deliberately — each names what its surface
  actually saw — and all share `ConstructError.build` for structure. A shared
  message-building helper beyond that would homogenize wording whose differences
  carry information (which surface caught it), for zero deduplication of validity
  logic (already centralized). Not adopted.
- **External audit of the decorator's output needs no new mechanism — already
  naturally true.** `@node` returns a plain `Node`, and
  `classify_modifiers(item)` (modifiers.py:169-192) reads `item.modifier_set`
  directly — the SAME classifier `compiler.py`, `runner.py`, and the export path
  trust. Any auditor of a decorated node calls it as-is;
  `tests/decorator/test_fanout_oracle.py:1191` already audits decorator output
  exactly this way (`classify_modifiers(verify)[0] is ModifierCombo.EACH_OPERATOR`).
  The design adds no comparison utility and needs none.

---

## 3. Invalid combinations: complete the kwarg-named vocabulary, keep `_COMBO_MAP` the only authority

Decision rule: **set-validity has one source of truth, `_COMBO_MAP`**; the decorator
owns kwarg-*vocabulary* errors (naming the kwargs the user actually typed) for every
invalid pair, and never re-derives validity. What changes vs today: the vocabulary
table becomes complete.

| Combination | Under the design | Rationale |
|---|---|---|
| `map_over`+`loop_when`; `portal`+`map_over`; `portal`+`loop_when`; `map_over`⊕`map_key` | Existing guard clauses kept verbatim (`decorators.py:243-295`), running before `derive_combo` | Messages already kwarg-named and asserted-against; churn buys nothing. |
| oracle-kwargs + `loop_when`; oracle-kwargs + `portal` | **NEW kwarg-named guard clauses** in the same block, same style ("ensemble_n=/models=/merge_fn=/merge_prompt= (Oracle) and loop_when= (Loop) cannot be combined on the same node"), so `derive_combo` never sees the pair | Completes the six-pair table. Verified safe: every test asserting the old pipe-layer message uses the programmatic surface (§1.3), which keeps its `_SLOT_RULES` raise untouched — no pinned assertion moves. Still a *message* change for the @node input path (today: "Cannot combine Oracle and Loop" from the pipe; after: kwarg-named from the guard block) — named in §4's accounting, not smuggled. |
| `max_hops`/`route` dangling | Existing guard clauses kept; also subsumed by §2.5 | Redundant coverage is fine; the specific hints are better than the generic contract error. |
| `interrupt_when` + `portal` in dispatch mode | Stays with `with_modifier`'s dynamic `is_dispatch` check (`modifiers.py:857-865`) | Depends on the Portal *instance's* `route` value, not kwarg presence; a static contract cannot express it and must not try. |
| Any future invalid set no clause names | `combo_for_modifier_names` raises "Invalid modifier combination" (`modifiers.py:243-251`) inside `derive_combo` | The total backstop: a sixth modifier row added without a `_COMBO_MAP` entry fails loud, never `KeyError`s. |
| Any valid shape + a foreign kwarg | §2.5's contract error, naming kwarg and shape | The new strictness — §4. |

---

## 4. Strictness vs zero-behavior-change: the honest accounting

**This design is NOT zero-behavior-change in the accepted-input sense, and cannot
be** — §2.5 is *specified* as stricter validation than today's code has. Precisely
what changes, named:

1. **Newly rejected inputs** (today silently ignored, §1.7): `max_iterations` without
   `loop_when`; `on_exhaust` without `loop_when`/`portal`; `map_on_error='collect'`
   without `map_over`; `merge_model`/`merge_pre_process`/`merge_post_process`/
   `merge_fallback` without an Oracle trigger. Each becomes a decoration-time
   `ConstructError` naming the kwarg and the shape.
2. **Re-sourced errors, same inputs** (today rejected by the pipe layer inside the
   decorator call, after this design rejected by the new guard clauses — identical
   timing, better message): oracle-kwargs+`loop_when`, oracle-kwargs+`portal`.
3. **Everything else**: unchanged — same `Node(...)` construction, same builders and
   `model_fields_set` parity, same pipe order (verified for all 12 combos, §2.2),
   same warnings (stacklevel re-tuned, §6.4), same eager-shim behavior, one sidecar
   site observationally identical to five (§1.6).

**Is it achievable against the current suite?** Predicted yes, with evidence but not
proof: grep (2026-08-04) found no test in `tests/decorator/`, `tests/modes/`,
`tests/modifiers/`, or `tests/check_fixtures/` that passes a §1.7-dangling kwarg and
asserts success, and no @node-path test pinning the two re-sourced messages (§1.3 —
all pipe-message assertions are programmatic-surface). The proof is empirical and is
built into the plan: the strictness lands as its own phase (§7 Phase 3b) whose gate
run either comes back green — confirming suite-neutrality — or names the exact
test(s) relying on leniency, which then get listed here and decided by the
maintainer, never silently accommodated. Per the 0.x policy (no shims, one known
downstream) the *user-facing* break is acceptable by default; piarch's constructs
should be grepped for §1.7 patterns before the phase lands (one `rg` over
`piarch/src/derive_ensemble/constructs/`).

Phases 0-2 (registry, dispatch rewrite, sidecar centralization) remain strictly
zero-behavior-change; the two behavior-touching steps (new guard clauses; the
contract check) are Phases 3a/3b, separately shippable and separately revertable.

---

## 5. Why the s7zt3.10 class is unrepresentable here — argued, not asserted

The bug class: *kwargs implied a modifier; the returned node silently lacks it.*
Today it is representable because "which modifiers the kwargs imply" and "which
modifiers get applied" are computed by DIFFERENT code (five branch conditions vs
five imperative pipe sites), separable by any control-flow edit — the early
`return` — with no error.

Under §2.2 the two computations read **one value**. `members =
modifier_names_for_combo(combo)` is computed once; §2.5 validates the passed kwargs
against it (via `valid_kwargs`, which reads the same membership), and step 3
applies exactly its members — five independent, unconditional checks on the same
frozenset. The divergence the bug class requires has no home:

- **No check can drop a sibling.** Each membership check applies exactly ONE
  modifier and falls through unconditionally to the next; no branch is responsible
  for more than one pipe, so "handled the first modifier, forgot the second" —
  s7zt3.10's exact shape, and the residual risk any hand-written multi-modifier
  case would carry — is structurally impossible. `EACH_ORACLE` is not a special
  case that could be mis-handled; it is the `oracle` check firing and then the
  `each` check firing.
- **Operator specifically cannot be omitted from the shape.** Its presence is an
  axis of the combo's identity (`_COMBO_MAP` partitions the 12 combos into 6
  operator-free / 6 `*_OPERATOR`): `interrupt_when` passed → `"operator" ∈ wanted`
  (a registry row, not a branch) → `derive_combo` returns a `*_OPERATOR` combo or
  raises. A combo omitting Operator when `interrupt_when` was passed is not in the
  function's image for that input; then `"operator" in members` fires like any
  other member. Dropping it now requires corrupting `_COMBO_MAP` itself — an
  IR-layer edit, pinned by §6.3's round-trip guard and
  `tests/test_combo_decomposition.py`.
- **Editing the dispatch back into a droppable shape is guard-banned.** The two
  regrowth signatures — a `return` inside a membership check, or a membership
  check whose condition stops reading `members` — are what the retargeted RULE 1
  scans for (§6.2), with the same failing-first meta-test discipline the current
  guard has.

Because the applied set IS the validated set by construction, a runtime
"assert what-was-applied == what-was-validated" post-condition would compare a
value against itself — **provably redundant, and therefore not part of the
design**. (An earlier working note kept such an assert as a safety net; the
independent-checks structure is precisely what makes it unnecessary. Phase 1 uses
a temporary cross-check of the OLD branches during migration — a scaffold deleted
in Phase 2, not a permanent fixture.)

What remains representable is a **different bug class**: a builder mis-constructing
its OWN modifier — `_build_loop_node` ignoring `max_iterations`, or degenerately
returning `n` unpiped. That is builder value-correctness, not composition
correctness — and the existing suites' coverage of it is THIN, not (as one might
assume) handled "by the per-modifier suites". Census of @node-path assertions that
a passed kwarg VALUE actually landed on the built modifier instance (grep
2026-08-04 over `tests/decorator/`, `tests/modifiers/`, `tests/modes/`):

| Modifier | @node-path value assertions today |
|---|---|
| Each | `over`/`key` asserted (`tests/decorator/test_fanout_oracle.py:74-75`); `map_on_error` never |
| Oracle | `models`/`n`/`merge_fn` asserted once, body-as-merge case (`tests/modifiers/test_oracle.py:461-464`); `ensemble_n`→`n`, `merge_model`, the three hooks never |
| Operator | `when` asserted (`tests/decorator/test_fanout_oracle.py:1163`, `:1190`) |
| Loop | **none, anywhere** — no `modifier_set.loop.*` / `get_modifier(Loop).*` assertion exists on any surface; `tests/decorator/test_edge_cases.py:158-168` passes `max_iterations=5` and asserts only `has_modifier(Loop)` |
| Portal | **none on the @node path** — `tests/modifiers/test_portal.py:105-110` asserts defaults on a programmatic `Portal(to=["a"])`; `tests/test_agent_spec_construct_member.py:101` is the import path; no `portal=` sugar value assertion exists |

So today's suite verifies *composition* (which modifier attached) but, for Loop and
Portal, not *value-threading* (did `5` reach `loop.max_iterations`). The 2^5 grid
(§6.1) is therefore extended to assert VALUE fidelity, not just presence — closing
this gap in the guard that already owns the composition question rather than a new
file.

---

## 6. Structural guards (failing-first; mutation via Edit+revert, never `git checkout` on uncommitted work)

### 6.1 NEW: the kwarg-composition-and-value grid — `tests/test_guards_node_kwarg_grid.py`

The test that would have caught s7zt3.10 before it shipped, in this suite's
derived-not-enumerated style (`test_combo_decomposition.py:131-133` is the
precedent): enumerate **all 2^5 = 32 subsets** of the five trigger groups with valid
values; expected outcome DERIVED from `_COMBO_MAP`, never hand-listed. Each row
carries both the kwargs to pass AND the modifier-instance fields those kwargs must
land on, so every valid case asserts **membership AND value fidelity** — the §5
census showed value-threading is currently unasserted for Loop and Portal entirely:

```python
LOOP_PRED = lambda d: d is None
TRIGGER_ROWS = {
    #  name       kwargs to pass                                  slot     expected instance fields
    "each":     (dict(map_over="up.items", map_key="label"),      "each",
                 {"over": "up.items", "key": "label"}),
    "oracle":   (dict(ensemble_n=3, merge_prompt="m/merge"),      "oracle",
                 {"n": 3, "merge_prompt": "m/merge"}),
    "operator": (dict(interrupt_when="needs_review"),             "operator",
                 {"when": "needs_review"}),
    "loop":     (dict(loop_when=LOOP_PRED, max_iterations=4),     "loop",
                 {"when": LOOP_PRED, "max_iterations": 4}),
    "portal":   (dict(portal=["peer_a"], max_hops=5),             "portal",
                 {"to": ["peer_a"], "max_hops": 5}),
}

@pytest.mark.parametrize("subset", all_subsets(TRIGGER_ROWS))         # 32 cases
def test_every_trigger_subset_yields_the_implied_modifiers_with_the_passed_values(subset):
    if frozenset(subset) not in _COMBO_MAP:                            # the ONE authority
        with pytest.raises(ConstructError):
            make_node(**merged_kwargs(subset))
        return
    n = make_node(**merged_kwargs(subset))
    assert modifier_names_for_combo(n.modifier_set.combo) == frozenset(subset)
    for name in subset:                                                # value fidelity
        _, slot, expected = TRIGGER_ROWS[name]
        mod = getattr(n.modifier_set, slot)
        for field, value in expected.items():
            assert getattr(mod, field) == value, f"{name}.{field} did not thread through @node"
```

Pre-s7zt3.10-fix, subset `{"each", "operator"}` returns an EACH node and the
membership equality fails — the bug caught by a test that never mentions it. The
value loop is the §5 residual-class guard with teeth: a builder that fails to pipe
its modifier fails membership; a builder that pipes it but drops a kwarg on the
floor (the today-unasserted `max_iterations=5` case) fails the field assertion.
Satellite choices are deliberate: `max_iterations` (Loop) and `max_hops` (Portal)
are the two per-row satellites NOT shared across rows, so merged subsets stay
unambiguous (`on_exhaust` is excluded from the grid rows for exactly that reason —
its cross-shape value rules are pinned by the §3 fixtures instead); `max_hops` at
decoration is safe because `_validation_portal`'s entry-only rule runs at mesh
assembly, which the grid never reaches. A totality assertion
(`set(TRIGGER_ROWS) ==` the modifier-name universe of `_COMBO_MAP`) fails the file
loudly when a sixth modifier gains a combo without a grid row — and the grid
extends to 2^6 automatically once the row is added. Lands in **Phase 0 against the
CURRENT code**, where it must pass (the value assertions hold today — today's
branches DO thread values; they are merely unasserted); non-vacuity demonstrated
by Edit-reinserting the old early `return`, watching `{"each","operator"}` fail,
Edit-reverting — plus one value-mutation control (Edit `loop_kwargs` to drop
`max_iterations`, watch the `{"loop"}` case fail, revert).

### 6.2 EVOLVED: `test_guards_modifier_composition_completeness.py` RULE 1

The scanner keys on `decorator(f)`'s five branch-test names; after Phase 2 those
branches are gone and an unmodified scanner is silently vacuous. In the SAME
commit, retarget it to the successor invariants in `apply_node_sugar`, pure-AST as
now: (i) exactly one `return` (the terminal one) — no membership check may
early-return past its siblings; (ii) every `if` whose body calls a
`_build_*_node` function has a test of the form `"<name>" in members` — the
applied set is READ from the one authority-derived value, never from a re-derived
kwarg condition (a check regrown as `if loop_when is not None:` is the disease
signature: it re-splits "implied" from "applied"); (iii) the set of
membership-checked names equals `{r.name for r in MODIFIER_KWARGS}` — no member
silently unhandled. Keep the meta-test pattern against synthetic sources of the
new shape (positive control: a `return` inside a check / a check testing a raw
kwarg; negative control: the five-check fall-through). RULE 2 (`_agent_spec.py`'s
Operator postlude outside the match) is untouched — it remains correct THERE
because export lowering, like the compiler, genuinely dispatches on shape.

### 6.3 NEW: registry integrity — same guard file

1. **Round-trip**: for every valid frozenset `S` in `_COMBO_MAP`,
   `modifier_names_for_combo(combo_for_modifier_names(S)) == S` — the §5 argument's
   load-bearing bijectivity, currently exercised only implicitly
   (`test_combo_decomposition.py:62-65` inverts the map but does not pin
   injectivity as such).
2. **Row/universe totality**: `{r.name for r in MODIFIER_KWARGS} ==` the union of
   `_COMBO_MAP`'s keys — a new modifier cannot gain @node sugar without a row, and
   a row cannot name a modifier the combo table doesn't know. (The
   every-member-checked half lives in §6.2's item iii, where the AST is already in
   hand.)
3. **The anti-flat-explosion ratchet**: every keyword parameter of `node()` (via
   `inspect.signature`) appears in ≥1 row's triggers/satellites or in
   `IDENTITY_KWARGS`, and the three sets are mutually exclusive except for
   documented shared satellites (`on_exhaust`). A 33rd kwarg added without
   declaring its owning shape(s) fails CI — the regrowth signature of the disease.
4. **`field_map` honesty**: every `field_map` value is a real field of the row's
   modifier class (`model_fields`), so the rename table cannot rot.

### 6.4 Carried hazards

`_build_oracle_kwargs`'s body-as-merge warning uses `stacklevel=4`
(`_node_modifier_kwargs.py:97`), tuned to today's call depth; `apply_node_sugar` +
`_build_oracle_node` add frames. Phase 2 re-tunes and adds one assertion on the
warning's reported filename (the existing `pytest.warns` tests pass regardless of
stacklevel, so attribution is currently unpinned). The strictness phase adds
should_fail check fixtures per new error class (per the check-fixture rule: one
fixture per new validation rule, plus a should_pass twin).

---

## 7. Phased build plan (each phase separately shippable, guard-first)

**Phase 0 — pin current behavior (no src change).** `tests/test_guards_node_kwarg_grid.py`
(§6.1) against the current tree; all 32 cases pass; Edit-mutation proves non-vacuity.
Commit. Safety net for everything after.

**Phase 1 — registry, additive (zero-behavior-change).** Add `ModifierKwargs`,
`MODIFIER_KWARGS`, `IDENTITY_KWARGS`, `valid_kwargs`, `derive_combo` to
`_node_modifier_kwargs.py`; §6.3 guards written first, failing until the tables
exist. As a migration scaffold only, `decorator(f)` additionally derives the combo
and cross-checks `modifier_names_for_combo(n.modifier_set.combo)` against it at the
existing single exit — auditing the OLD branches for one phase, so any latent drop
the grid's fixed values miss surfaces now. Branches untouched. Gate: bare
`uv run pytest`.

**Phase 2 — dispatch rewrite (zero-behavior-change).** Extract
`_build_operator_kwargs`/`_build_loop_kwargs` verbatim; add the five named
`_build_*_node` functions and the five independent membership checks (§2.2);
replace the six branches; delete the Phase-1 scaffold cross-check (§5 — the
dispatch now reads the same value it would compare against); collapse to one
sidecar site; re-tune stacklevels + attribution assertion; retarget RULE 1 in the
same commit (§6.2). Run `tests/test_node_sidecar_contract.py`, `tests/decorator/`,
`tests/modes/`, `tests/modifiers/`, then full gate. Note: `derive_combo` must run
after the kept guard clauses, and the two §1.3 gap-pairs would now hit its generic
"Invalid modifier combination" message before the pipe layer's — so Phase 2 either
carries a temporary pre-derivation re-raise mapping those two sets to the pipe
layer's exact message (deleted in Phase 3a), or Phase 3a lands first. Sequencing is
the implementer's choice; the constraint is that no phase changes a message unless
it is the phase that owns message changes.

**Phase 3a — complete the kwarg-named vocabulary (behavior: message re-sourcing
only).** New guard clauses for oracle-kwargs+`loop_when` / oracle-kwargs+`portal`
(§3); should_fail fixtures for both; confirm the programmatic-surface tests still
pin the pipe-layer messages (untouched by construction).

**Phase 3b — the contract check (behavior: new strictness, maintainer-gated).**
Enable §2.5; should_fail fixtures per §1.7 pattern + should_pass twins; grid
extended with dangling-satellite cases; `rg` piarch's constructs for §1.7 patterns
first; full gate either green (suite-neutrality proven) or the relying tests named
here for an explicit decision. Ceiling bookkeeping: re-measure `decorators.py`
(sheds ~130-150 lines; update or delete its `ALLOWLIST` entry per the exact-ceiling
rule) and `_node_modifier_kwargs.py` (~360-400, no entry needed while under 500).
Update AGENTS.md's @node sections; close neograph-jtawq.4 citing this doc.

---

## 8. Refusal-criteria check

- **No capability monopoly widened.** No `Command(` (G1); no IR-field writes
  (`fan_out_param`/`handoff_*` untouched, G3 moot); the `oracle_gen_type` write
  stays in `_node_modifier_kwargs.py` (`ALLOWED_PREPOP` unchanged — a re-key of
  nothing). New imports are all along the existing decorator→IR edge — combo
  vocabulary from `modifiers.py`, the same consumption pattern `compiler.py:67-71`
  established; no function-local imports, so `FUNCTION_LOCAL_IMPORT_ALLOWLIST`
  never grows.
- **`modifiers.py` untouched** — the layering answer (§2.4) keeps every @node
  vocabulary fact out of the off-limits modules, per AGENTS.md's rule, verbatim.
- **No second taxonomy, no borrowed mismatched pattern.** Zero new enums and zero
  new combo-composition data: shapes are `ModifierCombo`, validity is
  `_COMBO_MAP`, membership is `modifier_names_for_combo`, and the dispatch is five
  independent membership checks on that one value — composition encoded as
  composition, not forced through the compiler's mutually-exclusive
  topology-selection pattern, which answers a different question (§2.2). The only
  new declaration is the per-modifier kwarg rename/contract rows, information that
  exists today only as branch conditions and is decorator-layer by nature.
- **Behavior changes are enumerated, phase-isolated, and gated** (§4): two message
  re-sourcings (3a) and one strictness rule (3b), each with its own fixtures and
  its own revert line; Phases 0-2 provably inert under the Phase-0 grid.

## 9. Summary

The 32-kwarg flat namespace becomes a declared contract: five per-modifier rows (the
only new data) + `_COMBO_MAP` (unchanged, still the only validity authority) derive
a total `valid_kwargs: ModifierCombo → frozenset[str]`, against which every passed
kwarg is validated — expected parameters from the known shape, never shape guessed
from parameters. Dispatch mirrors the domain: `derive_combo(kwargs)`, then five
independent, plainly-named membership checks on the one authority-derived `members`
set — modifiers compose, so no mutually-exclusive match, no fusion carve-out, and a
future combinable modifier costs one `_COMBO_MAP` entry, one registry row, and one
membership line. The s7zt3.10 class loses its structural precondition (the
validated and applied modifier sets are one value, making a runtime
what-was-applied assert provably redundant); builder value bugs remain a distinct
class guarded by the 2^5 grid; the §1.7 silent-satellite seams become loud; five
sidecar sites become one; and the §6.3 ratchet makes the next kwarg declare its
owning shape or fail CI.
