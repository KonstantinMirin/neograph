# File-size refactor cluster C — node.py / spec_types.py / di.py

Read in full: `src/neograph/node.py` (522 lines), `src/neograph/spec_types.py` (521
lines), `src/neograph/di.py` (506 lines). Cross-checked against epic-active files
`_agent_spec.py` (1561), `loader.py` (1393), `_wiring.py` (1461), `modifiers.py`
(1116), `factory.py` (1007), `_agent_cycle.py` (1054) for overlap.

None of the three assigned files sit on the active epic's Phase 8 critical path
(`neograph-s7zt3.11`, fusion `ModifierCombo` lowering to `Construct`). Confirmed
by grep: nothing in `_agent_spec.py`/`_wiring.py` references `ModifierCombo` in
the same breath as `spec_types`/`di`/`node`.py internals — the only touch points
are ordinary one-directional imports (`lookup_type`, `model_to_agent_spec_properties`).

---

## 1. `src/neograph/node.py` (522 lines)

Sections, in file order:

| Lines | Responsibility |
|---|---|
| 39–90 | Node lifecycle **Protocols**: `SkipPredicate`, `SkipValueFactory`, `RawNodeFn`, `HasName` — pure `typing.Protocol` definitions, no dependency on `Node` itself |
| 93–149 | **TypeSpec machinery**: `_validate_type_spec`, `_is_type_like`, the `TypeSpec` / `TypeSpecStatic` type aliases — self-contained validation logic for the `inputs`/`outputs` field type |
| 152–357 | The **`Node` Pydantic model** itself: field declarations, `_normalize_raw_base_tools` validator, `__init__`, `_validate_gate_tools_when`, `_validate_skip_callables`, `Node.scripted()` classmethod |
| 359–522 | **`Node.run_isolated()`** — a single ~164-line method, a testing/inspection utility that bypasses `compile()`/`run()`, builds a minimal state dict, invokes the factory-built node fn directly |

### Extraction candidates

**A. Protocols → `src/neograph/_node_protocols.py`** (SAFE NOW)
Moves `SkipPredicate`, `SkipValueFactory`, `RawNodeFn`, `HasName` (~55 lines) verbatim.
Zero logic, no state, referenced only as field-type annotations on `Node`. `node.py`
already imports `Protocol`/`runtime_checkable` for this cluster only — after the
move those imports drop too. Removes ~55–60 lines from `node.py`.

**B. TypeSpec/TypeSpecStatic → `src/neograph/_type_spec.py`** (SAFE NOW)
Moves `_validate_type_spec`, `_is_type_like`, `TypeSpec`, `TypeSpecStatic` (~55
lines) verbatim. This is already written as free functions with no reference to
`Node`; `Node.inputs`/`Node.outputs` just annotate with the imported `TypeSpec`.
The `TypeSpecStatic` alias is also consumed elsewhere (grep for other consumers
before moving, but the alias itself doesn't change). Removes ~55 lines.

**C. `run_isolated` body → `src/neograph/_node_run_isolated.py`** (SAFE NOW,
slightly more care needed)
Extract the body into a module-level function `_run_isolated(node: Node, *,
input=None, config=None, llm_factory=None, prompt_compiler=None, scripted=None,
conditions=None, tool_factories=None) -> Any`, leave `Node.run_isolated` as a
5-line delegating wrapper (`return _run_isolated(self, input=input, ...)`).
Pure cut-and-paste + one indirection; no behavior change, no signature change on
the public method. All the local imports it already does (`from neograph.factory
import make_node_fn`, `from neograph._runtime_registry import
_decoration_registry`) move unchanged. Removes ~150 lines net (164 minus the
5-line wrapper kept behind).
Reasoning it's still SAFE NOW despite being "a method, not a free cluster of
functions": it has no dependency on private `Node` state beyond `self` passed
explicitly (`self.modifier_set`, `self.mode`, `self.name`, `self.scripted_fn`,
`self._scripted_shim`) — trivially passed as the first arg. This is a textbook
mechanical extraction, not a redesign.

Combined A+B+C removes **~260 lines**, taking `node.py` from 522 → roughly
**265 lines** (Node class + `__init__` + validators + `scripted()` classmethod),
comfortably under the 500-line cap with headroom for Phase 8 or any other future
`Node`-field additions.

### DEFER

- **Splitting the `Node` Pydantic model itself** (fields for LLM config vs.
  Portal/handoff vs. modifiers vs. tool-gating) into a mixin or composed-model
  shape. This is a real design decision (Pydantic multiple-inheritance/mixin
  field ordering, `model_config` merge behavior, whether `PrivateAttr`s move
  cleanly) and isn't needed once A+B+C land — the remaining class is ~265 lines,
  under budget. Don't do this now; only revisit if the class re-grows.

---

## 2. `src/neograph/spec_types.py` (521 lines)

Sections, in file order:

| Lines | Responsibility |
|---|---|
| 34–95 | **Core type registry**: `_type_registry` dict, `_fields_match`, `register_type`, `lookup_type` |
| 97–199 | **JSON-Schema-dict → Python type walker**: `_no_repr_check`, `_resolve_field_type` (the "DIRECT-tier" walker referenced throughout as the Core Invariant's canonical walker) |
| 202–235 | `load_project_types` — consumes the above two clusters to build models from a project config's `types:` section |
| 238–431 | **Agent Spec `Property` → neograph type bridge (IMPORT direction)**: `_import_agent_spec_property_classes`, `_property_type_signature`, `_structural_type_name`, `_property_to_field_type`, `agent_spec_properties_to_types` |
| 434–521 | **neograph type → Agent Spec `Property` bridge (EXPORT direction)**: `_annotation_to_property`, `model_to_agent_spec_properties` |

The Agent Spec bridge (import + export, lines 238–521, ~284 lines) is a
structurally distinct concern from the JSON-Schema/registry core (lines 34–235,
~202 lines) — the module's own docstring calls this out as a secondary
responsibility ("Also bridges Agent Spec `Property`... via..."). It only shares
two things with the core: calling `register_type`/`lookup_type` (public
functions, trivially imported) and one explicit documented fallback
(`_property_to_field_type`'s `Property` branch calls `_resolve_field_type` as a
last resort for the no-discriminator round-trip gap).

### Extraction candidate

**Agent Spec bridge → `src/neograph/_spec_types_agent_spec.py`** (DEFER, not
SAFE NOW — see reasoning)
Moves `_import_agent_spec_property_classes`, `_property_type_signature`,
`_structural_type_name`, `_property_to_field_type`,
`agent_spec_properties_to_types`, `_annotation_to_property`,
`model_to_agent_spec_properties` (~284 lines) to a new module that imports
`register_type`, `lookup_type`, `_resolve_field_type`, `_no_repr_check` from
`spec_types.py`. Mechanically this is clean (one-directional import, no cycle) —
`spec_types.py` would drop to ~235 lines and the new module would sit at ~290.

Reasoning for **DEFER despite being mechanically clean**: this is the Agent
Spec import/export bridge, and the active epic (`neograph-s7zt3`, Agent Spec /
Portal architecture rebuild) is *specifically* about Agent Spec surfaces. Even
though grep confirms Phase 8 (`ModifierCombo` → `Construct` fusion) doesn't
currently reference these particular functions, the epic's whole
`_agent_spec.py`/`loader.py` surface is volatile enough this session/next that
introducing a new module boundary in the Agent Spec bridge risks import-path
churn the epic didn't ask for (every future patch touching `spec_types.py`'s
Property-conversion code would need to know it moved). Land this only once
Phase 8 is closed and confirmed not to still be touching `_agent_spec.py`
import paths.

### Duplication found with epic-active file `_agent_spec.py`

`_agent_spec.py:97-100` explicitly says its `_import_agent_spec_flow_classes`
"Copies `spec_types._import_agent_spec_property_classes()`'s exact import-guard
shape." This is real, acknowledged pattern duplication: two independent
function-local-import-with-fail-loud-ConfigurationError guards for the same
`pyagentspec` optional dependency, one importing `property`, the other importing
`flows.edges`/`flows.flow`/`flows.nodes`/`property`/`tools`. A single shared
helper (e.g. `_pyagentspec_import.py:import_pyagentspec_submodules(*names) ->
tuple[ModuleType, ...]`, raising the shared `ConfigurationError` with the
`[agent-spec]` extra hint) would collapse both call sites to one line each and
remove the near-duplicate try/except/raise blocks (~15 lines each) from both
files. **DEFER** — same rationale as above, this touches `_agent_spec.py`
directly, which the epic is actively editing.

---

## 3. `src/neograph/di.py` (506 lines)

Sections, in file order:

| Lines | Responsibility |
|---|---|
| 33–121 | **Resource-fetcher config plumbing**: `RESOURCE_FETCHER_KEY`, `RESOURCE_REPLAYER_KEY`, `_get_configurable`, `_FETCHER_HINT`, `_require_fetcher` |
| 124–267 | **Resource hydration + layered expiry**: `parse_resource_content`, `_configurable_dict`, `_enforce_max_bytes`, `hydrate_resource_ref` (read → replay → fail-loud) |
| 269–307 | **Value-unwrap helpers shared across the runtime**: `_unwrap_loop_value`, `_unwrap_each_dict` — explicitly documented as "single source of truth ... used by `_extract_input`, `_resolve_merge_args`, and `loop_router`" |
| 309–327 | `_isinstance_safe` — generic Union/Optional/generic-origin-aware isinstance check |
| 329–426 | **`DIKind` enum + `DI_TEMPLATE_KINDS`** (defined near top, line 48–84) and **`DIBinding` dataclass + `resolve()`** — the core sync resolution path for all 6 DI kinds |
| 435–507 | **`DIBinding.aresolve()` + `_aresolve_from_manifest`** — async resolution twin, manifest-driven hydration for `FromResource(ref=...)` |

### Extraction candidate

**Resource hydration cluster → `src/neograph/_resource_hydration.py`** (SAFE NOW)
Moves `RESOURCE_FETCHER_KEY`, `RESOURCE_REPLAYER_KEY`, `_get_configurable`,
`_FETCHER_HINT`, `_require_fetcher`, `parse_resource_content`,
`_configurable_dict`, `_enforce_max_bytes`, `hydrate_resource_ref` (~230 lines,
roughly lines 33–267) into a new module. This cluster is genuinely
self-contained: it only touches `config['configurable']` dicts and
`ResourceRef`-shaped objects, never `DIBinding`/`DIKind`. The two call sites
that need it (`DIBinding.aresolve`/`_aresolve_from_manifest` in `di.py`, and
`tool.py`'s `resource_reader`, per the module's own docstring cross-reference at
line 142) both already do a plain top-level import — swapping the import
source is a one-line change per call site. Removes ~230 lines, taking `di.py`
from 506 → **~276 lines**.

Verify before landing: `tool.py` currently imports these names from `di.py`
directly (confirm via `grep -n "from neograph.di import" src/neograph/tool.py`)
— if so, update that import line too as part of the same mechanical commit.

`_isinstance_safe`, `_unwrap_loop_value`, `_unwrap_each_dict` should **stay** in
`di.py` — they're small, and `_unwrap_loop_value`/`_unwrap_each_dict` are used
by non-DI callers (`_extract_input`, `loop_router`) whose docstrings already
point at `di.py` as the canonical home; moving them would just relocate the
"shared helper module" problem one level sideways for ~35 lines of gain, not
worth a second new-module boundary in the same pass.

### DEFER

- **`DIBinding`/`DIKind`/`resolve`/`aresolve` split (sync vs async cores)**:
  the sync/async resolution logic is tightly coupled (aresolve delegates to
  resolve for 5 of 6 kinds) and both are genuinely part of "the one resolver
  path" the module docstring promises. Splitting sync/async into separate files
  would fight the module's own design invariant ("Single module with one
  resolver path for all DI parameters"). Not worth doing — after the resource
  extraction this whole remaining cluster is ~270 lines, under budget anyway.

### Duplication check against epic-active files

No real duplication found between `di.py` and `_agent_cycle.py`/`factory.py`/
`_wiring.py`/`modifiers.py` — `_agent_cycle.py`'s DI injection
(`_turn_prep_kwargs`) *calls into* `DIBinding.resolve`/the shared
`_inject_di_inputs`, it doesn't reimplement resolution logic. This is the
intended one-resolver-path architecture working as documented (AGENTS.md
"di_inputs" section), not a duplication finding.

---

## Summary of line-count impact if all SAFE NOW items land

| File | Before | After SAFE NOW extractions | New modules created |
|---|---|---|---|
| `node.py` | 522 | ~265 | `_node_protocols.py` (~60), `_type_spec.py` (~55), `_node_run_isolated.py` (~160) |
| `spec_types.py` | 521 | 521 (no SAFE NOW extraction — see DEFER) | none |
| `di.py` | 506 | ~276 | `_resource_hydration.py` (~230) |

`spec_types.py` has a mechanically clean split available but it's classified
DEFER solely because it sits inside the Agent Spec surface the active epic is
touching — re-evaluate once Phase 8 closes.
