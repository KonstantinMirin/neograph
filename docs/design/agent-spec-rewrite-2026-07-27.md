# Agent Spec export/import: from-scratch rewrite spec

**Status**: DRAFT — awaiting adversarial review (same process as Option F / `neograph-cbpyx`). No code, no branch, no deletion has happened as part of producing this document.

**Answers**: `neograph-s7zt3` (epic), `neograph-s7zt3.2` (bug — PORTAL_OPERATOR's dropped approval gate), `neograph-s7zt3.3` (task — compositional rebuild of `_lower_construct_item`).

**Supersedes nothing, preserves everything correct**: this spec does not re-litigate `agent-spec-interop-2026-07-09.md`'s positioning (still valid), or `agent-spec-placeholder-translation-2026-07-26.md`'s Option F design (still valid, ships as `neograph-cbpyx`, still passing at `tests/test_agent_spec_placeholder_translation.py`). It **replaces** the flat, hand-derived `_lower_construct_item`/`_lower_portal_mesh_to_swarm` dispatch structure that both `s7zt3.2` and `s7zt3.3` diagnosed as the second occurrence of the "duplicated source of truth" anti-pattern first found in `agent-spec-oracle-inputs-2026-07-25-architecture-retrospective.md`.

---

## 0. Recap: why this rewrite exists (one paragraph, so this doc stands alone)

`_agent_spec.py`'s `to_agent_spec()` intercepts Portal-mesh constructs before `_lower_construct_item` ever runs, and `_lower_construct_item` itself is a flat 5-branch dispatch (`ORACLE`, `EACH`, `LOOP`, `OPERATOR`, `BARE`) over `ModifierCombo`, with every composed combo (`EACH_ORACLE`, `EACH_OPERATOR`, `ORACLE_OPERATOR`, `LOOP_OPERATOR`, `EACH_ORACLE_OPERATOR`) falling through to a generic `ConfigurationError`, and Portal combos (`PORTAL`, `PORTAL_OPERATOR`) never reaching this dispatch at all. `compiler.py` — the thing that actually makes these pipelines run on LangGraph — proves all 12 `ModifierCombo` values are legitimate, running compositions. `compiler.py` actually contains **two** independent `ModifierCombo` match statements, not one: `_add_node_to_graph` (Node-level: a single item's own modifiers) and `_add_subgraph` (Construct-level: modifiers attached to a Construct used as one item inside another Construct). Each has 6 case arms (5 plain `X | X_OPERATOR` groupings for `EACH`, `ORACLE`, `LOOP`, `BARE`, `PORTAL`, plus a 6th, separately-cased `EACH_ORACLE | EACH_ORACLE_OPERATOR` fusion arm), and — critically — **the two match statements have different bodies for the same combo**: `EACH_ORACLE`/`EACH_ORACLE_OPERATOR` gets a real fused Node-level lowering but an unconditional `CompileError` at the Construct level (the fusion is defined only in terms of a single Node's `map_over`/`ensemble_n` fields, which a multi-node Construct doesn't have); `PORTAL`/`PORTAL_OPERATOR` is unreachable-by-construction at the Node level (Portal membership is resolved by a separate mesh-detection pre-pass before per-node dispatch ever runs — a defense-in-depth arm) and unconditionally rejected at the Construct level (a Portal mesh member must be a bare Node, never a Construct). So "what a `ModifierCombo` means" is genuinely two related but distinct facts — its Node-level decomposition (universal, all 12 combos meaningful) and its Construct-level validity (a narrower subset) — and `_agent_spec.py` encoded neither correctly: it approximates 5 of the 12 Node-level decompositions, has no Construct-level modifier check on a Construct-as-item at all today (the `isinstance(item, Construct)` branch in `_lower_construct_item` wraps the sub-flow before `classify_modifiers` is ever called on it), and routes Portal through a hand-written pre-pass with no awareness of Operator composition. This is the exact anti-pattern the retrospective already named for `_lower_node`/`_lower_oracle` at the per-mode level, recurring one level up. `s7zt3.2` found the sharpest symptom: a `PORTAL_OPERATOR` mesh member (Portal peer + human-approval gate) exports as a bare `Agent` inside a `Swarm` with the approval gate **silently dropped** — a safety-critical control-flow seam vanishing without error, discovered only by empirically testing the combo, not by reading code. The maintainer's ruling: **fail-loud is not an acceptable resolution** for any combo `compiler.py` can run at the Node level — the North Star ("any neograph structure that compiles to LangGraph must be representable in Agent Spec — anything less is poor design") forecloses that option the same way it foreclosed Option B's fail-loud for placeholder mismatches. The fix must be **compositional and structurally shared, at both the Node level and the Construct level**, not a hand-mirrored copy — `s7zt3.3`'s note is explicit: "mirroring it once by hand is insufficient — that's how it drifted the first time."

---

## 1. The shared decomposition table

### 1.1 Exact shape

```python
# src/neograph/modifiers.py (extends the existing single-source-of-truth module)

@dataclass(frozen=True)
class ComboDecomposition:
    """Describes how a ModifierCombo decomposes into a primary body-shape
    plus an optional orthogonal Operator wrapper. This is the ONE place
    that answers "what does this combo mean" — compiler.py and
    _agent_spec.py both consult it; neither re-derives it.
    """
    primary: PrimaryShape          # BARE | EACH | ORACLE | LOOP | PORTAL
    has_operator: bool             # True for every *_OPERATOR combo


class PrimaryShape(Enum):
    BARE = auto()
    EACH = auto()
    ORACLE = auto()
    LOOP = auto()
    PORTAL = auto()


COMBO_DECOMPOSITION: dict[ModifierCombo, ComboDecomposition] = {
    ModifierCombo.BARE:                  ComboDecomposition(PrimaryShape.BARE,   False),
    ModifierCombo.EACH:                  ComboDecomposition(PrimaryShape.EACH,   False),
    ModifierCombo.ORACLE:                ComboDecomposition(PrimaryShape.ORACLE, False),
    ModifierCombo.LOOP:                  ComboDecomposition(PrimaryShape.LOOP,   False),
    ModifierCombo.OPERATOR:              ComboDecomposition(PrimaryShape.BARE,   True),
    ModifierCombo.PORTAL:                ComboDecomposition(PrimaryShape.PORTAL, False),
    ModifierCombo.EACH_ORACLE:           ComboDecomposition(PrimaryShape.EACH,   False),  # fused, see 2.1
    ModifierCombo.EACH_OPERATOR:         ComboDecomposition(PrimaryShape.EACH,   True),
    ModifierCombo.ORACLE_OPERATOR:       ComboDecomposition(PrimaryShape.ORACLE, True),
    ModifierCombo.LOOP_OPERATOR:         ComboDecomposition(PrimaryShape.LOOP,   True),
    ModifierCombo.EACH_ORACLE_OPERATOR:  ComboDecomposition(PrimaryShape.EACH,   True),   # fused + operator
    ModifierCombo.PORTAL_OPERATOR:       ComboDecomposition(PrimaryShape.PORTAL, True),
}
```

This is a **total function over `ModifierCombo`** (all 12 keys present — a `frozenset(COMBO_DECOMPOSITION) == set(ModifierCombo)` assertion is the module-level partition guard, the same shape as `_COMBO_MAP`'s existing exhaustiveness). `EACH_ORACLE`/`EACH_ORACLE_OPERATOR` are flagged `primary=EACH` with a comment because `compiler.py`'s Arm 1 fuses them (`_add_each_oracle_fused`) rather than nesting — see §2.1 for why this is a real special case, not a table gap.

### 1.2 Where it lives

**Extends `modifiers.py`.** `ModifierCombo` and `_COMBO_MAP` already live there as the single source of truth for combo *classification* (which combo a given modifier set is). `COMBO_DECOMPOSITION` is the single source of truth for combo *meaning* (what body-shape + wrapper it decomposes to) — a natural, adjacent addition, not a new module. Putting it anywhere else (e.g. a new `_combo_decomposition.py`) would recreate exactly the "two things that should be one" problem this rewrite exists to close, just moved one file over. `modifiers.py` is already imported by both `compiler.py` and `_agent_spec.py` (both need `ModifierCombo`/`classify_modifiers` today), so no new import edge is introduced.

### 1.3 Single-writer discipline

`COMBO_DECOMPOSITION` is a **module-level frozen dict literal**, authored once, in `modifiers.py`, alongside `_COMBO_MAP`. Neither `compiler.py` nor `_agent_spec.py` writes to it — both are read-only consumers. This mirrors the **PORTAL exception precedent** (`Node.handoff_param`/`Node.handoff_channel` written only by `_ir_normalize.py`) in spirit: a genuinely-new shared capability gets exactly one writer and every consumer reads the same instance. Unlike the PORTAL fields (which are per-`Node` runtime data), `COMBO_DECOMPOSITION` is static enum-keyed metadata — closer in shape to `_COMBO_MAP` itself, so "single writer" here means "single *definition site*," pinned by a guard test rather than a runtime write-path guard.

### 1.4 CLAUDE.md layer-discipline clearance (verified, not assumed)

CLAUDE.md's off-limits list for @node-layer features is verified verbatim: **"`node.py`, `construct.py`, `_construct_validation.py`, `factory.py`, `modifiers.py` are off-limits for @node-layer features."** This list is about *@node-decorator-layer* features leaking into IR modules — it is not a blanket freeze on those modules for all time. Two things clear this rewrite:

1. **`compiler.py` is not on the off-limits list at all.** It sits one layer below (IR → Compiler in the stated pipeline: `User code → @node/ForwardConstruct/Node|Modifier → Construct → compile() → factory → LangGraph`). `compiler.py` is explicitly a sanctioned site for new IR-consuming logic — the PORTAL exception precedent already added `_add_portal_mesh`/`_add_portal_dispatch` there. Changing `compiler.py`'s match arms to *read* `COMBO_DECOMPOSITION` instead of hand-encoding the same facts is a refactor of an already-modifiable module, not a boundary violation.
2. **`modifiers.py` IS on the off-limits list, but the restriction is "off-limits for @node-layer features."** Adding `COMBO_DECOMPOSITION` is not a @node-layer feature — it is IR-level metadata describing IR-level enum semantics, the same category `ModifierCombo`/`_COMBO_MAP` already occupy. This is squarely inside what `modifiers.py` is *for*, not an exception to the rule barring it.

No CLAUDE.md rule needs to be waived; the rule doesn't apply to this change.

### 1.5 Two dispatch surfaces, one table each side of a second, narrower shared set

`compiler.py` has two `ModifierCombo`-driven functions, and they answer two different questions:

- **`_add_node_to_graph`** (Node-level): "how does THIS Node's own modifier combo lower?" — every one of the 12 combos is meaningful here (Operator composes onto any of the 5 primary shapes; `EACH_ORACLE`/`EACH_ORACLE_OPERATOR` get a real fused M×N lowering because a Node's own `map_over`/`ensemble_n` fields carry everything the fusion needs; `PORTAL`/`PORTAL_OPERATOR` are unreachable here by construction — Portal membership is resolved by the mesh pre-pass before per-node dispatch runs, so this function's `PORTAL` arm is defense-in-depth, asserting unreachability rather than lowering anything).
- **`_add_subgraph`** (Construct-level): "is a Construct, used as one item inside another Construct, ALLOWED to carry this modifier combo, and if so how does it lower?" — a strict subset of the 12. `EACH_ORACLE`/`EACH_ORACLE_OPERATOR` are unconditionally rejected here (`CompileError`, `compiler.py:511-516`) because the fusion has no meaning without a single Node's `map_over`/`ensemble_n` fields. `PORTAL`/`PORTAL_OPERATOR` are unconditionally rejected here too (a Portal mesh member must be a bare Node, never a Construct). Every other combo lowers the same way conceptually as the Node-level case, just wrapping a sub-`Flow`/subgraph instead of a leaf step.

`COMBO_DECOMPOSITION` (§1.1) answers the first question — it is a total, universal function over all 12 combos and is used, unconditionally, by both `_add_node_to_graph` and the new `_agent_spec.py`'s per-Node dispatch. The second question needs its own, much smaller, explicit set — also living in `modifiers.py`, next to `COMBO_DECOMPOSITION`, same single-writer discipline:

```python
# src/neograph/modifiers.py

SUB_CONSTRUCT_UNSUPPORTED_COMBOS: frozenset[ModifierCombo] = frozenset(
    {ModifierCombo.EACH_ORACLE, ModifierCombo.EACH_ORACLE_OPERATOR}
)
"""ModifierCombo values that are meaningful on a bare Node but have no
defined lowering when the SAME combo is attached to a Construct used as one
item inside another Construct. Consulted by BOTH compiler.py's
_add_subgraph and _agent_spec.py's Construct-item handling before any
Construct-level lowering is attempted -- checked FIRST, unconditionally.
Portal combos are excluded from this set deliberately: a Portal mesh member
being a Construct is already impossible by construction (mesh membership
requires a bare Node), so there is nothing for this set to reject for
PORTAL/PORTAL_OPERATOR -- that rejection lives in the mesh-detection
pre-pass itself, not here."""
```

`compiler.py`'s `_add_subgraph` becomes: check `combo in SUB_CONSTRUCT_UNSUPPORTED_COMBOS` first (raise `CompileError` naming the combo if so), otherwise proceed with the same `COMBO_DECOMPOSITION`-driven dispatch as the Node-level case, applied to the sub-`Flow`/subgraph shape instead of a leaf step:

```python
# compiler.py, sketch -- _add_node_to_graph (Node-level, all 12 combos meaningful)
combo, mods = classify_modifiers(item)
decomp = COMBO_DECOMPOSITION[combo]
match decomp.primary:
    case PrimaryShape.ORACLE:
        last_name = _wire_oracle(graph, name, ..., mods["oracle"], prev_node)
    case PrimaryShape.EACH:
        if "oracle" in mods:           # EACH_ORACLE / EACH_ORACLE_OPERATOR fusion
            last_name = _add_each_oracle_fused(graph, node, mods["each"], mods["oracle"], prev_node)
        else:
            last_name = _wire_each(graph, name, ..., mods["each"], prev_node)
    case PrimaryShape.LOOP:
        last_name = _add_loop_back_edge(graph, name, ..., mods["loop"])
    case PrimaryShape.PORTAL:
        assert False, "unreachable -- Portal membership resolved by mesh pre-pass"
    case PrimaryShape.BARE:
        last_name = _add_plain_or_agent_node(graph, name, node, prev_node)
if decomp.has_operator:
    last_name = _add_operator_check(graph, last_name, mods["operator"], condition_lookup=...)
```

```python
# compiler.py, sketch -- _add_subgraph (Construct-level, narrower)
combo, mods = classify_modifiers(item)
if combo in SUB_CONSTRUCT_UNSUPPORTED_COMBOS:
    raise CompileError.build(
        f"Construct {item.name!r} has modifier combination {combo.name} -- "
        "not supported when applied to a sub-Construct",
        expected="a combo in ModifierCombo - SUB_CONSTRUCT_UNSUPPORTED_COMBOS",
        found=combo.name,
    )
decomp = COMBO_DECOMPOSITION[combo]
match decomp.primary:
    case PrimaryShape.PORTAL:
        assert False, "unreachable -- a Portal mesh member must be a bare Node"
    # ... remaining arms wrap a subgraph instead of a leaf step; same decomp-driven shape
if decomp.has_operator:
    last_name = _add_operator_check(graph, last_name, mods["operator"], condition_lookup=...)
```

The **structural guard** (§1.7) pins that both match statements' case sets are exactly `set(PrimaryShape)`, that `has_operator` postludes are unconditional across all arms in both, and that `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` is checked before any Construct-level dispatch — so a future modifier addition that updates `ModifierCombo`/`COMBO_DECOMPOSITION` without updating either match, or without deciding its Construct-level validity, fails loud immediately.

> **STATUS (2026-07-28, `neograph-tjpn4`).** The prescription in §1.5-§1.7 that
> `has_operator` postludes are **unconditional across all arms** is the Phase 7 end
> state (`neograph-s7zt3.10`), NOT what is landed today. As of `neograph-tjpn4` the
> landed `_agent_spec.py` shape dispatches on `COMBO_DECOMPOSITION[combo].primary`
> with five unguarded arms + `assert_never`, but implements the Operator postlude
> **only on the BARE arm**; every other arm raises for its `*_OPERATOR` combo via a
> single-sited `NoReturn` helper. Deciding what the five fusion combos MEAN is
> Phase 7's job, so implementing the unconditional postlude earlier would have
> silently executed that phase. The §1.7 guard clause pinning unconditional
> postludes must therefore be written WITH Phase 7, not before it — against this
> code it would fail by design. The prescription below is deliberately left
> unedited: it remains the target.

### 1.6 How the new `_agent_spec.py` derives its dispatch from the same tables — including fixing a gap that exists today

`_lower_construct_item` (rewritten) performs the **identical two-step lookup** `_add_node_to_graph` does, for a bare `Node` item: `classify_modifiers(item)` → `COMBO_DECOMPOSITION[combo]` → dispatch on `.primary`, then unconditionally check `.has_operator` and wrap with `_lower_operator` (unchanged — already generic over "the node/subgraph that came before it"). The **only** thing that differs from `compiler.py`'s consumption is the *target primitive* each `PrimaryShape` case builds (LangGraph `Send`/`Command`/conditional-edges vs pyagentspec `MapNode`/`LlmNode`-ensemble/`BranchingNode`/`Swarm`) — never the *decomposition decision*.

For a `Construct` item, the rewrite fixes a gap that exists in `_agent_spec.py` **today**: the current `isinstance(item, Construct)` branch wraps the sub-`Flow` into a `FlowNode` and returns immediately — `classify_modifiers(item)` is never called on a Construct-as-item at all, so a `Construct(...) | Each() | Oracle()` used as one item inside another Construct currently has its modifiers silently ignored on export (they are legal IR — `_construct_validation.py` does not reject building it — so this is a real, currently-live gap, not a hypothetical one). The rewrite adds the same `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` check `_add_subgraph` performs, in the same order: classify the Construct item's modifiers, reject if the combo is in `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` (mirroring `compiler.py`'s own `CompileError`, as a `ConfigurationError` naming the combo), otherwise wrap the `FlowNode` and apply the `.has_operator` pause composite around it exactly as the Node-level case does around a leaf step.

### 1.7 The structural guard (mandatory, ships with the rewrite, not deferred)

Three guard tests, extending the existing structural-guard suite (`tests/test_guards_*.py`):

1. **Partition guard** (`modifiers.py`-local): `assert frozenset(COMBO_DECOMPOSITION) == frozenset(ModifierCombo)` — every enum value has a decomposition, no orphans. Already-precedented shape (`_COMBO_MAP`'s own exhaustiveness, `test_modifier_combo_axis_is_a_loud_partition` in the matrix). A second assertion pins `SUB_CONSTRUCT_UNSUPPORTED_COMBOS <= frozenset(ModifierCombo)` (a subset, not a partition — most combos ARE Construct-level valid).
2. **Cross-module derivation guard** (new, `tests/test_guards_agent_spec_combo_decomposition.py` or folded into an existing guards file): asserts that `compiler.py`'s TWO match statements and `_agent_spec.py`'s Node-level AND Construct-level dispatch are all driven by `COMBO_DECOMPOSITION`/`SUB_CONSTRUCT_UNSUPPORTED_COMBOS` — not by independently re-listing `ModifierCombo` values. Concretely: an AST-based check (same technique as `tjupj`'s exhaustive `LlmNode`/`_make_agent` construction-site guard) that (a) all four dispatch sites import `COMBO_DECOMPOSITION`/`SUB_CONSTRUCT_UNSUPPORTED_COMBOS` from `modifiers.py`, and (b) none of the four contains a **second, hand-written frozenset/dict literal enumerating `ModifierCombo` members for dispatch purposes** (a regex/AST scan for `ModifierCombo\.[A-Z_]+` occurring inside a `case`/`if`/dict-literal context outside of `modifiers.py` itself, allowlisted only for the `match decomp.primary` / `if decomp.has_operator` / `if combo in SUB_CONSTRUCT_UNSUPPORTED_COMBOS` sites). This directly answers `s7zt3.3`'s mandate: *"whether that's literally a shared lookup table both modules import, or two independent implementations pinned equal by a guard test, is a design-pass decision"* — this spec's decision is **the shared tables, additionally pinned by a guard that forbids a second table from growing back**. A guard pinning two independently-authored tables "equal" was rejected because equality-pinning doesn't stop someone from editing both files by hand in lockstep and never actually removes the duplication (exactly what happened after `2s2o6`'s content-only fix left the structural duplication in place) — the table-plus-anti-regrowth-guard is strictly stronger.
3. **Construct-item modifier check guard** (new, small): a regression test asserting `to_agent_spec` actually raises for `Construct(...) | Each() | Oracle()` used as a sub-item — pinning the gap-fix in §1.6 so it cannot silently regress back to the current silently-ignored-modifiers behavior.

---

## 2. Per-`ModifierCombo` Agent Spec feasibility (all 12, verified against real pyagentspec source)

Reference primitives (verified from `.venv/.../pyagentspec/`, not assumed):
- `LlmNode`/`AgentNode`/`ToolNode` — leaf generation/tool steps (`text_scan` input-inference family for `LlmNode`/`Agent`, `structural` for `MapNode`, `echo` for `ToolNode`).
- `MapNode(subflow: Flow, reducers: dict[str, ReductionMethod])` — fan-out over an arbitrary `Flow`; `_get_inferred_inputs` reads `subflow.start_node.inputs` **structurally** (not a text scan), so the subflow's shape is unconstrained by `MapNode` itself.
- `ParallelFlowNode(subflows: list[Flow])` and `ParallelMapNode` — both real parallel-fan-out primitives distinct from `MapNode`'s single-subflow-many-items fan-out, both present in pyagentspec and both classified `structural` (not text-scan-coupled) in `tests/agent_spec_capabilities.py`'s `NODE_FAMILIES`. Neither is used by any current lowering, and neither is chosen over `MapNode`-wraps-a-Flow for `EACH_ORACLE`'s fusion (§2.11) — `MapNode`-wrapping more directly mirrors `compiler.py`'s own per-item-fused-generation semantics and avoids introducing a second, currently-unconsumed parallel primitive into the exporter's vocabulary.
- `BranchingNode(mapping: dict[str, str])` + `InputMessageNode(message: Optional[str])` — the generic "pause composite": any `Flow` node reachable via ordinary `ControlFlowEdge`/`DataFlowEdge` wiring can precede a `BranchingNode`, and the `BranchingNode`'s branches are just string keys — nothing pins the composite to a specific predecessor node type. This is the load-bearing fact for Operator-orthogonality: **the pause composite generalizes to wrap any body**, because it's just ordinary graph wiring, not a body-type-specific primitive.
- `Swarm(first_agent, relationships: list[tuple[Agent, Agent]], handoff: HandoffMode)` — Portal's target; `AgentNode.agent: SerializeAsAny[AgenticComponent]` accepts a `Swarm` because both `Agent` and `Swarm` are `AgenticComponent` subclasses — so an `AgentNode` can wrap a `Swarm`, meaning a `Swarm` CAN be embedded as one step inside an enclosing `Flow`.

### 2.1 `BARE` — feasible (shipped)
Lowers to `LlmNode`/`AgentNode`/`ToolNode` via `_lower_generation_step`, wired by plain `ControlFlowEdge`. No change needed.

### 2.2 `EACH` — feasible (shipped)
Lowers to `MapNode(subflow=<1-node Flow wrapping the body>)`. `MapNode`'s structural input-inference is exactly why this works cleanly for any body shape (unlike `LlmNode`'s text-scan coupling). No change needed.

### 2.3 `ORACLE` — feasible (shipped)
N variant nodes (via shared `_lower_generation_step`) + one merge node (`LlmNode` for `merge_prompt`, `ToolNode` for `merge_fn`), fan-in `DataFlowEdge`s. No change needed.

### 2.4 `LOOP` — feasible (shipped)
`BranchingNode(mapping={"continue": ..., "done": ...})` + back-edge. No change needed.

### 2.5 `OPERATOR` — feasible (shipped), and this IS the reusable wrapper
`BranchingNode(mapping={"true": PAUSE, "false": DEFAULT})` → `InputMessageNode` on the pause branch, wired after the primary body. **This composite is already, structurally, "wrap any preceding node in a pause gate"** — nothing in `_lower_operator`'s current implementation assumes the predecessor is a bare scripted/think node. The rewrite's job is to *call* `_lower_operator` uniformly after every `PrimaryShape` case (§1.6), not to redesign it.

### 2.6 `PORTAL` — feasible, but requires a genuine per-member composite (fixes `s7zt3.1`'s class of bug structurally, not ad hoc)
`Swarm(first_agent, relationships, handoff)` is the target; `Agent` per mesh member. **This already ships** (via `_lower_portal_mesh_to_swarm`), and `neograph-s7zt3.1` (raw `${var}` shipping into `Agent.system_prompt`) is **already fixed** — verified via `bd show neograph-s7zt3.1` (status CLOSED): the export path now calls `_translate_placeholders` per member exactly like every other `_make_agent` call site, stamps `_MARK_PROMPT_SPEC` on the `Agent`'s own `metadata` (proven to survive — `Agent` is a `Component` and markers round-trip via `metadata`), and the import path (`_reconstruct_swarm_mesh`) prefers the marker's `original_text` when present. **This rewrite must preserve this fix exactly as-is** — it is correct, already landed, and not in scope to redo. The rewrite's job for `PORTAL` (undecorated) is purely structural: `_lower_portal_mesh_to_swarm` becomes the `PrimaryShape.PORTAL` case in the unified dispatch (§1.6), reached via the same `COMBO_DECOMPOSITION` lookup rather than a separate pre-dispatch interception — this is what makes `PORTAL_OPERATOR` (§2.7) representable at all, since today's pre-dispatch interception is precisely why `Operator` on a mesh member is invisible to any dispatch logic.

### 2.7 `PORTAL_OPERATOR` — feasible, verified composable, this is `s7zt3.2`'s fix
**The core question**: does pyagentspec have ANY primitive to represent "this mesh member pauses for approval before/after its turn"? Verified: yes, via the same generic pause composite as §2.5, applied **per-member inside the enclosing `Flow`**, not inside the `Swarm` itself. Concretely: `Swarm` has no native pause primitive of its own (`Swarm`/`Agent` are `AgenticComponent`s, not `Flow` nodes with `BranchingNode`-style branches) — but `AgentNode.agent: SerializeAsAny[AgenticComponent]` accepts a `Swarm`, so the *enclosing* `Flow` can hold `AgentNode(agent=swarm)` followed by a `BranchingNode`/`InputMessageNode` pause — this represents "pause after the mesh's turn." For "pause **before** a specific member's turn" (arguably the closer reading of neograph's `Portal(...) + Operator(when=...)` on one member), the honest verdict is: **pyagentspec's `Swarm`/`Agent` handoff model has no interior pause point between one member's turn and the next** — `Swarm.relationships` govern which agent hands off to which, but there's no node-level hook to interpose a `BranchingNode` mid-mesh without leaving the `Swarm` abstraction. So the rewrite's concrete design is:

- **Mesh-level Operator (any member carries Operator)**: wrap the *entire* `Swarm` (via `AgentNode(agent=swarm)`) with a mesh-exit pause composite — `BranchingNode`/`InputMessageNode` after the `AgentNode`, gated by `_MARK_PORTAL_SPEC`-adjacent metadata recording *which* member's Operator condition triggered it (so import can reconstruct which member the `Operator` belongs to). This is a faithful-but-flattening lowering (same spirit as Each/Oracle's existing flattening) — the approval semantics ("pause when member X's condition is true") is preserved via marker, the mesh-exit-pause structure is a lossy-but-honest approximation of "pause somewhere in this mesh," not a silent drop.
- **Marker**: extend `_MARK_PORTAL_SPEC` with an `operator_members: dict[member_name, {"when": ...}]` field (additive, doesn't disturb existing Portal-only meshes where this key is absent/empty).
- **This replaces `s7zt3.2`'s two originally-proposed options** ("represent it somehow" vs "fail loud") with a concretely verified "represent it, mesh-exit-scoped, marker-carries-the-precise-semantics" — chosen because fail-loud is ruled out by the maintainer, and a per-member interior pause is verified **not** to exist in pyagentspec's `Swarm` model, so mesh-exit is the closest faithful approximation, not a cop-out.

### 2.8 `EACH_OPERATOR` — feasible
`MapNode(subflow=...)` wrapped by the pause composite (`BranchingNode`/`InputMessageNode`) after the `MapNode` step in the enclosing `Flow` — structurally identical to `BARE|OPERATOR`, just with a `MapNode` as the preceding node instead of an `LlmNode`. No pyagentspec constraint blocks this; `BranchingNode`'s predecessor is unconstrained by type.

### 2.9 `LOOP_OPERATOR` — feasible
`BranchingNode` (Loop's continue/done gate) → body → back-edge, then the *separate* Operator pause composite chained after the Loop's `"done"` branch. Two `BranchingNode`s in sequence (one for loop-continue, one for approval-pause) is ordinary `Flow` wiring — no primitive conflict. Verified via `BranchingNode.mapping` being a plain `dict[str,str]`, unconstrained on multiplicity per `Flow`.

### 2.10 `ORACLE_OPERATOR` — feasible
Oracle's variant-fan-out + merge structure, followed by the pause composite after the merge node — identical shape to `BARE|OPERATOR`, with the Oracle merge node as predecessor.

### 2.11 `EACH_ORACLE` — feasible at the Node level, unconditionally rejected at the Construct level, matching `compiler.py`'s own two-surface split exactly
`compiler.py`'s `_add_node_to_graph` (Node-level match) explicitly special-cases `EACH_ORACLE`/`EACH_ORACLE_OPERATOR` as a **fused M×N Send topology**, not "Each wrapping an Oracle-shaped body." `compiler.py`'s `_add_subgraph` (Construct-level match, `compiler.py:511-516`) unconditionally rejects the identical combo with `CompileError.build("Each x Oracle fusion is not supported on sub-constructs", ..., hint="Use a Node with map_over + ensemble_n instead")` — no sub-condition narrows this further; it fires for `combo in (EACH_ORACLE, EACH_ORACLE_OPERATOR)` regardless of anything else about the Construct. This means the Agent Spec lowering must mirror the **same two-surface split**, not invent broader support `compiler.py` itself doesn't have — the North Star claim is "representable if `compiler.py` can run it," and `compiler.py` explicitly does NOT run `EACH_ORACLE` on sub-constructs.

Concrete design, Node-level (feasible, implement): `MapNode(subflow=<Flow containing the N-variant Oracle fan-out+merge for ONE item>)` — i.e., nest Oracle's existing variant+merge Flow *inside* the Each `MapNode`'s subflow, since `MapNode.subflow` is structurally unconstrained. Concrete design, Construct-level (reject, implement as rejection): `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` (§1.5) contains exactly `{EACH_ORACLE, EACH_ORACLE_OPERATOR}` — `_lower_construct_item`'s Construct-item branch (§1.6) checks this set first and raises `ConfigurationError` naming the combo, mirroring `compiler.py`'s `CompileError` message shape, before attempting any lowering `compiler.py` itself would reject.

### 2.12 `EACH_ORACLE_OPERATOR` — feasible, same fusion as 2.11 plus a mesh-exit-style pause after
Same `MapNode`-wraps-per-item-Oracle-fusion as §2.11, with the Operator pause composite chained after the `MapNode` step.

### Summary: **zero of the 12 combos are genuinely infeasible.** Every combo that `compiler.py` can run has a verified pyagentspec-primitive-backed lowering. This is a materially different conclusion than the epic's original `neograph-00447` UNSUPPORTED bucket implied (which meant "not yet attempted," not "verified infeasible") — consistent with `s7zt3.3`'s finding that the original partition was a scoping convenience, not a feasibility verdict, and with the maintainer's explicit rejection of fail-loud for any compiler-representable combo. The one place this spec introduces a **narrower-than-compiler.py** lowering on purpose is §2.11/2.12 (`EACH_ORACLE*` sub-construct case), and there it narrows to *match* `compiler.py`'s own restriction, not to fall short of it — preserving "no combo `compiler.py` can run silently loses fidelity in Agent Spec" while also preserving "no combo `compiler.py` itself rejects gets a broader claim in Agent Spec than the compiler makes."

---

## 3. Placeholder translation (Option F) — preserved as a required, unmodified subsystem

`_translate_placeholders`/`_node_translation`/`_emit_input_edges`/`_MARK_PROMPT_SPEC` (the corrected, §8.5-addendum version from `agent-spec-placeholder-translation-2026-07-26.md`, shipped as `neograph-cbpyx`) is **preserved verbatim** as the sole mechanism translating neograph's `${var}`/`${var.field}` syntax to pyagentspec's `{{ flat_name }}` syntax. Every new call site the rewrite introduces (§2.7's mesh-exit pause construction touches no prompts directly, so no new site there; §2.11/2.12's nested-Oracle-inside-`MapNode` reuses `_lower_oracle`'s existing per-variant translation unchanged) must route through this exact mechanism — **not re-derive it**. The rewrite's structural guard (§1.7) is extended to also re-run `tjupj`'s exhaustive `LlmNode`/`AgentNode`/`Agent` construction-site census (`tests/test_guards_agent_spec_llm_construction_sites.py`) against the rewritten module, with the expected-paired-site count bumped to reflect any new prompt-carrying construction the rewrite adds (verified: none — §2's new combos add no new prompt-emitting construction beyond what `_lower_generation_step`/`_lower_oracle` already emit per variant).

**`neograph-s7zt3.1` (Portal mesh raw-`${var}` bug) is CONFIRMED ALREADY FIXED** — verified via `bd show neograph-s7zt3.1` (status: CLOSED, fix landed 2026-07-27): `_lower_portal_mesh_to_swarm` now calls `_translate_placeholders` per member, stamps `_MARK_PROMPT_SPEC` on the `Agent`'s `metadata`, and `_reconstruct_swarm_mesh` prefers the marker's `original_text` on import. This spec's job regarding `s7zt3.1` is **preservation, not re-fixing**: the rewrite must not regress this fix while restructuring `_lower_portal_mesh_to_swarm` into the `PrimaryShape.PORTAL` dispatch case (§2.6) — the existing per-member `_translate_placeholders` call carries over unchanged into the new dispatch shape.

---

## 4. Marker/round-trip convention

**Verdict: consolidate the *documentation* of markers as a compositional set, but do not restructure the markers themselves.** The 13 `_MARK_*` constants (`_MARK_MODE`, `_MARK_AGENT_SPEC`, `_MARK_TOOL_SPEC`, `_MARK_REMOTE_AGENT`, `_MARK_MODIFIER`, `_MARK_GROUP_ID`, `_MARK_VARIANT`, `_MARK_ORACLE_SPEC`, `_MARK_EACH_SPEC`, `_MARK_LOOP_SPEC`, `_MARK_OPERATOR_SPEC`, `_MARK_BRANCH`, `_MARK_PORTAL_SPEC`, `_MARK_PROMPT_SPEC`) each carry one modifier's or mode's round-trip data — they are already, structurally, "per-`PrimaryShape`-or-orthogonal-wrapper" markers (`_MARK_ORACLE_SPEC` for `ORACLE`, `_MARK_EACH_SPEC` for `EACH`, `_MARK_LOOP_SPEC` for `LOOP`, `_MARK_OPERATOR_SPEC` for the orthogonal wrapper, `_MARK_PORTAL_SPEC` for `PORTAL`, `_MARK_MODE`/`_MARK_AGENT_SPEC`/`_MARK_PROMPT_SPEC`/`_MARK_TOOL_SPEC` for the leaf-generation-step, `_MARK_MODIFIER`/`_MARK_GROUP_ID`/`_MARK_VARIANT`/`_MARK_BRANCH` as cross-cutting group-membership tags). This mapping is now **exactly** the `COMBO_DECOMPOSITION` structure — one marker family per `PrimaryShape`, plus one for the orthogonal Operator wrapper — so no restructuring is needed; the compositional dispatch (§1) already lines the markers up correctly by construction. The one addition is §2.7's `operator_members` field on `_MARK_PORTAL_SPEC` (additive, not a restructure). **`_MARK_REMOTE_AGENT` is dead code, confirmed, not merely possibly dead**: `_MARK_REMOTE_AGENT = "neograph/remote_agent"` (`_agent_spec.py:74`) has zero other references anywhere in `_agent_spec.py` or `loader.py`. `_REMOTE_AGENT_ENDPOINT_ATTRS` (`loader.py:277`) is an unrelated dict (endpoint-attribute-name lookup for reconstructing remote-agent Node kinds) that only shares a naming pattern with it — the two never touch each other. This should be removed as a small, separate cleanup, filed as its own ticket rather than left for the rewrite's implementer to re-discover; it is not part of this rewrite's scope and not a blocker for it.

---

## 5. Test/doc asset disposition

| File | Disposition | Reasoning |
|---|---|---|
| `tests/test_agent_spec_matrix.py` | **PRESERVED, extended to all 12 combos** | This is the epic's own acceptance target (`neograph-s7zt3.3`'s scope). Its `SUPPORTED_COMBOS`/`UNSUPPORTED_COMBOS` partition becomes obsolete under this spec — replace with a single `GENERATES_REAL_LOWERING = set(ModifierCombo)` (all 12) driven by `COMBO_DECOMPOSITION`, since §2 found zero genuinely infeasible combos. The loud-partition guard (`test_modifier_combo_axis_is_a_loud_partition`) stays, now trivially `True` (partition against a total function). |
| `tests/agent_spec_capabilities.py` | **PRESERVED unchanged** | The two-tier introspection registry (`NODE_FAMILIES` Tier A, live-pyagentspec-subclass-walk Tier B) is orthogonal to the dispatch rewrite — it classifies pyagentspec node types by inference family (`text_scan`/`structural`/`echo`), which doesn't change. `PLACEHOLDER_PROMPT_EMITTING_CONSTRUCTORS`'s derivation also survives unchanged. |
| `tests/test_agent_spec_export.py` | **Assertions on `BARE`/`EACH`/`ORACLE`/`LOOP`/`OPERATOR` survive unchanged** (these 5 combos' lowerings are not being redesigned, only re-routed through the shared table). New assertions needed for the 7 newly-real combos — these are additive, not replacing existing ones. The `TestPortalMeshMemberPromptNeverShipsRawPlaceholder` class (from `s7zt3.1`) survives unchanged — its fix is preserved verbatim (§3). |
| `tests/test_agent_spec_roundtrip.py` | **Survives unchanged for existing combos; extended for new ones** | Round-trip fidelity for `BARE`/`EACH`/`ORACLE`/`LOOP`/`OPERATOR` is untouched by this rewrite (same lowering functions, just reached via a shared table). New round-trip tests needed per §2's newly-real combos, especially `PORTAL_OPERATOR`'s marker-based `operator_members` reconstruction. |
| `tests/test_agent_spec_oracle_agent_export.py` | **Survives unchanged** | Pins Oracle+agent/act lowering (from `neograph-i7k7j`), unrelated to the `_lower_construct_item` dispatch structure being rewritten. |
| `tests/test_agent_spec_refactor_snapshot.py` | **PRESERVED, and promoted to a required Step-0/Step-1 gate** | Read in full: this file is a byte snapshot of `to_agent_spec`'s canonicalized, id-free `Flow.to_dict()` output (`_canonicalize`) for a representative cell set spanning `scripted/think/agent/act × {bare, oracle-merge_fn, oracle-merge_prompt} × {single, dict}`, diffed against a committed golden fixture — it asserts only observable EXPORT behavior, never `_lower_construct_item`'s internal branch structure. It is definitively not the kind of implementation-detail snapshot that would spuriously break under a compositional dispatch refactor, and it is exactly the enforcement mechanism the "zero behavior change" claims in §7's Steps 0-1 need — it is named there explicitly as a required gate, not merely re-verified. One limit: it covers only `BARE` and `ORACLE`, not `EACH`/`LOOP`/`OPERATOR` — necessary but not sufficient proof of zero-behavior-change across all 5 currently-shipped combos, so Steps 0-1's gate also includes the existing `test_agent_spec_export.py`/`test_agent_spec_roundtrip.py` assertions for the other 3. |
| `tests/test_agent_spec_placeholder_translation.py` | **Survives unchanged** | Pins Option F itself, which this spec explicitly preserves untouched (§3). |
| `tests/test_agent_spec_import.py` | **Survives largely unchanged; `_group_flow_items` gets a NEW recognition-then-classify design (§6), not a mechanical mirror of `_lower_construct_item`** | `loader.py`'s `_group_flow_items` (`loader.py:559-638`) contains zero references to `ModifierCombo`/`classify_modifiers` — it reconstructs pipeline shape by pattern-matching `_MARK_MODIFIER` marker strings plus `ControlFlowEdge`/`DataFlowEdge` shape evidence, the OPPOSITE direction from what `COMBO_DECOMPOSITION` (a `ModifierCombo -> shape` lookup) answers. §6 gives this its own concrete design (recognize evidence → classify via the existing `_COMBO_MAP` → cross-validate structural shape via `COMBO_DECOMPOSITION`) rather than assuming it can "consult the same table the same way" the export side does. Existing reconstruct-tests for the 5 already-supported combos survive under this design; new tests are needed for the 7 newly-real reconstruct paths (especially reconstructing `operator_members` back into per-mesh-member `Operator` modifiers for `PORTAL_OPERATOR`). |
| `tests/test_agent_spec_types.py` | **Survives unchanged** | Pins `_properties_for`/`model_to_agent_spec_properties` type-conversion behavior, orthogonal to the modifier-composition dispatch. |

---

## 6. Import-side (`loader.py`) design — recognition, not dispatch

`compiler.py` and the rewritten `_agent_spec.py` both go **forward**: given a known `ModifierCombo`, decide what to build. `loader.py`'s `_group_flow_items` goes **backward**: given an Agent Spec `Flow`'s already-built primitives, decide what `ModifierCombo` (if any) produced them. These are genuinely different problems, and `COMBO_DECOMPOSITION`'s `ModifierCombo -> (primary, has_operator)` shape is not directly invertible for the second one — `(primary, has_operator)` alone does not distinguish `EACH` from `EACH_ORACLE` (both are `(EACH, False)`; the presence of a fused Oracle body inside the `MapNode`'s subflow is the only thing that tells them apart), so "look up `COMBO_DECOMPOSITION` symmetrically" was never a coherent instruction. The correct design keeps recognition and classification as two explicit steps, sharing the SAME classification authority the export side uses, rather than re-deriving "what does this evidence mean" a second time:

1. **Recognition (unchanged from today, this logic is genuinely different work and doesn't need to change in kind)**: `_group_flow_items` walks `flow.nodes` in order, reads each primitive's `metadata` for the marker keys (`_MARK_MODIFIER`, `_MARK_GROUP_ID`, `_MARK_OPERATOR_SPEC`, etc.), and verifies the marker's claim against the actual `ControlFlowEdge`/`DataFlowEdge` shape present (e.g., a claimed Loop back-edge must actually exist). This produces, per recognized group, the SET of modifier names actually present-and-verified — a `frozenset[str]` like `{"each", "oracle"}` — exactly the same shape `classify_modifiers`' internal `mods` dict keys already have on the export side.
2. **Classification (NEW, shared)**: that `frozenset[str]` is looked up in `_COMBO_MAP` (`modifiers.py`, already the single source of truth `classify_modifiers()` itself uses) to produce the actual `ModifierCombo` — the SAME classification authority both directions consult, closing the "two independently-derived facts" risk without pretending the recognition mechanism is symmetric to export-side dispatch.
3. **Cross-validation (NEW, a real correctness improvement, not just symmetry for its own sake)**: once the `ModifierCombo` is known, look up `COMBO_DECOMPOSITION[combo]` and assert the recognized STRUCTURAL shape (which concrete pyagentspec primitive types were actually found in this group — a `MapNode`, an Oracle variant-fan-out-plus-merge shape, a `BranchingNode` continue/done pair, a `BranchingNode`+`InputMessageNode` pause pair, an `AgentNode(agent=Swarm)`) matches what `(primary, has_operator)` predicts for that combo. This is new, genuine value: today, `_group_flow_items` trusts a marker string without cross-checking that the structure it claims actually matches — this turns a malformed or foreign (non-neograph, hand-authored) Agent Spec import with an inconsistent marker into a clear, fail-loud reconstruction error instead of a silent misreconstruction.
4. **Construct-level restriction, mirrored**: if step 2 classifies a MULTI-NODE group (i.e., the marker pattern spans what would reconstruct as a sub-Construct, not a single Node) into a combo in `SUB_CONSTRUCT_UNSUPPORTED_COMBOS`, reject — defense-in-depth, since a neograph-authored export never emits this shape (the export side refuses to produce it), but a hand-authored or foreign Agent Spec artifact could claim it.
5. **Where this lives**: the recognition logic (step 1) stays in `loader.py` — it is specific to interpreting Agent-Spec-primitive evidence, not IR-level metadata, so it does not belong in `modifiers.py`. Steps 2-4 are a small new `loader.py` helper (e.g. `_classify_recognized_group(names: frozenset[str], structural_evidence) -> ModifierCombo`) that IMPORTS `_COMBO_MAP`/`COMBO_DECOMPOSITION` from `modifiers.py` as its classification/validation authority — it does not redefine either table, and the cross-module derivation guard (§1.7 #2) is extended to also assert `loader.py` imports these tables and contains no independent re-listing of `ModifierCombo` members for classification purposes.

This design should itself go through its own focused review before implementation (it is the one genuinely new piece of design in this document, verified sound in shape but not yet battle-tested against every one of the 12 combos' reconstruction paths the way §2's export-side feasibility was) — but it is no longer an open question deferred to the rewrite's implementer; it is a concrete, three-step, shared-authority design.

---

## 7. Migration/build plan (design-only outline, no execution)

Ordering principle: **land the shared table and its guard first (currently-supported combos unaffected), then extend combo-by-combo, each landing behind its own matrix-cell gate — never a long red stretch.**

1. **Step 0 — shared tables + guards, zero behavior change, TWO refactors not one.** Add `COMBO_DECOMPOSITION`/`PrimaryShape`/`SUB_CONSTRUCT_UNSUPPORTED_COMBOS` to `modifiers.py` with the partition guards (§1.7 #1). Refactor `compiler.py`'s `_add_node_to_graph` (Node-level match, 6 arms) to read `.primary`/`.has_operator` from `COMBO_DECOMPOSITION` instead of hand-encoding the same grouping, AND separately refactor `_add_subgraph` (Construct-level match, 6 arms) to check `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` first and otherwise read the same table (behavior-preserving refactor of BOTH match statements — the existing compiler test suite plus `tests/test_agent_spec_refactor_snapshot.py` (now a named required gate, §5) is the gate; zero new tests required, all existing `test_check_fixtures.py`/compiler tests must stay green). This step alone proves both tables match `compiler.py`'s real behavior, on both dispatch surfaces, before `_agent_spec.py` ever consumes either.
2. **Step 1 — `_agent_spec.py` consumes the tables for the 5 already-supported combos, zero behavior change, AND fixes the live Construct-item-modifier gap.** Rewrite `_lower_construct_item`'s Node-item branch to dispatch via `COMBO_DECOMPOSITION` for `BARE`/`EACH`/`ORACLE`/`LOOP`/`OPERATOR` — output must be byte-identical to today's for these 5 (gate: `test_agent_spec_refactor_snapshot.py` plus existing `test_agent_spec_export.py`/`test_agent_spec_roundtrip.py`/matrix cells for these 5 combos, unchanged, must stay green). Additionally fix the Construct-item branch (§1.6) to call `classify_modifiers`/check `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` before wrapping a `FlowNode` — today it does not check modifiers at all, so this is a real (if narrow) behavior CHANGE, not zero-behavior-change: it starts rejecting `Construct(...) | Each() | Oracle()` used as a sub-item, which the current code silently mis-exports (drops the modifiers). TDD this specific change with its own failing-test-first (§1.7 #3). Add the cross-module derivation guard (§1.7 #2) here — it can now assert real non-trivial structure since all four dispatch sites exist.
3. **Step 2 — Portal folds into the unified dispatch (structural only, `s7zt3.1`'s fix preserved).** Move `_lower_portal_mesh_to_swarm` to be the `PrimaryShape.PORTAL` case, reached via `classify_modifiers`/`COMBO_DECOMPOSITION` instead of the current pre-dispatch interception. Gate: existing Portal export/roundtrip tests (including the `s7zt3.1` regression test) must stay green with zero behavior change — this step is purely "where is this code reached from," not "what does it produce."
4. **Step 3 — `PORTAL_OPERATOR` (the `s7zt3.2` fix).** Implement §2.7's mesh-exit pause composite + `operator_members` marker extension. TDD: write the failing repro first (2-member mesh, one member with `Operator`) — this is the exact `s7zt3.2` repro already in beads notes — then implement. Gate: new matrix cell for `PORTAL_OPERATOR` goes GREEN; `s7zt3.2` closes.
5. **Step 4 — the 5 non-Portal composed combos (`EACH_OPERATOR`, `ORACLE_OPERATOR`, `LOOP_OPERATOR`, `EACH_ORACLE`, `EACH_ORACLE_OPERATOR`).** Each is independent (different `PrimaryShape` + wrapper combination) and can be built in any order, ideally one PR/commit per combo so each has its own red→green cycle rather than one giant change. `EACH_ORACLE`/`EACH_ORACLE_OPERATOR` (§2.11/2.12) are the trickiest (fused nesting, plus the Construct-level rejection path from Step 1) — sequence them last, after the simpler `*_OPERATOR` wrapper cases prove the "any body + pause composite" pattern generically.
6. **Step 5 — matrix + capabilities extension (`s7zt3.3`'s literal scope).** Update `tests/agent_spec_capabilities.py`'s combo classification and `tests/test_agent_spec_matrix.py`'s cell generator to treat all 12 combos as `GENERATES_REAL_LOWERING`, removing the `UNSUPPORTED_COMBOS` bucket entirely. This step is the **end-state gate**: full matrix green across all 12 combos × mode × input-shape × round-trip is the acceptance criterion for closing `s7zt3`/`s7zt3.2`/`s7zt3.3` together.
7. **Step 6 — `loader.py` import-side rebuild per §6's recognize→classify→cross-validate design.** Add the `_classify_recognized_group` helper (§6, importing `_COMBO_MAP`/`COMBO_DECOMPOSITION` from `modifiers.py`) alongside `_group_flow_items`'s existing marker-recognition walk; land reconstruct-side support in lockstep with each export-side combo from Steps 3-5 (not deferred to the end — an export-only combo with no matching import support would fail the matrix's round-trip axis). Extend the cross-module derivation guard (§1.7 #2) to cover `loader.py` once this lands.

**Incremental-without-long-red-stretch property**: Steps 0-2 are zero-behavior-change refactors gated by *existing* tests (never red), except Step 1's Construct-item-modifier fix, which is a narrow, TDD'd, intentional behavior change gated by its own new failing-test-first. Steps 3-6 each add exactly one new capability gated by its own new test, landing green before the next step starts — at no point is more than one combo's worth of new surface red at once.
