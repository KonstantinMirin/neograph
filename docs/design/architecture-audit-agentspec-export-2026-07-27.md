# Architecture audit: `_agent_spec.py` export-side inventory (2026-07-27)

Scope: complete cell-by-cell inventory of `to_agent_spec()`'s current export
behavior, cross-referenced against `tests/test_agent_spec_matrix.py` and
`tests/agent_spec_capabilities.py`. Read the whole file
(`src/neograph/_agent_spec.py`, 1155 lines) plus the matrix test, the
`ModifierCombo` enum (`modifiers.py`), and `compiler.py`'s dispatch to check
whether "not supported" verdicts reflect genuine infeasibility or an unwired
existing capability. Companion angle to
`docs/design/modifier-combo-single-source-of-truth-2026-07-27.md` — read that
first; this file is the export-specific drill-down, independently verified
against current `develop` HEAD source, not re-derived from it blindly.

## 1. The per-node `ModifierCombo` axis — the central finding

`_lower_construct_item` (line 793) dispatches on `classify_modifiers(item)`.
It has exactly 5 handled arms — `BARE`, `EACH`, `ORACLE`, `LOOP`, `OPERATOR`
(lines 846–870) — and one catch-all `else: raise ConfigurationError` (872–877):

```
raise ConfigurationError.build(
    f"node {item.name!r} has modifier combination {combo.name} — no Agent Spec lowering yet",
    expected="BARE, ORACLE, EACH, LOOP, or OPERATOR",
    found=combo.name,
    hint="composed modifier lowering (e.g. Each+Oracle) is out of scope for i3zsh's primitive-level export",
)
```

`ModifierCombo` (`modifiers.py:65-84`) has **12** members. The 7 NOT handled:
`PORTAL`, `EACH_ORACLE`, `EACH_OPERATOR`, `ORACLE_OPERATOR`, `LOOP_OPERATOR`,
`EACH_ORACLE_OPERATOR`, `PORTAL_OPERATOR`.

**Cross-check against `compiler.py` (the proven-correct reference)**:
`compiler.py`'s two dispatch `match` blocks (lines ~505-560 and ~590-670) each
have an explicit `case` arm for **every one** of these combos:
- `EACH_ORACLE | EACH_ORACLE_OPERATOR` (lines 510, 596)
- `ORACLE | ORACLE_OPERATOR` (518, 608)
- `EACH | EACH_OPERATOR` (537, 619)
- `LOOP | LOOP_OPERATOR` (541, 630)
- `BARE | OPERATOR` (544, 642)
- `PORTAL | PORTAL_OPERATOR` (552, 665)

**Verdict: this is exactly the "capability exists, just not wired to
export" pattern the investigation is looking for.** All 5 fusion combos
(`EACH_ORACLE`, `EACH_OPERATOR`, `ORACLE_OPERATOR`, `LOOP_OPERATOR`,
`EACH_ORACLE_OPERATOR`) compile to real, running LangGraph pipelines today —
`compiler.py` has dedicated lowering for each — but `to_agent_spec()` rejects
every one of them with a generic message that even mischaracterizes `PORTAL`
(which is not a "composed modifier" fusion at all, just an unhandled arm) as
if it were the same kind of gap as `EACH_ORACLE`. Flag for cross-check phase:
confirm with `verify-compiler-arms`/`verify-meta-question` whether these 5
fusion combos are genuinely reachable via all three authoring surfaces (the
`test_example_agent_spec_each_oracle_loop.py` regression test explicitly
builds Each, Oracle, and Loop as **three separate panels**, never fused on one
node — i.e. there is no existing test proving today's `compiler.py` fusion
lowering is even exercised beyond unit level, but the `match` arms are real
code, not stubs).

**`PORTAL` (dispatch-mode, `is_dispatch=True`) member**: not part of the
mesh-export path (see §2) since it's filtered out of `mesh_members`; falls
into the generic per-item loop, hits `classify_modifiers` → `PORTAL` (or
`PORTAL_OPERATOR`) → the same catch-all raise. So a dispatch-mode Portal node
mixed into an otherwise-plain pipeline is fail-loud today, but via the wrong
error message (implies "composed modifier", when the real gap is "Portal
dispatch nodes have no Flow-node lowering at all").

## 2. Portal mesh export (`_lower_portal_mesh_to_swarm`, line 880) — a silent-loss finding

`to_agent_spec` (line 948) collects `mesh_members` as every item where
`portal is not None and not portal.is_dispatch` (960-967). If **any** mesh
members exist, and they are not the WHOLE construct, it fail-louds
("mixes a Portal peer mesh with non-mesh nodes", 970-976) — correctly
conservative pending a real answer for a mixed mesh+Flow shape (flag as a
genuine open question, not obviously wrong: `Swarm` is pyagentspec's own
top-level `AgenticComponent`, distinct from `Flow`, so a "FlowNode wrapping a
Swarm" representability question is a real one to resolve, not an
unwired-capability call).

**But**: if the mesh IS the whole construct, `_lower_portal_mesh_to_swarm`
builds one pyagentspec `Agent` per member via `_make_agent` (lines 914-924)
and **never inspects `member.modifier_set.operator` at all**. A mesh member
classified `PORTAL_OPERATOR` (Portal + Operator combo — a legitimate,
`compiler.py`-supported combo, human-approval gate on the dynamic path per
CLAUDE.md's own description of `PORTAL_OPERATOR`) exports as a bare `Agent`
with **no trace of the Operator HITL gate** — not a `ConfigurationError`, a
**silently wrong** export: the resulting Swarm looks fully exportable and
"passes," but the interrupt-when condition that governs the real running
pipeline is dropped entirely. This cell is **not in the test matrix** (the
matrix's `SUPPORTED_COMBOS`/`UNSUPPORTED_COMBOS` partition classifies
`PORTAL_OPERATOR` as `UNSUPPORTED` — i.e. the matrix assumes it never reaches
`_lower_construct_item`'s per-node dispatch — but it never tests the
mesh-export path for this combo either, so the silent drop is untested in
both directions).

## 3. The Oracle-variant guard-skip — a second silent-loss finding

`_lower_node` (line 377) is the ONLY caller of `_reject_unrepresentable_fields`
(line 111) — the guard that fail-louds on `raw_fn`, `skip_when`/`skip_value`,
`renderer`, `handoff_param`/`handoff_channel`, and callable `gate_tools_when`.

Every combo arm in `_lower_construct_item` routes its primary/body node
through `_lower_node` (and therefore the guard) **except Oracle**:
`_lower_oracle`'s per-variant loop (lines 521-543) calls
`_lower_generation_step` **directly**, bypassing `_lower_node` and therefore
bypassing `_reject_unrepresentable_fields` entirely. `_lower_generation_step`'s
own scripted/raw fallback comment (line 367: `"scripted / raw already
rejected raw_fn upstream"`) is **only true for the `_lower_node` call path** —
false for the Oracle path.

Concretely: an Oracle-modified node with `raw_fn` set (mode forced to `'raw'`
by the framework's own mode rules) hits the final branch of
`_lower_generation_step` (367-374) and is silently exported as a name-only
`ToolNode` (`_make_server_tool(node, ...)` uses `node.scripted_fn or
node.name`) — the actual Python callable is dropped with **no error at all**.
Same silent loss applies to an Oracle variant with `skip_when`/`skip_value`,
a custom `renderer`, or Portal `handoff_param`/`handoff_channel` set (the last
is likely unreachable in practice since Portal and Oracle combos are disjoint
per `_COMBO_MAP`, but `skip_when`/`renderer`/`raw_fn` are all independently
settable on any `Node` regardless of modifier).

**Confirmed untested**: `tests/test_agent_spec_export.py:256`
(`test_raw_fn_node_is_rejected`) builds a **BARE** node with `raw_fn` — the
guarded path. No test builds an Oracle-modified node with `raw_fn`/`skip_when`
/`renderer` set, so this silent-loss gap is invisible to the current suite.

## 4. Dict-form (multi-output) producer referenced by a downstream dict-form input — explicit reject, likely unwired

In `to_agent_spec`'s edge-wiring sweep (lines 1093-1102 and, for the
single-type fallback, 1115-1117), any upstream `Node.outputs` that is
dict-form (multi-output, `{key: type}`) referenced by a downstream dict-form
`inputs` entry is rejected:

```
raise ConfigurationError.build(
    f"node {item.name!r}'s dict-form inputs references upstream "
    f"{upstream_name!r}, whose outputs are not a single exportable type",
    ...
    hint="multi-output (dict-form outputs) producers referenced by a "
    "downstream dict-form input have no Agent Spec representation yet",
)
```

Dict-form `Node.outputs` is a fully first-class, compiler-supported IR
feature (CLAUDE.md: "N named outputs" — one state field per output key,
`{node_name}_{key}` naming, referenced downstream via `inputs={"{upstream}_
{key}": type}`). Nothing in `_properties_for` (line 275) is structurally
incapable of emitting one `Property` per dict key with a `{key}.{field}`-style
prefix (it already does exactly this for dict-form `Node.inputs`, lines
283-290) — the gap looks like "downstream consumer side of dict-form
*outputs* was never wired into the edge-emission sweep," not a genuine Agent
Spec representability wall. Flag as a **high-confidence candidate** for
cross-check: this is the same shape of gap as §1 (an unwired existing
capability), just discovered independently rather than via the ModifierCombo
enum.

## 5. `tests/test_agent_spec_matrix.py` coverage vs. this inventory

The matrix (`CELLS`, mechanically generated from `MODES × SUPPORTED_COMBOS ×
{oracle configs} × {input shapes}`) is **deliberately restricted to
`SUPPORTED_COMBOS = {BARE, EACH, ORACLE, LOOP, OPERATOR}`**
(`UNSUPPORTED_COMBOS` = the same 7 combos this audit's §1 identifies, listed
explicitly at lines 96-105 of the matrix file). Its own docstring says so:
"UNSUPPORTED = PORTAL ... + every fusion combo that raises
ConfigurationError" — i.e. the matrix ratifies today's reject-everything-fused
behavior as ground truth rather than testing whether it's a genuine limit.
`RED_EXPORT` is currently **empty** — every generated (supported-combo) cell
is either `GREEN` (exports+round-trips) or `UNREPRESENTABLE` (neograph's own
assembly validator rejects the construct before export is reached — e.g.
agent/act mode Each/Loop over >1 upstream input, a real "can't write it"
case, not an export gap). So: **within the 5-combo world the matrix tests,
export is fully green**; **outside it (7 unhandled combos, dict-form-output
fan-in, Oracle-variant guard-skip, mesh+Operator silent drop), there is zero
matrix coverage** and 3 of the 4 gaps found here are not fail-loud — they are
either silently wrong (§2 mesh+Operator, §3 Oracle+raw_fn/skip_when/renderer)
or reject with a misleading message (§1's generic "composed modifier... out
of scope" hint applied to plain `PORTAL`).

`tests/agent_spec_capabilities.py` is a different axis entirely — it
registers/classifies pyagentspec's own concrete `flows.nodes.Node` subclasses
(`assert_registry_complete()`, consumed by `test_pyagentspec_registry_is_complete`
and an AST guard) to ensure every *pyagentspec primitive* is accounted for
somewhere in the export/import code. It says nothing about neograph's
`ModifierCombo` coverage — the two completeness checks are orthogonal axes,
both necessary, neither sufficient alone.

## 6. Full genuinely-permanent fail-loud list (Core Invariant, by design)

These are stated in the module docstring and confirmed by reading
`_reject_unrepresentable_fields` + the two Oracle-merge-hook / callable-Loop
checks — no cross-check flag, these look like real, permanent Agent Spec gaps
(callable-valued fields have no serialization target in any spec):
- `raw_fn` (any Python callable body) — `_reject_unrepresentable_fields` (117-123), guarded only via `_lower_node` (see §3 for the bypass).
- `skip_when`/`skip_value` (124-130) — same bypass caveat.
- `renderer` (131-137) — same bypass caveat.
- Portal `handoff_param`/`handoff_channel` (138-146) — dispatch-mode-specific; peer-mesh members are exempted (they go through `_lower_portal_mesh_to_swarm`, not this guard, correctly per the module's Swarm design).
- callable `gate_tools_when` (147-153).
- Oracle `merge_pre_process`/`merge_post_process`/`merge_fallback` (508-515, checked directly in `_lower_oracle`, unconditionally — this one is NOT bypassed since it's checked before the variant loop).
- callable `Loop.when` (689-695, checked in `_lower_loop`, itself only called from the LOOP arm of `_lower_construct_item`, i.e. always reached for a Loop node).

## Summary table

| Combo / field | Export verdict | Message quoted | Compiler supports it? | Cross-check flag |
|---|---|---|---|---|
| BARE/EACH/ORACLE/LOOP/OPERATOR (single, non-fused) | GREEN (matrix-proven) | n/a | yes | none — genuinely solid |
| EACH_ORACLE, EACH_OPERATOR, ORACLE_OPERATOR, LOOP_OPERATOR, EACH_ORACLE_OPERATOR | fail-loud, generic message | `"has modifier combination {combo.name} — no Agent Spec lowering yet"` | **yes** (compiler.py has explicit `case` arms for all 5) | **HIGH — likely unwired, not infeasible** |
| PORTAL (dispatch mode) | fail-loud, same generic message (misleading — not a fusion) | same as above | yes (`compiler.py` PORTAL arm) | **HIGH — message is wrong AND capability likely unwired** |
| PORTAL_OPERATOR (peer-mesh) | **no error — silently drops Operator gate** | n/a | yes | **HIGH — silent information loss, untested in either direction** |
| Oracle variant + `raw_fn`/`skip_when`/`renderer` | **no error — silently exports name-only stub / drops field** | n/a | yes (compiles fine outside Oracle too) | **HIGH — guard-skip bug, not a spec limitation** |
| dict-form-outputs producer -> dict-form-inputs consumer | fail-loud | `"...whose outputs are not a single exportable type"` / `"...have no Agent Spec representation yet"` | yes (core IR feature) | **MEDIUM-HIGH — looks unwired, `_properties_for` already has the dict-key-prefix machinery for the input side** |
| mixed Portal-mesh + non-mesh construct | fail-loud | `"mixes a Portal peer mesh with non-mesh nodes"` | n/a (structural, not lowering) | LOW — plausible genuine Swarm-vs-Flow type-shape limit, worth a second look but not obviously wrong |
| raw_fn / skip_when / skip_value / renderer / handoff_param|channel / callable gate_tools_when / Oracle merge hooks / callable Loop.when (non-Oracle-bypass path) | fail-loud, all quoted in §6 | see §6 | n/a (callable-valued, no serialization target in any spec) | none — permanent by design |
