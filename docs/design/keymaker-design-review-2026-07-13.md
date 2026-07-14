> **RENAMED 2026-07-14 (neograph-1t0zh):** the construct designed here as *Keymaker* shipped as **`Portal`** — a functional name that reads for newcomers and spans both modes (peer routing + dynamic flow). This document is preserved verbatim as the historical design record; its body still uses the original *Keymaker* name. See `website/src/content/docs/concepts/portal.mdx` for the shipped API.

# Independent architect review — KEYMAKER design (dynamic-handoff-2026-07-13.md)

Reviewer: independent (did not author). Verified against `src/neograph/` at develop HEAD,
`bd show neograph-07inf`, AGENTS.md, and the research doc. Adversarial pass.

**Overall verdict: GO-WITH-CORRECTIONS.** The design is thorough, decision-complete, and
mostly well-grounded; the reserved-key, budget, fingerprint, scope-cut, and mode-(b)-reduced
choices are sound. But three claims fail against source and must be fixed BEFORE tasks are
filed, because two of them are correctness bugs (not implementation detail) and one re-introduces
the exact ts7 parity bug the design elsewhere warns against.

---

## HIGH — must fix before filing tasks

### H1. KEYMAKER+Operator ("approval gate on a hop") is unwireable via the existing postlude
Evidence: `_add_operator_check` (`_wiring.py:939-961`) appends a node and a **static** edge
`graph.add_edge(node_name, check_name)`, and the compiler runs it as a POSTLUDE after the
primary modifier returns `last_name` (`compiler.py:519-520`, `:622-623`). A KEYMAKER member's
wrapper returns `Command(goto=peer)` (design §4.1 factory row). A `Command(goto)` does not
traverse the static `member→member__operator` edge on the way to `peer` — the interrupt is
either bypassed (override semantics) or fires concurrently with the goto (additive semantics);
neither is "approve, THEN hop." Worse, `_add_keymaker_mesh` consumes the whole mesh at the
entry and returns the mesh EXIT as `last_name`, so the postlude actually appends the Operator
check after the mesh exit → the combo as wired is an approval gate at **mesh exit**, contradicting
§5/§7's "approval gate on a hop." And an Operator on a NON-entry member is silently dropped (the
walk skips subsequent members, §4.1). Fix: either (a) drop KEYMAKER_OPERATOR from v1, or (b)
redesign it to raise `interrupt()` INSIDE the keymaker wrapper before returning the Command (not
via the static-edge postlude), and rewrite §7's checkpoint-resume claim and the T2 sqlite test
accordingly. The current text and combo table assert it works "for free" — it does not.

### H2. `handoff_param` written in `_construct_builder.py` (T3) re-creates the ts7 parity bug
Evidence: the design pitches `Node.handoff_param` as "the exact `fan_out_param` parallel" (§3.3,
§4.1). But the ESTABLISHED fan_out_param invariant is that **`_ir_normalize.py` is the SOLE
writer** — `_construct_builder.py:172-177` explicitly says "fan_out_param is NOT written here —
it is owned exclusively by [the normalizer]... See neograph-k7bg," and `_ir_normalize.py:127-147`
is where it lands, precisely so all three surfaces converge. T3 (§10) instead lists
`_construct_builder.py` as the file that "wire[s] handoff_param from the param named handoff."
`_construct_builder` is the `@node`-decoration assembly path; declarative `Node(inputs={"handoff":X})`
and programmatic pipe do NOT pass through it. Wiring handoff_param there sets it for `@node` only
→ declarative/programmatic meshes fail reserved-key resolution → the exact neograph-ts7 divergence
the design cites as the thing to avoid. Fix: move the handoff_param write into the normalizer
(`_ir_normalize.py`), keyed off the presence of the reserved `handoff` inputs key (which all three
surfaces carry explicitly), and pin it with the three-surface parity test T3 already promises.
Re-scope T3's file list and G3's guard ("sole writer is the normalizer," not "assembly/decorator
code").

### H3. Command(goto) lowering buys nothing in v1 given D-MESH-LEVEL, and imports its own risk
The research's ONLY hard reasons to prefer `Command(goto)+destinations=` over the router+`path_map`
fallback are (i) cross-sub-construct mesh via `Command.PARENT` and (ii) unifying modes (a)+(b) on
one factory change (research §6, §7). In v1 BOTH are cut: D-MESH-LEVEL forbids cross-boundary peers
(§1), and mode (b) reduced uses "plain `graph.add_node` + static next edge... **no Command needed**"
(§4.2). So v1 Command(goto) delivers neither benefit while costing: a brand-new "wrapper returns
Command" capability (nothing in `src/` does this today — verified, `factory.py:76,79` all return
`dict`), a new Command-construction monopoly guard (G1), sync+async Command twins, AND it is the
direct cause of H1. The router+`path_map` form (identical to Loop, `_wiring.py:544-569`) composes
cleanly with the existing Operator postlude and gets peer-existence checking free from the static
path_map. Recommendation: either adopt router+`path_map` for the v1 same-level mesh and defer
Command(goto) to the v2 cross-boundary work (a contained migration — the mesh wiring is new code
either way), OR consciously accept the cost and fix H1. As written the doc keeps the invasive
lowering while cutting both things that justified it. At minimum this tension must be surfaced to
the maintainer, not buried. (This is why I rate D-MESH-LEVEL's "we still lower to Command(goto) so
v2 needs no re-lowering" as the weakest argument in the doc.)

---

## MEDIUM — fix during implementation, note in the task

### M1. The compile WALK, not just the dispatch arm, must become mesh-aware
`compiler.py:243-277` iterates every `construct.nodes` item and calls `_add_node_to_graph` per
node, threading `prev_node`. "`_add_keymaker_mesh` called once... subsequent members skipped by
the walk" (§4.1) requires editing this loop to detect a contiguous mesh, dispatch once, skip the
remaining members, and thread `prev_node` from the exit. The §4.1 per-layer table attributes the
compiler change only to the dispatch arms (`:549`/`:472`) and omits the walk-loop edit. Add it to
T1/T2 explicitly, or duplicate-`add_node` errors will surface at integration.

### M2. `ModifierSet.model_post_init` exclusions are HARD-CODED, not table-driven
The design says Keymaker×Each/Oracle/Loop is "rejected via `_SLOT_RULES` excludes" (§5.6). But the
table drives only `with_modifier` (the pipe path, `modifiers.py:660+`). Direct `ModifierSet(...)`
construction validates via `model_post_init` (`modifiers.py:646-658`), which is hand-coded pairwise
(each/loop, oracle/loop) and would silently ALLOW keymaker+loop unless new arms are added there too.
This is itself a three-surface-parity hazard (pipe rejects, direct-construct allows). T1 must add
model_post_init arms (or refactor it to read `_SLOT_RULES`). Also: adding the `keymaker` slot means
new fields on `ModifierSet` + the `combo` property (`:632-644`) + both `_COMBO_MAP` rows.

### M3. Five exhaustive match sites, not "compiler ×2"
`assert_never` bites at `compiler.py:516,620` AND `state.py:202,512,569` — five sites need
`KEYMAKER`/`KEYMAKER_OPERATOR` arms, plus non-exhaustive membership checks at `state.py:142,211-233`.
The §4.1 prose note ("grep all ModifierCombo matches... everywhere assert_never bites") and T1's AC
("full mypy green... at every combo match site") do cover this, but the per-layer state.py row is
framed as "add fields" and hides the three match-arm edits. Make it explicit in T1.

### M4. T2 is over-sized for one execute-molecule
T2 bundles: factory Command wrapper (sync+async), `_add_keymaker_mesh` wiring, compiler arm
replacement, runner recursion floor, full runtime test half, budget error/exit tests, silent-drop
closure, G1+G2 guards, AND a sqlite mid-mesh Operator interrupt/resume test. With H1 unresolved this
is the riskiest task. Split: T2a = mesh wiring + Command/budget/target-check + runtime tests +
G1/G2; T2b = checkpoint + (redesigned) Operator-on-hop. Keeps each write-stage one molecule
(matches the maintainer's execute-molecule rule).

---

## LOW — style / acknowledged

- **L1. `on_exhaust="exit"` diverges from Loop's `"last"`** (`modifiers.py:564`). The design claims
  parallelism but uses a different literal. Either reuse `"last"` or note the deliberate rename.
- **L2. G2 (handoff keys via StateKeys only) is largely redundant** with the existing Layer-A
  centralization guard (`test_guards_function_local_imports.py:450-464`, bans any inline `neo_`
  fragment anywhere). Fine to keep as a focused pin, but it is not net-new coverage.
- **L3. AGENTS.md mis-locates `_extract_input`.** The design inherits AGENTS.md's "`factory._extract_input`"
  phrasing (§3.3), but it lives in `_input_shape.py:133` (fan_out at `:108`); `_build_state_update`
  is in `_state_write.py:47`. Cosmetic, but implementers grep by these names — use the real paths.
- **L4. Swarm-import driver does not actually unblock in v1.** §11 correctly notes it is blocked on
  agent-member support (D-MEMBER-MODES), since Agent Spec Swarm members are Agents. Honest, but worth
  the maintainer knowing the marquee downstream benefit is still one epic away.
- **L5. "union frontier / fixpoint" oversells the mechanism.** In practice members are flat contiguous
  Nodes, so each registers its own output linearly (§5); there is no _BranchNode-style arm expansion.
  The guarantee is fine; the framing is grander than the code.

---

## JUDGMENT-CALL verdicts

- **D-SCOPE** (mode b reduced): AGREE. Deferred parts each depend on unbuilt seams (01i0g, mrb2y);
  the reduced core still demonstrates E2's rejection path.
- **D-MESH-LEVEL** (same-level only): AGREE on the cut — but see H3: it undercuts the Command(goto)
  lowering it claims to preserve. The cut is right; the lowering justification is not.
- **D-MEMBER-MODES** (no agent/act): AGREE. Sound; ReAct terminal-router plumbing is genuinely separate.
- **D-DICT-OUTPUTS** (reject on member): AGREE. Uniform single payload keeps the static check crisp.
- **D-ONE-CLASS** (one class, discriminator): AGREE. Matches the bead's pinned surface sketch and
  mirrors `Loop.model_post_init`.
- **D-FORWARD-EXEMPT**: AGREE — STRONGER than the di_inputs analogy. A runtime mesh genuinely has no
  static dataflow for the proxy tracer to thread; this is a real exemption, not a punt. Parity rule
  (which names @node/declarative/programmatic) is satisfied.
- **D-RESERVED-KEY** (`handoff` name-based): AGREE. Zero collision verified (grep of `src/` AND `tests/`
  is empty), greppable, parallel to `neo_each_item`/`fan_out_param`. A typed marker would be marginally
  safer but is inconsistent with existing precedent — keep the name.
- **D-SINGLE-MESH / D-UNIFORM-PAYLOAD**: AGREE for v1.
- **D-DISPATCH-REGISTRIES** (emitted flow uses only `Keymaker(scripted=,conditions=)`): AGREE. Sound
  capability boundary; sidesteps the global-registry-mutation concern the research flagged.

## Sanity checks that PASSED
- `handoff` collides with nothing (§3.3): confirmed, grep empty in src and tests.
- Fingerprint treatment (§7): `neo_`-prefixed fields excluded (`state.py:418`), member outputs
  fingerprint via `_type_signature`, format untouched → no gratuitous invalidation. Sane.
- Budget reuses `ExecutionError` like Loop (`_wiring.py:548-556`), no new exception class — an
  improvement over the research's "KeymakerBudgetError."
- Compiler dispatch-site line refs (node `:549`, subgraph `:472`) are correct.
