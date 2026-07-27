# Audit: fail-loud "not supported" verdicts (Portal / Agent Spec) vs real capability

Scope: every `ConfigurationError`/`CompileError` "not supported"-style raise site touching
Portal, Agent Spec export (`_agent_spec.py`), or Agent Spec import (`loader.py`), cross-checked
against current source (`compiler.py`, `_wiring.py`, `_validation_portal.py`) and the design-doc
history (`agent-spec-oracle-inputs-2026-07-25*`, `agent-spec-placeholder-translation-2026-07-26*`,
`agent-spec-rewrite-2026-07-27*`, `modifier-combo-single-source-of-truth-2026-07-27*`).

## Ledger

### 1. `to_agent_spec()` "mixes a Portal peer mesh with non-mesh nodes" — **FALSE, verified**
`src/neograph/_agent_spec.py:948-977`. `mesh_members` is computed as
`isinstance(item, Node) and item.modifier_set.portal is not None and not is_dispatch` — it only
ever counts `Node` members. `Construct` also carries `modifier_set` (`construct.py:137`) and
`do0d9` (`_validation_portal.py`'s `_check_one_mesh_group`) explicitly **admits a `Construct` as a
first-class non-entry Portal mesh member** ("do0d9 (§4 Q2): a Construct member is ADMITTED... The
former blanket `isinstance(member, Node)` rejection is relaxed"). So a homogeneous mesh containing
one Portal-carrying `Construct` member is legal IR (do0d9-admitted, has a passing fixture per
`agent-spec-rewrite-2026-07-27-review-2.md:127-153`), but `to_agent_spec()`'s mesh-membership scan
silently miscounts it as a non-mesh node, `len(mesh_members) != len(all_items)` fires, and the
construct is rejected with "mixes a Portal peer mesh with non-mesh nodes" — a diagnosis that is
factually wrong (nothing is mixed; the detector is do0d9-blind), not merely an unimplemented
lowering. **This is the single most consequential finding**: the check is a false NEGATIVE at the
gate that decides whether Swarm-export is even attempted, so it can silently misroute a legal mesh
into the Flow path or reject it outright depending on ordering, not "fail loud on a genuine gap."

### 2. `compiler.py` "Portal on a sub-construct is not supported" (`_add_subgraph`, line ~556) — **PARTIALLY FALSE, verified**
`compiler.py:551-560`. The per-item loop in `compile()` (`compiler.py:251-254`) detects a Portal
mesh **entry** only via `isinstance(item, Node) and classify_modifiers(item)[0] in (PORTAL,
PORTAL_OPERATOR)` — a `Construct` can never satisfy this, so a Portal-carrying `Construct` can
never be treated as the mesh **entry**; it falls through to `_add_subgraph`'s modifier match, hits
`ModifierCombo.PORTAL | ModifierCombo.PORTAL_OPERATOR`, and raises unconditionally with the stale
message "mesh members must be sibling Nodes (D-MESH-LEVEL)" — a rule do0d9 already relaxed. But
for a **non-entry** Construct mesh member: `_contiguous_portal_mesh` (called from a `Node` entry)
collects subsequent members regardless of type, `_add_portal_mesh` is proven to route to a
Construct member (do0d9 design doc: "an actual `Construct` object in a `Portal` member list, routed
to, running on the routed payload... before do0d9 closes"), and those members are marked in
`meshed` so the `_add_subgraph` match arm above is **never reached for them**. So: Construct-as-
non-entry mesh member — TRUE capability, compiler.py runs it; Construct-as-mesh-**entry** — TRUE
infeasible **today**, but as a genuine compiler.py bug/gap (mesh-entry detection is `isinstance`-
gated on `Node`), not a fundamental architectural limit — it is a missing `isinstance` relaxation
in the same family as do0d9's other four sites, not a re-derivation of new semantics.

### 3. `_agent_spec.py` "no Agent Spec lowering yet" for composed non-Portal combos (`_lower_construct_item`, line 872-873) — **FALSE for 5 of 5 combos, verified against compiler.py**
`_agent_spec.py:848-876` dispatches only `ORACLE`, `EACH`, `LOOP`, `OPERATOR`, `BARE` and raises
`ConfigurationError` for everything else, which catches `EACH_OPERATOR`, `ORACLE_OPERATOR`,
`LOOP_OPERATOR`, `EACH_ORACLE`, `EACH_ORACLE_OPERATOR` at the Node level. `compiler.py`'s
`_add_node_to_graph` match (verified: it has case arms for every one of the 12 `ModifierCombo`
values, per the design doc's own audit and confirmed live in source) actually runs all five of
these as legitimate LangGraph compositions — `compiler.py` is the proven reference implementation,
so "no lowering yet" is accurate only about `_agent_spec.py`'s own dispatch table, not about
whether the combination is representable/runnable. Confirmed textbook case of the systemic
duplicated-ModifierCombo-logic problem: the export layer's fail-loud message reads as "this isn't
supported" when the true state is "this isn't wired up here yet."

### 4. `compiler.py` "Each x Oracle fusion is not supported on sub-constructs" (`_add_subgraph`, line 511-516) — **TRUE-infeasible, verified, not a wiring gap**
Unlike #3, this one is genuine: `EACH_ORACLE` fusion is defined entirely in terms of a single
`Node`'s `map_over`/`ensemble_n` fields (`modifiers.py`'s `Each`/`Oracle` classes), which a
multi-node `Construct` structurally does not have — there is no Construct-level analog of
"fan out N ways, ensemble the results" to reuse; `_wiring.py`'s Node-level fused lowering
(`_add_each_oracle_fused`) has no Construct-shaped counterpart to call. This is the one Construct-
level rejection in the ledger that the design docs themselves treat as an accepted, intentional
scope boundary (`agent-spec-rewrite-2026-07-27.md:83`), not a stale check. Verdict stands as
genuinely infeasible under the current modifier model, not a "just not wired up" case.

### 5. `_agent_spec.py`'s blanket `_reject_unrepresentable_fields` handoff_param/handoff_channel rejection (line 138-148) — **MISLEADING BUT NOT LIVE-FALSE, verified as dead-in-practice**
The per-node guard called from `_lower_node`/Oracle-variant construction rejects ANY node with
`handoff_param`/`handoff_channel` set, commented "Portal mesh members cannot be exported to Agent
Spec." Taken alone this reads as a blanket claim contradicted by `_lower_portal_mesh_to_swarm`
existing at all. In practice it is not reached for genuine mesh members in the common case, because
`to_agent_spec()` intercepts all-Node meshes earlier (finding #1's code path) before any member
reaches `_lower_node`. However, because of finding #1's blind spot, a mesh containing a Construct
member does NOT get intercepted as a mesh — its Node siblings' `handoff_param`/`handoff_channel`
ARE still set, so if such a mixed-type mesh ever reached `_lower_construct_item`/`_lower_node`
(e.g. via a future fix to #1 that reclassifies it as "mixed" and tries the Flow path anyway), THIS
guard would fire and reject with a message that is accurate for a genuinely orphaned handoff node
but misleading as a description of Portal-export capability in general. Flag as **UNCERTAIN /
needs-cross-check** for the exact reachability of this path post-fix-of-#1, since fixing #1
changes what can reach this guard.

### 6. Placeholder-mismatch fail-loud, Option B (`agent-spec-placeholder-translation-2026-07-26.md`) — **RESOLVED historical FALSE claim, already fixed, cite as precedent**
Earlier proposal (Option B) would have made any `${var}` prompt with real upstream inputs
permanently unexportable ("cannot be expressed") on the theory that pyagentspec's `{{ jinja }}`
grammar and neograph's `${var}` grammar are incompatible. Verified resolved: Option F (translate,
don't reject) shipped — `_translate_placeholders` is called from every `_make_agent`/lowering site,
markers round-trip via `Component.metadata`, and `neograph-s7zt3.1` (raw `${var}` shipping into
`Agent.system_prompt`) is closed. Cited here because it is the clearest precedent that a "cannot be
represented" verdict in this codebase's Agent Spec layer has repeatedly turned out to be "the two
grammars are trivially inter-translatable and no one wrote the translator yet," reinforcing that
findings #1 and #3 above fit an established pattern rather than being isolated.

### 7. Oracle + agent/act mode — **explicitly out-of-scope, not a resolved verdict either way**
`agent-spec-oracle-inputs-2026-07-25.md:146`: "`agent`/`act` → reject (Oracle+agent/act is not
today's failure and can stay out of scope / filed separately — no test cell exercises it)." This is
a deliberate scope deferral, not a tested/verified infeasibility claim in either direction. Left as
**UNCERTAIN** — not audited further here; flagging so it isn't mistaken for a settled verdict.

## Summary table

| # | Site | Claim | Verdict |
|---|------|-------|---------|
| 1 | `_agent_spec.py:948-977` to_agent_spec mesh detection | "mixes mesh with non-mesh nodes" | **FALSE** — do0d9-blind detector, most consequential |
| 2 | `compiler.py:556` Portal-on-subconstruct | "not supported" | **PARTIALLY FALSE** — true only for mesh-entry (real bug, not fundamental limit); false for non-entry |
| 3 | `_agent_spec.py:872-873` composed combo dispatch | "no Agent Spec lowering yet" | **FALSE** for all 5 non-Portal composed combos — compiler.py runs them |
| 4 | `compiler.py:511-516` Each×Oracle on subconstruct | "not supported" | **TRUE-infeasible** — no Construct-level analog exists |
| 5 | `_agent_spec.py:138-148` handoff field rejection | "Portal mesh members cannot be exported" | **UNCERTAIN** — reachability depends on fix to #1 |
| 6 | placeholder Option B (historical) | "cannot be expressed" | **RESOLVED-FALSE** — fixed via Option F translation |
| 7 | Oracle+agent/act | (deferred, not asserted) | **UNCERTAIN / out of scope**, not audited |
