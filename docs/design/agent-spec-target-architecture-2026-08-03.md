# Agent Spec target architecture: one vocabulary, one bridge, one oracle

**Status**: design synthesis, 2026-08-03. Supersedes the *remaining-work* framing of
`docs/design/agent-spec-portal-master-architecture-2026-07-27.md` (its landed phases stand) and
extends `docs/design/modifier-combo-single-source-of-truth-2026-07-27.md` from one table to the
full classification layer.

**Scope**: the remaining Agent-Spec-relevant debt — neograph-jtawq.1, .2, .3, .7, .9, .10,
neograph-wq9nn, neograph-qtfof. Out of scope: jtawq.4 (decorators.py), .5 (forward.py), .6
(sync/async twins), .8 (lint.py) — no evidenced Agent-Spec coupling was found by any probe; the
sync/async twin duplication in factory.py is explicitly *not* collapsed here (see Refusals).

**Inputs**: six adversarially verified probe findings, then three independent reviews
(completeness, correctness-via-repro, consistency) whose required corrections are incorporated in
place (see the Consolidated Review Verdict, final section). Every citation below was carried from
a verified finding or re-checked this cycle; treat any "all consumers" list as provisional until
re-grepped at implementation time (the verification pass itself reproduced the missed-`_wiring.py`
failure mode once — a bad `grep -v '_portal.py'` filter silently excluded `_validation_portal.py`
and `_agent_spec_portal.py` — and only a second pass corrected it).

---

## 0. Ticket ground truth (all eight, corrected against current code)

The tickets' own descriptions have rotted. Build from this table, not from the ticket text:

| Ticket | Filed as | Actually is (verified) |
|---|---|---|
| jtawq.2 | "dispatch core redesign, 6 files incl. runner.py/factory.py" | Mostly landed via COMBO_DECOMPOSITION (modifiers.py:132-146, guard PENDING == frozenset()). Residual: fusion pre-split hand-spelled at 7+ sites in 4 structural positions; fusion membership has **two live authorities** (`is_each_oracle_fused` modifiers.py:288, `modifier_names_for_combo` :254-270) plus a loader shim (loader.py:558); `is_each_oracle_fused`/`combo_for_modifier_names` escape the consumer census. runner.py and factory.py hold **zero** combo vocabulary — strike them from the span. |
| jtawq.3 | "_SLOT_RULES single-sourcing" | Real and unstarted: **three** encodings of pair legality (`_SLOT_RULES` excludes :727-754; five hand-coded `model_post_init` arms :791-812; Portal(is_dispatch)+Operator dynamic rule duplicated verbatim at :818-822 and :857-865) because Pydantic `model_copy` skips `model_post_init` (:856-860). Five additional hand-written per-slot enumerations, incl. *both* `classify_modifiers` paths (:181-191, :199-215). The ":694-696 'ONE row'" comment is false today, and the "Unknown modifier type" expected-list (:838) is stale — it omits Portal. |
| jtawq.1 | "_wiring.py:703-1038 + factory.py:109-719" | Line refs dead (files are 817/949 lines). Routing *decision* already single-sourced (`_portal_route_to_command` factory.py:225-310, `_tool_handoff_to_command` :313-371). The suspected missing agent-cycle abstraction **already exists** (`_wire_agent_cycle_body` _wiring.py:557-681, shared by linear :684-702 and mesh :783-792). Real residuals: no member-class taxonomy (zero hits for `portal_member_class` anywhere; state.py:225-227 carries a private `_is_dispatch`); 4 copy-pasted closure-param blocks + **10** kwarg-threaded route-call sites (8× `_portal_route_to_command` factory.py:184,203,481,498,567,584,686,703; 2× `_tool_handoff_to_command` :658,:672); export/import re-derive both dispatch axes independently (`type(agent).__name__ == 'Flow'` loader.py:373; hard-coded rule :380-384). |
| jtawq.7 | "spec_types bridge extraction + guard idiom" | Primary half landed (commit 7d53427; `_agent_spec_types.py` 383 lines, `spec_types.py` 252 registry-only). Remainder: **four** (not three) copy-pasted pyagentspec import guards — loader.py:92, _agent_spec_markers.py:52-56, _agent_spec_types.py:48, **factory.py:826-833** (absent from the ticket). Plus: the string-name external-binding refusal hint is restated per module (_agent_spec_modifier_lowering.py:372, _agent_spec_node_lowering.py:77). |
| jtawq.9 | "relocate portal-dispatch without diluting G1" | Option (b) (factory.py exits G1) is impossible — 10 non-dispatch `Command(` sites remain (factory.py:280,290,303,307,350,355,363,371,423,425). Option (a) is now cheap: the Agent-Spec *gate* cluster (`_resolve_expected` :793-798, `_prepare` :800-874, `_finish` :876-889, `_check_and_increment_depth` :891-916) constructs no Command; the four dispatch `Command(` sites (:926,:931,:940,:945) stay put inside `dispatch_wrapper`/`adispatch_wrapper`. Two guards must be re-keyed in the same commit: `FUNCTION_LOCAL_IMPORT_ALLOWLIST` (tests/test_guards_sidecar_imports.py:73) and `TestPortalDispatchRoutesThroughCanonicalGate` (tests/test_guards_assembly.py:205-262) — the latter as a **two-sided** re-key, see G-GATE in section 6. |
| jtawq.10 | "swarm-import relocation, needs injection design" | Injection design **already exists**: `_construct_from_subflow(subflow, name, from_spec)` seam at _agent_spec_group_import.py:48; the other two former blockers moved to `_agent_spec_node_import.py` (:44, :187). Remaining move: loader.py:237-491, six functions, one injected callable. |
| wq9nn | "registered-name conditions don't round-trip" | **Fixed** by neograph-ijyjr (commit 110e8d6; discrimination at _agent_spec_group_import.py:279-295), verified by fresh repro this cycle. Operator half already pinned (tests/test_agent_spec_loop_operator.py:177-195, registered name). Missing: one NODE-level registered-name Loop round-trip pin. Adjacent: `_spec_loader.py:258` (native YAML path) still calls `parse_condition` unconditionally — pre-ijyjr shape. The ticket body's OPEN DECISION (v1-gating vs v2-deferred registered-name round-trip) is **mooted by the fix**; the closure note must say so explicitly so the dangling decision request does not survive the close. |
| qtfof | "Portal × Agent Spec v2 deferred features + docs" | All deps and children closed. Live scope is exactly **one item**: RemoteAgent export-side lowering. `node._remote_agent_endpoint` is written at _agent_spec_node_import.py:266-267 and has **zero readers** — an imported RemoteAgent silently re-exports as a plain scripted node, a real silent-lossiness seam under the faithful-export policy. (Also: node.py:191-193's comment cites the wrong module — it says "stashed by loader.py"; the actual writer is `_agent_spec_node_import.py`.) |

---

## 1. The single root disease

All six findings are the same disease at two altitudes:

> **A fact about the IR is re-derived locally by each consumer instead of being read from one
> declared authority — and the only oracle checking the derivations agree is another consumer
> that re-derives it leniently.**

**Altitude 1 — compile-side vocabulary fragmentation.** The Portal rollout applied single-writer
discipline to IR *fields* (G3) and, after the s7zt3 fix, to *combo decomposition*
(COMBO_DECOMPOSITION + consumer ratchet). But decomposition is only one of the classification
facts the dispatch layer consumes. The others never got an authority:

- **Fusion** ("is this combo Each∘Oracle fused?") — three spellings: `is_each_oracle_fused`
  (predicate over mods, modifiers.py:288), `modifier_names_for_combo` (table-derived,
  self-declared "the ONLY sanctioned way", modifiers.py:254-270), and loader.py:558's
  `dict.fromkeys` shim. The pre-split is hand-positioned differently at every consumer
  (guard-clause: compiler.py:548, _agent_spec.py:227, loader.py:558; case-guard: state.py:400;
  inside-arm: state.py:462, _state_write.py:90; boolean conjunction: _subconstruct.py:103). A
  second fused combo trips no `assert_never` and no guard.
- **Pair legality** ("may Portal combine with Loop?") — three encodings inside one file
  (jtawq.3 row above), kept in parity only by a comment admitting the hazard
  (modifiers.py:802-806).
- **Portal member class** ("what kind of mesh participant is this?") — re-derived by
  isinstance/mode/is_dispatch checks in **seven** src files (_wiring.py:197,748;
  _validation_portal.py ×3; _construct_validation.py:249; _agent_spec_portal.py ×2;
  compiler.py:280; modifiers.py:818,861; state.py:225-268 with a *private local helper*), plus
  loader.py deriving it a fourth way (`type(agent).__name__ == 'Flow'`, :373).
- **External-binding contract** ("a string names a Python callable the importing side must
  supply") — restated as near-copy refusal hints per lowering module; `_spec_loader.py:258`
  never joined the ijyjr discrimination at all.

**Altitude 2 — export-side oracle absence.** The m57mn → rh5fb / p7dyq / 8zvd1 / s7zt3.15 →
s7zt3.17 chain shares one cause, stated in the maintainer's own policy header
(tests/test_agent_spec_reachability.py:1-8): the exporter was validated only against neograph's
own importer, which "never executes a control edge at all, so it is blind to every defect below."
Three unchecked contract tiers, each producing its bug family: pyagentspec construction/wire
contract (8zvd1, p7dyq, commits 9b2843d/603f0f4), control-vs-data-edge semantics (rh5fb, commit
c9396d1), literal-executor semantics (s7zt3.15/.17, commits 41c9d1f/a734217 — "82 of 104 cases
were red" the moment a partial oracle appeared). The reachability BFS
(test_agent_spec_reachability.py:81-113) structurally cannot catch the .17 class — flattened
straight-line code is fully reachable.

The two altitudes are one disease because the exporter/importer are themselves *consumers of the
fragmented vocabulary*: they re-derive fusion (_agent_spec.py:227), member class
(_agent_spec_portal.py:93-104 vs loader.py:373), and the binding contract, with no authority
above them and no oracle beneath them. The master audit (s7zt3) could claim "every cell verified,
none TODO" and still be followed by five correctness bugs precisely because *verified* meant
"against the lenient importer."

---

## 2. Target architecture — three pillars

### Pillar A: complete the classification layer (`modifiers.py` becomes the IR vocabulary module, fully)

`modifiers.py` already owns two authorities (`_COMBO_MAP` = legality domain,
`COMBO_DECOMPOSITION` = meaning of legal combos). The target state is **five tables/functions,
one file, every consumer reads, no consumer re-derives**:

| Authority | Answers | Status |
|---|---|---|
| `_COMBO_MAP` (modifiers.py:87-101) | which modifier sets are legal | exists |
| `COMBO_DECOMPOSITION` (:132-146) | what a legal combo means (primary + has_operator + **new: `fused`**) | extend |
| `_CONFLICT_DIAGNOSTICS` + `_DYNAMIC_RULES` (new) | *why* illegal pairs are illegal, incl. instance-dependent rules | build (jtawq.3) |
| `_SLOT_RULES` (shrunk to roster: mod_type/slot/label) | which modifier types exist | shrink (jtawq.3) |
| `PortalMemberClass` + `portal_member_class(item)` (new) | what kind of mesh participant an item is | build (jtawq.1) |

**A1 — fusion as one authority, not a third** (jtawq.2 residual). Add `fused: bool` to
`ComboDecomposition` as a column *derived at definition* from `modifier_names_for_combo` (so the
table and the sanctioned membership function provably agree), plus a partition guard:
`COMBO_DECOMPOSITION[c].fused == ({'each','oracle'} <= modifier_names_for_combo(c))` for every
combo. Then: (a) delete loader.py:558's shim → `decomp.fused` (loader.py:547 already holds
`decomp = COMBO_DECOMPOSITION[combo]` four lines above, so this is a one-line coherent edit);
(b) rewrite the guard-clause sites (compiler.py:548, _agent_spec.py:227) to `decomp.fused`;
(c) keep `is_each_oracle_fused` only for genuine mods-in-hand reads inside EACH arms
(state.py:400/:462, _state_write.py:90, _subconstruct.py:103), reimplemented to consult the
table; (d) normalize compiler.py's `mods.get('operator')` postlude reads (:448, :533) to
`decomp.has_operator` so the pinned spelling is uniform. Size: one NamedTuple field, one
partition guard, ~6 call-site touch-ups.

**A2 — one gate function, two thin callers** (jtawq.3). Extract module-level
`_validate_slot_set(slots) -> None`: present-name frozenset ∈ `_COMBO_MAP` → pass; else raise
the pair-specific `ConstructError` from `_CONFLICT_DIAGNOSTICS` (generic `ConstructError` —
never KeyError — for uncovered sets); then run `_DYNAMIC_RULES` (Portal.is_dispatch+Operator).
`model_post_init`'s five arms + duplicated dynamic rule collapse to one call (direct
construction keeps its ValidationError-wrapped shape — established convention,
tests/modifiers/test_portal.py:213). `with_modifier` **keeps `model_copy`** but calls
`_validate_slot_set(prospective_slots)` first, preserving pipe-path `ConstructError` exactly.
**Explicitly rejected**: routing the pipe path through the constructor — Pydantic v2 wraps
`model_post_init` exceptions in `ValidationError` (empirically verified), so that "unification"
is a behavior change, violating the never-let-a-pure-refactor-change-behavior refusal.
`_SlotRule.excludes` is deleted; the field set, `combo`'s has-set, `to_list`, *both*
`classify_modifiers` paths, and the "Unknown modifier type" expected-list (:838 — currently
stale, missing Portal) derive from the roster. Canonical phrasing = the already-pinned
direct-path order ("Cannot combine Portal and X", test_portal.py:222-230); breakage shrinks to
docstring updates (test_portal.py, test_portal_operator_approval.py:17-20,
test_forward_parity.py:836, fusion-guard rationale prose).

**A3 — `PortalMemberClass`** (jtawq.1 axis 1). Enum `{ATOMIC, ATOMIC_OPERATOR,
AGENT_CYCLE_OUTPUT, AGENT_CYCLE_TOOL, SUB_CONSTRUCT}` (DISPATCH a sibling non-member arm) +
`portal_member_class(item)`, living **beside `classify_modifiers` in modifiers.py, not in
`_portal.py`** — `_portal.py`'s raw discriminators (is_dispatch :80, is_tool_triggered :93-106)
deliberately don't know about node modes or Construct-ness; those stay the single source for
their own axis and `portal_member_class` composes them with `COMBO_DECOMPOSITION`.

**Import mechanism (binding, not optional)**: `modifiers.py` sits *below* `node.py` and
`construct.py` in the import DAG — construct.py:33 and node.py:73 import modifiers.py at module
level, and modifiers.py refers to `ConstructItem` only under `TYPE_CHECKING` (modifiers.py:43).
Therefore `portal_member_class` MUST NOT import `Node` or `Construct` at module level (hard
cycle), and MUST NOT take a function-local import (that grows
`FUNCTION_LOCAL_IMPORT_ALLOWLIST`, which this doc's own decision ladder refuses — a deferred
import is never the answer). The prescribed mechanism is **structural discrimination via the
existing `ConstructItem`-protocol pattern that `classify_modifiers` already uses**
(modifiers.py:169: `getattr(item, "modifier_set", None)` + duck-typed fallback): discriminate
node-vs-construct by structure (e.g. `getattr(item, "nodes", None) is not None` for a
sub-construct, or a `runtime_checkable` protocol in `_ir_protocols.py`), and read the mode axis
via `getattr(item, "mode", None)`. If implementation shows structural discrimination is
insufficient for a case, the fallback is relocating the classifier to a module *above*
node/construct in the DAG — never a function-local import.

First-wave migrations: state.py's private `_is_dispatch` (:225-227), compiler.py:280's
arm-selection branch, _wiring.py:333-425's inline if/elif chain, loader.py:373's type-name
derivation.

### Pillar B: one bridge — layered modules, one guard idiom, one binding contract, mechanical plumbing collapsed

Target layer map (top depends on bottom; RULE B: extracted import modules receive
`from_spec: Callable` by injection, never import loader). The map is acyclic as drawn —
verified: loader.py, _agent_spec_group_import.py, and _agent_spec_node_import.py import no
factory symbol, so the new edge factory.py → _agent_spec_dispatch.py → loader.py introduces no
cycle:

```
factory.py ──> _agent_spec_dispatch.py ──> loader.py ──> _agent_spec_swarm_import.py
                                                              │
                                          _agent_spec_group_import.py
                                                              │
                                          _agent_spec_node_import.py
                                                              │
                              {_agent_spec_types.py, _agent_spec_markers.py (+ helper)}
                                                              │
                                          spec_types.py  (native registry, Agent-Spec-free)

export side (unchanged homes): _agent_spec.py ─> _agent_spec_{node,modifier}_lowering.py,
                               _agent_spec_portal.py ─> _agent_spec_markers.py
```

**B1 — one guarded-import helper** (jtawq.7 remainder): `import_pyagentspec(*submodules)` in
`_agent_spec_markers.py` (already the neutral, module-level-pyagentspec-free home hosting one
copy at :52-56). Collapses all four sites (loader.py:92, _agent_spec_markers.py:52-56,
_agent_spec_types.py:48, factory.py:826-833). Verified compatible with the core-purity guard,
which flags only *module-level* Import/ImportFrom AST nodes
(tests/test_guards_agent_spec_core_purity.py:70-100) — the probe's open question is closed YES.

**B2 — swarm-import extraction** (jtawq.10): move loader.py:237-491 (six functions, 255 lines)
to `_agent_spec_swarm_import.py`, injecting the single `from_spec` callable at the two dispatch
points (loader.py:525, :533) via the established `_construct_from_subflow` pattern. Keep
loader.py's F401 re-export idiom (:617-667). loader.py drops under 500 → **delete** its
ALLOWLIST entry (never lower to a sub-500 number).

**B3 — Agent-Spec runtime gate extraction** (jtawq.9, refined option (a)): move
`_resolve_expected`/`_prepare`/`_finish`/`_check_and_increment_depth` (factory.py:793-916) to
`_agent_spec_dispatch.py`. `make_portal_dispatch_fn`, its twins, and its four `Command(` sites
stay in factory.py; **G1 stays exactly `{factory.py, runner.py}`**
(tests/test_guards_assembly.py:104). Two mandatory same-commit re-keys:
`FUNCTION_LOCAL_IMPORT_ALLOWLIST` entry factory.py → _agent_spec_dispatch.py
(tests/test_guards_sidecar_imports.py:73, relocation already anticipated in its comment; a
genuine re-key, not a widen — the `compile` + `AgentSpecDeserializer` function-local imports at
factory.py:824-833 move with `_prepare`, and the cycle rationale transfers intact) and
`TestPortalDispatchRoutesThroughCanonicalGate` re-keyed **two-sided** (see G-GATE, section 6):
the emitted spec still passes only the canonical `from_agent_spec` + `compile` gate, AND
factory's wrapper is structurally pinned to call the extracted gate. This move is
behaviour-adjacent (closure-to-builder), so it gets its own tests per file-split-procedure.md;
fallback on difficulty is a smaller extraction, never a G1 widening.

**B4 — routing plumbing collapse** (jtawq.1 axis 2): a frozen `PortalRouteSpec`
(channel_key/count_field/payload_field/route_field/valid_targets/max_hops/on_exhaust/exit_name/
entry_name/…) with `to_command(...)` delegating to the *existing* decision functions, plus a
`MeshContext` built once in `_add_portal_mesh` (replacing the 6-kwarg threading at
_wiring.py:305-331 → :342-396, :749-777). Collapses 4 param-block copies + **10** routed-call
sites into thin per-class adapters keyed by `PortalMemberClass`. Pure split: no restructuring of
sync/async twins, no change to hop-budget or approval accounting (increment-only-on-approve,
factory.py:424-425). Read-side plumbing only: nothing here writes `handoff_param` or
`handoff_channel`, so guard G3's single-writer invariant (`_ir_normalize.py`,
tests/test_guards_llm_runtime.py:1051) is untouched.

**B5 — export/import read the vocabulary** (jtawq.1 axis 3, qtfof extension point): one two-way
table mapping `PortalMemberClass` ↔ Agent-vs-Flow member encoding, trigger axis ↔ `HandoffMode`
(via the existing `_swarm_trigger` inverse pair), entry knobs ↔ `_MARK_PORTAL_SPEC`. Replaces
loader.py:373's type-name check and the hard-coded :380-384 rule. The runtime must **not**
mirror Swarm's encoding — Swarm stays a documented lossy adapter (loader.py:411-442 warnings;
_agent_spec_portal.py:158-170 exit-gate approximation).

**B6 — the external-binding contract, named once** (wq9nn policy + qtfof): one bridge-level
statement (module docstring + shared refusal-raise helper in `_agent_spec_markers.py`
territory) of the contract *"a string names a Python callable the importing side supplies;
expression conditions are the closed grammar of conditions.py:93-129; callables fail loud at
export"* — consumed by Loop.when, Operator.when, gate_tools_when, scripted_fn, merge_fn
lowerings instead of per-module hint restatements (_agent_spec_modifier_lowering.py:372,
_agent_spec_node_lowering.py:77). `_spec_loader.py:258` is dispositioned **in P5, by P5's
implementer, not deferred**: default outcome is that it joins the parse-then-fallback
discrimination (native YAML specs may carry registered names; consistency beats a documented
divergence); the expression-only alternative is permitted only if the native-YAML schema is
shown to already reject non-expression condition strings at validation time, in which case the
expression-only status is documented in code at that site. Either way the divergence does not
survive P5 silently. RemoteAgent export lowering lands per the pt85t refined plan
(class-discriminator sidecar, per-family attrs verified against installed pyagentspec,
fail-loud on mismatch), consuming `node._remote_agent_endpoint` — closing qtfof's sole live
item and the silent re-export-as-scripted seam.

### Pillar C: the differential execution oracle (the disease's backstop)

Tests-only. Over the mechanically derived matrix registry (tests/test_agent_spec_matrix.py —
CELLS :414 / GREEN :433 / build_cell :215, already imported by the reachability suite at
test_agent_spec_reachability.py:51,128):

- **(a) WIRE — universal, day one**:
  `Flow.from_dict(to_agent_spec(build_cell(...)).to_dict())` for every GREEN cell. ~3 lines in
  `TestAgentSpecRoundTripMatrix` (:714-752), replacing the in-memory flow with the
  wire-reloaded one. Kills the 8zvd1/p7dyq tier per-cell instead of per hand-typed
  `TITLE_SHAPES` entry (test_agent_spec_roundtrip.py:458-489, 515-522).
- **(b) EXECUTE — all GREEN Flow-exporting cells**: a literal Agent-Spec mini-executor in
  tests/ — the semantic extension of the reachability BFS (:81-113): walk ControlFlowEdges from
  StartNode, evaluate BranchingNode conditions, pass values only via DataFlowEdges, iterate
  MapNode subflows. Semantics pinned to `_wiring`'s runtime (do-while entry, gate-after-body,
  pause reconvergence) exactly as the reachability docstrings already document (:35-38,
  :139-150, :185). Catches the rh5fb and s7zt3.15/.17 tiers even before output comparison.
- **(c) COMPARE — staged, ratcheted**: mini-executor outputs == `run(compile(construct))`
  outputs. **Immediately possible only for `mode=scripted` cells** — `build_cell`'s
  `_llm_kwargs` (:193-206) gives think/agent/act cells `model='fast'` + prompt, *not* scripted
  fakes (a probe premise refuted in verification). LLM-mode GREEN cells enter an explicit
  `EXEC_EXEMPT` frozenset with per-entry reasons, partition-asserted against GREEN, shrink-only,
  burned toward `frozenset()` — copying the PENDING pattern verbatim. Burn-down path: wire
  tests/fakes.py (`configure_fake_llm`/`StructuredFake`/`ReActFake`) identically into both
  sides.

Guardrails: the executor constructs no LangGraph `Command` (G1 untouched); it must not import
neograph's importer path (independence from the forgiving marker-reader is the point); its own
possible wrongness is defended by (c)'s two-sided comparison. **Expected side effect, budgeted
not feared**: per history (82/104 red on the last new oracle), a nonzero initial red set is
likely — findings, not blockers. Recount cell totals; do not reuse 104.

This is the answer to "what would have caught m57mn→s7zt3.17 earlier": (a) catches the
wire-contract tier at the first bad cell, (b) catches straight-line-flattened branches
(a734217's bug is *live and reachable* — only a literal walk that evaluates conditions sees the
missing BranchingNode), (c) catches everything semantic the first two miss.

---

## 3. Conflict resolutions between probes

1. **Single gate *entry point* vs single gate *function*** (jtawq.3): the probe's "make
   with_modifier construct via `ModifierSet(**slots)` so `model_post_init` re-runs" is
   rejected — verified to flip pipe-path rejections from `ConstructError` to Pydantic
   `ValidationError`. Resolution: one *function* (`_validate_slot_set`), two thin callers.
   Parity becomes structural without a behavior change.
2. **`fused` field vs "third membership authority"**: adding a column naively creates a third
   spelling next to `is_each_oracle_fused` and `modifier_names_for_combo`. Resolution: the
   column is *derived from* `modifier_names_for_combo` at definition and pinned by a partition
   guard; the predicate is reimplemented over the table. One authority, three views that
   provably agree.
3. **Classifier home — `_portal.py` vs `modifiers.py`**: the jtawq.1 probe said "next to Portal
   where the raw discriminators live"; verification showed that conflates two files. Resolution:
   `portal_member_class` goes in `modifiers.py` beside `classify_modifiers`; `_portal.py:80-106`
   remains the single source for the raw is_dispatch/is_tool_triggered axes, which the
   classifier *reads*. Because modifiers.py sits below node.py/construct.py in the import DAG
   (construct.py:33, node.py:73 import it), the classifier discriminates node-vs-construct
   **structurally** (the `classify_modifiers` duck-typing pattern, modifiers.py:169, plus
   `getattr(item, "mode", None)`) — never via a `Node`/`Construct` import, module-level (cycle)
   or function-local (allowlist growth, refused). See A3 for the full mechanism and its
   sanctioned fallback.
4. **Visitor/Strategy unification vs in-place tables**: both the jtawq.2 and jtawq.1 probes
   flirted with a unifying dispatch mechanism; both verifications rejected it. Resolution:
   **rejected, recorded as resolved-by-events.** The per-shape arm bodies bind genuinely
   divergent tuples (export's 7-tuple `_LoweredItem` vs compiler graph mutations vs loader
   items vs state field dicts), the export/import mirror is documented as deliberate on both
   sides (_agent_spec.py:212-225; loader.py:540-545), and the two route-decision functions
   differ in signature for a documented reason (factory.py:326-345). A visitor would be a
   *second parallel dispatch mechanism* — the disease, not the cure. Consumers stay where they
   are; only the *facts they consult* centralize.
5. **jtawq.9: new module vs stay-in-place vs G1 exit**: G1 exit is impossible (10 residual
   Command sites); staying in place leaves factory.py Agent-Spec-coupled. Resolution: extract
   the gate (no Command construction) to `_agent_spec_dispatch.py`; the Command-emitting
   wrappers stay; G1 never widens, two guards re-key in the same commit (G-GATE two-sided).
   This also severs factory.py's module-level Agent-Spec imports entirely (:31 `from
   neograph.loader import from_agent_spec`, :35 `lookup_type` — all uses fall inside the gate
   region).
6. **Swarm encoding vs runtime taxonomy**: export/import could have become the member-class
   authority (they already encode both axes). Resolution: no — Swarm is a documented *lossy*
   adapter; the runtime vocabulary (`PortalMemberClass`) is the authority and the Swarm mapping
   is one two-way table reading it.
7. **`_SLOT_RULES` vs `COMBO_DECOMPOSITION` merge**: rejected. Legality (set-membership
   domain), meaning (decomposition of legal combos), and diagnostics (why illegal pairs fail)
   are three deliberately separate tables — folding exclusions into `COMBO_DECOMPOSITION` would
   require illegal enum members poisoning every `PrimaryShape` dispatch site.

---

## 4. Phased build plan

Ordered to keep red stretches short: bookkeeping first, vocabulary before movers (so the
consumer inventory follows moved code, per the runner.py→_recursion_budget.py re-key precedent),
movers before features, oracle as an early parallel track with a budgeted red set. Every phase
lands green with its guards written failing-first.

| Phase | Work | Closes / supersedes |
|---|---|---|
| **P1 — Close the books** | Add the node-level registered-name Loop round-trip pin (beside test_agent_spec_loop_operator.py's expression form; assert `when` verbatim + reimport compiles with `compile(conditions=...)`); do NOT duplicate the existing Operator pin (:177-195). Fix node.py:191-193's rotted module reference. Rewrite qtfof's description to its single live item. Close **wq9nn** as fixed-by-ijyjr, with the closure note explicitly recording the ticket's OPEN DECISION (v1-gating vs v2-deferral) as **mooted by the fix**. Rescope jtawq.2 (strike runner.py/factory.py; record visitor rejection), jtawq.7 (four sites), jtawq.1/.10 (current line refs). | wq9nn closed; ticket text de-rotted |
| **P2 — Oracle step (a)** | Wire round-trip for every GREEN cell in `TestAgentSpecRoundTripMatrix`. Fix or file anything it turns red. | Pillar C(a); supersedes hand-typed TITLE_SHAPES coverage |
| **P3 — A2: slot-set gate** | `_validate_slot_set` + `_CONFLICT_DIAGNOSTICS` + `_DYNAMIC_RULES`; roster shrink; both `classify_modifiers` paths + `to_list` + field set derive/pin to roster; guards G-SLOT (i)-(iv) failing-first; docstring updates in the four affected test files. | **jtawq.3** |
| **P4 — A1: fusion column** | `ComboDecomposition.fused` + partition guard; delete loader shim; normalize compiler postlude spelling; guard extensions G-VOCAB + G-OP. | **jtawq.2** (residual; then close) |
| **P5 — B1: import helper** | `import_pyagentspec` in `_agent_spec_markers.py`; collapse 4 sites (factory.py's site noted as relocating in P8). Shared external-binding refusal helper + contract docstring (B6 first half). `_spec_loader.py:258` is dispositioned here, by this phase's implementer: joins the discrimination by default, or is documented expression-only in code if the native-YAML schema provably rejects non-expression strings — the fork does not fall between phases. | **jtawq.7** |
| **P6 — B2: swarm-import move** | `_agent_spec_swarm_import.py`; one injected `from_spec`; F401 re-exports; DELETE loader.py ALLOWLIST entry (re-derive with `wc -l` at commit time). Must follow P4 so the moved code already reads `decomp.fused`. | **jtawq.10** |
| **P7 — A3: PortalMemberClass** | Enum + classifier in modifiers.py via **structural discrimination** (ConstructItem-protocol pattern per A3 — no Node/Construct import, no function-local import); migrate state.py `_is_dispatch`, compiler.py:280, _wiring.py member chain, loader derivation; new consumer ratchet G-PMC built from a **fresh** sweep (currently 7 discriminator files + loader). | jtawq.1 (axis 1) |
| **P8 — B3: dispatch gate extraction** | `_agent_spec_dispatch.py`; twins + Command sites stay; re-key `FUNCTION_LOCAL_IMPORT_ALLOWLIST` + re-key `TestPortalDispatchRoutesThroughCanonicalGate` **two-sided** (gate-calls-from_agent_spec AND wrapper-calls-gate) same commit; own tests (behaviour-adjacent). Relocates P5's fourth guard-site. | **jtawq.9** |
| **P9 — B4: PortalRouteSpec/MeshContext** | Collapse 4 param blocks + 10 call sites into per-class adapters keyed by `PortalMemberClass` (hence after P7). Pure split; no twin restructuring. | jtawq.1 (axis 2) |
| **P10 — B5: export/import table** | Two-way `PortalMemberClass` ↔ Swarm-encoding table replacing loader.py:373/:380-384; lossy-adapter status unchanged. | **jtawq.1** (close) |
| **P11 — B6: RemoteAgent export** | Per pt85t refined plan; consumes `_remote_agent_endpoint`; touches no combo consumer, no monopoly. Definition of done includes the website-docs touchpoint: RemoteAgent round-trip is a consumer-visible interop capability, and AGENTS.md requires website content updates when API surfaces change — one pass over the Agent-Spec interop page ships in the same phase. | **qtfof** (close) |
| **P12 — Oracle steps (b)+(c)** | Mini-executor + scripted-cell comparison + `EXEC_EXEMPT` ratchet; burn-down via fakes on both sides. Can start in parallel any time after P2; listed last only because its red set is the largest and should not gate the mechanical phases. | Pillar C complete |

---

## 5. Refusals (declared complexity that stays)

Per docs/file-split-procedure.md's two refusal rules, the following are **not** collapsed into
the unification:

1. **No visitor / no Strategy hierarchy** over the per-shape dispatch arms or the two
   route-decision functions (see Conflict 4). The mirrored export/import dispatch is
   documented, deliberate layering.
2. **G1 never widens.** factory.py keeps its 10 non-dispatch `Command(` sites and its 949-line
   ceiling; the dispatch-gate extraction is scoped so no new module joins `_ALLOWED`. If P8
   turns out to need Command construction in the new module, take a smaller extraction instead.
3. **Sync/async twin duplication in factory.py stays** (the 8× route-call collapse in P9
   reduces *plumbing*, not the twin structure). Twin dedup is jtawq.6 — out of scope, no
   evidenced Agent-Spec coupling.
4. **Three tables stay three tables**: `_COMBO_MAP` (legality) / `COMBO_DECOMPOSITION`
   (meaning) / `_CONFLICT_DIAGNOSTICS` (diagnostics). Merging them was considered and rejected
   (Conflict 7). Likewise `_SLOT_RULES`-as-roster stays a sibling, not a column of either.
5. **Swarm remains a lossy adapter** with its documented warnings and exit-gate approximation;
   fidelity lives in `_MARK_PORTAL_SPEC` metadata, not in forcing the runtime taxonomy into
   Swarm's shape.
6. **Direct-construction `ValidationError` wrapping stays** (test_portal.py:213's "established
   convention") — unifying the exception *shape* across surfaces would be a behavior change
   bought for cosmetics.
7. **No function-local imports to solve layering.** `portal_member_class`'s node-vs-construct
   need is met structurally (A3); growing `FUNCTION_LOCAL_IMPORT_ALLOWLIST` to buy a
   convenient home is the same trade as widening G1 — refused.
8. **File-size numbers are re-derived at commit time** (`wc -l`, exact-ceiling rule); every
   post-move estimate in this doc is provisional. A file landing slightly over with clean
   boundaries beats a wrong-module function landing it under (the `_oracle.py` precedent).

---

## 6. New structural guards

All failing-first, all in the house ratchet shapes (`MIGRATED`/`PENDING == frozenset()`, exact
allowlists, AST scans):

| Guard | Pins | Phase |
|---|---|---|
| **G-SLOT** (4 assertions, new file or test_guards_assembly extension) | (i) totality: every 2-subset of roster slot names ∈ `_COMBO_MAP` ∪ dom(`_CONFLICT_DIAGNOSTICS`), domains disjoint — kills the modifiers.py:787 raw-KeyError drift hazard for modifier #6 (drift protection, not a live bug); (ii) AST ban on `'Cannot combine'` literals outside the tables; (iii) ModifierSet field names == roster; (iv) both `classify_modifiers` paths + `to_list` iterate the roster. Extends, does not contradict, `TestNoRedundantValidation` (test_guards_assembly.py:1423): "ModifierSet only" strengthens to "`_validate_slot_set` only". | P3 |
| **G-FUSE** (partition guard in test_guards_combo_decomposition_consumers.py) | `decomp.fused == ({'each','oracle'} <= modifier_names_for_combo(c))` ∀ combos — the three fusion spellings provably agree; a future fused combo is one table-row edit. | P4 |
| **G-VOCAB** (census closure) | `is_each_oracle_fused` and `combo_for_modifier_names` join the tracked vocabulary (TABLE_SYMBOLS or sibling set) — a fusion-predicate-only consumer can no longer escape the inventory (a real hole today). | P4 |
| **G-OP** (consumption pinning) | Every MIGRATED file matching on `.primary` also references `decomp.fused`/`is_each_oracle_fused` AND `decomp.has_operator` — OR sits in a hand-written exemption literal with per-file reasons. The literal is **six** files (state.py, _state_write.py, _subconstruct.py, _input_shape.py, _recursion_budget.py, _wiring.py — operator adds no state field, key-wrapping, input shape, hop cost, or mesh wiring), re-verified by grep at implementation time, provisional until then. | P4 |
| **G-PMC** (new consumer ratchet, copies the PENDING pattern) | Every file touching is_dispatch/is_tool_triggered/member-classification consults `portal_member_class` or sits in an exemption literal; enumeration from a **fresh sweep** (today: 7 src files + loader's name-based derivation), never copied from this doc. Additionally pins the classifier's import discipline: `modifiers.py` contains no module-level or function-local `Node`/`Construct` import (AST scan) — the structural-discrimination mechanism of A3 is a guard clause, not a convention. | P7 |
| **G-GATE re-key (two-sided)** | The canonical-gate invariant is pinned end-to-end across the extraction seam, in the same AST-walker style as the current tests/test_guards_assembly.py:231-243. **Assertion (a)** — gate side: the extracted gate function in `_agent_spec_dispatch.py` calls `from_agent_spec` and calls none of the `_BANNED` bespoke-validator names (`load_spec`, `_validate_spec`, `_build_construct`). **Assertion (b)** — wrapper side: `make_portal_dispatch_fn` in factory.py calls the extracted gate function (AST-level call-name check). Without (b), `dispatch_wrapper` could later bypass the gate or grow a bespoke path with the guard green — a one-sided re-point does NOT preserve the invariant. `FUNCTION_LOCAL_IMPORT_ALLOWLIST` entry re-keyed in the same commit. | P8 |
| **G-ORACLE** | Wire step universal over GREEN (partition-asserted: a new GREEN cell must round-trip the wire); `EXEC_EXEMPT` partition-asserted against GREEN, shrink-only, burned toward `frozenset()`. | P2/P12 |
| **AGENTS.md rule amendment** | Generalize the "Lesson from the Portal rollout" paragraph: *any new classification fact about the IR* (not just a ModifierCombo) — fusion-ness, member-class, legality, binding-kind — gets ONE table/function in modifiers.py's vocabulary layer + a failing-first consumer ratchet before the feature is done; and *any new export capability* is not done until its matrix cells pass the wire + execute oracle. | P4/P12 |

**Why this set would have caught the historical chain**: 8zvd1/p7dyq die at G-ORACLE step (a)
on the first bad cell (wire contract exercised per-cell, not per-remembered-shape); rh5fb and
s7zt3.15/.17 die at step (b) (a literal walk that *evaluates* branch conditions cannot miss a
flattened `_BranchNode` — the a734217 bug produced reachable, live, semantically wrong code
that only execution sees); the m57mn class ("found while fixing something else") shrinks
because the vocabulary guards (G-FUSE/G-VOCAB/G-OP/G-PMC) make "who else consumes this fact" a
mechanical census instead of a hand-picked list — the exact failure mode the s7zt3 audit's
"complete matrix, every cell verified" claim could not survive.

---

## 7. Ticket disposition table

One row per remaining Agent-Spec-relevant ticket, stating exactly what happens to it under this
design:

| Ticket | Disposition | Detail |
|---|---|---|
| **neograph-jtawq.1** (Portal mesh routing + agent-cycle wiring) | **Reshaped-into** P7 + P9 + P10 | The filed framing ("design the missing agent-cycle abstraction") is stale — `_wire_agent_cycle_body` already exists (_wiring.py:557-681). The ticket's real content decomposes into three axes: member-class taxonomy (`PortalMemberClass` + classifier, P7), routing-plumbing collapse (`PortalRouteSpec`/`MeshContext`, P9), and export/import reading the same vocabulary (two-way Swarm table, P10). Closes at P10. Rescope the ticket text in P1 to these three axes with current line refs. |
| **neograph-jtawq.2** (ModifierCombo/PrimaryShape dispatch core) | **Reshaped-into** P4, then closed | The 6-file span is stale: runner.py and factory.py hold zero combo vocabulary; the COMBO_DECOMPOSITION table + consumer ratchet already landed the bulk. The live residual is fusion-as-one-authority (`ComboDecomposition.fused` + G-FUSE/G-VOCAB/G-OP). The visitor/Strategy idea the ticket gestured at is **rejected, recorded as resolved-by-events** (Conflict 4). P1 rescopes; P4 closes. |
| **neograph-jtawq.3** (ModifierSet/_SLOT_RULES single-sourcing) | **Reshaped-into** P3 (substantially as filed) | The only ticket whose filed diagnosis still matches code: three encodings of pair legality, five per-slot enumerations, stale expected-list. Resolved by `_validate_slot_set` + `_CONFLICT_DIAGNOSTICS` + `_DYNAMIC_RULES` + roster shrink + G-SLOT. The probe's "re-run model_post_init" mechanism is rejected (behavior change); one gate *function*, two callers instead. Closes at P3. |
| **neograph-jtawq.7** (Agent-Spec bridge extraction + guard idiom) | **Reshaped-into** P5 (remainder only) | Primary half already landed (commit 7d53427). Remainder: `import_pyagentspec` helper collapsing **four** guard sites (the ticket lists three; factory.py:826-833 is the missed fourth, which then relocates in P8) + the shared external-binding refusal helper (B6 first half) + the `_spec_loader.py:258` disposition. Closes at P5. |
| **neograph-jtawq.9** (relocate portal-dispatch factory, G1 intact) | **Reshaped-into** P8 | G1-exit variant is **impossible** (10 non-dispatch `Command(` sites) — strike it from the ticket. Landing shape: extract the no-Command Agent-Spec gate cluster to `_agent_spec_dispatch.py`; Command-emitting wrappers stay; G1 `_ALLOWED` unchanged; two-sided G-GATE re-key + allowlist re-key same commit. Closes at P8. |
| **neograph-jtawq.10** (loader.py swarm-import relocation) | **Reshaped-into** P6; "needs injection design" clause **closed-as-stale** | The injection design the ticket asks for already exists (`_construct_from_subflow(…, from_spec)` seam, _agent_spec_group_import.py:48); the remaining work is the mechanical move of loader.py:237-491 with one injected callable, after P4 so moved code reads `decomp.fused`. loader.py's ALLOWLIST entry is deleted. Closes at P6. |
| **neograph-wq9nn** (registered-name conditions don't round-trip) | **Closed-as-stale (fixed-by-ijyjr)** in P1 | The bug is already fixed (commit 110e8d6, discrimination at _agent_spec_group_import.py:279-295), verified by fresh repro. P1 adds the one missing node-level registered-name Loop pin and closes the ticket, with the closure note explicitly recording the ticket's OPEN DECISION as mooted by the fix. Its policy content (the external-binding contract) is absorbed into B6/P5. |
| **neograph-qtfof** (Portal × Agent Spec v2) | **Reshaped-into** P11 (single live item), then closed | All deps/children closed; the epic's only live scope is RemoteAgent export-side lowering (zero readers of `_remote_agent_endpoint` today = silent lossy re-export). P1 rewrites the description to this single item; P11 implements it per the pt85t refined plan, including the website interop-docs touchpoint, and closes the epic. |

---

## 8. Consolidated Review Verdict (2026-08-03)

Three independent reviews were run against this document — completeness, correctness
(repro-based, not read-only), and consistency with the design's own stated principles and the
codebase's standing guards. This section consolidates their combined findings into one verdict
and one action list.

### Overall verdict: **SOUND (as corrected in place)**

The completeness review returned SOUND with zero required corrections: all eight in-scope
tickets receive concrete, phase-assigned dispositions; the wq9nn fix, the qtfof
single-live-item claim, and the jtawq.2 residual framing were independently re-verified against
code and beads rather than taken from the synthesis. The correctness review returned SOUND with
zero required corrections: every load-bearing claim was confirmed by live repro — the exact
10+4 `Command(` census in factory.py, the 8+2 route-call sites, the four (not three)
pyagentspec import guards, the zero-hit `portal_member_class` grep, the wq9nn fix at
_agent_spec_group_import.py:279-295, the `_spec_loader.py:258` divergence, the zero-reader
`_remote_agent_endpoint` sidecar, and all guard/oracle anchors. Its only finding was 1-7-line
citation drift on six references, corrected in this text (is_each_oracle_fused :288,
compiler.py:548, _state_write.py:90, loader.py:92, _agent_spec_markers.py:52-56,
_agent_spec_types.py:48); all consumer lists remain declared provisional-until-re-swept.

The consistency review verified the design's hardest composition claims directly — G1 never
widens (the P8 extraction cluster constructs no Command; all four dispatch sites stay in
factory.py), G3 is untouched (no proposal element writes `handoff_param`/`handoff_channel`),
the layer map is acyclic as drawn (no loader→factory import exists), and the fusion pillar
composes coherently with loader.py:547's existing `decomp` binding — but returned
NEEDS_REVISION on two blocking inconsistencies, **both corrected in place in this document**:

1. **`portal_member_class` import mechanism (was unspecified, dead-ended into a refusal).**
   The chosen home (modifiers.py) sits *below* node.py/construct.py in the import DAG
   (construct.py:33 and node.py:73 import modifiers.py at module level; modifiers.py sees
   `ConstructItem` only under TYPE_CHECKING at :43), so a module-level `Node`/`Construct`
   import is a hard cycle and a function-local import grows
   `FUNCTION_LOCAL_IMPORT_ALLOWLIST` — which this document's own decision ladder refuses.
   Resolution now specified in A3 and pinned by G-PMC: structural discrimination via the
   `ConstructItem`-protocol pattern `classify_modifiers` already uses (modifiers.py:169), with
   relocation above node/construct as the sanctioned fallback and the function-local import
   explicitly ruled out (Refusal 7).
2. **G-GATE re-key was one-sided (would have let the wrapper bypass the gate with the guard
   green).** `TestPortalDispatchRoutesThroughCanonicalGate` currently AST-walks
   `make_portal_dispatch_fn` in factory.py for a `from_agent_spec` call; re-pointing only at
   the extracted gate function would prove gate→from_agent_spec but drop the wrapper→gate
   linkage. Resolution now specified in section 6: the re-keyed guard is **two-sided** —
   (a) the extracted gate calls `from_agent_spec` and none of `_BANNED`; (b)
   `make_portal_dispatch_fn` calls the extracted gate, AST-level, same walker style as
   tests/test_guards_assembly.py:231-243.

### Remaining minor items (folded into the plan, no further action)

- wq9nn's closure note records its OPEN DECISION as mooted-by-fix (P1, completeness reviewer).
- The `_spec_loader.py:258` fork is assigned an owner and a default outcome inside P5 so it
  cannot fall between phases (B6, completeness reviewer).
- P11's definition of done includes the website interop-docs touchpoint per AGENTS.md's
  API-surface rule (completeness reviewer).

### Ready to implement vs. needs more design

**Ready to implement now, as designed**: P1 (bookkeeping), P2 (wire oracle — additive test
change), P3 (slot-set gate — single-file, guard-first), P4 (fusion column — one field + ~6
touch-ups), P5 (import helper), P6 (mechanical move over an existing seam).

**Ready, with the corrected mechanism as a hard constraint**: P7 (classifier must use
structural discrimination — G-PMC's import-discipline assertion is written failing-first),
P8 (two-sided G-GATE lands in the same commit as the move).

**Ready, with a budgeted red set**: P12 (mini-executor + comparison — a nonzero initial red set
is expected and is the point; cell totals are recounted, not reused).

No phase in this plan requires further design work before implementation. Every "all consumers"
enumeration in this document is provisional by declared policy: re-grep at implementation time,
never copy the list.
