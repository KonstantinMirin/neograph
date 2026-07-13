# KEYMAKER overnight build — decision log

Autonomous run, night of 2026-07-13 → 2026-07-14. Epic: neograph-t1f7z (feature bead: neograph-07inf).
Every judgment call made without the maintainer in the loop is recorded here, with rationale,
so the morning review can ratify or reverse each one. Grounding research:
docs/design/dynamic-handoff-research-2026-07-13.md.

Format: `D<n> — <decision>` / why / alternatives rejected / where it landed.

---

## Process decisions

**D1 — Design-first, independent review gate honored.** The 07inf bead mandates the spike
doc + independent architect review BEFORE implementation is greenlit. The overnight run
keeps that gate: design doc drafted (context-inheriting agent), reviewed by an independent
fresh-context architect, corrections adjudicated here, and only then are implementation
tasks filed and executed. Alternative rejected: skipping straight to code to maximize
overnight throughput — violates the bead's explicit gate and the maintainer's
independent-review preference.

**D2 — Execution discipline.** Every implementation task runs through
/dev-practices:execute via a dedicated fresh-context agent (main-agent context stays
clean); write-stages strictly sequential per the standing "execute molecule" rule;
full consolidation gates (mypy + full pytest + ruff, keyless examples) re-run on the
merged tree at the end regardless of per-task gates.

## Design-gate adjudication (independent review returned GO-WITH-CORRECTIONS)

Review artifact: scratchpad design-review.md (findings H1-H3, M1-M4; per-judgment-call verdicts).
The reviewer AGREED with 9 of the design's 10 judgment calls (incl. D-RESERVED-KEY after a
zero-collision grep, D-FORWARD-EXEMPT, D-DISPATCH-REGISTRIES); the one contested area was the
lowering strategy (H3) and its Operator fallout (H1).

**D3 — Command(goto)+destinations= KEPT for v1 (reviewer dissented; recorded for morning).**
The reviewer showed the two stated justifications (Command.PARENT mesh, mode-a/b unification)
are both cut from v1 scope, and recommended router+path_map (Loop-identical, composes with
Operator). Overruled: the maintainer pinned Command(goto) in the 07inf bead title and notes,
and the research consolidation adjudicated Command(goto)+destinations= deliberately
(forward-compat to v2 mesh-across-subconstructs; destinations= closes LangGraph's
silent-drop hole). Reversing the maintainer's written direction overnight is the larger risk.
The real cost the reviewer surfaced (H1) is removed by D4 instead. If the morning review
prefers the cheaper router, the blast radius of switching is one wiring site + the factory
wrapper — acceptable.

**D4 — KEYMAKER + Operator combo CUT from v1 (fixes H1).** `_add_operator_check` appends a
static post-edge (_wiring.py:939-961) which a Command(goto)-returning member bypasses, so
"approval gate on a hop" as designed was false — worse, silently false. v1 makes the combo
ILLEGAL at assembly with a clear ConstructError naming the workaround (Operator on the node
before mesh entry / after exit). v2 path documented: raise interrupt() inside the keymaker
wrapper. Alternative rejected: shipping the silently-dropped gate (production-quality rule).

**D5 — handoff_param written ONLY by the normalizer (accepts H2 fully).** The design had
_construct_builder.py writing it (decorator path only) — the exact ts7 three-surface parity
bug shape, and a violation of the fan_out_param single-writer invariant (neograph-k7bg,
sole writer = _ir_normalize). Re-scoped: _ir_normalize writes handoff_param for all three
surfaces; structural guard pins the single-writer rule.

**D6 — MEDIUMs folded into task text, T2 split (accepts M1-M4).** Compile WALK must be
mesh-aware (skip members; not just a new dispatch arm); ModifierSet.model_post_init needs
explicit keymaker arms (exclusion table is hard-coded); all FIVE assert_never sites get arms
(compiler x2, state.py:202/512/569); T2 split into core-lowering and checkpoint/budget-test
tasks to stay one-molecule-sized.

<!-- Implementation decisions appended below as they are made. -->

## Implementation decisions (T1 — neograph-rwion)

**D7 — T1 legal mesh ASSEMBLES but does not COMPILE; the compile-staging error is
pinned, and the should_pass fixture is deferred to T2.** The T1 scope pins the
compiler arms as fail-loud `CompileError("Keymaker lowering lands in T2")` (bead
scope + D6). This collides with two artifacts the write-test author produced that
expect a legal mesh to `compile()` successfully: the three `TestLegalMeshAssembles`
tests and the `should_pass/keymaker_mesh_minimal.py` fixture (the check-fixtures
harness compiles every should_pass Construct). Resolution, smallest defensible:
(1) the three `TestLegalMeshAssembles` tests keep their three-surface-parity intent
but assert the T1 reality — the `Construct(...)` ASSEMBLES cleanly (all §5 validation
passes) and `compile()` raises `CompileError` matching "lands in T2"; a new
`test_keymaker_compile_is_staged_to_t2` pins the staged error directly (T2 replaces
it). (2) The legal-mesh fixture moves from `should_pass/` to `should_fail/` with
`# CHECK_ERROR: lands in T2` for T1; T2 (which "replaces the pinning test") moves it
back to `should_pass/` once lowering lands. Rationale: honoring the pinned fail-loud
staging (never silently mis-lower a mesh member as a bare node) outranks the bead's
"should_pass ×1" AC detail, which is only satisfiable once T2 lowering exists.
Alternatives rejected: (a) compiling a mesh member as a bare node in T1 — a silent
wrong-compile, violates the production-quality/fail-loud rule; (b) dodging the
harness by not binding a module-level Construct in the fixture — a misleading
should_pass that proves nothing.

**D9 — a SIXTH `ModifierCombo` exhaustive match site exists (`_state_write.py:94`).**
Review M3 / the T1 bead enumerated FIVE `assert_never` sites needing a KEYMAKER arm
(compiler.py ×2, state.py:202/512/569). mypy flagged a sixth: the Each-wrapping
`match combo` in `_build_state_update` (`_state_write.py:94`). Folded KEYMAKER into
the no-Each-wrapping arm (`BARE|OPERATOR|ORACLE|...`): a mesh member writes its own
output plainly, never fan-out-key-wrapped. Runtime-unreachable in T1 (compile
fail-loud-stages before a member executes), but the arm is required for
exhaustiveness. Recorded so T2/T3 know the site exists.

**D8 — direct-`ModifierSet(...)` conflict tests assert on message, not exception
type.** The write-test author's four `TestDirectModifierSetConflicts` tests expected
`ConstructError`, but a `ModifierSet(...)` construction raises the exclusion from
`model_post_init`, where Pydantic wraps it into a `ValidationError` (a `ValueError`
subclass) — the exact established convention in
`test_modifier_edge_cases.test_each_loop_rejected_at_construction`
(`pytest.raises(Exception, match=...)`). Updated the four tests to
`pytest.raises(Exception, match="Cannot combine Keymaker and ...")` so both the
wrapped (direct) and unwrapped (pipe) forms pass while still asserting the offender-
naming message. The Core Invariant ("ConstructError naming the offender") is
satisfied in spirit: the ConstructError is the wrapped cause on the direct path and
raised directly on the pipe path.

## Implementation decisions (T2 — neograph-on6jt)

*(For the morning: T2 raised the compiler.py size-cap guard 690→720 with the
guard's own remedy honored (mesh wiring helpers `_add_keymaker_mesh` +
`_contiguous_keymaker_mesh` placed in `_wiring.py`; only the walk orchestration
and two exhaustiveness arms — core compiler responsibility — remain in
compiler.py) and the raise history recorded in the guard docstring. A size cap,
not a disease-baseline growth, but flagged here for ratification.)*

**D10 — the mesh-channel READ side is a node-self-contained IR field
(`Node.handoff_channel`) stamped by the `_ir_normalize` normalizer, NOT plumbed
through `_execute_node`.** The T2 architect review (dp-reviewer, MEDIUM) found the
design's §3.3 read side under-specified: a non-entry member's `_extract_input`
receives only `(bus, node)` with `node.handoff_param == "handoff"` (the bare
literal) carrying NO entry-field, so it cannot build the entry-keyed channel key
`neo_handoff_<entry_field>` — the channel is keyed off the mesh ENTRY
(one-mesh-per-level, D-SINGLE-MESH), not the member's own field. **Decision
(lead-adjudicated):** apply the `fan_out_param` precedent exactly — a node-level
IR field the extract path reads WITHOUT signature changes. `_ir_normalize`
(`normalize_ir`) already has the construct-level view and is the single writer of
`handoff_param`; it now also computes the entry-keyed channel once
(`StateKeys.handoff_payload(field_name_for(entry.name))`) and stamps it onto every
Keymaker member's new `Node.handoff_channel` field, alongside `handoff_param`.
`_extract_fan_in_dict` reads `bus.get(node.handoff_channel)` (optional — a member
reached via a hop always has it populated by the previous hop's `Command` update;
an entry declaring a `handoff` param on first activation legitimately sees `None`)
for the reserved `"handoff"` input key. This keeps `_execute_node`,
`_aexecute_node`, and `make_node_fn` signatures untouched (wrapper uniformity
preserved), and preserves the single-writer invariant (H2/D5): `_ir_normalize` is
the SOLE writer of both `handoff_param` and `handoff_channel`, since the
entry-keyed key is a construct-level fact no assembly path can compute per-node.
**Alternative rejected — plumbing `entry_field` through `make_node_fn →
_execute_node → _extract_input`** (the executor's initial implementation): wider
signature blast radius across the shared dispatch path and it breaks wrapper
uniformity by adding a Keymaker-specific kwarg to the generic node factory; the
node-self-contained field mirrors `fan_out_param` more faithfully and localizes
all Keymaker read-side knowledge to `_ir_normalize` + `_input_shape`. Also
rejected: state-scanning for the sole `neo_handoff_*` field (non-deterministic by
design; the reviewer forbade it). `Node.handoff_channel` is the SECOND sanctioned
new IR field for Keymaker (beside `handoff_param`); `_input_shape.py` is in-scope
for T2 (design §4.1's SCOPE table omits it, but §3.3 requires it and the AC
"genuine cycle end-to-end" forces it — a doc-table gap, not a contradiction).

---

## T3 decisions (hop budget + checkpoint semantics — neograph-0umvg)

**D11 — Hop-budget semantics: a "hop" is a peer continuation; HANDOFF_END exit is never budget-gated.**
The shared, entry-keyed counter `neo_handoff_hops_<entry_field>` counts member→PEER
handoffs (mesh continuations). The entry member's own first execution (triggered by the
static `prev→entry` edge) is NOT a hop; the first peer route out of any member is hop 1.
Check is BEFORE emitting a peer goto (Loop parity `count >= max_iterations`): if
`current >= max_hops` apply `on_exhaust`. `max_hops=N` ⇒ exactly N peer-hops allowed,
then exhaust. A member routing to `HANDOFF_END` leaves the mesh cleanly and is NOT
budget-checked (it terminates rather than continues; gating it would penalize a
legitimate exit at the boundary) and does NOT increment the counter. On the allowed
peer hop the wrapper writes `count_field: current+1`. Counter is read from the INCOMING
`state` (not the local update dict) so hops accumulate across DIFFERENT members' wrappers.
- `on_exhaust=="error"` → `ExecutionError.build("handoff exceeded max_hops", node=<entry.name>, ...)`; no new exception class (Loop parity, design §3.4/§6).
- `on_exhaust=="exit"` → `Command(goto=exit_name, update={**update, channel_key: payload, count_field: current})`; last payload stays on the bus.
why: matches design §3.4 "checked before emitting the goto" while scoping "the goto" to
peer-continuation gotos; keeps the existing T2 cycle test green (triage→billing→triage-exit
= 2 peer-hops under max_hops=6). Boundary pinned test-first.
alternatives rejected: gating HANDOFF_END too (raises on a legitimate boundary exit);
check-after-increment (off-by-one vs Loop parity).
landed: factory.py make_keymaker_fn / _to_command (threads `state` + entry budget params).

**D12 — Budget knobs threaded entry→every-member as closure params (T2 closure-capture analog).**
`max_hops`/`on_exhaust` are entry-only (T1 validation), but `make_keymaker_fn` runs
per-member. `_add_keymaker_mesh` sources them from `entry=members[0]`
(`entry.modifier_set.keymaker.max_hops`/`.on_exhaust`) plus `entry.name` (for the error
`node=`) and threads them into `make_keymaker_fn` for EVERY member as closure params —
same closure-capture threading T2 uses for `entry_field`/`exit_name` (not literally D10's
normalizer stamp; the appropriate analog). Do NOT read the member's own keymaker for the
budget (non-entry members default max_hops=10).
landed: _wiring.py:_add_keymaker_mesh, factory.py:make_keymaker_fn signature.

**D13 — Recursion floor covers mesh-only constructs via a level-ordered mesh walk (team-lead adjudicated).**
`_ensure_agent_recursion_limit` computes `mesh_cost` = Σ entry.max_hops ONCE per mesh, via
a level-ordered walk that reuses the compiler's contiguous-mesh grouping and recurses
sub-constructs (NOT `iter_nodes`, which leaf-flattens and cannot identify the entry nor
find a defaulted-max_hops entry). The zero-guard becomes
`if agent_cost == 0 and mesh_cost == 0: return config`, so a mesh-ONLY construct still
raises the floor; required = base + agent_cost + mesh_cost; floor only RAISES (a larger
user limit is kept).
why: iter_nodes+model_fields_set detection MISSES a mesh whose entry left max_hops at the
default 10, and a per-member sum under-counts when entry.max_hops>10 (members report
default 10) — both unsafe; the level-ordered entry-once walk is exact.
alternatives rejected: per-member iter_nodes over-approximation (under-counts, unsafe);
model_fields_set entry detection (misses defaulted entry).
landed: runner.py:_ensure_agent_recursion_limit + a mesh-cost helper.
