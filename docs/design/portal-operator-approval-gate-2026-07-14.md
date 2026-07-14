# Portal + Operator: human-approval gate on a dynamic handoff (D4 v2 spike)

Spike bead: neograph-airpr (deferred from KEYMAKER v1, epic neograph-t1f7z).
Runnable proof: `tests/test_spike_portal_operator_approval.py` (4 tests, passing).
Grounding: docs/design/dynamic-handoff-2026-07-13.md, docs/design/keymaker-decision-log-2026-07-13.md (D4),
docs/design/durable-execution-replay-research-2026-07-02.md.

## Verdict

**Portal + Operator is buildable without re-invoking the member's LLM on resume,
and should become a first-class composition in v2.** The D4 combo-cut is
temporary, not permanent — which strengthens (not weakens) the 7t7tf
`Command(goto)` ratification: the same route-through-a-node-I-name-at-runtime
power that creates the D4 bug is exactly what fixes it.

The assembly ban (D-NO-OPERATOR-COMBO) **stays in place until the v2 shape is
implemented**. Nothing in this spike lifts it; acceptance #4 of the bead is
conditional on landing the feature, which is filed as follow-up work.

## The problem (why v1 cut the combo)

`_add_operator_check` (`_wiring.py:1082`) wires the approval interrupt via a
STATIC edge: `graph.add_edge(node_name, check_name)`. A Portal member returns
`Command(goto=peer)`; LangGraph follows the goto and the static
`node -> {node}__operator` edge is **never traversed**. The approval node is
wired, looks correct, and silently never fires. Silent-never-fires is worse
than illegal, so v1 made the combo illegal at assembly.

## Core invariant

> The approval pause must sit **on** the dynamic `Command(goto)` path, never on
> a statically-appended edge.

## Two candidate shapes, one proof each

### Naive (REJECTED): `interrupt()` inside the member wrapper

Call `interrupt()` in `make_portal_fn`'s wrapper right before returning the
`Command`. Simple, but fails the cost crux: LangGraph resumes an interrupted
node **from the top**, so everything before the `interrupt()` — the member's
LLM turn and tool calls — re-executes on resume. Double model spend, and the
re-run may produce a *different* answer than the one the human just approved.

Proven, not assumed: `TestNaiveInWrapperShapeFailsTheCrux` asserts the member
body runs **twice** across pause+resume in this shape. (If LangGraph's resume
semantics ever change and that test starts failing, revisit this document.)

### Approval-node-on-the-dynamic-path (TARGET): splice a dedicated node into the routed path

```
member ──Command(goto="{member}__approve", update={proposed_target})──▶ {member}__approve
                                                                            │ interrupt({proposed_target})   ← the ONLY interrupt site
                                                              approve ──Command(goto=proposed_target)──▶ peer
                                                              reject  ──Command(goto=exit)──▶ __handoff_exit_<entry>
```

- The Operator-guarded member computes its handoff normally, but its wrapper
  emits `Command(goto="{member}__approve")` instead of `goto=peer`, carrying
  the proposed target in the update.
- `{member}__approve` is the **only** `interrupt()` site. It does no expensive
  work, so LangGraph's re-run-from-the-top resume re-executes only this cheap
  node. The member's LLM/tool spend happens exactly once.
- The approval node itself returns `Command(goto=...)`, so the pause sits on
  the live routed path — the goto cannot skip it (invariant satisfied by
  construction, not by edge-wiring discipline).

Proven by `TestApprovalNodeOnDynamicPath` against a real file-backed
`SqliteSaver` (sync + async):

1. the run pauses at the approval node and surfaces
   `{"proposed_target": ...}` as the `__interrupt__` payload;
2. `Command(resume="approve")` routes to the approved target;
   `Command(resume="reject")` routes to the mesh exit;
3. the member body executed **exactly once** across pause+resume in every case.

This is the same shape as the agent/act tool-approval gate
(`gate_tools_when`, `_wiring.py:1035+`): a dedicated cheap gate node that is
the only interrupt site, inserted on the live path before the guarded action.
Prior art, not a new pattern.

## Decision (spike acceptance #3): first-class composition, surface shape

**Recommendation: make `Portal + Operator` first-class in v2** rather than
keeping it permanently illegal with a nicer error. The composition is the
natural reading — "pause for a human before this member hands off" — and the
proof shows it is implementable without semantic surprises.

Surface: **reuse `Operator` as-is** (`member | Portal(to=[...]) | Operator(when=...)`),
no new `approve=` kwarg on Portal. Rationale:

- `Operator(when=...)` already means "conditionally pause here for a human";
  the composition changes *where* the pause is spliced (dynamic path instead of
  static postlude), not what it means. One concept, two lowerings.
- A Portal-specific `approve=` kwarg would fork the HITL vocabulary and leave
  `Operator` still-illegal on Portal for no user benefit.
- Three-surface parity comes free: `@node` (operator kwarg), declarative, and
  programmatic pipe all already produce `ModifierSet(portal=..., operator=...)`
  — only the assembly ban and the lowering change.

Semantics of the composition:

- `Operator.when` gates whether the approval node interrupts (falsy predicate
  → pass-through `Command(goto=proposed_target)` without pausing), mirroring
  `operator_check`'s conditional interrupt.
- Interrupt payload: the proposed target plus the member's payload (what the
  human needs to judge the handoff). Resume value: approve/reject decision —
  exact schema to be settled at implementation time against `ask_human`'s
  typed-resume pattern.
- Rejection routes to the mesh exit (`__handoff_exit_<entry>`) with the last
  payload on the bus. A "redirect to a different peer" resume is a possible
  later extension; not required for v2.
- Hop budget: an approved handoff counts as one peer-hop exactly as today; the
  approval node itself never increments the counter.

## Implementation sketch (for the follow-up feature bead)

1. **Lowering** (`_wiring._add_portal_mesh`): for each member whose
   `modifier_set.operator` is set, add `{member}__approve` (a
   `make_portal_approval_fn` from `factory.py` — G1 keeps `Command(`
   construction confined to factory.py/runner.py) with
   `destinations = peers ∪ {exit}`, and thread the approval-node name into that
   member's `make_portal_fn` so `_to_command` emits
   `goto={member}__approve` for PEER routes (HANDOFF_END exits stay direct and
   unguarded, or become guarded — decide at implementation; leaning unguarded:
   Operator guards the *handoff*, and exiting the mesh is not a handoff).
2. **Hop-budget placement**: keep the budget check in the member wrapper
   (before proposing), and carry the already-incremented counter through the
   approval node's update — so an approved hop costs exactly one, and a
   rejected hop should refund/not-count (settle in TDD).
3. **Ban lift** (`modifiers.py`): remove the `operator` ↔ `portal` reciprocal
   excludes from `_SLOT_RULES` and the pairwise arm in `ModifierSet._validate`;
   keep every other Portal exclusion. Update
   `tests/check_fixtures/should_fail/portal_operator_combo.py` → a
   `should_pass` fixture; add should_fail fixtures for any still-illegal shape.
4. **Validation** (`_validation_portal.py` / `_validation_keymaker` rules):
   mesh checks must accept an Operator on any member; `Operator.when` string
   conditions resolve via the same `condition_lookup` as today.
5. **Tests**: three-surface parity (decorator/declarative/programmatic),
   checkpoint interrupt+resume with real sqlite (copy
   `test_checkpoint_portal.py` conventions), exactly-once member assertion
   (port the spike's crux test to the real surface), lint/`operator_check`
   interplay, and the spike tests stay as the mechanics pin.
6. **Docs**: website `concepts/portal.mdx` + hitl page; note the D4 lift in the
   decision-log's renamed-construct addendum style.

## Linkage

- **7t7tf (Command(goto) ratification)**: this spike proves the D4 cost is
  temporary — an argument FOR ratifying `Command(goto)`.
- Prior art reused: agent/act tool-gate (`_wiring.py:1035+`), turn-boundary
  idempotency, `hitl.ask_human` typed resume, `_add_operator_check` predicate
  shape.
