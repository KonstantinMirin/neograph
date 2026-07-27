# Adversarial review round 2: `portal-tool-triggered-handoff-2026-07-27.md` (revised)

Reviewer: independent pass, read-only, re-derived against real source
(`src/neograph/_agent_cycle.py`, `factory.py`, `_wiring.py`, `tool.py`,
`compiler.py`, `test_guards_llm_runtime.py`, `test_portal_cross_subconstruct.py`).
Verifies the four resolutions claimed in round 1's follow-up (§9 of the
revision), not the revision's own narrative.

---

## Finding 1 — Tool-binding fix (item 1). PARTIALLY-CONFIRMED — right outcome, wrong call chain described.

Read `_agent_cycle.py` in full. Real structure:

- `_agent_caller(prep_prep, node, budget)` (line 240) is a **top-level module
  function with explicit positional params**, not a closure capturing `node`/
  `tracker` implicitly — confirms the design's claim (b) that adding a 4th
  param is clean.
- `_tracker_from_budget`/`ToolBudgetTracker.can_call` (`tool.py:224`): an
  unregistered tool name (`self._budgets.get(tool_name, 0)`) defaults to
  budget `0` → `can_call` returns `True` (treated as unlimited). So even if a
  handoff tool name were accidentally run through `can_call`, nothing breaks —
  confirms (c): the "never call `tracker.can_call` for handoff tools" choice
  is a clean simplification, not a required workaround for a crash.
- **The design's stated chain is factually wrong about which functions carry
  the parameter.** `_agent_caller` is called directly inside `agent_body`/
  `aagent_body` (lines 573, 584) — nested closures defined *inside*
  `make_agent_cycle_bodies`, as siblings of the `_build_turn_prep`/
  `_abuild_turn_prep` call, not consumers of its return value for tool
  binding. `_build_turn_prep`/`_abuild_turn_prep`/`_turn_prep_kwargs` build
  `_TurnPrep` (LLM instance, rendered prompt, tool_instances dict) — none of
  that path constructs the `active` list `bind_tools` receives; that's built
  fresh, separately, in `_agent_caller`. **Only two functions need signature
  changes**: `make_agent_cycle_bodies` (new `handoff_portal: Portal | None =
  None` param, closed over by `agent_body`/`aagent_body`) and `_agent_caller`
  (new `handoff_portal` param, called with it). `_build_turn_prep`,
  `_abuild_turn_prep`, and `_turn_prep_kwargs` need **zero** changes — the
  design's §3.1 diagram routing `handoff_portal` through all four is not what
  the real call graph requires.

This doesn't invalidate the fix — the actual implementation is *simpler* than
described, and the "does it reach `bind_tools`" question is genuinely
resolved (the tool ends up in `active`, which is passed to
`llm.bind_tools(active)` at line 250). But "verify against these functions'
REAL current signatures and call sites" turns up a concrete discrepancy: an
implementer following §3.1 literally would touch three functions that don't
need touching, and would likely get confused when `_build_turn_prep` has no
natural place to *use* `handoff_portal` (it doesn't need it at all — only
`agent_body`/`aagent_body`'s closure and the `_agent_caller` call do).

**Caller-preservation check**: `make_agent_cycle_bodies` has exactly two call
sites — `_wiring.py:1325` (`_add_agent_cycle`, plain non-mesh agent/act
nodes) and `factory.py:496` (`make_portal_agent_cycle_fn`, today's
`trigger="output"` mesh path). Neither passes anything resembling
`handoff_portal` today, so a new param with default `None` is zero-behavior-
change for both, confirmed by reading both call sites directly.

**Verdict: PARTIALLY-CONFIRMED.** The capability genuinely reaches
`bind_tools` under the design's plan, and the parameter-threading composes
cleanly with real call sites and defaults — but the doc's own claimed chain
(`make_agent_cycle_bodies → _build_turn_prep/_abuild_turn_prep →
_turn_prep_kwargs → _agent_caller`) misdescribes the real call graph.
Correct before implementation: `make_agent_cycle_bodies → (closure) →
_agent_caller`, only two signatures change.

---

## Finding 2 — Structural guard framing (item 2). CONFIRMED, concretely testable.

`compiler.py:426-431` (`_print_dag_summary`) already introspects
`compiled.get_graph().edges` (LangGraph's own edge-inspection API,
`edge.source`/`edge.target`/`edge.conditional`) as an established pattern in
this codebase. A guard asserting "`{node}__tools` has no static outgoing edge
to `{node}__agent` when `is_tool_triggered`" is directly expressible against
this same API (or the pre-compile `StateGraph.edges` set `add_edge`
populates) — unambiguous, not two-implementers-diverge vague. The design
doesn't cite the exact API, but the mechanism is discoverable and singular
given existing precedent in the same file.

**Verdict: CONFIRMED.**

---

## Finding 3 — Budget exemption (item 3) and Finding 4 — idempotency exemption (item 4). CONFIRMED, zero new code.

- `idempotent_by_tool = {spec.name: bool(getattr(spec, "idempotent", False))
  for spec in (node.tools or [])}` — still at `_agent_cycle.py:563`, keyed
  exclusively off `node.tools`. `_idempotent_repeat_key` (line 418) returns
  `None` when `idempotent_by_tool.get(name)` is falsy. A handoff tool name is
  never a key in this dict (never in `node.tools`), so the repeat-cache path
  is a structural no-op for it — matches the design's claim exactly, line
  numbers still accurate.
- `ToolBudgetTracker.__init__` (`tool.py:217-222`) only registers budgets for
  tools passed to its constructor — `_tracker_from_budget` passes `node.tools`
  only. A handoff tool is never in that list, so it never appears in
  `_budgets`/`_counts`, and (per Finding 1 above) `can_call` on an unknown
  name defaults to `True` rather than erroring — no pre-registration
  requirement exists, so the exemption needs no skip-list anywhere.
  `budget["calls"] = dict(tracker._counts)` (line 645) also never picks up a
  handoff tool's count, so no bookkeeping leak either.

**Verdict: CONFIRMED for both** — genuinely free consequences of "handoff
tools are absent from `node.tools`," exactly as claimed, re-derived
independently against current line numbers and logic (not just re-reading
the design's own citations).

---

## Finding 5 — NEW-GAP-CHECK: `_synthesize_handoff_tools` implementation reality. PARTIALLY-CONFIRMED — real precedent exists but the doc doesn't cite it.

`tool.py` already builds ad hoc, non-factory-registered LangChain tool
objects directly via `StructuredTool(name=, description=, args_schema=,
func=..., coroutine=...)` — see `resource_reader` (line 379) and
`_build_read_blob` (line 408), both bypassing `register_tool_factory`
entirely. This is a real, directly reusable, checkable pattern for "a bare
LangChain tool object with no side-effecting body": a `StructuredTool` with a
trivial stub `func` (e.g. `lambda **_: "transfer requested"`, never actually
invoked because `_tool_call_precheck`'s new handoff branch intercepts before
`tool_instances.get(name)`). One nuance the design doesn't surface:
`StructuredTool` requires at least one of `func`/`coroutine` non-`None` to
construct — a schema-only tool with neither is not directly constructible
this way, so "no side-effecting body" must mean "a body that runs a no-op
stub, never actually reached at runtime" rather than "no body at all." This
is a minor precision gap in the prose, not a blocking one — the mechanism is
real and available, just uncredited in the doc (an implementer who doesn't
grep `tool.py` first might genuinely wonder how to build a schema-only
tool).

**Verdict: PARTIALLY-CONFIRMED** — implementable, not hand-waved, but the doc
should cite `tool.py`'s `resource_reader`/`_build_read_blob` `StructuredTool`
pattern as precedent and correct "no side-effecting body" to "a body whose
result is never observed, since the handoff branch short-circuits before
invocation."

---

## Finding 6 — §2 narrowed justification: reachable or vacuous? CONFIRMED reachable, not vacuous.

Traced `compiler.py:251-303`: the per-`Construct` compile loop detects a
contiguous Portal run (`_add_portal_mesh`) and a `Construct`-typed item
(`_add_subgraph` → recursive `compile()`) as **siblings in the same loop**,
handled independently. `_add_subgraph` recurses into `compile()` on the
inner `Construct`, which builds its own fresh `StateGraph` and runs the
**same** top-level loop again — including the same Portal-mesh-detection
branch. So "a Portal mesh sitting inside a sub-construct nested one level
inside an outer Construct" (§2's scenario: the *whole mesh* one level
deeper, as opposed to C1's *one mesh member* being a Construct) is reachable
today via ordinary Construct nesting + an internal Portal mesh — it needs no
C1 support at all, since C1 is about a mesh *member* being a Construct
(`test_portal_cross_subconstruct.py`, currently red/WIP for do0d9), a
different and unrelated axis. I did not find an existing test exercising
this exact "mesh nested one level inside a sub-construct" shape, but the
architecture makes it trivially reachable by construction (recursive
`compile()`, no special-casing needed) — so the §2 narrower invariant is
correct-and-exercisable, not correct-but-vacuous.

**Verdict: CONFIRMED.**

---

## Minor note

§6's guard 3 description ("the `IR_FIELDS` frozenset ... `{handoff_param,
handoff_channel}`") slightly overstates precision: the real frozenset at
`test_guards_llm_runtime.py:1018` is `{"fan_out_param", "oracle_gen_type",
"handoff_param", "handoff_channel"}` — four fields, not two. The design's
claim ("stays exactly that frozenset") still holds correctly (no field is
added), just double-check the guard text names the real four-element set
when writing the test, not a fictional two-element one.

---

## Overall verdict: **NEEDS-REVISION** (minor) — not REJECT, not ready to implement byte-for-byte as written.

All four originally-blocking/should-fix findings from round 1 are
substantively resolved: the handoff tool genuinely reaches `bind_tools`
(Finding 1), the static-edge framing is now accurately a neograph convention
with a concretely testable guard (Finding 2), and both budget/idempotency
exemptions are real, zero-new-code consequences of keeping the tool out of
`node.tools` (Findings 3-4), independently re-verified against current line
numbers and logic. No new *blocking* gap was found (Finding 5's precedent
gap and Finding 6's reachability are non-blocking documentation
improvements).

However, §3.1's own call-chain diagram — the exact thing this round was
asked to re-verify "concretely" — misdescribes the real call graph
(Finding 1): the fix is real but simpler than stated, threading through 2
functions, not 4. Shipping the spec as literally written risks an
implementer wiring `handoff_portal` through `_build_turn_prep`/
`_abuild_turn_prep`/`_turn_prep_kwargs` unnecessarily, or stalling when those
functions have nothing to do with it.

### Prioritized action list

1. **(Should-fix, before implementation)** Correct §3.1's chain to:
   `make_agent_cycle_bodies(handoff_portal=...)` → `agent_body`/`aagent_body`
   close over it → `_agent_caller(..., handoff_portal=...)`. Drop
   `_build_turn_prep`/`_abuild_turn_prep`/`_turn_prep_kwargs` from the
   threading list entirely.
2. **(Should-fix, cheap)** Cite `tool.py`'s `resource_reader`/
   `_build_read_blob` `StructuredTool(func=, coroutine=)` pattern in §3.1 as
   the concrete precedent for synthesizing the handoff tool, and soften "no
   side-effecting body" to "a stub body never invoked, since
   `_tool_call_precheck` intercepts before `tool_instances.get(name)`."
3. **(Nit)** Fix the `IR_FIELDS` frozenset citation in §6 guard 3 to the real
   four-element set.
4. Once (1) is corrected, this is ready for implementation — no further
   design-level rework needed; (2)-(3) are documentation polish that can be
   folded in during implementation rather than blocking a third review round.
