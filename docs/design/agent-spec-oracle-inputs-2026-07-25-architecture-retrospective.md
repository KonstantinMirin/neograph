# Architecture retrospective: why the Agent Spec export surface took 6+ discovery rounds to converge (neograph-m57mn and its lineage)

Date: 2026-07-26
Status: retrospective / process analysis (no code changes)
Scope: `neograph-2ev48` / `neograph-154vl` (Agent Spec interop epics), specifically the
`to_agent_spec()` export module `src/neograph/_agent_spec.py`.

This document does **not** re-litigate whether the `m57mn` fix (commit `a109a7e`) is
correct — that is settled (design pass + adversarial review + implementation + all
gates green). It answers a different question the maintainer asked directly:
**what underlying architecture violations let this flaw recur through six discovery
rounds despite planning, decomposition, and a test harness?**

---

## 1. Timeline reconstruction

| Date | Event | What broke | How it was found |
|---|---|---|---|
| 2026-07-23 | `2ev48`/`154vl` marked "substantially complete" | — | closed prematurely, then reopened same window |
| 2026-07-24 | `neograph-ozxqw` | `to_agent_spec`'s dict-form fan-in used the raw inputs-dict **key** as a `DataFlowEdge` Property title instead of resolving the real Property via `_properties_for` | matrix probe on `@node`'s primary dict-form shape |
| 2026-07-24 | `neograph-qtfof.1` (sibling of ozxqw, found *while fixing* ozxqw) | dict-form fan-in loop didn't skip the Each `fan_out_param` key before treating it as an upstream node name | trace-similar step of ozxqw's own fix |
| 2026-07-24 | `neograph-qtfof.2` (same sibling wave) | `_lower_loop`'s self-edge assumed a bare destination Property title; dict-form inputs need a `{key}.` prefix | same trace-similar step |
| 2026-07-24 | `neograph-qtfof.3` (same wave) | round-trip re-derived `scripted_fn` from the exported node's hyphenated `.name`, breaking `@node` self-registration | same wave |
| 2026-07-24 | `neograph-hf505` (found *while fixing* qtfof.1) | Each-modified node with a non-fan-out **context** input: `MapNode` infers its own inputs as `iterated_`-prefixed from the sub-flow `StartNode`; the emitted edge used the bare/dict-prefixed title instead | qtfof.1's own trace-similar step surfaced a "deeper, separate bug" explicitly logged as out of scope for qtfof.1 |
| 2026-07-24 | `neograph-3lk2l` (found independently) | round-trip type-identity loss for `map_over=` Each fan-out receivers (synthesized `AgentSpecType_` classes didn't match the producer's element type) | independent matrix probe, "not covered by the closed qtfof.1 (export-only)" |
| 2026-07-24 | `neograph-wqb5t` (found alongside 3lk2l) | `spec_types.register_type`'s idempotency check compared field **names** only, not field **types** | same matrix probe |
| 2026-07-24 | **`neograph-sdfgz` filed** | — a parametrized **modifier × input-shape × round-trip** coverage matrix, filed explicitly because *"the fix-one-find-a-sibling pattern... shows the surface has NOT converged"* | maintainer's own diagnosis of the preceding chain |
| 2026-07-24→25 | `neograph-wq9nn` filed, NOT wired as a hard blocker | Loop/Operator registered-**name** conditions don't round-trip (inline/expression conditions do) | surfaced by the sdfgz sweep itself; "filed so it isn't lost" |
| 2026-07-25 | `neograph-m57mn` found **by the sdfgz matrix** | `_lower_oracle` built every Oracle variant as an unconditional `LlmNode`; pyagentspec's `LlmNode`/`Agent` couple `inputs=` to literal `{{ }}` placeholders in the prompt text, which neograph's `${var}`/template-ref prompts never produce | the matrix's `oracle-single`/`oracle-dict` cells, `xfail(strict=True)` |
| 2026-07-25 | Design doc written, identifies a **second** bug (mode-dispatch asymmetry) as root cause | `_lower_oracle` never dispatched on `node.mode` the way `_lower_node` does | design pass, reading `_lower_node` vs `_lower_oracle` side by side |
| 2026-07-25 | **Adversarial review finds a 4th, previously-unnamed guard call site** | `_lower_oracle`'s `merge_prompt` branch is gated on `oracle.merge_prompt` truthiness, **not** `node.mode` — a scripted-mode node can legally carry `merge_prompt=`, and the merge node hits the identical placeholder wall, 100% of the time, with zero test coverage (no matrix cell) | reviewer traced `_lower_oracle` "by hand, line by line" rather than re-deriving from the bug title |
| 2026-07-25 | Fix lands (`a109a7e`) | Option A (mode dispatch in `_lower_oracle`) + Option B (`_check_placeholder_inputs` guard) at all 4 sites + a `loader.py` round-trip fix (discovered mid-implementation: `_reconstruct_oracle_group` assumed every variant is an `LlmNode`) | TDD, all 16 matrix cells green |

Eight independent bugs (ozxqw, qtfof.1, qtfof.2, qtfof.3, hf505, 3lk2l, wqb5t, m57mn)
plus one deferred gap (wq9nn) plus one *unplanned* fix inside m57mn's own
implementation (`loader.py`), across roughly 36 hours, all in one ~1000-line module.

---

## 2. Root architectural causes

### 2.1 A second, unshared node-mode dispatch table (DRY violation, the same shape CLAUDE.md names elsewhere)

`_lower_node` (`_agent_spec.py:214-254`) is the canonical per-`node.mode` dispatch:
`scripted`/`raw` → `ToolNode`, `think` → `LlmNode`, `agent`/`act` → `AgentNode`+`Agent`.
`_lower_oracle` (`_agent_spec.py:355-497`) is a **second**, independently-written
lowering function that has to make the exact same "what pyagentspec primitive does
this generation step lower to" decision for Oracle's N variants — and until
`a109a7e`, it did not consult `_lower_node`'s dispatch or share a helper with it at
all; it unconditionally built `LlmNode`. This is precisely the pattern the project's
own CLAUDE.md calls out and prohibits elsewhere in the codebase:

> "One validator walker, not two." (`_check_fan_in_inputs`)
> "the single source of truth... Do NOT re-inline modifier checks elsewhere." (`effective_producer_type`)

`_lower_oracle` re-inlined a node-mode-to-primitive decision instead of calling
`_lower_node`'s dispatch (or a shared table both call). The design doc and the
post-fix code both acknowledge this directly — the fix's own code comment at
`_agent_spec.py:385-389` says the fix is there specifically because "an unconditional
LlmNode was itself the root cause... mirroring `_lower_node`'s existing dispatch" —
but the fix, notably, **still does not converge the two dispatches into one shared
function**. It duplicates the three-way branch a second time inline inside
`_lower_oracle` (`_agent_spec.py:390-433`), now with variant-naming and metadata
differences layered on top. The duplication that caused the bug is fixed in content
but not eliminated in structure — a second bug of the same shape (e.g. a future
`agent`/`act`-mode Oracle lowering, or a fifth modifier that needs the same
dispatch) can reintroduce the identical class of divergence, because there is still
no single function both `_lower_node` and `_lower_oracle` call.

**Was there a structural guard that could have caught this?** No, and the reason is
itself diagnostic: `tests/test_guards_agent_spec_lowering.py` — the one guard file
whose docstring explicitly says its purpose is "a modifier-lowering function in
`_agent_spec.py` silently drops a Node field" — is composed of **one test class per
already-discovered bug** (`TestAgentActModeLowersToAgentNode` for `i3zsh.1`,
`TestDictFormFanInResolvesRealPropertyTitles` for `ozxqw`,
`TestDictFormFanInSkipsFanOutReceiver` for `qtfof.1`,
`TestLoopSelfEdgeResolvesDictFormDestinationTitle` for `qtfof.2`). Every one of
these guards is a **symptom-shaped regression pin** (grep for a specific string
that must/must-not appear), not a **structural invariant** stated once (e.g. "every
place a Node lowers to a pyagentspec primitive must consult one shared
mode-dispatch table"). None of the four existing guard classes would have caught
`_lower_oracle`'s unconditional-`LlmNode` bug, because none of them assert anything
about `_lower_oracle` at all — the guard file's own scope silently stopped at the
bugs already found, not at the general shape of the risk.

### 2.2 The `LlmNode`/`Agent` inputs↔placeholder coupling was never named as a single testable invariant

pyagentspec's `LlmNode._get_inferred_inputs` / `Agent._get_inferred_inputs`
(`llmnode.py:92`, `agent.py:61`) both couple `inputs=` to literal `{{ }}` jinja
placeholders scanned out of `prompt_template`/`system_prompt`
(`ComponentWithIO._validate_inputs`, `component.py:1672-1683`, run as a Pydantic
validator on the primitive instance itself, invisible to any enclosing `Flow`). This
is a **downstream API constraint** that every `LlmNode`/`Agent` construction site in
`_agent_spec.py` must independently satisfy — but before `a109a7e` there was no
single, named, enforced rule for it anywhere in the module. It was an implicit
assumption baked separately into each of (at least) four call sites:

1. `_lower_node`'s `think` branch (`:222-230`)
2. `_make_agent`, covering `_lower_node`'s `agent`/`act` branch (`:257-274`)
3. `_lower_oracle`'s `think`-mode variant loop (`:390-401`)
4. `_lower_oracle`'s `merge_prompt` branch (`:436-457`)

Three of these four (2, 3, 4) were **completely untested** before `m57mn`'s
implementation — the doc's own §1 states this explicitly ("No existing test
exercises an agent/act-mode node with real upstream `Node.inputs`"; "the only
existing think+Oracle export test... uses a single node with no upstream
`Node.inputs`"). This is the exact same "single source of truth vs. re-inlined
checks in N places" pattern CLAUDE.md names for `effective_producer_type`
("teach `effective_producer_type` about the new rule... do NOT re-inline modifier
checks elsewhere") — except here there was no `effective_producer_type`-equivalent
function to teach in the first place. The invariant ("any real `Node.inputs`/
`gen_outputs` reaching an `LlmNode`/`Agent` construction must either be provably
placeholder-covered, or fail loud") existed only in the reviewer's head, discovered
by inspection, not encoded as a single checked rule the four call sites shared by
construction. `_check_placeholder_inputs` (the `a109a7e` fix) *is* that shared rule
— but it had to be invented reactively, at the same time as the bug that proved it
was missing, rather than being in place before the first `LlmNode`/`Agent`
construction site was ever written (`i3zsh`, weeks earlier).

### 2.3 The modifier × input-shape × round-trip decomposition was derived empirically, through failure, not from the module's own structure

`_agent_spec.py`'s actual shape is a genuine 2-axis structure: `_lower_node`'s
mode-dispatch table (`scripted`/`think`/`agent`/`act`) **crossed with** each
modifier-lowering function (`_lower_each`, `_lower_oracle`, `_lower_loop`,
`_lower_operator`), further crossed with input shape (none / single-type /
dict-form / dict-form-with-fan-out-receiver / dict-form-with-context-input) and
direction (export / round-trip). That is exactly the decomposition
`tests/test_agent_spec_matrix.py` (`neograph-sdfgz`) now enforces — but it was
filed and built **after** five siblings (ozxqw → qtfof.1/.2/.3 → hf505 → 3lk2l) had
already been found one at a time, each only after the previous one's own
trace-similar step happened to stumble onto it. The matrix's own docstring names
this precisely: *"Every one hid behind the same blind spot: the existing
export/round-trip tests exercise the SINGLE-TYPE-INPUT workaround shape... never
the PRIMARY `@node` shapes real users write."* That blind spot was visible from
day one by simply asking "which of `_lower_node`'s 4 modes × which of the 4
modifier-lowering functions × which of the input shapes has a test?" — a table
that could have been built from the function signatures alone, without running a
single one to failure first.

**Even the matrix, once built, still had a gap along a THIRD axis it hadn't yet
named**: Oracle has two independent *merge strategies* (`merge_fn` vs
`merge_prompt`), and the matrix's `oracle-single`/`oracle-dict` cells are both
`merge_fn`-only. There was no `oracle-merge-prompt` cell. This is exactly why the
adversarial reviewer's 4th-site finding was possible at all — a currently-100%-broken,
completely-uncovered path survived not just the design doc but the very matrix
built to end this pattern. **The decomposition converged on "modifier × input-shape ×
round-trip" but missed "and also modifier-specific CONFIG variant" as a 4th axis** —
a second-order instance of the same discovery-through-failure pattern the matrix
itself was meant to close.

### 2.4 The bug report's own framing anchored the design pass away from the 4th site

`m57mn`'s title — "Oracle-modified node with ANY external input fails export" — and
its body frame the defect entirely in terms of `Node.inputs` reaching an `LlmNode`.
The design doc, asked to "read the full `_agent_spec.py` file" and enumerate call
sites, produced exactly the sites reachable by re-deriving the bug from that
framing: `_lower_node`'s `think`/`agent`/`act` branches (same field, `node.inputs`)
and `_lower_oracle`'s variant loop (same field, gated by the same `node.mode`
value the ticket names). The `merge_prompt` branch is gated on a **different**
field (`oracle.merge_prompt` truthiness) feeding a **different** input set
(`gen_outputs`, not `node.inputs`), and is reachable **independent of `node.mode`
entirely** — a scripted-mode node can legally carry `merge_prompt=`. Nothing in the
ticket's title or problem statement points there. The design doc's own Option A
section symptomatically conflates the two ("the merge node, for the `merge_fn`
branch" — correctly scoping only the `merge_fn` side — while Option B's section
says the guard should apply "once from `_lower_oracle`'s think-mode variant/merge
construction," phrasing that reads `merge_prompt` construction as gated by
`node.mode`, which it is not). The adversarial reviewer found the 4th site only by
tracing `_lower_oracle` **by hand, line by line**, independent of the ticket's own
framing — i.e., by treating "does *any* `LlmNode`/`Agent` construction in this
function receive an unconstrained input set" as the actual search key, not
"where does `node.inputs`/`node.mode` reach an `LlmNode`."

**The general lesson**: a bug report framed around a specific *field* (`node.inputs`
+ `node.mode`) will systematically anchor a design pass's grep/read pattern onto
that field, and will under-search for a **structurally identical** defect gated on
a *different* field feeding the *same* downstream API constraint
(pyagentspec's `{{ }}`-placeholder coupling, which doesn't care whether the input
list came from `node.inputs` or `oracle.gen_outputs`, or whether the gate is
`node.mode == "think"` or `oracle.merge_prompt` truthiness). Tickets for this class
of bug should be framed around the **downstream invariant being violated**
("any Property list reaching `LlmNode`/`Agent` construction without matching
placeholders"), not around the **call site where it was first observed** — the
former is search-complete over grep-for-all-construction-sites; the latter invites
exactly the anchoring that happened here.

---

## 3. Why the design pass + the adversarial review (which JUST happened) still needed a second look to find the 4th site

This is not "the reviewer should have been more thorough" — both passes were
genuinely rigorous (the review confirmed every pyagentspec source claim against
live code and ran empirical repros). The specific process gap is narrower and
mechanical:

- The design doc's own investigative method (§1) was **top-down from the bug
  report**: read `_lower_oracle`'s variant loop (the site the xfail'd matrix cells
  exercise), then reason "what else looks like this" by checking the **other
  known LLM-primitive constructors** (`_lower_node`'s `think`, `_make_agent`).
  That method finds every site keyed on `node.mode`, because it was generated by
  asking "where else does `node.mode` branch to `LlmNode`/`Agent`."  It does not
  find a site keyed on a **different** boolean (`oracle.merge_prompt` truthiness)
  sitting three lines below the variant loop in the **same function**, because
  the search was never "grep for every `nodes_mod.LlmNode(` / `Agent(`
  construction call in the file and check each unconditionally," it was
  "trace `node.mode`'s influence."
- The adversarial review's method **was** bottom-up over the specific function
  (§2: "Confirmed... Verified: with the variant-mode fix hypothetically applied...
  Trace by hand, both cells") — but it only escalated to full-function,
  construction-site-exhaustive tracing for `_lower_oracle` specifically, because
  that was the function under review. It is not stated anywhere in either
  document that the same exhaustive-construction-site sweep was run over
  `_lower_node`/`_make_agent` a second time (it wasn't necessary there, since
  those are single-branch functions with only one `LlmNode`/`Agent` call each);
  the gap was specific to `_lower_oracle` being the one function in the module
  with **two independent** `LlmNode`-construction sites gated on two **different**
  conditions (`node.mode` for variants, `oracle.merge_prompt` for merge) — a
  shape none of the other lowering functions have, and therefore a shape neither
  the design doc's method nor a first skim of the review would think to check
  for by generalizing from any other function in the file.
- **The actual, reusable process fix**: neither pass ran the mechanical query "for
  every function in `_agent_spec.py`, list every `nodes_mod.LlmNode(` and
  `Agent(` construction call site, and for each one, name the boolean condition
  that gates reaching it." That query is exhaustive by construction (there is a
  finite, small number of call sites in a ~1000-line file) and would have
  surfaced the `merge_prompt` branch as a 4th, distinctly-gated site on the first
  pass, without needing the review's line-by-line trace to catch what the design
  pass's mode-centric search missed. This is a **repeatable technique**, not a
  "be more careful" exhortation — see the guard recommendation in §4.1 below,
  which turns this exact query into a permanent structural test.

---

## 4. Concrete, actionable recommendations

### 4.1 New guard: `test_guards_agent_spec_llm_construction_sites.py` — exhaustive-construction-site enumeration + guard-call pairing

Add a **positive, exhaustive** structural guard (not another symptom-shaped regex
pin like the four in `test_guards_agent_spec_lowering.py`) that:

1. AST-walks `_agent_spec.py` and collects **every** call expression whose callee
   resolves to `nodes_mod.LlmNode`, `Agent` (the pyagentspec class, not neograph's
   `Agent`-shaped types), or any future primitive added to
   `pyagentspec.templating`'s placeholder-inference family (this makes the guard
   forward-compatible with a new primitive being added to that family later, not
   just pin the current four).
2. For each such call site, asserts that a call to `_check_placeholder_inputs(`
   appears **textually before it, in the same function body** — i.e. the guard
   verifies the *pairing*, not just that the helper exists somewhere in the file.
   This is the exhaustive version of what the design pass's method and the first
   review pass both did manually and incompletely.
3. Includes an explicit count assertion: `assert len(construction_sites) == 4`
   (or whatever the current count is) with a comment that says *"if this number
   changes, a new call site was added — go verify it either has a paired
   `_check_placeholder_inputs` call or is deliberately exempt (e.g. Portal mesh
   Agents at `:760`, which intentionally pass `[]` for inputs and are exempt by
   construction, not by omission)."* This converts "did we forget a guard call"
   from a silent gap into a loud, unmissable diff the next time someone adds an
   `LlmNode`/`Agent` construction site to this module — which is exactly the
   failure mode that let the `merge_prompt` branch go three lines unguarded next
   to the (correctly guarded) variant loop.

This single guard would have caught the `merge_prompt` gap **before** the
adversarial review had to find it by hand, and it survives the next Oracle-like
modifier or the next LLM-primitive pyagentspec adds.

### 4.2 Refactor `_lower_oracle`'s per-mode variant dispatch to share `_lower_node`'s table now — a real design change, not "consider it"

Recommend doing this now, not deferring it:

- Extract a single function, e.g. `_lower_generation_step(node: Node, *, name: str,
  outputs: list[Property], metadata: dict) -> SpecNode`, that contains **exactly
  one** copy of the `think`/`agent-act`/`scripted-or-raw` three-way dispatch
  (`ToolNode`+`ServerTool` / `LlmNode` (guarded) / `AgentNode`+`Agent` (guarded)),
  parameterized by the caller-supplied `name` (so Oracle can pass
  `f"{node.name}__variant_{i}"` while `_lower_node` passes `node.name`) and
  `metadata` (so Oracle can attach its group/variant markers).
- `_lower_node` becomes a thin wrapper: build `inputs`/`outputs` from
  `node.inputs`/`node.outputs`, call `_reject_unrepresentable_fields`, then call
  the shared function with `name=node.name`.
- `_lower_oracle`'s variant loop calls the same shared function per variant tier,
  passing `model_tier or node.model` into a synthesized `Node` the way it already
  does today for `_make_llm_config`.
- This is not a cosmetic rename — it structurally **removes** the possibility of
  the two dispatches drifting again (e.g. if `agent`/`act` Oracle variants are
  ever implemented, they get `_make_agent`'s exact lowering for free instead of a
  third hand-written copy of the `AgentNode`+`Agent` construction). It directly
  answers the CLAUDE.md-mirrored diagnosis in §2.1: the fix that shipped
  duplicated the *content* correctly but left the *structure* duplicated: this
  recommendation removes the structural duplication itself, and the guard in
  §4.1 makes any future re-duplication loud immediately rather than silent until
  the next matrix sweep.

### 4.3 Add the missing matrix axis: modifier-specific CONFIG variants (`merge_fn` vs `merge_prompt`), not just modifier × input-shape

Extend `tests/test_agent_spec_matrix.py` with an `oracle-merge-prompt` cell
(`Oracle(n=2, merge_prompt=...)`, crossed with at least single-type and dict-form
inputs on the generating node) so the matrix's own stated purpose — "the permanent
regression guard that the surface stays converged" — actually covers the
config-variant axis that let the 4th site go uncovered even after the matrix
existed. Generalize the naming: the matrix's docstring should name this as a
4th, explicit axis (**modifier × input-shape × round-trip × modifier-specific
config variant**) rather than leave it implicit, so the next modifier with two
internal strategies (if any) is checked for the same gap during its own design
pass, not discovered by a future adversarial review repeating this exact incident.

### 4.4 Ticket-framing convention (process, not code): frame NO-REPR/coupling bug reports around the violated invariant, not the first call site found

Per §2.4's finding: when filing a bug about a downstream-API coupling constraint
(pyagentspec's placeholder inference, or any similar "field X's shape gates
whether field Y's values are accepted" coupling), the ticket title and problem
statement should name the **general invariant being violated** ("any Property
list reaching an `LlmNode`/`Agent` construction must satisfy pyagentspec's
placeholder-coupling constraint") rather than the **specific call site** where it
was first observed ("Oracle-modified node... fails export"). The former framing
is naturally search-complete (grep all construction sites); the latter invites
exactly the anchoring the design pass exhibited here. This is a one-line addition
to how future `_agent_spec.py`/coupling-constraint bugs get filed, not a new
process step.

---

## 5. What this retrospective does NOT claim

- It does not claim the `m57mn` fix is wrong — it isn't; all 16 matrix cells pass,
  the guard sites are correctly identified post-review, and the `loader.py` fix
  correctly handles the round-trip inverse.
- It does not claim "more tests" as a generic prescription — §4.1/4.3 name the
  exact guard/matrix cell to add and what each specifically asserts.
- It does not claim the design pass or adversarial review were careless — both
  were methodical and empirically grounded; §3 identifies the specific mechanical
  gap in the design pass's search method (mode-centric trace vs. exhaustive
  construction-site enumeration), which is a repeatable, fixable technique, not
  a diligence failure.
