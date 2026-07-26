# Review: `agent-spec-oracle-inputs-2026-07-25.md` (neograph-m57mn)

Reviewer: independent skeptical pass, all claims checked against real source
(`.venv/.../pyagentspec/*`, `src/neograph/_agent_spec.py`,
`tests/test_agent_spec_matrix.py`, `tests/test_agent_spec_export.py`) plus
three empirical repros run against the live code (scripts under
`/private/tmp/.../scratchpad/repro_*.py`, discarded after the run — not
committed anywhere).

---

## 1. pyagentspec source claims (§1 of the doc)

**CONFIRMED** — `ComponentWithIO.model_post_init` (`component.py:1589`) infers
`inputs`/`outputs` when `None`, then `_validate_inputs`/`_validate_outputs`
(`:1672`, `:1690`) run as `@model_validator_with_error_accumulation` methods
**on the instance itself**, calling `_validate_no_extra_property` (`:1625`),
which raises the exact `"did not expect any properties"` / `"expected only
properties with the titles: [...]"` messages quoted in the doc. Nothing here
reads any enclosing `Flow`/`StartNode`/`DataFlowEdge` — validation is entirely
local to the one `ComponentWithIO` instance being constructed.

**CONFIRMED (a)** — `ToolNode._get_inferred_inputs` (`toolnode.py:80`):
`return (self.tool.inputs or []) if hasattr(self, "tool") else []`. Zero text
coupling — it is a structural echo of whatever `tool.inputs` Properties were
passed in. Confirmed empirically too (Oracle `merge_fn` branch, already a
`ToolNode`, already accepts real `gen_outputs` today with no error).

**CONFIRMED (b)** — `LlmNode._get_inferred_inputs` (`llmnode.py:92`) and
`Agent._get_inferred_inputs` (`agent.py:61`) are both pure
`get_placeholder_properties_from_json_object(self.<one string field>)` calls,
with no reference to any surrounding structure. `MapNode._get_inferred_inputs`
(`mapnode.py:308`, delegating to the module helper at `:329`) is genuinely
different in kind — it reads `self.subflow.start_node`'s Properties, i.e. it
*is* structural in terms of its own wrapping sub-`Flow`. The doc's claim that
`_lower_each`'s `StartNode`+subflow trick "does not transfer" to `LlmNode`/
`Agent` is correct and is the right explanation for *why* it doesn't transfer
(different `_get_inferred_inputs` implementation family, not a bug in
`_lower_each` itself).

**CONFIRMED (c)** — Read `apinode.py:155` (`ApiNode`, same string-scan family,
over `url`/`http_method`/`api_spec_uri`/`data`/`query_params`/`headers`
instead of one field) and all node classes under
`pyagentspec/flows/nodes/`. No node type combines "represents an LLM
generation" with "unconstrained/structural inputs" the way `ToolNode` combines
"tool call" with "unconstrained inputs". Also checked `Component.model_config
= ConfigDict(extra="forbid")` (`component.py:157`) and found no
per-field or per-model opt-out flag anywhere in `ComponentWithIO` that would
let a caller suppress `_validate_inputs`/`_validate_outputs` — Option D's "no
escape hatch" conclusion is CONFIRMED, not just asserted.

---

## 2. `_agent_spec.py` + matrix tracing (§2 of the task, §3–5 of the doc)

**CONFIRMED** — `_lower_oracle` (`_agent_spec.py:311`) unconditionally builds
`LlmNode` for every variant regardless of `node.mode`, unlike `_lower_node`'s
mode dispatch (`:173`). Empirically reproduced: a scripted-mode Oracle node
(`ensemble_n=2, merge_fn=...`) with a real upstream input fails at variant
construction with exactly the pydantic message the doc quotes.

**CONFIRMED — merge_fn branch needs no change.** It is already `ToolNode` +
`ServerTool` with `inputs=gen_outputs or None` (`_agent_spec.py:379-395`), and
`ToolNode` has zero placeholder coupling, so this branch already accepts real
`gen_outputs` today. Verified: with the variant-mode fix hypothetically
applied (dispatch scripted→ToolNode), the `oracle-single`/`oracle-dict` cells
have no other bug in their path — the ONLY blocking construction site for
those two specific matrix cells is the variant loop. Option A, scoped exactly
as described, is mechanically sufficient to flip both cells green. **Trace by
hand, both cells**: `build_oracle_single`/`build_oracle_dict` (matrix
`:134`/`:146`) are `@node(outputs=Out, ensemble_n=2, merge_fn="m_combine")`
with no `prompt=`/`model=` → mode="scripted" on the generating node. Variants
→ ToolNode (post-fix), no coupling. Merge → ToolNode already (pre-fix, no
coupling). No third site touches these two cells. **This answers task item 5
directly: yes, Option A alone flips `oracle-single`/`oracle-dict` to
passing, with no residual issue in those two specific cells.**

**PARTIALLY-CONFIRMED / NEW FINDING not named in the doc — the `merge_prompt`
branch is a THIRD, independent bug site, orthogonal to `node.mode`, and it is
currently broken 100% of the time it's exercised, not just "exposed to the
same wall" as the doc's prose implies.**

The doc's Option A section says "the merge node, for the `merge_fn` branch"
(correctly scoping the merge-side fix to only the ToolNode branch) and its
Option B section says to add the guard "once from `_lower_oracle`'s
think-mode variant/merge construction" — this phrasing conflates the merge
node's primitive choice with the *variant's* `node.mode`. In reality,
`oracle.merge_prompt` vs `oracle.merge_fn` is independent of `node.mode`
entirely (`_construct_validation.py:337-340` only validates merge-hook
*signatures* when `merge_prompt` is set — it imposes no `node.mode`
restriction; `Oracle` itself, `modifiers.py:393-463`, requires exactly one of
`merge_fn`/`merge_prompt` but never ties either to `node.mode`). So a
**scripted-mode** node (no `prompt=`/`model=`) can legally carry
`ensemble_n=2, merge_prompt=...`, and I confirmed empirically that this
already fails today, independent of any upstream input on the generating node
at all:

```python
@node(outputs=Out, ensemble_n=2, merge_prompt="pick best: ${variants}")
def gen() -> Out: ...
```
→ `ValidationError: The LlmNode component received a property titled 'ok', but did not expect any properties`
(raised at the **merge** node, name matches `f"{node.name}"`, i.e.
`_agent_spec.py:350-364`, NOT the variant loop — `gen` itself has zero
upstream inputs, so the variant nodes here are empty-input and clean).

This happens because `gen_outputs` (from `node.oracle_gen_type` or
`node.outputs`, effectively always non-empty for any real node) is passed as
`merge_node.inputs`, and `oracle.merge_prompt` is real neograph prompt text
(`${variants}` inline syntax) that essentially never contains a literal
`{{ title }}` for each `gen_outputs` property. **Since this branch is gated
on `oracle.merge_prompt` truthiness, not on `node.mode`, Option A's scoped
fix (a per-`node.mode` dispatch over variant construction) does not touch it
at all** — it remains an `LlmNode` unconditionally whenever
`oracle.merge_prompt` is set, for every mode. No existing test (matrix or
`test_agent_spec_export.py`) exercises `to_agent_spec` with `merge_prompt`
set at all — it is a **currently-100%-broken, completely untested** path,
worse than the doc's framing of "the harder think-mode-with-inputs case,"
because it doesn't require `node.mode="think"` or even a real upstream input
on the generating node — merely `merge_prompt` + any non-empty `outputs`
(i.e., virtually every real Oracle+merge_prompt pipeline).

**Verdict on this finding**: this is real scope creep risk if unaddressed —
whichever of Option B/C is chosen for "the harder case" must ALSO cover this
merge_prompt branch explicitly (by `oracle.merge_prompt` truthiness, not by
variant `node.mode`), or `neograph-m57mn` will close while leaving a
still-100%-broken, silently-uncovered Oracle+merge_prompt export path with an
ugly raw pydantic `ValidationError` instead of a clean `ConfigurationError`.
This should be named explicitly in the implementation task, not left to be
inferred from "think-mode residual."

**Additional empirical confirmation of the doc's "untested latent gap"
claim** (task item 2, `_lower_node`'s plain think/agent/act branches):
reproduced independently —
a bare `think`-mode `Node(mode="think", inputs={"prod": Out}, prompt=...)`
fails at `LlmNode` construction with the same message pattern, and a bare
`agent`-mode `Node(mode="agent", inputs={"prod": Out}, prompt=...)` fails at
`Agent` construction (`_make_agent` → `Agent(...)`) with the same message
pattern, both via direct programmatic `Node(...)` construction (bypassing
`@node`'s mode-inference quirks around empty `tools=[]` lists, which is a
separate, minor decorator wrinkle not touching this bug). **CONFIRMED**: this
gap is real, currently unrepresented by any test, and — importantly for the
Option B "cost" framing — since it ALWAYS fails today (there is no
currently-passing case this would newly break), Option B is a strict UX
improvement here (clean `ConfigurationError` replacing a raw pydantic
`ValidationError` bubbling out of `to_agent_spec`), not a new restriction on
previously-working pipelines.

---

## 3. Scope-boundary questions (task item 3)

The doc argues Option A + Option B must land in the same task/PR. Given the
`merge_prompt` finding above, I largely agree, with one adjustment:

- Option A alone is well-defined, safe, and **sufficient to close the two
  currently-xfailed matrix cells** — it could theoretically ship standalone.
- But shipping Option A alone and closing `m57mn` would leave the (newly
  surfaced-by-this-review) `merge_prompt` branch and the `_lower_node`
  think/agent/act branches in their CURRENT state: silently crashing with a
  raw, unfriendly pydantic `ValidationError` from inside `to_agent_spec`,
  which is arguably a worse contributor-experience regression risk to leave
  open under a CLOSED bug than to fix — anyone hitting Oracle+merge_prompt or
  bare think/agent-with-inputs export today gets an opaque, non-neograph
  exception with no `ConfigurationError.build(...)` hint machinery, which is
  inconsistent with every other NO-REPR gap in this module (`raw_fn`,
  `skip_when`, callable `Loop.when`, Portal handoff, callable
  `gate_tools_when` — all fail loud through `ConfigurationError`, not a raw
  third-party `ValidationError`).
- Given the `merge_prompt` finding shows Option B needs THREE call sites more
  than the doc counted (it wrote two: "`_lower_node`'s think/agent/act" +
  "`_lower_oracle`'s think-mode variant/merge" — but the merge call site is
  driven by `oracle.merge_prompt`, independent of variant mode, so it isn't
  really "think-mode" scoped and needs to be named as its own site), landing
  A and B together in the same task is justified — NOT scope creep, because
  without B, Option A's own new code path (scripted Oracle → ToolNode) does
  nothing to fix the adjacent, still-broken `merge_prompt` site sitting three
  lines below it in the same function, and a reviewer/future maintainer would
  reasonably ask "why did we fix the scripted variant path but leave the
  merge path exactly as broken as before, still with a raw pydantic
  exception?" Shipping B's guard alongside A converts silent-ish crashes
  into intentional, documented NO-REPR gaps — consistent with the module's
  established convention, and cheap (one guard function, ~3-4 call sites, no
  new marker fields).
- I would NOT fold Option C into this same task; the doc's deferral of C is
  correctly scoped and justified (see §4 below).

**Revision needed**: the beads task/PR for m57mn should enumerate the guard's
call sites as: `_lower_node` think branch, `_lower_node` agent/act branch
(via `_make_agent`), AND `_lower_oracle`'s think-mode variant loop AND
`_lower_oracle`'s `merge_prompt` branch (four sites, not the doc's implied
three), with the last one gated on `oracle.merge_prompt` truthiness rather
than variant `node.mode`.

---

## 4. Missed simpler option (task item 4)

No Option E found beyond what the doc's Option D already ruled out. I
independently verified `ComponentWithIO` has no validation-skip flag
(`model_config = ConfigDict(extra="forbid")` is a Pydantic-level "no unknown
model fields" setting, unrelated to the `_validate_inputs`/`_validate_outputs`
custom validators — it does not gate them and there's no sibling flag that
does). The doc's Option D search of `apinode.py`/`agent.py`/`toolnode.py`/
`mapnode.py` is complete for the node types under
`pyagentspec/flows/nodes/`; I did not find an additional node type doc missed.
One idea worth naming as a non-option: `_properties_for`'s `"{key}.{field}"`
dict-form naming could theoretically be changed to strip dots and use bare
`field` titles matching a synthesized `{{ field }}` placeholder — but this
degenerates into Option C (still requires synthesizing matching placeholder
text into `prompt_template`) and doesn't sidestep the underlying validator;
it's a variant of C, not a distinct Option E.

---

## 5. Test-file honesty (task item 5)

Already addressed above under §2: **CONFIRMED**, traced by hand — Option A
alone flips `oracle-single`/`oracle-dict` (matrix `:134`/`:146`,
`_XFAIL_EXPORT`/`_XFAIL_ROUND_TRIP` at `:204-205`) to passing with no residual
issue, because both cells are `merge_fn`-only and the merge_fn branch is
already unaffected. The xfail comment's framing ("an Oracle(ensemble) node
with ANY external input fails export") is accurate for what it pins, but the
matrix does NOT have a `merge_prompt` cell at all, so it gives no visibility
into the third bug site found in §2 — that gap should get a new matrix cell
(or at minimum a dedicated regression test) as part of this same task, not
deferred, since it's a currently-100%-broken and currently-invisible path.

---

## 6. Anti-band-aid framing / steelman Option C (task item 6)

The doc's invocation of `feedback_production_quality.md`/the North star to
reject Option C is RIGOROUS, not merely dressed-up preference — the specific
argument (a synthesized `{{ }}` suffix is inline in the literal prompt text a
foreign, non-neograph Agent-Spec/WayFlow consumer will send to a real LLM,
unlike a `metadata["neograph/*"]` key a foreign runtime can freely ignore) is
a correct, falsifiable claim about the concrete mechanism
(`get_placeholders_from_json_object` recurses into the same `prompt_template`
string field a foreign runtime reads verbatim — confirmed at
`templating.py:35-50`), not an appeal to the North star in the abstract. This
is a good example of the CLAUDE.md principle applied correctly: the
"band-aid" here isn't "adds work," it's "makes an already-passing case worse
in a way specific to the exporter's stated purpose," which is the right test.

**Steelman attempted**: the least-bad version of Option C would be to make
the synthesized suffix a genuinely-inert appendix (e.g., trailing HTML-comment
`<!-- neograph:inputs {{a}} {{b}} -->` after a real sentinel, so a
human/foreign-runtime skim of the prompt visibly reads it as decoration, not
instruction) AND gate it behind an explicit `to_agent_spec(..., allow_prompt_synthesis=False)`-style opt-in flag defaulting to Option B's fail-loud
behavior. This is roughly what the doc's own deferred-follow-up path already
proposes ("revisit Option C ONLY if product pressure... needs its own
marker-fidelity design"). I don't think this changes the recommendation —
even an HTML-comment-wrapped suffix still literally appears in the string a
foreign LLM API call sends as its prompt (comments are not stripped by a
generic template-and-send consumer, only by ones that specifically know to
strip them) — so the steelman doesn't actually escape the core objection,
it just makes the leak slightly less semantically confusing. Agreed the doc's
choice to default-reject C and defer it is correct.

---

## Overall verdict: **NEEDS-REVISION** (not REJECT, not ship-as-is)

Option A is sound and should ship as described — verified mechanically
correct and sufficient for the two currently-xfailed matrix cells, with no
residual issue in those two cells specifically.

Option B (fail-loud over Option C) is the right call, for the reasons the doc
gives, reinforced by empirical confirmation that all three doc-named failure
sites are ALREADY broken today (so B is a UX improvement — friendly
`ConfigurationError` replacing raw pydantic `ValidationError` — not a new
restriction on anything currently working).

**What must change before implementation starts:**

1. **Add the `oracle.merge_prompt` branch as an explicitly-named fourth guard
   call site** in `_lower_oracle` (§2 finding above) — it is NOT covered by
   "think-mode variant/merge construction" phrasing as written, because it is
   gated on `oracle.merge_prompt` truthiness, independent of `node.mode`, and
   is currently 100%-broken (not merely "exposed"), with zero test coverage
   today (no matrix cell, no export test uses `merge_prompt`).
2. **Add a `merge_prompt` matrix cell (or dedicated regression test)** so this
   third site has permanent coverage the way `oracle-single`/`oracle-dict`
   do — currently it is invisible to CI in either direction (broken silently,
   not pinned as xfail, not caught if "fixed" by accident, not caught if it
   regresses further).
3. Ship Option A + the (now four-site) Option B guard in the SAME task, per
   the doc's own reasoning (§3 above) — do not split B into a follow-up bead;
   splitting would leave `to_agent_spec` throwing raw pydantic
   `ValidationError`s from three-going-on-four sites inside a module whose
   stated convention is `ConfigurationError` for every other NO-REPR gap.
4. Everything else in the doc (Option D exhaustiveness, Option C rejection,
   the three-surface-parity note, the suggested-beads list) is confirmed
   sound as written and needs no change.
