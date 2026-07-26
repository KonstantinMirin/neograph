# Agent Spec export: Oracle/LlmNode input-placeholder coupling (neograph-m57mn)

Date: 2026-07-25
Status: design / decision record (no code changes)
Author: Constantine Mirin (with agent-assisted analysis)
Blocks: neograph-2ev48, neograph-154vl

---

## 1. The bug, grounded in real pyagentspec source

`to_agent_spec()` (`src/neograph/_agent_spec.py`) lowers an `Oracle`-modified
node via `_lower_oracle` (lines ~311-407) into N variant nodes + 1 merge node.
Today `_lower_oracle` builds every variant **unconditionally as
`pyagentspec.flows.nodes.LlmNode`**, passing `inputs=inputs or None` (the
node's real upstream `Node.inputs`, via `_properties_for`) and
`prompt_template=node.prompt or ""`.

`LlmNode` overrides `ComponentWithIO._get_inferred_inputs`
(`.venv/.../pyagentspec/flows/nodes/llmnode.py:92`):

```python
def _get_inferred_inputs(self) -> List[Property]:
    return get_placeholder_properties_from_json_object(getattr(self, "prompt_template", ""))
```

`get_placeholder_properties_from_json_object` (`pyagentspec/templating.py:35`)
scans the string for literal jinja-style `{{ name }}` placeholders — nothing
else. `ComponentWithIO.model_post_init` / `_validate_inputs`
(`pyagentspec/component.py:1589,1672-1683`) then runs
**at `LlmNode.__init__` time, as a Pydantic model validator on the `LlmNode`
instance itself** — it does not look at any enclosing `Flow`, `StartNode`, or
`DataFlowEdge`. It compares the `inputs=` you passed against
`_get_inferred_inputs()` and calls `_validate_no_extra_property`
(`component.py:1625`), which raises:

```
ValueError: The LlmNode component received a property titled '<x>', but
did not expect any properties          # when 0 placeholders
ValueError: ... but expected only properties with the titles: [...]   # when >0
```

neograph's own prompt syntax is `${var}` (inline) or a template-ref name
(`node.prompt` resolved via the prompt-compiler seam) — **never** pyagentspec's
`{{ var }}` jinja syntax. So a real `Node.inputs` essentially never survives
as an `LlmNode.inputs=` regardless of mode.

### Confirmed failing today (pinned in `tests/test_agent_spec_matrix.py`)

- `oracle-single` / `oracle-dict`: `@node(outputs=Out, ensemble_n=2, merge_fn=...)`
  with **no** `prompt=`/`model=`. Per the documented `@node` mode-inference rule
  ("neither `prompt=` nor `model=` → `scripted`"), these nodes are
  **mode="scripted"** — zero prompt text, zero placeholders, so *any* input is
  "extra". `xfail(strict=True)` in `TestAgentSpecExportMatrix`/
  `TestAgentSpecRoundTripMatrix`.

### A second latent bug this task surfaced (not yet pinned by any test)

`_lower_oracle` is unconditionally an `LlmNode` factory — it never dispatches
on `node.mode`, unlike `_lower_node` (`scripted`→`ToolNode`, `think`→`LlmNode`,
`agent`/`act`→`AgentNode`, `_agent_spec.py:173-212`). **This mode-dispatch
asymmetry is itself root-causing both currently-xfailed matrix cells**: a
scripted-mode Oracle node should never reach `LlmNode` construction at all.

Tracing the parallel path in `_lower_node` and `_make_agent` shows the
placeholder-coupling bug is **not Oracle-specific and not fully fixed by the
dispatch asymmetry alone**:

- `ToolNode._get_inferred_inputs` (`toolnode.py:80`) returns `self.tool.inputs`
  — i.e. whatever `ServerTool.inputs=` you constructed it with. **No text
  coupling at all.** This is why ordinary scripted nodes, scripted Oracle
  merge (`merge_fn`), and (once the dispatch asymmetry is fixed) scripted
  Oracle variants already work / will work with zero changes to the
  placeholder mechanism.
- `Agent._get_inferred_inputs` (`pyagentspec/agent.py:61-63`) — used by
  `_make_agent`, which every `agent`/`act`-mode node AND every Oracle variant's
  hypothetical agent-mode form would go through — has **the exact same**
  `get_placeholder_properties_from_json_object(self.system_prompt)` coupling,
  scanning `system_prompt` instead of `prompt_template`. `_lower_node`'s
  `agent`/`act` branch and `_make_agent` (`_agent_spec.py:190-230`) pass
  `inputs=inputs or None` into `Agent(...)` exactly like the Oracle bug does.
  **No existing test exercises an agent/act-mode node with real upstream
  `Node.inputs`** (`test_agent_act_node_lowers_to_agent_node_not_tool_node`
  and its sibling use a single top-level node with no upstream producer), so
  this is untested, not verified-working.
- `_lower_node`'s plain `think` branch (`_agent_spec.py:181-188`) has the
  **identical** bug for any bare (non-Oracle) think-mode node with real
  inputs. `test_oracle_lowers_to_variant_nodes_plus_merge_with_group_marker`
  (`test_agent_spec_export.py:511`) — the only existing think+Oracle export
  test — uses a **single node with no upstream `Node.inputs`**, so it never
  exercises this path either.

**Conclusion:** the true defect is "any `LlmNode`/`Agent` construction in
`_agent_spec.py` that receives real `Node.inputs` and a prompt that doesn't
literally spell out matching `{{ }}` placeholders" — Oracle is where it was
*caught* (the matrix's systematic parametrization), but the mode-dispatch
asymmetry is a *distinct*, narrower bug that happens to make both currently
tested cells fail, and the placeholder coupling is a *broader*, currently
untested latent bug across `think`/`agent`/`act` export generally.

---

## 2. Did wrapping (the `_lower_each` StartNode/subflow precedent) get verified against real behavior?

Yes — read directly from source, no throwaway script needed; the validator's
location settles it definitively.

`_lower_each` sidesteps `MapNode`'s own input-inference coupling by declaring
the inner node's inputs on a synthetic `StartNode` inside a sub-`Flow`
(`_agent_spec.py:410-447`). This works for `MapNode` specifically because
`MapNode._get_inferred_inputs` (`mapnode.py:308-309`, delegating to the
module-level `_get_inferred_inputs` helper at `mapnode.py:329`) is
**structural**: it reads `self.subflow.start_node.inputs` and computes
`iterated_{title}` from them. `MapNode`'s inference is defined in terms of its
own wrapping substructure, so populating the `StartNode` changes what
`MapNode` infers.

`LlmNode._get_inferred_inputs` and `Agent._get_inferred_inputs` are **not**
structural in this sense — they are hardcoded to scan a single string field
(`prompt_template` / `system_prompt`) on `self`, full stop. Nothing about a
surrounding `Flow`, `StartNode`, or `DataFlowEdge` is visible to
`ComponentWithIO._validate_inputs`, which runs at `LlmNode.__init__`/`Agent.__init__`
time as a plain Pydantic model validator on that instance in isolation
(`component.py:1672-1683`, `model_post_init` at `:1589`). **Wrapping an
`LlmNode` in a `StartNode`+sub-`Flow` the way `_lower_each` wraps its inner
node does not change what the `LlmNode` itself infers or validates** — the
`LlmNode`'s own `inputs=` argument (whatever real `Node.inputs` we pass it)
is still checked against `{{ }}` scan of its own `prompt_template`,
independent of any enclosing structure. This option, as literally described
in the task ("wrap via StartNode+subflow... sidestepping LlmNode's own
inferred-input coupling"), **does not work** for `LlmNode`/`Agent` — it only
works for components whose `_get_inferred_inputs` is itself structural
(`MapNode`, `FlowNode` — inferred from `subflow.start_node`/`EndNode`s).

---

## 3. Design options

### Option A — dispatch-per-mode in `_lower_oracle` (fixes the tested cells, orthogonal to the placeholder bug)

Make `_lower_oracle` build variants (and the merge node, for the `merge_fn`
branch) by calling the SAME per-mode dispatch `_lower_node` already has,
instead of hardcoding `LlmNode`: `scripted` → `ToolNode` wrapping a
`ServerTool` (inputs declared freely, since `ToolNode`'s inference just
echoes `tool.inputs`), `think` → `LlmNode` (still exposed to the placeholder
bug, see Option C below), `agent`/`act` → reject (Oracle+agent/act is not
today's failure and can stay out of scope / filed separately — no test cell
exercises it).

- **Fixes `oracle-single`/`oracle-dict` completely, with zero placeholder
  hackery**, because both cells are `merge_fn`-only (mode="scripted"): once
  variants lower to `ToolNode`+`ServerTool` (mirroring `_make_server_tool`),
  `_get_inferred_inputs` reads `tool.inputs` and the real `Node.inputs`
  Properties pass straight through, no coupling at all.
- Uniform across all three surfaces automatically: Oracle's mode/inputs shape
  is IR-level (`Node.mode`, `Node.inputs`), identical regardless of whether
  the node was built via `@node`, declarative `Node.scripted(...)`, or
  `Node() | Oracle()`.
- Round-trip: unaffected either way — the existing `_MARK_ORACLE_SPEC` marker
  already carries `n`/`models`/`merge_fn` for reconstruction; switching the
  variant's Agent Spec primitive from `LlmNode` to `ToolNode` for scripted
  mode doesn't change what the marker carries.
- Cost: small — replace the unconditional `LlmNode(...)` construction with a
  per-`node.mode` branch reusing `_make_server_tool`/`_make_llm_config`,
  mirroring `_lower_node`'s existing dispatch. No new marker fields.
- **Does not, by itself, touch the harder think-mode-with-inputs case** (a
  `think`-mode Oracle variant, or a bare think-mode node, still hits the
  `LlmNode` placeholder wall). That is Option B/C/D's job.

This is necessary regardless of which option is chosen for the harder case,
because it is the actual, minimal, in-scope fix for the two currently-xfailed
matrix cells and it removes a genuine architectural inconsistency
(`_lower_oracle` should have dispatched on mode from day one, matching
`_lower_node`).

### Option B — fail loud (`ConfigurationError`) for LlmNode/Agent construction with real inputs and no literal placeholder coverage

Add a shared guard — call it once from `_lower_node`'s `think`/`agent`/`act`
branches and once from `_lower_oracle`'s `think`-mode variant/merge
construction — that computes the literal `{{ }}` placeholder titles actually
present in `node.prompt` (reuse pyagentspec's own
`get_placeholders_from_json_object`, don't hand-roll a second scanner) and,
if any of the real input Property titles are **not** among them, raises
`ConfigurationError` naming the missing placeholder(s) — same shape as the
existing `merge_pre_process`/`raw_fn`/callable-`Loop.when`/callable-
`gate_tools_when` rejections in `_reject_unrepresentable_fields`/`_lower_oracle`/
`_lower_loop`.

- Uniform across `think` Oracle variants, bare `think` nodes, and
  `agent`/`act` nodes (same guard, three call sites) — closes the untested
  latent gap in `_lower_node`'s `think`/`agent`/`act` branches too, not just
  Oracle.
- Zero round-trip risk: nothing is silently transformed; a NO-REPR construct
  simply cannot export, exactly the established pattern.
- Cheapest to implement and audit; smallest new surface area (one guard
  function + call sites, no new marker fields, no synthetic text).
- **Cost is real and possibly severe**: neograph's actual authoring surface
  is `${var}` inline and template-ref prompts, essentially never `{{ }}`
  jinja. Under this option, almost every realistic think/agent/act node with
  real upstream inputs becomes **permanently unexportable** unless the author
  redundantly hand-writes a `{{ var }}` reference into the prompt purely to
  satisfy Agent Spec's inference (awkward dual-syntax authoring burden, and
  brittle — the `{{ var }}` text has no relationship to what neograph
  actually renders into the prompt at runtime, so it can silently drift out
  of sync with the real input set as the node evolves).
- This may be the *right* long-term invariant for a first cut (fail loud
  beats silent corruption), but it substantially narrows what "export a
  realistic pipeline" means in practice, more than the epics blocking on this
  (`2ev48`, `154vl`) may be able to tolerate — worth an explicit product call,
  not just an engineering one.

### Option C — synthesize a placeholder-referencing suffix on `prompt_template`/`system_prompt`

For every `think`/`agent`/`act` `LlmNode`/`Agent` construction that has real
`Node.inputs`, deterministically append a fixed, delimited block to the
prompt/system-prompt text that literally references every input Property
title as a `{{ title }}` placeholder (e.g.
`f"{node.prompt}\n\n<!--neograph:inputs-->\n{{ pa.a }} {{ pb.b }}"`, using a
private sentinel `neograph.naming`-style constant, never an ad hoc string
literal per the marker-key discipline this module already follows). This
makes `LlmNode`/`Agent`'s own structural (text-scan) inference match the real
`Node.inputs` Properties, because — grounded in §2 — the coupling is a pure
string scan (`get_placeholders_from_json_object`, which recurses into
str/bytes/dict/list/etc., `pyagentspec/templating.py:50`), so it has no
opinion on WHY a `{{ }}` token appears in the text.

- Applies uniformly to `think`-mode Oracle variants/merge, bare `think`
  nodes, and `agent`/`act` nodes — one small shared helper
  `_synthesize_placeholder_suffix(prompt_text, input_props)` called from all
  three `_agent_spec.py` sites currently vulnerable.
- Round-trip: lossless IN PRINCIPLE if the importer strips everything after
  the fixed sentinel before restoring `node.prompt` — same "marker carries
  what the primitive can't" convention already used everywhere else in this
  module (here the "marker" is the sentinel-delimited suffix itself, not a
  separate `metadata` key, since the information — the real prompt text vs.
  the synthesized suffix — is trivially splittable back apart).
- **Real cost, and the reason this is not a clean win**: the exported
  `prompt_template`/`system_prompt` field is the literal text a foreign
  (non-neograph) Agent Spec / WayFlow consumer will send to an LLM. Per
  `docs/design/agent-spec-interop-2026-07-09.md`, Agent Spec's whole point is
  cross-ecosystem portability — "a portable flat Agent Spec (markers are
  ignorable by foreign runtimes)". A synthesized `{{ pa.a }} {{ pb.b }}`
  suffix is **not** ignorable in the same sense a `metadata["neograph/*"]`
  key is: it is inline in the actual prompt text a foreign runtime will
  execute, so a WayFlow/other-ecosystem consumer opening this Flow gets a
  visibly corrupted prompt (leaking internal Property titles into what an
  LLM actually sees) unless it also happens to already know to strip the
  sentinel. That is a real, not-fully-portable degradation of exactly the
  property this exporter's Core Invariant is supposed to protect — arguably
  a "silent-ish" seam (it round-trips fine for neograph, but silently
  corrupts the artifact for the declared cross-ecosystem use case) — worth
  weighing against Option B's fail-loud honesty.

### Option D — check for another pyagentspec primitive with unconstrained inputs that still models an LLM call

Explicitly checked and ruled out:

- `ToolNode` has unconstrained inputs (echoes `tool.inputs`) but represents a
  **tool call**, not an LLM call — using it for an Oracle `think`-mode variant
  would misrepresent what actually executes (an LLM generation, not a tool
  invocation), which is itself a Core-Invariant violation (never a second,
  divergent lowering that doesn't reflect what the node does).
- `ApiNode._get_inferred_inputs` (`apinode.py:155-163`) has the SAME
  text-scan coupling, just over more fields (`url`, `http_method`,
  `api_spec_uri`, `data`, `query_params`, `headers`) instead of one — no
  better, and also not an LLM-call primitive.
- `AgentNode`/`Agent` (checked in §1) has the identical coupling via
  `system_prompt` — no escape hatch there either.
- No pyagentspec node type in `.venv/.../pyagentspec/flows/nodes/` combines
  "represents an LLM call" with "unconstrained inputs" the way `ToolNode`
  combines "represents a tool call" with "unconstrained inputs". **There is
  no fourth option hiding in the API surface** — pyagentspec's design
  intentionally couples every LLM-call-shaped component's inputs to its own
  prompt text; that coupling is `ComponentWithIO`'s stated purpose ("save
  time... by not needing to specify inputs explicitly" — `component.py:1644`),
  not an oversight neograph can route around structurally.

---

## 4. Recommendation

**Ship Option A now** (fix the mode-dispatch asymmetry in `_lower_oracle`).
It is unambiguously correct regardless of what's decided for the harder case,
it is IN SCOPE for `neograph-m57mn` (the bug report explicitly flags it as
"may itself be part of the root cause"), and it **fully resolves both
currently-xfailed matrix cells** (`oracle-single`, `oracle-dict` — both
`merge_fn`-only, i.e. mode="scripted") with no placeholder synthesis, no new
marker fields, and no portability compromise. Remove the two `xfail(strict=True)`
markers in `tests/test_agent_spec_matrix.py` once implemented — they are
designed to flip to a hard failure the moment this lands, per the matrix's
own acceptance-signal convention.

**For the harder think-mode-with-inputs case (both Oracle variants AND the
untested plain `_lower_node`/`_make_agent` latent gap), recommend Option B
(fail loud) over Option C (synthesize placeholders), with a documented
follow-up path to revisit Option C ONLY if product pressure from `2ev48`/
`154vl` makes "think-mode nodes with real inputs simply cannot export"
unacceptable in practice:**

- Option B is the only option that doesn't compromise Agent Spec's stated
  cross-ecosystem-portability purpose (the entire reason this exporter
  exists per `agent-spec-interop-2026-07-09.md` §1). Option C's synthesized
  placeholder suffix leaks into the literal prompt text a foreign consumer
  sends to a real LLM — that is a materially different, and worse, kind of
  seam than a `metadata["neograph/*"]` marker a foreign runtime can freely
  ignore. The Core Invariant's explicit list of already-accepted NO-REPR
  gaps (`raw_fn`, `skip_when`, callable `Loop.when`, Oracle merge hooks,
  `renderer`, Portal handoff, callable `gate_tools_when`) shows this exporter
  already treats "genuinely no faithful Agent Spec representation" as a
  first-class, expected, fail-loud outcome — not an exceptional case to be
  engineered around at all costs. A prompt/inputs mismatch with pyagentspec's
  `{{ }}`-coupled inference belongs in that same list.
  Reference: `feedback_production_quality.md` — a band-aid identified in
  production code is immediate work, not deferred polish; the anti-band-aid
  reading of Option C is that synthesizing invisible template text to make a
  validator pass, at the cost of corrupting the exported artifact for its
  stated cross-ecosystem purpose, is exactly the kind of "band-aid that
  leaves a silent seam" the project's North star calls an existential
  defect, not a shortcut worth taking to avoid a fail-loud error message.
- Option B is also cheaper and safer to land: one guard function, reused at
  three call sites, with no new marker/round-trip machinery to get subtly
  wrong. It closes the untested `agent`/`act`/`think` gap in `_lower_node`
  too, which Option A alone does not touch.
- The cost is real (most realistic think/agent/act pipelines with real
  inputs become unexportable until the author also references those inputs
  literally as `{{ }}` in the prompt, which neograph's own `${var}`/
  template-ref authoring never naturally produces) — this is a product
  decision as much as an engineering one. If `2ev48`/`154vl` need think-mode
  Oracle/agent export to actually round-trip usable prompts (not just avoid
  a crash), Option C should be revisited as a SEPARATE, explicitly-scoped
  follow-up bead, not folded into this fix — it needs its own marker-fidelity
  design (what sentinel, how the importer strips it, whether it's
  acceptable that non-neograph consumers see the synthesized suffix) that
  is out of scope for closing `m57mn` cleanly.

**On the mode-dispatch-asymmetry question specifically: it is IN SCOPE for
this bug**, not a separate issue — it's the actual root cause of the two
tests this bug report is anchored on, and fixing `_lower_oracle` to dispatch
per-`node.mode` (mirroring `_lower_node`) is a small, self-contained change
that should land in the SAME task as whichever of B/C is chosen for the
think-mode residual, since both touch the same function and the guard/dispatch
interact (Option A must run BEFORE the think-mode guard, since only `think`
variants reach the placeholder check at all — `scripted` variants exit via
`ToolNode` first).

### Addendum (post-review, 2026-07-25): a fourth guard call site

Independent adversarial review (`agent-spec-oracle-inputs-2026-07-25-review.md`)
found a bug site this doc missed: `_lower_oracle`'s `oracle.merge_prompt`
branch (`_agent_spec.py:350-364`) builds an `LlmNode` merge node **gated on
`oracle.merge_prompt` truthiness, independent of `node.mode`** — a
scripted-mode node can legally carry `merge_prompt=...`, and
`merge_node.inputs=gen_outputs` (virtually always non-empty) then fails the
exact same placeholder-coupling validation, 100% of the time, with **zero**
existing test coverage (no matrix cell, no export test uses `merge_prompt`).
This is NOT "the think-mode residual" as originally phrased — it doesn't
require `node.mode="think"` or any real upstream input on the generating
node at all. Confirmed by empirical repro in the review.

**Revision to the recommendation**: Option B's guard has FOUR call sites, not
three — `_lower_node`'s `think` branch, `_lower_node`'s `agent`/`act` branch
(via `_make_agent`), `_lower_oracle`'s think-mode variant loop, AND
`_lower_oracle`'s `merge_prompt` branch (gated on `oracle.merge_prompt`
truthiness, not variant `node.mode`). Add a dedicated `merge_prompt`
regression test/matrix cell in the same task — this path is currently
invisible to CI in either direction. Reviewer's verdict: **NEEDS-REVISION**,
otherwise sound; ship Option A + all four Option B guard sites in one task,
per the reasoning above (splitting B out would leave a raw pydantic
`ValidationError` on three-going-on-four sites in a module whose convention
is `ConfigurationError` for every other NO-REPR gap).

### Suggested follow-up beads (not filed by this design task; recommend filing at implementation time)

1. Implement Option A (mode-dispatch in `_lower_oracle`) + Option B (fail-loud
   guard for `think`/`agent`/`act` LlmNode/Agent construction with
   placeholder-input mismatch), shared across `_lower_node` and
   `_lower_oracle`. Un-xfail `oracle-single`/`oracle-dict` in
   `tests/test_agent_spec_matrix.py`. Add a NEW matrix-style test (or extend
   the matrix) for a bare `think`-mode node with real inputs and an
   `agent`/`act`-mode node with real inputs, to close the untested-latent-gap
   finding from §1 — both should now hit the Option B `ConfigurationError`,
   not silently corrupt.
2. (Deferred, separate bead, only if product-pressured) Revisit Option C as
   a scoped placeholder-synthesis design for think-mode export, with its own
   round-trip/portability tradeoff write-up.

---

## 5. Three-surface parity note

Oracle's `n`/`models`/`merge_fn`/`merge_prompt` and the node's `mode`/`inputs`
are all IR-level fields (`Oracle`, `Node`), identical regardless of which of
the three API surfaces (`@node`, declarative `Node.scripted(...)`,
programmatic `Node() | Oracle()`) constructed them. Both Option A (dispatch
fix) and Option B (fail-loud guard) operate purely on `node.mode` /
`node.inputs` / `oracle.*` — no surface-specific branching is needed or
appropriate; the fix is automatically uniform across all three, and the
matrix's `oracle-single`/`oracle-dict` cells (built via `@node` +
`construct_from_functions`) plus the existing `test_agent_spec_export.py`
declarative-`Node(...)` Oracle test both exercise the same code path.
