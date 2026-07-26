# Review: `agent-spec-placeholder-translation-2026-07-26.md` (Option F)

Reviewer: independent skeptical pass. All claims checked against real source
(`.venv/.../pyagentspec/templating.py`, `.../pyagentspec/flows/edges.py`,
`.../pyagentspec/flows/flow.py`, `src/neograph/_agent_spec.py`,
`src/neograph/_placeholders.py`, `src/neograph/_llm_render.py`,
`src/neograph/loader.py`, `src/neograph/spec_types.py`,
`tests/test_agent_spec_matrix.py`) plus empirical repros run against the live
code (inline, not committed).

---

## 1. pyagentspec / neograph scanner claims (§1.1-§1.4 of the doc)

**CONFIRMED** — `TEMPLATE_PLACEHOLDER_REGEXP = r"{{\s*(\w+)\s*}}"`
(`templating.py:15`), pure flat `re.findall`, no Jinja engine, no scoping.
`ComponentWithIO._validate_inputs`/`_validate_no_extra_property` require an
exact set-match against `inputs=`. Matches the doc's §1.1 verbatim.

**CONFIRMED** — `_placeholders.py`'s `DOLLAR_RE = r"\$\{([^}]+)\}"` and
`apply_scanner` are a single shared scanner, reused (not re-implemented) by
`_llm_render.py`, `lint.py`, `prompt.py`. Read `DefaultPromptCompiler` and
confirmed its default `syntax="brace"` (single-brace `{var}`, `BRACE_RE`), not
`${var}` — the doc's §1.3 claim that there are exactly two neograph-native
grammars (inline `${...}`, template-ref `{...}` by default) is accurate.

**CONFIRMED, with an important nuance the doc glosses over (§1.4)** —
`_properties_for`'s dict-form prefixing (`_agent_spec.py:203-210`) is
confirmed to produce EXACTLY one dot level: `f"{key}.{p.title}"`, where
`p.title` itself comes from `model_to_agent_spec_properties`
(`spec_types.py:503-521`), which emits one flat Property per TOP-LEVEL Pydantic
field only (via `model_json_schema()`'s `properties` dict) — it does **not**
recurse into a nested `BaseModel` field to emit further dotted sub-Properties.
Meanwhile `_walk_var_path` (`_llm_render.py:50-79`) does `path.split(".")` with
**unbounded** depth and walks via `getattr` for every segment after the first.
This means a `${prod.a.b}` reference (where `Node.inputs={"prod": Foo}`, `Foo.a`
is itself a nested `BaseModel`, and `.b` is one of ITS fields) is perfectly
resolvable at runtime, but **`_properties_for` never emits a Property titled
`"prod.a.b"`** — only `"prod.a"` (as a structured/object-typed Property
representing the whole nested model). The doc's own §1.4 headline claim —
"neograph's `${path}` naming and `_properties_for`'s Property-title naming are
ALREADY the same string" — is true only for path depth ≤ 2 (bare, or one dict
key + one field). For depth ≥ 3 there is **no corresponding declared
Property at all**; Option F's own §2.1 algorithm quietly stops depending on
`input_props` correspondence for these cases (it derives Properties purely
from what's scanned in the text, not from `input_props` — see §2.1 rule 5),
so mechanically it still produces a plausible flat name, but the doc's
framing ("re-encoding an existing correspondence" rather than "synthesizing
new information", the entire basis for the §9 confidence claim) is
**overstated** for any path deeper than one dot. This is a documentation
precision gap, not by itself fatal — see §2 below for why it compounds into a
real bug.

**CONFIRMED** — `tests/test_agent_spec_matrix.py`'s cell builder uses
`prompt = "process ${x}"` / `"combine ${variants}"` (`test_agent_spec_matrix.py:168,252`)
for every LLM-mode cell — always inline `${...}`, never a template-ref name.
The doc's §1.5 claim that all 35 red cells are inline-prompt cases is
correct.

---

## 2. THE HEADLINE FINDING — Option F's Property-emission strategy is not reconciled with `to_agent_spec`'s DataFlowEdge wiring, and this breaks the "27 cells go GREEN" claim as written

**REFUTED** (with a concrete, reproduced counter-example). The doc's §2.2
`_translate_placeholders` explicitly emits Properties **only for names
actually scanned out of the prompt text** (§2.1 rule 5, restated and defended
in §3 as "not a required catch-all" and "byte-for-byte faithful"). But
`to_agent_spec`'s outer wiring — `_emit_input_edges`
(`_agent_spec.py:853-878`, called from `to_agent_spec` at `:880-914`) —
computes each `DataFlowEdge.destination_input` **independently**, directly
from `_properties_for(node.inputs)`'s full, un-filtered, DOTTED title set
(`dest_input = f"{upstream_name}.{source_title}"` at `:867`), with **no
awareness of what `_translate_placeholders` decided to keep, drop, or
rename**. The doc's four call sites (`_lower_node` think branch, `_make_agent`,
both `_lower_oracle` sites) are the only places `_translate_placeholders`
would be called under Option F — `_emit_input_edges` is never mentioned
anywhere in the doc (§2, §4, §6, §7 — the entire implementation plan).

I verified empirically that pyagentspec's `DataFlowEdge` **validates**
`destination_input` against the destination node's own declared
`inputs`/Property titles at construction time, and raises if it doesn't
match:

```
DataFlowEdge(name='d1', source_node=prod, source_output='a',
             destination_node=target, destination_input='prod.a')
# target.inputs = [Property(title='x')]   (what Option F would emit for
#                                           "process {{ x }}", since only
#                                           "x" was scanned)
```//
raises:
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for DataFlowEdge
Value error, Flow data connection named `d1` is connected to a property named
`prod.a` of the destination node `target`, but the node does not have any
property with that name.
```

Concretely, running the ACTUAL matrix fixture (`think-bare-single`, i.e.
`@node(mode="think", model="fast", prompt="process ${x}")` consuming
`prod: Alpha`) through `to_agent_spec` TODAY produces
`ConfigurationError ... missing placeholder(s) for: ['prod.a']` (reproduced).
Under Option F as literally specified, this exact cell's prompt text
("process ${x}") never references `prod`/`prod.a` at all — only the
unrelated name `"x"`. Two independent problems follow, either of which keeps
this cell red:

1. Per the doc's own §2.3 residual-fail-loud rule ("a `${path}`'s first
   segment doesn't correspond to any declared input at all ... was already an
   error class"), `"x"` is not a declared input key (`"prod"` is), so a
   faithful implementation of §2.3 would still raise — just with a different
   message ("dangling placeholder `x`") instead of today's "missing
   placeholder `prod.a`". Still red.
2. Even if that check were dropped or made lenient, the resulting
   `LlmNode(inputs=[Property(title="x")], prompt_template="process {{ x }}")`
   has **no** Property titled `"prod.a"` — so `_emit_input_edges`'s
   independently-computed `DataFlowEdge(destination_input="prod.a", ...)`
   (still built, unconditionally, from `_properties_for(node.inputs)`, per
   `_agent_spec.py:867`) **raises a raw pydantic `ValidationError`** at Flow
   construction, exactly as reproduced above.

So under a straightforward reading of Option F's own algorithm, **the
`think-bare-single`/`think-bare-dict`/`agent-bare-*`/`act-bare-*`/etc. cells
do NOT become green "under Option F alone, no other feature work needed"** —
the doc's §1.5/§8 claim is false as written for the actual test fixture in
the repo today, and more importantly the underlying mechanism gap (Property
emission decoupled from DataFlowEdge title computation) is real and would
resurface even if the matrix's prompt text were rewritten to reference the
correct names, for the **general, un-referenced-input case** the doc itself
poses in §3 (`inputs={"prod": T, "unused": U}` where `${unused}` never
appears) — in that scenario `_emit_input_edges` still unconditionally builds
a `DataFlowEdge` targeting `"unused.field"` on a node whose translated
`.inputs=` no longer declares it, and pydantic rejects the `Flow`
construction the same way. §3's own worked example is un-exportable under
Option F as specified, contradicting §3's conclusion that it's "not an error"
— it's not silently dropped either (as the doc predicts); it's a hard crash
one layer removed from where the doc looked.

**Verdict on this finding**: this is not a minor implementation nit. It means
`_emit_input_edges` must ALSO be rewired to route through the same flat-name
translation `_translate_placeholders` computes (so `destination_input` uses
the flat name for a referenced path, and — for an unreferenced-but-declared
input — the corresponding `DataFlowEdge` must be either omitted or the design
must accept that omission is a REAL topology change, not the "byte-for-byte
faithful" no-op §3 claims). The doc's implementation plan (§7) enumerates
`_agent_spec.py` changes at exactly the four `_check_placeholder_inputs`
call sites and `loader.py` changes at three reconstruction functions — it
never mentions `_emit_input_edges` at all. This is a real, exhaustive-sweep
gap of the same *shape* the architecture retrospective diagnosed in the
07-25 doc (see §5 below).

---

## 3. Collision rule (task item 2 / doc §2.1 rule 6)

**CONFIRMED sound as specified.** The doc's own worked example — a literal
`${prod_a}` reference (a top-level input key legitimately containing an
underscore) colliding with a dotted `${prod.a}` reference in the SAME prompt,
both flattening to `"prod_a"` — is a real, constructible scenario
(`Node.inputs={"prod_a": Foo, "prod": Bar}` with `Bar.a: str`), and the doc
explicitly names it and prescribes fail-loud `ConfigurationError` (not a
silent `__2` suffix). I could not construct a collision the doc's rule 6
misses: the check is over the SET of unique paths scanned from one prompt
string, independent of `input_props`, so it is complete for that string.
**No refutation found here** — this part of the design is sound as written.

Three-level paths (`${a.b.c}`): confirmed resolvable at runtime (§1 above),
and confirmed the doc's flattening rule (`path.replace(".", "_")`) produces a
syntactically valid pyagentspec name for them — but see the headline finding
in §2: whether that Property ever gets WIRED to anything via a matching
`DataFlowEdge` is unaddressed, so "handles 3-level paths" is true only in the
narrow sense of "produces a valid string," not "produces a working export."

---

## 4. Unreferenced-input claim (task item 3 / doc §3)

**PARTIALLY CONFIRMED, conclusion REFUTED.** The premise is correct:
`_check_fan_in_inputs`/`_construct_validation.py` requires only that a
declared upstream producer exist, never that the prompt reference it — grep
confirms no such requirement anywhere in `_construct_validation.py` or
`lint.py`. For `think`/`agent`/`act` modes the node body never runs, so an
unreferenced input is inert in neograph's own runtime too — this much is
right.

But the conclusion — "Option F should emit a Property ONLY when actually
scanned... this is not an under-declaration, it is byte-for-byte faithful...
no catch-all needed" — is **REFUTED** by §2 above: the "faithful" framing
implicitly assumes the exported `Flow`'s edges are unaffected by which
Properties get declared on the destination node, which is false
(`_emit_input_edges` still tries to wire the FULL, untranslated input set).
The doc treats "declare fewer Properties" as a clean no-op; it is not, given
the existing edge-construction code, without further changes the doc never
specifies.

---

## 5. Marker / round-trip design (task item 4 / doc §4, §7.2)

**PARTIALLY CONFIRMED — the doc's loader.py plan is not wrong, but it is
imprecise about where the NEW marker is actually needed, because it doesn't
notice two of its four "call sites" are ALREADY solved by pre-existing
markers, independent of Option F:**

- **Agent/act reconstruction already stores the untranslated prompt
  verbatim and already prefers it.** `_agent_spec_marker()`
  (`_agent_spec.py:314-336`) stores `"prompt": node.prompt` — the ORIGINAL
  `${...}` text — in `_MARK_AGENT_SPEC`, entirely independent of whatever
  `system_prompt=` gets passed to the `Agent(...)` constructor. `loader.py`'s
  `_node_from_spec_agent` (`:192-203`) already reads `marker["prompt"]`
  directly when the marker is present, NEVER `agent.system_prompt`. So the
  doc's §7.2 instruction ("`_reconstruct_agent_node` (`:228`): same
  preference for `system_prompt`") describes a change that is **already
  in place** and requires no new marker for this site at all.
- **Oracle `merge_prompt` reconstruction already stores the untranslated
  text verbatim and already prefers it.** `_MARK_ORACLE_SPEC`
  (`_agent_spec.py:450-455`) already stores `"merge_prompt": oracle.merge_prompt`
  (the original text) at construction time, and `_reconstruct_oracle_group`
  (`loader.py:371-372`) already reads `spec["merge_prompt"]` from that marker
  to rebuild `Oracle(merge_prompt=...)` — never `merge_node.prompt_template`.
  This is untouched by whatever Option F does to `prompt_template=`.

The genuinely-needed sites are narrower than the doc implies: (1)
`_reconstruct_primitive_node`'s `LlmNode` branch (`loader.py:302`,
`prompt=spec_node.prompt_template` verbatim, no marker fallback today) for a
bare `think` node, and (2) `_reconstruct_oracle_group`'s
`base_prompt = base_variant.prompt_template` (`loader.py:349`, same gap) for
the GENERATING node's own prompt in a think-mode Oracle variant. Both of
these genuinely read the primitive field directly today with no marker
preference, and DO need the new `_MARK_PROMPT_SPEC` + loader change the doc
describes.

This is not fatal, but implementing the doc's §7.2 literally (adding a
preference check unconditionally at all four "sites") risks a real
**redundant-source-of-truth** hazard the marker convention this module
otherwise avoids: the original prompt text would end up stored in TWO places
for two of the four sites (`_MARK_AGENT_SPEC["prompt"]` /
`_MARK_ORACLE_SPEC["merge_prompt"]` AND the new
`_MARK_PROMPT_SPEC["original_text"]`), with no stated single source of truth
between them if they ever drift. The task must name which of the four sites
actually need the new marker (2, not 4) rather than adding it uniformly.

---

## 6. `StringProperty`-always claim (task item 5 / doc §2.1 point 4)

**CONFIRMED, with the caveat that it is currently moot given §2's blocker.**
Traced `_reconstruct_primitive_node`'s input-reconstruction path: it prefers
`_inputs_from_data_edges` (`loader.py:100-118`), which derives each input's
type from the UPSTREAM producer's own registered output type
(`output_types[source_name]`) — never from the destination `LlmNode`'s own
declared `.inputs=` Property types — and falls back to
`_agent_spec_props_to_type(spec_node.inputs)` only when no `DataFlowEdge`
exists at all. So a `StringProperty`-only `LlmNode.inputs=` does NOT, by
itself, cause type-fidelity loss on the round-trip **input** side, because
type recovery already routes around the destination node's own Property
types via edge-tracing. No existing test currently exercises a real
(non-string) type surviving through an `LlmNode.inputs=` Property, so no
currently-passing test is at risk here specifically. However, this whole
path is moot in practice until §2's `_emit_input_edges` gap is fixed, since
`_inputs_from_data_edges` depends on the `DataFlowEdge` existing at all,
which — per §2 — currently fails to construct.

---

## 7. Scope-boundary honesty / 27-vs-8 arithmetic (task item 6 / doc §8)

**CONFIRMED as pure arithmetic against `neograph-i7k7j`'s actual notes.**
The beads task's 35-cell list matches the doc's 27+8 partition exactly:
`act-oracle-{merge_fn,merge_prompt}-{single,dict}` (4) +
`agent-oracle-{merge_fn,merge_prompt}-{single,dict}` (4) = 8 cells staying
red for the Oracle+agent/act missing-lowering reason; the remaining 27 named
cells match the doc's list cell-for-cell. **However**, per §2 above, the
CONCLUSION drawn from that correct arithmetic — that these 27 "become GREEN
under Option F alone" — does not hold as specified; the split itself
(which cells belong to which root family) is correctly computed, but the
claimed resolution for family 1/2 is unverified/refuted.

Deferring the template-ref `prompt_compiler=` case (§5 item 1) is
independently justified: confirmed `to_agent_spec()` takes no
`prompt_compiler` parameter today, so ALL template-ref prompts are
uniformly out of scope regardless of Option F, and the matrix has zero such
cells (confirmed in §1). This deferral is honest and correctly scoped.

---

## 8. The meta-question: does this doc repeat its predecessor's exact process failure? (task item 7)

**YES — confirmed, and it is the direct cause of the §2 headline finding.**
The 07-25 architecture retrospective's core diagnosis (§2.4, §3) was that the
07-25 design doc anchored its search on the bug's own framing
(`node.mode`/`node.inputs` reaching `LlmNode`/`Agent`) and, as a result, missed
a construction site (`merge_prompt`) gated on a *different* field feeding the
*same* downstream constraint. The prescribed fix (retrospective §4.1) was an
exhaustive, mechanical sweep: "for every function in `_agent_spec.py`, list
every `LlmNode(`/`Agent(` construction call site... and for each one, name the
boolean condition that gates reaching it" — and, by extension for THIS doc,
every **consumer** of the Properties that construction site declares.

This doc DOES correctly enumerate the four `LlmNode`/`Agent` construction
sites (confirmed exhaustive by grep — no fifth site exists in
`_agent_spec.py` today). But it never asks the next question the same
discipline demands: "what ELSE in this file reads `_properties_for(node.inputs)`
independently and assumes it lines up with whatever the `LlmNode`/`Agent`
declares?" A grep for `_properties_for(` in `_agent_spec.py` turns up 9 call
sites, including `_emit_input_edges`'s caller (`to_agent_spec` itself, via
`ni.by_name` / dict-form fan-in resolution at `:887-914`) — a consumer this
doc's four-site, construction-centric sweep never reached, because (exactly
as the retrospective predicted) the search was anchored on "where does
`_check_placeholder_inputs`/its replacement get called," not "what in this
module treats `_properties_for(node.inputs)`'s title set as authoritative for
something else." This is the SAME anchoring failure, one hop further removed
— the retrospective's own §4.1 recommendation (an exhaustive
construction-site-and-consumer sweep, not a hand-picked four) would have
caught it, and this doc did not run that sweep despite being written
specifically in response to that retrospective.

---

## Overall verdict: **NEEDS-REVISION**

The core insight — `${var}`/`{{ var }}` are the same substitution operation
modulo delimiter/dot-encoding, and Option B over-generalized a narrow,
correct objection to Option C into a blanket "no representation" policy — is
right, and rejecting it in favor of the doc's own Option F direction is the
correct call. The collision rule (§2.1 rule 6), the fan-in-validator claim
(§3's premise), the 27/8 cell-split arithmetic (§8), and the template-ref
deferral (§5 item 1) are all independently verified sound.

**But the doc is not implementable as written**, for one decisive,
source-verified reason: `_translate_placeholders`'s scan-derived,
subset-of-declared-inputs Property emission is never reconciled with
`_emit_input_edges`'s independent, full-declared-input `DataFlowEdge.destination_input`
computation, and pyagentspec's `DataFlowEdge` pydantic validator provably
rejects the mismatch (reproduced). Concretely, for the ACTUAL matrix fixture
in the repo today (`prompt="process ${x}"`, never referencing the real input
name), none of the 27 "become GREEN" cells would pass under a literal
implementation of this doc — either via the doc's own §2.3 dangling-reference
rule, or via the `DataFlowEdge` construction crash reproduced in §2, and the
general unreferenced-input case (§3, §4 above) hits the same wall regardless
of what the matrix's prompt text says.

**What must change before implementation starts:**

1. Add `_emit_input_edges` (and its caller's `_properties_for(node.inputs)`
   dict-form/single-type resolution, `_agent_spec.py:843-914`) to the scope
   of this design as a FIFTH site requiring changes, not just the four
   `LlmNode`/`Agent` construction sites. Decide and document explicitly: for
   a declared-but-unreferenced input, does the `DataFlowEdge` get dropped
   (a real topology change — no longer "byte-for-byte faithful", §3's
   framing needs rewriting) or does `_translate_placeholders` need to change
   its own contract to also validate/require full coverage (contradicting
   §3's "not a required catch-all" claim, effectively reintroducing part of
   Option B's restriction for this specific sub-case)? Either resolution is
   viable; the doc currently picks neither and asserts a "no problem" outcome
   that the reproduced pydantic error disproves.
2. Fix the matrix's own fixture as part of this task (or explicitly scope it
   as a prerequisite): `prompt="process ${x}"` does not reference any real
   declared input name for the BARE/EACH/LOOP/OPERATOR cells (`"prod"`,
   `"pa"`/`"pb"`, not `"x"`) — a faithful implementation of Option F cannot
   make these cells GREEN without EITHER changing the fixture to reference
   real names OR resolving item 1's DataFlowEdge/Property reconciliation in a
   way that tolerates it. Currently unaddressed by the doc.
3. Narrow the loader.py marker plan (§5 above) to the two sites that
   genuinely lack a pre-existing original-text marker
   (`_reconstruct_primitive_node`'s `LlmNode` branch,
   `_reconstruct_oracle_group`'s `base_prompt`), rather than uniformly
   touching all four reconstruction paths — two of the four
   (`_reconstruct_agent_node`, the `oracle.merge_prompt` branch of
   `_reconstruct_oracle_group`) already read the untranslated text from a
   pre-existing marker and need no change; adding a second, redundant
   marker there is an avoidable drift risk.
4. Re-run the depth-limitation analysis (§1 above) once item 1 is resolved:
   confirm whether any REAL neograph pipeline (not just the matrix fixture)
   needs `${a.b.c}`-depth resolution through Option F's mechanism, and if so,
   verify the `DataFlowEdge`/Property reconciliation handles it too (a
   3-level path has no corresponding `_properties_for`-emitted Property at
   any level, so item 1's fix must handle "Property doesn't exist in
   `_properties_for(node.inputs)` at all" as a first-class case, not just
   "Property exists but is unreferenced").

Everything else in the doc (§1.1-§1.3 mechanism verification, the collision
rule, the fan-in-validator premise, Option D exhaustiveness inherited from
07-25, the 27/8 arithmetic) is confirmed sound and needs no change.
