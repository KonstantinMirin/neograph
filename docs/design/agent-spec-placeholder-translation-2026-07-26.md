# Agent Spec export: translating `${var}` to `{{ var }}` instead of fail-loud rejection (Option F)

Date: 2026-07-26
Status: design / decision record (no code changes)
Supersedes: the Option-B ("fail loud") recommendation in
`docs/design/agent-spec-oracle-inputs-2026-07-25.md` §4, as reinforced by
`agent-spec-oracle-inputs-2026-07-25-review.md` and analyzed in
`agent-spec-oracle-inputs-2026-07-25-architecture-retrospective.md`. Those
three documents are NOT wrong about the facts they verified (pyagentspec's
placeholder coupling, the four guard call sites, the four-bug lineage) — they
are wrong about the CONCLUSION drawn from those facts. This doc does not
re-litigate the verified mechanism; it re-derives the right response to it.

Related beads: `neograph-m57mn` (closed, commit `a109a7e`, shipped Option B),
`neograph-s7zt3` (epic), `neograph-00447` (closed, matrix generator),
`neograph-i7k7j` (open, 35 red cells — this doc changes its scope, see the
notes update at the end), `neograph-tjupj`, `neograph-2s2o6` (dispatch
unification, unaffected by this doc).

---

## 0. Why Option B was wrong, not just costly

The 07-25 doc's own Option C section states the real objection correctly: a
synthesized suffix is bad because it **corrupts the literal prompt text a
foreign consumer sends to an LLM**. That objection is sound — and it is
*specific to Option C's mechanism* (decorative, disconnected fake
placeholders appended after the real prompt). It is not an objection to
translation in general. Option B converted that narrow, correct objection
into a blanket policy — "any think/agent/act node with real inputs has no
faithful Agent Spec representation" — which is not the same claim and does
not follow from it.

The maintainer's pushback is the right frame: **neograph's `${var}` and
pyagentspec's `{{ var }}` are two syntaxes for the identical semantic
operation** — "substitute this upstream value into the prompt text at this
position." `${prod.a}` and `{{ prod_a }}` denote the same substitution site;
they differ in delimiter and dot-encoding, not in what they mean. A construct
that compiles to LangGraph and *renders a prompt with these exact
substitutions* is, by construction, representable in Agent Spec's own prompt
templating model — because Agent Spec's model **is** flat string
substitution into a template (verified below, not assumed). Rejecting it as
NO-REPR is "poor design" in exactly the sense the maintainer named: it
mistakes "our current lowering doesn't do the syntax conversion" for "this
cannot be expressed", when the two grammars are trivially inter-translatable
for the common case (verified in §2).

This is the same shape as the marker convention this module already uses
everywhere else: when a primitive's native vocabulary can't hold a piece of
information as-is, translate what CAN move natively and stash the
difference in a `neograph/*` marker for round-trip. Option B skipped that
step for `${var}` → `{{ var }}` specifically, even though the mapping is
mechanical and lossless. Option F is that missing translation step, done
properly — not Option C's decorative hack, and not Option B's refusal.

---

## 1. Verified mechanism (re-confirmed from source, not re-derived from the prior docs)

### 1.1 pyagentspec's inference is flat string substitution, nothing more

`.venv/lib/python3.12/site-packages/pyagentspec/templating.py`:

```python
TEMPLATE_PLACEHOLDER_REGEXP = r"{{\s*(\w+)\s*}}"
```

`get_placeholders_from_json_object` recursively scans str/bytes/dict/list/
tuple/set for this pattern and returns the flat token list;
`get_placeholder_properties_from_json_object` wraps each into a
`Property(json_schema={"title": name, "type": "string"})`. There is no
Jinja engine here, no attribute-path resolution, no scoping — it is one
`re.findall` over one flat token grammar with `\w+`-only names (no dots, no
brackets). `ComponentWithIO._validate_inputs`/`_validate_no_extra_property`
(`component.py:1625,1672-1683`) then requires the declared `inputs=` list's
titles to be an EXACT set-match against what the scan found.

### 1.2 neograph's own scanner is the same shape, one substitution rule already shared everywhere

`src/neograph/_placeholders.py`:

```python
DOLLAR_RE = re.compile(r"\$\{([^}]+)\}")     # ${var} / ${var.field}
BRACE_RE  = re.compile(r"\{([a-zA-Z_][\w.]*)\}")  # {var} / {var.field}
```

`apply_scanner(template, pattern, resolve)` is a single-pass `pattern.sub`
substitution — structurally identical to what pyagentspec's own regex scan
implies its `{{ }}` grammar is for. The module docstring states the
anti-duplication invariant explicitly: "one substitution rule, one scanner
— never a parallel second rendering path." Building a *second*, ad hoc
regex inside `_agent_spec.py` to detect `${...}` would violate that
invariant; reusing `DOLLAR_RE`/`apply_scanner` (already imported in
`_llm_render.py` and `prompt.py`) is not a new mechanism, it's importing the
existing one into a third consumer.

### 1.3 `_compile_prompt`/`_is_inline_prompt` confirm which grammar is live where

`src/neograph/_llm_render.py:_is_inline_prompt` — a template is inline iff it
contains a space or `${`; everything else is a **template-ref name** handed
to `runtime.prompt_compiler`. Inline prompts render via `_substitute_vars`
→ `apply_scanner(template, DOLLAR_RE, ...)` against `ri.raw` (raw Pydantic
objects, dotted `getattr` chains). Template-ref prompts render via
`_render_and_compile` → the compiler's own substitution, using
`ri.for_template_ref` (BAML-rendered strings + flattened fields).

`src/neograph/prompt.py:DefaultPromptCompiler` is the default file-ref
compiler. Its constructor takes `syntax: SyntaxSpec = "brace"` — i.e. the
**default template-ref grammar is `{var}` (single-brace, `BRACE_RE`), not
`${var}`**. `render_messages` calls `substitute(template_text, vars,
strict=self.strict, syntax=self.syntax)`, which for a named syntax resolves
to `apply_scanner(template, _NAMED_SYNTAX[syntax], resolve)` — the SAME
`apply_scanner` primitive, different pattern. So there are, in total, only
two neograph-native grammars ever in play: `${...}` (inline) and `{...}`
(template-ref, `DefaultPromptCompiler`'s default) — both single-pass regex
substitutions over a flat token name, structurally the same *shape* of
mechanism pyagentspec's `{{ ... }}` is.

### 1.4 `_properties_for`'s dot-naming already lines up with `${path}`'s dot-naming — this is the key finding that makes Option F mechanical, not speculative

`src/neograph/_agent_spec.py:_properties_for`:

```python
if isinstance(type_spec, dict):
    for key, typ in type_spec.items():
        props = model_to_agent_spec_properties(typ)
        for p in props:
            p.title = f"{key}.{p.title}"
```

Dict-form `Node.inputs={"prod": SomeModel}` produces Property titles
`"prod.a"`, `"prod.b"`, ... — one per field of `SomeModel`, prefixed by the
dict key. Single-type `Node.inputs=SomeModel` produces bare `"a"`, `"b"`
(via `model_to_agent_spec_properties` with no prefix).

Independently, `src/neograph/_input_shape.py` confirms what a `${path}`
reference actually resolves against at RUNTIME:
- Dict-form (`_extract_fan_in_dict`): `input_data` is a
  `dict[str, Any]` keyed by the `Node.inputs` dict key (e.g. `"prod"`).
  `_walk_var_path` (`_llm_render.py`) treats `parts[0]` as a dict lookup,
  so `${prod.a}` walks `input_data["prod"].a` — matching the SAME
  `"prod.a"` string `_properties_for` already emits as a Property title.
- Single-type (`_extract_single_type`): `input_data` IS the raw model (no
  wrapping dict), so `${a}` resolves via `getattr(model, "a")` directly —
  matching the bare `"a"` Property title `_properties_for` emits for the
  single-type shorthand.

**This is the load-bearing fact**: neograph's own `${path}` naming and
`_properties_for`'s Property-title naming are ALREADY the same string, up to
the dot-vs-underscore encoding pyagentspec's `\w+`-only grammar forces. There
is no semantic gap to bridge — only a character-encoding gap (`.` is not a
word character). Translating `${prod.a}` → `{{ prod_a }}` is not
"inventing" a placeholder that wasn't there; it is re-encoding the exact
same, already-present reference in a different grammar.

### 1.5 Confirmed via commit `a109a7e` and the matrix fixtures: every currently-red placeholder-coupling cell is an INLINE prompt

`git show a109a7e` added `_check_placeholder_inputs` at four call sites
(`_lower_node`'s think branch, `_make_agent`, `_lower_oracle`'s think-variant
loop, `_lower_oracle`'s `merge_prompt` branch). `tests/test_agent_spec_matrix.py`
builds every cell's prompt as `"process ${x}"` (line 168) or
`"combine ${variants}"` (line 252) — **always inline `${...}` text, never a
template-ref name**. All 35 red cells in `neograph-i7k7j`'s scope are
therefore inline-prompt cases. This matters for scoping §4/§5 below: Option
F's inline mechanism alone, with zero new public API surface, is sufficient
to address every currently-known red cell that is placeholder-coupling
(not lowering-completeness) in nature.

---

## 2. The translation mechanism (inline case — the primary, common, immediately-buildable path)

### 2.1 Flat-name encoding rule

For a prompt/system-prompt/`merge_prompt` text and its associated
`input_props: list[Property]` (from `_properties_for(node.inputs)` /
`_properties_for(node.oracle_gen_type or node.outputs)` per call site):

1. Scan the text with the EXISTING scanner: `DOLLAR_RE` + a collecting
   `resolve` passed to `apply_scanner` (do not use a second, hand-rolled
   regex — reuse `_placeholders.DOLLAR_RE`/`apply_scanner` directly,
   matching the module's own anti-duplication invariant it already applies
   to `get_placeholders_from_json_object` reuse).
2. For each **unique** `${path}` match, compute
   `flat_name = path.replace(".", "_")`. `\w+` already accepts underscores,
   letters, digits — this is the one substitution rule needed because
   `.` is the only Property-title character pyagentspec's grammar forbids
   (dict-form dotted titles never contain any other non-`\w` character —
   confirmed by `_properties_for`'s only two title shapes, `"{key}.{field}"`
   and bare `"{field}"`).
3. Rewrite the text in a SINGLE PASS via `apply_scanner(text, DOLLAR_RE,
   lambda path: "{{ " + flat_name_for(path) + " }}")` — brace-safe by
   construction (the same guarantee `apply_scanner`'s docstring already
   states: a value containing `{...}` is never re-scanned).
4. Emit `Property(json_schema={"title": flat_name, "type": "string"})` —
   `StringProperty`, matching pyagentspec's OWN inference convention (its
   `get_placeholder_properties_from_json_object` always infers `"type":
   "string"` regardless of the real field type; `_properties_for` for
   OTHER primitives, like `ToolNode`, still carries the real type, but for
   an `LlmNode`/`Agent` specifically, string-typed is what the foreign
   consumer's OWN inference would produce too — the correct-fidelity choice
   here is to match, not to over-claim a richer type pyagentspec's own
   scanner would never assign).
5. Emit Properties ONLY for names actually scanned out of the text — not
   for every entry in `input_props`. See §3 for why this is correct, not a
   silent narrowing.
6. Collision rule: if two DISTINCT original paths (e.g. a literal
   `${prod_a}` reference and a dotted `${prod.a}` reference in the SAME
   prompt) flatten to the same `flat_name`, raise `ConfigurationError`
   naming both original paths and the collided flat name — the fail-loud
   convention this module already uses everywhere else (`raw_fn`,
   `skip_when`, callable `Loop.when`), not a silent disambiguation
   (appending `__2` would produce a placeholder name with no relationship
   to either original path, defeating the entire point of a faithful
   translation).

### 2.2 The shared helper

Replace `_check_placeholder_inputs` with `_translate_placeholders`:

```python
def _translate_placeholders(
    prompt_text: str,
    input_props: list[Property],
    node_name: str,
) -> tuple[str, list[Property], dict[str, str]]:
    """Rewrite ${path} -> {{ flat_name }} for every path referenced in
    prompt_text, returning (rewritten_text, referenced_properties,
    flat_to_original_map). Properties are emitted ONLY for paths actually
    scanned out of the text (see design doc §3) -- not for every declared
    input_prop, so pyagentspec's exact-set-match validator (extra AND
    missing) is satisfied by construction, never merely checked after the
    fact.
    """
```

Called at the same four sites `_check_placeholder_inputs` occupies today:
`_lower_node`'s `think` branch, `_make_agent` (`agent`/`act`), `_lower_oracle`'s
think-mode variant loop, `_lower_oracle`'s `merge_prompt` branch (gated on
`oracle.merge_prompt` truthiness, independent of `node.mode` — same
four-site enumeration the 07-25 review already established, still correct
under Option F). It is a REPLACEMENT, not an addition: nowhere does the
guard's old fail-loud path still make sense for a case Option F can now
resolve mechanically.

### 2.3 Residual fail-loud cases for the inline path

Even for inline prompts, `ConfigurationError` still fires when:
- A `${path}` reference and a distinct input key collide under flattening
  (§2.1 rule 6).
- A `${path}`'s first segment doesn't correspond to any declared input at
  all (e.g. a typo, or a reference to a framework-only var inline prompts
  don't even have access to per `_predict_input_keys(include_flattened=False)`)
  — this was already an error class (a dangling reference), now surfaced at
  export time with a clear message instead of silently emitting an orphan
  placeholder.

---

## 3. Inputs declared but never referenced in the prompt: NOT an error, and NOT a required catch-all

Verified via `_input_shape.py` and the mode table in `AGENTS.md`: for
`think`/`agent`/`act` modes, **the node's Python body never runs** ("dead
code" per the mode table). The ONLY mechanism by which a declared
`Node.inputs` value can affect what the LLM sees is (a) prompt-template
substitution, or (b) `di_inputs` (a DIFFERENT, DI-sourced channel, not
`Node.inputs`). Fan-in validation (`_check_fan_in_inputs`,
`_construct_validation.py`) requires only that a declared upstream producer
EXISTS — it does not require the prompt to reference it. So a real neograph
pipeline can legitimately declare `inputs={"prod": T, "unused": U}` where
`${unused}` never appears in the prompt text; that input is inert in
neograph's OWN runtime too (the value is computed, fanned in, and then
never read by anything, because the node body doesn't execute).

**Conclusion**: Option F should emit a Property for a name ONLY when it is
actually scanned out of the prompt text — this is not an under-declaration,
it is a byte-for-byte faithful reflection of what the node's Agent Spec
lowering (an `LlmNode`/`Agent` whose only data path IS its prompt text) can
possibly do with that input. No `${_all}`-style catch-all reference is
needed or correct: inventing one would mean asserting a data dependency the
exported `LlmNode` does not actually have, which is the OPPOSITE of
faithful — the same kind of decorative-suffix problem Option C had, just on
the input-declaration side instead of the prompt-text side.

The full original `Node.inputs`/`oracle.gen_outputs` TypeSpec (including any
unreferenced keys) is preserved in the round-trip marker (§4) for
`from_agent_spec()` to reconstruct the ORIGINAL `Node(...)` faithfully —
round-trip fidelity and "what the exported LlmNode actually consumes" are
two different questions, and Option F answers them from two different
places (marker vs. primitive `.inputs=`) rather than conflating them the
way a catch-all Property would.

---

## 4. Marker / round-trip design

Following the established convention (`_MARK_ORACLE_SPEC`,
`_MARK_EACH_SPEC`, etc. — "marker carries what the primitive can't"), add:

```python
_MARK_PROMPT_SPEC = "neograph/prompt_spec"
```

Stamped into the `metadata` of every `LlmNode`/`Agent` construction that
went through `_translate_placeholders` (i.e. had `input_props` and a
non-empty translation to do — an LLM-mode node with ZERO real inputs needs
no marker, mirroring `_check_placeholder_inputs`'s existing
`if not input_props: return` early-out):

```python
{
    "original_text": prompt_text,           # the untranslated ${...} text
    "placeholder_map": {                    # flat_name -> original ${path}
        "prod_a": "prod.a",
        "x": "x",
    },
}
```

**Why this shape and not something richer**: `original_text` lets
`from_agent_spec()` restore `node.prompt`/`oracle.merge_prompt` VERBATIM —
byte-identical to what the author wrote, not a re-derivation from the
translated `{{ }}` text (re-deriving would require inverting the
flattening, which is lossy exactly when the collision rule in §2.1 would
otherwise have fired — storing the original sidesteps that entirely).
`placeholder_map` is what lets the loader reconstruct the ORIGINAL
Property-title shape (dotted dict-form titles) from the flat Properties the
`LlmNode`/`Agent` primitive actually carries — `loader.py`'s existing
`_reconstruct_primitive_node`/`_reconstruct_oracle_group` read
`spec_node.inputs`/`outputs` Property lists directly today (e.g.
`_reconstruct_primitive_node:302` uses `spec_node.prompt_template` verbatim
as `node.prompt`); under Option F those reads must instead prefer
`metadata[_MARK_PROMPT_SPEC]["original_text"]` for the prompt and rebuild
dict-form `Node.inputs` keys by un-flattening each `spec_node.inputs[i].title`
through `placeholder_map` (falling back to the flat title unchanged if no
marker is present — the pre-Option-F wire format, so old exports without
this marker still import, same backward-compat posture the module already
takes for markers added in later commits).

This is a strictly ADDITIVE marker (a foreign, non-neograph Agent Spec
consumer ignores `metadata["neograph/*"]` entirely, per the module's Core
Invariant) — it does not touch the literal `prompt_template`/`system_prompt`
string a foreign runtime sends to an LLM, which now contains the REAL,
functioning `{{ prod_a }}` substitution the foreign runtime's own templating
engine will correctly fill in. This is the exact property Option C failed to
have (a decorative, non-functional suffix) and Option B gave up on
entirely (no export at all) — Option F's exported text is genuinely
portable AND genuinely correct for a foreign consumer, not merely inert.

---

## 5. What genuinely stays NO-REPR after Option F

Precisely, not "most":

1. **Custom (non-`DefaultPromptCompiler`) `prompt_compiler`s for
   template-ref prompts**, where the raw template text cannot be resolved
   without live `input_data` (an arbitrary callable
   `prompt_compiler(template, input_data, **kw) -> list[dict]` has no
   separable "give me the raw text" step neograph can call statically).
   `to_agent_spec()` has no `prompt_compiler` parameter today at all, so
   this is currently uniformly NO-REPR for ALL template-ref prompts,
   regardless of compiler. Confirmed via `prompt.py`: `DefaultPromptCompiler`
   is the ONE built-in compiler with a separable, no-input-data
   `load_template(template) -> str` method — verified by reading its body
   (`Path(self.loader)/f"{template}{suffix}"` read, or `self.loader(template)`
   call, neither touching `input_data`). A future extension could thread an
   optional `to_agent_spec(construct, *, prompt_compiler=None)` so a
   `DefaultPromptCompiler` instance's `load_template()` output feeds the
   SAME `_translate_placeholders` machinery using `BRACE_RE` (its default
   `syntax="brace"`) instead of `DOLLAR_RE` — but this needs a new public
   parameter threaded through all four call sites, a decision about what
   happens when `syntax` is a custom callable (still NO-REPR — no static
   regex to scan with), and is NOT needed to resolve any of the 35 currently
   known red cells (§1.5: the matrix uses inline prompts exclusively). This
   doc recommends scoping it as an explicit, separate follow-up (working
   title: "Agent Spec export: template-ref prompt text resolution via
   `prompt_compiler=` accessor"), not bundling it into the immediate fix —
   same "don't fold speculative scope into a landing fix" discipline the
   07-25 doc itself used correctly for deferring Option C.
2. **Oracle + `agent`/`act` mode**: `_lower_oracle`'s `agent`/`act` branch
   does not raise because of placeholder coupling — it raises because
   `_lower_oracle` has **no lowering implemented at all** for that mode
   combination (`"Oracle+agent/act export has no Agent Spec lowering yet"`,
   `_agent_spec.py:405-412`). This is a missing-feature gap, not a
   translation gap; Option F does not touch it, and none of the mechanism
   above applies until an `Agent`-based Oracle-variant lowering is written
   (mirroring `_make_agent`, the same way the `think`-mode variant loop
   mirrors `_lower_node`'s `think` branch). Once that lowering exists,
   Option F's `_translate_placeholders` applies to it uniformly — no new
   translation logic needed, only the missing dispatch arm.
3. **A `${path}` referencing something inline prompts structurally cannot
   see** — anything beyond `_predict_input_keys(include_flattened=False)`'s
   raw-key set (framework extras, `di_inputs`-only names) was already a
   dangling reference before Option F; it stays an error, now raised
   earlier and more precisely (§2.3) rather than silently producing an
   unmatched `{{ }}` scan miss deep inside pyagentspec's constructor.

Everything else in the 07-25 doc's Option D exhaustiveness search (no
pyagentspec primitive combines "LLM call" with "unconstrained inputs")
still stands and is irrelevant to Option F — Option F does not need such a
primitive; it makes `LlmNode`/`Agent`'s OWN placeholder-coupled inference
succeed on the merits, by construction, instead of routing around it.

---

## 6. Uniformity across the four sites and Oracle's `merge_prompt` vs `merge_fn`

`_translate_placeholders` is called with the SAME signature at:

- `_lower_node`'s `think` branch: `(node.prompt or "", _properties_for(node.inputs), node.name)`.
- `_make_agent` (`agent`/`act`): `(node.prompt or "", inputs, node.name)` — identical shape, `system_prompt` instead of `prompt_template` on the output side only.
- `_lower_oracle`'s think-mode variant loop: `(node.prompt or "", inputs, variant_name)` — per-variant, same `node.inputs`, since every variant consumes the SAME upstream input set (per `_lower_construct_item`'s documented Oracle input-fan-out rule: "ORACLE → EVERY variant node ... independently consumes the external input").
- `_lower_oracle`'s `merge_prompt` branch: `(oracle.merge_prompt, gen_outputs, node.name)` — the ONLY site whose `input_props` source is `gen_outputs` (the variants' outputs) rather than `node.inputs`, exactly as the 07-25 review's 4th-site finding established; Option F's translation mechanism is indifferent to which Property list it's given, so this asymmetry in SOURCE does not require asymmetry in the TRANSLATION FUNCTION itself — one function, four call sites, four different `(text, props, name)` triples.

`Oracle.merge_fn` (the `ToolNode` branch) needs no change under Option F,
same as under Option B — `ToolNode._get_inferred_inputs` echoes `tool.inputs`
with zero text coupling, so it was never broken by the placeholder issue in
the first place.

---

## 7. Concrete implementation plan

1. **`_agent_spec.py`**:
   - Add `_MARK_PROMPT_SPEC` to the marker-key block (§64-83 today).
   - Replace `_check_placeholder_inputs` with `_translate_placeholders`
     (§2.2 signature). Keep the function name change deliberate — a
     grep for the old name should find zero hits post-migration (mirrors
     the project's "grep should return zero hits" convention for fully
     retired symbols, e.g. `@raw_node`).
   - `_lower_node`'s `think` branch: call `_translate_placeholders`, use
     the rewritten text for `prompt_template=`, the returned Properties for
     `inputs=`, and stamp `_MARK_PROMPT_SPEC` (only if non-empty) into the
     `LlmNode`'s `metadata=` (a `metadata=` kwarg does not exist on the
     `LlmNode` construction today for the `think` branch — add it,
     mirroring `_lower_oracle`'s variant/merge `metadata=` usage).
   - `_make_agent`: same shape, `system_prompt=` instead of
     `prompt_template=`, marker on the `AgentNode` wrapper's `metadata=`
     dict (which already carries `_MARK_MODE`/`_MARK_AGENT_SPEC`) — extend
     that same dict with `_MARK_PROMPT_SPEC` rather than adding a second
     metadata dict.
   - `_lower_oracle`'s think-mode variant loop and `merge_prompt` branch:
     same call, folded into each's existing `metadata` dict (both already
     build one: `variant_metadata`, and the `merge_node`'s dict carrying
     `_MARK_ORACLE_SPEC`).
   - Delete the `oracle.merge_prompt`-gated raw-string coupling comment
     that currently justifies fail-loud (`_agent_spec.py:436-440`) —
     replace with the translate-and-emit call.
2. **`loader.py`**:
   - `_reconstruct_primitive_node` (`:302`, think case): prefer
     `metadata[_MARK_PROMPT_SPEC]["original_text"]` for `prompt=` when the
     marker is present, falling back to `spec_node.prompt_template`
     verbatim when absent (pre-Option-F wire compat).
   - `_reconstruct_agent_node` (`:228`): same preference for
     `system_prompt`.
   - `_reconstruct_oracle_group` (`:316`): same preference, reading the
     marker off whichever variant/merge node carries it.
   - All three additionally need to un-flatten `Node.inputs`' dict-form
     structure using `placeholder_map` when rebuilding `inputs=` from
     `spec_node.inputs` Property titles, instead of assuming the titles are
     already dotted (which they no longer are, post-Option-F, for any node
     that went through translation).
3. **Tests** (design-only doc; NOT written here, but scoped for the
   implementation task): a single-file round-trip test per the four call
   sites, plus a dedicated collision-detection test (§2.1 rule 6) and an
   unreferenced-input test (§3: declare `inputs={"prod": T, "unused": U}`,
   assert the exported `LlmNode.inputs` has ONLY the `prod`-derived
   Property, and the marker's original TypeSpec still reconstructs
   `unused` on round-trip).
4. **`tests/test_agent_spec_matrix.py`**: flip the RED_EXPORT cells listed
   in `neograph-i7k7j`'s notes to GREEN as each is fixed, EXCEPT the 8
   `{agent,act}-oracle-{merge_fn,merge_prompt}-{single,dict}` cells, which
   stay red under a DIFFERENT, correctly-scoped xfail reason
   ("Oracle+agent/act lowering not yet implemented" — not "placeholder
   coupling NO-REPR") until the separate Oracle+agent/act lowering feature
   lands.

---

## 8. Concrete effect on the 35-cell matrix

Of the 35 `RED_EXPORT` cells in `neograph-i7k7j`'s notes:

- **27 cells become GREEN** under Option F alone (no other feature work
  needed): `act-bare-{dict,single}`, `act-each-single`, `act-loop-single`,
  `act-operator-{dict,single}` (6); `agent-bare-{dict,single}`,
  `agent-each-single`, `agent-loop-single`, `agent-operator-{dict,single}`
  (6); `scripted-oracle-merge_prompt-{dict,single}` (2);
  `think-bare-{dict,single}`, `think-each-{context,dict,single}`,
  `think-loop-{dict,single}`, `think-operator-{dict,single}`,
  `think-oracle-merge_fn-{dict,single}`,
  `think-oracle-merge_prompt-{dict,single}` (13). 6+6+2+13 = 27.
- **8 cells stay red, but for a re-scoped reason**:
  `act-oracle-merge_fn-{dict,single}`, `act-oracle-merge_prompt-{dict,single}`,
  `agent-oracle-merge_fn-{dict,single}`, `agent-oracle-merge_prompt-{dict,single}`
  — these are blocked on the missing Oracle+agent/act lowering (§5 item 2),
  not on placeholder coupling. `neograph-i7k7j`'s acceptance criterion
  ("full matrix green ... zero xfail markers except explicitly
  deferred-and-justified ones") is satisfiable by implementing that
  lowering as its own, separate, in-scope fix within `i7k7j` (it was
  already root family 3 in the handoff notes) — Option F does not need to
  wait for it, and the 8-cell gap does not need a NEW deferred-xfail bead;
  it is the ALREADY-NAMED root family 3 in `i7k7j`'s own notes, now
  correctly separated from root family 1 (which Option F resolves) instead
  of being conflated with it.

---

## 8.5 Addendum (post-review, 2026-07-26): `_emit_input_edges` is a 5th required site, and the loader marker plan is narrower than §7 stated

Independent adversarial review
(`agent-spec-placeholder-translation-2026-07-26-review.md`) found the doc as
originally written is **not implementable**: `_translate_placeholders`
emits Properties only for names actually scanned out of the prompt text
(§2.1 rule 5), but `to_agent_spec`'s `_emit_input_edges`
(`_agent_spec.py:853-878`) independently computes every
`DataFlowEdge.destination_input` from the FULL, untranslated
`_properties_for(node.inputs)` title set, with no awareness of what
`_translate_placeholders` decided to keep. pyagentspec's `DataFlowEdge`
validator hard-rejects a `destination_input` that doesn't match a declared
Property on the destination node — reproduced empirically. This is the same
anchoring failure the architecture retrospective diagnosed in the
predecessor doc, one hop further removed (construction sites were swept
exhaustively; *consumers* of the same Property set were not).

**Required revisions before implementation:**

1. **`_emit_input_edges` becomes a 5th site this design must change**, not
   just the four `LlmNode`/`Agent` construction sites. For each upstream
   input reference, `destination_input` must use the SAME flat name
   `_translate_placeholders` computed (via its returned
   `flat_to_original_map`), not the raw dotted `_properties_for` title. For a
   declared-but-**unreferenced** input (§3's `unused` example), the
   corresponding `DataFlowEdge` must be **dropped**, not built — this is a
   real, acknowledged topology change (the exported `LlmNode`/`Agent`
   genuinely has no data path to that value, since prompt substitution is
   its only channel and the prompt doesn't reference it), not the "byte-for-byte
   faithful no-op" §3 originally claimed. §3's framing should be corrected to:
   "the DataFlowEdge is correctly absent, matching the primitive's true data
   dependencies — not a silent drop, an accurate one."
2. **The matrix's fixture prompts must be updated to reference their real
   input names** (e.g. `"process ${prod.a}"`, not the placeholder `"process
   ${x}"` currently in every cell builder) as part of this same
   implementation — Option F cannot turn a cell green whose prompt never
   references its own declared input, regardless of how correctly the
   translation mechanism itself is built.
3. **Narrow the `loader.py` marker plan to 2 sites, not 4**: review confirmed
   `_reconstruct_agent_node` and `_reconstruct_oracle_group`'s
   `oracle.merge_prompt` branch **already** read the untranslated original
   text from a pre-existing marker (`_MARK_AGENT_SPEC["prompt"]` /
   `_MARK_ORACLE_SPEC["merge_prompt"]`) and need NO new marker — adding
   `_MARK_PROMPT_SPEC` there too would create a redundant, driftable second
   source of truth. Only `_reconstruct_primitive_node`'s `LlmNode` branch and
   `_reconstruct_oracle_group`'s `base_prompt` (the think-mode variant's OWN
   prompt) genuinely lack a marker today and need the new
   `_MARK_PROMPT_SPEC` + loader preference-check.
4. **Depth ≥ 3 dotted paths** (`${prod.a.b}` where `prod.a` is itself a
   nested `BaseModel`): confirmed runtime-resolvable via `_walk_var_path`'s
   unbounded `getattr` chase, but `_properties_for` never emits a
   corresponding Property beyond one dot level. No current test needs this;
   implementation should treat "no `_properties_for`-emitted Property exists
   for this path at any prefix" as its own explicit case in the
   `_emit_input_edges` rewrite (item 1) — likely: no `DataFlowEdge` can be
   wired for it (same as the unreferenced-input case), documented as a known
   depth limitation, not silently mismatched.

---

## 9. Confidence and residual risk

High confidence this is the right design, for a falsifiable reason: §1.4's
finding (neograph's `${path}` naming already coincides with
`_properties_for`'s Property-title naming, modulo dot-encoding) was verified
against real source in both directions (`_properties_for`'s title
construction AND `_input_shape.py`'s runtime resolution), not assumed. That
finding is what makes Option F a *translation of an existing 1:1
correspondence* rather than a *synthesis of new information* — which is
exactly the distinction that made Option C wrong (it invented placeholder
references with no runtime correspondence) and makes Option F sound (it
re-encodes a correspondence that was already there). The one place this
doc recommends NOT resolving immediately (template-ref via a new
`prompt_compiler=` parameter, §5 item 1) is scoped as a deferral for
process-discipline reasons (new public API surface, zero current red-cell
need), not because its feasibility is in doubt — `DefaultPromptCompiler.load_template`'s
no-input-data signature was directly read, not inferred.
