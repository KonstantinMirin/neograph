# One rendering rule for everything a prompt_compiler sees

**Ticket**: neograph-l2a7w (P1, bug)
**Date**: 2026-08-11
**Branch / worktree**: `fix/prompt-compiler-input-shape` off `main` (0.7.x), `/Users/konst/projects/neograph-l2a7w`
**Status**: design agreed, not yet implemented

All line references are to `main` @ `105b165`, which is where this branch starts.

---

## 1. The defect

A `prompt_compiler` is handed `input_data` whose values are **sometimes an
already-BAML-rendered `str` and sometimes the raw Pydantic model**. The compiler
cannot tell which, and the two need opposite handling.

The failure is silent and produces a plausible wrong answer. A compiler written
the obvious way —

```python
prose = input_data.get('intake')
text = str(getattr(prose, 'text', '') or '')
```

— returns an EMPTY string when the value is a `str`, because every `getattr`
misses. The prompt renders with an empty payload, the model reads a blank input,
and answers coherently about nothing. Nothing raises, the graph completes, the
output is well-typed and confidently wrong.

**Real incident** (ox-troubleshooting-demo, qvoq.3, 2026-08-11): an extraction
node reading a prose diagnosis onto a typed taxonomy scored 0/6 on its accuracy
golden, every row answering `insufficient_data — the arm emitted no diagnosis`.
The prose was 5,334 chars and had been passed in correctly. Had that golden not
existed, the downstream A/B benchmark would have published "the third-party
prompt never reaches a conclusion" — a claim about someone else's system caused
entirely by this seam.

**It is a general tax, not one team's mistake.** The same repo carries
`context.render_intake`, whose entire body is the workaround (dict-or-not, then
str-or-model). It existed as three byte-identical private copies across three
arms before being extracted. Every single-agent arm independently rediscovered
the need for it.

---

## 2. Evidence

### 2.1 The rule already exists — implemented twice, differently, and skipped twice

The intended rule is uncontroversial and is what two of the four channels already
try to do: *BAML-render unless it is already a string; coerce primitives to
string; if you do not like the result, write your own presenter.* Measured
against that rule:

| Channel | Site | Total / coerces primitives | Honors `render_for_prompt()` |
|---|---|---|---|
| inline `${var}` | `_llm_render.py:82` `_resolve_var` | yes — `describe_value` or `str(obj)`, `None -> ""` | **no — ignored** |
| template-ref | `renderers.py:455` `_render_single` | **no — primitives pass through as `int`/`list`** | yes |
| `di_inputs` | `_dispatch.py:40` `_inject_di_inputs` | **no — not rendered at all** | **no** |
| Oracle `merge_prompt` | `_oracle.py:293` `_merge_prompt_input` | **no — not rendered at all** | **no** |

Reproduced (`compile_prompt` / `render_inputs`, model carrying a
`render_for_prompt()` presenter):

```
inline ${ctx}      -> {  node_id: "n1"  root: "/tmp" }   # presenter ignored
template-ref value -> 'MY PRESENTER: n1'                  # presenter honored
primitive passthru -> {'count': 5, 'items': [1, 2]}       # not coerced
```

A user who writes `render_for_prompt()` on their context model gets it applied or
ignored depending on whether the node's prompt happens to contain a space
(`_is_inline_prompt` = *contains a space or `${`*).

### 2.2 The reported str-vs-model split, reproduced

One probe compiler, one pipeline, an Oracle ensemble over a think node:

```
node='extract'  template='extract/taxon'  -> {'intake': 'str'}
node=''         template='merge/taxon'    -> {'variants': 'list', 'intake': 'Prose'}
```

Same compiler. Same logical name `intake`. `str` on one call, `Prose` on the
next. `node_name` is also empty on the merge path, so the compiler cannot even
branch on it.

Three further shape variants confirmed:

- **dict vs not-a-dict** — a node with single-type `inputs=Prose` hands the
  compiler a *bare* rendered value, no key at all. This is the other half of the
  consumer's `isinstance(input_data, dict)` dance.
- **mixed dict** — when a model defines `render_for_prompt() -> BaseModel`,
  `RenderedInput.for_template_ref` merges rendered strings *and* raw `BaseModel`
  children: `{'intake_e': 'str', 'inner': 'Other', 'text': 'str'}`
  (`renderers.py:_render_with_flattening`, deliberate, for dotted access).
- **`di_inputs` are raw** — `di_inputs['ctx']` arrives as a live `RunCtx`;
  `DefaultPromptCompiler.build_vars` (`prompt.py:190`) lays rendered strings on
  top of it and `substitute` (`prompt.py:46`) does `str(vars[name])`, so `{ctx}`
  ships a Pydantic repr while `{intake}` ships BAML.

### 2.3 Why the shape diverges: the decision is made per call site

Four runtime entry points feed a compiler. All four funnel through
`_llm_render.py:151` `_compile_prompt` — but three of them pre-render at the call
site and the fourth does not:

| Path | Site | Renders first? |
|---|---|---|
| think mode | `_dispatch.py` `ThinkDispatch` -> `_render_input` | yes |
| agent/act mode | `_agent_cycle.py:114` `_turn_prep_kwargs` -> `_render_input` | yes |
| public `compile_prompt` / `render_prompt` | `_llm_render.py:218` `_render_and_compile` | yes — its own second copy of the same split |
| **Oracle `merge_prompt`** | `_oracle.py:309` -> `invoke_structured(input_data=…)` | **no** |

`_dispatch.py:341` `_render_input` and `_llm_render.py:218` `_render_and_compile`
are two copies of the same decision (`build_rendered_input` + `_is_inline_prompt`
-> `.raw` or `.for_template_ref`). This is the duplicated-source-of-truth pattern
CLAUDE.md's `ModifierCombo` lesson bans: N consumers each re-deriving instead of
one shared table every consumer reads.

`_llm_render.py:218`'s own docstring asserts the invariant that is not in fact
held: *"the ONE seam render_prompt, compile_prompt, and the runtime ThinkDispatch
path all funnel through — no second rendering path, no second compile path (the
hjwv anti-duplication invariant)"*. It omits the agent cycle and the Oracle merge,
and there is no structural guard over the claim.

### 2.4 The tax is already visible inside this repo

- `examples/observable_pipeline.py` — **one compiler, both conventions**:
  `input_data.text` for one template, `input_data["variants"]` iterated as models
  for the other. It only survives because the first node has no inputs (so
  `input_data` is `""`, falsy, and hits a fallback). Give that node an upstream
  and it is the exact reported failure.
- `website/src/content/docs/concepts/prompt-compiler.mdx` — the documented
  "production-shaped pattern" *teaches* the dance:
  `if isinstance(input_data, dict): … # custom handling per type` /
  `elif isinstance(input_data, str):`.
- **neograph-iu05** (closed P3, 2026-06-04) is the same seam biting before. It was
  fixed with an AST guard over *examples*
  (`tests/test_guards_examples.py::TestExampleMergeCompilerReadsVariants`) telling
  authors to read `data["variants"]` — the symptom, not the shape. Its root-cause
  note ("neograph now BAML-renders upstream output to a STRING before the compiler
  sees it") is only true for three of the four paths.
- **neograph-hjwv** is the precedent for the framing: two consumers hand-rolled
  the compiler seam and diverged on a safety property; the answer was to ship the
  primitive. This is the same argument one level down — they hand-roll the *input
  shape* instead of the *substitution policy*.
- **lint has zero coverage of `merge_prompt` templates** — no `merge_prompt` in
  `lint.py` / `_lint_predict.py`, so `{variants}` / `{intake}` placeholders in a
  merge template are unchecked.

---

## 3. The rule

> Every value handed to a prompt_compiler is prompt-ready text. It is BAML-rendered
> unless it is already a string; primitives are coerced to string; `None` is `""`;
> and `render_for_prompt()` on the model is the one presenter hook that overrides
> the default.

One rule, one implementation, four call sites. There is no per-channel variation:
`input_data`, `di_inputs` and `context` are the same namespace and obey the same
rule. (`context` is already `dict[str, str]` by contract, so it is conformant
today — evidence the rule is already the norm where someone typed it.)

---

## 4. Design

### 4.1 One normalizer, at the one choke point

Move the render decision **out of the call sites and into `_compile_prompt`**,
the only place that knows whether a compiler will actually be invoked:

- Call sites thread **raw input + the effective renderer**, not a pre-rendered
  view. Safe to do: `input_data` in `_llm.py` and `_tool_loop.py` is pure
  passthrough to `_compile_prompt` — nothing else reads it, and retries reuse the
  built `messages`, not `input_data`.
- `_compile_prompt` performs the split once: inline -> `ri.raw` (unchanged; the
  compiler is never called for inline templates), file-ref ->
  `PromptInput(ri.for_template_ref)`.
- `_dispatch.py:341` `_render_input`'s view-selection and
  `_llm_render.py:218` `_render_and_compile`'s duplicate are deleted.

The Oracle merge path then gets correct rendering **with no edit in `_oracle.py`**.
That is the test of whether the fix is structural rather than another patch: if
`_oracle.py` needs a line, the decision is still living in the call sites.

### 4.2 The rendering function itself

`_llm_render.py:82` `_resolve_var`'s leaf rendering and `renderers.py:455`
`_render_single` collapse into one function implementing §3:

0. **already `Rendered` -> return it unchanged** (AMENDMENT A0, see §9). This rung
   runs BEFORE any `hasattr` probe, so a `Rendered` value never meets
   `hasattr(value, "render_for_prompt")` and never trips its loud `__getattr__`.
   The function is therefore IDEMPOTENT: rendering twice equals rendering once.
1. `render_for_prompt()` if present — wins over everything, including an explicit
   `renderer=` (this is already `_render_single`'s precedence; it is
   `_resolve_var` that must adopt it).
2. explicit `renderer=` (Xml / Delimited / Json).
3. BAML `describe_value` for models and containers of models, plus the two
   framework container shapes (`list[ToolInteraction]`, `dict[str, BaseModel]`).
4. `str(value)` for everything else; `None -> ""`.

Step 4 is the change: `_render_single`'s current "primitives pass through
unchanged" branch is the anomaly, and `_resolve_var`'s missing step 1 is the
other. Everything else already exists and moves unchanged.

**Why rung 0 matters more than it looks.** Idempotence is what converts this
design's central invariant from a rule people must remember ("nobody may render
twice") into a property of the function ("rendering is safe to do anywhere"). The
first is unenforceable across consumer code — the public `render_inputs` is
exported, and every compiler written the way the docs teach calls it. The second
needs no memory at all. It also dissolves three separate edges this design
originally carried: the `build_vars` "prerequisite" in §5, the need to thread a
`renderer=` parameter through `_llm.invoke_structured` and
`_tool_loop._build_loop_preamble` purely so one blessed site can see
`node.renderer`, and the brittleness of an invariant enforced by prose.

### 4.3 The type, with teeth

```python
class Rendered(str):
    """Prompt-ready text. Already rendered; it has no fields."""
    def __getattr__(self, name: str):        # only fires for names str lacks
        raise PromptInputError.build(
            f"{name!r} on already-rendered prompt text",
            hint="input_data values are rendered strings — use the value directly, "
                 "or move structured work into render_for_prompt()/merge_pre_process.",
        )

PromptInput = Mapping[str, Rendered]

class PromptCompiler(Protocol):                       # _llm_protocols.py:33
    def __call__(self, template: str, input_data: PromptInput, *a, **kw) -> list[Any]: ...
```

Typing and rendering are **both required, and they do different jobs**: the
annotation makes the shape discoverable (mypy, the API manifest, the docs); the
`Rendered` subclass makes a wrong assumption *loud at runtime*, which the
annotation alone cannot do because `PromptCompiler` is a structural Protocol and
every consumer compiler annotates `input_data: Any`.

Verified behaviour of the subclass:

```
str ops still work: PROSE 5 prose!
LOUD via getattr-with-default -> 'text' on rendered text
LOUD via hasattr -> 'model_dump' on rendered text
```

`getattr(x, name, default)` and `hasattr` swallow only `AttributeError`, so a
`NeographError` subclass propagates. This is exactly the ticket's acceptance
criterion: the compiler that scored 0/6 would have crashed on row one instead of
answering about nothing, and the `hasattr(v, 'model_dump')` half of the
hand-rolled both-ways renderer announces itself instead of silently taking the
wrong branch.

**Totality is deliberate.** `Mapping[str, Rendered]` is only true if the boundary
coerces every value, including primitives and non-renderable objects (a
`FROM_CONFIG` live handle such as a `RateLimiter` can legally be a di_input). A
`Rendered | int | list | Any` union would be `Any` wearing a hat and would
re-admit the "check before you use it" branching this ticket exists to remove.
Coercion is free for the default compiler, which already does `str(vars[name])`
at `prompt.py:46`.

A live handle rendering to `<RateLimiter object at 0x…>` is ugly but must **not**
be a hard failure: di_inputs are injected for every template-usable binding, not
only the ones a template references, so failing loud there would break nodes that
declare a `FromConfig` resource and never mention it in the prompt. The loudness
budget is spent on attribute access, which is where the silent bug lives.

### 4.4 Bare values get a key — all four sources of them

`_input_shape.py:121` `_extract_single_type` already iterates `state.keys()` and
matches on `attr_name` — it just discards the name. Return `(name, value)` and key
the normalized dict by it, so single-type and fan-in produce the same dict under
the same key and the "sometimes a dict, sometimes a bare value" branch disappears
along with the str-vs-model branch.

**AMENDMENT A2 (see §9): `_extract_single_type` is one of FOUR producers of bare
values, and the original text of this section wrongly implied it was the only
one.** The other three are:

- `merge_pre_process`, which "fully replaces input_data" and is typed
  `BaseModel | dict[str, Any] | str` (`_oracle.py:293-313`);
- the public `compile_prompt(template, input_data=SomeModel)`, whose docstring
  explicitly advertises "a Pydantic model … or `None`";
- `render_prompt(node, input_data)`, same shape.

`build_rendered_input`'s non-dict branch returns a scalar and `for_template_ref`
passes it through unwrapped, so all four reach the compiler unkeyed today.

**The naming rule for a bare value therefore belongs in `renderers.py`, at the
normalizer — NOT at any of the four call sites.** Putting it in `_oracle.py` to
serve `merge_pre_process` would be this design's own definition of failure
(§4.1). Note also that `prompt.render_inputs` currently resolves this case by
silently dropping a bare value to `{}` (`prompt.py:96-102`) — a quiet seam of
exactly this family, to be decided deliberately rather than inherited.

`Construct.input` port values keep their existing `neo_subgraph_input` key plus
the type-name alias (`renderers._alias_subgraph_input_port`) — unchanged, already
additive.

### 4.5 di_inputs render through the same path

`_inject_di_inputs` currently stashes `binding.resolve(config)` raw. Route the
resolved map through the same renderer before it reaches the compiler.

Consequences, all wanted:

- A DI model with its own `render_for_prompt()` finally controls its presentation.
  Today that method is ignored entirely on the DI channel, so a user who wrote a
  presenter for their context bundle gets `node_id='n1' project_root='/tmp'`
  regardless.
- A DI model whose `render_for_prompt()` returns a `BaseModel` also flattens its
  fields into the namespace, same as upstream inputs.
- The documented collision rule is unchanged: di_inputs (and their flattened
  fields) are the base layer, upstream outputs shadow on top.

### 4.6 `raw_inputs`: an opt-in second channel (REINSTATED — amendment A1)

An opt-in `raw_inputs` kwarg, introspection-gated through the existing
`prompt_compiler_params` / `_ACCEPT_ALL` filter exactly as `di_inputs` and
`context` already are:

```python
def prompt_compiler(template, input_data: PromptInput, *, raw_inputs, ...):
    text  = input_data["claims"]   # always Rendered
    model = raw_inputs["claims"]   # opt-in, and the compiler DECLARED it
```

**This section previously rejected it. That rejection was wrong and is
withdrawn** (decision: user, 2026-08-11; see §9/A1).

The rejection argued that structured manipulation has two homes upstream of the
compiler — `render_for_prompt()` and `merge_pre_process`. Neither can serve the
ordinary case: **`render_for_prompt()` is per-MODEL, not per-NODE**, so it cannot
give node A a `{% for c in claims %}` loop and node B a flat summary of the same
upstream model; `merge_pre_process` exists only on the merge path.

The rejection also mis-stated the invariant. The ambiguity this ticket kills is
**one channel with two possible shapes** — not **two channels with one shape
each**. The latter is precisely the design this repo already accepted for
`di_inputs`, and a compiler that declares `raw_inputs` has unambiguously asked
for objects. `input_data` stays totally `Mapping[str, Rendered]` either way.

Consequently `renderers.py:443-447` (BaseModel / `list[BaseModel]` children
preserved for dotted access) **stays**, and with it the documented feature at
`website/src/content/docs/concepts/renderers.mdx:229` and `lint.py:580`'s
tolerance of dotted placeholders. Had totality won instead, all three would have
had to be deleted **in this same commit** — shipping totality while three
artifacts still advertise structure is the worst of both.

---

## 5. Blast radius

- **Behaviour change**: `{variants}` in merge templates goes from
  `[T(label='a'), T(label='b')]` (Python repr via `substitute`'s `str()`) to a BAML
  block. Affects `examples/observable_pipeline.py`,
  `examples/03_oracle_ensemble.py`, `examples/vs_langgraph/03_map_reduce.py`,
  `examples/lead-outreach/pipeline.py`.
- **Behaviour change**: primitives reaching a compiler become text. A custom
  compiler doing `input_data["count"] + 1` or iterating a plain list must move
  that work upstream. Compile-time-visible via the new type.
- **Behaviour change**: inline `${var}` now honors `render_for_prompt()`. A model
  with a presenter renders differently in inline prompts than it did.
- **Tests to move**: `tests/modifiers/test_oracle.py`
  `TestMergePromptUpstreamContext` (asserts the merge dict — keys survive, value
  types change); `tests/test_example_map_reduce.py` (docstring states the models
  contract).
- **Guard to retire**: `tests/test_guards_examples.py`
  `TestExampleMergeCompilerReadsVariants` (the iu05 AST band-aid) exists only
  because the shape was ambiguous. It is replaced by the type-level rule, not
  carried alongside it.
- **Prerequisite**: `DefaultPromptCompiler.build_vars` (`prompt.py:190`) re-renders
  via `render_inputs(input_data)`, which does
  `hasattr(value, "render_for_prompt")` and would now raise on a `Rendered`. It
  must pass an already-normalized `PromptInput` through untouched. Killing that
  double-render is wanted independently: it is what hides the bug from the
  framework's own compiler.
- **Docs**: `concepts/prompt-compiler.mdx` (the isinstance-dance pattern),
  `concepts/evaluating-prompts.mdx` (the `input_data` row),
  `concepts/renderers.mdx`, `walkthrough/oracle-ensemble.mdx` and
  `node-api/modifier-kwargs.mdx` (the `merge_pre_process` rows), plus the
  regenerated API manifest.
- **Lint gap closes on the way**: once the merge path shares the normalizer it can
  share the placeholder prediction, which it has never had.

---

## 6. Guards (write failing first)

1. **Behavioural, red first**: one compiler, one pipeline, a think node and an
   Oracle merge — assert both invocations receive `Rendered` under the same key.
   This is the ticket's repro and must fail on `main` before anything is written.
2. **Attribute access on a rendered value raises** — pinning that
   `getattr(v, 'text', '')` does not silently return `""`.
3. **Always-on totality assertion at the seam** (replaces the two guards this
   section originally listed — amendment A3). One check in `_compile_prompt`
   immediately before `runtime.prompt_compiler(...)`: every value in the mapping
   is `Rendered`. This is worth more than both original guards combined because
   it fires in **every consumer's process**, not only in this repo's CI.

   The two it replaces, and why:
   - *"`runtime.prompt_compiler(...)` may be called only from `_compile_prompt`"*
     is **already true and already vacuous** — grep finds exactly one call site
     (`_llm_render.py:215`). It pins a property that was never violated, and it
     does not constrain the actual failure mode, which is call sites
     **pre-rendering**, not call sites **invoking**.
   - *"no module outside `_llm_render.py` may call `build_rendered_input` to
     choose a compiler-facing view"* is **not mechanically checkable as worded**:
     an AST guard cannot read intent, and `prompt.py:101` calls
     `build_rendered_input` legitimately today.
4. **Call-site allowlist, by concrete name**: a structural guard listing the
   functions permitted to call `build_rendered_input`. A plain allowlist over
   names is checkable; "to choose a view" is not.
5. **Totality**: every value in the mapping handed to a compiler is `Rendered` —
   asserted across the four channels, not one.
6. **Three-surface parity** (`@node` / declarative / programmatic) on the
   single-type keying change in `_extract_single_type`, per CLAUDE.md's rule.
7. Check-fixtures: a `should_pass` for a compiler written the obvious way.

---

## 7. Work order

1. Red tests (guards 1 and 2).
2. The one rendering function (§4.2) — collapse `_resolve_var`'s leaf rendering
   and `_render_single`; coerce primitives; presenter first.
3. `Rendered` + `PromptInput` + `PromptCompiler` annotation + `PromptInputError`,
   wired through `neograph/__init__.py`'s `__all__` (the public contract; the
   module `_` prefix is advisory only).
4. Normalizer in `_compile_prompt`; delete the two call-site copies; verify
   `_oracle.py` needs no edit.
5. `_extract_single_type` name plumbing (§4.4).
6. di_inputs through the renderer (§4.5); `build_vars` double-render removal.
7. Guards 3-7.
8. Examples, tests, docs, API manifest; retire the iu05 AST guard.
9. Lint coverage for `merge_prompt` templates (may split to its own ticket).

---

## 8a. Amendments from the codebase disease scan (2026-08-11/12)

Four review lenses ran against this branch's tree (artifacts:
`.claude/code-review/neograph-l2a7w/`), each building its own inventory without
being shown this document's. They found **24 instances** where this document's
author had found 7, and invalidated four of its sections. Six of six spot-checked
claims verified independently.

| ID | Amendment | Section |
|----|-----------|---------|
| A0 | Normalizer is IDEMPOTENT — rung 0, `Rendered` in -> `Rendered` out, before any `hasattr` probe | §4.2 |
| A1 | `raw_inputs` REINSTATED as an opt-in second channel; the rejection was wrong | §4.6 |
| A2 | Bare values come from FOUR sources, not one; the naming rule lives in `renderers.py` | §4.4 |
| A3 | Guard 3 was vacuous, guard 4 unmechanizable; replaced by an always-on totality assertion | §6 |
| A4 | `context` — decide explicitly (see below) | this section |
| A5 | `PromptInputError` must NEVER subclass `AttributeError`, at any point | §4.3 |

**A4, `context`.** §3 asserts `input_data`, `di_inputs` and `context` are one
namespace under one rule, but `context` is passed as a separate kwarg typed
`dict[str, str]`, so its type already pins it to text. Decision: **exempt, because
the type pins it** — and separately, `context` does not even reach the template
today (`DefaultPromptCompiler.__call__` swallows it in `**_kw`; filed as
neograph-cbfd9), and its `str`-ness is asserted by a `cast` that no validation
backs (neograph-ufqr7). Both are fixed on their own tickets, not here.

**A5** is small and load-bearing: making `PromptInputError` subclass
`AttributeError` "for compatibility" at any point during implementation silently
restores the exact swallow the loudness is bought to defeat.

**What the scan added that this document did not have** — the four findings that
most change the work:

1. `renderers.render_input` is **public, documented, has zero callers in `src/`**,
   and diverges from `build_rendered_input` (different flatten-merge discipline,
   no port alias). Left alone it guarantees a fourth divergence.
2. `RenderedInput.available_keys_inline` / `available_keys_template` **already
   compute** the key sets `lint._predict_input_keys` independently re-predicts,
   and nothing reads them. `lint` returns `set()` for single-type inputs, which
   §4.4 silently makes wrong — so lint must move in lockstep with §4.4.
3. `_oracle._build_upstream_context` reads the same keys as
   `_extract_fan_in_dict` **without** its Each/Loop unwrapping, so the merge path
   is wrong in *shape*, not only in rendering (neograph-13k4i).
4. The disease is an instance of a rule this repo has already written down —
   `CLAUDE.md`'s `effective_producer_type` single-source-of-truth rule and the
   `ModifierCombo` retrospective. This is the third recorded occurrence.

## 8. Open questions

- Whether the lint extension for `merge_prompt` templates lands here or as a
  follow-up ticket citing this document.
- Whether `Rendered.__getattr__` should allow a dunder/protocol allowlist. Current
  position: no allowlist — values are wrapped at the last step before the
  compiler, so no framework code probes them, and a consumer's both-ways helper
  *should* fail loudly.
