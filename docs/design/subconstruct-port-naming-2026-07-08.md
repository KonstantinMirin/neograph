# Sub-construct port naming in template-ref `input_data` (neograph-bluv, F3.4)

**Date**: 2026-07-08
**Source**: `docs/design/prompt-surface-consumer-survey-2026-07-08.md`, F3.4/R3.

## The problem

A sub-construct's single-value input boundary is always delivered under the
framework-internal key `neo_subgraph_input` (`StateKeys.SUBGRAPH_INPUT`). Three
independent IR-level mechanisms rewrite a node's `inputs` dict to this literal
key:

1. **`@node` sub-construct ports** — `_cleanup_inputs_and_register` in
   `_construct_builder.py` rewrites a function parameter matching
   `construct_input=` to `{StateKeys.SUBGRAPH_INPUT: T}` (drops the original
   parameter name).
2. **Declarative sub-constructs** — the user authors
   `inputs={"neo_subgraph_input": T}` directly; there is no rewrite, the
   literal key is the only spelling that ever existed.
3. **The qot6 fan-over-agent auto-wrap** (`_fan_agent_wrap.py`,
   `_synthesize_port`) — Oracle/Each/Loop over an agent/act node with a
   single-key dict-form input gets isolated into a sub-construct whose port
   key is rewritten from the original dict-form key to `neo_subgraph_input`.
   This auto-wrap made the leak MORE common: what used to be a plain peer
   producer read is now, silently, a port read.

By the time a node's prompt reaches a consumer's custom `prompt_compiler`, its
`input_data` dict has the literal `neo_subgraph_input` key and nothing else for
that value. ox-troubleshooting-demo's compilers hand-roll
`_unwrap(input_data, "neo_subgraph_input", "state", "seed")` in every one of
its four prompt compilers — a framework-internal implementation detail four
independent hand-written functions must each know by name.

## Options considered

**(a) Alias the port value under a stable friendly name in template-ref
`input_data`.** Two sub-variants were on the table:
  - (a1) the sub-construct's declared input model's class name (e.g.
    `VerifyClaim`), computed at the rendering layer from the runtime value's
    type;
  - (a2) the original dict-form key preserved through the rewrite (e.g.
    `claim` from `inputs={"claim": VerifyClaim}`).

**(b) Have `build_rendered_input`/`for_template_ref` translate the key** — a
variant of (a) really, differing only in whether the alias is added
(additive) or the key is renamed (destructive). Renaming was rejected outright
because `neo_subgraph_input` is read directly today by declarative pipelines,
scripted shims (`tests/test_composition.py`'s `parity_score_fn` does
`_in["neo_subgraph_input"]`), and the whole qot6/1h8c convention — renaming
would be a breaking change with no migration path for the framework's own
internal contract, not just consumer code.

**(c) Document `neo_subgraph_input` as public-stable.** Rejected per the task
brief — this enshrines the leak instead of killing it; every future consumer
would still need to know a framework-internal literal.

## Decision: (a1) — alias by declared type name, computed at the rendering layer only

Ship (a1), **not** (a2). Reasoning:

- **(a2) requires new IR-level plumbing that the task's file scope excludes.**
  The original key is DELIBERATELY dropped by both `_cleanup_inputs_and_register`
  and `_synthesize_port` today — their docstrings say so explicitly ("The
  original prompt-var name `k` is not preserved... matching manual
  `construct_from_functions(input=...)` wrapping"). Preserving it would mean
  threading a new piece of metadata (the pre-rewrite name) through `Node`,
  surviving `model_copy`, and being available at render time — a change to
  `_construct_builder.py`, `_param_classify.py`, and `_fan_agent_wrap.py`, all
  of which are explicitly OUT of scope for this task (owned by other agents /
  a separate change class). It would also need to agree across THREE
  independent rewrite sites, i.e. three places that could drift out of sync
  again — exactly the class of bug the `effective_producer_type`/`_declared_output`
  monopolies in this codebase exist to prevent.
- **(a1) needs zero IR changes.** The declared port type is already sitting in
  `node.inputs[StateKeys.SUBGRAPH_INPUT]` (or, since the render layer only ever
  sees the runtime *value*, is derivable as `type(value).__name__` directly off
  the dict value) at the exact point `build_rendered_input` runs. One
  render-layer change covers all three rewrite sites for free, because all
  three converge on the same `{neo_subgraph_input: <value>}` dict shape before
  rendering ever happens.
- **A Pydantic model's class name is always a legal template identifier**
  (`BRACE_RE`/`DOLLAR_RE` both accept `[a-zA-Z_][\w.]*`, and a class name is
  already a valid Python identifier by construction) and it never depends on
  which of the three surfaces produced the node -- so it doesn't invent a
  second naming convention consumers have to learn per surface.
- **Every three-surface parity concern collapses to one code path.** Because
  the fix lives in `build_rendered_input` (the single choke point `_dispatch.py`,
  `_llm_render.py`'s `_render_and_compile`/`compile_prompt`, and `render_prompt`
  all funnel through), the `@node` port rewrite, the declarative manual port
  key, and the qot6 auto-wrap rewrite are automatically covered by ONE change,
  with no per-surface branching. This is the strongest form of three-surface
  parity: not "tested separately and kept in sync by discipline" but
  "structurally identical because it's the same code."

## Mechanics

- Scope: **template-ref only** (`RenderedInput.for_template_ref`), matching the
  task's framing ("alias the port value... in template-ref input_data") and
  the actual bug report (consumer `prompt_compiler`s only ever see
  `for_template_ref`). Inline `${var}` prompts are UNCHANGED — they read
  `ri.raw`, and `${neo_subgraph_input}` already works there today via plain
  dict access; there is no leak to fix on the inline side.
- In `build_rendered_input`'s dict branch (`renderers.py`): after the existing
  per-key render + flattening loop, if `StateKeys.SUBGRAPH_INPUT` is present in
  `input_data`, compute `alias = type(input_data[StateKeys.SUBGRAPH_INPUT]).__name__`
  and add `flattened[alias] = rendered_dict[StateKeys.SUBGRAPH_INPUT]` **only
  if `alias` doesn't already collide with an existing `rendered_dict`/`flattened`
  key**. Landing it in `flattened` (not `rendered_dict`) means `for_template_ref`'s
  existing collision rule ("`k not in merged`") makes a real producer field of
  the same name win over the synthesized alias for free — no new precedence
  rule to invent, it reuses the exact mechanism `di_inputs` shadowing already
  established.
- `available_keys_template` picks up the new `flattened` key automatically
  (it's already `inline_keys | set(flattened.keys())`). `available_keys_inline`
  is untouched (built from `raw_dict` only).
- **Back-compat**: `neo_subgraph_input` is NEVER removed from `rendered_dict` —
  this is purely additive. Existing consumers (ox-demo's `_unwrap`, and any
  scripted shim reading `input_data["neo_subgraph_input"]`) keep working
  unchanged. The deprecation note is: `neo_subgraph_input` stays fully
  supported and readable; new consumer code SHOULD prefer the friendly alias
  going forward, but there is no removal timeline forcing a migration.
- **Lint** (`lint.py:_predict_input_keys`): when `include_flattened=True`
  (template-ref prediction) and `StateKeys.SUBGRAPH_INPUT` is a key in
  `ni.by_name`, also add `ni.by_name[StateKeys.SUBGRAPH_INPUT].__name__` to the
  predicted key set — mirroring the runtime rule exactly so lint and runtime
  can't drift (the same failure class `di_inputs`'s lockstep note in
  AGENTS.md calls out). `include_flattened=False` (inline prediction) is
  untouched, matching the runtime scope decision above.

## What this does NOT change

- `neo_subgraph_input` remains the literal IR-level dict key everywhere
  upstream of rendering (state bus, scripted shims, `_extract_input`,
  `_fan_agent_wrap.py`, `_construct_builder.py`). No IR file in the exclusion
  list (`_fan_agent_wrap.py`, `_construct_builder.py`, `_param_classify.py`,
  `_construct_graph.py`) is touched.
- The original dict-form param name (e.g. `claim`, `group`, `text_input`) is
  still NOT resurrected as a template var — `${text_input}`/`{text_input}`
  stay correctly flagged as unresolvable by the existing lint tests
  (`test_sub_construct_port_remapping_flagged`,
  `test_node_decorator_sub_construct_remapping`). Those tests assert on names
  that are NOT the port type's class name, so this change does not touch
  their expected outcomes.
