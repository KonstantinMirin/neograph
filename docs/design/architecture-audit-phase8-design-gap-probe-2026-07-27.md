# Phase 8 design-verification probe (neograph-s7zt3.11)

Date: 2026-07-27
Verdict: **NOT fully scoped as filed** — a genuine, verified design gap exists,
inherited from Phase 4's (neograph-s7zt3.8) equally-unverified premise, and
compounded by Phase 7 (neograph-s7zt3.10).

## The claim under test

Phase 8's filed description (and Phase 4's, which it depends on) asserts that
`_lower_each`/`_lower_oracle`/`_lower_loop`/`_lower_operator` can be reused
"the same way the Node-level path already does" once the item is a
`Construct` instead of a `Node`. This assumes those four functions are already
polymorphic over the wrapped item's type. They are not — three of the four
hard-crash on a `Construct` today.

## Repro (built and run against real source, then deleted)

Built a `Construct(name="sub", nodes=[leaf], input=Out, output=Out) | Loop(...)`
/ `| Each(...)` / `| Oracle(...)` and called `_lower_loop`, `_lower_each`,
`_lower_oracle` directly (bypassing `_lower_construct_item`'s current
Construct branch, which today ignores modifiers entirely — the Phase 4 bug).
All three raised `AttributeError` immediately:

- `_lower_loop(sub, loop, body)` → `'Construct' object has no attribute 'inputs'`
  (line `ni = normalize_inputs(node.inputs)`, then a second crash site two
  lines later on `node.outputs`).
- `_lower_each(sub, each)` → `'Construct' object has no attribute 'raw_fn'`
  (from `inner = _lower_node(node)` → `_lower_generation_step`, which reads
  Node-only mode/prompt/raw_fn fields).
- `_lower_oracle(sub, oracle)` → `'Construct' object has no attribute 'model'`
  (from `variant_models = oracle.models if oracle.models else [node.model] * oracle.n`).

`_lower_operator(sub, operator)` was NOT probed for a crash because reading it
confirms it only ever touches `node.name` — it is already Construct-safe.

**Root cause of the AttributeErrors**: `Node.inputs`/`Node.outputs` (plural,
dict-or-type) vs `Construct.input`/`Construct.output` (singular, type-only) —
the exact plural/singular split AGENTS.md documents. `_lower_loop`'s
self-feedback-edge code reads the plural names unconditionally;
`_lower_each`'s inner-lowering step is hardcoded to `_lower_node` (a
Node-only, `_lower_generation_step`-based dispatch); `_lower_oracle`'s
variant generation is hardcoded to per-variant `model`/`model_tier` swap
(`node.model`, `oracle.models`) — a concept a `Construct` has no equivalent
of.

## Why this isn't just an implementation detail Phase 8 can absorb silently

For **LOOP_OPERATOR** and **EACH_OPERATOR** the fix is a bounded, mechanical
dispatch (below) — no new *semantics*, just missing polymorphism. But for
**ORACLE_OPERATOR** the gap is semantic, not mechanical, and Phase 8
explicitly lists ORACLE_OPERATOR as one of the three combos going through
"the same primitives Phase 7 built":

- Node-level `_lower_oracle` means "N variants of the same node, each hitting
  a different LLM model tier, then merge" — it literally emits N distinct
  `LlmNode`s built via `_lower_generation_step(node, model_tier=...)`.
- Construct-level Oracle, verified in `compiler.py::_add_subgraph`'s
  `ORACLE | ORACLE_OPERATOR` arm, means something structurally different: the
  *same* compiled sub-graph (`subgraph_fn`) is redirected/fanned-out N times
  via `make_oracle_redirect_fn`/`make_oracle_merge_fn` — there is no
  per-variant model swap because a `Construct` has no `.model` field to swap.
- Therefore a correct Construct-level Oracle export is **N copies of the same
  sub-Flow FlowNode wired in parallel + the existing merge-node logic** — not
  a reuse of `_lower_oracle`'s per-variant `LlmNode` generation step. Only the
  merge half of `_lower_oracle` transfers as-is; the variant-generation half
  needs a genuinely new lowering path. This is exactly the class of "claims of
  reuse were wrong" pattern the master doc's own executive summary warns
  about — it just wasn't checked for this specific pairing before Phase 8 was
  filed.

## Concrete design to close the gap

1. **`_lower_operator`**: no change needed — already dispatches only on
   `item.name`, safe for both Node and Construct.
2. **`_lower_loop`**: gate the entire dict-form self-key-detection block
   (`ni = normalize_inputs(...)` through the `dest_prefix` loop) behind
   `isinstance(node, Node)`. For a `Construct` item, `dest_prefix` stays `""`
   unconditionally — a `Construct.input` is a single type, never dict-form,
   so the "which upstream key does the self-edge target" question the
   dict-form logic answers for `Node` simply does not arise for `Construct`.
   Read `_properties_for(node.output)` (singular) in the Construct branch
   instead of `_properties_for(node.outputs)`.
3. **`_lower_each`**: replace the hardcoded `inner = _lower_node(node)` with a
   dispatch: `inner = _lower_node(item) if isinstance(item, Node) else
   nodes_mod.FlowNode(name=item.name, subflow=to_agent_spec(item))` (i.e.
   reuse exactly the Construct branch `_lower_construct_item` already has for
   the bare case). Same dispatch for `inner_inputs`:
   `_properties_for(item.inputs)` for Node vs `_properties_for(item.input)`
   for Construct. `_is_translation_eligible`/`_node_translation` already
   return False-safe for a Construct (isinstance-gated), so no change needed
   there.
4. **`_lower_oracle`**: needs a real branch, not a one-line dispatch swap.
   For `Construct` items: build N `FlowNode`s, each wrapping a *fresh*
   `to_agent_spec(item)` call (verify whether pyagentspec permits the same
   `Flow` object reused across multiple parent `FlowNode`s without name
   collisions, or whether each variant needs its own independently-named
   `Flow` — untested here, flagged as the one remaining open verification
   item), then reuse the existing merge-node construction unchanged (the
   merge half only consumes `gen_outputs`/`outputs` Properties, agnostic to
   how the variant was produced). `oracle.models`/per-variant `model_tier`
   has no meaning for a Construct variant and must be rejected or ignored
   with a clear rule (recommend: reject `Oracle.models` set + Construct item
   as a `ConfigurationError`, since "which model" is meaningless when the
   variant is a whole sub-flow, not a single generation step) — this rule
   itself is new design, not present anywhere in the master doc's matrix.

## Recommendation

Do not implement Phase 8 (or Phase 4) as filed without folding in the above.
Concretely:
- Phase 4's ticket should be corrected to include the `_lower_loop`/
  `_lower_each` dispatch fixes above (mechanical, no design risk) and to
  explicitly scope Construct-level bare ORACLE as its own open design item
  (the N-copies-of-FlowNode variant generation), not a straight reuse.
- Phase 8's ticket should be corrected to drop the implicit assumption that
  ORACLE_OPERATOR-on-Construct is "same primitives, single wrap" — it
  requires the new Oracle-on-Construct variant-generation design from Phase 4
  to exist first, and inherits the open pyagentspec-Flow-reuse verification
  question above. LOOP_OPERATOR and EACH_OPERATOR-on-Construct, by contrast,
  ARE fully closed by the mechanical dispatch fixes in items 2–3 and can be
  implemented as designed once Phase 4 lands correctly.
- The two EACH_ORACLE(_OPERATOR)-on-Construct fail-loud-mirror rows are
  confirmed correct as filed: `compiler.py:511-516`'s rejection is
  unconditional and Construct-agnostic-in-the-relevant-sense (no `.model`/
  variant concept exists to fuse with `Each` either), so mirroring it in
  export needs no new design — this part of Phase 8 is ready to implement.
