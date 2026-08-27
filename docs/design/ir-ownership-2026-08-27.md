# IR ownership: "which value is meant" is answered once, as data

Date: 2026-08-27. Status: proposed. Scope: the internal representation of value
resolution — which state field satisfies an input, which member supplies a
boundary, which key receives a fed-back or fanned-out value — and the contract
by which every other module consumes that answer.

Companion tickets: neograph-5suot, neograph-t1nbp (closed), neograph-fnlrx,
neograph-af8ro, neograph-22jvj, neograph-5fvsu, neograph-6ars9, neograph-lmjn5.

---

## 1. The decision

**`_ir_normalize.py`, running inside `Construct.__init__`, owns every answer to
"which value is meant." The answer is reified as a value of a closed sum type,
`Source`, stored on the IR in a single normalizer-written table. Every
consumer — runtime extraction, both exporters, validation, lint, and error
diagnostics — reads `Source` values through one query surface and interprets
them by named reads. No module outside the normalizer may apply a
type-compatibility predicate to *choose* among values, fields, producers, or
markers.**

Two capability removals make the wrong answer unwriteable where neograph owns
the bag, rather than merely banned:

1. **`StateBus` loses `keys()`.** The state bag becomes non-enumerable by
   interface. A runtime scan over state stops being a bad pattern and becomes
   code that does not compile against the bus.
2. **`Source` values are constructible only in `_ir_normalize.py`** — the same
   closed-set construction ban G3 already applies to the five string fields,
   applied to one typed field instead. A consumer cannot fabricate a
   resolution; it can only read one.

Where neograph does *not* own the bag (`Annotated` metadata), the rule is
different and stated in §6: one ingestion site parses the foreign bag into
owned IR and **refuses** authored ambiguity.

This is Information Expert applied literally: assembly is the only moment when
declaration order, the modifier-adjusted producer set
(`effective_producer_type`), and the enclosing boundary all coexist. The
runtime has the values but has lost the declarations' intent; the exporters
have the declarations but every re-derivation has measurably diverged from the
runtime. The expert is the assembly-time normalizer, so the logic sits there —
all of it, in one module, which is also the High Cohesion answer: t1nbp showed
the rules interlock (port-vs-peer precedence, producer ordering, framework-key
exclusion), and interlocking rules split across five modules is exactly the
five-fix history.

---

## 2. The disease, re-verified

The task's evidence checks out, with three corrections recorded in §7.

Five independent fixes converged on resolve-once-store-as-IR-field:
`fan_out_param` (2026-04), `handoff_param` and `handoff_channel` (2026-07),
`output_from` (2026-08), `input_source_field` (2026-08, commit `1adaad0`).
Guard G3 (`IR_FIELDS` frozenset, `tests/test_guards_llm_runtime.py:1090`) makes
each field single-writer. It has not stopped the disease, because it governs
**writing** the resolution while nothing governs **asking the question**: every
consumer still has the raw materials (`construct.nodes`, `ProducerMap`, the
state dict, `FieldInfo.metadata`) and independently decides whether to read the
field or scan the bag. neograph-fnlrx is the proof that this leaks even when
the pattern is applied correctly: `output_from` shipped with three readers
(`_subconstruct.py:260`, `_lint_consumers.py:143`, `_validation_outputs.py:203`)
and both exporters simply never asked — `resolve_end_node_sources`
(`_agent_spec_boundary.py:238`) takes `construct.nodes[-1]` positionally, and
`_dump_sub_construct` (`_spec_dump.py:366`) drops the field, so two constructs
differing only in `output_from` dump identically.

**Why the correct pattern leaks: a field is storage, not a contract.** A
consumer must *know* the field exists to read it, and the consumer that
doesn't know compiles, runs, and exports without complaint. Three false
docstring citations ("mirrors the assembly-time validator", "agrees with
`_agent_spec_boundary`...one rule, not a fifth answer" —
`_subconstruct.py:58-62`, still present) survived review precisely because
parity was asserted in prose instead of achieved by a shared call. The closure
is to stop exposing fields and start exposing **answers**: functions whose
return type can only be produced by the resolver. Then a new rule lands inside
an answer every consumer already calls, and "the exporters did not learn the
field" becomes structurally impossible — there is no per-consumer learning
step left.

---

## 3. Two kinds of ambiguity, two mechanisms

The refusal experiment (neograph-5fvsu: 33 suite failures, Portal-downstream
consumers made unwritable) and the Portal design that already works both point
at the same taxonomy, which this decision adopts as the rule:

- **Authored ambiguity** — visible at assembly, fixable by the author naming
  what they mean: two port-typed params, two same-typed dict-form keys on a
  Loop, two `Carried` markers on one field. **Refuse at assembly**, naming the
  candidates. This is already house posture at `_param_classify.py:66-76`,
  `_construct_graph.py:86-93`, `_di_classify.py:245-256`; three other sites
  silently picked, and they migrate to refusal.
- **Emergent ambiguity** — a runtime fact no author can name at assembly:
  which mesh member ran last, which branch arm executed. **Funnel**: the
  mechanism that creates the fact writes the answer into one framework-owned
  named channel at the moment it happens, and consumers read that channel by
  name. `handoff_channel` (written only by the `Command` update, read at
  `_input_shape.py:127`) is this mechanism, already shipped and guarded.
- **Everything else** — one right answer derivable at assembly. **Resolve
  once**, store a `Source`.

Scanning was only ever a degenerate funnel whose "channel" is the whole bag.
The taxonomy also explains why universal refusal failed: it treated emergent
ambiguity as authored, demanding a name the author cannot know. Any future
"should this refuse?" question is answered by classifying the ambiguity, not
by re-running the debate.

---

## 4. The resolved structure and the consumption contract

### 4.1 `Source` — the closed sum (new module `_ir_source.py`, ~80 lines)

```python
@dataclass(frozen=True)
class PeerField:          # a named upstream producer's state field
    field: str

@dataclass(frozen=True)
class Port:               # the enclosing construct's input port (neo_subgraph_input)
    pass

@dataclass(frozen=True)
class EachItem:           # the fanned-out item (neo_each_item)
    pass

@dataclass(frozen=True)
class LoopCarry:          # the node's/construct's own append-list, latest element
    pass

@dataclass(frozen=True)
class HandoffChannel:     # Portal mesh channel, entry-keyed
    channel: str

@dataclass(frozen=True)
class LastPresent:        # ordered candidates, type-filtered AT ASSEMBLY;
    fields: tuple[str, ...]   # runtime takes the last one present. Presence
                              # is the only runtime question left.

Source = PeerField | Port | EachItem | LoopCarry | HandoffChannel | LastPresent
```

`LastPresent` is the honest encoding of the two legitimate cases the task's
constraints protect. A multi-arm branch boundary is satisfied by a different
node per arm: the *candidate set and its order* are assembly facts (the
type-compatible members of `item_field_names`, declaration order); only *which
arm ran* is a runtime fact, and that reduces to field presence. The runtime
interpreter does named reads in reverse order and returns the first present
value — no `isinstance` at runtime, because the type filtering already
happened where the types are known. The same variant carries the loop-carry
and nested-port precedence residues that 5suot's measurement characterized
(explicitly ordered: carry before seed, inner producer before outer port),
turning "later wins" from an emergent property of dict ordering into a written
rule.

### 4.2 Where it lives

- `Node.sources: dict[str, Source] | None` — one entry per consumer slot:
  every dict-form input key, or the single-type input under its one key. This
  **absorbs and retires** `fan_out_param` (an `EachItem` entry),
  `handoff_param`/`handoff_channel` (a `HandoffChannel` entry), and
  `input_source_field` (a `PeerField`/`Port` entry). `oracle_gen_type` stays a
  separate field — it is a type inference, not a which-value choice.
- `Construct.boundary_source: Source | None` — `PeerField` when `output_from`
  names the member; otherwise `LastPresent` over the type-filtered
  `item_field_names`. Replaces the runtime reverse scan as the *authority*
  (the scan survives only as the interpreter of `LastPresent`).
- `Construct.input_source: Source | None` — the parent-side answer to "which
  parent field feeds this sub-construct's port". This is the static binding
  5suot's unknown #5 requires; it is the same rule as
  `resolve_single_type_source` with the sub-construct as the consumer.

Stamping on the IR (rather than a parallel `ResolvedConstruct`) follows the
established D10 shape — `handoff_channel` is "read here WITHOUT any signature
threading" — and `normalize_ir` in `Construct.__init__` already guarantees no
un-normalized Construct is observable, which is the property a separate
lowered IR would otherwise buy (§8).

### 4.3 The consumption contract

One rule, four consumer classes, no exemptions:

> A module that needs "which value is meant" does exactly one of:
> **(a)** read a `Source` and interpret it via `read_source(bus, source)` —
> the single ~25-line runtime interpreter in `_input_shape.py`, the only place
> `Source` meets state; **(b)** serialize `Source` values obtained from
> `iter_resolved_edges(construct)`; **(c)** render the contents of a
> `Resolution` object the resolver returned. Nothing else. In particular, no
> module outside `_ir_normalize.py` applies `_types_compatible` /
> `issubclass` / `isinstance` to **select** among candidates.
> Type-compatibility for **verification** of an already-named pair (what
> `_check_fan_in_inputs` does) remains legal everywhere — checking is not
> choosing.

- **Runtime** (`_input_shape`, `_subconstruct`, `factory`): `read_source`.
  Named reads with optional presence, exactly the shape
  `_extract_single_type`/`_source_candidates` already has; the remaining
  scans (`_scan_subgraph_input`, `_scan_subgraph_output`, the
  `_extract_loop_reentry` presence-probe fill) become interpretations of
  stamped `Source` values and their scan bodies are deleted.
- **Export** (both `_agent_spec*` and `_spec_dump`):
  `iter_resolved_edges(construct) -> Iterator[tuple[ConsumerSlot, Source]]`,
  a total fold over the tables. Exporters serialize edges; they do not derive
  them. `resolve_end_node_sources` reads `boundary_source` instead of
  `nodes[-1]`. When a new resolution rule lands in the normalizer, the edge
  stream changes and both exporters change behavior **without an edit** — the
  fnlrx class closes because there is no exporter-side rule left to update.
- **Validation and lint**: read the same tables. `_framework_field_reads`
  (lmjn5) reads `boundary_source`'s candidate fields instead of re-deriving
  naming.
- **Diagnostics**: the resolver's entry points return a `Resolution` sum —
  `Resolved(source)` or `Unresolved(candidates: tuple[Candidate, ...])`,
  where each `Candidate` carries the producer label and a resolver-computed
  near-miss reason (`each_dict_of_compatible_element`,
  `list_field_of_compatible_element`, plain incompatible). `_suggest_hint`
  and `_build_no_producer_error` become renderers of `Unresolved`: the
  Each-dict and list-field probes they currently run themselves
  (`_validation_inputs.py:348-374`) move into the resolver, computed by the
  same `_types_compatible` at the single site. This is the shape that honours
  the maintainer's correction: the read-only diagnostic consumer and the
  authoritative runtime consumer follow the **same** pattern — both read the
  resolver's output object, neither touches a bag. "Diagnostic-only" stops
  being a category, because there is nothing left it would exempt you from.

A corollary, enforced in review rather than by tooling: **parity is achieved
by calling, never by asserting.** A docstring claiming a module "agrees with"
or "mirrors" a named rule is deleted and replaced by a call to that rule; the
three false citations fnlrx documents are the measured cost of the prose form.

---

## 5. What makes the wrong answer unwriteable — and what it does not cover

**Literal impossibility, runtime.** Delete `StateBus.keys()`
(`_state_bus.py:52,72,94`). Verified: its one caller outside the bus module is
`_scan_subgraph_input` (`_subconstruct.py:39`) —
`grep -rn "\.keys()" src/neograph` shows every other hit is `model_fields`
introspection, `snapshot_state`'s internal copy, or a docstring. After step 2
of the migration the Protocol offers `get`, `get_required`, and whole-bag copy
via `snapshot_state` (which copies without choosing; enumeration stays private
to `_state_bus.py`). A future runtime scan requires re-adding a method to the
`StateBus` Protocol — a visible interface change. A narrow guard pins the
Protocol's method set as a closed set; that is the sanctioned construction-ban
form, not an AST pattern-hunt.

**Literal impossibility, IR.** `Source` construction is confined to
`_ir_normalize.py` by extending G3's existing mechanism from five string
fields to the `sources`/`boundary_source`/`input_source` writers plus the six
`Source` classes. A second resolver cannot exist because its output type
cannot be minted; consumers hold values they can only have received.

**Not impossibility — the export side, stated honestly.** `construct.nodes`
must remain readable (topology, prompts, tools are legitimately walked), so a
determined new exporter *can* still hand-roll `issubclass` over `.nodes`.
Coverage there is threefold, and it is containment, not impossibility:
(a) the paved road — serializing `iter_resolved_edges` is strictly less code
than re-derivation; (b) the **differential-export harness**: fnlrx's
acceptance criterion ("two constructs differing only in `output_from` must not
export identically") generalizes into one parameterized test written once over
every axis of the source table, for both export formats. A leaking consumer
then fails a test that already exists instead of a test someone must remember
to write — the per-fix "did the exporters learn it?" review-memory step is
replaced by a mechanical red; (c) the closed-set guard means the hand-rolled
result cannot masquerade as a `Source`, so the divergence is at least typed as
foreign at review.

**Also not covered:** a wrong rule *inside* the normalizer — a single source
of truth is a single point of failure, mitigated only by the resolver being
the most-tested module in the tree and by refusal shrinking the rule surface
for the authored-ambiguity class; semantic misuse of a correctly-read
`Source`; and foreign bags read outside their ingestion site (§6's rule is a
module-boundary convention plus one guard, not a capability removal, because
`FieldInfo.metadata` is an open Python list neograph cannot close).

---

## 6. The ownership boundary: bags neograph does not own

`_is_carried` (`_output_classify.py:101`) scans `FieldInfo.metadata`, which
user code populates before neograph sees it, and returns the first `Carried`
(neograph-22jvj). The resolve-once rule cannot apply — there is no single
assembly site that *constructs* this bag, and neograph cannot define
precedence semantics ("first marker wins") on markers it did not order: any
such rule invents intent the author never expressed. All ambiguity in an
authored bag is authored ambiguity, so by §3 the answer is **refusal**, and
the Portal counter-case cannot arise here because nothing about `Annotated`
metadata is a runtime fact.

The general rule: **a foreign bag crosses the boundary exactly once, through a
parser that refuses ambiguity and emits owned IR; no interior module touches
the raw bag.** `output_markers` is already the single predicate both consumers
use — the ingestion site exists; it just doesn't refuse yet. The fix is
`_di_classify.py:245-256`'s collect-then-refuse copied onto `_is_carried`,
plus confining `FieldInfo.metadata` access to `_output_classify.py` and
`_di_classify.py` (the two parsers), pinned by extending the closed-set guard.
This differs from §5 deliberately: for owned bags we remove the capability to
enumerate; for foreign bags we cannot, so we confine enumeration to named
parser modules and make the parsers refuse.

---

## 7. Corrections to the commissioning findings

Verified against `develop` at `df3d4ff`. Findings 2, 3, 4, 5 are correct as
stated. Two need amendment and one constraint is overstated:

1. **`StateBus.keys()` is not the only runtime enumeration surface.**
   `_scan_subgraph_output(eligible=None)` scans a raw `dict` returned by the
   child graph's invoke — `sub_result.values()` at `_subconstruct.py:112` —
   reached from `_agent_spec_dispatch.py:150` and documented at
   `factory.py:551`. And `_extract_loop_reentry`'s multi-key branch
   (`_input_shape.py:70-88`, af8ro's runtime half) chooses destinations by
   per-iteration presence-probing without ever calling `keys()`. Deleting
   `keys()` closes the bus; these two are closed by steps 2 and 3 below.
2. **The Portal mode-(b) constraint is true at parent assembly but not at
   dispatch.** The emitted flow passes `from_agent_spec(flow)` →
   `Construct(...)` → `normalize_ir` at `_agent_spec_dispatch.py:118` *before*
   it is compiled and invoked — the same gate as a hand-written pipeline, as
   the module itself documents. So its `boundary_source` exists before
   `_finish` runs; `_finish` is simply not handed `sub` (it is in scope in
   `_prepare`). "Assembly time" in this design means Construct-construction
   time, wherever that happens — late assembly goes through the same single
   site. The `eligible=None` scan is therefore removable, not load-bearing,
   and the docstring claim that "the type scan is the only resolution
   available" (`_subconstruct.py:96-99`) joins the false-citation list.
3. **`snapshot_state` framing confirmed**: its three callers
   (`_wiring_oracle_each.py:145,203,273`) copy the whole bag into `Send`
   payloads and never choose from it. Whole-bag copy is compatible with a
   non-enumerable public interface.

---

## 8. Considered and rejected

- **Refusal-on-ambiguity as the universal mechanism.** Measured (5fvsu): 33
  failures; makes a correct Portal-downstream program unwritable — the inverse
  of the product's restriction. Retained only for the authored-ambiguity
  class, where it is already house posture.
- **Capability-token / NewType state keys** (keys mintable only by `naming`).
  Addresses *forging* a key, but the disease is *choosing* among values, and a
  typed key does not prevent iterating a dict of them; enumeration survives.
  Also fights Python's type erasure at runtime instead of using the cheaper,
  stronger move of removing the enumeration method. Rejected.
- **AST/lint guard as primary mechanism.** Maintainer-rejected form.
  Retained only as narrow construction bans on closed sets (the existing G3
  mechanism, the StateBus Protocol method set, the `Source` classes, the two
  metadata-parser modules) — pins on what the architecture already makes
  structural, not the mechanism itself.
- **A separate lowered IR** (`ResolvedConstruct` distinct from `Construct`).
  The cleanest theoretical shape — immutable, total, impossible to observe
  half-resolved. But it doubles the object model, changes every consumer
  signature, and buys a property `normalize_ir`-in-`__init__` already
  provides (no un-normalized Construct is observable). Revisit only if
  multi-stage lowering becomes real; 5suot's experiments did not need it.
- **Threading resolutions through call signatures** instead of stamping the
  IR. Explicitly rejected by precedent D10 (`handoff_channel` exists to avoid
  signature threading); it also scatters the answer across call chains, which
  is Low Coupling inverted.
- **Fixing the six tickets individually and stopping.** That is the measured
  status quo: five convergent fixes in five months, three false parity
  citations, and a sixth site (fnlrx) that leaked while copying the pattern
  correctly. The pattern is right; its delivery as per-fix fields is what
  fails.

---

## 9. Migration path

Each step lands independently, gated by `make release-gate`, TDD-first. Order
matters only where stated.

- **Step 0 — foundations.** `_ir_source.py` (`Source`, `Resolution`,
  `Candidate`); extend G3 to the new writers and the `Source` construction
  ban; land the **differential-export harness** parameterized over source-table
  axes, initially covering `output_from` (this is fnlrx's regression test,
  written multi-node so the positional rule cannot agree by coincidence).
- **Step 1 — neograph-fnlrx (P1).** Normalizer stamps
  `Construct.boundary_source` (`PeerField` from `output_from`, else
  `LastPresent` over type-filtered `item_field_names`). `_subconstruct.py:260`
  interprets it; `resolve_end_node_sources` reads it (also fixing its other
  two divergence axes: `construct.output` for EndNode properties,
  `iter_with_arms` for the walk); `_dump_sub_construct` dumps `output_from`.
  Delete the false docstrings at `_subconstruct.py:58-62` and `:96-99`.
- **Step 2 — neograph-5suot unknown #5, then the tooth.** Normalizer stamps
  `Construct.input_source` at parent assembly (same rule as
  `resolve_single_type_source`, with the loop-carry and nested-port residues
  encoded as ordered `LastPresent`). `_scan_subgraph_input` becomes
  `read_source`; thread `sub` to `_agent_spec_dispatch._finish` so the
  `eligible=None` path dies. **Then delete `StateBus.keys()`** and pin the
  Protocol method set. This step also hands 5suot its static channel binding.
- **Step 3 — neograph-af8ro.** Loop self-feedback destination resolved at
  assembly into `Node.sources` (`LoopCarry` entry); refuse when two same-typed
  non-self keys make it authored-ambiguous (extending the
  `_construct_graph.py:86` refusal to all three surfaces); export and
  `_extract_loop_reentry` read the stamp; delete the per-iteration
  presence-probe fill.
- **Step 4 — neograph-6ars9 and the error builder.** Resolver entry points
  return `Resolution`; `_suggest_hint`'s two probes move into
  `Unresolved.candidates`; `_build_no_producer_error` and the hint render
  resolver output. Last `ProducerMap` selection scans outside the normalizer
  deleted.
- **Step 5 — neograph-lmjn5.** `_framework_field_reads` reads
  `boundary_source` candidates.
- **Step 6 — neograph-22jvj and the foreign-bag rule.** `_is_carried`
  collect-then-refuse; confine `FieldInfo.metadata` to the two parser
  modules; guard the closed set.
- **Step 7 — collapse.** Retire `fan_out_param`, `handoff_param`,
  `handoff_channel`, `input_source_field` into `Node.sources` (0.x, no
  shims); G3's field set shrinks to the table writers plus `oracle_gen_type`.
  Rewrite AGENTS.md's "the pattern to copy" paragraph as the **invariant**:
  the taxonomy of §3, the contract of §4.3, and the rule that a new
  which-value question is a new `Source` variant plus normalizer rule — never
  a new consumer-side predicate.
- **Step 8 — close neograph-5fvsu** with the recorded decision: single-type
  `inputs=` after a mesh is emergent ambiguity, so resolve-to-immediate-
  upstream plus the existing DeprecationWarning is the endpoint; no narrower
  refusal is added because §3 now answers the classification question
  generally.

One fork genuinely needs the maintainer, at step 7: collapse the four legacy
fields into `sources`, or keep them as fields with `sources` derived.
**Recommendation: collapse.** Two representations of one fact is the disease
in miniature, 0.x tolerates the break, and G3 guarding one field is stronger
than G3 guarding five.

---

## 10. The falsifiable claim

By 2027-02, this design has **worked** if all of the following hold, each
mechanically checkable:

1. Re-running the t1nbp disease sweep (positional type-match selection over a
   bag) finds instances only inside `_ir_normalize.py`. Zero new tickets in
   the family; the five-months/five-fixes rate drops to zero.
2. `StateBus` still has no enumeration method, and the differential-export
   harness covers every `Source` axis for both export formats.
3. The next feature that adds a which-value rule ships as a diff touching
   `_ir_normalize.py`/`_ir_source.py` and tests — with **no edit to either
   exporter, `_input_shape.py` beyond `read_source`, or any lint module** —
   and the exporters' behavior changes anyway, proven by the harness.

It has **failed** if a sixth convergent fix appears outside the normalizer, if
any closed-set guard allowlist grows to admit a second writer or parser, or if
claim 3's feature needs a per-consumer edit — that would mean answers are
still being delivered as storage, and the design should be revisited rather
than patched.
