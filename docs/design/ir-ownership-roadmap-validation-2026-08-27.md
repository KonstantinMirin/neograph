# IR ownership vs. the roadmap: validation of `ir-ownership-2026-08-27.md`

Date: 2026-08-27. Status: review artifact. Subject: does the IR-ownership
decision (`docs/design/ir-ownership-2026-08-27.md`) survive the work already
on the roadmap — Agent Spec export AND import, Hybrid Data Flow
(neograph-ftnxl), the Portal tail (neograph-qtfof), and the GH-filed features
(neograph-iq4a3, neograph-8v1r0, neograph-5suot, neograph-mtwy7/rlapj)?

Method: every claim below was checked against `develop` at `0cbd19c` (file:line
cited) or against the bead text (`bd show <id>`), not against the design doc's
own summary of them.

---

## 1. Verdict

**ADOPT WITH AMENDMENTS.** The core of the decision — normalizer-owned
resolution, the `Source` sum, `read_source`, refusal-vs-funnel taxonomy, the
`StateBus.keys()` removal — survives every roadmap item examined, and in two
places (provides/requires, arm-aware wiring) it is the *enabler* the roadmap
item was going to have to invent anyway. But three of its claims are narrower
than the doc states, and the roadmap exposes exactly where. Five amendments,
each a change to the decision's own artifacts:

### Amendment A1 — `iter_resolved_edges` must be total over ALL data edges, not just the which-value tables

§4.3(b) defines the fold as "a total fold over the tables"
(`Node.sources` / `boundary_source` / `input_source`). Three P1/P2 roadmap
items require exporters to emit data edges that live in **named IR
declarations outside those tables**:

- `Each.over` — a dotted path whose root is a state field name
  (`_each.py:47`, `split_each_path`), read by name at runtime
  (`_wiring_oracle_each.py:61-62`). neograph-qtfof.7 (P1, open) requires the
  exporter to emit a real `DataFlowEdge` from that root's producer into the
  MapNode's `iterated_item`.
- Branch-decision reads — neograph-qtfof.6 (P1, open, ToolNode-synthesis
  decision recorded 2026-08-14) requires a real edge feeding
  `branching_mapping_key`.
- `context=` back-references — neograph-a1x7n item 2: the runtime reads them,
  the exported Flow has no edge for them.

Under the fold-as-specced, each of these lands as a **second, exporter-side
edge enumeration next to the fold** — which is structurally the fnlrx disease
with a named field instead of a positional rule, and a1x7n is the proof it is
already happening. §10 claim 3 ("no edit to either exporter") would be false
for all three features.

**Change**: define `iter_resolved_edges` as *the* data-edge stream of a
Construct: every consumer slot, where a slot is a dict-form input key, a
single-type input, a sub-construct port, a construct boundary, a loop-carry
destination, an `Each.over` collection read, and a `context=` reference.
Named declarations enter the stream as trivially-resolved Sources
(`PeerField` minted by the normalizer from the declared name — resolution is
verification, which the validator already performs). Exporters then serialize
one stream; qtfof.6/.7 and a1x7n become new slot kinds in the fold, not
exporter edits. The differential-export harness parameterizes over slot kinds
for free.

### Amendment A2 — write the growth rule for the closed sum into §4.1

The roadmap implies exactly **one** new variant in the next two quarters:
`Accumulated(channel)` for neograph-iq4a3 (see §2.4). Everything else
resolves into existing variants (§2). That is healthy — but only if the
growth rule is stated, otherwise the sum becomes a tagged union gaining a
variant per feature. **Change**: add to §4.1: *Source variants are in
bijection with state-channel kinds the runtime physically has — peer field,
port channel, each-item channel, carry list, mesh channel, (future)
accumulator channel — never with features. A new variant is admissible only
alongside a sanctioned new-IR-capability event (the AGENTS.md
`_BranchNode`/Portal bar), and the interpreter gains exactly one named-read
arm per variant.* Pre-register `Accumulated` for iq4a3 so it is minted in
`_ir_source.py` and nowhere else.

### Amendment A3 — LastPresent admission criteria

`LastPresent` is the variant every hard case will reach for, and two roadmap
items would misuse it (§2.4, §2.6). **Change**: add to §4.1: *a
`LastPresent` may be stamped only when (a) at most one candidate can be
present at runtime by construction (branch-arm exclusivity), or (b) the
ordering encodes a named, documented precedence rule (carry-before-seed,
inner-producer-before-outer-port). It is never the answer to a many-present
union read (that is `Accumulated`, with reducer semantics), and never an
escape from an authored-ambiguity refusal.* Without this, `LastPresent` is
last-write-wins under a blessed name — the exact Agent Spec semantics the
hybrid-dataflow doc §4.2 identifies as the disease
("behaviorally equivalent to a public variable that every connected node
overwrites").

### Amendment A4 — the import direction gets a section (extend §6)

The decision addresses import only via correction #2 (an imported flow passes
`normalize_ir` — verified, `_agent_spec_dispatch.py:118`). That is necessary
but not sufficient. A foreign `Flow` is a **foreign bag** in exactly §6's
sense, and `loader.from_agent_spec` is its ingestion site. The current
baseline is lossier than anyone has written down:
`_inputs_from_data_edges` (`_agent_spec_node_import.py:75-83`) discards
`source_output` and `destination_input` entirely and silently drops any edge
whose source is not an already-reconstructed top-level item. Two foreign
edges targeting the same `destination_input` — which Agent Spec resolves
last-write-wins — are reinterpreted as neograph fan-in. The hybrid-dataflow
doc's "Import is fine" (§5.1) is overstated against this code.

**Change**: state the import contract as the §6 rule applied to `Flow`:
*foreign edges cross the boundary exactly once, through `from_agent_spec`,
which translates them into IR declarations rich enough that `normalize_ir`
re-derives the same answer the flow declared; any edge it cannot express as a
declaration is refused or recorded as an explicit loss — never silently
re-derived differently.* Two consequences to name explicitly:

1. **Refusal reaches the hot path.** Every silent-pick site migrated to
   refusal (§3) now refuses at *dispatch time* for a mode-(b) planner-emitted
   spec — `_prepare` catches `ConstructError` and routes to
   `on_invalid='route_to_error'` or raises (`_agent_spec_dispatch.py:120-137`).
   This is fail-loud and north-star-consistent, but it is a behavior change on
   a runtime path: each new refusal added by the migration needs a dispatch-
   mode test axis (an emitted spec exercising it), not only an assembly test.
2. **The harness needs an import leg.** The differential-export harness is
   export-only; fnlrx's dump→load reversion shows the loader can silently
   undo a distinction the dumper preserves. For every axis the harness
   claims lossless, add `load(dump(x))` / `from_agent_spec(to_agent_spec(x))`
   re-normalization equality on the stamped tables.

### Amendment A5 — classify output-field population under §3, explicitly

`Carried` / `Selected` (ftnxl.4 shipped, ftnxl.7 open) and the projection
features (`emits=` neograph-8v1r0, `appends=` iq4a3) are the *population*
question — "how does this output field get its value" — not the
*consumption* question the tables answer. They are not homeless, but the
decision never says where they live, and someone will reach for a `Source`
variant. **Change**: add one paragraph: *output population is owned by the
`_output_classify` monopoly (`output_markers` / `project_output_model` /
`splice_carried`) plus the assembly stage. A `Carried` root is a second-order
read: it may root only at a name the node itself declares
(`_output_classify.py:20-23` — an `inputs` key or DI param), so its value
arrives through the root slot's already-stamped `Source`; the path walk after
the root is deterministic field access, not selection. `Selected` is §3's
funnel: the model's pick is a runtime fact written into the named projected
key field; materialization is a membership-checked keyed lookup — checking,
not choosing. Neither gets a `Source` variant.* This also gives
neograph-ftnxl.15 (marker-blind Agent Spec schema export — the "third
divergent schema consumer") its frame: it is the *projection* monopoly
leaking to a consumer that didn't call it, the same disease shape in the
sibling dimension, fixed by routing `model_to_agent_spec_properties` through
`project_output_model` — not by this decision's tables.

---

## 2. The roadmap items, one by one

### 2.1 provides/requires (neograph-ftnxl.8) — the hardest probe, and the decision absorbs it cleanly

What it demands: a contract-name indirection so a consumer binds to
`evidence: list[FetchResult]`, not to a producer's name. The commissioning
question: does this invalidate `PeerField(field: str)`?

**No — it lands on top of it, and the decision supplies the mechanism ftnxl.8
was going to have to invent.** The bead's own grounding says the blocker is
that `ProducerMap` is `OrderedDict[field_name, Producer]`
(`_validation_types.py:72`) with "no indirection table … anywhere", and that
phase 1 "genuinely crosses the sugar/IR line" needing "new Node field(s),
single-writer via the normalizer, structural guard". `Node.sources` **is that
field**: the table already decouples slot key from state field (the entry is
slot → `Source`; nothing forces the slot key to equal the producer's field
name). Phase 1 (strict aliasing, duplicate provider = `ConstructError`)
becomes: the normalizer consumes the contract registry, refuses duplicate
providers (authored ambiguity, §3), and stamps
`sources['evidence'] = PeerField(resolved_provider_field)`. `PeerField` does
not change meaning — it remains "a named state field", which is correct,
because assembly is precisely where contract substitution must be resolved.
Phase 2 (per-path providers under branching) is `LastPresent` over per-arm
provider fields with the every-path check as a normalizer `Unresolved` —
legal under A3(a).

Two consequences: (i) **ordering** — ftnxl.8 built before steps 0–2 mints a
fifth parallel field that step 7 then collapses; re-scope its "new IR field"
to target `sources` (§4). (ii) **export loss** — the resolved edge stream
serializes the concrete field, so the contract *name* is a marker/loss-manifest
item, consistent with the Option-1 (lossy, loudly classified) posture already
decided in ftnxl.1.

### 2.2 Carried / Selected / tool ledger (ftnxl.4 shipped, .7, .5, 8v1r0)

Covered by Amendment A5: population, not consumption; owned by the
`_output_classify` monopoly; no variant. The tool-log consumer side
(`{node}_tool_log` per-key state field) and 8v1r0's `emits=` projection both
produce ordinary named output fields read via `PeerField`. ftnxl.16
(Carried rooted at `context=`) stays second-order once `context=` refs are
slots in the fold (A1). No stress on the sum.

### 2.3 Arm-aware wiring (ftnxl.22) and multi-exit branches (qtfof.14)

ftnxl.22's recorded failure mode is exactly the split the decision closes:
"state and export layers already partially implement arm modifiers while
wiring does not, and that split is exactly what produced the silent seam."
The normalizer already walks arms (`_ir_normalize.py` imports
`iter_item_slots`) and already uses `effective_producer_type`, so arm-scoped
stamping with modifier-adjusted types is the existing mechanism, extended.
After the decision, ftnxl.22's lockstep set shrinks: normalizer stamps,
wiring/state implement, export and lint *read* — two consumers drop out of
the must-move-together set. Built before the decision, ftnxl.22 adds more
scan/derivation sites that steps 1–4 then delete. Sequence it after step 3.

qtfof.14 (divergent-arm terminal types): `LastPresent` accommodates it
mechanically — candidates ordered, arm-exclusive presence, legal under
A3(a) — provided the assembly-time type filter runs against the union of arm
types rather than a single `construct.output`. That is a normalizer-rule
tweak, not a shape change; and the stamp makes the current silent behavior
(an incompatible arm simply never supplies the boundary) visible at assembly,
which is an improvement qtfof.14's design pass should lean on. No redesign.

### 2.4 Accumulator channel (neograph-iq4a3, P1) — the real structural stressor

What it demands: a channel declared once, appended by many nodes (including
parallel Each branches), read downstream as the union. Two of the decision's
premises bend here:

1. **Every current variant is one→one.** The read side survives: "read the
   union of channel C" is still a single named binding — consumer slot →
   `Accumulated(channel)`, a named read with reducer semantics, interpreted
   by `read_source` in one new arm. The sum grows by exactly one variant,
   legal under A2 because an accumulator channel is a genuinely new
   state-channel kind (a sanctioned new-IR-capability event on the
   `_BranchNode`/Portal bar — which iq4a3 will be regardless of this
   decision).
2. **The resolver's data model breaks, not the sum.** The normalizer's world
   is "a value enters state only as a node's output":
   `ProducerMap = OrderedDict[field_name, Producer]` — one producer per
   field, last-writer-wins (`_validation_types.py:72`), and
   `_producer_pairs`/`resolve_single_type_source` are built on it. A
   many-writer channel is not a producer in that model. iq4a3 therefore
   requires teaching the *resolver's inputs* (producer enumeration,
   `effective_producer_type`, single-type resolution — must an `Accumulated`
   channel be eligible for a single-type `inputs=list[X]` match, or
   name-only?) about channel declarations. That work lands **inside
   `_ir_normalize.py`/`_ir_source.py`**, which is the decision working as
   designed — but it is resolver-core surgery, not an appended variant, and
   it is the item most likely to force revisiting the decision's internals
   (§5).

The misuse risk A3 exists for: encoding the union read as `LastPresent` over
per-branch fields reintroduces last-write-wins — refused by A3.

### 2.5 Native subgraphs (neograph-5suot, P0)

The decision claims `Construct.input_source` answers 5suot's unknown #5.
Verified against the bead: unknown #5 is literally "The input binding becomes
static… Resolve the producer at compile time instead… the validator already
resolves producers for fan-in, so the information exists" — that is
`resolve_single_type_source` with the sub-construct as consumer, i.e. step 2
verbatim. The claim holds. Native subgraphs then *strengthen* the design:
LangGraph delivers values to named channels itself, so `Port` interpretation
moves from a runtime read to compile-time channel wiring — fewer runtime
reads, same authority. Two watch-items: (i) 5suot condition 4 ("reducers
agree on shared channels") must keep `LoopCarry`'s "own append-list, latest
element" read valid across the parent/child schema split; (ii) **5suot is P0
and must not ship its own binding mechanism** — see ordering (§4).

### 2.6 The rest, briefly

- **neograph-fnlrx / af8ro / a1x7n / lmjn5 / 6ars9 / 5fvsu** — these are the
  decision's own commissioning tickets; steps 1–5 are their fixes. a1x7n item
  1's "unfixable until the sub-construct port has a NAMED source … sequence
  after 5suot unknown #5 / the t1nbp resolver" is exactly step 2. Consistent.
- **qtfof.6/.7** — covered by A1. qtfof.6's ToolNode-synthesis half
  (fabricating a predicate node in the exported Flow) is *lowering*, not
  which-value selection, and stays exporter-side legitimately; only its edge
  wiring should come from the fold.
- **neograph-pt85t / Node.metadata** — closed via the sidecar 3-tuple without
  adding `Node.metadata`. If a metadata field is ever added, it is an
  authored foreign-ish bag; §6's parser-confinement rule extends to it. No
  stress on the tables. 5x43u/kdr1u (closed) touch Swarm/mesh import and the
  Command path, both already funneled (`HandoffChannel`, G1). No stress.
- **neograph-mtwy7/rlapj (dump_spec)** — the Core Invariant recorded on
  mtwy7 ("ONE walker per output format, ONE lowering semantics behind them
  all", loss manifest) is the same instinct as this decision; under A1,
  dump_spec's data-edge phase consumes `iter_resolved_edges` and its
  structural/topology phase keeps walking raw `.nodes` (per its correction
  C1), which the decision explicitly permits ("topology, prompts, tools are
  legitimately walked"). Compatible — but a live coordination point (§4).
- **neograph-r8yh7, 738dj** — measurement-harness and loader-wiring tickets;
  no contact with the resolution machinery beyond what A4's import leg
  already covers.

---

## 3. Homeless questions (roadmap questions with no owner under the decision)

1. **The exported *loss manifest* for resolution facts.** Option-1 export
   (ftnxl.1, shipped) classifies conformance; mtwy7 defines a loss manifest
   for dump_spec. When `iter_resolved_edges` yields a Source the target
   format cannot express (a `LastPresent` with >1 candidate → Agent Spec has
   no "one of these, by presence"; an `Accumulated` → no many-writer
   channel; a contract name → edges only), *who decides refuse-vs-degrade
   per variant per format?* The decision makes exporters pure serializers
   but never says what a serializer does with an unserializable Source.
   Owner needed: a per-format `Source → emit | degrade(marker) | refuse`
   table, living next to the conformance classifier, pinned by the harness.
2. **Import-side reinterpretation policy** (A4): is two-edges-into-one-input
   a refusal, a documented reinterpretation-as-fan-in, or a loss entry?
   Today it is a silent reinterpretation. The decision must assign this to
   the ingestion parser explicitly.
3. **Schema-projection parity across consumers** (ftnxl.15): out of this
   decision's scope, but unowned by any other design doc — the projection
   monopoly has three consumers and one of them (`_agent_spec_types.py`)
   doesn't call it. Recommend a one-line scope note in the decision naming
   `_output_classify` as the owning monopoly so the pattern ("parity by
   calling, never by asserting") is cited from both dimensions.

---

## 4. Ordering constraints

- **Step 2 is on 5suot's (P0) critical path.** If 5suot starts first, its
  static-input-binding work IS step 2's content — do it as step 2 (stamp
  `input_source` in the normalizer), or 5suot ships a second binding
  mechanism the migration then deletes. Cheapest resolution: reorder steps
  1↔2 if 5suot is picked up before fnlrx.
- **ftnxl.8 after steps 0–2** (§2.1), else a fifth legacy field. Its bead
  should be edited now to target `Node.sources`.
- **qtfof.7 and a1x7n after the A1 fold exists**; qtfof.6's edge-wiring half
  likewise (its ToolNode synthesis can proceed independently). Done in the
  other order, each writes an exporter-side edge derivation that steps 1–4
  rewrite — the measured fnlrx shape.
- **ftnxl.22 after step 3** (§2.3).
- **iq4a3 after step 0** so `Accumulated` is minted in `_ir_source.py`; its
  resolver-core surgery (§2.4) wants the tables and G3 extension in place
  first, or it lands as yet another parallel mechanism.
- **~~rlapj/dump_spec is in flight NOW~~ — RETRACTED, unverified.** Checked
  against the repo on the day this was written: `rlapj` is OPEN, not
  in_progress, and untouched since 2026-08-19; `bd list --status=in_progress`
  returns nothing; `src/neograph/_spec_dump.py` was last committed 2026-08-20
  (`58dcf4b`); `git status` is clean and `git worktree list` shows no
  checkout for it. There is no live collision and nothing to coordinate.

  The ORDERING point underneath it still stands on its own: `rlapj`'s
  data-edge phase and step 1's `_dump_sub_construct` fix do touch the same
  module, so whichever starts second should build on the first, and landing
  step 0's harness first is worth doing because it is fnlrx's regression test
  and grades both. That is a sequencing preference, not an urgent conflict.

  Retained rather than deleted, because this was the one claim in the
  document that would have changed what a reader did in the next hour, and it
  was wrong. A document asserting an operational fact is exactly the
  load-bearing misinformation `neograph-avmx4` is about.
- **Step 7 (field collapse) last, after ftnxl.8 and iq4a3 have landed or
  been re-scoped** — collapsing four fields into `sources` while two roadmap
  items are adding entries to it is churn; the collapse is cheap once the
  writers are stable.

Rework if ordered wrongly, concretely: 5suot-first-without-step-2 = a second
binding mechanism (~the whole of step 2 redone); qtfof.7-before-fold = one
exporter edge-derivation written then deleted; ftnxl.8-before-tables = a new
G3 field plus its guard, both retired at step 7.

---

## 5. The one thing most likely to force a redesign within six months

**neograph-iq4a3 (accumulator channel).** P1, GH-filed with a real silent
data-loss incident behind it, and the only roadmap item that breaks a
*premise* of the resolver rather than adding a case to it: one producer per
state field (`ProducerMap`, `_validation_types.py:72`, last-writer-wins) is
load-bearing in `_producer_pairs`, `resolve_single_type_source`,
`declared_output_fields`, and `effective_producer_type` — all of which the
normalizer is built on. The Source sum survives (+`Accumulated`); the
resolver's producer model does not survive unchanged. If iq4a3 is designed
without Amendments A2/A3 in force, the likely failure is an accumulator
encoded through existing variants (a `LastPresent`, or per-branch
`PeerField`s hand-merged) — reintroducing last-write-wins inside the very
mechanism built to kill it.

Named runner-up, beyond six months: ftnxl.11 (`scope_path` lexical
addressing). If it ever graduates from its P4 design gate, every variant
that names a flat state field becomes scope-relative — `(name, scope_path,
iteration)` keys replace the flat bus and the sum's variants all gain a scope
dimension. That is a genuine redesign, but the bead is explicitly a
decision-document-only child, "not committed implementation", so it is a
horizon note, not a six-month risk.

---

## 6. The one fork for the maintainer

**Is `iter_resolved_edges` the total data-edge stream (Amendment A1) from
step 0, or which-value-only with named-declaration edges deferred?**

Recommendation: **total from step 0.** The marginal cost is small — the
named-declaration slots (`Each.over` root, `context=` refs) are
verification-only resolutions the validator already performs, stamped as
trivial `PeerField`s — and it converts three open P1/P2 tickets
(qtfof.6-edges, qtfof.7, a1x7n) from exporter edits into fold slot-kinds
graded by the harness on day one. Deferring it means §10 claim 3 ships with
an unstated exemption list, and a1x7n is already the measured cost of that
exemption.
