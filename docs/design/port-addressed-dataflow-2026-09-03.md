# Port-addressed data flow

**Status**: proposed. Supersedes the resolution model in
`ir-ownership-2026-08-27.md` and folds in the five amendments from
`ir-ownership-roadmap-validation-2026-08-27.md`.

**Date**: 2026-09-03

---

## 1. The decision

Every value that moves between nodes has an **address**: the producing member
and the output it came from. neograph resolves that address once, in
`_ir_normalize.py`, and stores it on the IR. Every other part of the
system — runtime, both exporters, the importer, validation, lint, and error
messages — reads the stored address. Nothing re-derives it.

Four things follow, and the rest of this document works them out.

1. **Addresses name ports, not members.** A member declaring dict-form
   outputs writes one state field per key, so a member name does not identify
   a value. Agent Spec already addresses ports: a `DataFlowEdge` carries
   `source_output` and `destination_input`. neograph adopts the same model, so
   its own vocabulary stops being less expressive than the format it
   serialises to.
2. **Naming is optional, and it is the answer to ambiguity.** You declare
   inputs and outputs as you do today. When two candidates match, assembly
   refuses and the message tells you to name the port. A name you supply is
   honoured exactly, and a type mismatch on a named pair fails loudly.
3. **Some ambiguity has no name, and refusing it is wrong.** Which mesh member
   handed you a value is a runtime fact. Those cases funnel into a named
   channel written by the mechanism that creates the fact.
4. **The wrong answer stops being writeable.** `StateBus` loses `keys()`, so
   no consumer can enumerate the bag. Address values are constructible only in
   the normalizer.

---

## 2. What goes wrong today

Several parts of the framework each answer "which value is meant" by scanning
the shared state bag and matching types. They disagree, and a disagreement
returns a wrong value rather than raising.

Two failures reached a user, both silent.

A sub-construct declared `output=Case` and `context=["opening"]`. The context
declaration puts a second `Case` into the bag, and the scan matched it:

```
the node computed    : Case(docket='D-1', readings=5, claims=3)
the boundary returned: Case(docket='D-1', readings=5, claims=0)

context=None        -> boundary = SETTLED
context=['opening'] -> boundary = OPENING      no error, exit 0
```

An input declaration re-pointed an unrelated output.

The runtime scanned forward and took the first match. The Agent Spec exporter
scanned in reverse and took the last:

```
runtime : sink consumes a
export  : wires  b -> sink
```

The exported artifact computes a different answer than the pipeline it came
from.

### 2.1 The recurrence

Five people found this independently over five months, and each applied the
same fix — resolve once at assembly, store a field, read the field:

```
fan_out_param        2026-04-06
handoff_param        2026-07-13
handoff_channel      2026-07-14
output_from          2026-08-25
input_source_field   2026-08-27
```

`AGENTS.md` records that pattern as a precedent to copy, never as a rule. A
precedent is optional, and the shortcut stays available: while code can
enumerate the bag, re-deriving locally is easier than finding a stamped field.

---

## 3. Addresses

### 3.1 A port is the unit

A member produces one or more outputs. Each output has a name. Together they
form an address:

```
settle.result        the 'result' output of member 'settle'
settle.tool_log      the 'tool_log' output of the same member
verify               shorthand for the sole output of a single-output member
```

The shorthand exists so that the common case stays short. It expands during
normalisation, so nothing downstream sees a member-shaped address.

### 3.2 Why ports, and not members

Three systems already model outputs per port.

| System | How a value is addressed |
|---|---|
| neograph | Dict-form outputs write one state field per key: `settle_result`, `settle_tool_log` |
| LangGraph | A node returns a dict, and each key updates a separate channel |
| Agent Spec | Outputs are a list of `Property`, and a `DataFlowEdge` carries `source_output` and `destination_input` |

`Construct.output_from` names a member. A member with dict-form outputs writes
two fields, so the name does not say which one crosses the boundary. Assembly
accepts it (`neograph-kgndo`).

The importer discards `source_output` and `destination_input` outright
(`_agent_spec_node_import.py:75-83`), so a port-addressed edge cannot survive a
round trip. Adopting port addresses removes both problems at once.

### 3.3 The address type

```python
@dataclass(frozen=True)
class PortRef:
    member: str            # the producing member's name
    output: str | None     # the output key; None means "the sole output"
```

`output=None` is accepted at declaration and resolved during normalisation. A
member with more than one output and no key is an error that names the
available ports.

---

## 4. Sources

An address says where a value comes from when a member produced it. Not every
value comes from a member. `Source` is the closed set of places a value can
come from, and `PortRef` is one of them.

```python
Peer(ref: PortRef)          # a named member's output
Port()                      # the enclosing construct's input port
EachItem()                  # the fanned-out item
LoopCarry()                 # this member's own append-list, latest element
HandoffChannel(channel)     # a Portal mesh channel
LastPresent(refs)           # ordered candidates; take the last one present
Accumulated(channel)        # reserved for the accumulator channel
```

### 4.1 The growth rule

Variants correspond to **kinds of state channel the runtime physically has** —
peer field, port channel, each-item channel, carry list, mesh channel,
accumulator channel. They never correspond to features. A new variant is
admissible only alongside a sanctioned new-IR-capability event, the bar
`AGENTS.md` already sets for `_BranchNode` and Portal. The interpreter gains
exactly one named-read case per variant.

The known roadmap implies one new variant, `Accumulated`, for the accumulator
channel. Pre-register it so the normalizer mints it and nothing else does.

### 4.2 When `LastPresent` is admissible

`LastPresent` is the variant every hard case reaches for, so it carries
admission criteria. Stamp it only when one of these holds:

- At most one candidate can be present at run time by construction, which
  branch-arm exclusivity guarantees.
- The ordering encodes a named, documented precedence rule, such as
  carry-before-seed or inner-producer-before-outer-port.

It is never the answer to a read that unions several present values; that is
`Accumulated`. It is never an escape from refusing authored ambiguity.

Type filtering happens at assembly, so the runtime asks only whether a field is
present. Presence cannot select the wrong one of two values. This is the move
SSA makes with a `phi` node: name the alternatives instead of searching for
them.

---

## 5. Three kinds of ambiguity

Refusing every ambiguity fails. A measured attempt broke 33 tests, and the
number understates it: refusal makes a correct program unwritable, because a
consumer placed after a Portal mesh has no name it could give. Which member ran
last is a runtime fact.

The fix is to separate three cases.

| Kind | Example | Mechanism |
|---|---|---|
| Authored | Two candidates of one type; two markers on one field | Refuse at assembly, and name the port in the message |
| Emergent | Which mesh member ran; which branch arm executed | Funnel into one named channel, written by the mechanism that creates the fact |
| Unambiguous | One candidate | Resolve once, store a `Source` |

The earlier attempt treated emergent ambiguity as authored, and demanded a name
nobody can supply.

---

## 6. Naming, layered on top

Naming is the disambiguator, not the default. You declare as you do today, and
you reach for a name when neograph tells you it needs one.

### 6.1 The four states

| Declaration | Behaviour |
|---|---|
| Unnamed, one candidate | Resolve once. Nothing changes for you. |
| Unnamed, several candidates | Refuse at assembly. The message lists the candidate ports and shows the named form. |
| Named | Resolve to that port exactly. |
| Named, type mismatch | Fail at assembly, naming the port, its type, and the expected type. |

The third and fourth rows do not hold today. `output_from` is stored and then
ignored: when the named member's type does not match and another member's type
does, assembly accepts the construct and the scan picks the other member
(`neograph-x8i3s`). An escape hatch that silently does nothing is worse than no
escape hatch, because you believe you disambiguated.

### 6.2 One spelling, both directions

The output side has `output_from`. The input side has no field, so
disambiguating an input means rewriting the declaration from `inputs=Claims`
to `inputs={"settle": Claims}`. That asymmetry costs adoption: "add a name" and
"rewrite the declaration" are different asks, and an author reading an error
does the cheaper one.

Both directions take a port address:

```python
Construct("verify", nodes=[...], output=Claim, output_from="settle.result")

Node.scripted("judge", fn="f", inputs=Claims, input_from="settle.result")
```

`input_from` does not exist today. Adding it makes the two directions
symmetrical and keeps the unnamed form as the default.

Dict-form inputs keep working and stay the preferred form when you know the
producers, because `@node` already generates them from parameter names. Use
`input_from` when you hold a bare type and want to disambiguate without
restructuring.

### 6.3 What naming does not replace

You cannot name every producer. Topology changes before definition and at run
time, and a Portal mesh member cannot know which peer hands it a value.
Requiring names everywhere makes correct programs unwritable and damages the
developer experience for the common case, where one candidate exists and no
name is needed.

So naming answers authored ambiguity. It does not replace the resolver.

---

## 7. Who owns resolution, and how everything else reads it

### 7.1 The owner

`_ir_normalize.py`, running inside `Construct.__init__`, owns every answer.
Assembly is the only moment when declaration order, the modifier-adjusted
producer set, and the enclosing boundary all exist together. The runtime holds
the values and has lost the intent. The exporters hold the declarations and
have measurably diverged from the runtime.

### 7.2 The contract

A module that needs to know which value is meant does exactly one of the
following:

- Reads a `Source` and interprets it through `read_source(bus, source)`, the
  single runtime interpreter and the only place a `Source` meets state.
- Serialises `Source` values obtained from `iter_resolved_edges(construct)`.
- Renders a `Resolution` the resolver returned.

No module outside `_ir_normalize.py` applies a type-compatibility test to
**select** among candidates. Verifying that an already-named pair is
type-compatible stays legal everywhere. Checking is not choosing.

### 7.3 The edge stream is total

`iter_resolved_edges` yields every data edge of a construct, not only those in
the resolution tables. A slot is any of: a dict-form input key, a single-type
input, a sub-construct port, a construct boundary, a loop-carry destination, an
`Each.over` collection read, and a `context=` reference.

Named declarations enter the stream as trivially resolved addresses. Without
this, three open tickets each add a second, exporter-side edge enumeration
beside the fold, which is the same disease with a named field instead of a
positional rule. One of them is already that bug.

Both exporters then serialise one stream. A new resolution rule changes
behaviour in both, with no edit to either.

### 7.4 Diagnostics read the same object

The resolver returns `Resolved(source)` or `Unresolved(candidates)`, where each
candidate carries the port address and a near-miss reason the resolver
computed. Error builders render that object.

Being diagnostic-only is not a licence to re-derive. `_suggest_hint` runs its
own type probes today; those probes move into the resolver and run at the
single site. The diagnostic consumer and the authoritative consumer follow the
same pattern, so "diagnostic-only" stops being a category that exempts anything.

### 7.5 Parity by calling, never by asserting

A comment claiming a module "agrees with" or "mirrors" a named rule gets
deleted and replaced by a call to that rule. Three such citations hid a
four-way divergence through at least one review, and a fourth was written by
the fix that was supposed to end it (`neograph-avmx4`).

---

## 8. Import

A foreign Flow is a bag neograph does not own, and `from_agent_spec` is its
ingestion site. Import runs on a hot path: Portal dispatch imports a
runtime-emitted flow on every call.

The baseline loses more than anyone recorded. `_inputs_from_data_edges` reads
only `edge.source_node.name`. It discards `source_output` and
`destination_input`, and it drops any edge whose source is not an already
reconstructed top-level item. Two foreign edges into one input become a
neograph fan-in.

Port addressing fixes the representational half: a `DataFlowEdge` maps to a
`Peer(PortRef(source_node, source_output))` bound to `destination_input`.

Two decisions remain, and this document records them rather than leaving them
to the first person who hits them.

- **Two edges, one input.** Agent Spec resolves this as last-write-wins.
  neograph refuses it, because it is authored ambiguity in the source document.
  The refusal message names both edges.
- **An edge neograph cannot express.** Refuse at import, naming the edge.
  Silent narrowing is what produced the current loss.

Each new refusal reaches the dispatch path, where `on_invalid="route_to_error"`
catches `ConstructError`. Every refusal added here needs a dispatch-mode test.

---

## 9. What makes the wrong answer unwriteable

**Runtime, literally.** Deleting `StateBus.keys()` makes the bag
non-enumerable through the interface. A scan stops being a bad pattern and
becomes code that does not compile against the bus. One caller remains
(`_subconstruct.py:39`), and it converts in step 2.

Two enumeration paths survive `keys()` deletion and get their own steps:
`_scan_subgraph_output` scans a raw invoke-result dict, and
`_extract_loop_reentry` selects by presence-probing with a positional fallback.

**Construction, by ban.** `Source` and `PortRef` values are constructible only
in `_ir_normalize.py`, extending the closed-set construction ban that already
covers five string fields. A consumer cannot fabricate a resolution.

**Export, by containment.** A construct's member list stays walkable, so
nothing prevents a future exporter from deriving its own edges. The substitute
is a differential harness: two constructs differing only in one address must
not export identically. That converts "the exporter did not learn the field"
from a reviewer's memory into a failing test. This is a weaker guarantee than
the runtime side, and it is stated rather than glossed.

---

## 10. Foreign bags

`_is_carried` scans Pydantic `FieldInfo.metadata`, which user code populates
before neograph sees it. neograph cannot close a bag it does not construct, and
every ambiguity there is authored.

So a foreign bag crosses the boundary once, through a parser that refuses
ambiguity and emits owned IR. Two conflicting markers on one field is an error
that names both. Access to `FieldInfo.metadata` stays confined to the parser
modules.

Output-field population — `Carried`, `Selected`, and the proposed projection
features — is a different question and gets no `Source` variant. It belongs to
the output-classification monopoly. Recording that prevents someone filing it
under this decision later.

---

## 11. What changes for you

### 11.1 Behaviour

| Change | Effect |
|---|---|
| Boundaries return the declared value | Adding `context=` no longer re-points an unrelated `output=` |
| Export matches runtime | An exported artifact wires the edges the pipeline takes |
| Wrapper types stop being necessary | You no longer invent a type so that it is unique in a bag |
| Errors name candidates | A refusal lists the candidate ports and the named form that resolves it |
| Some ambiguity refuses at assembly | A pipeline that relied on an accidental scan win fails loudly and tells you the fix |

### 11.2 Surface

`@node`, `Construct`, `compile`, and `run` keep their signatures. Two additions
and one change:

- `input_from="member.output"` is new.
- `output_from` accepts a port address, and rejects a bare member name when
  that member has more than one output.
- `StateBus.keys()` is removed, and four IR fields collapse into the address
  tables. Both are internal, and affect you only if you read neograph's IR
  directly.

This is a 0.x project with one downstream consumer, so the break is the
sanctioned kind. Keeping both representations reproduces the disease being
removed.

---

## 12. What it means for Agent Spec

| Item | Before | After |
|---|---|---|
| `Each` fan-out source edge | Exporter-side derivation | A slot kind in one stream |
| Branch-decision edge | Derived separately | The same stream |
| Port and `context=` edges | Missing | The same stream |
| `output_from` | Ignored by both exporters | Read from the stored address |
| Port fidelity on import | `source_output` discarded | Preserved as a `PortRef` |
| Adding a resolution rule | Edit both exporters | No exporter edit |

Adopting Agent Spec's port model also removes an expressiveness gap: neograph
stops being coarser than the format it serialises to.

---

## 13. Alternatives rejected

| Alternative | Reason |
|---|---|
| Refuse all ambiguity | Measured: 33 failures, and it makes a correct program unwritable after a Portal mesh |
| Require names everywhere | Topology changes at run time, so some producers have no name. It also damages the common case, which needs none |
| Delete the unnamed input form | 575 uses, though 471 sit in tests. It removes the guess for nameable cases and does nothing for emergent ones, so the resolver stays either way |
| A lint or AST guard as the mechanism | Detects the symptom after someone writes it. Such a guard exists, and five instances happened anyway |
| Capability-token state keys | Stops forging a key. Does not stop choosing among keys you hold, which is the defect |
| A separate lowered IR | Buys a property that normalising in `Construct.__init__` already provides, and doubles the object model |
| Thread resolutions through signatures | A missed call site falls back to old behaviour silently |
| Fix the open tickets individually | The measured status quo: five convergent fixes, four false citations, a sixth instance last week |

---

## 14. Migration

Each step lands independently and passes the full gate.

| Step | Work | Closes |
|---|---|---|
| 0 | Add `PortRef`, `Source`, `Resolution`, the construction ban, and the differential-export harness | — |
| 1 | Honour `output_from` in validation; require a port for multi-output members | `x8i3s`, `kgndo` |
| 2 | Stamp the boundary address; both exporters and `dump_spec` read it; delete the false citations | `fnlrx`, `avmx4` |
| 3 | Stamp the sub-construct port; delete `_scan_subgraph_input`; remove `StateBus.keys()` | `5suot` unknown 5 |
| 4 | Add `input_from`; refuse authored input ambiguity with a message naming the ports | `5fvsu` |
| 5 | Resolve the loop self-feedback destination | `af8ro` |
| 6 | Import: map `source_output` to `PortRef`; refuse what neograph cannot express | `a1x7n` |
| 7 | Diagnostics render `Resolution` | `6ars9`, `lmjn5` |
| 8 | Foreign-bag parser refuses conflicting markers | `22jvj` |
| 9 | Collapse the four legacy fields into the address tables | — |

Step 1 comes first among the fixes because it makes naming trustworthy. Every
later refusal tells an author to name a port, and that instruction has to work
before the refusals ship.

### 14.1 Ordering constraints

- `provides`/`requires` lands after step 3, or it mints a fifth legacy field.
- The accumulator channel lands after step 0, and it breaks the
  one-producer-per-field premise rather than adding a case. The closed set
  survives with `Accumulated`; the producer model does not survive unchanged.
- Native subgraphs share step 3's work. Whichever starts second builds on the
  first, and neither ships its own binding mechanism.

---

## 15. The falsifiable claim

**Success, by 2027-02.** A sweep finds type-match selection only in
`_ir_normalize.py`. The bag stays non-enumerable. The next feature that answers
"which value is meant" touches only the normalizer and its types, while
exporter behaviour changes anyway, proven by the harness.

**Failure.** A sixth convergent fix appears outside the normalizer, or a
closed-set ban grows, or a feature needs per-consumer edits. Any of those means
answers still arrive as storage rather than as a contract.
