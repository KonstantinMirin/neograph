# IR-item dispatch — the DDD lens

**Date:** 2026-06-04
**Status:** DDD-lens companion to `ir-item-dispatch-2026-06-04.md`. Feeds
`neograph-8cqd`. Read the seed doc first; this challenges and deepens it, it does
not restate it.
**Scope:** read-only architectural analysis. No production code touched.

---

## 0. TL;DR

The seed doc is right that the disease is real and right to reject "methods on the
types" (Option C). But from a DDD lens its framing has one load-bearing error: it
treats this as a *polymorphism-vs-helpers* dispatch problem, when it is actually a
**domain-services-without-a-home** problem with a **bounded-context leak**
(`_BranchNode` flowing upward from the DX context into the compiler core). Those
are two distinct DDD smells stacked on top of each other, and they have different
fixes. Option B is the right *direction* but the seed describes it as "a single
adapter / dispatch table" — which undersells it. What B actually is, named
correctly, is **a single IR Domain Service module with an anti-corruption boundary
around the forward.py sentinel**. I recommend **B'** (below), which is B with the
ACL made explicit and the operation set scoped to genuine domain rules only.

---

## 1. What is the domain, what is infrastructure

The ubiquitous language is real and consistent across the code: **Node**,
**Construct**, **Modifier** (Each/Oracle/Loop/Operator), **producer**, **boundary
port** (`Construct.input`/`output`), **fan-out**, **leaf node**, **sub-construct**.
That vocabulary is stable — it appears identically in `node.py`, `construct.py`,
`modifiers.py`, the validation cluster, and AGENTS.md. A stable, shared ubiquitous
language is the precondition for a domain model, and neograph has it.

The **domain** of neograph is *the meaning of a typed pipeline graph*: what a Node
produces, what an Each does to a producer's type (`dict[str, X]`), what a boundary
port exposes, how fan-in compatibility is decided. This is genuine domain logic —
it is the thing neograph *is*, independent of LangGraph.

The **infrastructure** is *LangGraph compilation mechanics*: `StateGraph`,
`add_conditional_edges`, `Send`, checkpointer wiring, scripted-fn registration,
the `_add_node_to_graph`/`_add_subgraph`/`_add_branch_to_graph` lowering in
`compiler.py:207-215`.

Critically: **the IR is the domain model, not an infrastructure DTO.** The seed
doc and AGENTS.md call the IR "dumb data," which invites the reading that it is a
DTO. It is not. A DTO carries no invariants; `Construct.__init__`
(`construct.py:129-176`) runs `normalize_ir` + `_validate_node_chain` and *refuses
to construct an invalid graph*. It enforces "Each+Loop is illegal,"
"Construct has no nodes," fan-in type compatibility. That is an **aggregate root
guarding its invariants**, which is precisely a rich domain object — not a DTO.
The "dumb data" instinct is about *where the methods live*, not about whether
invariants exist.

So the picture is:

- **Domain layer:** the IR types (Node/Construct/Modifier, value objects + an
  aggregate root) **and** the rules about them (`effective_producer_type`,
  `_declared_output`, fan-in compatibility, `iter_nodes`).
- **Infrastructure layer:** `compiler.py`, `_wiring.py`, `state.py` (the LangGraph
  adapter), the scripted/runtime registries.
- **DX / application layer:** `decorators.py`, `forward.py` — surfaces that *build*
  domain objects from ergonomic inputs.

The problem is that the domain *rules* are currently scattered into the validation
cluster (`_validation_types.py`), a normalization module (`_normalize.py`), and the
aggregate root's own module (`construct.py:iter_nodes`), with no single domain-layer
home — and one of them (`_declared_output`) is locked inside a private cluster so
the DX layer can't reach it.

---

## 2. Anemic-model verdict: NO — and that is the point

DDD's anemic-domain-model anti-pattern is: data bags with all behavior pushed into
"service" classes that manipulate them from outside, so the objects can't defend
their own state. **neograph is not that**, and the proof is in the constructor:

- `Construct.__init__` (`construct.py:139-176`) rejects empty node lists, propagates
  llm_config, normalizes IR fields, and runs full chain validation. The aggregate
  *cannot exist in an invalid state*.
- `ModifierSet.with_modifier` (`modifiers.py:602-671`) and `model_post_init`
  (`modifiers.py:586-600`) make illegal modifier combos *structurally impossible* —
  Each+Loop raises at composition time, not at use time.
- `Node.__init__` validates skip callables; `Oracle.model_post_init` enforces
  "exactly one of merge_prompt/merge_fn."

These are textbook **rich value objects and aggregate invariants**. An anemic model
would let you build `Construct(nodes=[])` and discover the problem in the compiler.
neograph refuses at the boundary.

**The correct DDD classification of the dumb-data rule is: the IR types are a
*published language* shared by two infrastructure adapters (the Python compiler and
the planned TypeScript port over LangGraphJS).** A published language is a
deliberately stable, behavior-light contract that multiple contexts agree on. That
is *exactly* why you don't bolt `effective_producer_type` onto `Node` as a Python
method: the method wouldn't exist in the TS port, so the contract would diverge.
The dumb-data rule is not an anemic-model mistake — it is a published-language
discipline, and it is correct.

What is *wrong* is not the data types. It is that the **domain rules over the
published language have no single domain-service home**, so they leaked into the
validation cluster and got fenced off from the DX layer.

---

## 3. Per-operation classification: domain vs infra

The seed doc lumps all four operations as "IR-item dispatch." From a DDD lens they
are not the same kind of thing, and conflating them is why the fix has felt
elusive.

| Operation | Where it lives now | DDD classification | Belongs in |
|---|---|---|---|
| `iter_nodes(construct)` | `construct.py:57-71` | **Domain** — "the leaf nodes of a pipeline" is a domain concept (a graph traversal over the aggregate). Pure model question. | Domain service (or aggregate query method — see §6). |
| `_declared_output(item)` | `_validation_types.py:102-110` (cluster-private) | **Domain** — "what type does this item declare it produces" is the meaning of Node.outputs vs Construct.output. Has nothing to do with LangGraph. | Domain service. Currently mislocated *and* over-fenced. |
| `effective_producer_type(item)` | `_validation_types.py:76-99` | **Domain** — "an Each modifier turns producer X into `dict[str, X]`" is a core neograph rule (the TS port must replicate it). AGENTS.md already names this "the single source of truth for modifier-aware type effects." | Domain service. Currently correct *home* (validation), wrong *visibility* (should be reusable by the state builder and DX). |
| `primary_output_field(base, outputs)` | `_normalize.py:109-126` | **Mixed, leaning infra** — "which *state field name* holds the primary output" is half domain (primary = first dict key) and half LangGraph state-bus naming (`output_field_name`). The naming convention is a compiler-adapter concern. | Domain computes "primary key"; infra maps key→field name. The current helper fuses both. |

The sharpest consequence: **`effective_producer_type` and `_declared_output` are
domain rules currently imprisoned in the validation package.** `_validation_types.py`
documents itself as a "leaf module of the validation cluster" and `_declared_output`
is not exported (`grep` confirms it is in no `__all__`). forward.py:559-562
re-implements the selector by hand *specifically because it cannot import the
cluster-private domain rule across the layer boundary*. That is the seed doc's
"cross-layer reuse blocked" observation — correct — but the DDD diagnosis is
sharper: a **domain rule is misfiled as a validation-implementation detail**, so a
peer layer that legitimately needs the same domain rule is forced to duplicate it.
The duplication is a *symptom of misplaced domain logic*, not of missing
polymorphism.

Note also `effective_producer_type_for` (`_validation_types.py:113-137`) already
**duck-types `modifier_set` via `getattr(..., "each", ...)` to avoid importing
`modifiers`**. The codebase is *already* writing anti-coupling shims by hand to keep
a domain rule from depending sideways. That is a smell pointing straight at "this
rule wants a neutral domain home where it can depend on the modifier vocabulary
cleanly."

---

## 4. Bounded contexts and the `_BranchNode` leak

There are effectively three contexts:

1. **IR / domain core** — Node, Construct, Modifier, the rules.
2. **DX context** — `forward.py` (`ForwardConstruct`, `_Tracer`, symbolic proxies)
   and `decorators.py`. These *author* domain objects from ergonomic Python.
3. **Compiler / infra context** — `compiler.py`, `_wiring.py`, `state.py`.

`_BranchNode` is defined in `forward.py:350` — **inside the DX context** — because
it is a tracing artifact: the `_Tracer` records `if/else` branches discovered while
symbolically executing `forward()`. But it then flows *upward and sideways*:

- `compiler.py:44` and `:209` import and dispatch on it,
- `_wiring.py:35,542` lower it to `add_conditional_edges`,
- `state.py:16,102` partition the node list on it,
- `_ir_protocols.py` and `construct.py` bend the *aggregate's own type contract*
  (`arbitrary_types_allowed`, the `ConstructItem` Protocol, the `BeforeValidator`)
  around the fact that this DX sentinel is non-Pydantic.

**This is a bounded-context leak.** A DX-context tracing artifact has become a
first-class member of the domain aggregate's `nodes` list, forcing the domain type
(`Construct.nodes`) to widen its contract to accommodate it, and forcing every
infra walker to special-case it. The constraint the seed doc treats as a fixed law
of nature — *"`_BranchNode` is non-Pydantic, so methods won't cover it"* — is
**itself the leak's fingerprint**. The reason it can't be a clean domain member is
that it was born in the wrong context.

The honest DDD move is an **anti-corruption layer (ACL)**: the DX context should
translate its tracing artifact into a *domain-native* branch representation at the
boundary where `forward()` produces a `Construct`, instead of smuggling the raw
sentinel into the aggregate. Whether that means promoting branches to a proper IR
node type (a `BranchNode(BaseModel)` in the domain) or keeping a sentinel but
behind a translator is a real design choice — but recognizing it as a *context
boundary* reframes the whole problem. Today there is no ACL; the sentinel is
trusted everywhere.

This matters for the A/B decision because **the TS port will hit this too.** If
`_BranchNode` is a DX artifact, the TS DX layer will have its own; if it's a domain
node type, both ports share it. The published-language argument that justifies
dumb-data IR *also* argues for `_BranchNode`-as-domain-node, or at least for an
explicit ACL — not for leaving a Python-DX sentinel embedded in the shared IR.

### Is plural-outputs vs singular-output a smell?

No. `Node.outputs` (plural, may fan to N state fields) vs `Construct.output`
(singular boundary port) is a **legitimate aggregate-boundary distinction**, and
AGENTS.md already articulates the reason: a Node can write multiple named producer
fields to the *shared* state bus; a sub-construct is an isolated aggregate with
*one* typed exit port. Different cardinalities because different scopes. This is
correct domain modeling, not drift. The only cost it imposes — that
`_declared_output` must bridge `.outputs`/`.output` — is trivial and belongs in the
domain service. Do not "unify" these fields; that would erase a real distinction.

---

## 5. Reconciling DDD's "push behavior onto the model" with the two constraints

DDD's instinct is *rich models*, and the naive expression is "methods on Node." The
two hard constraints block that:

- **C1: `_BranchNode` is non-Pydantic.** Methods on Node/Construct don't cover it.
- **C2: published-language discipline.** Methods bake Python behavior into a
  contract the TS port must mirror.

DDD has a *standard* answer for "the behavior is domain logic but it can't live on
the entity": the **Domain Service**. A domain service is for domain operations that
(a) don't naturally belong to one entity, or (b) span several. "Compute the
effective producer type of an item, given its modifiers" spans Node + Construct +
_BranchNode + Modifier — it belongs to *none* of them individually. That is the
canonical domain-service signature.

The second standard answer is **double dispatch / Visitor as ACL**: keep the data
types dumb, put the per-type behavior in one visitor whose dispatch table *is* the
single home. The seed doc's "single adapter" is a visitor in disguise; calling it
that clarifies the design.

So the reconciliation is: **the domain rules get exactly one home — a domain
service module — that the data types stay innocent of, and that both the validation
cluster and the DX layer import from.** This satisfies C2 (no methods on the IR,
contract stays portable; the TS port reimplements the *service*, which it has to do
for the compiler anyway) and addresses C1 by making the service responsible for the
`_BranchNode` case in one place (or, better, by the ACL in §4 removing `_BranchNode`
from the dispatch entirely).

The key DDD insight the seed doc misses: **the "single adapter module" of Option B
is not an infrastructure adapter at all — it is a domain service.** Naming it as
infrastructure ("adapter," "dispatch table") is what makes Option B feel like
mechanical dedup rather than the correct domain factoring it actually is.

---

## 6. Recommendation: B' — an IR domain-service module + a `_BranchNode` ACL

Adopt a sharpened Option B. Concretely:

### Shape

1. **One domain-service module** — call it `_ir_ops.py` (neutral low-level layer,
   below validation and below DX, importable by both). It owns *all* IR-item
   domain queries:
   - `iter_nodes(construct)` — moved out of `construct.py` (the aggregate keeps a
     thin `def leaf_nodes(self)` that delegates, if an aggregate-query method reads
     better at call sites; the *rule* lives in the service).
   - `declared_output(item)` — promoted from cluster-private `_declared_output`,
     made public, single home.
   - `effective_producer_type(item)` and `effective_producer_type_for(...)` —
     moved here from `_validation_types.py`. The validation cluster imports them;
     it no longer *owns* them.
   - `primary_output_key(outputs)` — the **domain** half of
     `primary_output_field`. The **infra** half (key → state-field-name via
     `output_field_name`) stays in `_normalize.py`/state-naming, because field
     naming is a compiler-adapter concern (§3).
   - `is_container(item)` / `child_nodes(item)` — the `_is_construct_like` TypeGuard
     content, generalized.

2. **The dispatch lives in exactly one place per operation, inside the service.**
   `isinstance(item, Node)` is *allowed here and nowhere else*. The service is the
   visitor; callers are branch-free and import the verb they need.

3. **`_BranchNode` ACL.** At the `forward()` → `Construct` boundary, translate the
   tracing sentinel into the domain representation the service understands. Minimum
   viable: a `branch_meta(item)` accessor in the service that is the *only* place
   that knows `_BranchNode`'s shape, so compiler/state/wiring ask the service
   instead of `isinstance`-ing the DX type. Stretch goal (separate ticket): promote
   branches to a real domain node type and delete the leak.

4. **One structural guard** replaces the N per-helper guards: `isinstance(_, Node |
   Construct | _BranchNode)` is banned outside `_ir_ops.py` (+ the compiler's final
   lowering switch, which is legitimately the infra dispatch and should be
   allowlisted by name).

### Why B' over A

A (status-quo helpers + one guard) *freezes the misfiling*. It accepts that domain
rules live in the validation cluster, accepts that forward.py must hand-roll
`_declared_output` forever (the guard would have to *allowlist* that duplication or
forbid the DX layer from the operation it legitimately needs), and never addresses
the `_BranchNode` leak. A's "one guard banning isinstance outside allowlisted
homes" still has *multiple* allowlisted homes across multiple modules — that is not
one home, it is a fence around scattered logic. A is a holding action; the seed doc
correctly files it as the fallback.

B' gives the domain rules one home, unblocks DX reuse (forward.py:561 imports
`declared_output` and the hand-rolled selector dies — this is the seed's §7 folded
symptom, now a clean two-line migration), and names the `_BranchNode` problem so it
can be paid down rather than worked around forever.

### Migration sketch (matches the seed's spike gate)

1. Create `_ir_ops.py`. Move `effective_producer_type[_for]` + `_declared_output`
   (→ public `declared_output`) + `iter_nodes` + `primary_output_key`. Re-export
   from their old homes as thin shims initially so the diff is reviewable.
2. **Proof migration 1:** forward.py:559-562 — replace the inline isinstance/getattr
   selector with `declared_output(source_node)` (also kills the
   `getattr(., 'output', None)` default the ticket flags).
3. **Proof migration 2:** the `state.py:100-102` three-way partition — route through
   `is_container` / `iter_nodes` / a service `is_branch` instead of three
   `isinstance` comprehensions.
4. **Proof migration 3:** `_validation_types.py` producer-registration loop and
   `_validation_inputs.py:286` — consume the service verbs.
5. **Decision gate (unchanged from seed §6.4):** if these collapse ≥~40 of the 52
   sites while keeping `_BranchNode`/layering clean → adopt B', delete the shims,
   retire per-op helpers behind the service. If the `_BranchNode` ACL fights back
   harder than expected, fall to A *plus* file the `_BranchNode`-as-domain-node
   ticket regardless — the leak is real independent of the dispatch decision.

### Risks

- **Over-scoping the service.** The temptation is to drag *every* IR query into
  `_ir_ops.py`. Keep it to genuine domain rules (§3 table). The compiler's final
  lowering switch (`compiler.py:207-215`) is *infra dispatch* — it maps domain items
  to LangGraph calls — and should stay in the compiler, allowlisted, not pulled into
  the service. Don't confuse "the meaning of an item" (domain) with "how this
  runtime lowers it" (infra).
- **TS-port parity.** `_ir_ops.py` becomes a module the TS port must mirror. That is
  *fine and intended* — the TS compiler already has to replicate these rules; a
  named service makes the parity surface explicit instead of hiding it inside the
  Python validation cluster. Document it as a published-language sibling.
- **`primary_output_field` split.** Splitting domain-key from infra-field-name is the
  fiddliest part; if it churns call sites more than it helps, leave
  `primary_output_field` whole in `_normalize.py` and only expose `primary_output_key`
  additively. Don't let the cleanest case block the high-value moves.

---

## 7. Where I agree and disagree with the seed doc

**Agree:**
- The disease is real; count-and-dispersion is the right test (seed §1–2).
- Option C (methods on types) is correctly rejected (seed §4).
- The forward.py:556 boundary is a symptom to migrate, not fix standalone (seed §7).
- The spike-with-a-gate process and the ≥40/52 threshold are sound (seed §6).

**Disagree / sharpen:**
- The seed frames this as **polymorphism vs centralized helpers**. From a DDD lens
  it is **misplaced domain logic + a bounded-context leak**. That reframing changes
  what "B" *is*: not an infrastructure adapter, but a **domain service** — and it
  surfaces a problem the seed never names (the `_BranchNode` leak / missing ACL).
- The seed calls the IR "dumb data," flirting with the DTO reading. It is a **rich
  aggregate + value objects published as a contract** — the dumb-*methods* rule is
  published-language discipline, not anemia. Getting this right is what lets you
  confidently move the *rules* without touching the *types*.
- The seed's "one guard, multiple allowlisted homes" (Option A) is not actually one
  home. B' delivers a genuine single home; A only fences scattered logic.
- The seed treats C1 (`_BranchNode` non-Pydantic) as a fixed constraint to design
  around. It is better read as **evidence of the leak** — worth a paydown ticket
  regardless of which dispatch option wins.

**Net recommendation:** pursue **B'** (IR domain-service module + explicit
`_BranchNode` ACL), measured against A via the seed's spike gate. File the
`_BranchNode`-as-domain-node ACL as its own ticket so it isn't held hostage to the
dispatch decision.

---

*Documentation © Constantine Mirin, mirin.pro — CC BY-ND 4.0.*
