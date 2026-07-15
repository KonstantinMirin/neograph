# Portal addressability: the port-taxonomy Core Invariant

Date: 2026-07-15
Motivates: `neograph-nnds9` (Portal: agent/act mesh members), `neograph-do0d9` (Portal: cross-subconstruct mesh via `Command.PARENT`), `neograph-a37vk` (Portal: `ForwardConstruct.self.handoff(members=[...])` builder).
Origin: `docs/design/keymaker-decision-log-2026-07-13.md` (D-MESH-LEVEL, D-MEMBER-MODES, D-FORWARD-EXEMPT, D-LOWERING-DISSENT).

This doc is the Core Invariant the three tasks above implement AGAINST, not
as bespoke, independently-invented mechanisms (anti-band-aid). It writes no
code — the taxonomy and the two mechanisms below are what every future
addressability extension must be checked against.

---

## Core Invariant

Runtime routing operates on **BOUNDARY PORTS**, never region interiors — a
routing target must always resolve to a construct's declared entry
(`Construct.input`'s compiled entry node, an entry-label-map destination, or
a Portal peer name), so an "addressable region" is defined by having
**exactly one such entry AND one reconverging exit**. Modifiers (`Each`,
`Oracle`, `Loop`, `Operator`) are internal wiring — they never qualify as a
port. This is why bare fork/join/interrupt regions stay permanently
non-addressable unless wrapped in a sub-Construct.

neograph already has the port concept: `Construct.input`/`Construct.output`
(singular) is the existing boundary-port pair (`src/neograph/construct.py`,
as of 2026-07-15 at lines 113–123). Everything below extends that concept —
it does not introduce a new one.

**On the lowering substrate**: this doc's two mechanisms name a
*destination*, not how that destination is reached. Whether a routing
target is lowered via `Command(goto=...)+destinations=` or a
router+`path_map` equivalent is orthogonal to this taxonomy — see
"A note on lowering substrate" below (the lowering is settled:
`Command(goto)`, ratified).

---

## The three-class addressability taxonomy

### ATOMIC

A single scripted/think node, one LangGraph node, `goto` already works.

**Shipped example**: any scripted/think node under `Portal(to=[...])` —
`make_portal_fn` (`src/neograph/factory.py`, as of 2026-07-15 around line
103) is the exemplar `Command(goto=...)` lowering with fail-loud
invalid-target handling. This is the "atomic" addressability class already
shipped; the two classes below are the extension frontier.

### PORT-BEARING REGION

Multi-node, but with a single defined entry and a reconverging exit —
addressable **with machinery** (the two mechanisms this doc specifies).

**Worked example (a) — sub-Construct**: compiled via `_add_subgraph`
(`src/neograph/compiler.py`, as of 2026-07-15 around line 438). A single
typed entry (`sub.input`) is already enforced fail-loud at compile time.

**Worked example (b) — agent/act ReAct cycle**: `_add_agent_cycle`
(`src/neograph/_wiring.py`, as of 2026-07-15 around line 989). One IR node
lowers to three parent LangGraph nodes (`{node}__agent` / `{node}__tools` /
`{node}__parse`), with a single entry (`{node}__agent`) and a reconverging
exit via the 3-way conditional router. Today this is **not** Portal
addressable (D-MEMBER-MODES rejected agent/act as mesh members in v1)
precisely because neither mechanism below existed yet — this is exactly
what motivates `neograph-nnds9`.

Both (a) and (b) are candidates the entry-label map + mesh-transparent-exit
mechanisms unlock, motivating `neograph-nnds9` (agent/act mesh members) and
`neograph-a37vk` (`self.handoff` builder needs a stable name to target).

### PORTLESS REGION

A bare `Each` / `Oracle` / `Loop` / `Operator` applied **inline** (not
wrapped as a sub-Construct). No single entry/exit exists structurally — it
is a fan-out barrier or a static post-edge, not a port. This region is
**permanently non-addressable by design**. The prescribed fix is: wrap it
in a sub-Construct, which then becomes PORT-BEARING via mechanism (a) above
— not a bespoke routing rule invented per-modifier.

---

## The two mechanisms

### Mechanism 1: entry-label map

A compile-time artifact stashed on the compiled graph, the same way
`schema_fingerprint`/`node_fingerprints` are stashed (`compiler.py`'s
`CompiledNeograph` construction). Shape: `dict[str, str]`, DX-facing name →
LangGraph entry node name. Computed once per `compile()` level — it never
crosses a sub-Construct boundary. Crossing that boundary is explicitly
`neograph-do0d9`'s `Command.PARENT` mechanism; keep the two mechanisms
textually separate so implementers don't conflate them.

### Mechanism 2: mesh-transparent exit

Extends the existing Portal exit pattern: a region's exit-node's
successor-resolution step additionally checks the entry-keyed mesh channel
(`StateKeys.handoff_payload(field_name)`, `src/neograph/_state_keys.py`, as
of 2026-07-15 around lines 130–139) before falling through to its static
next-node edge — the same channel/state-key convention Portal peer routing
already uses. No new state-key scheme.

**Exit-node identification is a per-region-class detail this taxonomy doc
does not pin.** For a multi-node PORT-BEARING region (agent/act cycle,
sub-Construct), which node counts as the "exit node" — the parse node? the
router's done-arm? the subgraph's opaque-node boundary? — is deferred to
`neograph-nnds9`/`neograph-do0d9` to resolve per region class. This doc
establishes that an exit node must exist and be singular for a region to
qualify as PORT-BEARING; it does not enumerate every region's exit node.

### Constraints on implementers

Any `Command(goto=...)` emission either mechanism requires must land in
`factory.py` or `runner.py` only — guard G1,
`TestCommandConstructionMonopoly` in `tests/test_guards_assembly.py`. This
is a constraint on `neograph-nnds9`/`neograph-do0d9`/`neograph-a37vk`'s
implementations, not something this doc's own text needs to satisfy (it
writes no code).

Any new mesh-assembly validation rule the taxonomy implies (e.g. rejecting
a Portal target that names a PORTLESS node) belongs in
`src/neograph/_validation_portal.py`'s `_check_portal_mesh`, per the
existing single-sited-validation-cluster discipline — not inlined in the
compiler or decorators.

---

## A note on lowering substrate (D-LOWERING-DISSENT)

**D-LOWERING-DISSENT is RATIFIED** (`neograph-7t7tf`, closed 2026-07-15):
`Command(goto=...)+destinations=` is the confirmed lowering. The reviewer's
router+`path_map` alternative was overruled because `path_map` cannot express
runtime-computed or cross-construct targets (Portal mode (b),
`Command.PARENT`), so it cannot carry Portal's identity; `path_map` stays the
right lowering only for build-time-known fan-out (`Each`/`Oracle`). The
historical dissent is recorded in
`docs/design/keymaker-decision-log-2026-07-13.md`.

The entry-label-map and mesh-transparent-exit mechanisms above are
nonetheless **lowering-substrate-agnostic** by design: they name a
*destination*, not how it is reached. That is a durability property — the
taxonomy survives untouched even if the ratified `Command(goto)` substrate is
ever revisited — not a hedge on an open decision. The lowering choice is
settled; the taxonomy simply does not depend on it.

---

## Origin: the decision log

This doc's taxonomy directly motivates and is justified by three entries in
`docs/design/keymaker-decision-log-2026-07-13.md`, cited here verbatim
rather than re-derived:

- **D-MESH-LEVEL**: "Peers = sibling Nodes at same construct level;
  sub-construct member / `Command.PARENT` cross-boundary = `ConstructError`.
  Lower to `Command(goto)` so v2 cross-boundary needs no re-lowering." —
  origin of `neograph-do0d9`.
- **D-MEMBER-MODES**: "v1 members `scripted`/`think`/`raw`; `agent`/`act`
  rejected (multi-node ReAct terminal-hop plumbing)." — origin of
  `neograph-nnds9`.
- **D-FORWARD-EXEMPT**: "ForwardConstruct exempt (no static dataflow for a
  runtime mesh); `self.handoff(...)` builder is fast-follow. Three-surface
  parity satisfied." — origin of `neograph-a37vk`.

Do not re-derive the rationale behind these three entries here; the
decision log is the authoritative record.

---

## Bounding the claim

Ports make invalid-entry **unrepresentable by construction** — the same
class of guarantee Portal's existing peer-target validation already
provides (fail-loud `ExecutionError` on an invalid target, never a silent
bad jump). `max_hops` (already shipped — `Portal.max_hops`/`on_exhaust`,
`src/neograph/factory.py`, as of 2026-07-15 around lines 146–199) is what
handles the residual **nonterminating-but-legally-routed** class.

These are two different guarantees. **Addressability is not termination.**
A region can be perfectly addressable (single entry, single reconverging
exit, every routing target valid) and still loop forever if the mesh
routing logic never converges — that is what `max_hops` bounds, not what
the port taxonomy bounds. This doc must not be read as claiming the port
taxonomy makes every mesh terminate; it only makes every mesh target valid.
