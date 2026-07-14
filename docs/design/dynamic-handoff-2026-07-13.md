> **RENAMED 2026-07-14 (neograph-1t0zh):** the construct designed here as *Keymaker* shipped as **`Portal`** — a functional name that reads for newcomers and spans both modes (peer routing + dynamic flow). This document is preserved verbatim as the historical design record; its body still uses the original *Keymaker* name. See `website/src/content/docs/concepts/portal.mdx` for the shipped API.

# KEYMAKER solution design — dynamic handoff + runtime flow-definition

Spike deliverable for **neograph-07inf** (epic neograph-t1f7z, design task neograph-5aqk8).
Grounded in `docs/design/dynamic-handoff-research-2026-07-13.md` (the adversarially
consolidated research) and direct source verification. This document is
**decision-complete**: an implementer with no other context can build from it.
Reversible calls made without the maintainer are marked `> JUDGMENT CALL:` and
mirrored in `docs/design/keymaker-decision-log-2026-07-13.md`.

KEYMAKER is one modifier with two modes:

- **Mode (a) — peer routing**: a node picks its successor at runtime from a
  declared peer set. Lowers to `Command(goto=<peer>, update=...)` with
  `add_node(..., destinations=...)` (the adjudicated lowering, research §6).
- **Mode (b) — dynamic flow definition**: a node emits the spec of the next flow;
  neograph validates → compiles → dispatches it. "If it dispatches, it was
  validated."

---

## 1. Scope for the overnight build

> JUDGMENT CALL (D-SCOPE): **Mode (a) ships fully** (IR + validation + lowering +
> budget + checkpoint tests + @node sugar + example + docs). **Mode (b) ships in a
> REDUCED v1** and only after mode (a) is green: emitted **neograph Spec** dicts via
> the existing `load_spec` seam (`loader.py:44`), dispatched inside the node wrapper,
> validated by `Construct(...)` + `compile()` at dispatch time. Explicitly OUT of v1:
> Agent Spec input (`from_agent_spec`, blocked on neograph-01i0g), Tier-2 durable
> recompile-resume (owned by neograph-mrb2y), self-extending flows (`max_depth`),
> `on_invalid="route_to_error"`, and Keymaker-in-spec schema (an emitted spec cannot
> itself contain a Keymaker). Rationale: mode (b)'s deferred parts each depend on
> unbuilt seams; the reduced core still demonstrates the whole thesis (E2's
> rejection path).

Further v1 boundary cuts, each reversible:

> JUDGMENT CALL (D-MESH-LEVEL): peers must be **sibling Nodes at the same construct
> level**. A `Construct` (sub-construct) as a mesh member, or a peer reference across
> a sub-construct boundary (`Command.PARENT`), is a `ConstructError` in v1. We still
> lower to `Command(goto)` (not router+path_map) so the v2 cross-boundary mesh needs
> no re-lowering — this preserves the research's adjudication while cutting v1 scope.

> JUDGMENT CALL (D-LOWERING-DISSENT, **ESCALATED TO MORNING REVIEW**): the independent
> architect review (H3) dissents from Command(goto)+destinations= for v1: with
> D-MESH-LEVEL cutting cross-boundary meshes and reduced mode (b) needing no Command,
> v1 gets neither of the two hard benefits that justified the invasive lowering, while
> paying for a brand-new "wrapper returns Command" capability and losing free
> composition with the Operator postlude (the cause of review finding H1). The
> reviewer recommends router+path_map (Loop's exact pattern, `_wiring.py:544-569`) for
> the v1 same-level mesh. ADJUDICATED KEEP for tonight: the 07inf bead pins
> Command(goto) as the lowering, and it is forward-compatible with the v2
> cross-boundary mesh. Switch cost if the maintainer reverses in the morning: one
> wiring site (`_add_keymaker_mesh`) + the factory wrapper (`make_keymaker_fn`) — the
> mesh detection, validation, budget, and channel machinery are lowering-agnostic and
> survive unchanged.

> JUDGMENT CALL (D-NO-OPERATOR-COMBO): **Keymaker+Operator is ILLEGAL in v1**
> (assembly-time `ConstructError`). The review (H1) proved the "approval gate on a
> hop" cannot be wired via the existing Operator postlude: `_add_operator_check`
> (`_wiring.py:939-961`) appends a static `member → member__operator` edge, and a
> member returning `Command(goto=peer)` does not traverse it — the interrupt is
> bypassed or mis-ordered, and because the mesh lowering consumes all members at the
> entry, the postlude would actually land at mesh EXIT while an Operator on a
> non-entry member would be silently dropped. The ConstructError names the supported
> workaround: place the Operator on the node BEFORE the mesh entry or AFTER the mesh
> exit. v2 path: raise `interrupt()` INSIDE the keymaker wrapper before returning the
> Command (a wrapper-level gate, not a postlude node).

> JUDGMENT CALL (D-MEMBER-MODES): v1 mesh members may be `scripted`, `think`, or
> `raw` nodes. `agent`/`act` members are rejected at assembly (`ConstructError`,
> hint: fast-follow) because they compile to a multi-node ReAct cycle
> (`_agent_cycle.py`) whose terminal hop needs separate plumbing. This narrows the
> "LLM swarm" demo to think-mode agents for now; the handoff mechanics are identical.

> JUDGMENT CALL (D-DICT-OUTPUTS): dict-form (`outputs={...}`) on a Keymaker node is
> rejected in v1 (`ConstructError`). The mesh payload contract (§3) requires one
> payload type; multi-output interaction is deferred.

---

## 2. Surface (all forms, pinned)

### 2.1 The modifier

```python
class Keymaker(Modifier, frozen=True):
    # -- mode (a): peer routing --
    peers: list[str] | None = None      # declared successor names (directed, per-node)
    route: str = "goto"                 # mode (a): name of the routing FIELD on the
                                        # node's output model; mode (b): the literal
                                        # sentinel "decide"
    max_hops: int = 10                  # mesh budget; settable ONLY on the entry member
    on_exhaust: Literal["error", "exit"] = "error"

    # -- mode (b): dynamic flow definition (route="decide") --
    spec_field: str | None = None       # output-model field holding the emitted Spec dict
    input_field: str | None = None      # output-model field holding the dispatch input dict
    output: type[BaseModel] | str | None = None   # REQUIRED in mode (b): the type the
                                                  # dispatched flow must produce
    scripted: dict[str, Callable] | None = None   # building-block registry for the
    conditions: dict[str, Callable] | None = None # emitted flow's compile()
    on_invalid: Literal["raise"] = "raise"        # v1: raise only (kwarg reserved)
```

Mode discrimination in `model_post_init` (mirrors `Loop.model_post_init`,
`modifiers.py:570`): `peers is not None` ⇒ peer mode (`route` must not be
`"decide"`); `route == "decide"` ⇒ dispatch mode (requires `spec_field`,
`input_field`, `output`; forbids `peers`/`max_hops`/`on_exhaust`). Neither/both ⇒
`ConfigurationError`. `max_hops >= 1` validated like `Loop.max_iterations`
(`modifiers.py:576`).

> JUDGMENT CALL (D-ONE-CLASS): one `Keymaker` class with a mode discriminator, not
> two sibling constructs. The bead pins the surface sketch (`Keymaker(peers=[...])` /
> `Keymaker(route='decide')`); validation and lowering fork internally on the mode.

A module-level sentinel is exported:

```python
HANDOFF_END = "__end__"   # route-field value meaning "leave the mesh"
```

### 2.2 The three API surfaces + ForwardConstruct

| Surface | Form | Ships v1 |
|---|---|---|
| Declarative | `Node("billing", mode="scripted", inputs={"handoff": HandoffDecision}, outputs=HandoffDecision, ...) \| Keymaker(peers=["triage"])` | YES |
| Programmatic pipe | identical (`Modifiable.__or__`, `modifiers.py:215` — free once the slot exists) | YES |
| `@node` decorator | `@node(outputs=HandoffDecision, peers=["billing", "technical"], max_hops=6)` — presence of `peers=` builds the Keymaker, mirroring `loop_when=`→`Loop` (`decorators.py:280,661`). Additional sugar kwargs: `route=`, `max_hops=`, `on_exhaust=`. Conflicts with `map_over=`/`loop_when=` rejected at decoration (same pattern as `decorators.py:378`) | YES |
| ForwardConstruct | `self.handoff(...)` builder | **NO — exempt in v1** |

> JUDGMENT CALL (D-FORWARD-EXEMPT): ForwardConstruct is exempt in v1, argued like the
> `di_inputs` decorator-only exemption (AGENTS.md): the tracer expresses dataflow by
> threading proxies through a linear/branching trace, and a runtime mesh has **no
> static dataflow to trace** — every member is simultaneously producer and consumer
> of the mesh channel. A `self.handoff(members=[...], ...)` builder (copying
> `self.interrupt`, `forward.py:1283`) is a fast-follow once mesh semantics are
> proven; nothing in v1's IR blocks it. The three-surface parity rule (which names
> @node / declarative / programmatic) is fully satisfied.

Mode (b) has no decorator sugar in v1 (declare the modifier via pipe); the kwarg
surface (`spec_field`, registries) is too wide to flatten into `@node` kwargs.

### 2.3 Public exports

`Keymaker` and `HANDOFF_END` are added to `neograph/__init__.py` `__all__` beside
Oracle/Each/Loop/Operator (`__init__.py:124-129`). Per the naming policy, the
facade `__all__` is the public contract.

---

## 3. Runtime contract (mode a)

### 3.1 The mesh

The **mesh** is the set of Keymaker-modified nodes at one construct level, closed
under `peers` references. Rules (all assembly-time `ConstructError`s, §5):

1. Every name in any member's `peers` must be a Keymaker-modified sibling Node.
   (A terminal specialist that only exits declares `peers=[]`.)
2. Mesh members must be **contiguous** in `Construct.nodes`; the first member is
   the **entry**. The incoming static edge (`prev → entry`) is the only static
   edge into the mesh; the mesh's single **exit node** (`__handoff_exit_<entry>`)
   is where the linear chain resumes.
3. **One mesh per construct level** in v1.
4. All members declare the **same single-type output** — the *payload model*.
5. `max_hops`/`on_exhaust` settable only on the entry member.

> JUDGMENT CALL (D-SINGLE-MESH): one mesh per construct level. Two disjoint meshes
> at the same level would make the reserved `handoff` input key (§3.3) ambiguous.
> Named meshes are the v2 relaxation.

> JUDGMENT CALL (D-UNIFORM-PAYLOAD): all members share one payload output type.
> Heterogeneous payloads would make the mesh channel type a union and the peer's
> input contract unverifiable; uniform payload keeps the static guarantee crisp
> ("router output type-compatible with every peer's input" collapses to one check).

### 3.2 The routing signal

The `route` kwarg names a **field on the payload model** (default `"goto"`). Its
annotation must be `str` or `Literal[...]`:

- `Literal[...]`: members must be ⊆ `peers ∪ {HANDOFF_END}` — verified at assembly;
  a stray member is a `ConstructError` naming it. This is the typed-swarm story:
  a typo'd target fails at *assembly*, not silently at runtime.
- plain `str`: assembly accepts; the runtime wrapper verifies each hop's target
  `∈ peers ∪ {HANDOFF_END}` and raises `ExecutionError` on a miss — **closing
  LangGraph's silent-drop hole** (`_algo.py:312`, the research's #1 constraint).

### 3.3 The mesh channel and the reserved `handoff` input key

Because any member can be entered from any caller, a peer cannot read a specific
upstream's field (it would read a stale producer when entered from elsewhere).
Instead the wrapper writes each hop's payload to a shared **mesh channel**
`StateKeys.handoff_payload(entry_field)` = `neo_handoff_<entry_field>`, and a
member consumes it via the **reserved inputs key `"handoff"`**:

- `@node` form: a parameter literally named `handoff` (typed as the payload model).
- Declarative/programmatic: `inputs={"handoff": PayloadModel}`.

Resolution mirrors the existing `fan_out_param`/`neo_each_item` IR concession
(AGENTS.md: "the only IR-level concession to the @node layer"): a new
`Node.handoff_param` field tells `_extract_input` (`_input_shape.py:133`; fan_out
handling at `:108`) to read the mesh channel. **Single-writer invariant (review
H2)**: `handoff_param` is written ONLY by the normalizer (`_ir_normalize.py`,
keyed off the presence of the reserved `handoff` inputs key) — exactly the
`fan_out_param` ownership rule (`_construct_builder.py:172-177` documents that
`_ir_normalize.py` is the sole fan_out_param writer, neograph-k7bg). All three
surfaces carry the `handoff` inputs key explicitly, so all three converge in the
normalizer; writing it in any assembly path would re-create the neograph-ts7
parity bug. The
entry member typically has no `handoff` param (its first activation comes from the
outer pipeline via normal upstream inputs) but MAY declare one (re-entry hops).
Each member's normal output field is *also* written (standard `_build_state_update`),
so downstream and the final `run()` result see member outputs as usual.

> JUDGMENT CALL (D-RESERVED-KEY): name-based reserved key `handoff` (not
> type-matching). Explicit, greppable, and parallel to how `fan_out_param` is keyed.
> The name is validated as colliding with nothing (a producer named `handoff` in a
> mesh-bearing construct is a `ConstructError`).

### 3.4 The hop budget

Copied verbatim from Loop's counter mechanism: one shared counter per mesh,
`StateKeys.handoff_hops(entry_field)` = `neo_handoff_hops_<entry_field>`, declared
as a plain `(int, 0)` state field (like `state.py:147`), incremented by every
member's wrapper (read-modify-write, like `_state_write.py:133-139`), checked
**before emitting the goto**:

- `count >= max_hops` and `on_exhaust == "error"` → `ExecutionError.build("handoff
  exceeded max_hops", node=..., ...)` (exact parallel of `_make_loop_router`,
  `_wiring.py:546-556`; same error class as Loop — no new exception type).
- `on_exhaust == "exit"` → route to the exit node (the payload on the bus is the
  last hop's output). Deliberate literal rename vs Loop's `"last"` (review L1): Loop's
  `"last"` means "return the last RESULT"; here the semantics are "leave via the exit
  NODE" — `"exit"` states what actually happens.

Default `max_hops=10` (matches `Loop.max_iterations` default, `modifiers.py:566`).

### 3.5 Runtime contract (mode b, reduced)

The node's declared output (e.g. `EmittedFlow`) carries two fields named by
`spec_field` / `input_field`: a **neograph Spec dict** and the dispatch input dict.
The wrapper, after the node body produces its output:

1. `load_spec(spec_dict)` → `Construct(...)` — **the validation gate** (eager
   `_validate_node_chain` at `construct.py:194`); a bad spec raises `ConstructError`
   here, before anything executes.
2. Verify the built construct's declared output type equals `Keymaker.output`
   (by registry name or class) — `ExecutionError` on mismatch. This is the typed
   dispatch boundary.
3. `compile(sub, scripted=keymaker.scripted, conditions=keymaker.conditions)` —
   the emitted flow can only wire **pre-registered building blocks**; an unknown
   `scripted_fn` fails loud at compile (existing check, `factory.py:63-69`).
   No checkpointer is passed (see §7).
4. `run` the compiled flow with `input_field`'s dict; write the result to a new
   state field `{node_field}_dispatch: <Keymaker.output>` alongside the node's own
   output field. Downstream consumes `inputs={"<node>_dispatch": OutputType}` /
   an `@node` param named `<node>_dispatch` (the existing dict-output param
   resolution pattern, `_resolve_dict_output_param`).

`on_invalid="raise"` (v1): the `ConstructError` propagates wrapped in
`ExecutionError` with the spec's name and the underlying message — the E2 demo
catches it at `run()` level. Types referenced by the emitted spec resolve via the
global `spec_types` registry (`register_type`, `spec_types.py:41`) — a real
authoring constraint, documented in the example.

> JUDGMENT CALL (D-DISPATCH-REGISTRIES): the emitted flow's callables come ONLY
> from `Keymaker(scripted=..., conditions=...)` declared at assembly time. This is
> a deliberate capability boundary — machine-authored flows compose pre-approved
> blocks; they cannot inject new code. It also sidesteps the global-registry
> mutation concern the research flagged.

---

## 4. Lowering

### 4.1 Mode (a) — per-layer

| Layer | File | Change |
|---|---|---|
| Modifier decl | `modifiers.py` | `Keymaker(Modifier)`; `ModifierSet.keymaker` slot; `_SlotRule(Keymaker, "keymaker", "Keymaker", excludes=(each, oracle, loop, operator))` (`modifiers.py:604`); `ModifierCombo.KEYMAKER` only (no Operator combo — D-NO-OPERATOR-COMBO); ONE `_COMBO_MAP` row (`:89`); `ModifierSet.combo` property extended (`:632-644`) |
| Modifier decl (direct-construct path) | `modifiers.py:646-658` | **`ModifierSet.model_post_init` gets explicit keymaker exclusion arms** (review M2): the hard-coded pairwise checks (each/loop, oracle/loop) do NOT read `_SLOT_RULES`, so without new arms a direct `ModifierSet(keymaker=..., loop=...)` would silently pass while the pipe path rejects — itself a parity hazard. Add keymaker×each/oracle/loop/operator arms (or refactor post_init to read `_SLOT_RULES`) |
| State keys | `_state_keys.py` | `StateKeys.handoff_hops(field)`, `StateKeys.handoff_payload(field)` (beside `loop_count`, `:126`) |
| State model | `state.py` | for the mesh entry: hop counter `(int, 0)` + mesh channel `(PayloadModel \| None, None)` fields, both `neo_`-prefixed (excluded from schema fingerprint — no gratuitous invalidation; member output fields carry the fingerprint). PLUS combo-match arms — see the assert_never enumeration below |
| IR | `node.py` | `Node.handoff_param: str \| None = None` (exact `fan_out_param` parallel) |
| IR normalizer | `_ir_normalize.py` | **SOLE writer of `handoff_param`** (review H2): set it when the node's `inputs` dict carries the reserved `handoff` key — the same single-writer ownership as `fan_out_param` (`_construct_builder.py:172-177`, neograph-k7bg). No assembly path (decorator, builder, loader) may write it |
| Validation | `_construct_validation.py` (+ helpers) | mesh rules §3.1, route-field check §3.2, reserved-key rules §3.3; `effective_producer_type` **unchanged** (members produce their declared output; pinned by test) |
| Compiler walk | `compiler.py:243-277` | **the per-node walk loop becomes mesh-aware** (review M1): detect the contiguous mesh at its entry, dispatch `_add_keymaker_mesh` ONCE, skip the remaining members in the walk, and thread `prev_node` from the mesh exit. This is a walk-loop edit, not just a new dispatch arm — without it every non-entry member gets double-added |
| Compiler dispatch | `compiler.py` (~`:549` node match, `:472` subgraph match) | node match: `KEYMAKER` → `_add_keymaker_mesh`. Subgraph match: reject (D-MESH-LEVEL). `assert_never` forces both arms |
| Wiring | `_wiring.py` | `_add_keymaker_mesh(graph, members, ...)`: adds every member with `graph.add_node(name, fn, destinations=tuple(peers)+(exit,))`; static edge `prev → entry`; pass-through exit node; **no static inter-member edges** |
| Factory | `factory.py` | `make_keymaker_fn(node, keymaker, mesh_ctx)`: wraps the standard `make_node_fn` result; after the inner update dict: read routing field from the node's output; hop-budget check (§3.4); target check (§3.2); return `Command(goto=target_or_exit, update={**inner_update, hops: n+1, mesh_channel: payload})`. Sync + async twins via `RunnableLambda(fn, afunc=...)` (the `factory.py:76` dual-path pattern) |
| Runner | `runner.py` | extend the recursion-limit floor (`_ensure_agent_recursion_limit`, `:51`) to add each mesh's `max_hops` — a K-hop mesh consumes K supersteps |
| Public | `__init__.py` | `Keymaker`, `HANDOFF_END` in `__all__` |

Notes:

- **Five exhaustive match sites need a `KEYMAKER` arm** (review M3): `compiler.py:516`
  (node dispatch), `compiler.py:620` (subgraph dispatch), `state.py:202`,
  `state.py:512`, `state.py:569` — plus the non-exhaustive membership checks at
  `state.py:142` and `:211-233`. mypy's `assert_never` makes the five mechanical;
  the membership checks must be grepped by hand (the exhaustiveness is a feature).
- LangGraph 1.2.4's `add_node(..., destinations=...)` gives compile-time target
  validation (research §2b) — `_add_keymaker_mesh` always declares it.
- The mesh members return `Command`, so LangGraph derives control flow from the
  Command; the exit node keeps the downstream chain wiring unchanged (mirrors
  `__loop_exit_<name>`, `_wiring.py:656`).

### 4.2 Mode (b) — per-layer (reduced)

| Layer | Change |
|---|---|
| `modifiers.py` | same class; dispatch mode discriminated (§2.1) |
| Compiler | `KEYMAKER` combo + dispatch-mode ⇒ `_add_keymaker_dispatch`: plain `graph.add_node` + static next edge (linear; **no Command needed** in reduced v1) |
| Factory | `make_keymaker_dispatch_fn(node, keymaker)`: wraps `make_node_fn`; steps §3.5; sync twin uses `graph.invoke`, async twin `ainvoke` on the freshly compiled sub-flow |
| State model | `{node_field}_dispatch: <Keymaker.output>` regular field (fingerprinted — an output-contract change invalidates, correctly) |
| Validation | dispatch-mode kwarg completeness; `output` resolvable (class or registered name); downstream `_dispatch` consumers validated like dict-form output keys |

---

## 5. Validation — what is static, what is runtime (the honest boundary)

Assembly-time (`ConstructError`, all with specific names in the message):

1. Unknown peer name (lists available siblings, mirroring the fan-in "no upstream
   node named X" error, `_validation_inputs.py:166-173`).
2. Peer not Keymaker-modified / mesh not contiguous / two meshes at one level /
   `max_hops` on a non-entry member / mesh member is a Construct / agent-act member /
   dict-form outputs on a member / producer named `handoff` at a mesh level.
3. Payload uniformity: any member whose output type differs from the entry's.
4. Route field: missing on the payload model; annotation not `str`/`Literal`;
   `Literal` member ∉ `peers ∪ {HANDOFF_END}`.
5. `handoff`-keyed input on a non-mesh node, or typed ≠ payload model.
6. Modifier legality: Keymaker×Each/Oracle/Loop/**Operator** all rejected — via
   `_SLOT_RULES` excludes on the pipe path AND explicit `model_post_init` arms on the
   direct-construct path (review M2). Keymaker owns the outgoing edge, exactly like
   Loop owns the back-edge; the Operator postlude's static edge cannot compose with a
   Command-returning member (D-NO-OPERATOR-COMBO). The error names the workaround:
   Operator on the node before mesh entry or after mesh exit.
7. Mode (b): kwarg completeness, `output` resolvability.

Runtime (`ExecutionError`):

- plain-`str` route value not in `peers ∪ {HANDOFF_END}` (§3.2);
- budget exhaustion with `on_exhaust="error"`;
- mode (b): emitted-spec `ConstructError` (wrapped), output-contract mismatch,
  unregistered type/scripted_fn in the emitted spec.

**Documented weakening (must appear in the website docs verbatim in spirit):** the
linear walker's guarantee ("exactly one type-checked producer feeds each consumer")
does not hold inside a mesh. What mode (a) statically guarantees is: every declared
peer exists, every hop's payload type-checks against every peer's `handoff` input,
and the mesh terminates (budget). Which *specific* member produced the payload a
peer sees is runtime information. This generalizes the documented branch cross-arm
limitation (`_construct_validation.py:139-147`) from two arms to a peer closure —
same honesty, stated once in docs and once in the validator's docstring. Producers
registered by mesh members join the frontier as a union (the `iter_with_arms`
analog), so post-mesh consumers of member outputs validate as today.

Lint: no new lint rules in v1 (mesh checks are hard assembly errors, not advisories).

---

## 6. Termination summary

| Knob | Value | Parallel |
|---|---|---|
| `max_hops` | default 10, entry-only | `Loop.max_iterations` (`modifiers.py:566`) |
| counter | `neo_handoff_hops_<entry_field>`, plain `(int, 0)` | `neo_loop_count_<field>` (`_state_keys.py:126`) |
| exhaustion | `on_exhaust="error"` → `ExecutionError`; `"exit"` → exit node | `_make_loop_router` (`_wiring.py:546-556`) |
| engine backstop | recursion floor raised by `max_hops` per mesh | `_ensure_agent_recursion_limit` (`runner.py:51`) |

---

## 7. Checkpointing

- **Mode (a)**: every hop is one superstep → one checkpoint; the hop counter and
  mesh channel persist in `channel_values` like Loop counters. Operator on a mesh
  member is ILLEGAL (D-NO-OPERATOR-COMBO — the postlude's static edge cannot compose
  with a Command-returning member); HITL around a mesh is expressed by an Operator on
  the node before the entry or after the exit, which composes exactly as today. No
  new checkpoint machinery; a sqlite-backed test proves hops are individual
  supersteps (checkpoint-history count), hop-counter persistence across resume, and
  that an interrupted run (Operator after mesh exit) resumes correctly (test
  conventions: real file-backed savers).
- **Fingerprints**: `neo_`-prefixed mesh fields are excluded (existing rule,
  `state.py:410-419`); member output fields fingerprint as today; the fingerprint
  FORMAT is untouched → no upgrade invalidation.
- **Mode (b) v1 durability is documented-opaque**: the dispatched sub-flow runs
  inside the node wrapper with **no checkpointer**; on resume the parent re-executes
  the whole dispatch node. This is stated in the example and docs. Durable Tier-2
  (recompile + `_auto_resume_from_divergence` on the same thread) is neograph-mrb2y's
  deliverable and is out of scope here (compose, don't rebuild — research §5b).

---

## 8. Tests and guards

Per suite (BDD naming; TDD—failing test first for every behavior):

- `tests/modifiers/test_keymaker.py` (new): modifier legality (slot conflicts,
  duplicate, Keymaker+Operator REJECTED on both the pipe and direct-`ModifierSet`
  paths — the M2 parity hazard pinned), mode discrimination, mesh runtime behavior
  (routing, cycle, budget error/exit, silent-drop closure), reserved-key resolution.
- `tests/test_validation.py`-adjacent (new `test_keymaker_validation.py`): every
  §5 assembly error, message content asserted.
- **Three-surface parity** (`test_fanin_validation.py` pattern): the E1 mesh built
  via `@node`, declarative `Node.scripted()`, and programmatic pipe — same IR, same
  runtime result. ForwardConstruct exemption documented in the test docstring.
- `tests/check_fixtures/should_fail/`: `keymaker_unknown_peer.py`,
  `keymaker_literal_stray_target.py`, `keymaker_loop_combo.py`,
  `keymaker_operator_combo.py`, `keymaker_nonuniform_payload.py`,
  `keymaker_dispatch_missing_output.py`.
  `should_pass/`: `keymaker_mesh_minimal.py`, `keymaker_dispatch_minimal.py`.
- Checkpoint: `test_checkpoint_keymaker.py` — sqlite saver; hops-as-supersteps
  (checkpoint-history count), hop-counter persistence across resume, Operator AFTER
  mesh exit interrupt + resume, sync + async.
- Structural guards (guard-first — written failing before the implementation they
  pin): (G1) **Command-construction monopoly** — `Command(` may be constructed only
  in `factory.py` and `runner.py` (ratchets the new capability); (G2) handoff state
  keys built only via `StateKeys.handoff_*` (no inline f-strings; a focused pin on
  top of the existing Layer-A `neo_`-fragment guard); (G3) **`Node.handoff_param`
  written ONLY by `_ir_normalize.py`** — the single-writer invariant, the exact
  sibling of the `fan_out_param` ownership rule (`_construct_builder.py:172-177`,
  neograph-k7bg).
- Existing gates that will bite (expected, handled in-task): mypy `assert_never`
  exhaustiveness at every combo match; `__all__` contract tests; factory-kwargs and
  StateKeys-centralization guards; the website build + `npm test` for the docs page;
  `scripts/gen_api_manifest.py` regeneration for the verifiable-docs plugin.

---

## 9. Examples and docs

- `examples/28_keymaker_swarm.py` — research E1 adapted to this design: params named
  `handoff`, `HANDOFF_END` imported, `max_hops=6` on entry only. Keyless (scripted).
  Pinned by a test (`test_example_keymaker.py`) like example 27.
- `examples/29_keymaker_dynamic_flow.py` — research E2 adapted: `Keymaker(
  route="decide", spec_field="spec", input_field="dispatch_input", output=Summary,
  scripted={...})`; the rejection path is a real second `run()` that catches the
  wrapped `ConstructError`. Keyless. Ships only with mode (b).
- Website: new page `website/src/content/docs/modifiers/keymaker.mdx` (surface,
  both modes, the documented weakening from §5, the durability note from §7);
  cross-link from `runtime/llm-driven.mdx` ("the dispatch is now a construct, not
  app glue"). `npm run build` + `npm test` must pass; API manifest regenerated.
- AGENTS.md: document the sanctioned new-IR exception (the `_BranchNode` precedent
  clause requires it): `Node.handoff_param`, the mesh channel, and the Command
  monopoly.

---

## 10. Task breakdown (beads-ready, ordered)

Each task is one execute-molecule run; write-stages sequential; every task leaves
the tree green (gates: `uv run --extra dev pytest` scoped + full at consolidation,
mypy, ruff).

**T1 — Keymaker IR: modifier, slots, combo, state keys, normalizer, assembly
validation.** INVARIANT: a Keymaker mesh that violates any §5 assembly rule
(including Keymaker+Operator, D-NO-OPERATOR-COMBO) fails at `Construct(...)` with a
ConstructError naming the offender, on BOTH the pipe path (`_SLOT_RULES` excludes)
and the direct-`ModifierSet` path (`model_post_init` arms — review M2);
`effective_producer_type` is untouched; `handoff_param` is written ONLY by
`_ir_normalize.py` (review H2). ANTI-BAND-AID: mesh checks live in the
`_construct_validation.py` walk (a new `_check_keymaker_mesh` helper), NOT inline in
decorators or compiler; do not special-case any single surface. Files:
`modifiers.py` (class, slot, ONE combo, post_init arms), `node.py` (`handoff_param`
field), `_ir_normalize.py` (sole writer), `_state_keys.py`, `state.py` (fields +
the three combo-match arms `:202,512,569` + membership checks `:142,211-233`),
`_construct_validation.py` (+`_validation_*` helpers), `__init__.py`, fixtures
(should_fail ×6 incl. `keymaker_operator_combo.py`, should_pass ×1 — mesh only),
tests (`test_keymaker_validation.py`, modifier-legality half of
`test_keymaker.py`). Compiler arms at `compiler.py:516,620` land here as explicit
`CompileError("Keymaker lowering lands in T2")` with a pinning test that T2
replaces — fail-loud staging, not a silent gap. AC: new tests green; full mypy
green (all FIVE assert_never sites handled — review M3); no existing test broken.

**T2 — Mode (a) core lowering: mesh-aware walk + Command(goto) + destinations +
exit/END.** INVARIANT: an out-of-set route value raises `ExecutionError` — never
LangGraph's silent drop (`_algo.py:312`); the compile WALK (`compiler.py:243-277`)
detects the contiguous mesh, dispatches `_add_keymaker_mesh` ONCE, skips the
remaining members, and threads `prev_node` from the mesh exit (review M1 — a
walk-loop edit, not just a dispatch arm). GUARD-FIRST: write G1 (Command monopoly)
+ G2 (handoff-key centralization) failing first. Files: `factory.py`
(`make_keymaker_fn` — Command(goto, update), target check, sync+async twins),
`_wiring.py` (`_add_keymaker_mesh`, destinations=, exit node, HANDOFF_END mapping),
`compiler.py` (walk edit + replace T1 placeholder arms), guards, runtime routing
half of `test_keymaker.py` (bounded routes; budget lands in T3). AC: E1-shaped
mesh routes end-to-end with a genuine cycle; unknown-target ExecutionError tested;
G1/G2 green; no double-added nodes (walk test).

**T3 — Hop budget + checkpoint semantics.** INVARIANT: the budget is enforced
BEFORE the goto is emitted; every hop is one checkpointed superstep; counters
persist across resume. Files: `factory.py`/`_state_write.py` (counter
read-modify-write in the wrapper), `modifiers.py` (nothing new — max_hops already
declared), `runner.py` (recursion floor + max_hops per mesh), tests
(`test_checkpoint_keymaker.py`: sqlite saver, hops-as-supersteps history count,
counter persistence, Operator-AFTER-exit interrupt + resume, sync + async; budget
error/exit paths in `test_keymaker.py`). AC: budget `error` raises ExecutionError
naming the entry; `exit` leaves via the exit node with the last payload on the bus;
checkpoint suite green.

**T4 — Three-surface parity + @node sugar.** INVARIANT: the same mesh built via
@node, declarative, and programmatic surfaces produces identical IR and identical
runtime results (the neograph-ts7 lesson); `handoff_param` converges in the
normalizer for all three (NO writes in `decorators.py`/`_construct_builder.py` —
review H2; pin with guard G3). Files: `decorators.py`
(`peers=`/`route=`/`max_hops=`/`on_exhaust=` kwargs + conflicts with
`map_over=`/`loop_when=`), G3 guard (sole-writer), parity tests. ForwardConstruct
exemption documented in the test docstring. AC: parity suite green across all
three surfaces; G3 green.

**T5 — Example 28 + website page + manifest.** Files: `examples/28_keymaker_swarm.py`,
`tests/test_example_keymaker.py`, `website/.../modifiers/keymaker.mdx` (including
the §5 documented weakening and the D-NO-OPERATOR-COMBO workaround),
llm-driven.mdx cross-link, api-manifest regen, AGENTS.md new-IR documentation.
AC: example runs keyless; `npm run build` + `npm test` green; keyless example
sweep (01…10) green.

**T6 — Mode (b) reduced: dispatch wrapper + validation + rejection path.**
INVARIANT: an invalid emitted spec raises BEFORE any sub-flow node executes, via
the same `Construct(...)` gate as hand-written pipelines; the dispatched flow can
only reference pre-registered building blocks; the result is typed by
`Keymaker.output`. ANTI-BAND-AID: dispatch goes through `load_spec`/`compile()` —
no bespoke validator, no schema subset. Files: `modifiers.py` (dispatch-mode
validation), `factory.py` (`make_keymaker_dispatch_fn`), `compiler.py`/`_wiring.py`
(linear arm), `state.py` (`_dispatch` field), tests + 2 fixtures. AC: dispatch
happy path + rejection path + output-contract mismatch tested, sync + async.

**T7 — Example 29 + docs update for mode (b).** Ships only if T6 lands. AC as T5.

**T8 — Consolidation: full gates + bead/docs closure.** Full pytest, full mypy,
ruff, website build, keyless examples; decision-log completeness pass; update
neograph-07inf (spike delivered), file the deferred-work beads (ForwardConstruct
builder, agent/act members, cross-subconstruct mesh via Command.PARENT, named
meshes, Keymaker+Operator wrapper-level `interrupt()` (the H1 v2 path),
`on_invalid="route_to_error"`, `max_depth`, Keymaker-in-spec schema, the
D-LOWERING-DISSENT morning decision, Swarm-import re-scope note on
neograph-2ev48/01i0g referencing research §7); `bd sync` + push.

Dependency chain: T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 (T6/T7 skippable on time
pressure without blocking T8).

---

## 11. Swarm-import re-scope (recorded, not executed tonight)

Once mode (a) lands, neograph-2ev48's Swarm decision flips from REJECT to faithful
handoff-mode import: `relationships` (caller, recipient) pairs → per-member directed
`peers=`; `first_agent` → mesh entry; `send_message`-mode stays lossy
(subagent-as-tool, a different primitive). Export: a mesh serializes to a Swarm with
`neograph/swarm_mode` + `max_hops` metadata. Filed as a follow-up bead in T8 —
blocked on agent-member support (D-MEMBER-MODES) since Agent Spec Swarm members are
Agents, not scripted nodes.
