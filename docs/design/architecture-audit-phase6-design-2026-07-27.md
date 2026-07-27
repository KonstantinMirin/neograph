# Phase 6 / neograph-s7zt3.2 — PORTAL_OPERATOR HITL-gate preservation on Agent Spec Swarm export: closing design

Date: 2026-07-27
Status: design-verification pass (no implementation). Supplements
`docs/design/agent-spec-portal-master-architecture-2026-07-27.md` §5/§6 Phase 6,
which explicitly flagged this phase as needing real design before an
implementer could pick it up.

**Verdict up front**: the master doc's high-level sketch ("branch on
`member.modifier_set.operator`, attach via a mesh-exit pause composite —
`AgentNode` wrapping the `Swarm` → `BranchingNode`/`InputMessageNode`, reusing
`_lower_operator`'s gate logic") is the *correct* shape, and I verified it
constructs and validates cleanly against the installed `pyagentspec` package
(live repro, not just plausible-looking code — see §1). But the doc left
three things unresolved that an implementer would have had to invent on the
spot: (1) the exact object graph including how the composite integrates with
`to_agent_spec`'s current bare-`Swarm`-return shape, (2) the marker field(s)
and their exact name/shape, and (3) — the real gap — whether the design
generalizes past a single gated member. All three are resolved below with
verified evidence. §4 also surfaces one fidelity fact the master doc did not
state explicitly: neograph's *own* runtime gates each Operator-tagged member
at its own interior turn (a real per-member pause), which the Swarm
mesh-exit composite structurally cannot reproduce — this is an inherent
representability ceiling of Agent Spec's `Swarm` primitive, not a bug, but
the design must say so rather than imply full behavioral fidelity.

---

## 0. What was verified against real source (not assumed)

- `Swarm` (`pyagentspec/swarm.py`) is an `AgenticComponent` with `first_agent`
  + `relationships: List[Tuple[AgenticComponent, AgenticComponent]]`, no
  interior node list, no per-member pause primitive. Confirms the master
  doc's stated premise.
- `AgentNode` (`pyagentspec/flows/nodes/agentnode.py`) declares
  `agent: SerializeAsAny[AgenticComponent]` — **not** `Agent` specifically.
  Since `Swarm(AgenticComponent)`, `AgentNode(agent=swarm_instance)` is
  type-legal, not just "probably works."
  `_get_inferred_inputs`/`_get_inferred_outputs` read `self.agent.inputs`/
  `.outputs`, and `Swarm`/`AgenticComponent` inherit `ComponentWithIO`'s
  default (`getattr(self, "inputs", []) or []`) — so an un-annotated `Swarm`
  infers `inputs=[] outputs=[]` on the `AgentNode`, verified live.
- `BranchingNode` (`pyagentspec/flows/nodes/branchingnode.py`) routes on a
  single string-valued input matched against `mapping`; declares no
  `DataFlowEdge` requirement — `Flow` does not validate that `from_branch`
  matches a node's declared branches, confirmed by reading
  `pyagentspec/flows/edges/controlflowedge.py` (plain
  `Optional[str] = None`, no cross-check) and by the fact that the
  **already-shipped** `_lower_operator` (single-node case, `_agent_spec.py:765`)
  never emits a `DataFlowEdge` into its `check` node either — the `when`
  condition rides purely as `_MARK_OPERATOR_SPEC` metadata for round-trip
  fidelity, not as an executable condition for a foreign engine. This is the
  established, already-shipped pattern (not something Phase 6 invents), and
  the design below mirrors it exactly rather than inventing new executable
  semantics.
- Live repro (`AgentNode(agent=Swarm(...))` → `BranchingNode` →
  `InputMessageNode`, wired into a `Flow` with `StartNode`/`EndNode`s)
  constructed cleanly AND round-tripped through `Flow.to_dict()` — both for
  **one** gated member and for **two members with two different `when`
  strings** (see §3 for the exact multi-member repro and its captured
  output). Deleted after verification per instructions.
- `Operator.when` (`modifiers.py:545`) is declared `when: str` — always a
  plain registered-condition-name string, never a callable. This matters:
  the marker never needs callable-serialization handling (unlike `Loop.when`,
  which `_reconstruct_loop_item` special-cases with `parse_condition`/
  `isinstance(..., str)`).
- `_validation_portal.py:142-152` confirms the assembly-time rule precisely:
  an Operator-gated Portal mesh member must be an atomic Node in
  scripted/think/raw mode — `agent`/`act` mode AND `Construct` members with
  Operator are both rejected at assembly. So **the mesh can legally contain
  any number of independently Operator-gated atomic members** — nothing in
  the validator caps it at one. This is the concrete evidence behind §2's
  answer to "does it generalize."
- `factory.py:307` (`make_portal_approval_fn`) confirms neograph's *own*
  compiled-graph semantics: **each** Operator-gated member gets its own
  `{member}__approve` node spliced onto *that specific member's* own
  `Command(goto)` edge, with `operator.when` evaluated against **that
  member's own turn**, pausing via `interrupt()` **before** the member's
  proposed target is reached. This is a genuine interior, per-member,
  per-condition pause — Agent Spec's `Swarm` has no analog to this at all
  (confirmed above), which is exactly why the master doc calls the
  mesh-exit composite "the correct, faithful approximation, not a cop-out"
  — and exactly why §4 below states the fidelity boundary explicitly.

---

## 1. (a) Exact object construction — verified function body

### Export (`_agent_spec.py`)

Replace the current unconditional bare-`Swarm`-return branch in
`_lower_portal_mesh_to_swarm` with a conditional: if **no** member carries
`.modifier_set.operator`, keep today's behavior (return the bare `Swarm`,
zero behavior change, so the currently-green pure-`PORTAL` cell in §5 of the
master doc is untouched). If **any** member does, wrap the `Swarm` in a
one-shot `Flow` mesh-exit composite:

```python
def _lower_portal_mesh_to_swarm(construct: Construct, members: list[Node], tools_mod: Any) -> Any:
    nodes_mod, flow_mod, edges_mod, property_mod, _tools_mod = _import_agent_spec_flow_classes()

    entry = members[0]
    entry_portal = entry.modifier_set.portal
    assert entry_portal is not None

    agents_by_name: dict[str, Any] = {}
    for member in members:
        rewritten, ref_props, flat_to_original = _translate_placeholders(
            member.prompt or "", _properties_for(member.inputs), member.name
        )
        agent = _make_agent(member, tools_mod, ref_props, [], rewritten)
        agent.metadata = {
            **(agent.metadata or {}),
            _MARK_PROMPT_SPEC: _prompt_spec_marker(member, flat_to_original),
        }
        agents_by_name[member.name] = agent

    relationships = [
        (agents_by_name[member.name], agents_by_name[peer])
        for member in members
        for peer in (member.modifier_set.portal.to or [])  # type: ignore[union-attr]
    ]

    from pyagentspec.swarm import Swarm

    swarm = Swarm(
        name=construct.name,
        first_agent=agents_by_name[entry.name],
        relationships=relationships,
        metadata={
            _MARK_PORTAL_SPEC: {
                "max_hops": entry_portal.max_hops,
                "on_exhaust": entry_portal.on_exhaust,
                "route": entry_portal.route,
            }
        },
    )

    # NEW: per-member Operator gates, collected in member order (deterministic,
    # matches how every other _MARK_*_SPEC marker in this module is built).
    gated: dict[str, str] = {
        member.name: member.modifier_set.operator.when
        for member in members
        if member.modifier_set.operator is not None
    }
    if not gated:
        return swarm  # unchanged today's-behavior path

    # Mesh-exit pause composite: Swarm has no interior per-member pause
    # primitive (verified live, pyagentspec 26.1.2 -- Swarm.relationships is
    # a flat AgenticComponent adjacency list, no Node graph to splice a check
    # into), so the gate is approximated at the point control returns to the
    # enclosing Flow, mirroring _lower_operator's existing BranchingNode +
    # InputMessageNode shape one-for-one, with the Swarm wrapped in an
    # AgentNode (legal: AgentNode.agent: SerializeAsAny[AgenticComponent],
    # and Swarm IS an AgenticComponent -- verified live in this doc's repro).
    agent_node = nodes_mod.AgentNode(name=f"{construct.name}__mesh", agent=swarm)
    check = nodes_mod.BranchingNode(
        name=f"{construct.name}__portal_operator_check",
        mapping={"true": _PAUSE_BRANCH, "false": _DEFAULT_BRANCH},
        metadata={
            _MARK_MODIFIER: "portal_operator",
            _MARK_PORTAL_OPERATOR_SPEC: gated,  # {member_name: when}, ALL gated members
        },
    )
    input_message = nodes_mod.InputMessageNode(
        name=f"{construct.name}__portal_operator_pause",
        outputs=[property_mod.StringProperty(title="user_input")],
    )
    start = nodes_mod.StartNode(name=f"{construct.name}__start")
    end_default = nodes_mod.EndNode(name=f"{construct.name}__end_default")
    end_paused = nodes_mod.EndNode(name=f"{construct.name}__end_paused")

    return flow_mod.Flow(
        name=construct.name,
        start_node=start,
        nodes=[start, agent_node, check, input_message, end_default, end_paused],
        control_flow_connections=[
            edges_mod.ControlFlowEdge(name=f"{construct.name}__start_to_mesh", from_node=start, to_node=agent_node),
            edges_mod.ControlFlowEdge(name=f"{construct.name}__mesh_to_check", from_node=agent_node, to_node=check),
            edges_mod.ControlFlowEdge(
                name=f"{construct.name}__check_to_pause", from_node=check, from_branch=_PAUSE_BRANCH, to_node=input_message
            ),
            edges_mod.ControlFlowEdge(
                name=f"{construct.name}__check_to_default", from_node=check, from_branch=_DEFAULT_BRANCH, to_node=end_default
            ),
            edges_mod.ControlFlowEdge(name=f"{construct.name}__pause_to_end", from_node=input_message, to_node=end_paused),
        ],
    )
```

This is not hypothetical glue: every primitive call above (`AgentNode(agent=...)`,
the `BranchingNode`+`InputMessageNode`+`ControlFlowEdge` shape) is byte-for-byte
what `_lower_operator` (single-node case) already does today, plus the one new
fact verified in §0 — `AgentNode(agent=swarm)` type-checks and infers empty I/O
cleanly. I built and validated this exact shape (`Flow(...)` construction +
`.to_dict()`) in a throwaway repro before writing this section; see §3 for the
multi-member variant's captured output. `to_agent_spec`'s return-type annotation
(`-> Flow`) was already inaccurate before this change (today's bare-`Swarm`-only
case returns a `Swarm`, not a `Flow`) — Phase 6 does not introduce this
inaccuracy, it inherits it; not itself a design gap to close here, worth a
one-line docstring/type note in the same PR.

---

## 2. (c) Does the design generalize to multiple gated members? — resolved, yes, with an explicit fidelity boundary

**This is the part the master doc left open, and it is a real question, not a
paperwork gap**: `_validation_portal.py:142-152` places no cap on how many
atomic mesh members may independently carry `Operator` — a 5-member mesh
could legally have members 2, 3, and 5 each gated with a *different*
`when` condition. The high-level sketch's wording ("attach via a mesh-exit
pause composite") does not by itself say whether that is one composite for
the whole mesh or one per member — and one-per-member is structurally
impossible (there is exactly one exit point from an opaque `Swarm`; there is
no way to attach N distinct `AgentNode`s around N distinct interior turns of
a black-box multi-agent conversation).

**Resolution**: the design generalizes as **one shared mesh-exit composite,
carrying ALL gated members' conditions as a dict-valued marker** —
`_MARK_PORTAL_OPERATOR_SPEC: {member_name: when, ...}` on the single `check`
`BranchingNode`'s metadata (§1's `gated` dict, built by iterating every
member, not just `members[0]` or the first gated one). This was verified
constructible with **two members carrying two different conditions** in a
live repro (§3) — both survived `Flow` construction and `.to_dict()`
round-trip serialization intact.

**The fidelity boundary that must be stated, not silently implied**: per §0's
`make_portal_approval_fn` finding, neograph's *own* compiled LangGraph gates
each Operator-tagged member **individually and precisely** — member 2's gate
fires only on member 2's own turn, evaluated against member 2's own outgoing
`Command(goto)` decision, independent of whether member 3 or 5 ever run in
the same execution. The Agent Spec mesh-exit composite **cannot** reproduce
this as executable behavior for a foreign Agent Spec runtime: there is
exactly one `BranchingNode` downstream of the entire `Swarm`'s black-box
conversation, so a foreign engine has no way to know, from the exported
graph alone, *which* member's turn is what triggered `check`, nor can it
evaluate N different `when` conditions against N different interior moments
it never sees. This is not a defect to fix in Phase 6 — it is the same kind
of approximation the single-member case already makes (§0: even the
single-`Operator` `BranchingNode` carries an unconnected/unevaluated
condition as metadata-only, not real executable branching) — but multiplied
across members it becomes more visible and must be called out so nobody
later assumes the exported `Flow` behaviorally reproduces per-member gating
on a foreign runtime. **What the design DOES guarantee**: lossless
round-trip back into neograph (§ below) — every gated member and its exact
condition string survives export→import, because the marker is read
structurally, not inferred behaviorally. That is the correct, honest scope:
Agent Spec Swarm export of `PORTAL_OPERATOR` is a **fidelity-preserving
serialization**, not a portable multi-agent HITL runtime.

---

## 3. Multi-member live repro (verified, then deleted)

Built and ran (then removed) a throwaway script constructing exactly the §1
shape with **two** independently-gated members (`a2` with
`"lambda d: d.risk > 0.5"`-style ad hoc text and `a3` with a registered
condition name `"requires_human_review"`, to prove the marker is opaque to
`Operator.when`'s content — it never inspects or re-interprets the string,
matching `Operator.when: str`'s plain-string contract from `modifiers.py:545`):

```
MULTI-MEMBER FLOW BUILT OK
to_dict keys: ['component_type', 'id', 'name', 'description', 'metadata', 'inputs']
recovered swarm agents: ['a1', 'a1', 'a2', 'a2', 'a3']
recovered per-member operator marker: {'a2': 'lambda d: d.risk > 0.5', 'a3': 'requires_human_review'}
```

(The `agents` list has duplicates because `_swarm_agents_ordered`-equivalent
naive enumeration over `first_agent` + both sides of each relationship tuple
was used ad hoc in the repro, not deduplicated — irrelevant to the point
being verified, which is that the `Flow` constructs and the marker survives
`to_dict()` round-trip unmodified for 2 differently-valued keys.) This
confirms §2's generalization is real, not asserted.

---

## 4. (b) Exact marker field(s) — precise extension of `_MARK_PORTAL_SPEC`'s neighborhood

Do **not** overload `_MARK_PORTAL_SPEC` itself (that marker lives on the
`Swarm` and carries entry-only knobs — `max_hops`/`on_exhaust`/`route` —
which are orthogonal to per-member Operator gating and unrelated to which
object in the composite carries them). Add a **new**, sibling constant,
following the exact naming convention every other combo marker in
`_agent_spec.py` already uses (`_MARK_OPERATOR_SPEC`, `_MARK_LOOP_SPEC`,
`_MARK_PROMPT_SPEC`):

```python
_MARK_PORTAL_OPERATOR_SPEC = "neograph/portal_operator_spec"
```

Placed on the **new `check` `BranchingNode`'s** `metadata`, alongside
`_MARK_MODIFIER: "portal_operator"` (a **new** `_MARK_MODIFIER` value,
distinct from `"operator"` — the existing `"operator"` value is what
`_group_flow_items` already keys its per-item Loop/Operator lookahead on;
reusing it here would make the mesh-exit check indistinguishable from a
plain single-node Operator check during import, which is exactly the kind
of "never trust a marker without checking the structure it claims to
describe" mistake the master doc's own root-cause section (§1) warns about).

Shape: `dict[str, str]` — `{member_name: operator.when}`, one entry per
Operator-gated member, keyed by the member's own `.name` (unambiguous:
`_check_portal_mesh` already enforces unique names within a mesh). No
callable-serialization concern (`Operator.when` is always `str`, verified
§0), so unlike `_MARK_LOOP_SPEC`'s `when` (which can be a callable and is
therefore NOT put in metadata verbatim in the Loop case per existing
`_lower_loop` handling), the Operator marker is directly JSON-safe.

The `Swarm`'s own `_MARK_PORTAL_SPEC` is untouched — it continues to live on
the `Swarm` object nested inside the `AgentNode`, exactly as today.

---

## 5. (d) Loader-side reconstruction — where it lands and how it reads the marker back

**Current shape**: `from_agent_spec` (`loader.py:737`) special-cases a
top-level `Swarm` at the very top (`if type(flow).__name__ == "Swarm": return
_reconstruct_swarm_mesh(flow)`) *before* the `Flow.nodes` walk. Once Phase 6
lands, a `PORTAL_OPERATOR` mesh export is no longer a bare `Swarm` — it is a
`Flow` wrapping one, so that early dispatch will **not** fire for this case,
and the generic `_group_flow_items` walk (which iterates `flow.nodes`, a
`Swarm` has none) would also not apply directly, since this is a
whole-Flow shape recognition problem, not a per-item one.

**New dispatch, added to `from_agent_spec` immediately after the existing
bare-`Swarm` check**, recognizing the exact composite shape structurally
(never trusting the marker alone — same discipline `_group_flow_items`
already applies to Loop/Operator lookahead at line ~608-633):

```python
def _reconstruct_swarm_mesh_with_operator_gates(flow: Any) -> Construct | None:
    """Recognize the Phase 6 mesh-exit pause composite
    (AgentNode(Swarm) -> BranchingNode['portal_operator'] -> InputMessageNode)
    and reconstruct the underlying Portal mesh with per-member Operator gates
    re-attached. Returns None if the structure does not match -- caller falls
    back to treating `flow` as a plain (possibly foreign) Flow, never trusting
    the marker blindly."""
    agent_nodes = [n for n in flow.nodes if type(n).__name__ == "AgentNode"]
    if len(agent_nodes) != 1 or type(agent_nodes[0].agent).__name__ != "Swarm":
        return None
    agent_node = agent_nodes[0]
    swarm = agent_node.agent

    check = next(
        (n for n in flow.nodes if (n.metadata or {}).get(_MARK_MODIFIER) == "portal_operator"),
        None,
    )
    if check is None or _MARK_PORTAL_OPERATOR_SPEC not in (check.metadata or {}):
        return None

    # Structural confirmation, not marker trust: the AgentNode really leads
    # into this specific check via a real ControlFlowEdge.
    edge_ok = any(
        e.from_node.name == agent_node.name and e.to_node.name == check.name
        for e in flow.control_flow_connections
    )
    if not edge_ok:
        return None

    base = _reconstruct_swarm_mesh(swarm)  # existing helper, unchanged
    gated: dict[str, str] = check.metadata[_MARK_PORTAL_OPERATOR_SPEC]
    updated_nodes = [
        (member | Operator(when=gated[member.name])) if member.name in gated else member
        for member in base.nodes
    ]
    return base.model_copy(update={"nodes": updated_nodes})
```

And in `from_agent_spec`:

```python
if type(flow).__name__ == "Swarm":
    return _reconstruct_swarm_mesh(flow)

reconstructed = _reconstruct_swarm_mesh_with_operator_gates(flow)
if reconstructed is not None:
    return reconstructed
```

placed before the `output_types`/`pipeline_items` walk begins.

**Why re-run `_reconstruct_swarm_mesh` rather than duplicating its body**:
`_reconstruct_swarm_mesh` already handles agent ordering
(`_swarm_agents_ordered`), payload synthesis (`_synthesize_swarm_payload`),
Portal-peer wiring (`member | Portal(to=peers)`), and the Option-F
prompt-marker recovery — all orthogonal to Operator gating. Piping its output
through one more `| Operator(...)` composition per gated member is the same
"compose existing modifiers" pattern `_reconstruct_operator_item`
(loader.py:538) already uses for the plain single-node case (`primary |
Operator(when=...)`), and the same pattern the master doc's §1 executive
summary identifies as the fix for nearly every other gap in this audit
("reuse, not new design"). No `parse_condition` call is needed here (unlike
`_reconstruct_loop_item`'s `when`, which can be a callable) because
`Operator.when` is always `str` (§0) — passing the marker's string straight
through matches `_reconstruct_operator_item`'s existing `Operator(when=operator_spec["when"])`
line exactly.

**Re-validation**: per the master doc's Phase-5 item A5 (loader
re-validation via `_check_portal_mesh`), the reconstructed `Construct` from
this new path must ALSO be routed through that same gate once A5 lands —
this function does not itself need to duplicate that call; it returns a
`Construct` that gets the same downstream re-validation every other
`from_agent_spec` reconstruction path is getting under A5. Not adding a
second, local re-validation call here avoids yet another instance of the
"ad hoc re-derivation instead of one shared checkpoint" anti-pattern the
whole master doc is about.

---

## 6. Test-fixture implication (not new design, just noting the surface)

Two new round-trip fixtures are needed once implemented (neither exists
today, confirmed by grep — `test_agent_spec_matrix.py`'s `SUPPORTED_COMBOS`
does not include `PORTAL_OPERATOR`):
1. Single gated member — export produces the `Flow` composite (not a bare
   `Swarm`), re-import recovers `Operator(when=...)` on exactly that member.
2. **Two or more independently-gated members with different `when` strings**
   — export produces one shared composite with a 2+-entry marker dict,
   re-import recovers each member's own distinct condition. This second
   fixture is the one that would have silently been missing had §2's
   generalization question been left unresolved — an implementer following
   only the master doc's single-member-flavored prose might reasonably have
   built and tested only the N=1 case.

---

## Summary

All three of the task's open items are closed with verified evidence, not
just plausible code:

- **(a)** Exact `AgentNode`/`BranchingNode`/`InputMessageNode`/edge
  construction given in §1, live-repro-verified (construction +
  `.to_dict()` round-trip) both for one and for two gated members.
- **(b)** New marker `_MARK_PORTAL_OPERATOR_SPEC = "neograph/portal_operator_spec"`,
  `dict[str, str]` keyed by member name, on the new `check` node's metadata,
  under a new `_MARK_MODIFIER` value `"portal_operator"` distinct from the
  existing `"operator"` so import lookahead can't conflate the two shapes.
- **(c)** Generalizes to any number of independently-gated members as ONE
  shared mesh-exit composite carrying a multi-entry marker — verified
  live with 2 members and 2 different conditions — but this is explicitly a
  **round-trip-lossless, not behaviorally-faithful-to-a-foreign-runtime**
  guarantee; §2 states why (neograph's own runtime gates each member
  individually and precisely via `make_portal_approval_fn`, which `Swarm`
  cannot represent at all) so the scope boundary is documented, not silently
  overclaimed.
- **(d)** `loader.py` gains a new whole-`Flow` shape-recognition function
  (§5, placed before `_group_flow_items`'s per-item walk, since this is a
  top-level dispatch decision like the existing bare-`Swarm` check, not a
  per-item concern), which structurally confirms the composite (never
  trusting the marker alone, mirroring `_group_flow_items`'s own existing
  discipline) before re-attaching `Operator` per gated member onto
  `_reconstruct_swarm_mesh`'s existing output.

**Is Phase 6 now implementation-ready?** Yes. All the design decisions the
master doc left open are resolved above with source-verified, repro-verified
answers; an implementer could TDD this directly from §1/§4/§5's function
bodies plus §6's two required fixtures, with no further open design
questions.
