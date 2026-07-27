# Phase design-gap probe summary (2026-07-27)

Consolidates five independent design-verification passes (Phases 1, 6, 8, 9, 10 of
the Agent-Spec-Portal master architecture), incorporating the adversarial reviews
filed for Phases 6 and 9. No implementation performed; no beads touched.

## Phase 1 (neograph-s7zt3.5) — Portal Construct-as-mesh-entry

**Verdict: NEEDS-MORE-DESIGN (now closed by this probe, not yet folded into the bead).**
The filed two-site fix (`compiler.py:254` + `state.py`'s `portal_members` filter) is
necessary but incomplete. Verified live: (1) the bead's proposed `state.py` fix
literally as worded (`nodes_only + sub_constructs` concatenation) reintroduces the
exact wrong-entry bug via list-reordering — must instead filter `construct.nodes`
directly, preserving order; (2) a genuine third site, `_ir_normalize.py:normalize_ir`'s
own separate `portal_members` collector (sole writer of `Node.handoff_channel`), has
the identical `isinstance(item, Node)` gate and, left unfixed, stamps the wrong
`handoff_channel` on Node peers even after sites 1+2 are patched — reproduced as a
live runtime failure (`handoff` resolving to `None`). All three sites' concrete fixes
are given, verified end-to-end via an in-memory-patched repro (compiled and ran).

## Phase 6 (neograph-s7zt3.2) — PORTAL_OPERATOR HITL-gate on Swarm export

**Verdict: READY.** The design (mesh-exit `AgentNode(Swarm)`→`BranchingNode`→
`InputMessageNode` composite, new `_MARK_PORTAL_OPERATOR_SPEC` dict-valued marker
keyed by member name, generalizing to any number of independently-gated members as
one shared composite, with an explicit round-trip-lossless-but-not-behaviorally-
faithful-to-a-foreign-runtime boundary) is verified constructible (live repro, 1 and
2 gated members, `.to_dict()` round-trip). The adversarial review independently
re-derived every claim from scratch (including a stronger `Flow.from_dict(to_dict())`
full round-trip, not just key inspection) and found no corrections needed — only a
non-blocking polish suggestion (cite `_check_portal_mesh`'s single-connected-component
invariant explicitly in §2). Both source docs agree: implementation-ready as-is.

## Phase 8 (neograph-s7zt3.11) — LOOP/EACH/ORACLE _OPERATOR-on-Construct export

**Verdict: NEEDS-MORE-DESIGN.** The filed premise — that `_lower_each`/`_lower_loop`/
`_lower_oracle` already dispatch polymorphically over Node-vs-Construct "the same way"
the Node path does — is false for 3 of 4: all three hard-crash (`AttributeError`) on a
`Construct` today (verified by direct repro calls). `_lower_operator` is the only
one already Construct-safe (touches only `.name`). LOOP_OPERATOR and EACH_OPERATOR
are closeable with bounded, mechanical isinstance-dispatch fixes (given concretely).
ORACLE_OPERATOR is not mechanical: Node-level Oracle means "N variants via per-variant
LLM model swap"; Construct-level Oracle (per `compiler.py`'s `ORACLE`/`ORACLE_OPERATOR`
arm) means "same compiled sub-graph fanned out N times via redirect/merge," with no
`.model` concept to swap — this needs a genuinely new variant-generation lowering
path (N `FlowNode`s wrapping fresh `to_agent_spec()` calls + reuse only the merge
half), plus an explicit new rule rejecting/ignoring `Oracle.models` for Construct
variants. One remaining unverified sub-question is flagged: whether pyagentspec
permits reusing the same `Flow` object across multiple parent slots without name
collision. This gap also implicates Phase 4 (the dependency this phase inherits it
from) — both tickets need correction before implementation.

## Phase 9 (neograph-s7zt3.12) — Construct-as-Portal-mesh-member + dispatch-mode Portal

**Verdict: READY-WITH-MINOR-GAP.** Both C1 (Construct as Swarm member, export+import)
and C2 (dispatch-mode Portal lowering) are resolved with live-verified evidence:
`Flow` IS-A `AgenticComponent` (so a Construct's exported `Flow` slots directly into
`Swarm.first_agent`/`relationships` unwrapped, no new primitive), and no installed
pyagentspec primitive supports runtime-synthesized subflow reference (ruling out
`BranchingNode`/`FlowNode` for C2, confirmed by source read), so C2 lowers via each
dispatch node's existing per-mode path plus a new `_MARK_PORTAL_DISPATCH_SPEC` marker
for statically-representable fields. Two small, explicitly-flagged (not
structural-unknown) decisions remain open, requiring a maintainer call rather than
further investigation: (1) whether foreign (non-neograph-originated) Swarm import
should fail-loud or coerce when a reconstructed Construct member's output doesn't
structurally match the mesh payload type (round-trip of neograph's own export has
no such gap — `_check_portal_mesh` already guarantees the match pre-export); (2) the
naming/registry convention for how `Portal.scripted`/`.conditions` (callable-valued,
dispatch-mode-only fields) get marker-encoded on export. The adversarial review
independently re-derived every load-bearing claim (including the Swarm+Flow
construction) from a fresh repro and found zero corrections needed.

## Phase 10 (neograph-s7zt3.13) — remove dead `_MARK_REMOTE_AGENT`

**Verdict: NEEDS-MORE-DESIGN (trivially, mechanically closeable).** The bead's
"zero references anywhere outside its own definition" claim is false repo-wide:
`tests/test_guards_agent_spec_markers.py` hard-references the constant twice (as a
member of `_EXPECTED_MARKER_VALUES` and a direct `getattr` assertion). Verified by
repro that deleting the constant alone flips those two guard assertions red. The
correct fix is a 3-step, still-fully-mechanical change (delete constant; remove the
two test references + fix a docstring count from "four" to "three" constants; leave
the two unrelated prospective-A2A-design docs alone, since they describe a different,
unimplemented, coincidentally-same-string-valued future marker). No architectural
ambiguity — just an undercounted blast radius.

---

## Status line

- Phase 1: NEEDS-MORE-DESIGN (now closed above; not yet folded into the bead)
- Phase 6: READY
- Phase 8: NEEDS-MORE-DESIGN (Phase 4 dependency also needs correcting)
- Phase 9: READY-WITH-MINOR-GAP (2 small maintainer decisions, not structural unknowns)
- Phase 10: NEEDS-MORE-DESIGN (mechanical 3-file scope correction, no ambiguity)

## Single most important open question across all five

**Phase 8's ORACLE_OPERATOR-on-Construct lowering is the one real structural gap
left in the group** — it requires genuinely new design (N-copies-of-`FlowNode`
variant generation replacing the per-variant-model-swap semantics `_lower_oracle`
currently hardcodes), not just a mechanical dispatch fix, and it silently
propagates from an equally-unverified Phase 4 premise. Every other phase's open
item (Phase 1's third site, Phase 9's two flagged decisions, Phase 10's blast
radius) is either fully closed by this probe or a small, bounded, non-structural
choice — Phase 8 is the one place where an implementer would need to invent real
new architecture, not just fill in a scoping gap.
