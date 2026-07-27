# Consistency review: agent-spec-portal-master-architecture-2026-07-27.md

Date: 2026-07-27. Lens: consistency with the investigation's own stated principles.

## (a) compiler.py as sole ground truth
Holds throughout. §1 states it explicitly ("wherever another module's answer
disagrees with compiler.py's, the other module is wrong"). §5 applies it
correctly both directions: EACH/ORACLE/LOOP/OPERATOR-on-Construct are marked
"target is correct export" because compiler.py runs them; EACH_ORACLE(_OPERATOR)-
on-Construct are marked "target is fail-loud parity, not support" because
compiler.py itself structurally rejects them (`compiler.py:511-516`, verified —
Each/Oracle fusion needs a single Node's `map_over`/`ensemble_n`, which
Construct lacks). No cell claims Agent Spec should exceed or should be capped
below what compiler.py can run. Verified against source: `compiler.py:253-254`
isinstance(Node) gate, `compiler.py:552-560` PORTAL arm's stale
"already rejected at assembly" comment, `state.py:109-116` (`nodes_only +
sub_constructs` union for Oracle/Each) vs `state.py`'s Portal block (`nodes_only`
only) — all match the document's citations exactly.

## (b) no "not supported" verdict without checking reuse first
Holds. Dispatch-mode Portal export ("NO lowering exists at all") explicitly
defers to checking `tests/agent_spec_capabilities.py`'s primitive registry
before assuming infeasibility (§5, §6 Phase 9) rather than asserting
non-support outright. The one place a hard non-support verdict IS asserted
(EACH_ORACLE/EACH_ORACLE_OPERATOR on Construct, PORTAL_OPERATOR-on-Construct-
member) is explicitly grounded in a verified compiler-level or validator-level
structural rejection, not an unreviewed assumption — the document is careful
to distinguish "permanent, structural" from "duplication gap" in both cases.
No slip back into the old pattern found.

## (c) build-plan/matrix internal consistency
Holds. The three items §1 flags as needing genuinely new design (C1 Construct-
mesh-member lowering, dispatch-mode Portal Flow-node lowering, B1
PORTAL_OPERATOR gate-preservation) are scheduled last/separately (Phase 6 for
B1, Phase 9 for C1/C2/dispatch) rather than bundled into the "pure reuse"
phases. Phase 7 (A4, the 5 fusion combos) is scheduled as reuse/glue work,
matching §5's characterization of those cells as composition of existing
`_lower_each`/`_lower_oracle`/`_lower_operator`/`_lower_loop`, not new design —
no combo is scheduled for a build step the matrix marks as still-needing-design.
Phase ordering (0 additive -> 1 compiler bug -> 2 reference migration -> 3
consumer migration -> 4 Construct-drop fix -> 5 reuse fixes -> 6 B1 -> 7 A4 ->
8 Construct-level fusion -> 9 C1/C2/dispatch -> 10 cleanup) is self-consistent
with dependencies stated elsewhere in the doc (e.g. Phase 8 explicitly waits
on Phase 4 + Phase 7; Phase 1's A3 is called out in §2/§5 as required before
any Agent-Spec Portal-mesh work, and Phase 6/9 correctly come after Phase 1).

## Verdict
No inconsistencies found across all three checks. Source spot-checks
(compiler.py, state.py, _agent_spec.py, _validation_portal.py) confirm the
document's line-level citations are accurate, not re-derived-from-memory
claims.
