# KEYMAKER overnight build — decision log

Autonomous run, night of 2026-07-13 → 2026-07-14. Epic: neograph-t1f7z (feature bead: neograph-07inf).
Every judgment call made without the maintainer in the loop is recorded here, with rationale,
so the morning review can ratify or reverse each one. Grounding research:
docs/design/dynamic-handoff-research-2026-07-13.md.

Format: `D<n> — <decision>` / why / alternatives rejected / where it landed.

---

## Process decisions

**D1 — Design-first, independent review gate honored.** The 07inf bead mandates the spike
doc + independent architect review BEFORE implementation is greenlit. The overnight run
keeps that gate: design doc drafted (context-inheriting agent), reviewed by an independent
fresh-context architect, corrections adjudicated here, and only then are implementation
tasks filed and executed. Alternative rejected: skipping straight to code to maximize
overnight throughput — violates the bead's explicit gate and the maintainer's
independent-review preference.

**D2 — Execution discipline.** Every implementation task runs through
/dev-practices:execute via a dedicated fresh-context agent (main-agent context stays
clean); write-stages strictly sequential per the standing "execute molecule" rule;
full consolidation gates (mypy + full pytest + ruff, keyless examples) re-run on the
merged tree at the end regardless of per-task gates.

## Design-gate adjudication (independent review returned GO-WITH-CORRECTIONS)

Review artifact: scratchpad design-review.md (findings H1-H3, M1-M4; per-judgment-call verdicts).
The reviewer AGREED with 9 of the design's 10 judgment calls (incl. D-RESERVED-KEY after a
zero-collision grep, D-FORWARD-EXEMPT, D-DISPATCH-REGISTRIES); the one contested area was the
lowering strategy (H3) and its Operator fallout (H1).

**D3 — Command(goto)+destinations= KEPT for v1 (reviewer dissented; recorded for morning).**
The reviewer showed the two stated justifications (Command.PARENT mesh, mode-a/b unification)
are both cut from v1 scope, and recommended router+path_map (Loop-identical, composes with
Operator). Overruled: the maintainer pinned Command(goto) in the 07inf bead title and notes,
and the research consolidation adjudicated Command(goto)+destinations= deliberately
(forward-compat to v2 mesh-across-subconstructs; destinations= closes LangGraph's
silent-drop hole). Reversing the maintainer's written direction overnight is the larger risk.
The real cost the reviewer surfaced (H1) is removed by D4 instead. If the morning review
prefers the cheaper router, the blast radius of switching is one wiring site + the factory
wrapper — acceptable.

**D4 — KEYMAKER + Operator combo CUT from v1 (fixes H1).** `_add_operator_check` appends a
static post-edge (_wiring.py:939-961) which a Command(goto)-returning member bypasses, so
"approval gate on a hop" as designed was false — worse, silently false. v1 makes the combo
ILLEGAL at assembly with a clear ConstructError naming the workaround (Operator on the node
before mesh entry / after exit). v2 path documented: raise interrupt() inside the keymaker
wrapper. Alternative rejected: shipping the silently-dropped gate (production-quality rule).

**D5 — handoff_param written ONLY by the normalizer (accepts H2 fully).** The design had
_construct_builder.py writing it (decorator path only) — the exact ts7 three-surface parity
bug shape, and a violation of the fan_out_param single-writer invariant (neograph-k7bg,
sole writer = _ir_normalize). Re-scoped: _ir_normalize writes handoff_param for all three
surfaces; structural guard pins the single-writer rule.

**D6 — MEDIUMs folded into task text, T2 split (accepts M1-M4).** Compile WALK must be
mesh-aware (skip members; not just a new dispatch arm); ModifierSet.model_post_init needs
explicit keymaker arms (exclusion table is hard-coded); all FIVE assert_never sites get arms
(compiler x2, state.py:202/512/569); T2 split into core-lowering and checkpoint/budget-test
tasks to stay one-molecule-sized.

<!-- Implementation decisions appended below as they are made. -->
