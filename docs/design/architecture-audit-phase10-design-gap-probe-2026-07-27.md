# Phase 10 design-gap probe — remove dead `_MARK_REMOTE_AGENT` marker

Bead: `neograph-s7zt3.13`. Master doc: `docs/design/agent-spec-portal-master-architecture-2026-07-27.md`, Build Plan Phase 10 ("remove `_MARK_REMOTE_AGENT`, confirmed dead code").

## Claim being checked

Bead description: "`_MARK_REMOTE_AGENT` (`_agent_spec.py`) confirmed dead code -- zero references anywhere outside its own definition (re-confirmed across two independent review passes this epic)."

## Verdict: the "zero references" claim is FALSE for the test suite. Genuine (small) design gap found and closed below.

### Grep evidence (whole repo, `src/` + `tests/` + `docs/`)

```
src/neograph/_agent_spec.py:74:_MARK_REMOTE_AGENT = "neograph/remote_agent"
tests/test_guards_agent_spec_markers.py:53:    "neograph/remote_agent",
tests/test_guards_agent_spec_markers.py:110:    assert getattr(ags, "_MARK_REMOTE_AGENT", None) == "neograph/remote_agent"
```

`_agent_spec.py` and `loader.py` production code: confirmed truly zero uses of the constant beyond its own definition line (`_agent_spec.py:74`). No `_MARK_REMOTE_AGENT` name anywhere in `loader.py`. `_REMOTE_AGENT_ENDPOINT_ATTRS` (`loader.py:277`) is a distinct, unrelated dict (endpoint-attribute lookup for reconstructing remote-agent `Node` kinds on import) — confirmed by reading `loader.py:277-321`; it never touches `_MARK_REMOTE_AGENT`. So the production-code half of the claim holds.

**But `tests/test_guards_agent_spec_markers.py` hard-references it twice:**

- Line 53: `"neograph/remote_agent"` is a member of `_EXPECTED_MARKER_VALUES`, the set checked by `test_every_marker_wire_value_is_a_module_constant` (asserts every value in this set is bound to *some* module-level constant in `_agent_spec.py` via `vars(ags)` scan).
- Line 110: `test_marker_constants_pin_the_exact_wire_values` does `assert getattr(ags, "_MARK_REMOTE_AGENT", None) == "neograph/remote_agent"` — a direct, named assertion on the constant's existence and value.

This is a real coupling the bead's "zero references outside its own definition" language misses (it was scoped to `_agent_spec.py`/`loader.py` production code, which is literally true, but the phase's actual blast radius is bigger).

### Repro verification (mechanical, not just inspection)

Ran the guard suite at baseline: `uv run --extra dev pytest tests/test_guards_agent_spec_markers.py -q` → **4 passed**.

Built a throwaway copy of `_agent_spec.py` with the `_MARK_REMOTE_AGENT = "neograph/remote_agent"` line mechanically stripped, loaded it via `importlib.util.spec_from_file_location` (bypassing package `__init__` machinery to isolate the single module), and confirmed `hasattr(mod, "_MARK_REMOTE_AGENT")` → `False`. Repro deleted after the check (no files retained).

Given that, if the constant is deleted from `_agent_spec.py` verbatim with no other change:
- `test_marker_constants_pin_the_exact_wire_values` → **FAILS** (`getattr(..., None) == "neograph/remote_agent"` becomes `None == "neograph/remote_agent"`).
- `test_every_marker_wire_value_is_a_module_constant` → **FAILS** (`"neograph/remote_agent"` is in `_EXPECTED_MARKER_VALUES` but no longer in `bound_values = {v for v in vars(ags).values() if ...}`).

So the phase as filed ("remove the dead constant") is a red-to-green cleanup only if the test file is edited in lockstep — the bead text does not say this, and a naive implementer who greps only `src/` (as the bead itself instructs: "zero references anywhere") would delete the constant, run the *full* suite, and get two red guard-test failures they'd have to re-diagnose from scratch. That's not a "genuine open design *question*" in the sense of an ambiguous architectural decision — there's only one correct fix — but it IS a scoping gap: the phase is not "fully and correctly scoped" as filed, because the stated blast radius (production code only) is incomplete.

### Third loose end: prose docs describe a *different, future* `neograph/remote_agent` metadata key

`docs/design/agent-spec-api/04-remote-agents-a2a.md:52,77` and `docs/design/agent-spec-ratification-2026-07-13.md:97` describe a **prospective A2A design**: stamping `metadata["neograph/remote_agent"]` on a scripted/raw `@node` (or on an imported `Node`) to preserve "this was a remote agent call" semantic intent across export/import. This is design prose only — grep confirms no code anywhere implements it (no `metadata["neograph/remote_agent"]` or `metadata.get("neograph/remote_agent")` use-site exists in `src/`). It is unrelated to the `_MARK_REMOTE_AGENT` module constant except for sharing the same string value by coincidence/reuse-of-naming-convention (both chose `"neograph/remote_agent"` as a plausible marker key for the same future concept, but only one of them — the dead constant — was ever wired into the export-side marker registry `_EXPECTED_MARKER_VALUES`/`vars(ags)` scan). Since the A2A design is not implemented, removing the constant does not break it; but an implementer should NOT read "confirmed dead code, zero references" and then also assume the *string value* `"neograph/remote_agent"` is free to reuse for something else without checking these two docs — the value is documented product vocabulary even though the constant binding it is unused.

## The concrete fix (closing the gap, not just flagging it)

Phase 10, fully scoped, is a **three-file** change, not a one-file deletion:

1. `src/neograph/_agent_spec.py:74` — delete `_MARK_REMOTE_AGENT = "neograph/remote_agent"`.
2. `tests/test_guards_agent_spec_markers.py`:
   - Remove `"neograph/remote_agent",` from `_EXPECTED_MARKER_VALUES` (line 53).
   - Remove the `assert getattr(ags, "_MARK_REMOTE_AGENT", ...)` line (line 110) from `test_marker_constants_pin_the_exact_wire_values`.
   - Update the docstring "The four aa5gq-named constants pin their exact wire strings" → "three" (it enumerates `_MARK_MODE`, `_MARK_AGENT_SPEC`, `_MARK_TOOL_SPEC` after the removal).
   - No change needed to the module docstring comment block in `_agent_spec.py` (lines 63-69) beyond removing the one constant line — it already says "every ... marker key", not an enumerated count.
3. No change needed to `docs/design/agent-spec-api/04-remote-agents-a2a.md` or `agent-spec-ratification-2026-07-13.md` — they describe an unimplemented future key, independent of this constant; leave as-is (do not conflate the two `"neograph/remote_agent"` string appearances).

TDD framing for the implementer: this is a rare "delete-first" cleanup, not red→green in the usual sense — there's no new behavior to test-drive. The correct TDD-flavored sequence is: (a) confirm `test_guards_agent_spec_markers.py` currently 4/4 green (baseline), (b) delete the constant, (c) update the two lines + one docstring word in the guard test so it's back to 4/4 green with no `_MARK_REMOTE_AGENT` reference anywhere, (d) full-suite grep for `_MARK_REMOTE_AGENT` and `"neograph/remote_agent"` returns only the two doc files describing the unimplemented A2A feature.

## Answer to the assigned confirmation question

- Grep entire repo for `_MARK_REMOTE_AGENT`: confirmed present only at its definition (`_agent_spec.py:74`) in production code, but **also asserted-on twice in `tests/test_guards_agent_spec_markers.py`** — the bead's "zero references outside its own definition" is true for `src/` only, not true repo-wide.
- Grep for `"neograph/remote_agent"`: same test file (as a literal, twice) plus two unimplemented-feature design docs (unrelated future key, not code).
- Does removing the constant break a test? **Yes — two assertions in `test_guards_agent_spec_markers.py` fail** unless updated in the same change (verified by a throwaway module-load repro, deleted after use).
- Does it break a docstring reference or round-trip fixture? No round-trip fixture references it (checked `tests/` broadly via the repo-wide grep above — only the guard test hits). The `_agent_spec.py` module docstring block does not need editing beyond deleting the one line.

**Net: not a zero-design-question trivial cleanup as filed** — it is still small and mechanical, but the bead undercounts its own blast radius by one file (the guard test) and should be re-scoped as a 3-file change (constant + guard test + verification grep), not a 1-file deletion, before an implementer picks it up.
