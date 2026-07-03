# Empirical research: output_schema vs. the neo_* strip tax, and STREAM_CUSTOM vs. writer-presence

**Date**: 2026-07-03
**Ticket**: neograph-pjqe (Item A: retire hand-stripping; Item B: retire STREAM_CUSTOM flag)
**Method**: research-only agent; minimal StateGraph experiments run via `uv run python` against the repo's locked versions. No repo modifications.
**Environment**: langgraph **1.2.4**, langgraph-checkpoint 4.1.1, langgraph-checkpoint-sqlite 3.1.0, langgraph-prebuilt 1.1.0, Python 3.12.9.
**Constructor**: `StateGraph(state_schema, context_schema=None, *, input_schema=None, output_schema=None)` — keyword-only `input_schema`/`output_schema`; old `input=`/`output=` names deprecated.

---

## R1 — Does `output_schema` filter each surface?

State = `result` (user) + `neo_fp`, `neo_counter` (framework); node writes all three; compiled with `output_schema` = user fields only.

| Surface | Observed output | Filtered? |
|---|---|---|
| `invoke` | `{'result': 'R'}` | **YES** |
| `ainvoke` | `{'result': 'R'}` | **YES** |
| `stream(stream_mode="values")` | `{'result': 'R', 'neo_fp': 'FP123', 'neo_counter': 7}` | **NO** |
| `stream(stream_mode="updates")` | `{'a': {'result': 'R', 'neo_fp': 'FP123', 'neo_counter': 7}}` | **NO** |
| `astream` values / updates | same as sync | **NO** |

Confirmed identically with a Pydantic state model + Pydantic `output_schema` (matching neograph's real `compile_state_model` output), sync and async.

**Input side (critical)**:
- `input_schema` = user-fields-only: caller-seeded `neo_fp` was **silently dropped** — `input_schema` is a hard input filter.
- `output_schema` alone (no `input_schema`): seeded `neo_fp` **survived** into internal state.

**Verdict**: `output_schema` filters `invoke`/`ainvoke` ONLY. Values/updates chunks (sync + async) are NOT filtered. neograph must declare `output_schema` alone and **NEVER declare `input_schema`** — `run()` injects fingerprints through the initial input dict; an `input_schema` would drop them before they reach state.

## R2 — Do excluded channels persist in checkpoints / get_state?

With `output_schema` filtering active (MemorySaver):
- `checkpointer.get_tuple(cfg).checkpoint["channel_values"]` → full keys including `neo_fp`/`neo_counter`. **Persisted.**
- `graph.get_state(cfg).values` → **unfiltered**; `neo_fp` present.
- Two-node resume: downstream node read `state.get("neo_fp")` successfully; post-run `get_state` still carried it.

**Verdict**: **YES on all three, unconditionally.** `output_schema` affects only the invoke/ainvoke return value; checkpoints and `get_state` are untouched. Auto-resume (fingerprint read-back via `get_state`/`channel_values`) is unaffected. Exactly the "keys stay, strip dies" property required.

## R3 — Sub-construct interaction

Two-level experiment mirroring `make_subgraph_fn` (child graph invoked inside a parent node body):
- Child WITHOUT `output_schema`: `child.invoke()` returns raw internals (`neo_subgraph_input`, `neo_child_internal`) — this is why `_strip_internals` wraps the child invoke today.
- Child WITH `output_schema=ChildOut`: `child.invoke()` → `{'child_out': ...}` — **no strip needed**.
- Parent-level `output_schema` is orthogonal (filtering applies only at parent exits; parent updates chunks still raw, consistent with R1).

**Verdict**: the CHILD compile can use `output_schema` to replace both `_strip_internals` calls in `make_subgraph_fn` (`_subconstruct.py:169` sync, `:178` async). Caveat: declare the child `output_schema` to include exactly `sub.output`'s field, which is what `_scan_subgraph_output`'s type-scan wants.

## R4 — `get_stream_writer()` no-op detection (Item B)

`get_stream_writer()` returns `runtime.stream_writer`. Across drivers:

| Driver | is public no-op sentinel? | writer freevars | live stream target |
|---|---|---|---|
| `invoke` / `ainvoke` | False | `[]` | no |
| `stream(values/updates/messages)` | False | `[]` | no |
| `stream(custom)` | False | `['stream']` | **yes** |
| `stream(['values','custom'])` | False | `['stream']` | **yes** |
| `astream(custom)` | False | `['aioloop','stream']` | **yes** |

Two decisive facts:
1. `get_stream_writer()` is **never** the public `langgraph.config._no_op_stream_writer` — even under plain `invoke` it returns a live `Pregel.stream.<locals>.stream_writer` closure. No clean public sentinel exists to identity-compare in 1.2.4.
2. The only discriminator is `'stream' in writer.__code__.co_freevars` — a private implementation detail with no cross-version stability guarantee, strictly more fragile than the explicit `STREAM_CUSTOM` config flag neograph owns.

**Verdict**: **NO clean replacement.** Keep the `STREAM_CUSTOM` flag, with this report as the citation. Item B resolves as "documented: no stable engine seam exists in langgraph 1.2.4; the flag is the correct Layer-2 mechanism" — not a deletion.

## R5 — Managed values as an alternative

`langgraph.managed` (`IsLastStep`, `RemainingSteps`, `ManagedValue` ABC) are ephemeral, computed-per-step channels, explicitly **not persisted** in checkpoints — the opposite of the fingerprint requirement (R2). Managed values solve "hide from output AND don't persist"; neograph needs "hide from output but KEEP persisted." `output_schema` fits; managed values do not.

---

## Feasibility summary

Real strip-site map (grep-confirmed — the code has already consolidated to one strip per batch verb, not the 4+4 the ticket text assumed; **six** sites total):

| Strip site | File:line | Dies via output_schema? |
|---|---|---|
| `run()` batch exit | `runner.py:509` | **YES** — invoke filtered |
| `arun()` batch exit | `runner.py:682` | **YES** — ainvoke filtered |
| `_finalize_by_mode` values arm | `runner.py:443` | **NO** — chunks unfiltered (R1); residue stays, cite this doc |
| `_finalize_by_mode` updates arm | `runner.py:447` | **NO** — same |
| sub-construct sync strip | `_subconstruct.py:169` | **YES** — child compiles with output_schema (R3) |
| sub-construct async strip | `_subconstruct.py:178` | **YES** — same |

**What lands**: declare `output_schema` = user output fields at the main compile (`compiler.py` StateGraph construction) and at the sub-construct child compile; `input_schema` MUST stay unset; delete the four dead strip wraps. `_strip_internals` survives with its caller set shrunk from 6 to 2 (the two stream arms), each commented as a cited engine gap.

**What stays**: the values/updates stream-arm strips (langgraph 1.2.4 does not filter streamed chunks by output_schema) and the `STREAM_CUSTOM` flag (no stable engine seam for writer-presence detection).

**Untouched by design**: `get_state`/checkpoint path — never filtered (R2), so auto-resume's fingerprint read-back keeps working with zero change.

Experiment scripts (session scratchpad, ephemeral): `r1_output_schema.py`, `r2_checkpoint.py`, `r3_subconstruct.py`, `r4_stream_writer.py`, `r4b_writer_delivery.py`, `r4c_noop_identity.py`, `r4d_runtime.py`, `r_pydantic.py`. The observed outputs above are the durable record.
