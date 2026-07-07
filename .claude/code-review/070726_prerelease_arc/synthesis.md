# Code Review Synthesis — 2026-07-07 (pre-release arc)

**Scope**: cf14801^..HEAD on develop (24 commits, 71 files, +8021/−489 — everything since the 2026-07-06 review: di_inputs, twin remediation, hgpt v1+v2, fan-over-agent complete, run_id, json_mode native, span hygiene, dedup). Whole-codebase audit before 0.6.0.
**Agents**: 5 ran, 5 produced findings.
**Date**: 2026-07-07

## The Headline

The arc's structure held: layering is clean for the second consecutive review,
testing quality is high (real compiled-graph E2Es, provider-faithful fakes),
and — the thing the last review existed to fix — code written *after* the twin
remediation honored extract-then-thin. What survived verification clusters
into three patterns: the error hierarchy grew by ad-hoc parenting (no rule
exists, so a consumer's `except ExecutionError` now silently misses two new
runtime failures), the thinness guard's builder set has a blind spot exactly
one indirection wide (the config-carrier idiom + an un-tabled twin), and the
new `_run_cache` primitive shipped minimal (bounded LRU but no run-end
eviction, no single-flight). All three close at the pattern level with small,
well-scoped work — nothing here questions the release's architecture.

## Patterns (load-bearing)

### PAT-01: New error types are parented ad hoc — no hierarchy rule exists

**Shape**: each ticket that adds an error type picks its parent locally.
`PromptVarMissing` (hjwv) chose `ExecutionError`; `ResourceExpiredError` and
`NonIdempotentReplayError` (a5nh/lhc6) chose bare `NeographError` — all three
are runtime failures raised during node execution. A consumer writing
`except ExecutionError` around `run()` catches the first and misses the other
two. Pre-existing siblings (`StateMissingError`, `CheckpointSchemaError`)
also sit directly under `NeographError` with no documented rationale.

**Evidence (illustrations, not the work queue)**:
- `errors.py:143` — `PromptVarMissing(ExecutionError)` [runtime, correctly parented]
- `errors.py:224` — `NonIdempotentReplayError(NeographError)` [runtime, mis-parented — VERIFIED live: `issubclass(..., ExecutionError) == False`]
- `errors.py:259` — `ResourceExpiredError(NeographError)` [runtime, mis-parented — VERIFIED]
- `di.py` — legacy raw `_ExecutionError(...)` message style alongside new `.build(hint=)` style [CON-02, same root: no rule]

**What closes it at the pattern level**: write the rule into `errors.py`'s
module docstring ("failures raised during graph execution subclass
`ExecutionError`; assembly-time → `ConstructError`; config/setup →
`ConfigurationError`"), re-parent the two new types, migrate di.py's legacy
raises to `.build`, decide-and-document the two pre-existing direct children,
and add a guard test pinning parentage-by-category so the next new error
cannot re-drift.

**Convergence**: consistency (High CON-01 + Medium CON-02); 4+ sites.

### PAT-02: The thinness guard's builder set is one indirection narrower than its purpose

**Shape**: the ykun guard checks twins for duplicated *value-builder blocks*
(error builders, log events, usage dicts) — but the config-carrier idiom
(`{**config, "configurable": {**configurable, K: v}}`) is not in its builder
set, and twins not registered in `TWIN_TABLE` are invisible to it entirely.
The arc re-inlined the carrier at 4 sites (verified), including the
`_inject_di_inputs`/`_ainject_di_inputs` pair, which is un-tabled. Classic
Pattern-B: the guard lands, the semantically-equivalent shape slips around it.

**Evidence**:
- 4 × `"configurable": {**configurable, ...}` sites across `_dispatch.py`/`runner.py` (grep-verified count: 4)
- `_inject_di_inputs`/`_ainject_di_inputs` — twin pair absent from `TWIN_TABLE`
- Low siblings: RUN_ID reader duplicated ×2; `_fan_agent` fan-label re-derivation

**What closes it at the pattern level**: extract one `_with_configurable(config, **kv)`
helper (kills the 4 inlines), table the injector twin pair, and add the
carrier shape to the guard's builder set so the *next* re-inline fails at
authoring time.

**Convergence**: dry (Medium + 2 Low); the same guard-scope lesson as the
2026-07-06 review's PAT-01, one level up.

### PAT-03: The run-cache primitive shipped minimal — correct but unhardened

**Shape**: `_run_cache` is sound for its two consumers today (bounded LRU,
lock-guarded map, run_id-keyed) but lacks the two properties a shared cache
primitive needs before more consumers arrive: run-end eviction (loop-bound
MCP/LLM handles linger until LRU pressure, holding event-loop-affine objects
past their run) and per-key single-flight (concurrent fan-out branches
double-build handles / double-fetch resources — waste, and for non-idempotent
providers a correctness edge).

**Evidence**:
- `_run_cache.py:66-71` — `_store` evicts only by LRU bound; no run-completion hook
- `_run_cache.py:74-105` — `get_or_build`/`aget_or_build`: lock guards the map, not the build; two concurrent misses on one key both build
- Un-pinned interaction properties already noted in yc38 (parallel-arun isolation; expiry-vs-cache) belong to this same hardening

**What closes it at the pattern level**: per-key single-flight (sync
`Lock`-per-key / async `asyncio.Lock`-per-key, minted under the map lock),
eviction hook in the runner verbs' `finally` (the same finalize seam observe=
uses), and the two concurrency pins as tests.

**Convergence**: python-practices (2 Medium); overlaps the yc38-noted unpinned
properties from Wave-8 verification.

## Cross-Agent Convergence

| Shape / site | dry | consistency | layering | python | testing |
|---|---|---|---|---|---|
| Error-raise discipline (parentage + style) | — | CON-01, CON-02 | — | — | — |
| Config-carrier idiom + un-tabled twin | DRY-Med | — | — | — | — |
| _run_cache lifecycle | — | — | (noted sound-by-uuid) | PP-01, PP-02 | — |

No multi-agent site convergence this round — the three patterns are each
single-dimension, consistent with a structurally-clean arc where remaining
issues are specialty concerns.

## Findings Catalog (illustrations)

### Critical (verified)
None.

### High (verified)

#### HIGH-01: Runtime errors split parents — `except ExecutionError` misses two new failures [PAT-01]
- **Source**: consistency CON-01. **Files**: `errors.py:224,259` vs `:143`.
- **Verification**: live `issubclass` check — `ResourceExpiredError`/`NonIdempotentReplayError` → False, `PromptVarMissing` → True.
- **Impact**: consumer error handling written against the documented hierarchy silently misses expiry/replay failures.
- **Pattern-level close**: hierarchy rule + re-parent + parentage guard (see PAT-01).

### Medium (verified / spot-checked)

- **MED-01 / CON-02** [PAT-01] di.py mixes legacy raw `_ExecutionError(...)` with `.build(hint=)` style.
- **MED-02 / DRY** [PAT-02] config-carrier idiom ×4 + un-tabled injector twin (grep-verified).
- **MED-03 / PP-01** [PAT-03] no run-end eviction (bounded LRU only — softer than "never evicts" as originally phrased; verified `_store` has an LRU bound).
- **MED-04 / PP-02** [PAT-03] no per-key single-flight in `get_or_build`/`aget_or_build`.

### Low (summary only)

| ID | Agent | Description |
|----|-------|-------------|
| TQ-01 | testing | Oracle-over-agent isolation test docstrings overclaim "isolated message channels" (the count assertion catches collapse; Each suite is the content-isolation model) |
| TQ-02 | testing | observability payloads pin schema not values — by design, no action |
| DRY-L1/L2 | dry | RUN_ID reader ×2; fan-label re-derivation |
| CON-L1..L4 | consistency | minor naming/docstring parity items (see report) |
| LAY-L1/L2 | layering | informational notes (see report) |
| PP-L1..L4 | python | minor idiom notes (see report) |

## False Positives Discarded

| Original | Why |
|---|---|
| PP-01 as "never evicts / unbounded" | Partially — `_run_cache` HAS a bounded LRU (`_store` popitem). Re-scoped to "no run-end eviction." |

## Validation Summary

| Agent | Raw | Verified | False Pos |
|-------|-----|----------|-----------|
| testing | 2 Low | n/a | 0 |
| dry | 1 Med, 2 Low | Med grep-verified (4 sites) | 0 |
| consistency | 1 High, 1 Med, 4 Low | High live-verified; Med read | 0 |
| layering | 2 Low | n/a | 0 |
| python | 2 Med, 4 Low | both read at source; PP-01 re-scoped | 0 (1 re-scoped) |
| **Total** | **17** | **all C/H/M verified** | **0 discarded, 1 re-scoped** |

## Metrics

- Suite: 2650 passing; 589 arc tests re-run green by the testing reviewer
- Layering: clean, second consecutive review
- Release verdict: **no architectural blocker** — three pattern-level tickets close everything verified; all are small and land before the reformat + release per the maintainer's everything-in-scope rule
