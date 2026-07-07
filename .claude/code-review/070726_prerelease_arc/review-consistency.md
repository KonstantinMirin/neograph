# Consistency Review

**Scope**: `cf14801^..HEAD` on `develop` (24 commits, 71 files) — di_inputs, hgpt v1+v2, fan-over-agent, run_id, native json_mode, span hygiene, wiring dedup. Whole-codebase consistency audit before 0.6.0, with the emphases the lead flagged: error-type boundary across new raises, `_neo_`/`neo_` config-key family, new public export surface + `__all__`, lint kind vocabulary, structlog event naming.
**Date**: 2026-07-07
**Reviewer**: review-consistency (read-only; wrote only this file)

## Convention Inventory

| Convention | Canonical Form | Following | Diverging |
|------------|---------------|-----------|-----------|
| Structured error build | `Err.build(what_lowercase, *, expected/found/hint/node/construct)` | all arc code in `prompt.py`, `tool.py`, `_fan_agent*.py`, `_llm*.py`, resource errors in `di.py` | legacy `DIBinding.resolve` (5× raw `_ExecutionError("Sentence. Provide via run().")`) |
| Diagnostic-error classmethod ctor | `Err.of(...)` returning a typed instance | `PromptVarMissing.of`, `NonIdempotentReplayError.of`, `ResourceExpiredError.of` | `StateMissingError.build` (same idea, `.build` name, different shape) |
| Runtime-error parent | `class XError(ExecutionError)` for execution-time failures | `PromptVarMissing(ExecutionError)` | `ResourceExpiredError(NeographError)`, `NonIdempotentReplayError(NeographError)` |
| `neo_` = state-bus key, `_neo_` = config-only key | leading `_` ⇒ never enters state | `DI_INPUTS`, `RESOURCE_MANIFEST_INJECT`, `RUN_ID`, `STREAM_CUSTOM`, `CONFIG_INPUT` | `ISOLATED_INPUT="_neo_isolated_input"` is a state-DICT key (documented historical) |
| `_neo_` key docstring triple | "NOT a state-bus key" + "Mirrors X pattern" + "cannot touch fingerprint" | `DI_INPUTS`, `RESOURCE_MANIFEST_INJECT`, `RUN_ID` | `STREAM_CUSTOM`, `CONFIG_INPUT` (older, partial) |
| Lint `kind=` vocabulary | snake_case `<subject>_<condition>` | all 10 kinds | — (consistent) |
| structlog event name | snake_case verb/noun literal first arg | all events incl. new `json_mode_native_unsupported`, `each_over_absent` | — (consistent) |
| `__all__` lists every public import | new: `ResourceRef`, `ProducingCall`, `BlobResult`, `FromResource`, `resource_reader`, `read_blob`, `ResourceExpiredError`, `NonIdempotentReplayError`, `PromptVarMissing` all present | `ToolInteraction` imported but absent from `__all__` (pre-existing) |
| Tool-metadata key prefix | `ng_` (`ng_idempotent`, `ng_tool_budget`) | consistent within itself | differs from framework `neo_` (justified: foreign LangChain namespace) |

## Findings

### CON-01: New runtime errors split their parent class inconsistently
- **Severity**: High
- **Convention**: `ExecutionError` is documented as the base for "Runtime errors during graph execution" (`errors.py:96-101`). Downstream code catching `except ExecutionError` expects to catch every execution-time failure.
- **Files**:
  - `errors.py:143` `PromptVarMissing(ExecutionError)` — raised at runtime from `substitute()` during prompt compilation.
  - `errors.py:224` `NonIdempotentReplayError(NeographError)` — raised at runtime from `hydrate_resource_ref` (`di.py:209`).
  - `errors.py:259` `ResourceExpiredError(NeographError)` — raised at runtime from `hydrate_resource_ref` (`di.py:213,223`).
- **Description**: All three are new fail-loud errors raised **during graph execution** (on the `arun()` node-execution path). One of them (`PromptVarMissing`) is placed under `ExecutionError`; the other two are placed directly under `NeographError`, skipping `ExecutionError`. A consumer doing `except ExecutionError` to handle runtime failures catches the prompt-var miss but silently misses resource-expiry/non-idempotent-replay — even though the `ExecutionError` docstring explicitly promises to cover exactly this class ("LLM response parse failures", "duplicate fan-out keys", etc., all execution-time). The split is not principled: the two that skip `ExecutionError` are precisely the ones raised deepest in the run path. `NodeOutputError` and `StateMissingError` also sit directly under `NeographError`, so there is an existing "diagnostic errors bypass ExecutionError" drift, but this arc widened it with two more.
- **Reproduction**:
  ```bash
  grep -nE '^class .*Error\(' src/neograph/errors.py
  # PromptVarMissing(ExecutionError)  vs  ResourceExpiredError(NeographError) / NonIdempotentReplayError(NeographError)
  ```
- **Recommended fix**: Pick one rule and apply it to all three. Given the `ExecutionError` docstring's promise, the coherent choice is `ResourceExpiredError(ExecutionError)` and `NonIdempotentReplayError(ExecutionError)` (both are execution-time). If instead the intent is that these carry structured diagnostic payloads and should be catchable independently of the broad `ExecutionError` bucket, then document that rule on `NeographError` and move `PromptVarMissing` to match — but do not leave the three arc-siblings straddling the boundary.

### CON-02: Two error-construction/message styles co-located in `di.py`
- **Severity**: Medium
- **Convention**: `errors.py:39` documents `.build()` as the structured path — "All neograph errors go through here" — producing lowercase phrase + `hint=` remediation (e.g. `di.py:122` `_ConfigurationError.build("resource mime '...' cannot be parsed into X", hint="pass parse=...")`).
- **Files**:
  - New style: `di.py:122`, `:190`, `:213`, `:391`, `:427` — all `.build(...)`/`.of(...)` with `hint=`.
  - Legacy style: `di.py:328`, `:336`, `:354`, `:364`, `:454` — 5× raw `_ExecutionError(f"Required DI parameter '{name}' (from input) is missing ... Provide it via run(input={...}).")`.
- **Description**: `DIBinding.resolve`/`aresolve` mixes both conventions in the same class. The legacy DI messages are sentence-case, terminate with a period, and inline the remediation ("Provide it via run(...)") into the message body; the arc's new resource messages are lowercase phrases with the remediation carried in a separate `hint=` and (for `.of()` builders) a node/param pinpoint. Same module, same method family, two shapes. This is the "are error messages formatted consistently?" checklist item — a `pytest.raises(match=...)` or a downstream log parser sees two grammars from one resolver.
- **Reproduction**:
  ```bash
  grep -n 'raise _ExecutionError(' src/neograph/di.py   # 5 raw, sentence-style
  grep -nE '\.(build|of)\(' src/neograph/di.py           # new, hint=-style
  ```
- **Recommended fix**: Route the 5 legacy DI raises through `ExecutionError.build(what_lowercase, hint="provide it via run(input={...})", node=...)`. This also lets them carry the node context the resource errors already carry, closing the "fail-loud messages consistently name node+param" gap the lead asked about — currently the resource errors name `node=`, the DI errors name only the param.

### CON-03: `.build()` is one method name with two different contracts + a duplicated body
- **Severity**: Low
- **Convention**: On `NeographError`/`ExecutionError`, `build(what, *, expected, found, hint, location, node, construct)` is the structured formatter.
- **Files**: `errors.py:28` (base), `errors.py:108` (`ExecutionError.build`, verbatim copy of the base body to thread `validation_errors`), `errors.py:182` (`StateMissingError.build(*, key, node_label)` — a *different* signature and a different message shape `[Node 'X'] required state key 'Y' not found`, bypassing the expected/found/hint format).
- **Description**: Two consistency risks. (1) `ExecutionError.build` re-implements the entire `[Node…] what / expected / found / hint / at` assembly (lines 121-140 duplicate 49-68); if the canonical message layout ever changes, the `ExecutionError` variant silently drifts. (2) `StateMissingError.build` reuses the `.build` name for an unrelated `(key, node_label)` contract that does not produce the structured layout — so "`.build`" no longer means one thing across the hierarchy. Neither is arc-introduced, but the arc leans heavily on `.build()`/`.of()` and inherits the ambiguity.
- **Recommended fix**: Have `ExecutionError.build` delegate to `super().build(...)` and only attach `validation_errors` to the returned instance, eliminating the copied body. Consider renaming `StateMissingError.build` → `StateMissingError.of` to reserve `.build` for the structured-format contract (matches the `.of` convention the diagnostic errors already use).

### CON-04: `_neo_` config-key docstring triple is followed by the arc's keys but not the two older ones
- **Severity**: Low
- **Convention**: The arc established a tight 3-part docstring for config-only `_neo_` keys in `_state_keys.py`: (a) "is a config['configurable'] key (NOT a state-bus key)", (b) "Mirrors the … config-injection pattern", (c) "Never enters state … cannot touch the schema fingerprint".
- **Files**: Following all three — `DI_INPUTS` (`:70`), `RESOURCE_MANIFEST_INJECT` (`:79`), `RUN_ID` (`:87`). Partial — `STREAM_CUSTOM` (`:56`, has (a) and the fingerprint note but not the "Mirrors X" line) and `CONFIG_INPUT` (`:54`, has (a) only).
- **Description**: The lead asked specifically about "docstring convention parity in `_state_keys.py`". The three keys this arc added are mutually consistent and set a good template; the two predecessors that the same family should match are the laggards. Minor, docstring-only.
- **Recommended fix**: Add the "Mirrors …" + fingerprint sentence to `STREAM_CUSTOM` and `CONFIG_INPUT` so the whole `_neo_` block reads uniformly. No behavior change.

### CON-05: `ToolInteraction` is public but missing from `__all__`
- **Severity**: Low
- **Convention**: `__init__.py` lists every intentionally-public symbol in `__all__`; the arc correctly added `ResourceRef`, `ProducingCall`, `BlobResult`, `resource_reader`, `read_blob`, `FromResource`, and the three new errors.
- **Files**: `__init__.py:90` imports `ToolInteraction`; it is absent from `__all__` (`:97-184`), unlike its freshly-added siblings `ResourceRef`/`ProducingCall` two lines below it in the same import block.
- **Description**: `ToolInteraction` is a documented public type (AGENTS.md's "Gather tool collection" section; the `tool_log` output contract). It is reachable as `neograph.ToolInteraction` but excluded from `from neograph import *` and from the public surface listing. Pre-existing, but the arc surfaced it by adding `ResourceRef`/`ProducingCall`/`BlobResult` right next to it — the one sibling left out now stands out.
- **Recommended fix**: Add `"ToolInteraction"` to `__all__` near `ResourceRef`/`ProducingCall`.

### CON-06 (informational): tool-metadata prefix `ng_` vs framework `neo_`
- **Severity**: Low (informational — likely justified, flagging for a conscious decision)
- **Files**: `tool.py:379,381` write `ng_idempotent` / `ng_tool_budget` into LangChain `BaseTool.metadata`; the rest of the framework namespaces internal keys with `neo_`/`_neo_` (`_state_keys.py`).
- **Description**: Two framework-owned prefixes exist. `ng_` is used only on the foreign LangChain tool-metadata dict, where a distinct short prefix is defensible (avoids colliding with LangChain's own `neo`-agnostic keys, and it is namespaced away from state). Internally consistent (2 keys, both `ng_`). Raising only so the divergence is a decision, not an accident.
- **Recommended fix**: None required. If uniformity is preferred, `neo_idempotent`/`neo_tool_budget` would align the prefix; otherwise document `ng_` = "neograph tool-metadata namespace" alongside the `neo_` note in `_state_keys.py`.

## Positive observations (consistency held)

- **Lint kind vocabulary** is coherent across all 10 kinds; the ~6 arc additions (`resource_hydration_kind_unmatched`, `template_placeholder_unresolvable`, `template_placeholder_known_vars_only`, `template_var_requires_async_driver`, `tool_requires_async_driver`, `ask_human_in_mutating_node`, `act_mode_all_idempotent_tools`) all use snake_case and the two parallel `*_requires_async_driver` kinds share the exact suffix for the same predicate — good parallel naming.
- **structlog event names** are uniformly snake_case literal first-args; the new `json_mode_native_unsupported` (`_llm.py:105`) and `each_over_absent` fit the existing `<noun>_<state>` house style.
- **`.of()` classmethod parity** across the three diagnostic errors (`PromptVarMissing.of`, `NonIdempotentReplayError.of`, `ResourceExpiredError.of`) — same construction idiom, same typed-payload approach.
- **DI marker parity**: `FromInput`/`FromConfig`/`FromResource` all export from `neograph.decorators` via `__init__` and share the `Annotated[T, Marker]` grammar.
- **Config-callable key parity**: `mcp_resource_fetcher`/`mcp_resource_replayer` share the `mcp_` prefix and the "consumer-owned, no session ownership" docstring framing.

## Summary

- Critical: 0
- High: 1 (CON-01 error-hierarchy parent split)
- Medium: 1 (CON-02 two error-construction styles in `di.py`)
- Low: 4 (CON-03 `.build` overload/dup, CON-04 docstring-triple laggards, CON-05 `ToolInteraction` `__all__`, CON-06 `ng_` prefix — informational)

The single item worth resolving before 0.6.0 is **CON-01**: it changes what `except ExecutionError` catches at runtime, and the two arc-new resource errors are on the wrong side of a boundary the `ExecutionError` docstring explicitly claims. CON-02 is the natural companion fix (routing the legacy DI raises through `.build()` both aligns message format and lets them carry `node=`). The rest are docstring/export hygiene.
