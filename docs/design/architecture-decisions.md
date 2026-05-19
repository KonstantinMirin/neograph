# Architecture decisions

Authoritative spec for neograph's architecture. Tickets cite section numbers from this file. New design that this file does not cover adds a section here first.

---

## 1. LangGraph commitment

neograph commits to LangGraph-family runtimes — Python on LangGraph, TypeScript on LangGraphJS. Provider-neutrality lives at the model-invocation layer (user's `llm_factory` returns any `BaseChatModel`), not at the graph-execution layer.

- The IR may import `RunnableConfig` directly.
- Private imports from `langgraph.*` (any `_`-prefixed path) are forbidden; structural guard enforces.
- LangGraph version range is pinned in `pyproject.toml` to a supported public-API window.
- Feature additions answer "does this exist in both LangGraph and LangGraphJS?" Python-only features ship with the asymmetry documented; the IR is not preemptively shrunk to the minimum common subset.
- `CLAUDE.md` / `AGENTS.md` and website positioning state the LangGraph commitment. "Backend-neutral" framing does not appear anywhere.

---

## 2. compile() reads inputs from keyword arguments

`compile()` is the sole entry point for runtime configuration. Module-level registration functions do not exist; the kwargs below are the only registration path.

```python
compile(
    construct,
    llm_factory=...,
    prompt_compiler=...,
    cost_callback=...,
    tool_factories={...},
    conditions={...},
)
```

These kwargs are closed over into factory closures at compile time. The compiled graph has no hidden process state.

- Two `compile()` calls with different kwargs produce fully isolated compiled graphs that can coexist in one process.
- `run(graph, ...)` does not accept runtime overrides. Multi-tenant servers recompile per tenant.
- `lint()` and `Node.run_isolated()` accept the same kwargs as `compile()`. For LLM-mode nodes the LLM kwargs are required (same fail-loud rule as `compile()`).
- `@node` scripted shims register into a per-compile lookup; no process-global state.
- `compile()` fails loudly when the construct contains LLM-mode nodes and the required LLM kwargs are missing.

---

## 3. Retry concerns

Each retry concern lives in exactly one layer.

| Concern | Where it lives | Mechanism |
|---|---|---|
| Transient API failures (network, 429, 5xx, timeouts) | User's `llm_factory` | `model.with_retry(...)` or SDK-level retry |
| LLM output-quality failures (malformed JSON, schema violations) | Per-node `LlmConfig.max_retries` | `_invoke_json_with_retry` with BAML-rendered feedback |
| Flaky external calls inside a scripted node | Inside the node function | Local explicit retry; not a framework concern |

- `compile()` does not accept a `retry_policy` kwarg. LangGraph's `RetryPolicy` is not exposed at the framework boundary; if a future need surfaces, it becomes an opt-in per-node field, never a blanket compile-level knob.
- Output-quality retry uses neograph's BAML-rendered feedback path. No foreign parser libraries (LangChain `OutputFixingParser`, `RetryWithErrorOutputParser`, etc.); BAML rendering beats JSON-Schema format instructions where it matters most.
- Output-quality retry is safe in `act` mode: it reformats the already-produced answer and does not re-run tool calls.

Open: `LlmConfig.total_call_budget` for a shared retry budget. Not implemented until a consumer reports confusing token spend.

---

## 4. Module and function responsibilities

Each module and each function owns one named responsibility. God-modules and god-functions are forbidden.

- Each module's docstring states its single responsibility in one line. Code that doesn't fit goes to a sibling module.
- Each function does one job. Multi-axis dispatch (e.g., "factory construction + state I/O + observability + skip logic") splits before it grows.
- Imports form a DAG. Cycles are broken by extracting Protocol modules — small files holding the types and interfaces that participants depend on, instead of participants depending on each other.
- Third-party imports declared in `pyproject.toml` appear at module top, never inside function bodies.
- Function-local `from neograph...` imports are forbidden in new code; a structural guard enforces, with a shrinking documented allowlist for existing cycles.

---

## 5. Type and error discipline

`Any` is forbidden in public APIs of IR modules (`node.py`, `construct.py`, `modifiers.py`, `_construct_validation.py`) and in framework dispatch layers (`factory.py`, `_dispatch.py`, `_oracle.py`, `_wiring.py`).

- `Any` appears only at user-supplied-data boundaries — state values whose types the user declares, raw model outputs before parsing.
- Approved external types (e.g., `RunnableConfig` from `langchain_core.runnables`, per §1) may appear in IR public APIs.
- Polymorphic IR fields (e.g., `Node.outputs: type | dict[str, type] | None`) are accessed through normalizers; see §8.
- `arbitrary_types_allowed=True` on Pydantic models is justified case-by-case in a module-level comment; new uses require sign-off.
- Public functions raise typed `NeographError` on failure.

---

## 6. Test discipline

Tests target behavior, not dispatch path.

- Pure functions and isolated algorithms (renderers, fingerprint walks, normalizers, validators) earn direct tests on their inputs and outputs. Internal helpers with non-trivial logic are tested on their contract.
- Mock plumbing that asserts a specific call sequence ("invoke called 3 times, then bind_tools, then invoke") is forbidden. The contract is the result, not the order of internal calls.
- Tests do not assert on internal LangGraph dispatch routing or on private node-naming conventions. The public introspection surface (e.g., `compiled.get_graph()` via `describe_graph`) is fair game when the test asserts on a contract neograph exposes to users.
- When a test fails after a refactor that did not change user-visible behavior, the test was pinning implementation. Rewrite or delete.

---

## 7. State bus

`neo_*` state-bus keys are defined in one place. A structural guard prevents `neo_*` string literals from appearing outside that module.

- The default state read raises `NeographError` on missing fields.
- An explicitly-optional read returns `None` silently. Used only where the missing case is meaningful (e.g., reading a not-yet-populated collector field during fan-out, or reading an output a downstream node has not produced yet within a partial run).
- Silent-`None` reads of required fields are a bug.

`run()` does not mutate the caller's input dict. Framework keys (e.g., schema fingerprint) are injected into a defensive copy.

---

## 8. Polymorphic Node.outputs / Node.inputs

`Node.outputs` and `Node.inputs` are user-facing polymorphic (`type | dict[str, type] | None`). The framework accesses them only through a normalizer that returns a tagged result.

- No `isinstance(node.outputs, dict)` or `isinstance(node.inputs, dict)` checks exist in `src/neograph/` outside the normalizer module.
- Structural guard enforces.

---

## 9. Schema fingerprinting

`_compute_invalidated_nodes` returns the transitive closure of changed nodes.

- Adjacency is keyed by state-field name, so modifier-bearing nodes (Each, Oracle, Loop) participate under their state-field names. A change to A invalidates A and all transitive consumers, even when downstream nodes' own type signatures are unchanged.

`Node.run_isolated` raises `NeographError` on missing output fields for in-scope nodes. Modifier-bearing nodes (Each, Oracle, Loop) are out of scope for `run_isolated` and raise `NeographError` at entry with a message pointing to `compile()` + `run()`.
