# Changelog

All notable changes to NeoGraph will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 0.2.0.dev (develop branch)

### Changed — BREAKING

**Dependency-injection surface switched from `FromInput[T]` to `Annotated[T, FromInput]`.**
The previous form used `FromInput` / `FromConfig` as `typing.Generic` subscriptions,
which had a hidden rule: `FromInput[str]` meant "pull the parameter by name" but
`FromInput[SomePydanticModel]` silently meant "bundle — pull every field of the
model". Same syntax, two different resolution strategies based on whether the inner
type happened to be a `BaseModel`.

The 0.2 surface uses `typing.Annotated` with `FromInput` / `FromConfig` as
markers — the FastAPI dependency-injection pattern (`Annotated[User, Depends(...)]`).
The primary annotation is the real type; the marker tells neograph where the value
comes from:

```python
from typing import Annotated
from neograph import node, FromInput, FromConfig

# Before (0.1.x):
def my_node(topic: FromInput[str]) -> ...: ...
def my_node(ctx: FromInput[RunCtx]) -> ...: ...         # bundled (silently different)

# After (0.2.x):
def my_node(topic: Annotated[str, FromInput]) -> ...: ...
def my_node(ctx: Annotated[RunCtx, FromInput]) -> ...: ...  # bundled (same syntax)
```

**Why:** one resolution path, no hidden BaseModel rule, primary annotation is the
real type (IDE autocomplete sees `ctx: RunCtx` directly), standard typing semantics,
matches the FastAPI pattern Python developers already know. The internal
classifier is simpler and has fewer failure modes.

**Migration:** mechanical — wrap every existing `FromInput[T]` or `FromConfig[T]`
in `Annotated[T, FromInput]` / `Annotated[T, FromConfig]`. The old Generic-
subscription form is removed entirely (no deprecation shim — we are at 0.2).
Attempting `FromInput[str]` now raises `TypeError: type 'FromInput' is not
subscriptable`.

### Added

- **`@merge_fn` decorator** for Oracle merge functions with `FromInput` /
  `FromConfig` dependency injection. The first parameter receives the list of
  variants; subsequent parameters are resolved the same way as `@node`
  parameters. Legacy `(variants, config) -> output` merge functions still work
  unchanged. See `TestOracleMergeFnDI` for end-to-end examples.
- **`FromInput[PydanticModel]` bundles** (via the new `Annotated` surface) —
  constructs the model by pulling each of its declared fields from
  `config['configurable']` under the field's name. Eliminates per-field
  boilerplate for pipeline metadata (`node_id`, `project_root`, tenant
  context, etc.).
- **Frame-walking classifier** — handles locally-defined Pydantic model classes
  (e.g. `class RunCtx` inside a test method) by walking the caller's frame
  stack at decoration time. Needed because `from __future__ import annotations`
  strips closure references, the same technique Pydantic uses for forward-ref
  resolution.

### Fixed

- **`_validate_fan_in_types` unwraps Each-modified upstream outputs as
  `dict[str, output]`** before the type-compatibility check (`neograph-ayq`).
  Previously, a downstream `@node` parameter annotated `dict[str, MatchResult]`
  against an upstream with `.map()` would raise a false-positive rejection
  because the fan-in walker ignored the modifier. The `_construct_validation`
  walker already had this rule (fixed in 0.1.0 via `neograph-8k3`); this brings
  the `@node` walker in line.

## [0.1.0] - 2026-04-05

Initial public release.

### Added

**Three API surfaces, one compiler.**

- **`@node` decorator** — Dagster-style functions-as-nodes API. Parameter names
  are edges, the framework infers the topology from your function signatures.
  - `construct_from_module()` / `construct_from_functions()` assemble pipelines
    from `@node`-decorated functions
  - Mode inference: `prompt=` + `model=` → `produce`; neither → `scripted`
  - Five modes: `scripted`, `produce`, `gather`, `execute`, `raw`
  - Modifier kwargs: `map_over=` / `map_key=` (fan-out), `ensemble_n=` /
    `merge_fn=` / `merge_prompt=` (Oracle), `interrupt_when=` (human-in-the-loop)
  - Non-node parameters: `FromInput[T]`, `FromConfig[T]`, default-value constants
  - Full fan-in parameter type validation across all upstreams
  - Decoration-time validation with source-location error messages
  - Cross-module composition and name-collision detection

- **`ForwardConstruct`** — DSPy/PyTorch-style class-based API with Python control flow.
  - Subclass `ForwardConstruct`, declare `Node` class attributes, override `forward()`
  - Python `if` compiles to LangGraph conditional edges (symbolic-proxy tracing)
  - Python `for` over proxy attributes compiles to Each fan-out
  - Call `forward()` directly in tests with real values (not the traced graph)

- **Programmatic IR** — `Node` + `Construct` + `|` pipe syntax for runtime construction.
  - Runtime pipeline assembly from LLM output, config files, or routing layers
  - Assembly-time `ConstructError` validation on every IR-level construction
  - Construct-level default `llm_config` inherited by all produce nodes

- **Shared infrastructure**
  - `compile(construct)` → LangGraph `StateGraph`
  - `run(graph, input=..., resume=..., config=...)` → execution
  - `configure_llm(llm_factory, prompt_compiler)` → one-time LLM setup
  - `@tool` decorator for tool registration with per-tool budgets
  - `FromConfig[T]` for observability providers, rate limiters, shared resources
  - `Node.run_isolated()` for unit-testing individual nodes
  - `structlog`-based structured logging on every node execution

### Dependencies

- Python >= 3.11
- pydantic >= 2.0
- langgraph >= 0.2
- langchain-core >= 0.3
- structlog >= 23.0

Optional: `langfuse>=3.0` for observability integration.

[0.1.0]: https://github.com/KonstantinMirin/neograph/releases/tag/v0.1.0
