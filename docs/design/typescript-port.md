# neograph TypeScript Port -- Solution Design

## Executive Summary

neograph's core architecture (Construct IR -> compile -> LangGraph StateGraph -> run) ports cleanly to TypeScript. LangGraph.js is mature and mirrors the Python API. The highest-risk items are all in the DX layer (@node auto-wiring), not the runtime. The programmatic API (Node + Construct + .pipe()) is the natural TS-first surface.

**Estimated effort**: 8-12 weeks for feature parity on the runtime. The @node DX layer requires a redesign (explicit declarations instead of signature introspection).

---

## Feature Parity Matrix

### Legend
- **Direct** -- same logic, different syntax. < 1 day per feature.
- **Redesign** -- different approach needed due to TS limitations. 2-5 days.
- **Skip** -- Python-specific, not applicable to TS.
- **New** -- TS-specific feature not in Python.

### Core IR Layer

| Feature | Python | TypeScript | Effort | Notes |
|---------|--------|------------|--------|-------|
| Node model (all fields) | `Node(BaseModel)` | `Node` class + Zod schema | Direct | All 18 fields map 1:1 |
| Construct model | `Construct(BaseModel)` | `Construct` class + Zod | Direct | |
| Modifier base | `Modifier(BaseModel, frozen=True)` | Immutable class / `readonly` | Direct | |
| Oracle modifier | `Oracle(n, models, merge_fn/merge_prompt)` | Same fields | Direct | |
| Each modifier | `Each(over, key)` | Same fields | Direct | |
| Loop modifier | `Loop(when, max_iterations, on_exhaust, history)` | Same fields | Direct | |
| Operator modifier | `Operator(when)` | Same fields | Direct | |
| Tool model | `Tool(name, budget, config)` | Same fields | Direct | |
| Pipe syntax `\|` | `__or__` operator | `.pipe()` method chain | Redesign | `node.pipe(Oracle({n: 3}))` |
| Mutual exclusion guards | Each+Loop, Oracle+Loop, Oracle+Each, duplicates | Same logic in `.pipe()` | Direct | |
| Node.scripted() factory | classmethod | Static factory | Direct | |
| Node.run_isolated() | Method | Method | Direct | |

### Three API Surfaces

| Surface | Python | TypeScript | Effort | Notes |
|---------|--------|------------|--------|-------|
| **Programmatic** (Node + Construct + pipe) | `Node(...) \| Oracle(...)` | `Node({...}).pipe(Oracle({...}))` | Direct | **TS-first surface**. Maps naturally. |
| **@node decorator** (signature = DAG) | `@node(outputs=X)` + params = edges | `node({inputs, outputs}, fn)` | Redesign | Explicit input/output schemas required. See below. |
| **ForwardConstruct** (class + tracing) | `forward()` with `_Proxy.__bool__` | `forward()` with JS Proxy + fluent branch API | Redesign | Branch detection needs `.gt()/.lt()` + `.then()/.else()`. |
| **YAML spec** (load_spec) | `load_spec(yaml, project)` | `loadSpec(yaml, project)` | Direct | js-yaml + ajv |

### @node Equivalent in TypeScript

Python auto-wires from function signatures. TypeScript erases types at runtime. The TS equivalent uses explicit declarations:

```typescript
// Python
@node(outputs=Claims, model="reason", prompt="decompose")
def decompose(topic: RawText) -> Claims: ...

// TypeScript
const decompose = node({
  inputs: { topic: RawTextSchema },
  outputs: ClaimsSchema,
  model: "reason",
  prompt: "decompose",
}, async ({ topic }) => { ... });
```

| @node Feature | Python | TypeScript | Effort |
|---------------|--------|------------|--------|
| Mode inference (think/scripted) | Auto from prompt+model presence | Same logic from config fields | Direct |
| Output type inference | Return annotation | Explicit `outputs:` field | Direct |
| Input type inference | Parameter annotations | Explicit `inputs:` field | Redesign |
| Auto-wiring (param name = edge) | `inspect.signature` | Explicit `inputs: {name: Schema}` | Redesign |
| Fan-out (`map_over`) | `@node(map_over="x.items")` | `node({mapOver: "x.items", ...})` | Direct |
| Ensemble (`ensemble_n`) | `@node(ensemble_n=3)` | `node({ensembleN: 3, ...})` | Direct |
| Interrupt (`interrupt_when`) | `@node(interrupt_when="cond")` | `node({interruptWhen: "cond", ...})` | Direct |
| Loop (`loop_when`) | `@node(loop_when=fn)` | `node({loopWhen: fn, ...})` | Direct |
| Body-as-merge | Function body IS the merge | Same -- callback IS the merge | Direct |
| Dead-body warning | AST inspection | Skip (or ESLint plugin) | Skip |
| Sidecar pattern | `id(Node)` + weakref | WeakRef + FinalizationRegistry | Direct |

### DI System

| Feature | Python | TypeScript | Effort |
|---------|--------|------------|--------|
| FromInput (scalar) | `Annotated[str, FromInput]` | `fromInput: { topic: z.string() }` | Redesign |
| FromConfig (scalar) | `Annotated[T, FromConfig]` | `fromConfig: { limiter: LimiterSchema }` | Redesign |
| FromInput bundled (BaseModel) | `Annotated[RunCtx, FromInput]` | `fromInput: { ctx: RunCtxSchema }` | Redesign |
| required=True | `FromInput(required=True)` | `fromInput: { topic: z.string().required() }` | Direct |
| DI collision check | DI name matches upstream | Same logic | Direct |
| Double marker rejection | FromInput + FromConfig on same param | N/A (explicit fields can't collide) | Skip |

### Compiler Layer

| Feature | Python | TypeScript | Effort | Notes |
|---------|--------|------------|--------|-------|
| compile() | `compile(construct, checkpointer)` | `compile(construct, {checkpointer})` | Direct | Same structure |
| State model generation | `pydantic.create_model()` | `Annotation.Root({...})` with Zod | Redesign | LangGraph.js Annotation pattern |
| Simple node wiring | `add_node + add_edge` | `addNode + addEdge` | Direct | |
| Oracle fan-out (Send) | `Send(name, state)` N times | `Send(name, state)` | Direct | LangGraph.js has Send |
| Each fan-out (Send) | Router + Send per item | Same | Direct | |
| Loop back-edge | Conditional edge + counter | Same | Direct | |
| Subgraph compilation | Recursive compile() | Same | Direct | |
| Branch wiring (ForwardConstruct) | `_BranchNode` sentinel | Same approach | Direct | |
| Operator interrupt | interrupt() after node | interrupt() in LangGraph.js | Direct | |
| Reducers (last-write, append, collect, merge) | Annotated[type, fn] | `Annotation({reducer, default})` | Direct | LangGraph.js native |
| Pre-compile validations | 6 checks | Same checks | Direct | |
| Msgpack type registration | `register_msgpack_types` | Serialization for checkpointer | Medium | |

### Factory Layer (Runtime Dispatch)

| Feature | Python | TypeScript | Effort |
|---------|--------|------------|--------|
| Registries (scripted, condition, tool) | Global dicts | Module-level Maps | Direct |
| _extract_input | State field scanning | Same logic | Direct |
| _build_state_update | Dict-form outputs, Each keying | Same logic | Direct |
| _apply_skip_when | Predicate with error wrapping | Same | Direct |
| make_oracle_merge_fn | With @merge_fn DI + state params | Same closure pattern | Direct |
| make_subgraph_fn | Recursive invocation | Same | Direct |
| Scripted wrapper | register_scripted + shim | Same | Direct |
| LLM wrappers (produce, gather, execute) | _make_*_wrapper | Same | Direct |

### LLM Integration

| Feature | Python | TypeScript | Effort | Notes |
|---------|--------|------------|--------|-------|
| configure_llm() | Global factory + compiler | Same pattern | Direct | |
| Tier-based LLM creation | llm_factory(tier) | Same | Direct | LangChain.js has ChatOpenAI etc. |
| Structured output | `with_structured_output(Model)` | `withStructuredOutput(zodSchema)` | Direct | LangChain.js native |
| JSON mode | Schema in prompt + parse | Same | Direct | |
| Text mode (extract JSON) | Brace-counting extraction | Same algorithm | Direct | |
| Error-feedback retry | Append bad response + retry | Same | Direct | |
| Inline prompt substitution | `${var}` in prompt text | Template literals or same regex | Direct | |
| Renderers (XML, Delimited, JSON) | 3 renderer classes | Same 3 classes | Direct | |
| describe_type | TypeScript-style schema emitter | Already TS-native notation | Direct | |
| ReAct tool loop | invoke_with_tools | bindTools + ToolMessage loop | Direct | |
| Tool budget tracking | ToolBudgetTracker | Same class | Direct | |

### Validation

| Feature | Python | TypeScript | Effort |
|---------|--------|------------|--------|
| _validate_node_chain | Producer/consumer walker | Same walker, Zod schemas | Direct |
| effective_producer_type | Modifier-aware type | Same logic | Direct |
| _types_compatible | issubclass + Union unwrap | Zod `.safeParse()` or schema comparison | Redesign |
| Fan-in validation | Dict-form inputs check | Same logic | Direct |
| Each.key field existence | Pydantic model_fields | Zod `.shape` introspection | Direct |
| Context= validation | Field name check | Same | Direct |
| Output boundary contract | Sub-construct output check | Same | Direct |
| merge_fn state param validation | From_state field checks | Same | Direct |
| Optional/Union handling | typing.Union unwrapping | Zod `.unwrap()` / `._def.typeName` | Redesign |
| list[X] / dict[str,X] compat | Element-type recursion | Same logic | Direct |

### Spec Loader

| Feature | Python | TypeScript | Effort |
|---------|--------|------------|--------|
| YAML parsing | yaml.safe_load | js-yaml | Direct |
| JSON Schema validation | jsonschema | ajv | Direct |
| Type registry | register_type / lookup_type | Same Map pattern | Direct |
| Dynamic model generation | Pydantic create_model from schema | Zod schema from JSON Schema (zod-to-json-schema inverse) | Medium |
| Size limit (YAML bomb) | 1MB check | Same | Direct |
| Condition parser | parse_condition("field > value") | Same regex/parser | Direct |

### CLI & Tooling

| Feature | Python | TypeScript | Effort |
|---------|--------|------------|--------|
| `neograph check` | __main__.py + compile + lint | `npx neograph check` via Commander.js | Direct |
| lint() | DI binding checker | Same walker | Direct |
| Error hierarchy | 5 error classes | Same class hierarchy | Direct |

### ForwardConstruct (Class-Based Tracing)

| Feature | Python | TypeScript | Effort | Notes |
|---------|--------|------------|--------|-------|
| Proxy symbolic tracing | `_Proxy.__getattr__` | JS `Proxy` handler (get trap) | Direct | |
| Branch detection | `__bool__` interception | `.gt()/.lt()` + `.then()/.else()` fluent API | Redesign | Can't intercept `if (proxy)` in JS |
| Loop detection | `__iter__` protocol | `.loop()` explicit method | Redesign | |
| Re-trace strategy | All-true + flip each branch | Same algorithm | Direct | |
| _merge_branch_traces | Shared prefix/suffix + BranchNode | Same | Direct | |
| Max 8 branches | Limit check | Same | Direct | |

---

## Architecture Decisions

### AD-1: Zod as the schema layer (not TypeBox, not io-ts)

Zod is the industry standard for TS runtime validation. LangChain.js uses Zod natively for `withStructuredOutput()`. This means neograph-ts gets structured output for free -- no adapter layer.

### AD-2: Programmatic API is the primary surface

The Python `@node` decorator surface relies on runtime introspection that doesn't exist in TS. Instead:

1. **Primary**: `Node({...}).pipe(Oracle({...}))` -- programmatic composition
2. **Secondary**: `node({inputs, outputs, ...}, fn)` -- wrapper function (not a decorator)
3. **Tertiary**: YAML spec via `loadSpec()` -- same as Python

The programmatic API is already the primary path for LLM-driven runtime pipeline construction, so this aligns with the core use case.

### AD-3: Type checking via Zod schema comparison

Python uses `issubclass()` and `typing.get_origin/get_args`. TS uses Zod schema comparison:

```typescript
function typesCompatible(producer: z.ZodType, target: z.ZodType): boolean {
  // Generate a sample value from producer, validate against target
  // Or compare schema shapes structurally
}
```

### AD-4: LangGraph.js Annotation for state models

Instead of `pydantic.create_model()`, use LangGraph.js's native state definition:

```typescript
const state = Annotation.Root({
  decompose: Annotation<Claims | null>({
    reducer: (_, update) => update,
    default: () => null,
  }),
  verify: Annotation<Record<string, MatchResult>>({
    reducer: (existing, update) => ({...existing, ...update}),
    default: () => ({}),
  }),
});
```

### AD-5: Branch API for ForwardConstruct

Replace Python's `if proxy:` with an explicit fluent API:

```typescript
class MyPipeline extends ForwardConstruct {
  forward(input: Proxy) {
    const claims = this.extract(input);
    return claims.score.gt(0.7)
      .then(() => this.expand(claims))
      .else(() => this.simplify(claims));
  }
}
```

---

## Effort Estimate by Module

| Module | Files | Days | Dependencies |
|--------|-------|------|-------------|
| Error hierarchy | 1 | 0.5 | None |
| Core IR (Node, Construct, Modifiers, Tool) | 4 | 3 | Zod |
| Pipe composition (.pipe method) | 1 | 1 | Core IR |
| Renderers (XML, Delimited, JSON, describe_type) | 2 | 2 | Zod |
| Spec loader (YAML + type registry) | 3 | 3 | Core IR, Zod, js-yaml, ajv |
| Validation (_construct_validation equivalent) | 1 | 4 | Core IR, Zod |
| State model generation (Annotation.Root) | 1 | 3 | LangGraph.js |
| Compiler (compile -> StateGraph wiring) | 1 | 5 | State model, LangGraph.js |
| Factory (runtime dispatch, _extract_input) | 1 | 5 | Compiler, LangGraph.js |
| LLM integration (structured/json/text output) | 1 | 3 | LangChain.js |
| ReAct tool loop + budget | 1 | 2 | LLM integration |
| node() wrapper function (TS @node equivalent) | 1 | 3 | Core IR, validation |
| DI system (fromInput/fromConfig) | 1 | 2 | Factory |
| construct_from_functions equivalent | 1 | 2 | node() wrapper |
| Runner (run + config injection) | 1 | 1 | Compiler |
| Lint | 1 | 1 | Core IR |
| CLI (neograph check) | 1 | 1 | Compiler, Lint |
| ForwardConstruct (Proxy tracing) | 1 | 5 | Compiler |
| Tests | -- | 10 | All |
| **Total** | **~24** | **~56 days** | |

**With parallelism (2 engineers)**: ~8 weeks.
**Solo**: ~12 weeks.

---

## What Ships in v0.1.0-ts

### Phase 1: Core (weeks 1-4)
- Node, Construct, Modifiers (all 4), Tool
- .pipe() composition
- compile() -> LangGraph.js StateGraph
- State model generation via Annotation.Root
- Factory dispatch (scripted + LLM modes)
- Basic validation (fan-in, type compat, mutual exclusion)
- run() + config injection
- Renderers

### Phase 2: Features (weeks 5-8)
- node() wrapper function with explicit schemas
- DI system (fromInput/fromConfig)
- construct_from_functions equivalent
- LLM integration (structured/json/text output strategies)
- ReAct tool loop + budget tracking
- Spec loader (YAML/JSON)
- Lint + CLI
- Error-feedback retry

### Phase 3: Advanced (weeks 9-12)
- ForwardConstruct with JS Proxy + fluent branch API
- Full validation suite (all checks from Python)
- Oracle merge_fn with state params
- Dev-mode warnings
- Test suite (port fixture-based safety net)

### Not in v0.1.0-ts
- construct_from_module (no module introspection in TS)
- Dead-body AST warning (ESLint plugin if needed)
- Frame stack walking (not needed with explicit declarations)

---

## Shared Artifacts

These Python artifacts work unchanged in TS:
- YAML spec format (cross-language by design)
- Prompt templates (Markdown with `${var}` substitution)
- Renderer output (XML/JSON/Delimited are format strings)
- `describe_type` output (already emits TypeScript-style notation)
- Error message formats
- Validation error messages
- Fixture-based test methodology (should_fail/should_pass pattern)
