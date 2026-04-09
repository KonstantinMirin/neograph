# neograph TypeScript Port -- Solution Design

## Executive Summary

neograph's core architecture (Construct IR -> compile -> LangGraph StateGraph -> run) ports cleanly to TypeScript. LangGraph.js is mature and mirrors the Python API.

**Key insight**: a TS compiler transformer (ts-patch) can extract function signatures at build time -- preserving the Python "signature IS the DAG" DX with zero runtime overhead. This eliminates the need for explicit Zod declarations on every node.

**Estimated effort**: 8-12 weeks for feature parity.

---

## AD-0: Compiler Transformer for Signature Extraction (CRITICAL DECISION)

### The problem

Python's `inspect.signature` + `get_type_hints` gives neograph full parameter metadata at runtime. TypeScript erases types at runtime. The naive solution (require explicit `inputs: {name: Schema}` on every node) works but degrades the DX.

### The solution: ts-patch transformer

A TypeScript compiler transformer intercepts `@node`-decorated functions during `tsc`, calls `program.getTypeChecker()` to resolve parameter names/types/return type, and emits runtime metadata alongside the function.

**This is proven technology.** [typia](https://github.com/samchon/typia) does exactly this -- it extracts full function signatures (parameter names, types, nested object shapes) at compile time and emits JSON Schema. typia's `llm.application<T>()` is structurally identical to what neograph needs.

### What the user writes (near-identical to Python):

```typescript
// Python:
@node(outputs=Claims, model="reason", prompt="decompose")
def decompose(topic: RawText) -> Claims: ...

// TypeScript (wrapper function -- TS decorators don't work on standalone fns):
const decompose = node({ model: "reason", prompt: "decompose" },
  (topic: RawText): Claims => { ... }
);
```

### What the transformer emits:

```typescript
decompose.__neo_meta = {
  inputs: { topic: { typeName: "RawText", schema: RawTextSchema } },
  output: { typeName: "Claims", schema: ClaimsSchema },
  name: "decompose",
};
```

### Setup (one-time):

```json
// tsconfig.json
{
  "compilerOptions": {
    "plugins": [{ "transform": "@neograph/transform" }]
  }
}
```

Or via `@neograph/unplugin` for Vite/Webpack/Bun/Next.js (typia's proven pattern).

### tsgo risk and fallback

TypeScript 7 is being rewritten in Go. The Go version will NOT support JS-based plugins initially (filed as "Post-7.0"). However:
- TypeScript 6.0 (bridge release) keeps current tsc -- transformer works
- tsgo initially targets `--noEmit` only -- emit-side plugins irrelevant
- **Runway: 2-3 years minimum**

**Fallback**: `neograph generate` via ts-morph (TS Compiler API wrapper). Same metadata shape, different extraction mechanism. Reads source files, writes `_neo_generated.ts`. Similar to `prisma generate` but for node signatures.

### Ruled out alternatives

| Approach | Why not |
|----------|---------|
| SWC plugins | Syntax-only, no TypeChecker access. Cannot resolve type aliases or follow imports. |
| TC39 decorator metadata | Spec deliberately excludes `design:paramtypes`. No parameter type capture. |
| typescript-rtti | Project in hibernation. Maintainer uncertain about future. |
| `Reflect.metadata` (legacy) | Deprecated, tied to legacy decorator proposal. Not available with Stage 3. |

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

With the compiler transformer (AD-0), TypeScript preserves Python's "signature IS the DAG" DX:

```typescript
// Python:
@node(outputs=Claims, model="reason", prompt="decompose")
def decompose(topic: RawText) -> Claims: ...

// TypeScript (compiler transformer extracts inputs/output from signature):
const decompose = node({ model: "reason", prompt: "decompose" },
  (topic: RawText): Claims => { ... }
);
// Build step emits: { inputs: { topic: RawText }, output: Claims }
```

Without the transformer (fallback), explicit declarations are needed:

```typescript
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
| Output type inference | Return annotation | **Compiler transformer extracts** | Direct (with transformer) |
| Input type inference | Parameter annotations | **Compiler transformer extracts** | Direct (with transformer) |
| Auto-wiring (param name = edge) | `inspect.signature` | **Compiler transformer emits `__neo_meta`** | Medium (transformer dev) |
| Fan-out (`map_over`) | `@node(map_over="x.items")` | `node({mapOver: "x.items", ...})` | Direct |
| Ensemble (`ensemble_n`) | `@node(ensemble_n=3)` | `node({ensembleN: 3, ...})` | Direct |
| Interrupt (`interrupt_when`) | `@node(interrupt_when="cond")` | `node({interruptWhen: "cond", ...})` | Direct |
| Loop (`loop_when`) | `@node(loop_when=fn)` | `node({loopWhen: fn, ...})` | Direct |
| Body-as-merge | Function body IS the merge | Same -- callback IS the merge | Direct |
| Dead-body warning | AST inspection | Skip (or ESLint plugin) | Skip |
| Sidecar pattern | `id(Node)` + weakref | WeakRef + FinalizationRegistry | Direct |

### DI System

With the compiler transformer, DI markers can use branded types that the transformer recognizes:

```typescript
import { FromInput, FromConfig } from "@neograph/core";

const myNode = node({ model: "reason", prompt: "test" },
  (upstream: Claims, topic: FromInput<string>, limiter: FromConfig<RateLimiter>): Result => { ... }
);
// Transformer extracts: { upstream: "Claims", topic: { kind: "from_input", type: "string" }, ... }
```

| Feature | Python | TypeScript | Effort |
|---------|--------|------------|--------|
| FromInput (scalar) | `Annotated[str, FromInput]` | `FromInput<string>` branded type | Direct (with transformer) |
| FromConfig (scalar) | `Annotated[T, FromConfig]` | `FromConfig<T>` branded type | Direct (with transformer) |
| FromInput bundled (BaseModel) | `Annotated[RunCtx, FromInput]` | `FromInput<RunCtx>` (transformer detects BaseModel-like) | Direct (with transformer) |
| required=True | `FromInput(required=True)` | `FromInput<string, {required: true}>` | Direct |
| DI collision check | DI name matches upstream | Same logic | Direct |
| Double marker rejection | FromInput + FromConfig on same param | Transformer rejects at build time | Direct |

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

### AD-2: All three API surfaces supported

With the compiler transformer, all three Python surfaces have TS equivalents:

1. **Programmatic**: `Node({...}).pipe(Oracle({...}))` -- runtime composition by LLMs
2. **node() wrapper**: `node({model, prompt}, (topic: RawText): Claims => {...})` -- transformer extracts signature
3. **YAML spec**: `loadSpec(yaml, project)` -- same as Python

The DX gap between Python and TS is minimal: `@node(...)` decorator syntax becomes `node({...}, fn)` wrapper syntax. The transformer ensures parameter names/types are still auto-wired.

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
| **Compiler transformer (@neograph/transform)** | **3** | **8** | **ts-patch, TS Compiler API** |
| **Unplugin wrapper (@neograph/unplugin)** | **1** | **2** | **Transformer** |
| Renderers (XML, Delimited, JSON, describe_type) | 2 | 2 | Zod |
| Spec loader (YAML + type registry) | 3 | 3 | Core IR, Zod, js-yaml, ajv |
| Validation (_construct_validation equivalent) | 1 | 4 | Core IR, Zod |
| State model generation (Annotation.Root) | 1 | 3 | LangGraph.js |
| Compiler (compile -> StateGraph wiring) | 1 | 5 | State model, LangGraph.js |
| Factory (runtime dispatch, _extract_input) | 1 | 5 | Compiler, LangGraph.js |
| LLM integration (structured/json/text output) | 1 | 3 | LangChain.js |
| ReAct tool loop + budget | 1 | 2 | LLM integration |
| node() wrapper function + auto-wiring | 1 | 2 | Core IR, transformer |
| DI system (FromInput/FromConfig branded types) | 1 | 2 | Factory, transformer |
| construct_from_functions equivalent | 1 | 2 | node() wrapper |
| Runner (run + config injection) | 1 | 1 | Compiler |
| Lint | 1 | 1 | Core IR |
| CLI (neograph check) | 1 | 1 | Compiler, Lint |
| ForwardConstruct (Proxy tracing) | 1 | 5 | Compiler |
| Tests | -- | 10 | All |
| **Total** | **~27** | **~65 days** | |

**With parallelism (2 engineers)**: ~9 weeks.
**Solo**: ~14 weeks.

The transformer adds ~10 days vs the explicit-declarations approach, but preserves the "signature IS the DAG" DX which is neograph's core differentiator.

---

## What Ships in v0.1.0-ts

### Phase 1: Core + Transformer (weeks 1-5)
- Node, Construct, Modifiers (all 4), Tool, .pipe()
- **@neograph/transform**: compiler transformer for signature extraction
- **@neograph/unplugin**: Vite/Webpack/Bun adapter
- compile() -> LangGraph.js StateGraph
- State model generation via Annotation.Root
- Factory dispatch (scripted + LLM modes)
- run() + config injection
- Renderers

### Phase 2: Features + DX (weeks 6-9)
- node() wrapper + auto-wiring via transformer metadata
- DI system (FromInput<T>/FromConfig<T> branded types)
- construct_from_functions equivalent
- LLM integration (structured/json/text output strategies)
- ReAct tool loop + budget tracking
- Full validation suite
- Spec loader (YAML/JSON)
- Lint + CLI + error-feedback retry

### Phase 3: Advanced (weeks 10-14)
- ForwardConstruct with JS Proxy + fluent branch API
- Oracle merge_fn with state params
- Dev-mode warnings
- ts-morph fallback codegen (`neograph generate` for tsgo future)
- Test suite (port fixture-based safety net)

### Not in v0.1.0-ts
- construct_from_module (no module introspection in TS; use explicit lists)
- Dead-body AST warning (ESLint plugin later)
- Frame stack walking (transformer eliminates the need)

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
