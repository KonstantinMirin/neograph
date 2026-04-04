# NeoGraph DX Research: What to Steal From the Rest of the World

**Date:** 2026-04-04
**Scope:** API/DX improvements inspired by other workflow/graph frameworks

## Core insight

Every framework that is pleasant to use has **one** of these two properties:

1. **The graph is inferred from Python semantics** (Dagster param names, DSPy forward-tracing, Keras callable layers, LlamaIndex type dispatch).
2. **The graph is a first-class data structure you can manipulate** (sklearn Pipeline, NeoGraph's current Construct).

Frameworks that are unpleasant are the ones stuck in the middle — where the graph is built imperatively with side-effecting calls (`graph.add_node`, `pipeline.connect`) that happen to produce a data structure but lose type information along the way.

**NeoGraph is already in category 2.** The biggest wins come from layering category 1 on top — not replacing.

## Tier S — These changed how I think about the API

### 1. Dagster `@asset` — parameter-name dependency inference

```python
@dg.asset
def daily_sales() -> DailySales: ...

@dg.asset
def weekly_sales(daily_sales: DailySales) -> WeeklySales: ...
```

**Insight:** The function signature IS the dependency graph. No `add_node`, no `add_edge`, no `nodes=[...]` list, no string names. Pyright/Pylance give red squiggles the moment you type a parameter name that doesn't match an upstream asset.

**Wins:**
- Eliminates the ordered list (topology computed from signatures)
- Eliminates string coupling between stages
- Makes refactors safe: rename the function → upstream refs break at import time
- Enables fan-in naturally: `def summarize(claims, scores, verified)`

### 2. LlamaIndex Workflows — type-dispatched event routing

```python
class Pipeline(Workflow):
    @step
    async def decompose(self, ev: StartEvent) -> ClaimsEvent: ...
    @step
    async def classify(self, ev: ClaimsEvent) -> ClassifiedEvent: ...
```

**Insight:** Same as Dagster but routed through *types*, not names. A step fires when an event of its input type is emitted. The dispatch mechanism is the type system.

NeoGraph's typed Pydantic `input`/`output` annotations are already *more* precise than LlamaIndex's Event classes. We're not using them for routing yet. We should.

### 3. DSPy `Module.forward()` — graphs ARE Python

```python
class RAG(dspy.Module):
    def __init__(self):
        self.query_gen = dspy.ChainOfThought("question -> query")
        self.answer_gen = dspy.ChainOfThought("question, context -> answer")
    def forward(self, question):
        q = self.query_gen(question=question).query
        ctx = search(q)
        return self.answer_gen(question=question, context=ctx)
```

**Insight:** Control flow is just Python. Loops are `for`, conditionals are `if`, error handling is `try`. No DSL for branching, no `add_conditional_edges`, no `Operator(when="...")` strings. DSPy still compiles/optimizes modules by walking `forward` via tracing.

This is where LangGraph got it wrong and where we can eat their lunch.

### 4. sklearn `Pipeline` — the original elegant composition

```python
pipe = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression())
```

**Insight:** A pipeline whose stages share a single protocol composes trivially.
- One constructor, positional args, no wiring
- Every step implements the same protocol (`fit` / `transform`)
- `pipe['pca']` indexing for introspection
- `pipe.set_params(logisticregression__C=0.5)` dunder param injection

## Tier A — Strong specific ideas

### 5. Keras Functional API — "layer is a function on tensors"

```python
inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(10)(x)
model = Model(inputs, outputs)
```

**Insight:** Two calls per layer: `Dense(64)` creates the layer (config), `(x)` applies it to data (wiring). The code literally looks like the data-flow diagram.

### 6. Prefect `.map()` / Airflow `.expand()` — first-class fan-out as a method

```python
# Prefect
results = verify_cluster.map(clusters)
# Airflow
verify_cluster.partial(config=cfg).expand(cluster=clusters)
```

Current `Each(over="cluster.groups", key="label")` is a **string-pathed access to state**, fragile and invisible to type checkers. `.map()` as a method on the task itself, with iterables as first-class expressions, is strictly better.

### 7. Pydantic AI `@agent.tool` — tools as decorators with introspected signatures

```python
agent = Agent('openai:gpt-4', deps_type=Database)

@agent.tool
def search_db(ctx: RunContext[Database], query: str) -> list[str]:
    """Search the database for matching records."""
    return ctx.deps.search(query)
```

Schema from type hints. Description from docstring. Dependencies via `RunContext[T]`. Zero metadata duplication. NeoGraph's current `Tool("search", budget=5)` + `register_tool_factory("search_code", ...)` has the same string-lookup disease.

### 8. Metaflow `FlowSpec` class + `self.next()`

```python
class TrainFlow(FlowSpec):
    @step
    def start(self): self.next(self.train, foreach='models')
    @step
    def train(self): self.next(self.join)
```

The value isn't `self.next(...)` — it's that a flow is a class, steps are methods, `self` is first-class shared state. You get docstrings on steps, inheritance, IDE jump-to-definition, testability.

### 9. Haystack `Pipeline.connect()` — the cautionary tale

```python
pipeline.connect("welcome_text_generator.welcome_text", "splitter.text")
```

GitHub issue #8862 is full of complaints that type mismatches crash pipelines with errors discovered at `connect()` time, not at import time, not in IDE. This is what `Each(over="cluster.groups")` will feel like when users have 40-node constructs.

**Lesson:** any string that references another part of the graph is a bug that hasn't been filed yet.

## Tier B — Minor but real wins

### 10. Effect-TS `Effect.gen` vs `pipe` — two modes, deliberate split

Effect explicitly tells users: use `pipe()` for linear transformations, use `Effect.gen` (generator functions) for business logic with conditionals. They don't pretend one idiom fits everything. NeoGraph's `|` pipe is great for single-node modifiers but falls apart the instant there's branching.

### 11. Rust typestate builder — compile-time invariant enforcement

```rust
let req = Request::new()
    .url("https://x")      // returns Builder<UrlSet>
    .method(Post)          // returns Builder<UrlSet, MethodSet>
    .build();              // only exists on Builder<UrlSet, MethodSet>
```

In Python: `typing.overload` + `Literal` + `Protocol` can get 80% of the way. `Node(mode="produce", output=...)` currently accepts half-built nodes silently; a few `@overload`s would make "produce without output" unrepresentable.

### 12. CrewAI `Process.sequential | hierarchical | parallel` — named topologies

Topology has a name. Currently Construct is implicitly `sequential`. Named `process=` parameter is cleaner than overloading `nodes=[...]` semantics.

### 13. RxJS / Apache Beam `|` — variadic pipe reads left-to-right

```python
(pipeline
  | 'ReadLines' >> ReadFromText('input.txt')
  | 'Split'     >> beam.FlatMap(split_words)
  | 'Count'     >> beam.combiners.Count.PerElement())
```

Beam's `'Label' >> transform` idiom: every step has a human-readable label inline. Huge for error messages.

### 14. Pulumi `Input[T]` / `Output[T]` — typed lazy values

A graph node's output isn't a value, it's a future value. Marking it with `Output[T]` at the type level forces `.apply()` when chaining. Makes `Node(...)(...)` call style type-safe: `claims: Output[Claims] = decompose()`.

## Tier C — Confirmed already doing it right

- Airflow TaskFlow `@task` — fine but XCom quirk we don't have
- LangGraph — pain points we know
- XState — state machines are wrong abstraction for LLM pipelines
- AWS CDK / Pulumi Components — tree-with-scope over-engineered for our domain
- Terraform HCL — not a Python lesson
- Haskell monads / F# computation expressions — Python syntax can't

## Ranked Top 10 DX Improvements

**#1. Parameter-name dependency inference via `@node` decorators**
- Source: Dagster `@asset`, LlamaIndex `@step`
- Effort: Large
- Breaks API: No (additive)

**#2. `.map()` method replacing `Each(over=..., key=...)` string path**
- Source: Prefect `.map()`, Airflow `.expand()`
- Effort: Medium
- Breaks API: No (additive)

**#3. `@neograph.tool` decorator with signature-inferred schemas**
- Source: Pydantic AI `@agent.tool`
- Effort: Small
- Breaks API: No

**#4. Compile-time type errors at Construct assembly, not at compile() call**
- Source: Haystack cautionary tale, Dagster static analysis
- Effort: Small-medium
- Breaks API: No (catches bugs earlier)

**#5. Callable Node API for fan-in DAGs: `node2(node1(...))`**
- Source: Keras Functional API, Pulumi Output[T]
- Effort: Medium
- Breaks API: No (additive)

**#6. `Construct.forward()` subclass mode for branching logic**
- Source: DSPy Module.forward, PyTorch nn.Module
- Effort: Large
- Breaks API: No (new class-based mode)

**#7. Typed `llm_config` via Literal-discriminated overloads**
- Source: Rust typestate builder
- Effort: Small-medium
- Breaks API: Mild (deprecation)

**#8. Inline error messages with source locations (Pydantic-style)**
- Source: Pydantic v2, Prefect, Dagster
- Effort: Medium
- Breaks API: No

**#9. First-class testing: `Node.run_isolated()` + `fake_llm` fixture**
- Source: Prefect `.fn`, Dagster direct-invocation
- Effort: Small
- Breaks API: No

**#10. Labeled step syntax à la Beam: `"decompose" >> Node.produce(...)`**
- Source: Apache Beam
- Effort: Small
- Breaks API: No

## Three things to NOT do

1. **Don't go YAML/config-first** like Kedro or CrewAI's YAML flavor
2. **Don't adopt Haystack's `connect("a.out", "b.in")`** string-pair wiring
3. **Don't follow LangGraph's `add_node`/`add_edge`** imperative builder

## The single most important insight

NeoGraph is already in "graph as first-class data structure" territory. Top move is to layer "graph inferred from Python semantics" on top via #1, #5, and #6. Don't have to pick — Dagster won by letting users choose between config-driven (`AssetsDefinition`) and code-driven (`@asset`), both compiling to the same IR. That's the roadmap.
