# NeoGraph

**A declarative LLM graph compiler.** Write Python functions. The framework builds the graph.

NeoGraph compiles typed node definitions into [LangGraph](https://github.com/langchain-ai/langgraph) state machines with automatic topology inference, assembly-time type validation, and structured error messages. You write Python; you get a production-grade execution graph with checkpointing, observability, and tool orchestration.

## Philosophy

**The graph is Python.** Not a DSL, not YAML, not `add_node`/`add_edge` calls. You express your pipeline in two ways — pick whichever fits the problem:

1. **`@node` decorator** — functions are nodes, parameter names are edges. The framework reads your function signatures and assembles the DAG. Rename a function, and downstream parameters break at import time. Fan-in is free: `def report(claims, scores, verified)`.

2. **`ForwardConstruct`** — a class with a `forward()` method. Python `if`/`for`/`try` compile to conditional edges and fan-out. No string dispatch, no `add_conditional_edges`. The type checker sees everything.

Both compile to the same IR and the same LangGraph backend. They coexist in the same project — use `@node` for most pipelines, reach for `ForwardConstruct` when you need branching logic.

## Quickstart: `@node`

```python
import sys
from neograph import node, construct_from_module, compile, run

@node(output=Claims, prompt='rw/decompose', model='reason')
def decompose(topic: RawText) -> Claims: ...

@node(output=Classified, prompt='rw/classify', model='fast')
def classify(decompose: Claims) -> Classified: ...

pipeline = construct_from_module(sys.modules[__name__])
graph = compile(pipeline)
result = run(graph, input={'node_id': 'doc-001'})
```

`classify(decompose: Claims)` — the parameter name `decompose` matches the upstream function. No `nodes=[...]` list, no ordering, no wiring. Mode is inferred: `prompt=` present means LLM call; absent means the function body runs.

## Quickstart: `ForwardConstruct`

```python
from neograph import ForwardConstruct, Node, compile, run

class Analysis(ForwardConstruct):
    check    = Node(output=CheckResult, prompt='check', model='fast')
    deep     = Node(output=DeepResult, prompt='deep', model='reason')
    shallow  = Node(output=ShallowResult, prompt='shallow', model='fast')

    def forward(self, topic):
        checked = self.check(topic)
        if checked.score > 0.7:          # compiles to conditional edge
            return self.deep(checked)
        else:
            return self.shallow(checked)

graph = compile(Analysis())
result = run(graph, input={'node_id': 'analysis-001'})
```

The `if` is real Python. At compile time, NeoGraph traces `forward()` with symbolic proxies (torch.fx-style) and emits the branch as a conditional edge with a runtime router. At execution time, the LangGraph graph runs — `forward()` doesn't re-run. For testing, call `Analysis().forward(topic)` directly with fakes.

## Modifier kwargs

Modifiers are `@node` keyword arguments — no pipe `|` syntax needed:

| Pattern | `@node` kwarg | What it does |
|---------|--------------|--------------|
| Fan-out over a collection | `map_over='upstream.items', map_key='label'` | Runs the node once per item; results collected as `dict[str, output]` |
| N-way ensemble + merge | `ensemble_n=3, merge_fn='combine'` | N parallel generators, then a scripted or LLM merge step |
| Human-in-the-loop | `interrupt_when=lambda state: {...} if bad else None` | Pauses the graph; resume with `run(graph, resume={...})` |

```python
@node(output=MatchResult, map_over='clusters.groups', map_key='label')
def verify(cluster: ClusterGroup) -> MatchResult: ...

@node(output=Claims, prompt='rw/decompose', model='reason',
      ensemble_n=3, merge_fn='merge_claims')
def decompose() -> Claims: ...

@node(output=ValidationResult, interrupt_when=lambda s: (
    {'issues': s.validate.issues} if not s.validate.passed else None
))
def validate(claims: Claims) -> ValidationResult: ...
```

## Non-node parameters

Not every parameter is an upstream node. NeoGraph supports three non-node parameter types:

```python
from neograph import node, FromInput, FromConfig

@node(output=Report)
def summarize(
    claims: Claims,                        # upstream @node
    topic: FromInput[str],                 # from run(input={'topic': '...'})
    rate_limiter: FromConfig[RateLimiter], # from config['configurable']
    max_items: int = 10,                   # compile-time constant
) -> Report: ...
```

## Organizing large pipelines

### Module-per-pipeline

Each pipeline lives in its own module. `construct_from_module` walks the module's `@node` functions:

```
pipelines/
    ingestion.py      # @node functions → construct_from_module(sys.modules[__name__])
    analysis.py       # separate module, separate pipeline
    reporting.py
```

### Cross-module composition

Import `@node` functions from other modules — they're just Python symbols:

```python
from pipelines.ingestion import extract, normalize
from neograph import node, construct_from_module

@node(output=Report)
def analyze(normalize: NormalizedData) -> Report: ...

pipeline = construct_from_module(sys.modules[__name__])
# Finds extract, normalize (imported), and analyze (local).
```

### Sub-constructs for isolation boundaries

When a sub-pipeline needs its own state boundary (isolated state, typed I/O contract):

```python
from neograph import Construct, Node

enrich = Construct(
    "enrich",
    input=Claims,
    output=ScoredClaims,
    nodes=[lookup, verify, score],
)
```

Sub-constructs get their own compiled subgraph. The parent pipeline wires input/output at the boundary.

### ForwardConstruct for complex branching

When the pipeline has non-trivial control flow (retries, conditional paths, fallbacks):

```python
class QualityGate(ForwardConstruct):
    validate = Node(output=ValidationResult, prompt='validate', model='fast')
    fix      = Node(output=FixedClaims, prompt='fix', model='reason')
    accept   = Node.scripted("accept", fn="accept_fn", output=AcceptedClaims)

    def forward(self, claims):
        result = self.validate(claims)
        if result.passed:
            return self.accept(claims)
        else:
            fixed = self.fix(claims)
            return self.validate(fixed)  # re-check
```

## Modes

Every `@node` function operates in one of five modes:

| Mode | When | What happens |
|------|------|-------------|
| `scripted` | No `prompt=`/`model=` | Function body runs at execution time |
| `produce` | `prompt=` + `model=` present | Single LLM call, structured JSON output |
| `gather` | Same + `tools=[...]` | ReAct tool loop (read-only exploration) |
| `execute` | Same + `tools=[...]` (mutations) | ReAct tool loop (side effects allowed) |
| `raw` | `mode='raw'` explicit | Full LangGraph `(state, config) -> dict` escape hatch |

Mode is inferred from kwargs unless set explicitly. If you pass `prompt=` and `model=`, mode is `produce`. If you pass neither, mode is `scripted` and your function body runs.

## Observability

Every node execution emits structured logs via [structlog](https://www.structlog.org/):

```
node_start   node=decompose mode=produce output_type=Claims
node_complete node=decompose duration_s=1.2
```

For production tracing, pass a trace provider via `config['configurable']`:

```python
result = run(graph, input={...}, config={
    'configurable': {
        'node_id': 'analysis-001',
        'trace_provider': langfuse_tracer,
        'rate_limiter': my_limiter,
    }
})
```

Node functions access these via `FromConfig[T]` parameters. The observable_pipeline example shows Langfuse integration.

## Testing

### Unit testing individual nodes

```python
result = my_node.run_isolated(input=Claims(items=["test"]))
assert isinstance(result, ClassifiedClaims)
```

`run_isolated` bypasses `compile()`/`run()`. For LLM nodes, configure a fake first:

```python
from neograph import configure_llm
configure_llm(llm_factory=lambda tier: FakeLLM(), prompt_compiler=...)
result = decompose.run_isolated()
```

### End-to-end testing

```python
pipeline = construct_from_module(my_module)
graph = compile(pipeline)
result = run(graph, input={'node_id': 'test-001'})
assert isinstance(result['classify'], ClassifiedClaims)
```

### ForwardConstruct testing

Call `forward()` directly — it runs real Python with real values, not the traced graph:

```python
pipeline = MyPipeline()
result = pipeline.forward(Claims(items=["test"]))  # direct call, debuggable
```

## Assembly-time validation

NeoGraph catches type mismatches when you construct the pipeline, not at runtime:

```
ConstructError: Node 'verify' in construct 'pipeline' declares input=ClusterGroup
  but no upstream produces a compatible value.
  upstream producers:
    • node 'cluster': Clusters
  hint: did you forget to fan out? try .map(lambda s: s.cluster.groups, key='...')
  at my_pipeline.py:42
```

Validation runs at `Construct(nodes=[...])` time and at `construct_from_module()` time. Fan-in parameters are type-checked against their upstream outputs. `Each`-modified producers are tracked as `dict[str, X]`.

## Examples

See [`examples/`](examples/) for 13 runnable pipelines covering every feature, plus 5 LangGraph comparison scripts.

## Runtime graph construction

The `@node` decorator and `ForwardConstruct` are for humans writing pipelines in source code. But pipelines can also be assembled at runtime — by an LLM, a config file, or a routing layer. This is where the Node/Construct API and the `|` pipe syntax shine:

```python
from neograph import Node, Tool, Construct, Oracle, Each, Operator, compile, run

# An LLM emits this as structured output (tool call or config):
spec = {
    "nodes": [
        {"name": "decompose", "mode": "produce", "output": "Claims",
         "prompt": "rw/decompose", "model": "reason",
         "modifiers": [{"type": "Oracle", "n": 3, "merge_fn": "merge_claims"}]},
        {"name": "verify", "mode": "gather", "output": "MatchResult",
         "prompt": "match/verify", "model": "fast", "tools": ["search"],
         "modifiers": [{"type": "Each", "over": "decompose.items", "key": "label"}]},
        {"name": "report", "mode": "produce", "output": "Report",
         "prompt": "rw/report", "model": "fast"},
    ]
}

# Your runtime builds the graph from the spec:
nodes = []
for s in spec["nodes"]:
    n = Node(s["name"], mode=s["mode"], output=resolve_type(s["output"]),
             prompt=s.get("prompt"), model=s.get("model"),
             tools=[lookup_tool(t) for t in s.get("tools", [])])
    for mod in s.get("modifiers", []):
        if mod["type"] == "Oracle":
            n = n | Oracle(n=mod["n"], merge_fn=mod.get("merge_fn"))
        elif mod["type"] == "Each":
            n = n | Each(over=mod["over"], key=mod["key"])
        elif mod["type"] == "Operator":
            n = n | Operator(when=mod["when"])
    nodes.append(n)

pipeline = Construct("llm-defined-pipeline", nodes=nodes)
graph = compile(pipeline)
result = run(graph, input={"node_id": "dynamic-001"})
```

The `|` pipe syntax composes at runtime without modules, function signatures, or class definitions — the LLM just emits a JSON spec and the system builds, validates, compiles, and runs it. Assembly-time validation (`ConstructError`) catches malformed specs before execution starts.

This is the same IR that `@node` and `ForwardConstruct` compile to internally. Three surfaces, one compiler.

## Install

```bash
pip install neograph
```

## License

MIT
