# NeoGraph

**Write Python. Get a production graph.**

```bash
pip install neograph
```

Define your LLM pipeline as Python functions. The framework infers the topology, validates types at assembly time, and compiles to [LangGraph](https://github.com/langchain-ai/langgraph) with checkpointing, observability, and tool orchestration. No DSL. No YAML. No `add_node` / `add_edge`.

**A function is a node. A parameter name is an edge. An `if` is a branch.**

---

## Functions are nodes

```python
from neograph import node, construct_from_module, compile, run

@node(output=Claims, prompt='rw/decompose', model='reason')
def decompose(topic: RawText) -> Claims: ...

@node(output=Classified, prompt='rw/classify', model='fast')
def classify(decompose: Claims) -> Classified: ...

@node(output=Report)
def report(classify: Classified) -> Report:
    return Report(summary=f"{len(classify.items)} claims processed")

pipeline = construct_from_module(sys.modules[__name__])
graph = compile(pipeline)
result = run(graph, input={'node_id': 'doc-001'})
```

`classify(decompose: Claims)` — the parameter name IS the dependency. Rename a function, downstream breaks at import time. Fan-in is just more parameters: `def report(claims, scores, verified)`.

Mode is inferred. `prompt=` + `model=` means LLM call. Neither means the function body runs.

## `if` is a branch

```python
from neograph import ForwardConstruct, Node, compile

class Analysis(ForwardConstruct):
    check   = Node(output=CheckResult, prompt='check', model='fast')
    deep    = Node(output=Result, prompt='deep-analysis', model='reason')
    shallow = Node(output=Result, prompt='quick-scan', model='fast')

    def forward(self, topic):
        checked = self.check(topic)
        if checked.confidence > 0.8:
            return self.shallow(checked)
        else:
            return self.deep(checked)

graph = compile(Analysis())
```

The `if` compiles to a conditional edge. `for` compiles to fan-out. Python is the graph language. Your type checker sees everything. Your debugger works.

## Everything else is a keyword

```python
# Fan-out over a collection
@node(output=MatchResult, map_over='clusters.groups', map_key='label')
def verify(cluster: ClusterGroup) -> MatchResult: ...

# N-way ensemble with merge
@node(output=Claims, prompt='decompose', model='reason',
      ensemble_n=3, merge_fn='merge_claims')
def decompose() -> Claims: ...

# Human-in-the-loop interrupt
@node(output=ValidationResult,
      interrupt_when=lambda s: {'issues': s.validate.issues} if not s.validate.passed else None)
def validate(claims: Claims) -> ValidationResult: ...

# Non-node parameters: runtime input, config, constants
@node(output=Report)
def summarize(
    claims: Claims,                        # upstream node
    topic: FromInput[str],                 # from run(input={...})
    rate_limiter: FromConfig[RateLimiter], # from config
    max_items: int = 10,                   # constant
) -> Report: ...
```

## Catches mistakes before you run

```
ConstructError: Node 'verify' declares input=ClusterGroup but no upstream
  produces a compatible value.
  upstream producers:
    • node 'cluster': Clusters
  hint: did you forget to fan out? try .map(lambda s: s.cluster.groups, key='...')
  at my_pipeline.py:42
```

Types are validated at assembly time — when you define the pipeline, not when you execute it.

## Scales to real systems

**Organize by module.** Each pipeline is a Python module. Import nodes across modules. `construct_from_module` finds them all.

**Isolate with sub-constructs.** Typed I/O boundaries for sub-pipelines: `Construct("enrich", input=Claims, output=ScoredClaims, nodes=[...])`.

**Observe everything.** Structured logs on every node. Pass trace providers and shared resources via `FromConfig[T]`.

**Test at every level.** `node.run_isolated()` for unit tests. `compile()` + `run()` for integration. `forward()` direct-call for debugging.

## LLMs can build the graph too

For runtime construction — an LLM emitting a pipeline via tool calls, a config system defining workflows — use the programmatic API with the `|` pipe syntax:

```python
from neograph import Node, Construct, Oracle, Each, compile, run

decompose = Node("decompose", mode="produce", output=Claims,
                 prompt="rw/decompose", model="reason") | Oracle(n=3, merge_fn="merge")
verify = Node("verify", mode="gather", output=MatchResult,
              prompt="verify", model="fast") | Each(over="decompose.items", key="label")

pipeline = Construct("dynamic", nodes=[decompose, verify])
graph = compile(pipeline)
```

Three surfaces — `@node`, `ForwardConstruct`, `Node | Modifier` — one compiler.

## Examples

See [`examples/`](examples/) for runnable pipelines and [`examples/vs_langgraph/`](examples/vs_langgraph/) for side-by-side comparisons.

## License

MIT
