# NeoGraph

**Write Python. Get a production graph.**

**Docs & guides: [neograph.pro](https://neograph.pro)** — full documentation site with tutorials, API reference, and side-by-side LangGraph comparisons.

```bash
# uv (recommended)
uv add neograph

# pip
pip install neograph
```

Define your LLM pipeline as Python functions. The framework infers the topology, validates types at assembly time, and compiles to [LangGraph](https://github.com/langchain-ai/langgraph) with checkpointing, observability, and tool orchestration. No DSL. No YAML. No `add_node` / `add_edge`.

**A function is a node. A parameter name is an edge. An `if` is a branch.**

---

## Functions are nodes

```python
from neograph import node, construct_from_module, compile, run

@node(outputs=Claims, prompt='rw/decompose', model='reason')
def decompose(topic: RawText) -> Claims: ...

@node(outputs=Classified, prompt='rw/classify', model='fast')
def classify(decompose: Claims) -> Classified: ...

@node(outputs=Report)
def report(classify: Classified) -> Report:
    return Report(summary=f"{len(classify.items)} claims processed")

pipeline = construct_from_module(sys.modules[__name__])
graph = compile(pipeline)
result = run(graph, input={'node_id': 'doc-001'})
```

`classify(decompose: Claims)` — the parameter name IS the dependency. Rename a function, downstream breaks at import time. Fan-in is just more parameters: `def report(claims, scores, verified)`.

Mode is inferred. `prompt=` + `model=` means LLM call (`think` mode). Neither means the function body runs (`scripted` mode).

## `if` is a branch

```python
from neograph import ForwardConstruct, Node, compile

class Analysis(ForwardConstruct):
    check   = Node(outputs=CheckResult, prompt='check', model='fast')
    deep    = Node(outputs=Result, prompt='deep-analysis', model='reason')
    shallow = Node(outputs=Result, prompt='quick-scan', model='fast')

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
@node(outputs=MatchResult, map_over='clusters.groups', map_key='label')
def verify(cluster: ClusterGroup) -> MatchResult: ...

# N-way ensemble with merge
@node(outputs=Claims, prompt='decompose', model='reason',
      ensemble_n=3, merge_fn='merge_claims')
def decompose() -> Claims: ...

# Human-in-the-loop interrupt
@node(outputs=ValidationResult,
      interrupt_when=lambda s: {'issues': s.validate.issues} if not s.validate.passed else None)
def validate(claims: Claims) -> ValidationResult: ...

# Agent with tools — typed tool results preserved
@node(outputs={"result": ExplorationResult, "tool_log": list[ToolInteraction]},
      mode='agent', model='research', prompt='explore',
      tools=[Tool("search", budget=5)],
      context=["catalog"])     # verbatim state injection
def explore(claim: VerifyClaim) -> ExplorationResult: ...

# Non-node parameters: runtime input, config, constants
from typing import Annotated
from neograph import FromInput, FromConfig

@node(outputs=Report)
def summarize(
    claims: Claims,                                   # upstream node
    topic: Annotated[str, FromInput],                 # from run(input={...})
    rate_limiter: Annotated[RateLimiter, FromConfig], # from config
    max_items: int = 10,                              # constant
) -> Report: ...
```

## Catches mistakes before you run

```
ConstructError: Node 'verify' declares inputs=ClusterGroup but no upstream
  produces a compatible value.
  upstream producers:
    • node 'cluster': Clusters
  hint: did you forget to fan out? try .map(lambda s: s.cluster.groups, key='...')
  at my_pipeline.py:42
```

Types are validated at assembly time — when you define the pipeline, not when you execute it.

## Scales to real systems

**Organize by module.** Each pipeline is a Python module. Import nodes across modules. `construct_from_module` finds them all.

**Sub-constructs from @node functions.** `construct_from_functions("verify", [explore, score], input=Claim, output=Result)` builds a sub-construct with port param resolution. Mix @node functions and sub-constructs in one `construct_from_functions` call.

**Observe everything.** Structured logs on every node. Pass trace providers and shared resources via `Annotated[T, FromConfig]`.

**Retry on failure.** `compile(pipeline, retry_policy=RetryPolicy(max_attempts=3))` retries LLM nodes on malformed JSON, validation errors, and transient API failures.

**Test at every level.** `node.run_isolated()` for unit tests. `compile()` + `run()` for integration. `forward()` direct-call for debugging.

## LLMs can build the graph too

For runtime construction — an LLM emitting a pipeline via tool calls, a config system defining workflows — use the programmatic API with the `|` pipe syntax:

```python
from neograph import Node, Construct, Oracle, Each, compile, run

decompose = Node("decompose", mode="think", outputs=Claims,
                 prompt="rw/decompose", model="reason") | Oracle(n=3, merge_fn="merge")
verify = Node("verify", mode="agent", outputs=MatchResult,
              prompt="verify", model="fast") | Each(over="decompose.items", key="label")

pipeline = Construct("dynamic", nodes=[decompose, verify])
graph = compile(pipeline)
```

Three surfaces — `@node`, `ForwardConstruct`, `Node | Modifier` — one compiler.

## Documentation

Full documentation is at **[neograph.pro](https://neograph.pro)**:

- [Quick Start](https://neograph.pro/getting-started/quick-start/) — install, configure, build a pipeline, run it
- [The @node API](https://neograph.pro/node-api/functions-as-nodes/) — functions as nodes, modifier kwargs, FromInput/FromConfig, organizing pipelines
- [ForwardConstruct](https://neograph.pro/forward/control-flow/) — class-based pipelines with Python `if`/`for`/`try`
- [Runtime Construction](https://neograph.pro/runtime/programmatic/) — LLM-driven pipeline assembly, programmatic API
- [vs LangGraph](https://neograph.pro/comparison/overview/) — side-by-side for five common patterns
- [API Reference](https://neograph.pro/reference/api/)

## Examples

16 runnable examples in [`examples/`](examples/) and 5 side-by-side comparisons in [`examples/vs_langgraph/`](examples/vs_langgraph/). Each example is narrated on [neograph.pro](https://neograph.pro/walkthrough/scripted-pipeline/) as a walkthrough.

## License

Code: MIT

Documentation content &copy; 2025-2026 Constantine Mirin, [mirin.pro](https://mirin.pro). Licensed under [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/).
