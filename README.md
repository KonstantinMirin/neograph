# NeoGraph

Declarative LLM graph compiler. Declare nodes with `@node`, assemble from function signatures, compile to LangGraph.

## Quickstart

```python
import sys
from neograph import node, construct_from_module, compile, run

@node(output=Claims, prompt='rw/decompose', model='reason')
def decompose(topic: RawText) -> Claims: ...

@node(output=Classified, prompt='rw/classify', model='fast')
def classify(decompose: Claims) -> Classified: ...

@node(mode='gather', output=MatchResult, prompt='match/explore',
      tools=[search], each=Each(over='clusters', key='label'))
def verify(classify: Classified) -> MatchResult: ...

pipeline = construct_from_module(sys.modules[__name__])
graph = compile(pipeline)
result = run(graph, input={'node_id': 'doc-001'})
```

Dependencies are inferred from parameter names: `classify(decompose: Claims)`
means *classify* depends on *decompose*. No explicit wiring needed.

## Modifiers

Pass modifiers as `@node` kwargs:

| Modifier | Purpose | Example kwarg |
|----------|---------|---------------|
| `Oracle` | N-way ensemble + judge-merge | `oracle=Oracle(n=3, merge_prompt='rw/merge')` |
| `Each` | Fan-out over a collection | `each=Each(over='clusters', key='label')` |
| `Operator` | Human-in-the-loop interrupt | `operator=Operator(when='has_failures')` |

## Examples

See [`examples/`](examples/) for runnable pipelines covering every feature.

## Advanced Use

For IR-level tests, programmatic construction from config, or sub-constructs,
use `Node` and `Construct` directly:

```python
from neograph import Node, Construct, compile, run

decompose = Node("decompose", mode="produce", output=Claims, prompt="rw/decompose")
classify = Node("classify", mode="produce", output=Classified, prompt="rw/classify")
pipeline = Construct("my-pipeline", nodes=[decompose, classify])
```

## Vocabulary

| Term | What it is |
|------|------------|
| **Node** | Typed processing block (mode: produce / gather / execute / scripted) |
| **Tool** | LLM-callable tool with per-tool budget |
| **Construct** | Ordered composition of Nodes (the blueprint) |
| **compile()** | Construct to executable LangGraph |
| **run()** | Execute with checkpointing |

## Install

```bash
pip install neograph
```

## License

MIT
