# NeoGraph

Declarative LLM graph compiler. Define typed Nodes, compose into Constructs, compile to LangGraph.

```python
from neograph import Node, Tool, Construct, Oracle, Replicate, Operator, compile, run

search = Tool("search_nodes", budget=5)

classify = Node(
    "classify",
    mode="produce",
    input=DecompositionResult,
    output=ClassificationResult,
    model="reason",
    prompt="rw/classify",
)

decompose = Node(
    "decompose",
    mode="produce",
    output=DecompositionResult,
    model="reason",
    prompt="rw/decompose",
) | Oracle(n=3, merge_prompt="rw/merge")

match_verify = Node(
    "match-verify",
    mode="gather",
    output=ClusterMatchResult,
    model="reason",
    prompt="match-verify/explore",
    tools=[search],
) | Replicate(over="clusters.clusters", key="label")

validate = Node(
    "validate",
    mode="produce",
    output=ValidationResult,
    prompt="validate",
) | Operator(when="has_failures")

pipeline = Construct("my-pipeline", nodes=[decompose, classify, match_verify, validate])
graph = compile(pipeline)
result = run(graph, input={"node_id": "BR-001", "project_root": "."})
```

## Vocabulary

| Term | What it is |
|---|---|
| **Node** | Typed processing block (mode: produce / gather / execute) |
| **Tool** | LLM-callable tool with per-tool budget |
| **Construct** | Ordered composition of Nodes (the blueprint) |
| **Oracle** | Modifier: N-way ensemble + judge-merge |
| **Replicate** | Modifier: fan-out over collection |
| **Operator** | Modifier: human-in-the-loop interrupt |
| **compile()** | Construct to executable LangGraph |
| **run()** | Execute with checkpointing |

## Install

```bash
pip install neograph
```

## License

MIT
