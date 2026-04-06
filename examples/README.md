# NeoGraph Examples

All examples use `@node` as the primary construction API unless noted.
Examples marked "declarative" use `Node`/`Construct` directly — either because they
demonstrate sub-constructs (which require explicit I/O boundaries) or because they
exercise config-injection patterns not yet ported to `Annotated[T, FromConfig]`.

## Core examples

| # | File | What it demonstrates | API |
|---|------|---------------------|-----|
| 01 | `01_scripted_pipeline.py` | Basic 3-node scripted pipeline — the "hello world" | @node |
| 01c | `01c_decorator_fan_in.py` | Fan-in with diamond DAG — @node's killer feature over declarative | @node |
| 02 | `02_produce_and_gather.py` | LLM produce + gather modes with tool use | @node |
| 03 | `03_oracle_ensemble.py` | Oracle modifier — N parallel generators + scripted merge | @node |
| 04 | `04_each_fanout.py` | Each modifier — fan-out over a collection via `map_over=` | @node |
| 05 | `05_subgraph_composition.py` | Isolated sub-pipelines with typed I/O boundaries | declarative |
| 06 | `06_raw_node_escape_hatch.py` | `@node(mode='raw')` for full LangGraph control | @node |
| 07 | `07_llm_configuration.py` | Per-node model routing, temperature, and token budgets | @node |
| 08 | `08_structured_output_coercion.py` | Output strategies: default, json_mode, text-parse | @node |
| 09 | `09_operator_human_in_loop.py` | Human-in-the-loop interrupt + resume via `interrupt_when=` | @node |
| 10 | `10_full_pipeline.py` | Every feature in one scenario (Oracle + Each + Operator + sub-construct) | mixed |
| 11 | `11_pipeline_metadata_and_prompts.py` | Pipeline metadata + config injection via `config['configurable']` | declarative |
| 12 | `12_input_rendering.py` | Pluggable input rendering (XML, delimited, JSON) | @node |
| 13 | `13_gather_produce_subconstruct.py` | Agent+think inside a sub-construct with tool_log flow + Each fan-out | @node + sub-construct |
| 14 | `14_context_injection.py` | Verbatim context= in sub-constructs (catalog forwarded from parent) | @node + sub-construct |
| -- | `observable_pipeline.py` | Observable LLM pipeline with Langfuse tracing | @node |

### Why some examples stay declarative

- **Example 05** — sub-constructs use `Construct(input=X, output=Y, nodes=[...])` for isolation boundaries. Can also use `construct_from_functions(input=, output=)` with @node (see example 13).
- **Example 10** — the `enrich` sub-construct requires declarative `Construct(input=..., output=...)`. The top-level producer uses `@node`.
- **Example 11** — uses `config['configurable']` in scripted nodes. Migrating requires `Annotated[T, FromConfig]` annotation on the decorated function.

## `vs_langgraph/`

Side-by-side comparisons showing NeoGraph vs raw LangGraph for common patterns.
Each file has a NeoGraph version and a LangGraph version of the same pipeline.

| File | Pattern |
|------|---------|
| `01_sequential_pipeline.py` | Sequential pipeline |
| `02_tool_agent.py` | Tool-calling agent |
| `03_map_reduce.py` | Map-reduce fan-out |
| `04_human_in_the_loop.py` | Human-in-the-loop |
| `05_subgraph.py` | Subgraph composition |

## Running examples

Most examples run standalone:

```bash
python examples/01_scripted_pipeline.py
```

Examples that use LLM modes (02, 03, 07, 08, 10) require either:
- A configured LLM via `neograph.configure_llm()` (shown in each example), or
- The `--fake` flag if the example supports it (07)

Example 09 demonstrates interrupt/resume — it runs both phases automatically.
