# NeoGraph Examples

All examples use `@node` as the primary construction API unless noted otherwise.

| # | File | What it demonstrates | API |
|---|------|---------------------|-----|
| 01 | `01_scripted_pipeline.py` | Basic 3-node scripted pipeline (no LLM) | declarative |
| 01b | `01b_scripted_pipeline_decorator.py` | Same as 01 rewritten with `@node` decorator | @node |
| 01c | `01c_decorator_fan_in.py` | Fan-in with diamond DAG — the decorator's killer feature | @node |
| 02 | `02_produce_and_gather.py` | LLM produce + gather modes with tool use | @node |
| 03 | `03_oracle_ensemble.py` | Oracle modifier — N parallel generators + merge | @node |
| 04 | `04_each_fanout.py` | Each modifier — fan-out over a collection | @node |
| 05 | `05_subgraph_composition.py` | Isolated sub-pipelines with modifiers | declarative |
| 06 | `06_raw_node_escape_hatch.py` | `@node(mode='raw')` for custom LangGraph logic | @node |
| 07 | `07_llm_configuration.py` | Per-node model routing, temperature, and budgets | @node |
| 08 | `08_structured_output_coercion.py` | Output strategies for models that don't do JSON well | @node |
| 09 | `09_operator_human_in_loop.py` | Operator modifier — human-in-the-loop interrupt and resume | declarative |
| 10 | `10_full_pipeline.py` | Every NeoGraph feature in one realistic scenario | declarative |
| 11 | `11_pipeline_metadata_and_prompts.py` | Pipeline metadata + rich prompt compilation | declarative |
| -- | `observable_pipeline.py` | Observable LLM pipeline with Langfuse tracing | declarative |

### `vs_langgraph/`

Side-by-side comparisons of NeoGraph vs raw LangGraph for common patterns:

| File | Pattern |
|------|---------|
| `01_sequential_pipeline.py` | Sequential pipeline |
| `02_tool_agent.py` | Tool-calling agent |
| `03_map_reduce.py` | Map-reduce |
| `04_human_in_the_loop.py` | Human-in-the-loop |
| `05_subgraph.py` | Subgraph composition |
