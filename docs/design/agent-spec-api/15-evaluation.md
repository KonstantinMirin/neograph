# Agent Spec → neograph mapping: Evaluation

## Source

Oracle Agent Spec v26.1.2 Evaluation API: https://oracle.github.io/agent-spec/development/api/evaluation.html
How-to guide: https://oracle.github.io/agent-spec/development/howtoguides/howto_evaluation.html

Agent Spec Eval is an **extension** of Agent Spec that standardizes evaluation of agentic systems with Datasets, Metrics, Evaluators, and Aggregators. It is a **runtime/test harness**, not a Flow component.

---

## Class-by-class mapping

| Agent Spec class | What it does | LangGraph primitive | neograph concept | Ecosystem | Status |
|------------------|--------------|---------------------|------------------|-----------|--------|
| `Dataset` | Wrapper around `_DataSource`; holds samples with feature consistency controls | None (data container) | Not a core concept; user passes raw data to scripted nodes | Langfuse datasets, DeepEval `EvaluationDataset`, LangSmith datasets | **OUT OF SCOPE** — test harness data, not pipeline component |
| `Evaluator` | Orchestrates metrics over a dataset with concurrency control | None (test runner) | Not a core concept; eval is external to compiled graphs | Langfuse `score()`, LangSmith `evaluate()`, Promptfoo CLI | **OUT OF SCOPE** — runtime harness, not a Flow node |
| `EvaluationResults` | Container for metric results (sample_id × metric_name matrix) | None (result container) | Not a core concept; results are returned from test harness | Langfuse `Observation` scores, LangSmith evaluation runs | **OUT OF SCOPE** — output container, not a pipeline component |
| `Aggregator` (abstract) | Combine a collection of metric values into a single aggregate | None (reducer) | Not a core concept; users write reduction functions | DeepEval `MeanAggregator`, custom reducers | **OUT OF SCOPE** — eval-time only |
| `HarmonicMeanAggregator` | Harmonic mean of non-negative numeric values | None | Not a core concept | DeepEval has mean aggregators | **OUT OF SCOPE** — eval-time only |
| `MeanAggregator` | Arithmetic mean of numeric values | None | Not a core concept | DeepEval has mean aggregators | **OUT OF SCOPE** — eval-time only |
| `Intermediate` | Reusable intermediate values shared across metrics (e.g., embeddings) | None (shared computation) | Not a core concept | DeepEval `GEval`, custom intermediates | **OUT OF SCOPE** — eval-time only |
| `Metric` (base) | Base for implementing metrics and metric wrappers; `compute_metric()` returns (value, details) | None | Not a core concept; scripted `@node` can implement any metric | DeepEval `Metric` base, LangSmith evaluators, Langfuse metrics | **OUT OF SCOPE** — eval-time only |
| `LlmBasedMetric` | Metric via LLM invocation with `ask_llm()` helper | None | Not a core concept; a `@node(mode='think')` with an LLM can act as judge | DeepEval `LLMAsAJudge`, LangSmith LangChain evaluators, Langfuse LLM metrics | **OUT OF SCOPE** — eval-time only |
| `LlmAsAJudgeMetric` | LLM-as-judge with system prompt + user prompt template + regex value extraction | None | Not a core concept; `@node(mode='think')` with judge prompt | DeepEval `LLMAsAJudge`, LangSmith criteria evaluators | **OUT OF SCOPE** — eval-time only |

---

## Status legend used

| Status | Meaning |
|--------|---------|
| **OUT OF SCOPE** | Not a pipeline/graph component; runtime/test harness construct that does NOT serialize in Agent Spec Flow YAML. Import/export ignores these. |

---

## Serialization notes

**Critical:** Agent Spec Eval classes are **not Flow components**. They are part of the evaluation harness, NOT part of a serialized Flow.

Evidence from Agent Spec docs:
- "Agent Spec Eval is an extension of Agent Spec that standardizes how agentic systems are evaluated"
- Datasets, Metrics, Evaluators are **runtime constructs** used in test harnesses
- They do NOT appear in Flow YAML serialization
- The Flow serialization includes: Nodes (LlmNode, ToolNode, etc.), ControlFlowEdge, DataFlowEdge — **not** Dataset/Evaluator/Metric

**Therefore:**
- **Export (Construct → Agent Spec):** No lowering needed. Eval constructs never exist in a Construct — they're external test code. Export emits Flow components only; eval is ignored.
- **Import (Agent Spec → Construct):** No reconstruction needed. Eval classes aren't in the imported Flow. If a user wants to eval an imported Agent Spec, they use neograph-scripted `@node` functions + ecosystem eval harnesses (Langfuse/DeepEval/LangSmith) separately.
- **Wire format:** Eval is OUT OF SCOPE for the Agent Spec wire format. It lives in the runtime/test layer, not the pipeline definition layer.

---

## Export lowering

**No lowering required.** Eval is not part of a Construct. A user writes:

```python
# neograph eval pattern (separate from pipeline)
@node(scripted_fn="exact_match")
def exact_match(reference: str, response: str) -> float:
    return 1.0 if reference == response else 0.0

# Run eval externally via ecosystem tools
# langfuse.score(), deepeval.evaluate(), etc.
```

The pipeline itself (Construct → compiled LangGraph) does NOT carry eval metadata. Export serializes the pipeline to Agent Spec Flow YAML; eval harness code stays external.

---

## Import reconstruction

**No reconstruction needed.** When importing an Agent Spec Flow:
- Deserialize Flow → Construct (nodes, edges)
- Compile to LangGraph
- **Eval is separate:** user writes `@node` metric functions or uses ecosystem tools

No attempt is made to reconstruct Dataset/Evaluator/Metric from the Agent Spec Eval extension because those classes **do not exist in Flow YAML**. They're runtime-only constructs.

---

## Verdict for interop

**Fidelity impact:** None. Eval is orthogonal to pipeline definition. A Flow defined in Agent Spec has no eval metadata; eval runs as a separate concern using the runtime ecosystem (Langfuse, DeepEval, LangSmith). Round-trip fidelity for **pipelines** is unaffected.

**Single biggest risk:** None, because eval is explicitly out of scope for the wire format. The only "risk" is user confusion about where eval lives — documentation must clarify that eval is a **harness layer** concern, not a **pipeline definition** concern.

**In/out-of-scope call:** Eval is **OUT OF SCOPE** for the Agent Spec Flow wire format. It is a runtime/test harness extension, not a component of pipeline serialization. Import/export ignores it entirely.

**Expressibility:** Can neograph express eval-pipelines as Each+@node+Oracle? **YES**. The Agent Spec Eval pattern of "evaluate each dataset sample with metrics, then aggregate" maps cleanly to:
- `Each(over=dataset)` → fan-out over samples
- Per-sample `@node(scripted_fn="metric")` → compute metric value
- Oracle or merge_fn → aggregate results

But this is a **demonstrable pattern**, not a gap. The actual Agent Spec Eval classes remain runtime harness constructs that don't serialize.
