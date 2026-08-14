# Hybrid Data Flow in Graph-Based Agents

**Deterministic carry, LLM reasoning, and scoped back-references — problem statement, worked examples, proposed architecture, and prior art**

Audience: neograph engineering team, Agent Spec interop workstream
Status: design proposal, for discussion
Date: August 2026

---

## 0. Executive summary

A node in an agentic graph almost never produces output that is *purely* LLM-authored. In practice its output is a mix:

- fields the model genuinely reasons about (a verdict, a summary, a classification), and
- fields that must arrive **verbatim** from a tool result or an upstream step (a source URL, a record ID, an exact excerpt, a timestamp).

Today, most frameworks — including neograph — weld the node's output type to the LLM's structured-output schema. The consequence is that carried fields must be **re-transcribed by the model**, which is unreliable, expensive, and destroys provenance.

We propose three changes:

1. **Split the two types.** The LLM's schema becomes a *compile-time projection* of the node's output type, not the type itself.
2. **Introduce three field kinds** — `Reasoned`, `Carried`, `Selected` — and a deterministic assembly pass that fills the non-reasoned ones after the model call.
3. **Address upstream data by scoped artifact name**, not by node identity, so back-references survive loops and branching.

Plus one supporting primitive: the **tool ledger**, which makes a ReAct node's tool calls addressable without exposing its turn structure.

Prior art check: every individual primitive exists somewhere in the ecosystem. Nothing combines them behind a compiler that can check them statically. **`Selected` appears to have no prior art at all.**

A significant interop consequence is documented in §5: **the scoped-addressing design does not round-trip through Agent Spec as it stands today.** This needs an explicit decision from the team before export work goes deep.

---

## 1. The problem

### 1.1 The conflation

In LangGraph, and in neograph today, a node that produces structured output typically looks like this:

```python
class Finding(BaseModel):
    verdict: str
    rationale: str
    source_url: str
    excerpt: str

# The same class is used for BOTH:
#   - what the node contributes to graph state
#   - what the LLM is asked to emit
finding = llm.structured(Finding, prompt=...)
return {"finding": finding}
```

Because one class serves both roles, **every field is an obligation on the model.** There is no way to say "the model authors these two; the runtime fills those two."

### 1.2 Why re-transcription is the wrong default

Four independent reasons, in rough order of severity:

**Correctness.** The tool call is deterministic. The type conversion is deterministic. Inserting a token generator between them makes a deterministic pipeline probabilistic. Observed failure modes when a model re-emits a URL: scheme normalization (`http` → `https`), dropped trailing slash, silently stripped tracking parameters, percent-encoding changes. For an excerpt: whitespace normalization, quiet truncation, paraphrase-instead-of-copy. None of these throw. They are silent corruption.

**Provenance.** Once a field has passed through the model, you can no longer assert it is byte-identical to its source. For anything audit-facing — and the troubleshooting agent is audit-facing — that is disqualifying.

**Cost and latency.** Carried payload is generated twice: once into the prompt, once out of the completion. On large tool results this dominates the node's token bill.

**Schema pressure.** Every carried field added to the output schema increases the model's chance of a malformed emission on the *reasoned* fields too. The reasoning surface degrades as the transcription surface grows.

### 1.3 Scope of the problem

This is not an edge case. In current neograph pipelines it is closer to the rule: extract-then-reason, merge-then-forward, and verify-against-source all have this shape. The team should assume roughly every non-trivial node needs it.

---

## 2. Worked examples

These are the five cases that motivated the design. Each shows the naive form, what breaks, and the intended form.

### Example 1 — Research node: extract, reason, carry the source

**Task.** Fetch a page, decide whether it supports a claim, and pass forward the verdict *plus* the exact source and the exact supporting excerpt.

**Tool available:**

```python
def fetch(url: str) -> FetchResult: ...

class FetchResult(BaseModel):
    url: str            # final URL after redirects
    body: str           # full page text
    fetched_at: datetime
```

**Naive form — everything through the model:**

```python
class Finding(BaseModel):
    verdict: Literal["supports", "contradicts", "unrelated"]
    rationale: str
    source_url: str     # ⚠ model re-types the URL
    excerpt: str        # ⚠ model re-types a span of body
    fetched_at: datetime  # ⚠ model re-formats a timestamp

finding = llm.structured(Finding, tools=[fetch], prompt=...)
```

**What breaks.** `source_url` comes back as the URL the model was *asked* to fetch, not the post-redirect URL the tool actually returned. `excerpt` comes back paraphrased about one time in twenty, and is not detectably wrong without re-reading the body. `fetched_at` comes back in whatever format the model felt like. And the entire `body` was paid for on input and partially paid for again on output.

**Intended form.** Note that three different mechanisms are needed here — this example is the reason all three field kinds exist:

```python
class Finding(BaseModel):
    # authored by the model
    verdict:   Reasoned[Literal["supports", "contradicts", "unrelated"]]
    rationale: Reasoned[str]

    # copied by the runtime, byte-identical, never seen as an output obligation
    source_url: Carried["tools.fetch[0].result.url"]
    fetched_at: Carried["tools.fetch[0].result.fetched_at"]

    # model CHOOSES which span; runtime MATERIALIZES the text
    excerpt: Selected["tools.fetch[0].result.body", by="span"]
```

**What the model actually sees as its output contract** (the compile-time projection):

```json
{
  "type": "object",
  "properties": {
    "verdict":   {"enum": ["supports", "contradicts", "unrelated"]},
    "rationale": {"type": "string"},
    "excerpt_span": {
      "type": "object",
      "properties": {"start": {"type": "integer"},
                     "end":   {"type": "integer"}}
    }
  },
  "required": ["verdict", "rationale", "excerpt_span"]
}
```

`source_url` and `fetched_at` are **absent from the schema entirely.** `excerpt` has been replaced by an index pair. After the call, the runtime performs assembly: copies the two carried fields, slices `body[start:end]` into `excerpt`, then validates the full `Finding`.

**Guarantee gained:** `finding.source_url == fetch_result.url` is now assertable as an equality, not a hope.

---

### Example 2 — Fan-in: three summarizers converging on a dedup node

**Task.** Three nodes each categorize a support ticket against a strict vocabulary and summarize it. A merge node removes duplicates by category and forwards the survivors unchanged.

```python
class Ticket(BaseModel):
    category: Literal["billing", "outage", "complaint", "feature_request"]
    summary:  str
    source_id: str
    raw_text:  str
```

Suppose the three nodes return:

| # | category  | source_id | summary                      |
|---|-----------|-----------|-------------------------------|
| 1 | complaint | T-1041    | Customer unhappy with SLA    |
| 2 | complaint | T-1041    | Repeated SLA breach reported |
| 3 | outage    | T-1042    | Region eu-west-1 unavailable |

**Naive form.** Feed all three to an LLM and ask for the deduplicated list. What comes back: merged/rewritten summaries the user never wrote, occasionally a `source_id` that does not exist, and no way to tell which of the two `complaint` records actually survived.

**Case 2a — the dedup rule is purely mechanical.** If "same category wins by first occurrence" is the whole rule, **there should be no LLM in this node at all.** It compiles to a reducer:

```python
@reducer
def dedup_by_category(items: list[Ticket]) -> list[Ticket]:
    seen, out = set(), []
    for t in items:
        if t.category not in seen:
            seen.add(t.category)
            out.append(t)
    return out
```

This is worth calling out separately because a meaningful fraction of "merge nodes" in existing pipelines are this, and are currently implemented with a model call for no reason.

**Case 2b — the dedup rule is semantic.** Suppose survivors must be chosen by judgement ("keep the record that better characterizes the incident"). Then the model decides *which*, never *what*:

```python
class MergedTickets(BaseModel):
    keep: Selected["inbound.tickets", by="source_id", cardinality="many"]
```

The model's projected schema is just:

```json
{"keep": {"type": "array",
          "items": {"enum": ["T-1041", "T-1041", "T-1042"]}}}
```

It returns `["T-1041", "T-1042"]`. The runtime materializes the two full `Ticket` records from the inbound set. `summary` and `raw_text` are untouched, byte-identical to what the upstream nodes produced.

**Validation property:** any key the model returns that is not in the offered set is a hard validation failure with a cheap, well-targeted retry ("`T-9999` is not one of the available IDs"). Compare this to detecting a subtly rewritten `summary`, which is not detectable at all.

---

### Example 3 — Back-reference: step 2 reasoning over step 1's evidence

**Task.** Step 1 researches and produces a verdict. Step 2 must critique that verdict, and to do so it needs the *evidence* step 1 used — which is not part of step 1's declared output.

**Three ways to handle it, only one of which is good:**

**(a) Widen step 1's output.** Add every possibly-useful field to `Finding`. State grows without bound; every consumer pays for every producer's speculation about what might be needed later. Rejected.

**(b) Reach into step 1 by node identity.**

```python
evidence = state["step_1"]["tools"]["fetch"]   # ⚠
```

This is what n8n-style tools do. It couples step 2 to step 1's *implementation* and to graph topology. Rename a tool, insert a node, wrap step 1 in a loop — step 2 breaks, usually silently. Rejected.

**(c) Declared export, referenced by contract name.**

```python
# step 1 publishes a named, typed artifact
@node(provides={"evidence": trace.tools.fetch})
def research(...) -> Finding: ...

# step 2 declares what it needs, by name and type
@node(requires={"evidence": list[FetchResult]})
def critique(evidence, ...) -> Critique: ...
```

Step 2 is now coupled to the **name and type of a published contract**, not to a producer or a position in the graph. Any node that provides `evidence: list[FetchResult]` can be substituted. Because `requires` is declarative and topology is known at compile time, the compiler can verify that `evidence` is provided on *every* path reaching `critique` — a class of bug that n8n and Agent Spec can only discover at runtime.

**This is the core reframe: back-referencing is not the anti-pattern. Referencing a node is.**

---

### Example 4 — ReAct nodes: the tool ledger

**The difficulty.** A ReAct node has five tools and runs until it decides to stop. You do not know in advance whether it took one turn or seven, which tools it hit, or how many times. Crucially, **turn count does not correlate with call count** — a single turn can legitimately issue five parallel `search` calls with different parameters.

**The insight.** A downstream consumer should never be able to observe the turn structure. Turn count is an artifact of the loop's stopping behavior; if it were observable, a prompt tweak would become a breaking change. What the consumer cares about is *which tool produced what*.

**The ledger.** Every node maintains an append-only, per-tool record:

```python
class ToolCall(BaseModel):
    ordinal: int          # 0-based, across the whole node invocation
    args:    dict         # arguments as actually sent
    result:  Any          # typed to the tool's declared return type
    started_at: datetime

# addressable as:
trace.tools.search   # -> list[ToolCall], length 0..n
trace.tools.fetch    # -> list[ToolCall], length 0..n
```

**Store calls, not results.** A result without its arguments is unprovenanced. If `search` was called five times with different queries, `tools.search[2].result` is meaningless to a verifier without `tools.search[2].args`.

**Length is genuinely 0..n.** A tool that was never called yields an empty list. The type system should force consumers to handle that rather than assert a call happened.

**No new machinery required.** The ledger is an append-only channel with list-monoid semantics — the same reducer mechanism the graph already has, with the runtime as the writer instead of user code. It inherits scope resolution for free (see Example 5).

**Selector vocabulary — keep it to three.** `all` (default), `first`, `last`. Anything conditional ("the search whose query mentioned the customer ID") is a pure Python predicate over the ledger, not a path expression. See §3.7.

**Do not expose the raw ledger across node boundaries.** Downstream code writing `step_1.tools.fetch` reintroduces exactly the implementation coupling rejected in Example 3. The ledger stays internal; nodes export from it by contract:

```python
@node(provides={"evidence": trace.tools.fetch})
```

---

### Example 5 — Loops and branches: why naive addressing fails

**Setup.** A refine loop: `draft → critique → refine → (loop back to critique | exit)`. It runs an unknown number of times.

**The ambiguity.** After the loop, what does `refine.summary` mean? Iteration 1's? The last one? All of them? Agent Spec's answer today is last-write-wins (§4.2). n8n's answer is effectively the same, and is the reason back-references break there. Neither is a semantics; both are an accident of execution order.

**The proposal — treat it as lexical scope.** Every artifact is keyed by:

```
(name, scope_path, iteration_index)
```

where `scope_path` is the loop/branch nesting path. Resolution walks the scope chain, nearest enclosing binding first — exactly like variable lookup in a programming language.

**Three selectors, and only three:**

| Selector             | Meaning                                              |
|-----------------------|-------------------------------------------------------|
| `latest`             | most recent binding in the nearest enclosing scope    |
| `all_in_scope`       | every binding in the current scope, ordered           |
| `from_enclosing(n)`  | skip `n` scope levels outward before resolving        |

**Worked resolution.** Nested case: a `map_over` across 3 documents, each running the refine loop twice.

```
map_over[0] / refine_loop[0] / summary
map_over[0] / refine_loop[1] / summary
map_over[1] / refine_loop[0] / summary
map_over[1] / refine_loop[1] / summary
map_over[2] / refine_loop[0] / summary
map_over[2] / refine_loop[1] / summary
```

From inside `map_over[1] / refine_loop[1]`:

- `summary@latest` → `map_over[1]/refine_loop[1]/summary`
- `summary@all_in_scope` → the two bindings under `map_over[1]` only
- `summary@from_enclosing(1)` → resolves in the `map_over[1]` scope, skipping the loop

The fan-in node after `map_over` uses `summary@all_in_scope` and gets exactly three values (one final summary per document), with no ambiguity and no manual index threading.

**Why this matters for the tool ledger.** `trace.tools.search` inside a node nested in `map_over` resolves within the current map scope automatically. Same key shape, same lookup, no special case.

---

## 3. Proposed architecture

### 3.1 Three field kinds

| Kind        | Who decides | Who writes the value | In the LLM schema?          |
|-------------|-------------|-----------------------|-------------------------------|
| `Reasoned`  | model       | model                 | yes, as-is                    |
| `Carried`   | author      | runtime               | **no — absent entirely**      |
| `Selected`  | model       | runtime               | yes, but as a key/index only  |

The unifying principle: **the model emits only things it authored — judgements, prose it wrote, and choices among a closed set. It never emits data it merely saw.**

### 3.2 Compile-time: schema projection

The compiler derives the LLM's structured-output contract from the node's output type:

1. Take the full output model.
2. **Drop** every `Carried` field.
3. **Replace** every `Selected` field with its key type — an enum over the offered set, or an index/span object.
4. Keep `Reasoned` fields unchanged.
5. Emit the result as the BAML/JSON-schema contract.

Also at compile time:

- Resolve every `Carried` path against the producing node's declared types. **A path that does not type-check is a build error, not a runtime error.**
- Verify every `requires` is satisfied on every path through the graph.

### 3.3 Runtime: the assembly pipeline

```
render → call → parse → assemble → validate
                          ↑
                     (new stage)
```

`assemble` fills `Carried` fields from their bound paths and expands `Selected` keys into materialized values. Then the *full* output model is validated.

**This belongs in node-runtime, not the engine.** Assembly is pure, cheap, and replayable — it passes the purity test already established for the node-internal layer. It requires no new LangGraph surface. Projection is compile-time emission, which is where the compiler already lives.

### 3.4 Scoped artifact store

Append-only, content-addressed, keyed by `(name, scope_path, iteration_index)`. Gives, as byproducts: cheap dedup, caching, replay, and a provenance graph for free.

### 3.5 provides / requires contracts

Nodes publish named typed artifacts and declare named typed needs. Coupling is to the contract, never to the producer or to graph position. Statically checkable.

### 3.6 Tool ledger

Per §2, Example 4. Runtime-written append-only channel, exported by contract.

### 3.7 Constraint: keep the binding language tiny

**This is the constraint most likely to be violated, and the one that will do the most damage if it is.**

Permitted in a `Carried` / `Selected` path:

- field access and array indexing (`tools.fetch[0].result.url`)
- one of three scope selectors (`latest`, `all_in_scope`, `from_enclosing(n)`)
- one of three ledger selectors (`all`, `first`, `last`)

**Not permitted:** conditionals, arithmetic, comparisons, string manipulation, filters, arbitrary expressions.

Anything requiring computation is a pure Python function over the resolved artifact. It is testable, debuggable, type-checked by the existing toolchain, and does not require us to build, document, and maintain a language. The moment a path expression contains an `if`, we have started building a bad DSL.

---

## 4. Prior art

### 4.1 Summary

| System | Mechanism | Carry? | Field-level? | Select? | Loop semantics |
|---|---|---|---|---|---|
| LangGraph | shared state + reducers, partial writes | implicit | no (channel-level) | no | reducers; no per-iteration addressing |
| Google ADK | `output_key` + `{key}` prompt templating | implicit | no | no | shared state; races on parallel |
| Agent Spec | explicit `DataFlowEdge` per property | **yes** | **yes** | no | **last-write-wins** |
| Mastra | `.map()` + `mapVariable({step, path})` | **yes** | **yes** | no | manual per-iteration wiring |
| Griptape | `off_prompt` → Task Memory reference | n/a | no | no | n/a |
| Anthropic code exec / Code Mode | tool results stay as sandbox variables | **yes** | **yes** | no | host language scoping |
| DSPy | Input/Output field split in signature | no | n/a | no | n/a |
| Airflow | XCom + `map_index` | yes | yes | no | indexed, with known defects |

### 4.2 Detail

**LangGraph.** Node signature is `State → Partial<State>`. When a node returns a partial update, the reducer is invoked per updated key with accumulated state on the left and the node's update on the right. The carry is therefore *implicit*: a node writes only the keys it authored, and everything else survives untouched. Limitation: granularity is the channel, not the field. There is no way to express "this field of this model is carried" — only "this channel is not written."

**Google ADK.** Same shape, more explicit. An agent's `output_key` writes its result to a named slot in shared session state; `{key}` placeholders in the instruction string are substituted from session state before the prompt is sent. Reads are templated in, writes are one named key out — so carried data never round-trips through the model. Weakness: parallel children share one session state and must use distinct keys to avoid races; the fan-in problem is pushed onto the user.

**Agent Spec (Oracle).** The most direct prior art for `Carried`. Separates `ControlFlowEdge` from `DataFlowEdge`, on the stated grounds that a component id alone cannot identify a data relationship — so a `DataFlowEdge` names `source_node` + `source_output` → `destination_node` + `destination_input`. That *is* field-level deterministic carry: wire a tool node's output directly to the consumer, bypassing the LLM. Also supports both models: setting `data_flow_connections` to `None` falls back to a shared name-based variable space, and the spec notes that the name-based scheme is always expressible as explicit edges while the reverse is not. There is additionally a "flow state" of values available in every node without an edge.

**Where Agent Spec fails, and it is exactly our case:** when multiple data outputs connect to the same input, the most recently executed node's value wins — behaviorally equivalent to a public variable that every connected node overwrites. That is last-write-wins, and it is the n8n failure mode formalized.

**Mastra.** The most developed implementation in code. Provides a dedicated deterministic `.map()` step between steps; `mapVariable({step, path})` extracts a named field from a specific step's output and renames it; `getStepResult(step)` and `getInitData()` give typed back-references. Notably the reference is to the **step instance**, not a string id — which eliminates a whole class of rename bugs and is worth copying. Mastra also retains a workflow-level `state` for sharing values without threading them through every step's schemas, i.e. even the explicit-edges camp keeps a namespace escape hatch. Loops are handled by wiring the previous iteration's value in explicitly each turn.

**Griptape.** The purest expression of "the model emits references, not data." With `off_prompt` set on a tool, its output is stored in Task Memory and only a reference is returned to the LLM. Motivated by output length blowing the token limit and degrading model precision, and by data that should not enter the prompt at all.

**Anthropic code execution with MCP / programmatic tool calling; Cloudflare Code Mode.** Same principle at the orchestration layer: the model writes a script, the script runs in a sandbox, tool results are processed by the script rather than consumed by the model, and the model sees only the final output. The chained call never round-trips the first tool's response through the model — faster, more reliable, fewer tokens, and no exposure of sensitive payload. Reported token reductions on real workflows are large (Anthropic reports a ~150k → ~2k token case).

**DSPy.** A signature is a tuple of input fields and output fields; the adapter renders inputs into the prompt and parses back only the declared output fields. This is the projection idea at the schema level — the closest conceptual match to §3.2. But there is no carry: an input field is a prompt input, not a field that reappears in the prediction.

**Airflow.** Closest thing to scope-indexed addressing. Dynamic task mapping assigns each mapped instance an index and the reduce task gathers XComs lazily rather than materializing thousands of outputs. However, there are reported defects where a downstream mapped task pulls the XCom belonging to a different `map_index` when not all upstream instances ran — index-based addressing without proper scope resolution. Instructive as a warning, not a model.

### 4.3 Where the gap is

- **`Selected` has no prior art.** No surveyed system offers "the model picks keys from a closed set, the runtime materializes the values." This is the piece that makes semantic dedup and span selection safe, and it is genuinely novel.
- **Nobody has real scope semantics for loops.** Every system is last-write-wins, manual rewiring, or flat indexing with known defects. Lexical scoping with a scope chain is the largest open gap in the ecosystem.
- **Nobody does static checking.** Because `requires` is declarative and topology is known at build time, we can verify reachability of every reference on every path. No surveyed system does this.

### 4.4 Reading list

- Open Agent Specification technical report — https://arxiv.org/abs/2510.04173
- Agent Spec language spec — https://oracle.github.io/agent-spec/
- LangGraph Graph API (state, reducers) — https://docs.langchain.com/oss/python/langgraph/graph-api
- ADK session state — https://google.github.io/adk-docs/sessions/state/
- Mastra input data mapping — https://mastra.ai/en/docs/workflows/input-data-mapping
- Griptape Task Memory / off-prompt — https://docs.griptape.ai/latest/griptape-framework/structures/task-memory/
- Anthropic, code execution with MCP — https://www.anthropic.com/engineering/code-execution-with-mcp
- Anthropic, advanced tool use — https://www.anthropic.com/engineering/advanced-tool-use
- Airflow dynamic task mapping — https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/dynamic-task-mapping.html
- DSPy signatures — https://dspy.ai/learn/programming/signatures/

---

## 5. Agent Spec interop: what this means for the export path

**This section is the action item for the interop workstream.**

### 5.1 Fidelity matrix

| neograph construct | Agent Spec equivalent | Round-trips? |
|---|---|---|
| `Reasoned` field | LlmNode output property | ✅ |
| `Carried` field | `DataFlowEdge` (source_output → destination_input) | ✅ |
| Deterministic reducer node | ToolNode / ServerTool | ✅ (as an opaque tool) |
| `map_over` | `MapNode` | ✅ |
| Shared-namespace mode | `data_flow_connections=None` | ✅ |
| `Selected` field | — | ❌ no equivalent |
| Scoped `(name, scope, iteration)` addressing | — collapses to last-write-wins | ❌ **lossy** |
| `provides` / `requires` contracts | — approximated by explicit edges | ⚠️ partial |
| Static reachability checking | — | ❌ |
| Tool ledger | — | ❌ |

**Import is fine. Export is lossy on exactly our differentiators.** Any graph using loops with back-references will export to a specification whose semantics differ from what it does in neograph. That is worse than not exporting, if it happens silently.

### 5.2 Two options — pick one deliberately

**Option 1 — accept lossy export.** Treat Agent Spec as a lowest-common-denominator interop format. Complex graphs are neograph-native. The exporter emits a documented fidelity warning, and refuses (loudly) to export graphs whose semantics it cannot preserve. Cheapest, honest, shippable now.

Concretely: the exporter needs a **conformance classifier** that partitions a graph into `exportable` / `exportable-with-degradation` / `not-exportable`, and CI that asserts the classification for every example in the repo.

**Option 2 — extend the spec.** Agent Spec has a serialization plugin mechanism for component types outside the standard set, and its published roadmap already includes memory, datastores, and MCP tools — so the spec is not frozen. A scoped-reference component and a selection node are plausible upstream proposals.

Slower, and it is committee work. But if the strategic framing is *neograph as a spec with per-engine bindings* rather than *neograph as a library*, then being the party that fixed Agent Spec's loop semantics is worth substantially more than being one more runtime adapter.

**These are not mutually exclusive** — Option 1 is the 2026 shipping path, Option 2 is the 2027 positioning play. But Option 1 should be built assuming Option 2 happens, i.e. the exporter's degradation points should be the exact places we would later propose extensions.

---

## 6. Open questions for the team

1. **Carried path syntax.** Annotation-based (`Carried["path"]`) versus a separate binding table on the node decorator. Annotations keep the binding co-located with the type, which is better ergonomics than Agent Spec's edge objects; but they are harder to generate programmatically from an imported Agent Spec graph. Do we need both surfaces?
2. **Selected cardinality.** Do we need `one` / `many` / `at_most_one` as distinct forms, or is a single form with a type-level list/optional sufficient?
3. **Ledger retention.** The ledger is append-only and can be large. Do we checkpoint it, or keep it in a side-store keyed by run id and reference it from checkpoints?
4. **Retry semantics on assembly failure.** If a `Selected` key is invalid, we retry the model with the invalid key surfaced. If a `Carried` path resolves to nothing (tool was never called), that is a graph-design error — should it be a compile error where statically provable, and a runtime error otherwise?
5. **Migration.** Existing script-only workaround nodes need a mechanical path to the new form. Is that a codemod or hand migration?
6. **Does the ledger export at all?** Recommendation is no, but a debug/trace export mode may be worth it for the troubleshooting agent specifically.

---

## 7. Glossary

- **Assembly** — the deterministic post-model stage that fills `Carried` fields and expands `Selected` keys.
- **Carried** — a field bound to an upstream path, copied byte-identically by the runtime, absent from the model's schema.
- **Ledger** — a node's append-only per-tool record of calls (`args` + `result` + `ordinal`).
- **Projection** — the reduced JSON schema the model is actually asked to fill, derived at compile time from the node's output type.
- **Reasoned** — a field the model authors.
- **Scope chain** — nearest-enclosing-first resolution of an artifact name across loop/branch nesting.
- **Selected** — a field where the model chooses a key from a closed set and the runtime materializes the value.

---

## Addendum: codebase grounding + implementation plan

Companion document: `docs/design/hybrid-dataflow-workflow-findings-2026-08-14.md` — a 12-agent workflow's mapping of every primitive above against the actual `develop` codebase (file:line citations), 5 independent feasibility passes per primitive, and a synthesized, sequenced epic breakdown. Filed as beads epic `neograph-ftnxl` with 10 children (`neograph-ftnxl.1`–`.11`, `.3` unused): `.1` conformance classifier, `.2` branch-arm reachability fix, `.4` Carried+Reasoned markers, `.5` tool-ledger ordinals, `.6` free Loop-scope docs, `.7` Selected marker, `.8` provides/requires v1, `.9` Each iteration_index (parked), `.10` Agent Spec metadata docs, `.11` scope_path design gate. See that epic for live, current implementation status — this file and its companion are the point-in-time design record.
