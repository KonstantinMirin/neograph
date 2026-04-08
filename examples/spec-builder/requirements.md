# Spec Builder — describe a pipeline, get a running pipeline

## What this does

Takes a natural language description of a workflow and produces a running neograph pipeline. The user describes what they want ("research these companies, score them, draft outreach for the good ones"), and an LLM generates the pipeline spec (YAML) + type definitions. neograph compiles and executes it.

This is the "Clay for arbitrary LLM workflows" demo. Clay has fixed nodes (enrichment, scoring, emailing). This has arbitrary nodes — the LLM defines them based on what you ask for.

## Who uses this

Founders, ops people, anyone who can describe a workflow but doesn't write code. Also: developers prototyping — describe the pipeline in English, get a working first draft, then refine the YAML directly.

## The flow

### Step 1: User describes the workflow

Plain text. Examples:

> "I have a list of 50 companies. For each one, look up their website and recent news. Score them 0-100 on how well they match my ICP (B2B SaaS, 50-200 employees, raised Series A-B). For companies scoring above 60, draft a cold email referencing something specific from the research."

> "Take this incident alert JSON. Pull logs, metrics, and recent deploys in parallel. Correlate them into a timeline. Form a hypothesis about the root cause. Write an RCA draft."

> "Review this git diff. For each changed file, check for security issues, logic bugs, and style problems independently. Synthesize into a prioritized report."

### Step 2: LLM generates the spec

An LLM call with:
- **Input**: the user's description + the JSON Schema for pipeline specs + the JSON Schema for project types
- **Output**: two YAML documents — `pipeline.yaml` and `project.yaml`

The LLM knows the spec format because the JSON Schema IS the tool schema. It defines types (Pydantic models as JSON Schema), nodes (with mode, prompt, inputs/outputs, modifiers), and wiring.

The generation prompt teaches the LLM:
- Available modes: `think` (LLM call), `scripted` (Python function), `agent` (LLM + tools)
- Available modifiers: `Oracle` (ensemble), `Each` (fan-out), `Loop` (iterate)
- How nodes wire: parameter names match upstream node names
- How types work: JSON Schema properties become Pydantic fields
- Inline prompts: each `think` node needs a prompt that tells the LLM what to do

### Step 3: Validate the spec

- JSON Schema validation of both documents (catches structural errors)
- Type-reference validation (every node's inputs/outputs reference defined types)
- Wiring validation (compile-time type checking)

On validation failure, feed the errors back to the LLM and retry (same pattern as the json_mode retry with validation details).

### Step 4: Compile and run

```python
construct = load_spec(pipeline_yaml, project=project_yaml)
graph = compile(construct)
result = run(graph, input=user_params)
```

### Step 5: Show the result

Structured output from the pipeline — the types the LLM defined determine the shape.

## What makes this different from "just prompting an LLM"

1. **Type safety**: the generated spec goes through neograph's compile-time validation. Type mismatches between nodes are caught before execution, not at 3 AM.
2. **Parallelism**: Each modifier means independent steps run in parallel automatically. The LLM doesn't need to think about async/threading.
3. **Ensemble**: Oracle modifier means you get multiple model perspectives merged. The LLM just says "use 3 models" in the spec.
4. **Iteration**: Loop modifier means quality-gated refinement. The LLM defines when to stop, neograph handles the back-edge.
5. **Durability**: checkpointing means a 20-minute pipeline can resume from where it left off.
6. **Observability**: structlog at every node, timing data, token usage — all automatic.

None of these exist when you "just prompt an LLM." The spec is the contract between human intent and structured execution.

## Architecture

```
User description (text)
        |
        v
   [generate-spec]  LLM call with JSON Schema as structured output
        |
        v
   pipeline.yaml + project.yaml
        |
        v
   [validate]  JSON Schema + compile-time type checking
        |
        v  (retry with errors if validation fails)
   [compile]   load_spec → compile
        |
        v
   [execute]   run with user parameters
        |
        v
   Structured result (typed, observable, checkpointed)
```

The generate-spec step is itself a neograph pipeline (meta-pipeline):
- Node 1: `parse_request` (scripted) — extract intent, entities, constraints from user description
- Node 2: `generate` (think, Oracle models=) — LLM generates the spec, multiple models for diversity
- Node 3: `validate` (scripted) — JSON Schema + compile validation
- Node 4: `compile_and_run` (scripted) — load_spec → compile → run

With a Loop on nodes 2-3: generate → validate → if errors, retry with feedback.

## Demo experience

This needs a walkthrough page on neograph.pro showing:

1. The user's plain-text description (input box or code block)
2. The generated YAML spec (syntax-highlighted, collapsible)
3. The generated types (what Pydantic models the LLM created)
4. The compiled graph (visual — mermaid diagram or similar)
5. The execution log (structlog output, timing, token usage)
6. The structured result (JSON output)

The walkthrough should show 2-3 examples:
- Simple: "summarize this document in 3 bullet points" (single node, no modifiers)
- Medium: "research and score these companies" (Each + Oracle)
- Complex: "review this code diff across 3 dimensions with refinement" (Each + Loop + sub-constructs)

Each example shows the full flow from description to result.

## What neograph features this demonstrates

- **load_spec**: YAML → Construct (the whole point)
- **JSON Schema validation**: specs validated at load time
- **Compile-time type checking**: type mismatches caught before execution
- **All modifiers**: Each, Oracle, Loop available in the spec format
- **Inline prompts**: LLM-generated prompts embedded in the spec
- **Type auto-generation**: JSON Schema → Pydantic models
- **The thesis**: "the agent generates the workflow, the workflow executes the agent"
