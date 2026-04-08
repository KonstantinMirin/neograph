# Spec Builder — describe a workflow, get a running pipeline

## The point

Demonstrate that given a fixed runtime surface (tools + models), a user can describe a workflow in plain English and get a compiled, type-safe, observable pipeline back. The LLM does the wiring. neograph does the execution.

This is NOT a product. It is a demo that shows the spec format + load_spec + compile in action.

## System boundaries

### Fixed runtime surface (the "platform")

Three tools, pre-registered:

| Tool | What it does | Implementation |
|------|-------------|----------------|
| `web_search` | Search the web, return top 5 results with titles + snippets | Serper API (real) or fake results from `data/search_results.json` |
| `read_page` | Fetch a URL, extract text content | httpx + readability (real) or fake from `data/pages/` |
| `linkedin_lookup` | Look up a person's LinkedIn profile | Fake — returns from `data/linkedin/{name}.json` |

Three model tiers, pre-configured:

| Tier | Model | Use case |
|------|-------|----------|
| `reason` | Claude Sonnet | Complex analysis, scoring |
| `fast` | Gemini Flash | Quick extraction, classification |
| `creative` | GPT-4o | Writing, personalization |

That is it. No file system access, no databases, no email sending, no other APIs. The LLM can only use these tools and models when generating the spec.

### Fixed test data

`data/linkedin/` contains 5-10 fake LinkedIn profiles (JSON). Different roles, companies, industries. Enough variety for the demo to be interesting.

`data/search_results.json` contains canned web search results for a few company names. Used when running without Serper API key.

### One use case for the demo

"Research [person] — find out what their company does, what they care about, and write a personalized LinkedIn connection request."

This is concrete, uses all three tools, produces a tangible output (the connection request), and naturally exercises:
- **Agent mode**: web_search + read_page (tool-calling nodes)
- **Think mode**: scoring, writing (structured LLM output)
- **Each**: if given multiple people, fan-out
- **Loop**: refine the connection request until quality threshold met

## How it works

### The project surface file

`project.yaml` — declares what the platform offers:

```yaml
tools:
  - name: web_search
    description: "Search the web. Returns top 5 results with title, url, snippet."
    args:
      query: {type: string}
  - name: read_page
    description: "Fetch a URL and extract the main text content."
    args:
      url: {type: string}
  - name: linkedin_lookup
    description: "Look up a person's LinkedIn profile by name."
    args:
      name: {type: string}

models:
  reason: "Complex analysis, scoring, evaluation"
  fast: "Quick extraction, classification, parsing"
  creative: "Writing, personalization, tone"
```

This file IS the system prompt context for the spec-generating LLM. The LLM sees exactly what it can use.

### The flow

```
1. User: "Research Sarah Chen and write a connection request"
2. Script loads project.yaml (available tools + models)
3. Script calls LLM with:
   - The user's request
   - The project surface (tools + models)
   - The pipeline JSON Schema (what a valid spec looks like)
4. LLM generates: pipeline.yaml + types
5. Script validates: JSON Schema + compile-time type checking
6. If validation fails: retry with errors (same pattern as json_mode retry)
7. Script runs: load_spec → compile → run
8. Output: structured result (research + connection request)
```

### What the LLM generates (example)

Given "Research Sarah Chen and write a connection request":

```yaml
# types (generated)
types:
  ProfileData:
    properties:
      name: {type: string}
      headline: {type: string}
      company: {type: string}
      recent_activity: {type: string}
  CompanyInfo:
    properties:
      description: {type: string}
      recent_news: {type: string}
  ConnectionRequest:
    properties:
      message: {type: string}
      score: {type: number}
      reasoning: {type: string}

# pipeline (generated)  
name: research-and-connect
nodes:
  - name: lookup
    mode: agent
    model: fast
    tools: [linkedin_lookup]
    prompt: "Look up the LinkedIn profile for the person specified in the input."
    outputs: ProfileData

  - name: research
    mode: agent
    model: reason
    tools: [web_search, read_page]
    prompt: "Research this person's company. Search for recent news, product launches, funding. Read the most relevant pages."
    inputs: {lookup: ProfileData}
    outputs: CompanyInfo

  - name: write
    mode: think
    model: creative
    prompt: "Write a personalized LinkedIn connection request..."
    inputs:
      lookup: ProfileData
      research: CompanyInfo
    outputs: ConnectionRequest
    loop:
      when: "score < 0.8"
      max_iterations: 3
```

The LLM decided the types, the nodes, the prompts, and the wiring. neograph validates and runs it.

## What this is NOT

- NOT a product or platform
- NOT a web UI (it is a Python script you run from the terminal)
- NOT supporting arbitrary tools (3 tools, that is the boundary)
- NOT generating Python code (it generates YAML specs)
- NOT doing anything that load_spec + compile + run can not already do

## Files

```
examples/spec-builder/
  requirements.md          (this file)
  project.yaml             (fixed runtime surface)
  builder.py               (the script — takes user request, generates spec, runs it)
  prompts/
    generate_spec.md       (system prompt for the spec-generating LLM)
  data/
    linkedin/              (5-10 fake profiles)
    search_results.json    (canned web search results)
    pages/                 (canned page content)
```

## Demo walkthrough (for neograph.pro)

One page showing:

1. "Here is what the platform offers" — the project.yaml (3 tools, 3 models)
2. "Here is what the user asked" — plain text request
3. "Here is what the LLM generated" — the YAML spec (syntax-highlighted)
4. "Here is what neograph validated" — compile log showing type checking
5. "Here is what ran" — execution log with timing and tool calls
6. "Here is the result" — the structured output

One use case, shown end to end. That is the demo.
