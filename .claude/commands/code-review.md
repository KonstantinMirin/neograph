---
name: code-review
description: Launch multi-agent code review covering architecture, security, testing, and consistency
arguments:
  - name: scope
    description: "Review scope: 'full' (all src/), a directory path, or a file glob (default: full)"
    required: false
---

# Multi-Agent Code Review: $ARGUMENTS

## MANDATORY: Use Agent Teams with Review Agents

You MUST use the `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` feature with the
8 review agent types (`.claude/agents/review-*.md`). Review agents are
read-only for source code — they scan, analyze, and write findings to their
assigned output file. They do NOT modify source code.

### What you must NOT do:
- Do NOT launch independent background subagents without `team_name`
- Do NOT use `run_in_background: true` with standalone Task calls
- Do NOT use `subagent_type: "general-purpose"` for review — use the specific review agent types

## Protocol

### Step 1: Load team tools
```
ToolSearch: "select:TeamCreate"
ToolSearch: "select:TaskCreate"
ToolSearch: "select:SendMessage"
```

### Step 2: Create output directory and team

```bash
REVIEW_DIR=".claude/code-review/$(date +%d%m%y_%H%M)"
mkdir -p "$REVIEW_DIR"
```

```
TeamCreate: team_name="code-review-<DDMMYY>", description="Multi-agent code review"
```

### Step 3: Determine scope

The review scope is: `$ARGUMENTS`

If no scope was provided, default to `full` (all of `src/`).

### Step 4: Create tasks and spawn 8 reviewer teammates

For each of the 8 review dimensions, create a task and spawn a teammate.
All 8 MUST be spawned in a single message (parallel launch).

```
Task:
  team_name: "<team-name>"
  name: "review-architecture"
  subagent_type: "review-architecture"
  prompt: |
    Review scope: <REVIEW_SCOPE>
    Output file: <REVIEW_DIR>/review-architecture.md

    Read your agent definition for the full checklist and output format.
    Write your findings to the output file. Do NOT modify any other file.
```

The 8 agents:

| name | subagent_type | Output file |
|------|---------------|-------------|
| review-architecture | review-architecture | `<dir>/review-architecture.md` |
| review-python-practices | review-python-practices | `<dir>/review-python-practices.md` |
| review-layering | review-layering | `<dir>/review-layering.md` |
| review-execution-excellence | review-execution-excellence | `<dir>/review-execution-excellence.md` |
| review-consistency | review-consistency | `<dir>/review-consistency.md` |
| review-dry | review-dry | `<dir>/review-dry.md` |
| review-security | review-security | `<dir>/review-security.md` |
| review-testing | review-testing | `<dir>/review-testing.md` |

### Step 5: Monitor and coordinate

- Reviewers send messages when they complete or get stuck
- Messages are delivered automatically — no polling needed
- Use SendMessage to communicate with reviewers by name
- When all 8 report done, proceed to synthesis

### Step 6: Synthesize findings

After ALL 8 agents complete, read all 8 output files and produce a synthesis
report at `<REVIEW_DIR>/synthesis.md`.

**Synthesis is NOT a simple count.** Follow this protocol:

#### 6a: Validate Critical and High findings

For every Critical and High finding across all 8 reports:
1. Run the reproduction command the agent provided
2. If it reproduces → mark **verified**
3. If it doesn't reproduce → mark **false positive** with reason
4. Check if another agent found the same issue → **deduplicate**

#### 6b: Spot-check Medium findings

For Medium findings, spot-check ~30%. Verify the pattern holds for the rest.

#### 6c: Summarize Low findings

Low findings get a summary table without individual verification.

#### 6d: Cross-reference across agents

Identify systemic patterns — issues flagged independently by multiple agents.
These are the most important findings because they indicate structural problems.

#### 6e: Write synthesis.md

Use this exact structure:

```markdown
# Code Review Synthesis — <DATE>

**Scope**: <what was reviewed>
**Agents**: 8 ran, N produced findings
**Date**: YYYY-MM-DD

## Validation Summary

| Agent | Raw Findings | Verified | False Positives | Deduped |
|-------|-------------|----------|-----------------|---------|
| architecture | N | N | N | N |
| python-practices | N | N | N | N |
| layering | N | N | N | N |
| execution-excellence | N | N | N | N |
| consistency | N | N | N | N |
| dry | N | N | N | N |
| security | N | N | N | N |
| testing | N | N | N | N |
| **Total** | **N** | **N** | **N** | **N** |

## Critical Findings (verified)

### CRIT-01: <title>
- **Source agent**: <which agent found this>
- **Original ID**: <agent's finding ID, e.g. CR-01>
- **File**: `path:line`
- **Verification**: <what the synthesizer did to confirm this is real>
  - Ran: `<reproduction command from agent's report>`
  - Result: <confirmed / false positive / partially correct>
- **Cross-references**: <other agents that flagged related issues>
- **Impact**: <concrete consequence>
- **Recommended action**: <specific fix>

## High Findings (verified)

### HIGH-01: ...

## Medium Findings (verified)

### MED-01: ...

## Low Findings (summary only)

| ID | Agent | File | Description |
|----|-------|------|-------------|
| LOW-01 | consistency | path:line | ... |

## Patterns Observed

<Cross-cutting themes that emerged from multiple agents. E.g., "3 agents
independently flagged tenant isolation gaps in the creative module.">

## False Positives Discarded

| Original ID | Agent | Why discarded |
|-------------|-------|---------------|
| CR-02 | security | Not a vulnerability: input is already validated at line X |

## Metrics

- **Architecture compliance**: N/7 patterns fully adopted
- **Pattern adoption**: N% repository, N% UoW, N% factory fixtures
- **Test coverage shape**: unit=N, integration=N, e2e=N (pyramid/diamond/inverted?)
- **Obligation coverage**: N/M behavioral obligations covered (N% allowlist remaining)
- **Security posture**: N critical, N high open items
```

### Step 7: Shut down team and report

1. Send shutdown requests to all 8 reviewers
2. Delete the team after all teammates shut down
3. Present the synthesis summary to the user
4. Do NOT auto-file beads issues — the user reviews first and decides what to act on
