---
name: team
description: Launch a coordinated agent team to work on parallel tasks
arguments:
  - name: prompt
    description: What the team should do (e.g., "process BR-RW-001, BR-RW-002, BR-RW-003 in parallel")
    required: true
---

# Launch Agent Team: $ARGUMENTS

## MANDATORY: Use Agent Teams with Executor Agents

You MUST use the `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` feature with the
**executor** agent type (`.claude/agents/executor.md`). Executors are
self-contained: they read their beads task, follow the skill protocol,
run the linter, and close the task.

### What you must NOT do:
- Do NOT launch independent background subagents without `team_name`
- Do NOT use `run_in_background: true` with standalone Agent calls
- Do NOT use `subagent_type: "general-purpose"` for beads task execution — use `executor`

## Protocol

### Step 1: Create the team
```
TeamCreate: team_name="<descriptive-name>", description="<what the team does>"
```

### Step 2: Plan the work

Analyze the user's prompt and break it into parallel work items.

For beads task IDs, run `bd show <id>` to read descriptions and verify
they're unblocked.

For bulk work (e.g., "process all unprocessed raw inputs"), query the graph:
```bash
contextgit status --format=json
contextgit impact <node> --depth=1 --format=json
```

**Pre-assign artifact IDs** to avoid conflicts between parallel executors:
- UC numbers: check `ls docs/requirements/use-cases/` for next available
- BR-RULE numbers: check `ls docs/requirements/business-rules/` for next available
- NFR/ADR numbers: check respective directories

Each work item becomes:
- A task in the team's task list (via TaskCreate)
- An executor teammate spawned to handle it (via Agent with team_name)

### Step 3: Spawn executor teammates

For each work item, spawn an executor:
```
Agent:
  team_name: "<team-name>"
  name: "executor-<short-id>"
  subagent_type: "executor"
  prompt: |
    Execute beads task piarch-<id>.

    Pre-assigned artifact ID: <ID> (use this exact ID, do not generate your own)

    Run `bd show <id>` for full description and acceptance criteria.
    Read the skill file indicated by the task label.
    Follow the skill protocol end-to-end.
    Run the linter on your output.
    Close the task when done.
    Do NOT commit.
```

**Spawn all executors in a single message** to maximize parallelism.

### Step 4: Monitor and coordinate
- Executors send messages when they complete tasks or get stuck
- Messages are delivered automatically — no polling needed
- Use SendMessage to communicate with executors by name
- When all executors report done, review their output

### Step 5: Verify and commit

After all executors complete:
1. Run full linter: `uv run python scripts/lint_body_refs.py docs/requirements/`
2. Rescan index: `contextgit scan docs/requirements/ --recursive`
3. Stage all new/modified requirement files
4. Commit with descriptive message
5. `bd sync --flush-only`

### Step 6: Shutdown team

Send shutdown requests to all executors, then:
```
TeamDelete
```

## User's Request

$ARGUMENTS
