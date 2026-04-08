# Code Review Agent

## What this does

Takes a git diff (from a PR, a branch comparison, or a local uncommitted change) and produces a structured code review with findings categorized by severity, grouped by dimension (style, security, logic), with actionable fix suggestions.

This is not a linter. Linters catch syntax and formatting. This catches: "you're using string concatenation for SQL queries" (security), "this function does three things" (design), "the error message doesn't tell the user what to do" (UX), "this loop is O(n^2) but the collection can be 100k items" (performance).

## Who uses this

Individual developers reviewing their own code before pushing, or tech leads reviewing PRs from the team. The person wants a second pair of eyes that doesn't get tired at 5 PM and doesn't forget to check for SQL injection.

## The process

### Phase 1: Parse the diff

Extract the list of changed files from the git diff. For each file, capture:
- File path
- Language (inferred from extension)
- Changed lines (with context — 5 lines above and below each hunk)
- The full file content (for understanding the context of the change)

Filter out: binary files, lock files, generated code, vendor directories.

### Phase 2: Per-file analysis (parallel, multi-dimension)

For each changed file, run three independent review passes:

- **Style review**: Naming conventions, function length, comment quality, dead code, unnecessary complexity. The "is this code readable by someone who didn't write it?" check.
- **Security review**: Injection vulnerabilities (SQL, XSS, command), hardcoded secrets, insecure deserialization, missing input validation, open redirects. OWASP top 10 focus.
- **Logic review**: Off-by-one errors, null/undefined handling, race conditions, resource leaks, incorrect error handling, broken edge cases.

Each dimension runs independently (fan-out). Each produces a list of findings with severity (critical/high/medium/low), location (file:line), description, and suggested fix.

### Phase 3: Iterative refinement (per file, when needed)

If any reviewer flags a critical or high finding, re-review that file with expanded context. The re-review gets:
- The original diff
- The full file
- The findings from the first pass
- Surrounding files that import/call the changed code

The re-review either confirms or downgrades the finding. This prevents false positives on critical issues. Loop up to 2 times.

### Phase 4: Synthesis

Aggregate findings across all files. Deduplicate (same pattern in multiple files = one finding, not N). Prioritize by:
1. Critical security findings
2. High logic bugs
3. Medium style/design issues
4. Low suggestions

Use two models for the synthesis (Oracle models=) — different models catch different patterns in aggregation.

### Phase 5: Format output

Produce a markdown report:
- Executive summary (1-2 sentences: "3 findings, 1 critical SQL injection in auth.py")
- Critical findings with code snippets and fix suggestions
- High findings
- Medium/low in a table
- "What's good" section (positive feedback on well-written code)

## Data sources

| Source | How | What it provides |
|--------|-----|------------------|
| Git diff | `git diff` or GitHub API | Changed files + hunks |
| Full file content | `git show` or file read | Context for the change |
| Import graph | Simple grep/AST | Which files import/call the changed code |

No external APIs needed — everything is local git operations. This makes it runnable offline and in CI.

## Input format

One of:
- A git ref range: `main..feature-branch`
- A PR URL: `https://github.com/owner/repo/pull/123`
- A directory with uncommitted changes: `.`

## Output format

Markdown report to stdout. Optionally: JSON for CI integration, GitHub PR comment.

## What makes this a good neograph example

- **Each**: fan-out over changed files (N files analyzed in parallel)
- **Each (nested)**: fan-out over review dimensions per file (style + security + logic in parallel)
- **Loop**: re-review with expanded context when critical findings detected
- **Oracle models=**: multi-model synthesis for dedup and prioritization
- **Sub-constructs**: each review dimension is an isolated sub-pipeline
- **Tools**: real git operations (read diff, read file, grep)
- **Compile-time validation**: change the Finding schema and see the error
- **No API keys needed for the basic version** — runs with local models or any API
