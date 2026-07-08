# Prompt rendering/management: three-consumer survey + the rethink

**Date**: 2026-07-08
**Method**: three parallel survey agents, one questionnaire, file:line-cited reports (agent-stark, ox-troubleshooting-demo, piarch/derive_ensemble).
**Question**: is "pass the model, receive BAML" the feature everyone wants, and what should neograph ship for prompt rendering/management?

## The one-paragraph answer

The thesis is **fully validated** — all three consumers independently converged on
`describe_value` for inputs and `describe_type` for output schemas (zero
`model_json_schema`, zero `json.dumps` in any pipeline prompt path), and each
hand-rolled the same ~40–150 lines of glue around them: load template, render
models, inject schema, fail-loud substitute, wrap messages. hjwv/euyh already
shipped that glue as `DefaultPromptCompiler` + primitives + `di_inputs` —
**the problem is adoption (currently zero) plus four small deltas, not a missing
feature**. The one genuinely unshipped need that recurs (2 of 3 consumers) is
prompt *management* — filename-versioned variants wired into promptfoo evals —
and the survey's sharpest discovery is that both eval harnesses **rebuild
prompts outside the graph** (piarch's evals even use a *different* schema
mechanism than its pipeline), which a public `compile_prompt` kills without
neograph becoming a prompt-management platform.

## Consumer reality matrix

| Dimension | agent-stark | ox-troubleshooting-demo | piarch |
|---|---|---|---|
| Compiler | ~90 lines, one seam (`bridge.py:149-207`) | 4 near-identical compilers (~40 ea) | 148 lines (`neograph_bridge.py:91-238`) — the GH-5 compiler |
| Adoption of shipped surface | `describe_type/value` only | `describe_type/value` only | `describe_value` only |
| Template store | 20 flat `.txt`, git-versioned | 34 `.txt`, **filename-versioned** (`explain-v7`) + flip constants | **83 `.txt`, filename-versioned** (`rubric-v2..v12`) |
| Substitution | `.replace` loop (brace-safe by construction) | `str.format` (brace-fragile) | `str.format` (brace-fragile) |
| Strictness | fail-loud regex → RuntimeError | strict (KeyError → RuntimeError) | fail-loud RuntimeError w/ remediation (the `PromptVarMissing` ancestor) |
| Schema injection | `describe_type` → `{json_schema}` | `describe_type` → `{x_schema}` under `## Output` | `describe_type` (+ a dead `model_json_schema` fallback branch) |
| Message shaping | constant system + user | single user message | **per-node roles** (explore→user-only; else system+user w/ node line) |
| DI → templates | **seed nodes** (the `{domain}` incident; filed GH-5) | one hand-read of `config['configurable']` | config + context hand-marshalling |
| Deletable under the shipped surface | **~85 of 90 lines** + the seed nodes | the 4 compilers + `evals/prompts/*` rebuilds | **~100 of 148** + the dead schema branch |
| Migration blockers | none (2-line `.txt` loader adapter) | none (pure consolidation) | message-role shaping, legacy aliases, tool-log folding, model-as-object dotted access |
| Beyond-rendering needs | none | fan-in dict rendering; eval hooks | **variant/eval apparatus** (the big one); reason-then-coerce prose; think-tag strip |

## Findings

### F1 — The thesis is the consumers' existing practice, not a wish
Every pipeline prompt in all three repos is already `describe_value(input)` +
`describe_type(output)` + template. agent-stark's ADR-0004 frames adopting the
neograph renderers as "a deletion, not construction"; ox-demo's ADR-0002 is
titled "the schema IS the model". The convergence is total.

### F2 — Shipped-but-unadopted: the gap is migration, not capability
The hjwv/euyh surface (`DefaultPromptCompiler`, `substitute`+`PromptVarMissing`,
`render_inputs`, `inject_schema`, `di_inputs`) replaces each hand-rolled
compiler with near-zero blockers — agent-stark (which *filed* GH-5) deletes ~95%
of its prompt layer including the seed-node workaround. Nobody migrated because
the surface shipped days ago mid-flight. Adoption needs recipes + the deltas in F3.

### F3 — Four small deltas unblock full migration (the actionable list)
1. **`render_inputs` must render fan-in dicts and tool-interaction lists** —
   ox-demo's `_render_verifications` (`dict[str, Model]` from `Each` merges) and
   stark/piarch's `list[ToolInteraction]` → research-packet folding are the same
   shape: the framework knows these container types; the renderer should.
2. **Loader convenience**: `.txt` template dirs (all three use `.txt`, the
   shipped dir-loader assumes `.md`) — suffix param or documented 2-line callable.
3. **Message-shaping recipe (mostly shipped)**: `DefaultPromptCompiler(system=)`
   + the `render_messages` override cover stark and piarch's per-node roles —
   verify `node_name` reaches the override and document the recipe.
4. **Kill the `neo_subgraph_input` leak**: ox-demo's compilers do
   `_unwrap(input_data, "neo_subgraph_input", "state", ...)` — a framework-internal
   key every consumer compiler must know. Sub-construct port values should
   surface under a stable, friendly name (or an alias) in template-ref input_data.

### F4 — The eval-parity discovery (highest new-feature value per line)
Both eval-running consumers rebuild prompts *outside* the graph to eval them:
ox-demo duplicates builders in `evals/prompts/`; piarch's evals hand-roll
literal JSON-Schema + `response_format` — **a different schema mechanism than
its own pipeline**, meaning its evals do not test the prompts production sends.
A **public `compile_prompt(...)`** — the exact function the graph calls,
callable standalone (the `render_prompt` inspection path, promoted and
documented) — gives eval harnesses byte-identical prompts. This is the
"pass the model, receive BAML" thesis *as a public function*.

### F5 — Prompt management: draw the line at rendering
Variant/experiment lifecycle (filename-versioned templates, promptfoo providers,
per-eval variant selection) is the biggest unaddressed need in 2 of 3 consumers —
and it is exactly where neograph should NOT build a platform (Langfuse/promptfoo
own experiments; the three-layer instinct applies: use the ecosystem). What
neograph should do: **bless the convention** (filename-versioned `.txt` +
flip-constant, documented), and make `compile_prompt` accept a template-source
override so eval harnesses can parameterize variants without rebuilding
rendering. Registry/A-B/experiment tracking: explicitly out of scope.

### F6 — The old #1 pains are (mostly) already fixed
Both ox-demo's and piarch's dominant prompt-adjacent pain was output-strategy
reliability, not rendering: json_mode native `response_format` (shipped, 15s2)
and structured-retry-on-ValidationError (shipped, zcxd). Residual to verify with
the consumer: ox-demo's `nyt.19` note ("structured fallback does not fail closed
on a second bad turn") — likely subsumed by zcxd's bounded-retry-then-
ExecutionError + their `safe_run`; confirm and close the loop. piarch's
`strip_think_tags`/`json_repair` eval hacks shrink once evals call the real path.

## Roadmap (proposed)

| # | Item | Size | Notes |
|---|---|---|---|
| R1 | `render_inputs` container rendering (fan-in dict, ToolInteraction list) + `.txt` loader convenience + message-shaping recipe verification | S | F3.1–3 in one molecule |
| R2 | Public `compile_prompt` for eval harnesses + template-source override | S–M | F4+F5; the eval-parity unlock |
| R3 | Sub-construct port naming/alias in template-ref input_data (kill `neo_subgraph_input` in consumer code) | M | F3.4 — needs a design atom (naming + back-compat) |
| R4 | Migration recipes: one short doc per consumer pattern (stark swap, ox 4-compiler collapse, piarch bridge w/ overrides) + the blessed variant convention | S | docs; makes 0.6.0 the "adopt me" release |
| — | Variant registry / experiments / A-B | — | **out of scope**, documented as such (F5) |

**Gate recommendation**: R1+R2+R4 are small and make the release the adoption
moment for all three consumers; R3 needs design and can be 0.6.x. Maintainer
decides.
