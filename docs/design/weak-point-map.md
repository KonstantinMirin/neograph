# Weak Point Map — Bug Pattern Catalog

Every historical bug reveals a **coupling pattern** — a pair of features or
parameters whose interaction was under-tested. This map catalogs those patterns
so `/test-obligations` can systematically target them.

## Pattern 1: Modifier Composition (22 bugs)

Bugs where two modifiers interact on the same item.

| Pair | Bugs | Pattern |
|------|------|---------|
| Oracle + Loop | neograph-3oxx, neograph-ltqj | Second modifier silently dropped or crashes |
| Each + Loop | neograph-c4b9 | Counter not incremented on skip |
| Oracle + Each | neograph-tpgi (feature, not bug) | Needed flat M x N fusion |
| Each + skip_when | neograph-gpng | skip_value not wrapped in Each dict key |
| Loop + dict-form | neograph-ltqj, neograph-o5gd | _extract_input reads wrong field on re-entry |
| Oracle + dict-form | neograph-7ft | Concurrent write error |
| Duplicate modifiers | neograph-li9b | Silently dropped |

**Obligation pattern**: For every modifier M1, test M1 alone AND M1 x M2 for
all valid M2 combinations, with both single-type and dict-form inputs/outputs.

## Pattern 2: DI Resolution Across Boundaries (12 bugs)

Bugs where FromInput/FromConfig resolves differently depending on execution context.

| Context | Bugs | Pattern |
|---------|------|---------|
| After Operator resume | neograph-pd8j | Config not re-injected |
| After Each fan-out | neograph-iio2 | Config lost across Send boundary |
| FromInput shadows upstream | neograph-shsr | Silent dependency drop |
| Double DI marker | neograph-3ep3 | Silently picks first |
| Bundled model required | neograph-7nzv | Not enforced at runtime |
| merge_fn DI | neograph-f70z, neograph-s2h8 | Invisible to lint |

**Obligation pattern**: For every DI kind (from_input, from_config, from_input_model,
from_config_model, from_state, constant), test resolution in: normal execution,
after Each barrier, after Oracle merge, after Operator resume, inside sub-construct,
inside Loop re-entry.

## Pattern 3: Dict-Form Inputs/Outputs (10 bugs)

Bugs specific to the `dict[str, type]` form of Node.inputs or Node.outputs.

| Area | Bugs | Pattern |
|------|------|---------|
| Fan-in validation | neograph-d8w | dict-form outputs not registered as per-key producers |
| Loop re-entry | neograph-ltqj, neograph-o5gd | Wrong field read / wrong key placement |
| Oracle + dict-form | neograph-7ft | Collector holds full result dict |
| Each + dict-form | (multiple) | Per-key wrapping in _build_state_update |

**Obligation pattern**: Every code path that reads/writes node outputs should be
tested with BOTH single-type and dict-form. This includes: _extract_input,
_build_state_update, effective_producer_type, make_oracle_redirect_fn, make_each_redirect_fn.

## Pattern 4: Sub-construct Boundary (8 bugs)

Bugs at the boundary between parent and child constructs.

| Area | Bugs | Pattern |
|------|------|---------|
| Output boundary check | neograph-luzc, neograph-c4se | Input port counts as producer |
| Context forwarding | neograph-j66m | Context= not forwarded to sub-construct |
| Oracle models on Construct | neograph-e481 | neo_oracle_model not forwarded |
| Port param ambiguity | neograph-vih | Multiple port params silently overwrite |

**Obligation pattern**: For sub-constructs, test: input port vs internal producer,
context forwarding, Oracle model forwarding, Each item forwarding, Loop counter
propagation. Test with the sub-construct as first node AND as middle node.

## Pattern 5: State Reducer Edge Cases (6 bugs)

Bugs in the LangGraph state reducers.

| Reducer | Bugs | Pattern |
|---------|------|---------|
| _merge_dicts | neograph-o0tv, neograph-kipb | Duplicate keys, non-dict input |
| _collect_oracle_results | neograph-pq4b | Results not collected |
| _append_loop_result | (none yet) | Untested with None existing |
| _append_tagged | (none yet) | Untested duplicate handling |

**Obligation pattern**: Every reducer should be tested with: None existing,
empty collection existing, populated existing, wrong-type input (None, list, str).

## Pattern 6: LLM Response Parsing (6 bugs)

Bugs in JSON extraction, retry, and structured output.

| Area | Bugs | Pattern |
|------|------|---------|
| R1 XML after budget | neograph-irv3 | Tool-call XML in content |
| JSON extraction | neograph-hqhw | Lenient parsing needed |
| Retry token tracking | neograph-xcwd | Retry usage invisible to callback |
| Empty usage | neograph-pz2x | Empty dict is falsy |

**Obligation pattern**: For every LLM call site, test with: successful response,
malformed JSON, empty usage_metadata, missing usage_metadata, retry scenario.

## Pattern 7: Validation Gaps (15 bugs from adversarial testing)

Patterns caught by the fixture-based safety net.

| Area | Bugs | Pattern |
|------|------|---------|
| Type compatibility | neograph-pbyz | Optional/Union crashes |
| Reference validation | neograph-r135 | context= not validated |
| Compile-time checks | neograph-fn5x, -9513, -mn41, -c4se | Missing pre-compile validation |

**Obligation pattern**: Every new IR field that references another entity
(node name, field name, type) needs a validation check AND a should_fail fixture.

---

## How to use this map

1. Pick a pattern category (e.g., "Modifier Composition")
2. For each coupling pair, run `/test-obligations` on the functions involved
3. Focus on combinations NOT in the existing test suite
4. File bugs for confirmed behavioral gaps

The highest-yield targets are Pattern 1 (modifier composition) and Pattern 2
(DI across boundaries) — they account for 34 of 128 bugs (27%).
