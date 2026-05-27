# §6 Test-discipline audit — 2026-05-20

Audit ticket: **neograph-5fqg**.
Audit scope: every test in `tests/modes/` and `tests/modifiers/`, with `tests/modes/test_llm_internals.py` (134 tests) as the primary target.
Audit rubric: `docs/design/architecture-decisions.md` §6 (Test discipline).

## Method

Walked function-by-function through each in-scope file. For each test, classified the *strongest assertion(s)* against §6:

- **KEEP** — assertions express a user-visible contract (`isinstance(result, X)`, `result.field == "expected"`, `pytest.raises(ExecutionError)`, "tool scripted N calls so N tool invocations fired", "Oracle(n=3) ran 3 generators", "checkpoint resume did not re-execute"). Includes pure-function tests on extractors/parsers (§6 explicitly allows).
- **REWRITE** — at least one assertion pins an internal call-sequence count, structlog event name+payload, or private LangGraph node name, where the user-visible contract could be expressed without those pins. Keep the test; soften or replace the assertion to express the actual contract.
- **DELETE** — the entire test is *only* exercising private dispatch or log text that retests no contract beyond what the surrounding tests already cover. Conservative: when in doubt, REWRITE not DELETE.

Where a class has a consistent pattern across multiple methods, I clustered them in §2. Section §1 lists each test individually for the primary file.

---

## Section 1 — Per-test triage table

Format: `class::method | verdict | rationale | effort`.

### `tests/modes/test_llm_internals.py` (134 tests)

| Test | Verdict | Rationale | Effort |
|---|---|---|---|
| TestNodeContext::test_context_passed_to_prompt_compiler_when_declared | KEEP | Asserts on prompt_compiler kwargs contract (`context["build_catalog"].text == ...`). User-visible. | 0 |
| TestNodeContext::test_no_context_kwarg_when_node_has_no_context | KEEP | Behavioral: absence of `context` kwarg when not declared. | 0 |
| TestNodeContext::test_context_works_with_agent_mode | KEEP | Same shape as above, agent mode. | 0 |
| TestToolRegistrationError::test_clear_error_raised_when_tool_not_registered | KEEP | `pytest.raises(CompileError, match="nonexistent_tool")` — error contract. | 0 |
| TestToolRegistrationError::test_clear_error_raised_when_execute_tool_not_registered | KEEP | Same shape, act mode. | 0 |
| TestSkipWhenOnToolNodes::test_node_skipped_when_skip_when_true_on_gather | KEEP | `len(fake_tool.calls) == 0` because skip fired — contract is "tool not invoked", count==0 is the right form. | 0 |
| TestSkipWhenOnToolNodes::test_node_runs_when_skip_when_false_on_gather | KEEP | `len(fake_tool.calls) == 1` and `final_text == "llm-produced"` — output + control-flow contract. | 0 |
| TestExtractJsonEdgeCases::test_plain_json_parsed_when_no_wrapping | KEEP | Pure function on `_extract_json`. §6 explicitly allows. | 0 |
| TestExtractJsonEdgeCases::test_json_extracted_when_wrapped_in_markdown_fences | KEEP | Pure function. | 0 |
| TestExtractJsonEdgeCases::test_first_object_extracted_when_multiple_objects_in_text | KEEP | Pure function. | 0 |
| TestExtractJsonEdgeCases::test_nested_braces_parsed_when_json_has_nested_objects | KEEP | Pure function. | 0 |
| TestExtractJsonEdgeCases::test_text_returned_when_no_json_present | KEEP | Pure function. | 0 |
| TestExtractJsonEdgeCases::test_json_extracted_when_surrounded_by_prose | KEEP | Pure function. | 0 |
| TestParseJsonResponseLenientParsing::test_control_characters_in_strings_parsed | KEEP | Pure function on `_parse_json_response`. | 0 |
| TestParseJsonResponseLenientParsing::test_null_coerced_to_default_for_string_field | KEEP | Pure function, bug repro. | 0 |
| TestParseJsonResponseLenientParsing::test_null_coerced_in_nested_model | KEEP | Pure function. | 0 |
| TestParseJsonResponseLenientParsing::test_null_without_default_still_fails | KEEP | Pure function, error contract. | 0 |
| TestParseJsonResponseLenientParsing::test_trailing_comma_parsed | KEEP | Pure function. | 0 |
| TestParseJsonResponseLenientParsing::test_single_quotes_parsed | KEEP | Pure function. | 0 |
| TestParseJsonResponseLenientParsing::test_unescaped_newlines_in_strings_parsed | KEEP | Pure function. | 0 |
| TestCallStructuredFallback::test_fallback_succeeds_when_include_raw_raises_type_error | REWRITE | `call_count["n"] == 2` pins exact call sequence on a MagicMock. The contract is "TypeError on include_raw=True is recoverable" — express as `result == expected` and `with_structured_output` was called at least twice. | 3 min |
| TestCallStructuredFallback::test_result_correct_when_include_raw_supported | REWRITE | `mock_llm.with_structured_output.assert_called_once_with(Claims, include_raw=True)` is exact-call-sequence pinning. The contract is "result equals parsed" — drop the `assert_called_once_with`. | 2 min |
| TestRetryPromptIncludesSchema::test_retry_msg_includes_schema_for_nested_model | KEEP | Pure function on `_build_retry_msg`; asserts schema text appears. | 0 |
| TestRetryPromptIncludesSchema::test_retry_msg_without_model_still_works | KEEP | Pure function. | 0 |
| TestRetryPromptIncludesSchema::test_default_max_retries_is_2 | KEEP | Default-value signature check; this default is a documented contract (cited in CLAUDE.md and architecture docs). | 0 |
| TestGetGraphEdges::test_oracle_edges_visible_in_get_graph | REWRITE | Asserts private node names (`"merge_gen"`, `"post"`) — §6 explicitly forbids "private node-naming conventions". The user contract is reachability: every declared user node appears as a `target` somewhere reachable from `__start__` and there is a path to `__end__`. Rewrite to assert reachability without spelling internal node names. | 8 min |
| TestGetGraphEdges::test_each_edges_visible_in_get_graph | REWRITE | Same as above (`"assemble_verify"` is a synthesized barrier name). | 8 min |
| TestR1XmlAfterBudgetExhaustion::test_xml_tool_call_in_content_parsed_when_json_mode_gather | KEEP | `pytest.raises(ExecutionError, match=...)` — error contract. | 0 |
| TestConditionErrorHandling::test_loop_condition_wraps_attribute_error_when_value_is_none | KEEP | Error contract via `pytest.raises`. | 0 |
| TestConditionErrorHandling::test_loop_condition_wraps_type_error_when_value_is_none | KEEP | Error contract. | 0 |
| TestLlmIntrospectParams::test_c_extension_returns_empty_set | KEEP | Pure function on `_accepted_params`. | 0 |
| TestLlmNotConfigured::test_get_llm_raises_when_not_configured | KEEP | Error contract. | 0 |
| TestLlmNotConfigured::test_get_llm_accepts_all_kwargs_factory | KEEP | Behavioral: factory received kwargs. | 0 |
| TestExtractJsonEdgeCases2::test_escape_char_in_json_string | KEEP | Pure function. | 0 |
| TestExtractJsonEdgeCases2::test_unbalanced_braces_first_to_last | KEEP | Pure function. | 0 |
| TestExtractJsonEdgeCases2::test_unbalanced_with_trailing_closing_brace | KEEP | Pure function. | 0 |
| TestExtractJsonEdgeCases2::test_no_closing_brace_at_all | KEEP | Pure function. | 0 |
| TestParseJsonException::test_generic_exception_during_parse | KEEP | Pure function error path. | 0 |
| TestTruncatedArraySilentEmpty::test_truncated_array_raises_not_empty_result | KEEP | Pure-function bug repro: tests the contract "do not silently produce empty". | 0 |
| TestTruncatedArraySilentEmpty::test_truncated_array_extract_json_does_not_return_inner_dict | KEEP | Pure function. | 0 |
| TestToolCallArgsCoercion::test_string_args_retried_and_recovered | KEEP | `parsed.answer == "done"` + `len(tool_invoked) > 0` — behavioral, no exact-count pin. | 0 |
| TestToolCallArgsCoercion::test_consistent_string_args_always_coerced | KEEP | `tool_calls_received == ["BR-UC-008", "FLOW-006"]` — fake scripted exactly 2 → assertion mirrors scripted intent. | 0 |
| TestToolCallArgsCoercion::test_non_tool_calls_validation_error_reraised | KEEP | Error contract. | 0 |
| TestToolCallArgsCoercion::test_multiple_tool_calls_mixed_args | KEEP | `args_received == ["first", "second"]` matches scripted 2 calls. | 0 |
| TestToolCallArgsCoercion::test_coerced_args_parsed_correctly | KEEP | `received_args[0] == {...}` is the coercion-correctness contract. | 0 |
| TestToolCallArgsCoercion::test_malformed_json_in_string_args | KEEP | Behavioral: doesn't crash + `hasattr(response, "tool_calls")`. | 0 |
| TestToolCallArgsCoercion::test_empty_string_args | KEEP | Same. | 0 |
| TestCallStructuredUnknownStrategy::test_unknown_strategy_raises | KEEP | Error contract. | 0 |
| TestToolResultRendering::test_plain_value_returns_str | KEEP | Pure function. | 0 |
| TestToolResultRendering::test_list_of_models_rendered | KEEP | Pure function. | 0 |
| TestUnregisteredToolInReact::test_unregistered_tool_raises | KEEP | Error contract. | 0 |
| TestUsageTokenAccumulation::test_usage_tokens_accumulated_from_messages | KEEP | `len(interactions) == 1` mirrors fake's 1 scripted tool call → contract. | 0 |
| TestRenderPromptEdgeCases::test_render_prompt_not_configured_raises | KEEP | Error contract. | 0 |
| TestRenderPromptEdgeCases::test_render_prompt_with_message_objects | KEEP | Behavioral: rendered output contains expected text. | 0 |
| TestToolBudgetTrackerAllExhausted::test_all_exhausted_no_tools | KEEP | Pure function on tracker. | 0 |
| TestReActToolReturnsListOfModels::test_tool_returning_list_of_models_renders | KEEP | `len(interactions) == 1` mirrors scripted 1 call; `"a" in result and "b" in result` is rendering contract. | 0 |
| **TestReActMaxIterationsGuard::test_max_iterations_default_stops_at_20** | **REWRITE** | `assert len(interactions) == 19 and fake.call_count == 21` plus narrative comment "Iterations 1-19 execute tools (19 interactions). Iteration 20 hits the guard, skips tool execution. Iteration 21..." — textbook §6 violation (dispatch-sequence pin). Contract: "default cap stops infinite loop and returns a valid result". | 5 min |
| **TestReActMaxIterationsGuard::test_max_iterations_custom_value** | **REWRITE** | `len(interactions) == 2` + narrative "Iterations 1-2 execute tools, iteration 3 hits guard". Custom `max_iterations=3` is a user-visible knob; assert ≤3 interactions and valid result, not exact 2. | 4 min |
| TestReActMaxIterationsGuard::test_max_iterations_does_not_affect_normal_completion | KEEP | `len(interactions) == 1` matches scripted "stops after 1 tool call" intent — fake-scripted contract. | 0 |
| **TestReActMaxIterationsGuard::test_max_iterations_equals_one** | **REWRITE** | `len(interactions) == 0` with narrative on guard mechanics. Contract: "max_iterations=1 → loop terminates without infinite-loop, no successful tool execution". Soften to `interactions == [] and isinstance(result, Claims)` (which the current assertions already do — but drop the dispatch narrative in the comment). Borderline KEEP, but the narrative makes it brittle. | 2 min |
| TestReActMaxIterationsGuard::test_guard_fired_llm_ignores_wrap_up_still_calls_tools | KEEP | `len(interactions) == 0` is the user-visible "no rogue dispatch" contract. Stays. | 0 |
| **TestReActMaxIterationsGuard::test_max_iterations_guard_logs_warning** | **DELETE** | Tests `structlog` event name `react_max_iterations_exceeded` and payload key `max_iterations`. §6: "Tests do not assert on internal LangGraph dispatch routing". A specific structlog event name is implementation detail; the user-visible contract (loop terminates) is already covered by sibling tests. The event-name pin will break on any rename. | 1 min |
| **TestReActTokenBudgetGuard::test_token_budget_stops_loop** | **REWRITE** | `len(interactions) == 2` with dispatch narrative. Contract: "token_budget terminates loop". Soften to `len(interactions) < 50 and isinstance(result, Claims)`. | 4 min |
| TestReActTokenBudgetGuard::test_token_budget_none_is_no_limit | KEEP | `len(interactions) == 1` matches fake's scripted 1 call. | 0 |
| **TestReActTokenBudgetGuard::test_both_guards_fire_simultaneously** | **REWRITE** | `len(interactions) == 2` with narrative on which guard fires first. Contract: "both knobs together still terminate cleanly". | 4 min |
| TestReActTokenBudgetGuard::test_token_budget_missing_usage_metadata | KEEP | `len(interactions) == 1` matches scripted 1 call. | 0 |
| TestBareArrayExtraction::test_extract_json_finds_bare_array | KEEP | Pure function. | 0 |
| TestBareArrayExtraction::test_extract_json_bare_array_with_markdown_fence | KEEP | Pure function. | 0 |
| TestBareArrayExtraction::test_extract_json_bare_array_with_prose | KEEP | Pure function. | 0 |
| TestBareArrayExtraction::test_parse_json_response_bare_array_auto_wraps | KEEP | Pure function. | 0 |
| TestBareArrayExtraction::test_parse_json_response_bare_array_multi_field_model_raises | KEEP | Pure function error contract. | 0 |
| TestBareArrayExtraction::test_extract_json_prefers_object_over_array | KEEP | Pure function. | 0 |
| TestDSMLTrailingToolCallRecovery::test_dsml_markup_retried_with_targeted_directive | KEEP | `parsed.answer == "recovered"` is the user-visible recovery contract; `len(interactions) == 1` matches one scripted successful tool call. | 0 |
| TestDSMLTrailingToolCallRecovery::test_custom_budget_exhausted_message | KEEP | `"CUSTOM" in content` — directly user-supplied string surfaces in retry. | 0 |
| **TestDSMLDoubleFailure::test_double_dsml_falls_through_to_generic_retry** | **REWRITE** | `assert call_count[0] == 4` pins exact LLM-invoke count. Contract: "double DSML still recovers through the generic retry path". Soften to `parsed.answer == "recovered-via-generic" and call_count[0] >= 3` (need at least the tool call + DSML + generic-retry-success). | 3 min |
| TestDSMLAllRetriesFail::test_exhausted_retries_raise_execution_error_with_dsml_hint | KEEP | `pytest.raises(ExecutionError, match=...)` + `call_count[0] >= 3` (lower-bound not equality). Already conservative. | 0 |
| **TestNonDSMLParseFailureTakesGenericRetry::test_plain_json_parse_failure_bypasses_dsml_branch** | **REWRITE** | Three "discriminator" assertions inspect structlog events (`trailing_tool_call_markup`), DSML-branch-specific user-message phrase ("contained tool-call markup"), and the generic-retry user-message phrase ("could not be parsed as valid JSON" / "failed validation"). Phrase pins are brittle and §6-forbidden. The contract is: plain garbled response still recovers. Soften to `parsed.answer == "recovered-via-generic"` and drop the structlog discriminators (or keep ONE structural discriminator: the absence of DSML log event, if that's load-bearing for the bug regression). | 6 min |
| **TestE2EDSMLRecoveryViaAgentMode::test_agent_mode_recovers_from_dsml_after_budget_exhaustion_e2e** | **REWRITE** | `assert call_count[0] == 3` pins exact invoke count. The E2E contract is `result["research"].answer == "recovered-e2e"`. Drop the `call_count == 3` pin. | 2 min |
| TestCoercingToolWrapperGenerateNotAvailable::test_generate_not_available_falls_back_to_empty | REWRITE | Asserts `tool_calls_coercion_generate_failed` in caplog records. Event-name pin. Contract: wrapper falls back to empty AIMessage. Drop the log-event assertion. | 3 min |
| TestCoercingToolWrapperGenerateRaises::test_generate_raises_exception_falls_back_to_empty | REWRITE | Asserts the structlog event name + payload (`log_level=="warning"`, `"simulated network failure" in event["error"]`). Event-name + log-level + error-substring pin. Soften to "response is empty AIMessage" (already asserted) and drop event introspection. | 3 min |
| TestCoercingToolWrapperMixedDictAndStringArgs::test_mixed_dict_and_string_args_are_all_coerced_to_dicts | KEEP | Pure-ish unit test of `_CoercingToolWrapper`'s coercion loop. Inputs/outputs only. | 0 |
| **TestDSMLInStructuredStrategyPath::test_structured_path_has_no_dsml_recovery_when_provider_returns_dsml** | **REWRITE** | `assert structured_calls[0] == 2, "must fire EXACTLY twice — once for include_raw=True, once for except-TypeError compat fallback"` + `assert call_count[0] == 2`. Textbook forbidden pattern (the explanatory string IS the dispatch sequence). Contract: "structured strategy raises TypeError without DSML recovery". `pytest.raises(TypeError)` already covers it. | 4 min |
| TestDSMLInStructuredStrategyPath::test_structured_path_happy_baseline_no_dsml | REWRITE | `structured_calls[0] == 1 and len(interactions) == 1`. Happy-path: just `parsed.answer == "ok"` and `len(interactions) == 1` (scripted-1 contract — KEEP that). Drop `structured_calls[0] == 1`. | 2 min |
| **TestDSMLAfterMaxIterationsGuard::test_dsml_after_max_iterations_recovers_via_targeted_retry** | **REWRITE** | `parsed.answer == "recovered"` KEEP, `call_count[0] == 3` REWRITE to `>=`, structlog events `react_max_iterations_exceeded` and `trailing_tool_call_markup` REWRITE/DELETE. The contract is "DSML after guard still recovers" — already covered by the answer assertion. | 5 min |
| **TestDSMLAfterTokenBudget::test_dsml_after_token_budget_recovered_via_targeted_retry** | **REWRITE** | Same pattern as above: `call_count[0] == 4` pin + structlog event-name pins. | 5 min |
| TestMultipleIndependentDSMLRecoveries::test_two_sequential_calls_recover_independently | KEEP | The whole point is state isolation. `interactions1 is not interactions2`, `interactions1[0].tool_name != interactions2[0].tool_name`, distinct budget tracker exhausted sets, distinct answers. `fake1.calls == 3 and fake2.calls == 3` is a per-fake counter (not framework-internal) — borderline, but consistent with "fake scripted exactly N invokes" pattern. Mark KEEP with a note: if reviewing, consider dropping the per-fake `calls == 3` pins. | 0 |
| TestBudgetExhaustedMessageFallback::test_missing_key_uses_default_message | KEEP | `"tool-call markup" in content and "All tool budgets are exhausted" in content` — direct user-visible message content. | 0 |
| TestBudgetExhaustedMessageFallbackPostRnjw::test_none_falls_back_to_default_message | KEEP | Same shape. | 0 |
| TestBudgetExhaustedMessageFallbackPostRnjw::test_empty_string_falls_back_to_default_message | KEEP | Same shape. | 0 |
| TestLlmConfigTypedView::test_rejects_wrong_type_on_known_field | KEEP | Pure pydantic validation. | 0 |
| TestLlmConfigTypedView::test_rejects_unknown_top_level_key | KEEP | Pure pydantic. | 0 |
| TestLlmConfigTypedView::test_provider_kwargs_round_trips | KEEP | Pure pydantic. | 0 |
| TestLlmConfigTypedView::test_none_budget_message_resolves_to_default | KEEP | Pure helper. | 0 |
| TestLlmConfigTypedView::test_empty_budget_message_resolves_to_default | KEEP | Pure helper. | 0 |
| TestLlmConfigTypedView::test_nonempty_budget_message_is_returned_verbatim | KEEP | Pure helper. | 0 |
| TestDefaultBudgetExhaustedMessageRendersModelName::test_default_message_includes_output_model_name | KEEP | `content.count("ExplorationResult") == 1` — message-content contract. KEEP, but the `== 1` exact count is mildly brittle; if a future change adds a second mention this fails. Recommend KEEP for now (user-visible message text). | 0 |
| **TestSafetyBreakOnGuardWithRogueToolCalls::test_safety_break_fires_when_guard_set_but_rogue_llm_emits_tool_calls** | **REWRITE** | Asserts on structlog event `react_guard_forced_break`, event payload keys `loops`/`tool_calls`, and `log_level == "warning"`. Event-name + payload pin. User contract: `parsed.answer == "breakthrough"`, `interactions == []`, `dispatched == []` (no rogue dispatch). All three already pass without the structlog assertions; drop the log assertions. | 4 min |
| TestToolExceptionPropagates::test_tool_exception_propagates_out_of_invoke_with_tools | KEEP | Error contract via `pytest.raises`. | 0 |
| TestToolCallsShapeEdgeCases::test_empty_list_exits_loop_on_first_iteration | KEEP | `call_count[0] == 1 and parsed.answer == "done" and interactions == []` — contract is "exits on first iteration" and the `MutationProofBound` exception raised on call #2 is a runaway-detector. Reasonable. | 0 |
| TestToolCallsShapeEdgeCases::test_absent_attribute_current_behavior_raises_attribute_error | KEEP | Documents current behavior; `pytest.raises(AttributeError, match="tool_calls")`. | 0 |
| TestLlmConfigAsIRType::test_typo_on_known_field_raises_at_node_construction | KEEP | Pydantic validation. | 0 |
| TestLlmConfigAsIRType::test_provider_kwarg_at_top_level_raises_after_promotion | KEEP | Pydantic. | 0 |
| TestLlmConfigAsIRType::test_provider_kwargs_namespace_round_trips | KEEP | Pure. | 0 |
| TestLlmConfigAsIRType::test_dict_input_preserves_model_fields_set | KEEP | Pydantic round-trip. | 0 |
| TestLlmConfigAsIRType::test_default_factory_produces_empty_model_fields_set | KEEP | Pydantic. | 0 |
| TestLlmConfigAsIRType::test_as_factory_kwargs_flattens_provider_into_top_level | KEEP | Pure helper. | 0 |
| TestLlmConfigAsIRType::test_as_factory_kwargs_framework_takes_precedence_on_collision | KEEP | Pure helper. | 0 |
| TestLlmConfigAsIRType::test_merged_with_child_wins_on_set_fields | KEEP | Pure helper. | 0 |
| TestLlmConfigAsIRType::test_merged_with_child_unset_inherits_parent | KEEP | Pure helper. | 0 |
| TestLlmConfigAsIRType::test_merged_with_provider_kwargs_collision_child_wins | KEEP | Pure helper. | 0 |
| TestLlmConfigAsIRType::test_merged_with_provider_kwargs_disjoint_unions | KEEP | Pure helper. | 0 |
| TestLlmConfigAsIRType::test_invalid_output_strategy_raises_at_node_construction | KEEP | Pydantic. | 0 |
| TestLlmConfigAsIRType::test_construct_propagation_uses_typed_merge | KEEP | Pure: IR-level merge behavior. | 0 |
| TestLlmConfigAsIRType::test_node_llm_config_is_typed_field | KEEP | Structural guard. | 0 |
| TestLlmConfigAsIRType::test_construct_llm_config_is_typed_field | KEEP | Structural guard. | 0 |
| TestLlmConfigAsIRType::test_normalize_llm_config_not_referenced_in_src | KEEP | Structural guard. | 0 |
| TestLlmConfigAsIRType::test_factory_receives_full_typed_config_as_dict | KEEP | Behavioral contract: factory boundary. | 0 |
| TestLlmConfigAsIRType::test_three_surface_parity_decorative_declarative_programmatic | KEEP | Three-surface parity (project-mandated). | 0 |
| TestCallbackProtocols::test_llm_factory_protocol_field_type | KEEP | Structural. | 0 |
| TestCallbackProtocols::test_prompt_compiler_protocol_field_type | KEEP | Structural. | 0 |
| TestCallbackProtocols::test_cost_callback_protocol_field_type | KEEP | Structural. | 0 |
| TestCallbackProtocols::test_oracle_merge_hooks_protocol_field_types | KEEP | Structural. | 0 |
| TestCallbackProtocols::test_node_skip_predicate_protocol_field_type | KEEP | Structural. | 0 |
| TestCallbackProtocols::test_node_skip_value_protocol_field_type | KEEP | Structural. | 0 |
| TestCallbackProtocols::test_node_raw_fn_protocol_field_type | KEEP | Structural. | 0 |
| TestCallbackProtocols::test_runtime_checkable_isinstance_works | KEEP | Structural. | 0 |
| TestCallbackProtocols::test_simple_lambda_factory_satisfies_protocol | KEEP | Structural. | 0 |
| TestCallbackProtocols::test_kwargs_factory_satisfies_protocol | KEEP | Structural. | 0 |
| TestCallbackProtocols::test_protocols_publicly_exported | KEEP | Structural. | 0 |
| TestCallbackProtocols::test_existing_runtime_introspection_unchanged | KEEP | Pure introspection check. | 0 |
| TestCallbackProtocols::test_validate_skip_callables_still_runs | KEEP | Error contract. | 0 |
| TestDictCoercionTypoRejection::test_invoke_structured_rejects_dict_with_typo | KEEP | Error contract + "LLM not reached" — short-circuit contract. | 0 |
| TestDictCoercionTypoRejection::test_invoke_with_tools_rejects_dict_with_typo | KEEP | Same shape. | 0 |
| TestDictCoercionTypoRejection::test_coerce_llm_config_rejects_dict_with_typo_directly | KEEP | Pure validation. | 0 |

`test_llm_internals.py` subtotal: 134 tests → **KEEP 112, REWRITE 21, DELETE 1**.

---

### `tests/modes/test_execution.py` (49 tests) — selective sweep

Quick scan: this file's pattern is dominated by behavioral assertions (`result["foo"] == "bar"`, `pytest.raises(...)`, `isinstance(result, X)`). The few `call_count[0] == 1` assertions appear in checkpoint-resume tests where "did not re-execute" IS the contract. I verified the four call_count assertions in this file (lines 1065, 1071, 1301, 1306) — all four pin the resume-skipped-the-node contract.

| Cluster | Verdict | Notes |
|---|---|---|
| TestExecuteMode (1 test) | KEEP | Behavioral. |
| TestErrorPaths (8 tests) | KEEP | All `pytest.raises(...)` error contracts. |
| TestLLMUnknownToolCall (1 test) | KEEP | E2E behavioral. |
| TestFirstNodeEdgeCases (3 tests) | KEEP | E2E behavioral. |
| TestLLMConfig (4 tests) | KEEP | Behavioral. |
| TestCheckpointResume (8 tests) | KEEP | `call_count==1` pins "resume skipped re-execution" — the contract. |
| TestRunIsolated (8 tests) | KEEP | Behavioral. |
| TestStateGet (4 tests) | KEEP | Pure helper. |
| TestConfigInjectionPatterns (5 tests) | KEEP | Behavioral. |
| TestCheckpointSchemaValidation (3 tests) | KEEP | Behavioral. |
| TestPerNodeCheckpointInvalidation (1 test) | KEEP | Behavioral. |
| TestAutoResumeFromSchemaDivergence (4 tests) | KEEP | Behavioral. |

`test_execution.py` subtotal: 49 tests → **KEEP 49, REWRITE 0, DELETE 0**.

---

### `tests/modes/test_node_io.py` (49 tests) — selective sweep

| Cluster | Verdict | Notes |
|---|---|---|
| TestCostCallback (10 tests) | KEEP | All assert on user-visible cost_callback receipt (kwargs, payload). One test asserts `usage["input_tokens"] == 200` (100 + 100 across two retries) — that IS the accumulation contract (regression neograph-xcwd). |
| TestMergeDictsReducer (3 tests) | KEEP | Pure reducer. |
| TestModifierCombo (5 tests) | KEEP | Pure helper. |
| TestUnwrapHelpers (9 tests) | KEEP | Pure helpers. |
| TestSkipWhenOnScriptedNode (1 test) | KEEP | Behavioral. |
| TestSkipWhenOnThinkNode (1 test) | KEEP | Behavioral. |
| TestSingleTypeListInputFromEach (1 test) | KEEP | Behavioral. |
| TestNodeInput (7 tests) | KEEP | Pure helpers (input/output unwrapping). |
| TestNodeOutput (5 tests) | KEEP | Pure helpers. |
| TestModeDispatch (7 tests) | KEEP | Pure helpers. Instantiation + protocol conformance. |

`test_node_io.py` subtotal: 49 tests → **KEEP 49, REWRITE 0, DELETE 0**.

---

### `tests/modes/test_output_strategies.py` (16 tests) — full sweep

| Test | Verdict | Notes |
|---|---|---|
| TestOutputStrategyStructured::test_structured_output_used_when_no_strategy_specified | KEEP | Behavioral. |
| TestOutputStrategyJsonMode::test_json_parsed_when_json_mode_strategy_set | KEEP | Behavioral. |
| TestOutputStrategyJsonMode::test_fences_stripped_when_json_response_wrapped_in_markdown | KEEP | Behavioral. |
| TestOutputStrategyJsonMode::test_error_feedback_retry_recovers_when_first_response_is_garbage | REWRITE | `call_n["n"] == 2` is borderline — the contract is "retry recovered" (covered by `items == ["recovered"]`). Soften to `call_n["n"] >= 2` or drop. | 2 min |
| TestOutputStrategyJsonMode::test_validation_errors_included_in_retry_when_fields_wrong | KEEP | `len(retry_messages_seen) == 1` matches scripted 1 retry → contract. The `"items" in retry_messages_seen[0]` is the actual retry-content contract. |
| TestOutputStrategyJsonMode::test_max_retries_configurable_when_set_in_llm_config | REWRITE | `call_n["n"] == 3` (3 attempts = 1 + 2 retries). Drop or soften to `>=`. The user-visible contract `items == ["third-time"]` is sufficient. | 2 min |
| TestOutputStrategyText::test_json_extracted_when_text_strategy_with_embedded_json | KEEP | Behavioral. |
| TestOutputStrategyOnGather::test_json_parsed_when_gather_mode_with_json_strategy | KEEP | Behavioral. |
| TestPromptCompilerReceivesOutputModel::test_output_model_passed_when_produce_node_compiled | KEEP | `len(compiler_calls) == 1` matches scripted 1 LLM call. |
| TestPromptCompilerReceivesOutputModel::test_llm_config_passed_when_produce_node_compiled | KEEP | Same. |
| TestPromptCompilerReceivesOutputModel::test_schema_injected_when_compiler_uses_json_mode | KEEP | Behavioral. |
| TestGatherToolCollection::test_tool_interaction_has_expected_fields | KEEP | Structural. |
| TestGatherToolCollection::test_tool_log_written_when_gather_with_dict_outputs | KEEP | `len(tool_log) == 1` matches scripted 1 tool call. |
| TestGatherToolCollection::test_typed_result_preserved_when_tool_returns_pydantic_model | KEEP | Same shape. |
| TestGatherToolCollection::test_typed_result_holds_string_when_tool_returns_string | KEEP | Same shape. |
| TestGatherToolCollection::test_tool_result_rendered_with_schema_when_pydantic_model | KEEP | Behavioral. |

`test_output_strategies.py` subtotal: 16 tests → **KEEP 14, REWRITE 2, DELETE 0**.

---

### `tests/modes/test_core_modes.py` (6 tests) — full sweep

All six tests assert on user-visible contracts (final output equality, budget enforcement counts that match user-supplied budget). All **KEEP** (6/6).

---

### `tests/modifiers/test_each.py` (12 tests), `test_oracle.py` (55 tests), `test_compositions.py` (36 tests), `test_operator.py` (9 tests), `test_modifier_edge_cases.py` (33 tests) — sampled sweep

Searched for the §6 anti-patterns:

```
grep -n "fake.call_count|fake.calls|call_count\[0\] ==|len(interactions) ==|invoke_count|capture_logs" tests/modifiers/*.py
```

Result: 1 hit only — `test_oracle.py:91 assert gen_call_count[0] == 3`. That assertion mirrors `ensemble_n=3` (Oracle(n=3) → 3 generator calls) which IS the user-visible contract. KEEP.

No `capture_logs` or `caplog` usage in modifiers tests. No private node-name assertions. No mock-call-sequence pinning.

All five modifier files (145 tests total) → **KEEP all**.

---

## Section 2 — Aggregate counts + clusters

### Totals

| File | Tests | KEEP | REWRITE | DELETE |
|---|---:|---:|---:|---:|
| modes/test_llm_internals.py | 134 | 112 | 21 | 1 |
| modes/test_execution.py | 49 | 49 | 0 | 0 |
| modes/test_node_io.py | 49 | 49 | 0 | 0 |
| modes/test_output_strategies.py | 16 | 14 | 2 | 0 |
| modes/test_core_modes.py | 6 | 6 | 0 | 0 |
| modifiers/test_each.py | 12 | 12 | 0 | 0 |
| modifiers/test_oracle.py | 55 | 55 | 0 | 0 |
| modifiers/test_compositions.py | 36 | 36 | 0 | 0 |
| modifiers/test_operator.py | 9 | 9 | 0 | 0 |
| modifiers/test_modifier_edge_cases.py | 33 | 33 | 0 | 0 |
| **Total** | **399** | **375** | **23** | **1** |

REWRITE+DELETE = 24 tests = 6.0% of in-scope tests. All concentrated in `test_llm_internals.py` and one minor cluster in `test_output_strategies.py`.

### Clusters by anti-pattern

1. **`react_max_iterations_*` / `react_token_budget_*` / `trailing_tool_call_markup` / `react_guard_forced_break` structlog event-name pinning (7 tests).** These are the most common §6 violations in the suite. Concentrated in `TestReActMaxIterationsGuard`, `TestDSMLAfterMaxIterationsGuard`, `TestDSMLAfterTokenBudget`, `TestNonDSMLParseFailureTakesGenericRetry`, `TestSafetyBreakOnGuardWithRogueToolCalls`, `TestCoercingToolWrapperGenerateNotAvailable`, `TestCoercingToolWrapperGenerateRaises`. The structlog event names are private. Rewrite to drop event-name assertions and rely on the user-visible parse/recovery/return-value contract.

2. **`len(interactions) == N` with narrative dispatch-sequence comments (5 tests).** Concentrated in `TestReActMaxIterationsGuard` (3 of 6 tests) and `TestReActTokenBudgetGuard` (2 of 4 tests). The pattern is: `# Iterations 1-2 execute tools, iteration 3 hits guard...` followed by `assert len(interactions) == 2`. Rewrite to `len(interactions) < threshold` and drop the iteration-sequence narrative.

3. **`call_count[0] == N` pinning total LLM invocations across dispatch+retry paths (5 tests).** `TestDSMLDoubleFailure`, `TestE2EDSMLRecoveryViaAgentMode`, `TestDSMLAfterMaxIterationsGuard`, `TestDSMLAfterTokenBudget`, `TestDSMLInStructuredStrategyPath`. The recovery contract is the user-visible `parsed.answer == "recovered-*"` value. Drop the call-count pin or change to `>=` lower bound.

4. **`MagicMock.assert_called_once_with(...)` and `mock.call_count == 2` (2 tests).** `TestCallStructuredFallback` (both methods). Replace with behavioral result assertions.

5. **Private LangGraph node-name assertions (2 tests).** `TestGetGraphEdges` asserts on `"merge_gen"` and `"assemble_verify"` strings, which are framework-internal naming. Rewrite to assert reachability (declared-node-name appears in graph and has path to `__end__`) without spelling internal synthetic names.

6. **`call_n == 2/3` for retry recovery tests (2 tests).** `TestOutputStrategyJsonMode` (`test_error_feedback_retry_recovers_when_first_response_is_garbage`, `test_max_retries_configurable_when_set_in_llm_config`). Soften to `>=` lower bound; the parse-success result assertion already covers the contract.

7. **DELETE candidate (1 test).** `TestReActMaxIterationsGuard::test_max_iterations_guard_logs_warning` — exclusively asserts on a structlog event name and a payload field. No behavioral assertion. Sibling tests already cover that the loop terminates. The structlog event is implementation detail; deleting removes a brittle log-text pin without losing any contract.

### Clusters by file

`test_llm_internals.py` is the single source of REWRITE+DELETE in this audit. The §2 mechanical migration ran straight through 134 tests and preserved several pre-§2 patterns (especially in the ReAct-guard cluster) that pre-date the §6 discipline. Every other file in scope is clean.

---

## Section 3 — Concrete rewrite shapes

For each REWRITE class, here is the proposed shape.

### Shape A — drop the iteration-sequence narrative + count pin

```python
# Original (TestReActMaxIterationsGuard::test_max_iterations_default_stops_at_20):
# Iterations 1-19 execute tools (19 interactions). Iteration 20 hits
# the guard, skips tool execution. Iteration 21: unbound LLM returns
# no tool calls, loop ends. Plus 1 structured parse call. Total = 21.
assert len(interactions) == 19
assert fake.call_count == 21

# Rewrite:
# Contract: an unboundedly-eager fake (calls tools forever) must terminate
# at the default cap with a valid Claims result, without hanging.
assert isinstance(result, Claims)
assert len(interactions) < 25  # default cap is 20; small upper bound, not exact pin
```

### Shape B — drop structlog event-name assertions, keep behavioral

```python
# Original (TestSafetyBreakOnGuardWithRogueToolCalls):
forced_break_events = [
    e for e in cap_logs if e.get("event") == "react_guard_forced_break"
]
assert len(forced_break_events) == 1
evt = forced_break_events[0]
assert evt["log_level"] == "warning"
assert "loops" in evt and "tool_calls" in evt

# Rewrite (drop the structlog probe entirely):
assert parsed.answer == "breakthrough"      # recovery succeeded
assert interactions == []                    # no rogue tool execution
assert dispatched == []                      # safety break prevented dispatch
```

### Shape C — drop `MagicMock.assert_called_once_with`

```python
# Original (TestCallStructuredFallback::test_result_correct_when_include_raw_supported):
mock_llm.with_structured_output.assert_called_once_with(Claims, include_raw=True)

# Rewrite:
# Contract: when include_raw is supported, we get the parsed result back.
# (Drop the assert_called_once_with — the result assertion already covers
# the user-visible behavior.)
# (No new line needed; the existing `assert result == expected` is sufficient.)
```

### Shape D — soften call-count to lower bound

```python
# Original (TestE2EDSMLRecoveryViaAgentMode):
assert result["research"].answer == "recovered-e2e"
assert call_count[0] == 3

# Rewrite:
assert result["research"].answer == "recovered-e2e"
# Recovery requires at least the initial tool call + DSML + targeted retry.
assert call_count[0] >= 3
```

### Shape E — reachability instead of private node names (TestGetGraphEdges)

```python
# Original:
edge_set = {(e.source, e.target) for e in dg.edges}
assert ("merge_gen", "post") in edge_set
assert ("post", "__end__") in edge_set

# Rewrite:
nodes = {n.id for n in dg.nodes}
edges = list(dg.edges)
# Declared user nodes are reachable
assert "post" in nodes
# A path leads from each declared node toward __end__
def reaches_end(start: str, edges, *, end="__end__", seen=None) -> bool:
    seen = seen or set()
    if start == end: return True
    if start in seen: return False
    seen.add(start)
    return any(reaches_end(e.target, edges, end=end, seen=seen)
               for e in edges if e.source == start)
assert reaches_end("post", edges)
# (No assertion on the synthesized name "merge_gen" or "assemble_verify".)
```

### Shape F — DELETE candidate

```python
# TestReActMaxIterationsGuard::test_max_iterations_guard_logs_warning
# DELETE. The sibling test `test_max_iterations_default_stops_at_20`
# (rewritten per Shape A) already covers "guard terminates the loop".
# The structlog event-name+payload assertions retest nothing user-visible.
```

---

## Section 4 — DELETE rationale (1 test)

`TestReActMaxIterationsGuard::test_max_iterations_guard_logs_warning` is the only DELETE candidate. Justification:

1. The test's only behavioral check is that a `structlog` event whose name starts with `react_max_iterations_exceeded` was emitted, with payload `max_iterations == 2`.
2. §6 explicitly forbids assertions on internal log routing.
3. The loop-termination contract this test purportedly covers is already covered by `test_max_iterations_default_stops_at_20`, `test_max_iterations_custom_value`, and `test_max_iterations_equals_one` (all four tests share the same `GuardFake` infrastructure).
4. No other test in the suite asserts on the `react_max_iterations_exceeded` structlog event name. Removing it does not strand any other coverage.
5. If a future operator wants to debug the guard firing, they can read the source code or set a temporary `caplog` probe; the framework does not promise the event name as a public surface.

No other tests are proposed for deletion. Every other REWRITE preserves the test (and its boilerplate setup) while changing only the load-bearing assertion.

---

## Section 5 — Estimated effort

| Verdict | Count | Per-test estimate | Subtotal |
|---|---:|---|---:|
| KEEP | 375 | 0 min | 0 min |
| REWRITE | 23 | 2–8 min (median ~4 min) | ~95 min |
| DELETE | 1 | 1 min | 1 min |
| **Total** | **399** | | **~95 min** |

Add ~15 min for running `make quality` after each REWRITE batch (3 batches of ~8 tests each = 3 quality runs at ~5 min each), plus ~10 min for the final commit + closing notes.

**Total estimated effort: ~2 hours of focused work.**

This matches the original "10-test sample triage" pace: ~6 min/test for REWRITE work, dominated by reading the test carefully to confirm which assertion is the load-bearing user contract and which is the pinned implementation detail.

---

## Section 6 — Methodology for the executor

When this ticket executes, follow this order:

1. **Read this audit and §6 of architecture-decisions.md first.** Confirm the rewrite shapes A–F before touching any test.
2. **Apply REWRITEs in clusters, by anti-pattern, not by file order.** Suggested batches:
   - Batch 1 (Shape A — iteration-sequence narrative cluster, 5 tests in TestReActMaxIterationsGuard / TestReActTokenBudgetGuard).
   - Batch 2 (Shape B — structlog event-name cluster, 7 tests spread across guard + DSML retry test classes + CoercingToolWrapper).
   - Batch 3 (Shape D — call_count >= N softening, 5 tests in DSML recovery cluster).
   - Batch 4 (Shape C + Shape F — TestCallStructuredFallback's 2 mock-assert tests + DELETE the log-warning test).
   - Batch 5 (Shape E — TestGetGraphEdges' 2 tests for private node names).
   - Batch 6 (test_output_strategies.py — 2 retry-count softening, Shape D).
3. **Run `make quality` after each batch.** If a batch reveals a test now passes because the production behavior moved underneath, that's a green-light not a regression — the rewrite was correct to drop the implementation pin.
4. **Apply the single DELETE last.** The TDD habit is to DELETE only when sibling coverage is verified passing first.
5. **Final report** (in the beads ticket close-out): list the deleted test, the cluster names rewritten, and the post-rewrite test count. The pre-rewrite count is 399 in-scope; the post-rewrite count is 398 (one delete).

**Quality gate:** the suite should still pass `make quality` end-to-end. If any test starts failing after a rewrite, the rewrite was too aggressive — restore the load-bearing assertion and consider a tighter rewrite.

**Three-surface parity reminder:** none of the REWRITE candidates touch the IR layer or modifier semantics. The audit is purely an assertion-shape rewrite. No production-code changes expected.

---

## Section 7 — Open questions for maintainer

These items are flagged for explicit maintainer decision before the executor proceeds:

1. **`TestNonDSMLParseFailureTakesGenericRetry::test_plain_json_parse_failure_bypasses_dsml_branch`** — the test author was deliberate about the three "discriminator" assertions because they're the only way to prove the DSML branch did NOT fire (i.e., a negative assertion about which code path was taken). The simplest behavioral contract is "garbled response still recovers", which is covered by `parsed.answer == "recovered-via-generic"`. But the *point* of the test is to verify the routing decision, not the recovery. Two options:
   - Option A (aggressive): drop all three discriminators; the test becomes a sibling of `TestDSMLAllRetriesFail` but with garbled-non-DSML input. Loses the routing-decision claim.
   - Option B (conservative): keep ONE structural discriminator — the negative log-event assertion (`"trailing_tool_call_markup" not in events`) — and drop the two phrase-substring assertions. Preserves the routing claim with one structlog pin.
   - Audit recommendation: Option B. The negative log assertion is the cleanest signal of "this code path did not run", and is less brittle than positive event-name pins because the event-name only needs to exist (not be renamed) for the assertion to remain valid.

2. **`TestMultipleIndependentDSMLRecoveries::test_two_sequential_calls_recover_independently`** — I classified this KEEP, but it does pin `fake1.calls == 3 and fake2.calls == 3`. The per-fake counters are not framework-internal, but the `== 3` is similar to the call-count anti-pattern elsewhere. Two options:
   - Keep as-is (current verdict): the value is "exactly 3 invocations per fake confirms no state leakage across the two invoke_with_tools calls". The contract IS the count.
   - Soften to `>= 3`: less precise but still catches the leakage. Recommendation: keep as-is. The fake is fully scripted; "exactly 3" matches "tool call + DSML + targeted retry = 3 invocations" which mirrors the user-visible recovery shape.

3. **`TestDefaultBudgetExhaustedMessageRendersModelName::test_default_message_includes_output_model_name`** — asserts `content.count("ExplorationResult") == 1`. The `== 1` exact count is mildly brittle (a future change adding a second mention of the model name in the template would fail). The substantive contract is "the model name appears at least once". Two options:
   - Keep as-is: the `== 1` precision is part of the contract (do not redundantly mention).
   - Soften to `>= 1`: easier to maintain.
   - Audit recommendation: keep as-is. The default template is documented and the `== 1` is a sensible guard against accidental duplication.

4. **`TestSkipWhenOnToolNodes::test_node_skipped_when_skip_when_true_on_gather`** — `len(fake_tool.calls) == 0` is the contract ("tool not called"). I kept it. If the maintainer prefers a stricter `not fake_tool.calls` form, that's a stylistic choice not a §6 issue.

5. **`TestCoercingToolWrapperGenerateNotAvailable` and `TestCoercingToolWrapperGenerateRaises`** — both REWRITE candidates. The user contract is "wrapper falls back to empty AIMessage on ValidationError". Both tests already assert that. The structlog event assertions in both tests are arguably valuable for confirming the warning *was* emitted (operators monitor these events). Two options:
   - Drop the event assertions entirely (audit recommendation).
   - Keep the existence-of-event assertion (`assert any(e["event"] == "tool_calls_coercion_generate_failed" for e in cap)`) but drop the payload (`log_level == "warning"`, `"simulated network failure" in event["error"]`) which is the brittle part.
   - Audit recommendation: aggressive drop. The behavioral fallback is the contract; the structlog event name is a quality-of-life signal for operators, not a tested invariant.

---

## Appendix — files NOT in scope

The audit scope was specified as `tests/modes/` and `tests/modifiers/`. Other test files (hypothesis/, decorator/, root-level `test_*.py`) were excluded by the ticket. A brief grep across those directories suggests they may have similar issues (especially `tests/hypothesis/` where property-based testing can pin implementation details), but they're out of scope for neograph-5fqg.

If the maintainer wants a follow-up audit, suggested targets in order of likely yield:
1. `tests/hypothesis/` (property tests can inadvertently pin implementation).
2. `tests/decorator/` (decorator tests often mock internals).
3. `tests/test_forward.py` (ForwardConstruct tracer tests can pin AST internals).
