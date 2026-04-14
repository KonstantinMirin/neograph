# Testing Review — NeoGraph

## High

### TEST-01: TestLLMUnknownToolCall assertion is vacuous
`test_unknown_tool_call_handled` asserts `result["explore"] is not None` but the value is an `AIMessage("ok done")`, not a `Claims` model. The test passes because AIMessage is not None, but the production output is completely wrong — the node returned an AIMessage when it should have returned a Claims instance. The assertion should verify the type: `assert isinstance(result["explore"], Claims)`.

### TEST-02: TestOperator.test_interrupt_on_failure has bare except
The test catches `except Exception: pass` around the interrupt. This means any error (not just GraphInterrupt) is silently swallowed. The test would pass even if interrupt never fired. The resume is commented out. This test proves almost nothing.

## Medium

### TEST-03: FakeLLM structured output doesn't match real behavior
Most FakeLLMs implement `with_structured_output` by setting `self._model` and returning `self`, then `invoke` returns `self._model(items=[...])`. Real LLMs with `include_raw=True` return `{"parsed": model, "raw": AIMessage}`. The fakes return the model directly. This means the `include_raw` path is never exercised by fakes — only by the real observable_pipeline example.

### TEST-04: No test verifies output type correctness
Tests check `result["x"] is not None` and `result["x"].items == [...]` but never verify `isinstance(result["x"], ExpectedModel)`. A node returning the wrong Pydantic model type would pass all current tests.

### TEST-05: Global registry pollution between examples
The examples (01-11) share the same process registries. Running multiple examples sequentially (as in the vs_langgraph suite) pollutes registries. The conftest fixture handles tests but not examples.

## Low

### TEST-06: 8+ FakeLLM classes with identical structure
Could share a base class or factory function.

### TEST-07: configure_llm called inline in 15+ tests
Could be a parameterized fixture.
