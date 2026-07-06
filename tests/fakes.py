"""Shared test fakes — FakeLLM variants and helpers.

Three fake LLM patterns cover all production scenarios:

StructuredFake  — implements with_structured_output() for produce mode.
                  The `respond` callable is called with the output model
                  and should return an instance of it.

ReActFake       — implements bind_tools() + invoke() with a scriptable
                  sequence of responses. For gather/execute mode tests.

TextFake        — returns plain text via invoke(). For json_mode and
                  text output strategies where the framework parses.

Also provides helpers:

configure_fake_llm(factory, prompt="test") — one-line configure_llm
setup with a simple prompt compiler.
"""

from __future__ import annotations

from collections.abc import Callable

# Post-§2 (ticket ezqz): `configure_llm` no longer exists. Tests pass LLM
# configuration as kwargs to `compile()`. Helpers below return those kwargs
# instead of mutating module state.
#
# `register_scripted`/`register_condition`/`register_tool_factory` were also
# removed from src/neograph/. They now live HERE as test-only convenience
# that writes to test-local dicts. compile() reads them via the `scripted=`,
# `conditions=`, `tool_factories=` kwargs (tests splat via
# `compile(c, **build_test_compile_kwargs())` or pass dicts explicitly).
from collections.abc import Callable as _Callable
from typing import Any

# dyp3: the fake-LLM contract now lives in the public package. Re-export the
# migrated doubles so the 122 `from tests.fakes import ...` sites keep working
# with ZERO churn AND resolve to the SAME object as neograph.testing.fakes (one
# implementation, two consumers). Do NOT redefine these here.
from neograph.testing.fakes import (  # noqa: F401  (re-export shim)
    FakeLLM,
    FakeTool,
    GatedAsyncFake,
    GuardFake,
    ReActFake,
    StringArgsFake,
    StructuredFake,
    StructuredFakeWithRaw,
    StubbornFake,
    TextFake,
    _final_json_content,
    event_loop_lag_watchdog,
    install_fake_llm,
)

_TEST_SCRIPTED: dict[str, _Callable] = {}
_TEST_CONDITIONS: dict[str, _Callable] = {}
_TEST_TOOL_FACTORIES: dict[str, _Callable] = {}




def register_scripted(name: str, fn: _Callable) -> None:
    """Test-side scripted-shim registration (post-§2).

    Writes into a test-local dict in `tests/fakes.py`; the autouse fixture in
    `tests/conftest.py` resets these dicts between tests. Tests pass the
    accumulated registrations to compile() via `**build_test_compile_kwargs()`.
    """
    _TEST_SCRIPTED[name] = fn


def register_condition(name: str, fn: _Callable) -> None:
    """Test-side condition registration. See `register_scripted` for context."""
    _TEST_CONDITIONS[name] = fn


def register_tool_factory(name: str, fn: _Callable) -> None:
    """Test-side tool-factory registration. See `register_scripted` for context."""
    _TEST_TOOL_FACTORIES[name] = fn


def build_test_compile_kwargs(**extra) -> dict[str, Any]:
    """Build compile() kwargs from the test-local registration dicts.

    Tests that registered scripted/conditions/tool_factories pass the result
    to `compile(c, **build_test_compile_kwargs())`. Extra kwargs (e.g.
    `llm_factory=...`) are merged on top.
    """
    out: dict[str, Any] = {}
    if _TEST_SCRIPTED:
        out["scripted"] = dict(_TEST_SCRIPTED)
    if _TEST_CONDITIONS:
        out["conditions"] = dict(_TEST_CONDITIONS)
    if _TEST_TOOL_FACTORIES:
        out["tool_factories"] = dict(_TEST_TOOL_FACTORIES)
    out.update(extra)
    return out


def reset_test_registry() -> None:
    """Clear the test-local registration dicts. Called by conftest fixture."""
    _TEST_SCRIPTED.clear()
    _TEST_CONDITIONS.clear()
    _TEST_TOOL_FACTORIES.clear()


def lookup_scripted(name: str) -> _Callable:
    """Test-only mirror of the removed src/ helper.

    Looks up in both the test-local dict AND the decoration-time registry
    (which holds inline body-merge / @merge_fn / interrupt shims).
    """
    from neograph._runtime_registry import _decoration_registry
    from neograph.errors import ConfigurationError
    fn = _TEST_SCRIPTED.get(name) or _decoration_registry.scripted.get(name)
    if fn is None:
        raise ConfigurationError.build(
            f"Scripted function '{name}' not registered",
            hint="Use tests.fakes.register_scripted() or pass scripted= to compile().",
        )
    return fn


def lookup_condition(name: str) -> _Callable:
    """Test-only mirror of the removed src/ helper."""
    from neograph._runtime_registry import _decoration_registry
    from neograph.errors import ConfigurationError
    fn = _TEST_CONDITIONS.get(name) or _decoration_registry.condition.get(name)
    if fn is None:
        raise ConfigurationError.build(
            f"Condition '{name}' not registered",
            hint="Use tests.fakes.register_condition() or pass conditions= to compile().",
        )
    return fn




# The fake DOUBLES (StructuredFake, StructuredFakeWithRaw, ReActFake,
# StringArgsFake, TextFake, FakeTool, GuardFake, StubbornFake, GatedAsyncFake,
# event_loop_lag_watchdog, _final_json_content) moved to
# src/neograph/testing/fakes.py (dyp3) and are re-exported at the top of this
# module. Only the test-only registry plumbing + compile-kwargs helpers remain.


# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _default_fake_prompt_compiler(template, data, **kw):
    """A minimal prompt compiler that returns a single user message.
    Test sites that need a custom compiler pass their own."""
    return [{"role": "user", "content": "test"}]


def build_fake_llm_kwargs(
    factory: Callable,
    prompt_compiler: Callable | None = None,
    *,
    cost_callback: Callable | None = None,
    renderer: Any = None,
) -> dict[str, Any]:
    """Build the LLM kwargs dict that gets splatted into `compile(c, **kwargs)`.

    Replaces the old `configure_fake_llm(...)` (which mutated module state).
    Test sites that previously did::

        configure_fake_llm(lambda tier: fake)
        graph = compile(pipeline)

    now do::

        graph = compile(pipeline, **build_fake_llm_kwargs(lambda tier: fake))
    """
    pc = prompt_compiler if prompt_compiler is not None else _default_fake_prompt_compiler
    kwargs: dict[str, Any] = {
        "llm_factory": factory,
        "prompt_compiler": pc,
    }
    if cost_callback is not None:
        kwargs["cost_callback"] = cost_callback
    if renderer is not None:
        kwargs["renderer"] = renderer
    return kwargs


# Backward-compat alias to ease the migration. New code prefers
# `build_fake_llm_kwargs(...)`. The name `configure_fake_llm` no longer
# mutates anything — it now returns the same kwargs dict.
def configure_fake_llm(
    factory: Callable,
    prompt_compiler: Callable | None = None,
) -> dict[str, Any]:
    """Migration alias for `build_fake_llm_kwargs`.

    Old shape mutated module state via `configure_llm`. New shape returns
    a kwargs dict — every old call site must add `**` to splat the result
    into `compile()` or accept the dict and pass it explicitly.
    """
    return build_fake_llm_kwargs(factory, prompt_compiler)


def build_fake_runtime(
    factory: Callable | None = None,
    prompt_compiler: Callable | None = None,
    *,
    cost_callback: Callable | None = None,
    renderer: Any = None,
) -> Any:
    """Build an `LlmRuntime` backed by fakes — for tests that call helpers
    like `invoke_structured(runtime, ...)` directly without going through
    `compile()`.
    """
    from neograph._llm_runtime import LlmRuntime

    if factory is None:
        factory = lambda tier: StructuredFake(lambda m: m())  # noqa: E731
    if prompt_compiler is None:
        prompt_compiler = _default_fake_prompt_compiler
    return LlmRuntime.build(
        llm_factory=factory,
        prompt_compiler=prompt_compiler,
        cost_callback=cost_callback,
        renderer=renderer,
    )


def build_fake_tool_lookup() -> dict[str, _Callable]:
    """Snapshot of the test-local tool-factory registry.

    For tests that call `invoke_with_tools(...)` directly (without going
    through `compile()`), pass the result as `tool_factory_lookup=`. Mirrors
    what `compile()` would have built from `tool_factories=` kwarg.
    """
    return dict(_TEST_TOOL_FACTORIES)


def drive_agent_via_cycle(
    *args: Any,
    model_tier: str | None = None,
    prompt_template: str | None = None,
    input_data: Any = None,
    output_model: Any = None,
    tools: list | None = None,
    budget_tracker: Any = None,  # ignored — the cycle derives budget from tools
    config: Any = None,
    node_name: str = "agent",
    llm_config: Any = None,
    renderer: Any = None,
    context: Any = None,
    runtime: Any = None,
    tool_factory_lookup: dict[str, _Callable] | None = None,
) -> tuple[Any, list]:
    """Test driver for the inline agent cycle (neograph-m6d3.3).

    Drop-in replacement for the deleted ``_tool_loop.invoke_with_tools``: builds a
    single agent node, compiles it to the real inline ReAct cycle (agent/tools/
    parse), runs it, and returns ``(result, tool_interactions)`` — the same shape
    the monolith returned. Lets the tool-loop unit tests keep their assertions
    verbatim while exercising the NEW mechanism end-to-end through run().

    ``budget_tracker`` is accepted for signature compatibility but ignored (the
    cycle derives budget from each Tool's ``budget``). ``context=`` is not
    supported here — migrate such tests to a full compile()/run() with an upstream
    context producer.
    """
    from langgraph.checkpoint.memory import MemorySaver

    from neograph import Construct, Node, ToolInteraction, compile, run
    from neograph._llm_runtime import LlmRuntime

    if args and isinstance(args[0], LlmRuntime):
        runtime = args[0]
    if context is not None:
        raise NotImplementedError(
            "drive_agent_via_cycle: context= not supported — migrate to compile()/run() "
            "with an upstream context producer node."
        )

    node_kwargs: dict[str, Any] = {
        "name": node_name or "agent",
        "mode": "agent",
        "outputs": {"result": output_model, "tool_log": list[ToolInteraction]},
        "model": model_tier or "fast",
        "prompt": prompt_template or "test",
        "tools": tools or [],
    }
    if llm_config is not None:
        node_kwargs["llm_config"] = llm_config
    if renderer is not None:
        node_kwargs["renderer"] = renderer
    node = Node(**node_kwargs)
    pipeline = Construct("agent_unit", nodes=[node])

    ckwargs: dict[str, Any] = {}
    if runtime is not None:
        ckwargs["_runtime"] = runtime
    if tool_factory_lookup:
        ckwargs["tool_factories"] = tool_factory_lookup

    graph = compile(pipeline, checkpointer=MemorySaver(), **ckwargs)

    run_input = input_data if isinstance(input_data, dict) else {}
    run_config = {**(config or {})}
    run_config.setdefault("configurable", {})
    run_config["configurable"] = {**run_config["configurable"]}
    run_config["configurable"].setdefault("thread_id", "agent-unit")

    result = run(graph, input=run_input, config=run_config)
    field = node_name or "agent"
    return result.get(f"{field}_result"), result.get(f"{field}_tool_log", []) or []
