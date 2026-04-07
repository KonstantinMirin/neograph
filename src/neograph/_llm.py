"""LLM invocation layer — thin abstraction over LangChain chat models.

This module is intentionally minimal. NeoGraph does NOT own the LLM client
configuration — consumers register their model routing. This module provides
the dispatch mechanics (structured output, ReAct loop, tool budget enforcement).

Observability: callbacks (e.g. Langfuse CallbackHandler) flow through via
RunnableConfig. If the consumer wires them at the run() call site, every
LLM call here inherits them automatically. If not, nothing happens.
Structlog captures the framework-level view regardless.
"""

from __future__ import annotations

import inspect
import re
import time
from typing import Any, Callable

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph.errors import ConfigurationError, ExecutionError
from neograph.tool import Tool, ToolBudgetTracker

log = structlog.get_logger()

# Consumer-provided LLM factory
_llm_factory: Callable[[str], Any] | None = None
_llm_factory_params: set[str] = set()

# Consumer-provided prompt compiler
_prompt_compiler: Callable[[str, Any], list] | None = None
_prompt_compiler_params: set[str] = set()

# Consumer-provided renderer (set via configure_llm(renderer=...))
_global_renderer: Any = None


def _get_global_renderer() -> Any:
    """Return the globally configured renderer, or None."""
    return _global_renderer


_ACCEPT_ALL = frozenset({"__all__"})  # sentinel for **kwargs functions


def _accepted_params(fn: Callable) -> set[str]:
    """Inspect a callable and return the set of parameter names it accepts.

    If the function accepts **kwargs, returns a sentinel that matches all keys.
    """
    try:
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return _ACCEPT_ALL  # accepts everything
        return set(sig.parameters.keys())
    except (ValueError, TypeError):
        # Builtins, C extensions — assume simple signature
        return set()


def configure_llm(
    llm_factory: Callable,
    prompt_compiler: Callable,
    *,
    renderer: Any = None,
) -> None:
    """Configure NeoGraph's LLM layer.

    Args:
        llm_factory: Creates LLM instances per node.
            Simple:   (tier) → BaseChatModel
            Advanced: (tier, node_name=, llm_config=) → BaseChatModel

        prompt_compiler: Builds message lists for LLM calls.
            Simple:   (template, input_data) → list[BaseMessage]
            Advanced: (template, input_data, node_name=, config=) → list[BaseMessage]
            The config contains everything from run()'s input + config["configurable"],
            so the compiler can access node_id, project_root, shared resources, etc.

        renderer: Global renderer for input data. Lowest priority in the
            dispatch hierarchy (model method > node.renderer > global).

    Usage:
        # Simple
        configure_llm(
            llm_factory=lambda tier: ChatOpenAI(model=MODELS[tier]),
            prompt_compiler=lambda template, data: [HumanMessage(content=str(data))],
        )

        # Production: full context access
        def my_compiler(template, data, *, node_name=None, config=None):
            node_id = config["configurable"]["node_id"]
            project_root = config["configurable"]["project_root"]
            return get_generator_prompt(
                atom_type=template,
                node_id=node_id,
                context_files=load_context(project_root, node_id),
                analysis_notes=format_notes(data),
            )

        def my_factory(tier, node_name=None, llm_config=None):
            return ChatOpenAI(
                model=MODELS[tier],
                temperature=(llm_config or {}).get("temperature", 0),
            )

        configure_llm(llm_factory=my_factory, prompt_compiler=my_compiler)
    """
    global _llm_factory, _prompt_compiler, _llm_factory_params, _prompt_compiler_params, _global_renderer  # noqa: PLW0603
    _llm_factory = llm_factory
    _llm_factory_params = _accepted_params(llm_factory)
    _prompt_compiler = prompt_compiler
    _prompt_compiler_params = _accepted_params(prompt_compiler)
    _global_renderer = renderer


def _get_llm(tier: str, node_name: str = "", llm_config: dict | None = None) -> Any:
    if _llm_factory is None:
        msg = "LLM not configured. Call neograph.configure_llm() first."
        raise ConfigurationError(msg)
    all_kwargs = {"node_name": node_name, "llm_config": llm_config or {}}
    if _llm_factory_params is _ACCEPT_ALL:
        kwargs = all_kwargs
    else:
        kwargs = {k: v for k, v in all_kwargs.items() if k in _llm_factory_params}
    return _llm_factory(tier, **kwargs)


def _is_inline_prompt(template: str) -> bool:
    """Detect whether a prompt template is inline text vs a file reference.

    Inline text contains a space character or a ``${`` substitution marker.
    Everything else (e.g. ``"rw/summarize"``) is treated as a file reference
    and delegated to the consumer-provided prompt compiler.
    """
    return " " in template or "${" in template


_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_var(path: str, input_data: Any) -> str:
    """Resolve a single ``${path}`` variable against *input_data*.

    *path* may be a plain name (``claim``) or dotted (``claim.text``).

    When *input_data* is a dict the first segment is looked up as a key;
    when it is a single value (non-dict) the whole value is used as the root,
    and subsequent segments are resolved via ``getattr``.
    """
    parts = path.split(".")

    if isinstance(input_data, dict):
        root = input_data.get(parts[0], "")
        rest = parts[1:]
    else:
        root = input_data
        rest = parts[1:]

    obj = root
    for attr in rest:
        obj = getattr(obj, attr, "")
    return str(obj)


def _substitute_vars(template: str, input_data: Any) -> str:
    """Replace all ``${...}`` placeholders in *template*."""
    return _VAR_RE.sub(lambda m: _resolve_var(m.group(1), input_data), template)


def _compile_prompt(
    template: str,
    input_data: Any,
    *,
    node_name: str = "",
    config: dict | None = None,
    output_model: type[BaseModel] | None = None,
    llm_config: dict | None = None,
    output_schema: str | None = None,
    context: dict[str, Any] | None = None,
) -> list:
    # Inline prompt — resolve ${} variables and return directly
    if _is_inline_prompt(template):
        rendered = _substitute_vars(template, input_data)
        return [{"role": "user", "content": rendered}]

    all_kwargs = {
        "node_name": node_name,
        "config": config,
        "output_model": output_model,
        "llm_config": llm_config,
        "output_schema": output_schema,
    }
    if context is not None:
        all_kwargs["context"] = context
    # Only pass kwargs the compiler accepts — inspected at configure_llm() time
    if _prompt_compiler_params is _ACCEPT_ALL:
        kwargs = all_kwargs
    else:
        kwargs = {k: v for k, v in all_kwargs.items() if k in _prompt_compiler_params}
    return _prompt_compiler(template, input_data, **kwargs)


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response text — strips markdown fences, finds JSON object."""
    import re

    # Strip markdown code fences
    cleaned = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```\s*$', '', cleaned, flags=re.MULTILINE)

    # Try the cleaned text directly
    cleaned = cleaned.strip()
    if cleaned.startswith('{'):
        return cleaned

    # Find first { ... } block in the text
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
    if match:
        return match.group(0)

    return cleaned


def _parse_json_response(text: str, output_model: type[BaseModel]) -> BaseModel:
    """Parse a JSON string into a Pydantic model.

    Uses json_repair to handle common LLM JSON malformations (control
    characters, trailing commas, single quotes, unescaped newlines) before
    passing to Pydantic validation.
    """
    from json_repair import repair_json

    extracted = _extract_json(text)
    repaired = repair_json(extracted, return_objects=False)
    return output_model.model_validate_json(repaired)


def _call_structured(
    llm: Any,
    messages: list,
    output_model: type[BaseModel],
    strategy: str,
    config: RunnableConfig,
) -> tuple[BaseModel, Any]:
    """Dispatch structured output by strategy. Returns (result, usage_metadata)."""
    usage = None

    if strategy == "structured":
        try:
            structured_llm = llm.with_structured_output(output_model, include_raw=True)
            raw_result = structured_llm.invoke(messages, config=config)
            if isinstance(raw_result, dict) and "parsed" in raw_result:
                result = raw_result["parsed"]
                raw_msg = raw_result.get("raw")
                usage = getattr(raw_msg, "usage_metadata", None) if raw_msg else None
            else:
                result = raw_result
        except TypeError:
            structured_llm = llm.with_structured_output(output_model)
            result = structured_llm.invoke(messages, config=config)

    elif strategy in ("json_mode", "text"):
        response = llm.invoke(messages, config=config)
        raw_text = response.content if hasattr(response, "content") else str(response)
        usage = getattr(response, "usage_metadata", None)
        result = _parse_json_response(raw_text, output_model)

    else:
        msg = f"Unknown output_strategy: {strategy}. Use 'structured', 'json_mode', or 'text'."
        raise ExecutionError(msg)

    return result, usage


def invoke_structured(
    model_tier: str,
    prompt_template: str,
    input_data: Any,
    output_model: type[BaseModel],
    config: RunnableConfig,
    node_name: str = "",
    llm_config: dict | None = None,
    context: dict[str, Any] | None = None,
) -> BaseModel:
    """Single LLM call with structured JSON output. Mode: produce.

    Output strategy (from llm_config["output_strategy"]):
        "structured" — llm.with_structured_output(model) (default, widest LangChain support)
        "json_mode"  — inject schema into prompt, LLM returns raw JSON, framework parses
        "text"       — LLM returns plain text, framework extracts and parses JSON from it
    """
    llm_config = llm_config or {}
    strategy = llm_config.get("output_strategy", "structured")
    llm_log = log.bind(tier=model_tier, prompt=prompt_template, output=output_model.__name__, strategy=strategy)

    output_schema = None
    if strategy == "json_mode":
        from neograph.describe_type import describe_type

        output_schema = describe_type(output_model)

    llm = _get_llm(model_tier, node_name=node_name, llm_config=llm_config)
    messages = _compile_prompt(
        prompt_template, input_data,
        node_name=node_name, config=config,
        output_model=output_model, llm_config=llm_config,
        output_schema=output_schema,
        context=context,
    )

    t0 = time.monotonic()
    result, usage = _call_structured(llm, messages, output_model, strategy, config)
    elapsed = time.monotonic() - t0

    usage_info = {}
    if usage:
        usage_info = {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    llm_log.info("llm_call", mode="think", duration_s=round(elapsed, 3), **usage_info)
    return result


def _render_tool_result_for_llm(result: Any, renderer: Any = None) -> str:
    """Render a typed tool result for the LLM's ToolMessage content.

    When ``renderer`` is provided, it is used for Pydantic models and lists
    of models.  Otherwise falls back to ``describe_value`` (BAML-style
    notation with field descriptions as inline ``//`` comments).
    Falls back to str() for non-Pydantic returns.
    """
    from pydantic import BaseModel as _BM

    if isinstance(result, _BM) or (
        isinstance(result, list) and result and isinstance(result[0], _BM)
    ):
        if renderer is not None:
            return renderer.render(result)
        from neograph.describe_type import describe_value
        return describe_value(result, prefix="Tool result:")

    return str(result)


def invoke_with_tools(
    model_tier: str,
    prompt_template: str,
    input_data: Any,
    output_model: type[BaseModel],
    tools: list[Tool],
    budget_tracker: ToolBudgetTracker,
    config: RunnableConfig,
    node_name: str = "",
    llm_config: dict | None = None,
    renderer: Any = None,
    context: dict[str, Any] | None = None,
) -> tuple[BaseModel | None, list]:
    """ReAct tool loop with per-tool budget enforcement. Mode: agent/act.

    Returns (parsed_result, tool_interactions) where tool_interactions is a
    list of ToolInteraction records from the ReAct loop.

    When ``renderer`` is set, typed tool results (Pydantic models) are
    rendered using ``describe_type`` (schema header) + the renderer
    (instance) instead of raw ``model_dump_json``.
    """
    from langchain_core.messages import ToolMessage

    from neograph.factory import _tool_factory_registry

    llm_log = log.bind(
        tier=model_tier,
        prompt=prompt_template,
        tools=[t.name for t in tools],
        budgets={t.name: t.budget for t in tools},
    )

    llm = _get_llm(model_tier, node_name=node_name, llm_config=llm_config)
    messages = list(_compile_prompt(
        prompt_template, input_data,
        node_name=node_name, config=config,
        output_model=output_model, llm_config=llm_config,
        context=context,
    ))  # copy — the loop appends to this list

    # Create tool instances from registered factories
    tool_instances = {}
    for tool_spec in tools:
        if tool_spec.name not in _tool_factory_registry:
            msg = f"Tool '{tool_spec.name}' not registered. Use register_tool_factory()."
            raise ConfigurationError(msg)
        factory = _tool_factory_registry[tool_spec.name]
        tool_instances[tool_spec.name] = factory(config, tool_spec.config)

    active_tools = list(tool_instances.values())
    llm_with_tools = llm.bind_tools(active_tools)

    from neograph.tool import ToolInteraction

    loop_count = 0
    total_tool_calls = 0
    tool_interactions: list[ToolInteraction] = []
    t0 = time.monotonic()

    while True:
        loop_count += 1
        response = llm_with_tools.invoke(messages, config=config)
        messages.append(response)

        if not response.tool_calls:
            llm_log.debug("react_final_response", loop=loop_count)
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]

            if not budget_tracker.can_call(tool_name):
                llm_log.info("tool_budget_exhausted", tool=tool_name)
                messages.append(ToolMessage(
                    content=f"Tool '{tool_name}' budget exhausted. Use remaining tools or respond.",
                    tool_call_id=tool_call["id"],
                ))
                continue

            tool_fn = tool_instances.get(tool_name)
            if tool_fn is None:
                messages.append(ToolMessage(
                    content=f"Unknown tool: {tool_name}",
                    tool_call_id=tool_call["id"],
                ))
                continue

            tool_t0 = time.monotonic()
            result = tool_fn.invoke(tool_call["args"])
            tool_elapsed = time.monotonic() - tool_t0

            budget_tracker.record_call(tool_name)
            total_tool_calls += 1
            llm_log.debug(
                "tool_call",
                tool=tool_name,
                call_num=total_tool_calls,
                duration_s=round(tool_elapsed, 3),
            )

            # Render for LLM: Pydantic → describe_type schema + renderer,
            # list[Pydantic] → schema + rendered items, else str()
            from pydantic import BaseModel as _BM
            if isinstance(result, _BM):
                rendered = _render_tool_result_for_llm(result, renderer)
            elif isinstance(result, list) and result and isinstance(result[0], _BM):
                rendered = _render_tool_result_for_llm(result, renderer)
            else:
                rendered = str(result)

            tool_interactions.append(ToolInteraction(
                tool_name=tool_name,
                args=tool_call.get("args", {}),
                result=rendered,
                typed_result=result,
                duration_ms=int(tool_elapsed * 1000),
            ))

            messages.append(ToolMessage(
                content=rendered,
                tool_call_id=tool_call["id"],
            ))

        # Check if all budgeted tools are spent
        if budget_tracker.all_exhausted():
            llm_log.info("all_tools_exhausted",
                         exhausted=budget_tracker.exhausted_tools(),
                         forcing_response=True)
            llm_with_tools = llm  # unbind tools, force final response
        else:
            # Rebind with only tools that still have budget
            active_tools = [
                tool_instances[t.name]
                for t in tools
                if budget_tracker.can_call(t.name)
            ]
            llm_with_tools = llm.bind_tools(active_tools)

    elapsed = time.monotonic() - t0

    # Parse final response as structured output — strategy-aware
    llm_config = llm_config or {}
    strategy = llm_config.get("output_strategy", "structured")

    if strategy in ("json_mode", "text"):
        # Already have the final response in messages[-1] — parse directly
        last_msg = messages[-1]
        raw_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        parse_result = _parse_json_response(raw_text, output_model)
        usage = getattr(last_msg, "usage_metadata", None)
    else:
        parse_result, usage = _call_structured(llm, messages, output_model, strategy, config)

    # Collect total usage from all LLM calls in the loop
    total_input_tokens = 0
    total_output_tokens = 0
    for msg in messages:
        msg_usage = getattr(msg, "usage_metadata", None)
        if msg_usage:
            total_input_tokens += msg_usage.get("input_tokens", 0)
            total_output_tokens += msg_usage.get("output_tokens", 0)
    # Add final structured parse call
    if usage:
        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)

    usage_info = {}
    if total_input_tokens or total_output_tokens:
        usage_info = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }

    llm_log.info(
        "llm_call",
        mode="react",
        loops=loop_count,
        tool_calls=total_tool_calls,
        duration_s=round(elapsed, 3),
        output=output_model.__name__,
        **usage_info,
    )
    return parse_result, tool_interactions


def render_prompt(
    node: Any,
    input_data: Any,
    *,
    config: dict | None = None,
) -> str:
    """Render the exact prompt a node would send to the LLM, without calling it.

    Applies the renderer dispatch hierarchy (node.renderer > global > None),
    compiles via the registered prompt_compiler, and formats messages as a
    readable string. Useful for prompt engineering and debugging.

    Args:
        node: A Node instance with prompt, model, and output fields.
        input_data: The input data (Pydantic model, dict, or primitive).
        config: Optional RunnableConfig-style dict for the prompt compiler.

    Returns:
        A human-readable string of the compiled messages.
    """
    if _prompt_compiler is None:
        msg = "Prompt compiler not configured. Call neograph.configure_llm() first."
        raise ConfigurationError(msg)

    # Apply renderer dispatch: node.renderer > global > None
    effective_renderer = getattr(node, "renderer", None) or _global_renderer
    if effective_renderer is not None:
        from neograph.renderers import render_input

        input_data = render_input(input_data, renderer=effective_renderer)

    # Generate output_schema for json_mode
    output_schema = None
    llm_config = getattr(node, "llm_config", None) or {}
    strategy = llm_config.get("output_strategy", "structured")
    output_model = getattr(node, "outputs", None)
    if strategy == "json_mode" and output_model is not None:
        from neograph.describe_type import describe_type

        output_schema = describe_type(output_model)

    messages = _compile_prompt(
        getattr(node, "prompt", "") or "",
        input_data,
        node_name=getattr(node, "name", ""),
        config=config,
        output_model=output_model,
        llm_config=llm_config,
        output_schema=output_schema,
    )

    # Format messages as a readable string (supports both LangChain message
    # objects and plain dicts from simple prompt_compilers).
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", str(msg))
        else:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", str(msg))
        parts.append(f"[{role}]\n{content}")

    return "\n\n".join(parts)
