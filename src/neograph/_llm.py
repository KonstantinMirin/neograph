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
from collections.abc import Callable
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ValidationError

from neograph.describe_type import describe_type, describe_value
from neograph.errors import ConfigurationError, ExecutionError
from neograph.tool import Tool, ToolBudgetTracker

log = structlog.get_logger()

# Consumer-provided LLM factory
_llm_factory: Callable[[str], Any] | None = None
_llm_factory_params: set[str] | frozenset[str] = set()

# Consumer-provided prompt compiler
_prompt_compiler: Callable[[str, Any], list] | None = None
_prompt_compiler_params: set[str] | frozenset[str] = set()

# Consumer-provided renderer (set via configure_llm(renderer=...))
_global_renderer: Any = None

# Consumer-provided cost callback (set via configure_llm(cost_callback=...))
_cost_callback: Callable | None = None


def _notify_cost(
    tier: str,
    usage: dict | None,
    *,
    node_name: str | None = None,
    mode: str | None = None,
    duration_s: float | None = None,
) -> None:
    """Dispatch cost callback if configured.

    Extra kwargs (node_name, mode, duration_s) are passed through to the
    callback. Old-style callbacks that only accept (tier, input_tokens,
    output_tokens) are supported via TypeError fallback.
    """
    if _cost_callback is None or usage is None:
        return
    kwargs: dict = {
        "tier": tier,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }
    if node_name is not None:
        kwargs["node_name"] = node_name
    if mode is not None:
        kwargs["mode"] = mode
    if duration_s is not None:
        kwargs["duration_s"] = duration_s
    try:
        _cost_callback(**kwargs)
    except TypeError:
        # Old-style callback doesn't accept extra kwargs — fall back
        try:
            _cost_callback(
                tier=tier,
                input_tokens=kwargs["input_tokens"],
                output_tokens=kwargs["output_tokens"],
            )
        except TypeError as exc:
            log.warning("cost_callback_failed", error=str(exc))
    except TypeError as exc:
        log.warning("cost_callback_failed", error=str(exc))


def _get_global_renderer() -> Any:
    """Return the globally configured renderer, or None."""
    return _global_renderer


_ACCEPT_ALL = frozenset({"__all__"})  # sentinel for **kwargs functions


def _accepted_params(fn: Callable) -> set[str] | frozenset[str]:
    """Inspect a callable and return the set of parameter names it accepts.

    If the function accepts **kwargs, returns a sentinel that matches all keys.
    """
    try:
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return _ACCEPT_ALL  # accepts everything
        return set(sig.parameters.keys())
    except (ValueError, TypeError):  # pragma: no cover
        # Builtins, C extensions — assume simple signature
        return set()  # pragma: no cover


def configure_llm(
    llm_factory: Callable,
    prompt_compiler: Callable,
    *,
    renderer: Any = None,
    cost_callback: Callable | None = None,
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
    global _llm_factory, _prompt_compiler, _llm_factory_params, _prompt_compiler_params, _global_renderer, _cost_callback  # noqa: PLW0603
    _llm_factory = llm_factory
    _llm_factory_params = _accepted_params(llm_factory)
    _prompt_compiler = prompt_compiler
    _prompt_compiler_params = _accepted_params(prompt_compiler)
    _global_renderer = renderer
    _cost_callback = cost_callback


def _get_llm(tier: str, node_name: str = "", llm_config: dict | None = None) -> Any:
    if _llm_factory is None:
        raise ConfigurationError.build(
            "LLM not configured",
            hint="Call neograph.configure_llm() first.",
        )
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

    BaseModel values at any resolution stage are BAML-rendered via
    describe_value() instead of using str() (which gives Pydantic repr).
    This makes inline prompt output symmetric with template-ref rendering.
    """
    from pydantic import BaseModel as _BM

    parts = path.split(".")

    if isinstance(input_data, dict):
        if parts[0] not in input_data:
            log.warning(
                "prompt_var_missing",
                var=path,
                available=sorted(input_data.keys()),
            )
        root = input_data.get(parts[0], "")
        rest = parts[1:]
    else:
        root = input_data
        rest = parts[1:]

    obj = root
    for attr in rest:
        if not hasattr(obj, attr):
            log.warning("prompt_var_missing", var=path, segment=attr)
        obj = getattr(obj, attr, "")
    if obj is None:
        return ""
    # BAML-render Pydantic models instead of using repr
    if isinstance(obj, _BM):
        return describe_value(obj)
    return str(obj)


def _substitute_vars(template: str, input_data: Any) -> str:
    """Replace all ``${...}`` placeholders in *template*."""
    return _VAR_RE.sub(lambda m: _resolve_var(m.group(1), input_data), template)


def _compile_prompt(
    template: str,
    input_data: Any,
    *,
    node_name: str = "",
    config: RunnableConfig | dict[str, Any] | None = None,
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
    assert _prompt_compiler is not None, "prompt compiler not configured; call configure_llm() first"
    return _prompt_compiler(template, input_data, **kwargs)


def _extract_json(text: str) -> str:
    """Extract the first balanced JSON value (object or array) from LLM text.

    Finds the first '{' or '[' and tracks depth to find the matching closer.
    Handles markdown fences, thinking tags, prose before/after JSON.
    When both '{' and '[' are present, picks whichever comes first.
    """
    # Find earliest potential JSON start — either '{' or '['
    brace_pos = text.find("{")
    bracket_pos = text.find("[")

    # Try array first if '[' comes before '{' (or '{' is absent)
    if bracket_pos != -1 and (brace_pos == -1 or bracket_pos < brace_pos):
        result = _extract_balanced(text, bracket_pos, "[", "]")
        if result is not None:
            return result

    # Try object extraction
    pos = 0
    while True:
        start = text.find("{", pos)
        if start == -1:
            break

        # Quick check: JSON objects start with {"  or {digit or {[  or { "
        # Prose braces like {a + b} start with a letter after {.
        after_brace = text[start + 1 : start + 2].lstrip()
        if after_brace and after_brace not in ('"', "'", "[", "]", "{", "}", ""):
            pos = start + 1
            continue

        result = _extract_balanced(text, start, "{", "}")
        if result is not None:
            return result

        pos = start + 1

    # Unbalanced — fall back to first-to-last (let json_repair handle it)
    first_open = brace_pos if brace_pos != -1 else bracket_pos
    if first_open != -1:
        close_char = "}" if first_open == brace_pos else "]"
        last = text.rfind(close_char)
        if last > first_open:
            return text[first_open : last + 1]
    return text.strip()


def _extract_balanced(text: str, start: int, open_ch: str, close_ch: str) -> str | None:
    """Extract a balanced JSON value starting at *start* with given delimiters."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _is_list_annotation(annotation: Any) -> bool:
    """Check if a type annotation is a list type (list[X], List[X], etc.)."""
    import typing
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return True
    # Handle Optional[list[X]] → Union[list[X], None]
    if origin is typing.Union:
        args = getattr(annotation, "__args__", ())
        return any(_is_list_annotation(a) for a in args if a is not type(None))
    return annotation is list


def _apply_null_defaults(data: dict, model: type[BaseModel]) -> None:
    """Replace null values with field defaults, recursively.

    Mutates *data* in place. Only applies when the field has an explicit
    default (not PydanticUndefined) and the JSON value is None.
    Also recurses into nested BaseModel fields and list[BaseModel] items.
    """
    from pydantic_core import PydanticUndefined

    for field_name, field_info in model.model_fields.items():
        if field_name not in data:
            continue
        val = data[field_name]

        # Null → default coercion
        if val is None and field_info.default is not PydanticUndefined:
            data[field_name] = field_info.default
            continue

        # Recurse into nested BaseModel
        annotation = field_info.annotation
        if isinstance(val, dict) and isinstance(annotation, type) and issubclass(annotation, BaseModel):
            _apply_null_defaults(val, annotation)
            continue

        # Recurse into list[BaseModel]
        if isinstance(val, list):
            from typing import get_args, get_origin
            origin = get_origin(annotation)
            if origin is list:
                args = get_args(annotation)
                if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                    inner_model = args[0]
                    for item in val:
                        if isinstance(item, dict):
                            _apply_null_defaults(item, inner_model)


def _parse_json_response(text: str, output_model: type[BaseModel]) -> BaseModel:
    """Parse LLM response text into a Pydantic model.

    Uses json_repair to handle common LLM JSON malformations (control
    characters, trailing commas, single quotes, unescaped newlines) before
    passing to Pydantic validation.

    Raises ExecutionError with a clear message when the LLM produces non-JSON
    content (e.g., XML tool-call markup from DeepSeek R1 after budget exhaustion).
    """
    import re

    from json_repair import repair_json

    extracted = _extract_json(text)

    # Detect non-JSON content: empty result or XML-like tool-call markup.
    # DeepSeek R1 emits <｜DSML｜function_calls>... after budget exhaustion.
    stripped = extracted.strip()
    if not stripped or (
        not stripped.startswith(("{", "["))
        and re.search(r"<[^>]*(?:function_call|invoke|DSML)[^>]*>", text, re.IGNORECASE)
    ):
        raise ExecutionError.build(
            "LLM returned non-JSON content instead of structured output",
            expected=f"JSON object for {output_model.__name__}",
            found=f"XML tool-call markup (first 200 chars): {text[:200]!r}",
            hint="Check model compatibility with structured output; some models emit XML tool-call markup after budget exhaustion.",
        )

    repaired = repair_json(extracted, return_objects=False)

    # Coerce null → default for fields that have defaults.
    # LLMs (especially R1 on long prompts) return null for optional-like fields
    # even when the Pydantic type is `str` (not Optional[str]). Pydantic rejects
    # null for str. JSON Schema semantics: null + default → use default.
    import json as _json
    try:
        parsed = _json.loads(repaired)
        if isinstance(parsed, dict) and hasattr(output_model, "model_fields"):
            _apply_null_defaults(parsed, output_model)
            repaired = _json.dumps(parsed)
        elif isinstance(parsed, list) and hasattr(output_model, "model_fields"):
            # Bare array auto-wrap: if the model has exactly one list field,
            # wrap the array into {"field_name": array}.
            list_fields = [
                fname for fname, finfo in output_model.model_fields.items()
                if _is_list_annotation(finfo.annotation)
            ]
            if len(list_fields) == 1:
                wrapped = {list_fields[0]: parsed}
                _apply_null_defaults(wrapped, output_model)
                repaired = _json.dumps(wrapped)
            else:
                raise ExecutionError.build(
                    f"LLM returned a bare JSON array but {output_model.__name__} "
                    f"has {len(list_fields)} list fields (expected exactly 1 for auto-wrap)",
                    expected=f"JSON object for {output_model.__name__}",
                    found=f"bare array (first 200 chars): {repaired[:200]!r}",
                    hint="Either make the LLM return a JSON object, or ensure the model has exactly one list field.",
                )
    except ExecutionError:
        raise
    except (ValueError, TypeError):
        pass  # if json.loads fails, let model_validate_json handle it

    try:
        return output_model.model_validate_json(repaired)
    except ValidationError as exc:
        # Preserve field-level details for error-feedback retry.
        details = "; ".join(f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise ExecutionError.build(
            f"Validation failed for {output_model.__name__}",
            expected=f"valid {output_model.__name__} fields",
            found=details,
            hint="Check that the LLM response matches the output model schema.",
            validation_errors=str(exc),
        ) from exc
    except Exception as exc:
        raise ExecutionError.build(
            f"Failed to parse LLM response as {output_model.__name__}",
            expected=f"valid JSON for {output_model.__name__}",
            found=f"unparseable content (first 200 chars): {repaired[:200]!r}",
            hint=f"Underlying error: {exc}",
        ) from exc


def _build_retry_msg(
    error: ExecutionError,
    output_model: type[BaseModel] | None = None,
) -> str:
    """Build a retry message with validation details and schema.

    Includes the full output schema so the LLM sees the expected structure.
    This is critical when the LLM simplifies nested objects to flat strings
    on long prompts — the schema shows exactly what fields are required.
    """
    schema_block = ""
    if output_model is not None:
        schema_block = (
            f"\n\nExpected output schema:\n{describe_type(output_model, prefix='')}\n"
        )

    details = getattr(error, "validation_errors", None)
    if details:
        return (
            f"Your response failed validation:\n{details}"
            f"{schema_block}\n"
            "Fix these errors and respond with ONLY the corrected JSON object. "
            "No markdown fences, no XML, no explanation. "
            "Every nested object MUST include ALL required fields — do not "
            "simplify objects to plain strings."
        )
    return (
        "Your previous response could not be parsed as valid JSON."
        f"{schema_block}\n"
        "Respond with ONLY the JSON object. No markdown fences, no XML, "
        "no explanation."
    )


def _invoke_json_with_retry(
    llm: Any,
    messages: list,
    output_model: type[BaseModel],
    config: RunnableConfig,
    max_retries: int = 2,
) -> tuple[BaseModel, Any]:
    """Invoke LLM for json_mode/text, with error-feedback retries.

    On parse failure, appends the bad response + validation errors + the
    full output schema and calls the LLM again. The LLM sees exactly what
    fields failed, why, and what the expected structure looks like.
    """
    response = llm.invoke(messages, config=config)
    raw_text = response.content if hasattr(response, "content") else str(response)
    usage = getattr(response, "usage_metadata", None) or {}
    # Accumulate usage across retries
    total_input = usage.get("input_tokens", 0)
    total_output = usage.get("output_tokens", 0)

    for _attempt in range(max_retries):
        try:
            combined_usage = {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_input + total_output,
            } if (total_input or total_output) else usage
            return _parse_json_response(raw_text, output_model), combined_usage
        except ExecutionError as exc:
            retry_messages = messages + [
                {"role": "assistant", "content": raw_text},
                {"role": "user", "content": _build_retry_msg(exc, output_model)},
            ]
            response = llm.invoke(retry_messages, config=config)
            raw_text = response.content if hasattr(response, "content") else str(response)
            retry_usage = getattr(response, "usage_metadata", None) or {}
            total_input += retry_usage.get("input_tokens", 0)
            total_output += retry_usage.get("output_tokens", 0)

    # Final attempt — no more retries, let it raise
    combined_usage = {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
    } if (total_input or total_output) else None
    return _parse_json_response(raw_text, output_model), combined_usage


def _call_structured(
    llm: Any,
    messages: list,
    output_model: type[BaseModel],
    strategy: str,
    config: RunnableConfig,
    max_retries: int = 1,
) -> tuple[BaseModel, Any]:
    """Dispatch structured output by strategy. Returns (result, usage_metadata)."""
    if strategy == "structured":
        usage = None
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
        return result, usage

    if strategy in ("json_mode", "text"):
        return _invoke_json_with_retry(llm, messages, output_model, config, max_retries=max_retries)

    raise ExecutionError.build(
        "Unknown output_strategy",
        expected="'structured', 'json_mode', or 'text'",
        found=repr(strategy),
        hint="Set output_strategy in llm_config to one of the supported values.",
    )


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
        prompt_template,
        input_data,
        node_name=node_name,
        config=config,
        output_model=output_model,
        llm_config=llm_config,
        output_schema=output_schema,
        context=context,
    )

    max_retries = llm_config.get("max_retries", 1)
    t0 = time.monotonic()
    result, usage = _call_structured(llm, messages, output_model, strategy, config, max_retries=max_retries)
    elapsed = time.monotonic() - t0

    usage_info = {}
    if usage:
        usage_info = {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    llm_log.info("llm_call", mode="think", duration_s=round(elapsed, 3), **usage_info)
    _notify_cost(model_tier, usage, node_name=node_name, mode="think", duration_s=round(elapsed, 3))
    return result


def _render_tool_result_for_llm(result: Any, renderer: Any = None) -> str:
    """Render a typed tool result for the LLM's ToolMessage content.

    When ``renderer`` is provided, it is used for Pydantic models and lists
    of models.  Otherwise falls back to ``describe_value`` (BAML-style
    notation with field descriptions as inline ``//`` comments).
    Falls back to str() for non-Pydantic returns.
    """
    from pydantic import BaseModel as _BM

    if isinstance(result, _BM) or (isinstance(result, list) and result and isinstance(result[0], _BM)):
        if renderer is not None:
            return renderer.render(result)
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

    Loop guards (configured via llm_config):

    - ``max_iterations`` (int, default 20): maximum ReAct iterations before
      forcing a final response. When hit, tool calls are skipped and the LLM
      is asked to respond immediately.
    - ``token_budget`` (int or None, default None): if set, abort the loop
      when cumulative input tokens exceed this threshold. Cumulative input
      tokens are the sum of ``input_tokens`` across all LLM responses in
      the loop (each call replays the full conversation, so this tracks
      total cost linearly).

    When ``renderer`` is set, typed tool results (Pydantic models) are
    rendered using ``describe_type`` (schema header) + the renderer
    (instance) instead of raw ``model_dump_json``.
    """
    from langchain_core.messages import ToolMessage

    from neograph._registry import registry

    llm_log = log.bind(
        tier=model_tier,
        prompt=prompt_template,
        tools=[t.name for t in tools],
        budgets={t.name: t.budget for t in tools},
    )

    llm = _get_llm(model_tier, node_name=node_name, llm_config=llm_config)
    messages = list(
        _compile_prompt(
            prompt_template,
            input_data,
            node_name=node_name,
            config=config,
            output_model=output_model,
            llm_config=llm_config,
            context=context,
        )
    )  # copy — the loop appends to this list

    # Create tool instances from registered factories
    tool_instances = {}
    for tool_spec in tools:
        if tool_spec.name not in registry.tool_factory:
            raise ConfigurationError.build(
                f"Tool '{tool_spec.name}' not registered",
                hint="Call register_tool_factory() to register the tool before using it.",
            )
        factory = registry.tool_factory[tool_spec.name]
        tool_instances[tool_spec.name] = factory(config, tool_spec.config)

    active_tools = list(tool_instances.values())
    llm_with_tools = llm.bind_tools(active_tools)

    from neograph.tool import ToolInteraction

    llm_config = llm_config or {}
    max_iterations = llm_config.get("max_iterations", 20)
    token_budget = llm_config.get("token_budget")  # None = no limit
    cumulative_input_tokens = 0

    loop_count = 0
    total_tool_calls = 0
    tool_interactions: list[ToolInteraction] = []
    _guard_fired = False
    t0 = time.monotonic()

    while True:
        loop_count += 1
        response = llm_with_tools.invoke(messages, config=config)
        messages.append(response)

        # Track token usage across iterations
        response_usage = getattr(response, "usage_metadata", None) or {}
        cumulative_input_tokens += response_usage.get("input_tokens", 0)

        if not response.tool_calls:
            llm_log.debug("react_final_response", loop=loop_count)
            break

        # Safety break: if the guard already fired (tools unbound) but the LLM
        # still returns tool_calls, force-break instead of looping forever.
        if _guard_fired:
            llm_log.warning("react_guard_forced_break", loops=loop_count, tool_calls=total_tool_calls)
            break

        # Check iteration and token budget guards
        max_iter_hit = loop_count >= max_iterations
        budget_hit = token_budget is not None and cumulative_input_tokens > token_budget

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]

            # If we've hit a loop guard, skip execution and tell LLM to wrap up
            if max_iter_hit or budget_hit:
                messages.append(
                    ToolMessage(
                        content=(
                            f"React loop limit reached"
                            f" ({'max iterations' if max_iter_hit else 'token budget'})."
                            f" Provide your final answer now."
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            if not budget_tracker.can_call(tool_name):
                llm_log.info("tool_budget_exhausted", tool=tool_name)
                messages.append(
                    ToolMessage(
                        content=f"Tool '{tool_name}' budget exhausted. Use remaining tools or respond.",
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            tool_fn = tool_instances.get(tool_name)
            if tool_fn is None:
                messages.append(
                    ToolMessage(
                        content=f"Unknown tool: {tool_name}",
                        tool_call_id=tool_call["id"],
                    )
                )
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

            tool_interactions.append(
                ToolInteraction(
                    tool_name=tool_name,
                    args=tool_call.get("args", {}),
                    result=rendered,
                    typed_result=result,
                    duration_ms=int(tool_elapsed * 1000),
                )
            )

            messages.append(
                ToolMessage(
                    content=rendered,
                    tool_call_id=tool_call["id"],
                )
            )

        # Unbind tools if a loop guard was hit — next iteration forces final response
        if max_iter_hit or budget_hit:
            reason = (
                "max_iterations+token_budget"
                if max_iter_hit and budget_hit
                else ("max_iterations" if max_iter_hit else "token_budget")
            )
            llm_log.warning(
                f"react_{reason}_exceeded",
                max_iterations=max_iterations,
                token_budget=token_budget,
                cumulative_input_tokens=cumulative_input_tokens,
                loops=loop_count,
                tool_calls=total_tool_calls,
            )
            llm_with_tools = llm
            _guard_fired = True
            continue

        # Check if all budgeted tools are spent
        if budget_tracker.all_exhausted():
            llm_log.info("all_tools_exhausted", exhausted=budget_tracker.exhausted_tools(), forcing_response=True)
            llm_with_tools = llm  # unbind tools, force final response
        else:
            # Rebind with only tools that still have budget
            active_tools = [tool_instances[t.name] for t in tools if budget_tracker.can_call(t.name)]
            llm_with_tools = llm.bind_tools(active_tools)

    elapsed = time.monotonic() - t0

    # Parse final response as structured output — strategy-aware
    strategy = llm_config.get("output_strategy", "structured")

    max_retries = llm_config.get("max_retries", 1)

    if strategy in ("json_mode", "text"):
        # Already have the final response in messages[-1] — try parsing it,
        # with error-feedback retry via the shared helper.
        last_msg = messages[-1]
        raw_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        try:
            parse_result = _parse_json_response(raw_text, output_model)
        except ExecutionError:
            parse_result, _ = _invoke_json_with_retry(llm, messages, output_model, config, max_retries=max_retries)
        usage = getattr(last_msg, "usage_metadata", None)
    else:
        parse_result, usage = _call_structured(llm, messages, output_model, strategy, config, max_retries=max_retries)

    # Collect total usage from all LLM calls in the loop
    total_input_tokens = 0
    total_output_tokens = 0
    for msg in messages:
        msg_usage = getattr(msg, "usage_metadata", None)
        if msg_usage:
            total_input_tokens += msg_usage.get("input_tokens", 0)
            total_output_tokens += msg_usage.get("output_tokens", 0)
    # Add final structured parse call ONLY for the structured strategy path,
    # which makes a separate LLM call not in the messages list.
    # json_mode/text path reuses messages[-1] — already counted above.
    if usage and strategy not in ("json_mode", "text"):
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
    _notify_cost(model_tier, usage_info if usage_info else None,
                 node_name=node_name, mode="react", duration_s=round(elapsed, 3))
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
        raise ConfigurationError.build(
            "Prompt compiler not configured",
            hint="Call neograph.configure_llm() first.",
        )

    # Apply renderer dispatch via RenderedInput — matches runtime path
    from neograph.renderers import build_rendered_input
    prompt = getattr(node, "prompt", "") or ""
    effective_renderer = getattr(node, "renderer", None) or _global_renderer
    ri = build_rendered_input(input_data, renderer=effective_renderer)
    if _is_inline_prompt(prompt):
        input_data = ri.raw
    else:
        input_data = ri.for_template_ref

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
