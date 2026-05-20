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

import re
import time
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ValidationError

from neograph._image import resolve_image
from neograph._llm_config import LlmConfig
from neograph._llm_protocols import CostCallback, LlmFactory, PromptCompiler  # noqa: F401 — re-exported
from neograph._llm_runtime import (
    _ACCEPT_ALL,
    EMPTY_RUNTIME,
    LlmRuntime,
    _accepted_params,  # noqa: F401 — re-exported
)
from neograph.describe_type import describe_type, describe_value
from neograph.errors import ConfigurationError, ExecutionError

log = structlog.get_logger()


def _notify_cost(
    *args: Any,
    runtime: LlmRuntime | None = None,
    node_name: str | None = None,
    mode: str | None = None,
    duration_s: float | None = None,
) -> None:
    """Dispatch cost callback if configured.

    Extra kwargs (node_name, mode, duration_s) are passed through to the
    callback. Old-style callbacks that only accept (tier, input_tokens,
    output_tokens) are supported via TypeError fallback.

    Accepts both call shapes:
        _notify_cost(runtime, tier, usage, ...)  # post-§2
        _notify_cost(tier, usage, ...)           # legacy — uses compat runtime
    """
    if args and isinstance(args[0], LlmRuntime):
        runtime = args[0]
        tier = args[1] if len(args) > 1 else ""
        usage = args[2] if len(args) > 2 else None
    else:
        tier = args[0] if args else ""
        usage = args[1] if len(args) > 1 else None
        if runtime is None:
            runtime = EMPTY_RUNTIME
    cb = runtime.cost_callback
    if cb is None or usage is None:
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
        cb(**kwargs)
    except TypeError:
        # Old-style callback doesn't accept extra kwargs — fall back
        try:
            cb(
                tier=tier,
                input_tokens=kwargs["input_tokens"],
                output_tokens=kwargs["output_tokens"],
            )
        except TypeError as exc:
            log.warning("cost_callback_failed", error=str(exc))


def _coerce_llm_config(llm_config: LlmConfig | dict | None) -> LlmConfig:
    """Internal helper: accept LlmConfig/dict/None and produce LlmConfig.

    Test harnesses and external callers often pass dicts; internal pipelines
    pass typed LlmConfig. This keeps the entry points ergonomic while the
    rest of the module operates on the typed form.
    """
    if llm_config is None:
        return LlmConfig()
    if isinstance(llm_config, LlmConfig):
        return llm_config
    return LlmConfig(**llm_config)


def _get_llm(
    *args: Any,
    runtime: LlmRuntime | None = None,
    node_name: str = "",
    llm_config: LlmConfig | dict | None = None,
) -> Any:
    """Resolve an LLM client for *tier* from the runtime's factory.

    Backward-compatible argument shapes:
        _get_llm(runtime, tier, ...)   # post-§2 (positional)
        _get_llm(tier, ...)            # legacy — runtime defaults to compat
        _get_llm(tier, runtime=...)    # explicit keyword
    """
    if args and isinstance(args[0], LlmRuntime):
        runtime = args[0]
        tier = args[1] if len(args) > 1 else None
    else:
        tier = args[0] if args else None
        if runtime is None:
            runtime = EMPTY_RUNTIME
    if runtime.llm_factory is None:
        raise ConfigurationError.build(
            "LLM not configured",
            hint="Pass llm_factory= to compile().",
        )
    cfg = _coerce_llm_config(llm_config)
    all_kwargs = {"node_name": node_name, "llm_config": cfg.as_factory_kwargs()}
    if runtime.llm_factory_params is _ACCEPT_ALL:
        kwargs = all_kwargs
    else:
        kwargs = {k: v for k, v in all_kwargs.items() if k in runtime.llm_factory_params}
    assert tier is not None and isinstance(tier, str)
    return runtime.llm_factory(tier, **kwargs)


def _is_inline_prompt(template: str) -> bool:
    """Detect whether a prompt template is inline text vs a file reference.

    Inline text contains a space character or a ``${`` substitution marker.
    Everything else (e.g. ``"rw/summarize"``) is treated as a file reference
    and delegated to the consumer-provided prompt compiler.
    """
    return " " in template or "${" in template


_VAR_RE = re.compile(r"\$\{([^}]+)\}")
_IMAGE_RE = re.compile(r"\$\{image:([^}]+)\}")


def _compile_multimodal_prompt(template: str, input_data: Any) -> list[dict[str, Any]]:
    """Compile an inline prompt that contains ``${image:...}`` placeholders.

    Splits the template into alternating text / image-var segments, resolves
    each, and returns a single user message with content blocks.
    """
    parts = _IMAGE_RE.split(template)
    # re.split with a capturing group: [text, img_var, text, img_var, ...]
    content_blocks: list[dict[str, Any]] = []

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Text segment — substitute remaining ${var} placeholders
            rendered = _substitute_vars(part, input_data).strip()
            if rendered:
                content_blocks.append({"type": "text", "text": rendered})
        else:
            # Image variable name — resolve the raw value (not BAML-rendered)
            raw_val = _resolve_var_raw(part, input_data)
            if isinstance(raw_val, BaseModel):
                log.warning("image_resolved_to_model", var=part,
                            model_type=type(raw_val).__name__,
                            hint="image field resolved to a BaseModel, not a string; "
                                 "use dotted access like ${image:model.field}")
                continue  # skip — don't produce a corrupt image block
            val_str = str(raw_val) if raw_val is not None else ""
            if not val_str or not val_str.strip():
                log.warning("image_field_empty", var=part,
                            hint="image field is empty or None; skipping image block")
                continue  # skip — don't produce an empty image block
            uri = resolve_image(val_str)
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": uri},
            })

    return [{"role": "user", "content": content_blocks}]


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


def _resolve_var_raw(path: str, input_data: Any) -> Any:
    """Like _resolve_var but returns the raw object without BAML rendering.

    Used for image resolution where the resolved value must be a string
    (file path or base64), not a BAML-rendered model description.
    """
    parts = path.split(".")

    if isinstance(input_data, dict):
        if parts[0] not in input_data:
            log.warning("prompt_var_missing", var=path, available=sorted(input_data.keys()))
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
    return obj


def _substitute_vars(template: str, input_data: Any) -> str:
    """Replace all ``${...}`` placeholders in *template*."""
    return _VAR_RE.sub(lambda m: _resolve_var(m.group(1), input_data), template)


def _compile_prompt(
    template_or_runtime: Any,
    input_data_or_template: Any = None,
    input_data: Any = None,
    *,
    runtime: LlmRuntime | None = None,
    node_name: str = "",
    config: RunnableConfig | None = None,
    output_model: type[BaseModel] | None = None,
    llm_config: LlmConfig | dict | None = None,
    output_schema: str | None = None,
    context: dict[str, Any] | None = None,
) -> list:
    """Compile a prompt to message list.

    Two call shapes are accepted to preserve backward compatibility with the
    inline-prompt tests that predate the runtime threading:

        _compile_prompt(template, input_data, ...)              # legacy
        _compile_prompt(runtime, template, input_data, ...)     # post-§2

    The first positional is the discriminator: if it's an `LlmRuntime`,
    the new shape is used; otherwise the call is treated as the legacy
    inline-prompt path with `runtime=EMPTY_RUNTIME` and no prompt_compiler.
    """
    if isinstance(template_or_runtime, LlmRuntime):
        runtime = template_or_runtime
        template = input_data_or_template
        # input_data already named correctly
    else:
        template = template_or_runtime
        input_data = input_data_or_template
        if runtime is None:
            runtime = EMPTY_RUNTIME
    # Inline prompt — resolve ${} variables and return directly
    if _is_inline_prompt(template):
        if _IMAGE_RE.search(template):
            return _compile_multimodal_prompt(template, input_data)
        rendered = _substitute_vars(template, input_data)
        return [{"role": "user", "content": rendered}]

    cfg = _coerce_llm_config(llm_config)
    all_kwargs = {
        "node_name": node_name,
        "config": config,
        "output_model": output_model,
        "llm_config": cfg.as_factory_kwargs(),
        "output_schema": output_schema,
    }
    if context is not None:
        all_kwargs["context"] = context
    if runtime.prompt_compiler is None:
        raise ConfigurationError.build(
            "prompt compiler not configured",
            hint="Pass prompt_compiler= to compile().",
        )
    # Only pass kwargs the compiler accepts — inspected at compile time
    if runtime.prompt_compiler_params is _ACCEPT_ALL:
        kwargs = all_kwargs
    else:
        kwargs = {
            k: v
            for k, v in all_kwargs.items()
            if k in runtime.prompt_compiler_params
        }
    return runtime.prompt_compiler(template, input_data, **kwargs)


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
        # Array was truncated (no closing ]). Do NOT fall through to object
        # extraction — that would return the first inner dict, causing silent
        # empty results for list-field target models. Return the raw text and
        # let json_repair handle truncation recovery.
        return text.strip()

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
    *args: Any,
    model_tier: str | None = None,
    prompt_template: str | None = None,
    input_data: Any = None,
    output_model: Any = None,
    config: RunnableConfig | None = None,
    node_name: str = "",
    llm_config: LlmConfig | dict | None = None,
    context: dict[str, Any] | None = None,
    runtime: LlmRuntime | None = None,
) -> BaseModel:
    """Single LLM call with structured JSON output. Mode: produce.

    Output strategy (from llm_config.output_strategy):
        "structured" — llm.with_structured_output(model) (default, widest LangChain support)
        "json_mode"  — inject schema into prompt, LLM returns raw JSON, framework parses
        "text"       — LLM returns plain text, framework extracts and parses JSON from it

    Accepts both call shapes:
        invoke_structured(runtime, model_tier=..., ...)   # post-§2
        invoke_structured(model_tier=..., ...)            # legacy — uses compat runtime
    """
    if args and isinstance(args[0], LlmRuntime):
        runtime = args[0]
    elif runtime is None:
        runtime = EMPTY_RUNTIME
    cfg = _coerce_llm_config(llm_config)
    strategy = cfg.output_strategy
    llm_log = log.bind(tier=model_tier, prompt=prompt_template, output=output_model.__name__, strategy=strategy)

    output_schema = None
    if strategy == "json_mode":
        from neograph.describe_type import describe_type

        output_schema = describe_type(output_model)

    llm = _get_llm(runtime, model_tier, node_name=node_name, llm_config=cfg)
    messages = _compile_prompt(
        runtime,
        prompt_template,
        input_data,
        node_name=node_name,
        config=config,
        output_model=output_model,
        llm_config=cfg,
        output_schema=output_schema,
        context=context,
    )

    max_retries = cfg.max_retries
    t0 = time.monotonic()
    assert config is not None
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
    _notify_cost(runtime, model_tier, usage, node_name=node_name, mode="think", duration_s=round(elapsed, 3))
    return result


def render_prompt(
    node: Any,
    input_data: Any,
    *,
    config: RunnableConfig | None = None,
    runtime: LlmRuntime | None = None,
) -> str:
    """Render the exact prompt a node would send to the LLM, without calling it.

    Applies the renderer dispatch hierarchy (node.renderer > runtime.renderer > None),
    compiles via the supplied prompt_compiler, and formats messages as a
    readable string. Useful for prompt engineering and debugging.

    Args:
        node: A Node instance with prompt, model, and output fields.
        input_data: The input data (Pydantic model, dict, or primitive).
        config: Optional RunnableConfig-style dict for the prompt compiler.
        runtime: LLM runtime bundle (llm_factory/prompt_compiler/renderer).
            If omitted, falls back to the legacy compat slot set via
            `configure_llm()`. Pass `runtime=` for the new API.

    Returns:
        A human-readable string of the compiled messages.
    """
    if runtime is None:
        runtime = EMPTY_RUNTIME

    if runtime.prompt_compiler is None:
        raise ConfigurationError.build(
            "Prompt compiler not configured",
            hint="Pass prompt_compiler= to compile(), or supply runtime= here.",
        )

    # Apply renderer dispatch via RenderedInput — matches runtime path
    from neograph.renderers import build_rendered_input
    prompt = getattr(node, "prompt", "") or ""
    effective_renderer = getattr(node, "renderer", None) or runtime.renderer
    ri = build_rendered_input(input_data, renderer=effective_renderer)
    if _is_inline_prompt(prompt):
        input_data = ri.raw
    else:
        input_data = ri.for_template_ref

    # Generate output_schema for json_mode
    output_schema = None
    cfg = _coerce_llm_config(getattr(node, "llm_config", None))
    strategy = cfg.output_strategy
    output_model = getattr(node, "outputs", None)
    if strategy == "json_mode" and output_model is not None:
        from neograph.describe_type import describe_type

        output_schema = describe_type(output_model)

    messages = _compile_prompt(
        runtime,
        getattr(node, "prompt", "") or "",
        input_data,
        node_name=getattr(node, "name", ""),
        config=config,
        output_model=output_model,
        llm_config=cfg,
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
        # Multimodal content blocks → readable summary
        if isinstance(content, list):
            block_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        block_parts.append(block.get("text", ""))
                    elif block.get("type") == "image_url":
                        block_parts.append("[image]")
                    else:
                        block_parts.append(str(block))
                else:
                    block_parts.append(str(block))
            content = " ".join(block_parts)
        parts.append(f"[{role}]\n{content}")

    return "\n\n".join(parts)
