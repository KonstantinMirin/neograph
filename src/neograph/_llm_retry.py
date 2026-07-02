"""JSON parsing + error-feedback retry for json_mode / text output strategies.

Single-responsibility: LLM responses come back as text; this module repairs,
extracts, parses, and -- on parse failure -- retries with a structured error
message containing the validation errors and expected schema.

Change axis: how LLM-emitted JSON is repaired or normalized, and the retry
escalation policy on parse failure.
"""

from __future__ import annotations

import json as _json
from typing import Any

from json_repair import repair_json
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined

from neograph._dsml import contains_dsml
from neograph.describe_type import describe_type
from neograph.errors import ExecutionError


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


def _extract_json(text: str) -> str:
    """Extract the first balanced JSON value (object or array) from LLM text.

    Finds the first '{' or '[' and tracks depth to find the matching closer.
    Handles markdown fences, thinking tags, prose before/after JSON.
    When both '{' and '[' are present, picks whichever comes first.
    """
    brace_pos = text.find("{")
    bracket_pos = text.find("[")

    if bracket_pos != -1 and (brace_pos == -1 or bracket_pos < brace_pos):
        result = _extract_balanced(text, bracket_pos, "[", "]")
        if result is not None:
            return result
        # Array was truncated (no closing ]). Do NOT fall through to object
        # extraction — that would return the first inner dict, causing silent
        # empty results for list-field target models. Return the raw text and
        # let json_repair handle truncation recovery.
        return text.strip()

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

    first_open = brace_pos if brace_pos != -1 else bracket_pos
    if first_open != -1:
        close_char = "}" if first_open == brace_pos else "]"
        last = text.rfind(close_char)
        if last > first_open:
            return text[first_open : last + 1]
    return text.strip()


def _is_list_annotation(annotation: Any) -> bool:
    """Check if a type annotation is a list type (list[X], List[X], etc.)."""
    import typing
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return True
    if origin is typing.Union:
        args = getattr(annotation, "__args__", ())
        return any(_is_list_annotation(a) for a in args if a is not type(None))
    return annotation is list


def _apply_null_defaults(data: dict, model: type[BaseModel]) -> None:
    """Replace null values with field defaults, recursively.

    Mutates *data* in place. Applies when the JSON value is None and the field
    has either an explicit default or a default_factory. Also recurses into
    nested BaseModel fields and list[BaseModel] items.
    """
    for field_name, field_info in model.model_fields.items():
        if field_name not in data:
            continue
        val = data[field_name]

        if val is None and field_info.default is not PydanticUndefined:
            data[field_name] = field_info.default
            continue

        # LLMs emit null for default_factory list/dict fields (their default is
        # PydanticUndefined, so the branch above skips them). Coerce to the
        # factory result. Zero-arg first: a data-accepting factory (Pydantic
        # 2.10+) raises TypeError -> factory(data); a zero-arg one like list must
        # NOT get the data dict (list(data) returns keys, not []). neograph-s1u4.
        if val is None and field_info.default_factory is not None:
            # default_factory is typed as a union (zero-arg | data-accepting);
            # the try/except resolves the arity at runtime, so both calls need
            # the call-arg ignore.
            factory = field_info.default_factory
            try:
                data[field_name] = factory()  # type: ignore[call-arg]
            except TypeError:
                data[field_name] = factory(data)  # type: ignore[call-arg]
            continue

        annotation = field_info.annotation
        if isinstance(val, dict) and isinstance(annotation, type) and issubclass(annotation, BaseModel):
            _apply_null_defaults(val, annotation)
            continue

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
    extracted = _extract_json(text)

    # Detect non-JSON content: empty result or XML-like tool-call markup.
    # DeepSeek R1 emits <｜DSML｜function_calls>... after budget exhaustion.
    stripped = extracted.strip()
    if not stripped or (
        not stripped.startswith(("{", "["))
        and contains_dsml(text)
    ):
        raise ExecutionError.build(
            "LLM returned non-JSON content instead of structured output",
            expected=f"JSON object for {output_model.__name__}",
            found=f"XML tool-call markup (first 200 chars): {text[:200]!r}",
            hint="Check model compatibility with structured output; some models emit XML tool-call markup after budget exhaustion.",
        )

    repaired = repair_json(extracted, return_objects=False)

    # Coerce null → default for fields that have defaults.
    try:
        parsed = _json.loads(repaired)
        if isinstance(parsed, dict) and hasattr(output_model, "model_fields"):
            _apply_null_defaults(parsed, output_model)
            repaired = _json.dumps(parsed)
        elif isinstance(parsed, list) and hasattr(output_model, "model_fields"):
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
        pass

    try:
        return output_model.model_validate_json(repaired)
    except ValidationError as exc:
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

    combined_usage = {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
    } if (total_input or total_output) else None
    return _parse_json_response(raw_text, output_model), combined_usage


def recover_dsml(
    raw_text: str,
    output_model: type[BaseModel],
    llm: Any,
    messages: list,
    config: RunnableConfig,
    cfg: Any,  # LlmConfig; duck-typed for .resolved_budget_exhausted_message + .max_retries
    *,
    strategy: str,
) -> BaseModel | None:
    """Recover from DSML/XML tool-call markup via a targeted re-prompt.

    This is the *retry* side of the DSML story: the compat shim
    (`_llm_structured_compat`) classifies output as DSML; this function acts on
    it by re-prompting the LLM with a budget-exhausted directive and re-parsing.
    Strategy-orthogonal — called by both the `structured` dispatch path and the
    `json_mode`/`text` tool-loop path. Detection is delegated to
    `_dsml.contains_dsml` (the single definition site).

    Returns:
        Parsed model on successful recovery; None if no DSML markup detected.

    Raises:
        ExecutionError if DSML detected but the targeted retry also failed
        AND the generic retry path also failed.
    """
    import structlog
    _log = structlog.get_logger("neograph")

    if not contains_dsml(raw_text):
        return None

    _log.warning(
        "trailing_tool_call_markup",
        strategy=strategy,
        hint="model emitted tool-call markup; retrying with targeted directive",
    )
    budget_msg = cfg.resolved_budget_exhausted_message(output_model.__name__)
    retry_messages = list(messages)
    retry_messages.append({"role": "assistant", "content": raw_text})
    retry_messages.append({"role": "user", "content": budget_msg})
    try:
        retry_response = llm.invoke(retry_messages, config=config)
        retry_text = (
            retry_response.content if hasattr(retry_response, "content")
            else str(retry_response)
        )
        return _parse_json_response(retry_text, output_model)
    except ExecutionError:
        # Targeted retry also failed — try generic retry path
        max_retries = getattr(cfg, "max_retries", 1)
        parse_result, _ = _invoke_json_with_retry(
            llm, retry_messages, output_model, config, max_retries=max_retries,
        )
        return parse_result
