"""Prompt rendering — inline `${var}` substitution, multimodal blocks, and
template-ref delegation; plus the public `render_prompt` introspection entry.

Single-responsibility: how a prompt template is turned into a message list.
Change axis: prompt template syntax (inline vs file-ref, new content blocks
like image/audio/streaming) is the only scenario that should force changes
here.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph._image import resolve_image
from neograph._llm_config import LlmConfig, _coerce_llm_config
from neograph._llm_runtime import _ACCEPT_ALL, EMPTY_RUNTIME, LlmRuntime
from neograph._placeholders import DOLLAR_RE as _VAR_RE
from neograph._placeholders import apply_scanner
from neograph._state_keys import StateKeys
from neograph.describe_type import describe_type, describe_value
from neograph.errors import ConfigurationError
from neograph.renderers import build_rendered_input

log = structlog.get_logger()


# The ``${var}`` scanner is the shared one in _placeholders (fail-SOFT resolver
# here, fail-LOUD in prompt.substitute — one scanner, two policies). ``_IMAGE_RE``
# is a distinct multimodal grammar (``${image:...}`` builds image content blocks,
# not text substitution) and stays local.
_IMAGE_RE = re.compile(r"\$\{image:([^}]+)\}")


def _is_inline_prompt(template: str) -> bool:
    """Detect whether a prompt template is inline text vs a file reference.

    Inline text contains a space character or a ``${`` substitution marker.
    Everything else (e.g. ``"rw/summarize"``) is treated as a file reference
    and delegated to the consumer-provided prompt compiler.
    """
    return " " in template or "${" in template


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
    """Replace all ``${...}`` placeholders in *template* (fail-soft resolver)."""
    return apply_scanner(template, _VAR_RE, lambda name: _resolve_var(name, input_data))


def _compile_multimodal_prompt(template: str, input_data: Any) -> list[dict[str, Any]]:
    """Compile an inline prompt that contains ``${image:...}`` placeholders.

    Splits the template into alternating text / image-var segments, resolves
    each, and returns a single user message with content blocks.
    """
    parts = _IMAGE_RE.split(template)
    content_blocks: list[dict[str, Any]] = []

    for i, part in enumerate(parts):
        if i % 2 == 0:
            rendered = _substitute_vars(part, input_data).strip()
            if rendered:
                content_blocks.append({"type": "text", "text": rendered})
        else:
            raw_val = _resolve_var_raw(part, input_data)
            if isinstance(raw_val, BaseModel):
                log.warning(
                    "image_resolved_to_model",
                    var=part,
                    model_type=type(raw_val).__name__,
                    hint="image field resolved to a BaseModel, not a string; "
                    "use dotted access like ${image:model.field}",
                )
                continue
            val_str = str(raw_val) if raw_val is not None else ""
            if not val_str or not val_str.strip():
                log.warning("image_field_empty", var=part, hint="image field is empty or None; skipping image block")
                continue
            uri = resolve_image(val_str)
            content_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": uri},
                }
            )

    return [{"role": "user", "content": content_blocks}]


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
    else:
        template = template_or_runtime
        input_data = input_data_or_template
        if runtime is None:
            runtime = EMPTY_RUNTIME
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
    # di_inputs: the dispatch layer resolves a node's FromInput/FromConfig params
    # once and stashes them on config under DI_INPUTS (the _oracle_model-style
    # side-channel); reading here keeps DI resolution single-sourced.
    di_inputs = config.get("configurable", {}).get(StateKeys.DI_INPUTS) if isinstance(config, dict) else None
    if di_inputs:
        all_kwargs["di_inputs"] = di_inputs
    if runtime.prompt_compiler is None:
        raise ConfigurationError.build(
            "prompt compiler not configured",
            hint="Pass prompt_compiler= to compile().",
        )
    if runtime.prompt_compiler_params is _ACCEPT_ALL:
        kwargs = all_kwargs
    else:
        kwargs = {k: v for k, v in all_kwargs.items() if k in runtime.prompt_compiler_params}
    return runtime.prompt_compiler(template, input_data, **kwargs)


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

    prompt = getattr(node, "prompt", "") or ""
    effective_renderer = getattr(node, "renderer", None) or runtime.renderer
    ri = build_rendered_input(input_data, renderer=effective_renderer)
    if _is_inline_prompt(prompt):
        input_data = ri.raw
    else:
        input_data = ri.for_template_ref

    output_schema = None
    cfg = _coerce_llm_config(getattr(node, "llm_config", None))
    strategy = cfg.output_strategy
    output_model = getattr(node, "outputs", None)
    if strategy == "json_mode" and output_model is not None:
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

    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", str(msg))
        else:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", str(msg))
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
