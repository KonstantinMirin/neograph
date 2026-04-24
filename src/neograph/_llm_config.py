"""Typed view over the dict-valued `llm_config` that Nodes and Constructs carry.

The public surface (Node.llm_config, Construct.llm_config, the llm_factory
callback) stays a dict for provider-knob pass-through (temperature,
max_tokens, etc.). This module provides the internal typed view used by
invoke_structured / invoke_with_tools / render_prompt so framework-read
keys get one normalized source of truth for defaults and type rejection.

Core invariant: the raw dict is never mutated. LlmConfig is read-only.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from neograph.errors import ConfigurationError


class LlmConfig(BaseModel):
    """Typed view over the llm_config dict. Internal consumers only.

    Pydantic validation rejects wrong types on known fields (``max_retries=``
    ``"five"`` raises, instead of silently degrading downstream). Unknown
    keys are preserved via ``extra="allow"`` so provider-specific knobs
    (temperature, max_tokens, model_tier) survive the round-trip any
    downstream consumer might need.
    """

    model_config = ConfigDict(extra="allow")

    output_strategy: Literal["structured", "json_mode", "text"] = "structured"
    max_retries: int = 1
    max_iterations: int = 20
    token_budget: int | None = None
    # ``None`` is the only legitimate sentinel here — it means
    # ``use the framework default, interpolated with output_model_name``.
    # Resolver method handles None and "" symmetrically.
    budget_exhausted_message: str | None = None

    def resolved_budget_exhausted_message(self, output_model_name: str) -> str:
        """Return the effective message for the Layer-C retry branch.

        ``None`` and empty string both fall back to the default template; any
        non-empty caller-supplied string is returned verbatim.
        """
        return self.budget_exhausted_message or (
            "Your previous response contained tool-call markup. "
            "All tool budgets are exhausted. Do NOT invoke any more tools. "
            f"Produce the final response as a {output_model_name} object. "
            "No markup, no tool calls. Output ONLY the structured response."
        )


def normalize_llm_config(
    raw: dict[str, Any] | None,
    *,
    node_name: str = "",
) -> LlmConfig:
    """Construct an LlmConfig from a raw dict, raising a friendly error on failure.

    Wraps pydantic.ValidationError in ConfigurationError so users see which
    node produced the bad config and what was expected. Returns a typed view
    over the dict; callers continue to pass the raw dict to the llm_factory
    callback and the prompt compiler -- those surfaces stay dict.
    """
    try:
        return LlmConfig.model_validate(raw or {})
    except ValidationError as exc:
        raise ConfigurationError.build(
            "invalid llm_config",
            node=node_name or None,
            found=str(raw),
            hint=f"pydantic: {exc.errors()[0]['msg']} on field "
                 f"'{'.'.join(str(p) for p in exc.errors()[0]['loc'])}'",
        ) from exc
