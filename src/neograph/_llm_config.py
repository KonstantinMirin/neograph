"""Typed IR for LLM configuration.

``LlmConfig`` is the IR type carried by ``Node.llm_config`` and
``Construct.llm_config``. It parses inputs at construction time and rejects
unknown keys via ``extra='forbid'`` -- typos like ``max_retires`` surface at
Node construction instead of silently defaulting.

Provider-specific knobs (``temperature``, ``max_tokens``, ``model_tier``, ...)
live in the separate ``provider_kwargs`` namespace and cannot collide with
framework keys.

At the ``llm_factory`` / ``prompt_compiler`` boundary, :py:meth:`as_factory_kwargs`
flattens framework fields and provider knobs into a single dict to preserve
the documented consumer contract.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LlmConfig(BaseModel):
    """Typed IR for the per-Node / per-Construct LLM configuration.

    Framework fields live here as typed attributes. Provider knobs go into
    ``provider_kwargs``. Pydantic's ``extra='forbid'`` ensures any unknown
    top-level key (including typos) raises ``ValidationError`` at Node
    construction time.
    """

    model_config = ConfigDict(extra="forbid")

    # Known framework fields -- the ONLY place defaults live.
    # output_strategy: how a node's typed output is produced.
    #   Single-shot (think): 'structured' = provider constrained decoding
    #   (with_structured_output); 'json_mode'/'text' = schema in prompt + parse.
    #   Agent/act: the ReAct loop's final turn is parsed directly (0 extra calls);
    #   output_strategy selects the PARSE-FAILURE fallback only -- 'structured' =
    #   constrained-decoding fallback (weak-model recourse), 'json_mode'/'text' =
    #   parse-retry fallback. It is NOT inert for agent/act; it picks the recovery.
    output_strategy: Literal["structured", "json_mode", "text"] = "structured"
    max_retries: int = 1
    max_iterations: int = 20
    token_budget: int | None = None
    # Opt-in: when True, the tool loop prepends a framework-generated
    # {role:system} preamble announcing per-tool budgets + the step cap.
    # Announced numbers are computed at the enforcement site, never
    # hand-written, so they cannot drift. See _tool_budget_preamble.py.
    announce_tool_budget: bool = False
    # ``None`` means "use the framework default template, interpolated with
    # the output model name". ``""`` also falls back to the default.
    budget_exhausted_message: str | None = None

    # Separate namespace for provider-specific knobs (temperature, max_tokens,
    # top_p, stop, model_tier, ...). Kept distinct so framework keys cannot
    # accidentally shadow or be shadowed by provider keys.
    provider_kwargs: dict[str, Any] = Field(default_factory=dict)

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

    def as_factory_kwargs(self) -> dict[str, Any]:
        """Flatten to a dict for the user ``llm_factory`` / ``prompt_compiler``.

        Framework fields take precedence over ``provider_kwargs`` on key
        collision -- framework semantics are authoritative, so a user cannot
        shadow ``max_retries`` with a provider knob.
        """
        framework = self.model_dump(exclude={"provider_kwargs"})
        return {**self.provider_kwargs, **framework}

    def merged_with(self, child: LlmConfig) -> LlmConfig:
        """Construct-level propagation: parent default + child overrides.

        Child wins on every framework field it sets explicitly (tracked via
        ``model_fields_set``). Unset child fields inherit the parent value.
        ``provider_kwargs`` are merged with child winning on key collisions.

        The result has ``model_fields_set`` populated for every framework
        field written -- callers that chain further merges see the full set
        of "has been configured" markers.
        """
        parent_dump = self.model_dump(exclude={"provider_kwargs"})
        child_overrides = child.model_dump(exclude_unset=True, exclude={"provider_kwargs"})
        merged_framework = {**parent_dump, **child_overrides}
        merged_provider = {**self.provider_kwargs, **child.provider_kwargs}
        return LlmConfig(provider_kwargs=merged_provider, **merged_framework)


def _coerce_llm_config(llm_config: LlmConfig | dict | None) -> LlmConfig:
    """Accept ``LlmConfig`` / ``dict`` / ``None`` and produce ``LlmConfig``.

    Test harnesses and external callers often pass dicts; internal pipelines
    pass typed ``LlmConfig``. This keeps entry points ergonomic while the
    rest of the codebase operates on the typed form.
    """
    if llm_config is None:
        return LlmConfig()
    if isinstance(llm_config, LlmConfig):
        return llm_config
    return LlmConfig(**llm_config)
