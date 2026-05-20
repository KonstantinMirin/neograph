"""Typed Protocol classes for user-supplied LLM callbacks.

Extracted from `_llm.py` to keep the Protocol definitions in a leaf
module that other layers can import without pulling in LLM machinery.

Protocols are structural and erased at runtime; `runtime_checkable` enables
`isinstance(fn, ProtocolName)` in tests.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LlmFactory(Protocol):
    """Factory callback for creating LLM client instances per node tier.

    Backward-compatible with both shapes:

    * Simple:   ``(tier) -> BaseChatModel``
    * Advanced: ``(tier, *, node_name=, llm_config=) -> BaseChatModel``

    Uses ``*args``/``**kwargs`` catch so the Simple form (no kwargs) still
    satisfies the Protocol structurally. ``_accepted_params`` filters
    actual kwargs at call site.
    """

    def __call__(self, tier: str, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class PromptCompiler(Protocol):
    """Builds message lists for LLM calls.

    Backward-compatible with both shapes:

    * Simple:   ``(template, input_data) -> list``
    * Advanced: ``(template, input_data, *, node_name=, config=, ...) -> list``
    """

    def __call__(self, template: str, input_data: Any, *args: Any, **kwargs: Any) -> list[Any]: ...


@runtime_checkable
class CostCallback(Protocol):
    """Cost telemetry hook called after each LLM invocation.

    Modern shape (preferred): keyword-only — mypy validates the required
    keys. The legacy 3-arg fallback at ``_notify_cost`` catches ``TypeError``
    for callbacks that only accept ``(tier, input_tokens, output_tokens)``.
    """

    def __call__(
        self,
        *,
        tier: str,
        input_tokens: int,
        output_tokens: int,
        node_name: str = ...,
        mode: str = ...,
        duration_s: float = ...,
        **kw: Any,
    ) -> None: ...
