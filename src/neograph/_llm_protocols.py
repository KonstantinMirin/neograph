"""Typed Protocol classes for user-supplied LLM callbacks.

Extracted from `_llm.py` to keep the Protocol definitions in a leaf
module that other layers can import without pulling in LLM machinery.

Protocols are structural and erased at runtime; `runtime_checkable` enables
`isinstance(fn, ProtocolName)` in tests.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from neograph._rendered import PromptInput


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

    ``input_data`` is a :data:`~neograph._rendered.PromptInput` — a TOTAL mapping
    of name -> prompt-ready text. Every channel obeys it: node inputs (fan-in
    dict form and single-type, the latter keyed by its type name), the Oracle
    merge payload, and ``di_inputs``. A compiler never has to ask what shape it
    is holding, which is the whole point of neograph-l2a7w: the values used to be
    a rendered ``str`` on one path and the raw Pydantic model on another, and the
    obvious ``getattr`` silently yielded an empty payload on whichever path the
    author did not expect.

    Reaching for a field on a value now raises ``PromptInputError`` rather than
    returning ``""``. A compiler that genuinely needs the objects declares a
    ``raw_inputs`` parameter and receives them alongside, keyed identically.

    Backward-compatible with both shapes:

    * Simple:   ``(template, input_data) -> list``
    * Advanced: ``(template, input_data, *, node_name=, config=, ...) -> list``

    The annotation is the DISCOVERABLE half of the contract; because this is a
    structural Protocol and consumer compilers annotate ``Any``, the enforcing
    half is the runtime totality assertion at the seam plus ``Rendered``'s loud
    ``__getattr__``. Both are required — neither alone closes the hole.
    """

    def __call__(self, template: str, input_data: PromptInput, *args: Any, **kwargs: Any) -> list[Any]: ...


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
