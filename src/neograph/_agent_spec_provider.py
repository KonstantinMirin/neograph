"""Derive an Agent Spec ``LlmConfig`` triple from the client a caller's own
``llm_factory`` resolves a neograph model TIER to (neograph-qtfof.13).

``Node.model`` is an opaque tier string (``"fast"``), deliberately: the real
provider is kept OUT of the IR. But at ``compile()`` time the tier IS resolved,
by the caller's ``llm_factory``, to a concrete client that already knows its own
provider, model name and base URL. This module reads that triple back off the
resolved instance so ``to_agent_spec(construct, llm_factory=...)`` can emit a
standard LLM configuration instead of a tier string no vendor has.

Three things a reader should not have to reconstruct:

**Identity vs capability.** ``_llm.py:50-58`` bans identifying a
``response_format`` rejection by provider CLASS NAME ("that drifts") — that rule
is about CAPABILITY sniffing, where behaviour varies within a class and between
versions. Provider IDENTITY is a different question, and the class is the only
signal a resolved client carries; ``_llm.py:111`` already records
``provider=type(self._unbound).__name__`` for exactly that purpose. The
distinction is deliberate, not an oversight.

**Why the table has one row.** ``pyagentspec``'s ``LlmConfig.api_provider`` is a
free-form ``Optional[str]`` with no enum, and all five shipped adapters dispatch
a BARE ``LlmConfig`` on ``api_provider == "openai"`` and raise
``NotImplementedError`` otherwise (``_langgraphconverter.py:1325-1342``, crewai
``:326``, openaiagents ``:105``, agent_framework ``:261``, autogen ``:220``).
Other providers are represented by ``LlmConfig`` SUBCLASSES that freeze their own
literal (``OllamaConfig``, ``OciGenAiConfig``, ...). So deriving ``"anthropic"``
from a ``ChatAnthropic`` would emit an artifact that still fails to load, with a
worse message than an honest ``None``. Unclassified stays ``None`` until
neograph grows subclass lowering (separate ticket).

**Why strings, never ``isinstance``.** ``langchain-openai`` lives in
``[dependency-groups].dev``; core ``[project].dependencies`` carries
``langchain-core`` only. An ``isinstance`` check would drag a provider package
into the core install.

The import edge is ONE-WAY — ``_agent_spec_provider`` imports ``_llm``, never the
reverse. This is the first runtime-layer edge into the export layer, and keeping
it in this leaf module is what contains it.

One version caveat, safe today but not asserted anywhere else: ``pyagentspec``'s
``LlmConfig._versioned_model_fields_to_exclude`` drops ``api_provider``, ``url``
and ``model_id`` for ``agentspec_version < v26_1_2``. A bare ``LlmConfig``
self-infers ``v26_1_2``, so the derived triple cannot silently vanish — but a
future version pin could erase the very field this module exists to populate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from ._llm_runtime import LlmRuntime
    from .node import Node

log = structlog.get_logger()


#: Resolved-client class (``module.QualName``) -> Agent Spec ``api_provider``.
#:
#: EXACT match only. One row today, and that is the honest size of it — see the
#: module docstring on why a second row would emit a worse artifact than
#: ``None`` rather than a better one.
PROVIDER_BY_CLIENT_CLASS: dict[str, str] = {
    "langchain_openai.chat_models.base.ChatOpenAI": "openai",
}

#: Attribute pairs read off a resolved client for the non-provider half of the
#: triple. Tried in order; the first present, non-empty value wins.
_MODEL_ATTRS = ("model_name", "model_id", "model")
_URL_ATTRS = ("openai_api_base", "base_url", "api_base")


@dataclass(frozen=True)
class DerivedLlmConfig:
    """What a resolved client tells us about itself. All three fields are
    ``None`` when the tier could not be resolved or the class is unclassified —
    the caller then keeps today's honest behaviour (tier string as ``model_id``,
    no provider, no url)."""

    api_provider: str | None = None
    model_id: str | None = None
    url: str | None = None


#: The "we learned nothing" answer, shared so the no-factory path allocates nothing.
_NOTHING_DERIVED = DerivedLlmConfig()


def _read_first(client: Any, names: tuple[str, ...]) -> str | None:
    for name in names:
        value = getattr(client, name, None)
        if isinstance(value, str) and value:
            return value
    return None


def describe_client(client: Any) -> DerivedLlmConfig:
    """Read the Agent Spec triple off an already-resolved LLM client.

    Split out from :class:`ApiProviderResolver` so the classification is
    testable without a runtime, and so the resolver holds only caching +
    precedence.
    """
    qualname = f"{type(client).__module__}.{type(client).__qualname__}"
    provider = PROVIDER_BY_CLIENT_CLASS.get(qualname)
    if provider is None:
        log.warning(
            "agent_spec_provider_unclassified",
            client_class=qualname,
            detail=(
                "no Agent Spec api_provider row for this client class; exporting api_provider=None "
                "(honest) — export_conformance() reports it via llm_config_missing_api_provider"
            ),
        )
        return _NOTHING_DERIVED
    return DerivedLlmConfig(
        api_provider=provider,
        model_id=_read_first(client, _MODEL_ATTRS),
        url=_read_first(client, _URL_ATTRS),
    )


@dataclass
class ApiProviderResolver:
    """Decides one node's exported ``LlmConfig`` triple.

    Precedence lives HERE and nowhere else: an explicitly-supplied
    ``api_provider`` wins, then whatever the resolved client says, then
    ``None``. With a one-row table the explicit kwarg stays the only route for a
    factory this module cannot classify, so it must not be shadowed by
    derivation.

    Resolution is memoised PER TIER: a 20-node construct on 2 tiers constructs 2
    clients, not 20. Resolving a tier is not free — it reads env and may open a
    connection pool.
    """

    explicit: str | None = None
    runtime: LlmRuntime | None = None
    _by_tier: dict[str, DerivedLlmConfig] = field(default_factory=dict, repr=False)

    @property
    def _can_derive(self) -> bool:
        # Short-circuit BEFORE calling _get_llm, which raises ConfigurationError
        # when llm_factory is None -- otherwise "no factory supplied" and "the
        # caller's factory blew up" arrive as the same swallowed exception, and
        # the common no-factory path pays a raise per tier.
        return self.runtime is not None and self.runtime.llm_factory is not None

    def _derive_for_tier(self, tier: str, node: Node) -> DerivedLlmConfig:
        if tier in self._by_tier:
            return self._by_tier[tier]
        # REUSE the single canonical tier->client path. A bare
        # runtime.llm_factory(tier) here would skip the llm_factory_params
        # signature filter LlmRuntime.build populates, silently breaking every
        # factory whose signature is (tier, node_name=None, llm_config=None) --
        # which is what examples/07, 08 and 10 all use.
        from ._llm import _get_llm

        try:
            client = _get_llm(self.runtime, tier, node_name=node.name, llm_config=node.llm_config)
            derived = describe_client(client)
        except Exception as exc:
            # Exporting a spec must never require credentials: examples/07 does
            # os.environ["OPENROUTER_API_KEY"], a KeyError with no key set.
            log.warning(
                "agent_spec_provider_resolution_failed",
                tier=tier,
                node=node.name,
                error=f"{type(exc).__name__}: {exc}",
                detail="exporting api_provider=None for this tier; export does not require credentials",
            )
            derived = _NOTHING_DERIVED
        self._by_tier[tier] = derived
        return derived

    def config_for(self, node: Node) -> DerivedLlmConfig:
        """The triple to export for *node*, precedence already applied."""
        tier = node.model
        derived = self._derive_for_tier(tier, node) if (self._can_derive and tier) else _NOTHING_DERIVED
        if self.explicit is not None:
            return DerivedLlmConfig(api_provider=self.explicit, model_id=derived.model_id, url=derived.url)
        return derived


def build_resolver(api_provider: str | None, llm_factory: Any) -> ApiProviderResolver:
    """Construct the ONE resolver a whole export shares.

    Builds the runtime with the SAME call ``compile()`` makes
    (``LlmRuntime.build``), so ``llm_factory_params`` is populated exactly as it
    is at runtime.
    """
    from ._llm_runtime import LlmRuntime

    runtime = LlmRuntime.build(llm_factory=llm_factory) if llm_factory is not None else None
    return ApiProviderResolver(explicit=api_provider, runtime=runtime)
