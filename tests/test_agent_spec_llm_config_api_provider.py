"""Regression tier for neograph-qtfof.8 -- an exported ``LlmConfig`` must be
convertible by a third-party Agent Spec runtime WHEN the caller opts in.

``_agent_spec_node_lowering._make_llm_config`` built every exported
``LlmConfig`` with ``model_id`` alone, leaving ``api_provider`` at its ``None``
default -- unconditionally, with no way to change it. ``pyagentspec``'s own
LangGraph adapter dispatches a *bare* ``LlmConfig`` (one that is not a concrete
subclass like ``OpenAiConfig`` / ``OciGenAiConfig``) on that string and raises
``NotImplementedError: LlmConfig with api_provider='None' is not yet supported``
for anything that is not ``"openai"``.

MAINTAINER DECISION (2026-08-14, see neograph-qtfof.8's design field): FAIL-LOUD
OPT-IN, not a guessed default. neograph deliberately keeps the real LLM
provider out of the graph layer (``Node.model`` is an opaque tier string,
resolved via ``llm_factory`` at RUNTIME) -- the exporter cannot know the
provider and must not guess one. So ``to_agent_spec(construct)`` (no
``api_provider=``) KEEPS today's honest behavior: no ``api_provider`` is
emitted, and the third-party runtime cannot convert the config. That is the
right permanent behavior for the default case, not the bug. The bug -- and
what this tier actually tests -- is that there was previously no way for the
caller to opt in at all. ``to_agent_spec(construct, api_provider="openai")``
must now make the SAME cells convertible.

**Why the assertion is scoped to the api_provider gap and not "it works".**
The defect surfaces at two DIFFERENT points, because the adapter converts an
``LlmNode``'s config eagerly at ``load_dict`` but an ``AgentNode``'s lazily on
first use:

  * ``think`` and the Oracle ``merge_prompt`` merge (both lower to ``LlmNode``)
    fail during **load**;
  * ``agent`` / ``act`` (which lower to ``AgentNode``) **load cleanly** and fail
    during **invoke**.

A load-only test would therefore report a false green for half the surface --
which is exactly why the cells below are driven through the harness's
load-*and*-invoke entry point rather than a bare ``load_dict``.

Once the config names a real API provider the adapter builds a live client, so
anything past the conversion is credentials + network, which this tier has no
business asserting on. The fixture pins a dummy key and an unroutable base URL
(so nothing leaves the machine) and each test asserts only that the *api_provider
conversion gap* is gone -- any other failure is out of scope and tolerated.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("pyagentspec")

from neograph._agent_spec import to_agent_spec  # noqa: E402
from tests.agent_spec_loader_harness import run_via_agent_spec_loader, stub_registry  # noqa: E402
from tests.test_agent_spec_matrix import CELLS, build_cell  # noqa: E402

#: Every exported-cell shape whose lowering calls ``_make_llm_config``. Covers
#: both manifestations: LlmNode (load-time) and AgentNode (invoke-time), plus
#: the Oracle merge lowering, which is a third call site of the same helper.
LLM_BEARING_CELLS: tuple[str, ...] = (
    "think-bare-single",
    "scripted-oracle-merge_prompt-single",
    "agent-bare-single",
    "act-bare-single",
)


@pytest.fixture(autouse=True)
def _offline_openai_credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A syntactically-valid key + an unroutable endpoint.

    A fixed export names an API provider, at which point the adapter constructs a
    real chat client -- which refuses to build at all without a key. The key makes
    the CONVERSION reachable; the base URL guarantees that any call made past it
    dies on a local connection refusal instead of reaching a vendor endpoint from
    the default test gate.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-neograph-offline-not-a-real-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
    yield


class TestExportedLlmConfigIsConvertibleWhenApiProviderIsSupplied:
    """neograph-qtfof.8: ``to_agent_spec(construct, api_provider=...)`` must make
    the exported ``LlmConfig`` convertible by the third-party adapter."""

    @pytest.mark.parametrize("cell_id", LLM_BEARING_CELLS)
    def test_the_third_party_runtime_can_convert_the_exported_llm_config(self, cell_id: str) -> None:
        construct = build_cell(*CELLS[cell_id])
        flow = to_agent_spec(construct, api_provider="openai")

        try:
            run_via_agent_spec_loader(flow, cell_id, stub_registry(flow))
        except NotImplementedError as exc:  # noqa: PERF203 -- one cell per test
            if "api_provider" in str(exc):
                pytest.fail(
                    f"neograph-qtfof.8: cell {cell_id!r} exported an LlmConfig the third-party "
                    f"Agent Spec runtime cannot convert even with api_provider='openai' supplied -- {exc}"
                )
            raise
        except Exception:
            # Credentials, network, or any other downstream failure is outside
            # this tier's claim: the api_provider conversion gap is what is
            # being asserted, and reaching here means it did not fire.
            pass


class TestExportedLlmConfigStaysHonestByDefault:
    """neograph-qtfof.8: WITHOUT ``api_provider=``, the export must keep its
    honest, non-guessed default -- no ``api_provider`` emitted, third-party
    runtime cannot convert. This is the intentional, permanent behavior
    (maintainer decision), not something the fix atom removes."""

    @pytest.mark.parametrize("cell_id", LLM_BEARING_CELLS)
    def test_default_export_still_has_no_api_provider(self, cell_id: str) -> None:
        construct = build_cell(*CELLS[cell_id])
        flow = to_agent_spec(construct)

        with pytest.raises(NotImplementedError, match="api_provider"):
            run_via_agent_spec_loader(flow, cell_id, stub_registry(flow))


# ---------------------------------------------------------------------------
# neograph-qtfof.13 -- PER-NODE DERIVATION from the runtime the caller supplies
# ---------------------------------------------------------------------------
#
# The tier above proves the UNIFORM opt-in string works. It cannot express a
# mixed-provider construct: one ``api_provider=`` is stamped on every exported
# ``LlmConfig``, so a pipeline whose ``fast`` tier resolves to a real
# ``ChatOpenAI`` and whose ``reason`` tier resolves to something else must
# either lie about one of them or stay honest about both.
#
# neograph-qtfof.13's corrected five-step scope (maintainer, 2026-08-24):
#
#   1. ``to_agent_spec(construct, api_provider=None, *, llm_factory=None)`` --
#      the public surface builds the resolver ONCE (the fourth instance of the
#      established ``llm_factory=`` kwarg, after ``compile()``, ``lint()`` and
#      ``Node.run_isolated()``).
#   2. Resolve tier -> client through the EXISTING ``_llm._get_llm`` -- never a
#      second ``llm_factory(...)`` call site.
#   3. Read provider identity + ``model_name`` + ``openai_api_base`` off the
#      resolved instance.
#   4. Emit all three at the single ``_make_llm_config`` / ``SpecLlmConfig(``
#      site.
#   5. Add ``model_tier`` to ``_prompt_spec_marker`` so ``from_agent_spec``
#      still restores the opaque tier string.
#
# **Measurement locus is LOAD, not a live invoke.** The gap these tests close
# fires while the third-party adapter CONSTRUCTS the model
# (``_langgraphconverter.py:1325-1342``'s ``if api_provider == "openai": ...
# else NotImplementedError``). No stub endpoint, no credentials, and no claim
# that the exported graph then runs against a real vendor -- that is a strictly
# weaker, separate question this ticket cannot settle. The fake factory below
# therefore points its ``ChatOpenAI`` at the same unroutable local address the
# autouse fixture uses, so a derived ``url`` cannot send anything off-machine.
#
# **What must NOT move**: ``TestExportedLlmConfigStaysHonestByDefault`` above
# and the 32-cell byte-for-byte golden
# (``tests/fixtures/agent_spec_refactor_snapshot.json``). Zero-arg
# ``to_agent_spec(construct)`` keeps emitting ``api_provider=None`` and the
# tier string as ``model_id``, forever -- the derived provider is a RENDERING
# of a caller-supplied runtime, never IR.

from typing import Any  # noqa: E402

from langchain_openai import ChatOpenAI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from neograph import construct_from_functions, node  # noqa: E402
from neograph.loader import from_agent_spec  # noqa: E402
from tests.agent_spec_flow_walk import all_flows  # noqa: E402

#: The address the derived ``url`` must carry. Deliberately the unroutable one
#: the autouse fixture pins: a derived url that DID reach a vendor endpoint
#: would make this tier network-dependent, which is the whole thing the LOAD
#: measurement locus exists to avoid.
STUB_BASE_URL = "http://127.0.0.1:1/v1"

#: The real model name behind the ``fast`` tier. The point of deriving
#: ``model_id`` at all: without it the adapter builds ``ChatOpenAI(model="fast")``
#: -- a graph that loads and then 404s on its first real call.
DERIVED_MODEL_NAME = "openai/gpt-4o-mini"


class _UnclassifiableChatModel:
    """A resolved client no provider row can classify.

    Stands in for every non-``ChatOpenAI`` client: ``pyagentspec``'s five
    shipped adapters all dispatch a BARE ``LlmConfig`` on ``api_provider ==
    "openai"`` and raise otherwise, so deriving ``"anthropic"`` here would emit
    an artifact that still fails to load, with a worse message than an honest
    ``None``. Unclassified must stay ``None``.
    """


def _openai_client() -> ChatOpenAI:
    return ChatOpenAI(
        model=DERIVED_MODEL_NAME,
        base_url=STUB_BASE_URL,
        api_key="sk-neograph-offline-not-a-real-key",
    )


def _mixed_provider_factory(tier: str, node_name: str | None = None, llm_config: Any = None) -> Any:
    """``fast`` resolves to a real ``ChatOpenAI``; anything else does not.

    Signature is the three-parameter form ``examples/07``, ``08`` and ``10``
    all use, precisely so the export path has to go through
    ``_llm._get_llm``'s ``llm_factory_params`` filter rather than calling the
    factory bare (which would pass ``node_name``/``llm_config`` to factories
    that do not accept them).
    """
    if tier == "fast":
        return _openai_client()
    return _UnclassifiableChatModel()


def _openai_only_factory(tier: str, node_name: str | None = None, llm_config: Any = None) -> Any:
    return _openai_client()


def _raising_factory(tier: str, node_name: str | None = None, llm_config: Any = None) -> Any:
    """A factory that needs a credential it does not have -- ``examples/07``'s
    ``os.environ["OPENROUTER_API_KEY"]`` with no key set. Exporting a spec must
    never require credentials, so this must degrade to ``None``, not propagate."""
    raise KeyError("OPENROUTER_API_KEY")


class _Alpha(BaseModel, frozen=True):
    a: str


class _Out(BaseModel, frozen=True):
    ok: str


def _mixed_tier_construct() -> Any:
    """Two ``think`` nodes on two different tiers -- the mixed-provider shape a
    single uniform ``api_provider=`` string structurally cannot express."""

    @node(outputs=_Alpha, mode="think", model="fast", prompt="produce alpha")
    def first() -> _Alpha: ...

    @node(outputs=_Out, mode="think", model="reason", prompt="judge ${first.a}")
    def second(first: _Alpha) -> _Out: ...

    return construct_from_functions("qtfof13_mixed_tier", [first, second])


def _llm_configs_by_node(flow: Any) -> dict[str, Any]:
    """Every exported ``LlmConfig`` in ``flow``, keyed by the SpecNode carrying it.

    Reads both manifestations of the single ``_make_llm_config`` site: an
    ``LlmNode``'s own ``llm_config`` (think, Oracle merge) and an
    ``AgentNode``'s ``agent.llm_config`` (agent/act).
    """
    found: dict[str, Any] = {}
    for sub in all_flows(flow):
        for spec_node in sub.nodes:
            config = getattr(spec_node, "llm_config", None)
            if config is None:
                config = getattr(getattr(spec_node, "agent", None), "llm_config", None)
            if config is not None:
                found[spec_node.name] = config
    return found


class TestApiProviderIsDerivedPerNodeFromTheResolvedClient:
    """neograph-qtfof.13 step 1-3: the provider is READ OFF the client the
    caller's own ``llm_factory`` resolves each tier to -- per node, never
    guessed, never uniform."""

    def test_a_mixed_provider_construct_derives_a_provider_per_tier(self) -> None:
        flow = to_agent_spec(_mixed_tier_construct(), llm_factory=_mixed_provider_factory)
        configs = _llm_configs_by_node(flow)

        assert configs["first"].api_provider == "openai", (
            "neograph-qtfof.13: the 'fast' tier resolves to a real langchain_openai.ChatOpenAI, "
            "so its exported LlmConfig must name api_provider='openai'"
        )
        assert configs["second"].api_provider is None, (
            "neograph-qtfof.13: the 'reason' tier resolves to a client no provider row classifies -- "
            "honest None, not the other node's provider, and not a guess"
        )

    def test_explicit_api_provider_still_wins_over_the_derived_one(self) -> None:
        """Precedence, in ONE place: explicit -> derived -> None. With a
        one-row mapping, the explicit kwarg stays the only route for a factory
        the introspector cannot classify, so it must not be shadowed."""
        flow = to_agent_spec(_mixed_tier_construct(), api_provider="oci", llm_factory=_openai_only_factory)
        configs = _llm_configs_by_node(flow)

        assert {c.api_provider for c in configs.values()} == {"oci"}, (
            "neograph-qtfof.13: an explicitly-supplied api_provider must win over derivation"
        )

    def test_a_factory_that_raises_degrades_to_none_without_propagating(self) -> None:
        """Exporting a spec must never require credentials."""
        flow = to_agent_spec(_mixed_tier_construct(), llm_factory=_raising_factory)
        configs = _llm_configs_by_node(flow)

        assert {c.api_provider for c in configs.values()} == {None}
        assert {c.model_id for c in configs.values()} == {"fast", "reason"}, (
            "a factory that raised must leave the tier string in place, exactly as no factory does"
        )

    def test_each_tier_is_resolved_once_however_many_nodes_share_it(self) -> None:
        """Amendment A: the public surface builds the resolver ONCE and the
        internal recursion carries it, so a construct on N tiers constructs N
        clients -- not one per node. Resolving a tier has side effects (reads
        env, may open a connection pool); doing it per node is the cost this
        pins against."""
        calls: list[str] = []

        def _counting_factory(tier: str, node_name: str | None = None, llm_config: Any = None) -> Any:
            calls.append(tier)
            return _openai_client()

        @node(outputs=_Alpha, mode="think", model="fast", prompt="one")
        def one() -> _Alpha: ...

        @node(outputs=_Out, mode="think", model="fast", prompt="two ${one.a}")
        def two(one: _Alpha) -> _Out: ...

        to_agent_spec(construct_from_functions("qtfof13_same_tier", [one, two]), llm_factory=_counting_factory)

        assert calls == ["fast"], f"expected the 'fast' tier resolved exactly once, got {calls}"


class TestModelIdAndUrlComeFromTheSameResolvedClient:
    """neograph-qtfof.13 steps 3-4: a derived ``api_provider`` ALONE yields an
    artifact that loads and cannot run -- ``model_id`` stays the neograph tier
    string ``"fast"``, so the adapter builds ``ChatOpenAI(model="fast")``. The
    same resolved instance carries the real ``model_name`` and
    ``openai_api_base``, and the bare-``LlmConfig`` adapter path honours ``url``
    (``base_url=llm_config.url``). All three travel together or the export is
    portable-looking rather than portable."""

    def test_model_id_is_the_resolved_model_name_not_the_neograph_tier(self) -> None:
        flow = to_agent_spec(_mixed_tier_construct(), llm_factory=_mixed_provider_factory)
        configs = _llm_configs_by_node(flow)

        assert configs["first"].model_id == DERIVED_MODEL_NAME, (
            "neograph-qtfof.13: exporting the tier string as model_id names a model no provider has"
        )

    def test_url_travels_with_the_derived_provider(self) -> None:
        """An OpenRouter-pointed ``ChatOpenAI`` is genuinely
        ``api_provider="openai"`` ONLY if its base URL travels with it --
        deriving the provider without the url is the case that most looks right
        and most is not."""
        flow = to_agent_spec(_mixed_tier_construct(), llm_factory=_mixed_provider_factory)
        configs = _llm_configs_by_node(flow)

        assert configs["first"].url == STUB_BASE_URL

    def test_an_unclassifiable_client_keeps_the_tier_string_and_no_url(self) -> None:
        flow = to_agent_spec(_mixed_tier_construct(), llm_factory=_mixed_provider_factory)
        configs = _llm_configs_by_node(flow)

        assert configs["second"].model_id == "reason"
        assert configs["second"].url is None


class TestTheDerivedProviderNeverEntersTheIR:
    """neograph-qtfof.13 step 5 + the Core Invariant: the derived triple is a
    RENDERING of a caller-supplied runtime. ``from_agent_spec`` must still
    collapse it back to the opaque tier string, which is why
    ``_prompt_spec_marker`` has to carry ``model_tier`` BEFORE a concrete
    ``model_id`` is emitted -- a ``think`` node's ``Node.model`` otherwise
    round-trips THROUGH ``llm_config.model_id`` and comes back as
    ``"openai/gpt-4o-mini"``."""

    def test_round_trip_restores_the_original_tier_string(self) -> None:
        construct = _mixed_tier_construct()
        reimported = from_agent_spec(to_agent_spec(construct, llm_factory=_mixed_provider_factory))

        assert {n.name: n.model for n in reimported.nodes} == {"first": "fast", "second": "reason"}, (
            "neograph-qtfof.13: a derived provider/model must never re-enter the IR -- from_agent_spec "
            "must restore the opaque tier string the construct was written with"
        )


class TestDerivedExportClearsTheApiProviderGapAtLoad:
    """neograph-qtfof.13's closing measurement: the SAME cells the opt-in tier
    fixes must become convertible from a supplied runtime alone, with no
    caller-typed provider string.

    Scoped exactly as the opt-in tier above: the assertion is that the
    ``api_provider`` conversion gap is gone. Everything past the conversion is
    credentials + network against an unroutable address and is tolerated -- this
    tier proves the artifact LOADS, never that it runs against a real vendor."""

    @pytest.mark.parametrize("cell_id", LLM_BEARING_CELLS)
    def test_a_supplied_runtime_alone_makes_the_export_convertible(self, cell_id: str) -> None:
        construct = build_cell(*CELLS[cell_id])
        flow = to_agent_spec(construct, llm_factory=_openai_only_factory)

        try:
            run_via_agent_spec_loader(flow, cell_id, stub_registry(flow))
        except NotImplementedError as exc:  # noqa: PERF203 -- one cell per test
            if "api_provider" in str(exc):
                pytest.fail(
                    f"neograph-qtfof.13: cell {cell_id!r} exported an LlmConfig the third-party Agent "
                    f"Spec runtime cannot convert even though the caller supplied the llm_factory that "
                    f"resolves its tier -- {exc}"
                )
            raise
        except Exception:
            # Out of scope, exactly as in the opt-in tier: reaching here means
            # the api_provider gap did not fire.
            pass


class TestTheWholeApiProviderExemptPopulationClearsTheGate:
    """neograph-qtfof.13's closing measurement, over the POPULATION rather than a
    sample.

    ``TestDerivedExportClearsTheApiProviderGapAtLoad`` above proves 4 representative
    cells. The gap this ticket exists to close is 72 cells -- every ``EXEC_EXEMPT``
    entry whose recorded cause is qtfof.8's ``LlmConfig.api_provider=None``. Proving
    4 and reporting 72 would be precisely the self-certification the ticket's own
    notes forbid, so the assertion is driven off ``EXEC_EXEMPT`` itself: the set
    cannot drift out from under this test, and a cell that regresses re-appears here
    automatically.

    SCOPE, stated exactly. Passing means the third-party adapter no longer REFUSES
    the artifact over ``api_provider`` -- it dispatches, and proceeds to build a
    real client. It does NOT mean the exported graph runs: every cell then stops on
    ``OpenAIError: Missing credentials``, because ``api_key`` is a pyagentspec
    SensitiveField replaced by a reference on export. That is correct behaviour (a
    config export ships no secret; the consumer supplies their own) and is the
    strictly weaker claim this ticket deliberately does not make.
    """

    def test_no_api_provider_exempt_cell_is_still_refused_over_api_provider(self) -> None:
        from tests.test_agent_spec_execute import EXEC_EXEMPT

        targets = [c for c, cause in EXEC_EXEMPT.items() if "qtfof.8" in cause or "api_provider" in cause]
        assert targets, "EXEC_EXEMPT no longer records an api_provider cause -- re-derive this test's population"

        still_refused = []
        for cell_id in targets:
            flow = to_agent_spec(build_cell(*CELLS[cell_id]), llm_factory=_openai_only_factory)
            try:
                run_via_agent_spec_loader(flow, cell_id, stub_registry(flow))
            except NotImplementedError as exc:
                if "api_provider" in str(exc):
                    still_refused.append(cell_id)
            except Exception:  # noqa: BLE001,S110 -- past the gate; credentials/network are out of scope
                pass

        assert not still_refused, (
            f"neograph-qtfof.13: {len(still_refused)} of {len(targets)} api_provider-exempt cells are STILL "
            f"refused by a third-party Agent Spec runtime over api_provider even though the caller supplied "
            f"the llm_factory that resolves their tiers: {sorted(still_refused)}"
        )
