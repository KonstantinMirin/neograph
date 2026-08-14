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
