"""Phase 1f observability-across-await E2E — CHARACTERIZATION for neograph-w74k.2.6.

CHARACTERIZATION, not TDD red. Phase 1c/1d already threaded ``config`` end-to-end
through the async LLM vertical (see the w74k.2.6 codebase scan: every ``.ainvoke``
hop passes ``config=config``), so this test PASSES on first run. It LOCKS the
already-working observability parity against regression. A FAIL means a real
async config-drop bug in the driver — report it loudly; do NOT paper over it.

SCOPE OF THE CLAIM (refinement mybm.12, re-scoped per the architect review):
this E2E proves ONLY that ``config["callbacks"]`` is NOT dropped through
``arun``'s node fan-out — i.e. the SAME node-level ``on_chain_start`` events that
LangGraph emits under ``run()`` also fire under ``asyncio.run(arun())``. It does
NOT by itself prove LLM-await observability: ``on_chain_start`` is a LangGraph
per-node guarantee under both ``invoke`` and ``ainvoke``, independent of whether
neograph re-threads ``config`` into ``llm.ainvoke``. The LLM-hop proof — that the
``configurable`` marker survives into the awaited ``.ainvoke`` — lives in
``tests/test_async_llm_tool.py:272`` ``TestAsyncThinkConfigThreading`` /
``_ConfigCapturingFake`` (cell 5 of the 1c harness). A true ``on_llm_start`` span
check needs a real ``BaseChatModel`` (the shared duck-typed fakes do not emit LLM
callbacks); that is a deferred manual/opt-in check, not covered here.

No langfuse coupling (optional dep): the handler is a hand-rolled
``langchain_core.callbacks.base.BaseCallbackHandler`` recording node-level chain
starts. Reuses ``StructuredFake`` for the think node.

Driver/runtime E2E — NOT an IR-level shape change, so the 6-cell three-surface
matrix does not apply; a scripted + think @node pipeline suffices (design NOTES).

PARTIAL-COVERAGE FLAGS (design step 6):
  * within-run MCP session reuse (H1/§5) is Phase 3, NOT covered here.
  * tool-invocation config is a symmetric pre-existing gap (sync + async both
    omit config on ``tool_fn.[a]invoke``) filed as neograph-ddb3 — out of 1f
    scope, not band-aided here.
"""

from __future__ import annotations

import asyncio
import types as _types

from langchain_core.callbacks.base import BaseCallbackHandler

import neograph
from neograph import compile, construct_from_module, node
from tests.fakes import build_fake_llm_kwargs, build_test_compile_kwargs
from tests.schemas import Claims, RawText


class _NodeStartRecorder(BaseCallbackHandler):
    """Records the ``name`` of every ``on_chain_start`` LangGraph emits.

    Node runs surface their node name in ``kwargs["name"]``; internal channel /
    graph chains surface other names. We keep the raw list and filter to the
    node names of interest at assertion time.
    """

    def __init__(self) -> None:
        self.chain_names: list[str] = []

    def on_chain_start(self, serialized, inputs, **kwargs) -> None:  # noqa: ANN001
        name = kwargs.get("name")
        if name:
            self.chain_names.append(name)


def _scripted_plus_think_pipeline(fake_factory):
    """fetch (scripted) -> gen (think) — exercises BOTH a scripted node and an
    LLM node so node-level callbacks fire for each under run() and arun()."""
    mod = _types.ModuleType("test_async_obs_mod")

    @node(mode="scripted", outputs=RawText)
    def fetch() -> RawText:
        return RawText(text="hello")

    @node(mode="think", outputs=Claims, model="fast", prompt="test/extract")
    def gen(fetch: RawText) -> Claims: ...

    mod.fetch = fetch
    mod.gen = gen
    return compile(
        construct_from_module(mod, name="async-obs"),
        **build_test_compile_kwargs(),
        **build_fake_llm_kwargs(fake_factory),
    )


def test_node_level_callbacks_fire_the_same_under_run_and_arun():
    """``config["callbacks"]`` is not dropped through arun's node fan-out:
    the SAME node-level ``on_chain_start`` events fire under ``run()`` and under
    ``asyncio.run(arun())``. See module docstring for scope — the LLM-hop config
    proof is ``test_async_llm_tool.py::TestAsyncThinkConfigThreading`` (cell 5)."""
    from tests.fakes import StructuredFake

    def make_fake(tier):
        return StructuredFake(lambda m: m(items=["a", "b"]))

    graph = _scripted_plus_think_pipeline(make_fake)

    h_sync = _NodeStartRecorder()
    sync_result = neograph.run(
        graph,
        input={"node_id": "obs-sync"},
        config={"callbacks": [h_sync], "configurable": {}},
    )

    h_async = _NodeStartRecorder()

    async def _drive():
        return await neograph.arun(
            graph,
            input={"node_id": "obs-async"},
            config={"callbacks": [h_async], "configurable": {}},
        )

    async_result = asyncio.run(_drive())

    # Behavioral parity of the pipeline itself.
    assert sync_result["gen"] == Claims(items=["a", "b"])
    assert async_result["gen"] == sync_result["gen"]

    # The node-level callback events fired on BOTH surfaces for BOTH nodes —
    # config[callbacks] survived arun's node fan-out.
    node_names = {"fetch", "gen"}
    assert node_names <= set(h_sync.chain_names), (
        f"sync run missing node callbacks: saw {h_sync.chain_names}"
    )
    assert node_names <= set(h_async.chain_names), (
        "arun DROPPED config[callbacks] on its node fan-out — node-level "
        f"on_chain_start did not fire under arun: saw {h_async.chain_names}"
    )
    # Same node-level event set under both drivers (parity).
    assert node_names <= (set(h_sync.chain_names) & set(h_async.chain_names))
