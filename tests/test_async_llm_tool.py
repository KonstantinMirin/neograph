"""Phase 1c LLM/tool async-seam tests — TDD RED for neograph-w74k.2.3.

Pins the Core Invariant of neograph-w74k.2.3: ``ThinkDispatch.aexecute`` and
``ToolDispatch.aexecute`` (which today raise ``NotImplementedError("async
LLM/tool dispatch lands in Phase 1c")``) must actually AWAIT the async twins
``_llm.ainvoke_structured`` / ``_tool_loop.ainvoke_with_tools`` under
``graph.ainvoke`` — awaiting only the network seam (``.invoke`` -> ``.ainvoke``,
``_generate`` -> ``_agenerate``) while reusing every pure preamble/parse/classify
bookend, and threading ``config`` into every ``.ainvoke`` hop (M6a).

Why the async cells are driven at ``compiled.graph.ainvoke`` and NOT ``arun``:
``arun()`` does not exist until Phase 1d (it is the next atom, neograph-w74k.2.4),
and the design NOTES on 1c mandate the ``test_async_dual_path.py`` harness pattern
— drive the graph layer directly. So this suite splats ``build_test_compile_kwargs``
+ ``build_fake_llm_kwargs`` into ``compile()`` and drives ``graph.graph.ainvoke``.

Why parity alone is NOT sufficient (the M1 discriminator, cell 3): a
``ThinkDispatch.aexecute`` that blockingly called the SYNC ``invoke_structured``
on the event loop would ALSO return the right result and pass a parity check while
starving the loop. ``GatedAsyncFake`` is the ready-made discriminator: it has NO
sync ``invoke`` (so a sync-path regression would ``AttributeError``) and its
``ainvoke`` parks on an ``asyncio.Event`` until released. Running ``graph.ainvoke``
as a task and asserting it PARKS on the gate before release proves the ``await``
actually fires on the loop — distinguishing a real ``await`` from a blocking call.

Cell 4 (``StringArgsFake``) locks that ``_CoercingToolWrapper.ainvoke`` does the
explicit string-args coercion recovery rather than silently delegating through
``__getattr__`` to ``self._bound.ainvoke`` (which would bypass coercion under
``arun``). Cell 5 locks the M6a config-threading plumbing on the async think hop.

All cells FAIL NOW: cells 1/2/3/5 hit a stub that raises ``NotImplementedError``
before the seam is reached; cell 4 hits the same stub today and, once
``ToolDispatch.aexecute`` is wired, locks the coercion recovery on the async path.
Reuses the existing async fakes (fakes.py); invents no shared fake infra — the
only test-local class is the tiny config-capturing surface cell 5 requires
(``StructuredFake.with_structured_output`` hardcodes its own class on clone, so a
subclass cannot capture config through the clone).
"""

from __future__ import annotations

import asyncio

from neograph import (
    Construct,
    Each,
    Node,
    Oracle,
    Tool,
    compile,
    construct_from_functions,
    node,
)
from tests.fakes import (
    FakeTool,
    GatedAsyncFake,
    ReActFake,
    StringArgsFake,
    StructuredFake,
    build_fake_llm_kwargs,
    build_test_compile_kwargs,
    register_scripted,
    register_tool_factory,
)
from tests.schemas import Claims, ClusterGroup, Clusters, MatchResult

_CFG = {"configurable": {}}
_INPUT = {"node_id": "t"}


# ═══════════════════════════════════════════════════════════════════════════
# Cell 1 — THINK mode via graph.ainvoke (StructuredFake) + sync parity
# ═══════════════════════════════════════════════════════════════════════════


class TestThinkModeAsyncSeam:
    """``ThinkDispatch.aexecute`` must await ``_llm.ainvoke_structured`` and
    return the structured model under ``graph.ainvoke``. RED now — the stub
    raises ``NotImplementedError('async LLM/tool dispatch lands in Phase 1c')``.
    """

    def test_think_node_returns_structured_model_under_ainvoke(self):
        @node(mode="think", outputs=Claims, model="fast", prompt="test/extract")
        def gen() -> Claims: ...

        graph = compile(
            construct_from_functions("p", [gen]),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(
                lambda tier: StructuredFake(lambda m: m(items=["a", "b"]))
            ),
        )

        # Sync parity: the sync think path already works today (green pre/post).
        sync_result = graph.graph.invoke(dict(_INPUT), dict(_CFG))
        assert sync_result["gen"] == Claims(items=["a", "b"])

        # RED: async think seam raises NotImplementedError today; must return the
        # same structured model once ThinkDispatch.aexecute awaits the twin.
        async_result = asyncio.run(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)))
        assert async_result["gen"] == Claims(items=["a", "b"])
        assert async_result["gen"] == sync_result["gen"]


# ═══════════════════════════════════════════════════════════════════════════
# Cell 2 — AGENT/tool mode via graph.ainvoke (ReActFake + FakeTool)
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentModeAsyncSeam:
    """``ToolDispatch.aexecute`` must await ``_tool_loop.ainvoke_with_tools``:
    an agent node calls a tool then finalizes under ``graph.ainvoke``. RED now —
    the stub raises ``NotImplementedError``.
    """

    def test_agent_node_calls_tool_then_finalizes_under_ainvoke(self):
        search_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda config, tool_config: search_tool)

        fake = ReActFake(
            tool_calls=[
                [{"name": "search", "args": {"q": "test"}, "id": "c1"}],
                [],  # stop — final structured turn
            ],
            final=lambda m: m(items=["done"]),
        )

        @node(
            mode="agent",
            outputs=Claims,
            model="reason",
            prompt="test/explore",
            tools=[Tool(name="search", budget=2)],
        )
        def explore() -> Claims: ...

        graph = compile(
            construct_from_functions("p", [explore]),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: fake),
        )

        # RED: async tool loop raises NotImplementedError today.
        result = asyncio.run(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)))

        assert result["explore"] == Claims(items=["done"])
        # The awaited tool_fn.ainvoke hop actually ran the tool once.
        assert search_tool.calls == [{"q": "test"}]


# ═══════════════════════════════════════════════════════════════════════════
# Cell 3 — M1 DISCRIMINATOR: the await must FIRE on the loop (GatedAsyncFake)
# ═══════════════════════════════════════════════════════════════════════════


class TestThinkSeamAwaitFiresOnLoop:
    """Load-bearing (M1): parity cannot prove the ``await`` fired. Drive a THINK
    node under ``graph.ainvoke`` with ``GatedAsyncFake`` — it has NO sync
    ``invoke`` (a blocking sync-path regression would ``AttributeError``) and its
    ``ainvoke`` parks on a gate. Assert the run PARKS (``enter_count``) before
    release, proving a real ``await`` on the loop, then release and await the
    structured result. RED now — the stub raises ``NotImplementedError`` before
    ever reaching the gate, so the run never parks.
    """

    async def test_think_seam_parks_on_gated_ainvoke_then_completes(self):
        fake = GatedAsyncFake(lambda m: m(items=["gated"]))

        @node(mode="think", outputs=Claims, model="fast", prompt="test/extract")
        def gen() -> Claims: ...

        graph = compile(
            construct_from_functions("p", [gen]),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: fake),
        )

        task = asyncio.create_task(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)))

        # Give the loop several ticks for the think seam to reach the gate. Today
        # the task fails fast with NotImplementedError (task.done() True,
        # enter_count 0), so this loop breaks with the run NOT parked.
        for _ in range(50):
            if fake.enter_count or task.done():
                break
            await asyncio.sleep(0.01)

        assert fake.enter_count == 1 and not task.done(), (
            "think seam did not park on GatedAsyncFake.ainvoke — the async twin "
            "did not await the network seam on the loop (ThinkDispatch.aexecute "
            "still raises NotImplementedError, or blocks the loop synchronously)"
        )

        fake.release()
        result = await asyncio.wait_for(task, timeout=2.0)
        assert result["gen"] == Claims(items=["gated"])


# ═══════════════════════════════════════════════════════════════════════════
# Cell 4 — _CoercingToolWrapper.ainvoke string-args coercion recovery
# ═══════════════════════════════════════════════════════════════════════════


class TestCoercingToolWrapperAsyncRecovery:
    """``StringArgsFake`` emits ``tool_calls.args`` as JSON strings, so the
    initial ``ainvoke`` raises ``ValidationError``; ``_CoercingToolWrapper.ainvoke``
    must recover via ``await self._bound._agenerate(...)`` (the async mirror of the
    sync ``_generate`` path) rather than silently delegating through ``__getattr__``
    to ``self._bound.ainvoke`` (which bypasses coercion). RED now — the async tool
    loop stub raises ``NotImplementedError``; once wired, this locks that the
    coerced dict args reach the tool.
    """

    def test_string_args_coerced_on_async_tool_path(self):
        search_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda config, tool_config: search_tool)

        fake = StringArgsFake(
            tool_calls=[
                [{"name": "search", "args": {"q": "test"}, "id": "c1"}],
                [],  # final
            ],
            final=lambda m: m(items=["coerced"]),
        )

        @node(
            mode="agent",
            outputs=Claims,
            model="fast",
            prompt="test/explore",
            tools=[Tool(name="search", budget=2)],
        )
        def explore() -> Claims: ...

        graph = compile(
            construct_from_functions("p", [explore]),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: fake),
        )

        result = asyncio.run(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)))

        assert result["explore"] == Claims(items=["coerced"])
        # The string args were coerced back to a dict before the tool ran — the
        # __getattr__ passthrough would have skipped recovery and never coerced.
        assert search_tool.calls == [{"q": "test"}]


# ═══════════════════════════════════════════════════════════════════════════
# Cell 4b — _generate/_agenerate stay valid attributes (langchain-core contract)
# ═══════════════════════════════════════════════════════════════════════════


class TestCoercingWrapperGenerateAttributePin:
    """PP-01 (neograph-yc38): the string-args recovery re-invokes the model's
    low-level ``_generate``/``_agenerate`` — langchain-core private methods.

    If a langchain-core bump renames or removes them, ``getattr`` would raise
    inside the recovery ``try`` and the wrapper would SILENTLY take the
    empty-``AIMessage`` fallback (a real tool turn dropped to a blank response).
    Pin the attribute on a REAL ``BaseChatModel`` and through the wrapper's
    ``__getattr__`` forwarding so such a bump breaks CI here, loudly, instead."""

    def test_generate_is_a_valid_attribute_on_base_chat_model(self):
        from langchain_core.language_models.chat_models import BaseChatModel

        # The recovery depends on both hooks existing on the model base class.
        assert hasattr(BaseChatModel, "_generate")
        assert hasattr(BaseChatModel, "_agenerate")

    def test_wrapper_forwards_generate_to_a_real_bound_model(self):
        from langchain_core.language_models.fake_chat_models import (
            GenericFakeChatModel,
        )
        from langchain_core.messages import AIMessage

        from neograph._tool_loop import _CoercingToolWrapper

        model = GenericFakeChatModel(messages=iter([AIMessage(content="x")]))
        wrapper = _CoercingToolWrapper(model)

        # __getattr__ forwards to self._bound; a rename in langchain-core would
        # make these raise AttributeError instead of resolving to a callable.
        assert callable(wrapper._generate)
        assert callable(wrapper._agenerate)


# ═══════════════════════════════════════════════════════════════════════════
# Cell 5 — M6a config-threading on the async think hop
# ═══════════════════════════════════════════════════════════════════════════


class _ConfigCapturingFake:
    """Test-local think fake that records the ``config`` its ``ainvoke`` receives.

    Not shared infra: cell 5 explicitly needs a config-capturing surface, and the
    shared ``StructuredFake.with_structured_output`` hardcodes ``StructuredFake``
    on its clone, so a subclass cannot capture config through the clone. Kept
    minimal and local to this file.
    """

    def __init__(self, respond, captured: dict) -> None:
        self._respond = respond
        self._model = None
        self.captured = captured

    def with_structured_output(self, model, **kwargs) -> _ConfigCapturingFake:
        clone = _ConfigCapturingFake(self._respond, self.captured)
        clone._model = model
        return clone

    def invoke(self, messages, **kwargs):
        self.captured["config"] = kwargs.get("config")
        return self._respond(self._model)

    async def ainvoke(self, *a, **k):
        self.captured["config"] = k.get("config")
        return self._respond(self._model)


class TestAsyncThinkConfigThreading:
    """M6a: the async think twin must thread ``config`` into ``.ainvoke`` so
    observability survives the await. Assert the ``configurable`` marker set on
    ``graph.ainvoke``'s config reaches the LLM's ``ainvoke``. RED now — the stub
    raises ``NotImplementedError`` before any config is threaded.
    """

    def test_config_threads_into_ainvoke_on_think_seam(self):
        captured: dict = {}

        @node(mode="think", outputs=Claims, model="fast", prompt="test/extract")
        def gen() -> Claims: ...

        graph = compile(
            construct_from_functions("p", [gen]),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(
                lambda tier: _ConfigCapturingFake(lambda m: m(items=["x"]), captured)
            ),
        )

        cfg = {"configurable": {"_async_marker": "thread-42"}}
        asyncio.run(graph.graph.ainvoke(dict(_INPUT), dict(cfg)))

        threaded = captured.get("config")
        assert threaded is not None, "config was not threaded into the async .ainvoke hop"
        assert threaded.get("configurable", {}).get("_async_marker") == "thread-42", (
            "async think twin dropped the configurable marker on the .ainvoke hop "
            "— observability (callbacks/tracer) would not survive the await (M6a)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Cell 6 — Oracle merge barrier async seam (neograph-p3c7)
# ═══════════════════════════════════════════════════════════════════════════


async def _drive_until_parked_or_done(graph, fake):
    """Launch ``graph.ainvoke`` as a task; spin the loop until the merge either
    parks on ``fake.ainvoke`` or the task finishes. Returns the task."""
    task = asyncio.create_task(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)))
    for _ in range(60):
        if fake.enter_count or task.done():
            break
        await asyncio.sleep(0.01)
    return task


class TestOracleMergeAsyncSeam:
    """neograph-p3c7 — the Oracle merge BARRIER (LLM-judge merge) must await its
    async twin under ``graph.ainvoke``, same as the generator side already does.

    Today ``make_oracle_merge_fn`` (single-group) and ``_wiring.group_merge_barrier``
    (Each×Oracle fused) return a single sync closure calling ``invoke_structured``
    (a SYNC LLM call) — no ``afunc`` twin. Under ``graph.ainvoke`` LangGraph
    threadpools the barrier and the LLM merge runs on the sync path, blocking the
    loop (Phase-1 H2 invariant violation). The generator redirects were made
    dual-path in Phase 1a; the merge barriers were missed.

    Discriminator (gate-parking, NOT parity — see module docstring): ``GatedAsyncFake``
    has NO sync ``invoke``. Generators are SCRIPTED so the ONLY LLM call is the
    merge (``merge_prompt``). On the async path the merge awaits
    ``ainvoke_structured`` -> ``fake.ainvoke`` and PARKS (``enter_count == 1``,
    task not done); on the sync path (today) it calls ``invoke_structured`` ->
    ``fake.invoke`` and the run fails fast with ``AttributeError`` — the run never
    parks. RED now: ``enter_count == 0`` and the task is already done (failed).
    """

    def test_single_group_oracle_merge_parks_on_async_seam(self):
        """``make_oracle_merge_fn`` — programmatic ``Node.scripted | Oracle``."""
        register_scripted("mg_gen", lambda input_data, config: Claims(items=["v"]))
        fake = GatedAsyncFake(lambda m: m(items=["merged"]))

        pipeline = Construct(
            "p",
            nodes=[
                Node.scripted("mg_gen", fn="mg_gen", outputs=Claims)
                | Oracle(n=2, merge_prompt="test/merge"),
            ],
        )
        graph = compile(
            pipeline,
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: fake),
        )

        async def run_it():
            task = await _drive_until_parked_or_done(graph, fake)
            assert fake.enter_count == 1 and not task.done(), (
                "Oracle merge barrier did not park on GatedAsyncFake.ainvoke — the "
                "merge ran on the sync path (invoke_structured) under graph.ainvoke "
                "instead of awaiting ainvoke_structured (neograph-p3c7). "
                f"enter_count={fake.enter_count}, task.done={task.done()}"
            )
            fake.release()
            result = await asyncio.wait_for(task, timeout=2.0)
            return result

        result = asyncio.run(run_it())
        assert result["mg_gen"] == Claims(items=["merged"])

    def test_each_oracle_group_merge_parks_on_async_seam(self):
        """``_wiring.group_merge_barrier`` — FLAT Each×Oracle fusion (``map_over``
        + ``ensemble_n`` on ONE node) with an LLM ``merge_prompt``. This routes to
        ``ModifierCombo.EACH_ORACLE`` -> ``_wire_each_oracle`` ->
        ``group_merge_barrier`` — a distinct closure from make_oracle_merge_fn.

        Single group (one item) so exactly one merge fires: on the async path it
        awaits ``ainvoke_structured`` and parks; on the sync path (today) it calls
        ``invoke_structured`` -> ``fake.invoke`` and fails fast.
        """
        fake = GatedAsyncFake(lambda m: m(cluster_label="merged", matched=["x"]))

        register_scripted(
            "eo_mk",
            lambda _in, _cfg: Clusters(groups=[ClusterGroup(label="a", claim_ids=["c1"])]),
        )
        register_scripted(
            "eo_gen",
            lambda item, cfg: MatchResult(cluster_label="item", matched=["m"]),
        )
        pipeline = Construct(
            "p",
            nodes=[
                Node.scripted("eo_mk", fn="eo_mk", outputs=Clusters),
                Node.scripted("eo_gen", fn="eo_gen", inputs=ClusterGroup, outputs=MatchResult)
                | Oracle(n=2, merge_prompt="test/merge")
                | Each(over="eo_mk.groups", key="label"),
            ],
        )
        graph = compile(
            pipeline,
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: fake),
        )

        async def run_it():
            task = await _drive_until_parked_or_done(graph, fake)
            assert fake.enter_count >= 1 and not task.done(), (
                "Each×Oracle group_merge_barrier did not park on GatedAsyncFake."
                "ainvoke — the fused merge ran on the sync path under graph.ainvoke "
                "instead of awaiting ainvoke_structured (neograph-p3c7). "
                f"enter_count={fake.enter_count}, task.done={task.done()}"
            )
            fake.release()
            await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(run_it())
