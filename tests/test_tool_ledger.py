"""Tool ledger: per-tool ordinal stamping + the ``ToolLedger`` selector view (neograph-ftnxl.5).

TDD RED. Every test here is written against the INTENDED API and fails until
neograph-fe1d6.7 implements it:

- ``ToolInteraction.ordinal: int = 0`` — 1-based, per ``tool_name``, per node.
  ``0`` means "not a budgeted invocation" (a ``transfer_to_<peer>`` handoff ack,
  a synthetic record, or a pre-0.8 checkpointed one).
- ``ToolInteraction.key -> str | None`` — the canonical address
  ``f"{tool_name}#{ordinal}"``, ``None`` when ``ordinal == 0`` (unaddressable).
- ``neograph.ToolLedger`` — a pure read-time view over an existing
  ``list[ToolInteraction]``: ``first``/``last``/``all``/``grouped``/``by_key``
  plus ``__iter__``/``__len__``.

**Selector semantics pinned here** (decided before the tests were written, per
the design's step-0 instruction): absence is ``None``, a caller bug is a raise.
``first(name)``/``last(name)`` return ``None`` for a tool that was never called
(never raise); ``all(name)`` returns ``[]``. ``by_key(key)`` returns ``None`` for
a well-formed key with no match, but raises ``ValueError`` for a ``None``/empty
key argument — a caller that passes an unaddressable record's ``key`` straight
through must fail loud rather than silently look up nothing.

The load-bearing cell is the async batch one: the async twin PRE-RESERVES budget
in Phase 1 and only builds the ``ToolInteraction`` in Phase 3, so an ordinal read
at build time yields ``(2, 2)`` for two concurrent calls to the same tool instead
of ``(1, 2)``. Ordinals must be captured at the pre-reserve point.

The durability boundary is pinned too: ``tool_name``/``args``/``result``/
``duration_ms``/``ordinal`` survive a real checkpoint round-trip; ``typed_result``
is documented and TESTED as RESUME-VOLATILE (langgraph's serializer encodes the
record via ``model_dump()``, which flattens the ``Any``-typed field, so a
``BaseModel`` result reads back as a plain ``dict``). That test DOCUMENTS the
contract — it is not a bug repro.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, ConfigDict, Field

from neograph import (
    Construct,
    Node,
    Portal,
    Tool,
    ToolInteraction,
    compile,
    construct_from_functions,
    node,
    run,
)
from neograph._state_keys import StateKeys
from neograph.errors import CheckpointSchemaError
from tests.fakes import (
    ReActFake,
    build_fake_llm_kwargs,
    build_test_compile_kwargs,
    register_tool_factory,
)
from tests.schemas import Claims

_CFG = {"configurable": {}}
_INPUT = {"node_id": "t"}
_TOOL_LOG_OUTPUTS = {"result": Claims, "tool_log": list[ToolInteraction]}


class _CountingTool:
    """Counts real invocations and echoes its args, mirroring the ``.invoke`` /
    ``.ainvoke`` surface the agent cycle drives."""

    def __init__(self, name: str, result: Any | None = None):
        self.name = name
        self.calls: list[dict] = []
        self._result = result

    def _answer(self, args: dict) -> Any:
        if self._result is not None:
            return self._result
        return f"{self.name}:{args.get('q', '?')}"

    def invoke(self, args: dict, config=None) -> Any:
        self.calls.append(dict(args))
        return self._answer(args)

    async def ainvoke(self, args: dict, config=None) -> Any:
        self.calls.append(dict(args))
        return self._answer(args)


def _agent_node(tools: list[Tool], outputs: Any = None):
    @node(
        mode="agent",
        outputs=outputs if outputs is not None else _TOOL_LOG_OUTPUTS,
        model="reason",
        prompt="test/explore",
        tools=tools,
    )
    def explore() -> Claims: ...

    return explore


def _compile(tools: list[Tool], turns: list[list[dict]], outputs: Any = None, **kw):
    """Compile a one-agent-node pipeline driven by a scripted ReAct fake. A FRESH
    fake per compile so a re-compiled graph replays the same script."""
    fake = ReActFake(
        tool_calls=[*turns, []],  # scripted tool-using turns, then the final turn
        final=lambda m: m(items=["done"]),
    )
    return compile(
        construct_from_functions("p", [_agent_node(tools, outputs)]),
        **build_test_compile_kwargs(),
        **build_fake_llm_kwargs(lambda tier: fake),
        **kw,
    )


def _call(name: str, q: str, call_id: str) -> dict:
    return {"name": name, "args": {"q": q}, "id": call_id}


def _stamps(tool_log: list) -> list[tuple[str, int]]:
    """``[(tool_name, ordinal), ...]`` in tool_log order."""
    return [(t.tool_name, t.ordinal) for t in tool_log]


# =============================================================================
# (a) / (b) — ordinals increment per tool name, sync and async twins
# =============================================================================


class TestOrdinalIncrementsPerToolName:
    """The ordinal is per ``tool_name``, per node, 1-based, and keeps counting
    across ReAct turns (the counter it derives from is restored from the
    checkpointed budget on every superstep)."""

    def test_ordinals_increment_per_tool_name_when_sync_agent_runs_multiple_turns(self):
        search = _CountingTool("search")
        fetch = _CountingTool("fetch")
        register_tool_factory("search", lambda config, tool_config: search)
        register_tool_factory("fetch", lambda config, tool_config: fetch)

        graph = _compile(
            [Tool(name="search", budget=5), Tool(name="fetch", budget=5)],
            [
                [_call("search", "a", "c1"), _call("fetch", "x", "c2")],
                [_call("search", "b", "c3")],
            ],
        )
        result = run(graph, input=dict(_INPUT), config=dict(_CFG))

        assert _stamps(result["explore_tool_log"]) == [
            ("search", 1),
            ("fetch", 1),
            ("search", 2),
        ], "ordinals are 1-based and counted PER TOOL NAME, continuing across turns"

    def test_ordinals_increment_per_tool_name_when_async_agent_runs_multiple_turns(self):
        """Async twin of the cell above — the sync/async twin axis is the one that
        actually needs doubling here (three-surface parity is exempt: this is a
        dispatch/runtime-layer change)."""
        search = _CountingTool("search")
        fetch = _CountingTool("fetch")
        register_tool_factory("search", lambda config, tool_config: search)
        register_tool_factory("fetch", lambda config, tool_config: fetch)

        graph = _compile(
            [Tool(name="search", budget=5), Tool(name="fetch", budget=5)],
            [
                [_call("search", "a", "c1"), _call("fetch", "x", "c2")],
                [_call("search", "b", "c3")],
            ],
        )
        result = asyncio.run(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)))

        assert _stamps(result["explore_tool_log"]) == [
            ("search", 1),
            ("fetch", 1),
            ("search", 2),
        ], "the async twin must stamp the same ordinals as the sync twin"


# =============================================================================
# (c) — THE load-bearing cell: async batch ordinals come from the PRE-RESERVE
# =============================================================================


def test_concurrent_same_tool_calls_get_distinct_ordinals_when_batched_in_one_async_turn():
    """Two calls to the SAME tool in ONE async tool_call batch must be stamped
    (1, 2) in tool_call order.

    The async twin pre-reserves budget sequentially in Phase 1 and builds the
    ToolInteraction in Phase 3, AFTER the gather — by then the shared counter
    already holds the batch total. A build-time read therefore stamps (2, 2).
    The ordinal must be captured at the Phase-1 pre-reserve point and carried
    through the plan, mirroring the neograph-dyy7 budget pre-reserve.
    """
    search = _CountingTool("search")
    register_tool_factory("search", lambda config, tool_config: search)

    graph = _compile(
        [Tool(name="search", budget=5)],
        [[_call("search", "a", "c1"), _call("search", "b", "c2")]],
    )
    result = asyncio.run(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)))

    tool_log = result["explore_tool_log"]
    assert [t.args["q"] for t in tool_log] == ["a", "b"], "precondition: assembly is in tool_call order"
    assert _stamps(tool_log) == [("search", 1), ("search", 2)], (
        "concurrent calls to one tool must be stamped at PRE-RESERVE time; a build-time counter read gives (2, 2)"
    )


# =============================================================================
# (d) / (g) — the durability boundary, against a REAL file-backed Sqlite saver
# =============================================================================


class _Deal(BaseModel):
    """A typed tool result — a BaseModel, so it is msgpack-encodable (a result
    that is neither pydantic nor dataclass raises at checkpoint WRITE today)."""

    id: int
    name: str


def _read_back_tool_log(db: str, thread: dict, turns: list[list[dict]], tools: list[Tool]) -> list:
    """Reopen the sqlite file with a FRESH saver + a freshly compiled graph and
    read the persisted tool_log — a genuine msgpack round-trip, not the
    in-memory objects the first run happened to hold."""
    with SqliteSaver.from_conn_string(db) as saver:
        graph = _compile(tools, turns, checkpointer=saver)
        snapshot = graph.graph.get_state(thread)
    return snapshot.values["explore_tool_log"]


def test_ordinal_survives_when_tool_log_is_read_back_from_a_sqlite_checkpoint(tmp_path):
    """The ordinal is part of the DURABLE half of the contract: it must come back
    off a real file-backed checkpoint exactly as stamped, alongside tool_name /
    args / rendered result / duration_ms."""
    search = _CountingTool("search")
    register_tool_factory("search", lambda config, tool_config: search)
    tools = [Tool(name="search", budget=5)]
    turns = [[_call("search", "a", "c1")], [_call("search", "b", "c2")]]

    db = str(tmp_path / "ledger.db")
    thread = {"configurable": {"thread_id": "ledger-ordinal"}}
    with SqliteSaver.from_conn_string(db) as saver:
        graph = _compile(tools, turns, checkpointer=saver)
        run(graph, input=dict(_INPUT), config=thread)

    restored = _read_back_tool_log(db, thread, turns, tools)

    assert _stamps(restored) == [("search", 1), ("search", 2)], (
        "ordinals must survive a real checkpoint round-trip unchanged"
    )
    assert [t.args["q"] for t in restored] == ["a", "b"], "the durable recipe (args) survives too"


def test_typed_result_is_resume_volatile_when_read_back_from_a_sqlite_checkpoint(tmp_path):
    """DOCUMENTS the contract, does NOT repro a bug: ToolLedger promises durability
    only over (tool_name, args, ordinal, rendered result, duration_ms).

    langgraph encodes a pydantic object as ``model_dump()``, which flattens the
    ``Any``-typed ``typed_result`` one level — a ``_Deal`` comes back a plain
    ``dict``. The ledger follows ProducingCall's precedent (persist the RECIPE,
    not the result) rather than silently promising a durability it cannot keep.
    """
    search = _CountingTool("search", result=_Deal(id=1, name="x"))
    register_tool_factory("search", lambda config, tool_config: search)
    tools = [Tool(name="search", budget=5)]
    turns = [[_call("search", "a", "c1")]]

    db = str(tmp_path / "volatile.db")
    thread = {"configurable": {"thread_id": "ledger-volatile"}}
    with SqliteSaver.from_conn_string(db) as saver:
        graph = _compile(tools, turns, checkpointer=saver)
        live = run(graph, input=dict(_INPUT), config=thread)["explore_tool_log"][0]

    assert isinstance(live.typed_result, _Deal), "in-process, typed_result is the live object"

    restored = _read_back_tool_log(db, thread, turns, tools)[0]

    assert isinstance(restored.typed_result, dict), (
        "typed_result is RESUME-VOLATILE by contract: a BaseModel result reads "
        "back as a plain dict after a checkpoint round-trip"
    )
    # ...while everything the ledger DOES promise survives, ordinal included.
    assert restored.tool_name == "search"
    assert restored.args == {"q": "a"}
    assert restored.result == live.result
    assert restored.ordinal == 1


# =============================================================================
# (e) — an idempotent repeat consumes no ordinal
# =============================================================================


def test_repeat_served_call_consumes_no_ordinal_when_tool_is_idempotent():
    """An idempotent repeat is served from the cycle's own history: no re-invoke,
    no budget spend, no ToolInteraction — and therefore no ordinal. The NEXT
    genuinely-new call must be stamped 2, not 3."""
    sweep = _CountingTool("sweep")
    register_tool_factory("sweep", lambda config, tool_config: sweep)

    graph = _compile(
        [Tool(name="sweep", budget=3, idempotent=True)],
        [
            [_call("sweep", "FORMAT", "c1")],
            [_call("sweep", "FORMAT", "c2-retry")],  # repeat: served from cache
            [_call("sweep", "DEVICE", "c3")],
        ],
    )
    result = run(graph, input=dict(_INPUT), config=dict(_CFG))

    tool_log = result["explore_tool_log"]
    assert [t.args["q"] for t in tool_log] == ["FORMAT", "DEVICE"], "precondition: the repeat produced no record"
    assert _stamps(tool_log) == [("sweep", 1), ("sweep", 2)], (
        "a repeat-served call must consume no ordinal — DEVICE is the 2nd real invocation"
    )


# =============================================================================
# (f) — a handoff ack is a routing signal, never a budgeted invocation
# =============================================================================


class _Handoff(BaseModel, frozen=True):
    goto: str


def _tool_trigger_mesh() -> Construct:
    """Two-agent tool-triggered mesh. ``triage`` owns a real tool AND emits the
    synthesized ``transfer_to_researcher`` call in the same batch, so one
    tool_log holds both an ordinal-1 record and an ordinal-0 ack."""
    triage = Node(
        name="triage",
        mode="agent",
        model="router",
        prompt="test/triage",
        inputs={"handoff": _Handoff},
        outputs=_Handoff,
        tools=[Tool(name="search", budget=5)],
    ) | Portal(to=["researcher"], trigger="tool", max_hops=6)
    researcher = Node(
        name="researcher",
        mode="agent",
        model="worker",
        prompt="test/research",
        inputs={"handoff": _Handoff},
        outputs=_Handoff,
        tools=[],
    ) | Portal(to=["triage"], trigger="tool")
    return Construct("tool-trigger-mesh", nodes=[triage, researcher])


def test_handoff_ack_keeps_ordinal_zero_when_transfer_tool_is_called():
    """``transfer_to_<peer>`` is intercepted by NAME before budget/idempotency —
    it never touches the counter, so its ToolInteraction keeps ordinal 0 and is
    NOT addressable. A real tool call in the same batch is still stamped 1.

    Read off the internal ``neo_agent_tool_log_*`` channel via streamed updates:
    ``triage`` hands off before it ever parses a typed output, so nothing is
    exposed through a declared ``tool_log`` output field.
    """
    search = _CountingTool("search")
    register_tool_factory("search", lambda config, tool_config: search)
    fakes = {
        "router": ReActFake(
            tool_calls=[
                [
                    _call("search", "a", "c1"),
                    {"name": "transfer_to_researcher", "args": {}, "id": "tr1"},
                ]
            ],
            final=lambda m: m(goto="__end__"),
            output_model=_Handoff,
        ),
        "worker": ReActFake(tool_calls=[[]], final=lambda m: m(goto="__end__"), output_model=_Handoff),
    }
    graph = compile(
        _tool_trigger_mesh(),
        **build_test_compile_kwargs(),
        **build_fake_llm_kwargs(lambda tier: fakes[tier]),
    )

    tlog_key = StateKeys.agent_tool_log("triage")
    interactions: list = []
    for chunk in graph.graph.stream({}, dict(_CFG), stream_mode="updates"):
        for update in chunk.values():
            if isinstance(update, dict):
                interactions.extend(update.get(tlog_key) or [])

    assert _stamps(interactions) == [("search", 1), ("transfer_to_researcher", 0)], (
        "a handoff ack must keep ordinal 0 (never spends budget, never addressable) "
        "while a real call in the same batch is stamped 1"
    )
    assert interactions[1].key is None, "an ordinal-0 record has no address"


# =============================================================================
# (h) — the ToolLedger selector view
# =============================================================================


def test_ledger_selectors_address_records_when_ordinals_are_stamped():
    """``ToolLedger`` is a pure read-time view over the list a consumer already
    receives from a node's declared ``tool_log`` output.

    Semantics pinned: absence is ``None``/``[]`` (never a raise) for
    ``first``/``last``/``all``/``by_key``; a ``None``/empty key argument to
    ``by_key`` is a caller bug and raises ``ValueError``.
    """
    from neograph import ToolLedger  # local: the module does not exist yet (TDD red)

    search = _CountingTool("search")
    fetch = _CountingTool("fetch")
    register_tool_factory("search", lambda config, tool_config: search)
    register_tool_factory("fetch", lambda config, tool_config: fetch)

    graph = _compile(
        [Tool(name="search", budget=5), Tool(name="fetch", budget=5)],
        [
            [_call("search", "a", "c1"), _call("fetch", "x", "c2")],
            [_call("search", "b", "c3")],
        ],
    )
    tool_log = run(graph, input=dict(_INPUT), config=dict(_CFG))["explore_tool_log"]
    ledger = ToolLedger(tool_log)

    assert len(ledger) == 3
    assert list(ledger) == list(tool_log), "iteration preserves the underlying order"

    assert ledger.first("search").args["q"] == "a"
    assert ledger.last("search").args["q"] == "b"
    assert [t.args["q"] for t in ledger.all("search")] == ["a", "b"]
    assert ledger.first("fetch") is ledger.last("fetch"), "one call: first and last are the same record"

    assert set(ledger.grouped()) == {"search", "fetch"}
    assert [t.ordinal for t in ledger.grouped()["search"]] == [1, 2]

    # The canonical address, single-sited on the record and consumed by by_key.
    assert tool_log[0].key == "search#1"
    assert ledger.by_key("search#2").args["q"] == "b"
    assert ledger.by_key("fetch#1").tool_name == "fetch"

    # Absence -> None / []. Never a raise.
    assert ledger.first("never_called") is None
    assert ledger.last("never_called") is None
    assert ledger.all("never_called") == []
    assert ledger.by_key("search#99") is None


# =============================================================================
# (i) — the resume consequence of adding a field to a PUBLIC frozen model
# =============================================================================


class _PreOrdinalToolInteraction(BaseModel):
    """Stand-in for a pre-ordinal ``ToolInteraction``: the exact field set the
    model had before this ticket, spoofing ``ToolInteraction``'s
    ``module.Qualname`` so ``_type_signature`` compares FIELD SETS rather than
    trivially diverging on the class identity."""

    model_config = ConfigDict(frozen=True, from_attributes=True)

    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    typed_result: Any = None
    duration_ms: int = 0


_PreOrdinalToolInteraction.__module__ = "neograph.tool"
_PreOrdinalToolInteraction.__qualname__ = "ToolInteraction"


def _fingerprints(ti_type: type, **kw) -> tuple[str, dict[str, str]]:
    graph = _compile(
        [Tool(name="search", budget=5)],
        [[_call("search", "a", "c1")]],
        outputs={"result": Claims, "tool_log": list[ti_type]},
        **kw,
    )
    return graph.schema_fingerprint, graph.node_fingerprints


def test_fingerprints_change_when_tool_interaction_gains_a_field():
    """A node declaring ``tool_log`` in dict-form outputs gets a NON-``neo_``
    state field, which IS folded into both fingerprints. Adding ``ordinal`` to
    the public frozen ``ToolInteraction`` therefore CHANGES the schema
    fingerprint and that node's per-node fingerprint — the opposite of the
    "never triggers invalidation" claim the plan started from.

    Pinned so the user-visible resume consequence is a stated, tested fact
    rather than a silent ship.
    """
    register_tool_factory("search", lambda config, tool_config: _CountingTool("search"))

    old_schema_fp, old_node_fps = _fingerprints(_PreOrdinalToolInteraction)
    new_schema_fp, new_node_fps = _fingerprints(ToolInteraction)

    assert new_node_fps["explore_tool_log"] != old_node_fps["explore_tool_log"], (
        "the tool_log node fingerprint MUST change when ToolInteraction gains a field"
    )
    assert new_schema_fp != old_schema_fp, (
        "the schema fingerprint must change too — it is the GATE; if it matches, "
        "the enriched node fingerprint is never even consulted"
    )
    assert new_node_fps["explore_result"] == old_node_fps["explore_result"], (
        "the unrelated output key must NOT be invalidated"
    )


def test_old_checkpoint_does_not_silently_resume_when_tool_interaction_gains_a_field(tmp_path):
    """Resuming a PRE-CHANGE checkpoint of a tool_log-declaring node must not
    hand back a stale tip.

    Test (d) cannot catch this — it writes and resumes within one version. Here
    v1 declares the pre-ordinal field set and v2 declares the current
    ``ToolInteraction``, so the schema gate opens on resume. Either outcome is
    acceptable and both are fail-loud-or-recompute: a ``CheckpointSchemaError``
    carrying the invalidated node, or a genuine re-execution of the agent node.
    Silently returning the tip (what happens while the fingerprints still match)
    is the failure this pins.
    """
    search = _CountingTool("search")
    register_tool_factory("search", lambda config, tool_config: search)
    tools = [Tool(name="search", budget=5)]
    turns = [[_call("search", "a", "c1")]]

    db = str(tmp_path / "old_checkpoint.db")
    thread = {"configurable": {"thread_id": "ledger-old-checkpoint"}}
    raised: CheckpointSchemaError | None = None

    with SqliteSaver.from_conn_string(db) as saver:
        graph_v1 = _compile(
            tools, turns, outputs={"result": Claims, "tool_log": list[_PreOrdinalToolInteraction]}, checkpointer=saver
        )
        run(graph_v1, input=dict(_INPUT), config=thread)
        calls_after_v1 = len(search.calls)

        graph_v2 = _compile(tools, turns, checkpointer=saver)
        try:
            run(graph_v2, input=dict(_INPUT), config=thread, auto_resume=True)
        except CheckpointSchemaError as exc:
            raised = exc

    if raised is not None:
        assert raised.invalidated_nodes, "the error must name what was invalidated"
    else:
        assert len(search.calls) > calls_after_v1, (
            "a pre-change checkpoint must not silently resume from the tip: the "
            "invalidated agent node has to re-execute (or the resume must raise "
            "CheckpointSchemaError)"
        )


# =============================================================================
# (j) — stamping is unconditional; only EXPOSURE is demand-gated
# =============================================================================


def test_ordinal_is_stamped_when_node_does_not_declare_tool_log_output():
    """Collection is unconditional inside the ReAct loop — only EXPOSURE through
    a declared dict-form ``tool_log`` output is demand-gated. A node declaring a
    single output type still accumulates ordinal-stamped records on its internal
    channel (pins the AGENTS.md "no collection overhead" correction)."""
    search = _CountingTool("search")
    register_tool_factory("search", lambda config, tool_config: search)

    graph = _compile(
        [Tool(name="search", budget=5)],
        [[_call("search", "a", "c1")], [_call("search", "b", "c2")]],
        outputs=Claims,  # single-type: NO tool_log output declared
    )

    tlog_key = StateKeys.agent_tool_log("explore")
    interactions: list = []
    for chunk in graph.graph.stream(dict(_INPUT), dict(_CFG), stream_mode="updates"):
        for update in chunk.values():
            if isinstance(update, dict):
                interactions.extend(update.get(tlog_key) or [])

    assert _stamps(interactions) == [("search", 1), ("search", 2)], (
        "ordinals are stamped even when no consumer declares tool_log — only exposure is demand-gated, not collection"
    )


# =============================================================================
# (k) — ordinal 0 is UNADDRESSABLE, and by_key says so loudly
# =============================================================================


def test_by_key_refuses_unaddressable_records_when_ordinal_is_zero():
    """Two ordinal-0 records (a handoff ack and a pre-change checkpointed record)
    must not collide into one bogus address. ``key`` is ``None`` for both, and
    ``by_key`` refuses a ``None``/empty argument loudly instead of silently
    returning nothing — the offered set neograph-ftnxl.7 renders must never
    contain an unaddressable record."""
    from neograph import ToolLedger  # local: the module does not exist yet (TDD red)

    ack = ToolInteraction(tool_name="transfer_to_peer", result="Successfully transferred to peer")
    legacy = ToolInteraction(tool_name="search", args={"q": "a"}, result="ok")
    addressable = ToolInteraction(tool_name="search", args={"q": "b"}, result="ok", ordinal=1)
    ledger = ToolLedger([ack, legacy, addressable])

    assert ack.ordinal == 0 and legacy.ordinal == 0, "0 is the default: not a budgeted invocation"
    assert ack.key is None and legacy.key is None, "an ordinal-0 record is NOT addressable"

    assert ledger.by_key("search#1") is addressable, "only the stamped record is addressable"
    assert ledger.by_key("search#0") is None, "an ordinal-0 record is never reachable by key"
    assert ledger.by_key("transfer_to_peer#0") is None

    with pytest.raises(ValueError):
        ledger.by_key(None)
    with pytest.raises(ValueError):
        ledger.by_key("")

    # The unaddressable records are still VISIBLE through the ordinary selectors.
    assert [t.args.get("q") for t in ledger.all("search")] == ["a", "b"]
