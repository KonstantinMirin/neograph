"""PORTAL mode (b) — ``max_depth`` self-extending-flow budget (neograph-2aw5c).

TDD-RED regression tests for the atom neograph-cbtjk.22 / bead neograph-2aw5c
(Refined Plan). NONE of this is implemented yet: ``Portal`` has no ``max_depth``
field, and ``make_portal_dispatch_fn`` (factory.py) never reads or forwards a
depth budget. So every test in this file is expected to FAIL RED now and turn
GREEN once neograph-cbtjk.23 lands.

Core Invariant (2aw5c Refined Plan): ``max_depth`` is a LINEAGE property across
FRESH per-level compiled sub-flows (mode-b dispatch calls ``compile_construct``
+ ``compiled.invoke`` on a brand-new sub-flow each level, with fresh initial
state), so the budget can be carried ONLY on ``config['configurable']`` (a
state-bus counter would silently reset to 0 at every nesting level). The
resolved design (post-review):

  * ``Portal.max_depth`` is REQUIRED (no numeric default) in dispatch mode
    (``route="decide"``) and FORBIDDEN in peer mode (``to=[...]``) — the
    mirror image of the existing ``max_hops``/``on_exhaust`` guard
    (modifiers.py:671-690).
  * The depth is threaded through a single FLAT config-only key (this file
    assumes ``StateKeys.PORTAL_DISPATCH_DEPTH``, mirroring the
    ``StateKeys.DI_INPUTS`` / ``RESOURCE_MANIFEST_INJECT`` /
    ``ORACLE_MODEL_OVERRIDE`` pattern, _state_keys.py:70-93) — NEVER a
    per-field key like ``handoff_hops`` (that would reset every nesting level).
  * The check happens in ``dispatch_wrapper``/``adispatch_wrapper`` at the very
    TOP, BEFORE ``inner.invoke(state, config)`` runs — so an over-budget
    dispatch fails before the dispatcher's OWN body (a planner LLM call, in the
    general case) even executes, and before the emitted spec is
    deserialized/validated/compiled.
  * The child config handed to the nested ``compiled.invoke``/``ainvoke`` is a
    NEW dict (shallow-copy config + configurable, mirroring
    ``runner.py:153-197``'s copy-not-mutate pattern) with the depth
    incremented by exactly one — so sibling dispatch nodes sharing the SAME
    incoming config must NOT observe each other's increment.

Architectural note discovered while authoring this test: ``from_agent_spec``
(loader.py) has NO reconstruction path for a Portal dispatch (``route=
"decide"``) node inside an IMPORTED Flow (only a top-level ``Swarm`` -> Portal
PEER mesh is supported, ``_reconstruct_swarm_mesh``) and ``to_agent_spec``
explicitly rejects exporting a PORTAL-modified node ("no Agent Spec lowering
yet"). So a genuinely self-referential emitted spec (a dispatched flow whose
OWN node re-emits another dispatch) cannot be authored through the documented
spec_field channel today — that gap is orthogonal to this budget feature and
out of scope here. This file instead pins the underlying mechanism directly and
observably through the ONE real dispatch level the current architecture
supports: (a) the pre-body budget check, using an already-elevated incoming
depth to simulate "this invocation is itself a nested level", and (b) the
child-config increment, by inspecting what config the dispatched sub-flow's
own node actually receives. Both are exactly the moving parts the Core
Invariant describes, exercised through the real ``dispatch_wrapper`` runtime
path (no mocking of factory/dispatch internals).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import Construct, Node, Portal, compile, run
from neograph._state_keys import StateKeys
from neograph.errors import ConfigurationError, ExecutionError
from neograph.spec_types import register_type
from tests.fakes import build_test_compile_kwargs, register_scripted

# The 2aw5c Refined Plan pins this exact key name/shape (a single FLAT
# config-only key, StateKeys.PORTAL_DISPATCH_DEPTH). It does not exist yet, so
# reference it via getattr with the documented literal as a fallback -- this
# keeps every test's FAILURE at the intended behavioral assertion (raised /
# not raised, expected depth value) rather than a spurious AttributeError,
# while still pinning the exact name the implementation must add.
DEPTH_KEY = getattr(StateKeys, "PORTAL_DISPATCH_DEPTH", "_neo_portal_dispatch_depth")


class DispatchDecision(BaseModel):
    """A dispatcher node's own output: the emitted spec dict + input dict."""

    spec: dict
    dispatch_input: dict


class Summary(BaseModel, frozen=True):
    """The dispatched-flow output type -- the required ``Portal.output``."""

    text: str


@pytest.fixture(autouse=True)
def _register_max_depth_types():
    register_type("Summary", Summary)


def _make_summary(input_data, config):
    """Dispatched building block -> Summary."""
    return Summary(text="dispatched")


def _agent_spec_flavored_happy_spec() -> dict:
    """A trivially-valid one-node emitted spec, built by round-tripping a real
    Construct through ``to_agent_spec()`` (0la8v's Core Invariant) -- never a
    hand-authored dict literal."""
    from neograph._agent_spec import to_agent_spec

    return to_agent_spec(
        Construct("dispatched-flow", nodes=[Node.scripted("summarize", fn="_make_summary", outputs=Summary)])
    ).to_dict()


# ═══════════════════════════════════════════════════════════════════════════
# ASSEMBLY-TIME: max_depth is REQUIRED in dispatch mode, FORBIDDEN in peer mode
# ═══════════════════════════════════════════════════════════════════════════


class TestPortalMaxDepthAssemblyContract:
    """Mirrors the existing dispatch-mode 'requires spec_field/input_field/
    output' guard (modifiers.py:677-683) and its mirror-image peer-mode
    'forbids max_hops/on_exhaust' guard (modifiers.py:684-690)."""

    def test_dispatch_mode_requires_max_depth(self):
        """A route='decide' Portal with no max_depth fails loud at
        construction -- there is no numeric default to silently fall back to
        (2aw5c Refined Plan, Resolution to MEDIUM 2)."""
        with pytest.raises(ConfigurationError) as exc_info:
            Portal(
                route="decide",
                spec_field="spec",
                input_field="dispatch_input",
                output=Summary,
                scripted={"_make_summary": _make_summary},
                # max_depth intentionally omitted
            )
        assert "max_depth" in str(exc_info.value)

    def test_peer_mode_forbids_max_depth(self):
        """max_depth is a dispatch-mode-only knob -- setting it alongside
        peer-mode `to=[...]` is the mirror image of max_hops/on_exhaust being
        forbidden in dispatch mode, and must fail loud the same way."""
        with pytest.raises(ConfigurationError) as exc_info:
            Portal(to=["billing", "technical"], max_depth=3)
        assert "max_depth" in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════════
# RUNTIME: the config-only depth budget is checked BEFORE the dispatcher body
# runs, and forwarded incremented-by-one into the nested compiled.invoke
# ═══════════════════════════════════════════════════════════════════════════


def _dispatch_construct(*, planner_ran: list[str], max_depth: int, summary_fn=_make_summary) -> Construct:
    """A single dispatch node ('planner') emitting the happy one-node spec,
    with a sentinel recording whether the planner's OWN body executed.

    ``summary_fn`` goes into ``Portal.scripted`` directly (never a later
    ``register_scripted`` call) -- ``compile()``'s merge order applies
    explicit ``scripted=`` kwargs LAST (compiler.py), so ``Portal.scripted``
    (captured at construction time and passed as the dispatched sub-flow's
    own ``scripted=`` at compile time) always wins over anything registered
    into the global decoration-time registry afterward.
    """

    def _planner_body(input_data, config):
        planner_ran.append("ran")
        return DispatchDecision(spec=_agent_spec_flavored_happy_spec(), dispatch_input={})

    register_scripted("planner_max_depth_body", _planner_body)

    km = Portal(
        route="decide",
        spec_field="spec",
        input_field="dispatch_input",
        output=Summary,
        max_depth=max_depth,
        scripted={"_make_summary": summary_fn},
    )
    planner = Node.scripted("planner", fn="planner_max_depth_body", outputs=DispatchDecision) | km
    return Construct("dispatch-max-depth", nodes=[planner])


class TestPortalMaxDepthRuntimeBudget:
    def test_dispatch_raises_before_planner_body_when_incoming_depth_at_budget(self):
        """The check runs BEFORE inner.invoke (Resolution to MEDIUM 3): when
        the INCOMING config already carries a depth equal to max_depth (as if
        this invocation were itself an already-at-budget nested level), the
        dispatcher must raise ExecutionError WITHOUT ever running the
        planner's own body -- proven by the sentinel list staying empty."""
        planner_ran: list[str] = []
        c = _dispatch_construct(planner_ran=planner_ran, max_depth=2)
        graph = compile(c, **build_test_compile_kwargs())

        with pytest.raises(ExecutionError) as exc_info:
            run(graph, input={}, config={"configurable": {DEPTH_KEY: 2}})

        assert "planner" in str(exc_info.value) or "depth" in str(exc_info.value).lower()
        assert planner_ran == [], (
            "the dispatcher's own body ran even though the incoming depth "
            "already met max_depth -- the budget check must fire BEFORE "
            "inner.invoke, not after"
        )

    def test_dispatch_succeeds_within_budget_and_increments_child_depth(self):
        """When under budget, the dispatcher's body DOES run, and the child
        config handed to the dispatched sub-flow's own node carries the depth
        incremented by exactly one (never the unmodified incoming config)."""
        planner_ran: list[str] = []
        captured_configs: list[dict] = []

        def _make_summary_capturing(input_data, config):
            captured_configs.append(dict(config.get("configurable", {})))
            return Summary(text="dispatched")

        c = _dispatch_construct(planner_ran=planner_ran, max_depth=3, summary_fn=_make_summary_capturing)
        graph = compile(c, **build_test_compile_kwargs())

        result = run(graph, input={}, config={"configurable": {DEPTH_KEY: 1}})

        assert planner_ran == ["ran"]
        assert isinstance(result["planner_dispatch"], Summary)
        assert len(captured_configs) == 1, "the dispatched sub-flow's own node must have run exactly once"
        assert captured_configs[0].get(DEPTH_KEY) == 2, (
            "the nested compiled.invoke must receive a NEW config with the "
            "depth incremented by one over the incoming depth (1 -> 2), not "
            "the unmodified incoming config"
        )

    def test_sibling_dispatches_do_not_double_increment_each_others_depth(self):
        """Two independent (sibling, non-nested) dispatch nodes sharing the
        SAME incoming config must each independently see the SAME starting
        depth (0) incremented to exactly 1 for their own child invoke -- i.e.
        the child config must be built via copy, never by mutating the shared
        incoming config in place (2aw5c Refined Plan risk: 'sibling dispatches
        at the same depth must not double-increment each other's counter').
        If one sibling's wrapper mutated the shared config in place, the OTHER
        sibling (running after it) would incorrectly observe depth 2."""
        ran_a: list[str] = []
        ran_b: list[str] = []
        captured_a: list[dict] = []
        captured_b: list[dict] = []

        def _planner_a(input_data, config):
            ran_a.append("ran")
            return DispatchDecision(spec=_agent_spec_flavored_happy_spec(), dispatch_input={})

        def _planner_b(input_data, config):
            ran_b.append("ran")
            return DispatchDecision(spec=_agent_spec_flavored_happy_spec(), dispatch_input={})

        def _make_summary_a(input_data, config):
            captured_a.append(dict(config.get("configurable", {})))
            return Summary(text="dispatched-a")

        def _make_summary_b(input_data, config):
            captured_b.append(dict(config.get("configurable", {})))
            return Summary(text="dispatched-b")

        register_scripted("planner_a_body", _planner_a)
        register_scripted("planner_b_body", _planner_b)

        km_a = Portal(
            route="decide",
            spec_field="spec",
            input_field="dispatch_input",
            output=Summary,
            max_depth=1,  # exactly one level of dispatch allowed from depth 0
            scripted={"_make_summary": _make_summary_a},
        )
        km_b = Portal(
            route="decide",
            spec_field="spec",
            input_field="dispatch_input",
            output=Summary,
            max_depth=1,
            scripted={"_make_summary": _make_summary_b},
        )
        planner_a = Node.scripted("planner_a", fn="planner_a_body", outputs=DispatchDecision) | km_a
        planner_b = Node.scripted("planner_b", fn="planner_b_body", outputs=DispatchDecision) | km_b
        c = Construct("dispatch-siblings", nodes=[planner_a, planner_b])
        graph = compile(c, **build_test_compile_kwargs())

        # No pre-existing depth on the incoming config -- both siblings start at 0.
        result = run(graph, input={})

        assert ran_a == ["ran"], "sibling A's dispatcher body did not run"
        assert ran_b == ["ran"], "sibling B's dispatcher body did not run"
        assert isinstance(result["planner_a_dispatch"], Summary)
        assert isinstance(result["planner_b_dispatch"], Summary)
        assert len(captured_a) == 1 and len(captured_b) == 1
        assert captured_a[0].get(DEPTH_KEY) == 1, "sibling A must see depth 0 -> 1, independent of sibling B"
        assert captured_b[0].get(DEPTH_KEY) == 1, (
            "sibling B must see depth 0 -> 1, NOT 2 -- a shared-config mutation "
            "bug would make B observe A's increment"
        )
