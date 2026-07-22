"""PORTAL T6 — mode (b) "dispatch": dynamic flow definition (neograph-f27xo).

TDD-RED behavioral tests for the dispatch wrapper + the SAME-``Construct(...)``
validation gate + the rejection path. NONE of the production wiring exists yet:
``make_portal_dispatch_fn``, the compiler-walk dispatch discriminator, the
``{node_field}_dispatch`` state field, the M1/M2 validator fixes, and the M3
``_dispatch`` producer registration are all unimplemented. So every test here
FAILS RED now (the assembly ones raise ``ConstructError`` "route field 'decide'
missing from the payload model"; the runtime ones cannot even build the
Construct) and turns GREEN once T6 (neograph-s3vr3.7) lands.

Core Invariant (design §3.5 / §4.2, neograph-f27xo): a node marked
``Portal(route="decide", ...)`` emits, as its OWN typed output, a spec dict
(``spec_field``) + an input dict (``input_field``). The framework then
``load_spec`` -> validates via the SAME ``Construct(...)`` gate as a hand-written
pipeline -> checks the built flow's output == ``Portal.output`` ->
``compile(scripted=, conditions=)`` with NO checkpointer -> invokes -> writes the
typed result to the regular (fingerprinted) ``{node_field}_dispatch`` state
field. An INVALID emitted spec raises a ``ConstructError`` WRAPPED in
``ExecutionError`` BEFORE any dispatched sub-node body executes.

Surfaces tested — the two LIVE mode-(b) surfaces only:
    * declarative  ``Node(name=..., mode="scripted", ...) | Portal(route="decide", ...)``
    * programmatic ``Node.scripted(...) | Portal(route="decide", ...)``
These are the SAME pipe construction, so one builder covers both. **@node is
EXEMPT for mode (b)** (design §2.2: "Mode (b) has no decorator sugar in v1") —
``_param_res`` / the DI-and-sugar machinery is decorator-only, and there is no
static dataflow for a runtime-emitted flow, so there is nothing for the ``@node``
surface (or ForwardConstruct) to build. This mirrors the mode-(a) ForwardConstruct
exemption note in ``test_portal_parity.py``.

Durability (design §7, D-mrb2y deferral): the dispatched inner graph is compiled
with NO checkpointer, so on a parent resume the WHOLE dispatch node re-executes.
This is DOCUMENTED-OPAQUE in v1 (Tier-2 durability belongs to neograph-mrb2y);
these tests do not exercise checkpoint resume of the inner flow, by design.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import (
    Construct,
    Node,
    Portal,
    compile,
    run,
)
from neograph.errors import ConfigurationError, ExecutionError
from neograph.runner import arun
from neograph.spec_types import register_type
from tests.fakes import build_test_compile_kwargs, register_scripted

# ═══════════════════════════════════════════════════════════════════════════
# PAYLOAD + FLOW-OUTPUT MODELS
# ═══════════════════════════════════════════════════════════════════════════


class DispatchDecision(BaseModel):
    """A dispatcher node's OWN output: the emitted spec dict + input dict.

    ``spec_field="spec"`` and ``input_field="dispatch_input"`` on the Portal
    name these two fields — the framework reads them off this payload after the
    dispatcher body runs, then loads/validates/compiles/invokes the spec.
    """

    spec: dict
    dispatch_input: dict


class Summary(BaseModel, frozen=True):
    """The dispatched-flow output type — the required ``Portal.output``."""

    text: str


class Other(BaseModel, frozen=True):
    """A DIFFERENT flow-output type — drives the output-contract mismatch case."""

    val: int


class Final(BaseModel, frozen=True):
    """A downstream consumer's output — proves M3 producer registration:
    a consumer reading ``inputs={"planner_dispatch": Summary}`` receives the
    dispatched result."""

    echo: str


# The emitted spec's ``outputs: "<name>"`` strings resolve through the SAME type
# registry the loader uses (spec_types.lookup_type). The autouse
# ``_clean_registries`` conftest fixture CLEARS ``_type_registry`` before every
# test, so a module-import-time ``register_type`` would be wiped before the first
# test runs (and load_spec would fail "type 'Summary' is not registered" at
# runtime). Register inside an autouse fixture so the types are present for each
# test's runtime dispatch — this mirrors how a real app registers types before
# ``run()`` (the documented mode-b authoring constraint, design §3.5).
@pytest.fixture(autouse=True)
def _register_dispatch_types():
    register_type("Summary", Summary)
    register_type("Other", Other)


# ═══════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS for the emitted flows (registered on the Portal, per
# D-DISPATCH-REGISTRIES: the dispatched flow may reference ONLY these).
# ═══════════════════════════════════════════════════════════════════════════

# Sentinel: the rejection-path flow's body MUST NEVER run (an invalid spec is
# rejected BEFORE any dispatched node executes). Stays empty on rejection.
DISPATCHED_BODY_RAN: list[str] = []


def _make_summary(input_data, config):
    """Trivial dispatched building block -> Summary (the happy flow's node)."""
    return Summary(text="dispatched")


def _make_other(input_data, config):
    """Dispatched building block -> Other (the output-contract-mismatch flow)."""
    return Other(val=7)


def _sentinel_body(input_data, config):
    """Dispatched building block for the INVALID flow — records that it ran so
    the rejection test can prove it did NOT (list stays empty)."""
    DISPATCHED_BODY_RAN.append("ran")
    return Summary(text="should-never-be-produced")


# ── Emitted spec dicts (neograph-flavored Agent Spec format, 0la8v) ─────────
#
# Per 0la8v's Core Invariant, mode (b)'s runtime dispatch format IS the SAME
# neograph-flavored Agent Spec to_agent_spec()/from_agent_spec() share -- so
# every emitted spec dict below is generated by ROUND-TRIPPING a real
# Construct through the canonical exporter, never hand-authored (mirrors
# ``_agent_spec_flavored_happy_spec`` in ``TestDispatchAcceptsAgentSpecFlavoredSpec``
# below, per the architect review's MEDIUM finding + the f0j1e.39 refine).


def _agent_spec_flavored(name: str, nodes: list) -> dict:
    """Build an emitted ``spec_field`` dict by exporting a real Construct
    through ``to_agent_spec()`` -- the SAME single format a real mode-(b)
    planner emits and ``from_agent_spec()`` consumes."""
    from neograph._agent_spec import to_agent_spec

    return to_agent_spec(Construct(name, nodes=nodes)).to_dict()


def _happy_spec() -> dict:
    return _agent_spec_flavored(
        "dispatched-flow", [Node.scripted("summarize", fn="_make_summary", outputs=Summary)]
    )


def _mismatch_spec() -> dict:
    """Structurally valid + compiles, but its flow output type is Other != Summary."""
    return _agent_spec_flavored("mismatch-flow", [Node.scripted("produce", fn="_make_other", outputs=Other)])


def _invalid_spec() -> dict:
    """A Flow from_agent_spec cannot import: an agent/act-mode node lowers
    (i3zsh.1) to a real AgentNode on EXPORT, but from_agent_spec's IMPORT side
    fails loud on AgentNode (agent/act import is a tracked follow-up,
    neograph-aa5gq) -- the natural Agent-Spec-flavored equivalent of "this
    spec cannot be reconstructed into a runnable Construct", exercising the
    SAME wrapped-ExecutionError rejection path a native-format validation
    error used to (a dangling upstream reference has no equivalent in this
    format: from_agent_spec derives inputs strictly from real DataFlowEdges,
    so it cannot produce a "references a nonexistent node" Construct)."""
    return _agent_spec_flavored(
        "invalid-flow",
        [
            Node(
                name="tail",
                mode="agent",
                model="fast",
                prompt="unused",
                outputs=Summary,
                tools=[],
            )
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════
# DISPATCHER-CONSTRUCT BUILDERS (declarative + programmatic are the same pipe)
# ═══════════════════════════════════════════════════════════════════════════


def _dispatch_scripted(fn_name: str, decision: DispatchDecision):
    """Register a dispatcher body that emits ``decision`` (spec + input)."""

    def _body(input_data, config):
        return decision

    register_scripted(fn_name, _body)


def _happy_construct(*, programmatic: bool) -> Construct:
    """A dispatcher 'planner' emitting the happy spec, plus a downstream
    consumer reading ``planner_dispatch`` (pins M3 producer registration)."""
    _dispatch_scripted("planner_happy", DispatchDecision(spec=_happy_spec(), dispatch_input={}))

    def _consume(input_data, config):
        summary = input_data["planner_dispatch"]
        return Final(echo=summary.text)

    register_scripted("consume_dispatch", _consume)

    km = Portal(
        route="decide",
        spec_field="spec",
        input_field="dispatch_input",
        output=Summary,
        scripted={"_make_summary": _make_summary},
    )
    if programmatic:
        planner = Node.scripted("planner", fn="planner_happy", outputs=DispatchDecision) | km
    else:
        planner = Node(name="planner", mode="scripted", scripted_fn="planner_happy", outputs=DispatchDecision) | km

    consumer = Node.scripted(
        "consumer", fn="consume_dispatch", inputs={"planner_dispatch": Summary}, outputs=Final
    )
    return Construct("dispatch-happy", nodes=[planner, consumer])


def _mismatch_construct() -> Construct:
    """A dispatcher whose emitted flow produces Other, but Portal.output=Summary."""
    _dispatch_scripted("planner_mismatch", DispatchDecision(spec=_mismatch_spec(), dispatch_input={}))
    km = Portal(
        route="decide",
        spec_field="spec",
        input_field="dispatch_input",
        output=Summary,  # != the flow's Other -> output-contract mismatch
        scripted={"_make_other": _make_other},
    )
    planner = Node.scripted("planner", fn="planner_mismatch", outputs=DispatchDecision) | km
    return Construct("dispatch-mismatch", nodes=[planner])


def _reject_construct() -> Construct:
    """A dispatcher emitting a spec ``from_agent_spec`` cannot import (an
    AgentNode -- see ``_invalid_spec()``)."""
    _dispatch_scripted("planner_reject", DispatchDecision(spec=_invalid_spec(), dispatch_input={}))
    km = Portal(
        route="decide",
        spec_field="spec",
        input_field="dispatch_input",
        output=Summary,
    )
    planner = Node.scripted("planner", fn="planner_reject", outputs=DispatchDecision) | km
    return Construct("dispatch-reject", nodes=[planner])


# ═══════════════════════════════════════════════════════════════════════════
# ASSEMBLY / COMPILE — M1 + M2: a lone dispatch node ASSEMBLES + COMPILES
# ═══════════════════════════════════════════════════════════════════════════


class TestDispatchNodeAssemblesAndCompiles:
    """M1 (_validation_portal mesh-collector must NOT reject a dispatch node)
    and M2 (_contiguous_portal_mesh must NOT absorb it). A lone
    ``route="decide"`` node must pass the SAME ``Construct(...)`` gate and
    ``compile()`` as any hand-written pipeline.

    RED NOW: the mesh-member collector predicate is route-blind, so a dispatch
    node is treated as a degenerate mesh member and ``Construct(...)`` raises
    "Portal route field 'decide' is missing from the payload model".
    """

    def test_lone_dispatch_node_assembles(self):
        km = Portal(
            route="decide",
            spec_field="spec",
            input_field="dispatch_input",
            output=Summary,
            scripted={"_make_summary": _make_summary},
        )
        _dispatch_scripted("planner_solo", DispatchDecision(spec=_happy_spec(), dispatch_input={}))
        planner = Node.scripted("planner", fn="planner_solo", outputs=DispatchDecision) | km
        c = Construct("dispatch-solo", nodes=[planner])  # must NOT raise (M1)
        assert {n.name for n in c.nodes} == {"planner"}

    def test_lone_dispatch_node_compiles(self):
        km = Portal(
            route="decide",
            spec_field="spec",
            input_field="dispatch_input",
            output=Summary,
            scripted={"_make_summary": _make_summary},
        )
        _dispatch_scripted("planner_solo2", DispatchDecision(spec=_happy_spec(), dispatch_input={}))
        planner = Node.scripted("planner", fn="planner_solo2", outputs=DispatchDecision) | km
        c = Construct("dispatch-solo2", nodes=[planner])
        graph = compile(c, **build_test_compile_kwargs())  # must NOT raise (M1/M2)
        # M2: the dispatch node is lowered as a real linear graph node (not absorbed
        # into a mesh and dropped), so it appears in the compiled StateGraph.
        assert "planner" in graph.get_graph().nodes


# ═══════════════════════════════════════════════════════════════════════════
# HAPPY PATH — dispatched flow result lands typed on {planner}_dispatch
# ═══════════════════════════════════════════════════════════════════════════


class TestDispatchHappyPath:
    """A valid emitted spec is loaded -> validated -> compiled -> invoked; the
    typed Summary lands on ``planner_dispatch`` and a downstream consumer
    reading ``inputs={"planner_dispatch": Summary}`` receives it (M3)."""

    @pytest.mark.parametrize("programmatic", [False, True], ids=["declarative", "programmatic"])
    def test_dispatch_result_on_dispatch_field_sync(self, programmatic):
        c = _happy_construct(programmatic=programmatic)
        graph = compile(c, **build_test_compile_kwargs())
        result = run(graph, input={})
        assert isinstance(result["planner_dispatch"], Summary)
        assert result["planner_dispatch"].text == "dispatched"

    @pytest.mark.parametrize("programmatic", [False, True], ids=["declarative", "programmatic"])
    async def test_dispatch_result_on_dispatch_field_async(self, programmatic):
        c = _happy_construct(programmatic=programmatic)
        graph = compile(c, **build_test_compile_kwargs())
        result = await arun(graph, input={})
        assert isinstance(result["planner_dispatch"], Summary)
        assert result["planner_dispatch"].text == "dispatched"

    def test_downstream_consumer_receives_dispatch_result_sync(self):
        """M3: a consumer typed on ``planner_dispatch`` type-checks at assembly
        AND receives the dispatched Summary at runtime."""
        c = _happy_construct(programmatic=True)
        graph = compile(c, **build_test_compile_kwargs())
        result = run(graph, input={})
        assert isinstance(result["consumer"], Final)
        assert result["consumer"].echo == "dispatched"

    async def test_downstream_consumer_receives_dispatch_result_async(self):
        c = _happy_construct(programmatic=True)
        graph = compile(c, **build_test_compile_kwargs())
        result = await arun(graph, input={})
        assert isinstance(result["consumer"], Final)
        assert result["consumer"].echo == "dispatched"


# ═══════════════════════════════════════════════════════════════════════════
# REJECTION PATH — invalid spec -> wrapped ConstructError BEFORE any sub-node
# ═══════════════════════════════════════════════════════════════════════════


class TestDispatchRejectionPath:
    """An invalid emitted spec is rejected by the SAME ``from_agent_spec()`` +
    ``Construct(...)`` gate, surfaced as an ``ExecutionError`` WRAPPING the
    underlying error and naming the spec — raised BEFORE any dispatched
    sub-node body runs.

    0la8v NOTE: the native-format "references an unknown upstream" scenario
    (the pre-0la8v ``ghost_result`` case) has no equivalent in the
    Agent-Spec-flavored format -- ``from_agent_spec`` derives a node's
    ``inputs`` strictly from real ``DataFlowEdge``s, so it structurally
    cannot produce a Construct referencing a nonexistent upstream. The
    natural equivalent invalidity in this format is an AgentNode
    (agent/act mode is export-able since neograph-i3zsh.1, but import-side
    reconstruction is a tracked follow-up, neograph-aa5gq) -- see
    ``_invalid_spec()``."""

    def test_invalid_spec_raises_wrapped_execution_error_sync(self):
        DISPATCHED_BODY_RAN.clear()
        c = _reject_construct()
        graph = compile(c, **build_test_compile_kwargs())
        with pytest.raises(ExecutionError) as exc:
            run(graph, input={})
        msg = str(exc.value)
        # Wraps the underlying ConfigurationError message (AgentNode not supported) ...
        assert "AgentNode" in msg
        # ... and names the offending spec.
        assert "invalid-flow" in msg
        # Rejected BEFORE any dispatched node body executed.
        assert DISPATCHED_BODY_RAN == []

    async def test_invalid_spec_raises_wrapped_execution_error_async(self):
        DISPATCHED_BODY_RAN.clear()
        c = _reject_construct()
        graph = compile(c, **build_test_compile_kwargs())
        with pytest.raises(ExecutionError) as exc:
            await arun(graph, input={})
        msg = str(exc.value)
        assert "AgentNode" in msg
        assert "invalid-flow" in msg
        assert DISPATCHED_BODY_RAN == []

    def test_underlying_error_is_construct_error(self):
        """The wrapped cause is the real ``ConfigurationError`` from the shared
        gate (anti-band-aid: no bespoke validator, no schema subset)."""
        c = _reject_construct()
        graph = compile(c, **build_test_compile_kwargs())
        with pytest.raises(ExecutionError) as exc:
            run(graph, input={})
        chain = []
        cur = exc.value
        while cur is not None:
            chain.append(cur)
            cur = cur.__cause__
        assert any(isinstance(e, ConfigurationError) for e in chain), chain


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT-CONTRACT MISMATCH — flow compiles, but its output != Portal.output
# ═══════════════════════════════════════════════════════════════════════════


class TestDispatchOutputContractMismatch:
    """The emitted spec is structurally valid and compiles, but the dispatched
    flow's output type (Other) != ``Portal.output`` (Summary). This raises an
    ``ExecutionError`` (the output-contract check)."""

    def test_output_contract_mismatch_raises_sync(self):
        c = _mismatch_construct()
        graph = compile(c, **build_test_compile_kwargs())
        with pytest.raises(ExecutionError):
            run(graph, input={})

    async def test_output_contract_mismatch_raises_async(self):
        c = _mismatch_construct()
        graph = compile(c, **build_test_compile_kwargs())
        with pytest.raises(ExecutionError):
            await arun(graph, input={})


# ═══════════════════════════════════════════════════════════════════════════
# 0la8v — mode (b)'s runtime dispatch format IS the neograph-flavored Agent
# Spec (the SAME single format to_agent_spec() emits / from_agent_spec()
# consumes), not a bespoke native-Spec dict.
# ═══════════════════════════════════════════════════════════════════════════


def _agent_spec_flavored_happy_spec() -> dict:
    """Build the emitted ``spec_field`` dict by ROUND-TRIPPING a real,
    minimal single-node scripted ``Construct`` through the canonical
    exporter -- ``to_agent_spec() -> Flow -> flow.to_dict()`` -- never by
    hand-authoring an Agent-Spec-envelope dict literal.

    Per the architect review's MEDIUM finding + the f0j1e.39 refine notes:
    a hand-authored fixture risks silently drifting from what
    ``to_agent_spec()`` actually emits, which would falsify 0la8v's own
    invariant ("the SAME single machine format"). Generating it via the
    real exporter keeps the fixture honest to the export contract.
    """
    from neograph._agent_spec import to_agent_spec

    source = Construct(
        "dispatched-agent-spec-flow",
        nodes=[Node.scripted("summarize", fn="_make_summary", outputs=Summary)],
    )
    flow = to_agent_spec(source)
    return flow.to_dict()


class TestDispatchAcceptsAgentSpecFlavoredSpec:
    """0la8v Core Invariant: Portal mode (b) has exactly ONE modifier-aware
    runtime spec-loading path -- whatever ``from_agent_spec()`` (01i0g)
    accepts is the SAME single format both ``to_agent_spec()`` (export) and
    a mode-(b) planner (dispatch) use. There is never a second, parallel,
    bespoke dict-dispatch serializer living only inside ``factory.py``.

    A planner's emitted ``spec_field`` dict is now an Agent-Spec-flavored
    dict (built by round-tripping a real ``Construct`` through
    ``to_agent_spec()``, per ``_agent_spec_flavored_happy_spec`` above) --
    NOT a native neograph ``Spec`` dict (``{"nodes": [...], "pipeline":
    {...}}``) like ``_HAPPY_SPEC`` above. Dispatch must deserialize it via
    ``from_agent_spec()`` (which needs a live ``pyagentspec.flows.flow.Flow``
    object -- ``AgentSpecDeserializer().from_dict(spec_dict)``, NOT the
    dict itself, per neograph-0la8v's confirmed signature note) + compile,
    the SAME way it does for a native-format spec today.

    TDD RED (2026-07-22): ``make_portal_dispatch_fn``'s ``_prepare``
    (``factory.py`` ~line 456) still hard-codes ``sub = load_spec(spec_dict)``
    -- the native-Spec-only loader. Fed an Agent-Spec-flavored dict (whose
    top-level shape is `{"component_type": "Flow", "start_node": ...,
    "$referenced_components": {...}, ...}`, nothing like native `Spec`'s
    `{"nodes": [...], "pipeline": {...}}`), ``load_spec``'s Pydantic gate
    rejects EVERY field (missing `nodes[i].name`/`nodes[i].outputs`,
    `pipeline` missing, `component_type`/`$referenced_components`/etc. all
    `extra_forbidden`) and ``_prepare`` re-raises it wrapped as
    ``ExecutionError`` before the dispatched flow ever runs. This test pins
    the CORRECT future behavior (dispatch succeeds, typed ``Summary`` lands
    on ``{field}_dispatch``) and therefore FAILS today with
    ``ExecutionError`` until 0la8v re-points ``_prepare`` from ``load_spec``
    onto ``from_agent_spec`` (Implementation Plan step 2).

    Gated on ``pyagentspec`` -- ``uv run --extra dev --extra agent-spec
    pytest tests/test_portal_dispatch.py``.
    """

    @pytest.mark.parametrize("programmatic", [False, True], ids=["declarative", "programmatic"])
    def test_dispatch_result_on_dispatch_field_sync(self, programmatic):
        pytest.importorskip("pyagentspec")
        spec_dict = _agent_spec_flavored_happy_spec()
        _dispatch_scripted(
            f"planner_agent_spec_happy_sync_{programmatic}",
            DispatchDecision(spec=spec_dict, dispatch_input={}),
        )
        km = Portal(
            route="decide",
            spec_field="spec",
            input_field="dispatch_input",
            output=Summary,
            scripted={"_make_summary": _make_summary},
        )
        if programmatic:
            planner = (
                Node.scripted("planner", fn=f"planner_agent_spec_happy_sync_{programmatic}", outputs=DispatchDecision)
                | km
            )
        else:
            planner = (
                Node(
                    name="planner",
                    mode="scripted",
                    scripted_fn=f"planner_agent_spec_happy_sync_{programmatic}",
                    outputs=DispatchDecision,
                )
                | km
            )
        c = Construct("dispatch-agent-spec-happy-sync", nodes=[planner])
        graph = compile(c, **build_test_compile_kwargs())

        result = run(graph, input={})

        assert isinstance(result["planner_dispatch"], Summary), (
            "Portal dispatch must deserialize an Agent-Spec-flavored spec_field "
            "dict via from_agent_spec() (the SAME single format to_agent_spec() "
            "emits), not reject it via the native-Spec-only load_spec() gate"
        )
        assert result["planner_dispatch"].text == "dispatched"

    async def test_dispatch_result_on_dispatch_field_async(self):
        pytest.importorskip("pyagentspec")
        spec_dict = _agent_spec_flavored_happy_spec()
        _dispatch_scripted(
            "planner_agent_spec_happy_async", DispatchDecision(spec=spec_dict, dispatch_input={})
        )
        km = Portal(
            route="decide",
            spec_field="spec",
            input_field="dispatch_input",
            output=Summary,
            scripted={"_make_summary": _make_summary},
        )
        planner = Node.scripted("planner", fn="planner_agent_spec_happy_async", outputs=DispatchDecision) | km
        c = Construct("dispatch-agent-spec-happy-async", nodes=[planner])
        graph = compile(c, **build_test_compile_kwargs())

        result = await arun(graph, input={})

        assert isinstance(result["planner_dispatch"], Summary)
        assert result["planner_dispatch"].text == "dispatched"
