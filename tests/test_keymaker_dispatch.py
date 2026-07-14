"""KEYMAKER T6 — mode (b) "dispatch": dynamic flow definition (neograph-f27xo).

TDD-RED behavioral tests for the dispatch wrapper + the SAME-``Construct(...)``
validation gate + the rejection path. NONE of the production wiring exists yet:
``make_keymaker_dispatch_fn``, the compiler-walk dispatch discriminator, the
``{node_field}_dispatch`` state field, the M1/M2 validator fixes, and the M3
``_dispatch`` producer registration are all unimplemented. So every test here
FAILS RED now (the assembly ones raise ``ConstructError`` "route field 'decide'
missing from the payload model"; the runtime ones cannot even build the
Construct) and turns GREEN once T6 (neograph-s3vr3.7) lands.

Core Invariant (design §3.5 / §4.2, neograph-f27xo): a node marked
``Keymaker(route="decide", ...)`` emits, as its OWN typed output, a spec dict
(``spec_field``) + an input dict (``input_field``). The framework then
``load_spec`` -> validates via the SAME ``Construct(...)`` gate as a hand-written
pipeline -> checks the built flow's output == ``Keymaker.output`` ->
``compile(scripted=, conditions=)`` with NO checkpointer -> invokes -> writes the
typed result to the regular (fingerprinted) ``{node_field}_dispatch`` state
field. An INVALID emitted spec raises a ``ConstructError`` WRAPPED in
``ExecutionError`` BEFORE any dispatched sub-node body executes.

Surfaces tested — the two LIVE mode-(b) surfaces only:
    * declarative  ``Node(name=..., mode="scripted", ...) | Keymaker(route="decide", ...)``
    * programmatic ``Node.scripted(...) | Keymaker(route="decide", ...)``
These are the SAME pipe construction, so one builder covers both. **@node is
EXEMPT for mode (b)** (design §2.2: "Mode (b) has no decorator sugar in v1") —
``_param_res`` / the DI-and-sugar machinery is decorator-only, and there is no
static dataflow for a runtime-emitted flow, so there is nothing for the ``@node``
surface (or ForwardConstruct) to build. This mirrors the mode-(a) ForwardConstruct
exemption note in ``test_keymaker_parity.py``.

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
    Keymaker,
    Node,
    compile,
    run,
)
from neograph.errors import ConstructError, ExecutionError
from neograph.runner import arun
from neograph.spec_types import register_type
from tests.fakes import build_test_compile_kwargs, register_scripted

# ═══════════════════════════════════════════════════════════════════════════
# PAYLOAD + FLOW-OUTPUT MODELS
# ═══════════════════════════════════════════════════════════════════════════


class DispatchDecision(BaseModel):
    """A dispatcher node's OWN output: the emitted spec dict + input dict.

    ``spec_field="spec"`` and ``input_field="dispatch_input"`` on the Keymaker
    name these two fields — the framework reads them off this payload after the
    dispatcher body runs, then loads/validates/compiles/invokes the spec.
    """

    spec: dict
    dispatch_input: dict


class Summary(BaseModel, frozen=True):
    """The dispatched-flow output type — the required ``Keymaker.output``."""

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
# BUILDING BLOCKS for the emitted flows (registered on the Keymaker, per
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


# ── Emitted spec dicts (version-1 load_spec format) ─────────────────────────

_HAPPY_SPEC: dict = {
    "name": "dispatched-flow",
    "nodes": [
        {"name": "summarize", "mode": "scripted", "scripted_fn": "_make_summary", "outputs": "Summary"},
    ],
    "pipeline": {"nodes": ["summarize"]},
}

# Structurally valid + compiles, but its flow output type is Other != Summary.
_MISMATCH_SPEC: dict = {
    "name": "mismatch-flow",
    "nodes": [
        {"name": "produce", "mode": "scripted", "scripted_fn": "_make_other", "outputs": "Other"},
    ],
    "pipeline": {"nodes": ["produce"]},
}

# Structurally-parseable but INVALID node chain: 'tail' references an upstream
# named 'ghost_result' that no node produces -> _validate_node_chain raises
# ConstructError inside load_spec (verified empirically). This is the SAME gate
# hand-written pipelines pass through — no bespoke validator.
_INVALID_SPEC: dict = {
    "name": "invalid-flow",
    "nodes": [
        {"name": "start", "mode": "scripted", "scripted_fn": "_sentinel_body", "outputs": "Summary"},
        {
            "name": "tail",
            "mode": "scripted",
            "scripted_fn": "_sentinel_body",
            "inputs": {"ghost_result": "Summary"},
            "outputs": "Summary",
        },
    ],
    "pipeline": {"nodes": ["start", "tail"]},
}


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
    _dispatch_scripted("planner_happy", DispatchDecision(spec=_HAPPY_SPEC, dispatch_input={}))

    def _consume(input_data, config):
        summary = input_data["planner_dispatch"]
        return Final(echo=summary.text)

    register_scripted("consume_dispatch", _consume)

    km = Keymaker(
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
    """A dispatcher whose emitted flow produces Other, but Keymaker.output=Summary."""
    _dispatch_scripted("planner_mismatch", DispatchDecision(spec=_MISMATCH_SPEC, dispatch_input={}))
    km = Keymaker(
        route="decide",
        spec_field="spec",
        input_field="dispatch_input",
        output=Summary,  # != the flow's Other -> output-contract mismatch
        scripted={"_make_other": _make_other},
    )
    planner = Node.scripted("planner", fn="planner_mismatch", outputs=DispatchDecision) | km
    return Construct("dispatch-mismatch", nodes=[planner])


def _reject_construct() -> Construct:
    """A dispatcher emitting a structurally-invalid spec (unknown upstream)."""
    _dispatch_scripted("planner_reject", DispatchDecision(spec=_INVALID_SPEC, dispatch_input={}))
    km = Keymaker(
        route="decide",
        spec_field="spec",
        input_field="dispatch_input",
        output=Summary,
        scripted={"_sentinel_body": _sentinel_body},
    )
    planner = Node.scripted("planner", fn="planner_reject", outputs=DispatchDecision) | km
    return Construct("dispatch-reject", nodes=[planner])


# ═══════════════════════════════════════════════════════════════════════════
# ASSEMBLY / COMPILE — M1 + M2: a lone dispatch node ASSEMBLES + COMPILES
# ═══════════════════════════════════════════════════════════════════════════


class TestDispatchNodeAssemblesAndCompiles:
    """M1 (_validation_keymaker mesh-collector must NOT reject a dispatch node)
    and M2 (_contiguous_keymaker_mesh must NOT absorb it). A lone
    ``route="decide"`` node must pass the SAME ``Construct(...)`` gate and
    ``compile()`` as any hand-written pipeline.

    RED NOW: the mesh-member collector predicate is route-blind, so a dispatch
    node is treated as a degenerate mesh member and ``Construct(...)`` raises
    "Keymaker route field 'decide' is missing from the payload model".
    """

    def test_lone_dispatch_node_assembles(self):
        km = Keymaker(
            route="decide",
            spec_field="spec",
            input_field="dispatch_input",
            output=Summary,
            scripted={"_make_summary": _make_summary},
        )
        _dispatch_scripted("planner_solo", DispatchDecision(spec=_HAPPY_SPEC, dispatch_input={}))
        planner = Node.scripted("planner", fn="planner_solo", outputs=DispatchDecision) | km
        c = Construct("dispatch-solo", nodes=[planner])  # must NOT raise (M1)
        assert {n.name for n in c.nodes} == {"planner"}

    def test_lone_dispatch_node_compiles(self):
        km = Keymaker(
            route="decide",
            spec_field="spec",
            input_field="dispatch_input",
            output=Summary,
            scripted={"_make_summary": _make_summary},
        )
        _dispatch_scripted("planner_solo2", DispatchDecision(spec=_HAPPY_SPEC, dispatch_input={}))
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
    """An invalid emitted spec is rejected by the SAME ``Construct(...)`` gate,
    surfaced as an ``ExecutionError`` WRAPPING the underlying ``ConstructError``
    and naming the spec — raised BEFORE any dispatched sub-node body runs."""

    def test_invalid_spec_raises_wrapped_execution_error_sync(self):
        DISPATCHED_BODY_RAN.clear()
        c = _reject_construct()
        graph = compile(c, **build_test_compile_kwargs())
        with pytest.raises(ExecutionError) as exc:
            run(graph, input={})
        msg = str(exc.value)
        # Wraps the underlying ConstructError message (unknown upstream) ...
        assert "ghost_result" in msg
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
        assert "ghost_result" in msg
        assert "invalid-flow" in msg
        assert DISPATCHED_BODY_RAN == []

    def test_underlying_error_is_construct_error(self):
        """The wrapped cause is the real ``ConstructError`` from the shared gate
        (anti-band-aid: no bespoke validator, no schema subset)."""
        c = _reject_construct()
        graph = compile(c, **build_test_compile_kwargs())
        with pytest.raises(ExecutionError) as exc:
            run(graph, input={})
        chain = []
        cur = exc.value
        while cur is not None:
            chain.append(cur)
            cur = cur.__cause__
        assert any(isinstance(e, ConstructError) for e in chain), chain


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT-CONTRACT MISMATCH — flow compiles, but its output != Keymaker.output
# ═══════════════════════════════════════════════════════════════════════════


class TestDispatchOutputContractMismatch:
    """The emitted spec is structurally valid and compiles, but the dispatched
    flow's output type (Other) != ``Keymaker.output`` (Summary). This raises an
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
