"""Example 29: Keymaker dynamic flow dispatch -- a pipeline chosen at RUNTIME.

`Keymaker` has two modes. Example 28 covers mode (a), peer routing: a fixed set
of members hands off to declared peers. This example covers mode (b), *dynamic
flow dispatch* (`Keymaker(route="decide", ...)`): a "planner" node decides at
RUNTIME which sub-flow to run, emits a neograph *spec* (a plain dict) describing
that flow, and the framework loads -> validates -> compiles -> runs it, then
hands the typed result back to the outer pipeline.

The safety story is the whole point. A machine-emitted spec is not trusted blind:
it passes the SAME `Construct(...)` validation gate a hand-written pipeline does.
So mode (b) keeps neograph's promise even for a flow that did not exist until
runtime -- "if it dispatches, it was validated". And the emitted flow may wire
ONLY the building blocks you pre-registered on the `Keymaker` (`scripted=`): a
machine composes pre-approved code, it cannot inject new code.

Two self-contained demos:
  1. Dynamic dispatch  -- one planner emits DIFFERENT topologies (a one-node
                          brief flow vs a two-node thorough flow) depending on
                          the runtime request; each is compiled + run and its
                          typed result lands on the `{planner}_dispatch` channel.
  2. Rejection path    -- the planner emits a STRUCTURALLY INVALID spec (a node
                          consuming an upstream nothing produces). The shared
                          gate rejects it: a `ConstructError` surfaces WRAPPED in
                          an `ExecutionError` naming the spec, and it fires BEFORE
                          any dispatched sub-node body runs (a module-level
                          sentinel proves the body never executed).

Everything is scripted (keyless) so the example is deterministic and needs no
network -- but a real planner would be `mode='think'` emitting the spec as its
structured output, and the building blocks would be real agent stages. The
`route="decide"` wiring is byte-identical.

The dispatched inner flow compiles with NO checkpointer in v1, so on a parent
resume the whole dispatch node re-executes -- see the Keymaker docs (durability).

Run (keyless, no network):
    uv run --extra dev python examples/29_keymaker_dynamic_flow.py
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import (
    Construct,
    Keymaker,
    Node,
    compile,
    register_type,
    run,
)
from neograph.errors import ExecutionError

# ── the dispatcher's OWN output: the emitted spec + the flow's input ─────────
# A `route="decide"` member emits this as its typed output. `spec_field="spec"`
# names the flow to run (a load_spec dict); `input_field="flow_input"` names the
# dict the framework feeds the compiled flow. The framework reads both off this
# payload after the planner body runs, then loads/validates/compiles/invokes.


class DispatchDecision(BaseModel):
    spec: dict
    flow_input: dict


# ── the flow-output types the emitted specs reference by NAME ────────────────
# An emitted spec names its node outputs as strings ("Summary", "Notes"); those
# resolve through the SAME type registry load_spec uses. A real app registers
# them before run() -- we do it inside main() (see the note there).


class Notes(BaseModel, frozen=True):
    points: list[str]


class Summary(BaseModel, frozen=True):
    text: str


class Echo(BaseModel, frozen=True):
    """A downstream consumer's output -- proves the dispatched result is a real,
    typed value on the bus that a peer node can consume."""

    heard: str


# ═════════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS for the emitted flows.
#
# These are the ONLY code an emitted spec may wire (D-DISPATCH-REGISTRIES): they
# are pre-registered on the Keymaker via `scripted=`. A machine-authored flow
# COMPOSES these pre-approved blocks; it can never introduce new code. Each is a
# plain scripted body: (input_data, config) -> a Pydantic model.
# ═════════════════════════════════════════════════════════════════════════════


def brief_summary(_input_data, _config) -> Summary:
    """One-shot brief flow: a single node produces the summary directly."""
    return Summary(text="brief: one-line recap of the request")


def draft_notes(_input_data, _config) -> Notes:
    """Thorough flow, step 1: draft intermediate notes."""
    return Notes(points=["scope confirmed", "risks listed", "owners assigned"])


def polish_summary(input_data, _config) -> Summary:
    """Thorough flow, step 2: consume the upstream `draft` notes -> a summary.

    The emitted spec wires this node's input to the `draft` node, so at runtime
    `input_data["draft"]` is the `Notes` the first node produced."""
    draft = input_data["draft"]
    return Summary(text="thorough: " + "; ".join(draft.points))


# Rejection-path sentinel: the invalid flow's body MUST NEVER run -- an invalid
# spec is rejected at the gate, before any dispatched node executes. Demo 2
# asserts this list stays empty.
DISPATCHED_BODY_RAN: list[str] = []


def _never_runs(_input_data, _config) -> Summary:
    DISPATCHED_BODY_RAN.append("ran")
    return Summary(text="should-never-be-produced")


# ── the emitted specs (version-1 load_spec dict format) ──────────────────────
# A one-node flow and a two-node flow. The planner picks WHICH to emit at
# runtime -- that runtime choice of topology is the "dynamic flow definition".

_BRIEF_SPEC: dict = {
    "name": "brief-flow",
    "nodes": [
        {"name": "brief", "mode": "scripted", "scripted_fn": "brief_summary", "outputs": "Summary"},
    ],
    "pipeline": {"nodes": ["brief"]},
}

_THOROUGH_SPEC: dict = {
    "name": "thorough-flow",
    "nodes": [
        {"name": "draft", "mode": "scripted", "scripted_fn": "draft_notes", "outputs": "Notes"},
        {
            "name": "polish",
            "mode": "scripted",
            "scripted_fn": "polish_summary",
            "inputs": {"draft": "Notes"},  # consumes the `draft` node's output
            "outputs": "Summary",
        },
    ],
    "pipeline": {"nodes": ["draft", "polish"]},
}

# Structurally parseable but INVALID: `tail` consumes an upstream `ghost` that no
# node produces -> the SAME _validate_node_chain gate a hand-written pipeline hits
# raises ConstructError inside load_spec. No bespoke validator, no schema subset.
_INVALID_SPEC: dict = {
    "name": "invalid-flow",
    "nodes": [
        {"name": "head", "mode": "scripted", "scripted_fn": "_never_runs", "outputs": "Summary"},
        {
            "name": "tail",
            "mode": "scripted",
            "scripted_fn": "_never_runs",
            "inputs": {"ghost": "Summary"},  # <- no node named `ghost` exists
            "outputs": "Summary",
        },
    ],
    "pipeline": {"nodes": ["head", "tail"]},
}


# ═════════════════════════════════════════════════════════════════════════════
# Demo 1 -- DYNAMIC DISPATCH: one planner, a runtime-chosen flow topology.
# ═════════════════════════════════════════════════════════════════════════════
# The planner reads the run request off `config['configurable']` (run(input=...)
# merges there) and emits the brief OR the thorough spec. The Keymaker loads,
# validates, compiles, and runs whichever it got; the typed Summary lands on
# `planner_dispatch`, and a downstream `consumer` reads it as a normal upstream.


def _planner_body(_input_data, config) -> DispatchDecision:
    depth = config.get("configurable", {}).get("depth", "brief")
    spec = _THOROUGH_SPEC if depth == "deep" else _BRIEF_SPEC
    return DispatchDecision(spec=spec, flow_input={})


def _consumer_body(input_data, _config) -> Echo:
    summary = input_data["planner_dispatch"]  # the dispatched flow's typed result
    return Echo(heard=summary.text)


def _dispatch_pipeline() -> Construct:
    """A planner that dispatches a runtime-chosen flow, plus a consumer that
    reads the dispatched result off `planner_dispatch`."""
    km = Keymaker(
        route="decide",
        spec_field="spec",
        input_field="flow_input",
        output=Summary,
        # the emitted flow may wire ONLY these pre-registered blocks:
        scripted={
            "brief_summary": brief_summary,
            "draft_notes": draft_notes,
            "polish_summary": polish_summary,
        },
    )
    planner = Node.scripted("planner", fn="planner_body", outputs=DispatchDecision) | km
    consumer = Node.scripted(
        "consumer", fn="consumer_body", inputs={"planner_dispatch": Summary}, outputs=Echo
    )
    return Construct("dynamic-dispatch", nodes=[planner, consumer])


def demo_dynamic_dispatch() -> None:
    print("=" * 68)
    print("DEMO 1 -- dynamic dispatch: one planner emits a runtime-chosen flow")
    print("=" * 68)

    # The planner + consumer bodies are the OUTER pipeline's scripted code
    # (registered here); the emitted-flow blocks are registered on the Keymaker.
    graph = compile(
        _dispatch_pipeline(),
        scripted={"planner_body": _planner_body, "consumer_body": _consumer_body},
    )

    # A "brief" request -> the planner emits the ONE-node brief-flow spec.
    brief = run(graph, input={"depth": "brief"})
    print(f"request depth='brief' -> dispatched flow produced: {brief['planner_dispatch'].text}")
    assert isinstance(brief["planner_dispatch"], Summary)
    assert brief["planner_dispatch"].text.startswith("brief:")
    assert brief["consumer"].heard == brief["planner_dispatch"].text

    # A "deep" request -> the SAME planner emits a different TWO-node topology
    # (draft -> polish). A static graph could not swap its own shape at runtime.
    deep = run(graph, input={"depth": "deep"})
    print(f"request depth='deep'  -> dispatched flow produced: {deep['planner_dispatch'].text}")
    assert isinstance(deep["planner_dispatch"], Summary)
    assert deep["planner_dispatch"].text.startswith("thorough:")
    assert "scope confirmed" in deep["planner_dispatch"].text  # the polish node consumed draft
    assert deep["consumer"].heard == deep["planner_dispatch"].text

    print("same planner, two runtime-chosen topologies, each validated + run.\n")


# ═════════════════════════════════════════════════════════════════════════════
# Demo 2 -- REJECTION PATH: an invalid emitted spec is caught at the gate.
# ═════════════════════════════════════════════════════════════════════════════
# The planner emits a structurally-invalid spec. Because an emitted spec passes
# the SAME Construct(...) gate as a hand-written pipeline, the defect is caught
# at load_spec -- the ConstructError surfaces WRAPPED in an ExecutionError naming
# the offending spec, raised BEFORE any dispatched sub-node body runs. The
# sentinel list proves the invalid flow's body never executed.


def _rejecting_planner(_input_data, _config) -> DispatchDecision:
    return DispatchDecision(spec=_INVALID_SPEC, flow_input={})


def _rejecting_pipeline() -> Construct:
    km = Keymaker(
        route="decide",
        spec_field="spec",
        input_field="flow_input",
        output=Summary,
        scripted={"_never_runs": _never_runs},
    )
    planner = Node.scripted("planner", fn="rejecting_planner", outputs=DispatchDecision) | km
    return Construct("dynamic-reject", nodes=[planner])


def demo_rejection_path() -> None:
    print("=" * 68)
    print("DEMO 2 -- rejection path: an invalid emitted spec fails the gate")
    print("=" * 68)

    DISPATCHED_BODY_RAN.clear()
    graph = compile(_rejecting_pipeline(), scripted={"rejecting_planner": _rejecting_planner})

    try:
        run(graph, input={})
        raise AssertionError("expected the invalid emitted spec to be rejected")
    except ExecutionError as exc:
        first_line = str(exc).splitlines()[0]
        print(f"caught (before any sub-node ran): {first_line}")
        # The wrapped ConstructError names the unresolved upstream ...
        assert "ghost" in str(exc)
        # ... and the ExecutionError names the offending spec.
        assert "invalid-flow" in str(exc)

    # Proof the rejection fired at the gate, not mid-flow: no dispatched body ran.
    assert DISPATCHED_BODY_RAN == []
    print("the dispatched flow never started -- validation ran before execution.\n")


def main() -> None:
    # Register the flow-output types the emitted specs name as strings. A real
    # app does this once at startup, before run(); doing it here also keeps the
    # example robust when the pin test imports the module under a test harness
    # that resets the type registry between tests (the documented mode-b
    # authoring constraint).
    register_type("Notes", Notes)
    register_type("Summary", Summary)

    demo_dynamic_dispatch()
    demo_rejection_path()

    print("=" * 68)
    print("KEYMAKER dynamic-flow dispatch verified: a planner emitted a flow spec")
    print("at runtime, the SAME Construct(...) gate validated it, and the typed")
    print("result landed on the bus -- while an invalid spec was rejected before")
    print("any sub-node ran. If it dispatches, it was validated.")
    print("=" * 68)


if __name__ == "__main__":
    main()
