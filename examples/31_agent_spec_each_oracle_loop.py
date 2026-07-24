"""Example 31: Each + Oracle + Loop shown side-by-side with their Agent Spec Flow.

Cite docs/design/agent-spec-interop-2026-07-09.md s5 (Helper lowerings table),
s10.8 (this example) and neograph-e3b4j's refined plan (three separate compact
panels, not one fused construct -- the combined shape is unproven).

Scenario: three tiny, independent pipelines, ONE per neograph modifier
(Each, Oracle, Loop), each just a few lines. For each panel this example
prints the REAL Agent Spec ``Flow`` that ``to_agent_spec()`` lowers it to --
the exact JSON dump, and its ACTUAL line count (never a hand-typed number).
This is BOTH the fidelity proof and the marketing asset: a few lines of
neograph collapse to many lines of flat Agent Spec primitives -- Oracle in
particular has NO native Agent Spec node at all (it lowers to N parallel
LlmNodes + a merge node + fan-out/fan-in edges), which is why Oracle is the
flagship gap to advertise (design doc s5 line ~176-178).

Fidelity table this reproduces (design doc s5):
  - Each  -> MapNode wrapping an extracted sub-Flow
  - Oracle -> NO node; N LlmNodes (fan-out) + one merge node + edges
  - Loop  -> BranchingNode + cyclic ControlFlowEdges + self DataFlowEdges

Boundary note (Core Invariant): the "Agent Spec Flow: N lines" printout for
each panel is computed FROM the live ``to_agent_spec(pipeline).to_dict()``
dump, never a rhetorical/guessed number -- if the exporter's output changes,
this printout changes with it, so the shown text can never silently drift
from what neograph actually emits.

Known gap (design doc s6, lossy-callable-fields): Oracle only round-trips
correctly in "think" mode today -- a scripted-mode Oracle node reconstructs
into a broken think-mode node on import (tracked separately, neograph-aa5gq
note in tests/test_agent_spec_roundtrip.py). This example's Oracle panel
therefore uses mode="think" with a keyless fake LLM, matching the currently-
correct path -- not a scripted merge that would silently misrepresent the gap.

Each/Loop panel scripted functions use the DECLARATIVE ``Node.scripted(name=X,
fn=X)`` + a matching ``register_scripted(X, fn)`` call (mirroring
tests/test_agent_spec_roundtrip.py's own proven pattern) rather than @node's
sugar: a round-tripped scripted node's ``scripted_fn`` is always re-derived
from the EXPORTED node's own ``.name`` (loader.py), never from the original
Python callable's identity, so ``name == fn == the registered key`` is what
makes the SAME pipeline resolve identically whether compiled directly or
after an export -> import round-trip. ``register_scripted`` writes into the
shared decoration-time registry that ``compile()`` always merges in
(``neograph._runtime_registry``), so no caller (this file's own ``main()`` or
an external test) needs to pass an explicit ``scripted=`` kwarg.

Each panel's fan-out consumer and the Loop panel's looped node use explicit
single-type ``inputs=`` (piped ``| Each(...)``/``| Loop(...)`` afterward)
rather than @node's ``map_over=``/param-name-inferred dict-form sugar, and
the Oracle panel's ``@merge_fn`` is deliberately untyped and single-param --
each sidesteps a DISTINCT, separately-discovered Agent-Spec-interop gap this
docs/example task is not scoped to fix (see the inline comments at each
site): dict-form fan-in against an Each fan-out receiver, Loop's self-edge
requiring bare (unprefixed) Property titles, and typed merge_fn annotations
triggering an unwanted type-transform inference. None of these are silently
worked around -- each is called out where it bites, with a one-line reason.

Keyless: all three panels use either scripted nodes or a fake "think"-mode
LLM -- no API keys required. Requires the [agent-spec] extra.

Run:
    uv run --extra agent-spec python examples/31_agent_spec_each_oracle_loop.py
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from neograph import Construct, Each, Loop, Node, Oracle, compile, merge_fn, run
from neograph._agent_spec import to_agent_spec
from neograph._runtime_registry import register_scripted
from neograph.loader import from_agent_spec
from neograph.testing.fakes import StructuredFake

# -- Schemas ------------------------------------------------------------------


class Tagged(BaseModel, frozen=True):
    label: str


class Bag(BaseModel, frozen=True):
    items: list[Tagged]


class Result(BaseModel, frozen=True):
    value: str


class Draft(BaseModel, frozen=True):
    content: str
    iteration: int
    score: float


# Field named "claims", not "items" -- avoids an Agent-Spec-import-side type-
# synthesis hash collision with Bag/Result's "items" field: spec_types.py's
# _structural_type_name signature is (title, str(Property.type)), and
# pyagentspec's own Property.type is the bare JSON-schema keyword ("array"),
# which does not distinguish list[str] from list[Tagged] -- two same-named,
# same-JSON-schema-type fields synthesize the SAME reconstructed class even
# though their Python item types differ (a real, separate interop gap, out
# of scope for this docs/example task -- sidestepped by a distinct field
# name, not silently worked around).
class Claims(BaseModel, frozen=True):
    claims: list[str]


# -- Each panel -----------------------------------------------------------


def _each_seed_fn(input_data, config):
    return Bag(items=[Tagged(label="a"), Tagged(label="b")])


register_scripted("each_seed", _each_seed_fn)


def _each_step_fn(input_data, config):
    return Result(value=f"tagged-{input_data.label}")


register_scripted("each_step", _each_step_fn)


def build_each_panel() -> Construct:
    seed = Node.scripted("each_seed", fn="each_seed", outputs=Bag)
    step = Node.scripted("each_step", fn="each_step", inputs=Tagged, outputs=Result) | Each(
        over="each_seed.items", key="label"
    )
    return Construct("each-panel", nodes=[seed, step])


# -- Oracle panel -----------------------------------------------------------


@merge_fn
def oracle_merge(variants):
    # Deliberately UNTYPED and single-param (no `variants: list[Claims]`
    # annotation, no trailing `config`): a typed annotation makes
    # infer_oracle_gen_type() (_sidecar.py) treat this as a type-TRANSFORMING
    # merge (generators produce the annotated type, merge produces
    # node.outputs) -- which would then require this function to return
    # node.outputs's type while its OWN annotation says Claims, a mismatch.
    # Untyped + single-param (matching @merge_fn's own proven convention,
    # tests/test_composition.py) keeps gen_type == outputs, so this just
    # operates on whatever type it actually receives -- type(variants[0]),
    # never a hardcoded original class (the reconstructed output type is a
    # freshly-synthesized class with no back-reference to "Claims").
    all_claims: list[str] = []
    for v in variants:
        all_claims.extend(v.claims)
    return type(variants[0])(claims=all_claims)


def build_oracle_panel() -> Construct:
    ensemble = Node(name="ensemble", mode="think", model="fast", outputs=Claims, prompt="rw/ensemble") | Oracle(
        n=3, merge_fn="oracle_merge"
    )
    return Construct("oracle-panel", nodes=[ensemble])


# -- Loop panel -----------------------------------------------------------

_loop_call_count = [0]


def _loop_seed_fn(input_data, config):
    _loop_call_count[0] = 0
    return Draft(content="v0", iteration=0, score=0.0)


register_scripted("loop_seed", _loop_seed_fn)


def _loop_refine_fn(input_data, config):
    # from_agent_spec always reconstructs single-type inputs as dict-form
    # ({upstream_name: type}) -- the upstream/reentry value lives under the
    # "loop_seed" key (mirrors test_agent_spec_roundtrip.py's own comment).
    _loop_call_count[0] += 1
    prev = input_data["loop_seed"] if isinstance(input_data, dict) else input_data
    return Draft(content=f"v{_loop_call_count[0]}", iteration=prev.iteration + 1, score=prev.score + 0.3)


register_scripted("loop_refine", _loop_refine_fn)


def build_loop_panel() -> Construct:
    seed = Node.scripted("loop_seed", fn="loop_seed", outputs=Draft)
    refine = Node.scripted("refine", fn="loop_refine", inputs=Draft, outputs=Draft) | Loop(
        when="score < 0.8", max_iterations=10
    )
    return Construct("loop-panel", nodes=[seed, refine])


# -- Shared: real, mechanically-derived Agent Spec dump ----------------------


def _print_panel(title: str, pipeline: Construct) -> dict:
    flow = to_agent_spec(pipeline)
    dumped = json.dumps(flow.to_dict(), indent=2)
    line_count = dumped.count("\n") + 1
    print(f"\n=== {title} ===")
    print(f"neograph pipeline: {len(pipeline.nodes)} node(s)")
    print(f"Agent Spec Flow: {line_count} lines")
    return {"flow": flow, "dump": dumped}


def main() -> None:
    # -- EACH PANEL --
    each_pipeline = build_each_panel()
    _print_panel("EACH PANEL", each_pipeline)
    each_imported = from_agent_spec(to_agent_spec(each_pipeline))
    each_graph = compile(each_imported)
    each_result = run(each_graph, input={"node_id": "each-panel"})
    print(f"Round-trip result: {each_result['each_step']}")

    # -- ORACLE PANEL --
    oracle_pipeline = build_oracle_panel()
    _print_panel("ORACLE PANEL", oracle_pipeline)
    oracle_imported = from_agent_spec(to_agent_spec(oracle_pipeline))
    fake_llm = StructuredFake(lambda m: m(**{next(iter(m.model_fields)): ["variant"]}))
    oracle_graph = compile(
        oracle_imported,
        llm_factory=lambda tier, node_name=None, llm_config=None: fake_llm,
        prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": "generate"}],
    )
    oracle_result = run(oracle_graph, input={"node_id": "oracle-panel"})
    ensemble_field = next(k for k in oracle_result if not k.startswith("neo_") and "node_id" not in k)
    print(f"Round-trip result: {oracle_result[ensemble_field]}")

    # -- LOOP PANEL --
    loop_pipeline = build_loop_panel()
    _print_panel("LOOP PANEL", loop_pipeline)
    loop_imported = from_agent_spec(to_agent_spec(loop_pipeline))
    loop_graph = compile(loop_imported)
    loop_result = run(loop_graph, input={"node_id": "loop-panel"})
    print(f"Round-trip result: {loop_result['refine']}")


if __name__ == "__main__":
    main()
