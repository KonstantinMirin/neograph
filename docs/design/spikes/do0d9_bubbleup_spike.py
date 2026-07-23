"""do0d9 SPIKE: parent-scoped bubble-up from inside a sub-construct to a parent Portal peer.

Goal (from neograph-670qe acceptance): a REAL two-level neograph pipeline where an
inner sub-construct member, at runtime, decides to hand off to a DIFFERENT
PARENT-level Portal peer -- routed correctly, no silent seam, no swallowed
escape, no uncaught crash. Sync AND async.

Mechanism proven here (the do0d9 replacement for the dead Command.PARENT path):

  The sub-construct runs ISOLATED (its own `run()` / Pregel invocation, exactly
  like neograph's `make_subgraph_fn` does today). Its declared output payload
  carries a routing decision (`goto`) whose value may name a PARENT-level Portal
  peer. At the BOUNDARY, the sub-construct is invoked from inside a node that is
  itself a real parent Portal mesh member; that member RETURNS the sub-construct's
  payload, so neograph's own `make_portal_fn` -> `_portal_route_to_command` reads
  the payload's route field, writes it onto the PARENT mesh channel
  (`StateKeys.handoff_payload(parent_entry_field)`), and emits `Command(goto=<parent
  peer>)`. The parent mesh routes on it via the normal Portal path.

  In production do0d9 this boundary lives in a `make_portal_subgraph_fn` in
  factory.py (so the `Command(` construction stays inside guard G1), and the
  sub-construct is a first-class parent mesh member validated by an extended
  `_check_portal_mesh`. In THIS spike the boundary is expressed as a real
  `@node(portal=[...])` whose body invokes the compiled sub-construct -- i.e. a
  "node wrapping a sub-construct" -- so that 100% of the ROUTING mechanism is
  driven by neograph's real Portal machinery (make_portal_fn / _portal_route_to_command
  / the entry-keyed mesh channel / the reserved `handoff` input), and only the
  sub-invocation-and-payload-lift is hand-written prototype code.

Run:
    uv run --extra dev python spike_do0d9.py
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from pydantic import BaseModel

from neograph import (
    HANDOFF_END,
    FromInput,
    arun,
    compile,
    construct_from_functions,
    node,
    run,
)


# ---------------------------------------------------------------------------
# The uniform mesh payload spoken by the PARENT mesh. `goto` names the next
# parent peer or HANDOFF_END. This is also the sub-construct's declared output
# type, so the sub-construct's decision rides out on the SAME field the parent
# Portal machinery routes on -- that is the "escape intent".
# ---------------------------------------------------------------------------
class Handoff(BaseModel, frozen=True):
    goto: str  # a parent peer name, or HANDOFF_END
    subject: str
    trail: list[str] = []          # visible audit of who handled it
    resolution: str | None = None


# ===========================================================================
# The SUB-CONSTRUCT: a real isolated neograph pipeline with its OWN internal
# members. Its internal decider chooses, at runtime, whether to resolve
# locally (HANDOFF_END) or ESCALATE to a DIFFERENT PARENT-level peer.
# ===========================================================================
class SubDecision(BaseModel, frozen=True):
    """The sub-construct's own internal working type."""
    subject: str
    kind: Literal["billing", "escalate"]
    trail: list[str]


@node(outputs=SubDecision)
def sub_intake(subject: Annotated[str, FromInput]) -> SubDecision:
    """Inner member 1: classify the ticket inside the sub-construct."""
    low = subject.lower()
    kind: Literal["billing", "escalate"] = "billing" if "refund" in low else "escalate"
    return SubDecision(subject=subject, kind=kind, trail=["sub_intake"])


@node(outputs=Handoff)
def sub_decide(sub_intake: SubDecision) -> Handoff:
    """Inner member 2 (the internal member that decides the handoff).

    If the sub-construct can finish the work, it resolves locally and routes to
    HANDOFF_END. If it CANNOT (needs a parent-level specialist), it emits a
    payload whose `goto` names a PARENT peer -- the escape intent. The parent
    peer name ("specialist") is NOT a sub-local node; it is a parent mesh member.
    """
    trail = [*sub_intake.trail, "sub_decide"]
    if sub_intake.kind == "billing":
        # resolvable inside the sub-construct: leave the parent decision as a
        # clean local finish.
        return Handoff(
            goto=HANDOFF_END,
            subject=sub_intake.subject,
            trail=trail,
            resolution="refund issued inside sub-construct",
        )
    # ESCAPE: the sub-construct decides to hand off to a parent-level peer.
    return Handoff(goto="specialist", subject=sub_intake.subject, trail=trail)


# Compiled once; the boundary node invokes it in isolation (mirrors
# make_subgraph_fn's isolated sub_graph.invoke()).
_sub_graph = compile(
    construct_from_functions("resolver_sub", [sub_intake, sub_decide], output=Handoff)
)


def _extract_handoff(state) -> Handoff:
    """Pull the sub-construct's declared Handoff output from its result state.

    Mirrors make_subgraph_fn._scan_subgraph_output (reverse type-scan for the
    declared output type).
    """
    if isinstance(state, Handoff):
        return state
    for v in reversed(list(state.values())):
        if isinstance(v, Handoff):
            return v
    raise AssertionError("sub-construct produced no Handoff output")


def _run_sub(subject: str) -> Handoff:
    """Prototype boundary: invoke the sub-construct in isolation, return its payload.

    In production this is the body of factory.make_portal_subgraph_fn (isolated
    invoke + declared-output extraction), and it stays inside guard G1 because the
    Command wrapping happens in _portal_route_to_command, not here.
    """
    return _extract_handoff(run(_sub_graph, input={"subject": subject}))


async def _arun_sub(subject: str) -> Handoff:
    return _extract_handoff(await arun(_sub_graph, input={"subject": subject}))


# ===========================================================================
# The PARENT mesh: a real neograph Portal mesh with >=2 peers. `worker_region`
# is the peer that wraps the sub-construct. Its returned payload is piped
# through neograph's OWN make_portal_fn -> _portal_route_to_command, which does
# the channel-write + Command(goto=<parent peer>). No prototype routing code.
# ===========================================================================
def build_parent(async_mode: bool = False):
    @node(outputs=Handoff, portal=["worker"], max_hops=8)
    def dispatcher(
        handoff: Handoff,
        subject: Annotated[str, FromInput] = "",
    ) -> Handoff:
        # parent mesh entry: first activation seeds from FromInput, then routes
        # into the sub-construct-wrapping peer.
        if handoff is None:
            return Handoff(goto="worker", subject=subject, trail=["dispatcher"])
        # a peer bounced back (not exercised in the happy path) -- re-route.
        return Handoff(goto="worker", subject=handoff.subject, trail=[*handoff.trail, "dispatcher"])

    @node(name="worker", outputs=Handoff, portal=["specialist", "closer"])
    def worker_region(handoff: Handoff) -> Handoff:
        """Parent peer that wraps the sub-construct.

        Invokes the isolated sub-construct; the sub-construct's INTERNAL member
        decides to escalate to the PARENT peer `specialist`. This node returns
        that payload verbatim -- neograph's Portal wrapper then routes on its
        `goto` field. The escape crosses the sub-construct boundary here.
        """
        subject = handoff.subject
        sub_out = _run_sub(subject)  # <-- isolated sub-construct invocation
        # bubble the sub-construct's decision up: its `goto` may name a PARENT peer
        return Handoff(
            goto=sub_out.goto,
            subject=sub_out.subject,
            trail=[*handoff.trail, "worker_region", *sub_out.trail],
            resolution=sub_out.resolution,
        )

    # async twin of worker_region (neograph selects by driver)
    @node(name="worker", outputs=Handoff, portal=["specialist", "closer"])
    async def worker_region_async(handoff: Handoff) -> Handoff:
        subject = handoff.subject
        sub_out = await _arun_sub(subject)
        return Handoff(
            goto=sub_out.goto,
            subject=sub_out.subject,
            trail=[*handoff.trail, "worker_region", *sub_out.trail],
            resolution=sub_out.resolution,
        )

    @node(outputs=Handoff, portal=[])
    def specialist(handoff: Handoff) -> Handoff:
        """The PARENT peer the sub-construct escaped to. Proof-of-arrival."""
        return Handoff(
            goto=HANDOFF_END,
            subject=handoff.subject,
            trail=[*handoff.trail, "specialist"],
            resolution="escalated ticket resolved by parent specialist",
        )

    @node(outputs=Handoff, portal=[])
    def closer(handoff: Handoff) -> Handoff:
        return Handoff(
            goto=HANDOFF_END,
            subject=handoff.subject,
            trail=[*handoff.trail, "closer"],
            resolution="closed",
        )

    worker = worker_region_async if async_mode else worker_region
    return construct_from_functions(
        "parent_mesh", [dispatcher, worker, specialist, closer]
    )


def _check(tag: str, result: Handoff, subject: str, expect_escape: bool) -> None:
    trail = result.trail
    print(f"[{tag}] subject={subject!r}")
    print(f"       trail={trail}")
    print(f"       resolution={result.resolution!r}")
    if expect_escape:
        assert "sub_decide" in trail, f"{tag}: sub-construct internal member never ran"
        assert "specialist" in trail, f"{tag}: escape did NOT reach parent peer 'specialist'"
        assert result.resolution == "escalated ticket resolved by parent specialist", (
            f"{tag}: wrong resolver -- silent seam? got {result.resolution!r}"
        )
        # prove it did NOT just finish inside worker_region and did NOT hit 'closer'
        assert "closer" not in trail, f"{tag}: mis-routed to the wrong parent peer"
        print(f"       PASS: inner member escaped across boundary to parent peer 'specialist'")
    else:
        assert result.resolution == "refund issued inside sub-construct", f"{tag}: local finish broken"
        assert "specialist" not in trail, f"{tag}: unexpected escape"
        print(f"       PASS: sub-construct finished locally, no escape")
    print()


def main() -> None:
    print("=" * 72)
    print("do0d9 SPIKE -- parent-scoped bubble-up across the sub-construct boundary")
    print("=" * 72)

    # ---- SYNC ----
    parent = compile(build_parent(async_mode=False))

    # Case A: escalate -> must cross the boundary to parent peer 'specialist'
    res = run(parent, input={"subject": "login crash needs deep investigation"})
    _check("sync/escape", res["specialist"] if isinstance(res, dict) else res, "login crash...", expect_escape=True)

    # Case B: resolvable locally inside the sub-construct -> no escape
    res = run(parent, input={"subject": "please process my refund"})
    # local finish emerges on worker_region's field (HANDOFF_END exit)
    payload = _final_payload(res)
    _check("sync/local", payload, "refund...", expect_escape=False)

    # ---- ASYNC ----
    parent_a = compile(build_parent(async_mode=True))
    res = asyncio.run(arun(parent_a, input={"subject": "login crash needs deep investigation"}))
    _check("async/escape", _final_payload(res), "login crash...", expect_escape=True)

    res = asyncio.run(arun(parent_a, input={"subject": "please process my refund"}))
    _check("async/local", _final_payload(res), "refund...", expect_escape=False)

    print("ALL SPIKE ASSERTIONS PASSED")


def _final_payload(res) -> Handoff:
    """Pick the terminal Handoff from the returned parent state dict.

    The mesh writes each member's payload to its own field; the last one to run
    (the HANDOFF_END exiter) carries the final resolution. Prefer specialist >
    closer > worker_region > dispatcher.
    """
    if not isinstance(res, dict):
        return res
    for key in ("specialist", "closer", "worker_region", "dispatcher"):
        if res.get(key) is not None:
            # the terminal payload is the one with a resolution or HANDOFF_END goto
            pass
    # choose the payload with a resolution set, else the deepest trail
    candidates = [v for v in res.values() if isinstance(v, Handoff)]
    resolved = [c for c in candidates if c.resolution is not None]
    pool = resolved or candidates
    return max(pool, key=lambda h: len(h.trail))


if __name__ == "__main__":
    main()
