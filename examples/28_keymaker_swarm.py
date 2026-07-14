"""Example 28: Keymaker peer routing -- a typed swarm over declared specialists.

`Keymaker` turns a set of nodes into a *mesh*: each member declares a `peers`
list, and at RUNTIME it hands the conversation off to one of those peers with a
`Command(goto)` -- not a static conditional edge fixed at compile time. A
specialist can route BACK to the router (a genuine cycle), a `max_hops` budget
guarantees termination, and `HANDOFF_END` leaves the mesh. Yet every reachable
peer is type-checked at ASSEMBLY time, so a handoff to an undeclared peer is a
compile error, never a silent runtime miss.

The routing contract is uniform: every member consumes the shared mesh channel
through the reserved `handoff` input (the payload the previous member produced)
and emits the same payload model, whose `goto` field names the next peer or
`HANDOFF_END`. This is a mesh, not a star -- the router and the specialists
speak the same protocol.

Four self-contained demos:
  1. Basic routing            -- triage hands a ticket to the right specialist.
  2. Genuine cycle            -- a specialist bounces back to triage, which re-routes.
  3. max_hops budget          -- a runaway mesh terminates (raise or clean exit).
  4. Three-surface parity     -- the same mesh built via the declarative pipe form.

The nodes here are scripted (keyless) so the example is deterministic and needs
no network -- but each `@node(peers=...)` stands in for a real agent stage. In
production `triage` would be `mode='think'` and a specialist `mode='agent'` with
tools; the `peers=` wiring is byte-identical.

This example covers mode (a) peer routing. Mode (b), dynamic flow dispatch
(`Keymaker(route='decide', ...)`) -- a member dispatches a whole sub-flow chosen
at runtime from an emitted spec -- is shown in example 29.

Run (keyless, no network):
    uv run --extra dev python examples/28_keymaker_swarm.py
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel

from neograph import (
    HANDOFF_END,
    Construct,
    FromInput,
    Keymaker,
    Node,
    compile,
    construct_from_functions,
    node,
    run,
)
from neograph.errors import ExecutionError

# ── the uniform mesh payload ─────────────────────────────────────────────────
# One model, spoken by every member. `goto` names the next peer (or HANDOFF_END);
# the ticket fields ride along so a specialist that bounces back carries context;
# `hops` is a visible hand-off counter. A mesh member reads the previous payload
# through the reserved `handoff` input and returns the next one.


class Handoff(BaseModel, frozen=True):
    goto: str  # a declared peer name, or HANDOFF_END to leave the mesh
    subject: str
    body: str
    resolution: str | None = None
    hops: int = 0


def _classify(subject: str, body: str) -> str:
    """Pick the specialist a ticket should go to (a scripted stand-in for triage)."""
    text = (subject + " " + body).lower()
    if "refund" in text or "charge" in text:
        return "billing"
    if "error" in text or "crash" in text:
        return "technical"
    return "account"


# =============================================================================
# Demo 1 -- BASIC ROUTING: triage picks a peer, the specialist resolves.
# =============================================================================
# `triage` declares three peers and a hop budget. On its first activation the
# mesh channel is empty (`handoff` is None), so it classifies the incoming ticket
# -- seeded via `run(input=...)` and read with `FromInput` -- and routes. Each
# specialist declares `peers=[]` and resolves straight to HANDOFF_END.


def demo_basic_routing() -> None:
    print("=" * 68)
    print("DEMO 1 -- basic routing: triage -> specialist -> HANDOFF_END")
    print("=" * 68)

    @node(outputs=Handoff, peers=["billing", "technical", "account"], max_hops=6)
    def triage(
        handoff: Handoff,
        subject: Annotated[str, FromInput] = "",
        body: Annotated[str, FromInput] = "",
    ) -> Handoff:
        if handoff is None:  # first activation: the mesh channel is empty
            return Handoff(goto=_classify(subject, body), subject=subject, body=body, hops=1)
        # a specialist handed back; fall through to the general-purpose desk
        return Handoff(goto="account", subject=handoff.subject, body=handoff.body, hops=handoff.hops + 1)

    @node(outputs=Handoff, peers=[])
    def billing(handoff: Handoff) -> Handoff:
        if "charge" in handoff.body.lower():
            return Handoff(goto=HANDOFF_END, subject=handoff.subject, body=handoff.body,
                           resolution="Refund issued.", hops=handoff.hops + 1)
        # not actually billing -- hand back to triage for re-routing
        return Handoff(goto="triage", subject=handoff.subject, body=handoff.body, hops=handoff.hops + 1)

    @node(outputs=Handoff, peers=[])
    def technical(handoff: Handoff) -> Handoff:
        return Handoff(goto=HANDOFF_END, subject=handoff.subject, body=handoff.body,
                       resolution="Patch shipped.", hops=handoff.hops + 1)

    @node(outputs=Handoff, peers=[])
    def account(handoff: Handoff) -> Handoff:
        return Handoff(goto=HANDOFF_END, subject=handoff.subject, body=handoff.body,
                       resolution="Account updated.", hops=handoff.hops + 1)

    # `billing` declares peers=["triage"] so it can bounce back (used in demo 2);
    # here we keep peers=[] so demo 1 stays a straight route. Build + run:
    graph = compile(construct_from_functions("swarm-basic", [triage, billing, technical, account]))
    result = run(graph, input={"subject": "Double charge", "body": "I was charged twice, need a refund."})

    resolved = result["billing"]
    print(f"routed to billing at hop {result['triage'].hops}; resolved at hop {resolved.hops}")
    print(f"resolution: {resolved.resolution}\n")
    assert resolved.goto == HANDOFF_END
    assert resolved.resolution == "Refund issued."


# =============================================================================
# Demo 2 -- GENUINE CYCLE: a specialist routes BACK to triage (a real back-edge).
# =============================================================================
# A ticket looks like billing ("refund") but is not ("charge" absent), so billing
# hands BACK to triage -- a genuine runtime cycle a static edge could not express.
# triage sees the populated channel on re-entry and re-routes to the account desk.
# The hop counter proves the round trip: triage(1) -> billing(2) -> triage(3) ->
# account(4).


def demo_cycle() -> None:
    print("=" * 68)
    print("DEMO 2 -- genuine cycle: triage -> billing -> triage -> account")
    print("=" * 68)

    @node(outputs=Handoff, peers=["billing", "account"], max_hops=6)
    def triage(
        handoff: Handoff,
        subject: Annotated[str, FromInput] = "",
        body: Annotated[str, FromInput] = "",
    ) -> Handoff:
        if handoff is None:
            goto = "billing" if "refund" in (subject + " " + body).lower() else "account"
            return Handoff(goto=goto, subject=subject, body=body, hops=1)
        # re-entry after a bounce-back: send to the general-purpose account desk
        return Handoff(goto="account", subject=handoff.subject, body=handoff.body, hops=handoff.hops + 1)

    @node(outputs=Handoff, peers=["triage"])  # billing can hand BACK to triage
    def billing(handoff: Handoff) -> Handoff:
        if "charge" in handoff.body.lower():
            return Handoff(goto=HANDOFF_END, subject=handoff.subject, body=handoff.body,
                           resolution="Refund issued.", hops=handoff.hops + 1)
        return Handoff(goto="triage", subject=handoff.subject, body=handoff.body, hops=handoff.hops + 1)

    @node(outputs=Handoff, peers=[])
    def account(handoff: Handoff) -> Handoff:
        return Handoff(goto=HANDOFF_END, subject=handoff.subject, body=handoff.body,
                       resolution="Account updated.", hops=handoff.hops + 1)

    graph = compile(construct_from_functions("swarm-cycle", [triage, billing, account]))
    # "refund" routes to billing, but no "charge" -> billing bounces back to triage.
    result = run(graph, input={"subject": "refund request", "body": "please refund my subscription"})

    resolved = result["account"]
    print(f"billing bounced back (goto={result['billing'].goto!r}); triage re-routed to account")
    print(f"resolved at hop {resolved.hops}: {resolved.resolution}\n")
    assert result["billing"].goto == "triage"  # the back-edge fired
    assert resolved.goto == HANDOFF_END and resolved.hops == 4


# =============================================================================
# Demo 3 -- MAX_HOPS BUDGET: a runaway mesh always terminates.
# =============================================================================
# Two members that hand off to each other forever. `max_hops` (an entry-only knob)
# bounds the cycle. `on_exhaust='error'` raises when the budget is spent;
# `on_exhaust='exit'` instead leaves the mesh cleanly with the last payload. Either
# way the mesh is guaranteed to terminate -- the durability property a free-form
# agent loop cannot promise.


def demo_max_hops_budget() -> None:
    print("=" * 68)
    print("DEMO 3 -- max_hops budget guarantees termination")
    print("=" * 68)

    class Ping(BaseModel, frozen=True):
        goto: str
        hops: int = 0

    def _ping_pong(peers_exhaust: str):
        @node(outputs=Ping, peers=["pong"], max_hops=3, on_exhaust=peers_exhaust)  # type: ignore[arg-type]
        def ping(handoff: Ping) -> Ping:
            return Ping(goto="pong", hops=(0 if handoff is None else handoff.hops + 1))

        @node(outputs=Ping, peers=["ping"])
        def pong(handoff: Ping) -> Ping:
            return Ping(goto="ping", hops=handoff.hops + 1)

        return construct_from_functions(f"pingpong-{peers_exhaust}", [ping, pong])

    # on_exhaust='error' -> the budget breach raises, surfacing the runaway loop.
    graph_err = compile(_ping_pong("error"))
    try:
        run(graph_err, input={})
        raise AssertionError("expected the max_hops budget to raise")
    except ExecutionError as exc:
        print(f"on_exhaust='error': budget stopped the loop -> {str(exc).splitlines()[0]}")

    # on_exhaust='exit' -> the mesh leaves cleanly with the last payload on the bus.
    graph_exit = compile(_ping_pong("exit"))
    result = run(graph_exit, input={})
    print(f"on_exhaust='exit': mesh exited cleanly with state keys {sorted(k for k in result if not k.startswith('neo_'))}\n")
    assert "ping" in result and "pong" in result


# =============================================================================
# Demo 4 -- THREE-SURFACE PARITY: the same mesh via the declarative pipe form.
# =============================================================================
# The `@node(peers=...)` sugar is one of three surfaces; declarative
# `Node.scripted(...) | Keymaker(...)` produces the IDENTICAL mesh IR and routing.
# The keyless scripted functions get their bodies through `compile(scripted=...)`.


def demo_surface_parity() -> None:
    print("=" * 68)
    print("DEMO 4 -- three-surface parity: declarative Node | Keymaker")
    print("=" * 68)

    def fn_triage(input_data, _config):
        # first activation: the channel is empty; route to billing.
        incoming = input_data.get("handoff") if isinstance(input_data, dict) else None
        if incoming is None:
            return Handoff(goto="billing", subject="Double charge", body="charged twice", hops=1)
        return Handoff(goto=HANDOFF_END, subject=incoming.subject, body=incoming.body, hops=incoming.hops + 1)

    def fn_billing(input_data, _config):
        incoming = input_data["handoff"]
        return Handoff(goto=HANDOFF_END, subject=incoming.subject, body=incoming.body,
                       resolution="Refund issued.", hops=incoming.hops + 1)

    triage = Node.scripted("triage", fn="fn_triage", outputs=Handoff) | Keymaker(peers=["billing"], max_hops=6)
    billing = (
        Node.scripted("billing", fn="fn_billing", inputs={"handoff": Handoff}, outputs=Handoff)
        | Keymaker(peers=[])
    )
    mesh = Construct("swarm-declarative", nodes=[triage, billing])

    graph = compile(mesh, scripted={"fn_triage": fn_triage, "fn_billing": fn_billing})
    result = run(graph, input={})

    resolved = result["billing"]
    print(f"declarative mesh resolved at hop {resolved.hops}: {resolved.resolution}")
    print("same peers= wiring, same Command(goto) routing -- just a different surface.\n")
    assert resolved.goto == HANDOFF_END and resolved.resolution == "Refund issued."


def main() -> None:
    demo_basic_routing()
    demo_cycle()
    demo_max_hops_budget()
    demo_surface_parity()
    print("=" * 68)
    print("All KEYMAKER peer-routing surfaces ran: basic routing, genuine cycle,")
    print("max_hops budget termination, and declarative three-surface parity.")
    print("Routing decided at runtime (Command(goto)); every peer typed at compile.")
    print("=" * 68)


if __name__ == "__main__":
    main()
