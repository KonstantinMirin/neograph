"""Example 33: Run-scoped state -- reaching a value the previous step didn't hand you.

A step often needs a value produced earlier in the run that is NOT what its
input port carries: run identity, a session handle, a tenant. Threading it
through every intermediate type pollutes the domain model -- a `Claim` is about
a fault, not about which deal it belongs to -- and passing it as config only
works while the value is fixed for the whole run.

There are three routes, and the difference between them is the whole point of
this example. They are not alternatives; they answer different questions.

    context=["run_ctx"]            an LLM node SEES the value
    inputs={"item": X, "ctx": Y}   a scripted node READS the value (type-checked)
    Tool(bound_args={...})         a tool RECEIVES the value, and the model
                                   cannot substitute a different one

The third exists because the first is not a guarantee. An LLM composes its own
tool-call arguments, so being shown the right deal id does not stop a model
emitting a different one -- and that failure is silent, because a wrong id
returns an empty result that reads exactly like a legitimately empty one.

To make that concrete rather than asserted, the fake model below DELIBERATELY
asks for deal 1 when the run is about deal 4822. Watch what the tool receives.

Keyless -- uses a fake LLM.

Run:
    python examples/33_run_scoped_state.py
"""

from __future__ import annotations

import sys

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from neograph import Tool, compile, construct_from_module, node, run

# -- Schemas --------------------------------------------------------------


class RunCtx(BaseModel, frozen=True):
    """Produced once, early. Every later step needs it; no later step produces it."""

    deal_id: int
    tenant: str


class Claim(BaseModel, frozen=True):
    """A domain object. Note what it does NOT carry: any run identity."""

    text: str


class Claims(BaseModel, frozen=True):
    items: tuple[Claim, ...]


class Triage(BaseModel, frozen=True):
    label: str


class Verdict(BaseModel, frozen=True):
    text: str


# -- What the tool actually received ---------------------------------------

tool_calls_seen: list[dict] = []


class DealLookupTool:
    name = "get_deal"

    def invoke(self, args, config=None, **kwargs):
        tool_calls_seen.append(dict(args))
        return f"deal {args['deal_id']}: 3 open findings"


# -- A model that guesses ---------------------------------------------------


# Module-level so the count survives bind_tools(), which the cycle calls once
# per turn and which must return a bound clone.
turns = {"n": 0}


class GuessingLLM:
    """Emits deal_id=1. The run is about deal 4822.

    This is not a strawman: it is the reported production shape, where a
    fan-out branch made 49 calls whose deal id read 1, 2, 3, 4, 5, 1001.
    """

    def bind_tools(self, tools):
        return GuessingLLM()

    def invoke(self, messages, **kwargs):
        turns["n"] += 1
        if turns["n"] == 1:
            msg = AIMessage(content="")
            msg.tool_calls = [
                {"name": "get_deal", "args": {"deal_id": 1, "verbosity": "full"}, "id": "c1"}
            ]
            return msg
        return AIMessage(content=Verdict(text="reviewed").model_dump_json())

    def with_structured_output(self, model):
        return self


def llm_factory(tier):
    return GuessingLLM()


# -- Route 1 + 2: the pipeline ---------------------------------------------


@node(outputs=RunCtx)
def run_ctx() -> RunCtx:
    return RunCtx(deal_id=4822, tenant="acme")


@node(outputs=Claims)
def claims() -> Claims:
    return Claims(items=(Claim(text="missing index"), Claim(text="stale cache")))


@node(outputs=Triage, map_over="claims.items", map_key="text")
def triage(item: Claim, run_ctx: RunCtx) -> Triage:
    """ROUTE 2 -- a scripted fan-out branch.

    `item` is the fanned value; `run_ctx` is an ordinary upstream read. A
    scripted node needs no special mechanism: the port carries WHICH ITEM and a
    second fan-in key carries WHICH RUN. Both are type-checked by the validator,
    which is why this is the route to prefer when the node runs Python.
    """
    return Triage(label=f"{item.text} [deal {run_ctx.deal_id}/{run_ctx.tenant}]")


# ROUTE 1 + 3 -- an LLM node that both SEES the run context and cannot
# misreport it to a tool. The two declarations sit side by side on purpose:
# one is what the model is shown, the other is what it cannot change.
@node(
    mode="agent",
    outputs=Verdict,
    model="fast",
    prompt="review",
    # ROUTE 1: the model SEES the run context. Declared, so assembly fails if
    # nothing upstream produces it.
    context=["run_ctx"],
    tools=[
        Tool(
            name="get_deal",
            budget=3,
            # ROUTE 3: the framework SUPPLIES this argument. Whatever the model
            # emits for deal_id is overwritten from run state. `verbosity` is
            # not named here, so it stays exactly as the model composed it.
            bound_args={"deal_id": "run_ctx.deal_id"},
        )
    ],
)
def review(claims: Claims) -> Verdict:
    # body unused for mode='agent' -- the LLM drives the tool loop
    ...


pipeline = construct_from_module(sys.modules[__name__], name="run-scoped-state")


# -- Run --------------------------------------------------------------------

if __name__ == "__main__":
    graph = compile(
        pipeline,
        llm_factory=llm_factory,
        prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": "review"}],
        tool_factories={"get_deal": lambda config, tool_config: DealLookupTool()},
    )
    result = run(graph, input={"node_id": "REVIEW-1"})

    print("Route 2 -- scripted fan-out branches, each reading the run context:")
    for key, value in result["triage"].items():
        print(f"  {key}: {value.label}")

    args = tool_calls_seen[0]
    print("\nRoute 3 -- what the model asked for vs what the tool received:")
    print("  model emitted   : deal_id=1        <- an id that does not exist")
    print(f"  tool received   : deal_id={args['deal_id']}     <- supplied from run state")
    print(f"  left untouched  : verbosity={args['verbosity']!r}  <- not bound, so still the model's")

    assert args["deal_id"] == 4822, "the model's invented id reached the tool"
    assert args["verbosity"] == "full", "binding one argument swallowed another"

    print("\nWithout bound_args the first line would have been the tool call.")
    print("A wrong id returns an empty result that reads like a real one, so the")
    print("run completes, every gate stays green, and the answer is derived from")
    print("evidence about a deal that does not exist.")
