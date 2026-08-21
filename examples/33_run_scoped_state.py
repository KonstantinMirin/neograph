"""Example 33: Run-scoped state -- reaching a value the previous step didn't hand you.

A step often needs a value produced earlier in the run that is NOT what its
input port carries: which tenant, which warehouse, which session. Threading it
through every intermediate type pollutes the domain model -- a `Discrepancy` is
about a product, not about which warehouse the audit is running in -- and
passing it as config only works while the value is fixed for the whole run.

There are three routes, and the difference between them is the point of this
example. They are not alternatives; they answer different questions.

    context=["audit"]              an LLM node SEES the value
    inputs={"item": X, "ctx": Y}   a scripted node READS the value (type-checked)
    Tool(bound_args={...})         a tool RECEIVES the value, and the model
                                   cannot substitute a different one

The third exists because the first is not a guarantee. An LLM composes its own
tool-call arguments, so being shown the right warehouse does not stop a model
emitting a different one.

That failure is quiet, which is what makes it expensive. Ask for a SKU's stock
in the wrong warehouse and the answer is zero units -- indistinguishable from a
genuine stockout. The run completes, every gate stays green, and the report
says "out of stock" about a shelf nobody looked at.

To make that concrete rather than asserted, the fake model below DELIBERATELY
queries warehouse 1 when the audit is running in warehouse 4822. Watch what the
tool receives.

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


class Audit(BaseModel, frozen=True):
    """Produced once, early. Every later step needs it; no later step produces it."""

    warehouse_id: int
    region: str


class Discrepancy(BaseModel, frozen=True):
    """A domain object. Note what it does NOT carry: any run identity.

    A discrepancy is about a product. Adding `warehouse_id` here to get it
    downstream would make the domain model carry routing information, and doing
    that a few times is how a clean model turns into transport envelopes.
    """

    sku: str


class Discrepancies(BaseModel, frozen=True):
    items: tuple[Discrepancy, ...]


class Classified(BaseModel, frozen=True):
    label: str


class Report(BaseModel, frozen=True):
    text: str


# -- What the tool actually received ---------------------------------------

tool_calls_seen: list[dict] = []


class StockLookupTool:
    name = "check_stock"

    def invoke(self, args, config=None, **kwargs):
        tool_calls_seen.append(dict(args))
        return f"warehouse {args['warehouse_id']}: 14 units on hand"


# -- A model that guesses ---------------------------------------------------

# Module-level so the count survives bind_tools(), which the cycle calls once
# per turn and which must return a bound clone.
turns = {"n": 0}


class GuessingLLM:
    """Queries warehouse 1. The audit is running in warehouse 4822.

    Not a strawman -- a model fills in an argument it was never given a way to
    know, and a plausible-looking integer is the most likely thing it writes.
    """

    def bind_tools(self, tools):
        return GuessingLLM()

    def invoke(self, messages, **kwargs):
        turns["n"] += 1
        if turns["n"] == 1:
            msg = AIMessage(content="")
            msg.tool_calls = [
                {
                    "name": "check_stock",
                    "args": {"warehouse_id": 1, "include_reserved": True},
                    "id": "c1",
                }
            ]
            return msg
        return AIMessage(content=Report(text="audit complete").model_dump_json())

    def with_structured_output(self, model):
        return self


def llm_factory(tier):
    return GuessingLLM()


# -- The pipeline ----------------------------------------------------------


@node(outputs=Audit)
def audit() -> Audit:
    return Audit(warehouse_id=4822, region="eu-west")


@node(outputs=Discrepancies)
def discrepancies() -> Discrepancies:
    return Discrepancies(items=(Discrepancy(sku="SKU-114"), Discrepancy(sku="SKU-330")))


@node(outputs=Classified, map_over="discrepancies.items", map_key="sku")
def classify(item: Discrepancy, audit: Audit) -> Classified:
    """ROUTE 2 -- a scripted fan-out branch.

    `item` is the fanned value; `audit` is an ordinary upstream read. A scripted
    node needs no special mechanism: the port carries WHICH ITEM and a second
    fan-in key carries WHICH RUN. Both are type-checked by the validator, which
    is why this is the route to prefer when the node runs Python.
    """
    return Classified(label=f"{item.sku} [warehouse {audit.warehouse_id}/{audit.region}]")


# ROUTE 1 + 3 -- an LLM node that both SEES the run context and cannot
# misreport it to a tool. The two declarations sit side by side on purpose:
# one is what the model is shown, the other is what it cannot change.
@node(
    mode="agent",
    outputs=Report,
    model="fast",
    prompt="audit",
    # ROUTE 1: the model SEES the run context. Declared, so assembly fails if
    # nothing upstream produces it.
    context=["audit"],
    tools=[
        Tool(
            name="check_stock",
            budget=3,
            # ROUTE 3: the framework SUPPLIES this argument. Whatever the model
            # emits for warehouse_id is overwritten from run state.
            # `include_reserved` is not named here, so it stays exactly as the
            # model composed it.
            bound_args={"warehouse_id": "audit.warehouse_id"},
        )
    ],
)
def summarize(discrepancies: Discrepancies) -> Report:
    # body unused for mode='agent' -- the LLM drives the tool loop
    ...


pipeline = construct_from_module(sys.modules[__name__], name="run-scoped-state")


# -- Run --------------------------------------------------------------------

if __name__ == "__main__":
    graph = compile(
        pipeline,
        llm_factory=llm_factory,
        prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": "audit"}],
        tool_factories={"check_stock": lambda config, tool_config: StockLookupTool()},
    )
    result = run(graph, input={"node_id": "AUDIT-1"})

    print("Route 2 -- scripted fan-out branches, each reading the run context:")
    for key, value in result["classify"].items():
        print(f"  {key}: {value.label}")

    args = tool_calls_seen[0]
    print("\nRoute 3 -- what the model asked for vs what the tool received:")
    print("  model emitted   : warehouse_id=1        <- a warehouse this audit is not in")
    print(f"  tool received   : warehouse_id={args['warehouse_id']}     <- supplied from run state")
    print(f"  left untouched  : include_reserved={args['include_reserved']}  <- not bound, still the model's")

    assert args["warehouse_id"] == 4822, "the model's invented warehouse reached the tool"
    assert args["include_reserved"] is True, "binding one argument swallowed another"

    print("\nWithout bound_args the first line would have been the tool call.")
    print("Stock in the wrong warehouse reads as zero units, which is")
    print("indistinguishable from a genuine stockout -- so the run completes,")
    print("every gate stays green, and the report is about a shelf nobody looked at.")
