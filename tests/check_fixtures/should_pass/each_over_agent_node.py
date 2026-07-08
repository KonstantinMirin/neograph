# Each over an agent/act node compiles cleanly via auto-wrap (neograph-1h8c).
# The agent's ReAct cycle is isolated in a single-node sub-construct and Each fans
# over THAT via the existing subgraph path. Design call (2026-07-07): the fanned
# neo_each_item is delivered AS the sub-construct's single-value input port
# (neo_each_item -> neo_subgraph_input), mirroring the qot6 single-key dict-form
# rewrite, so each isolated cycle reads its OWN per-branch item value. The agent
# must declare a consumed input (here single-type Item) — a self-contained agent
# has no port for the item and stays fail-loud. Was a should_fail fixture while the
# item-delivery design was open (m6d3.6); 1h8c implemented it.
# See docs/design/fan-over-agent-node-2026-07-07.md.
from pydantic import BaseModel

from neograph import Construct, Each, Node, Tool
from tests.fakes import register_scripted, register_tool_factory


class Item(BaseModel, frozen=True):
    text: str


class Batch(BaseModel, frozen=True):
    items: list[Item]


class Result(BaseModel, frozen=True):
    text: str


register_scripted("batch_fn", lambda input_data, config: Batch(items=[Item(text="a")]))
register_tool_factory("search", lambda config, tool_config: None)


pipeline = Construct(
    "each-over-agent",
    nodes=[
        Node.scripted("make-batch", fn="batch_fn", outputs=Batch),
        Node(
            name="agent-gen",
            mode="agent",
            inputs=Item,  # single-value input port: receives the fanned Each item
            outputs=Result,
            model="fast",
            prompt="test",
            tools=[Tool(name="search", budget=3)],
        )
        | Each(over="make_batch.items", key="text"),
    ],
)
