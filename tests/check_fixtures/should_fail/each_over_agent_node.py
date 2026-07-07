# CHECK_ERROR: Each over an agent/act node is not supported
# neograph-m6d3.6 + neograph-1h8c: fan-over-agent auto-wrap supports ONLY Oracle
# over a self-contained agent. Each-over-agent needs a design call (delivering the
# fanned neo_each_item across the subgraph boundary), so it fails loud at assembly.
from pydantic import BaseModel

from neograph import Construct, Each, Node, Tool


class Result(BaseModel, frozen=True):
    text: str


pipeline = Construct("each-over-agent", nodes=[
    Node(
        name="agent-gen",
        mode="agent",
        outputs=Result,
        model="fast",
        prompt="test",
        tools=[Tool(name="search", budget=3)],
    )
    | Each(over="upstream", key="text"),
])
