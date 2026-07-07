# CHECK_ERROR: Loop over an agent/act node is not supported
# neograph-m6d3.6 + neograph-gk3e: fan-over-agent auto-wrap supports ONLY Oracle
# over a self-contained agent. Loop-over-agent needs a design call (the loop
# condition reading the wrapped subgraph output), so it fails loud at assembly.
from pydantic import BaseModel

from neograph import Construct, Loop, Node, Tool


class Result(BaseModel, frozen=True):
    text: str


pipeline = Construct("loop-over-agent", nodes=[
    Node(
        name="agent-gen",
        mode="agent",
        outputs=Result,
        model="fast",
        prompt="test",
        tools=[Tool(name="search", budget=3)],
    )
    | Loop(when="never", max_iterations=3),
])
