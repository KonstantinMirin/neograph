# Loop over an agent/act node compiles cleanly via auto-wrap (neograph-gk3e).
# The agent's ReAct cycle is isolated in a single-node sub-construct and Loop wires
# a conditional back-edge over THAT via the existing subgraph-loop path
# (_add_subgraph_loop). Design call (2026-07-07): the loop `when` condition reads
# the agent's typed output SURFACED onto the parent field (the sub-construct output
# boundary), not an internal node output field; re-entry feeds the prior typed
# output back as the sub-construct input port (refine pattern). Was a should_fail
# fixture while the design was open (m6d3.6); gk3e implemented it.
# See docs/design/fan-over-agent-node-2026-07-07.md.
from pydantic import BaseModel

from neograph import Construct, Loop, Node, Tool
from tests.fakes import register_scripted, register_tool_factory


class Draft(BaseModel, frozen=True):
    text: str
    score: float = 0.0


register_scripted("seed_fn", lambda input_data, config: Draft(text="seed"))
register_tool_factory("search", lambda config, tool_config: None)


pipeline = Construct(
    "loop-over-agent",
    nodes=[
        Node.scripted("seed", fn="seed_fn", outputs=Draft),
        Node(
            name="agent-gen",
            mode="agent",
            inputs=Draft,  # refine port: fed the prior typed output on re-entry
            outputs=Draft,
            model="fast",
            prompt="test",
            tools=[Tool(name="search", budget=3)],
        )
        # callable condition reads the agent's typed subgraph output (Draft.score)
        | Loop(when=lambda d: d is None or d.score < 0.8, max_iterations=3),
    ],
)
