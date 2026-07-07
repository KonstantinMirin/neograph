# CHECK_ERROR: Each over a self-contained agent/act node is not supported
# neograph-1h8c: Each-over-agent delivers the fanned item AS the wrapped
# sub-construct's single-value input port (neo_each_item -> neo_subgraph_input).
# A SELF-CONTAINED agent (inputs=None) declares no input, so the per-item value
# has nowhere to land — every isolated cycle would run on empty input and produce
# identical results keyed by distinct Each keys (a silent broken fan). Fail loud:
# require the agent to declare a consumed input.
from pydantic import BaseModel

from neograph import Construct, Each, Node, Tool


class Result(BaseModel, frozen=True):
    text: str


pipeline = Construct("each-over-self-contained-agent", nodes=[
    Node(
        name="agent-gen",
        mode="agent",
        outputs=Result,  # no inputs= — nothing to deliver the fanned item to
        model="fast",
        prompt="test",
        tools=[Tool(name="search", budget=3)],
    )
    | Each(over="upstream", key="text"),
])
