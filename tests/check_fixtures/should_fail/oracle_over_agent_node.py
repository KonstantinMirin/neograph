# CHECK_ERROR: Oracle over an agent/act node is not supported
# neograph-m6d3 (binding condition 3): an agent/act node compiles to a multi-node
# inline ReAct cycle; Oracle/Each/Loop fan to a single node and cannot wrap it.
# Deliberate compile-time error until the fan target is generalized (neograph-m6d3.6).
from pydantic import BaseModel

from neograph import Construct, Node, Oracle, Tool


class Result(BaseModel, frozen=True):
    text: str


pipeline = Construct("oracle-over-agent", nodes=[
    Node(
        name="agent-gen",
        mode="agent",
        outputs=Result,
        model="fast",
        prompt="test",
        tools=[Tool(name="search", budget=3)],
    )
    | Oracle(n=2, merge_fn="some_merge"),
])
