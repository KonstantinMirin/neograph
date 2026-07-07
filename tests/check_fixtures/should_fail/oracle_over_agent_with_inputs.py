# CHECK_ERROR: Oracle over an agent/act node with upstream inputs is not supported
# neograph-m6d3.6 + neograph-qot6: the auto-wrap synthesizes an EMPTY input port,
# so it only supports a SELF-CONTAINED agent. An agent that consumes upstream
# inputs needs port synthesis (open design Q), so it fails loud at assembly.
from pydantic import BaseModel

from neograph import Construct, Node, Oracle, Tool


class Result(BaseModel, frozen=True):
    text: str


pipeline = Construct("oracle-over-agent-inputs", nodes=[
    Node(
        name="agent-gen",
        mode="agent",
        inputs={"upstream": Result},
        outputs=Result,
        model="fast",
        prompt="test",
        tools=[Tool(name="search", budget=3)],
    )
    | Oracle(n=2, merge_fn="some_merge"),
])
