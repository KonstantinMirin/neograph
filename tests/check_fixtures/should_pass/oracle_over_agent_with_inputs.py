# Oracle over an agent/act node that consumes a SINGLE upstream producer compiles
# cleanly via input-port synthesis (neograph-qot6). The auto-wrap isolates the
# ReAct cycle in a single-node sub-construct and synthesizes the sub-construct's
# input= boundary port from the agent's single-key dict-form fan-in (here: the
# "seed" upstream of type Result). Was a should_fail fixture while port synthesis
# was an open design question (m6d3.6); qot6 implemented it, so it now belongs in
# should_pass. See docs/design/fan-over-agent-node-2026-07-07.md.
from pydantic import BaseModel

from neograph import Construct, Node, Oracle, Tool
from tests.fakes import register_scripted, register_tool_factory


class Result(BaseModel, frozen=True):
    text: str


register_scripted("seed_fn", lambda input_data, config: Result(text="seed"))
register_scripted("some_merge", lambda variants, config: variants[0])
register_tool_factory("search", lambda config, tool_config: None)


pipeline = Construct(
    "oracle-over-agent-inputs",
    nodes=[
        Node.scripted("seed", fn="seed_fn", outputs=Result),
        Node(
            name="agent-gen",
            mode="agent",
            inputs={"seed": Result},  # single-key dict-form fan-in: names the producer
            outputs=Result,
            model="fast",
            prompt="test",
            tools=[Tool(name="search", budget=3)],
        )
        | Oracle(n=2, merge_fn="some_merge"),
    ],
)
