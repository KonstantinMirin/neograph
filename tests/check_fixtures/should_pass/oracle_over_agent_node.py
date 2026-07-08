# Oracle over a SELF-CONTAINED agent/act node compiles cleanly via the auto-wrap
# (neograph-m6d3.6): the multi-node ReAct cycle is isolated in a single-node
# sub-construct and Oracle fans over it through the existing subgraph path, so
# each variant runs with its own message/budget channels. This was a should_fail
# fixture under the m6d3.2 blanket compile error; the principled fix (auto-wrap)
# replaced that error, so it now belongs in should_pass. See
# docs/design/fan-over-agent-node-2026-07-07.md.
from pydantic import BaseModel

from neograph import Construct, Node, Oracle, Tool
from tests.fakes import register_scripted, register_tool_factory


class Result(BaseModel, frozen=True):
    text: str


register_scripted("some_merge", lambda variants, config: variants[0])
register_tool_factory("search", lambda config, tool_config: None)


pipeline = Construct(
    "oracle-over-agent",
    nodes=[
        Node(
            name="agent-gen",
            mode="agent",
            outputs=Result,
            model="fast",
            prompt="test",
            tools=[Tool(name="search", budget=3)],
        )
        | Oracle(n=2, merge_fn="some_merge"),
    ],
)
