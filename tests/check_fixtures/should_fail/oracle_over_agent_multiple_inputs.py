# CHECK_ERROR: multiple upstream inputs
# neograph-qot6 implemented input-port synthesis for a SINGLE upstream producer,
# but the single-value subgraph boundary (neo_subgraph_input) cannot carry N
# distinct producers. Fan-in from MULTIPLE producers stays fail-loud until the
# bundle-port synthesis lands (neograph-qzrv). Fail loud at assembly, never a
# broken graph.
from pydantic import BaseModel

from neograph import Construct, Node, Oracle, Tool
from tests.fakes import register_scripted, register_tool_factory


class Alpha(BaseModel, frozen=True):
    a: str


class Beta(BaseModel, frozen=True):
    b: str


register_scripted("alpha_fn", lambda input_data, config: Alpha(a="x"))
register_scripted("beta_fn", lambda input_data, config: Beta(b="y"))
register_scripted("some_merge", lambda variants, config: variants[0])
register_tool_factory("search", lambda config, tool_config: None)


pipeline = Construct("oracle-over-agent-multi-inputs", nodes=[
    Node.scripted("alpha", fn="alpha_fn", outputs=Alpha),
    Node.scripted("beta", fn="beta_fn", outputs=Beta),
    Node(
        name="agent-gen",
        mode="agent",
        inputs={"alpha": Alpha, "beta": Beta},  # two distinct producers
        outputs=Alpha,
        model="fast",
        prompt="test",
        tools=[Tool(name="search", budget=3)],
    )
    | Oracle(n=2, merge_fn="some_merge"),
])
