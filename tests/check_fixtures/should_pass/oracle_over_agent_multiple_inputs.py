# Oracle over an agent with MULTIPLE distinct dict-form producers compiles cleanly
# via packer-port synthesis (neograph-qzrv). The single-value subgraph boundary
# (neo_subgraph_input) carries one typed value, so a synthesized parent "packer"
# node bundles the N upstreams into one port model, and inner per-key "unpacker"
# nodes re-expose the ORIGINAL keys so the agent's dict-form fan-in (and prompt-var
# surface) is unchanged. Was a should_fail fixture while bundle-port synthesis was
# an open design question (qot6); qzrv implemented it for Oracle.
# See docs/design/fan-over-agent-node-2026-07-07.md.
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
        inputs={"alpha": Alpha, "beta": Beta},  # two distinct producers -> packer
        outputs=Alpha,
        model="fast",
        prompt="test",
        tools=[Tool(name="search", budget=3)],
    )
    | Oracle(n=2, merge_fn="some_merge"),
])
