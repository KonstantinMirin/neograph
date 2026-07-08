# CHECK_ERROR: Oracle over an agent/act node with multi-output \(dict-form\) outputs is not supported
# neograph-qzrv design call: Oracle over an agent with dict-form (multi-output)
# OUTPUTS stays fail-loud. The isolating sub-construct has a SINGLE output boundary
# port, and an N-way merge of secondary outputs (e.g. tool_log) across fanned
# variants is undefined (the Oracle merge_fn contract is single-type). Keeping this
# fail-loud avoids silently dropping declared outputs. Use single-type outputs, or
# wrap the agent explicitly.
from pydantic import BaseModel

from neograph import Construct, Node, Oracle, Tool, ToolInteraction
from tests.fakes import register_scripted, register_tool_factory


class RawText(BaseModel, frozen=True):
    text: str


class Claims(BaseModel, frozen=True):
    items: list[str]


register_scripted("seed_fn", lambda input_data, config: RawText(text="seed"))
register_scripted("some_merge", lambda variants, config: variants[0])
register_tool_factory("search", lambda config, tool_config: None)


pipeline = Construct(
    "oracle-over-agent-multi-output",
    nodes=[
        Node.scripted("seed", fn="seed_fn", outputs=RawText),
        Node(
            name="agent-gen",
            mode="agent",
            inputs=RawText,
            outputs={"result": Claims, "tool_log": list[ToolInteraction]},  # dict-form
            model="fast",
            prompt="test",
            tools=[Tool(name="search", budget=3)],
        )
        | Oracle(n=2, merge_fn="some_merge"),
    ],
)
