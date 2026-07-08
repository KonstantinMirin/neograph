# CHECK_ERROR: merge_pre_process requires 1 positional parameter
# Attack vector: merge_pre_process with no params at all.

from pydantic import BaseModel

from neograph import Construct, Node, Oracle


class ModelA(BaseModel, frozen=True):
    value: str


def bad() -> dict:
    return {}


pipeline = Construct(
    "test",
    nodes=[
        Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast")
        | Oracle(n=2, merge_prompt="merge: ${variants}", merge_pre_process=bad),
    ],
)
