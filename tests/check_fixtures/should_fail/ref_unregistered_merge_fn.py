# CHECK_ERROR: Scripted function 'nonexistent_merge' not registered
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.modifiers import Oracle


class Result(BaseModel, frozen=True):
    text: str


pipeline = Construct(
    "broken",
    nodes=[
        Node(name="gen", mode="think", outputs=Result, model="fast", prompt="test")
        | Oracle(n=2, merge_fn="nonexistent_merge"),
    ],
)
