# CHECK_ERROR: Scripted function.*not registered
from pydantic import BaseModel
from neograph import Construct, Node


class Result(BaseModel, frozen=True):
    text: str


pipeline = Construct("broken", nodes=[
    Node.scripted("do-stuff", fn="totally_nonexistent_fn", outputs=Result),
])
