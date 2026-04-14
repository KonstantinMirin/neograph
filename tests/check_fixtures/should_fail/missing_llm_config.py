# CHECK_ERROR: LLM nodes require configure_llm
from pydantic import BaseModel

from neograph import Construct, Node


class Result(BaseModel, frozen=True):
    text: str

pipeline = Construct("broken", nodes=[
    Node(name="gen", mode="think", outputs=Result, model="fast", prompt="test"),
])
