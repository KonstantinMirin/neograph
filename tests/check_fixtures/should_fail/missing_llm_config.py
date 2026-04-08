# CHECK_ERROR: LLM nodes but.*not set.*configure_llm
from neograph import Construct, Node
from pydantic import BaseModel

class Result(BaseModel, frozen=True):
    text: str

pipeline = Construct("broken", nodes=[
    Node(name="gen", mode="think", outputs=Result, model="fast", prompt="test"),
])
