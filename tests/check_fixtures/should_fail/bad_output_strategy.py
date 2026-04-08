# CHECK_ERROR: output_strategy='yolo'.*Valid values
from neograph import Construct, Node, configure_llm
from pydantic import BaseModel

configure_llm(
    llm_factory=lambda tier: None,
    prompt_compiler=lambda t, d: [{"role": "user", "content": "x"}],
)

class Result(BaseModel, frozen=True):
    text: str

pipeline = Construct("broken", nodes=[
    Node(name="gen", mode="think", outputs=Result, model="fast",
         prompt="test", llm_config={"output_strategy": "yolo"}),
])
