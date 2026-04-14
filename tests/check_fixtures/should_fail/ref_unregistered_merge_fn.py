# CHECK_ERROR: Scripted function.*not registered|merge_fn.*not
from pydantic import BaseModel

from neograph import Construct, Node
from neograph._llm import configure_llm
from neograph.modifiers import Oracle

configure_llm(
    llm_factory=lambda tier: None,
    prompt_compiler=lambda t, d: [{"role": "user", "content": "x"}],
)


class Result(BaseModel, frozen=True):
    text: str


pipeline = Construct("broken", nodes=[
    Node(name="gen", mode="think", outputs=Result, model="fast",
         prompt="test") | Oracle(n=2, merge_fn="nonexistent_merge"),
])
