# CHECK_ERROR: merge_post_process.*return type mismatch|declared ModelB
# Attack vector: merge_post_process returns wrong type.
# At runtime the LangGraph state write would silently accept the wrong
# type (no validation), but downstream consumers see the wrong model.

from pydantic import BaseModel

from neograph import Construct, Node, Oracle


class ModelA(BaseModel, frozen=True):
    value: str

class ModelB(BaseModel, frozen=True):
    score: float


def bad_post(result: ModelA, variants: list[ModelA]) -> ModelB:
    return ModelB(score=0.5)


pipeline = Construct("test", nodes=[
    Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast")
    | Oracle(n=2, merge_prompt="merge: ${variants}", merge_post_process=bad_post),
])
