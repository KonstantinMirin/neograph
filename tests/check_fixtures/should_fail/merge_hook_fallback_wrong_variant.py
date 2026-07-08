# CHECK_ERROR: merge_fallback.*type mismatch|declared list\[ModelB\]
# Attack vector: merge_fallback annotated with wrong variant type.

from pydantic import BaseModel

from neograph import Construct, Node, Oracle


class ModelA(BaseModel, frozen=True):
    value: str


class ModelB(BaseModel, frozen=True):
    score: float


def bad_fb(variants: list[ModelB], error: Exception) -> ModelA:
    return ModelA(value="fb")


pipeline = Construct(
    "test",
    nodes=[
        Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast")
        | Oracle(n=2, merge_prompt="merge: ${variants}", merge_fallback=bad_fb),
    ],
)
