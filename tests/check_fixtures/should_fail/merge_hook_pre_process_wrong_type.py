# CHECK_ERROR: merge_pre_process.*type mismatch|declared list\[ModelB\]
# Attack vector: merge_pre_process annotated with wrong variant type.
# The real piarch-1kz2a bug: pre_process(variants: list[WrongType])
# silently filters all variants at runtime. Must be caught at assembly.

from pydantic import BaseModel

from neograph import Construct, Node, Oracle


class ModelA(BaseModel, frozen=True):
    value: str

class ModelB(BaseModel, frozen=True):
    score: float


def bad_pre(variants: list[ModelB]) -> dict:
    return {"items": variants}


pipeline = Construct("test", nodes=[
    Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast")
    | Oracle(n=2, merge_prompt="merge: ${variants}", merge_pre_process=bad_pre),
])
