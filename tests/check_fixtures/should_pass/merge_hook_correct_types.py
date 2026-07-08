# Happy path: all three merge hooks with correct type annotations.
# Scripted node + Oracle with merge_prompt hooks. Assembly validates
# hook signatures; scripted mode compiles without configure_llm.

from pydantic import BaseModel

from neograph import Construct, Node, Oracle
from tests.fakes import register_scripted


class Claims(BaseModel, frozen=True):
    items: list[str]


def pre(variants: list[Claims]) -> dict:
    return {"items": variants}


def post(result: Claims, variants: list[Claims]) -> Claims:
    return result


def fb(variants: list[Claims], error: Exception) -> Claims:
    return Claims(items=[])


register_scripted("_chk_hook_gen", lambda i, c: Claims(items=["a"]))

pipeline = Construct(
    "hook-types-ok",
    nodes=[
        Node.scripted("gen", fn="_chk_hook_gen", outputs=Claims)
        | Oracle(
            n=2, merge_prompt="merge: ${variants}", merge_pre_process=pre, merge_post_process=post, merge_fallback=fb
        ),
    ],
)
