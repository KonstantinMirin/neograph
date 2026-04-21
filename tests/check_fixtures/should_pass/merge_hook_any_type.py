# Happy path: list[Any] accepted for any output type.

from typing import Any

from pydantic import BaseModel

from neograph import Construct, Node, Oracle
from neograph.factory import register_scripted


class Claims(BaseModel, frozen=True):
    items: list[str]


def pre(variants: list[Any]) -> dict:
    return {"items": variants}


register_scripted("_chk_hook_any", lambda i, c: Claims(items=["a"]))

pipeline = Construct("hook-any-ok", nodes=[
    Node.scripted("gen", fn="_chk_hook_any", outputs=Claims)
    | Oracle(n=2, merge_prompt="merge: ${variants}", merge_pre_process=pre),
])
