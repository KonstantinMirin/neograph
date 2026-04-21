# Happy path: lambda hooks have no annotations — pass with arity-only check.

from pydantic import BaseModel

from neograph import Construct, Node, Oracle
from neograph.factory import register_scripted


class Claims(BaseModel, frozen=True):
    items: list[str]


register_scripted("_chk_hook_lam", lambda i, c: Claims(items=["a"]))

pipeline = Construct("hook-lambda-ok", nodes=[
    Node.scripted("gen", fn="_chk_hook_lam", outputs=Claims)
    | Oracle(n=2, merge_prompt="merge: ${variants}",
             merge_pre_process=lambda v: {"items": v},
             merge_post_process=lambda r, v: r,
             merge_fallback=lambda v, e: Claims(items=[])),
])
