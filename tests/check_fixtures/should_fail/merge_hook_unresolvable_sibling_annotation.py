# CHECK_ERROR: merge_pre_process variants param 'variants' type mismatch
# Regression (neograph-okli): the merge-hook type check must STILL fire when a
# SIBLING param of the hook carries an unresolvable annotation. The former
# all-or-nothing `get_type_hints()` swallow in _validation_modifiers skipped the
# WHOLE check on ANY unresolvable annotation — hiding a genuine variants-type
# mismatch (a silent false negative). Routing through `_hints.resolve_hints`
# degrades PER annotation: `variants` (list[ModelB]) still resolves and the
# mismatch against the Oracle's generated ModelA is caught; only the genuinely
# unresolvable `ctx` is dropped (with a debug breadcrumb).
from __future__ import annotations

from pydantic import BaseModel

from neograph import Construct, Node, Oracle


class ModelA(BaseModel, frozen=True):
    value: str = ""


class ModelB(BaseModel, frozen=True):
    score: float = 0.0


def bad_pre(variants: list[ModelB], ctx: UndefinedSiblingRef = None) -> dict:  # noqa: F821
    # `ctx`'s annotation is an undefined forward ref — unresolvable. The check
    # must not let that disable the `variants`-type verification.
    return {"items": variants}


pipeline = Construct(
    "bad-merge-hook",
    nodes=[
        Node("gen", mode="think", outputs=ModelA, prompt="gen", model="fast")
        | Oracle(n=2, merge_prompt="merge: ${variants}", merge_pre_process=bad_pre),
    ],
)
