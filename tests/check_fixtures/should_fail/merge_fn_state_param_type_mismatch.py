# CHECK_ERROR: type.*mismatch|not compatible|expects.*Context.*produces.*Claims|wrong type
# Attack vector: @merge_fn has a bare typed param whose name matches an
# upstream node, but the type annotation doesn't match what that node
# produces. State auto-wiring should catch the type incompatibility.
from pydantic import BaseModel

from neograph import merge_fn, node
from neograph.decorators import construct_from_functions


class Claims(BaseModel, frozen=True):
    items: list[str]


class Context(BaseModel, frozen=True):
    topic: str


@node(outputs=Claims)
def prepare() -> Claims: ...


@merge_fn
def combine(variants: list[Claims], prepare: Context) -> Claims:
    # 'prepare' matches upstream node name, but upstream produces Claims
    # while the annotation here says Context. Type mismatch.
    return variants[0]


@node(outputs=Claims, ensemble_n=2, merge_fn="combine")
def generate(prepare: Claims) -> Claims: ...


pipeline = construct_from_functions("type-mismatch", [prepare, generate])
