# Valid: @merge_fn param has BOTH a state-matching name AND a FromInput marker.
# The DI marker should take precedence — the param is resolved from
# config["configurable"], not auto-wired from state. No conflict.
from typing import Annotated

from pydantic import BaseModel

from neograph import FromInput, merge_fn, node
from neograph.decorators import construct_from_functions


class Claims(BaseModel, frozen=True):
    items: list[str]


@node(outputs=Claims)
def prepare() -> Claims: ...


@merge_fn
def combine(
    variants: list[Claims],
    prepare: Annotated[str, FromInput],  # name clashes with upstream 'prepare' node
) -> Claims:
    # DI marker wins: 'prepare' reads from config, not from state.
    return variants[0]


@node(outputs=Claims, ensemble_n=2, merge_fn="combine")
def generate(prepare: Claims) -> Claims: ...


pipeline = construct_from_functions("di-wins", [prepare, generate])
