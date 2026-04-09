# CHECK_ERROR: no.*producer|does not exist|not found|unknown.*field|no upstream
# Attack vector: @merge_fn has a bare typed param whose name doesn't match
# any node in the pipeline. State auto-wiring should fail at compile time
# because there is no state field to read from.
from pydantic import BaseModel

from neograph import merge_fn, node
from neograph.decorators import construct_from_functions


class Claims(BaseModel, frozen=True):
    items: list[str]


class Context(BaseModel, frozen=True):
    topic: str


@node(outputs=Context)
def setup() -> Context: ...


@merge_fn
def combine(variants: list[Claims], ghost: Context) -> Claims:
    # 'ghost' doesn't match any node name in the pipeline — no producer.
    return variants[0]


@node(outputs=Claims, ensemble_n=2, merge_fn="combine")
def generate(setup: Context) -> Claims: ...


pipeline = construct_from_functions("nonexistent", [setup, generate])
