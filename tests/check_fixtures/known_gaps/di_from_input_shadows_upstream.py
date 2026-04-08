# CHECK_ERROR: shadow|ambig|conflict|upstream
# Attack vector: FromInput param has the same name as an upstream @node.
# The classifier marks 'produce' as from_input, so it gets excluded from
# the inputs dict (line ~639 in decorators.py: "if p.name in param_res: continue").
# This silently breaks the dependency edge — 'consume' no longer depends on
# 'produce', so the topological sort may run them in wrong order or the
# validator may not verify the type compatibility.
from typing import Annotated

from pydantic import BaseModel

from neograph import FromInput, node
from neograph.decorators import construct_from_functions


class Data(BaseModel, frozen=True):
    value: str


class Output(BaseModel, frozen=True):
    result: str


@node(outputs=Data)
def produce() -> Data: ...


@node(outputs=Output)
def consume(produce: Annotated[Data, FromInput]) -> Output: ...


pipeline = construct_from_functions("shadow", [produce, consume])
