"""The dotted port address, which is the remedy every later refusal will name.

It did not parse before neograph-9axw6.2: the whole string was compared against
member names, so "settle.result" matched nothing and was refused as an unknown
member. Design 6.2 documents this spelling, and design 14 puts that step first
precisely so the instruction works before the refusals ship.
"""

from pydantic import BaseModel

from neograph import Construct, Node
from neograph._runtime_registry import register_scripted


class Seed(BaseModel, frozen=True):
    tag: str = "s"


class Case(BaseModel, frozen=True):
    label: str = "L"


register_scripted("npda_multi", lambda _i, _c: {"result": Case(), "extra": Seed()})

pipeline = Construct(
    "boundary",
    input=Seed,
    output=Case,
    output_from="settle.result",
    nodes=[
        Node.scripted("settle", fn="npda_multi", inputs=Seed, outputs={"result": Case, "extra": Seed}),
    ],
)
