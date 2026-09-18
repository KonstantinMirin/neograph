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


register_scripted("npda_seed", lambda _i, _c: Seed())
register_scripted("npda_multi", lambda _i, _c: {"result": Case(), "extra": Seed()})

boundary = Construct(
    "boundary",
    input=Seed,
    output=Case,
    output_from="settle.result",
    nodes=[
        Node.scripted("settle", fn="npda_multi", inputs=Seed, outputs={"result": Case, "extra": Seed}),
    ],
)

# A parent, because a boundary only resolves when something CONSUMES it: run
# standalone this construct reports settle's own per-key state fields and the
# dotted port is never exercised (neograph-36302).
pipeline = Construct(
    "dotted-parent",
    nodes=[
        Node.scripted("seed", fn="npda_seed", outputs=Seed),
        boundary,
    ],
)

# The claim, made assertable: "settle.result" addresses ONE of the node's two
# dict-form output keys, and that key's value is what crosses the boundary.
EXPECT = {"boundary": Case(label="L")}
