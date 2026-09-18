"""GH #17: output_from naming a declared item compiles cleanly, and is OBEYED.

The positive half of the boundary rule. Also covers the common case needing NO
declaration at all: the default already prefers the last declared item producing
the type, so `passthrough` below resolves without output_from.

The two sub-constructs sit under one parent because a boundary only resolves
when something CONSUMES it -- run standalone, each would report its member's own
state field and the port would never be exercised (neograph-36302).
"""

from pydantic import BaseModel

from neograph import Construct, Node
from tests.fakes import register_scripted


class Seed(BaseModel, frozen=True):
    tag: str


class Case(BaseModel, frozen=True):
    label: str


register_scripted("ofnri_seed", lambda i, c: Seed(tag="s"))
register_scripted("ofnri_settle", lambda i, c: Case(label="settled"))
register_scripted("ofnri_later", lambda i, c: Case(label="later"))


# Two members produce Case, so declaration order would pick `later`. This
# construct names `settle` and means it.
explicit = Construct(
    "explicit",
    input=Seed,
    output=Case,
    output_from="settle",
    nodes=[
        Node.scripted("settle", fn="ofnri_settle", inputs=Seed, outputs=Case),
        Node.scripted("later", fn="ofnri_later", inputs=Case, outputs=Case),
    ],
)

inferred = Construct(
    "inferred",
    input=Seed,
    output=Case,
    nodes=[Node.scripted("passthrough", fn="ofnri_settle", inputs=Seed, outputs=Case)],
)

# A Seed producer ahead of both, so each sub-construct's input port has an
# upstream to bind to.
pipeline = Construct(
    "boundaries",
    nodes=[Node.scripted("seed", fn="ofnri_seed", outputs=Seed), explicit, inferred],
)

EXPECT = {
    "explicit": Case(label="settled"),  # output_from beat declaration order
    "inferred": Case(label="settled"),  # the sole producer, resolved by the default
}
