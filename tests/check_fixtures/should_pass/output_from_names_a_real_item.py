"""GH #17: output_from naming a declared item compiles cleanly.

The positive half of the boundary rule. Also covers the common case needing NO
declaration at all: the default already prefers the last declared item producing
the type, so `passthrough` below resolves without output_from.
"""

from pydantic import BaseModel

from neograph import Construct, Node
from tests.fakes import register_scripted


class Seed(BaseModel, frozen=True):
    tag: str


class Case(BaseModel, frozen=True):
    label: str


register_scripted("ofnri_settle", lambda i, c: Case(label="settled"))


explicit = Construct(
    "explicit",
    input=Seed,
    output=Case,
    output_from="settle",
    nodes=[Node.scripted("settle", fn="ofnri_settle", inputs=Seed, outputs=Case)],
)

inferred = Construct(
    "inferred",
    input=Seed,
    output=Case,
    nodes=[Node.scripted("passthrough", fn="ofnri_settle", inputs=Seed, outputs=Case)],
)
