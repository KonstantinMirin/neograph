# CHECK_ERROR: output_from='settle_typo'[\s\S]*matches no item
"""GH #17: output_from must name an item THIS construct declares.

A typo must refuse at assembly, not resolve to whatever the type-scan would have
picked -- silently picking one is the bug output_from exists to prevent.
"""

from pydantic import BaseModel

from neograph import Construct, Node


class Seed(BaseModel, frozen=True):
    tag: str


class Case(BaseModel, frozen=True):
    label: str


branch = Construct(
    "branch",
    input=Seed,
    output=Case,
    output_from="settle_typo",
    nodes=[Node.scripted("settle", fn="settle_fn", inputs=Seed, outputs=Case)],
)
