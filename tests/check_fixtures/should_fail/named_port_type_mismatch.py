# CHECK_ERROR: output_from='settle'[\s\S]*whose type cannot satisfy output=
"""neograph-x8i3s: a NAMED port whose type cannot satisfy output=.

The two-producer shape is load-bearing. With a single producer the construct is
refused anyway, for the WRONG reason ("no internal node produces a compatible
type"), so a one-producer fixture passes without the fix.
"""

from pydantic import BaseModel

from neograph import Construct, Node
from neograph._runtime_registry import register_scripted


class Seed(BaseModel, frozen=True):
    tag: str = "s"


class Case(BaseModel, frozen=True):
    label: str = "L"


register_scripted("nptm_case", lambda _i, _c: Case())
register_scripted("nptm_seed", lambda _i, _c: Seed())

pipeline = Construct(
    "boundary",
    input=Seed,
    output=Case,
    output_from="settle",
    nodes=[
        # This peer DOES satisfy output=Case, which is what the old type scan found.
        Node.scripted("other", fn="nptm_case", inputs=Seed, outputs=Case),
        # But THIS is the member the author named, and it produces Seed.
        Node.scripted("settle", fn="nptm_seed", inputs=Case, outputs=Seed),
    ],
)
