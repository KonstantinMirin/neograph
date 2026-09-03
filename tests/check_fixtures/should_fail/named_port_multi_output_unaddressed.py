# CHECK_ERROR: output_from='settle'[\s\S]*names a member with several outputs
"""neograph-kgndo: output_from names a MEMBER, but a dict-form outputs= writes one
state field per key, so the member name does not say which VALUE crosses the
boundary. The remedy the message gives is the dotted address, which
tests/check_fixtures/should_pass/named_port_dotted_address.py exercises.
"""

from pydantic import BaseModel

from neograph import Construct, Node
from neograph._runtime_registry import register_scripted


class Seed(BaseModel, frozen=True):
    tag: str = "s"


class Case(BaseModel, frozen=True):
    label: str = "L"


register_scripted("npmo_multi", lambda _i, _c: {"result": Case(), "extra": Seed()})

pipeline = Construct(
    "boundary",
    input=Seed,
    output=Case,
    output_from="settle",
    nodes=[
        Node.scripted("settle", fn="npmo_multi", inputs=Seed, outputs={"result": Case, "extra": Seed}),
    ],
)
