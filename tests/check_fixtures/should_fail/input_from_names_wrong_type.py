# CHECK_ERROR: declares input_from='beta_maker', which produces Beta
"""Naming a REAL member is not enough: it must produce the declared type.

An existence-only check would accept this and hand the body a ``Beta`` where it
declared ``Alpha``. ``check_output_from``'s twin verifies both halves, so the two
directions of the same authored reference are now symmetric.
"""

from pydantic import BaseModel

from neograph import Construct, Node
from tests.fakes import register_scripted


class Alpha(BaseModel, frozen=True):
    tag: str = "a"


class Beta(BaseModel, frozen=True):
    other: str = "b"


register_scripted("ifw_beta", lambda _i, _c: Beta())
register_scripted("ifw_alpha", lambda _i, _c: Alpha(tag="RIGHT-TYPE"))
register_scripted("ifw_sink", lambda input_data, _c: Alpha(tag=f"saw-{input_data.tag}"))

pipeline = Construct(
    "input-from-wrong-type",
    nodes=[
        Node.scripted("beta_maker", fn="ifw_beta", outputs=Beta),
        Node.scripted("alpha_maker", fn="ifw_alpha", outputs=Alpha),
        Node(
            name="sink",
            mode="scripted",
            scripted_fn="ifw_sink",
            inputs=Alpha,
            outputs=Alpha,
            input_from="beta_maker",
        ),
    ],
)
