# CHECK_ERROR: declares input_from='nosuch', which names no producer visible here
"""``input_from`` is an authored name, and a name that resolves to nothing is refused.

The defect it replaces was not a missing value but a DISPLACED one: the resolver
stamped whatever the spelling said, so ``nosuch`` addressed a field nobody writes
while ``producer`` sat there producing exactly the declared type, and the body was
handed ``None`` on a green run.
"""

from pydantic import BaseModel

from neograph import Construct, Node
from tests.fakes import register_scripted


class Alpha(BaseModel, frozen=True):
    tag: str = "a"


register_scripted("ifn_producer", lambda _i, _c: Alpha(tag="REAL"))
register_scripted("ifn_sink", lambda input_data, _c: Alpha(tag=f"saw-{input_data.tag}"))

pipeline = Construct(
    "input-from-names-nothing",
    nodes=[
        Node.scripted("producer", fn="ifn_producer", outputs=Alpha),
        Node(
            name="sink",
            mode="scripted",
            scripted_fn="ifn_sink",
            inputs=Alpha,
            outputs=Alpha,
            input_from="nosuch",
        ),
    ],
)
