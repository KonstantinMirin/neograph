# CHECK_ERROR: no upstream produces a compatible value
"""``inputs=dict`` is refused when nothing in scope writes a mapping.

It used to return early from validation, commented as deferring to a runtime
isinstance scan -- which had nothing to scan: no field held a mapping, so the body
was handed ``None`` on a green run. The shape is still legal where a producer really
writes one (an ``Each``-modified node writes ``dict[str, X]``, and consuming the
whole fan with ``inputs=dict`` is the documented spelling). What is refused is the
DECLARATION NO PRODUCER CAN SATISFY, not the spelling.
"""

from pydantic import BaseModel

from neograph import Construct, Node
from tests.fakes import register_scripted


class Alpha(BaseModel, frozen=True):
    tag: str = "a"


register_scripted("dmp_maker", lambda _i, _c: Alpha(tag="A-MODEL"))
register_scripted("dmp_reader", lambda _input_data, _c: Alpha(tag="read"))

pipeline = Construct(
    "dict-input-no-mapping-producer",
    nodes=[
        Node.scripted("maker", fn="dmp_maker", outputs=Alpha),
        Node.scripted("reader", fn="dmp_reader", inputs=dict, outputs=Alpha),
    ],
)
