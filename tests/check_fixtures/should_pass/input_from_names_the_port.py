"""``input_from`` names which eligible producer a single-type input reads.

The input-side twin of ``output_from``. Before neograph-9axw6.5 the two directions
were asymmetric: disambiguating an output meant ADDING A NAME, while disambiguating
an input meant REWRITING ``inputs=Alpha`` into ``inputs={"first": Alpha}``. Those are
different asks, and an author reading an error does the cheaper one.

Both members produce ``Alpha``, so declaration order would pick ``second``. This
construct says ``first`` and means it.
"""

from pydantic import BaseModel

from neograph import Construct, Node
from neograph._runtime_registry import register_scripted


class Alpha(BaseModel, frozen=True):
    tag: str = "a"


register_scripted("if_a", lambda _i, _c: Alpha(tag="FIRST"))
register_scripted("if_b", lambda _i, _c: Alpha(tag="SECOND"))
register_scripted("if_sink", lambda input_data, _c: Alpha(tag=f"saw-{input_data.tag}"))

pipeline = Construct(
    "named-input",
    nodes=[
        Node.scripted("first", fn="if_a", outputs=Alpha),
        Node.scripted("second", fn="if_b", inputs=Alpha, outputs=Alpha),
        Node(
            name="sink",
            mode="scripted",
            scripted_fn="if_sink",
            inputs=Alpha,
            outputs=Alpha,
            input_from="first",
        ),
    ],
)
