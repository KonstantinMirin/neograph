# CHECK_ERROR: type.*compatible|no upstream produces
# Attack vector 3b: X output connected to list[X] consumer.
# Claims is not list[Claims].
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class Claims(BaseModel, frozen=True):
    text: str

register_scripted("elem_out_3b", lambda i, c: Claims(text="ok"))
register_scripted("list_in_3b", lambda i, c: [Claims(text="ok")])

pipeline = Construct("broken", nodes=[
    Node.scripted("first", fn="elem_out_3b", outputs=Claims),
    Node.scripted("second", fn="list_in_3b", inputs=list[Claims], outputs=list[Claims]),
])
