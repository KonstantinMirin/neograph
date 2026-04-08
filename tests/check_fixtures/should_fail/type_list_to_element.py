# CHECK_ERROR: type.*compatible|no upstream produces
# Attack vector 3: list[X] output connected to X consumer (not Each, plain list)
# list[Claims] is not Claims, should fail.
from neograph import Construct, Node
from neograph.factory import register_scripted
from pydantic import BaseModel

class Claims(BaseModel, frozen=True):
    text: str

register_scripted("list_out", lambda i, c: [Claims(text="a")])
register_scripted("elem_in", lambda i, c: Claims(text="ok"))

pipeline = Construct("broken", nodes=[
    Node.scripted("first", fn="list_out", outputs=list[Claims]),
    Node.scripted("second", fn="elem_in", inputs=Claims, outputs=Claims),
])
