# CHECK_ERROR: type.*compatible|no upstream produces
# Attack vector 2: str output connected to a BaseModel consumer
# str is not a BaseModel subclass, should fail type check.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class SomeModel(BaseModel, frozen=True):
    text: str

register_scripted("str_out", lambda i, c: "hello")
register_scripted("model_in", lambda i, c: SomeModel(text="ok"))

pipeline = Construct("broken", nodes=[
    Node.scripted("first", fn="str_out", outputs=str),
    Node.scripted("second", fn="model_in", inputs=SomeModel, outputs=SomeModel),
])
