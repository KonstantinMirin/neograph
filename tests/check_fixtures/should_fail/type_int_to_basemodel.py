# CHECK_ERROR: type.*compatible|no upstream produces
# Attack vector 2b: int output connected to BaseModel consumer
# int is not a BaseModel subclass, should fail type check.
from neograph import Construct, Node
from neograph.factory import register_scripted
from pydantic import BaseModel

class SomeModel(BaseModel, frozen=True):
    text: str

register_scripted("int_out_2b", lambda i, c: 42)
register_scripted("model_in_2b", lambda i, c: SomeModel(text="ok"))

pipeline = Construct("broken", nodes=[
    Node.scripted("first", fn="int_out_2b", outputs=int),
    Node.scripted("second", fn="model_in_2b", inputs=SomeModel, outputs=SomeModel),
])
