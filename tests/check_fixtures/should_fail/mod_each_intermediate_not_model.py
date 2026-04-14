# CHECK_ERROR: does not resolve|no field|not.*model
# Each.over path where an intermediate segment is a plain type (str), not a Pydantic
# model. Path resolution should fail because str has no model_fields.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted
from neograph.modifiers import Each


class Outer(BaseModel, frozen=True):
    label: str


class Result(BaseModel, frozen=True):
    value: str


register_scripted("mei_outer", lambda i, c: Outer(label="test"))
register_scripted("mei_proc", lambda i, c: Result(value="ok"))

# over="outer.label.chars" — label is a str, not a model, so "chars" can't resolve
pipeline = Construct("broken", nodes=[
    Node.scripted("outer", fn="mei_outer", outputs=Outer),
    Node.scripted("proc", fn="mei_proc", inputs=str, outputs=Result)
    | Each(over="outer.label.chars", key="x"),
])
