# Valid: skip_when with bad field compiles fine. Error is caught at runtime
# with a clear ExecutionError (neograph-hr4c).
from neograph import Construct, Node
from neograph.factory import register_scripted
from pydantic import BaseModel


class Claims(BaseModel, frozen=True):
    items: list[str]


class Result(BaseModel, frozen=True):
    text: str


register_scripted("swbf_proc", lambda i, c: Result(text="ok"))

pipeline = Construct("broken", nodes=[
    Node.scripted("extract", fn="swbf_proc", outputs=Claims),
    Node(
        "classify",
        mode="scripted",
        scripted_fn="swbf_proc",
        inputs=Claims,
        outputs=Result,
        skip_when=lambda data: data.nonexistent_field > 0,
    ),
])
