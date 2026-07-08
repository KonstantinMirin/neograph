# CHECK_ERROR: not registered|Condition.*not registered
# Operator.when with a condition name that was never registered.
# Uses a fake checkpointer to get past the checkpointer guard.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.modifiers import Operator
from tests.fakes import register_scripted


class Result(BaseModel, frozen=True):
    text: str


register_scripted("mouc_proc", lambda i, c: Result(text="ok"))


class FakeCheckpointer:
    """Minimal checkpointer stub to get past the compile guard."""

    pass


pipeline = Construct(
    "broken",
    nodes=[
        Node.scripted("proc", fn="mouc_proc", outputs=Result) | Operator(when="this_condition_does_not_exist"),
    ],
)
