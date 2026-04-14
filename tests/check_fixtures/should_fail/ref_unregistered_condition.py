# CHECK_ERROR: Condition.*not registered
# NOTE: Operator requires a checkpointer. The fixture harness doesn't pass one,
# so it hits the checkpointer check first. This fixture validates via direct
# compile() with a checkpointer in the inline test below.
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from neograph import Construct, Node, compile
from neograph.factory import register_scripted
from neograph.modifiers import Operator


class Result(BaseModel, frozen=True):
    text: str


register_scripted("ref_cond_fn", lambda i, c: Result(text="ok"))

n = Node.scripted("check", fn="ref_cond_fn", outputs=Result) | Operator(when="ghost_condition")
pipeline = Construct("broken", nodes=[n])

# Directly compile with checkpointer to bypass the checkpointer gate.
compile(pipeline, checkpointer=MemorySaver())
