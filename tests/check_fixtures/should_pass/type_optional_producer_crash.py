# KNOWN GAP: Optional[X] (typing.Union) as a producer type causes an
# unhandled TypeError crash in _types_compatible.
#
# Root cause: _types_compatible line 474 calls issubclass(producer_origin, target)
# where producer_origin is typing.Union (not a class). typing.Union is NOT
# types.UnionType — the PEP 604 form (X | Y) uses types.UnionType (a real class),
# but typing.Optional[X] and typing.Union[X, Y] use typing.Union (not a class).
#
# The issubclass() call raises TypeError which propagates uncaught because
# the try/except at line 494 only covers the final issubclass, not the one
# at line 474.
#
# Impact: any node with outputs=Optional[X] crashes assembly with a confusing
# TypeError instead of a clean ConstructError.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class Claims(BaseModel, frozen=True):
    text: str

register_scripted("opt_crash_a", lambda i, c: None)
register_scripted("opt_crash_b", lambda i, c: Claims(text="ok"))

pipeline = Construct("broken", nodes=[
    Node.scripted("first", fn="opt_crash_a", outputs=Claims | None),
    Node.scripted("second", fn="opt_crash_b", inputs=Claims, outputs=Claims),
])
