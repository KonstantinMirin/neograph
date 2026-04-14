# KNOWN GAP: Claims producer should satisfy Optional[Claims] consumer,
# but _types_compatible(Claims, Optional[Claims]) returns False because
# Optional[Claims] is Union[Claims, None] and is not isinstance(type).
# The validator falls through to the issubclass check which fails.
# This means Optional[X] as an input type is silently rejected at assembly time.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class Claims(BaseModel, frozen=True):
    text: str

register_scripted("opt_c_a", lambda i, c: Claims(text="ok"))
register_scripted("opt_c_b", lambda i, c: Claims(text="done"))

# This SHOULD pass (Claims satisfies Claims | None) but currently
# raises ConstructError because _types_compatible returns False.
pipeline = Construct("should-work", nodes=[
    Node.scripted("first", fn="opt_c_a", outputs=Claims),
    Node.scripted("second", fn="opt_c_b", inputs=Claims | None, outputs=Claims),
])
