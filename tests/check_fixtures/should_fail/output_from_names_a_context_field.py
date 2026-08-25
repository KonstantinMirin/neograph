# CHECK_ERROR: output_from='read'[\s\S]*matches no item
"""GH #17: a forwarded context= field is NOT an eligible boundary producer.

`read` names a real field in the child's state -- the value the construct was
HANDED -- and that is exactly what the reverse type-scan used to return. Naming it
must refuse rather than re-spell the original bug through the new field.
"""

from pydantic import BaseModel

from neograph import Construct, Node


class Seed(BaseModel, frozen=True):
    tag: str


class Case(BaseModel, frozen=True):
    label: str


_settle = Node.scripted("settle", fn="settle_fn", inputs=Seed, outputs=Case).model_copy(
    update={"context": ["read"]}
)

branch = Construct("branch", input=Seed, output=Case, output_from="read", nodes=[_settle])
