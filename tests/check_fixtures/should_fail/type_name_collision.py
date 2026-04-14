# CHECK_ERROR: collision|both map to
# Attack vector 7: Two nodes with same normalized name (hyphen vs underscore)
# "my-node" and "my_node" both normalize to state field "my_node"
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class TypeA(BaseModel, frozen=True):
    x: str

register_scripted("nc_a", lambda i, c: TypeA(x="hello"))
register_scripted("nc_b", lambda i, c: TypeA(x="world"))

pipeline = Construct("broken", nodes=[
    Node.scripted("my-node", fn="nc_a", outputs=TypeA),
    Node.scripted("my_node", fn="nc_b", outputs=TypeA),
])
