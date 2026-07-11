# CHECK_ERROR: loop_history= requires loop_when=
# @node(loop_history=True) without loop_when= composes no Loop modifier at
# all, so the kwarg would be silently dead — the decorator fails loud at
# decoration time instead (neograph-d5pvl, same pairing rule as map_over/map_key).
from pydantic import BaseModel

from neograph import construct_from_functions, node


class Draft(BaseModel, frozen=True):
    text: str


@node(outputs=Draft, loop_history=True)
def draft() -> Draft:
    return Draft(text="v1")


pipeline = construct_from_functions("broken-loop-history", [draft])
