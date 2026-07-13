# CHECK_ERROR: stray
# A Literal-typed route member not in peers∪{HANDOFF_END} (design §5.4).
from typing import Literal

from pydantic import BaseModel

from neograph import Construct, Keymaker, Node
from tests.fakes import register_scripted


class Handoff(BaseModel, frozen=True):
    goto: Literal["billing", "stray", "__end__"]


register_scripted("km_f", lambda i, c: Handoff(goto="__end__"))

pipeline = Construct(
    "swarm",
    nodes=[
        Node.scripted("triage", fn="km_f", outputs=Handoff) | Keymaker(peers=["billing"]),
        Node.scripted("billing", fn="km_f", inputs={"handoff": Handoff}, outputs=Handoff) | Keymaker(peers=["triage"]),
    ],
)
