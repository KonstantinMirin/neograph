# CHECK_ERROR: does_not_exist
# A Keymaker peer that isn't a sibling Node — unknown peer (design §5.1).
from pydantic import BaseModel

from neograph import Construct, Keymaker, Node
from tests.fakes import register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


register_scripted("km_f", lambda i, c: Handoff(goto="__end__"))

pipeline = Construct(
    "swarm",
    nodes=[
        Node.scripted("triage", fn="km_f", outputs=Handoff) | Keymaker(peers=["does_not_exist"]),
        Node.scripted("billing", fn="km_f", inputs={"handoff": Handoff}, outputs=Handoff) | Keymaker(peers=["triage"]),
    ],
)
