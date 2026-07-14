# CHECK_ERROR: billing
# A mesh member whose output type differs from the entry's (design §5.3).
from pydantic import BaseModel

from neograph import Construct, Node, Portal
from tests.fakes import register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


class OtherPayload(BaseModel, frozen=True):
    goto: str


register_scripted("km_f", lambda i, c: Handoff(goto="__end__"))

pipeline = Construct(
    "swarm",
    nodes=[
        Node.scripted("triage", fn="km_f", outputs=Handoff) | Portal(to=["billing"]),
        Node.scripted("billing", fn="km_f", inputs={"handoff": Handoff}, outputs=OtherPayload) | Portal(to=["triage"]),
    ],
)
