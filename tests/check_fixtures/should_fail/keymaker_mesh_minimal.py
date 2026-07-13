# CHECK_ERROR: lands in T2
# A legal minimal Keymaker mesh: it ASSEMBLES cleanly (all §5 validation passes —
# in particular MEDIUM-1's fan-in 'handoff' skip lets the non-entry member
# assemble), but T1 fail-loud-stages the mesh LOWERING as a CompileError (D7).
# T2 (neograph-on6jt) implements lowering and moves this fixture back to
# should_pass/ (it then imports AND compiles cleanly).
from pydantic import BaseModel

from neograph import Construct, Keymaker, Node
from tests.fakes import register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


register_scripted("km_f", lambda i, c: Handoff(goto="__end__"))

pipeline = Construct(
    "swarm",
    nodes=[
        Node.scripted("triage", fn="km_f", outputs=Handoff) | Keymaker(peers=["billing"], max_hops=6),
        Node.scripted("billing", fn="km_f", inputs={"handoff": Handoff}, outputs=Handoff) | Keymaker(peers=["triage"]),
    ],
)
