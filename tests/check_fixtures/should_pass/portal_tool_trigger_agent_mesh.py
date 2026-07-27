# A legal minimal Portal tool-triggered mesh (design portal-tool-triggered-handoff):
# two agent-mode peers, each carrying Portal(trigger="tool"). It ASSEMBLES cleanly
# (both members are agent-mode, so the §4 trigger="tool"-requires-agent/act rule
# passes) AND compiles cleanly (the tool-triggered agent-cycle wiring lowers the
# {node}__tools node into a Command-emitting handoff exit). The harness compiles
# every should_pass Construct with a placeholder LLM, so this pins the
# tool-triggered mesh lowering end to end.
from pydantic import BaseModel

from neograph import Construct, Node, Portal


class Handoff(BaseModel, frozen=True):
    goto: str


triage = Node(
    name="triage",
    mode="agent",
    model="fast",
    prompt="rw/triage",
    inputs={"handoff": Handoff},
    outputs=Handoff,
    tools=[],
) | Portal(to=["billing"], trigger="tool", max_hops=6)

billing = Node(
    name="billing",
    mode="agent",
    model="fast",
    prompt="rw/billing",
    inputs={"handoff": Handoff},
    outputs=Handoff,
    tools=[],
) | Portal(to=["triage"], trigger="tool")

pipeline = Construct("portal-tool-trigger-agent-mesh", nodes=[triage, billing])
