# CHECK_ERROR: references context='nonexistent_node' but no upstream node produces a field with that name
# GAP: context= references are never validated at compile time.
# state.py blindly creates Any-typed fields for each context name.
# factory.py reads them at runtime via the StateBus adapter, silently returning None.
# A typo in context=["nonexistant_node"] compiles fine but injects None
# into the LLM prompt at runtime.
from pydantic import BaseModel

from neograph import Construct, Node


class Input(BaseModel, frozen=True):
    text: str


class Output(BaseModel, frozen=True):
    result: str


pipeline = Construct(
    "broken",
    nodes=[
        Node(
            name="first",
            mode="think",
            inputs=Input,
            outputs=Output,
            model="fast",
            prompt="test",
            context=["nonexistent_node"],
        ),
    ],
)
