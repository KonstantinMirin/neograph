# CHECK_ERROR: output
# Dispatch mode (route='decide') requires output= (design §2.1 mode discrimination).
from pydantic import BaseModel

from neograph import Node, Portal
from tests.fakes import register_scripted


class Emitted(BaseModel, frozen=True):
    spec: dict
    dispatch_input: dict


register_scripted("km_f", lambda i, c: Emitted(spec={}, dispatch_input={}))

# route='decide' without output= is incomplete dispatch config — raises at
# Portal(...) construction (ConfigurationError).
pipeline_node = Node.scripted("dispatcher", fn="km_f", outputs=Emitted) | Portal(
    route="decide", spec_field="spec", input_field="dispatch_input"
)
