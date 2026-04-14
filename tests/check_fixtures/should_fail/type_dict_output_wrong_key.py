# CHECK_ERROR: no upstream node named|type.*compatible
# Attack vector 4: dict-form outputs where consumer references wrong key
# Producer has outputs={"result": Claims, "log": str}, state fields are
# "source_result" and "source_log". Consumer references "source_log" correctly
# but with wrong type expectation.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.factory import register_scripted


class Claims(BaseModel, frozen=True):
    text: str

register_scripted("dict_out", lambda i, c: {"result": Claims(text="ok"), "log": "done"})
register_scripted("bad_ref", lambda i, c: "ok")

# The dict-form output creates state fields: source_result (Claims) and source_log (str)
# Consumer references "source_wrong" which doesn't exist
pipeline = Construct("broken", nodes=[
    Node.scripted("source", fn="dict_out", outputs={"result": Claims, "log": str}),
    Node.scripted("consumer", fn="bad_ref",
                  inputs={"source_wrong": Claims},
                  outputs=str),
])
