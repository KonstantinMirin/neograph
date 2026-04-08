# Attack vector 4c: dict-form output with correct key reference (should pass)
# source has outputs={"result": Claims, "log": str}.
# State fields are "source_result" and "source_log".
# Consumer expects source_result=Claims (correct).
from neograph import Construct, Node
from neograph.factory import register_scripted
from pydantic import BaseModel

class Claims(BaseModel, frozen=True):
    text: str

register_scripted("dict_out_4c", lambda i, c: {"result": Claims(text="ok"), "log": "done"})
register_scripted("good_ref_4c", lambda i, c: "ok")

pipeline = Construct("valid-dict-ref", nodes=[
    Node.scripted("source", fn="dict_out_4c", outputs={"result": Claims, "log": str}),
    Node.scripted("consumer", fn="good_ref_4c",
                  inputs={"source_result": Claims},
                  outputs=str),
])
