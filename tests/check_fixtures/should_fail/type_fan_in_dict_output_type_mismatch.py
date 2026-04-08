# CHECK_ERROR: type.*compatible|produces
# Attack vector 4b: dict-form output, consumer references correct key name
# but with wrong type. source has outputs={"result": Claims, "log": str}.
# State fields are "source_result" and "source_log".
# Consumer expects source_result=int (wrong, should be Claims).
from neograph import Construct, Node
from neograph.factory import register_scripted
from pydantic import BaseModel

class Claims(BaseModel, frozen=True):
    text: str

register_scripted("dict_out_4b", lambda i, c: {"result": Claims(text="ok"), "log": "done"})
register_scripted("bad_type_4b", lambda i, c: "ok")

pipeline = Construct("broken", nodes=[
    Node.scripted("source", fn="dict_out_4b", outputs={"result": Claims, "log": str}),
    Node.scripted("consumer", fn="bad_type_4b",
                  inputs={"source_result": int},  # should be Claims
                  outputs=str),
])
