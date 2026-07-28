# CHECK_ERROR: Each x Oracle fusion is not supported on sub-constructs
# neograph-s7zt3.7 R4 pin. Each x Oracle fusion is defined entirely in terms of a
# single Node's map_over/ensemble_n (an M x N Send topology) which a multi-node
# Construct structurally lacks, so SUB_CONSTRUCT_UNSUPPORTED_COMBOS rejects the
# combo on a sub-construct. compiler.py's `_add_subgraph` is the ONE site that
# raises it (CompileError), and `compile_state_model` runs BEFORE it -- so
# state.py's sub-construct field builder must keep its NON-raising defensive
# fallback for this combo. If state.py ever pre-empts with its own raise, the
# user-visible error changes site, type and text and this fixture goes red.
from pydantic import BaseModel

from neograph import Construct, Node
from neograph.modifiers import Each, Oracle
from tests.fakes import register_scripted


class Item(BaseModel, frozen=True):
    label: str


class Batch(BaseModel, frozen=True):
    items: list[Item]


class Scored(BaseModel, frozen=True):
    text: str


register_scripted("seof_seed", lambda _i, _c: Batch(items=[Item(label="a")]))
register_scripted("seof_inner", lambda i, _c: Scored(text=i.label))
register_scripted("seof_merge", lambda variants, _c: variants[0])

sub = Construct(
    "seof-sub",
    input=Item,
    output=Scored,
    nodes=[Node.scripted("seof-inner", fn="seof_inner", inputs=Item, outputs=Scored)],
)

# Each x Oracle on a Construct — legal to assemble, rejected at compile.
fused_sub = sub | Each(over="seof_seed.items", key="label") | Oracle(n=2, merge_fn="seof_merge")

pipeline = Construct(
    "seof-parent",
    nodes=[
        Node.scripted("seof-seed", fn="seof_seed", outputs=Batch),
        fused_sub,
    ],
)
