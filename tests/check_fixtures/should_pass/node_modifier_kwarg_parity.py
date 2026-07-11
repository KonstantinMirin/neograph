# Valid: the three parity kwargs added under neograph-d5pvl — merge_model=
# (-> Oracle.merge_model), loop_history= (-> Loop.history, riding loop_when=),
# and map_on_error= (-> Each.on_error) — all compile through @node.
from pydantic import BaseModel

from neograph import construct_from_functions, node


class Item(BaseModel, frozen=True):
    label: str
    value: str


class Items(BaseModel, frozen=True):
    items: list[Item]


class Verdict(BaseModel, frozen=True):
    label: str
    ok: bool


class Draft(BaseModel, frozen=True):
    text: str
    score: float


@node(outputs=Items)
def produce() -> Items:
    return Items(items=[Item(label="a", value="1")])


@node(
    outputs=Verdict,
    prompt="judge ${produce}",
    model="fast",
    map_over="produce.items",
    map_key="label",
    map_on_error="collect",
    ensemble_n=3,
    merge_prompt="merge the verdicts: ${variants}",
    merge_model="fast",
)
def judge(item: Item) -> Verdict: ...


@node(outputs=Draft)
def seed() -> Draft:
    return Draft(text="v0", score=0.0)


@node(
    outputs=Draft,
    prompt="refine ${seed}",
    model="fast",
    loop_when=lambda d: d is None or d.score < 0.8,
    max_iterations=3,
    loop_history=True,
)
def refine(seed: Draft) -> Draft: ...


pipeline = construct_from_functions("parity-kwargs", [produce, judge, seed, refine])
