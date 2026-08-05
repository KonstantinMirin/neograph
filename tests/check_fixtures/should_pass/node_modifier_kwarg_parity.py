# Valid: the parity kwargs added under neograph-d5pvl — merge_model=
# (-> Oracle.merge_model) and map_on_error= (-> Each.on_error) — compile
# through @node. (loop_history= was removed as born-redundant, neograph-eef83.)
# Extended (Phase 3 strictness gate should_pass twin): merge_pre_process/
# merge_post_process/merge_fallback each paired with the merge_prompt trigger
# already on `judge`, and on_exhaust= paired with the loop_when trigger
# already on `refine` -- every one of Phase 3's 4 newly-gated satellites has
# a should_pass cell here, alongside the 4 should_fail fixtures.
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
    merge_pre_process=lambda variants: {"variants": variants},
    merge_post_process=lambda result, variants: result,
    merge_fallback=lambda variants, error: variants[0],
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
    on_exhaust="last",
)
def refine(seed: Draft) -> Draft: ...


pipeline = construct_from_functions("parity-kwargs", [produce, judge, seed, refine])
