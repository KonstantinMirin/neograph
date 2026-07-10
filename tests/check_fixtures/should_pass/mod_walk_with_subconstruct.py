# Valid: construct_from_module collects a module-level sub-construct (with
# its output= boundary declared) alongside @node functions and wires it via
# its output port. Pins the neograph-xv9ay fix: the module walk and the
# explicit list share one member-selection predicate.
import types

from pydantic import BaseModel

from neograph import construct_from_functions, construct_from_module, node


class Seed(BaseModel, frozen=True):
    text: str


class Verdict(BaseModel, frozen=True):
    label: str


@node(outputs=Verdict)
def judge(claim: Seed) -> Verdict:
    return Verdict(label=claim.text)


judge_sub = construct_from_functions("judge-sub", [judge], input=Seed, output=Verdict)


@node(outputs=Seed)
def seed() -> Seed:
    return Seed(text="hello")


_mod = types.ModuleType("mod_walk_subconstruct_demo")
_mod.seed = seed
_mod.judge_sub = judge_sub

pipeline = construct_from_module(_mod)
