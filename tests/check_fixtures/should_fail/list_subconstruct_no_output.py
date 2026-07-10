# CHECK_ERROR: has no output type
# An explicitly listed Construct missing output= raises: the functions list
# is a promise that every element is a wireable member. (The module-walk
# side of the same case SKIPS with a ConstructArtifactSkipped warning — a
# namespace Construct without output= is a stored top-level pipeline
# artifact, e.g. a re-walked notebook namespace. Pins decision #3 of
# neograph-xv9ay, revised.)
from pydantic import BaseModel

from neograph import construct_from_functions, node


class Seed(BaseModel, frozen=True):
    text: str


class Verdict(BaseModel, frozen=True):
    label: str


@node(outputs=Verdict)
def judge(claim: Seed) -> Verdict:
    return Verdict(label=claim.text)


broken_sub = construct_from_functions("broken-sub", [judge], input=Seed)


@node(outputs=Seed)
def seed() -> Seed:
    return Seed(text="hello")


pipeline = construct_from_functions("parent", [seed, broken_sub])
