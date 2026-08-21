"""Example 32: A node that returns a list of models.

Scenario: a node extracts several readings from a report. The natural way to
say that is `outputs=list[Reading]` -- the node produces a list of models, so
the declaration says so.

For a long time it was the only shape the framework had never exercised. Every
example declared a CONTAINER MODEL with one list field instead:

    class Readings(BaseModel):
        items: list[Reading]        # a class minted to have something declarable

Both roads work, and this example exists so the direct one has a signpost.
Which to reach for:

- `outputs=list[Reading]` -- the node's whole output IS the collection, and
  nothing else travels with it. Nothing new enters your domain model.
- `outputs=Readings` -- the collection travels WITH something else (a summary,
  a confidence, a cursor), or the container is a real domain concept you would
  have written anyway. `Each(over="node.items")` fans over its field.

The trap is minting a container per node during a bug fix. That is how a
ten-class domain model becomes eighteen, and how a reviewer ends up asking when
`RoundDelta` joined the ontology. If the container is not a concept you would
name on a whiteboard, declare the list.

Under `output_strategy="json_mode"` the model is shown an ARRAY schema and
returns a JSON array, which the framework validates through the declared type.
Under the default `structured` strategy the same declaration goes through
constrained decoding. Both accept `list[Reading]`.

Keyless -- uses a fake LLM.

Run:
    python examples/32_list_output_type.py
"""

from __future__ import annotations

import sys

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from neograph import compile, construct_from_module, node, run

# ── Schemas ──────────────────────────────────────────────────────────────


class Reading(BaseModel, frozen=True):
    sensor: str
    value: float


class Summary(BaseModel, frozen=True):
    text: str


# ── Fake LLM ─────────────────────────────────────────────────────────────


class FakeExtractLLM:
    """Returns a JSON ARRAY, which is what an array schema asks for."""

    def invoke(self, messages, config=None, **kwargs):
        return AIMessage(content='[{"sensor": "t1", "value": 21.5}, {"sensor": "t2", "value": 19.0}]')


def llm_factory(tier):
    return FakeExtractLLM()


# ── Pipeline ─────────────────────────────────────────────────────────────


@node(
    outputs=list[Reading],
    model="fast",
    prompt="report/extract",
    llm_config={"output_strategy": "json_mode"},
)
def extract() -> list[Reading]:
    # body unused for mode='think' -- the LLM produces the value
    ...


@node(outputs=Summary)
def summarize(extract: list[Reading]) -> Summary:
    """A downstream node consumes the list by name, like any other output."""
    hottest = max(extract, key=lambda r: r.value)
    return Summary(text=f"{len(extract)} readings, hottest {hottest.sensor} at {hottest.value}")


pipeline = construct_from_module(sys.modules[__name__], name="sensor-report")


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = compile(
        pipeline,
        llm_factory=llm_factory,
        prompt_compiler=lambda template, data: [{"role": "user", "content": "extract"}],
    )
    result = run(graph, input={"node_id": "REPORT-7"})

    print(f"extract -> {type(result['extract']).__name__} of {len(result['extract'])}:")
    for reading in result["extract"]:
        print(f"  {reading.sensor}: {reading.value}")

    print(f"\nsummarize -> {result['summarize'].text}")
