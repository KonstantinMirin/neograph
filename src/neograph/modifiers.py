"""Modifiers — composable pipeline behaviors applied via the | operator.

    node | Oracle(n=3, merge_prompt="rw/merge")
    node | Each(over="clusters.clusters", key="label")
    node | Operator(when="has_open_questions")
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Modifier(BaseModel, frozen=True):
    """Base class for node modifiers. Applied via Node.__or__."""


class Oracle(Modifier):
    """Ensemble modifier: N parallel generators + judge-merge.

    The compiler expands this into:
    1. Fan-out: Send(node, payload) x N with different generator IDs
    2. Barrier: merge node with defer=True
    3. Merge: LLM judge (merge_prompt) or scripted function (merge_fn)

    Exactly one of merge_prompt or merge_fn must be provided.

    Usage:
        # LLM merge:
        node | Oracle(n=3, merge_prompt="rw/decompose-merge")

        # Scripted merge:
        node | Oracle(n=3, merge_fn="combine_variants")
    """

    n: int = 3
    merge_prompt: str | None = None
    merge_model: str = "reason"
    merge_fn: str | None = None  # registered scripted function name

    def model_post_init(self, __context: Any) -> None:
        if not self.merge_prompt and not self.merge_fn:
            msg = "Oracle requires either merge_prompt (LLM judge) or merge_fn (scripted function)."
            raise ValueError(msg)
        if self.merge_prompt and self.merge_fn:
            msg = "Oracle accepts merge_prompt or merge_fn, not both."
            raise ValueError(msg)


class Each(Modifier):
    """Fan-out modifier: dispatch parallel instances over a collection.

    The compiler expands this into:
    1. Router node that iterates over the collection field in state
    2. Send() per item with the item as payload
    3. Barrier node with defer=True that collects results

    Usage:
        match_verify = Node(...) | Each(over="clusters.clusters", key="label")
    """

    over: str       # dotted path to collection in state (e.g., "clusters.clusters")
    key: str        # field on each item used as the dispatch key


class Operator(Modifier):
    """Human-in-the-loop modifier: pause graph for human review.

    The compiler inserts a check node after the modified node.
    If the condition is truthy, LangGraph interrupt() is called.
    The graph checkpoints and stops. Resume with:

        run(graph, resume={"approved": True, ...}, config=config)

    Usage:
        validate = Node(...) | Operator(when="any_test_failed")
    """

    when: str       # registered condition function name
