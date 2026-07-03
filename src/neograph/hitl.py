"""Human-in-the-loop sugar for agent/act tool bodies.

``ask_human`` is optional, first-class sugar over LangGraph's
``langgraph.types.interrupt()``. It is a *pure Layer-2 passthrough*: it adds a
typed payload/resume contract and makes the HITL pause a named marker the linter
can see (see ``ask_human_in_mutating_node`` in ``lint.py``), but it contains ZERO
execution logic. The pause/resume path is byte-identical to calling
``interrupt()`` directly — same ``__interrupt__`` surface, same raw resume value.

Calling raw ``interrupt()`` inside a tool remains fully supported; ``ask_human``
is the blessed path, not the only one. The mechanism that makes a mid-loop
interrupt work (checkpoint-at-node-boundary, exactly-once pre-interrupt tool
execution across resume) lives in the agent-cycle compiler and LangGraph — not
here. See ``tests/test_agent_subgraph_keystone.py`` for the raw-path guarantee
and ``tests/test_hitl.py`` for the parity of this sugar with it.
"""

from __future__ import annotations

from typing import TypeVar

from langgraph.types import interrupt
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def ask_human(payload: BaseModel, resume_model: type[T] | None = None) -> T | dict:
    """Pause an agent/act tool mid-loop to ask a human, and return their answer.

    Pass ``payload`` (any Pydantic model) — it surfaces to the caller as the
    ``__interrupt__`` value (``payload.model_dump()``), exactly as a raw
    ``interrupt(payload.model_dump())`` would. Resume the graph with the human's
    answer; that value comes back here.

    When ``resume_model`` is given, the resumed value is validated into that
    model and returned as a typed instance — a malformed answer raises
    ``pydantic.ValidationError`` at THIS boundary, not deep inside tool code.
    When ``resume_model`` is ``None``, the raw resume value is returned
    unchanged (byte-identical to the raw path).

    Args:
        payload: The question/context handed to the human. Surfaces as the
            interrupt value via ``payload.model_dump()``.
        resume_model: Optional Pydantic model to validate the resumed value
            into. Omit to receive the raw resume value unchanged.

    Returns:
        A validated ``resume_model`` instance when ``resume_model`` is given,
        otherwise the raw resume value (typically a ``dict``).

    Raises:
        pydantic.ValidationError: When ``resume_model`` is given and the resumed
            value fails validation.
    """
    returned = interrupt(payload.model_dump())
    if resume_model is not None:
        return resume_model.model_validate(returned)
    return returned
