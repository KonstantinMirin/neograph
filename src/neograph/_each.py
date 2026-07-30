"""``Each`` — fan-out over a collection, and its failure policy.

Extracted from ``modifiers.py`` (neograph-3ffdg.5) as a pure file split — the
classes and helper below are unchanged, only their home moved. ``modifiers.py``
re-exports them; external callers use the already-public ``split_each_path``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

from neograph._modifier_base import Modifier


class EachFailure(BaseModel, frozen=True):
    """Typed per-item failure written into an Each barrier under ``on_error='collect'``.

    Replaces a thrown item's result in the keyed barrier dict so the barrier
    always completes with one entry per planned key. Consumers assert
    set-equality over planned keys and branch on ``isinstance(v, EachFailure)``.
    """

    key: str  # the Each dispatch key of the item that failed
    error_type: str  # exception class name (e.g., "RuntimeError")
    message: str  # str() of the caught exception


class Each(Modifier, frozen=True):
    """Fan-out modifier: dispatch parallel instances over a collection.

    The compiler expands this into:
    1. Router node that iterates over the collection field in state
    2. Send() per item with the item as payload
    3. Barrier node with defer=True that collects results

    Usage:
        match_verify = Node(...) | Each(over="clusters.clusters", key="label")

    ``on_error`` controls per-item fault handling:
    - ``'raise'`` (default): a thrown item aborts the whole fan-out run.
    - ``'collect'``: a thrown item is caught and keyed into the barrier as a
      typed ``EachFailure`` instead of aborting; the barrier always completes.
    """

    over: str  # dotted path to collection in state (e.g., "clusters.clusters")
    key: str  # field on each item used as the dispatch key
    on_error: Literal["raise", "collect"] = "raise"

    @field_validator("over")
    @classmethod
    def _validate_over(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Each.over must not be empty")
        return v


def split_each_path(over: str) -> tuple[str, tuple[str, ...]]:
    """Parse an `Each.over` dotted path into (root_field, remaining_segments).

    Single point of truth for the path grammar. Both the assembly-time
    type walker in `construct.py` and the runtime value walker in
    `compiler.py` consume this so future extensions to the syntax (indexing,
    wildcards, escaping) land in one place.
    """
    parts = over.split(".")
    return parts[0], tuple(parts[1:])
