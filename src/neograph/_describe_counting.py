"""Pass 1 of the two-pass renderer: how often does each nested class appear.

Split from ``describe_type`` (which crossed its size ceiling) along the seam
the renderer already had. Pass 1 COUNTS -- it walks the model tree and tallies
each nested class, so pass 2 knows which classes to hoist into their own
declaration rather than inline. Pass 2 (``_render_model_body`` / ``_render_type``)
stays in ``describe_type``, next to the public functions it serves.

``_stable_sort`` lives here because it exists solely to give the counted set a
deterministic order.
"""

from __future__ import annotations

import enum
import types
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from neograph._output_classify import output_markers


def _count_classes(
    model: type[BaseModel],
    counts: dict[type, int],
    enum_classes: set[type],
    recursive: set[type],
    visited: set[type],
    path: set[type],
) -> None:
    """Recursively count how many times each nested BaseModel/Enum appears.

    ``visited`` is a global dedup set (each model's subtree is walked once, so
    counts stay accurate for repeated siblings). ``path`` is the current
    ancestor chain, used to detect back-edges — a model reachable from itself is
    recorded in ``recursive`` and later force-hoisted.
    """
    if model in visited:
        return
    visited.add(model)
    path.add(model)

    for _name, field_info in model.model_fields.items():
        if field_info.exclude or output_markers(field_info)[0]:
            continue
        _count_annotation(field_info.annotation, counts, enum_classes, recursive, visited, path)

    path.discard(model)


def _count_annotation(
    annotation: Any,
    counts: dict[type, int],
    enum_classes: set[type],
    recursive: set[type],
    visited: set[type],
    path: set[type],
) -> None:
    """Count occurrences within a single type annotation."""
    if annotation is None or annotation is type(None):
        return

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union or origin is types.UnionType:
        for arg in args:
            _count_annotation(arg, counts, enum_classes, recursive, visited, path)
        return

    if origin is Literal:
        return

    if origin in (list, tuple, frozenset, set):
        for arg in args:
            _count_annotation(arg, counts, enum_classes, recursive, visited, path)
        return

    if origin is dict:
        for arg in args:
            _count_annotation(arg, counts, enum_classes, recursive, visited, path)
        return

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        enum_classes.add(annotation)
        counts[annotation] = counts.get(annotation, 0) + 1
        return

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        counts[annotation] = counts.get(annotation, 0) + 1
        # A back-edge to an ancestor currently on the path is a recursion cycle.
        if annotation in path:
            recursive.add(annotation)
        _count_classes(annotation, counts, enum_classes, recursive, visited, path)
        return


# ---------------------------------------------------------------------------
# Pass 2: render
# ---------------------------------------------------------------------------


def _stable_sort(classes: set[type]) -> list[type]:
    """Sort classes by name for deterministic output."""
    return sorted(classes, key=lambda c: c.__name__)


# ---------------------------------------------------------------------------
# describe_value: BAML-style instance renderer
# ---------------------------------------------------------------------------
