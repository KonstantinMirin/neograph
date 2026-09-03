"""lint check: a model-authored field with a default whose type rejects null.

A Pydantic default applies only when the key is ABSENT. A present-and-null
value OVERRIDES it and fails validation -- and ``describe_type`` tells the model
it may send exactly that: ``_render_model_body`` appends `` or null`` whenever a
field is not required and its annotation does not already admit ``None``. So for
``status: ClaimStatus = 'proposed'`` neograph ships ``status: "proposed" or
"accepted" or null`` and the model, reasonably, sometimes sends null.

This check reports that field before a run. It is the STATIC half of GH #20;
the runtime half coerces the null back to the default on both output strategies.
Coercion makes the pipeline RUN, which is why this is a WARN -- but it does not
make the field a good idea, because a model that declines to author it yields
the default silently, which reads exactly like a real answer.

WHY IT RECURSES, and why that is the whole point: both production failures the
reporter measured were on types reachable FROM a node's declared output, not on
the output itself. Their first attempt walked top-level fields only and reported
clean on a tree that already contained the defect -- an instrument passing
because it did not look.

SCOPE -- what "model-authored" means here:
- think/agent/act nodes only. A scripted node's output is built in Python, where
  a default is ordinary and correct.
- the PRIMARY output only. For dict-form ``outputs=``, every key after the first
  is framework-collected (``tool_log``), never authored by the model.
- fields ``describe_type`` itself skips (``exclude=True`` / ``ExcludeFromOutput``)
  are skipped here too: the model never sees them, so it cannot send null for
  them.

``_admits_none`` is imported, never re-derived. It is the annotation-shape
authority that GH #7 established, and reusing it is what keeps this check in
lockstep with what the renderer actually tells the model.
"""

from __future__ import annotations

import types
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

from neograph._dispatch import _resolve_primary_output
from neograph._lint_kind_registry import LintIssue
from neograph.describe_type import _admits_none, _is_output_excluded
from neograph.node import Node

# The modes whose declared output the MODEL authors. A scripted node builds its
# output in Python, so a default there is not a defect.
_MODEL_AUTHORED_MODES = ("think", "agent", "act")


def _reachable_models(annotation: Any, visited: set[type]) -> list[type[BaseModel]]:
    """Every BaseModel reachable from *annotation*, one container layer at a time.

    Deliberately annotation-driven rather than model-driven: the walker's ENTRY
    is a type expression, so a node declaring ``outputs=list[Claim]`` -- a
    container, not a bare model -- is walked rather than silently skipped.
    *visited* makes the walk terminate on self-referential and mutually
    recursive models, and makes a model reachable by two paths yield its fields
    once.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation in visited:
            return []
        visited.add(annotation)
        return [annotation]

    origin = get_origin(annotation)
    if origin is None and not isinstance(annotation, type):
        return []
    args = get_args(annotation)
    if origin is Union or origin is types.UnionType:
        pass  # members are walked below like any other type argument
    found: list[type[BaseModel]] = []
    for arg in args:
        if arg is Ellipsis:
            continue
        found.extend(_reachable_models(arg, visited))
    return found


def _walk_model(
    model: type[BaseModel],
    node: Node,
    issues: list[LintIssue],
    *,
    path: str,
    visited: set[type],
) -> None:
    """Report every defective field of *model*, then descend into its field types.

    *path* is the dotted route from the node's declared output down to this
    model, so the report says WHERE. Without it a nested report is unactionable
    and, worse, indistinguishable from a top-level-only walk that got lucky.
    """
    for field_name, field_info in model.model_fields.items():
        if field_info.exclude or _is_output_excluded(field_info):
            continue
        annotation = field_info.annotation
        dotted = f"{path}{field_name}" if path else field_name
        # THE predicate, and it is describe_type.py's verbatim: "has a default"
        # (not required) AND "the type rejects null" (the annotation does not
        # admit None). `Any` is excluded because it genuinely holds null.
        if not field_info.is_required() and annotation is not Any and not _admits_none(annotation):
            issues.append(
                LintIssue(
                    node_name=node.name,
                    param=dotted,
                    kind="model_authored_null_rejecting_default",
                    message=(
                        f"'{dotted}' ({model.__name__}.{field_name}) carries a default and its type "
                        f"rejects null, but the model authors it -- describe_type renders it as "
                        f"'... or null', so a present-and-null value is a live outcome. The runtime "
                        f"coerces the null back to the default, which means the model can decline to "
                        f"author this field and the default ships as if it were an answer. Drop the "
                        f"default to make it required, make the type nullable, or exclude the field "
                        f"from the model-facing schema if the pipeline sets it."
                    ),
                    required=False,
                )
            )
        for nested in _reachable_models(annotation, visited):
            _walk_model(nested, node, issues, path=f"{dotted}.", visited=visited)


def _check_null_rejecting_defaults(item: Node, issues: list[LintIssue]) -> None:
    """Entry point called once per node from ``lint._walk``.

    *visited* is built FRESH here, per node. Hoisting it across nodes would make
    the SECOND node that shares a nested output model report nothing -- the same
    "passed because it did not look" failure, reintroduced by deduplication.
    """
    if item.mode not in _MODEL_AUTHORED_MODES:
        return
    declared, _primary_key = _resolve_primary_output(item)
    if declared is None:
        return
    visited: set[type] = set()
    for model in _reachable_models(declared, visited):
        _walk_model(model, item, issues, path="", visited=visited)
