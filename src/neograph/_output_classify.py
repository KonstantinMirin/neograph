"""Output-field marker family + the shared schema PROJECTION (neograph-ftnxl.4).

Exactly one function decides which fields of a node's output model the LLM is
asked to produce (``output_markers`` / ``project_output_model``), and exactly
one function fills the fields it removed (``splice_carried``) -- so the
schema the model is constrained by (``with_structured_output``), the schema
text it is shown (``describe_type``), and the instance written to the state
bus can never disagree.

Before this module, ``ExcludeFromOutput`` (``describe_type.py``) had exactly
ONE reader -- the text renderer -- so the promise in its own docstring ("the
LLM won't try to produce it") was FALSE under the default
``output_strategy="structured"``: ``with_structured_output`` received the
model unprojected, and an excluded field with no default was demanded of the
provider as ``required``. ``project_output_model`` is the fix; ``Carried``
joins the same projection so it never repeats the bug.

``Carried`` fields are spliced in by the FRAMEWORK after the LLM response is
parsed -- the model never authors them. A ``Carried`` path may root ONLY at a
name the node itself declares (a ``node.inputs`` key or a ``_param_res`` DI
param name) -- never another node's output by an undeclared path, which would
add a validator-invisible dataflow edge (the Portal-precedent escalation this
stays sugar to avoid). Depth-0 only: a marker on a NESTED model's field is
rejected at assembly time (``_validation_outputs.py``) -- ``describe_type``'s
strip is recursive, a flat splice is not, and a nested marker would strip
from the rendered text while staying demanded by the validating schema with
no way to fill it back in — the exact bug this module exists to close,
reproduced one level down.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Union

from pydantic import BaseModel, ValidationError, create_model
from pydantic.fields import FieldInfo

from neograph._state_keys import StateKeys
from neograph.errors import ConstructError, ExecutionError

if TYPE_CHECKING:
    from neograph._dispatch import NodeInput
    from neograph.node import Node

__all__ = ["Carried", "ExcludeFromOutput"]


class ExcludeFromOutput:
    """Marker: field is visible in input rendering but excluded from output schema.

    Use with Annotated on a Pydantic field::

        source_url: Annotated[str, ExcludeFromOutput] = ""

    The field will:
    - Be rendered when the model is shown as input (XmlRenderer, DelimitedRenderer)
    - Be EXCLUDED from describe_type() output schema AND from the schema the
      LLM's structured-output call is constrained by (``project_output_model``)
    - Must have a default value (since the LLM won't provide it)

    THE OWNER of this marker moved here (neograph-ftnxl.4) from
    ``describe_type.py``, re-exported there for backward compatibility --
    ``describe_type`` is a CONSUMER of the shared projection now, not the
    single reader it used to be.
    """


class Carried:
    """Marker: field is spliced in by the framework from the node's own
    ``input_data``/DI params after the LLM response is parsed -- the model
    never authors it.

    Use with ``Annotated`` on a Pydantic field::

        source_url: Annotated[str, Carried("claims.source")]

    ``path`` roots at a name the node's OWN declared inputs/DI params expose
    (a ``node.inputs`` key or a ``_param_res`` DI param name); the remaining
    dotted segments are attribute access on that value. See the module
    docstring for the scope fence and the depth-0-only restriction.
    """

    __slots__ = ("path", "segments")

    def __init__(self, path: str) -> None:
        if not path or not isinstance(path, str):
            # ConstructError subclasses ValueError — a marker-arg contract
            # error is a construction defect (mirrors FromResource.__init__).
            raise ConstructError.build(
                f"Carried(path=...) requires a non-empty string path, got {path!r}",
                hint="Carried('claims.source') -- dotted, rooted at a declared input/DI name",
            )
        self.path = path
        self.segments = tuple(path.split("."))

    def __repr__(self) -> str:
        return f"Carried({self.path!r})"


def _is_carried(field_info: FieldInfo, *, field_label: str = "a field") -> Carried | None:
    """The one ``Carried`` marker on a field, or ``None``.

    REFUSES two markers instead of taking the first. This returned
    ``metadata[0]`` and discarded the rest, so an author who wrote two ``Carried``
    paths on one field got the one that happened to be listed first and no
    indication a choice had been made -- the same fail-soft shape as the type scan
    this epic removes: an answer arriving by POSITION rather than by contract.

    A foreign bag is the one place neograph cannot close by construction. User code
    populates ``FieldInfo.metadata`` before neograph ever sees it, so the set cannot
    be made unrepresentable the way ``Source`` is. Every ambiguity there is
    AUTHORED, which is exactly the kind design section 5 says to refuse -- so the
    bag crosses the boundary once, through a parser that refuses rather than
    guesses. neograph-22jvj.
    """
    markers = [m for m in field_info.metadata if isinstance(m, Carried)]
    if len(markers) > 1:
        paths = [getattr(m, "path", None) or repr(m) for m in markers]
        raise ConstructError.build(
            f"{field_label} carries {len(markers)} Carried markers",
            expected="at most one Carried marker per field",
            found=f"paths: {paths}",
            hint=(
                "Two markers on one field is authored ambiguity: only one value can be "
                "spliced in, and taking the first silently is how the wrong one shipped. "
                "Remove the marker you do not mean."
            ),
        )
    return markers[0] if markers else None


def _is_output_excluded(field_info: FieldInfo) -> bool:
    """True if the field carries an ExcludeFromOutput marker in Annotated metadata."""
    return any(m is ExcludeFromOutput or isinstance(m, ExcludeFromOutput) for m in field_info.metadata)


def output_markers(field_info: FieldInfo, *, field_label: str = "a field") -> tuple[bool, Carried | None]:
    """Single predicate BOTH the text renderer (``describe_type``) and the
    schema projector (``project_output_model``) consume: ``(strip_from_output,
    carried_marker_or_None)`` for one field. ``strip_from_output`` is True for
    either ``ExcludeFromOutput`` or ``Carried`` -- both are fields the LLM
    must not be asked to produce.
    """
    carried = _is_carried(field_info, field_label=field_label)
    excluded = _is_output_excluded(field_info)
    return (excluded or carried is not None), carried


@lru_cache(maxsize=256)
def project_output_model(model: type[BaseModel]) -> type[BaseModel]:
    """A STRUCTURAL SIBLING of ``model`` (created via
    ``create_model(..., __base__=BaseModel, **fields)``, NEVER a subclass --
    pydantic cannot remove an inherited field in a subclass, so a
    subclass-based projection would be a silent no-op) with every
    Excluded/Carried field removed.

    Preserves ``model_config`` and field-level constraints (the same
    ``FieldInfo`` objects ride along unchanged for every kept field).
    DELIBERATELY drops model-LEVEL validators (``@model_validator``) -- one
    may reference a stripped field and cannot run meaningfully against the
    projection; the reconstructed ``declared`` instance (``splice_carried``)
    re-validates the full model, including those validators, at the end.

    ``lru_cache``-bounded (not a module-level mutable registry) -- pure
    function of the model class.
    """
    # A declared CONTAINER type (list[Claim], dict[str, Claim]) has no fields
    # of its own to project, so it passes through unchanged. Same assumption as
    # the parse tail once made -- that a declared output type is always a
    # BaseModel SUBCLASS -- and the same failure when it is not.
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        return model

    fields: dict[str, Any] = {}
    changed = False
    for name, info in model.model_fields.items():
        strip, _carried = output_markers(info)
        if strip:
            changed = True
            continue
        fields[name] = (info.annotation, info)
    if not changed:
        return model
    projected = create_model(f"{model.__name__}_Projected", __base__=BaseModel, **fields)
    projected.model_config.update(model.model_config)
    return projected


def _is_optional(annotation: Any) -> bool:
    origin = getattr(annotation, "__origin__", None)
    if origin is Union:
        return type(None) in getattr(annotation, "__args__", ())
    return False


def _resolve_carried_root(
    node: Node,
    carried: Carried,
    input_data: NodeInput | None,
    config: Any,
) -> Any:
    root, *rest = carried.segments
    node_name = getattr(node, "name", "?")
    fan_in = getattr(input_data, "fan_in", None) or {}
    di_inputs = ((config or {}).get("configurable") or {}).get(StateKeys.DI_INPUTS) or {}
    if root in fan_in:
        value: Any = fan_in[root]
    elif root in di_inputs:
        value = di_inputs[root]
    else:
        raise ExecutionError.build(
            f"node {node_name!r}: Carried({carried.path!r}) root {root!r} is not a "
            "declared input or DI param of this node",
            node=node_name,
        )
    for seg in rest:
        if value is None:
            raise ExecutionError.build(
                f"node {node_name!r}: Carried({carried.path!r}) -- {seg!r} accessed on None "
                f"while resolving {'.'.join(carried.segments)!r}",
                node=node_name,
            )
        value = getattr(value, seg)
    return value


def splice_carried(
    node: Node,
    declared: type[BaseModel],
    projected_result: BaseModel,
    input_data: NodeInput | None,
    config: Any,
) -> BaseModel:
    """Reconstruct a ``declared`` instance from ``projected_result`` (whatever
    ``project_output_model(declared)`` produced and the LLM call actually
    returned), splicing each Carried field's value in from ``input_data``/DI
    params. A shallow field-wise copy -- NOT ``declared(**projected_result.
    model_dump(), **carried)``, which would deep-dump nested models to plain
    dicts and re-run every (possibly non-idempotent) validator.

    MISSING (unresolvable root/attribute) always raises. A PRESENT-but-``None``
    value splices as ``None`` when the declared field is ``Optional``, raises
    otherwise -- "missing" and "legitimately None" are different failures.
    """
    # A declared CONTAINER type has no fields, so nothing was ever projected out
    # of it and there is nothing to splice back in. Checked before the identity
    # test below, which can never hold for one: type([...]) is `list`, while
    # `declared` is `list[Claim]`.
    if not (isinstance(declared, type) and issubclass(declared, BaseModel)):
        return projected_result

    if type(projected_result) is declared:
        return projected_result  # nothing was projected -- already the declared class

    merged: dict[str, Any] = {name: getattr(projected_result, name) for name in type(projected_result).model_fields}
    for name, info in declared.model_fields.items():
        if name in merged:
            continue
        _strip, carried = output_markers(info)
        if carried is None:
            continue  # ExcludeFromOutput only: takes its Pydantic default via declared(**merged)
        value = _resolve_carried_root(node, carried, input_data, config)
        if value is None and not _is_optional(info.annotation):
            raise ExecutionError.build(
                f"node {getattr(node, 'name', '?')!r}: Carried({carried.path!r}) resolved to None "
                f"for non-Optional field {name!r}",
                node=getattr(node, "name", None),
            )
        merged[name] = value

    try:
        return declared(**merged)
    except ValidationError as exc:
        raise ExecutionError.build(
            f"node {getattr(node, 'name', '?')!r}: failed to reconstruct {declared.__name__} "
            "after splicing Carried fields",
            found=str(exc),
            node=getattr(node, "name", None),
        ) from exc
