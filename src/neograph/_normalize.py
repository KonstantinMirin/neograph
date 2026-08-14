"""Normalization of polymorphic ``Node.outputs`` and ``Node.inputs``.

``Node.outputs`` is ``type | dict[str, type] | None``; ``Node.inputs`` is the
same trichotomy. Discriminating these forms via ``isinstance(node.outputs, dict)``
was repeated 18+ times across the codebase before this module existed.

This module is the single place where that discrimination happens. Every other
module accesses the normalized view (``NormalizedOutputs`` / ``NormalizedInputs``)
and never touches the raw polymorphic field.

A structural guard in ``tests/test_structural_guards.py`` enforces that no
other ``src/neograph/*.py`` file does ``isinstance(<expr>.outputs, dict)`` or
``isinstance(<expr>.inputs, dict)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

from neograph._ir_protocols import ConstructItem
from neograph.errors import ConfigurationError
from neograph.naming import output_field_name
from neograph.node import Node, TypeSpecStatic

_ItemT = TypeVar("_ItemT", bound=ConstructItem)


@dataclass(frozen=True)
class NormalizedOutputs:
    """Normalized view of ``Node.outputs``.

    ``primary`` / ``primary_key`` capture the LLM-facing output (first dict key
    for dict-form, the raw type for single-type). ``secondary`` carries the
    remaining dict entries (e.g. ``tool_log``). ``all_keys`` is the ordered
    mapping for dict-form callers that need to iterate every output field.
    """

    primary: Any
    primary_key: str | None
    secondary: dict[str, Any]
    all_keys: dict[str, Any]
    is_dict_form: bool
    is_none: bool


@dataclass(frozen=True)
class NormalizedInputs:
    """Normalized view of ``Node.inputs``.

    Dict-form inputs populate ``by_name``; single-type inputs populate
    ``single_type``. Both forms set ``is_none=False``.
    """

    by_name: dict[str, Any] = field(default_factory=dict)
    single_type: Any = None
    is_dict_form: bool = False
    is_none: bool = False


def normalize_outputs(outputs: Any) -> NormalizedOutputs:
    """Discriminate ``Node.outputs`` into a normalized view.

    - ``None`` → ``is_none=True``, primary=None, primary_key=None, secondary={}.
    - ``dict[str, type]`` → ``is_dict_form=True``, primary is the first value,
      primary_key the first key, secondary the rest.
    - Single type → ``primary=type``, primary_key=None, secondary={}.
    """
    if outputs is None:
        return NormalizedOutputs(
            primary=None,
            primary_key=None,
            secondary={},
            all_keys={},
            is_dict_form=False,
            is_none=True,
        )
    if isinstance(outputs, dict):
        items = list(outputs.items())
        primary_key, primary = items[0]
        secondary = dict(items[1:])
        return NormalizedOutputs(
            primary=primary,
            primary_key=primary_key,
            secondary=secondary,
            all_keys=dict(outputs),
            is_dict_form=True,
            is_none=False,
        )
    return NormalizedOutputs(
        primary=outputs,
        primary_key=None,
        secondary={},
        all_keys={},
        is_dict_form=False,
        is_none=False,
    )


def resolve_primary_output(node: Node) -> tuple[TypeSpecStatic, str | None]:
    """Resolve the LLM output model and primary key for dict-form outputs.

    For dict-form outputs, the LLM produces the primary type (first key).
    Secondary outputs (e.g. tool_log) are framework-collected.

    When ``node.oracle_gen_type`` is set (Oracle with type-transforming merge_fn),
    the generator type overrides ``node.outputs`` -- the LLM should produce the
    per-variant type, not the post-merge type.

    Returns (output_model, primary_key) where primary_key is None for
    single-type outputs.

    Relocated from ``_dispatch.py`` (neograph-ftnxl.17) so a sibling module
    that must NOT import ``_dispatch.py`` (``_llm_render.py`` -- ``_dispatch.py``
    already imports FROM it, so the reverse import would cycle) can reach the
    ONE canonical resolver instead of re-deriving it via raw ``node.outputs``
    access.
    """
    # Oracle generator type override: merge_fn transforms A -> B, generators produce A.
    if node.oracle_gen_type is not None:
        return node.oracle_gen_type, None

    no = normalize_outputs(node.outputs)
    return no.primary, no.primary_key


def normalize_inputs(inputs: Any) -> NormalizedInputs:
    """Discriminate ``Node.inputs`` into a normalized view.

    - ``None`` → ``is_none=True``.
    - ``dict[str, type]`` → ``is_dict_form=True``, ``by_name=inputs``.
    - Single type → ``single_type=inputs``.
    """
    if inputs is None:
        return NormalizedInputs(is_none=True)
    if isinstance(inputs, dict):
        return NormalizedInputs(by_name=dict(inputs), is_dict_form=True)
    return NormalizedInputs(single_type=inputs)


def primary_output_field(base_field: str, outputs: Any) -> str:
    """State field that holds a node's PRIMARY output value.

    Single source of truth for the dict-form field-name resolution: for
    dict-form ``outputs`` the primary value lands on
    ``output_field_name(base_field, primary_key)``; single-type / ``None``
    outputs keep ``base_field`` unchanged. Replaces the
    ``if no.is_dict_form: output_field_name(base, no.primary_key)`` block that
    was repeated across the loop, oracle, and wiring read paths.
    """
    no = normalize_outputs(outputs)
    if no.is_dict_form:
        assert no.primary_key is not None  # dict-form always has a primary key
        return output_field_name(base_field, no.primary_key)
    return base_field


def _declared_output(item: ConstructItem) -> TypeSpecStatic:
    """Return an item's declared output type, abstracting the Node/Construct split.

    Single source of truth: ``Node`` declares ``.outputs`` (plural);
    ``Construct`` / ``_BranchNode`` declare ``.output`` (singular).
    Lives here (a neutral low-level module reachable from every layer, incl. the
    DX layer ``forward.py``) so no caller re-inlines the
    ``getattr(item, 'output', None)`` selector.
    """
    return item.outputs if isinstance(item, Node) else getattr(item, "output", None)


def _with_declared_io(item: _ItemT, **fields: object) -> _ItemT:
    """Apply NODE-level ``inputs``/``outputs`` surgery, as a no-op on a Construct.

    The write counterpart of ``_declared_output``, and it exists for the same
    reason: to keep the Node-vs-Construct decision in ONE place instead of
    re-inlining ``isinstance`` at every ``model_copy`` site.

    Why a Construct is a NO-OP rather than a route-to-``input``/``output``: a
    Node's ``inputs`` is a fan-in MAPPING derived from the caller's data edges,
    whereas a Construct's ``input``/``output`` is a single boundary PORT already
    restored from the exported sub-flow. They are not the same thing, so writing
    one into the other would be wrong, not merely redundant.

    This is fail-loud by construction. Pydantic's ``model_copy(update=...)`` does
    NOT validate: on a Construct it silently attaches a phantom ``outputs``
    attribute that is not in ``model_fields`` at all, so a missed skip produces a
    quietly malformed item instead of an error. Unknown keys are therefore
    rejected here rather than trusted to the model.
    """
    unknown = set(fields) - {"inputs", "outputs"}
    if unknown:
        raise ConfigurationError.build(
            "_with_declared_io writes only Node-level inputs/outputs",
            expected="'inputs' and/or 'outputs'",
            found=f"{sorted(unknown)}",
            hint="set a Construct's boundary via input=/output= at reconstruction, not through this helper",
        )
    if not isinstance(item, Node) or not fields:
        return item
    return item.model_copy(update=fields)
