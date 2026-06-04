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
from typing import Any

from neograph._ir_protocols import ConstructItem
from neograph.naming import output_field_name
from neograph.node import Node, TypeSpecStatic


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
