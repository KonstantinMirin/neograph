"""Null-default coercion — repairing model output that says "null" in words.

Extracted from ``_llm_retry.py`` for neograph-5s8f6 as a pure file split — the
functions below are unchanged, only their home moved. ``_llm_retry.py``
re-exports them, so existing imports keep resolving.

Two callers, ONE coercion. A Pydantic default applies only when a key is
ABSENT; a present-and-null value overrides it and fails validation, and models
emit present-and-null routinely. :func:`_apply_null_defaults` repairs that
before validation on the json_mode path (``_parse_json_response``), and
:func:`recover_null_defaults` applies the SAME function to the payload a
structured-output provider already emitted, so the two output strategies agree
about what a declared output can be (the house rule from GH #14).

Some providers also emit the STRING "null" for an Optional field rather than a
real null. The descent here is shape-driven: one recursive classifier reaches
every leaf model dict at any depth, which is what replaced the hand-enumerated
(bare-model, bare-list-of-model) walk that silently skipped every container
shape it did not spell out.
"""

from __future__ import annotations

import copy
import json as _json
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined


def _is_list_annotation(annotation: Any) -> bool:
    """Check if a type annotation is a list type (list[X], List[X], etc.)."""
    import typing

    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return True
    if origin is typing.Union:
        args = getattr(annotation, "__args__", ())
        return any(_is_list_annotation(a) for a in args if a is not type(None))
    return annotation is list


_STRINGLY_NULL = frozenset({"null", "none", "nil", "n/a", "na"})


def _optional_inner_types(annotation: Any) -> tuple[Any, ...] | None:
    """Non-None member types of an Optional annotation, else None.

    Returns the tuple of non-``NoneType`` members when *annotation* is a Union
    that admits ``None`` (``int | None``, ``Optional[Enum]``, ``X | Y | None``);
    returns ``None`` when the field is not nullable (so callers leave it alone).
    """
    import types
    from typing import Union, get_args, get_origin

    origin = get_origin(annotation)
    if origin is Union or origin is getattr(types, "UnionType", ()):
        args = get_args(annotation)
        if type(None) in args:
            return tuple(a for a in args if a is not type(None))
    return None


def _unwrap_optional(annotation: Any) -> Any:
    """Peel a single ``X | None`` wrapper, returning ``X``; else the annotation.

    ``Company | None`` -> ``Company``; ``list[Product] | None`` -> ``list[Product]``;
    a plain ``list[Product]`` or ``Company`` is returned unchanged. An ambiguous
    nullable union with more than one non-None member (``A | B | None``) is left
    intact -- there is no single interior type to recurse into, so the caller's
    ``issubclass``/``origin`` checks correctly decline it.

    This is the single Optional-unwrapping seam for the nested-recursion descent:
    without it, ``isinstance(annotation, type)`` and ``get_origin(...) is list``
    both reject an Optional-wrapped model/list and the descent silently skips the
    interior. See neograph-zhwgh.
    """
    non_none = _optional_inner_types(annotation)
    if non_none is not None and len(non_none) == 1:
        return non_none[0]
    return annotation


def _is_stringly_null(val: Any, annotation: Any) -> bool:
    """True when *val* is a string sentinel meaning "no value" for a nullable field.

    LLMs (GLM 5.2) intermittently emit the *string* ``"null"`` (or ``"none"``,
    ``""``) for Optional numeric/enum/bool fields instead of a JSON ``null``.
    json_repair leaves the string intact and Pydantic then rejects it
    (``int_parsing`` / ``enum``), aborting the node. We coerce the sentinel to
    ``None`` — but ONLY when the field is Optional, so a legitimately-typed
    ``str`` value is never destroyed. The empty string is treated as a sentinel
    only when the field cannot itself be a plain ``str`` (where ``""`` is valid).
    """
    if not isinstance(val, str):
        return False
    non_none = _optional_inner_types(annotation)
    if non_none is None:
        return False
    low = val.strip().lower()
    if low in _STRINGLY_NULL:
        return True
    return low == "" and not any(t is str for t in non_none)


def _sequence_item_annotation(annotation: Any) -> Any | None:
    """The ONE element annotation of a homogeneous sequence, else ``None``.

    JSON decodes every array to a Python ``list``, so which interiors the
    descent can reach is decided by the ANNOTATION, not by the runtime
    container: ``list[X]``, ``set[X]``, ``frozenset[X]``, the variadic
    ``tuple[X, ...]`` and a fixed ``tuple[X, X]`` all have a single element type
    and are descended.

    ``tuple[A, B]`` has none, and is left alone. That heterogeneous case is what
    the original "tuples are out of scope" exclusion was actually about --
    ``tuple[X, ...]`` is a list with a different constructor, and it is what a
    FROZEN Pydantic domain model uses, which is the same reason 0.7.9 widened
    ``Each`` to accept it. neograph-sjwny.
    """
    from typing import get_args, get_origin

    if get_origin(annotation) not in (list, set, frozenset, tuple):
        return None
    args = [a for a in get_args(annotation) if a is not Ellipsis]
    if not args:
        return None
    return args[0] if all(a == args[0] for a in args) else None


def _descend_null_defaults(val: Any, annotation: Any) -> None:
    """Recurse :func:`_apply_null_defaults` into any BaseModel dict nested within
    *val*, driven by *annotation*'s container shape.

    ONE shape classifier instead of a per-shape branch list. It peels a single
    ``Optional`` wrapper, then dispatches on the concrete runtime value:

    - a model dict (``val`` is a dict, annotation a ``BaseModel``) -> recurse into it;
    - a ``list[...]`` -> recurse into each element against the item annotation;
    - a ``dict[K, V]`` -> recurse into each value against ``V``.

    The recursion re-peels ``Optional`` at every level, so *every* container
    composition -- ``list[Product] | None``, ``dict[str, Product]``,
    ``list[Product | None]``, ``Optional[dict[str, list[Product]]]`` -- reaches
    its leaf model dicts. This replaced a hand-enumerated (bare-model,
    bare-list-of-model) descent that silently skipped every shape it did not
    spell out, which is how the Optional-wrapped and dict-of-model interiors
    kept crashing. See neograph-zhwgh. (``tuple[...]`` is intentionally out of
    scope -- LLM structured output does not emit heterogeneous tuples.)
    """
    from typing import get_args, get_origin

    annotation = _unwrap_optional(annotation)

    if isinstance(val, dict):
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            _apply_null_defaults(val, annotation)
            return
        if get_origin(annotation) is dict:
            args = get_args(annotation)
            if len(args) == 2:
                for item in val.values():
                    _descend_null_defaults(item, args[1])
        return

    if isinstance(val, list):
        item_annotation = _sequence_item_annotation(annotation)
        if item_annotation is not None:
            for item in val:
                _descend_null_defaults(item, item_annotation)


def _apply_null_defaults(data: dict, model: type[BaseModel]) -> None:
    """Replace null values with field defaults, recursively.

    Mutates *data* in place. Applies when the JSON value is None and the field
    has either an explicit default or a default_factory. Also recurses (via
    :func:`_descend_null_defaults`) into nested BaseModel fields nested within
    ``list`` / ``dict`` / ``Optional`` containers to any depth. Stringly-null
    sentinels (the string ``"null"``/``"none"``/``""``) on Optional fields are
    first coerced to ``None`` so the same default/None disposition applies.
    """
    for field_name, field_info in model.model_fields.items():
        if field_name not in data:
            continue
        val = data[field_name]

        # GLM emits the STRING "null" for Optional numeric/enum fields; normalize
        # it to a real None BEFORE the null-disposition branches below run.
        if _is_stringly_null(val, field_info.annotation):
            data[field_name] = val = None

        if val is None and field_info.default is not PydanticUndefined:
            data[field_name] = field_info.default
            continue

        # LLMs emit null for default_factory list/dict fields (their default is
        # PydanticUndefined, so the branch above skips them). Coerce to the
        # factory result. Zero-arg first: a data-accepting factory (Pydantic
        # 2.10+) raises TypeError -> factory(data); a zero-arg one like list must
        # NOT get the data dict (list(data) returns keys, not []). neograph-s1u4.
        if val is None and field_info.default_factory is not None:
            # default_factory is typed as a union (zero-arg | data-accepting);
            # the try/except resolves the arity at runtime, so both calls need
            # the call-arg ignore.
            factory = field_info.default_factory
            try:
                data[field_name] = factory()  # type: ignore[call-arg]
            except TypeError:
                data[field_name] = factory(data)  # type: ignore[call-arg]
            continue

        # Recurse into nested model dicts wherever they sit -- directly, under an
        # Optional wrapper, or inside a list/dict container -- so stringly-null
        # interiors are normalized at every depth. ``val`` is a concrete
        # dict/list here (the None branches above returned).
        _descend_null_defaults(val, field_info.annotation)


def recover_null_defaults(raw_msg: Any, model: Any) -> BaseModel | None:
    """Re-validate a structured-output payload after the null-default coercion.

    The ``structured`` strategy hands validation to the provider adapter, so a
    present-and-null value that overrides a field default surfaces as a
    ``ValidationError`` its caller can only re-prompt — while ``json_mode``,
    parsing the same bytes itself, repairs it and succeeds. This closes that
    disagreement by applying the SAME :func:`_apply_null_defaults` to the
    payload the provider ALREADY produced. No re-prompt, no extra round-trip.

    Reads the payload wherever the provider put it: ``tool_calls[*]["args"]``
    for ``method="function_calling"`` (whose message content is empty), else the
    message content parsed as strict JSON. Strict, never ``repair_json``: a
    constrained decode emits machine-produced JSON, and running the repairer
    over arbitrary text here could manufacture a payload out of prose.

    Returns ``None`` — leaving the caller's re-prompt path exactly as it was —
    unless the coercion ACTUALLY changed the payload AND the changed payload
    validates. That guard is what stops an unrelated validation failure from
    being silently reclassified as a success.
    """
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        return None

    candidates: list[dict] = []
    for call in getattr(raw_msg, "tool_calls", None) or []:
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
        if isinstance(args, dict):
            candidates.append(args)
    content = getattr(raw_msg, "content", None)
    if isinstance(content, str) and content.strip():
        try:
            loaded = _json.loads(content)
        except ValueError:
            loaded = None
        if isinstance(loaded, dict):
            candidates.append(loaded)

    for payload in candidates:
        # Same exception set json_mode's coercion site swallows (ValueError /
        # TypeError from a hostile default_factory) — parity, not extra hardening.
        try:
            repaired = copy.deepcopy(payload)
            _apply_null_defaults(repaired, model)
            if repaired == payload:
                continue
            return model.model_validate(repaired)
        except (ValidationError, ValueError, TypeError):
            continue
    return None
