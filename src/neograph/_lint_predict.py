"""Pure predictors used by the lint template checks.

Extracted from ``lint.py`` (neograph-3ffdg.10) as a pure file split — the
functions below are unchanged, only their home moved. They emit no LintIssue:
each one answers a question about a node (what keys will it see at runtime, what
template vars its DI params expose, what its return type resolves to) and the
callers decide whether that is a problem.
"""

from __future__ import annotations

import string
from typing import Any

import structlog

from neograph._normalize import normalize_inputs
from neograph._sidecar import _get_param_res
from neograph._state_keys import StateKeys
from neograph.di import DI_TEMPLATE_KINDS, DIKind
from neograph.node import Node

log = structlog.get_logger()


def _extract_format_placeholders(text: str) -> list[str]:
    """Extract {placeholder} names from Python str.format-style template text.

    Returns a list of field names (may include dotted paths like 'claim.text').
    Skips empty/None field names (literal braces, positional args).
    """
    formatter = string.Formatter()
    names = []
    for _, field_name, _, _ in formatter.parse(text):
        if field_name is not None and field_name != "":
            names.append(field_name)
    return names


def _di_template_var_names(node: Node) -> set[str]:
    """The node's FromInput/FromConfig parameter names usable as template vars.

    These become valid template-ref placeholders when the prompt_compiler
    accepts ``di_inputs`` (the dispatch layer resolves them and the compiler
    binds them by parameter name). Bundled-model kinds contribute the bundle's
    parameter name (matching the first segment of a dotted ``{ctx.field}``).
    """
    param_res = _get_param_res(node)
    return {name for name, binding in (param_res or {}).items() if binding.kind in DI_TEMPLATE_KINDS}


def _di_resource_template_var_names(node: Node) -> set[str]:
    """The node's FromResource parameter names usable as template vars. See neograph-3q6j.

    Kept separate from ``_di_template_var_names`` because a FROM_RESOURCE var
    resolves ONLY on the async arun() driver (its fetch is awaited): it is a VALID
    template-ref placeholder when the compiler accepts ``di_inputs`` — the async
    injector twin stashes the fetched value — but the lint layer flags it with an
    async-driver WARN, in lockstep with the sync-driver fail-loud at runtime.
    """
    param_res = _get_param_res(node)
    return {name for name, binding in (param_res or {}).items() if binding.kind is DIKind.FROM_RESOURCE}


def _predict_input_keys(node: Node, *, include_flattened: bool = True) -> set[str]:
    """Predict the dict keys that _extract_input will produce for this node.

    For dict-form inputs: keys are the dict keys. When *include_flattened* is
    True (default), also adds flattened field names from ``render_for_prompt()``
    return annotations — these are available for template-ref prompts where
    ``_render_with_flattening`` runs. Set *include_flattened=False* for inline
    prompts, which skip flattening and only see raw input dict keys.

    For single-type inputs: the value's TYPE NAME, which is the key the runtime
    now produces for a bare value (``RenderedInput.for_template_ref``). This must
    stay in lockstep with that keying: before neograph-l2a7w a bare value reached
    the compiler unkeyed and lint correctly predicted no keys, so leaving this at
    ``set()`` would flag a valid ``{MyInput}`` placeholder as unresolvable on
    every single-type think node.

    For None inputs: empty set — there is nothing to name.
    """
    ni = normalize_inputs(node.inputs)
    if ni.is_none:
        return set()
    if ni.is_dict_form:
        keys = set(ni.by_name.keys())
        if include_flattened:
            for input_type in ni.by_name.values():
                keys |= _get_flattened_field_names(input_type)
            # Sub-construct port alias (neograph-bluv, F3.4): mirrors
            # renderers._alias_subgraph_input_port exactly, so lint's
            # template-ref prediction and the runtime alias cannot drift.
            port_type = ni.by_name.get(StateKeys.SUBGRAPH_INPUT)
            if port_type is not None:
                keys.add(port_type.__name__)
        return keys
    # Single-type: the runtime keys a bare value by its type name. Inline prompts
    # address the value's ATTRIBUTES (`${var.field}`) against the raw object and
    # never see this synthesized key, so it is template-ref only — the same
    # asymmetry the flattened fields already have.
    single = ni.single_type
    if include_flattened and isinstance(single, type):
        return {single.__name__}
    return set()


def _get_flattened_field_names(input_type: Any) -> set[str]:
    """Extract field names from a type's render_for_prompt() return annotation.

    If the type has ``render_for_prompt`` with a return annotation that is a
    BaseModel subclass, returns the non-excluded field names of that model.
    Otherwise returns an empty set.
    """

    from pydantic import BaseModel as _BM

    rfp = getattr(input_type, "render_for_prompt", None)
    if rfp is None:
        return set()

    ret_type = _resolve_return_type(rfp, input_type)
    if ret_type is None:
        return set()
    if not (isinstance(ret_type, type) and issubclass(ret_type, _BM)):
        return set()
    return {fname for fname, finfo in ret_type.model_fields.items() if not finfo.exclude}


def _resolve_return_type(fn: Any, owner_cls: Any) -> Any:
    """Resolve the return type annotation of a method.

    ``from __future__ import annotations`` turns annotations into strings.
    ``typing.get_type_hints`` resolves them from ``fn.__globals__`` but fails
    when the return type is defined in a local scope (e.g., inside a test).

    Fallback: scan the caller's frame stack (up to 10 frames) for the name.
    This mirrors the technique Pydantic and neograph's ``_di_classify.py``
    use for forward-ref resolution.
    """
    import sys
    import types
    import typing

    # Fast path: get_type_hints works for module-scoped types
    try:
        hints = typing.get_type_hints(fn)
        return hints.get("return")
    except (NameError, AttributeError, TypeError) as exc:
        # A real frame-walking fallback follows, so this is not fatal — but the
        # primary-path failure is otherwise invisible, turning a lint
        # false-negative into an undiagnosable one. Leave a breadcrumb naming
        # the function whose return hint failed to resolve.
        log.debug(
            "return_hint_resolution_failed",
            fn=getattr(fn, "__qualname__", None) or repr(fn),
            owner=getattr(owner_cls, "__qualname__", None) or repr(owner_cls),
            error=str(exc),
        )

    # Fallback: resolve string annotation from frame locals
    raw = getattr(fn, "__annotations__", {}).get("return")
    if raw is None or not isinstance(raw, str):
        return raw

    # Walk caller frames to find the name (handles test-local classes)
    frame: types.FrameType | None = sys._getframe(0)
    for _ in range(10):
        frame = frame.f_back if frame is not None else None
        if frame is None:
            break
        if raw in frame.f_locals:
            return frame.f_locals[raw]
    return None
