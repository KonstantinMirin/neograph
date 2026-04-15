"""lint() — validate DI bindings and template placeholders against config/inputs.

Walks all nodes in a Construct and checks:
1. Every FromInput/FromConfig parameter has a matching key in the config dict.
2. Every ${var} placeholder in inline prompts resolves to a known input key.

Returns a list of LintIssue dataclass instances (never raises — reports all problems).
"""

from __future__ import annotations

import re
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from neograph.construct import Construct
from neograph._sidecar import _get_param_res, get_merge_fn_metadata
from neograph.di import DIBinding, DIKind
from neograph.node import Node

# Standard keys always available in state / config
_KNOWN_EXTRAS: frozenset[str] = frozenset({
    "node_id", "project_root", "human_feedback",
})

# Matches ${var} and ${var.field} in inline prompts
_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


@dataclass
class LintIssue:
    """A single lint problem — DI binding or template placeholder."""

    node_name: str
    param: str
    kind: str
    message: str
    required: bool = False


def _check_binding(
    node_label: str,
    binding: DIBinding,
    config: dict[str, Any] | None,
    issues: list[LintIssue],
) -> None:
    """Check a single DI binding against config.

    ``node_label`` is pre-formatted by the caller — node and merge_fn paths
    use different naming conventions, so the caller supplies the label.
    """
    kind_str = binding.kind.value

    if binding.kind in (DIKind.FROM_INPUT, DIKind.FROM_CONFIG):
        if config is not None:
            if binding.name not in config:
                issues.append(LintIssue(
                    node_name=node_label,
                    param=binding.name,
                    kind=kind_str,
                    required=binding.required,
                    message=(
                        f"{node_label}: DI parameter '{binding.name}' "
                        f"({kind_str}) not found in config"
                    ),
                ))
        elif binding.required:
            issues.append(LintIssue(
                node_name=node_label,
                param=binding.name,
                kind=kind_str,
                required=True,
                message=(
                    f"{node_label}: required DI parameter '{binding.name}' "
                    f"({kind_str}) has no config to resolve from"
                ),
            ))

    elif binding.kind in (DIKind.FROM_INPUT_MODEL, DIKind.FROM_CONFIG_MODEL):
        model_cls: Any = binding.model_cls or binding.inner_type
        required = binding.required
        if config is not None:
            for fname in model_cls.model_fields:
                if fname not in config:
                    issues.append(LintIssue(
                        node_name=node_label,
                        param=fname,
                        kind=kind_str,
                        required=required,
                        message=(
                            f"{node_label}: bundled model field "
                            f"'{fname}' ({kind_str} via {model_cls.__name__}) "
                            f"not found in config"
                        ),
                    ))
        elif required:
            for fname in model_cls.model_fields:
                issues.append(LintIssue(
                    node_name=node_label,
                    param=fname,
                    kind=kind_str,
                    required=True,
                    message=(
                        f"{node_label}: required bundled model "
                        f"field '{fname}' ({kind_str} via "
                        f"{model_cls.__name__}) has no config"
                    ),
                ))


def lint(
    construct: Construct,
    *,
    config: dict[str, Any] | None = None,
    known_template_vars: set[str] | None = None,
    template_resolver: Callable[[str], str | None] | None = None,
) -> list[LintIssue]:
    """Validate DI bindings and template placeholders in *construct*.

    Walks every node (recursing into sub-constructs). Checks:
    1. FromInput/FromConfig parameters exist in the provided config dict.
    2. Inline prompt ``${var}`` placeholders resolve to known input keys.
    3. Template-ref prompt ``{placeholder}`` names resolve when a
       *template_resolver* is provided.

    *known_template_vars* is a set of extra variable names the consumer's
    prompt pipeline provides (e.g., ``{"topic", "json_schema"}``). These
    are accepted as valid alongside the standard framework extras
    (node_id, project_root, human_feedback).

    *template_resolver* maps a template name (e.g., ``"rw/summarize"``) to
    the template text string, or ``None`` if the template can't be found.
    When provided, lint reads the template text, extracts ``{placeholder}``
    names, and validates them against predicted input keys.

    Returns a list of LintIssue instances. An empty list means all bindings
    are satisfied.
    """
    issues: list[LintIssue] = []
    all_known = _KNOWN_EXTRAS | (known_template_vars or set())
    _walk(construct, config, issues, known_vars=all_known,
          template_resolver=template_resolver)
    return issues


def _walk(
    item: Construct | Node,
    config: dict[str, Any] | None,
    issues: list[LintIssue],
    *,
    known_vars: frozenset[str] | set[str] = _KNOWN_EXTRAS,
    template_resolver: Callable[[str], str | None] | None = None,
) -> None:
    """Recursively walk a construct and check DI bindings + template placeholders."""
    if isinstance(item, Construct):
        for child in item.nodes:
            _walk(child, config, issues, known_vars=known_vars,
                  template_resolver=template_resolver)
        return

    if not isinstance(item, Node):
        return

    param_res = _get_param_res(item)
    node_label = f"Node '{item.name}'"

    # 1. DI binding checks (existing)
    for binding in (param_res or {}).values():
        _check_binding(node_label, binding, config, issues)

    # Check merge_fn DI bindings for Oracle nodes.
    oracle = item.modifier_set.oracle
    if oracle is not None and isinstance(oracle.merge_fn, str):
        meta = get_merge_fn_metadata(oracle.merge_fn)
        if meta is not None:
            _, merge_param_res = meta
            merge_label = f"{item.name} merge_fn '{oracle.merge_fn}'"
            for binding in merge_param_res.values():
                _check_binding(merge_label, binding, config, issues)

    # 2. Template placeholder checks
    _check_template_placeholders(item, issues, known_vars=known_vars,
                                 template_resolver=template_resolver)


def _check_template_placeholders(
    node: Node,
    issues: list[LintIssue],
    *,
    known_vars: frozenset[str] | set[str],
    template_resolver: Callable[[str], str | None] | None = None,
) -> None:
    """Check that prompt placeholders resolve to known input keys.

    Two modes:
    - Inline prompts (space or ${} in prompt): extract ${var} placeholders.
    - Template-ref prompts (bare name like "rw/summarize"): if template_resolver
      is provided, read the template text and extract {placeholder} names.
    """
    prompt = node.prompt
    if not prompt or node.mode == "scripted":
        return

    is_inline = " " in prompt or "${" in prompt

    if is_inline:
        placeholders = _PLACEHOLDER_RE.findall(prompt)
    else:
        # Template-ref prompt — resolve text if resolver available
        if template_resolver is None:
            return
        text = template_resolver(prompt)
        if text is None:
            return
        placeholders = _extract_format_placeholders(text)

    if not placeholders:
        return

    predicted_keys = _predict_input_keys(node)
    valid_keys = predicted_keys | known_vars
    node_label = f"Node '{node.name}'"
    consumer_known = known_vars - _KNOWN_EXTRAS - predicted_keys
    placeholder_syntax = "${%s}" if is_inline else "{%s}"

    for placeholder in placeholders:
        first_segment = placeholder.split(".")[0]
        if first_segment not in valid_keys:
            issues.append(LintIssue(
                node_name=node_label,
                param=first_segment,
                kind="template_placeholder_unresolvable",
                required=True,
                message=(
                    f"{node_label}: prompt placeholder "
                    f"'{placeholder_syntax % first_segment}' "
                    f"not found in predicted input keys {sorted(predicted_keys)} "
                    f"or known extras {sorted(_KNOWN_EXTRAS)} "
                    f"(prompt: {prompt!r})"
                ),
            ))
        elif first_segment in consumer_known and first_segment not in predicted_keys and first_segment not in _KNOWN_EXTRAS:
            issues.append(LintIssue(
                node_name=node_label,
                param=first_segment,
                kind="template_placeholder_known_vars_only",
                required=False,
                message=(
                    f"{node_label}: placeholder "
                    f"'{placeholder_syntax % first_segment}' resolved only "
                    f"via known_vars — verify consumer bridge supplies it at runtime. "
                    f"Consider using the actual @node parameter name instead of a "
                    f"bridge alias."
                ),
            ))


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


def _predict_input_keys(node: Node) -> set[str]:
    """Predict the dict keys that _extract_input will produce for this node.

    For dict-form inputs: keys are the dict keys PLUS any flattened field
    names from ``render_for_prompt()`` return annotations on input types.
    This mirrors what ``_render_with_flattening`` produces at runtime.

    For single-type or None inputs: empty set (isinstance scan, no dict).
    """
    if node.inputs is None:
        return set()
    if isinstance(node.inputs, dict):
        keys = set(node.inputs.keys())
        for input_type in node.inputs.values():
            keys |= _get_flattened_field_names(input_type)
        return keys
    # Single-type inputs: no dict keys predictable
    return set()


def _get_flattened_field_names(input_type: Any) -> set[str]:
    """Extract field names from a type's render_for_prompt() return annotation.

    If the type has ``render_for_prompt`` with a return annotation that is a
    BaseModel subclass, returns the non-excluded field names of that model.
    Otherwise returns an empty set.
    """
    import typing

    from pydantic import BaseModel as _BM

    rfp = getattr(input_type, "render_for_prompt", None)
    if rfp is None:
        return set()

    ret_type = _resolve_return_type(rfp, input_type)
    if ret_type is None:
        return set()
    if not (isinstance(ret_type, type) and issubclass(ret_type, _BM)):
        return set()
    return {
        fname for fname, finfo in ret_type.model_fields.items()
        if not finfo.exclude
    }


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
    import typing

    # Fast path: get_type_hints works for module-scoped types
    try:
        hints = typing.get_type_hints(fn)
        return hints.get("return")
    except (NameError, AttributeError, TypeError):
        pass

    # Fallback: resolve string annotation from frame locals
    raw = getattr(fn, "__annotations__", {}).get("return")
    if raw is None or not isinstance(raw, str):
        return raw

    # Walk caller frames to find the name (handles test-local classes)
    frame = sys._getframe(0)
    for _ in range(10):
        frame = frame.f_back
        if frame is None:
            break
        if raw in frame.f_locals:
            return frame.f_locals[raw]
    return None
