"""lint() — validate DI bindings and template placeholders against config/inputs.

Walks all nodes in a Construct and checks:
1. Every FromInput/FromConfig parameter has a matching key in the config dict.
2. Every ${var} placeholder in inline prompts resolves to a known input key.

Returns a list of LintIssue dataclass instances (never raises — reports all problems).
"""

from __future__ import annotations

import re
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
) -> list[LintIssue]:
    """Validate DI bindings and template placeholders in *construct*.

    Walks every node (recursing into sub-constructs). Checks:
    1. FromInput/FromConfig parameters exist in the provided config dict.
    2. Inline prompt ``${var}`` placeholders resolve to known input keys.

    *known_template_vars* is a set of extra variable names the consumer's
    prompt pipeline provides (e.g., ``{"topic", "json_schema"}``). These
    are accepted as valid alongside the standard framework extras
    (node_id, project_root, human_feedback).

    Returns a list of LintIssue instances. An empty list means all bindings
    are satisfied.
    """
    issues: list[LintIssue] = []
    all_known = _KNOWN_EXTRAS | (known_template_vars or set())
    _walk(construct, config, issues, known_vars=all_known)
    return issues


def _walk(
    item: Construct | Node,
    config: dict[str, Any] | None,
    issues: list[LintIssue],
    *,
    known_vars: frozenset[str] | set[str] = _KNOWN_EXTRAS,
) -> None:
    """Recursively walk a construct and check DI bindings + template placeholders."""
    if isinstance(item, Construct):
        for child in item.nodes:
            _walk(child, config, issues, known_vars=known_vars)
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

    # 2. Template placeholder checks (new)
    _check_template_placeholders(item, issues, known_vars=known_vars)


def _check_template_placeholders(
    node: Node,
    issues: list[LintIssue],
    *,
    known_vars: frozenset[str] | set[str],
) -> None:
    """Check that inline prompt ${var} placeholders resolve to known input keys.

    Only checks inline prompts (containing space or ${}).
    Template-ref prompts are opaque — the consumer's prompt_compiler resolves them.
    """
    prompt = node.prompt
    if not prompt or node.mode == "scripted":
        return

    # Only check inline prompts
    if " " not in prompt and "${" not in prompt:
        return

    # Extract ${var} and ${var.field} placeholders
    placeholders = _PLACEHOLDER_RE.findall(prompt)
    if not placeholders:
        return

    # Predict runtime input keys
    predicted_keys = _predict_input_keys(node)

    # Validate each placeholder's first segment
    valid_keys = predicted_keys | known_vars
    node_label = f"Node '{node.name}'"

    for placeholder in placeholders:
        first_segment = placeholder.split(".")[0]
        if first_segment not in valid_keys:
            issues.append(LintIssue(
                node_name=node_label,
                param=first_segment,
                kind="template_placeholder_unresolvable",
                required=True,
                message=(
                    f"{node_label}: inline prompt placeholder '${{{{{{first_segment}}}}}}'  "
                    f"not found in predicted input keys {sorted(valid_keys)} "
                    f"(prompt: {prompt!r})"
                ).replace("{{", "{").replace("}}", "}"),
            ))


def _predict_input_keys(node: Node) -> set[str]:
    """Predict the dict keys that _extract_input will produce for this node.

    For dict-form inputs: keys are the dict keys.
    For single-type or None inputs: empty set (isinstance scan, no dict).
    """
    if node.inputs is None:
        return set()
    if isinstance(node.inputs, dict):
        return set(node.inputs.keys())
    # Single-type inputs: no dict keys predictable
    return set()
