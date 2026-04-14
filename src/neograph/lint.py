"""lint() — validate DI bindings against a sample config.

Walks all nodes in a Construct and checks that every FromInput/FromConfig
parameter has a matching key in the provided config dict. Returns a list
of LintIssue dataclass instances (never raises — reports all problems).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neograph.construct import Construct
from neograph._sidecar import _get_param_res, get_merge_fn_metadata
from neograph.di import DIBinding, DIKind
from neograph.node import Node


@dataclass
class LintIssue:
    """A single DI binding problem found by lint()."""

    node_name: str
    param: str
    kind: str  # "from_input", "from_config", "from_input_model", "from_config_model"
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
) -> list[LintIssue]:
    """Validate DI bindings in *construct* against *config*.

    Walks every node (recursing into sub-constructs). For each node with
    FromInput/FromConfig parameters, verifies the param name (or model
    fields for bundled BaseModel params) exist in the provided config dict.

    When *config* is None, only structural checks are performed: required=True
    params are flagged as missing since no config is available.

    Returns a list of LintIssue instances. An empty list means all bindings
    are satisfied.
    """
    issues: list[LintIssue] = []
    _walk(construct, config, issues)
    return issues


def _walk(
    item: Construct | Node,
    config: dict[str, Any] | None,
    issues: list[LintIssue],
) -> None:
    """Recursively walk a construct and check DI bindings."""
    if isinstance(item, Construct):
        for child in item.nodes:
            _walk(child, config, issues)
        return

    if not isinstance(item, Node):
        return

    param_res = _get_param_res(item)
    # Don't return early — merge_fn DI check below doesn't depend on node param_res.

    node_label = f"Node '{item.name}'"
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
