"""lint() — validate DI bindings against a sample config.

Walks all nodes in a Construct and checks that every FromInput/FromConfig
parameter has a matching key in the provided config dict. Returns a list
of LintIssue dataclass instances (never raises — reports all problems).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neograph.construct import Construct
from neograph.decorators import _get_param_resolutions
from neograph.node import Node


@dataclass
class LintIssue:
    """A single DI binding problem found by lint()."""

    node_name: str
    param: str
    kind: str  # "from_input", "from_config", "from_input_model", "from_config_model"
    message: str
    required: bool = False


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

    param_res = _get_param_resolutions(item)
    if not param_res:
        return

    for pname, (kind, payload) in param_res.items():
        if kind in ("from_input", "from_config"):
            required = bool(payload)
            if config is not None:
                # Config provided — check the key exists
                if pname not in config:
                    issues.append(LintIssue(
                        node_name=item.name,
                        param=pname,
                        kind=kind,
                        required=required,
                        message=(
                            f"Node '{item.name}': DI parameter '{pname}' "
                            f"({kind}) not found in config"
                        ),
                    ))
            elif required:
                # No config provided — only flag required params
                issues.append(LintIssue(
                    node_name=item.name,
                    param=pname,
                    kind=kind,
                    required=True,
                    message=(
                        f"Node '{item.name}': required DI parameter '{pname}' "
                        f"({kind}) has no config to resolve from"
                    ),
                ))

        elif kind in ("from_input_model", "from_config_model"):
            model_cls, required = payload
            if config is not None:
                # Check each model field exists in config
                for fname in model_cls.model_fields:
                    if fname not in config:
                        issues.append(LintIssue(
                            node_name=item.name,
                            param=fname,
                            kind=kind,
                            required=required,
                            message=(
                                f"Node '{item.name}': bundled model field "
                                f"'{fname}' ({kind} via {model_cls.__name__}) "
                                f"not found in config"
                            ),
                        ))
            elif required:
                for fname in model_cls.model_fields:
                    issues.append(LintIssue(
                        node_name=item.name,
                        param=fname,
                        kind=kind,
                        required=True,
                        message=(
                            f"Node '{item.name}': required bundled model "
                            f"field '{fname}' ({kind} via "
                            f"{model_cls.__name__}) has no config"
                        ),
                    ))
        # Skip 'constant' and 'upstream' kinds — not DI bindings
