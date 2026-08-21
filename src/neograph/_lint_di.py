"""DI-binding lint: what a caller must supply, and what no caller can.

Split out of ``lint.py`` so the module stays under its size ceiling, following
the ``_lint_*`` decomposition this package already uses.

With no config a ``FromInput``/``FromConfig`` parameter is the graph's INPUT
CONTRACT, reported informationally rather than as an error. Demanding a config
to reach a clean gate is what pushed one consumer to pad a fixture with a key no
caller could pass, which silenced a real defect (GH #13).
"""

from __future__ import annotations

from typing import Any

from neograph._lint_kind_registry import LintIssue
from neograph.di import DIBinding, DIKind


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
                issues.append(
                    LintIssue(
                        node_name=node_label,
                        param=binding.name,
                        kind=kind_str,
                        required=binding.required,
                        message=(f"{node_label}: DI parameter '{binding.name}' ({kind_str}) not found in config"),
                    )
                )
        elif binding.required:
            # No config supplied. This parameter is part of the graph's INPUT
            # CONTRACT: a caller supplies it at run time, so reporting it as an
            # error says only that the graph has inputs. Requiring a config to
            # reach a clean gate is what pushed one consumer to pad the fixture
            # with a key no caller could pass, which silenced a real
            # unsatisfiable binding (GH #12, GH #13).
            issues.append(
                LintIssue(
                    node_name=node_label,
                    param=binding.name,
                    kind=kind_str,
                    required=False,
                    message=(
                        f"{node_label}: DI parameter '{binding.name}' ({kind_str}) "
                        f"is part of this graph's input contract -- a caller supplies "
                        f"it at run time. Pass config= to check a specific payload."
                    ),
                )
            )

    elif binding.kind in (DIKind.FROM_INPUT_MODEL, DIKind.FROM_CONFIG_MODEL):
        model_cls: Any = binding.model_cls or binding.inner_type
        required = binding.required
        if config is not None:
            for fname in model_cls.model_fields:
                if fname not in config:
                    issues.append(
                        LintIssue(
                            node_name=node_label,
                            param=fname,
                            kind=kind_str,
                            required=required,
                            message=(
                                f"{node_label}: bundled model field "
                                f"'{fname}' ({kind_str} via {model_cls.__name__}) "
                                f"not found in config"
                            ),
                        )
                    )
        elif required:
            # Same input-contract reasoning as the scalar branch above.
            for fname in model_cls.model_fields:
                issues.append(
                    LintIssue(
                        node_name=node_label,
                        param=fname,
                        kind=kind_str,
                        required=False,
                        message=(
                            f"{node_label}: bundled model field '{fname}' "
                            f"({kind_str} via {model_cls.__name__}) is part of "
                            f"this graph's input contract -- a caller supplies it "
                            f"at run time. Pass config= to check a payload."
                        ),
                    )
                )


