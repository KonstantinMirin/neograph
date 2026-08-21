"""DI-binding lint: what a caller must supply, and what no caller can.

Split out of ``lint.py`` so the module stays under its size ceiling, following
the ``_lint_*`` decomposition this package already uses.

Two surfaces live here, and the split between them is the point:

- ``input_contract(construct)`` answers "what must a caller supply?". A correct
  graph has an answer, so the answer is not a problem and does not travel in
  ``lint()``'s list.
- ``_check_binding`` answers "does THIS payload satisfy the graph?". Only a
  caller who passes ``config=`` is asking, and only an unsatisfied binding is a
  defect.

``lint()`` used to report the first as low-severity issues. That left a correct
graph reporting issues, which forbids an all-output-fails gate and pushes a
consumer onto "fails on required only" -- the same trust-the-classification
posture a padded config exploits (GH #12, GH #13).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neograph._ir_branch import iter_with_arms
from neograph._lint_kind_registry import LintIssue
from neograph._sidecar import _get_param_res, get_merge_fn_metadata
from neograph._state_keys import StateKeys
from neograph.construct import Construct
from neograph.di import DI_CALLER_SUPPLIED_KINDS, DIBinding, DIKind
from neograph.node import Node

if TYPE_CHECKING:
    from neograph.construct import ConstructItem

__all__ = ["InputBinding", "input_contract"]

# Keys a caller may legitimately put in `config['configurable']` that no DI
# binding names: the framework extras neograph itself supplies, and
# LangGraph's own run identifiers. Anything `neo_`/`_neo_`-prefixed is a
# framework channel (StateKeys.FRAMEWORK_PREFIX) and is matched by prefix
# rather than listed, so a new one cannot be forgotten here.
_CONFIG_FRAMEWORK_KEYS: frozenset[str] = frozenset(
    {
        StateKeys.NODE_ID,
        StateKeys.PROJECT_ROOT,
        StateKeys.HUMAN_FEEDBACK,
        "thread_id",
        "checkpoint_id",
        "checkpoint_ns",
        "run_id",
        "run_name",
    }
)

_MODEL_KINDS = (DIKind.FROM_INPUT_MODEL, DIKind.FROM_CONFIG_MODEL)


@dataclass(frozen=True)
class InputBinding:
    """One key a caller supplies at run time -- part of the graph's input contract.

    Deliberately NOT a ``LintIssue`` and deliberately not a subclass of one: a
    record that is still an issue would keep travelling in the issue list and
    defeat the separation this type exists to make.
    """

    node_name: str
    """The node (or merge_fn) that consumes the value, pre-formatted as a label."""

    param: str
    """The config key a caller supplies. For a bundled model this is the FIELD name."""

    kind: str
    """The ``DIKind`` value: from_input, from_config, from_input_model, from_config_model."""

    source: str
    """Where the caller passes it: ``"input"`` for ``run(input={...})``, ``"config"`` for ``config=``."""

    type_name: str
    """Display name of the expected type."""

    required: bool
    """False when the parameter has a default, so a caller may omit it."""

    model_name: str | None = None
    """The bundling ``BaseModel`` when this key is one of its fields, else None."""


def _type_name(annotation: Any) -> str:
    """Display name for an annotation, falling back to ``str`` for generics."""
    return getattr(annotation, "__name__", None) or str(annotation)


def iter_di_bindings(item: ConstructItem) -> Iterator[tuple[str, DIBinding]]:
    """Every DI binding under *item*, paired with its pre-formatted node label.

    SINGLE SOURCE for "where do DI bindings live". Both readers -- ``lint()``'s
    payload check and ``input_contract()`` -- walk this, so a new binding site
    reaches both surfaces or neither. The two used to enumerate bindings
    separately, which is how a surface drifts.

    Recurses sub-constructs through ``iter_with_arms`` so a node inside a branch
    arm is reached, matching ``lint()``'s own traversal.
    """
    if isinstance(item, Construct):
        for child in iter_with_arms(item):
            yield from iter_di_bindings(child)
        return
    if not isinstance(item, Node):
        return

    node_label = f"Node '{item.name}'"
    for binding in (_get_param_res(item) or {}).values():
        yield node_label, binding

    # Oracle merge_fn bindings are resolved from the same config, so they are
    # part of the same contract and the same payload check.
    oracle = item.modifier_set.oracle
    if oracle is not None and isinstance(oracle.merge_fn, str):
        meta = get_merge_fn_metadata(oracle.merge_fn)
        if meta is not None:
            _, merge_param_res = meta
            merge_label = f"{item.name} merge_fn '{oracle.merge_fn}'"
            for binding in merge_param_res.values():
                yield merge_label, binding


def input_contract(construct: Construct) -> list[InputBinding]:
    """Every key a caller supplies to run *construct*.

    This is the positively-framed counterpart to ``lint()``: a correct graph has
    an input contract, so reporting one says nothing is wrong. Read it to
    document a graph's entry points, to build a config, or to check a payload
    against something other than ``lint()``.

    A bundled ``BaseModel`` parameter expands to one entry per FIELD, because a
    field name is what a caller actually puts in the config.

    ``FromResource`` and ``FromState`` parameters are absent: the framework
    resolves those, not the caller.
    """
    contract: list[InputBinding] = []
    for node_label, binding in iter_di_bindings(construct):
        if binding.kind not in DI_CALLER_SUPPLIED_KINDS:
            continue
        source = "input" if binding.kind in (DIKind.FROM_INPUT, DIKind.FROM_INPUT_MODEL) else "config"
        if binding.kind in _MODEL_KINDS:
            model_cls: Any = binding.model_cls or binding.inner_type
            for fname, field in model_cls.model_fields.items():
                contract.append(
                    InputBinding(
                        node_name=node_label,
                        param=fname,
                        kind=binding.kind.value,
                        source=source,
                        type_name=_type_name(field.annotation),
                        required=binding.required and field.is_required(),
                        model_name=model_cls.__name__,
                    )
                )
        else:
            contract.append(
                InputBinding(
                    node_name=node_label,
                    param=binding.name,
                    kind=binding.kind.value,
                    source=source,
                    type_name=_type_name(binding.inner_type),
                    required=binding.required,
                )
            )
    return contract


def _check_binding(
    node_label: str,
    binding: DIBinding,
    config: dict[str, Any] | None,
    issues: list[LintIssue],
) -> None:
    """Check a single DI binding against a caller-supplied *config*.

    With no config there is nothing to check: a ``FromInput``/``FromConfig``
    parameter is the graph's INPUT CONTRACT, which ``input_contract()`` reports
    positively. Demanding a config to reach a clean gate is what pushed one
    consumer to pad a fixture with a key no caller could pass, which silenced a
    real defect (GH #12, GH #13).

    ``node_label`` is pre-formatted by the caller -- node and merge_fn paths use
    different naming conventions, so the caller supplies the label.
    """
    if config is None:
        return

    kind_str = binding.kind.value

    if binding.kind in (DIKind.FROM_INPUT, DIKind.FROM_CONFIG):
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

    elif binding.kind in _MODEL_KINDS:
        model_cls: Any = binding.model_cls or binding.inner_type
        for fname in model_cls.model_fields:
            if fname not in config:
                issues.append(
                    LintIssue(
                        node_name=node_label,
                        param=fname,
                        kind=kind_str,
                        required=binding.required,
                        message=(
                            f"{node_label}: bundled model field "
                            f"'{fname}' ({kind_str} via {model_cls.__name__}) "
                            f"not found in config"
                        ),
                    )
                )


def _check_unmatched_config_keys(
    construct: Construct,
    config: dict[str, Any] | None,
    issues: list[LintIssue],
) -> None:
    """Report a config key that no binding in *construct* can consume.

    A key is accepted because a binding names it, never because it is present.
    Honouring an unmatched key is what makes a padded lint config a working
    silencer: the linter reports a binding no caller can satisfy, someone adds
    the demanded key to make the message stop, and the linter then agrees with a
    description of the world it was handed rather than with the graph (GH #12).

    ERROR, deliberately. A WARN leaves the hatch open for any gate that keys on
    ``required``, which is the gate a consumer falls back to.
    """
    if not config:
        return

    consumable = {binding.param for binding in input_contract(construct)}
    for key in config:
        if key in consumable or key in _CONFIG_FRAMEWORK_KEYS:
            continue
        if key.startswith(StateKeys.FRAMEWORK_PREFIX) or key.startswith("_" + StateKeys.FRAMEWORK_PREFIX):
            continue
        issues.append(
            LintIssue(
                node_name=construct.name,
                param=key,
                kind="config_key_unmatched",
                required=True,
                message=(
                    f"{construct.name}: config key '{key}' matches no DI binding in this "
                    f"construct, so nothing reads it. A key accepted because it is present "
                    f"rather than because a binding names it is how a padded config silences "
                    f"a real unsatisfiable binding. Remove it, or bind a parameter that "
                    f"consumes it."
                ),
            )
        )
