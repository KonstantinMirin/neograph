"""Supply-side lint checks: every supplied value must reach a consumer.

The linter historically checked one direction of the dataflow. Every kind it
emitted reported a reference with no source, or a missing config value. None
reported a value that arrives and reaches nothing.

Three checks live here, all derived from IR the compiler already holds. None
asks the pipeline author for an annotation:

- ``_check_unreferenced_inputs`` -- a bound input, DI parameter, or context
  field that the node's own template never names (GH #10).

``_check_unconsumed_outputs`` (GH #11) moved to ``_lint_consumers`` when this
file crossed its size ceiling; the two split along the seam the checks already
had -- supply here, demand there.
- ``_check_unsatisfiable_di`` -- a ``FromInput`` bound to an Each item or a Loop
  carry, which no caller can satisfy (GH #12).
"""

from __future__ import annotations

from typing import Any

from neograph._lint_kind_registry import LintIssue
from neograph._lint_predict import (
    _di_resource_template_var_names,
    _di_template_var_names,
    _get_flattened_field_names,
)
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph._state_keys import StateKeys
from neograph.construct import Construct
from neograph.di import DI_TEMPLATE_KINDS
from neograph.modifiers import classify_modifiers
from neograph.node import Node


def _port_supplied_by_modifier(construct: Construct) -> tuple[type, str] | None:
    """The sub-construct's port type, when a modifier supplies its value.

    An ``Each`` fans one item into the port per branch and a ``Loop`` carries the
    previous result back into it. In both cases the value comes from the
    construct, never from the caller, so a ``FromInput`` on a parameter of that
    type can never be satisfied (GH #12).
    """
    port = getattr(construct, "input", None)
    if not isinstance(port, type):
        return None
    _, mods = classify_modifiers(construct)
    if "each" in mods:
        return (port, "Each-fanned item")
    if "loop" in mods:
        return (port, "Loop carry")
    return None


def _check_unsatisfiable_di(
    node: Node,
    issues: list[LintIssue],
    *,
    port_supplied: tuple[type, str] | None,
) -> None:
    """Report a DI binding that the enclosing modifier makes unsatisfiable.

    Decided from the construct's structure alone. It takes no config, so no
    fixture can silence it -- which is the point: the reported bug survived
    because the lint config had been padded with a key no caller could pass, and
    the gate then graded its own answer key (GH #12).
    """
    supplied: list[tuple[Any, str]] = []
    if port_supplied is not None:
        supplied.append(port_supplied)

    # A node-level self-loop carries the node's OWN output back as its input, so
    # that type is supplied by the construct too. `Construct.input` must be a
    # BaseModel, so a bare `list[X]` carry can only arise this way.
    _, node_mods = classify_modifiers(node)
    if "loop" in node_mods:
        carry = normalize_outputs(node.outputs).primary
        if carry is not None:
            supplied.append((carry, "Loop carry"))

    if not supplied:
        return
    bindings = getattr(node, "_param_res", None) or {}

    for param, binding in bindings.items():
        if binding.kind not in DI_TEMPLATE_KINDS:
            continue
        inner = getattr(binding, "model_cls", None) or getattr(binding, "inner_type", None)
        match = next(
            (
                (t, src)
                for t, src in supplied
                if inner is t or _unwrap_sequence(inner) is t
            ),
            None,
        )
        if match is None:
            continue
        port_type, source = match
        issues.append(
            LintIssue(
                node_name=node.name,
                param=param,
                kind="from_input_unsatisfiable",
                required=True,
                message=(
                    f"Node '{node.name}': parameter '{param}' is bound with "
                    f"{binding.kind.value} but its value is the {source}, which no "
                    f"caller can supply. The run fails in the DI preflight, and "
                    f"padding a config to silence this makes every branch compute "
                    f"from the padded value instead. Bind it as a PORT parameter: "
                    f"drop the DI marker and type it as the sub-construct's "
                    f"input={getattr(port_type, '__name__', port_type)}."
                ),
            )
        )


def _unwrap_sequence(annotation: Any) -> Any:
    """``list[X]`` -> ``X``; anything else unchanged. A Loop carry may be either."""
    from typing import get_args, get_origin

    if get_origin(annotation) in (list, tuple, set, frozenset):
        args = get_args(annotation)
        if args:
            return args[0]
    return annotation


def _placeholder_root(placeholder: str) -> str:
    """Root name a placeholder references. ``claims.items`` references ``claims``."""
    body = placeholder.split(":", 1)[1] if placeholder.startswith("image:") else placeholder
    return body.split(".")[0]


def _supply_axes(node: Node, *, is_inline: bool) -> list[tuple[str, str, set[str]]]:
    """Every name supplied to *node*, as ``(axis, name, aliases)`` triples.

    Reads the accessors the compiler already exposes. It derives nothing the IR
    does not hold, and it asks the pipeline author for no annotation.
    """
    axes: list[tuple[str, str, set[str]]] = []

    inputs = normalize_inputs(node.inputs)
    if inputs.is_dict_form:
        for key, declared in inputs.by_name.items():
            aliases = {key}
            if not is_inline:
                aliases |= _get_flattened_field_names(declared)
                if key == StateKeys.SUBGRAPH_INPUT and isinstance(declared, type):
                    aliases.add(declared.__name__)
            axes.append(("upstream input", key, aliases))
    elif not inputs.is_none and isinstance(inputs.single_type, type) and not is_inline:
        # A single-type input reaches an inline prompt as the bare value, so it
        # has no name to reference. Only template-ref prompts key it by type.
        name = inputs.single_type.__name__
        axes.append(("upstream input", name, {name}))

    for name in sorted(_di_template_var_names(node) | _di_resource_template_var_names(node)):
        axes.append(("DI parameter", name, {name}))

    for name in getattr(node, "context", None) or ():
        axes.append(("context field", name, {name}))

    return axes


def _check_unreferenced_inputs(
    node: Node,
    issues: list[LintIssue],
    *,
    referenced: set[str],
    is_inline: bool,
    consumer_known: frozenset[str] | set[str],
    node_label: str,
) -> None:
    """Report a bound name that the node's own template never references.

    The mirror of :func:`_check_template_placeholders`'s demand check. A node
    binds an input, the data arrives, and the prompt never names it, so the model
    never sees the value while every gate passes.

    Demand is inferred from the template TEXT. A custom ``prompt_compiler``
    composes the final message and can consume a name the resolved template never
    spells, so this check reports a warning rather than an error.

    Three suppressions, each read from the IR:

    - ``skip_when``: the predicate receives the extracted input dict on the think
      path and on the agent path, so an input it reads is real demand. The
      callable is opaque, which makes suppression the only sound disposition.
    - An Oracle ``merge_prompt``: ``_build_upstream_context`` injects one entry
      per ``node.inputs`` key into that prompt, so it is a second demand surface
      over the same keys.
    - A bridge alias: when the template names a ``known_vars``-only placeholder,
      the consumer supplies names this walk cannot see. Scoped to the inputs
      axis, since a bridge alias cannot explain an unreferenced context field.
    """
    oracle = getattr(getattr(node, "modifier_set", None), "oracle", None)
    inputs_suppressed = (
        getattr(node, "skip_when", None) is not None
        or (oracle is not None and getattr(oracle, "merge_prompt", None))
        or bool(referenced & consumer_known)
    )

    fan_out = getattr(node, "fan_out_param", None)
    shown = ", ".join(sorted(referenced)) or "none"

    for axis, name, aliases in _supply_axes(node, is_inline=is_inline):
        if name == fan_out or aliases & referenced:
            continue
        if axis == "upstream input" and inputs_suppressed:
            continue
        issues.append(
            LintIssue(
                node_name=node.name,
                param=name,
                kind="template_input_unreferenced",
                required=False,
                message=(
                    f"{node_label}: {axis} '{name}' is bound but the node's "
                    f"template references none of {sorted(aliases)}. The value "
                    f"reaches the node and the model never sees it. Template "
                    f"references: {shown}. Demand is read from the template "
                    f"text, so a prompt_compiler that composes the message may "
                    f"consume this name without naming it."
                ),
            )
        )


