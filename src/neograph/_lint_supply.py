"""Supply-side lint checks: every supplied value must reach a consumer.

The linter historically checked one direction of the dataflow. Every kind it
emitted reported a reference with no source, or a missing config value. None
reported a value that arrives and reaches nothing.

Three checks live here, all derived from IR the compiler already holds. None
asks the pipeline author for an annotation:

- ``_check_unreferenced_inputs`` -- a bound input, DI parameter, or context
  field that the node's own template never names (GH #10).
- ``_check_unconsumed_outputs`` -- a produced field that nothing reads (GH #11).
- ``_check_unsatisfiable_di`` -- a ``FromInput`` bound to an Each item or a Loop
  carry, which no caller can satisfy (GH #12).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from neograph._ir_branch import iter_with_arms
from neograph._lint_kind_registry import LintIssue
from neograph._lint_predict import (
    _di_resource_template_var_names,
    _di_template_var_names,
    _extract_format_placeholders,
    _get_flattened_field_names,
)
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph._placeholders import DOLLAR_RE as _PLACEHOLDER_RE
from neograph._state_keys import StateKeys
from neograph.construct import Construct
from neograph.di import DI_TEMPLATE_KINDS
from neograph.modifiers import classify_modifiers
from neograph.naming import field_name_for, output_field_name
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


def _template_roots(node: Node, template_resolver: Any) -> set[str]:
    """Every ROOT name a node's template references, dotted or bare."""
    prompt = node.prompt
    if not prompt or node.mode == "scripted":
        return set()
    if " " in prompt or "${" in prompt:
        placeholders = _PLACEHOLDER_RE.findall(prompt)
    else:
        if template_resolver is None:
            return set()
        text = template_resolver(prompt)
        if text is None:
            return set()
        placeholders = _extract_format_placeholders(text)
    return {p.split(".")[0] for p in placeholders}


def _dotted_field_reads(node: Node, template_resolver: Any) -> set[tuple[str, str]]:
    """``(root, field)`` pairs a node's template reads, e.g. ``${triage.severity}``.

    The only FIELD-granular consumer axis. The other two -- a downstream node
    taking the whole model, and the terminal projection -- consume at model
    granularity.
    """
    prompt = node.prompt
    if not prompt or node.mode == "scripted":
        return set()

    if " " in prompt or "${" in prompt:
        placeholders = _PLACEHOLDER_RE.findall(prompt)
    else:
        if template_resolver is None:
            return set()
        text = template_resolver(prompt)
        if text is None:
            return set()
        placeholders = _extract_format_placeholders(text)

    reads: set[tuple[str, str]] = set()
    for placeholder in placeholders:
        parts = placeholder.split(".")
        if len(parts) >= 2:
            reads.add((parts[0], parts[1]))
    return reads


def _check_unconsumed_outputs(
    construct: Construct,
    issues: list[LintIssue],
    *,
    template_resolver: Callable[[str], str | None] | None = None,
) -> None:
    """Report an output field that nothing in the graph reads (GH #11).

    Construct-level, not per-node: a field is dead only when EVERY consumer axis
    misses it, so the decision needs the whole graph. Three axes:

    1. A downstream node input. A consumer declaring ``triage: Triage`` receives
       the whole model, and which fields its body reads is not derivable, so
       whole-model consumption marks every field consumed. This over-approximates
       deliberately: the opposite flags every scripted consumer in every pipeline.
    2. A dotted template placeholder. ``${triage.severity}`` reads one field, and
       this is the axis that gives the check its resolution.
    3. The terminal projection. The last node's output is the graph's output.

    Deriving fewer axes reports false cleanliness, which is worse than reporting
    nothing, because a guard that cannot fire is evidence of nothing.
    """
    nodes = [n for n in iter_with_arms(construct) if isinstance(n, Node)]
    if not nodes:
        return
    terminal = nodes[-1].name

    consumed_whole: set[str] = set()
    consumed_fields: set[tuple[str, str]] = set()

    for node in nodes:
        inputs = normalize_inputs(node.inputs)
        declared_names: set[str] = set()
        if inputs.is_dict_form:
            declared_names = set(inputs.by_name)
        elif not inputs.is_none and isinstance(inputs.single_type, type):
            declared_names = {inputs.single_type.__name__}

        if node.mode == "scripted" or not node.prompt:
            # A scripted body can read any field, and which ones is not
            # derivable, so taking the model consumes all of it. The opposite
            # would flag every scripted consumer in every pipeline.
            consumed_whole |= declared_names
            continue

        # An LLM-mode body never runs, so the TEMPLATE is the only reader. A
        # bare `${triage}` consumes the whole model; `${triage.severity}`
        # consumes one field. This is the same reasoning that scopes the
        # unreferenced-input check to LLM-mode nodes.
        reads = _dotted_field_reads(node, template_resolver)
        consumed_fields |= reads
        dotted_roots = {root for root, _ in reads}
        bare = _template_roots(node, template_resolver) - dotted_roots
        consumed_whole |= declared_names & bare
        if node.skip_when is not None:
            # A skip predicate receives the extracted input dict, so it can read
            # any field of it. Same opacity as a scripted body.
            consumed_whole |= declared_names

    for node in nodes:
        if node.name == terminal:
            continue
        outputs = normalize_outputs(node.outputs)
        for key, declared in (outputs.all_keys or {}).items() or (
            {node.name: outputs.primary}.items() if outputs.primary is not None else ()
        ):
            if not (isinstance(declared, type) and issubclass(declared, BaseModel)):
                continue
            # `@node` kebab-cases the function name, while a consumer's PARAM
            # keeps the underscore form. field_name_for owns that contract, so
            # both sides are compared in the same form.
            base = field_name_for(node.name)
            root = base if key == node.name else output_field_name(base, key)
            if root in consumed_whole or base in consumed_whole:
                continue
            for fname in declared.model_fields:
                if (root, fname) in consumed_fields or (base, fname) in consumed_fields:
                    continue
                issues.append(
                    LintIssue(
                        node_name=node.name,
                        param=fname,
                        kind="output_field_unconsumed",
                        required=False,
                        message=(
                            f"Node '{node.name}': output field '{fname}' of "
                            f"{declared.__name__} has no consumer. No downstream "
                            f"node takes the model, no template reads "
                            f"${{{root}.{fname}}}, and this is not the graph's "
                            f"terminal output. The field costs tokens on every "
                            f"call and cannot affect the answer."
                        ),
                    )
                )


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


