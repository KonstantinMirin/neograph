"""Who consumes an output: the ``output_field_unconsumed`` check and its axes.

Split from ``_lint_supply`` (which crossed its 500-line ceiling) along the seam
the checks already have -- this module answers "does anything read this produced
field", while ``_lint_supply`` answers "does this supplied value reach anything"
and "can any caller satisfy this binding".

The consumer axes live together on purpose. A field is dead only when EVERY axis
misses it, so a new axis added at one site and forgotten at another does not
report a smaller problem -- it reports a live field as dead. ``_framework_field_reads``
is the single place a FRAMEWORK-declared reader is derived; deriving one at a
second site is how four of them were missed one at a time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from neograph._ir_branch import _BranchNode, iter_with_arms
from neograph._ir_fields import item_field_names
from neograph._ir_normalize import resolve_output_from
from neograph._lint_kind_registry import LintIssue
from neograph._lint_predict import _extract_format_placeholders
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph._placeholders import DOLLAR_RE as _PLACEHOLDER_RE
from neograph.construct import Construct
from neograph.naming import field_name_for, output_field_name
from neograph.node import Node


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
        # `${image:seed.photo}` reads field `photo` of `seed`. The `image:`
        # prefix is a rendering directive, not part of the name, and
        # `_placeholder_root` already strips it -- both readers must agree or a
        # referenced field looks unread.
        body = placeholder.split(":", 1)[1] if placeholder.startswith("image:") else placeholder
        parts = body.split(".")
        if len(parts) >= 2:
            reads.add((parts[0], parts[1]))
    return reads


def _single_input_types(annotation: Any) -> list[type]:
    """Concrete model types a single-type ``inputs=`` annotation can bind.

    ``Claims`` yields ``[Claims]``; ``Claims | None`` yields ``[Claims]``. The
    runtime binds by isinstance, so a union binds to whichever member matches --
    treating the union as opaque made every Optional consumer invisible.
    """
    import types as _types
    from typing import Union, get_args, get_origin

    if isinstance(annotation, type):
        return [annotation]
    if get_origin(annotation) is Union or isinstance(annotation, _types.UnionType):
        return [a for a in get_args(annotation) if isinstance(a, type) and a is not type(None)]
    return []


def _framework_field_reads(construct: Construct) -> tuple[set[tuple[str, str]], set[str]]:
    """Field readers the FRAMEWORK declares in the IR, not a downstream node.

    SINGLE SOURCE for "what does neograph itself read off a node's output".
    Returns ``(field_reads, whole_roots)`` -- ``(root, field)`` pairs, and roots
    consumed whole.

    Every reader below is a NAME already sitting in the IR the walk passes over,
    which is what makes deriving them possible without an annotation. They were
    missed because the check derived its axes from what a pipeline AUTHOR writes
    and never asked what the runtime reads:

    - ``Each(over="clusters.groups")`` -- the fan-out reads that field to build
      its ``Send`` list.
    - ``Portal(route="goto")`` -- mode (a) routes on that field of the member's
      own output; ``route`` defaults to ``"goto"`` and IS the field name.
    - ``Portal(spec_field=..., input_field=...)`` -- mode (b) reads both off the
      emitting node's own output.
    - A branch condition's ``attr_chain`` -- ``_ConditionSpec`` names the source
      node and the attribute path the predicate reads.
    - ``Construct.output`` -- the declared boundary type surfaces to the parent,
      so EVERY node producing it is a terminal producer. ``members[-1]`` can name
      only one, which is wrong the moment a branch gives each arm its own.

    A ``Loop``/``Operator`` ``when=`` callable is deliberately absent: a lambda's
    field reads are not derivable, and the whole-model rule already covers the
    node it guards.

    RULE for a new modifier: if it names a field, teach THIS function. Do not
    re-derive readers at a second site -- that is how the first four were missed
    one at a time.
    """
    field_reads: set[tuple[str, str]] = set()
    whole_roots: set[str] = set()

    # This construct's OWN declared output=. Every arm that satisfies the
    # boundary is a terminal producer, not just the last member -- a branch has
    # one terminal per arm, and `members[-1]` can only ever name one of them.
    #
    # GH #17: when `output_from` names the producer, exactly ONE member is the
    # boundary, so this reads that member instead of every type-compatible one.
    # Without this the author's declared port would draw a false
    # `output_field_unconsumed` WARN -- the field is read, by the boundary itself.
    # One call, through the shared hop. Step 1 resolved the dotted form here by hand;
    # PortRef.field is now where that lives, so lint cannot drift from the runtime and
    # the exporters on what "settle.result" means.
    ref = resolve_output_from(construct)
    if ref is not None:
        whole_roots.add(ref.field)
    elif isinstance(construct.output, type):
        # The shared ELIGIBILITY SET, not a re-derivation. This used to add
        # field_name_for(inner.name) -- the BARE base -- which is wrong for a
        # dict-form-output member, whose boundary field is {base}_{key} and never the
        # bare base. So lint marked a field that does not exist as consumed, left the
        # real one unmarked, and reported the author's actual boundary as dead
        # neograph-lmjn5. item_field_names is the same set the runtime scopes the
        # boundary to, so the two cannot disagree about which fields exist.
        eligible = set(item_field_names(construct))
        for inner in iter_with_arms(construct):
            declared = normalize_outputs(getattr(inner, "outputs", None)).primary
            if isinstance(declared, type) and issubclass(declared, construct.output):
                base = field_name_for(inner.name)
                whole_roots.update(f for f in eligible if f == base or f.startswith(f"{base}_"))

    # RAW `.nodes`, deliberately: `iter_with_arms` drops the `_BranchNode`
    # sentinel, and the sentinel is what carries the condition's attr_chain --
    # the branch's field read would be invisible through the arm-aware
    # primitive. Arm CONTENTS are still visited, by expanding each sentinel
    # below, so no arm node's modifiers are missed either.
    members: list[Any] = []
    for member in construct.nodes:
        if isinstance(member, _BranchNode):
            meta = member._neo_branch_meta
            spec = meta.condition_spec
            source = getattr(spec, "source_node", None)
            if source is not None and spec.attr_chain:
                field_reads.add((field_name_for(source.name), spec.attr_chain[0]))
            members.extend(meta.true_arm_nodes)
            members.extend(meta.false_arm_nodes)
            continue
        members.append(member)

    for member in members:
        # A sub-construct applies the boundary rule from its own side, in the
        # recursive call -- the parent must not restate it.
        if isinstance(member, Construct):
            field_reads_inner, whole_inner = _framework_field_reads(member)
            field_reads |= field_reads_inner
            whole_roots |= whole_inner

        modifiers = getattr(member, "modifier_set", None)
        if modifiers is None:
            continue

        each = getattr(modifiers, "each", None)
        if each is not None and isinstance(getattr(each, "over", None), str) and "." in each.over:
            producer, _, field = each.over.partition(".")
            field_reads.add((field_name_for(producer), field.split(".")[0]))

        portal = getattr(modifiers, "portal", None)
        if portal is not None:
            own = field_name_for(member.name)
            # Mode (b) reads spec_field/input_field; mode (a) routes on `route`,
            # which names a field of the member's own output. `is_dispatch` is
            # the one authority on which mode this is -- in
            # dispatch mode `route` is the literal marker, not a field name.
            attrs = ("spec_field", "input_field") if portal.is_dispatch else ("route",)
            for attr in attrs:
                value = getattr(portal, attr, None)
                if isinstance(value, str) and value:
                    field_reads.add((own, value))

    return field_reads, whole_roots


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
    members = list(iter_with_arms(construct))
    nodes = [n for n in members if isinstance(n, Node)]
    if not nodes:
        return
    # The LAST member is a terminal, whether it is a Node or a sub-construct.
    # When a sub-construct is last, no leaf node is terminal. A BRANCH gives each
    # ARM its own terminal, and a single `members[-1]` can name only one of them:
    # the arm that happens to be expanded last. Every other arm's final producer
    # then looks dead while it is exactly what the graph returns on that path.
    terminals: set[str] = set()
    if construct.nodes:
        last = construct.nodes[-1]
        if isinstance(last, _BranchNode):
            meta = last._neo_branch_meta
            for arm in (meta.true_arm_nodes, meta.false_arm_nodes):
                if arm:
                    terminals.add(arm[-1].name)
        else:
            terminals.add(last.name)

    # A member sub-construct consumes its port BY TYPE: `_scan_subgraph_input`
    # matches any upstream value whose type is the declared `input=`. It is not a
    # Node, so filtering to Node alone makes it invisible as a consumer and every
    # producer feeding it looks dead.
    port_types = {m.input for m in members if isinstance(m, Construct) and isinstance(m.input, type)}

    # Axis 4: the FRAMEWORK itself. A field named by a modifier, a branch
    # condition, or a sub-construct boundary has a reader that is not a
    # downstream node, and deriving only author-written axes reported 18 live
    # fields as dead across the should_pass corpus.
    framework_fields, framework_whole = _framework_field_reads(construct)

    consumed_whole: set[str] = set(framework_whole)
    consumed_fields: set[tuple[str, str]] = set(framework_fields)
    consumed_types: set[type] = set()

    for node in nodes:
        inputs = normalize_inputs(node.inputs)
        declared_names: set[str] = set()
        if inputs.is_dict_form:
            declared_names = set(inputs.by_name)
        elif not inputs.is_none:
            # A single-type input is resolved by TYPE at runtime, not by the
            # producer's name. `inputs=Claims | None` is the same consumer
            # wearing a union, so unwrap it -- otherwise the model it takes
            # whole looks unread.
            for declared_type in _single_input_types(inputs.single_type):
                declared_names.add(declared_type.__name__)
                consumed_types.add(declared_type)

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
        if node.name in terminals:
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
            if any(declared is t or issubclass(declared, t) for t in port_types):
                continue
            if any(declared is t or issubclass(declared, t) for t in consumed_types):
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
