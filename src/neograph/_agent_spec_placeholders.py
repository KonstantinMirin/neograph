"""Placeholder translation + property derivation for the Agent Spec export.

Extracted from ``_agent_spec.py`` (neograph-3ffdg.3) as a pure file split — the
functions below are unchanged, only their home moved.

``_properties_for`` came along even though the ticket did not name it: all three
extracted clusters call it, so leaving it in ``_agent_spec.py`` would have made
every new module import back into its parent — a cycle the split exists to avoid.
It is a 17-line property-derivation helper and belongs with the translation code
that uses it most.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

from neograph._placeholders import DOLLAR_RE, apply_scanner
from neograph.errors import ConfigurationError
from neograph.node import Node
from neograph.spec_types import model_to_agent_spec_properties

if TYPE_CHECKING:
    from pyagentspec.property import Property


def _translate_placeholders(
    prompt_text: str, input_props: list[Property], node_name: str
) -> tuple[str, list[Property], dict[str, str]]:
    """Translate neograph ``${path}`` placeholders to pyagentspec ``{{ flat }}``
    form (Option F, neograph-cbpyx — amends m57mn's Option-B fail-loud guard).

    neograph's ``${var}``/``${var.field}`` and pyagentspec's ``{{ var }}`` are two
    syntaxes for the IDENTICAL flat, non-recursive text substitution. This rewrites
    each ``${path}`` to ``{{ path_with_dots_as_underscores }}`` and returns the
    Properties the exported ``LlmNode``/``Agent`` should declare — exactly the
    scanned names, so pyagentspec's own placeholder inference/validation
    (``ComponentWithIO._get_inferred_inputs`` / ``_validate_inputs``) passes by
    construction. REUSES the ONE ``${...}`` scanner (``_placeholders.DOLLAR_RE`` +
    ``apply_scanner``) — never a second grammar (the anti-duplication invariant).

    Returns ``(rewritten_text, referenced_props, flat_to_original)``:
      * ``rewritten_text`` — prompt with every ``${path}`` -> ``{{ flat }}``.
      * ``referenced_props`` — one ``StringProperty(title=flat)`` per unique scanned
        path (names only; pyagentspec infers inputs by NAME, and round-trip type
        fidelity rides the ``neograph/prompt_spec`` marker, not these props).
      * ``flat_to_original`` — ``{flat_name: original_dotted_path}``, consumed by the
        input-edge / StartNode consumer sweep to route ``destination_input`` through
        the SAME flat name (drop an edge whose source path is unreferenced).

    Fail loud (``ConfigurationError``) on a ``${path}`` whose first segment is not a
    declared input (dangling), and on two distinct paths flattening to one name
    (collision — names both paths AND the collided flat name).
    """
    from pyagentspec.property import StringProperty

    declared_keys = {p.title.split(".", 1)[0] for p in input_props}
    flat_to_original: dict[str, str] = {}
    ordered: list[str] = []

    def resolve(raw: str) -> str:
        path = raw.strip()
        first = path.split(".", 1)[0]
        if first not in declared_keys:
            raise ConfigurationError.build(
                f"node {node_name!r}'s prompt references ${{{path}}}, whose first segment "
                f"{first!r} is not a declared input",
                expected=f"a ${{...}} path rooted at one of the declared inputs {sorted(declared_keys)}",
                found=f"dangling placeholder ${{{path}}}",
                hint="every inline ${var} placeholder in an exported LLM-mode prompt must "
                "resolve to a declared Node.input (the value has no other data path).",
            )
        flat = path.replace(".", "_")
        prev = flat_to_original.get(flat)
        if prev is not None and prev != path:
            raise ConfigurationError.build(
                f"node {node_name!r}'s prompt has two distinct placeholders {prev!r} and "
                f"{path!r} that both flatten to the same name {flat!r}",
                expected="each ${path} to flatten (. -> _) to a unique placeholder name",
                found=f"collision: {prev!r} and {path!r} both -> {flat!r}",
                hint="rename one of the colliding upstream inputs/fields so the flattened "
                "pyagentspec placeholder names stay distinct.",
            )
        if flat not in flat_to_original:
            flat_to_original[flat] = path
            ordered.append(flat)
        return f"{{{{ {flat} }}}}"

    rewritten = apply_scanner(prompt_text, DOLLAR_RE, resolve)
    referenced_props: list[Property] = [StringProperty(title=flat) for flat in ordered]
    return rewritten, referenced_props, flat_to_original


def _node_translation(node: Node) -> tuple[str, list[Property], dict[str, str]]:
    """Recompute a node's placeholder translation (``rewritten_text``,
    ``referenced_props``, ``original_to_flat``) from its prompt + declared inputs.

    The SINGLE per-node translation seam every construction site AND every
    consumer (input edges, Loop self-edge, Oracle fan-in, Each StartNode)
    re-derives from — so ``destination_input`` names are computed by the ONE
    translator, never re-inferred per-symptom. Idempotent: the node was already
    translated during ``_lower_construct_item`` (any collision/dangling already
    raised there), so re-running here cannot introduce a new raise. Returns
    ``original_to_flat`` (dotted path -> flat name) — the inverse of
    ``_translate_placeholders``'s ``flat_to_original`` — for edge routing.
    """
    _, ref_props, flat_to_original = _translate_placeholders(node.prompt or "", _properties_for(node.inputs), node.name)
    original_to_flat = {path: flat for flat, path in flat_to_original.items()}
    return node.prompt or "", ref_props, original_to_flat


def _is_translation_eligible(item: Any) -> TypeGuard[Node]:
    """A construct item whose exported prompt is placeholder-translated: an
    LLM-mode (``think``/``agent``/``act``) ``Node``. Gates the consumer sweep on
    the CONSUMING ITEM's mode — NOT the destination SpecNode class (a MapNode
    wrapping a translated inner is still a translation target). Scripted/raw
    nodes have no ${var} prompt, so their edges keep the untranslated dotted form.

    A ``TypeGuard[Node]`` so the ``node`` arg of ``_lower_each``/``_lower_loop``
    (now ``Node | Construct``) narrows to ``Node`` inside the guarded branch that
    calls the Node-only ``_node_translation`` — a Construct item is never
    translation-eligible (it carries no ${var} prompt of its own).
    """
    return isinstance(item, Node) and item.mode in ("think", "agent", "act")


def _prompt_spec_marker(node: Node, flat_to_original: dict[str, str]) -> dict[str, Any]:
    """Build the strictly JSON-native ``neograph/prompt_spec`` round-trip marker.

    Carries the UNtranslated ``${var}`` text + the full original input TypeSpec so
    ``from_agent_spec`` reconstructs the exact original ``Node`` — including inputs
    the prompt never referenced (whose translated ``LlmNode`` drops both Property
    and DataFlowEdge, a real topology change). MUST stay JSON-native (str / dict /
    list only): ``p.json_schema`` is the plain JSON-Schema dict (NOT a live
    pyagentspec ``Property`` object, which would degrade to a dict across a
    JSON/YAML wire round trip and break the loader's un-flatten). The dotted
    ``title`` (``"{key}.{field}"``) is stored alongside so the loader can regroup
    by upstream key via the EXISTING ``_dict_form_inputs_from_props``.
    """
    return {
        "original_text": node.prompt or "",
        "placeholder_map": dict(flat_to_original),
        "original_inputs": [{"title": p.title, "json_schema": p.json_schema} for p in _properties_for(node.inputs)],
    }


def _properties_for(type_spec: Any) -> list[Property]:
    """Convert a Node.inputs/outputs TypeSpec (None | type | dict[str, type]) to Properties.

    Reuses ``spec_types.model_to_agent_spec_properties`` for every model —
    never a second type walker, per the Core Invariant.
    """
    if type_spec is None:
        return []
    if isinstance(type_spec, dict):
        result: list[Property] = []
        for key, typ in type_spec.items():
            props = model_to_agent_spec_properties(typ)
            for p in props:
                p.title = f"{key}.{p.title}"
            result.extend(props)
        return result
    return model_to_agent_spec_properties(type_spec)
