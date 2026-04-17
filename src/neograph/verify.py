"""verify_compiled() — post-compile structural verification.

Validates that a compiled graph's structural dependencies (registries,
state model, LLM factory) are still intact.  Call after compile() as a
defense-in-depth check before run().

Returns a list of VerifyIssue dataclass instances (never raises).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neograph.construct import Construct
from neograph.naming import field_name_for
from neograph.node import Node


@dataclass
class VerifyIssue:
    """A single post-compile verification finding."""

    node_name: str   # which node has the problem
    kind: str        # issue category
    message: str     # human-readable description


def verify_compiled(graph: Any) -> list[VerifyIssue]:
    """Validate structural integrity of a compiled graph.

    Checks that all runtime dependencies (scripted function registry,
    condition registry, LLM factory, state model fields) are still valid
    after compilation.

    Returns an empty list when everything checks out.
    """
    construct = getattr(graph, "_neo_construct", None)
    if construct is None:
        return [VerifyIssue(
            node_name="<graph>",
            kind="no_construct",
            message="Compiled graph has no _neo_construct — was it compiled with neograph.compile()?",
        )]

    issues: list[VerifyIssue] = []

    # Get state model from the compiled graph's builder
    state_schema = getattr(getattr(graph, "builder", None), "state_schema", None)
    state_fields = set(state_schema.model_fields.keys()) if state_schema else set()

    # Check if any LLM node exists (to validate LLM factory)
    has_llm_node = False

    _walk(construct, issues, state_fields, has_llm_ref=has_llm_node)

    # After walking, check if LLM factory is needed
    if _has_llm_nodes(construct):
        from neograph._llm import _llm_factory
        if _llm_factory is None:
            issues.append(VerifyIssue(
                node_name="<graph>",
                kind="llm_factory_missing",
                message="Graph contains LLM nodes but no LLM factory is configured. "
                        "Call configure_llm() before run().",
            ))

    return issues


def _walk(
    item: Construct | Node,
    issues: list[VerifyIssue],
    state_fields: set[str],
    *,
    has_llm_ref: bool,
) -> None:
    """Recursively walk construct tree and check each node."""
    if isinstance(item, Construct):
        for child in item.nodes:
            _walk(child, issues, state_fields, has_llm_ref=has_llm_ref)
        return

    if not isinstance(item, Node):
        return

    label = f"Node '{item.name}'"

    # 1. Scripted fn registration check
    if item.scripted_fn:
        from neograph._registry import registry
        if item.scripted_fn not in registry.scripted:
            issues.append(VerifyIssue(
                node_name=label,
                kind="scripted_fn_missing",
                message=f"Scripted function '{item.scripted_fn}' not found in registry. "
                        f"Was register_scripted() called, or was the registry cleared?",
            ))

    # 2. State field existence check
    field_name = field_name_for(item.name)
    if state_fields and field_name not in state_fields:
        issues.append(VerifyIssue(
            node_name=label,
            kind="state_field_missing",
            message=f"No state field '{field_name}' in compiled state model. "
                    f"Node output may not be stored.",
        ))

    # 3. Loop/Operator condition registration check
    ms = getattr(item, "modifier_set", None)
    if ms:
        _check_condition_registrations(item, ms, issues)


def _check_condition_registrations(
    node: Node,
    ms: Any,
    issues: list[VerifyIssue],
) -> None:
    """Check that string conditions for Loop/Operator are still registered."""
    from neograph._registry import registry

    label = f"Node '{node.name}'"

    loop = ms.loop
    if loop and isinstance(loop.when, str):
        if loop.when not in registry.condition:
            issues.append(VerifyIssue(
                node_name=label,
                kind="condition_missing",
                message=f"Loop condition '{loop.when}' not found in condition registry.",
            ))

    operator = ms.operator
    if operator and isinstance(operator.when, str):
        if operator.when not in registry.condition:
            issues.append(VerifyIssue(
                node_name=label,
                kind="condition_missing",
                message=f"Operator condition '{operator.when}' not found in condition registry.",
            ))


def _has_llm_nodes(construct: Construct) -> bool:
    """Check if any node in the construct tree uses an LLM mode."""
    for item in construct.nodes:
        if isinstance(item, Construct):
            if _has_llm_nodes(item):
                return True
        elif isinstance(item, Node):
            if item.mode in ("think", "agent", "act"):
                return True
    return False
