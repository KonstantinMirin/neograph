"""verify_compiled() — post-compile structural verification.

Validates that a compiled graph's structural dependencies (registries,
state model, LLM factory) are still intact.  Call after compile() as a
defense-in-depth check before run().

Returns a list of VerifyIssue dataclass instances (never raises).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neograph._ir_branch import iter_with_arms
from neograph._ir_protocols import ConstructItem
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
    construct = getattr(graph, "construct", None)
    if construct is None:
        return [VerifyIssue(
            node_name="<graph>",
            kind="no_construct",
            message="Compiled graph has no construct — was it compiled with neograph.compile()?",
        )]

    issues: list[VerifyIssue] = []

    # Get state model from the compiled graph's builder
    state_schema = getattr(getattr(graph, "builder", None), "state_schema", None)
    state_fields = set(state_schema.model_fields.keys()) if state_schema else set()

    # Check if any LLM node exists (to validate LLM factory)
    has_llm_node = False
    scripted_lookup: dict = getattr(graph, "scripted", {}) or {}
    condition_lookup: dict = getattr(graph, "conditions", {}) or {}

    _walk(
        construct, issues, state_fields,
        has_llm_ref=has_llm_node,
        scripted_lookup=scripted_lookup,
        condition_lookup=condition_lookup,
    )

    # After walking, check if LLM factory is needed.
    # The runtime is stashed on the compiled graph at compile time (§2).
    if _has_llm_nodes(construct):
        runtime = getattr(graph, "runtime", None)
        if runtime is None or runtime.llm_factory is None:
            issues.append(VerifyIssue(
                node_name="<graph>",
                kind="llm_factory_missing",
                message="Graph contains LLM nodes but no LLM factory is configured. "
                        "Pass llm_factory= to compile().",
            ))

    return issues


def _walk(
    item: ConstructItem,
    issues: list[VerifyIssue],
    state_fields: set[str],
    *,
    has_llm_ref: bool,
    scripted_lookup: dict | None = None,
    condition_lookup: dict | None = None,
) -> None:
    """Recursively walk construct tree and check each node."""
    if isinstance(item, Construct):
        # iter_with_arms expands _BranchNode sentinels so bare arm Nodes get
        # scripted-fn / state-field / condition-registration verification. See
        # neograph-vn5f (site 4).
        for child in iter_with_arms(item):
            _walk(
                child, issues, state_fields,
                has_llm_ref=has_llm_ref,
                scripted_lookup=scripted_lookup,
                condition_lookup=condition_lookup,
            )
        return

    if not isinstance(item, Node):
        return

    label = f"Node '{item.name}'"

    # 1. Scripted fn registration check (post-§2: per-compile only).
    if item.scripted_fn:
        per_compile = scripted_lookup or {}
        if item.scripted_fn not in per_compile:
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
        _check_condition_registrations(item, ms, issues, condition_lookup=condition_lookup)


def _check_condition_registrations(
    node: Node,
    ms: Any,
    issues: list[VerifyIssue],
    condition_lookup: dict | None = None,
) -> None:
    """Check that string conditions for Loop/Operator are still registered."""
    label = f"Node '{node.name}'"
    conditions = condition_lookup or {}

    loop = ms.loop
    if loop and isinstance(loop.when, str):
        if loop.when not in conditions:
            issues.append(VerifyIssue(
                node_name=label,
                kind="condition_missing",
                message=f"Loop condition '{loop.when}' not found in condition registry.",
            ))

    operator = ms.operator
    if operator and isinstance(operator.when, str):
        if operator.when not in conditions:
            issues.append(VerifyIssue(
                node_name=label,
                kind="condition_missing",
                message=f"Operator condition '{operator.when}' not found in condition registry.",
            ))


def _has_llm_nodes(construct: Construct) -> bool:
    """Check if any node in the construct tree uses an LLM mode.

    iter_with_arms expands _BranchNode sentinels so an LLM-mode node living only
    in a branch arm is counted. See neograph-vn5f (site 4).
    """
    for item in iter_with_arms(construct):
        if isinstance(item, Construct):
            if _has_llm_nodes(item):
                return True
        elif isinstance(item, Node):
            if item.mode in ("think", "agent", "act"):
                return True
    return False
