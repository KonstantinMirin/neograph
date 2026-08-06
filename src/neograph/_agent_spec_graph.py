"""Agent-Spec ``Flow`` graph predicates for the IMPORT side.

The importer's discipline is *never trust a marker -- confirm the structure it
claims to describe*: a node carrying ``metadata["neograph/modifier"] == "loop"``
only imports as a Loop if a real back-edge is actually there. That discipline is
correct and lives at the correct layer; the defect this module fixes is that its
central question -- *is there a control-flow edge X -> Y, optionally labelled
L?* -- had no name and was spelled out five times, so the copies could drift
independently.

Why a module of its own rather than a private function in ``loader.py``: one of
the five sites now lives in ``_agent_spec_swarm_import.py`` (neograph-jtawq.10
moved the swarm-import cluster out of ``loader.py``), so a loader-private helper
could not reach it and the duplication would have survived the fix. The
reconstruction modules were the wrong home for the opposite reason -- this is
RECOGNITION, not reconstruction. This module imports nothing from ``neograph``,
so it is cycle-free by construction and stays reachable from every import-side
module.
"""

from __future__ import annotations

from typing import Any


def _has_control_edge(
    flow: Any,
    from_name: str,
    to_name: str,
    branch: str | None = None,
) -> bool:
    """Is there a ``ControlFlowEdge`` from ``from_name`` to ``to_name``?

    ``branch=None`` means *any* branch label -- the edge's ``from_branch`` is
    not consulted at all. Pass a label to additionally require that arm, which
    is what distinguishes a Loop's ``continue`` back-edge from its ``done``
    exit, and an Operator's ``pause`` arm from its default one.

    Reads ``control_flow_connections`` directly rather than through a cached
    index: the recognition walk asks this a handful of times per group over a
    flow that is already fully in memory, and an index would need invalidating
    for no measurable gain.
    """
    return any(
        e.from_node.name == from_name
        and e.to_node.name == to_name
        and (branch is None or e.from_branch == branch)
        for e in flow.control_flow_connections
    )
