"""Pipeline-member selection — THE single predicate + the single policy site.

Extracted from _construct_builder.py. The silent sub-construct drop happened
because construct_from_module and construct_from_functions each carried their
own isinstance-classification ladder, which drifted on two axes (plain Node:
collect vs reject; Construct: silent-skip vs collect). Classification AND the
skip/warn/raise policy are monopolized here; the entry points only declare
their source kind. `TestMemberSelectionPredicateMonopoly` bans a second
ladder or a second policy site.

Same layer as _construct_builder.py (assembly/DX): imports Node, Construct,
and _get_sidecar cycle-free. Do NOT push this into the IR or a neutral
low-level module — Construct imports would cycle, and the IR must stay
module-unaware.
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Any, Literal

from neograph._construct_validation import ConstructError
from neograph._sidecar import _get_sidecar
from neograph.construct import Construct
from neograph.errors import ConstructArtifactSkipped
from neograph.node import Node


class _MemberKind(Enum):
    """Classification of a pipeline-member candidate. See `_classify_member`."""

    DECORATED_NODE = "decorated_node"
    PLAIN_NODE = "plain_node"
    SUB_CONSTRUCT = "sub_construct"
    ARTIFACT = "artifact"  # Construct with output=None — see _classify_member


def _classify_member(obj: Any) -> _MemberKind | None:
    """THE single selection predicate: what counts as a pipeline member.

    Everything classifies through here; `_bucket_members` is the only policy
    site that acts on the result. Duplicating this ladder at a second site
    re-creates the silent-drop drift of neograph-xv9ay.

    A Construct with `output=` set is a wireable SUB_CONSTRUCT (this holds
    for ForwardConstruct instances too — the key is the output boundary, not
    the type). A Construct with `output=None` is an ARTIFACT: in a walked
    namespace it is a stored top-level pipeline (the canonical
    ``pipeline = construct_from_module(sys.modules[__name__])`` pattern
    re-walked in a persistent namespace — notebook cell re-run), structurally
    indistinguishable from a sub-construct whose author forgot `output=`.
    The source-dependent skip-warn/raise policy lives in `_bucket_members`.
    """
    if isinstance(obj, Construct):
        return _MemberKind.SUB_CONSTRUCT if obj.output is not None else _MemberKind.ARTIFACT
    if isinstance(obj, Node):
        if _get_sidecar(obj) is not None:
            return _MemberKind.DECORATED_NODE
        return _MemberKind.PLAIN_NODE
    return None


def _bucket_members(
    members: list[Any],
    construct_name: str,
    source: Literal["module", "list"] = "list",
) -> tuple[list[Node], list[Node], list[Construct]]:
    """Classify each member once (via `_classify_member`) and bucket for the
    build pipeline. The SOLE skip/warn/raise policy site:

    | kind          | source="module" (namespace)   | source="list" (promise) |
    |---------------|-------------------------------|-------------------------|
    | DECORATED     | collect                       | collect                 |
    | PLAIN_NODE    | collect                       | collect                 |
    | SUB_CONSTRUCT | collect                       | collect                 |
    | ARTIFACT      | skip + ConstructArtifactSkipped | raise                 |
    | None          | skip silently                 | raise                   |

    A namespace accumulates artifacts and non-members; an explicit list is a
    promise that every element is a wireable member. That source-dependent
    difference is the ONE legitimate asymmetry — expressed here and nowhere
    else.
    """
    nodes: list[Node] = []
    plain_nodes: list[Node] = []
    sub_constructs: list[Construct] = []
    for item in members:
        kind = _classify_member(item)
        if kind is _MemberKind.SUB_CONSTRUCT:
            sub_constructs.append(item)
        elif kind is _MemberKind.DECORATED_NODE:
            nodes.append(item)
        elif kind is _MemberKind.PLAIN_NODE:
            plain_nodes.append(item)
        elif kind is _MemberKind.ARTIFACT:
            if source == "module":
                warnings.warn(
                    f"construct_from_module skipped module-level Construct '{item.name}' "
                    "(output=None): a Construct with no output boundary is treated as a "
                    "stored top-level pipeline, not a wireable sub-construct. This is "
                    "expected when re-walking a namespace that already holds a built "
                    "pipeline (re-run notebook cell, persistent execution namespace). "
                    f"If '{item.name}' was meant to be an inlined sub-construct, give it "
                    "output= and it will be collected.",
                    ConstructArtifactSkipped,
                    stacklevel=3,
                )
            else:
                raise ConstructError.build(
                    f"Construct '{item.name}' has no output type",
                    construct=construct_name,
                    hint="an explicitly listed Construct must declare output= to wire as a "
                    f"sub-construct; if '{item.name}' is a finished top-level pipeline, "
                    "don't pass it in the functions list",
                )
        elif source == "list":
            raise ConstructError.build(
                "argument is not decorated with @node, a Node, or a Construct",
                construct=construct_name,
                found=type(item).__name__,
                hint="every list element must be an @node function, a Node instance, or a Construct with declared output",
            )
        # source == "module" and kind is None: helper/constant/import — skip silently.
    return nodes, plain_nodes, sub_constructs
