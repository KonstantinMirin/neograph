"""Portal mesh cost across a Branch arm boundary (neograph-ftnxl.12).

ORIGINAL FINDING (superseded, kept for history): ``_recursion_budget.
_mesh_hop_cost`` walked ``iter_with_arms(construct)`` and accumulated a
contiguous run of Portal-shaped items, flushing on a non-Portal item. Its
docstring claimed the walk "mirrors the compiler's mesh detection", but the
compiler's ``_contiguous_portal_mesh`` (``_wiring.py``) collects a run from a
node LIST -- and a ``_BranchNode``'s two arms are two SEPARATE lists, never
one. ``iter_with_arms`` flattens the sentinel away, so a Portal node ending
the TRUE arm and a Portal node starting the FALSE arm landed adjacent in the
flat walk and merged into ONE run, silently dropping the false-arm member's
own hop budget from the ``recursion_limit`` floor.

SUPERSEDING FIX: reproduction (neograph-q14v5.2, ``TestPortalInsideBranchArmRejected``
in ``test_portal_validation.py``) found the true defect is more severe than a
mis-sized budget -- a Portal modifier placed directly on a branch-arm item is
completely INERT at compile time (``_add_arm_nodes``/``_wire_arm_edges`` never
wire ``Command``-based routing for arm items at all). Per the project's north
star ("unrepresentable > fail-loud > silent"), the fix makes this construction
shape unrepresentable (``_check_no_portal_in_branch_arm``, raises
``ConstructError`` at ``Construct(...)`` assembly) rather than correctly
re-costing a mesh that would never actually route. This test is repurposed
below to assert that rejection instead of a successful compile + budget floor
-- the original scenario (successful compilation of a Portal-in-arm construct)
is now unreachable.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import HANDOFF_END, Construct, Node, Portal
from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
from neograph._wiring import _contiguous_portal_mesh
from neograph.errors import ConstructError
from tests.fakes import register_scripted

TRUE_ARM_MAX_HOPS = 2
FALSE_ARM_MAX_HOPS = 7


class BranchSeed(BaseModel, frozen=True):
    flag: bool = True


class RouteHop(BaseModel, frozen=True):
    goto: str = HANDOFF_END


def _build_branch_with_portal_in_each_arm() -> tuple[Construct, _BranchMeta]:
    """A Construct whose TRUE arm ends with a Portal node and whose FALSE arm
    starts with a Portal node -- adjacent only in the arm-flattened walk."""
    register_scripted("ftnxl12_seed", lambda _in, _cfg: BranchSeed(flag=True))
    register_scripted("ftnxl12_hop", lambda _in, _cfg: RouteHop())

    seed = Node.scripted("seed", fn="ftnxl12_seed", outputs=BranchSeed)
    true_portal = Node(
        name="true_arm_portal",
        mode="scripted",
        inputs={"handoff": RouteHop},
        outputs=RouteHop,
        scripted_fn="ftnxl12_hop",
    ) | Portal(to=[], max_hops=TRUE_ARM_MAX_HOPS)
    false_portal = Node(
        name="false_arm_portal",
        mode="scripted",
        inputs={"handoff": RouteHop},
        outputs=RouteHop,
        scripted_fn="ftnxl12_hop",
    ) | Portal(to=[], max_hops=FALSE_ARM_MAX_HOPS)

    meta = _BranchMeta(
        condition_spec=_ConditionSpec(
            source_node=seed,
            attr_chain=["flag"],
            op_fn=lambda value, _t: bool(value),
            op_str="route",
            threshold=None,
        ),
        true_arm_nodes=[true_portal],
        false_arm_nodes=[false_portal],
    )
    construct = Construct("ftnxl12-arm-boundary", nodes=[seed, _BranchNode(meta, 0)])
    return construct, meta


def test_arm_boundary_construct_rejected_not_silently_mis_budgeted():
    """A Portal node ending the TRUE arm and a Portal node starting the FALSE
    arm must never reach a merged (or any) recursion-limit budget calculation
    at all -- the construct is rejected at assembly, before ``compile()`` runs.
    """
    with pytest.raises(ConstructError) as exc:
        _build_branch_with_portal_in_each_arm()
    assert "portal" in str(exc.value).lower() and "arm" in str(exc.value).lower()


def test_contiguous_portal_mesh_still_treats_the_two_arms_as_disjoint_runs():
    """Documents the compiler-level fact that motivated the original filing:
    ``_contiguous_portal_mesh``, applied to each arm's node list (the only
    lists it is ever handed), returns a one-member run per arm -- the two arms
    are disjoint lists and can never merge under the compiler's own
    definition. Construct assembly now rejects this shape before compilation,
    but the mesh-collector's list-scoped behavior is independently correct and
    worth pinning on its own list inputs (no ``Construct``/``_BranchNode``
    assembly involved, so the new guard does not apply here).
    """
    register_scripted("ftnxl12_hop2", lambda _in, _cfg: RouteHop())
    true_portal = Node(
        name="true_arm_portal2",
        mode="scripted",
        inputs={"handoff": RouteHop},
        outputs=RouteHop,
        scripted_fn="ftnxl12_hop2",
    ) | Portal(to=[], max_hops=TRUE_ARM_MAX_HOPS)
    false_portal = Node(
        name="false_arm_portal2",
        mode="scripted",
        inputs={"handoff": RouteHop},
        outputs=RouteHop,
        scripted_fn="ftnxl12_hop2",
    ) | Portal(to=[], max_hops=FALSE_ARM_MAX_HOPS)

    assert _contiguous_portal_mesh([true_portal], true_portal) == [true_portal]
    assert _contiguous_portal_mesh([false_portal], false_portal) == [false_portal]
