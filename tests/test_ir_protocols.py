"""Unit tests for the ``ConstructItem`` Protocol.

Verifies that all three IR item subtypes — ``Node`` (BaseModel), ``Construct``
(BaseModel), and ``_BranchNode`` (non-Pydantic sentinel) — conform to
``ConstructItem``. The Protocol replaces the bare ``list[Any]`` annotation on
``Construct.nodes`` so that the polymorphism is named, not erased.
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph._ir_protocols import ConstructItem
from neograph.construct import Construct
from neograph.forward import _BranchMeta, _BranchNode, _ConditionSpec
from neograph.node import Node


class _Payload(BaseModel):
    value: str = ""


def _make_branch_node() -> _BranchNode:
    """Build a minimal ``_BranchNode`` for isinstance checks (no compilation).

    The sentinel does not require a usable ``_BranchMeta``; only its
    ``name``/``modifier_set`` attributes matter for the structural check.
    """
    cond = _ConditionSpec(
        source_node=None,
        attr_chain=[],
        op_fn=lambda a, b: True,
        op_str=">",
        threshold=0,
    )
    meta = _BranchMeta(condition_spec=cond, true_arm_nodes=[], false_arm_nodes=[])
    return _BranchNode(meta, branch_id=0)


class TestConstructItemConformance:
    """Every IR-item subtype must satisfy the ``ConstructItem`` Protocol."""

    def test_node_conforms(self):
        node = Node.scripted("step", fn="f", inputs=_Payload, outputs=_Payload)
        assert isinstance(node, ConstructItem)

    def test_construct_conforms(self):
        producer = Node.scripted("p", fn="f", inputs=_Payload, outputs=_Payload)
        construct = Construct("c", nodes=[producer])
        assert isinstance(construct, ConstructItem)

    def test_branch_node_conforms(self):
        branch = _make_branch_node()
        assert isinstance(branch, ConstructItem)

    def test_unrelated_value_does_not_conform(self):
        assert not isinstance("not an item", ConstructItem)
        assert not isinstance(42, ConstructItem)
        assert not isinstance({"name": "x"}, ConstructItem)
