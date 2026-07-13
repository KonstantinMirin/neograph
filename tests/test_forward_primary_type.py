"""Behavior-pin for forward.py _primary_type extraction (neograph-2yi7q item a).

(a) is a ZERO-BEHAVIOR-CHANGE refactor: extract module-private ``_primary_type(item)``
over the 11x ``normalize_outputs(_declared_output(item)).primary`` chain. This pins
that the helper equals the chain it replaces and returns the declared primary type.

The tracer infers the post-merge downstream-bus type, which INTENTIONALLY diverges
from ``_dispatch._resolve_primary_output`` (the LLM-schema gen_type) for a
type-transforming Oracle — see ``_primary_type``'s docstring. Do NOT route these
sites through the resolver; the divergence docstring is the contract that keeps
them separate.
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import Node
from neograph._normalize import _declared_output, normalize_outputs
from neograph.forward import _primary_type


class _Out(BaseModel):
    x: str = ""


class TestPrimaryTypeHelper:
    """_primary_type(item) == normalize_outputs(_declared_output(item)).primary."""

    def test_helper_equals_the_chain_it_replaces(self):
        """The refactor is zero-behavior-change: the helper IS the chain."""
        node = Node.scripted("n", fn="f", outputs=_Out)
        assert _primary_type(node) == normalize_outputs(_declared_output(node)).primary

    def test_returns_declared_output_type(self):
        """For a plain node, _primary_type returns its declared output type."""
        node = Node.scripted("n", fn="f", outputs=_Out)
        assert _primary_type(node) is _Out

    def test_consistent_across_calls(self):
        """Repeated calls return the same type object (no drift)."""
        node = Node.scripted("n", fn="f", outputs=_Out)
        assert _primary_type(node) is _primary_type(node)
