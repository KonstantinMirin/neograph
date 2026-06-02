"""Contract test: lock the Pydantic ``PrivateAttr`` preservation behavior.

``Node._sidecar``, ``Node._param_res``, and ``Node._scripted_shim`` are
``PrivateAttr(default=None)`` fields (node.py:216-218). The programmatic API
(``Node() | Modifier()``) and the assembly layer rely on Pydantic v2 copying
``__pydantic_private__`` through ``model_copy`` so this metadata survives the
pipe operator, direct copies, modifier composition, and deep copies.

That behavior is *observed*, not *contracted* by Pydantic. A Pydantic 3 bump
could silently drop private attrs on copy and break the programmatic surface
with no error at the call site. This file pins the behavior so a version bump
fails loudly here instead of in production.

See neograph-fvlj. Complements the single-assertion smoke test in
``tests/decorator/test_edge_cases.py::TestAdjacencySidecarLost``.
"""

from __future__ import annotations

import copy
from typing import Annotated

from pydantic import BaseModel

from neograph import Each, FromInput, Operator, Oracle, node
from neograph._sidecar import _get_param_res, _get_sidecar, _register_sidecar
from neograph.di import DIBinding, DIKind
from neograph.node import Node

# ── Schemas ──────────────────────────────────────────────────────────


class RawText(BaseModel):
    text: str


class Claims(BaseModel):
    items: list[str]


# ── Helpers ──────────────────────────────────────────────────────────


def _decorated_node() -> Node:
    """A @node-decorated Node that populates BOTH _sidecar and _param_res.

    The ``topic`` DI param forces a non-empty ``_param_res`` (decorators.py
    only sets it ``if param_res``), so the test exercises both PrivateAttrs.
    """

    @node(mode="scripted", outputs=Claims)
    def classify(upstream: RawText, topic: Annotated[str, FromInput]) -> Claims:
        return Claims(items=[topic])

    assert isinstance(classify, Node)
    return classify


def _assert_sidecar_intact(n: Node, *, fn_expected, param_names_expected) -> None:
    sidecar = _get_sidecar(n)
    assert isinstance(sidecar, tuple), "_sidecar lost (became None or non-tuple)"
    assert sidecar[0] is fn_expected, "_sidecar function identity not preserved"
    assert sidecar[1] == param_names_expected, "_sidecar param names not preserved"


def _assert_param_res_intact(n: Node, *, keys_expected: set[str]) -> None:
    param_res = _get_param_res(n)
    assert isinstance(param_res, dict), "_param_res lost (became None or non-dict)"
    assert set(param_res) == keys_expected, "_param_res keys not preserved"
    for binding in param_res.values():
        assert isinstance(binding, DIBinding), "_param_res value is not a DIBinding"


# ── Contract: @node-populated PrivateAttrs survive every copy path ───


class TestSidecarSurvivesCopyPaths:
    """Node._sidecar / _param_res must survive model_copy, pipe, and deepcopy.

    Pins Pydantic's __pydantic_private__ copy behavior. If a Pydantic bump
    drops private attrs on copy, these assertions fail instead of the
    programmatic API silently breaking at runtime.
    """

    def test_baseline_both_private_attrs_populated_when_node_decorated(self):
        """@node with a DI param populates both _sidecar and _param_res."""
        n = _decorated_node()
        _assert_sidecar_intact(
            n, fn_expected=_get_sidecar(n)[0], param_names_expected=("upstream", "topic")
        )
        # topic is the only DI param; upstream is an upstream edge (not DI).
        _assert_param_res_intact(n, keys_expected={"topic"})

    def test_private_attrs_survive_single_pipe_when_each_applied(self):
        """Node | Each preserves _sidecar and _param_res (model_copy path)."""
        n = _decorated_node()
        fn, names = _get_sidecar(n)

        n2 = n | Each(over="upstream.items", key="k")

        assert n2 is not n, "pipe must produce a new instance (model_copy)"
        _assert_sidecar_intact(n2, fn_expected=fn, param_names_expected=names)
        _assert_param_res_intact(n2, keys_expected={"topic"})

    def test_private_attrs_survive_direct_model_copy(self):
        """A bare model_copy() preserves both PrivateAttrs."""
        n = _decorated_node()
        fn, names = _get_sidecar(n)

        n2 = n.model_copy()

        _assert_sidecar_intact(n2, fn_expected=fn, param_names_expected=names)
        _assert_param_res_intact(n2, keys_expected={"topic"})

    def test_private_attrs_survive_model_copy_with_update(self):
        """model_copy(update={...}) — the exact call the pipe operator makes."""
        n = _decorated_node()
        fn, names = _get_sidecar(n)

        n2 = n.model_copy(update={"name": "renamed"})

        assert n2.name == "renamed"
        _assert_sidecar_intact(n2, fn_expected=fn, param_names_expected=names)
        _assert_param_res_intact(n2, keys_expected={"topic"})

    def test_private_attrs_survive_chained_pipe_composition(self):
        """Repeated pipe (Node | Oracle | Operator) preserves both attrs.

        Each | step is a separate model_copy; the chain must not erode the
        private state across composition.
        """
        n = _decorated_node()
        fn, names = _get_sidecar(n)

        n2 = n | Oracle(n=3, merge_prompt="pick best: ${variants}") | Operator(
            when="needs_review"
        )

        assert n2.has_modifier(Oracle)
        assert n2.has_modifier(Operator)
        _assert_sidecar_intact(n2, fn_expected=fn, param_names_expected=names)
        _assert_param_res_intact(n2, keys_expected={"topic"})

    def test_private_attrs_survive_deepcopy(self):
        """copy.deepcopy preserves both PrivateAttrs.

        deepcopy clones the function object identity rules differently from
        model_copy: model_copy keeps the SAME callable, deepcopy may clone it.
        We assert the metadata is structurally present and callable either way.
        """
        n = _decorated_node()
        _, names = _get_sidecar(n)

        n2 = copy.deepcopy(n)

        sidecar = _get_sidecar(n2)
        assert isinstance(sidecar, tuple), "_sidecar lost after deepcopy"
        assert callable(sidecar[0]), "_sidecar function not callable after deepcopy"
        assert sidecar[1] == names, "_sidecar param names not preserved by deepcopy"
        _assert_param_res_intact(n2, keys_expected={"topic"})


# ── Contract: programmatically-registered sidecar (no @node) ─────────


class TestProgrammaticSidecarSurvivesCopyPaths:
    """The programmatic surface registers the sidecar by hand, then pipes.

    This is the runtime-construction path (LLM-driven / config systems) that
    does NOT go through @node. It must enjoy the same preservation guarantee.
    """

    def test_manually_registered_sidecar_survives_pipe_chain(self):
        def fn_a(upstream: RawText) -> Claims:
            return Claims(items=["x"])

        n = Node(name="a", mode="scripted", outputs=Claims)
        _register_sidecar(n, fn_a, ("upstream",))
        n._param_res = {
            "topic": DIBinding(
                name="topic", kind=DIKind.FROM_INPUT, inner_type=str, required=True
            )
        }

        n2 = n | Oracle(n=2, merge_prompt="pick best: ${variants}") | Operator(
            when="needs_review"
        )

        _assert_sidecar_intact(n2, fn_expected=fn_a, param_names_expected=("upstream",))
        _assert_param_res_intact(n2, keys_expected={"topic"})

    def test_scripted_shim_private_attr_survives_model_copy(self):
        """_scripted_shim (third PrivateAttr, same mechanism) survives copy."""
        def shim(state, config):  # noqa: ANN001
            return {}

        n = Node(name="a", mode="scripted", outputs=Claims)
        n._scripted_shim = shim

        n2 = n.model_copy(update={"name": "b"})

        assert n2._scripted_shim is shim, "_scripted_shim lost after model_copy"
