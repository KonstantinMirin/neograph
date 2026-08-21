"""A scripted node reads run state as a normal typed input, including under Each.

GH #15 asks how a step reaches a value produced earlier in the run when its
port carries something else. For an LLM-mode node the answer is `context=`.
For a SCRIPTED node the answer is that there was never a special mechanism to
find: an upstream read IS a normal declared input, and dict-form fan-in lets a
fanned branch declare both -- the fanned item and the upstream -- at once.

`neograph-7e065` proposed widening `context=` to scripted nodes. These tests
are why that was closed instead of built: the capability exists, and the
existing route is BETTER. A fan-in input is type-checked by the validator and
creates a real dataflow edge; a `context` field is typed `Any` in `state.py`
and declares no edge. Adding the second route would trade a checked mechanism
for an unchecked one and leave two ways to do one thing.

They are pinned rather than merely observed because "it already works" is only
a reason to skip building something if it keeps working.
"""

from __future__ import annotations

import sys
import types

from pydantic import BaseModel

from neograph import Construct, Each, Node, compile, construct_from_module, node, run
from tests.fakes import build_test_compile_kwargs, register_scripted


class RunCtx(BaseModel, frozen=True):
    deal_id: int


class Claim(BaseModel, frozen=True):
    text: str


class Claims(BaseModel, frozen=True):
    items: tuple[Claim, ...]


class Out(BaseModel, frozen=True):
    saw: str


class TestScriptedNodeReadsRunState:
    def test_a_fanned_scripted_branch_reads_both_its_item_and_an_upstream(self):
        """Programmatic surface. The port carries WHICH ITEM, the second
        fan-in key carries WHICH RUN -- the exact split GH #15 describes."""
        register_scripted("srs_ctx", lambda i, c: RunCtx(deal_id=4822))
        register_scripted("srs_claims", lambda i, c: Claims(items=(Claim(text="a"), Claim(text="b"))))
        register_scripted("srs_branch", lambda i, c: Out(saw=f"{i['item'].text}@{i['ctx'].deal_id}"))

        fan = Node(
            name="branch",
            mode="scripted",
            scripted_fn="srs_branch",
            inputs={"item": Claim, "ctx": RunCtx},
            outputs=Out,
            fan_out_param="item",
        ) | Each(over="claims.items", key="text")
        pipeline = Construct(
            "scripted-backref",
            nodes=[
                Node.scripted("ctx", fn="srs_ctx", outputs=RunCtx),
                Node.scripted("claims", fn="srs_claims", outputs=Claims),
                fan,
            ],
        )

        result = run(compile(pipeline, **build_test_compile_kwargs()), input={"node_id": "t"})

        assert {k: v.saw for k, v in result["branch"].items()} == {"a": "a@4822", "b": "b@4822"}

    def test_the_same_shape_through_the_node_decorator(self):
        """@node surface -- the one most people write. `item` is the fanned
        value and `ctx` is an ordinary upstream parameter."""
        module = types.ModuleType("srs_decorator_mod")

        @node(outputs=RunCtx)
        def ctx() -> RunCtx:
            return RunCtx(deal_id=4822)

        @node(outputs=Claims)
        def claims() -> Claims:
            return Claims(items=(Claim(text="a"), Claim(text="b")))

        @node(outputs=Out, map_over="claims.items", map_key="text")
        def branch(item: Claim, ctx: RunCtx) -> Out:
            return Out(saw=f"{item.text}@{ctx.deal_id}")

        for fn in (ctx, claims, branch):
            setattr(module, fn.name.replace("-", "_"), fn)
        sys.modules[module.__name__] = module
        try:
            pipeline = construct_from_module(module, name="decorator-backref")
            result = run(compile(pipeline, **build_test_compile_kwargs()), input={"node_id": "t"})
        finally:
            sys.modules.pop(module.__name__, None)

        assert {k: v.saw for k, v in result["branch"].items()} == {"a": "a@4822", "b": "b@4822"}
