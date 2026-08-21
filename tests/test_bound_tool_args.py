"""A run value bound into a tool call cannot be overridden by the model.

GH #15's production failure: a fan-out branch made 49 tool calls whose
``dealId`` read 1, 2, 3, 4, 5, 1001. The model composed them. Every one queried
a deal that does not exist, each returned ACCESS_DENIED with empty data, and
the pipeline read that as "this deal has no data" and concluded "blocked" from
evidence about nothing. Nothing errored; every gate stayed green.

``context=`` makes the model SEE the right value, which is a large improvement
over it seeing nothing. It is not a guarantee -- the tool call is still composed
by the model, so an invented argument stays representable. This suite is about
making it UNREPRESENTABLE for a declared argument, which is the subtractive move
the project exists to make.

The fake below emits a WRONG value deliberately. That is the point: a test where
the model cooperates proves nothing about a model that does not.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import Construct, Node, Tool, compile, run
from tests.fakes import (
    ReActFake,
    build_test_compile_kwargs,
    configure_fake_llm,
    register_scripted,
    register_tool_factory,
)


class RunCtx(BaseModel, frozen=True):
    deal_id: int


class Verdict(BaseModel, frozen=True):
    text: str


class _RecordingTool:
    """Captures the args the framework actually handed the tool."""

    name = "get_deal"

    def __init__(self, seen: list):
        self._seen = seen

    def invoke(self, args, config=None, **kwargs):
        self._seen.append(dict(args))
        return f"deal {args.get('deal_id')} ok"


def _pipeline(seen: list, *, bound: bool):
    register_scripted("bta_ctx", lambda i, c: RunCtx(deal_id=4822))
    register_tool_factory("get_deal", lambda config, tool_config: _RecordingTool(seen))

    tool_kwargs = {"bound_args": {"deal_id": "ctx.deal_id"}} if bound else {}
    agent = Node(
        name="probe",
        mode="agent",
        inputs=RunCtx,
        outputs=Verdict,
        model="fast",
        prompt="check",
        tools=[Tool(name="get_deal", budget=3, **tool_kwargs)],
    )
    return Construct("bound-args", nodes=[Node.scripted("ctx", fn="bta_ctx", outputs=RunCtx), agent])


def _run(pipeline):
    # The model emits deal_id=1 -- an id that does not exist. Exactly the
    # reported failure, scripted rather than hoped for.
    fake = ReActFake(
        tool_calls=[[{"name": "get_deal", "args": {"deal_id": 1}, "id": "c1"}], []],
        final=lambda m: m(text="done"),
    )
    kw = configure_fake_llm(lambda tier: fake)
    graph = compile(pipeline, **kw, **build_test_compile_kwargs())
    return run(graph, input={"node_id": "t"})


class TestBoundToolArguments:
    def test_the_model_cannot_override_a_bound_argument(self):
        """The acceptance: a declared argument comes from run state, and a model
        emitting something else does not change what the tool receives."""
        seen: list = []

        _run(_pipeline(seen, bound=True))

        assert seen, "the tool was never invoked"
        assert seen[0]["deal_id"] == 4822, (
            f"the model's invented deal_id survived into the tool call: {seen[0]}"
        )

    def test_without_the_binding_the_model_wins(self):
        """The control. Without `bound_args` the model's value reaches the tool
        -- which is today's behaviour and the whole reported defect. If this
        ever passes for the wrong reason, the test above proves nothing."""
        seen: list = []

        _run(_pipeline(seen, bound=False))

        assert seen and seen[0]["deal_id"] == 1, (
            "the control no longer reproduces the defect, so the guarded case is unproven"
        )

    def test_an_unbound_argument_is_left_alone(self):
        """Binding one argument must not swallow the others the model supplies."""
        seen: list = []
        register_scripted("bta_ctx", lambda i, c: RunCtx(deal_id=4822))
        register_tool_factory("get_deal", lambda config, tool_config: _RecordingTool(seen))

        agent = Node(
            name="probe",
            mode="agent",
            inputs=RunCtx,
            outputs=Verdict,
            model="fast",
            prompt="check",
            tools=[Tool(name="get_deal", budget=3, bound_args={"deal_id": "ctx.deal_id"})],
        )
        pipeline = Construct(
            "bound-mixed", nodes=[Node.scripted("ctx", fn="bta_ctx", outputs=RunCtx), agent]
        )
        fake = ReActFake(
            tool_calls=[[{"name": "get_deal", "args": {"deal_id": 1, "note": "keep me"}, "id": "c1"}], []],
            final=lambda m: m(text="done"),
        )
        kw = configure_fake_llm(lambda tier: fake)
        run(compile(pipeline, **kw, **build_test_compile_kwargs()), input={"node_id": "t"})

        assert seen[0] == {"deal_id": 4822, "note": "keep me"}

    def test_an_unresolvable_binding_fails_at_assembly(self):
        """Declared means checkable. A binding whose root no upstream produces
        must fail before a run, not resolve to None at call time."""
        from neograph import ConstructError

        register_scripted("bta_ctx2", lambda i, c: RunCtx(deal_id=1))
        register_tool_factory("get_deal", lambda config, tool_config: _RecordingTool([]))

        with pytest.raises(ConstructError, match="bound_args"):
            Construct(
                "bad-binding",
                nodes=[
                    Node.scripted("ctx", fn="bta_ctx2", outputs=RunCtx),
                    Node(
                        name="probe",
                        mode="agent",
                        inputs=RunCtx,
                        outputs=Verdict,
                        model="fast",
                        prompt="check",
                        tools=[Tool(name="get_deal", bound_args={"deal_id": "nope.deal_id"})],
                    ),
                ],
            )
