"""_notify_cost callback dispatch (neograph-dyy7).

``_notify_cost`` calls the user's ``cost_callback`` with a kwargs bundle that
includes optional fields (``node_name`` / ``mode`` / ``duration_s``). It must
decide which kwargs the callback accepts via signature introspection ONCE and
invoke the callback EXACTLY once — never via ``try cb(**full) except TypeError:
cb(**core)``. The except-based arity probe re-invoked the callback whenever the
callback BODY raised ``TypeError`` for a real reason, double-counting cost.
"""

from __future__ import annotations

from neograph._llm import _notify_cost
from neograph._llm_runtime import LlmRuntime

_USAGE = {"input_tokens": 10, "output_tokens": 20}


def _runtime(cb) -> LlmRuntime:
    return LlmRuntime.build(cost_callback=cb)


def test_callback_body_typeerror_does_not_reinvoke():
    """A callback that accepts the kwargs but raises TypeError in its BODY must
    be invoked exactly once — the old except-based fallback re-invoked it."""
    calls: list[dict] = []

    def cb(**kwargs):
        calls.append(kwargs)
        raise TypeError("real bug in user body")

    _notify_cost(_runtime(cb), "fast", _USAGE, node_name="n", mode="think", duration_s=1.0)

    assert len(calls) == 1, f"cost_callback re-invoked on a body TypeError (double-count): {len(calls)} calls"


def test_reduced_signature_callback_invoked_once_with_core_kwargs():
    """A callback declaring only the core triple is invoked once; the optional
    kwargs are filtered out via introspection, so no TypeError, no retry."""
    calls: list[dict] = []

    def cb(tier, input_tokens, output_tokens):
        calls.append({"tier": tier, "input_tokens": input_tokens, "output_tokens": output_tokens})

    _notify_cost(_runtime(cb), "fast", _USAGE, node_name="n", mode="think", duration_s=1.0)

    assert calls == [{"tier": "fast", "input_tokens": 10, "output_tokens": 20}]


def test_full_signature_callback_receives_optional_kwargs():
    """A callback declaring the optional kwargs receives them (single invocation)."""
    calls: list[dict] = []

    def cb(tier, input_tokens, output_tokens, node_name=None, mode=None, duration_s=None):
        calls.append(
            {
                "tier": tier,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "node_name": node_name,
                "mode": mode,
                "duration_s": duration_s,
            }
        )

    _notify_cost(_runtime(cb), "fast", _USAGE, node_name="n", mode="think", duration_s=2.5)

    assert calls == [
        {
            "tier": "fast",
            "input_tokens": 10,
            "output_tokens": 20,
            "node_name": "n",
            "mode": "think",
            "duration_s": 2.5,
        }
    ]


def test_kwargs_callback_receives_everything_once():
    """A **kwargs callback receives the full bundle, exactly once."""
    calls: list[dict] = []

    def cb(**kwargs):
        calls.append(kwargs)

    _notify_cost(_runtime(cb), "fast", _USAGE, node_name="n", mode="agent", duration_s=3.0)

    assert len(calls) == 1
    assert calls[0]["tier"] == "fast"
    assert calls[0]["node_name"] == "n"
    assert calls[0]["mode"] == "agent"
    assert calls[0]["duration_s"] == 3.0
