"""Agent-Spec dispatch GATE: validate + prepare a machine-emitted spec for
Portal DISPATCH-mode execution, through the canonical ``from_agent_spec`` path.

Extracted from ``factory.py`` (neograph-jtawq.9) -- the no-Command half of the
Portal dispatch mechanism (mode b, ``route="decide"``). The four
``Command(``-constructing dispatch sites (``dispatch_wrapper`` /
``adispatch_wrapper`` in ``factory.py``) STAY in ``factory.py``, so guard G1's
``_ALLOWED`` set ``{factory.py, runner.py}`` is untouched by this move.

``make_dispatch_gate`` is the ONE gate builder; ``factory.make_portal_dispatch_fn``
calls it and invokes the returned handle's ``prepare`` / ``finish`` /
``check_and_increment_depth`` members -- it never re-derives the gate logic
locally. The three functions stay NESTED inside ``make_dispatch_gate`` (not
module-level) so this module's zero-``Any`` audit costs no new
``ANY_ALLOWLIST`` entries -- ``tests/test_guards_any_audit.py``'s
``_walk_public_functions`` only scans module-level definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from neograph._agent_spec_markers import import_pyagentspec
from neograph._state_keys import StateKeys
from neograph._subconstruct import _scan_subgraph_output
from neograph.errors import ConfigurationError, ConstructError, ExecutionError
from neograph.loader import from_agent_spec
from neograph.spec_types import lookup_type

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.runnables import RunnableConfig

    from neograph.modifiers import Portal
    from neograph.node import Node


@dataclass(frozen=True)
class DispatchGate:
    """The gate handle ``make_portal_dispatch_fn`` must consume in full."""

    prepare: Callable[[dict[str, Any]], tuple[Any, type[BaseModel], str, Any, str | None]]
    finish: Callable[[dict[str, Any], dict[str, Any], type[BaseModel], str], dict[str, Any]]
    check_and_increment_depth: Callable[[RunnableConfig], RunnableConfig]


def make_dispatch_gate(
    node: Node,
    portal: Portal,
    *,
    payload_field: str,
    dispatch_field: str,
) -> DispatchGate:
    """Build the no-Command half of a Portal DISPATCH-mode wrapper.

    Sync/async parity: ``prepare`` (load/contract/compile) and ``finish``
    (scan/write) are shared; only ``compiled.invoke`` vs ``await
    compiled.ainvoke`` differs between the twins in ``factory.py``.
    """
    spec_field = portal.spec_field
    input_field = portal.input_field
    assert spec_field is not None and input_field is not None  # dispatch-mode invariant (T1 validation)

    def _resolve_expected() -> type[BaseModel]:
        out = portal.output
        if isinstance(out, str):
            return lookup_type(out)
        assert out is not None  # dispatch-mode invariant (T1 validation)
        return out

    def _prepare(update: dict[str, Any]) -> tuple[Any, type[BaseModel], str, Any, str | None]:
        """Shared pre-invoke: read the emitted spec/input, run the SAME gate, compile.

        Returns ``(compiled, expected_output, spec_name, dispatch_input,
        gate_error_message)``. ``gate_error_message`` is non-None ONLY when
        ``portal.on_invalid == 'route_to_error'`` and the spec-validation gate
        (deserialize + ``from_agent_spec``) failed -- in that case ``compiled``/
        ``dispatch_input`` are meaningless and the caller must route to
        ``portal.error_handler`` instead of invoking. Scope: route_to_error
        covers ONLY this gate failure, never the output-contract-mismatch
        check below it, which always raises regardless of ``on_invalid``.

        Contains NO invoke -- the sync/async twins in ``factory.py`` supply
        that so the gate + compile logic cannot drift between them.
        """
        # `compile` is the ONE cycle-avoidance function-local import here:
        # compiler.py imports _wiring -> factory -> this module, so a
        # module-level `from neograph.compiler import compile` would cycle.
        # AgentSpecDeserializer is function-local for the OTHER reason
        # function-local imports are allowlisted: an optional-dependency
        # guard, routed through the shared `import_pyagentspec` helper
        # (neograph-jtawq.7) -- src/neograph stays Agent-Spec-free by
        # default. from_agent_spec / _scan_subgraph_output / lookup_type are
        # module-level (their modules do not import factory or this module).
        from neograph.compiler import compile as compile_construct

        AgentSpecDeserializer = import_pyagentspec(
            "pyagentspec.serialization", found="ImportError on pyagentspec.serialization"
        ).AgentSpecDeserializer

        decision = update[payload_field]
        spec_dict = getattr(decision, spec_field)
        dispatch_input = getattr(decision, input_field)
        expected = _resolve_expected()
        spec_name = spec_dict.get("name", "<unnamed>") if isinstance(spec_dict, dict) else "<unnamed>"

        try:
            # ONE modifier-aware runtime spec-loading path (Core Invariant):
            # deserialize the Agent-Spec-flavored dict a mode-(b) planner
            # emits into a live Flow, then hand it to the SAME from_agent_spec
            # (01i0g) that reads any other Agent Spec Flow -- never a second,
            # bespoke native-Spec dict-dispatch serializer.
            flow = AgentSpecDeserializer().from_dict(spec_dict)
            sub = from_agent_spec(flow)
        except (ConstructError, ConfigurationError, ValidationError) as gate_error:
            # The emitted spec failed the SAME Construct(...) gate as a hand-written
            # pipeline. on_invalid='route_to_error': signal the caller to route to
            # error_handler instead of raising. on_invalid='raise' (default):
            # surface it wrapped, naming the spec, BEFORE anything runs.
            if portal.on_invalid == "route_to_error":
                return None, expected, spec_name, None, f"{spec_name}: {gate_error}"
            raise ExecutionError.build(
                "dispatched flow spec is invalid",
                construct=spec_name,
                found=str(gate_error),
                node=node.name,
                hint="the emitted spec failed the same Construct(...) validation gate as a hand-written pipeline",
            ) from gate_error

        if sub.output is not None and sub.output is not expected:
            raise ExecutionError.build(
                "dispatched flow output-contract mismatch",
                expected=getattr(expected, "__name__", str(expected)),
                found=f"flow '{spec_name}' declares output {getattr(sub.output, '__name__', sub.output)}",
                node=node.name,
                hint="the emitted flow's declared output must equal Portal.output",
            )

        compiled = compile_construct(sub, scripted=portal.scripted, conditions=portal.conditions)
        return compiled, expected, spec_name, dispatch_input, None

    def _finish(
        update: dict[str, Any], result: dict[str, Any], expected: type[BaseModel], spec_name: str
    ) -> dict[str, Any]:
        """Shared post-invoke: extract the typed output, write ``{node}_dispatch``."""
        out = _scan_subgraph_output(result, expected)
        if out is None:
            raise ExecutionError.build(
                "dispatched flow did not produce the required output type",
                expected=getattr(expected, "__name__", str(expected)),
                found=f"flow '{spec_name}' produced no value assignable to it",
                node=node.name,
                hint="a route='decide' flow must produce Portal.output",
            )
        return {**update, dispatch_field: out}

    def _check_and_increment_depth(config: RunnableConfig) -> RunnableConfig:
        """Read the incoming depth off config, raise if already at
        ``max_depth`` -- BEFORE the dispatcher's own body runs (louder and
        cheaper than checking after: no wasted planner call, no wasted
        spec validation/compile) -- and return a NEW config (copy-not-
        mutate, mirrors ``runner.py``'s ``_ensure_agent_recursion_limit``)
        carrying the depth incremented by exactly one, for the nested
        ``compiled.invoke``/``ainvoke``. Depth is a LINEAGE property across
        fresh per-level compiled sub-flows, so it MUST live on
        ``config['configurable']`` only -- a state-bus counter would reset
        to 0 at every nesting level.
        """
        assert portal.max_depth is not None  # dispatch-mode invariant (T1 validation)
        configurable = dict((config or {}).get("configurable") or {})
        depth = configurable.get(StateKeys.PORTAL_DISPATCH_DEPTH, 0)
        if depth >= portal.max_depth:
            raise ExecutionError.build(
                "Portal dispatch exceeded max_depth",
                construct=node.name,
                found=f"depth {depth} >= max_depth {portal.max_depth}",
                node=node.name,
                hint="a self-extending flow must bound its own recursion via Portal(max_depth=...)",
            )
        new_config: RunnableConfig = {**(config or {})}
        new_config["configurable"] = {**configurable, StateKeys.PORTAL_DISPATCH_DEPTH: depth + 1}
        return new_config

    return DispatchGate(
        prepare=_prepare,
        finish=_finish,
        check_and_increment_depth=_check_and_increment_depth,
    )
