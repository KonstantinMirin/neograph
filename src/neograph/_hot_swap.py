"""``resume_from_agent_spec()`` — durable Tier-2 hot-swap of a running graph.

Cite docs/design/agent-spec-interop-2026-07-09.md §1a (motivating use case) and
docs/design/dynamic-handoff-research-2026-07-13.md (Tier-2 compose note).

A running agent re-emits its graph as an Open Agent Spec ``Flow`` at runtime,
recompiles it, and resumes durably on the SAME ``thread_id``. This helper is a
THIN public COMPOSE over three primitives that already exist and already
validate/rewind — it adds NO rewind, fingerprint, or validation logic of its
own (that would fork the single seam the Core Invariant forbids):

    from_agent_spec(flow)  ->  Construct.__init__ (_validate_node_chain), the
                                type-channel gate that rejects a machine-authored
                                spec BEFORE any node runs
    compile(construct, checkpointer=<same>)  ->  schema/node fingerprints
    run(graph, config=<same thread_id>, auto_resume=True)  ->  the EXISTING
                                _auto_resume_from_divergence rewind re-runs only
                                fingerprint-invalidated nodes, reusing state.

In-graph analog: ``factory.py``'s ``make_portal_dispatch_fn._prepare`` (~440-490)
runs the same ``from_agent_spec(flow) -> Construct(...)`` gate BEFORE
``compiled.invoke`` — but that in-graph path WRAPS the gate error in
``ExecutionError``. This OUT-of-graph durable sibling deliberately raises the
gate error RAW (``ConstructError``/``ConfigurationError``) straight to the
caller, so the two paths stay one discipline without one forking into the
other's wrapping behavior.

FAIL LOUD ON MISSING DURABILITY: durable resume is DEFINITIONAL for Tier-2.
``run(..., auto_resume=True)`` with ``checkpointer=None`` or an absent
``thread_id`` silently short-circuits into a full re-run with zero state reuse
(``runner.py`` ``_verify_checkpoint_schema`` returns early) — the checkpointer-
less path is Tier-1, a different feature. The helper raises ``ConfigurationError``
in that case rather than let the silent seam through.

SCOPE BOUND: think / scripted (+ Oracle/Each/Loop/Operator modifiers) pipelines
only. ``from_agent_spec`` fails loud on an ``AgentNode`` (agent/act mode); a
hot-swap of an agent/act mesh awaits its importer support and is out of scope
here — the fail-loud on import is the honest, correct bound today.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from neograph.compiler import compile
from neograph.errors import ConfigurationError
from neograph.loader import from_agent_spec
from neograph.runner import arun, run

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from pyagentspec.flows.flow import Flow

log = structlog.get_logger()

__all__ = ["resume_from_agent_spec", "aresume_from_agent_spec"]


def _require_durability(
    checkpointer: BaseCheckpointSaver | None,
    config: RunnableConfig | None,
) -> None:
    """Fail loud when the durable-resume preconditions are absent.

    Durable resume is definitional for Tier-2: without a checkpointer or a
    ``thread_id`` the underlying ``run(..., auto_resume=True)`` would silently
    full-re-run with zero state reuse. The checkpointer-less path is Tier-1, a
    different feature — so we reject it here rather than let it degrade quietly.
    """
    if checkpointer is None:
        raise ConfigurationError.build(
            "resume_from_agent_spec requires a checkpointer",
            expected="a checkpointer (the SAME saver used for the original run)",
            found="checkpointer=None",
            hint="durable resume is definitional for a Tier-2 hot-swap; the "
            "checkpointer-less path is Tier-1 data-driven routing, a different feature.",
        )
    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    if not thread_id:
        raise ConfigurationError.build(
            "resume_from_agent_spec requires config['configurable']['thread_id']",
            expected="the SAME thread_id the original run used",
            found="no thread_id in config['configurable']",
            hint="durable resume rewinds the checkpoint for that thread.",
        )


def resume_from_agent_spec(
    flow: Flow,
    *,
    checkpointer: BaseCheckpointSaver,
    config: RunnableConfig,
    auto_resume: bool = True,
    **runtime_kwargs: Any,
) -> Any:
    """Recompile an Agent Spec ``Flow`` and resume it durably on the same thread.

    Composes ``from_agent_spec`` (validates before execution) -> ``compile``
    (same checkpointer, recomputes fingerprints) -> ``run`` (same ``thread_id``,
    ``auto_resume=True`` by default so the existing rewind re-runs only the
    fingerprint-invalidated nodes). See the module docstring for the in-graph
    analog and the fail-loud-on-missing-durability contract.

    Args:
        flow: The emitted Open Agent Spec ``Flow`` (e.g. from ``to_agent_spec``).
        checkpointer: The SAME checkpointer used for the original run (durable
            resume rewinds its stored checkpoint). Required.
        config: A ``RunnableConfig`` carrying ``configurable.thread_id`` — the
            SAME thread as the original run. Required.
        auto_resume: Forwarded to ``run``; ``True`` (default) selectively
            re-executes only invalidated nodes.
        **runtime_kwargs: Forwarded to ``compile`` for LLM-mode nodes and
            scripted lookups (``llm_factory`` / ``prompt_compiler`` / ``renderer``
            / ``scripted`` / ``conditions`` / ``tool_factories``).

    Returns:
        The ``run`` result of the resumed graph.

    Raises:
        ConfigurationError: no checkpointer or no ``thread_id`` (missing
            durability), or an unsupported node (e.g. ``AgentNode``) surfaced by
            ``from_agent_spec``.
        ConstructError: the reconstructed ``Construct`` fails type-channel
            validation — raised RAW before any node runs.
    """
    _require_durability(checkpointer, config)
    construct = from_agent_spec(flow)
    graph = compile(construct, checkpointer=checkpointer, **runtime_kwargs)
    return run(graph, config=config, auto_resume=auto_resume)


async def aresume_from_agent_spec(
    flow: Flow,
    *,
    checkpointer: BaseCheckpointSaver,
    config: RunnableConfig,
    auto_resume: bool = True,
    **runtime_kwargs: Any,
) -> Any:
    """Async twin of :func:`resume_from_agent_spec` (composes ``arun``).

    Driver-parallel to the sync helper: identical validate-before-execution and
    fail-loud-on-missing-durability contracts, diverging only at the engine verb
    (``arun`` instead of ``run``), so the ``arun``/``astream`` rewind twins get
    the same selective-rerun coverage.
    """
    _require_durability(checkpointer, config)
    construct = from_agent_spec(flow)
    graph = compile(construct, checkpointer=checkpointer, **runtime_kwargs)
    return await arun(graph, config=config, auto_resume=auto_resume)
