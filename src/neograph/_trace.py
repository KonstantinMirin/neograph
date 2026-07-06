"""Trace-span hygiene for the engine's own callback channel. See neograph-3fm1.

LangGraph already fires ``on_chain_start``/``on_chain_end`` per node to any
handler in ``config["callbacks"]`` (Langfuse, an OTEL bridge, etc.), so the full
DAG tree is already visible to trace consumers. What leaked was hygiene debt:
every explicitly-constructed wrapper ``RunnableLambda`` emitted a CHILD span
named after its inner function (``node_wrapper``, ``subgraph_node``,
``oracle_redirect_fn``, ``agent_sync`` …) — neograph's internals bleeding into
every consumer's trace tree — and those spans carried none of the node attributes
neograph already computes.

``named`` fixes both with a single ``.with_config`` binding:

* ``run_name`` overrides the leaking inner function name so the span reads as the
  user's node name (a ``run_name`` on the binding wins over the wrapped
  runnable's ``__name__``).
* ``tags``/``metadata`` attach the node's mode, declared output type, and id so
  callback backends can index them. Token usage is NOT touched — it stays on the
  LLM child spans where LangChain emits it.

This is a LATE, STATIC config binding (no runtime branching, no tracer
dependency, no neograph-emitted OTEL spans — the locked NON-GOALs of the
companion ``observe=`` ticket). When no callbacks are attached the binding is
inert, so there is zero runtime cost.
"""

from __future__ import annotations

from langchain_core.runnables import Runnable

#: Tag stamped on every neograph graph-node span so backends can filter to
#: "neograph nodes" across a mixed trace tree.
NODE_TAG = "neograph:node"


def named(
    runnable: Runnable,
    name: str,
    *,
    mode: str | None = None,
    output_type: str | None = None,
    node_id: str | None = None,
) -> Runnable:
    """Label a graph-node ``Runnable`` for the engine's callback spans.

    Parameters
    ----------
    runnable:
        The wrapper runnable that becomes a graph node (factory node fn, subgraph
        fn, oracle redirect/merge, each redirect, agent-cycle body, …).
    name:
        The user-facing node name — becomes the span ``run_name``, replacing the
        leaking inner function name.
    mode / output_type / node_id:
        Optional node attributes attached as span metadata + tags. ``None`` values
        are omitted so a metadata-less internal node (a redirect, a barrier) still
        gets its name without empty keys.

    Returns the config-bound runnable. Sync/async dual paths are preserved: the
    binding delegates ``invoke``/``ainvoke`` to the wrapped runnable, which keeps
    selecting its own sync/async twin.
    """
    tags = [NODE_TAG]
    if mode:
        tags.append(f"neograph:mode:{mode}")

    metadata: dict[str, str] = {"neograph_node": name}
    if mode:
        metadata["neograph_mode"] = mode
    if output_type:
        metadata["neograph_output_type"] = output_type
    if node_id:
        metadata["neograph_node_id"] = node_id

    return runnable.with_config(run_name=name, tags=tags, metadata=metadata)
