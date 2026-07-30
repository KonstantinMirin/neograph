"""Compile-time diagnostics — shim collection, DI requirements, DAG description.

Extracted from ``compiler.py`` (neograph-3ffdg.8) as a pure file split — the
functions below are unchanged, only their home moved. ``compiler.py``
re-exports them, so existing imports keep resolving.

None of these touch the StateGraph build or the modifier-combo dispatch: they
walk an already-assembled Construct to collect scripted shims, report which DI
bindings a pipeline requires, and render the DAG for humans.
"""

from __future__ import annotations

from typing import Any

import structlog

from neograph._sidecar import _get_param_res
from neograph.construct import Construct, iter_nodes
from neograph.di import DIKind

log = structlog.get_logger()


def _collect_scripted_shims(construct: Construct) -> dict[str, Any]:
    """Walk the construct tree and build the per-compile scripted dict.

    For each Node with a `_scripted_shim` PrivateAttr (attached by the
    `@node` decorator path via `_register_node_scripted`), insert the
    shim under `node.scripted_fn` into the returned dict. Sub-constructs
    are walked recursively.
    """
    lookup: dict[str, Any] = {}
    for item in iter_nodes(construct):
        shim = getattr(item, "_scripted_shim", None)
        if shim is not None and item.scripted_fn:
            lookup[item.scripted_fn] = shim
    return lookup


def _collect_required_di(construct: Construct) -> dict[str, set[str]]:
    """Walk all nodes and collect required DI param names by source (input/config).

    Returns {"input": {"topic", "node_id"}, "config": {"limiter"}} — the set of
    param names that must be present in run(input=) or config['configurable'].
    """
    required: dict[str, set[str]] = {"input": set(), "config": set()}
    for item in iter_nodes(construct):
        param_res = _get_param_res(item)
        if not param_res:
            continue
        for _pname, binding in param_res.items():
            if not binding.required:
                continue
            if binding.kind in (DIKind.FROM_INPUT, DIKind.FROM_INPUT_MODEL):
                if binding.kind == DIKind.FROM_INPUT_MODEL:
                    # Bundled model — each field is a required input key
                    model_cls = binding.model_cls
                    if model_cls is not None:
                        for fname in model_cls.model_fields:
                            required["input"].add(fname)
                else:
                    required["input"].add(binding.name)
            elif binding.kind in (DIKind.FROM_CONFIG, DIKind.FROM_CONFIG_MODEL):
                if binding.kind == DIKind.FROM_CONFIG_MODEL:
                    model_cls = binding.model_cls
                    if model_cls is not None:
                        for fname in model_cls.model_fields:
                            required["config"].add(fname)
                else:
                    required["config"].add(binding.name)
    return required


def describe_graph(compiled: Any) -> str:
    """Return a Mermaid diagram string for a compiled graph.

    Usage::

        graph = compile(pipeline)
        print(describe_graph(graph))

    Paste the output into any Mermaid renderer (GitHub, docs, mermaid.live).
    """
    try:
        return compiled.get_graph().draw_mermaid()
    except (AttributeError, TypeError, ValueError) as exc:
        log.debug("describe_graph_failed", error=str(exc))
        return "(graph visualization not available)"


def _print_dag_summary(compiled: Any, construct: Any) -> None:
    """Print a human-readable DAG summary to stderr in dev mode."""
    import sys

    try:
        lg_graph = compiled.get_graph()
    except (AttributeError, TypeError, ValueError):
        return

    nodes = [n for n in lg_graph.nodes if n not in ("__start__", "__end__")]
    edges = lg_graph.edges

    lines = [f"[neograph-dev] Compiled '{construct.name}' ({len(nodes)} nodes):"]

    for edge in edges:
        src = edge.source.replace("__start__", "START").replace("__end__", "END")
        tgt = edge.target.replace("__start__", "START").replace("__end__", "END")
        cond = " [conditional]" if edge.conditional else ""
        lines.append(f"  {src} -> {tgt}{cond}")

    print("\n".join(lines), file=sys.stderr)
