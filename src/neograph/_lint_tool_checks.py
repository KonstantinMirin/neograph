"""Tool-policy lint checks — ask_human placement, act-mode idempotence, async-only tools.

Extracted from ``lint.py`` (neograph-3ffdg.10) as a pure file split — the
functions below are unchanged, only their home moved. ``lint.py`` re-exports
them and remains the only caller.

``LintIssue`` comes from ``_lint_kind_registry`` rather than from ``lint.py`` so
this module does not import its own parent.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import structlog

from neograph._lint_kind_registry import LintIssue
from neograph.node import Node
from neograph.tool import Tool, is_async_only_tool

log = structlog.get_logger()

_TOOL_BODY_ATTRS = ("func", "coroutine", "_run", "_arun", "invoke", "ainvoke")


def _tool_references_ask_human(tool_obj: Any) -> bool:
    """True when any of a tool's callable bodies references ``ask_human`` by name.

    Direct-reference heuristic: scans each body's ``__code__.co_names`` (which
    includes imported and attribute-accessed names) for ``"ask_human"``. This is
    exactly why ``ask_human`` is a NAMED marker — a raw ``interrupt()`` call is
    invisible, but ``from neograph.hitl import ask_human`` shows up here. The
    heuristic misses alias imports and indirection through helpers; a consumer
    who alias-hides ``ask_human`` opts out of this safety net.
    """
    for attr in _TOOL_BODY_ATTRS:
        fn = getattr(tool_obj, attr, None)
        if fn is None:
            continue
        code = getattr(fn, "__code__", None) or getattr(getattr(fn, "__func__", None), "__code__", None)
        if code is not None and "ask_human" in code.co_names:
            return True
    return False


def _check_ask_human_in_mutating_node(
    node: Node,
    issues: list[LintIssue],
    *,
    tool_factories: dict[str, Callable] | None = None,
) -> None:
    """Warn when ``ask_human`` is reachable from an act-mode (mutating) node.

    A non-idempotent side effect performed *before* a mid-loop pause in the same
    node can double-fire on resume — LangGraph memoizes at node granularity, and
    a ReAct loop runs many tool steps inside one node (the residual "Level-B"
    case documented in docs/design/durable-execution-replay-research-2026-07-02.md).
    ``ask_human`` makes the pause a marker the linter can see, so an act-mode
    node carrying it is flagged.

    This is a WARN (``required=False``): the legitimate ask_human-then-idempotent
    -mutate pattern must not be blocked. The rule gates on the DECLARED
    ``node.mode == 'act'`` (act == mutations, agent == read-only); a mutating tool
    mislabeled ``mode='agent'`` escapes the net — an accepted limitation of
    trusting the declared mode.
    """
    if node.mode != "act" or not node.tools:
        return

    for spec in node.tools:
        tool_obj = _resolve_tool_object(spec, tool_factories)
        if tool_obj is None:
            continue
        if _tool_references_ask_human(tool_obj):
            tool_name = str(getattr(spec, "name", None) or getattr(tool_obj, "name", "?"))
            issues.append(
                LintIssue(
                    node_name=f"Node '{node.name}'",
                    param=tool_name,
                    kind="ask_human_in_mutating_node",
                    required=False,
                    message=(
                        f"Node '{node.name}': act-mode (mutating) tool '{tool_name}' "
                        "calls ask_human(). A non-idempotent side effect before the "
                        "mid-loop pause can double-fire on resume (node-granularity "
                        "replay). Ensure any mutation before the ask_human() is "
                        "idempotent, or move it after the pause."
                    ),
                )
            )


def _check_act_mode_all_idempotent(
    node: Node,
    issues: list[LintIssue],
) -> None:
    """Warn when an act-mode node's tools are ALL idempotent. See neograph-lhc6.

    ``act`` declares mutations, ``agent`` declares read-only. A node whose every
    tool is marked ``idempotent=True`` performs no non-idempotent side effect, so
    ``mode='act'`` is almost certainly a misclassification -- it should be
    ``agent`` (read-only). Flagging it keeps the act/agent distinction honest,
    which the replay-safety gate (hydration re-derivation) relies on.

    WARN (``required=False``): a genuinely idempotent mutation (HTTP PUT) is a
    legitimate act-mode-of-idempotent-tools shape, so this must not block. The
    rule fires only when EVERY tool is a ``Tool`` spec known to be idempotent; a
    raw BaseTool or any non-idempotent spec has unknown/mutating side effects, so
    the node cannot be concluded misclassified and the rule stays silent.
    """
    if node.mode != "act" or not node.tools:
        return

    if all(isinstance(spec, Tool) and spec.idempotent for spec in node.tools):
        tool_names = ", ".join(str(spec.name) for spec in node.tools)
        issues.append(
            LintIssue(
                node_name=f"Node '{node.name}'",
                param="mode",
                kind="act_mode_all_idempotent_tools",
                required=False,
                message=(
                    f"Node '{node.name}': mode='act' (mutations) but all tools "
                    f"({tool_names}) are idempotent=True (read-only). This is probably "
                    "a misclassification -- use mode='agent'. If a tool is a genuinely "
                    "idempotent mutation, this warning is expected and can be ignored."
                ),
            )
        )


def _spec_factory(spec: Any, tool_factories: dict[str, Callable] | None) -> Any:
    """The registered factory for a Tool spec, or None when the spec carries a
    pre-bound tool (raw BaseTool) or is not a Tool. Used to introspect a factory
    (e.g. detect a coroutine factory) WITHOUT calling it."""
    if isinstance(spec, Tool) and getattr(spec, "_bound_tool", None) is None:
        return (tool_factories or {}).get(spec.name)
    return None


def _resolve_tool_object(spec: Any, tool_factories: dict[str, Callable] | None) -> Any:
    """Resolve the concrete tool object for a Tool spec, or None if unavailable.

    Prefers the bound tool carried on a spec synthesized from a raw BaseTool;
    otherwise instantiates the registered factory. Never raises — lint must not
    fail because a factory misbehaves.
    """
    if isinstance(spec, Tool):
        bound = getattr(spec, "_bound_tool", None)
        if bound is not None:
            return bound
        factory = (tool_factories or {}).get(spec.name)
        if factory is None:
            return None
        try:
            return factory({}, spec.config)
        except Exception as exc:  # noqa: BLE001
            # lint must not crash because a tool factory misbehaves; a tool it
            # cannot instantiate simply yields no async-only finding.
            log.debug("lint_tool_factory_failed", tool=spec.name, error=str(exc))
            return None
    # A raw BaseTool that slipped through un-normalized — introspect directly.
    return spec


def _check_async_only_tools(
    node: Node,
    issues: list[LintIssue],
    *,
    tool_factories: dict[str, Callable] | None = None,
) -> None:
    """Flag agent/act nodes bound to an async-only (MCP) tool.

    An async-only tool (StructuredTool with a coroutine and no sync func — the
    langchain-mcp-adapters shape) cannot run under the sync ``run()`` driver;
    it requires ``arun()``. lint() cannot know the driver statically, so it
    warns whenever such a tool is bound. The tool object is resolved either from
    ``Tool._bound_tool`` (raw BaseTool passed in tools=) or by instantiating the
    registered factory.
    """
    if node.mode not in ("agent", "act") or not node.tools:
        return

    for spec in node.tools:
        factory = _spec_factory(spec, tool_factories)
        if factory is not None and asyncio.iscoroutinefunction(factory):
            # An async tool factory requires the arun() driver. Classify it
            # WITHOUT calling: invoking a coroutine factory here would create an
            # un-awaited coroutine (RuntimeWarning) and misintrospect that
            # coroutine object as the tool.
            tool_name = str(getattr(spec, "name", None) or "?")
            issues.append(
                LintIssue(
                    node_name=f"Node '{node.name}'",
                    param=tool_name,
                    kind="tool_requires_async_driver",
                    required=False,
                    message=(
                        f"Node '{node.name}': tool '{tool_name}' has an async tool "
                        "factory (e.g. it awaits a per-run token provider or builds "
                        "an MCP client) and cannot run under the sync run() driver. "
                        "Drive this graph with arun() so the async tool loop is used."
                    ),
                )
            )
            continue
        tool_obj = _resolve_tool_object(spec, tool_factories)
        if tool_obj is None:
            continue
        tool_name = str(getattr(spec, "name", None) or getattr(tool_obj, "name", "?"))
        if is_async_only_tool(tool_obj):
            issues.append(
                LintIssue(
                    node_name=f"Node '{node.name}'",
                    param=tool_name,
                    kind="tool_requires_async_driver",
                    required=False,
                    message=(
                        f"Node '{node.name}': tool '{tool_name}' is async-only "
                        "(e.g. an MCP tool) and cannot run under the sync run() "
                        "driver. Drive this graph with arun() so the async tool "
                        "loop is used."
                    ),
                )
            )
