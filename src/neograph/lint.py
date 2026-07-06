"""lint() — validate DI bindings and template placeholders against config/inputs.

Walks all nodes in a Construct and checks:
1. Every FromInput/FromConfig parameter has a matching key in the config dict.
2. Every ${var} placeholder in inline prompts resolves to a known input key.

Returns a list of LintIssue dataclass instances (never raises — reports all problems).
"""

from __future__ import annotations

import asyncio
import string
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog

from neograph._ir_branch import iter_with_arms
from neograph._ir_protocols import ConstructItem
from neograph._llm_runtime import (
    _ACCEPT_ALL,
    _accepted_params,
    collect_llm_nodes,
    missing_runtime_kwargs,
)
from neograph._normalize import normalize_inputs
from neograph._placeholders import DOLLAR_RE
from neograph._runtime_registry import _decoration_registry
from neograph._sidecar import _get_param_res, get_merge_fn_metadata
from neograph._state_keys import StateKeys
from neograph.construct import Construct
from neograph.di import DI_TEMPLATE_KINDS, DIBinding, DIKind
from neograph.node import Node
from neograph.tool import Tool, is_async_only_tool

log = structlog.get_logger()

# Standard keys always available in state / config
_KNOWN_EXTRAS: frozenset[str] = frozenset({
    StateKeys.NODE_ID, StateKeys.PROJECT_ROOT, StateKeys.HUMAN_FEEDBACK,
})

# The ${var} scanner is the ONE shared in _placeholders — imported, not redefined
# (byte-identical dedup: lint collects names, prompt.substitute fills them, both
# off one grammar). Aliased to preserve the existing local name.
_PLACEHOLDER_RE = DOLLAR_RE


@dataclass
class LintIssue:
    """A single lint problem — DI binding or template placeholder."""

    node_name: str
    param: str
    kind: str
    message: str
    required: bool = False


def _check_binding(
    node_label: str,
    binding: DIBinding,
    config: dict[str, Any] | None,
    issues: list[LintIssue],
) -> None:
    """Check a single DI binding against config.

    ``node_label`` is pre-formatted by the caller — node and merge_fn paths
    use different naming conventions, so the caller supplies the label.
    """
    kind_str = binding.kind.value

    if binding.kind in (DIKind.FROM_INPUT, DIKind.FROM_CONFIG):
        if config is not None:
            if binding.name not in config:
                issues.append(LintIssue(
                    node_name=node_label,
                    param=binding.name,
                    kind=kind_str,
                    required=binding.required,
                    message=(
                        f"{node_label}: DI parameter '{binding.name}' "
                        f"({kind_str}) not found in config"
                    ),
                ))
        elif binding.required:
            issues.append(LintIssue(
                node_name=node_label,
                param=binding.name,
                kind=kind_str,
                required=True,
                message=(
                    f"{node_label}: required DI parameter '{binding.name}' "
                    f"({kind_str}) has no config to resolve from"
                ),
            ))

    elif binding.kind in (DIKind.FROM_INPUT_MODEL, DIKind.FROM_CONFIG_MODEL):
        model_cls: Any = binding.model_cls or binding.inner_type
        required = binding.required
        if config is not None:
            for fname in model_cls.model_fields:
                if fname not in config:
                    issues.append(LintIssue(
                        node_name=node_label,
                        param=fname,
                        kind=kind_str,
                        required=required,
                        message=(
                            f"{node_label}: bundled model field "
                            f"'{fname}' ({kind_str} via {model_cls.__name__}) "
                            f"not found in config"
                        ),
                    ))
        elif required:
            for fname in model_cls.model_fields:
                issues.append(LintIssue(
                    node_name=node_label,
                    param=fname,
                    kind=kind_str,
                    required=True,
                    message=(
                        f"{node_label}: required bundled model "
                        f"field '{fname}' ({kind_str} via "
                        f"{model_cls.__name__}) has no config"
                    ),
                ))


def lint(
    construct: Construct,
    *,
    config: dict[str, Any] | None = None,
    known_template_vars: set[str] | None = None,
    template_resolver: Callable[[str], str | None] | None = None,
    llm_factory: Any = None,
    prompt_compiler: Any = None,
    conditions: dict[str, Callable] | None = None,
    tool_factories: dict[str, Callable] | None = None,
) -> list[LintIssue]:
    """Validate DI bindings and template placeholders in *construct*.

    Walks every node (recursing into sub-constructs). Checks:
    1. FromInput/FromConfig parameters exist in the provided config dict.
    2. Inline prompt ``${var}`` placeholders resolve to known input keys.
    3. Template-ref prompt ``{placeholder}`` names resolve when a
       *template_resolver* is provided.

    *config* is the FLAT inner configurable mapping that DI bindings
    resolve against (e.g., ``{"node_id": "x", "project_root": "/p"}``),
    NOT a full LangChain ``RunnableConfig`` envelope. This is intentional
    -- lint validates the user's resolved config payload, not the
    transport shape. Hence ``dict[str, Any]`` rather than ``RunnableConfig``.

    *known_template_vars* is a set of extra variable names the consumer's
    prompt pipeline provides (e.g., ``{"topic", "json_schema"}``). These
    are accepted as valid alongside the standard framework extras
    (node_id, project_root, human_feedback).

    *template_resolver* maps a template name (e.g., ``"rw/summarize"``) to
    the template text string, or ``None`` if the template can't be found.
    When provided, lint reads the template text, extracts ``{placeholder}``
    names, and validates them against predicted input keys.

    Returns a list of LintIssue instances. An empty list means all bindings
    are satisfied.

    Fail-loud LLM kwarg surfacing (§2): when the construct contains any
    LLM-mode node (think/agent/act) and neither the supplied kwargs nor the
    legacy `configure_llm()` compat slot provides `llm_factory` and
    `prompt_compiler`, `lint()` emits a `LintIssue(kind="llm_kwargs_missing")`
    naming the offending node(s). The compile-time path (`compile()`) raises;
    `lint()` surfaces the same contract as a discoverable issue.
    """
    issues: list[LintIssue] = []
    _emit_missing_llm_kwargs_issue(construct, llm_factory, prompt_compiler, issues)

    # Seed the tool-factory lookup the same way compile() does: decoration-time
    # registrations (@tool, auto-registered raw BaseTools) plus explicit kwargs.
    tool_factory_lookup: dict[str, Callable] = dict(_decoration_registry.tool_factory)
    if tool_factories:
        tool_factory_lookup.update(tool_factories)

    all_known = _KNOWN_EXTRAS | (known_template_vars or set())
    _walk(construct, config, issues, known_vars=all_known,
          template_resolver=template_resolver,
          conditions=conditions,
          tool_factories=tool_factory_lookup,
          di_inputs_enabled=_compiler_accepts_di_inputs(prompt_compiler))
    return issues


def _compiler_accepts_di_inputs(prompt_compiler: Any) -> bool:
    """True when *prompt_compiler* declares a ``di_inputs`` param (or ``**kwargs``).

    This is the third column of the inline/template-ref key asymmetry: a
    FromInput/FromConfig parameter name is a VALID template-ref placeholder only
    when the app's compiler opts in by accepting ``di_inputs`` — otherwise the
    resolved DI value never reaches the template and the placeholder is
    unresolvable. Reuses the ONE signature-introspection helper
    (``_accepted_params``) that the runtime uses to gate the kwarg.
    """
    if prompt_compiler is None:
        return False
    params = _accepted_params(prompt_compiler)
    return params is _ACCEPT_ALL or "di_inputs" in params


def _emit_missing_llm_kwargs_issue(
    construct: Construct,
    llm_factory: Any,
    prompt_compiler: Any,
    issues: list[LintIssue],
) -> None:
    """Surface a `llm_kwargs_missing` LintIssue when LLM-mode nodes lack runtime config.

    This is the lint-surface counterpart to compile()'s fail-loud raise: the
    contract is the same (§2 requires LLM kwargs), but lint() reports it as
    a discoverable issue rather than raising.
    """
    llm_nodes = collect_llm_nodes(construct)
    if not llm_nodes:
        return

    missing = missing_runtime_kwargs(llm_factory, prompt_compiler)
    if not missing:
        return

    issues.append(LintIssue(
        node_name=", ".join(llm_nodes),
        param="",
        kind="llm_kwargs_missing",
        message=(
            f"LLM-mode nodes ({', '.join(llm_nodes)}) require "
            f"{' and '.join(missing)} at compile() time. "
            "Pass these kwargs to compile() or configure them via "
            "configure_llm() (legacy)."
        ),
    ))


def _walk(
    item: ConstructItem,
    config: dict[str, Any] | None,
    issues: list[LintIssue],
    *,
    known_vars: frozenset[str] | set[str] = _KNOWN_EXTRAS,
    template_resolver: Callable[[str], str | None] | None = None,
    conditions: dict[str, Callable] | None = None,
    tool_factories: dict[str, Callable] | None = None,
    di_inputs_enabled: bool = False,
) -> None:
    """Recursively walk a construct and check DI bindings + template placeholders."""
    if isinstance(item, Construct):
        # Check Loop condition on the Construct itself (Construct | Loop)
        _check_loop_condition(item, issues, conditions=conditions)
        # iter_with_arms expands _BranchNode sentinels so a bare arm Node's DI
        # bindings + template placeholders are linted like any other node. See
        # neograph-vn5f (site 3).
        for child in iter_with_arms(item):
            _walk(child, config, issues, known_vars=known_vars,
                  template_resolver=template_resolver,
                  conditions=conditions,
                  tool_factories=tool_factories,
                  di_inputs_enabled=di_inputs_enabled)
        return

    if not isinstance(item, Node):
        return

    param_res = _get_param_res(item)
    node_label = f"Node '{item.name}'"

    # 1. DI binding checks (existing)
    for binding in (param_res or {}).values():
        _check_binding(node_label, binding, config, issues)

    # Check merge_fn DI bindings for Oracle nodes.
    oracle = item.modifier_set.oracle
    if oracle is not None and isinstance(oracle.merge_fn, str):
        meta = get_merge_fn_metadata(oracle.merge_fn)
        if meta is not None:
            _, merge_param_res = meta
            merge_label = f"{item.name} merge_fn '{oracle.merge_fn}'"
            for binding in merge_param_res.values():
                _check_binding(merge_label, binding, config, issues)

    # 2. Template placeholder checks
    _check_template_placeholders(item, issues, known_vars=known_vars,
                                 template_resolver=template_resolver,
                                 di_inputs_enabled=di_inputs_enabled)

    # 3. Loop condition checks
    _check_loop_condition(item, issues, conditions=conditions)

    # 4. Async-only (MCP) tool checks
    _check_async_only_tools(item, issues, tool_factories=tool_factories)

    # 5. ask_human reachable from an act-mode (mutating) node (A.5 safety)
    _check_ask_human_in_mutating_node(item, issues, tool_factories=tool_factories)


def _check_template_placeholders(
    node: Node,
    issues: list[LintIssue],
    *,
    known_vars: frozenset[str] | set[str],
    template_resolver: Callable[[str], str | None] | None = None,
    di_inputs_enabled: bool = False,
) -> None:
    """Check that prompt placeholders resolve to known input keys.

    Two modes:
    - Inline prompts (space or ${} in prompt): extract ${var} placeholders.
    - Template-ref prompts (bare name like "rw/summarize"): if template_resolver
      is provided, read the template text and extract {placeholder} names.

    The valid-key set for a template-ref prompt has THREE columns:
    predicted input keys (upstream outputs + flattened), consumer *known_vars*,
    and — when *di_inputs_enabled* (the compiler declares a ``di_inputs`` param)
    — the node's FromInput/FromConfig parameter names. Inline ``${var}`` prompts
    never see di_inputs (they resolve via raw attribute access, not the compiler
    seam), so the third column applies to template-ref prompts only.
    """
    prompt = node.prompt
    if not prompt or node.mode == "scripted":
        return

    is_inline = " " in prompt or "${" in prompt

    if is_inline:
        placeholders = _PLACEHOLDER_RE.findall(prompt)
    else:
        # Template-ref prompt — resolve text if resolver available
        if template_resolver is None:
            return
        text = template_resolver(prompt)
        if text is None:
            return
        placeholders = _extract_format_placeholders(text)

    if not placeholders:
        return

    node_label = f"Node '{node.name}'"
    placeholder_syntax = "${%s}" if is_inline else "{%s}"

    if is_inline:
        # Inline prompts only have access to raw input dict keys.
        # Flattened fields from render_for_prompt are NOT available (inline
        # skips _render_with_flattening). Known extras (node_id etc) are NOT
        # available (_resolve_var has no config/state access).
        predicted_keys = _predict_input_keys(node, include_flattened=False)
        valid_keys = predicted_keys | (known_vars - _KNOWN_EXTRAS)
    else:
        # Template-ref prompts get rendered data: flattened fields, known
        # extras, framework extras are all available to the prompt_compiler.
        # Third column: FromInput/FromConfig param names, but ONLY when the
        # compiler opted into di_inputs — otherwise the resolved DI value never
        # reaches the template and the placeholder is genuinely unresolvable.
        predicted_keys = _predict_input_keys(node)
        valid_keys = predicted_keys | known_vars
        if di_inputs_enabled:
            valid_keys = valid_keys | _di_template_var_names(node)

    consumer_known = known_vars - _KNOWN_EXTRAS - predicted_keys

    for placeholder in placeholders:
        first_segment = placeholder.split(".")[0]
        if first_segment not in valid_keys:
            issues.append(LintIssue(
                node_name=node_label,
                param=first_segment,
                kind="template_placeholder_unresolvable",
                required=True,
                message=(
                    f"{node_label}: prompt placeholder "
                    f"'{placeholder_syntax % first_segment}' "
                    f"not found in predicted input keys {sorted(predicted_keys)} "
                    f"or known extras {sorted(_KNOWN_EXTRAS)} "
                    f"(prompt: {prompt!r})"
                ),
            ))
        elif first_segment in consumer_known and first_segment not in predicted_keys and first_segment not in _KNOWN_EXTRAS:
            issues.append(LintIssue(
                node_name=node_label,
                param=first_segment,
                kind="template_placeholder_known_vars_only",
                required=False,
                message=(
                    f"{node_label}: placeholder "
                    f"'{placeholder_syntax % first_segment}' resolved only "
                    f"via known_vars — verify consumer bridge supplies it at runtime. "
                    f"Consider using the actual @node parameter name instead of a "
                    f"bridge alias."
                ),
            ))


def _check_loop_condition(
    item: Construct | Node,
    issues: list[LintIssue],
    *,
    conditions: dict[str, Callable] | None = None,
) -> None:
    """Check Loop modifier's when-condition for common issues.

    Three checks:
    1. String condition not in the `conditions=` kwarg (ERROR).
    2. Callable condition not None-safe — first iteration value is None (WARN).
    3. Registered string condition that resolves to a parse_condition result,
       which is inherently None-unsafe (ERROR).
    """
    conditions = conditions or {}

    ms = getattr(item, "modifier_set", None)
    if ms is None:
        return
    loop = ms.loop
    if loop is None:
        return

    item_label = (
        f"Construct '{item.name}'" if isinstance(item, Construct)
        else f"Node '{item.name}'"
    )
    condition = loop.when

    if isinstance(condition, str):
        # Check 1: is the string condition registered?
        resolved = conditions.get(condition)
        if resolved is None:
            issues.append(LintIssue(
                node_name=item_label,
                param="loop.when",
                kind="loop_condition_unregistered",
                required=True,
                message=(
                    f"Loop condition '{condition}' is not registered. "
                    f"Pass conditions={{'{condition}': fn}} to compile()."
                ),
            ))
            return  # can't test None-safety without the callable

        # Check 3: registered string condition — smoke-test with None.
        # parse_condition results always crash on None (getattr on None).
        # This is ERROR (required=True) because it WILL crash on first iteration.
        try:
            resolved(None)
        except (AttributeError, TypeError):
            issues.append(LintIssue(
                node_name=item_label,
                param="loop.when",
                kind="loop_condition_none_unsafe",
                required=True,
                message=(
                    f"Loop condition '{condition}' raises when called with None. "
                    f"The first iteration's value may be None. Use a None-safe "
                    f"wrapper: lambda d: d is None or {condition}(d)"
                ),
            ))
    elif callable(condition):
        # Check 2: callable condition — smoke-test with None.
        # WARN (required=False) because the callable might handle None
        # via other means we can't statically verify.
        try:
            condition(None)
        except (AttributeError, TypeError):
            issues.append(LintIssue(
                node_name=item_label,
                param="loop.when",
                kind="loop_condition_none_unsafe",
                required=False,
                message=(
                    "Loop condition raises when called with None. "
                    "The first iteration's value may be None — add a "
                    "None guard: lambda d: d is None or <condition>"
                ),
            ))


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
            issues.append(LintIssue(
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
            ))
            continue
        tool_obj = _resolve_tool_object(spec, tool_factories)
        if tool_obj is None:
            continue
        tool_name = str(getattr(spec, "name", None) or getattr(tool_obj, "name", "?"))
        if is_async_only_tool(tool_obj):
            issues.append(LintIssue(
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
            ))


# Attribute names under which a tool object may carry its callable body. Covers
# every shape ask_human can hide in: StructuredTool (@tool) uses .func/.coroutine,
# a BaseTool subclass uses ._run/._arun, and a duck-typed class tool (the keystone
# _AskTool shape) puts logic in .invoke/.ainvoke. Introspecting all of them keeps
# the rule from silently no-opping on a shape it doesn't recognize.
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
        code = getattr(fn, "__code__", None) or getattr(
            getattr(fn, "__func__", None), "__code__", None
        )
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
            issues.append(LintIssue(
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
            ))


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


def _extract_format_placeholders(text: str) -> list[str]:
    """Extract {placeholder} names from Python str.format-style template text.

    Returns a list of field names (may include dotted paths like 'claim.text').
    Skips empty/None field names (literal braces, positional args).
    """
    formatter = string.Formatter()
    names = []
    for _, field_name, _, _ in formatter.parse(text):
        if field_name is not None and field_name != "":
            names.append(field_name)
    return names


def _di_template_var_names(node: Node) -> set[str]:
    """The node's FromInput/FromConfig parameter names usable as template vars.

    These become valid template-ref placeholders when the prompt_compiler
    accepts ``di_inputs`` (the dispatch layer resolves them and the compiler
    binds them by parameter name). Bundled-model kinds contribute the bundle's
    parameter name (matching the first segment of a dotted ``{ctx.field}``).
    """
    param_res = _get_param_res(node)
    return {
        name for name, binding in (param_res or {}).items()
        if binding.kind in DI_TEMPLATE_KINDS
    }


def _predict_input_keys(node: Node, *, include_flattened: bool = True) -> set[str]:
    """Predict the dict keys that _extract_input will produce for this node.

    For dict-form inputs: keys are the dict keys. When *include_flattened* is
    True (default), also adds flattened field names from ``render_for_prompt()``
    return annotations — these are available for template-ref prompts where
    ``_render_with_flattening`` runs. Set *include_flattened=False* for inline
    prompts, which skip flattening and only see raw input dict keys.

    For single-type or None inputs: empty set (isinstance scan, no dict).
    """
    ni = normalize_inputs(node.inputs)
    if ni.is_none:
        return set()
    if ni.is_dict_form:
        keys = set(ni.by_name.keys())
        if include_flattened:
            for input_type in ni.by_name.values():
                keys |= _get_flattened_field_names(input_type)
        return keys
    # Single-type inputs: no dict keys predictable
    return set()


def _get_flattened_field_names(input_type: Any) -> set[str]:
    """Extract field names from a type's render_for_prompt() return annotation.

    If the type has ``render_for_prompt`` with a return annotation that is a
    BaseModel subclass, returns the non-excluded field names of that model.
    Otherwise returns an empty set.
    """

    from pydantic import BaseModel as _BM

    rfp = getattr(input_type, "render_for_prompt", None)
    if rfp is None:
        return set()

    ret_type = _resolve_return_type(rfp, input_type)
    if ret_type is None:
        return set()
    if not (isinstance(ret_type, type) and issubclass(ret_type, _BM)):
        return set()
    return {
        fname for fname, finfo in ret_type.model_fields.items()
        if not finfo.exclude
    }


def _resolve_return_type(fn: Any, owner_cls: Any) -> Any:
    """Resolve the return type annotation of a method.

    ``from __future__ import annotations`` turns annotations into strings.
    ``typing.get_type_hints`` resolves them from ``fn.__globals__`` but fails
    when the return type is defined in a local scope (e.g., inside a test).

    Fallback: scan the caller's frame stack (up to 10 frames) for the name.
    This mirrors the technique Pydantic and neograph's ``_di_classify.py``
    use for forward-ref resolution.
    """
    import sys
    import types
    import typing

    # Fast path: get_type_hints works for module-scoped types
    try:
        hints = typing.get_type_hints(fn)
        return hints.get("return")
    except (NameError, AttributeError, TypeError):
        pass

    # Fallback: resolve string annotation from frame locals
    raw = getattr(fn, "__annotations__", {}).get("return")
    if raw is None or not isinstance(raw, str):
        return raw

    # Walk caller frames to find the name (handles test-local classes)
    frame: types.FrameType | None = sys._getframe(0)
    for _ in range(10):
        frame = frame.f_back if frame is not None else None
        if frame is None:
            break
        if raw in frame.f_locals:
            return frame.f_locals[raw]
    return None
