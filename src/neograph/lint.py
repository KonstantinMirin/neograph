"""lint() — validate DI bindings and template placeholders against config/inputs.

Walks all nodes in a Construct and checks:
1. Every FromInput/FromConfig parameter has a matching key in the config dict.
2. Every ${var} placeholder in inline prompts resolves to a known input key.

Returns a list of LintIssue dataclass instances (never raises — reports all problems).
"""

from __future__ import annotations

# --- names lint.py imported and RE-EXPORTED before the split; the moved
# --- clusters were their only local consumers here.
import asyncio  # noqa: E402,F401
import string  # noqa: E402,F401
from collections.abc import Callable
from dataclasses import dataclass  # noqa: E402,F401
from typing import Any

import structlog

from neograph._ir_branch import iter_with_arms
from neograph._ir_protocols import ConstructItem
from neograph._lint_consumers import _check_unconsumed_outputs
from neograph._lint_di import _check_binding, _check_unmatched_config_keys, iter_di_bindings

# --- extracted clusters (neograph-3ffdg.10), re-exported so existing
# --- `from neograph.lint import ...` call sites keep resolving unchanged.
from neograph._lint_kind_registry import (  # noqa: E402,F401
    LINT_KIND_META,
    LintIssue,
    LintKindMeta,
)
from neograph._lint_predict import (  # noqa: E402,F401
    _di_resource_template_var_names,
    _di_template_var_names,
    _extract_format_placeholders,
    _get_flattened_field_names,
    _predict_input_keys,
    _resolve_return_type,
)
from neograph._lint_supply import (
    _check_unreferenced_inputs,
    _check_unsatisfiable_di,
    _placeholder_root,
    _port_supplied_by_modifier,
)
from neograph._lint_tool_checks import (  # noqa: E402,F401
    _TOOL_BODY_ATTRS,
    _check_act_mode_all_idempotent,
    _check_ask_human_in_mutating_node,
    _check_async_only_tools,
    _resolve_tool_object,
    _spec_factory,
    _tool_references_ask_human,
)
from neograph._llm_runtime import (
    _ACCEPT_ALL,
    _accepted_params,
    collect_llm_nodes,
    missing_runtime_kwargs,
)
from neograph._normalize import normalize_inputs, normalize_outputs  # noqa: E402,F401
from neograph._placeholders import DOLLAR_RE
from neograph._runtime_registry import _decoration_registry
from neograph._sidecar import _get_param_res
from neograph._state_keys import StateKeys
from neograph.construct import Construct
from neograph.di import (
    DI_TEMPLATE_KINDS,  # noqa: E402,F401
    DIKind,
)
from neograph.node import Node
from neograph.tool import Tool, is_async_only_tool  # noqa: E402,F401

log = structlog.get_logger()

# Standard keys always available in state / config
_KNOWN_EXTRAS: frozenset[str] = frozenset(
    {
        StateKeys.NODE_ID,
        StateKeys.PROJECT_ROOT,
        StateKeys.HUMAN_FEEDBACK,
    }
)

# The ${var} scanner is the ONE shared in _placeholders — imported, not redefined
# (byte-identical dedup: lint collects names, prompt.substitute fills them, both
# off one grammar). Aliased to preserve the existing local name.
_PLACEHOLDER_RE = DOLLAR_RE


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

    # Construct-level: consumability is a property of the whole binding set.
    _check_unmatched_config_keys(construct, config, issues)

    all_known = _KNOWN_EXTRAS | (known_template_vars or set())
    _walk(
        construct,
        config,
        issues,
        known_vars=all_known,
        template_resolver=template_resolver,
        conditions=conditions,
        tool_factories=tool_factory_lookup,
        di_inputs_enabled=_compiler_accepts_di_inputs(prompt_compiler),
        context_enabled=_compiler_accepts_context(prompt_compiler),
        resource_producer_present=_has_resource_link_producer(construct),
    )
    # Construct-level: a field is dead only when every consumer axis misses it.
    _check_unconsumed_outputs(construct, issues, template_resolver=template_resolver)
    return issues


def _has_resource_link_producer(construct: Construct) -> bool:
    """True when the construct contains any agent/act node bound to tools.

    Only an agent/act node running tools can emit an MCP ``resource_link`` that the
    manifest lift captures. This is the static approximation the
    ``resource_hydration_kind_unmatched`` check uses: a manifest-driven
    ``FromResource(ref=...)`` needs SOME upstream resource_link producer, else the
    manifest is empty and the ref hydration fails far from its cause at runtime.
    Per-kind matching is not statically knowable (kinds are lifted at runtime), so
    producer-existence is the honest static gate. See neograph-a5nh Risk 3."""
    stack: list[Any] = [construct]
    while stack:
        item = stack.pop()
        if isinstance(item, Construct):
            stack.extend(iter_with_arms(item))
        elif isinstance(item, Node) and item.mode in ("agent", "act") and item.tools:
            return True
    return False


def _check_resource_hydration(
    node: Node,
    param_res: Any,
    issues: list[LintIssue],
    resource_producer_present: bool,
) -> None:
    """Flag a manifest-driven FromResource(ref=<kind>) with no possible producer.

    ERROR (``required=True``): a node hydrates a manifest ``kind`` but no upstream
    agent/act node can emit a ``resource_link`` at all, so the manifest is
    guaranteed empty and the ref hydration would fail loud at runtime far from the
    cause. The documented fallback for a flat server (one that emits no
    ``resource_link``) is a templated ``FromResource(uri=...)`` / ``resource_reader``
    tool — the two mechanisms cover each other's gaps (Risk 3)."""
    if resource_producer_present or not param_res:
        return
    for binding in param_res.values():
        if binding.kind is DIKind.FROM_RESOURCE and binding.ref_kind is not None:
            issues.append(
                LintIssue(
                    node_name=f"Node '{node.name}'",
                    param=binding.name,
                    kind="resource_hydration_kind_unmatched",
                    required=True,
                    message=(
                        f"Node '{node.name}': parameter '{binding.name}' hydrates "
                        f"manifest kind='{binding.ref_kind}' but no upstream agent/act "
                        "node produces resource_links. Add a resource_link-emitting "
                        "agent/act producer upstream, or use a templated "
                        "FromResource(uri=...) / resource_reader tool (the flat-server "
                        "fallback)."
                    ),
                )
            )


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


def _compiler_accepts_context(prompt_compiler: Any) -> bool:
    """True when *prompt_compiler* declares a ``context`` param or ``**kwargs``.

    The FOURTH column, and the exact twin of :func:`_compiler_accepts_di_inputs`:
    a node's declared ``context=`` field name is a valid template-ref placeholder
    only when the compiler actually receives the channel. Until neograph-cbfd9 the
    shipped compiler swallowed it, so lint reporting the placeholder unresolvable
    was CORRECT; fixing the runtime turned that true positive into a false one.
    """
    if prompt_compiler is None:
        return False
    params = _accepted_params(prompt_compiler)
    return params is _ACCEPT_ALL or "context" in params


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

    issues.append(
        LintIssue(
            node_name=", ".join(llm_nodes),
            param="",
            kind="llm_kwargs_missing",
            message=(
                f"LLM-mode nodes ({', '.join(llm_nodes)}) require "
                f"{' and '.join(missing)} at compile() time. "
                "Pass these kwargs to compile() or configure them via "
                "configure_llm() (legacy)."
            ),
        )
    )


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
    context_enabled: bool = False,
    resource_producer_present: bool = False,
    port_supplied: tuple[type, str] | None = None,
) -> None:
    """Recursively walk a construct and check DI bindings + template placeholders."""
    if isinstance(item, Construct):
        # Check Loop condition on the Construct itself (Construct | Loop)
        _check_loop_condition(item, issues, conditions=conditions)
        # iter_with_arms expands _BranchNode sentinels so a bare arm Node's DI
        # bindings + template placeholders are linted like any other node. See
        # neograph-vn5f (site 3).
        child_port = _port_supplied_by_modifier(item)
        for child in iter_with_arms(item):
            _walk(
                child,
                config,
                issues,
                known_vars=known_vars,
                template_resolver=template_resolver,
                conditions=conditions,
                tool_factories=tool_factories,
                di_inputs_enabled=di_inputs_enabled,
                context_enabled=context_enabled,
                resource_producer_present=resource_producer_present,
                port_supplied=child_port,
            )
        return

    if not isinstance(item, Node):
        return

    _check_unsatisfiable_di(item, issues, port_supplied=port_supplied)

    param_res = _get_param_res(item)
    node_label = f"Node '{item.name}'"

    # 1. DI binding checks. iter_di_bindings is the SINGLE enumeration of where
    # DI bindings live; input_contract() reads the same one, so a new binding
    # site reaches both surfaces or neither. It covers the node's own params AND
    # its Oracle merge_fn's, which is why no separate merge_fn loop follows.
    for binding_label, binding in iter_di_bindings(item):
        _check_binding(binding_label, binding, config, issues)

    # 1b. Manifest-driven hydration: a FromResource(ref=<kind>) binding needs an
    # upstream resource_link producer somewhere in the construct neograph-a5nh.
    _check_resource_hydration(item, param_res, issues, resource_producer_present)

    # 2. Template placeholder checks
    _check_template_placeholders(
        item,
        issues,
        known_vars=known_vars,
        template_resolver=template_resolver,
        di_inputs_enabled=di_inputs_enabled,
        context_enabled=context_enabled,
    )

    # 3. Loop condition checks
    _check_loop_condition(item, issues, conditions=conditions)

    # 4. Async-only (MCP) tool checks
    _check_async_only_tools(item, issues, tool_factories=tool_factories)

    # 5. ask_human reachable from an act-mode (mutating) node (A.5 safety)
    _check_ask_human_in_mutating_node(item, issues, tool_factories=tool_factories)

    # 6. act-mode node whose tools are ALL idempotent (probably mode='agent')
    _check_act_mode_all_idempotent(item, issues)


def _check_template_placeholders(
    node: Node,
    issues: list[LintIssue],
    *,
    known_vars: frozenset[str] | set[str],
    template_resolver: Callable[[str], str | None] | None = None,
    di_inputs_enabled: bool = False,
    context_enabled: bool = False,
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

    node_label = f"Node '{node.name}'"
    placeholder_syntax = "${%s}" if is_inline else "{%s}"

    # FROM_RESOURCE param names that are valid template-ref vars but resolve ONLY
    # under the async arun() driver (their fetch is awaited). Populated for the
    # template-ref + di_inputs_enabled case below; used to emit the async-driver
    # WARN in lockstep with runtime. See neograph-3q6j.
    resource_vars: set[str] = set()

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
        # Fourth column: the node's declared context= fields, on the same opt-in
        # terms as di_inputs -- the seam offers the channel, the compiler must
        # declare it. See neograph-ait72.
        if context_enabled:
            valid_keys = valid_keys | set(getattr(node, "context", None) or ())
        if di_inputs_enabled:
            valid_keys = valid_keys | _di_template_var_names(node)
            resource_vars = _di_resource_template_var_names(node)
            valid_keys = valid_keys | resource_vars

    consumer_known = known_vars - _KNOWN_EXTRAS - predicted_keys

    # One reduced set read by both directions, so the demand check and the
    # supply check cannot disagree about what a template names. The demand loop
    # still iterates the placeholder LIST: it emits one issue per OCCURRENCE.
    referenced = {_placeholder_root(p) for p in placeholders}

    _check_unreferenced_inputs(
        node,
        issues,
        referenced=referenced,
        is_inline=is_inline,
        consumer_known=consumer_known,
        node_label=node_label,
    )

    if not placeholders:
        return

    for placeholder in placeholders:
        first_segment = placeholder.split(".")[0]
        if first_segment not in valid_keys:
            issues.append(
                LintIssue(
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
                )
            )
        elif first_segment in resource_vars:
            # A FROM_RESOURCE template var IS valid, but its fetch is awaited — it
            # resolves only under arun() (the sync run() driver fails loud at the
            # injector). WARN so lint stays in lockstep with runtime coverage.
            issues.append(
                LintIssue(
                    node_name=node_label,
                    param=first_segment,
                    kind="template_var_requires_async_driver",
                    required=False,
                    message=(
                        f"{node_label}: template var "
                        f"'{placeholder_syntax % first_segment}' is a FromResource DI "
                        f"param — its fetch is awaited, so it resolves ONLY under the "
                        f"async arun() driver (the sync run() driver fails loud). Drive "
                        f"this graph with arun()."
                    ),
                )
            )
        elif (
            first_segment in consumer_known
            and first_segment not in predicted_keys
            and first_segment not in _KNOWN_EXTRAS
        ):
            issues.append(
                LintIssue(
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
                )
            )


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

    item_label = f"Construct '{item.name}'" if isinstance(item, Construct) else f"Node '{item.name}'"
    condition = loop.when

    if isinstance(condition, str):
        # Check 1: is the string condition registered?
        resolved = conditions.get(condition)
        if resolved is None:
            issues.append(
                LintIssue(
                    node_name=item_label,
                    param="loop.when",
                    kind="loop_condition_unregistered",
                    required=True,
                    message=(
                        f"Loop condition '{condition}' is not registered. "
                        f"Pass conditions={{'{condition}': fn}} to compile()."
                    ),
                )
            )
            return  # can't test None-safety without the callable

        # Check 3: registered string condition — smoke-test with None.
        # parse_condition results always crash on None (getattr on None).
        # This is ERROR (required=True) because it WILL crash on first iteration.
        try:
            resolved(None)
        except (AttributeError, TypeError):
            issues.append(
                LintIssue(
                    node_name=item_label,
                    param="loop.when",
                    kind="loop_condition_none_unsafe",
                    required=True,
                    message=(
                        f"Loop condition '{condition}' raises when called with None. "
                        f"The first iteration's value may be None. Use a None-safe "
                        f"wrapper: lambda d: d is None or {condition}(d)"
                    ),
                )
            )
    elif callable(condition):
        # Check 2: callable condition — smoke-test with None.
        # WARN (required=False) because the callable might handle None
        # via other means we can't statically verify.
        try:
            condition(None)
        except (AttributeError, TypeError):
            issues.append(
                LintIssue(
                    node_name=item_label,
                    param="loop.when",
                    kind="loop_condition_none_unsafe",
                    required=False,
                    message=(
                        "Loop condition raises when called with None. "
                        "The first iteration's value may be None — add a "
                        "None guard: lambda d: d is None or <condition>"
                    ),
                )
            )


# Attribute names under which a tool object may carry its callable body. Covers
# every shape ask_human can hide in: StructuredTool (@tool) uses .func/.coroutine,
# a BaseTool subclass uses ._run/._arun, and a duck-typed class tool (the keystone
# _AskTool shape) puts logic in .invoke/.ainvoke. Introspecting all of them keeps
# the rule from silently no-opping on a shape it doesn't recognize.
