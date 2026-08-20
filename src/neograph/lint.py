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
from pydantic import BaseModel

from neograph._ir_branch import iter_with_arms
from neograph._ir_protocols import ConstructItem
from neograph._llm_runtime import (
    _ACCEPT_ALL,
    _accepted_params,
    collect_llm_nodes,
    missing_runtime_kwargs,
)
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph._placeholders import DOLLAR_RE
from neograph._runtime_registry import _decoration_registry
from neograph._sidecar import _get_param_res, get_merge_fn_metadata
from neograph._state_keys import StateKeys
from neograph.construct import Construct
from neograph.di import DI_TEMPLATE_KINDS, DIBinding, DIKind
from neograph.modifiers import classify_modifiers
from neograph.naming import field_name_for, output_field_name
from neograph.node import Node
from neograph.tool import Tool, is_async_only_tool

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


@dataclass
class LintIssue:
    """A single lint problem — DI binding or template placeholder."""

    node_name: str
    param: str
    kind: str
    message: str
    required: bool = False


@dataclass(frozen=True)
class LintKindMeta:
    """Severity + one-line human meaning for a single lint kind."""

    severity: str  # one of: ERROR | WARN | WARN/ERROR | varies
    meaning: str


# Authoritative metadata for every kind lint() can emit. Task neograph-uw54v.
#
# SINGLE SOURCE OF TRUTH for lint-kind severity + meaning. scripts/
# gen_api_manifest.py reads this to build the manifest's ``lint_issue_kinds`` as
# ``{kind, severity, meaning}`` objects; the website reference lint table
# renders from that manifest (Stage C, neograph-cvjfm) instead of a
# hand-authored, drift-prone copy.
#
# Severity discipline (refinement neograph-uqy66.52): for a kind emitted at a
# SINGLE LintIssue(...) site with a literal ``required=``, the severity below
# MUST equal ``'ERROR' if required else 'WARN'`` — the canonical rule at
# __main__.py:199. The manifest generator RE-DERIVES that from the ``ast.Call``
# at the emission site and FAILS LOUD if this registry drifts, so a future
# ``required=`` flip cannot silently diverge. Two sanctioned exceptions that
# have no single derived value:
#   - ``WARN/ERROR``: ``loop_condition_none_unsafe`` is emitted at two sites with
#     conflicting ``required=`` (ERROR for registered string conditions that
#     always crash on None, WARN for user callables that may guard None).
#   - ``varies``: the 4 DI kinds are emitted with ``kind=binding.kind.value`` (a
#     variable), and severity is the per-binding runtime ``required``.
LINT_KIND_META: dict[str, LintKindMeta] = {
    # DI bindings — kind=variable, severity is runtime binding.required-dependent.
    "from_input": LintKindMeta(
        "varies",
        "`Annotated[T, FromInput]` -- resolved from `config['configurable']`, "
        "originally from `run(input={...})`.",
    ),
    "from_config": LintKindMeta(
        "varies",
        "`Annotated[T, FromConfig]` -- resolved from `config['configurable']`, "
        "passed directly in `config=`.",
    ),
    "from_input_model": LintKindMeta(
        "varies",
        "Bundled `BaseModel` via `FromInput` -- each model field must exist in "
        "config.",
    ),
    "from_config_model": LintKindMeta(
        "varies",
        "Bundled `BaseModel` via `FromConfig` -- each model field must exist in "
        "config.",
    ),
    # Template placeholders.
    "output_field_unconsumed": LintKindMeta(
        "WARN",
        "A field of a node's typed output that nothing reads: no downstream node "
        "takes the model, no template reads it by name, and it is not the graph's "
        "terminal output. It costs tokens on every call and cannot affect the "
        "answer.",
    ),
    "from_input_unsatisfiable": LintKindMeta(
        "ERROR",
        "A `FromInput`/`FromConfig` parameter whose value is the Each-fanned item "
        "or the Loop carry. No caller can supply it, so the run fails in the DI "
        "preflight -- and padding a config to silence it makes every branch "
        "compute from the padded value. Bind it as a port parameter instead.",
    ),
    "template_input_unreferenced": LintKindMeta(
        "WARN",
        "A bound input, DI parameter, or context field that the node's own "
        "template never references. The value reaches the node and the model "
        "never sees it. Demand is read from the template text, so a "
        "`prompt_compiler` that composes the message may consume the name "
        "without naming it.",
    ),
    "template_placeholder_unresolvable": LintKindMeta(
        "ERROR",
        "Prompt placeholder not found in predicted input keys or known extras.",
    ),
    "template_placeholder_known_vars_only": LintKindMeta(
        "WARN",
        "Placeholder only resolvable via `known_template_vars`, not from actual "
        "`@node` parameter names. Advisory: verify the consumer bridge supplies "
        "it at runtime.",
    ),
    "template_var_requires_async_driver": LintKindMeta(
        "WARN",
        "A template var is a `FromResource` DI param whose fetch is awaited, so "
        "it resolves only under the async `arun()` driver (sync `run()` fails "
        "loud). Drive the graph with `arun()`.",
    ),
    # Loop conditions.
    "loop_condition_unregistered": LintKindMeta(
        "ERROR",
        "Loop `when` is a string that is not registered in the condition "
        "registry.",
    ),
    "loop_condition_none_unsafe": LintKindMeta(
        "WARN/ERROR",
        "Loop `when` callable raises when called with `None`. ERROR for "
        "registered string conditions (always crash), WARN for user-supplied "
        "callables (may handle None via other means).",
    ),
    # Tools.
    "tool_requires_async_driver": LintKindMeta(
        "WARN",
        "An `agent`/`act` node is bound to an async-only tool (e.g. an MCP tool) "
        "that cannot run under the sync `run()` driver. Drive the graph with "
        "`arun()`.",
    ),
    "ask_human_in_mutating_node": LintKindMeta(
        "WARN",
        "An `act`-mode (mutating) tool calls `ask_human()`; a non-idempotent "
        "side effect before the mid-loop pause can double-fire on resume. Make "
        "any pre-pause mutation idempotent, or move it after the pause.",
    ),
    "act_mode_all_idempotent_tools": LintKindMeta(
        "WARN",
        "`mode='act'` (mutations) but all tools are `idempotent=True` "
        "(read-only) -- probably a misclassification; use `mode='agent'` unless "
        "a tool is a genuinely idempotent mutation.",
    ),
    # Runtime configuration.
    "resource_hydration_kind_unmatched": LintKindMeta(
        "ERROR",
        "A node hydrates a manifest kind via `FromResource(ref=...)` but the "
        "construct has no upstream `agent`/`act` node that can emit a "
        "`resource_link`, so the manifest is guaranteed empty at runtime.",
    ),
    "llm_kwargs_missing": LintKindMeta(
        "WARN",
        "LLM-mode nodes require `llm_factory` and `prompt_compiler` at "
        "`compile()` time. Pass these kwargs to `compile()` (or configure via "
        "`configure_llm()`, legacy).",
    ),
}


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
                issues.append(
                    LintIssue(
                        node_name=node_label,
                        param=binding.name,
                        kind=kind_str,
                        required=binding.required,
                        message=(f"{node_label}: DI parameter '{binding.name}' ({kind_str}) not found in config"),
                    )
                )
        elif binding.required:
            # No config supplied. This parameter is part of the graph's INPUT
            # CONTRACT: a caller supplies it at run time, so reporting it as an
            # error says only that the graph has inputs. Requiring a config to
            # reach a clean gate is what pushed one consumer to pad the fixture
            # with a key no caller could pass, which silenced a real
            # unsatisfiable binding (GH #12, GH #13).
            issues.append(
                LintIssue(
                    node_name=node_label,
                    param=binding.name,
                    kind=kind_str,
                    required=False,
                    message=(
                        f"{node_label}: DI parameter '{binding.name}' ({kind_str}) "
                        f"is part of this graph's input contract -- a caller supplies "
                        f"it at run time. Pass config= to check a specific payload."
                    ),
                )
            )

    elif binding.kind in (DIKind.FROM_INPUT_MODEL, DIKind.FROM_CONFIG_MODEL):
        model_cls: Any = binding.model_cls or binding.inner_type
        required = binding.required
        if config is not None:
            for fname in model_cls.model_fields:
                if fname not in config:
                    issues.append(
                        LintIssue(
                            node_name=node_label,
                            param=fname,
                            kind=kind_str,
                            required=required,
                            message=(
                                f"{node_label}: bundled model field "
                                f"'{fname}' ({kind_str} via {model_cls.__name__}) "
                                f"not found in config"
                            ),
                        )
                    )
        elif required:
            # Same input-contract reasoning as the scalar branch above.
            for fname in model_cls.model_fields:
                issues.append(
                    LintIssue(
                        node_name=node_label,
                        param=fname,
                        kind=kind_str,
                        required=False,
                        message=(
                            f"{node_label}: bundled model field '{fname}' "
                            f"({kind_str} via {model_cls.__name__}) is part of "
                            f"this graph's input contract -- a caller supplies it "
                            f"at run time. Pass config= to check a payload."
                        ),
                    )
                )


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
    # Construct-level: a field is dead only when every consumer axis misses it,
    # so this decision needs the whole graph rather than one node (GH #11).
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


def _port_supplied_by_modifier(construct: Construct) -> tuple[type, str] | None:
    """The sub-construct's port type, when a modifier supplies its value.

    An ``Each`` fans one item into the port per branch and a ``Loop`` carries the
    previous result back into it. In both cases the value comes from the
    construct, never from the caller, so a ``FromInput`` on a parameter of that
    type can never be satisfied (GH #12).
    """
    port = getattr(construct, "input", None)
    if not isinstance(port, type):
        return None
    _, mods = classify_modifiers(construct)
    if "each" in mods:
        return (port, "Each-fanned item")
    if "loop" in mods:
        return (port, "Loop carry")
    return None


def _check_unsatisfiable_di(
    node: Node,
    issues: list[LintIssue],
    *,
    port_supplied: tuple[type, str] | None,
) -> None:
    """Report a DI binding that the enclosing modifier makes unsatisfiable.

    Decided from the construct's structure alone. It takes no config, so no
    fixture can silence it -- which is the point: the reported bug survived
    because the lint config had been padded with a key no caller could pass, and
    the gate then graded its own answer key (GH #12).
    """
    supplied: list[tuple[Any, str]] = []
    if port_supplied is not None:
        supplied.append(port_supplied)

    # A node-level self-loop carries the node's OWN output back as its input, so
    # that type is supplied by the construct too. `Construct.input` must be a
    # BaseModel, so a bare `list[X]` carry can only arise this way.
    _, node_mods = classify_modifiers(node)
    if "loop" in node_mods:
        carry = normalize_outputs(node.outputs).primary
        if carry is not None:
            supplied.append((carry, "Loop carry"))

    if not supplied:
        return
    bindings = getattr(node, "_param_res", None) or {}

    for param, binding in bindings.items():
        if binding.kind not in DI_TEMPLATE_KINDS:
            continue
        inner = getattr(binding, "model_cls", None) or getattr(binding, "inner_type", None)
        match = next(
            (
                (t, src)
                for t, src in supplied
                if inner is t or _unwrap_sequence(inner) is t
            ),
            None,
        )
        if match is None:
            continue
        port_type, source = match
        issues.append(
            LintIssue(
                node_name=node.name,
                param=param,
                kind="from_input_unsatisfiable",
                required=True,
                message=(
                    f"Node '{node.name}': parameter '{param}' is bound with "
                    f"{binding.kind.value} but its value is the {source}, which no "
                    f"caller can supply. The run fails in the DI preflight, and "
                    f"padding a config to silence this makes every branch compute "
                    f"from the padded value instead. Bind it as a PORT parameter: "
                    f"drop the DI marker and type it as the sub-construct's "
                    f"input={getattr(port_type, '__name__', port_type)}."
                ),
            )
        )


def _unwrap_sequence(annotation: Any) -> Any:
    """``list[X]`` -> ``X``; anything else unchanged. A Loop carry may be either."""
    from typing import get_args, get_origin

    if get_origin(annotation) in (list, tuple, set, frozenset):
        args = get_args(annotation)
        if args:
            return args[0]
    return annotation


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
    """Recursively walk a construct and check DI bindings + template placeholders.

    ``port_supplied`` carries the enclosing sub-construct's ``input=`` type and
    the modifier that feeds it, when that modifier supplies the value from the
    construct rather than from the caller. A ``FromInput`` on a parameter of that
    type is unsatisfiable by any caller (GH #12).
    """
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

    # 1. DI binding checks (existing)
    for binding in (param_res or {}).values():
        _check_binding(node_label, binding, config, issues)

    # 1b. Manifest-driven hydration: a FromResource(ref=<kind>) binding needs an
    # upstream resource_link producer somewhere in the construct neograph-a5nh.
    _check_resource_hydration(item, param_res, issues, resource_producer_present)

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


def _template_roots(node: Node, template_resolver: Any) -> set[str]:
    """Every ROOT name a node's template references, dotted or bare."""
    prompt = node.prompt
    if not prompt or node.mode == "scripted":
        return set()
    if " " in prompt or "${" in prompt:
        placeholders = _PLACEHOLDER_RE.findall(prompt)
    else:
        if template_resolver is None:
            return set()
        text = template_resolver(prompt)
        if text is None:
            return set()
        placeholders = _extract_format_placeholders(text)
    return {p.split(".")[0] for p in placeholders}


def _dotted_field_reads(node: Node, template_resolver: Any) -> set[tuple[str, str]]:
    """``(root, field)`` pairs a node's template reads, e.g. ``${triage.severity}``.

    The only FIELD-granular consumer axis. The other two -- a downstream node
    taking the whole model, and the terminal projection -- consume at model
    granularity.
    """
    prompt = node.prompt
    if not prompt or node.mode == "scripted":
        return set()

    if " " in prompt or "${" in prompt:
        placeholders = _PLACEHOLDER_RE.findall(prompt)
    else:
        if template_resolver is None:
            return set()
        text = template_resolver(prompt)
        if text is None:
            return set()
        placeholders = _extract_format_placeholders(text)

    reads: set[tuple[str, str]] = set()
    for placeholder in placeholders:
        parts = placeholder.split(".")
        if len(parts) >= 2:
            reads.add((parts[0], parts[1]))
    return reads


def _check_unconsumed_outputs(
    construct: Construct,
    issues: list[LintIssue],
    *,
    template_resolver: Callable[[str], str | None] | None = None,
) -> None:
    """Report an output field that nothing in the graph reads (GH #11).

    Construct-level, not per-node: a field is dead only when EVERY consumer axis
    misses it, so the decision needs the whole graph. Three axes:

    1. A downstream node input. A consumer declaring ``triage: Triage`` receives
       the whole model, and which fields its body reads is not derivable, so
       whole-model consumption marks every field consumed. This over-approximates
       deliberately: the opposite flags every scripted consumer in every pipeline.
    2. A dotted template placeholder. ``${triage.severity}`` reads one field, and
       this is the axis that gives the check its resolution.
    3. The terminal projection. The last node's output is the graph's output.

    Deriving fewer axes reports false cleanliness, which is worse than reporting
    nothing, because a guard that cannot fire is evidence of nothing.
    """
    nodes = [n for n in iter_with_arms(construct) if isinstance(n, Node)]
    if not nodes:
        return
    terminal = nodes[-1].name

    consumed_whole: set[str] = set()
    consumed_fields: set[tuple[str, str]] = set()

    for node in nodes:
        inputs = normalize_inputs(node.inputs)
        declared_names: set[str] = set()
        if inputs.is_dict_form:
            declared_names = set(inputs.by_name)
        elif not inputs.is_none and isinstance(inputs.single_type, type):
            declared_names = {inputs.single_type.__name__}

        if node.mode == "scripted" or not node.prompt:
            # A scripted body can read any field, and which ones is not
            # derivable, so taking the model consumes all of it. The opposite
            # would flag every scripted consumer in every pipeline.
            consumed_whole |= declared_names
            continue

        # An LLM-mode body never runs, so the TEMPLATE is the only reader. A
        # bare `${triage}` consumes the whole model; `${triage.severity}`
        # consumes one field. This is the same reasoning that scopes the
        # unreferenced-input check to LLM-mode nodes.
        reads = _dotted_field_reads(node, template_resolver)
        consumed_fields |= reads
        dotted_roots = {root for root, _ in reads}
        bare = _template_roots(node, template_resolver) - dotted_roots
        consumed_whole |= declared_names & bare
        if node.skip_when is not None:
            # A skip predicate receives the extracted input dict, so it can read
            # any field of it. Same opacity as a scripted body.
            consumed_whole |= declared_names

    for node in nodes:
        if node.name == terminal:
            continue
        outputs = normalize_outputs(node.outputs)
        for key, declared in (outputs.all_keys or {}).items() or (
            {node.name: outputs.primary}.items() if outputs.primary is not None else ()
        ):
            if not (isinstance(declared, type) and issubclass(declared, BaseModel)):
                continue
            # `@node` kebab-cases the function name, while a consumer's PARAM
            # keeps the underscore form. field_name_for owns that contract, so
            # both sides are compared in the same form.
            base = field_name_for(node.name)
            root = base if key == node.name else output_field_name(base, key)
            if root in consumed_whole or base in consumed_whole:
                continue
            for fname in declared.model_fields:
                if (root, fname) in consumed_fields or (base, fname) in consumed_fields:
                    continue
                issues.append(
                    LintIssue(
                        node_name=node.name,
                        param=fname,
                        kind="output_field_unconsumed",
                        required=False,
                        message=(
                            f"Node '{node.name}': output field '{fname}' of "
                            f"{declared.__name__} has no consumer. No downstream "
                            f"node takes the model, no template reads "
                            f"${{{root}.{fname}}}, and this is not the graph's "
                            f"terminal output. The field costs tokens on every "
                            f"call and cannot affect the answer."
                        ),
                    )
                )


def _placeholder_root(placeholder: str) -> str:
    """Root name a placeholder references. ``claims.items`` references ``claims``."""
    body = placeholder.split(":", 1)[1] if placeholder.startswith("image:") else placeholder
    return body.split(".")[0]


def _supply_axes(node: Node, *, is_inline: bool) -> list[tuple[str, str, set[str]]]:
    """Every name supplied to *node*, as ``(axis, name, aliases)`` triples.

    Reads the accessors the compiler already exposes. It derives nothing the IR
    does not hold, and it asks the pipeline author for no annotation.
    """
    axes: list[tuple[str, str, set[str]]] = []

    inputs = normalize_inputs(node.inputs)
    if inputs.is_dict_form:
        for key, declared in inputs.by_name.items():
            aliases = {key}
            if not is_inline:
                aliases |= _get_flattened_field_names(declared)
                if key == StateKeys.SUBGRAPH_INPUT and isinstance(declared, type):
                    aliases.add(declared.__name__)
            axes.append(("upstream input", key, aliases))
    elif not inputs.is_none and isinstance(inputs.single_type, type) and not is_inline:
        # A single-type input reaches an inline prompt as the bare value, so it
        # has no name to reference. Only template-ref prompts key it by type.
        name = inputs.single_type.__name__
        axes.append(("upstream input", name, {name}))

    for name in sorted(_di_template_var_names(node) | _di_resource_template_var_names(node)):
        axes.append(("DI parameter", name, {name}))

    for name in getattr(node, "context", None) or ():
        axes.append(("context field", name, {name}))

    return axes


def _check_unreferenced_inputs(
    node: Node,
    issues: list[LintIssue],
    *,
    referenced: set[str],
    is_inline: bool,
    consumer_known: frozenset[str] | set[str],
    node_label: str,
) -> None:
    """Report a bound name that the node's own template never references.

    The mirror of :func:`_check_template_placeholders`'s demand check. A node
    binds an input, the data arrives, and the prompt never names it, so the model
    never sees the value while every gate passes.

    Demand is inferred from the template TEXT. A custom ``prompt_compiler``
    composes the final message and can consume a name the resolved template never
    spells, so this check reports a warning rather than an error.

    Three suppressions, each read from the IR:

    - ``skip_when``: the predicate receives the extracted input dict on the think
      path and on the agent path, so an input it reads is real demand. The
      callable is opaque, which makes suppression the only sound disposition.
    - An Oracle ``merge_prompt``: ``_build_upstream_context`` injects one entry
      per ``node.inputs`` key into that prompt, so it is a second demand surface
      over the same keys.
    - A bridge alias: when the template names a ``known_vars``-only placeholder,
      the consumer supplies names this walk cannot see. Scoped to the inputs
      axis, since a bridge alias cannot explain an unreferenced context field.
    """
    oracle = getattr(getattr(node, "modifier_set", None), "oracle", None)
    inputs_suppressed = (
        getattr(node, "skip_when", None) is not None
        or (oracle is not None and getattr(oracle, "merge_prompt", None))
        or bool(referenced & consumer_known)
    )

    fan_out = getattr(node, "fan_out_param", None)
    shown = ", ".join(sorted(referenced)) or "none"

    for axis, name, aliases in _supply_axes(node, is_inline=is_inline):
        if name == fan_out or aliases & referenced:
            continue
        if axis == "upstream input" and inputs_suppressed:
            continue
        issues.append(
            LintIssue(
                node_name=node.name,
                param=name,
                kind="template_input_unreferenced",
                required=False,
                message=(
                    f"{node_label}: {axis} '{name}' is bound but the node's "
                    f"template references none of {sorted(aliases)}. The value "
                    f"reaches the node and the model never sees it. Template "
                    f"references: {shown}. Demand is read from the template "
                    f"text, so a prompt_compiler that composes the message may "
                    f"consume this name without naming it."
                ),
            )
        )


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

    # The set of ROOT names this template references. `${claims.items}` is a
    # reference to `claims`. Computed once and read by both directions, so the
    # demand check and the supply check can never disagree about what a template
    # names. The demand loop below still iterates the placeholder LIST: it emits
    # one issue per OCCURRENCE, and iterating the deduped set here would collapse
    # `${bogus.a}` and `${bogus.b}` into a single issue.
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
    return {name for name, binding in (param_res or {}).items() if binding.kind in DI_TEMPLATE_KINDS}


def _di_resource_template_var_names(node: Node) -> set[str]:
    """The node's FromResource parameter names usable as template vars. See neograph-3q6j.

    Kept separate from ``_di_template_var_names`` because a FROM_RESOURCE var
    resolves ONLY on the async arun() driver (its fetch is awaited): it is a VALID
    template-ref placeholder when the compiler accepts ``di_inputs`` — the async
    injector twin stashes the fetched value — but the lint layer flags it with an
    async-driver WARN, in lockstep with the sync-driver fail-loud at runtime.
    """
    param_res = _get_param_res(node)
    return {name for name, binding in (param_res or {}).items() if binding.kind is DIKind.FROM_RESOURCE}


def _predict_input_keys(node: Node, *, include_flattened: bool = True) -> set[str]:
    """Predict the dict keys that _extract_input will produce for this node.

    For dict-form inputs: keys are the dict keys. When *include_flattened* is
    True (default), also adds flattened field names from ``render_for_prompt()``
    return annotations — these are available for template-ref prompts where
    ``_render_with_flattening`` runs. Set *include_flattened=False* for inline
    prompts, which skip flattening and only see raw input dict keys.

    For single-type inputs: the value's TYPE NAME, which is the key the runtime
    now produces for a bare value (``RenderedInput.for_template_ref``). This must
    stay in lockstep with that keying: before neograph-l2a7w a bare value reached
    the compiler unkeyed and lint correctly predicted no keys, so leaving this at
    ``set()`` would flag a valid ``{MyInput}`` placeholder as unresolvable on
    every single-type think node.

    For None inputs: empty set — there is nothing to name.
    """
    ni = normalize_inputs(node.inputs)
    if ni.is_none:
        return set()
    if ni.is_dict_form:
        keys = set(ni.by_name.keys())
        if include_flattened:
            for input_type in ni.by_name.values():
                keys |= _get_flattened_field_names(input_type)
            # Sub-construct port alias (neograph-bluv, F3.4): mirrors
            # renderers._alias_subgraph_input_port exactly, so lint's
            # template-ref prediction and the runtime alias cannot drift.
            port_type = ni.by_name.get(StateKeys.SUBGRAPH_INPUT)
            if port_type is not None:
                keys.add(port_type.__name__)
        return keys
    # Single-type: the runtime keys a bare value by its type name. Inline prompts
    # address the value's ATTRIBUTES (`${var.field}`) against the raw object and
    # never see this synthesized key, so it is template-ref only — the same
    # asymmetry the flattened fields already have.
    single = ni.single_type
    if include_flattened and isinstance(single, type):
        return {single.__name__}
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
    return {fname for fname, finfo in ret_type.model_fields.items() if not finfo.exclude}


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
    except (NameError, AttributeError, TypeError) as exc:
        # A real frame-walking fallback follows, so this is not fatal — but the
        # primary-path failure is otherwise invisible, turning a lint
        # false-negative into an undiagnosable one. Leave a breadcrumb naming
        # the function whose return hint failed to resolve.
        log.debug(
            "return_hint_resolution_failed",
            fn=getattr(fn, "__qualname__", None) or repr(fn),
            owner=getattr(owner_cls, "__qualname__", None) or repr(owner_cls),
            error=str(exc),
        )

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
