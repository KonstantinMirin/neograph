"""``LintIssue`` and the lint-kind registry.

Extracted from ``lint.py`` (neograph-3ffdg.10) as a pure file split — the
dataclasses and the registry below are unchanged, only their home moved.
``lint.py`` re-exports them, so existing imports keep resolving.

``LINT_KIND_META`` is the authoritative description of every lint kind;
``scripts/gen_api_manifest.py`` reads it to generate the reference docs, so this
module has an external consumer besides the linter itself.
"""

from __future__ import annotations

from dataclasses import dataclass


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
        "`Annotated[T, FromInput]` -- resolved from `config['configurable']`, originally from `run(input={...})`.",
    ),
    "from_config": LintKindMeta(
        "varies",
        "`Annotated[T, FromConfig]` -- resolved from `config['configurable']`, passed directly in `config=`.",
    ),
    "from_input_model": LintKindMeta(
        "varies",
        "Bundled `BaseModel` via `FromInput` -- each model field must exist in config.",
    ),
    "from_config_model": LintKindMeta(
        "varies",
        "Bundled `BaseModel` via `FromConfig` -- each model field must exist in config.",
    ),
    # Template placeholders.
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
    "output_field_unconsumed": LintKindMeta(
        "WARN",
        "A field of a node's typed output that nothing reads: no downstream node "
        "takes the model, no template reads it by name, and it is not the graph's "
        "terminal output. It costs tokens on every call and cannot affect the "
        "answer.",
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
        "Loop `when` is a string that is not registered in the condition registry.",
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
