"""Shared utilities for example pipelines.

NOT part of the neograph public API. neograph's `prompt_compiler` contract is
deliberately minimal — a callable taking `(template, data, **kw)` and
returning `list[dict]` — and every production consumer writes their own
compiler shaped to their needs: system/user message splits, schema
injection, registry lookup, alias forwarding, multi-template chains.

This helper covers ONLY the simple shape that example pipelines need:
load a `.md` template by name and substitute neograph's BAML-rendered input.

It is built on the PUBLIC `neograph.DefaultPromptCompiler` — the productized
90%-case file-ref compiler — rather than a hand-rolled `string.Template`. The
only example-specific behavior layered on top is the `${input}` alias
convention (see below). The original hand-rolled form is preserved verbatim as
`_manual_template_prompt_compiler` at the bottom of this module: it documents
the escape hatch — how to write a `prompt_compiler` from scratch with nothing
but the stdlib, for when `DefaultPromptCompiler` doesn't fit.

Production code should NOT import from this module. Write your own compiler,
or use `DefaultPromptCompiler` directly. The full protocol — argument shapes,
what neograph guarantees about `data`, the inline-prompt detection rule — is
documented at `docs/concepts/prompt-compiler.mdx` (also at neograph.pro under
Concepts → Prompt Compiler).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from string import Template
from typing import Any

from neograph import DefaultPromptCompiler


class _ExamplePromptCompiler(DefaultPromptCompiler):
    """`DefaultPromptCompiler` + the example ``${input}`` alias convention.

    Every example prompt template references exactly one placeholder,
    ``${input}``, regardless of how many upstream producers feed the node.
    Two conveniences make that possible and both are layered here on top of the
    public compiler's ``build_vars``:

    - Single-input nodes receive ``data`` as a one-entry dict (or a bare
      rendered value); its value is exposed as ``${input}`` so templates never
      have to name the upstream ``@node`` — or a framework key like
      ``neo_subgraph_input`` for sub-graph port inputs.
    - Fan-in nodes still address each producer by ``${param_name}``; the
      ``${input}`` alias is only added for the single-entry case, so it never
      shadows a real multi-input binding.

    Configured for ``dollar`` syntax (``${var}``, matching neograph's own
    inline-prompt substitution) and ``strict=False`` (leave an unfilled
    ``${var}`` verbatim — the `safe_substitute` behavior the examples were
    written against; example prompts embed literal ``{curly}`` braces in
    YAML/JSON code samples that a fail-loud brace compiler would choke on).
    """

    def build_vars(self, input_data: Any, **kw: Any) -> dict[str, Any]:
        vars = super().build_vars(input_data, **kw)
        # Expose the lone upstream value as ${input} (see class docstring).
        if isinstance(input_data, dict):
            if len(input_data) == 1:
                key = next(iter(input_data))
                vars.setdefault("input", vars.get(key))
        elif isinstance(input_data, str):
            vars.setdefault("input", input_data)
        return vars


def make_template_prompt_compiler(
    prompt_dir: str | Path,
    *,
    extension: str = ".md",
) -> Callable[..., list[dict[str, str]]]:
    """Build a minimal prompt_compiler for example pipelines.

    Loads `{prompt_dir}/{name}{extension}` and substitutes neograph's
    BAML-pre-rendered input via the public :class:`DefaultPromptCompiler`
    (``${var}`` dollar syntax, non-strict).

    `${var}` style (not `{var}`) is used because example prompts contain
    code samples with literal `{curly}` braces in YAML/JSON blocks that
    would crash a brace substitution. `${var}` is the same syntax neograph
    uses for its own inline-prompt substitution.

    Conventions used by the example pipelines:

    - Multi-input fan-in nodes receive ``data`` as ``dict[str, str]``;
      templates use ``${param_name}`` for each @node parameter.
    - Single-input nodes receive ``data`` as ``str``; templates use ``${input}``.
    - Single-entry dicts (e.g. sub-graph port input ``neo_subgraph_input``)
      have their value ALSO exposed as ``${input}`` so prompts don't leak
      framework keys.

    This function is NOT part of neograph's public API. Production consumers
    should write their own prompt_compiler shaped to their needs (system/user
    message splits, JSON schema injection, registry lookup, alias forwarding,
    etc.), or use :class:`DefaultPromptCompiler` directly. See
    `docs/concepts/prompt-compiler.mdx` for the full contract. For the fully
    hand-rolled escape hatch (stdlib only), see
    ``_manual_template_prompt_compiler`` below.

    Args:
        prompt_dir: directory containing prompt files.
        extension: file extension (default ``".md"``).

    Returns:
        A callable suitable for ``compile(..., prompt_compiler=...)``.

    Example:
        >>> from pathlib import Path
        >>> from examples._shared import make_template_prompt_compiler
        >>>
        >>> graph = compile(
        ...     pipeline,
        ...     llm_factory=my_factory,
        ...     prompt_compiler=make_template_prompt_compiler(
        ...         Path(__file__).parent / "prompts"
        ...     ),
        ... )
    """
    prompt_dir = Path(prompt_dir)

    def _load(name: str) -> str:
        return (prompt_dir / f"{name}{extension}").read_text()

    return _ExamplePromptCompiler(_load, strict=False, syntax="dollar")


def _manual_template_prompt_compiler(
    prompt_dir: str | Path,
    *,
    extension: str = ".md",
) -> Callable[..., list[dict[str, str]]]:
    """Escape hatch: the same compiler hand-rolled with only the stdlib.

    Kept as living documentation. When :class:`DefaultPromptCompiler` doesn't
    fit — an exotic template grammar, a bespoke message shape, a registry-driven
    loader — a `prompt_compiler` is just a callable ``(template, data, **kw) ->
    list[dict]``. This is the entire contract, implemented from scratch:

    - `data` arrives already BAML-rendered by neograph (a ``dict[str, str]`` for
      fan-in nodes, a ``str`` for single-input nodes) — the compiler only
      substitutes, it does not render.
    - `string.Template(...).safe_substitute` leaves unmatched ``${var}``
      placeholders verbatim, which is why literal ``{curly}`` braces in prompt
      code samples survive.

    `make_template_prompt_compiler` (above) is the production default for the
    examples and produces byte-identical output to this; this form exists so the
    from-scratch knowledge is not lost.
    """
    prompt_dir = Path(prompt_dir)

    def compiler(template: str, data: Any, **_kw: Any) -> list[dict[str, str]]:
        raw = (prompt_dir / f"{template}{extension}").read_text()
        if isinstance(data, dict):
            if len(data) == 1:
                data = {**data, "input": next(iter(data.values()))}
            content = Template(raw).safe_substitute(**data)
        elif isinstance(data, str):
            content = Template(raw).safe_substitute(input=data)
        else:
            content = raw
        return [{"role": "user", "content": content}]

    return compiler
