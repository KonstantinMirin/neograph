"""Shared utilities for example pipelines.

NOT part of the neograph public API. neograph's `prompt_compiler` contract is
deliberately minimal — a callable taking `(template, data, **kw)` and
returning `list[dict]` — and every production consumer writes their own
compiler shaped to their needs: system/user message splits, schema
injection, registry lookup, alias forwarding, multi-template chains.

This helper covers ONLY the simple shape that example pipelines need:
load a `.md` template by name and substitute neograph's BAML-rendered input.

Production code should NOT import from this module. Write your own compiler.
The full protocol — argument shapes, what neograph guarantees about
`data`, the inline-prompt detection rule — is documented at
`docs/concepts/prompt-compiler.mdx` (also at neograph.pro under
Concepts → Prompt Compiler).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from string import Template
from typing import Any


def make_template_prompt_compiler(
    prompt_dir: str | Path,
    *,
    extension: str = ".md",
) -> Callable[..., list[dict[str, str]]]:
    """Build a minimal prompt_compiler for example pipelines.

    Loads `{prompt_dir}/{name}{extension}` and substitutes neograph's
    BAML-pre-rendered input via `string.Template.safe_substitute`.

    `${var}` style (not `{var}`) is used because example prompts contain
    code samples with literal `{curly}` braces in YAML/JSON blocks that
    would crash `str.format`. `${var}` is the same syntax neograph uses
    for its own inline-prompt substitution.

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
    etc.). See `docs/concepts/prompt-compiler.mdx` for the full contract.

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
