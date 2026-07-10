"""Public fail-loud prompt primitives + ``DefaultPromptCompiler`` (Layer-2 node DX).

Four composable pieces plus one typed error, so the 90%-case file-ref compiler is
``compile(construct, prompt_compiler=DefaultPromptCompiler(Path("prompts")))`` with
zero app compiler code, and a custom compiler is ~10 lines of primitive composition:

- ``substitute(template, vars, *, strict, syntax)`` — the ONE substitution
  primitive. ``strict=True`` (default) raises :class:`PromptVarMissing` on an
  unfilled placeholder; brace-safe by construction so an injected JSON schema's
  literal ``{}`` renders intact.
- ``render_inputs(input_data, *, renderer)`` — the exported view of
  ``renderers.build_rendered_input(...).for_template_ref`` (reuse, not a reimpl).
- ``inject_schema(vars, output_model, *, output_schema, key)`` — rides
  ``describe_type`` to add the output schema under ``vars[key]``.
- ``DefaultPromptCompiler`` — a callable satisfying the prompt_compiler seam;
  ``load_template`` / ``build_vars`` / ``render_messages`` are override hooks
  (override one, keep the rest).

Engine-free: no langgraph import. Reuses the ONE placeholder scanner
(``_placeholders``) and the ONE input-rendering path (``renderers``) — this module
never opens a second rendering path (the hjwv anti-duplication invariant).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from neograph._placeholders import BRACE_RE, DOLLAR_RE, apply_scanner
from neograph.describe_type import describe_type
from neograph.errors import ConfigurationError, PromptVarMissing
from neograph.renderers import Renderer, build_rendered_input

# A syntax is either a named grammar or a user tokenizer ``(template, resolve) -> str``
# where ``resolve(name) -> str`` is neograph's per-match resolver (strict policy
# carried by substitute, scanning carried by the callable).
SyntaxSpec = str | Callable[[str, Callable[[str], str]], str]

_NAMED_SYNTAX = {"brace": BRACE_RE, "dollar": DOLLAR_RE}
_VERBATIM = {"brace": lambda n: "{" + n + "}", "dollar": lambda n: "${" + n + "}"}


def substitute(
    template: str,
    vars: dict[str, Any],
    *,
    strict: bool = True,
    syntax: SyntaxSpec = "brace",
) -> str:
    """Fill ``template`` placeholders from ``vars`` in a single brace-safe pass.

    ``strict=True`` (default) raises :class:`PromptVarMissing` on the first token
    with no matching key — the fail-loud default that makes the
    ``{domain}``-reaches-the-model bug impossible without an explicit opt-out.
    ``strict=False`` leaves an unfilled token verbatim.

    ``syntax`` selects the grammar: ``"brace"`` (``{var}``, default), ``"dollar"``
    (``${var}``), or a callable tokenizer ``(template, resolve) -> str``.
    """
    verbatim = _VERBATIM.get(syntax if isinstance(syntax, str) else "", lambda n: n)

    def resolve(name: str) -> str:
        if name in vars:
            return str(vars[name])
        if strict:
            raise PromptVarMissing.of(name, sorted(vars))
        return verbatim(name)

    if callable(syntax):
        return syntax(template, resolve)
    pattern = _NAMED_SYNTAX.get(syntax)
    if pattern is None:
        raise ConfigurationError.build(
            f"unknown substitution syntax '{syntax}'",
            hint="use 'brace', 'dollar', or a callable (template, resolve) -> str",
        )
    return apply_scanner(template, pattern, resolve)


def render_inputs(
    input_data: Any,
    *,
    renderer: Renderer | None = None,
) -> dict[str, Any]:
    """The template-ref view of ``input_data``: BAML-rendered values + flattened
    fields, exactly what the prompt_compiler seam receives.

    A thin exported wrapper over ``build_rendered_input(...).for_template_ref`` —
    the same internal rendering path dispatch uses, not a re-implementation.

    Total dict contract: template-ref prompts bind vars by name, so any input
    with no named keys collapses to ``{}`` — ``None`` (an all-DI leaf think node
    fed purely by ``run(input=...)``) and single-type non-dict values (a bare
    ``str``/``BaseModel`` has no name to bind a var to). This owns the non-dict
    contract at the primitive so ``inject_schema``/``substitute`` downstream never
    see a non-dict (neograph-4tsd: ``dict(None)`` crashed ``inject_schema``).
    """
    view = build_rendered_input(input_data, renderer).for_template_ref
    return view if isinstance(view, dict) else {}


def inject_schema(
    vars: dict[str, Any],
    output_model: type[BaseModel] | None = None,
    *,
    output_schema: str | None = None,
    key: str = "json_schema",
) -> dict[str, Any]:
    """Return ``vars`` with ``vars[key]`` set to the output schema string.

    Uses the precomputed ``output_schema`` when the seam already threaded one;
    otherwise rides ``describe_type(output_model)``. Returns a new dict — the
    input is not mutated.
    """
    if output_schema is not None:
        schema = output_schema
    elif output_model is not None:
        schema = describe_type(output_model)
    else:
        raise ConfigurationError.build(
            "inject_schema needs output_model or output_schema",
            hint="pass output_model=YourModel (or the precomputed output_schema=).",
        )
    out = dict(vars)
    out[key] = schema
    return out


# A loader is either a directory of ``{name}.md`` templates or a callable name->text.
TemplateLoader = str | Path | Callable[[str], str]


class DefaultPromptCompiler:
    """The 90%-case file-ref prompt_compiler — no app compiler code required.

    ``DefaultPromptCompiler(Path("prompts"))`` loads ``prompts/{name}.md``, renders
    the input via ``render_inputs``, injects the output schema, and substitutes with
    ``substitute`` (fail-loud, brace-safe). Pass any callable ``loader`` for a custom
    template source. Override exactly one of ``load_template`` / ``build_vars`` /
    ``render_messages`` for a ~10-line custom compiler; keep the rest.

    Opt-in: this is just a callable passed as ``prompt_compiler=``. Consumers with
    their own compiler are untouched — the seam is unchanged.
    """

    def __init__(
        self,
        loader: TemplateLoader,
        *,
        strict: bool = True,
        syntax: SyntaxSpec = "brace",
        system: str | None = None,
        schema_var: str = "json_schema",
        suffix: str = ".md",
    ) -> None:
        self.loader = loader
        self.strict = strict
        self.syntax = syntax
        self.system = system
        self.schema_var = schema_var
        self.suffix = suffix

    def load_template(self, template: str) -> str:
        """Resolve ``template`` to its raw text (``{name}{suffix}`` under a dir
        loader, or ``loader(template)`` for a callable loader).

        ``suffix`` defaults to ``.md``; pass ``suffix='.txt'`` for a ``.txt``
        template dir (the shape all three shipped consumers use) instead of
        hand-rolling a callable loader.
        """
        if callable(self.loader):
            text = self.loader(template)
            if text is None:
                # A callable loader returns None for a name it cannot resolve (the
                # not-found contract, mirroring the dir loader's FileNotFoundError).
                # Fail loud HERE with the template name rather than letting None
                # surface as an opaque TypeError deep in substitution.
                raise ConfigurationError.build(
                    f"Prompt template '{template}' not found by the compiler's loader",
                    hint="The callable loader returned None for this name. Check the "
                    "node's prompt= against the names the loader can resolve (e.g. "
                    "the MCP server's declared prompts for mcp_prompt_source).",
                )
            return text
        return (Path(self.loader) / f"{template}{self.suffix}").read_text()

    def build_vars(
        self,
        input_data: Any,
        *,
        output_model: type[BaseModel] | None = None,
        output_schema: str | None = None,
        di_inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Render inputs and, when an output type is known, inject its schema.

        ``di_inputs`` are the resolved ``FromInput``/``FromConfig`` values keyed
        by the node's parameter names (e.g. ``{"domain": "oncology"}``). They
        form the BASE layer of the vars dict; upstream node OUTPUTS rendered from
        ``input_data`` are laid on top, so on a name collision the node-local
        output shadows the DI value. Rationale: an upstream producer named
        ``domain`` is the more specific, dataflow-derived value for this node,
        while ``di_inputs`` is run-wide ambient context — the narrower binding
        wins. This is ALSO the zero-behavior-change rule: di_inputs only fills
        names not already produced by an upstream node, so no existing pipeline's
        ``{name}`` binding changes meaning when a FromInput param happens to
        collide. ``None`` collapses to ``{}`` (the render_inputs total-dict
        contract), so an all-DI leaf node still gets its ``{domain}`` var.
        """
        vars: dict[str, Any] = dict(di_inputs or {})
        vars.update(render_inputs(input_data))
        if output_model is not None or output_schema is not None:
            vars = inject_schema(
                vars,
                output_model,
                output_schema=output_schema,
                key=self.schema_var,
            )
        return vars

    def render_messages(
        self,
        template_text: str,
        vars: dict[str, Any],
        *,
        node_name: str = "",
    ) -> list[dict[str, str]]:
        """Substitute ``vars`` into the template and wrap as a message list
        (optional system message + user message).

        ``node_name`` is the name of the node whose prompt is being compiled —
        threaded from the graph at runtime (and from ``__call__``). The default
        implementation ignores it; override to shape per-node roles (e.g. an
        ``explore`` node emitting a single user message while every other node
        gets a system line naming the node). See the message-shaping recipe in
        the prompt-compiler docs.
        """
        content = substitute(template_text, vars, strict=self.strict, syntax=self.syntax)
        messages: list[dict[str, str]] = []
        if self.system is not None:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": content})
        return messages

    def __call__(
        self,
        template: str,
        input_data: Any,
        *,
        output_model: type[BaseModel] | None = None,
        output_schema: str | None = None,
        di_inputs: dict[str, Any] | None = None,
        node_name: str = "",
        **_kw: Any,
    ) -> list[dict[str, str]]:
        text = self.load_template(template)
        vars = self.build_vars(
            input_data,
            output_model=output_model,
            output_schema=output_schema,
            di_inputs=di_inputs,
        )
        return self.render_messages(text, vars, node_name=node_name)
