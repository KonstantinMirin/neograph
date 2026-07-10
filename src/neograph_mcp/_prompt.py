"""MCP server prompts as a neograph prompt SOURCE — ``mcp_prompt_source``.

Closes the third MCP primitive: tools bind via ``mcp_tool_factory``/``mcp_session``,
resources hydrate via ``mcp_resource_fetcher``/``FromResource`` — and a server's
curated prompt template (``prompts/list`` + ``prompts/get``) now backs a neograph
node prompt through the UNCHANGED ``DefaultPromptCompiler(loader=)`` +
``lint(template_resolver=)`` seam. An MCP prompt is just another loader source
feeding the same compiler: no second prompt-resolution path, and the normal
template-ref placeholder rules (and lint coverage) apply untouched.

How a server template becomes a neograph template — **placeholder-echo**: the
loader reads the prompt's DECLARED argument names from ``prompts/list``, then
calls ``prompts/get`` passing each argument as its own brace placeholder
(``deal_context -> "{deal_context}"``). A pure-passthrough server prompt echoes
them, so the returned text IS the raw template carrying ``{arg}`` placeholders
for the normal compiler to bind from node inputs / ``di_inputs``.

Scope and caveats (deliberate):

- **Pure-template passthrough only.** A server prompt that TRANSFORMS its
  arguments (rather than interpolating them) will bake the literal placeholder
  strings into whatever it computes — placeholder-echo cannot recover a template
  from it.
- **Role collapse.** ``prompts/get`` returns role-tagged messages; the loader
  concatenates their TEXT into one template string (the loader contract is
  name->text). Role structure is dropped; a non-text content block fails loud.
- **Brace collision.** A server template carrying literal ``{...}`` (e.g. JSON
  examples) mis-parses under strict brace substitution and fails loud at
  compile — inherent to brace templating, same as any file template.
- **Lint goes LIVE.** ``lint(template_resolver=mcp_prompt_source(...))`` performs
  a real ``prompts/get`` round-trip — there is no offline way to read a server's
  declared arguments. The loader memoizes (name -> text) by default, so lint and
  compile share ONE connect per prompt name.

The ``MultiServerMCPClient``/session is created and owned strictly INSIDE the
loader closure (the nmb2 invariant), structurally copying ``mcp_resource_fetcher``:
zero network at construction; the connect fires on the first load of each name.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from neograph_mcp._client import (
    HttpServer,
    StdioServer,
    TokenProvider,
    _client_for,
    _resolve_token_no_config,
    _run_sync,
    _unwrap_single,
)


def mcp_prompt_source(
    server_key: str,
    spec: StdioServer | HttpServer,
    *,
    token_provider: TokenProvider | None = None,
    stdio_token_arg: str = "token",
    timeout: float | None = 30.0,
) -> Callable[[str], str | None]:
    """Build a ``DefaultPromptCompiler(loader=)`` / ``lint(template_resolver=)``
    loader that resolves prompt names against ``spec``'s MCP server.

    The returned closure uses its INCOMING template argument as the MCP prompt
    name (the same ``{name}.md`` name->text contract as the directory loader),
    so ONE compiler backs any number of MCP-backed nodes. A prompt the server
    does not declare returns ``None`` — not-found per the loader contract; the
    compiler fails loud upstream. Results are memoized per name (a server
    prompt template is stable within a run; lint + compile share one connect).

    ``token_provider`` mints identity per fetch (no config on the loader path —
    same shape as ``mcp_resource_fetcher``); if the prompt declares a
    ``stdio_token_arg`` argument, the minted token is passed for it instead of a
    placeholder (identity is framework-carried, mirroring stdio tool injection).
    ``timeout`` bounds the connect and the prompt round-trip. Construction is
    ZERO network I/O.
    """
    cache: dict[str, str | None] = {}

    async def _fetch(name: str) -> str | None:
        token = await _resolve_token_no_config(token_provider)
        try:
            async with asyncio.timeout(timeout):
                client = _client_for(server_key, spec, token)
                async with client.session(server_key) as session:
                    declared = await _declared_prompt_args(session, name)
                    if declared is None:
                        return None
                    arguments: dict[str, str] = {}
                    for arg in declared:
                        if arg == stdio_token_arg and token is not None:
                            arguments[arg] = token
                        else:
                            arguments[arg] = "{" + arg + "}"
                    result = await session.get_prompt(name, arguments=arguments)
                    return _messages_text(name, result.messages)
        except BaseException as exc:  # noqa: BLE001 - normalise the transport's group wrapping
            raise _unwrap_single(exc) from None

    def loader(template: str) -> str | None:
        if template not in cache:
            cache[template] = _run_sync(_fetch(template))
        return cache[template]

    return loader


async def _declared_prompt_args(session: Any, name: str) -> list[str] | None:
    """The declared argument names of prompt ``name`` via paginated
    ``prompts/list`` — or ``None`` when the server does not declare the prompt."""
    cursor: str | None = None
    while True:
        result = await session.list_prompts(cursor)
        for prompt in result.prompts:
            if prompt.name == name:
                return [a.name for a in (prompt.arguments or [])]
        cursor = getattr(result, "nextCursor", None)
        if not cursor:
            return None


def _messages_text(name: str, messages: Any) -> str:
    """Concatenate the TEXT of a ``prompts/get`` result into one template string.

    Role structure is collapsed (the loader contract is name->text); a non-text
    content block fails loud — there is no meaningful way to fold an image or
    resource block into a text template."""
    texts: list[str] = []
    for message in messages or []:
        content = message.content
        text = getattr(content, "text", None)
        if text is None:
            raise ValueError(
                f"MCP prompt '{name}' returned a non-text content block "
                f"({type(content).__name__}); mcp_prompt_source supports text-only "
                f"prompt templates (the loader contract is name->text)."
            )
        texts.append(str(text))
    return "\n\n".join(texts)
