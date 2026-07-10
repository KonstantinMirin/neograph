"""The single ``structuredContent`` -> ``output_model`` rehydration.

Both real MCP result paths — ``McpSession.call(output_model=)`` (``_session.py``)
and the bound-tool typed wrapper ``_wrap_output_model`` (``_client.py``) — and the
consumer test fakes (``neograph_mcp.testing``) rehydrate a server's
``structuredContent`` dict into the consumer's declared model the SAME way: run the
caller's ``parse`` if given, else ``output_model.model_validate``, and raise ONE
identical ``ValueError`` when the tool declared ``output_model=`` but the server
returned no ``structuredContent``.

Factoring it here (a neutral, dependency-light module — imports only ``pydantic``)
makes that parity STRUCTURAL rather than three hand-cloned copies drifting apart:
a fake green on this helper is green on the real server because it IS the real
rehydration. Prior to this the block was duplicated verbatim at two sites; the fake
would have been a third clone and the anti-echo-chamber invariant would rest on
copy-paste discipline instead of a shared call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

ParseFn = Callable[[dict[str, Any]], BaseModel]


def rehydrate(
    output_model: type[BaseModel],
    parse: ParseFn | None,
    structured: dict[str, Any] | None,
    tool_name: str,
) -> BaseModel:
    """Rehydrate ``structured`` into ``output_model`` (via ``parse`` if given).

    Raises the identical typed ``ValueError`` — naming the tool, mentioning
    ``structuredContent``, and pointing at the server return-annotation fix — when
    ``structured`` is ``None`` (the tool has a bare ``-> dict`` / ``-> list``
    annotation, so FastMCP emitted no ``structuredContent``). A ``ValidationError``
    from a genuine schema mismatch propagates unchanged.
    """
    if structured is None:
        raise ValueError(
            f"MCP tool '{tool_name}' was declared with output_model="
            f"{output_model.__name__} but the server returned no structuredContent. "
            f"The server tool likely has a bare '-> dict' / '-> list' return "
            f"annotation; use '-> dict[str, Any]' or a Pydantic return type so "
            f"FastMCP emits structuredContent."
        )
    return parse(structured) if parse is not None else output_model.model_validate(structured)
