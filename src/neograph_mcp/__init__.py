"""neograph_mcp — the optional ``neograph[mcp]`` client DX battery.

An OVERRIDABLE default that turns a typed MCP server config into the consumer-owned
``tool_factories`` + resource fetcher/replayer that ``compile()`` already accepts —
the ``DefaultPromptCompiler`` seam-plus-battery story, applied to MCP client
stitching. Ships OUTSIDE ``src/neograph`` (a second wheel package) so neograph core
stays MCP-free and the no-session-ownership guard stays green.

Requires the ``mcp`` extra (``mcp`` + ``langchain-mcp-adapters``)::

    pip install 'neograph[mcp]'

Importing this package without the extra fails loud with the install hint above
(the langfuse-observe fail-loud precedent) rather than surfacing a bare
``ModuleNotFoundError`` from deep in the import graph.
"""

from __future__ import annotations


def _require_mcp() -> None:
    """Fail loud with an install hint when the ``mcp`` extra is not installed."""
    try:
        import langchain_mcp_adapters  # noqa: F401
        import mcp  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "neograph_mcp requires the 'mcp' extra (mcp + langchain-mcp-adapters). "
            "Install it with:  pip install 'neograph[mcp]'  (or: uv add 'neograph[mcp]')"
        ) from exc


_require_mcp()

from neograph_mcp._auth import (  # noqa: E402 — after the fail-loud extra check by design
    client_credentials_auth,
)
from neograph_mcp._client import (  # noqa: E402 — after the fail-loud extra check by design
    HttpServer,
    StdioServer,
    ToolFactory,
    mcp_resource_fetcher,
    mcp_tool_factories,
    mcp_tool_factory,
)
from neograph_mcp._progress import (  # noqa: E402 — after the fail-loud extra check by design
    McpProgress,
)
from neograph_mcp._prompt import (  # noqa: E402 — after the fail-loud extra check by design
    mcp_prompt_source,
)
from neograph_mcp._run_context import (  # noqa: E402 — after the fail-loud extra check by design
    McpRunContext,
    mcp_run_context,
)
from neograph_mcp._session import (  # noqa: E402 — after the fail-loud extra check by design
    McpCallResult,
    McpSession,
    McpToolCallError,
    mcp_session,
)

__all__ = [
    "StdioServer",
    "HttpServer",
    "ToolFactory",
    "mcp_tool_factories",
    "mcp_tool_factory",
    "mcp_resource_fetcher",
    "mcp_session",
    "McpSession",
    "McpCallResult",
    "McpToolCallError",
    "mcp_run_context",
    "McpRunContext",
    "McpProgress",
    "client_credentials_auth",
    "mcp_prompt_source",
]
