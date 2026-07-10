"""MCP progress notifications -> neograph's custom stream (``McpProgress``).

MCP defines progress notifications (``progressToken`` + ``notifications/progress``)
for long-running tools; neograph's tool loop is request/response, so a 60s tool
call looked frozen to the caller. This module bridges the adapter's progress
callback seam into the EXISTING ``emit_progress`` custom-stream channel — one
seam, no parallel mechanism: a consumer driving ``astream(stream_mode='custom')``
sees each server notification as a typed :class:`McpProgress` event; a
non-streaming ``arun()`` is unaffected (``emit_progress`` warns once and drops).

Progress is OBSERVABILITY, never control flow: events never enter state, the
checkpoint, the tool result, or budgets — the ``emit_progress`` envelope is a
user payload (not ``neo_*``-prefixed), which is exactly the mechanism that keeps
it out of the state bus. A missing or partial progress stream cannot affect the
tool call's outcome.
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import emit_progress


class McpProgress(BaseModel):
    """One MCP progress notification, surfaced through ``emit_progress``.

    ``progress``/``total``/``message`` are the server's ``report_progress``
    values verbatim; ``server`` and ``tool`` tag which bound tool emitted it
    (from the adapter's per-tool callback context).
    """

    server: str
    tool: str
    progress: float
    total: float | None = None
    message: str | None = None


def _progress_callbacks(server_key: str):
    """Build the adapter ``Callbacks`` forwarding each MCP progress notification
    into ``emit_progress`` as a typed :class:`McpProgress` event.

    The adapter injects a per-tool ``CallbackContext`` (server + tool name) as
    the callback's last argument; ``server_key`` is the fallback tag if a
    context ever arrives without one. The callback runs inside the tool call's
    task (the adapter awaits it on the session's receive path), where
    ``emit_progress`` resolves the node's stream writer via its contextvar.
    """
    from langchain_mcp_adapters.callbacks import Callbacks

    async def _on_progress(progress, total, message, context) -> None:
        emit_progress(
            McpProgress(
                server=getattr(context, "server_name", None) or server_key,
                tool=getattr(context, "tool_name", None) or "unknown",
                progress=progress,
                total=total,
                message=message,
            )
        )

    return Callbacks(on_progress=_on_progress)
