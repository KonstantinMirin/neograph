"""Provider string-args coercion — repairing tool calls whose args arrive as text.

Extracted from ``_tool_loop.py`` (neograph-3ffdg.13) as a pure file split — the
wrapper and helpers below are unchanged, only their home moved. ``_tool_loop.py``
re-exports them, so existing imports keep resolving.

Some providers hand back a tool call whose arguments are a JSON *string* rather
than a decoded object, and some hand back nothing parseable at all. This module
wraps a tool so those two cases become a repairable result instead of a crash
inside the tool body.
"""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger()

# Sentinel key stamped onto a tool_call's ``args`` when the provider returned a
# non-JSON args string we could not parse. The AIMessage schema requires ``args``
# to be a dict, so we cannot leave the raw string in place — instead we stamp this
# marker (preserving the raw string under it) so the tool-execution seam
# (``_agent_cycle._tool_call_precheck``) can surface a RETRIABLE ToolMessage error
# to the LLM INSTEAD of running the tool with empty args neograph-arus. Blanking
# to ``{}`` used to silently run the tool with wrong (empty) arguments.
UNPARSEABLE_ARGS_MARKER = "__neo_unparseable_tool_args__"


def _to_lc_messages(messages: list) -> list:
    """Pure: convert dict messages to LangChain message objects for _generate."""
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

    lc_messages: list[BaseMessage | dict] = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content) if role == "assistant" else m)
        else:
            lc_messages.append(m)
    return lc_messages


class _CoercingToolWrapper:
    """Wraps a tool-bound LLM to coerce string tool_calls.args to dicts.

    Some providers (DeepSeek R1 via OpenRouter) emit tool_calls with
    ``args`` as a JSON string. LangChain AIMessage Pydantic validation
    rejects this. This wrapper catches the ValidationError and
    reconstructs the AIMessage via the ``additional_kwargs`` path which
    handles string arguments correctly (``default_tool_parser`` calls
    ``json.loads`` on them).

    Recovery caveat (documented intent): the ``_generate``/``_agenerate``
    recovery re-invokes the model's low-level generate method directly, which
    re-emits WITHOUT the bound tools — the ``bind_tools`` kwargs (the ``tools=``
    schema) are a ``RunnableBinding`` concern that ``_generate`` bypasses. This
    is acceptable because the recovery runs ONLY after the provider already
    emitted a full tool_calls turn (the string-args ValidationError proves the
    tool call happened); we are re-materializing that same already-returned
    message, not soliciting a fresh tool decision. ``_generate`` is also a
    langchain-core private method, so a bump that renames/removes it must
    surface in CI (see the pinned attribute test) instead of silently taking
    the empty-``AIMessage`` fallback branch below.

    Usage (automatic — applied by ``invoke_with_tools``)::

        wrapped = _CoercingToolWrapper(llm.bind_tools(tools))
        response = wrapped.invoke(messages)  # never raises for string args
    """

    def __init__(self, bound_llm: Any):
        self._bound = bound_llm

    def invoke(self, messages: list, **kwargs: Any) -> Any:
        from pydantic import ValidationError

        try:
            return self._bound.invoke(messages, **kwargs)
        except ValidationError as exc:
            if not _string_args_tool_errors(exc):
                raise
            try:
                raw_result = self._bound._generate(_to_lc_messages(messages), run_manager=None)
                coerced = _coerce_string_args_result(raw_result)
                if coerced is not None:
                    return coerced
                reason = "coercion recovery produced no message"
            except Exception as inner:
                reason = f"coercion _generate failed: {inner}"
            return _empty_recovery_message(reason)

    async def ainvoke(self, messages: list, **kwargs: Any) -> Any:
        # MUST be an explicit override — __getattr__ would forward `ainvoke` to
        # self._bound and silently bypass string-args coercion under arun.
        # **kwargs forwards config through (config rides in kwargs).
        from pydantic import ValidationError

        try:
            return await self._bound.ainvoke(messages, **kwargs)
        except ValidationError as exc:
            if not _string_args_tool_errors(exc):
                raise
            try:
                raw_result = await self._bound._agenerate(_to_lc_messages(messages), run_manager=None)
                coerced = _coerce_string_args_result(raw_result)
                if coerced is not None:
                    return coerced
                reason = "coercion recovery produced no message"
            except Exception as inner:
                reason = f"coercion _agenerate failed: {inner}"
            return _empty_recovery_message(reason)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bound, name)


def _string_args_tool_errors(exc: Any) -> list:
    """Pure: the tool_calls string-args ValidationError entries (empty if none).

    Shared by _CoercingToolWrapper.invoke and .ainvoke; a non-empty list logs
    the coercion warning and triggers the _generate/_agenerate recovery.
    """
    tool_call_errors = [
        e for e in exc.errors() if "tool_calls" in str(e.get("loc", "")) and e.get("type") == "dict_type"
    ]
    if tool_call_errors:
        log.warning(
            "tool_calls_args_coercion",
            error_count=len(tool_call_errors),
            hint="provider returned tool_calls.args as JSON string; reconstructing via additional_kwargs path",
        )
    return tool_call_errors


def _unparseable_args_raw(tc: dict) -> str | None:
    """Pure read: the raw un-parseable args string if ``tc``'s args carry the
    marker, else None. The tool-execution seam consults this to decide whether to
    emit a retriable error instead of invoking the tool. See neograph-arus."""
    args = tc.get("args")
    if isinstance(args, dict) and UNPARSEABLE_ARGS_MARKER in args:
        raw = args[UNPARSEABLE_ARGS_MARKER]
        return raw if isinstance(raw, str) else str(raw)
    return None


def _coerce_string_args_result(raw_result: Any) -> Any | None:
    """Pure: extract the message from a _generate/_agenerate result and json-load
    any string tool_call args. Returns the coerced message, or None if empty."""
    import json as _json

    if raw_result.generations:
        gen = raw_result.generations[0]
        raw_msg = gen.message if hasattr(gen, "message") else gen
        if hasattr(raw_msg, "tool_calls"):
            for tc in raw_msg.tool_calls:
                if isinstance(tc.get("args"), str):
                    raw = tc["args"]
                    try:
                        tc["args"] = _json.loads(raw)
                    except (_json.JSONDecodeError, TypeError):
                        # The provider returned tool_calls.args as a string that
                        # is ALSO not valid JSON — we cannot reconstruct the
                        # intended arguments. Rather than blank to {} (which
                        # silently runs the tool with empty args), stamp the
                        # unparseable marker with the raw string preserved: the
                        # tool-execution seam (_agent_cycle._tool_call_precheck)
                        # detects it and emits a retriable ToolMessage ERROR back
                        # to the LLM so it can re-emit valid args. See neograph-arus.
                        log.warning(
                            "tool_calls_args_unparseable",
                            tool=tc.get("name"),
                            raw_args=raw,
                            hint="provider returned tool_calls.args as a non-JSON string; "
                            "surfacing a retriable error to the model instead of running with empty args",
                        )
                        tc["args"] = {UNPARSEABLE_ARGS_MARKER: raw}
        return raw_msg
    return None


def _empty_recovery_message(reason: str) -> Any:
    """Build the empty-``AIMessage`` string-args coercion fallback, WARNING first.

    Both the ``_generate`` raised branch and the coercion-produced-nothing branch
    land here. Pre-audit only the raised branch logged; the produced-nothing
    branch shipped an empty "the model said nothing" turn silently (7ymj). The
    warning names the recovery failure so the empty turn is never a silent
    dead-end for the caller.
    """
    from langchain_core.messages import AIMessage

    log.warning(
        "tool_calls_coercion_empty_fallback",
        reason=reason,
        hint="string-args coercion could not recover a message; returning an empty AIMessage",
    )
    return AIMessage(content="")
