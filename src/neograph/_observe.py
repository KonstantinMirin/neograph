"""``observe=`` — opt-in Langfuse auto-attach (per-run config merge + flush).

Extracted from ``runner.py`` (neograph-3ffdg.9) as a pure file split — the
functions below are unchanged, only their home moved.

A THIN-VERB concern (docs/design/three-layer-principle): ``observe=`` merges a
CallbackHandler into config BEFORE the engine verb and flushes AFTER it — it
NEVER wraps the engine. langfuse is the OPTIONAL [langfuse] extra, so every
langfuse import stays FUNCTION-LOCAL inside the observe path (enforced by
TestNoModuleLevelLangfuseImports, which scans all of src/neograph and therefore
covers this module automatically); the core import graph stays langfuse-free.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.runnables import RunnableConfig

from neograph._config_carrier import _with_configurable, run_id_of
from neograph._run_cache import evict_run
from neograph._state_keys import StateKeys
from neograph.errors import ConfigurationError


def _observe_wants_langfuse(observe: bool | str | None) -> bool:
    """Normalize the ``observe`` argument to 'is Langfuse requested?'.

    ``None`` / ``False`` -> off; ``True`` / ``'langfuse'`` -> on; any other string
    is an explicit misconfiguration and raises (fail-loud on a typo'd backend)."""
    if observe is None or observe is False:
        return False
    if observe is True or observe == "langfuse":
        return True
    raise ConfigurationError.build(
        f"unknown observe backend {observe!r}",
        hint="use observe=True or observe='langfuse' (the only backend today).",
    )


def _langfuse_keys_present() -> bool:
    """Both Langfuse keys must be set before we attach or flush.

    Gating on BOTH LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY is the clean-no-op
    boundary: a half-configured handler warns and silently drops traces, and
    ``get_client().flush()`` would flush a mis-configured client. Absent/partial
    keys -> observe is a no-op, so offline/CI stays green."""
    return bool(os.environ.get("LANGFUSE_SECRET_KEY")) and bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))


def _merge_observe_callbacks(config: RunnableConfig | None, observe: bool | str | None) -> RunnableConfig | None:
    """Return a config with a Langfuse ``CallbackHandler`` MERGED into
    ``callbacks`` — fresh dict, fresh list, never mutating the caller's, never
    clobbering a user-supplied handler.

    The attached handler is pinned to a trace id DERIVED from this run's
    ``RUN_ID`` (``Langfuse.create_trace_id(seed=...)``, deterministic), and that
    id is carried back on ``configurable`` under ``StateKeys.TRACE_ID`` so the
    node lifecycle logs can emit it. Deriving this direction — rather than
    sourcing ``run_id`` from Langfuse — leaves ``_mint_run_id`` and everything
    keyed on ``run_id`` (``_run_cache`` / ``evict_run``) untouched, and makes the
    logs<->traces join computable from a bare log line.

    No-op (returns the config unchanged) when observe is off or the env gate
    fails. Deduplicates: if a Langfuse handler is already present (the user wired
    one manually), no second one is added. Both of those paths also carry NO
    trace id: in the dedupe case the user's handler owns a trace we did not
    derive, so advertising ours would name a trace that does not exist."""
    if not _observe_wants_langfuse(observe) or not _langfuse_keys_present():
        return config

    from langfuse import Langfuse  # function-local: optional extra
    from langfuse.langchain import CallbackHandler  # function-local: optional extra

    config = config or {}
    existing = config.get("callbacks")
    if existing is None:
        existing_list: list[Any] = []
    elif isinstance(existing, list):
        existing_list = existing
    else:
        # A BaseCallbackManager (or other non-list) — attaching to it would mean
        # mutating a shared object. Fail loud with the documented escape hatch.
        raise ConfigurationError.build(
            "observe= requires config['callbacks'] to be a list (or absent)",
            found=type(existing).__name__,
            hint="pass callbacks as a list, or attach the Langfuse handler manually.",
        )

    if any(isinstance(h, CallbackHandler) for h in existing_list):
        return config  # dedupe: user already wired a Langfuse handler (their trace, not ours)

    run_id = run_id_of(config)
    if run_id is None:
        # No run scope (graph invoked outside the verbs) -> nothing to seed from.
        return {**config, "callbacks": [*existing_list, CallbackHandler()]}

    trace_id = Langfuse.create_trace_id(seed=run_id)
    config = _with_configurable(config, **{StateKeys.TRACE_ID: trace_id})
    handler = CallbackHandler(trace_context={"trace_id": trace_id})
    return {**config, "callbacks": [*existing_list, handler]}


def _flush_observe(observe: bool | str | None) -> None:
    """Flush the Langfuse client on completion — symmetric with the attach gate
    (flush iff we would have attached), so a mis-configured client is never
    flushed. Safe to call unconditionally from a verb's ``finally``."""
    if not _observe_wants_langfuse(observe) or not _langfuse_keys_present():
        return
    from langfuse import get_client  # function-local: optional extra

    get_client().flush()


def _evict_run_cache(config: RunnableConfig | None) -> None:
    """Drop this run's per-run handle/resource cache entries — the SAME finalize
    seam as ``_flush_observe``, wired into every verb's ``finally``. ``_prepare``
    minted the RUN_ID into ``config['configurable']``; reading it back here (not
    threading it through) keeps one hook per verb. No RUN_ID (graph invoked
    directly) -> nothing was cached -> no-op."""
    run_id = run_id_of(config)
    if run_id is not None:
        evict_run(run_id)
