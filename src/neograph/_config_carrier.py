"""Helpers for reading from and writing to ``config['configurable']``.

Neutral low-level leaf: imports only ``RunnableConfig`` (a typed dict) and the
``StateKeys`` constants. Every framework layer that stashes a value onto the
config side-channel (``_inject_oracle_config``, ``_inject_di_inputs`` and its
async twin, ``_inject_resource_manifest``) or reads the framework-minted run id
routes through here, so the copy-not-mutate carrier idiom and the run-id read
each live at ONE site.

Why a single carrier helper: the fresh-dict rule
``{**config, "configurable": {**configurable, K: v}}`` was re-inlined at four
seams (review PAT-02). Inlining it per-site is a per-twin drift hazard — the
sync/async injector twins both built it verbatim, exactly the one-indirection
blind spot the thinness guard was built to close. Single-siting it makes the
copy-not-mutate contract un-forkable.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from neograph._state_keys import StateKeys


def _with_configurable(config: RunnableConfig, **kv: object) -> RunnableConfig:
    """Return a FRESH config with ``kv`` merged into ``config['configurable']``.

    Copy-not-mutate: neither ``config`` nor its ``configurable`` sub-dict is
    mutated, so a caller re-invoked per superstep (the agent cycle) is
    idempotent. This is the single source of the framework's config
    side-channel write idiom — used by ``_inject_oracle_config``,
    ``_inject_di_inputs`` / ``_ainject_di_inputs``, and
    ``_inject_resource_manifest``. Pass dynamic keys via ``**{KEY: value}``.
    """
    return {**config, "configurable": {**config.get("configurable", {}), **kv}}


def run_id_of(config: RunnableConfig | None) -> str | None:
    """The framework-minted per-run id from ``config['configurable']``, or None.

    Canonical reader for ``StateKeys.RUN_ID``. Absent when the graph was invoked
    directly, bypassing the runner's mint (``_mint_run_id`` in ``runner.py``) —
    in that case there is no sound per-run key, so callers treat None as "no run
    scope" (skip run-id log binding, skip the per-run cache).
    """
    if not config:
        return None
    configurable = config.get("configurable") or {}
    return configurable.get(StateKeys.RUN_ID)
