"""LIVE verification of neograph-s65y2 sub-claim (c) — Langfuse actually records
the trace under the id neograph derived from ``run_id``.

The offline suite (``test_observe_trace_correlation.py``) proves the in-process
half: the derivation, that the handler receives it, and that the field lands on
every node line. It CANNOT prove that Langfuse honours ``trace_context`` — that
is an external side effect. This file closes that gap against the real API.

RUN IT (keys required; mirrors the MCP e2e harness convention — "no-network !=
no-run", the credentials are the extra here):

    set -a && . .env && set +a
    uv run --extra dev --extra langfuse pytest tests/test_observe_trace_live.py

Without ``LANGFUSE_SECRET_KEY`` + ``LANGFUSE_PUBLIC_KEY`` the module skips. That
skip is a KNOWN, DOCUMENTED hole, not silent coverage: the offline file is the
default gate and this one is the periodic live check. Do not read a green default
run as evidence that (c) holds.
"""

from __future__ import annotations

import base64
import json
import os
import time
import types
import urllib.error
import urllib.request

import pytest
from structlog.testing import capture_logs

import neograph
from neograph import compile, construct_from_module, node
from tests.fakes import build_test_compile_kwargs
from tests.schemas import Claims, RawText

pytestmark = pytest.mark.skipif(
    not (os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY")),
    reason="live Langfuse keys absent — see this module's docstring for the run command",
)

_INGEST_ATTEMPTS = 12
_INGEST_INTERVAL_S = 5


def _auth_header() -> str:
    raw = f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
    return "Basic " + base64.b64encode(raw).decode()


def _get_trace(trace_id: str) -> tuple[int, dict | None]:
    base = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
    req = urllib.request.Request(
        f"{base}/api/public/traces/{trace_id}", headers={"Authorization": _auth_header()}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, None


def _await_trace(trace_id: str) -> dict:
    """Poll until ingestion lands. Langfuse ingests asynchronously — observed
    ~25s to first 200 — so a single immediate GET would false-negative."""
    for _ in range(_INGEST_ATTEMPTS):
        time.sleep(_INGEST_INTERVAL_S)
        status, body = _get_trace(trace_id)
        if status == 200 and body:
            return body
    raise AssertionError(
        f"trace {trace_id} never appeared after "
        f"{_INGEST_ATTEMPTS * _INGEST_INTERVAL_S}s — trace_context was not honoured"
    )


def _run_observed_pipeline() -> tuple[str, str]:
    """Run a real scripted pipeline with observe=True; return (run_id, trace_id)
    as read off the LOG LINES — the same surface a caller debugging a run has."""
    mod = types.ModuleType("test_observe_trace_live_mod")

    @node(mode="scripted", outputs=RawText)
    def fetch() -> RawText:
        return RawText(text="live-trace-check")

    @node(mode="scripted", outputs=Claims)
    def gen(fetch: RawText) -> Claims:
        return Claims(items=[fetch.text])

    mod.fetch = fetch
    mod.gen = gen
    graph = compile(
        construct_from_module(mod, name="s65y2-live-check"),
        **build_test_compile_kwargs(),
    )

    with capture_logs() as logs:
        neograph.run(graph, input={"node_id": "x"}, observe=True)

    lifecycle = [e for e in logs if e.get("event") in {"node_start", "node_complete"}]
    assert lifecycle, "expected node lifecycle log lines"
    trace_ids = {e.get("trace_id") for e in lifecycle}
    run_ids = {e.get("run_id") for e in lifecycle}
    assert len(trace_ids) == 1 and None not in trace_ids, f"unstable/absent trace_id: {trace_ids}"
    assert len(run_ids) == 1 and None not in run_ids, f"unstable/absent run_id: {run_ids}"
    return run_ids.pop(), trace_ids.pop()


class TestLangfuseRecordsTheDerivedTraceId:
    def test_trace_is_retrievable_under_the_derived_id(self):
        run_id, trace_id = _run_observed_pipeline()

        from langfuse import Langfuse

        assert trace_id == Langfuse.create_trace_id(seed=run_id), (
            "the logged trace_id is not the derivation of the logged run_id — "
            "the join a caller would compute offline is broken"
        )

        body = _await_trace(trace_id)
        assert body["id"] == trace_id, "Langfuse stored the trace under a DIFFERENT id"
        assert body.get("observations"), "trace landed but carries no observations"

    def test_the_run_id_is_not_itself_a_trace_id(self):
        """The control — and the original bug. Before this fix the run_id was the
        only thing a caller had, and it 404s. If this ever returns 200 the two
        identity spaces have collided and the derivation is not doing its job."""
        run_id, _trace_id = _run_observed_pipeline()
        status, _ = _get_trace(run_id)
        assert status == 404, f"expected the raw run_id to name no trace, got HTTP {status}"
