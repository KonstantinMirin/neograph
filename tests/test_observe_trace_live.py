"""LIVE verification of neograph-s65y2 sub-claim (c) — Langfuse actually records
the trace under the id neograph derived from ``run_id``.

The offline suite (``test_observe_trace_correlation.py``) proves the in-process
half: the derivation, that the handler receives it, and that the field lands on
every node line. It CANNOT prove that Langfuse honours ``trace_context`` — that
is an external side effect. This file closes that gap against the real API.

RUN IT (keys required; mirrors the MCP e2e harness convention — "no-network !=
no-run", the credentials are the extra here):

    set -a && . .env && set +a
    uv run --extra langfuse pytest tests/test_observe_trace_live.py

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

_KEYS_PRESENT = bool(os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY"))

# The release gate sets NEOGRAPH_REQUIRE_LIVE=1. In that context an absent-keys
# SKIP is the exact failure mode this file exists to prevent: `make quality`
# reported green on merged main for 0.7.4 while these two tests silently skipped,
# and a flaky live assertion rode the tag out. A skip that reads as coverage is
# the seam; so when live is REQUIRED, missing credentials fail collection loudly
# rather than quietly subtracting two tests from the count.
if os.environ.get("NEOGRAPH_REQUIRE_LIVE") and not _KEYS_PRESENT:
    raise RuntimeError(
        "NEOGRAPH_REQUIRE_LIVE=1 but LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY are unset. "
        "The release gate must not pass on a silent skip. Load the credentials first:\n"
        "    set -a && . .env && set +a && make release-gate"
    )

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _KEYS_PRESENT,
        reason="live Langfuse keys absent — see this module's docstring for the run command",
    ),
]

# DEADLINE-based, not attempt-based: a slow API multiplies an attempt budget into
# an unbounded wall time (12 attempts x a 120s timeout = 25 min). A release gate
# must have a predictable ceiling, so the poll runs until a fixed deadline.
_INGEST_DEADLINE_S = 180
_INGEST_INTERVAL_S = 5
# Generous per request: cloud Langfuse has been seen at 100s for a single GET
# when throttled after repeated runs.
_REQUEST_TIMEOUT_S = 60


def _auth_header() -> str:
    raw = f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
    return "Basic " + base64.b64encode(raw).decode()


def _api_get(path: str) -> tuple[int, dict | None]:
    """GET a Langfuse API path. Returns (status, body); status 0 = unreachable.

    A transient timeout or connection error must NOT fail the test: the API has
    been observed taking 100s under load (repeated release-gate runs throttle
    it), and a gate that cries wolf on one slow response is worse than no gate --
    it trains you to ignore it. Network trouble is reported as status 0 so the
    caller keeps polling; only the poll budget expiring is a real failure."""
    base = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
    req = urllib.request.Request(f"{base}{path}", headers={"Authorization": _auth_header()})
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, None


def _get_trace(trace_id: str) -> tuple[int, dict | None]:
    return _api_get(f"/api/public/traces/{trace_id}")


def _get_observations(trace_id: str) -> list[dict]:
    """Spans belonging to *trace_id*, via the dedicated observations API.

    NOT ``trace["observations"]``: that inline field is deprecated (the trace
    response says so in a ``_deprecation`` notice) and was observed returning
    ``[]`` for a trace whose spans the observations endpoint reports fine. An
    assertion on the inline field therefore fails for a reason that has nothing
    to do with neograph.
    """
    status, body = _api_get(f"/api/public/observations?traceId={trace_id}")
    return (body or {}).get("data", []) if status == 200 else []


def _await_trace_with_nodes(trace_id: str, expected: set[str]) -> tuple[dict, set[str]]:
    """Poll until the trace AND its spans have landed.

    Langfuse ingests asynchronously and in STAGES: the trace record appears
    before its observations do. Polling only for the trace (then reading spans
    once) is racy — it passed twice and failed on the third run with an empty
    span list for a trace that was fully populated seconds later. So the poll
    condition is the thing actually being asserted: the trace exists AND carries
    this run's nodes."""
    body: dict | None = None
    names: set[str] = set()
    deadline = time.monotonic() + _INGEST_DEADLINE_S
    while time.monotonic() < deadline:
        time.sleep(_INGEST_INTERVAL_S)
        status, fetched = _get_trace(trace_id)
        if status != 200 or not fetched:
            continue
        body = fetched
        names = {o.get("name") for o in _get_observations(trace_id)}
        if expected <= names:
            return body, names
    if body is None:
        raise AssertionError(
            f"trace {trace_id} never appeared within {_INGEST_DEADLINE_S}s — "
            "trace_context was not honoured (or the API was unreachable throughout)"
        )
    raise AssertionError(
        f"trace {trace_id} landed but its spans never included {sorted(expected)} "
        f"within {_INGEST_DEADLINE_S}s; saw {sorted(names)}"
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

        # Poll for the trace AND its spans; the assertion is that the spans under
        # the derived id are OUR nodes, so it cannot pass on an unrelated trace.
        body, names = _await_trace_with_nodes(trace_id, {"fetch", "gen"})
        assert body["id"] == trace_id, "Langfuse stored the trace under a DIFFERENT id"
        assert {"fetch", "gen"} <= names

    def test_the_run_id_is_not_itself_a_trace_id(self):
        """The control — and the original bug. Before this fix the run_id was the
        only thing a caller had, and it 404s. If this ever returns 200 the two
        identity spaces have collided and the derivation is not doing its job."""
        run_id, _trace_id = _run_observed_pipeline()
        deadline = time.monotonic() + _INGEST_DEADLINE_S
        while time.monotonic() < deadline:
            status, _ = _get_trace(run_id)
            if status != 0:  # 0 = unreachable; retry rather than misread it as "not 404"
                assert status == 404, f"expected the raw run_id to name no trace, got HTTP {status}"
                return
            time.sleep(_INGEST_INTERVAL_S)
        pytest.fail(f"Langfuse API unreachable for {_INGEST_DEADLINE_S}s — control not evaluated")
