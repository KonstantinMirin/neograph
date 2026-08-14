"""``export_conformance()`` -- the composer (neograph-ftnxl.1).

The ONLY function that imports BOTH ``_agent_spec_conformance`` (the IR/Flow
predicates) and ``_agent_spec`` (the exporter) -- one-way, exporter ->
classifier, so no import cycle exists: this module is never imported by
either of the two it composes.

``strict=`` lives HERE, not on ``to_agent_spec``. Measured at HEAD: every
GREEN matrix cell hits ``neograph-qtfof.9`` (the outermost EndNode gap fires
unconditionally for a top-level Construct), so PORTABLE is the empty set
today. A ``to_agent_spec(strict=True)`` default would raise for 100% of the
223 existing call sites across the test suite and examples 29/30/31, none of
which pass an opinion today. Keeping ``to_agent_spec``'s signature untouched
gives this feature zero blast radius; ``export_conformance`` is purely
additive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neograph._agent_spec import to_agent_spec
from neograph._agent_spec_conformance import (
    CONFORMANCE_PREDICATE_META,
    ConformanceFinding,
    ConformanceReport,
    ConformanceTier,
    _flow_findings,
    _structural_findings,
    _worst_tier,
)
from neograph.errors import ConfigurationError

if TYPE_CHECKING:
    from neograph.construct import Construct


def export_conformance(construct: Construct, *, strict: bool = True) -> ConformanceReport:
    """Classify how portable ``construct``'s Agent Spec export is to a genuine
    third-party runtime.

    Never re-derives the exporter's own raise-list: NOT_EXPORTABLE is decided
    by ATTEMPTING ``to_agent_spec(construct)`` and catching
    ``ConfigurationError`` -- the exporter stays the one authority on what it
    cannot represent at all. On success, the lowered artifact (a ``Flow`` or a
    ``Swarm``) is inspected for the tracked NEOGRAPH_ROUND_TRIP_ONLY gaps
    (neograph-qtfof.6/.7/.8/.9). PORTABLE is never self-certified from
    "no predicate fired" -- it is calibrated against the empirical execution
    tier (``tests/test_agent_spec_conformance.py``), which is how this ticket
    avoids relocating its own "didn't raise == portable" disease one layer out.

    ``strict=True`` (the default for this NEW function only -- see module
    docstring for why ``to_agent_spec`` itself keeps a different default)
    raises ``ConfigurationError`` naming the offending predicate(s) + their
    tracked bead(s) when the verdict is below PORTABLE. Pass ``strict=False``
    to always get the report back as data, never a raise.
    """
    findings: list[ConformanceFinding] = list(_structural_findings(construct))

    try:
        flow = to_agent_spec(construct)
    except ConfigurationError as exc:
        findings.append(ConformanceFinding("exporter_rejected", None, str(exc)))
        report = ConformanceReport(ConformanceTier.NOT_EXPORTABLE, tuple(findings))
    else:
        findings.extend(_flow_findings(flow))
        report = ConformanceReport(_worst_tier(findings), tuple(findings))

    if strict and report.verdict is not ConformanceTier.PORTABLE:
        offenders = [f for f in report.findings if CONFORMANCE_PREDICATE_META[f.predicate].tier == report.verdict]
        summary = "; ".join(f"{f.predicate} ({CONFORMANCE_PREDICATE_META[f.predicate].bead})" for f in offenders)
        raise ConfigurationError.build(
            f"construct {construct.name!r} is not PORTABLE to a third-party Agent Spec runtime",
            expected="PORTABLE",
            found=f"{report.verdict.value}: {summary}" if summary else report.verdict.value,
            hint="call export_conformance(construct, strict=False) to inspect the full report without raising",
            construct=construct.name,
        )
    return report
