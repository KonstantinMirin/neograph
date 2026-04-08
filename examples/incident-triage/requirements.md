# Incident Response Triage

## What this does

Takes an alert (from PagerDuty, Datadog, Grafana, or a raw JSON payload) and produces a structured Root Cause Analysis (RCA) draft — with correlated evidence from multiple data sources, a timeline of events, and recommended remediation steps.

On-call engineers get woken up at 3 AM by an alert that says "HTTP 500 rate > 5% on /api/checkout". They then spend 30-60 minutes manually checking: logs, metrics, recent deployments, related alerts, upstream dependencies. This pipeline does the investigation in parallel and presents a structured hypothesis with evidence — the engineer validates instead of searching from scratch.

## Who uses this

On-call engineers, SRE teams, DevOps. The person is stressed, sleep-deprived, and needs to quickly determine: is this a real incident or a false alarm? If real, what's the root cause and what do I do?

## The process

### Phase 1: Parse the alert

Extract structured information from the alert payload:
- **What**: which service/endpoint/metric is affected
- **When**: alert trigger time, duration
- **Severity**: P1-P4
- **Context**: any metadata (environment, region, customer segment)
- **Related alerts**: other alerts that fired in the same time window

### Phase 2: Gather evidence (parallel fan-out)

Query multiple data sources simultaneously:

- **Application logs**: Search for errors, exceptions, stack traces in the affected service around the alert time window. Look for patterns (same error repeated, new error type, error rate spike).
- **Infrastructure metrics**: CPU, memory, disk, network for the affected hosts. Check for resource exhaustion, sudden spikes, flatlines (process died).
- **Deployment history**: What was deployed in the last 2 hours? Any config changes? Feature flag toggles? Database migrations?
- **Dependency health**: Are upstream/downstream services healthy? Any related alerts from dependencies?

Each source is independent — fan-out in parallel. Each returns structured evidence (timestamped findings with severity).

### Phase 3: Correlate and hypothesize (loop)

Take all gathered evidence and:
1. **Build a timeline**: order all events chronologically
2. **Form a hypothesis**: "deployment X at 02:45 introduced a null pointer in the checkout handler, causing 500s starting at 02:47"
3. **Test the hypothesis**: does the evidence support it? Are there contradictions? Missing data?
4. **Refine**: if the hypothesis has gaps, identify what additional evidence would confirm or reject it

Loop until confidence >= 0.9 or max 3 iterations. Each iteration can request additional evidence queries (but in this example, we work with what we gathered in Phase 2).

### Phase 4: Draft RCA

Produce a structured RCA using two models (Oracle) for different perspectives:
- **Timeline**: ordered list of events with timestamps
- **Root cause**: the specific change/failure that caused the incident
- **Impact**: what was affected, for how long, how many users
- **Remediation**: immediate fix + long-term prevention
- **Action items**: who needs to do what, by when

### Phase 5: Format report

Produce a markdown report suitable for:
- Pasting into a Slack incident channel (summary)
- Attaching to the incident ticket (full RCA)
- Presenting at the post-mortem (structured narrative)

## Data sources

For the example, we simulate the data sources with local JSON files. In production, these would be real API calls:

| Source | Example simulation | Production API |
|--------|-------------------|----------------|
| Application logs | `data/logs.json` | Datadog, Elasticsearch, CloudWatch |
| Metrics | `data/metrics.json` | Prometheus, Grafana, Datadog |
| Deployments | `data/deploys.json` | GitHub Actions, ArgoCD, internal deploy API |
| Dependencies | `data/deps.json` | Status page APIs, internal health checks |

The sample data tells a story: a deployment at 02:45 introduced a bug, errors started at 02:47, metrics show memory spike at 02:46 (the new code leaks), a dependency (payment-service) is healthy (ruling it out).

## Input format

Alert JSON:
```json
{
  "alert_id": "INC-2024-0342",
  "service": "checkout-api",
  "metric": "http_5xx_rate",
  "threshold": 0.05,
  "current_value": 0.12,
  "triggered_at": "2024-03-15T02:47:00Z",
  "environment": "production",
  "region": "us-east-1"
}
```

## Output format

Markdown RCA report + JSON structured data.

## What makes this a good neograph example

- **Each**: fan-out over data sources (logs + metrics + deploys + deps in parallel)
- **Loop**: hypothesis refinement cycle (correlate → test → refine)
- **Oracle models=**: multi-model RCA drafting (different models see different patterns)
- **Tools**: log search, metrics query, deployment lookup (simulated with local JSON, swappable for real APIs)
- **Time-critical**: demonstrates observability (structlog at every node, timing data)
- **The story is compelling**: everyone who's been on-call understands the pain
- **Compile-time validation**: the Evidence schema must match between gatherer and correlator — type mismatch caught before the 3 AM incident
