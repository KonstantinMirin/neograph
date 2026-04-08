# Lead Outreach Email Sequence

## What this does

Takes a lead profile (LinkedIn JSON) and a set of key ideas/value propositions, and produces a sequence of 3-5 outreach emails. Each email goes through:

1. **Ensemble drafting** — multiple models draft subject + body independently
2. **Deterministic quality gate** — scripted checks (no em-dash, correct first name, length limits, forbidden phrases)
3. **LLM evaluation ensemble** — multiple models role-play as the prospect and grade the email (would I open? would I reply? does this feel generic?)
4. **Revision loop** — if grades are below threshold, revise using the evaluation feedback

The output is a ready-to-load email sequence with subject lines, bodies, send timing, and quality scores.

## Who uses this

Founders doing outbound themselves. SDRs at early-stage startups. Anyone who writes cold emails and knows that "Hi {first_name}, I noticed your company..." gets deleted instantly.

## Inputs

### Lead profile (LinkedIn JSON or equivalent)

```json
{
  "first_name": "Sarah",
  "last_name": "Chen",
  "headline": "VP Engineering at Finova | Building real-time payments infrastructure",
  "company": "Finova",
  "company_description": "Series B fintech building real-time payment rails for banks",
  "location": "San Francisco, CA",
  "experience": [
    {"title": "VP Engineering", "company": "Finova", "duration": "2 years"},
    {"title": "Staff Engineer", "company": "Stripe", "duration": "4 years"}
  ],
  "recent_posts": [
    "Excited to share we just hit 99.99% uptime on our core payments API",
    "Hiring senior backend engineers — Rust experience a plus"
  ],
  "skills": ["distributed systems", "Rust", "payments", "API design"]
}
```

### Key ideas (seller's value propositions)

```yaml
product: "neograph"
positioning: "Declarative LLM pipeline compiler — typed, durable, observable"
angles:
  - "Engineering teams waste 70% of agent code on wiring, not logic"
  - "Type mismatches between pipeline stages surface at 3 AM, not at build time"
  - "Every agent framework requires you to manage state manually"
relevant_to_lead:
  - "Finova is hiring backend engineers — they're scaling their engineering team"
  - "Sarah's Stripe background means she values reliability and type safety"
  - "Recent post about 99.99% uptime — she cares about observable, production-grade infra"
```

## The sequence

### Email 1: Cold open (day 0)

The hardest email. Must earn attention in 3 seconds. No pitch — just relevance.

- Reference something specific from their profile or recent activity
- Ask a question or share an insight, don't sell
- Subject line: must be curiosity-provoking, under 50 chars, no clickbait

### Email 2: Value drop (day 3)

Share something genuinely useful — not a product pitch.

- A relevant insight, benchmark, or resource related to their domain
- Connect to the key ideas without being salesy
- "I wrote this because I saw your post about X" energy

### Email 3: Soft ask (day 7)

The ask. Light, low-commitment.

- Reference emails 1-2 (continuity)
- One specific ask: "15 min chat" or "would it be useful if I showed you X?"
- Make it easy to say yes

### Email 4 (optional): Breakup (day 14)

If no reply. Humor or directness.

- "I'll stop emailing after this"
- One final angle or social proof
- Leave the door open

## The pipeline for EACH email

```
                    key_ideas + lead_profile
                            |
                    [draft] x3 models (Oracle ensemble)
                     /       |       \
                  draft_1  draft_2  draft_3
                     \       |       /
                    [merge] pick best subject + body
                            |
                    [quality_gate] deterministic checks
                            |
                        pass / fail
                            |
                    [evaluate] x2 models role-play as prospect
                     /               \
                  eval_reason      eval_fast
                     \               /
                    [merge_eval] aggregate scores
                            |
                    score >= 0.8? ----yes----> done
                            |
                           no
                            |
                    [revise] using eval feedback
                            |
                    loop back to quality_gate (max 3 times)
```

## Deterministic quality gate

Scripted node — no LLM. Checks:

| Check | Rule | Fail example |
|-------|------|-------------|
| First name correct | `body.contains(lead.first_name)` and NOT `body.contains(lead.last_name + ",")` (too formal) | "Dear Ms. Chen," |
| No em-dash | `"—" not in body` | "neograph — the fastest way to..." |
| No exclamation marks | `body.count("!") == 0` | "Would love to chat!" |
| Subject length | `len(subject) <= 50` | "Quick question about your real-time payments infrastructure at Finova" |
| Body length | `50 <= word_count(body) <= 150` | A 300-word essay |
| No forbidden phrases | not contains "I hope this email finds you", "I wanted to reach out", "I came across your profile", "synergy" | Self-explanatory |
| No company name in subject | `lead.company not in subject` | "Finova + neograph" (screams cold email) |
| Has question | `"?" in body` | Pure statement with no engagement hook |
| Personalization present | references at least one specific detail from lead profile | Generic template that works for anyone |

Returns pass/fail + list of violations. On fail, the violations become revision instructions.

## LLM evaluation ensemble

Two models independently role-play as the prospect and grade:

| Dimension | Scale | What it measures |
|-----------|-------|-----------------|
| **Would open** | 0-1 | Subject line effectiveness. Does it earn a click? |
| **Would read** | 0-1 | First sentence hook. Would I read past line 1? |
| **Would reply** | 0-1 | Call-to-action effectiveness + relevance. Is there a reason to respond? |
| **Feels personal** | 0-1 | Does this feel written for ME, or copy-pasted? |
| **Not annoying** | 0-1 | Tone. Would I be irritated receiving this? |

Overall score = weighted average (reply: 0.3, personal: 0.25, open: 0.2, read: 0.15, not_annoying: 0.1).

Threshold: >= 0.8 passes. < 0.8 triggers revision with the specific dimension scores as feedback.

## Output format

```json
{
  "lead": {"name": "Sarah Chen", "company": "Finova"},
  "sequence": [
    {
      "email_number": 1,
      "send_day": 0,
      "subject": "real-time payments uptime",
      "body": "Sarah, saw your post about hitting four nines on...",
      "quality_gate": {"passed": true, "violations": []},
      "evaluation": {
        "would_open": 0.85,
        "would_read": 0.9,
        "would_reply": 0.75,
        "feels_personal": 0.95,
        "not_annoying": 0.9,
        "overall": 0.87
      },
      "iterations": 2,
      "models_used": ["claude-sonnet", "gpt-4o", "claude-haiku"]
    }
  ]
}
```

## What makes this a good neograph example

- **Each**: fan-out over the email sequence (4 emails in parallel, or sequential if later emails reference earlier ones)
- **Oracle models=**: ensemble drafting (3 models) + ensemble evaluation (2 models)
- **Loop**: revise until evaluation score >= 0.8
- **Scripted node**: deterministic quality gate (no LLM needed)
- **Sub-constructs**: each email's draft→gate→eval→revise cycle is isolated
- **Mix of modes**: scripted (quality gate) + think (draft, evaluate) + scripted (merge)
- **Body-as-merge**: the merge function picks the best draft from the ensemble
- **Real and practical**: this is a pipeline people would actually use and pay for
- **Demonstrates the full thesis**: "the agent generates the workflow, the workflow executes the agent" — an LLM could generate this pipeline spec from the ICP description
