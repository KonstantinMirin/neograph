# Lead Research Pipeline

## What this does

Takes a list of target companies (from CSV, CRM export, or manual input) and produces qualified, scored leads with personalized outreach drafts — ready for a sales rep to send.

The pipeline replaces the manual process of: open LinkedIn, Google the company, read their blog, check Crunchbase, figure out if they're a fit, write a personalized email, review it, rewrite it, send it. That process takes 20-40 minutes per lead. This pipeline does it in under 2 minutes per lead, in parallel.

## Who uses this

SDRs and AEs at B2B SaaS companies doing outbound. The person running this has a list of 50-200 target accounts and needs to turn them into qualified, personalized outreach within a day.

## The process

### Phase 1: Research (per lead, parallel)

For each company in the input list, gather intelligence from multiple sources simultaneously:

- **Company website**: Scrape the homepage + about page + blog. Extract what they do, their tech stack signals, team size indicators, recent announcements.
- **News**: Search for recent news (last 90 days). Funding rounds, product launches, leadership changes, partnerships. These are conversation openers.
- **Job postings**: Search for open roles. Hiring patterns reveal priorities (e.g., hiring 5 ML engineers = investing in AI). This is the strongest intent signal.

Each source is independent — they fan out in parallel and results merge into a single company profile.

### Phase 2: Score and Qualify

Take the research profile and score it against an Ideal Customer Profile (ICP):

- **Firmographic fit**: Industry, company size, geography, tech stack match
- **Intent signals**: Hiring patterns, funding recency, tech stack changes
- **Timing signals**: Recent news, leadership changes, expansion indicators

Score 0-100. Threshold: >= 60 is qualified, < 60 is parked for later.

This is where model diversity helps — run the scoring on two different models (e.g., Claude Sonnet + GPT-4o) and take the average. Different models catch different signals.

### Phase 3: Personalize Outreach (qualified leads only)

For each qualified lead, draft a personalized email:

- Reference a specific finding from the research (not generic "I saw your company is growing")
- Connect to the seller's value proposition
- Include a specific call-to-action

Then review it: is it too long? Too generic? Does the personalization feel forced? If the review scores below 0.8, revise and re-review. This loop runs up to 3 times.

### Phase 4: Output

Write results to a structured output:
- Qualified leads with scores, research summaries, and outreach drafts
- Disqualified leads with reasons (for future nurturing)
- Statistics: total processed, qualified rate, average score

## Data sources

| Source | API | Free tier | What it provides |
|--------|-----|-----------|------------------|
| Company website | httpx + readability-lxml | Unlimited | Product info, tech signals, team size |
| News search | Serper API (serper.dev) | 2,500 free queries | Recent funding, launches, press |
| Job postings | Serper API (Google search) | Same quota | Hiring patterns, priorities |

Alternative: Tavily API (tavily.com) — 1,000 free searches/month, returns clean extracted content instead of raw search results.

## Input format

CSV with columns: `company_name`, `website`, `industry` (optional), `notes` (optional)

```csv
company_name,website,industry,notes
Acme Corp,https://acme.com,SaaS,Saw their Series B announcement
Globex,https://globex.io,FinTech,Met CEO at conference
```

## Output format

JSON with full research + score + outreach per lead. Also a summary CSV for quick review.

## What makes this a good neograph example

- **Each**: fan-out over the lead list (N leads processed in parallel)
- **Each (nested)**: fan-out over data sources per lead (website + news + jobs in parallel)
- **Oracle models=**: multi-model scoring for diversity
- **Loop**: outreach draft -> review -> revise cycle
- **Sub-construct**: research phase is an isolated sub-pipeline (input: company name, output: company profile)
- **Spec-driven**: the pipeline is defined in YAML — an LLM could generate it from a natural language description of the ICP
- **Compile-time validation**: change the Score model fields and see the error before running
