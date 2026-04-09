"""Lead Research Pipeline -- neograph mini-project.

Loads a batch of leads, researches each in parallel (website + news),
synthesizes findings, then qualifies and ranks them.

    OPENROUTER_API_KEY=sk-... python examples/lead-research/pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from neograph import (
    Construct,
    Each,
    compile,
    configure_llm,
    construct_from_functions,
    node,
    run,
)

from schemas import (
    CompanyProfile,
    Lead,
    LeadBatch,
    LeadReport,
    NewsList,
    NewsItem,
    QualifiedLeads,
)


# =============================================================================
# Data + LLM setup
# =============================================================================

def _prompt(name: str) -> str:
    return (_HERE / "prompts" / f"{name}.md").read_text()


MODELS = {
    "reason": "anthropic/claude-sonnet-4",
    "fast": "google/gemini-2.0-flash-001",
}


def _llm_factory(tier: str, *, node_name: str = "", llm_config: dict | None = None):
    from langchain_openai import ChatOpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY to run this example.")
    return ChatOpenAI(
        model=MODELS.get(tier, MODELS["fast"]),
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=(llm_config or {}).get("temperature", 0.7),
        max_tokens=(llm_config or {}).get("max_tokens", 4000),
    )


configure_llm(
    llm_factory=_llm_factory,
    prompt_compiler=lambda template, data: [{"role": "user", "content": template}],
)


# =============================================================================
# Pipeline nodes
# =============================================================================

@node(outputs=LeadBatch)
def load_leads() -> LeadBatch:
    with open(_HERE / "data" / "leads.json") as f:
        raw = json.load(f)
    return LeadBatch(items=[Lead(**entry) for entry in raw])


@node(outputs=CompanyProfile)
def scrape_website(load_leads: Lead) -> CompanyProfile:
    """Fetch and parse the company website. Scripted with mock data."""
    profiles = {
        "Arcline Systems": CompanyProfile(
            company_name="Arcline Systems",
            website_url="https://arcline.io",
            tagline="Developer tools for the AI-native stack",
            products=["Arcline CLI", "Arcline Cloud", "Arcline SDK"],
            recent_updates=[
                "Launched Arcline Cloud GA in Q1 2026",
                "Raised Series A ($18M) led by Gradient Ventures",
                "Open-sourced Arcline SDK on GitHub",
            ],
            tech_stack_hints=["Python", "Rust", "Kubernetes", "gRPC"],
        ),
        "Greenfield Analytics": CompanyProfile(
            company_name="Greenfield Analytics",
            website_url="https://greenfield-analytics.com",
            tagline="Carbon accounting that scales with your supply chain",
            products=["Greenfield Platform", "Emissions API", "Scope 3 Tracker"],
            recent_updates=[
                "Partnership with EU carbon registry for automated reporting",
                "Expanded to 12 countries in APAC",
                "Hired VP of Engineering from Stripe",
            ],
            tech_stack_hints=["Python", "PostgreSQL", "dbt", "AWS"],
        ),
        "MedBridge Health": CompanyProfile(
            company_name="MedBridge Health",
            website_url="https://medbridgehealth.com",
            tagline="Connecting patients to clinical trials faster",
            products=["TrialMatch", "PatientHub", "MedBridge API"],
            recent_updates=[
                "FDA clearance for AI-powered trial matching",
                "Series C ($45M) closed in March 2026",
                "Launched integration with Epic EHR",
            ],
            tech_stack_hints=["Python", "React", "FHIR", "GCP"],
        ),
    }
    return profiles.get(load_leads.company, CompanyProfile(
        company_name=load_leads.company,
        website_url=load_leads.website,
    ))


@node(outputs=NewsList)
def search_news(load_leads: Lead) -> NewsList:
    """Search for recent news about the company. Scripted with mock data."""
    news = {
        "Arcline Systems": NewsList(items=[
            NewsItem(
                headline="Arcline Systems raises $18M to build AI-native developer tools",
                source="TechCrunch",
                date="2026-02-15",
                summary="Series A funding to expand Arcline Cloud and hire 30 engineers.",
            ),
            NewsItem(
                headline="Arcline open-sources its SDK, bets on community adoption",
                source="The New Stack",
                date="2026-03-01",
                summary="Arcline SDK now Apache-2.0 licensed. CEO says open source is the moat.",
            ),
        ]),
        "Greenfield Analytics": NewsList(items=[
            NewsItem(
                headline="EU carbon registry partners with Greenfield for automated scope reporting",
                source="Reuters",
                date="2026-01-20",
                summary="Greenfield selected as technology partner for new EU carbon compliance framework.",
            ),
            NewsItem(
                headline="Greenfield Analytics expands to 12 APAC markets",
                source="Bloomberg Green",
                date="2026-03-10",
                summary="Climate tech startup targets rapid supply-chain decarbonization in Asia.",
            ),
        ]),
        "MedBridge Health": NewsList(items=[
            NewsItem(
                headline="MedBridge Health gets FDA clearance for AI trial matching",
                source="STAT News",
                date="2026-02-28",
                summary="First AI-powered clinical trial matching tool to receive FDA De Novo clearance.",
            ),
            NewsItem(
                headline="MedBridge closes $45M Series C to scale clinical trial platform",
                source="Fierce Biotech",
                date="2026-03-15",
                summary="Funding led by a]16z Bio to expand TrialMatch to oncology and rare diseases.",
            ),
        ]),
    }
    return news.get(load_leads.company, NewsList(items=[]))


@node(
    outputs=LeadReport,
    mode="think",
    model="reason",
    prompt=_prompt("synthesize"),
)
def synthesize(scrape_website: CompanyProfile, search_news: NewsList) -> LeadReport:
    ...


# -- Research sub-construct: one per lead, fanned out via Each --

research = construct_from_functions(
    "research",
    [scrape_website, search_news, synthesize],
    input=Lead,
    output=LeadReport,
) | Each(over="load_leads.items", key="name")


@node(
    outputs=QualifiedLeads,
    mode="think",
    model="reason",
    prompt=_prompt("qualify"),
)
def qualify(research: list[LeadReport]) -> QualifiedLeads:
    ...


# -- Top-level pipeline --

pipeline = construct_from_functions(
    "lead-research",
    [load_leads, research, qualify],
)


# =============================================================================
# Run
# =============================================================================

def main():
    print("Lead Research Pipeline")
    print("=" * 40)

    graph = compile(pipeline)
    result = run(graph, input={"node_id": "lead-research-batch"})

    qualified = result["qualify"]
    print(f"\nQualified {len(qualified.leads)} leads:\n")
    for ql in qualified.leads:
        print(f"  [{ql.tier.upper()}] {ql.lead_name} @ {ql.company} -- score {ql.score:.2f}")
        print(f"    {ql.reasoning}")
        print(f"    Next step: {ql.recommended_next_step}\n")


if __name__ == "__main__":
    main()
