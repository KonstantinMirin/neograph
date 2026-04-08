# RFP Response Generator

## What this does

Takes an RFP document (PDF or text) and produces a structured response — section by section — with each section drafted, reviewed for completeness, and revised until it meets a quality bar. The output is a response document ready for human final review.

RFP responses are one of the most time-consuming tasks in enterprise sales. A typical RFP has 15-40 sections, each requiring: read the requirement, find the relevant capability in your product, draft a response that matches the requirement language, include evidence/references, and make sure you didn't miss any sub-requirements. Teams spend 2-4 weeks on this. This pipeline produces a first draft in minutes.

## Who uses this

Sales engineers, proposal managers, and pre-sales teams at companies that respond to RFPs regularly (government contractors, enterprise SaaS, consulting firms). The person has a knowledge base of past responses and product documentation, and needs to produce a first draft that's 80% done — the human reviews and polishes the last 20%.

## The process

### Phase 1: Parse the RFP

Extract the RFP structure from the document:
- Sections and sub-sections with their numbering
- Requirements within each section (often bulleted or numbered)
- Evaluation criteria and weightings (if present)
- Submission requirements (format, page limits, deadlines)
- Key terms and definitions

This is an agent step — the LLM reads the document with a tool that extracts text from PDF, then structures it into a typed schema.

### Phase 2: Classify and prioritize

For each section, determine:
- **Category**: Technical capability, pricing, team/staffing, past performance, compliance, management approach
- **Complexity**: Simple (yes/no capability statement), moderate (requires explanation), complex (requires detailed solution design)
- **Weight**: How much does this section matter for evaluation? (from criteria if available, inferred if not)
- **Knowledge base coverage**: Do we have existing content for this? (search past responses)

### Phase 3: Draft sections (parallel, with review loop)

For each section (fan-out):
1. **Search knowledge base** for relevant past responses, product docs, case studies
2. **Draft the section** using the requirement text + knowledge base results + company context
3. **Review the draft** against:
   - Does it address every sub-requirement? (completeness)
   - Does it use the RFP's own language? (compliance)
   - Is it the right length? (not too brief, not padding)
   - Does it include evidence/examples? (credibility)
4. **Score**: 0-1 on completeness, compliance, quality
5. If score < 0.8, **revise** with specific feedback from the review → loop back to review

Each section is independent — they all run in parallel. The loop runs up to 3 times per section.

### Phase 4: Assemble

Combine all sections in order. Add:
- Cover letter (generated from the assembled content)
- Executive summary (generated from section highlights)
- Table of contents
- Compliance matrix (requirement → section reference)

### Phase 5: Output

Write the assembled document as:
- Markdown (for review in any editor)
- DOCX (using python-docx, for formal submission)
- JSON (structured data for further processing)

## Data sources

| Source | How | What it provides |
|--------|-----|------------------|
| RFP document | PyPDF2 / pdfplumber | Raw requirement text |
| Knowledge base | Local markdown/text files | Past responses, product docs, case studies |
| Company profile | Static config file | Company name, capabilities summary, key differentiators |

The knowledge base is a directory of markdown files. No vector database needed — the LLM searches by reading file names and content directly (tool: read_file, list_files). Simple and reproducible.

## Input format

- RFP document: PDF or text file
- Knowledge base: directory of markdown files
- Company profile: YAML config

## Output format

Markdown + DOCX + JSON. The markdown is the primary output for review.

## What makes this a good neograph example

- **Each**: fan-out over RFP sections (15-40 sections in parallel)
- **Loop**: draft → review → revise per section (quality gate at 0.8)
- **Sub-constructs**: each section's draft-review-revise cycle is isolated
- **Tools**: PDF extraction, knowledge base search
- **Spec-driven**: the pipeline structure could be generated from the RFP's own table of contents
- **Real-world value**: this is a pipeline people actually pay for ($50k+ RFP response tools exist)
