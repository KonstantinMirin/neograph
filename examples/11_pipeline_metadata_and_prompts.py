"""Example 11: Pipeline Metadata + Rich Prompt Compilation.

Scenario: A requirement analysis pipeline where every node needs:
  - node_id: which requirement is being analyzed
  - project_root: where to find source files for context
  - shared resources: rate limiter, cost tracker (passed by consumer)

And prompt compilation needs:
  - The template name (from Node.prompt)
  - The typed input data from the previous node
  - The node's identity (from Node.name)
  - Pipeline metadata (node_id, project_root from config)

This shows the production pattern: run() injects input into
config["configurable"], and the prompt_compiler receives full context.
No boilerplate in node functions — just read from config.

Run:
    python examples/11_pipeline_metadata_and_prompts.py
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import Construct, Node, compile, configure_llm, register_scripted, run


# ── Schemas ──────────────────────────────────────────────────────────────

class Claims(BaseModel, frozen=True):
    items: list[str]

class Report(BaseModel, frozen=True):
    text: str


# ══════════════════════════════════════════════════════════════════════════
# SHARED RESOURCES — consumer's infrastructure passed through config
# ══════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Example shared resource — consumer owns this, NeoGraph doesn't."""

    def __init__(self, max_rpm: int):
        self.max_rpm = max_rpm
        self.calls = 0

    def call(self, fn, *args, **kwargs):
        self.calls += 1
        return fn(*args, **kwargs)


# ══════════════════════════════════════════════════════════════════════════
# PROMPT COMPILER — receives full context from NeoGraph
#
# Full signature:
#   (template, data, *, node_name=None, config=None,
#                       output_model=None, llm_config=None) → messages
#
# config["configurable"] contains:
#   - Everything from run(input={...})  → node_id, project_root
#   - Everything from run(config={...}) → rate_limiter, custom fields
#
# output_model is the Pydantic class the node will parse into — use it to
# inject JSON schema for json_mode / text output strategies (models without
# native structured output support like DeepSeek).
#
# llm_config is the Node's llm_config dict — check output_strategy here.
# ══════════════════════════════════════════════════════════════════════════

import json


def my_prompt_compiler(template, data, *, node_name=None, config=None,
                       output_model=None, llm_config=None):
    """Production prompt compiler with full context access."""
    configurable = (config or {}).get("configurable", {})

    node_id = configurable.get("node_id", "unknown")
    project_root = configurable.get("project_root", ".")
    strategy = (llm_config or {}).get("output_strategy", "structured")

    print(f"  [prompt] template={template}, node={node_name}, "
          f"node_id={node_id}, strategy={strategy}")

    # Base prompt per template
    if template == "decompose":
        prompt = (
            f"Decompose requirement {node_id} from {project_root} into claims. "
            f"Previous analysis: {data}"
        )
    elif template == "summarize":
        prompt = f"Summarize: {data}"
    else:
        prompt = str(data)

    messages = [{"role": "user", "content": prompt}]

    # For json_mode / text: inject the output schema so the LLM knows what
    # shape to return. The framework will parse the JSON from the response.
    if strategy in ("json_mode", "text") and output_model is not None:
        schema = json.dumps(output_model.model_json_schema(), indent=2)
        messages.append({
            "role": "user",
            "content": f"Return ONLY a valid JSON object matching this schema:\n{schema}",
        })

    return messages


# ══════════════════════════════════════════════════════════════════════════
# LLM FACTORY — also receives node context
# ══════════════════════════════════════════════════════════════════════════

class FakeLLM:
    def with_structured_output(self, model, **kwargs):
        self._model = model
        return self

    def invoke(self, messages, **kwargs):
        return self._model(items=["claim-1", "claim-2"])


def my_llm_factory(tier, node_name=None, llm_config=None):
    print(f"  [factory] tier={tier}, node={node_name}, config={llm_config}")
    return FakeLLM()


configure_llm(llm_factory=my_llm_factory, prompt_compiler=my_prompt_compiler)


# ══════════════════════════════════════════════════════════════════════════
# SCRIPTED NODE — accesses pipeline metadata from config
# ══════════════════════════════════════════════════════════════════════════

def build_report(input_data, config):
    """Node function that needs pipeline metadata — reads from config."""
    configurable = config.get("configurable", {})

    # These are available because run() injected them
    node_id = configurable.get("node_id")
    project_root = configurable.get("project_root")

    # Consumer's shared resources are also available
    rate_limiter = configurable.get("rate_limiter")
    if rate_limiter:
        print(f"  [report] rate_limiter has made {rate_limiter.calls} calls")

    return Report(text=f"Report for {node_id} at {project_root}: {input_data.items}")


register_scripted("build_report", build_report)


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════

decompose = Node(
    name="decompose",
    mode="produce",
    outputs=Claims,
    model="fast",
    prompt="decompose",
)

report = Node.scripted("report", fn="build_report", inputs=Claims, outputs=Report)

pipeline = Construct("metadata-demo", nodes=[decompose, report])


# ══════════════════════════════════════════════════════════════════════════
# RUN — consumer passes everything through input + config
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    limiter = RateLimiter(max_rpm=60)

    graph = compile(pipeline)

    print("Running pipeline:\n")
    result = run(
        graph,
        # input fields → state AND config["configurable"]
        input={"node_id": "BR-RW-042", "project_root": "/my/project"},
        # extra config → config["configurable"]
        config={"configurable": {"rate_limiter": limiter}},
    )

    print(f"\nResult: {result['report'].text}")
    print(f"\nAll config accessible: node_id, project_root, rate_limiter — no boilerplate")
