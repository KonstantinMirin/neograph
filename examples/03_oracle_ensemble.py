"""Example 3: Oracle Ensemble — N parallel generators + merge.

Scenario: Decompose a complex requirement from multiple angles. Three
generators independently break down the same input, producing different
claim lists. A merge function combines and deduplicates them into a
single consensus list.

This demonstrates the Oracle modifier via @node kwargs: run a node N
times in parallel, then merge results. The merge can be:
1. Scripted — deterministic function (merge_fn)
2. LLM-powered — judge prompt (merge_prompt)
3. LLM with hooks — merge_prompt + pre_process/post_process/fallback

Run:
    python examples/03_oracle_ensemble.py
"""

from __future__ import annotations

import sys
import threading

from pydantic import BaseModel

from neograph import compile, construct_from_module, node, run

# ── Schemas ──────────────────────────────────────────────────────────────

class Topic(BaseModel, frozen=True):
    text: str

class Claims(BaseModel, frozen=True):
    items: list[str]


# ── Generator: each Oracle instance produces a different decomposition ──
# A thread-safe counter rotates through perspectives so each of the 3
# parallel generators returns a different claim list.

_perspectives = [
    ["security: must authenticate", "security: must encrypt"],
    ["reliability: must handle failures", "reliability: must log errors"],
    ["performance: must respond in 200ms", "security: must authenticate"],
]

_gen_counter_lock = threading.Lock()
_gen_counter = [0]


# ── Merge: combine and deduplicate claims from all generators ────────────

def merge_claims(variants, config):
    """Merge N claim lists into one deduplicated list."""
    seen = set()
    merged = []
    for variant in variants:
        for claim in variant.items:
            if claim not in seen:
                seen.add(claim)
                merged.append(claim)
    return Claims(items=merged)


# ── Build pipeline ───────────────────────────────────────────────────────
# @node with ensemble_n=3 and merge_fn= attaches the Oracle modifier.

@node(outputs=Claims, ensemble_n=3, merge_fn="merge_claims")
def decompose() -> Claims:
    with _gen_counter_lock:
        idx = _gen_counter[0] % len(_perspectives)
        _gen_counter[0] += 1
    return Claims(items=_perspectives[idx])


pipeline = construct_from_module(sys.modules[__name__], name="oracle-demo")


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _gen_counter[0] = 0  # reset for clean run
    graph = compile(pipeline, scripted={"merge_claims": merge_claims})
    result = run(graph, input={"node_id": "REQ-001"})

    merged = result["decompose"]
    print("=== Same-model ensemble (3 generators) ===")
    print(f"{len(merged.items)} unique claims:")
    for claim in merged.items:
        print(f"  - {claim}")
    # "security: must authenticate" appears in gen-0 and gen-2 but is deduplicated
    print()

    # ── Multi-model ensemble ────────────────────────────────────────────
    # Run the same task on different models and merge. Each generator
    # gets a different model tier via round-robin assignment.
    from neograph import Construct, Node, Oracle

    seen_models = []

    def multi_model_gen(input_data, config):
        model = config.get("configurable", {}).get("_oracle_model", "default")
        seen_models.append(model)
        return Claims(items=[f"claim-from-{model}"])

    def pick_best(variants, config):
        # In a real pipeline, you'd score and pick. Here we merge all.
        all_items = []
        for v in variants:
            all_items.extend(v.items)
        return Claims(items=all_items)

    gen_node = (
        Node.scripted("multi-gen", fn="multi_model_gen", outputs=Claims)
        | Oracle(models=["reason", "fast", "creative"], merge_fn="pick_best")
    )
    multi_pipeline = Construct("multi-model", nodes=[gen_node])
    multi_graph = compile(
        multi_pipeline,
        scripted={"multi_model_gen": multi_model_gen, "pick_best": pick_best},
    )
    multi_result = run(multi_graph, input={"node_id": "REQ-002"})

    multi_merged = multi_result["multi_gen"]
    print(f"=== Multi-model ensemble (models={['reason', 'fast', 'creative']}) ===")
    print(f"Models used: {seen_models}")
    print("Merged claims:")
    for claim in multi_merged.items:
        print(f"  - {claim}")
    print()

    # ── Merge hooks: pre_process, post_process, fallback ────────────────
    # When using merge_prompt (LLM judge), optional hooks let you
    # customize the merge without dropping to @merge_fn.
    #
    # This demo uses merge_fallback to show the pattern without requiring
    # an LLM key — the fallback fires because no LLM is configured.

    # A prompt compiler that just returns the template. No real LLM —
    # invoke_structured will fail, triggering our fallback. These are
    # passed to compile() below as llm_factory/prompt_compiler kwargs.

    def tag_variants(variants: list) -> dict:
        """Pre-process: tag each variant with a generator ID."""
        tagged = [
            {"gen_id": f"gen-{i}", "claims": v.items}
            for i, v in enumerate(variants)
        ]
        return {"tagged_claims": tagged}

    def validate_merge(result, variants: list):
        """Post-process: ensure every input claim appears in the output."""
        all_input = {c for v in variants for c in v.items}
        missing = all_input - set(result.items)
        if missing:
            return Claims(items=result.items + sorted(missing))
        return result

    def deterministic_fallback(variants: list, error: Exception) -> Claims:
        """Fallback: on LLM failure, deduplicate deterministically."""
        seen = set()
        merged = []
        for v in variants:
            for claim in v.items:
                if claim not in seen:
                    seen.add(claim)
                    merged.append(claim)
        return Claims(items=merged)

    def hooks_gen(input_data, config):
        with _gen_counter_lock:
            idx = _gen_counter[0] % len(_perspectives)
            _gen_counter[0] += 1
        return Claims(items=_perspectives[idx])

    _gen_counter[0] = 0  # reset
    hooks_node = (
        Node.scripted("decompose-hooks", fn="hooks_gen", outputs=Claims)
        | Oracle(
            n=3,
            merge_prompt="Pick the best decomposition: ${tagged_claims}",
            merge_pre_process=tag_variants,
            merge_post_process=validate_merge,
            merge_fallback=deterministic_fallback,
        )
    )
    hooks_pipeline = Construct("hooks-demo", nodes=[hooks_node])
    hooks_graph = compile(
        hooks_pipeline,
        llm_factory=lambda tier: None,  # type: ignore[arg-type]
        prompt_compiler=lambda tmpl, data, **kw: [{"role": "user", "content": tmpl}],
        scripted={"hooks_gen": hooks_gen},
    )
    hooks_result = run(hooks_graph, input={"node_id": "REQ-003"})

    hooks_merged = hooks_result["decompose_hooks"]
    print("=== Merge hooks (fallback fires — no LLM configured) ===")
    print(f"Fallback produced {len(hooks_merged.items)} claims:")
    for claim in hooks_merged.items:
        print(f"  - {claim}")
