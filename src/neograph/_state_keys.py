"""Named constants and builders for `neo_*` state-bus keys.

Single source of truth for framework-internal state-dict field names.

Why: LangGraph's state-bus returns `None` for unknown keys with no error,
so a typo in any read site is a silent runtime miss. Centralizing the
keys here makes every site refer to the same symbol and lets the
structural guard (`TestNeoStateKeysCentralized`) prevent regression.

No code outside this module should reference `neo_*` as a string literal.

Fixed names are class attributes; templated names (parameterized on a
node's `field_name`) are static methods.
"""

from __future__ import annotations

from typing import Any

from neograph.naming import output_field_name


class StateKeys:
    """Named `neo_*` state-bus keys.

    Use the class attributes for fixed names and the static methods for
    templated names. All names produced here include the `neo_` framework
    prefix; values are exactly the strings that previously appeared as
    literals.
    """

    # Framework prefix — distinguishes framework plumbing keys from user
    # node outputs on the state dict.
    FRAMEWORK_PREFIX = "neo_"

    # Schema-aware checkpoint resume.
    SCHEMA_FINGERPRINT = "neo_schema_fingerprint"
    NODE_FINGERPRINTS = "neo_node_fingerprints"

    # Oracle fan-out plumbing.
    ORACLE_GEN_ID = "neo_oracle_gen_id"
    ORACLE_MODEL = "neo_oracle_model"

    # Each modifier per-Send item.
    EACH_ITEM = "neo_each_item"

    # Sub-construct boundary port.
    SUBGRAPH_INPUT = "neo_subgraph_input"

    # Historical leading-underscore framework keys (predate this module). Kept
    # at their original literal values for zero behavior change; centralized
    # here so the StateKeys guard can no longer slip them (MED-06).
    #
    # ISOLATED_INPUT is a state-DICT key: run_isolated() seeds a typed instance
    # under it so _extract_input can find it by type (node.py).
    ISOLATED_INPUT = "_neo_isolated_input"
    # CONFIG_INPUT is a config['configurable'] key (NOT a state-bus key): the
    # runner stashes the run input there for re-injection on resume (runner.py).
    CONFIG_INPUT = "_neo_input"
    # STREAM_CUSTOM is a config['configurable'] key (NOT a state-bus key): the
    # streaming verbs (stream/astream) set it True when the driver is consuming
    # stream_mode='custom', so emit_progress (progress.py) can tell a live
    # progress consumer from a non-streaming driver and warn on the latter
    # instead of vanishing silently (review L1). Never enters state — a config
    # flag, so it cannot touch the schema fingerprint.
    # DOCUMENTED KEEP (neograph-pjqe Item B): this is NOT an engine-duplicating
    # hand-roll — langgraph 1.2.4's get_stream_writer() has no public no-op
    # sentinel (returns a private live closure when no consumer is attached), so
    # writer-presence cannot replace this flag. See docs/design/
    # langgraph-output-schema-research-2026-07-03.md (R4).
    STREAM_CUSTOM = "_neo_stream_custom"
    # DI_INPUTS is a config['configurable'] key (NOT a state-bus key): the LLM
    # dispatch layer stashes the resolved FromInput/FromConfig values (keyed by
    # the node's parameter names) here so the prompt-compilation seam
    # (`_compile_prompt`) can hand them to a prompt_compiler that declares a
    # `di_inputs` param — WITHOUT re-resolving DI or threading a new positional
    # through the `_llm`/`_tool_loop` call chain. Mirrors the `_oracle_model`
    # config-injection pattern (`_inject_oracle_config`). Never enters state — a
    # config-only key, so it cannot touch the schema fingerprint.
    DI_INPUTS = "_neo_di_inputs"
    # RESOURCE_MANIFEST_INJECT is a config['configurable'] key (NOT a state-bus
    # key): _execute._inject_resource_manifest collects every checkpointed
    # resource-manifest channel's ResourceRefs off state and stashes the merged
    # list here so the FROM_RESOURCE(ref=) DI resolver (di.aresolve) can look up a
    # ref by kind WITHOUT threading full state through the scripted-shim signature.
    # Mirrors the DI_INPUTS / _oracle_model config-injection pattern neograph-a5nh.
    # Never enters state — a config-only key, so it cannot touch the fingerprint.
    RESOURCE_MANIFEST_INJECT = "_neo_resource_manifest"
    # ORACLE_MODEL_OVERRIDE is a config['configurable'] key (NOT a state-bus key):
    # _inject_oracle_config copies the per-generator model tier off state
    # (ORACLE_MODEL, the neo_oracle_model STATE channel) into config here so the
    # LLM dispatch layer (_dispatch / _agent_cycle) can pick the effective model
    # WITHOUT threading it through the call chain. Distinct from ORACLE_MODEL: that
    # is the state-bus channel the value ARRIVES on; this is the config side-channel
    # it is FORWARDED on. Mirrors the DI_INPUTS / RESOURCE_MANIFEST_INJECT pattern.
    # Never enters state — a config-only key, so it cannot touch the schema
    # fingerprint. Renamed from the un-prefixed literal "_oracle_model" (CON-01 /
    # neograph-awor) so TestNeoStateKeysCentralized's `_neo_` matcher can see it —
    # config keys never persist, so the rename has no checkpoint/compat impact.
    ORACLE_MODEL_OVERRIDE = "_neo_oracle_model_override"
    # PORTAL_DISPATCH_DEPTH is a config['configurable'] key (NOT a state-bus
    # key): Portal mode-(b) dispatch (route="decide") self-extending flows
    # nest by calling compile_construct(...) + compiled.invoke(...) on a
    # BRAND-NEW sub-flow each level, with fresh initial state -- a state-bus
    # counter would reset to 0 at every nesting level, silently defeating a
    # depth budget. Depth is therefore a LINEAGE property carried ONLY on
    # config, incremented via a copy-not-mutate child config
    # (make_portal_dispatch_fn) before each nested compiled.invoke/ainvoke.
    # Mirrors the DI_INPUTS / RESOURCE_MANIFEST_INJECT / ORACLE_MODEL_OVERRIDE
    # config-injection pattern. Never enters state — cannot touch the schema
    # fingerprint.
    PORTAL_DISPATCH_DEPTH = "_neo_portal_dispatch_depth"
    # RUN_ID is a config['configurable'] key (NOT a state-bus key): a
    # framework-minted per-run correlation id, minted fresh per execution attempt
    # by ``_mint_run_id`` in the pre-engine brains (``_prepare`` / ``_aprepare``),
    # stable across every superstep of one run, and NEVER user-supplied. Two-
    # lifetime-correct BY CONSTRUCTION: a config-only key never enters state, so it
    # cannot touch the schema fingerprint or persist in a checkpoint; resume re-runs
    # ``_prepare`` -> a NEW id (invalidate-on-resume for free). Mirrors the
    # STREAM_CUSTOM / DI_INPUTS config-injection pattern. Surfaced read-only to
    # nodes via config['configurable'] (the node_id/project_root DI-context path).
    RUN_ID = "_neo_run_id"

    # Non-`neo_`-prefixed framework state keys (CON-01). These are DI-context
    # state-bus fields the compiler always adds (node_id, project_root) plus the
    # Operator interrupt channel (human_feedback). They predate the `neo_`
    # convention and stay un-prefixed because FromInput/FromConfig bind them by
    # their bare names; centralized here so every read/write site shares one
    # symbol and the schema-fingerprint exclusion list cannot drift.
    NODE_ID = "node_id"
    PROJECT_ROOT = "project_root"
    HUMAN_FEEDBACK = "human_feedback"

    # Shared prefix for the per-producer resource-manifest channels. A consumer
    # (di / _execute) enumerates state fields by this prefix to collect all refs.
    RESOURCE_MANIFEST_PREFIX = "neo_resource_manifest_"

    @staticmethod
    def loop_count(field_name: str) -> str:
        """Per-loop iteration counter field name."""
        return f"neo_loop_count_{field_name}"

    @staticmethod
    def handoff_hops(field_name: str) -> str:
        """Per-mesh hop-budget counter field name (Portal, design §3.4).

        Keyed off the mesh ENTRY's producer field. Plain ``(int, 0)`` state
        field, incremented by each member's wrapper (T2/T3), like ``loop_count``.
        """
        return f"neo_handoff_hops_{field_name}"

    @staticmethod
    def handoff_payload(field_name: str) -> str:
        """Per-mesh shared channel field name (Portal, design §3.3).

        Keyed off the mesh ENTRY's producer field. Each hop writes its payload
        here so a peer can read it via the reserved ``handoff`` inputs key
        regardless of which member routed to it.
        """
        return f"neo_handoff_{field_name}"

    @staticmethod
    def portal_proposed_target(field_name: str) -> str:
        """Portal+Operator approval gate: the routing target proposed by an
        Operator-guarded mesh member, pending human approval.

        Keyed off the MEMBER's own producer field (each approval-guarded
        member has its own approval node, unlike the mesh-entry-keyed
        handoff_payload/handoff_hops channels). Internal framework field --
        neo_-prefixed, never read by user code.
        """
        return f"neo_portal_proposed_{field_name}"

    @staticmethod
    def dispatch_error(field_name: str) -> str:
        """Portal dispatch on_invalid='route_to_error' payload field name. Keyed off the dispatch node's own producer field,
        via the SAME per-output-key naming convention as the sibling
        ``{field_name}_dispatch`` success field (``output_field_name``) --
        NOT ``neo_``-prefixed: unlike the mesh's internal handoff_payload/
        handoff_hops channels, this field is user-visible (the named
        error_handler sibling AND the top-level caller both read it off the
        run() result), so it must survive the engine's neo_* stripping.
        Carries the invalid spec's name + the underlying gate error message
        (mirrors ExecutionError.build(construct=, found=)'s shape). Does NOT
        coexist with the success field ({field_name}_dispatch) on the same
        invocation.
        """
        return output_field_name(field_name, "dispatch_error")


    @staticmethod
    def oracle_collector(field_name: str) -> str:
        """Oracle barrier/collector field name for a given producer field."""
        return f"neo_oracle_{field_name}"

    @staticmethod
    def agent_messages(field_name: str) -> str:
        """Agent-cycle message-history channel for an agent/act node.

        `neo_`-prefixed so the full ReAct message history is stripped from the
        returned state (_strip_internals) and excluded from the schema
        fingerprint — it is turn-by-turn plumbing, not durable user output.
        """
        return f"neo_agent_messages_{field_name}"

    @staticmethod
    def agent_tool_log(field_name: str) -> str:
        """Agent-cycle tool_log channel (accumulated ToolInteraction records)."""
        return f"neo_agent_tool_log_{field_name}"

    @staticmethod
    def resource_manifest(field_name: str) -> str:
        """Agent-cycle resource-manifest channel (accumulated ResourceRef records
        lifted from resource_link tool-result blocks).

        ``neo_``-prefixed so ``_strip_internals`` removes it from returned state
        and ``compute_schema_fingerprint`` excludes it — exactly like the sibling
        ``agent_tool_log`` channel. It IS a checkpointed state channel (built in
        ``state.py:_add_agent_channels`` with an append reducer), so a HITL pause
        preserves the manifest across resume. Contrast with config-only ``_neo_``
        keys (``DI_INPUTS``) which never enter state.
        """
        return f"{StateKeys.RESOURCE_MANIFEST_PREFIX}{field_name}"

    @staticmethod
    def agent_budget(field_name: str) -> str:
        """Agent-cycle budget/iteration channel (per-tool call counts, iteration
        count, cumulative input tokens) — survives per-turn checkpoints, unlike
        the in-memory ToolBudgetTracker the monolith used."""
        return f"neo_agent_budget_{field_name}"

    @staticmethod
    def eachoracle_collector(field_name: str) -> str:
        """Each+Oracle composed barrier/collector field name."""
        return f"neo_eachoracle_{field_name}"


def _strip_internals(result: Any) -> Any:
    """Remove `neo_*` framework plumbing from a result dict.

    Pure result-shaping utility. Lives here (a neutral low-level module) rather
    than in the run layer so both the run layer (``runner.py``) and the compile
    layer (``_subconstruct.py``, which strips internals off sub-graph results)
    can import it at module level without a compile->run inversion.
    """
    if not isinstance(result, dict):
        return result
    return {k: v for k, v in result.items() if not k.startswith(StateKeys.FRAMEWORK_PREFIX)}
