"""Focused Oracle+agent/act EXPORT + round-trip regression tests (neograph-i7k7j).

The exhaustive matrix (``tests/test_agent_spec_matrix.py``) already parametrizes
the 8 ``{agent,act}-oracle-{merge_fn,merge_prompt}-{single,dict}`` RED_EXPORT
cells (xfail(strict) in BOTH the export AND round-trip matrices), so those cells
are NOT re-authored here. This file pins the behaviors the matrix does NOT cover
because it varies INPUT shape only (single / dict / context = fan-in) and NEVER
output arity, nor the per-variant model-tier detail:

  1. EXPORT-PATH dict-form-OUTPUTS guard (the MEDIUM silent-seam, refined design
     step 1b). ``to_agent_spec`` walks the RAW construct via ``iter_with_arms`` and
     NEVER runs the compile pre-pass (``wrap_fan_over_agents``), so the qzrv
     compile-time rejection of dict-form Oracle+agent OUTPUTS
     (``_fan_agent.py`` -> "multi-output (dict-form) outputs is not supported") is
     UNREACHABLE on the export path. The incumbent unconditional raise at
     ``_agent_spec.py`` ("no Agent Spec lowering yet") is the ONLY thing catching
     it today. Once single-output Oracle+agent starts exporting (step 1), the NEW
     agent/act variant branch MUST itself reject dict-form OUTPUTS (step 1b),
     mirroring the qzrv message -- otherwise a semantically-undefined multi-output
     agent ensemble ships silently (a North-Star seam).

  2. Round-trip fidelity for single-output Oracle+agent (steps 1 + 2). The loader
     must gain an ``AgentNode`` branch in ``_reconstruct_oracle_group`` so
     export -> import rebuilds the ORIGINAL ``Node(mode='agent') | Oracle(...)``
     shape (base mode/prompt/model/tools + Oracle.n/merge).

  3. Per-variant model tiers from ``Oracle.models`` (step 1 detail). ``_make_agent``
     builds ``llm_config`` from ``node.model``; without a per-variant
     ``model_copy`` the tier axis is silently ignored.

TDD RED: ALL tests below fail TODAY because ``_lower_oracle``'s agent/act branch
raises unconditionally ("Oracle+agent/act export has no Agent Spec lowering yet")
BEFORE any of these behaviors can be observed. Test 1 additionally pins the
dict-form-specific message so it keeps guarding the seam AFTER single-output
export lands. Confirmed by running pytest, not by inspection.

Run with::

    uv run --extra dev --extra agent-spec pytest tests/test_agent_spec_oracle_agent_export.py
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyagentspec")

from neograph import Construct, Node, Oracle, Tool, ToolInteraction  # noqa: E402
from neograph._agent_spec import to_agent_spec  # noqa: E402
from neograph.errors import NeographError  # noqa: E402
from neograph.loader import from_agent_spec  # noqa: E402
from neograph.modifiers import classify_modifiers  # noqa: E402

from .schemas import Claims, RawText, _producer  # noqa: E402


class TestOracleAgentDictFormOutputsNeverExportsAFlow:
    """North-Star ratchet for the qzrv "dict-form Oracle+agent OUTPUTS is
    undefined / unrepresentable" combination: it must fail loud and NEVER yield an
    exported Flow. The Oracle merge_fn contract is single-type, so an N-way merge
    of secondary agent OUTPUTS (e.g. ``tool_log``) across fanned variants is
    undefined -- a multi-output agent ensemble must never ship.

    FINDING (surfaced by this test, contradicts refined design step 1b/step 5's
    premise): step 1b asserts the EXPORTER (``to_agent_spec``) is the SOLE guard
    for this combo, on the reasoning that ``to_agent_spec`` walks the raw construct
    via ``iter_with_arms`` and never runs the compile pre-pass, so the qzrv
    rejection in ``_fan_agent.py`` is unreachable via export. That reasoning misses
    that ``raise_if_unsupported_fan_over_agent`` (the SAME qzrv predicate) is called
    from ``_validate_node_chain`` at ``Construct.__init__`` -- i.e. at ASSEMBLY,
    upstream of BOTH compile and export. So the dict-form Oracle+agent OUTPUTS
    construct is genuinely UNBUILDABLE (``ConstructError`` at ``Construct(...)``),
    and ``to_agent_spec`` can never receive it. The seam is already closed at the
    stronger (unrepresentable) layer, so the step-1b export guard is likely
    unreachable dead code -- the implementer should confirm before adding it.

    Consequently this test is GREEN TODAY (the guarantee already holds via the
    assembly guard); it is a durable regression ratchet, NOT the TDD-red artifact
    for i7k7j (that role belongs to the two tests below, which fail because export
    raises today). It is written to be tolerant of WHERE the fail-loud fires
    (assembly today, or an added export guard tomorrow): a ``NeographError`` naming
    the multi-output condition, raised before any Flow escapes.
    """

    def _build_and_export(self) -> object:
        # Build AND export in one span so the assertion holds no matter which layer
        # (assembly ConstructError today, or an export-path ConfigurationError if
        # step 1b is ever added) enforces the fail-loud.
        seed = _producer("seed", RawText)  # RawText.text
        gen = Node(
            name="agent-gen",
            mode="agent",
            inputs={"seed": RawText},
            outputs={"result": Claims, "tool_log": list[ToolInteraction]},  # dict-form
            model="fast",
            prompt="analyze ${seed.text}",
            tools=[Tool(name="t_read", budget=1)],
        ) | Oracle(n=2, merge_fn="m_combine")
        return to_agent_spec(Construct("oracle-agent-dict-outputs", nodes=[seed, gen]))

    def test_dict_form_outputs_fail_loud_and_no_flow_escapes(self) -> None:
        with pytest.raises(NeographError) as exc_info:
            self._build_and_export()

        msg = str(exc_info.value).lower()
        assert "multi-output" in msg or "dict-form" in msg, (
            "the fail-loud for Oracle-over-agent dict-form OUTPUTS must name the "
            "multi-output / dict-form condition (the qzrv seam), so it stays a "
            "targeted rejection of undefined secondary-output merges rather than a "
            f"generic error. Got:\n{msg}"
        )
        assert "output" in msg, (
            "the rejection must be about OUTPUT arity (the qzrv seam), not merely "
            f"input fan-in or a generic missing-lowering message. Got:\n{msg}"
        )


class TestOracleAgentSingleOutputRoundTrips:
    """Steps 1 + 2: a single-output ``Node(mode='agent') | Oracle(...)`` must
    export to a Flow AND re-import to the SAME IR shape -- base mode 'agent',
    prompt/model/tools recovered, Oracle.n + merge preserved. This pins the
    loader's new ``AgentNode`` branch in ``_reconstruct_oracle_group`` (step 2),
    which is REQUIRED, not optional (the 8 round-trip matrix cells xfail(strict)
    ratchet it, but the matrix asserts only that import yields SOME Construct --
    it does not assert the reconstructed base is mode='agent' with tools).

    FAILS NOW at the ``to_agent_spec`` call: the agent/act variant branch raises.
    """

    def _pipeline(self) -> Construct:
        prod = _producer("prod", RawText)  # RawText.text
        gen = Node(
            name="gen",
            mode="agent",
            inputs={"prod": RawText},
            outputs=Claims,
            model="fast",
            prompt="analyze ${prod.text}",
            tools=[Tool(name="t_read", budget=1)],
        ) | Oracle(n=2, merge_fn="m_combine")
        return Construct("oracle-agent-rt", nodes=[prod, gen])

    def test_single_output_oracle_agent_round_trips_to_original_shape(self) -> None:
        flow = to_agent_spec(self._pipeline())
        imported = from_agent_spec(flow)

        reconstructed = next(
            (n for n in imported.nodes if getattr(n, "name", None) == "gen"), None
        )
        assert reconstructed is not None, (
            "the reconstructed Oracle group must surface a single combo node named "
            f"'gen' (the merge node's name), got nodes "
            f"{[getattr(n, 'name', n) for n in imported.nodes]!r}"
        )

        assert reconstructed.mode == "agent", (
            "the reconstructed base node must recover mode='agent' (the loader's new "
            f"AgentNode branch), got mode={reconstructed.mode!r}"
        )
        assert reconstructed.prompt == "analyze ${prod.text}", (
            "the ORIGINAL ${var} prompt must be recovered from the agent-spec marker, "
            f"got {reconstructed.prompt!r}"
        )
        assert reconstructed.model == "fast", (
            f"the base node model must round-trip, got {reconstructed.model!r}"
        )
        tool_names = {t.name for t in (reconstructed.tools or [])}
        assert "t_read" in tool_names, (
            "the agent's tools must be recovered from the agent-spec marker "
            f"(_reconstruct_agent_node), got tools {tool_names!r}"
        )

        _combo, mods = classify_modifiers(reconstructed)
        oracle = mods.get("oracle")
        assert oracle is not None, (
            "the reconstructed node must carry an Oracle modifier (base | Oracle), got "
            f"modifiers {sorted(mods)!r}"
        )
        assert oracle.n == 2, f"Oracle.n must round-trip, got {oracle.n!r}"
        assert oracle.merge_fn == "m_combine", (
            f"Oracle.merge_fn must round-trip, got {oracle.merge_fn!r}"
        )


class TestOracleAgentPerVariantModelTiers:
    """Step 1 detail: ``Oracle.models`` per-variant tiers must reach each exported
    AgentNode variant's ``Agent.llm_config`` -- ``_make_agent`` reads
    ``node.model``, so without a per-variant ``node.model_copy(update={'model':
    tier})`` the tier axis is silently ignored and every variant carries the base
    ``node.model``.

    FAILS NOW at the ``to_agent_spec`` call: the agent/act variant branch raises.
    """

    def _pipeline(self) -> Construct:
        prod = _producer("prod", RawText)
        gen = Node(
            name="gen",
            mode="agent",
            inputs={"prod": RawText},
            outputs=Claims,
            model="base",  # NOT the per-variant tiers below
            prompt="analyze ${prod.text}",
            tools=[Tool(name="t_read", budget=1)],
        ) | Oracle(n=2, models=["fast", "smart"], merge_fn="m_combine")
        return Construct("oracle-agent-tiers", nodes=[prod, gen])

    def test_each_agent_variant_carries_its_oracle_model_tier(self) -> None:
        from pyagentspec.flows.nodes import AgentNode

        flow = to_agent_spec(self._pipeline())

        variants = {
            n.name: n
            for n in flow.nodes
            if isinstance(n, AgentNode) and "__variant_" in n.name
        }
        assert set(variants) == {"gen__variant_0", "gen__variant_1"}, (
            f"expected two Oracle AgentNode variants, got {sorted(variants)!r}"
        )

        model_ids = {
            name: node.agent.llm_config.model_id for name, node in variants.items()
        }
        assert model_ids["gen__variant_0"] == "fast", (
            "variant 0 must carry Oracle.models[0]='fast' (per-variant model_copy), "
            f"not the base node.model='base'. Got {model_ids['gen__variant_0']!r}"
        )
        assert model_ids["gen__variant_1"] == "smart", (
            "variant 1 must carry Oracle.models[1]='smart' (per-variant model_copy), "
            f"not the base node.model='base'. Got {model_ids['gen__variant_1']!r}"
        )
