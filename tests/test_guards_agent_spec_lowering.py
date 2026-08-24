"""Structural guard: agent/act mode export must never regress to a
lossy placeholder (neograph-i3zsh.1's disease scan -- codebase-scan:complete).

Disease pattern: a modifier-lowering function in _agent_spec.py silently
drops a Node field instead of either lowering it to a real Agent Spec
primitive + a neograph/*_spec round-trip marker, or failing loud. The
motivating instance was _lower_node's agent/act branch constructing a
ToolNode placeholder (or, pre-i3zsh.1, failing loud with no real lowering
at all) that would have silently dropped prompt/model/tools.
"""

from __future__ import annotations

import ast

# neograph-3ffdg.3 split the exporter into four modules. These AST guards make
# claims about the export SURFACE, not about a filename, so they scan the whole
# cluster. The file list is single-sited in tests/agent_spec_capabilities.py.
from tests.agent_spec_capabilities import agent_spec_source


def _lower_node_source() -> str:
    tree = ast.parse(agent_spec_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_lower_node":
            return ast.get_source_segment(agent_spec_source(), node) or ""
    raise AssertionError("_lower_node not found in _agent_spec.py")


def _to_agent_spec_source() -> str:
    """The export BODY, wherever it currently lives.

    neograph-qtfof.13 split the public ``to_agent_spec`` into a thin wrapper
    (builds the ApiProviderResolver once) plus ``_to_agent_spec_with``, which
    holds the lowering body these guards scan. Re-keyed to the internal name,
    with the public one kept as a fallback: this guard follows the code, it does
    not pin which of the two the body sits in.
    """
    tree = ast.parse(agent_spec_source())
    by_name = {
        node.name: ast.get_source_segment(agent_spec_source(), node) or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    for candidate in ("_to_agent_spec_with", "to_agent_spec"):
        source = by_name.get(candidate, "")
        if "if ni.is_dict_form:" in source:
            return source
    raise AssertionError(
        "neither _to_agent_spec_with nor to_agent_spec in _agent_spec.py carries the "
        "dict-form fan-in branch these guards scan"
    )


def _dict_form_fan_in_branch(source: str) -> str:
    start = source.index("if ni.is_dict_form:")
    end = source.index("# Single-type inputs (convenience shorthand):", start)
    return source[start:end]


def _lower_loop_source() -> str:
    tree = ast.parse(agent_spec_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_lower_loop":
            return ast.get_source_segment(agent_spec_source(), node) or ""
    raise AssertionError("_lower_loop not found in _agent_spec.py")


def _lower_each_source() -> str:
    tree = ast.parse(agent_spec_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_lower_each":
            return ast.get_source_segment(agent_spec_source(), node) or ""
    raise AssertionError("_lower_each not found in _agent_spec.py")


def _lower_generation_step_source() -> str:
    """The shared per-node.mode dispatch (neograph-2s2o6). Post-refactor the
    think/agent-act/scripted construction lives here, NOT in _lower_node."""
    tree = ast.parse(agent_spec_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_lower_generation_step":
            return ast.get_source_segment(agent_spec_source(), node) or ""
    raise AssertionError("_lower_generation_step not found in _agent_spec.py")


class TestAgentActModeLowersToAgentNode:
    """The shared _lower_generation_step's agent/act branch must construct a real
    AgentNode, never a ToolNode placeholder or a bare fail-loud with no lowering at
    all. Retargeted from _lower_node to _lower_generation_step (neograph-2s2o6): the
    per-mode dispatch now lives in the shared function; _lower_node is a thin wrapper."""

    def test_agent_act_branch_constructs_agent_node(self):
        """Positive: the current source builds an AgentNode for agent/act mode."""
        source = _lower_generation_step_source()
        assert "AgentNode(" in source, (
            "_lower_generation_step's agent/act branch must construct a pyagentspec "
            "AgentNode -- the ToolNode placeholder silently dropped prompt/model/tools"
        )

    def test_agent_act_branch_does_not_construct_bare_tool_node_only(self):
        """Negative (regex-slip guard): a ToolNode-only agent/act lowering --
        i.e. a version of this function that constructs ToolNode but NOT
        AgentNode inside the agent/act branch -- must be caught. Simulates
        the exact pre-i3zsh.1 regression by checking the two constructors
        are not conflated: AgentNode must appear BEFORE the final bare
        ToolNode return (which is reached only by scripted/raw modes)."""
        source = _lower_generation_step_source()
        agent_idx = source.find('mode in ("agent", "act")')
        agent_node_idx = source.find("AgentNode(")
        tool_node_idx = source.rfind("nodes_mod.ToolNode(")
        assert agent_idx != -1, "agent/act mode dispatch branch not found"
        assert agent_node_idx != -1 and agent_node_idx > agent_idx, (
            "AgentNode construction must appear inside (after) the agent/act mode branch"
        )
        # The scripted/raw ToolNode fallback must come AFTER the agent/act
        # branch's own AgentNode construction -- i.e. agent/act mode returns
        # before ever reaching the bare ToolNode fallback.
        assert tool_node_idx > agent_node_idx, (
            "the scripted/raw ToolNode fallback must be structurally reachable only "
            "after the agent/act branch already returned its own AgentNode"
        )

    def test_agent_act_branch_stamps_reconstruction_marker(self):
        """Every irreversible flattening must carry a neograph/-prefixed marker
        (per the exporter's Core Invariant) -- agent/act is no exception.

        Strengthened (neograph-2s2o6, review MEDIUM-3): assert the EXECUTABLE
        stamping (_MARK_AGENT_SPEC constant + the _agent_spec_marker() call), NOT
        only the comment literal 'neograph/agent_spec'. The prior comment-keyed
        check was vacuous -- deleting the real stamping while keeping the comment
        stayed green. The comment migrated with the branch, but the token is what
        actually guarantees the marker is emitted."""
        source = _lower_generation_step_source()
        assert "_MARK_AGENT_SPEC: _agent_spec_marker(node)" in source, (
            "agent/act lowering must stamp the neograph/agent_spec marker via the "
            "_MARK_AGENT_SPEC constant + _agent_spec_marker(node) call (executable "
            "token, not a comment) so from_agent_spec() reconstructs the node losslessly"
        )


class TestDictFormFanInResolvesRealPropertyTitles:
    """Structural guard for neograph-ozxqw's codebase-scan MIGRATE row.

    Disease pattern: ``to_agent_spec()``'s dict-form fan-in branch (a
    downstream node with ``inputs={'seed': A}``, @node's PRIMARY fan-in
    shape) must build each ``DataFlowEdge`` from a resolved output/input
    Property TITLE (via ``_properties_for``) -- never the raw inputs-dict
    KEY (the upstream node's bare NAME), which is not itself a Property
    title and crashes pyagentspec's own ``DataFlowEdge`` validator.
    """

    def test_dict_form_branch_resolves_via_properties_for(self):
        branch = _dict_form_fan_in_branch(_to_agent_spec_source())
        assert "_properties_for(" in branch, (
            "dict-form fan-in must resolve Property titles via _properties_for(), "
            "not construct DataFlowEdges directly from the raw inputs-dict key"
        )

    def test_dict_form_branch_never_uses_bare_upstream_name_as_property_title(self):
        branch = _dict_form_fan_in_branch(_to_agent_spec_source())
        assert "source_output=upstream_name" not in branch, (
            "regression of neograph-ozxqw: source_output must be a real output "
            "Property title, never the bare upstream inputs-dict key"
        )
        assert "destination_input=upstream_name" not in branch, (
            "regression of neograph-ozxqw: destination_input must be a real input "
            "Property title, never the bare upstream inputs-dict key"
        )

    def test_meta_guard_catches_the_disease_pattern_if_reintroduced(self):
        """Meta-test (positive+negative pair, not regex-based -- plain
        substring checks have no regex-slip failure mode): prove the two
        assertions above actually flag the pre-fix disease pattern, so this
        guard isn't vacuously passing only because the current source
        happens to be fixed."""
        buggy_branch = (
            "if ni.is_dict_form:\n"
            "    for upstream_name in ni.by_name:\n"
            "        source_node = data_node_by_item_name.get(upstream_name)\n"
            "        data_edges.append(\n"
            "            edges_mod.DataFlowEdge(\n"
            "                source_output=upstream_name,\n"
            "                destination_input=upstream_name,\n"
            "            )\n"
            "        )\n"
        )
        assert "source_output=upstream_name" in buggy_branch
        assert "destination_input=upstream_name" in buggy_branch
        assert "_properties_for(" not in buggy_branch


class TestDictFormFanInSkipsFanOutReceiver:
    """Structural guard for neograph-qtfof.1's codebase-scan MIGRATE row.

    Disease pattern: ``to_agent_spec()``'s dict-form fan-in branch must skip
    the Each fan-out RECEIVER key (``item.fan_out_param``) before treating a
    dict-form inputs key as an upstream NODE name -- otherwise a legitimate
    ``map_over=``/programmatic-Each dict-form node raises a false
    ``ConfigurationError`` ("no exportable Agent Spec node") for its own
    fan-out item slot, which is populated per-item by the MapNode's own
    sub-flow wiring, not by a peer node.
    """

    def test_dict_form_branch_checks_fan_out_param_before_lookup(self):
        branch = _dict_form_fan_in_branch(_to_agent_spec_source())
        assert "fan_out_param" in branch, (
            "dict-form fan-in must read item.fan_out_param and skip that key "
            "before doing the upstream-node lookup -- mirrors "
            "_validation_inputs.py's fan_out_param skip"
        )

    def test_meta_guard_catches_the_disease_pattern_if_reintroduced(self):
        """Meta-test (positive+negative pair, not regex-based -- plain
        substring checks have no regex-slip failure mode): prove the
        assertion above actually flags a pre-fix branch with no
        fan_out_param skip at all, so this guard isn't vacuously passing
        only because the current source happens to be fixed."""
        buggy_branch = (
            "if ni.is_dict_form:\n"
            "    for upstream_name in ni.by_name:\n"
            "        upstream_item = item_by_name.get(upstream_name)\n"
            "        source_node = data_node_by_item_name.get(upstream_name)\n"
            "        if upstream_item is None or source_node is None:\n"
            "            raise ConfigurationError.build(...)\n"
        )
        assert "fan_out_param" not in buggy_branch


class TestLoopSelfEdgeResolvesDictFormDestinationTitle:
    """Structural guard for neograph-qtfof.2's codebase-scan MIGRATE row.

    Disease pattern: ``_lower_loop``'s self-edge must not construct
    ``destination_input`` from a BARE output Property title alone -- dict-form
    inputs qualify input Property titles with their upstream key (per
    ``_properties_for``'s dict-form convention), so a bare title crashes
    pyagentspec's own ``DataFlowEdge`` validator for any Loop-wrapped node
    declared with @node's PRIMARY dict-form inputs shape.

    neograph-8zvd1 renamed the resolved variable ``dest_prefix`` -> ``dest_key``
    (a KEY, not a pre-joined string) and routes it through the shared
    ``compose_property_title``, so the guard now pins those names.
    """

    def test_loop_self_edge_computes_a_dict_form_destination_prefix(self):
        source = _lower_loop_source()
        assert "dest_key" in source, (
            "_lower_loop's self-edge must resolve a dict-form destination "
            "key (via node.inputs' dict-form key), not assume the "
            "destination's input Property title is always bare"
        )
        assert "compose_property_title(dest_key, prop.title)" in source, (
            "the self-edge's destination title must be built from the resolved "
            "dest_key through the SHARED compose_property_title, never the bare "
            "prop.title alone and never an inline separator literal -- Option F "
            "(neograph-cbpyx) then routes the PROMPT-path form through the body's "
            "flat placeholder map when the loop body is a translated LLM node, "
            "but the dict-form key resolution is unchanged"
        )

    def test_meta_guard_catches_the_disease_pattern_if_reintroduced(self):
        """Meta-test (positive+negative pair, not regex-based -- plain
        substring checks have no regex-slip failure mode): prove the
        assertions above actually flag a pre-fix function body that
        constructs destination_input from the bare title alone."""
        buggy_source = (
            "def _lower_loop(node, loop, body):\n"
            "    data_edges = []\n"
            "    for prop in _properties_for(node.outputs):\n"
            "        data_edges.append(\n"
            "            edges_mod.DataFlowEdge(\n"
            "                source_node=body,\n"
            "                source_output=prop.title,\n"
            "                destination_node=body,\n"
            "                destination_input=prop.title,\n"
            "            )\n"
            "        )\n"
        )
        assert "dest_key" not in buggy_source
        assert "compose_property_title(dest_key, prop.title)" not in buggy_source


class TestLoopSelfEdgeDecidesAgainstTheBodysDeclaredProperties:
    """Structural guard for neograph-rh5fb.

    Disease pattern: ``_lower_loop`` constructs the self-feedback
    ``DataFlowEdge`` unconditionally and lets pyagentspec's own validator be
    the one to discover there is no such property on the destination -- which
    surfaces as a raw pydantic ``ValidationError`` for any Construct item whose
    ``input`` and ``output`` types differ (a shape that compiles and RUNS).

    The edge must be a DECISION taken per field against the body's declared
    Properties, and it must not be bought back by swallowing the validator's
    exception -- a suppressed export error is exactly the silent seam the North
    Star forbids.
    """

    def test_loop_self_edge_reads_the_bodys_declared_properties(self):
        source = _lower_loop_source()
        assert "body.inputs" in source and "body.outputs" in source, (
            "_lower_loop's self-edge must decide per field against the lowered "
            "body's own declared input/output Properties -- the destination "
            "property may not exist at all when the body's boundary types differ"
        )
        assert "property_is_castable_to" in source, (
            "the pairing rule must be pyagentspec's OWN property_is_castable_to "
            "(the same predicate DataFlowEdge validates with), never a re-derived "
            "type-compatibility check that can drift from it"
        )

    def test_loop_self_edge_does_not_suppress_the_validation_error(self):
        source = _lower_loop_source()
        assert "except" not in source, (
            "_lower_loop must not catch-and-suppress pyagentspec's DataFlowEdge "
            "ValidationError -- decide whether the destination property exists "
            "BEFORE constructing the edge"
        )

    def test_meta_guard_catches_the_disease_pattern_if_reintroduced(self):
        """Meta-test: prove the assertions above flag both the pre-fix body
        (unconditional construction) and the band-aid (try/except suppression)."""
        pre_fix = (
            "def _lower_loop(node, loop, body):\n"
            "    for prop in _properties_for(_item_outputs(node)):\n"
            "        data_edges.append(edges_mod.DataFlowEdge(destination_node=body))\n"
        )
        band_aid = (
            "def _lower_loop(node, loop, body):\n"
            "    for prop in _properties_for(_item_outputs(node)):\n"
            "        try:\n"
            "            data_edges.append(edges_mod.DataFlowEdge(destination_node=body))\n"
            "        except ValidationError:\n"
            "            continue\n"
        )
        assert "body.inputs" not in pre_fix
        assert "property_is_castable_to" not in pre_fix
        assert "except" in band_aid


def _func_def(name: str) -> ast.FunctionDef:
    tree = ast.parse(agent_spec_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in _agent_spec.py")


def _all_func_names() -> set[str]:
    tree = ast.parse(agent_spec_source())
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _oracle_variant_loop() -> ast.For:
    """The ``for i, model_tier in enumerate(variant_models):`` per-variant loop
    inside ``_lower_oracle`` -- the exact region that must delegate its
    per-node.mode construction, matched by its iterable (not by line number)."""
    fn = _func_def("_lower_oracle")
    for node in ast.walk(fn):
        if isinstance(node, ast.For) and "variant_models" in ast.unparse(node.iter):
            return node
    raise AssertionError("_lower_oracle's variant loop (over variant_models) not found")


def _construction_calls_in(node: ast.AST, class_names: frozenset[str]) -> list[str]:
    """Every ``Call`` under ``node`` whose callee is one of ``class_names``
    (matched as ``mod.ClassName(...)`` Attribute OR bare ``ClassName(...)``
    Name) -- an AST shape match, never a line-number or substring match."""
    found: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                name = None
            if name in class_names:
                found.append(name)
    return found


_GENERATION_NODE_CTORS: frozenset[str] = frozenset({"LlmNode", "AgentNode"})


class TestPerModeDispatchLivesInOneSharedFunction:
    """Anti-re-duplication structural guard for neograph-2s2o6 (disease-scan
    REQUIRED; refinement x02ki.12 step 5b) -- written FAILING-FIRST.

    Disease: the per-``node.mode`` THREE-WAY generation-step dispatch
    (``think`` -> LlmNode / ``agent``,``act`` -> AgentNode+_make_agent /
    ``scripted``,``raw`` -> ToolNode) is hand-written as TWO independent copies
    -- one in ``_lower_node``, one inline in ``_lower_oracle``'s variant loop.
    This is the exact 'one validator, not two' / 'single source of truth, do
    not re-inline elsewhere' pattern CLAUDE.md bans (``effective_producer_type``,
    ``_check_fan_in_inputs``); it was just never applied here.

    The invariant (north-star -- keep the disease UNWRITEABLE): the LlmNode/
    AgentNode construction dispatch must live in EXACTLY ONE shared function
    (``_lower_generation_step``); both ``_lower_node`` and the ``_lower_oracle``
    variant loop DELEGATE to it and construct NO LlmNode/AgentNode of their own.

    FAILS on current code (both functions still hold their own dispatch); the
    2s2o6 extract turns it GREEN. The ``_lower_oracle`` MERGE LlmNode/ToolNode
    (built OUTSIDE the variant loop, per the plan) is deliberately NOT in scope
    -- only the per-variant generation-step dispatch is.
    """

    def test_lower_node_body_constructs_no_llm_or_agent_node(self) -> None:
        found = _construction_calls_in(_func_def("_lower_node"), _GENERATION_NODE_CTORS)
        assert not found, (
            "_lower_node must be a thin wrapper that DELEGATES the per-node.mode "
            "generation-step dispatch to _lower_generation_step -- it still "
            f"constructs {sorted(set(found))} of its own (the disease: two copies of "
            "the three-way dispatch). Extract the dispatch into _lower_generation_step "
            "(neograph-2s2o6) so it lives in exactly one place."
        )

    def test_lower_oracle_variant_loop_constructs_no_llm_or_agent_node(self) -> None:
        found = _construction_calls_in(_oracle_variant_loop(), _GENERATION_NODE_CTORS)
        assert not found, (
            "_lower_oracle's per-variant loop must DELEGATE to _lower_generation_step "
            f"-- it still constructs {sorted(set(found))} inline (the SECOND hand-written "
            "copy of the three-way dispatch). Call the shared _lower_generation_step per "
            "variant tier instead (neograph-2s2o6)."
        )

    def test_shared_generation_step_function_holds_the_dispatch(self) -> None:
        names = _all_func_names()
        assert "_lower_generation_step" in names, (
            "the unified per-node.mode dispatch must live in a shared function named "
            "_lower_generation_step (neograph-2s2o6) -- it does not exist yet, so the "
            "dispatch is still duplicated across _lower_node and _lower_oracle."
        )
        constructed = set(_construction_calls_in(_func_def("_lower_generation_step"), _GENERATION_NODE_CTORS))
        assert constructed == _GENERATION_NODE_CTORS, (
            "_lower_generation_step must be the ONE home of the three-way dispatch: it "
            f"must construct both {sorted(_GENERATION_NODE_CTORS)} (think + agent/act "
            f"branches), found {sorted(constructed)}."
        )

    def test_meta_guard_detector_flags_construction_and_passes_delegation(self) -> None:
        """Positive+negative meta-test (mirrors the file convention): prove the AST
        detector actually flags a function that constructs LlmNode/AgentNode AND
        passes one that only delegates -- so the guard isn't vacuously green."""
        buggy = ast.parse(
            "def f():\n"
            "    if node.mode == 'think':\n"
            "        return nodes_mod.LlmNode(name=name)\n"
            "    return nodes_mod.AgentNode(name=name)\n"
        )
        delegating = ast.parse(
            "def f():\n    return _lower_generation_step(node, name=name, outputs=outputs, metadata={})\n"
        )
        assert sorted(set(_construction_calls_in(buggy, _GENERATION_NODE_CTORS))) == ["AgentNode", "LlmNode"], (
            "meta-guard: the detector must flag a body that constructs LlmNode/AgentNode"
        )
        assert _construction_calls_in(delegating, _GENERATION_NODE_CTORS) == [], (
            "meta-guard: the detector must pass a body that only delegates to _lower_generation_step"
        )


class TestEachSubflowStartInputsGatedOnTranslation:
    """Structural guard for neograph-cbpyx's MEDIUM-1 review finding.

    Disease pattern (the construction-centric anchoring blind spot the
    review caught): the Option F consumer sweep must cover NON-DataFlowEdge
    input-name consumers too, not just ``DataFlowEdge`` construction sites.
    ``_lower_each``'s sub-flow ``StartNode.inputs`` is such a consumer of
    ``_properties_for(node.inputs)``. When the inner node is
    placeholder-translated (an LLM-mode node), its declared inputs are the
    FLAT ``${var}->{{ flat }}`` names, so the StartNode MUST use the SAME
    flat titles (via ``_node_translation``) or the sub-flow ships an
    unfillable ``{{ item_v }}`` to a foreign consumer -- a silent seam worse
    than a red cell. A regression that unconditionally builds the StartNode
    from ``_properties_for(node.inputs)`` (the pre-fix shape) reintroduces
    exactly the dotted/flat mismatch this ticket eliminated.
    """

    def test_each_start_inputs_are_gated_on_translation_eligibility(self):
        source = _lower_each_source()
        assert "_is_translation_eligible(node)" in source, (
            "_lower_each must gate its sub-flow StartNode.inputs on "
            "_is_translation_eligible(node): a translated inner node's flat "
            "input titles must flow to the StartNode, not the untranslated "
            "dotted _properties_for(node.inputs) titles"
        )
        assert "_node_translation(node)" in source, (
            "the translation-eligible branch must derive the StartNode's "
            "inputs from _node_translation(node) (the SAME flat map the inner "
            "node was translated with), never a second, divergent computation"
        )
        # The StartNode must consume the gated `inner_inputs` variable, never a
        # direct _properties_for(node.inputs) call inline on the StartNode.
        assert "inputs=inner_inputs" in source, (
            "the StartNode must declare inputs=inner_inputs (the branch-resolved "
            "Property list), so the translation gate above actually reaches the "
            "sub-flow boundary"
        )

    def test_meta_guard_catches_the_disease_pattern_if_reintroduced(self):
        """Meta-test (positive+negative pair, not regex-based -- plain
        substring checks have no regex-slip failure mode): prove the
        assertions above actually flag a pre-fix body that builds the
        StartNode unconditionally from the untranslated dotted Properties."""
        buggy_source = (
            "def _lower_each(node, each):\n"
            "    inner = _lower_node(node)\n"
            "    start_node = nodes_mod.StartNode(\n"
            "        name=f'{node.name}__each_start',\n"
            "        inputs=_properties_for(node.inputs) or None,\n"
            "    )\n"
        )
        assert "_is_translation_eligible(node)" not in buggy_source
        assert "_node_translation(node)" not in buggy_source
        assert "inputs=inner_inputs" not in buggy_source
