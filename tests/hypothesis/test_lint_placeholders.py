"""Hypothesis property tests for lint template placeholder validation.

Stress-tests the invariant: _predict_input_keys must match what _extract_input
actually produces at runtime for any valid pipeline topology.
"""

from __future__ import annotations

import re

import hypothesis.strategies as st
from hypothesis import given, settings
from pydantic import BaseModel

from neograph import Construct, Node, compile
from neograph.factory import register_scripted
from neograph.lint import _predict_input_keys, lint

from .conftest import Alpha, Beta, Gamma, _make_fn, _uid


# ── Strategies ──────────────────────────────────────────────────────────

@st.composite
def inline_prompt_node(draw):
    """Generate a Node with dict-form inputs and an inline prompt that
    references some valid and some invalid placeholders."""
    tag = _uid()
    n_inputs = draw(st.integers(min_value=1, max_value=4))
    types = [Alpha, Beta, Gamma]
    input_dict = {}
    for i in range(n_inputs):
        key = f"inp{i}_{tag}".replace("-", "_")
        input_dict[key] = draw(st.sampled_from(types))

    valid_keys = list(input_dict.keys())
    # Pick some valid refs and some invalid refs
    n_valid = draw(st.integers(min_value=0, max_value=min(len(valid_keys), 3)))
    n_invalid = draw(st.integers(min_value=0, max_value=3))

    valid_refs = draw(st.permutations(valid_keys))[:n_valid]
    invalid_refs = [f"bad{i}_{tag}".replace("-", "_") for i in range(n_invalid)]

    all_refs = list(valid_refs) + invalid_refs
    if not all_refs:
        all_refs = valid_keys[:1]  # at least one ref
        invalid_refs = []

    prompt = " ".join(f"${{{ref}}}" for ref in all_refs)

    node = Node(
        f"hyp-{tag}", prompt=prompt, model="default",
        outputs=Alpha, inputs=input_dict,
    )
    return node, set(valid_refs), set(invalid_refs)


@st.composite
def pipeline_with_inline_prompts(draw):
    """Generate a valid 2-3 node pipeline where the terminal LLM node has
    an inline prompt with some valid and some invalid placeholders."""
    tag = _uid()
    n_intermediates = draw(st.integers(min_value=0, max_value=1))

    src_name = f"src-{tag}"
    register_scripted(f"hyp_src_{tag}", _make_fn(Alpha))

    nodes = [Node.scripted(src_name, fn=f"hyp_src_{tag}", outputs=Alpha)]

    prev_type = Alpha
    prev_field = src_name.replace("-", "_")
    for i in range(n_intermediates):
        mid_name = f"mid{i}-{tag}"
        register_scripted(f"hyp_mid{i}_{tag}", _make_fn(Beta))
        nodes.append(Node.scripted(
            mid_name, fn=f"hyp_mid{i}_{tag}",
            inputs={prev_field: prev_type}, outputs=Beta,
        ))
        prev_type = Beta
        prev_field = mid_name.replace("-", "_")

    # Terminal LLM node with inline prompt
    n_invalid = draw(st.integers(min_value=0, max_value=2))
    invalid_refs = [f"bogus{i}" for i in range(n_invalid)]
    all_refs = [prev_field] + invalid_refs
    prompt = " ".join(f"${{{ref}}}" for ref in all_refs)

    terminal = Node(
        f"term-{tag}", prompt=prompt, model="default",
        outputs=Gamma, inputs={prev_field: prev_type},
    )
    nodes.append(terminal)

    construct = Construct(f"hyp-pipe-{tag}", nodes=nodes)
    return construct, set(invalid_refs)


# ── Property tests ──────────────────────────────────────────────────────

class TestPredictInputKeysInvariant:
    """_predict_input_keys must match what the runtime would see."""

    @given(data=inline_prompt_node())
    @settings(max_examples=50)
    def test_predicted_keys_match_dict_form_inputs(self, data):
        """For any node with dict-form inputs, predicted keys == input dict keys."""
        node, valid_refs, invalid_refs = data
        predicted = _predict_input_keys(node)
        assert isinstance(node.inputs, dict)
        assert predicted == set(node.inputs.keys())

    @given(data=inline_prompt_node())
    @settings(max_examples=50)
    def test_valid_placeholders_never_flagged(self, data):
        """Placeholders that match input keys are never reported as issues."""
        node, valid_refs, invalid_refs = data
        # Build a minimal construct with a source + this node
        tag = _uid()
        register_scripted(f"hyp_vsrc_{tag}", _make_fn(Alpha))

        src = Node.scripted(f"vsrc-{tag}", fn=f"hyp_vsrc_{tag}", outputs=Alpha)
        # Need source nodes for all input keys
        src_nodes = []
        for key, typ in node.inputs.items():
            fn_name = f"hyp_vs_{key}_{tag}"
            register_scripted(fn_name, _make_fn(typ))
            src_nodes.append(Node.scripted(key, fn=fn_name, outputs=typ))

        try:
            c = Construct(f"hyp-valid-{tag}", nodes=[*src_nodes, node])
        except Exception:
            return  # invalid topology, skip

        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        flagged_params = {i.param for i in template_issues}

        for ref in valid_refs:
            assert ref not in flagged_params, (
                f"Valid placeholder '{ref}' should not be flagged. "
                f"Input keys: {set(node.inputs.keys())}, flagged: {flagged_params}"
            )

    @given(data=inline_prompt_node())
    @settings(max_examples=50)
    def test_invalid_placeholders_always_flagged(self, data):
        """Placeholders that DON'T match input keys are always reported."""
        node, valid_refs, invalid_refs = data
        if not invalid_refs:
            return  # nothing to test

        tag = _uid()
        src_nodes = []
        for key, typ in node.inputs.items():
            fn_name = f"hyp_is_{key}_{tag}"
            register_scripted(fn_name, _make_fn(typ))
            src_nodes.append(Node.scripted(key, fn=fn_name, outputs=typ))

        try:
            c = Construct(f"hyp-invalid-{tag}", nodes=[*src_nodes, node])
        except Exception:
            return

        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        flagged_params = {i.param for i in template_issues}

        for ref in invalid_refs:
            assert ref in flagged_params, (
                f"Invalid placeholder '{ref}' should be flagged. "
                f"Input keys: {set(node.inputs.keys())}, flagged: {flagged_params}"
            )


class TestLintPipelineProperty:
    """End-to-end: lint on generated pipelines catches exactly the invalid placeholders."""

    @given(data=pipeline_with_inline_prompts())
    @settings(max_examples=30)
    def test_lint_catches_exactly_invalid_placeholders(self, data):
        """lint() flags exactly the bogus placeholders, nothing else."""
        construct, expected_invalid = data
        issues = lint(construct)
        template_issues = [i for i in issues if "template" in i.kind]
        flagged = {i.param for i in template_issues}

        assert flagged == expected_invalid, (
            f"Expected to flag {expected_invalid}, actually flagged {flagged}"
        )

    @given(data=pipeline_with_inline_prompts())
    @settings(max_examples=30)
    def test_all_template_issues_are_required(self, data):
        """Every template placeholder issue has required=True (runtime crash)."""
        construct, _ = data
        issues = lint(construct)
        template_issues = [i for i in issues if "template" in i.kind]
        for issue in template_issues:
            assert issue.required is True, (
                f"Template issue should be required=True: {issue}"
            )
