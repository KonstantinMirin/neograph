"""Structural guards for example pipelines.

Pins the canonical NOW pattern (neograph-jq2t): every example that uses
external prompt files (markdown loaded into ``compile(..., prompt_compiler=...)``)
must NOT ship a ``lambda template, data: ..., template, ...`` shape that
discards ``data``. The discipline is mirrored from piarch's
``neograph_bridge.prompt_compiler`` — see
/Users/konst/projects/piarch/src/derive_ensemble/integration/neograph_bridge.py:62-210.

The guard scans ``examples/**/pipeline.py`` (and ``examples/*.py``) for the
bypass-``data`` lambda pattern and fails when found.

Allowlist: inline-prompt-only examples (where the prompt contains
``${var}`` markers and neograph auto-substitutes) AND intentional minimal
demos that pass a fixed string like ``"analyze"`` — see notes on
neograph-jq2t for the per-file disposition.
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES_ROOT = REPO_ROOT / "examples"


# Files where the bypass-data lambda is INTENTIONAL (see scan disposition).
# Each entry MUST have a one-line reason; opaque entries fail review.
ALLOWLIST = {
    "examples/03_oracle_ensemble.py":
        "only prompt is inline ${tagged_claims}; neograph auto-substitutes; lambda never reached",
    "examples/20_oracle_merge_hooks.py":
        "only prompts are inline ${...}; same as above",
    "examples/02_produce_and_gather.py":
        "intentional minimal demo with hardcoded 'analyze' string",
    "examples/08_structured_output_coercion.py":
        "intentional minimal demo",
    "examples/10_full_pipeline.py":
        "intentional minimal demo",
    "examples/18_typed_projections.py":
        "no LLM call — pure type-projection demo, lambda returns []",
    "examples/vs_langgraph/02_tool_agent.py":
        "tool-agent demo: model gets seed instruction, then iterates via tools",
}


def _iter_example_pipelines() -> list[pathlib.Path]:
    """All .py files in examples/ that could call compile(prompt_compiler=...)."""
    return [
        p for p in EXAMPLES_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts
    ]


def _is_bypass_data_lambda(call: ast.Call) -> tuple[bool, str | None]:
    """Detect the disease pattern: ``lambda template, data, ...: [..., template, ...]``
    where the returned content is bound to the bare ``template`` (or ``tmpl``)
    parameter — i.e., the lambda ignores ``data``.

    Returns (is_bypass, snippet) — snippet is the truncated source representation
    when is_bypass is True.
    """
    # Looking for: prompt_compiler=lambda ARG, ARG2, ...: [{"role": "user", "content": <bare ARG>}]
    if not call.keywords:
        return False, None
    for kw in call.keywords:
        if kw.arg != "prompt_compiler":
            continue
        if not isinstance(kw.value, ast.Lambda):
            return False, None
        lam = kw.value
        # Must have at least 2 args (template, data)
        args = lam.args.args
        if len(args) < 2:
            return False, None
        template_name = args[0].arg
        # The lambda body should be a list of message dicts.
        # We're looking for the simplest broken case: content is the bare
        # template parameter (no .format, no f-string interpolation referencing data).
        body = lam.body
        if not isinstance(body, ast.List):
            return False, None
        # Walk into the first dict literal; check the 'content' value.
        for elt in body.elts:
            if not isinstance(elt, ast.Dict):
                continue
            for key_node, val_node in zip(elt.keys, elt.values, strict=False):
                if not isinstance(key_node, ast.Constant) or key_node.value != "content":
                    continue
                # Disease shape: content is just the bare template name (ast.Name)
                if isinstance(val_node, ast.Name) and val_node.id == template_name:
                    return True, ast.unparse(lam)
        return False, None
    return False, None


def _scan_file(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return list of (lineno, snippet) for every bypass-data lambda in path."""
    findings: list[tuple[int, str]] = []
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return findings
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_bug, snippet = _is_bypass_data_lambda(node)
        if is_bug:
            findings.append((node.lineno, snippet or ""))
    return findings


class TestExampleCanonicalPromptCompiler:
    """neograph-jq2t — examples must use the canonical NOW pattern.

    The bypass-``data`` lambda shape (``prompt_compiler=lambda template, data:
    [{"role": "user", "content": template}]``) discards the BAML-pre-rendered
    input that neograph provides. The model receives static instructions but
    no content to analyze. This guard fails when the disease shape is found
    outside the documented allowlist.
    """

    def test_no_bypass_data_lambda_in_examples(self):
        offenders: list[str] = []
        for path in _iter_example_pipelines():
            rel = path.relative_to(REPO_ROOT).as_posix()
            findings = _scan_file(path)
            if not findings:
                continue
            if rel in ALLOWLIST:
                # Verify the allowlist reason is non-empty
                assert ALLOWLIST[rel], (
                    f"ALLOWLIST entry for {rel} has empty reason — "
                    f"add a one-line justification"
                )
                continue
            for lineno, snippet in findings:
                offenders.append(f"{rel}:{lineno} — {snippet}")
        assert not offenders, (
            "Bypass-data prompt_compiler lambda found in examples that are NOT "
            "in ALLOWLIST. Rewrite to canonical pattern "
            "(template.format(**data) for dict-input, template.format(input=data) "
            "for single-input) — see piarch's neograph_bridge.py:62-210.\n"
            + "\n".join(offenders)
        )

    def test_meta_scanner_catches_bypass_pattern(self, tmp_path):
        """Positive meta-test: scanner must catch the disease shape."""
        synthetic = tmp_path / "bypass.py"
        synthetic.write_text(
            "from neograph import compile\n"
            "graph = compile(\n"
            "    pipeline,\n"
            "    prompt_compiler=lambda template, data: ["
            '{"role": "user", "content": template}'
            "],\n"
            ")\n"
        )
        findings = _scan_file(synthetic)
        assert findings, (
            "Scanner must catch synthetic `lambda template, data: [..., template]` "
            "pattern; got nothing"
        )

    def test_meta_scanner_passes_canonical_format(self, tmp_path):
        """Negative meta-test: canonical .format(**data) shape must not be flagged."""
        synthetic = tmp_path / "canonical.py"
        synthetic.write_text(
            "from neograph import compile\n"
            "graph = compile(\n"
            "    pipeline,\n"
            "    prompt_compiler=lambda template, data, **kw: ["
            '{"role": "user", "content": template.format(**data)}'
            "],\n"
            ")\n"
        )
        findings = _scan_file(synthetic)
        assert not findings, (
            "Scanner must NOT flag canonical .format(**data) shape; "
            f"got false positives: {findings}"
        )

    def test_meta_scanner_passes_hardcoded_string(self, tmp_path):
        """Negative meta-test: hardcoded-string demos must not be flagged
        (they don't reference the bare template parameter)."""
        synthetic = tmp_path / "hardcoded.py"
        synthetic.write_text(
            "from neograph import compile\n"
            "graph = compile(\n"
            "    pipeline,\n"
            "    prompt_compiler=lambda template, data: ["
            '{"role": "user", "content": "analyze"}'
            "],\n"
            ")\n"
        )
        findings = _scan_file(synthetic)
        assert not findings, (
            "Scanner must NOT flag hardcoded-string demos; "
            f"got false positives: {findings}"
        )

    def test_meta_scanner_resists_tmpl_alias_slip(self, tmp_path):
        """Regex-slip meta-test: the disease shape uses ``tmpl`` instead of
        ``template`` as the first parameter name. Scanner must still catch it
        (it binds to whatever the first parameter is)."""
        synthetic = tmp_path / "tmpl_alias.py"
        synthetic.write_text(
            "from neograph import compile\n"
            "graph = compile(\n"
            "    pipeline,\n"
            "    prompt_compiler=lambda tmpl, d, **kw: ["
            '{"role": "user", "content": tmpl}'
            "],\n"
            ")\n"
        )
        findings = _scan_file(synthetic)
        assert findings, (
            "Scanner must catch the bypass pattern regardless of parameter naming "
            "(tmpl/template/t — first-param-bound content is the disease)"
        )
