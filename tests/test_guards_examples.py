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


# ═══════════════════════════════════════════════════════════════════════════
# neograph-yi0n — example LLM-structured output models must be typed (no open
# dict[...] maps). Open mappings have no fixed json-schema `properties` and so
# violate OpenAI structured-output (json_schema) invariants -> runtime 400.
# Scripted-node outputs are NOT the disease (no structured-output call), so
# this guard only flags models produced by LLM nodes (prompt= present).
# ═══════════════════════════════════════════════════════════════════════════

import re as _re

# Output models that legitimately keep an open mapping despite being an LLM
# output (none today). Each entry MUST carry a one-line reason.
OPEN_MAPPING_OUTPUT_ALLOWLIST: dict[str, str] = {}


def _class_field_annotations(tree: ast.AST) -> dict[str, list[str]]:
    """Map every class name -> list of its annotated field type sources."""
    out: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            anns = [
                ast.unparse(stmt.annotation)
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and stmt.annotation is not None
            ]
            out[node.name] = anns
    return out


def _annotation_has_open_mapping(annotation: str) -> bool:
    """True if the annotation contains a dict[...] / Dict[...] open mapping."""
    return bool(_re.search(r"\bdict\[", annotation, _re.IGNORECASE))


def _model_has_open_mapping(
    name: str, fields_map: dict[str, list[str]], _seen: set[str] | None = None
) -> bool:
    """True if `name` or any model it transitively references declares a dict[...] field."""
    if _seen is None:
        _seen = set()
    if name in _seen or name not in fields_map:
        return False
    _seen.add(name)
    for ann in fields_map[name]:
        if _annotation_has_open_mapping(ann):
            return True
        for ref in _re.findall(r"[A-Za-z_][A-Za-z0-9_]*", ann):
            if ref != name and ref in fields_map and _model_has_open_mapping(ref, fields_map, _seen):
                return True
    return False


def _llm_output_model_refs(tree: ast.AST) -> list[tuple[int, str]]:
    """(lineno, model_name) for every LLM node whose output model is a bare Name.

    LLM node = a Node(...)/node(...) call OR an @node-decorated function that
    carries a `prompt=` keyword and is not mode='scripted'/'raw'. Output models
    come from `outputs=`/`output=` (explicit) or, for an @node function with no
    explicit output kwarg, the function's return annotation (inference).
    """
    refs: list[tuple[int, str]] = []

    def _is_llm_kwargs(kws: dict[str, ast.expr]) -> bool:
        if "prompt" not in kws:
            return False
        mode = kws.get("mode")
        return not (isinstance(mode, ast.Constant) and mode.value in ("scripted", "raw"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            kws = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            if not _is_llm_kwargs(kws):
                continue
            for out_kw in ("outputs", "output"):
                v = kws.get(out_kw)
                if isinstance(v, ast.Name):
                    refs.append((node.lineno, v.id))
        elif isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                fname = dec.func.id if isinstance(dec.func, ast.Name) else getattr(dec.func, "attr", None)
                if fname != "node":
                    continue
                kws = {kw.arg: kw.value for kw in dec.keywords if kw.arg}
                if not _is_llm_kwargs(kws):
                    continue
                if any(k in kws for k in ("outputs", "output")):
                    continue  # explicit output already captured by the Call branch
                if isinstance(node.returns, ast.Name):
                    refs.append((node.lineno, node.returns.id))
    return refs


def _scan_open_mapping_outputs(source: str) -> list[tuple[int, str]]:
    """Return (lineno, model_name) for every LLM-structured output model in
    `source` that (transitively) declares an open dict[...] field."""
    tree = ast.parse(source)
    fields_map = _class_field_annotations(tree)
    return [
        (lineno, name)
        for lineno, name in _llm_output_model_refs(tree)
        if _model_has_open_mapping(name, fields_map)
    ]


class TestExampleStructuredOutputModelsAreTyped:
    """neograph-yi0n — LLM nodes route structured output through OpenAI's
    json_schema API (on OpenRouter/OpenAI). An output model with an open
    dict[...] field has no fixed json-schema `properties` and 400s at runtime.
    Examples must declare named, typed output models. Scripted-node outputs are
    exempt (no structured-output call is made).
    """

    def test_no_open_mapping_in_llm_output_models(self):
        offenders: list[str] = []
        for path in _iter_example_pipelines():
            rel = path.relative_to(REPO_ROOT).as_posix()
            for lineno, model_name in _scan_open_mapping_outputs(path.read_text()):
                key = f"{rel}:{model_name}"
                if key in OPEN_MAPPING_OUTPUT_ALLOWLIST:
                    assert OPEN_MAPPING_OUTPUT_ALLOWLIST[key], f"empty allowlist reason for {key}"
                    continue
                offenders.append(f"{rel}:{lineno} — LLM output model {model_name} has an open dict[...] field")
        assert not offenders, (
            "LLM-structured output model with an open dict[...] field found in examples "
            "(neograph-yi0n). Use a named Pydantic model with typed fields, or "
            "output_strategy='json_mode'.\n" + "\n".join(offenders)
        )

    def test_meta_catches_open_dict_llm_output(self):
        """Positive: an LLM node whose output model has a dict[...] field is flagged."""
        source = (
            "from neograph import node\n"
            "from pydantic import BaseModel\n"
            "class Bad(BaseModel):\n"
            "    classified: list[dict[str, str]]\n"
            "@node(prompt='classify', model='fast', outputs=Bad)\n"
            "def classify(x): ...\n"
        )
        assert _scan_open_mapping_outputs(source), "must flag open-dict LLM output model"

    def test_meta_passes_named_model_output(self):
        """Negative: a named, typed output model is not flagged."""
        source = (
            "from neograph import node\n"
            "from pydantic import BaseModel\n"
            "class Classification(BaseModel):\n"
            "    claim: str\n"
            "    category: str\n"
            "class Good(BaseModel):\n"
            "    classified: list[Classification]\n"
            "@node(prompt='classify', model='fast', outputs=Good)\n"
            "def classify(x): ...\n"
        )
        assert not _scan_open_mapping_outputs(source), "named-model output must not be flagged"

    def test_meta_passes_scripted_open_dict_output(self):
        """Negative: a SCRIPTED node with an open-dict output is exempt (no
        structured-output call) -- the ALLOWLIST examples (01/01c/05/10 scripted)."""
        source = (
            "from neograph import Node\n"
            "from pydantic import BaseModel\n"
            "class Scored(BaseModel):\n"
            "    scored: list[dict[str, str]]\n"
            "n = Node.scripted('score', fn='score_fn', outputs=Scored)\n"
        )
        assert not _scan_open_mapping_outputs(source), "scripted open-dict output must not be flagged"

    def test_meta_catches_nested_open_dict(self):
        """Would-be-missed: the dict[...] hides one model level below the LLM
        output model. Transitive check must still catch it."""
        source = (
            "from neograph import node\n"
            "from pydantic import BaseModel\n"
            "class Inner(BaseModel):\n"
            "    attrs: dict[str, str]\n"
            "class Outer(BaseModel):\n"
            "    inner: Inner\n"
            "@node(prompt='p', model='fast', outputs=Outer)\n"
            "def f(x): ...\n"
        )
        assert _scan_open_mapping_outputs(source), "must catch dict[...] nested one model level below the output"
