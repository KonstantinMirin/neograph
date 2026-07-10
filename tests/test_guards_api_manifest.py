"""Freshness guard for the verifiable-docs API manifest (Stage A, neograph-ryn4h).

The committed ``website/src/data/api-manifest.json`` and
``website/src/data/api-manifest-mcp.json`` are generated SOLELY by introspecting
``neograph.__all__`` + ``neograph_mcp.__all__`` + ``errors.py`` + the
``lint.py`` AST. Any drift between the committed files and a fresh in-memory
regeneration fails the default pytest suite -- the rustdoc "docs build from the
code" contract that Stage B (remark plugin) and Stage C (generated reference
sections) depend on.

Mirrors ``tests/test_spec_loader.py::test_pipeline_schema_in_sync`` (single
on-disk-vs-generated diff) and the wider ``tests/test_guards_*.py`` culture.
The generator lives in ``scripts/gen_api_manifest.py`` and is expected to expose:

  - ``build_manifest()``          -> dict   (core: neograph.__all__ + errors + lint kinds)
  - ``build_mcp_manifest()``      -> dict   (neograph_mcp.__all__; raises ImportError w/o extra)
  - ``extract_lint_issue_kinds()`` -> list[{kind, severity, meaning}]  (enriched from
    lint.py kind= literals + DI kinds, severity/meaning owned by LINT_KIND_META)
  - ``slug(name: str)``           -> str    (github-slugger 2.0.0-compatible; the anchor contract)
  - ``main()``                    -> int    (writes both files to website/src/data/)

Cross-stage contracts pinned by this module (from the ryn4h Refinements):

  - M1: field annotations use Python ``str(annotation)`` (NOT describe_type TS notation).
  - M2: declared-only model fields via MRO subtraction -- ``ForwardConstruct``
        (pure subclass of ``Construct``) emits ZERO fields, not Construct's 8.
  - M3: per-symbol ``anchor = slug(symbol NAME)``, NOT slug(signature) -- stable
        across signature changes (adding a kwarg to ``compile()`` keeps its anchor).
  - L1: slug snapshot values are JS-authoritative, generated via the REAL
        ``github-slugger`` npm package (a transitive Starlight dep), not a
        tautological Python reimplementation.
  - Option (c): core + mcp split into two files; the mcp guard is
        ``skipif not _HAS_MCP`` so the default ``uv run --extra dev pytest`` stays light.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "gen_api_manifest.py"
WEBSITE_DATA = REPO_ROOT / "website" / "src" / "data"
CORE_MANIFEST_PATH = WEBSITE_DATA / "api-manifest.json"
MCP_MANIFEST_PATH = WEBSITE_DATA / "api-manifest-mcp.json"
API_MDX_PATH = REPO_ROOT / "website" / "src" / "content" / "docs" / "reference" / "api.mdx"

# Sentinel comment pair delimiting the ONE contiguous generated reference region
# inside api.mdx (design "ONE GENERATED REFERENCE BLOCK", refine atom uqy66.54).
# render_reference_sections() output goes VERBATIM between them; the freshness
# guard extracts exactly the bytes between the two markers and diffs against a
# fresh render. MDX comment syntax so Starlight ignores it.
GEN_REGION_START = "{/* GEN:reference-sections START */}"
GEN_REGION_END = "{/* GEN:reference-sections END */}"

# Fence-aware markdown heading matcher -- mirrors website/plugins/remark-api.mjs
# lines 78-79 byte-for-byte (`/^#{1,6}\s+(.+?)\s*$/`) so the guard slugs the SAME
# heading text github-slugger sees at build time. Levels 1-6, trailing whitespace
# trimmed, capture group 1 is the heading TEXT (which slug() maps to the anchor).
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")

# Mirror tests/test_mcp_battery.py:43 -- re-defined locally per L3 (RE-DEFINE local).
_HAS_MCP = bool(importlib.util.find_spec("mcp")) and bool(importlib.util.find_spec("langchain_mcp_adapters"))
requires_mcp = pytest.mark.skipif(not _HAS_MCP, reason="requires the mcp extra (mcp + langchain-mcp-adapters)")


def _load_gen_api_manifest():
    """Load ``scripts/gen_api_manifest.py`` as a module.

    ``scripts/`` is not a package (mirrors ``scripts/regen_schema.py``), so the
    module is loaded by file path via ``importlib.util`` -- the same pattern
    ``tests/test_guards_examples.py:628`` and
    ``tests/test_example_map_reduce.py:27`` use for path-located modules.
    """
    assert SCRIPT_PATH.exists(), (
        f"scripts/gen_api_manifest.py is missing -- Stage A generator not yet "
        f"implemented (neograph-ryn4h). Expected at {SCRIPT_PATH}."
    )
    spec = importlib.util.spec_from_file_location("gen_api_manifest", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, (
        f"could not build a module spec for {SCRIPT_PATH}"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The 10 verified kind= string literals in src/neograph/lint.py today (grep
# `kind="[a-z_]+"` | sort -u). The extractor must yield AT LEAST these -- the
# floor catches kind=variable drift (an AST walk for string-literal kwargs
# silently misses a variable assignment).
_KNOWN_LINT_KIND_FLOOR = frozenset({
    "act_mode_all_idempotent_tools",
    "ask_human_in_mutating_node",
    "llm_kwargs_missing",
    "loop_condition_none_unsafe",
    "loop_condition_unregistered",
    "resource_hydration_kind_unmatched",
    "template_placeholder_known_vars_only",
    "template_placeholder_unresolvable",
    "template_var_requires_async_driver",
    "tool_requires_async_driver",
})


# ════════════════════════════════════════════════════════════════════════════
# Class 1 -- the freshness guard + the M2/M3 manifest-structure invariants
# ════════════════════════════════════════════════════════════════════════════
class TestApiManifestFreshness:
    """The committed manifests must equal a fresh in-memory regeneration.

    Pins the rustdoc "docs build from the code" contract: any public-surface
    change without a manifest commit fails the default pytest suite. Drift
    message MUST name the regen script verbatim -- the same shape as
    ``test_pipeline_schema_in_sync`` (tests/test_spec_loader.py:1296).
    """

    def test_core_manifest_is_in_sync_with_committed_file(self):
        """website/src/data/api-manifest.json == build_manifest() byte-for-byte.

        Drift fails loudly with a message pointing at
        ``python scripts/gen_api_manifest.py``. This is the foundation Stage B/C
        depend on: every anchor and signature the remark plugin validates comes
        from this committed file.
        """
        gen = _load_gen_api_manifest()
        assert CORE_MANIFEST_PATH.exists(), (
            f"core manifest not committed at {CORE_MANIFEST_PATH}. "
            f"Run: python scripts/gen_api_manifest.py"
        )
        on_disk = json.loads(CORE_MANIFEST_PATH.read_text())
        generated = gen.build_manifest()
        assert on_disk == generated, (
            "api-manifest.json drifted from build_manifest(). "
            "Regenerate with:  python scripts/gen_api_manifest.py"
        )

    def test_core_manifest_was_generated_with_sort_keys_for_byte_stability(self):
        """Both the file and build_manifest() output MUST be sort_keys=True so
        dict iteration order cannot churn the committed file across Python
        versions or runs. Belt-and-braces per the ryn4h JSON-stability risk.
        """
        gen = _load_gen_api_manifest()
        if not CORE_MANIFEST_PATH.exists():
            pytest.skip("core manifest not committed yet (TDD red)")
        # If the file was written by main(), it is already json.dumps(indent=2,
        # sort_keys=True) + "\n". Re-serializing the loaded dict with sort_keys
        # must produce byte-identical text (otherwise keys are NOT stable).
        raw = CORE_MANIFEST_PATH.read_text()
        on_disk_obj = json.loads(raw)
        canonical = json.dumps(on_disk_obj, indent=2, sort_keys=True) + "\n"
        assert raw == canonical, (
            "api-manifest.json is not in canonical sort_keys=True form. "
            "Regenerate with:  python scripts/gen_api_manifest.py"
        )
        # build_manifest() must round-trip through the same canonical form.
        generated_obj = gen.build_manifest()
        assert json.dumps(generated_obj, indent=2, sort_keys=True) + "\n" == canonical, (
            "build_manifest() output is not byte-identical to the canonical file. "
            "Drift between generator and committed artifact."
        )

    def test_freshness_check_is_not_a_noop_stale_manifest_is_detected(self):
        """Negative meta-test: the freshness guard actually catches drift.

        A structural guard that only passes on a correct file is unproven until
        it is shown to FAIL on a stale one. Simulate the canonical failure mode
        -- a public symbol removed in code but the committed manifest not
        regenerated -- and assert the equality check the guard uses would no
        longer hold. Without this, the in-sync test could pass on a generator
        that ignores the symbol set entirely.
        """
        gen = _load_gen_api_manifest()
        current = gen.build_manifest()
        # Deep-copy + drop one symbol = the stale-committed-manifest shape.
        stale = json.loads(json.dumps(current))
        assert stale["symbols"], "manifest has no symbols -- generator walk is broken"
        stale["symbols"].pop()
        assert stale != current, (
            "freshness guard is a no-op: dropping a symbol did not change "
            "build_manifest() output -- the generator is not sensitive to the "
            "public surface, so a stale committed manifest would pass undetected."
        )

    @requires_mcp
    def test_mcp_manifest_is_in_sync_with_committed_file(self):
        """website/src/data/api-manifest-mcp.json == build_mcp_manifest() byte-for-byte.

        Per Option (c): the mcp surface is a SEPARATE committed file, guarded
        only when the ``mcp`` extra is present (skipif-not-_HAS_MCP). Default
        ``uv run --extra dev pytest`` stays light; full surface guarded in
        ``uv run --extra dev --extra mcp pytest``.
        """
        gen = _load_gen_api_manifest()
        assert MCP_MANIFEST_PATH.exists(), (
            f"mcp manifest not committed at {MCP_MANIFEST_PATH}. "
            f"Run with the extra installed:  uv run --extra dev --extra mcp python scripts/gen_api_manifest.py"
        )
        on_disk = json.loads(MCP_MANIFEST_PATH.read_text())
        generated = gen.build_mcp_manifest()
        assert on_disk == generated, (
            "api-manifest-mcp.json drifted from build_mcp_manifest(). "
            "Regenerate with:  uv run --extra dev --extra mcp python scripts/gen_api_manifest.py"
        )

    def test_forward_construct_emits_no_inherited_fields_from_construct_parent(self):
        """M2 -- declared-only fields via MRO subtraction.

        ``ForwardConstruct`` subclasses ``Construct`` and declares NO new
        Pydantic fields, so its manifest entry must contribute an EMPTY ``fields``
        list -- not ``Construct``'s 8 inherited fields. Without the MRO
        subtraction the manifest duplicates fields under every subclass entry,
        which is the exact drift risk the architect review's MEDIUM finding flags.

        Verified live counts (per ryn4h Refinements M2):
          Construct        -> 8 declared fields
          ForwardConstruct -> 0 declared fields (pure subclass)
          Node             -> 18 declared fields
        """
        gen = _load_gen_api_manifest()
        manifest = gen.build_manifest()
        symbols = manifest.get("symbols") or manifest.get("api_symbols") or []
        by_name = {s["name"]: s for s in symbols}
        assert "ForwardConstruct" in by_name, (
            "ForwardConstruct missing from manifest symbols -- __all__ walk is broken"
        )
        forward_construct = by_name["ForwardConstruct"]
        # ForwardConstruct is a Pydantic BaseModel subclass -> it has a "fields" key.
        assert "fields" in forward_construct, (
            "Pydantic model entry missing the 'fields' key -- schema drift"
        )
        assert forward_construct["fields"] == [], (
            f"ForwardConstruct must emit ZERO declared fields (pure subclass of "
            f"Construct), saw {len(forward_construct['fields'])} -- the MRO "
            f"subtraction filter is missing or wrong. "
            f"Use: set(cls.model_fields) - {{k for base in cls.__mro__[1:] for k in "
            f"getattr(base, 'model_fields', {{}})}}"
        )

    def test_every_symbol_anchor_is_slug_of_its_name(self):
        """M3 (Stage C kind-namespace refinement, neograph-rfl7b DECISION 1).

        The anchor is NAME-based, never signature-based (adding a kwarg to
        ``compile()`` must not change its anchor). The refinement Stage C adds:
        when two distinct public symbols share one bare ``slug(name)`` (VERIFIED:
        node/Node and tool/Tool), each colliding symbol's anchor is
        kind-namespaced to ``f"{slug(name)}-{tag}"`` so the manifest owns a
        DISTINCT anchor per symbol. The tightened contract:

          - non-colliding symbol:  ``anchor == slug(name)`` (unchanged from M3).
          - colliding symbol:      ``anchor == f"{slug(name)}-{tag}"`` where
            ``tag`` derives from the symbol's kind (``gen._KIND_ANCHOR_TAG``).
          - every anchor is slug-stable (``slug(anchor) == anchor``) and UNIQUE.

        This is the deliberate Stage-A contract change Stage B/C consume; the L1
        JS-authoritative slug snapshot is untouched (a suffix built from the same
        ``slug()`` is appended -- ``slug()`` itself is unchanged).
        """
        from collections import Counter

        gen = _load_gen_api_manifest()
        manifest = gen.build_manifest()
        symbols = manifest.get("symbols") or manifest.get("api_symbols") or []
        assert symbols, "manifest symbols list is empty -- __all__ walk is broken"
        base_counts = Counter(gen.slug(s["name"]) for s in symbols)
        seen_anchors: set[str] = set()
        for symbol in symbols:
            name = symbol.get("name")
            anchor = symbol.get("anchor")
            assert name is not None and anchor is not None, (
                f"symbol entry missing name/anchor keys: {symbol!r}"
            )
            base = gen.slug(name)
            if base_counts[base] == 1:
                assert anchor == base, (
                    f"symbol {name!r} anchor is {anchor!r} but its bare slug does "
                    f"not collide, so it must equal slug(name)={base!r} (M3)."
                )
            else:
                tag = gen._KIND_ANCHOR_TAG.get(symbol["kind"], symbol["kind"])
                expected = f"{base}-{tag}"
                assert anchor == expected, (
                    f"symbol {name!r} (kind={symbol['kind']}) bare slug {base!r} "
                    f"collides, so its anchor must be kind-namespaced to "
                    f"{expected!r}, got {anchor!r} (Stage C DECISION 1)."
                )
            # Every anchor must be slug-stable so Starlight reproduces it exactly.
            assert gen.slug(anchor) == anchor, (
                f"symbol {name!r} anchor {anchor!r} is not slug-stable "
                f"(slug({anchor!r}) == {gen.slug(anchor)!r})."
            )
            # And distinct across the whole surface (the Stage C crux).
            assert anchor not in seen_anchors, (
                f"symbol {name!r} anchor {anchor!r} is shared by >1 symbol -- "
                f"disambiguation failed to make anchors unique."
            )
            seen_anchors.add(anchor)


# ════════════════════════════════════════════════════════════════════════════
# Class 2 -- slug fidelity snapshot (L1: JS-authoritative values)
# ════════════════════════════════════════════════════════════════════════════
class TestAnchorSlugFidelity:
    """The Python ``slug()`` port must produce byte-identical output to the
    ``github-slugger`` 2.0.0 npm package -- the package Starlight uses
    (transitive via ``@astrojs/markdown-remark``; see ``website/package-lock.json``)
    to slug reference-page headings. This is the anchor contract Stage B/C rely on.

    Expected values below are JS-AUTHORITATIVE: generated against the real
    installed ``github-slugger`` package via::

        cd website && node --input-type=module -e \\
            "import {slug} from 'github-slugger'; \\
             ['node','FromResource',...].forEach(t => console.log(t+'\t'+slug(t)))"

    NOT a tautological Python reimplementation -- L1 in the ryn4h Refinements.
    """

    # (raw_input, expected_slug) -- expected values verified against
    # github-slugger 2.0.0 in website/node_modules on 2026-07-09.
    _SNAPSHOTS = [
        # --- representative symbol names from neograph.__all__ (the actual
        #     anchor use case per M3 -- all ASCII identifiers, trivially slug'd)
        ("node", "node"),
        ("compile", "compile"),
        ("arun", "arun"),
        ("FromResource", "fromresource"),
        ("Node", "node"),
        ("Each", "each"),
        ("MergePreProcess", "mergepreprocess"),
        ("NeographError", "neographerror"),
        ("CheckpointSchemaError", "checkpointschemaerror"),
        ("DefaultPromptCompiler", "defaultpromptcompiler"),
        ("ask_human", "ask_human"),  # underscores preserved
        ("BlobResult", "blobresult"),
        ("render_prompt", "render_prompt"),
        # --- non-identifier edge cases that exercise the slug algorithm itself.
        #     These catch divergence from the JS package on whitespace, dashes,
        #     and the strip-punctuation character class.
        ("a_quick_brown_fox", "a_quick_brown_fox"),  # underscores preserved
        ("hello world", "hello-world"),              # whitespace -> dash
        ("foo.bar", "foobar"),                       # dots stripped
        ("UPPER CASE", "upper-case"),                # case-fold + whitespace
        ("with-dash", "with-dash"),                  # dashes preserved
        ("with/slash", "withslash"),                 # slashes stripped
        ("foo!bar", "foobar"),                       # punctuation stripped
        ("foo#bar", "foobar"),                       # hash stripped
    ]

    @pytest.mark.parametrize("raw, expected", _SNAPSHOTS)
    def test_slug_matches_github_slugger_npm_package(self, raw, expected):
        """slug(raw) must equal the byte-identical output of github-slugger 2.0.0.

        Divergence here silently breaks every doc anchor Stage B/C emit or
        validate -- a single mismatched character becomes a dead link in the
        generated reference docs.
        """
        gen = _load_gen_api_manifest()
        actual = gen.slug(raw)
        assert actual == expected, (
            f"slug({raw!r}) returned {actual!r}; github-slugger 2.0.0 produces "
            f"{expected!r}. The Python port has diverged from the npm package -- "
            f"re-verify against: cd website && node --input-type=module -e "
            f"\"import {{slug}} from 'github-slugger'; console.log(slug({raw!r}))\""
        )


# ════════════════════════════════════════════════════════════════════════════
# Class 3 -- lint-issue-kind AST extraction floor
# ════════════════════════════════════════════════════════════════════════════
class TestLintIssueKindExtraction:
    """The manifest's ``lint_issue_kinds`` section is AST-extracted from
    ``src/neograph/lint.py`` ``kind='...'`` string literals (the zero-drift
    approach per ryn4h -- mirrors ``tests/test_guards_meta.py:60-80``'s AST walk).

    The extraction MUST yield at least the known floor of 10 kinds. This catches
    kind=variable drift: if ``lint.py`` ever switches a kind to a variable
    assignment or f-string, the AST walk silently misses it. The floor assertion
    makes that silent miss a loud test failure.
    """

    def test_extracted_kinds_are_superset_of_known_floor(self):
        """Enriched extraction must yield every one of the 10 known kind= literals.

        Pin against the floor set, not the exact set, so none of the current 10
        literal kinds may silently disappear. NOTE (neograph-uw54v): extraction
        now returns ``{kind, severity, meaning}`` OBJECTS, and a NEW literal kind
        is no longer picked up with zero other action -- it must ALSO get a
        ``neograph.lint.LINT_KIND_META`` entry, or ``extract_lint_issue_kinds()``
        raises at regen (the completeness + severity-binding cross-checks).
        """
        gen = _load_gen_api_manifest()
        extracted = {e["kind"] for e in gen.extract_lint_issue_kinds()}
        missing = _KNOWN_LINT_KIND_FLOOR - extracted
        assert not missing, (
            f"lint.py AST extraction missed {len(missing)} known kind(s): "
            f"{sorted(missing)}. Either lint.py switched a kind to a variable/"
            f"f-string (AST walk for ast.Constant kwargs cannot see it), or the "
            f"extractor's AST walk is malformed."
        )


# ════════════════════════════════════════════════════════════════════════════
# Class 4 -- enriched lint_issue_kinds: {kind, severity, meaning} objects,
#            COMPLETE 14-kind set incl. the 4 DI kinds, code-derived severity
#            (neograph-uw54v; refinement neograph-uqy66.52)
# ════════════════════════════════════════════════════════════════════════════

# The 4 dynamically-constructed DI kinds the literal AST walk STRUCTURALLY
# cannot see (emitted as ``kind=binding.kind.value`` -- a variable, lint.py:86).
# Sourced from the authoritative frozenset, not hardcoded here.
from neograph.di import DI_TEMPLATE_KINDS  # noqa: E402

_DI_KIND_NAMES = frozenset(k.value for k in DI_TEMPLATE_KINDS)

# The COMPLETE kind set = 10 literal (the floor) + 4 DI. This is the manifest's
# zero-drift target vs what lint() actually emits.
_COMPLETE_LINT_KIND_SET = _KNOWN_LINT_KIND_FLOOR | _DI_KIND_NAMES

_ALLOWED_SEVERITIES = frozenset({"ERROR", "WARN", "WARN/ERROR", "varies"})

# Sanctioned non-derivable severities (refinement neograph-uqy66.52):
#  - loop_condition_none_unsafe is emitted at TWO sites with conflicting
#    required (True@lint.py:579, False@:599) -> merged to the dual value.
#  - the 4 DI kinds are emitted with kind=variable, so the literal walk cannot
#    see their Call -> severity is a runtime binding.required -> 'varies'.
_DUAL_SEVERITY_KIND = "loop_condition_none_unsafe"


def _derive_literal_kind_severities() -> dict[str, set[bool]]:
    """Co-derive per-kind ``required`` booleans from lint.py's LintIssue sites.

    Walks the same ``ast.Call`` nodes the extractor visits. For every Call that
    carries a LITERAL ``kind=<str>`` keyword, records the LITERAL ``required=``
    value at that same Call (defaulting to ``False`` when no ``required=`` keyword
    is present -- LintIssue.required defaults False, e.g. llm_kwargs_missing).

    Returns ``{kind: {required_bool, ...}}`` -- a set because a kind may be
    emitted at multiple sites with conflicting ``required`` (the dual case).
    DI kinds use ``kind=variable`` and are invisible to this literal walk.
    """
    lint_path = REPO_ROOT / "src" / "neograph" / "lint.py"
    tree = ast.parse(lint_path.read_text())
    out: dict[str, set[bool]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kind_val: str | None = None
        required_val: bool = False  # LintIssue.required default
        for kw in node.keywords:
            if (
                kw.arg == "kind"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                kind_val = kw.value.value
            elif (
                kw.arg == "required"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, bool)
            ):
                required_val = kw.value.value
        if kind_val is not None:
            out.setdefault(kind_val, set()).add(required_val)
    return out


class TestLintIssueKindEnrichment:
    """The manifest's ``lint_issue_kinds`` must be enriched from bare name
    strings to ``{kind, severity, meaning}`` objects (neograph-uw54v), covering
    the COMPLETE 14-kind set (10 literal + 4 DI) with a severity that STAYS
    code-derived from the ``required=`` at each emission site.

    This is the Stage-A precursor that makes Severity/Meaning manifest-owned so
    Stage C (neograph-cvjfm) can render the reference lint table from the
    manifest instead of a hand-authored, drift-prone doc table.
    """

    def _entries(self) -> list:
        assert CORE_MANIFEST_PATH.exists(), (
            f"core manifest not committed at {CORE_MANIFEST_PATH}. "
            f"Run: python scripts/gen_api_manifest.py"
        )
        manifest = json.loads(CORE_MANIFEST_PATH.read_text())
        assert "lint_issue_kinds" in manifest, "manifest missing lint_issue_kinds"
        return manifest["lint_issue_kinds"]

    def test_entries_are_objects_with_kind_severity_meaning(self):
        """Each lint_issue_kinds entry is a ``{kind, severity, meaning}`` object.

        FAILS today: the committed manifest is a bare sorted list of name
        strings, so the entries are ``str`` not ``dict``.
        """
        entries = self._entries()
        assert entries, "lint_issue_kinds is empty -- generator walk is broken"
        for entry in entries:
            assert isinstance(entry, dict), (
                f"lint_issue_kinds entry is {type(entry).__name__} {entry!r}, "
                f"expected a {{kind, severity, meaning}} object -- the manifest "
                f"is still a bare list of name strings (not enriched)."
            )
            assert {"kind", "severity", "meaning"} <= set(entry), (
                f"lint_issue_kinds entry {entry!r} missing one of "
                f"kind/severity/meaning keys."
            )

    def test_kind_set_is_complete_including_the_four_di_kinds(self):
        """The manifest kind-set == the COMPLETE 14 kinds (10 literal + 4 DI).

        FAILS today: the AST walk misses the 4 dynamically-constructed DI kinds
        (from_input/from_config/from_input_model/from_config_model), so the
        committed set has only 10.
        """
        entries = self._entries()
        # Object shape required first; guard against bare-string entries so the
        # failure names the enrichment gap, not a TypeError.
        assert all(isinstance(e, dict) for e in entries), (
            "lint_issue_kinds entries are not objects yet -- enrichment missing."
        )
        kinds = {e["kind"] for e in entries}
        missing = _COMPLETE_LINT_KIND_SET - kinds
        assert not missing, (
            f"manifest lint_issue_kinds is INCOMPLETE, missing {sorted(missing)} "
            f"(expected all 14 = 10 literal + 4 DI kinds "
            f"{sorted(_DI_KIND_NAMES)}). The AST walk cannot see kind=variable DI "
            f"emissions; they must be unioned in from neograph.di.DI_TEMPLATE_KINDS."
        )
        extra = kinds - _COMPLETE_LINT_KIND_SET
        assert not extra, (
            f"manifest lint_issue_kinds has unexpected kinds {sorted(extra)} not "
            f"in the known 14-kind set."
        )

    def test_every_entry_has_nonempty_meaning_and_allowed_severity(self):
        """Each entry carries a non-empty ``meaning`` and a ``severity`` in the
        allowed set {ERROR, WARN, WARN/ERROR, varies}.

        FAILS today: entries are bare strings with no meaning/severity at all.
        """
        for entry in self._entries():
            assert isinstance(entry, dict), (
                f"entry {entry!r} is not an object -- enrichment missing."
            )
            meaning = entry.get("meaning")
            assert isinstance(meaning, str) and meaning.strip(), (
                f"lint kind {entry.get('kind')!r} has empty/missing meaning "
                f"{meaning!r}; meaning must be manifest-owned (moved out of "
                f"api.mdx into the lint.py LINT_KIND_META registry)."
            )
            severity = entry.get("severity")
            assert severity in _ALLOWED_SEVERITIES, (
                f"lint kind {entry.get('kind')!r} severity {severity!r} not in "
                f"the allowed set {sorted(_ALLOWED_SEVERITIES)}."
            )

    def test_severity_is_code_derived_from_required_at_emission_site(self):
        """Anti-drift binding (refinement neograph-uqy66.52): the stored severity
        for every SINGLE-SITE literal kind must equal ``'ERROR' if required else
        'WARN'`` co-derived from the ``required=`` at its LintIssue ast.Call, so a
        future ``required=`` flip breaks regen instead of silently drifting.

        Sanctioned exceptions (asserted as constants, not derived):
          - loop_condition_none_unsafe -> 'WARN/ERROR' (dual: True@579 + False@599)
          - the 4 DI kinds             -> 'varies' (kind=variable, runtime-bound)

        FAILS today: no severity is stored at all (bare-string entries).
        """
        derived = _derive_literal_kind_severities()
        by_kind = {}
        for entry in self._entries():
            assert isinstance(entry, dict), (
                f"entry {entry!r} is not an object -- enrichment missing."
            )
            by_kind[entry["kind"]] = entry["severity"]

        for kind in _COMPLETE_LINT_KIND_SET:
            assert kind in by_kind, f"kind {kind!r} missing from manifest"
            stored = by_kind[kind]

            if kind in _DI_KIND_NAMES:
                assert stored == "varies", (
                    f"DI kind {kind!r} is emitted with kind=variable so its "
                    f"severity is runtime binding.required-dependent; the "
                    f"registry must store 'varies', got {stored!r}."
                )
                continue

            if kind == _DUAL_SEVERITY_KIND:
                # Must genuinely be dual in the code for the exception to hold.
                assert derived.get(kind) == {True, False}, (
                    f"{kind!r} is the sanctioned dual-severity exception but the "
                    f"code no longer emits it at both required=True and "
                    f"required=False sites (saw {derived.get(kind)}). Re-derive."
                )
                assert stored == "WARN/ERROR", (
                    f"dual-severity kind {kind!r} must store 'WARN/ERROR', "
                    f"got {stored!r}."
                )
                continue

            required_values = derived.get(kind)
            assert required_values is not None, (
                f"literal kind {kind!r} not found at any LintIssue site in "
                f"lint.py -- the co-derivation walk cannot bind its severity."
            )
            assert len(required_values) == 1, (
                f"literal kind {kind!r} unexpectedly emitted with conflicting "
                f"required= values {required_values}; only "
                f"{_DUAL_SEVERITY_KIND!r} may be dual."
            )
            expected = "ERROR" if next(iter(required_values)) else "WARN"
            assert stored == expected, (
                f"lint kind {kind!r} stored severity {stored!r} DRIFTED from the "
                f"code-derived value {expected!r} ('ERROR' if required else "
                f"'WARN', __main__.py:199). The registry severity is not bound to "
                f"the required= at the emission site."
            )


# ════════════════════════════════════════════════════════════════════════════
# Class 5 -- Stage C: ONE generated per-symbol reference region (neograph-kec0k)
#
# kec0k renders, for every NON-exception manifest symbol, a reference section
# (heading whose slug == the manifest anchor + fenced signature + field table for
# the 14 fielded symbols) into a SINGLE contiguous sentinel-delimited region of
# api.mdx via a NEW gen.render_reference_sections(). Two guards pin it:
#
#   1. FRESHNESS -- the committed region == render_reference_sections() byte-for-
#      byte (the rustdoc "docs build from the code" contract, one region).
#   2. SINGLE-HEADING-PER-ANCHOR -- every non-exception anchor is emitted by
#      EXACTLY ONE heading whose slug == the anchor (turns the silently-inert
#      node/Node/tool/Tool collision refs into a loud fail). Exception anchors are
#      excluded (uorb4 owns the fenced error tree, whose headings are unharvested).
#
# Both FAIL today: render_reference_sections() + the sentinel region do not yet
# exist, and the current signature-in-heading sections slug to garbage so the 4
# collision anchors (and 50+ others) have NO matching heading. That is the TDD red.
# ════════════════════════════════════════════════════════════════════════════


def _extract_reference_region(mdx_text: str) -> str:
    """Return the bytes strictly between the two GEN sentinel markers.

    The freshness contract is a single contiguous region: everything after the
    START marker literal and before the END marker literal is owned verbatim by
    ``render_reference_sections()``. Asserts BOTH markers are present (and in
    order) so a missing region fails loudly rather than returning ``""``.
    """
    start = mdx_text.find(GEN_REGION_START)
    end = mdx_text.find(GEN_REGION_END)
    assert start != -1 and end != -1, (
        f"api.mdx is missing the generated-reference sentinel markers "
        f"{GEN_REGION_START!r} / {GEN_REGION_END!r}. Stage C (neograph-kec0k) must "
        f"add ONE contiguous generated region delimited by them. Expected in "
        f"{API_MDX_PATH}."
    )
    assert start < end, (
        f"api.mdx sentinel markers are out of order: START at {start}, END at "
        f"{end}. The generated region must be START ... END."
    )
    return mdx_text[start + len(GEN_REGION_START):end]


def _reference_heading_slugs(mdx_text: str, gen) -> list[str]:
    """Fence-aware slugs of every markdown heading in api.mdx.

    Mirrors the remark harvester (website/plugins/remark-api.mjs:74-80): track
    a ``` fence toggle, skip lines inside fenced code (so signatures rendered as
    code blocks emit no anchor), match ``#{1,6} <text>`` on the remaining lines,
    and slug the captured TEXT via the generator's slug() -- the exact anchor
    contract Starlight reproduces. Returns a list (not a set) so duplicate
    headings are countable by the single-heading guard.
    """
    slugs: list[str] = []
    fence = False
    for line in mdx_text.split("\n"):
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        match = _HEADING_RE.match(line)
        if match:
            slugs.append(gen.slug(match.group(1)))
    return slugs


class TestReferenceSectionRendering:
    """Stage C: the generated per-symbol reference region (neograph-kec0k)."""

    def _non_exception_symbols(self, gen) -> list[dict]:
        manifest = gen.build_manifest()
        symbols = manifest.get("symbols") or manifest.get("api_symbols") or []
        assert symbols, "manifest symbols list is empty -- __all__ walk is broken"
        return [s for s in symbols if s.get("kind") != "exception"]

    # ── Guard 1: freshness (committed region == regenerated, byte-for-byte) ──
    def test_generated_reference_region_matches_render_reference_sections(self):
        """The committed sentinel region in api.mdx equals a fresh
        ``render_reference_sections()`` byte-for-byte.

        FAILS today: neither ``render_reference_sections()`` (the renderer) nor
        the sentinel-delimited region exists yet -- the TDD red. Once green, any
        drift between the committed reference sections and the manifest-driven
        render (a new symbol, a changed signature, an added field) fails the
        default pytest suite, the rustdoc "docs build from the code" contract.
        """
        gen = _load_gen_api_manifest()
        assert hasattr(gen, "render_reference_sections"), (
            "scripts/gen_api_manifest.py must expose render_reference_sections() "
            "-> str (Stage C, neograph-kec0k): the manifest-driven renderer that "
            "emits one section per NON-exception symbol (heading + fenced "
            "signature + field table). Not implemented yet."
        )
        assert API_MDX_PATH.exists(), f"reference page missing at {API_MDX_PATH}"
        rendered = gen.render_reference_sections()
        region = _extract_reference_region(API_MDX_PATH.read_text())
        assert region == rendered, (
            "api.mdx generated-reference region drifted from "
            "render_reference_sections(). Regenerate the region with the Stage C "
            "renderer (the region between the GEN:reference-sections sentinels "
            "must equal render_reference_sections() byte-for-byte)."
        )

    def test_render_reference_sections_excludes_exception_symbols(self):
        """The renderer must NOT emit a heading for any exception-kind symbol.

        Exceptions live in the FENCED error tree (uorb4); double-emitting a
        heading for them would break single-heading-per-anchor. Assert no
        exception anchor's heading appears in the rendered region.

        FAILS today: render_reference_sections() does not exist.
        """
        gen = _load_gen_api_manifest()
        assert hasattr(gen, "render_reference_sections"), (
            "render_reference_sections() not implemented yet (neograph-kec0k)."
        )
        rendered = gen.render_reference_sections()
        rendered_slugs = set(_reference_heading_slugs(rendered, gen))
        manifest = gen.build_manifest()
        symbols = manifest.get("symbols") or manifest.get("api_symbols") or []
        exception_anchors = {s["anchor"] for s in symbols if s.get("kind") == "exception"}
        leaked = exception_anchors & rendered_slugs
        assert not leaked, (
            f"render_reference_sections() emitted heading(s) for exception "
            f"symbol anchor(s) {sorted(leaked)}; exceptions are owned by the "
            f"fenced error tree (uorb4) and must be excluded from per-symbol "
            f"heading generation."
        )

    # ── Guard 2: single heading per anchor (fence-aware, manifest-scoped) ──
    def test_every_non_exception_anchor_has_exactly_one_matching_heading(self):
        """Each NON-exception manifest anchor is emitted by EXACTLY ONE heading
        in api.mdx whose slug == the anchor.

        Parse ALL headings (levels 1-6, fence-aware -- mirroring remark-api.mjs)
        and slug each heading TEXT via the generator's slug(). Organizer headings
        that map to no manifest anchor are IGNORED (the guard is scoped to the
        intersection with manifest anchors, per refine MEDIUM-2). Exception
        anchors are excluded (uorb4's fenced tree).

        FAILS today: the current headings embed full signatures (e.g.
        '### `Node(name, *, mode, ...)`') which slug to garbage, so the 4
        collision anchors (node-function/node-model/tool-function/tool-model) and
        50+ other symbols have ZERO matching headings.
        """
        gen = _load_gen_api_manifest()
        assert API_MDX_PATH.exists(), f"reference page missing at {API_MDX_PATH}"
        heading_slugs = _reference_heading_slugs(API_MDX_PATH.read_text(), gen)
        counts = Counter(heading_slugs)

        problems: list[str] = []
        for symbol in self._non_exception_symbols(gen):
            anchor = symbol["anchor"]
            seen = counts.get(anchor, 0)
            if seen != 1:
                problems.append(
                    f"symbol {symbol['name']!r} (kind={symbol['kind']}, "
                    f"anchor={anchor!r}) is emitted by {seen} heading(s), "
                    f"expected EXACTLY 1 heading whose slug == the anchor"
                )
        assert not problems, (
            "single-heading-per-anchor violated -- every NON-exception manifest "
            "anchor must have exactly one api.mdx heading slugging to it "
            "(Stage C, neograph-kec0k). Signature-in-heading sections slug to "
            "garbage; replace them with the manifest-generated headings:\n"
            + "\n".join(problems)
        )

    def test_collision_anchors_each_have_exactly_one_kind_namespaced_heading(self):
        """The 4 disambiguated collision anchors (node-function/node-model/
        tool-function/tool-model) each have exactly one matching heading.

        This is the motivating defect (the addendum): today the collision-pair
        prose refs stay INERT (0 autolinks) because no heading slugs to these
        anchors. Kind-namespaced heading TEXT ('node (function)' etc.) fixes it.

        FAILS today: 0 headings slug to any of the 4 collision anchors.
        """
        gen = _load_gen_api_manifest()
        heading_slugs = Counter(_reference_heading_slugs(API_MDX_PATH.read_text(), gen))
        manifest = gen.build_manifest()
        symbols = manifest.get("symbols") or manifest.get("api_symbols") or []
        collision_anchors = sorted(
            s["anchor"]
            for s in symbols
            if s.get("kind") != "exception" and "-" in s["anchor"]
            and s["anchor"].rsplit("-", 1)[1] in gen._KIND_ANCHOR_TAG.values()
        )
        assert collision_anchors, (
            "expected disambiguated collision anchors in the manifest (e.g. "
            "node-function/node-model) -- anchor scheme regressed."
        )
        bad = {a: heading_slugs.get(a, 0) for a in collision_anchors if heading_slugs.get(a, 0) != 1}
        assert not bad, (
            f"collision anchors without exactly one matching heading: {bad}. "
            f"Each must have a kind-namespaced heading ('node (function)' -> "
            f"#node-function, 'Node (model)' -> #node-model, etc.) so the prose "
            f"refs resolve instead of staying silently inert."
        )

    # ── PROC-2 slip meta-test for the sole regex constant introduced here ──
    def test_slip_heading_re(self):
        """Slip meta-test for _HEADING_RE (PROC-2): prove it matches real
        markdown headings across levels 1-6 and REJECTS non-headings, so it
        cannot silently harvest the wrong lines (or miss real headings).
        """
        # Matches all levels 1-6, capturing the TEXT with trailing ws trimmed.
        assert _HEADING_RE.match("# Top").group(1) == "Top"
        assert _HEADING_RE.match("### node (function)  ").group(1) == "node (function)"
        assert _HEADING_RE.match("###### Deep").group(1) == "Deep"
        # Rejects a 7th-level "heading" (github markdown caps at 6).
        assert _HEADING_RE.match("####### too deep") is None
        # Rejects a hash with no following whitespace (a code comment like
        # '#nospace', not a heading) and ordinary prose.
        assert _HEADING_RE.match("#nospace") is None
        assert _HEADING_RE.match("not a heading") is None
        # Rejects an empty-text heading marker (needs \s+ then >=1 char).
        assert _HEADING_RE.match("### ") is None
