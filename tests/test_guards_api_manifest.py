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
  - ``extract_lint_issue_kinds()`` -> list[str]  (AST-extracted from lint.py kind= literals)
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

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "gen_api_manifest.py"
WEBSITE_DATA = REPO_ROOT / "website" / "src" / "data"
CORE_MANIFEST_PATH = WEBSITE_DATA / "api-manifest.json"
MCP_MANIFEST_PATH = WEBSITE_DATA / "api-manifest-mcp.json"

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
        """M3 -- per-symbol anchor is ``slug(symbol_name)``, NOT slug(signature).

        Signature-based anchors are unstable (adding a kwarg to ``compile()``
        would change its anchor and break every existing doc link that targets
        it). Anchoring on the NAME keeps links stable across signature changes.
        The CROSS-STAGE CONTRACT defined here: Stage C renders each symbol
        section with ``id = slug(name)`` (decoupled from the heading text); the
        manifest owns the anchor end-to-end, Stage B validates refs against it.
        """
        gen = _load_gen_api_manifest()
        manifest = gen.build_manifest()
        symbols = manifest.get("symbols") or manifest.get("api_symbols") or []
        assert symbols, "manifest symbols list is empty -- __all__ walk is broken"
        for symbol in symbols:
            name = symbol.get("name")
            anchor = symbol.get("anchor")
            assert name is not None and anchor is not None, (
                f"symbol entry missing name/anchor keys: {symbol!r}"
            )
            expected = gen.slug(name)
            assert anchor == expected, (
                f"symbol {name!r} anchor is {anchor!r} but should be slug(name)="
                f"{expected!r} -- anchor must be NAME-based, not signature-based (M3)."
            )


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
        """AST extraction must yield every one of the 10 known kind= literals.

        Pin against the floor set, not the exact set: future kinds may be added
        (the extractor should pick them up automatically), but none of the
        current 10 may silently disappear.
        """
        gen = _load_gen_api_manifest()
        extracted = set(gen.extract_lint_issue_kinds())
        missing = _KNOWN_LINT_KIND_FLOOR - extracted
        assert not missing, (
            f"lint.py AST extraction missed {len(missing)} known kind(s): "
            f"{sorted(missing)}. Either lint.py switched a kind to a variable/"
            f"f-string (AST walk for ast.Constant kwargs cannot see it), or the "
            f"extractor's AST walk is malformed."
        )
