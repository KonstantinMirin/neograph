"""Stage C (neograph-rfl7b) RED guard: manifest anchors must be page-unique.

Stage C renders committed reference-doc sections (per-symbol signature blocks,
Pydantic field tables, the error-hierarchy tree) FROM
``website/src/data/api-manifest.json``. The Core Invariant of the ticket:

    every generated heading's Starlight slug MUST equal the manifest anchor for
    that symbol -- so the manifest owns both ends of every cross-link.

That invariant is UNSATISFIABLE while two distinct public symbols share one
anchor: on a single rendered page github-slugger dedups the second heading to
``<anchor>-1``, which is NOT in the manifest, so the Stage B HARD tier's
owner-anchor target no longer exists and the SOFT-tier autolink points at the
wrong symbol. The manifest MUST therefore assign every symbol a distinct anchor.

VERIFIED collisions in the committed manifest (2026-07-10):
  - ``node`` (function, the @node decorator)  and ``Node`` (pydantic_model)
    both slug to ``node``.
  - ``tool`` (function, the @tool decorator)  and ``Tool`` (pydantic_model)
    both slug to ``tool``.

Step 1 of the rfl7b plan (DECISION 1 = kind-namespace anchors in Stage A)
disambiguates both pairs so each colliding symbol gets a distinct deterministic
anchor. This module pins that end state and FAILS today (the two duplicates
above still exist), which is the intended TDD-red step.

Companion contracts (M3 anchor=slug(name), the JS-authoritative slug snapshot,
freshness) live in ``tests/test_guards_api_manifest.py`` and are re-pinned by
the implementer for the new scheme.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_MANIFEST_PATH = REPO_ROOT / "website" / "src" / "data" / "api-manifest.json"


def _load_committed_manifest() -> dict:
    """Load the committed core manifest -- the artifact Stage B/C consume.

    Stage C renders FROM this committed JSON (never re-introspects neograph), and
    the Stage B remark plugin validates/autolinks against it, so the anchor
    uniqueness contract must hold on the committed file itself.
    """
    assert CORE_MANIFEST_PATH.exists(), (
        f"core manifest not committed at {CORE_MANIFEST_PATH}. "
        f"Run: python scripts/gen_api_manifest.py"
    )
    return json.loads(CORE_MANIFEST_PATH.read_text())


class TestManifestAnchorUniqueness:
    """Every public symbol needs a distinct manifest anchor (the Stage C crux).

    Without this, the Core Invariant slug(heading) == manifest anchor cannot hold
    for both members of a colliding pair on one page.
    """

    def test_all_symbol_anchors_are_unique(self):
        """No two symbols may share an anchor.

        The manifest owns both ends of every cross-link: a generated heading's
        Starlight slug is looked up as the anchor. Two symbols with the same
        anchor mean one of the two generated headings slugs to a value the
        manifest does not contain (github-slugger appends ``-1``), breaking the
        HARD-tier owner-anchor target and mis-pointing the SOFT-tier autolink.

        FAILS today: node/Node -> 'node' and tool/Tool -> 'tool' collide.
        """
        manifest = _load_committed_manifest()
        symbols = manifest["symbols"]
        counts = Counter(s["anchor"] for s in symbols)
        duplicates = {anchor: n for anchor, n in counts.items() if n > 1}
        # Human-readable map of which symbol names share each duplicated anchor.
        offenders = {
            anchor: sorted(
                f"{s['name']} (kind={s['kind']})"
                for s in symbols
                if s["anchor"] == anchor
            )
            for anchor in duplicates
        }
        assert not duplicates, (
            f"{len(duplicates)} manifest anchor(s) are shared by >1 symbol: "
            f"{offenders}. Kind-namespace the colliding anchors in "
            f"scripts/gen_api_manifest.py so every symbol has a distinct, "
            f"deterministic anchor (Stage C DECISION 1)."
        )

    def test_verified_name_class_collisions_are_disambiguated(self):
        """The two VERIFIED function/class name collisions must have distinct anchors.

        node(function, the @node decorator) vs Node(class) and tool(function, the
        @tool decorator) vs Tool(class) currently both slug to a single anchor.
        After kind-namespacing they must differ, or the class section and the
        decorator section cannot both live on the reference page with
        manifest-owned anchors.

        FAILS today: both pairs still share one anchor.
        """
        manifest = _load_committed_manifest()
        by_name: dict[str, dict] = {s["name"]: s for s in manifest["symbols"]}
        for fn_name, cls_name in (("node", "Node"), ("tool", "Tool")):
            assert fn_name in by_name, f"{fn_name!r} missing from manifest symbols"
            assert cls_name in by_name, f"{cls_name!r} missing from manifest symbols"
            fn_anchor = by_name[fn_name]["anchor"]
            cls_anchor = by_name[cls_name]["anchor"]
            assert fn_anchor != cls_anchor, (
                f"{fn_name!r} (kind={by_name[fn_name]['kind']}) and "
                f"{cls_name!r} (kind={by_name[cls_name]['kind']}) share anchor "
                f"{fn_anchor!r}. Kind-namespace one of the pair so each gets a "
                f"distinct deterministic anchor."
            )

    def test_every_anchor_is_slug_stable(self):
        """Each anchor must be a fixed point of ``slug`` -- 'manifest owns both ends'.

        A heading whose text slugs to X only resolves to anchor X if X is itself
        slug-stable (``slug(X) == X``); otherwise Starlight would emit a
        different id than the manifest records. This holds today and must keep
        holding after the anchors are kind-namespaced -- it pins that the
        disambiguated anchors remain reproducible by the github-slugger port.
        """
        import importlib.util

        script = REPO_ROOT / "scripts" / "gen_api_manifest.py"
        spec = importlib.util.spec_from_file_location("gen_api_manifest", script)
        assert spec is not None and spec.loader is not None
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)

        manifest = _load_committed_manifest()
        for symbol in manifest["symbols"]:
            anchor = symbol["anchor"]
            assert gen.slug(anchor) == anchor, (
                f"symbol {symbol['name']!r} anchor {anchor!r} is not slug-stable "
                f"(slug({anchor!r}) == {gen.slug(anchor)!r}); Starlight would emit "
                f"a different heading id than the manifest records."
            )
