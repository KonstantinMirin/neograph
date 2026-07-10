"""Regenerate the verifiable-docs API manifest from the live public surface.

Walks ``neograph.__all__`` (+ ``neograph_mcp.__all__`` when the ``mcp`` extra is
installed), introspects each symbol, and emits committed JSON the docs build
verifies against. This is the Stage A foundation (neograph-ryn4h) that Stage B
(remark plugin) and Stage C (generated reference sections) consume: every anchor
and signature the docs cross-link comes from this manifest.

Run after any change to the public surface::

    uv run python scripts/gen_api_manifest.py                 # core only
    uv run --extra dev --extra mcp python scripts/gen_api_manifest.py   # core + mcp

A freshness guard (``tests/test_guards_api_manifest.py``) regenerates the
manifest in-memory and diffs it against the committed files, so any
public-surface change without a manifest commit fails the default pytest suite
-- the rustdoc "docs build from the code" contract displaced into the test
suite (because Amplify cannot run Python).

Design decisions pinned by the ryn4h architect review (see bead notes
``## Refinements``):

  - M1: field annotations use Python ``str(annotation)`` (the audience is human
    Python devs, not LLMs -- describe_type's TS notation is the wrong audience).
  - M2: declared-only model fields via MRO subtraction (``ForwardConstruct``, a
    pure subclass of ``Construct``, emits zero fields, not Construct's 8).
  - M3: per-symbol ``anchor = slug(name)`` (stable across signature changes).
  - Option (c): core + mcp split into two committed files; the mcp file is
    regenerated + guarded only when the ``mcp`` extra is present.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import sys
from dataclasses import MISSING
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any

# Object reprs embed process-specific addresses (``<function f at 0x10d111260>``)
# that would make the committed manifest non-deterministic across runs. Collapse
# every ``0x...`` to a stable placeholder so regeneration is byte-identical.
_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


def _sanitize_addresses(text: str) -> str:
    return _ADDRESS.sub("0x…", text)

# Put src/ on the path (mirrors scripts/regen_schema.py:17).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic_core import PydanticUndefined  # noqa: E402

import neograph  # noqa: E402
import neograph.errors as _errors_module  # noqa: E402
from neograph import NeographError  # noqa: E402
from neograph.di import DI_TEMPLATE_KINDS  # noqa: E402
from neograph.lint import LINT_KIND_META  # noqa: E402

try:  # pydantic v2 lives under pydantic.fields
    from pydantic import BaseModel  # noqa: E402
except ImportError:  # pragma: no cover - pydantic is a hard dep
    BaseModel = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
WEBSITE_DATA = REPO_ROOT / "website" / "src" / "data"
CORE_MANIFEST_PATH = WEBSITE_DATA / "api-manifest.json"
MCP_MANIFEST_PATH = WEBSITE_DATA / "api-manifest-mcp.json"

# -- slug (github-slugger 2.0.0-compatible) -----------------------------------

# github-slugger 2.0.0 strips these characters (its `specials` char class) and
# replaces whitespace with '-'. ASCII letters, digits, '_' and '-' survive. The
# Python port must be byte-identical so anchors match the headings Starlight
# (transitive via @astrojs/markdown-remark) emits -- pinned by the slug snapshot
# test in tests/test_guards_api_manifest.py (JS-authoritative values).
_SPECIAL_CHARS: set[str] = set()
_SPECIAL_CHARS.update(chr(cp) for cp in range(0x2000, 0x2070))  #  -⁯
_SPECIAL_CHARS.update(chr(cp) for cp in range(0x2E00, 0x2E80))  # ⸀-⹿
_SPECIAL_CHARS.update(
    c
    for c in (
        "\\", "'", '"', "!", "#", "$", "%", "&", "(", ")", "*", "+", ",",
        ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "]", "^", "`",
        "{", "|", "}", "~",
    )
)


def slug(name: str) -> str:
    """github-slugger 2.0.0-compatible slug -- the cross-stage anchor contract.

    Lowercase, drop the github-slugger ``specials`` character class, then turn
    every whitespace run into a single ``-``. Underscores and dashes survive.
    """
    lowered = name.lower()
    stripped = "".join(ch for ch in lowered if ch not in _SPECIAL_CHARS)
    # Collapse any run of whitespace into a single dash (matches \s -> '-').
    out: list[str] = []
    prev_dash = False
    for ch in stripped:
        if ch.isspace():
            if not prev_dash:
                out.append("-")
                prev_dash = True
        else:
            out.append(ch)
            prev_dash = False
    return "".join(out)


# -- helpers ------------------------------------------------------------------


def _first_line(doc: str | None) -> str:
    """First non-empty line of a docstring, stripped; '' if absent."""
    if not doc:
        return ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _render_annotation(ann: Any) -> str:
    """Render a type annotation as a clean Python type expression for docs.

    ``str()`` on a bare type yields ``<class 'str'>`` (wrong for a reference);
    on typing generics it yields ``typing.List[...]`` (verbose). Return the
    qualname for bare classes (``None`` for NoneType) and strip the ``typing.``
    prefix otherwise. Deterministic and address-sanitized.
    """
    if ann is type(None):
        return "None"
    if isinstance(ann, type):
        return getattr(ann, "__qualname__", ann.__name__)
    return _sanitize_addresses(str(ann).replace("typing.", ""))


def _safe_signature(obj: Any) -> str:
    """Best-effort ``inspect.signature`` string, address-sanitized; '' if unavailable."""
    try:
        return _sanitize_addresses(str(inspect.signature(obj)))
    except (TypeError, ValueError):
        return ""


def _pydantic_field_default(info: Any) -> str | None:
    """Render a Pydantic FieldInfo default deterministically (no addresses)."""
    if info.default is not PydanticUndefined:
        return _sanitize_addresses(repr(info.default))
    factory = getattr(info, "default_factory", None)
    if factory is not None:
        fname = getattr(factory, "__name__", None) or _sanitize_addresses(repr(factory))
        return f"<factory:{fname}>"
    return None


def _pydantic_signature(cls: type) -> str:
    """Deterministic constructor signature built from ``model_fields``.

    ``inspect.signature(cls)`` on a Pydantic model renders the field validators
    as ``Annotated[..., PlainValidator(func=<function _validate_type_spec at
    0x...>)]`` -- addresses that change every run and break byte-stability.
    Building from ``model_fields`` (all of them -- a constructor takes every
    field) is both deterministic and a cleaner reference rendering.
    """
    parts: list[str] = []
    for name, info in cls.model_fields.items():
        ann = _render_annotation(info.annotation)
        default = _pydantic_field_default(info)
        if info.is_required():
            parts.append(f"{name}: {ann}")
        elif default is None:
            parts.append(f"{name}: {ann} = None")
        else:
            parts.append(f"{name}: {ann} = {default}")
    return "(" + ", ".join(parts) + ")"


def _classify(obj: Any) -> str:
    """Classify a public symbol into a render hint for Stage C."""
    if inspect.isclass(obj):
        if issubclass(obj, BaseException):
            return "exception"
        if BaseModel is not None and issubclass(obj, BaseModel):
            return "pydantic_model"
        if hasattr(obj, "__dataclass_fields__"):
            return "dataclass"
        return "class"
    if callable(obj):
        return "function"
    return "other"


def _declared_pydantic_fields(cls: type) -> list[dict[str, Any]]:
    """Declared-only Pydantic v2 fields (M2: MRO subtraction).

    ``cls.model_fields`` INCLUDES inherited fields; subtract every ancestor's
    ``model_fields`` so a pure subclass (e.g. ForwardConstruct) contributes an
    empty list instead of duplicating its parent's fields. ``model_fields``
    already excludes PrivateAttrs (``_sidecar``/``_param_res``/``_scripted_shim``).
    """
    inherited = {
        field_name
        for base in cls.__mro__[1:]
        for field_name in getattr(base, "model_fields", {})
    }
    entries: list[dict[str, Any]] = []
    for name, info in cls.model_fields.items():
        if name in inherited:
            continue
        entries.append(
            {
                "name": name,
                "annotation": _render_annotation(info.annotation),  # M1: clean Python notation
                "required": info.is_required(),
                "default": _pydantic_field_default(info),
            }
        )
    return entries


def _declared_dataclass_fields(cls: type) -> list[dict[str, Any]]:
    """Declared dataclass fields (Leaf classes in practice; MRO-agnostic)."""
    entries: list[dict[str, Any]] = []
    for field in dc_fields(cls):
        required = field.default is MISSING and field.default_factory is MISSING
        if field.default is not MISSING:
            default: Any = repr(field.default)
        elif field.default_factory is not MISSING:  # type: ignore[misc]
            default = f"<factory:{field.default_factory.__name__}>"
        else:
            default = None
        entries.append(
            {
                "name": field.name,
                "annotation": _render_annotation(field.type),
                "required": required,
                "default": default,
            }
        )
    return entries


def _bases_in_all(cls: type, all_names: set[str]) -> list[str]:
    """Direct base class names that are themselves public (linkable in docs)."""
    return [
        base.__name__
        for base in cls.__bases__
        if isinstance(base, type) and base.__name__ in all_names
    ]


def _symbol_entry(name: str, obj: Any, all_names: set[str]) -> dict[str, Any]:
    """Build one manifest entry for a public symbol."""
    kind = _classify(obj)
    entry: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "anchor": slug(name),  # M3: anchor on NAME, not signature
        "doc": _first_line(inspect.getdoc(obj)),
    }
    if kind == "pydantic_model":
        entry["bases"] = _bases_in_all(obj, all_names)
        entry["signature"] = _pydantic_signature(obj)
        entry["fields"] = _declared_pydantic_fields(obj)
    elif kind in ("dataclass", "exception", "class"):
        entry["bases"] = _bases_in_all(obj, all_names)
        entry["signature"] = _safe_signature(obj)
    elif kind == "function":
        entry["signature"] = _safe_signature(obj)
    if kind == "dataclass":
        entry["fields"] = _declared_dataclass_fields(obj)
    if kind == "pydantic_model":
        entry["fields"] = _declared_pydantic_fields(obj)
    elif kind == "dataclass":
        entry["fields"] = _declared_dataclass_fields(obj)
    return entry


# Kind -> anchor-disambiguation tag (Stage C / neograph-rfl7b, DECISION 1).
#
# Two DISTINCT public symbols can share one bare ``slug(name)`` -- VERIFIED:
# ``node`` (the @node decorator function) / ``Node`` (the pydantic model) and
# ``tool`` / ``Tool`` each slug to a single value. On one rendered reference
# page github-slugger would dedup the second heading to ``<anchor>-1``, a value
# the manifest does not contain -- breaking the Core Invariant that the manifest
# owns both ends of every cross-link. For each symbol in a colliding group the
# anchor is kind-namespaced: ``f"{slug(name)}-{tag}"``. The tag is chosen so the
# result is (a) slug-stable -- ``slug(anchor) == anchor`` -- and (b) reproducible
# as the slug of a real heading, e.g. heading ``node (function)`` -> github-
# slugger -> ``node-function``. Non-colliding symbols keep the bare ``slug(name)``.
_KIND_ANCHOR_TAG: dict[str, str] = {
    "function": "function",
    "pydantic_model": "model",
    "dataclass": "dataclass",
    "exception": "exception",
    "class": "class",
    "other": "other",
    "missing": "missing",
}


def _disambiguate_anchors(entries: list[dict[str, Any]]) -> None:
    """Kind-namespace anchors for symbols whose bare ``slug(name)`` collides.

    Two-pass, in place: pass 1 counts how many symbols share each base slug;
    pass 2 rewrites the anchor of every symbol in a colliding group to
    ``f"{base}-{tag}"`` where ``tag`` derives from the symbol's kind
    (``_KIND_ANCHOR_TAG``). Non-colliding symbols are left untouched (their
    anchor stays ``slug(name)`` as set by ``_symbol_entry``). Every rewritten
    anchor is slug-stable and reproducible as ``slug("<name> (<tag>)")``.
    """
    from collections import Counter

    base_counts = Counter(slug(entry["name"]) for entry in entries)
    for entry in entries:
        base = slug(entry["name"])
        if base_counts[base] > 1:
            tag = _KIND_ANCHOR_TAG.get(entry["kind"], entry["kind"])
            entry["anchor"] = f"{base}-{tag}"


def _build_symbols(module: Any, names: list[str]) -> list[dict[str, Any]]:
    """Walk an ``__all__`` against a module into manifest symbol entries."""
    all_names = set(names)
    entries: list[dict[str, Any]] = []
    for name in names:
        obj = getattr(module, name, None)
        if obj is None:
            # Declared in __all__ but absent -- surface loudly rather than skip.
            entries.append(
                {"name": name, "kind": "missing", "anchor": slug(name), "doc": ""}
            )
            continue
        entries.append(_symbol_entry(name, obj, all_names))
    # Two-pass anchor disambiguation: kind-namespace any colliding base slugs so
    # every symbol has a distinct, deterministic, slug-stable anchor. Applied
    # here (not in build_manifest) so BOTH the core and mcp manifests inherit the
    # same scheme wherever anchors are emitted (neograph-rfl7b DECISION 1).
    _disambiguate_anchors(entries)
    return entries


def _exception_hierarchy() -> list[dict[str, Any]]:
    """The NeographError tree, parent-linked (mirrors test_error_hierarchy.py)."""
    exc_classes = [
        obj
        for obj in vars(_errors_module).values()
        if isinstance(obj, type) and issubclass(obj, NeographError)
    ]
    hierarchy: list[dict[str, Any]] = []
    for exc in sorted(exc_classes, key=lambda c: c.__name__):
        neo_parents = [
            base.__name__
            for base in exc.__bases__
            if isinstance(base, type) and issubclass(base, NeographError)
        ]
        hierarchy.append(
            {
                "name": exc.__name__,
                "parent": neo_parents[0] if neo_parents else None,
                "anchor": slug(exc.__name__),
                "doc": _first_line(inspect.getdoc(exc)),
            }
        )
    return hierarchy


_DI_KIND_NAMES: frozenset[str] = frozenset(k.value for k in DI_TEMPLATE_KINDS)


def _literal_kind_required_sites() -> dict[str, set[bool]]:
    """Co-derive ``{kind: {required_bool, ...}}`` from lint.py's LintIssue sites.

    Walks every ``ast.Call`` that carries a LITERAL ``kind=<str>`` and records the
    LITERAL ``required=`` at that same Call (defaulting to ``False`` — the
    LintIssue.required default). A set because a kind may be emitted at multiple
    sites with conflicting ``required`` (the sanctioned dual case). DI kinds use
    ``kind=binding.kind.value`` (a variable) and are invisible to this walk.
    """
    lint_path = REPO_ROOT / "src" / "neograph" / "lint.py"
    tree = ast.parse(lint_path.read_text())
    out: dict[str, set[bool]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kind_val: str | None = None
        required_val = False
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


def extract_lint_issue_kinds() -> list[dict[str, str]]:
    """Build the enriched ``lint_issue_kinds`` manifest section (neograph-uw54v).

    Emits one ``{kind, severity, meaning}`` object per kind lint() can emit,
    sorted by ``kind``. Severity + meaning are MANIFEST-OWNED, sourced from the
    authoritative ``neograph.lint.LINT_KIND_META`` registry so the reference doc
    table (neograph-cvjfm) renders from here instead of hand-authoring it.

    Zero-drift is enforced two ways, both FAIL LOUD at regen time:
      * COMPLETENESS — the kind set discovered from the code (literal ``kind=``
        sites AST-walked from lint.py, UNIONed with the 4 DI kinds from
        ``DI_TEMPLATE_KINDS`` that the walk structurally cannot see) must equal
        ``LINT_KIND_META.keys()``; a kind added without a registry entry (or vice
        versa) breaks regen.
      * SEVERITY BINDING (refinement neograph-uqy66.52) — for every single-site
        literal kind the registry severity must equal ``'ERROR' if required else
        'WARN'`` co-derived from the ``required=`` at its emission site. Sanctioned
        exceptions: ``loop_condition_none_unsafe`` is dual (``WARN/ERROR``); the 4
        DI kinds are ``varies``. So a future ``required=`` flip cannot silently
        drift the severity column.
    """
    literal_sites = _literal_kind_required_sites()
    code_kinds = set(literal_sites) | _DI_KIND_NAMES

    if code_kinds != set(LINT_KIND_META):
        missing = sorted(code_kinds - set(LINT_KIND_META))
        extra = sorted(set(LINT_KIND_META) - code_kinds)
        raise ValueError(
            "LINT_KIND_META is out of sync with the kinds lint() emits. "
            f"missing registry entries for {missing}; "
            f"registry has entries with no emission site {extra}. "
            "Add/remove entries in neograph.lint.LINT_KIND_META."
        )

    for kind, required_values in literal_sites.items():
        stored = LINT_KIND_META[kind].severity
        if kind in _DI_KIND_NAMES:  # unreachable (DI kinds are not literal), defensive
            continue
        if kind == "loop_condition_none_unsafe":
            if required_values != {True, False}:
                raise ValueError(
                    f"{kind!r} is the sanctioned dual-severity exception but is no "
                    f"longer emitted at both required=True and required=False "
                    f"(saw {required_values}); re-check LINT_KIND_META."
                )
            if stored != "WARN/ERROR":
                raise ValueError(
                    f"dual-severity kind {kind!r} must be 'WARN/ERROR', got {stored!r}."
                )
            continue
        if len(required_values) != 1:
            raise ValueError(
                f"literal kind {kind!r} emitted with conflicting required= "
                f"{required_values}; only loop_condition_none_unsafe may be dual."
            )
        expected = "ERROR" if next(iter(required_values)) else "WARN"
        if stored != expected:
            raise ValueError(
                f"LINT_KIND_META[{kind!r}].severity={stored!r} DRIFTED from the "
                f"code-derived {expected!r} ('ERROR' if required else 'WARN'). "
                "Fix the registry severity to match the required= at the site."
            )

    for kind in _DI_KIND_NAMES:
        if LINT_KIND_META[kind].severity != "varies":
            raise ValueError(
                f"DI kind {kind!r} severity must be 'varies' (runtime "
                f"binding.required-dependent), got {LINT_KIND_META[kind].severity!r}."
            )

    return [
        {"kind": kind, "severity": meta.severity, "meaning": meta.meaning}
        for kind, meta in sorted(LINT_KIND_META.items())
    ]


def build_manifest() -> dict[str, Any]:
    """Core manifest: neograph.__all__ + the error tree + lint issue kinds."""
    names = list(neograph.__all__)
    return {
        "generated_from": "neograph.__all__",
        "symbols": _build_symbols(neograph, names),
        "exception_hierarchy": _exception_hierarchy(),
        "lint_issue_kinds": extract_lint_issue_kinds(),
    }


def _reference_heading(entry: dict[str, Any]) -> str:
    """The '### ' heading whose github-slugger slug == the symbol's manifest anchor.

    Non-colliding symbols use the bare name ('### compile' -> #compile). Symbols
    whose bare ``slug(name)`` collided (anchor was kind-namespaced to
    ``f"{base}-{tag}"`` by ``_disambiguate_anchors``) use ``'### {name} ({tag})'``
    so github-slugger reproduces the anchor: '### node (function)' -> #node-function.
    """
    name = entry["name"]
    anchor = entry["anchor"]
    base = slug(name)
    if anchor != base and anchor.startswith(f"{base}-"):
        tag = anchor[len(base) + 1:]
        return f"### {name} ({tag})"
    return f"### {name}"


def _reference_field_table(fields: list[dict[str, Any]]) -> list[str]:
    """A markdown field table (Field/Type/Required/Default) for a fielded symbol."""
    rows = ["| Field | Type | Required | Default |", "|-------|------|----------|---------|"]
    for f in fields:
        default = "" if f.get("default") is None else f"`{f['default']}`"
        required = "yes" if f.get("required") else "no"
        rows.append(f"| `{f['name']}` | `{f['annotation']}` | {required} | {default} |")
    return rows


def render_reference_sections() -> str:
    """Render the per-symbol reference region for api.mdx (Stage C, neograph-kec0k).

    Emits ONE contiguous block: for every NON-exception manifest symbol (exceptions
    are owned by uorb4's fenced error tree), in a stable order sorted by anchor, a
    section with a heading whose slug == the manifest anchor, the signature in a
    fenced ``python`` code block (fenced so the remark harvester emits no anchor
    from it), and a Pydantic/dataclass field table when the symbol has fields.

    The return value is the exact text that lives between the
    ``{/* GEN:reference-sections START */}`` / ``END`` sentinels in api.mdx; the
    freshness guard asserts committed-region == this, byte-for-byte.
    """
    symbols = [s for s in _build_symbols(neograph, list(neograph.__all__)) if s.get("kind") != "exception"]
    symbols.sort(key=lambda s: s["anchor"])
    sections: list[str] = []
    for entry in symbols:
        block = [_reference_heading(entry), ""]
        signature = entry.get("signature")
        if signature:
            # Skip-mark the fence: a signature is a declaration, not a runnable
            # statement, so Stage D (test_docs_snippets.py) must not execute it.
            block += [
                "{/* test-skip: generated API signature (declaration, not runnable) */}",
                "```python",
                f"{entry['name']}{signature}",
                "```",
                "",
            ]
        fields = entry.get("fields")
        if fields:
            block += _reference_field_table(fields) + [""]
        sections.append("\n".join(block).rstrip())
    # Leading + trailing newline so the sentinel markers sit on their own lines.
    return "\n" + "\n\n".join(sections) + "\n"


def build_mcp_manifest() -> dict[str, Any]:
    """MCP manifest: neograph_mcp.__all__ (raises ImportError w/o the extra)."""
    import neograph_mcp  # noqa: F401  -- _require_mcp() gates at import time

    names = list(neograph_mcp.__all__)
    return {
        "generated_from": "neograph_mcp.__all__",
        "symbols": _build_symbols(neograph_mcp, names),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write canonical JSON: indent=2, sort_keys=True, trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    """Regenerate both manifest files. Returns 0 on success."""
    _write_json(CORE_MANIFEST_PATH, build_manifest())
    print(f"wrote {CORE_MANIFEST_PATH.relative_to(REPO_ROOT)}")
    try:
        _write_json(MCP_MANIFEST_PATH, build_mcp_manifest())
        print(f"wrote {MCP_MANIFEST_PATH.relative_to(REPO_ROOT)}")
    except ImportError:
        # Option (c): the mcp extra is absent -- leave the committed mcp file
        # untouched (it persists from the last extra-present regeneration) and
        # warn. The mcp guard is skipif-not-_HAS_MCP, so default CI stays green.
        print(
            f"skip {MCP_MANIFEST_PATH.relative_to(REPO_ROOT)}: "
            "mcp extra not installed (run with --extra mcp to regenerate it)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
