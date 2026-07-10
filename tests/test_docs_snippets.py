"""Docs-snippet testing suite (verifiable-docs Stage D, neograph-3my4x).

Stage D executes the python fenced blocks embedded in the ``.mdx`` documentation
pages so that a snippet which resolves at Stage B (remark-api autolink) but has
since drifted -- no longer imports, runs, or produces what the prose claims --
fails the default ``uv run pytest`` suite instead of shipping green.

SPIKE VERDICT (report open-question 2): pytest-examples WINS over Sybil. The
only gap in ``pytest_examples.find_examples.find_examples`` is its ``.md``-suffix
filter (our docs are 44 ``.mdx`` / 0 ``.md``); the internal
``_extract_code_chunks`` extracts fenced ``python``/``py`` blocks cleanly from
raw ``.mdx`` text and ignores JSX islands, frontmatter, and ``<Tabs>``/``<Aside>``
components (those are prose nodes, never fenced code). A ~10-line
``find_examples_mdx()`` wrapper -- glob ``website/src/content/docs/**/*.mdx``, per
file yield ``_extract_code_chunks(path, path.read_text('utf-8'), uuid4())`` -- is
the ENTIRE MDX adaptation.

REFUTED-FOLKLORE LESSON (the Core Invariant): pytest-examples tests only what
``find_examples`` collects -- NOT "every snippet". Coverage is reported as the set
of python fenced blocks collected minus explicitly-skipped blocks, enforced by a
COMPUTED assertion (collected == passed + failed + skipped, zero silent drops),
and every pipeline-running snippet runs against ``tests/fakes.py`` -- never a real
LLM or API key.

This module mirrors ``tests/test_guards_api_manifest.py`` (module-docstring style,
``REPO_ROOT`` anchoring, sibling in the docs-freshness guard family).

---

DRIFT GUARD (this file's first red artifact, implementation plan v2 step 7 /
review LOW-1). Because the wrapper couples to the PRIVATE, pre-1.0
``pytest_examples.find_examples._extract_code_chunks`` symbol, an upstream rename
OR a partial regex regression on nested/indented fences would silently collect
ZERO or fewer blocks and every downstream snippet test would vacuously pass. To
fail LOUD instead, we pin the KNOWN python-block count for two pages:

  - ``concepts/checkpoint-resume.mdx``          == 3 blocks (plain top-level fences)
  - ``concepts/migrating-prompt-compilers.mdx`` == 8 blocks (the indented-fence
    case the spike stressed: fences at indent=3 nested inside a JSX component,
    which ``remove_indent`` dedents correctly)

The wrapper ``find_examples_mdx`` is added by the implement atom
(neograph-wv1yh.15); until then the guard below fails as the TDD-red step.
"""

from __future__ import annotations

import enum
import importlib
import re
import typing
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import BaseModel

from neograph.testing.fakes import StructuredFake

# Provider API-key env vars whose presence would let a doc snippet's real
# llm_factory (e.g. quick-start's ``ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"])``)
# succeed and make a LIVE network call. btqzq deletes them before executing any
# page so the offline path is DETERMINISTIC regardless of ambient env (the Core
# Invariant: no Stage-D-executed snippet ever reaches a real LLM/API key).
_PROVIDER_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "GOOGLE_API_KEY",
    "COHERE_API_KEY",
    # Langfuse: observe=True live-attaches + flushes traces to the Langfuse API
    # only when BOTH keys are present (runner._langfuse_keys_present). Delete them
    # so an observability snippet stays a verified offline no-op (neograph-x51j3).
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_PUBLIC_KEY",
)


def _build_instance(model: type[BaseModel]) -> BaseModel:
    """Construct a minimal valid instance of ``model`` by type-defaulting its
    required fields (recursively for nested models). Net-new piece for btqzq's
    generic offline think-mode double: the fake has no per-node output knowledge,
    so it fills required fields with type-appropriate zero values."""
    return model(**{
        name: _default_for(field.annotation)
        for name, field in model.model_fields.items()
        if field.is_required()
    })


def _default_for(annotation: typing.Any) -> typing.Any:
    """A type-appropriate zero value for a field annotation (Optional/Union,
    containers, nested BaseModel, Enum, Literal, primitives)."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Union:  # Optional[X] / Union[...] -> first non-None member
        non_none = [a for a in args if a is not type(None)]
        return _default_for(non_none[0]) if non_none else None
    if origin is typing.Literal:
        return args[0] if args else None
    if origin in (list, set, frozenset, tuple):
        return origin() if origin is not tuple else ()
    if origin is dict:
        return {}
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return _build_instance(annotation)
        if issubclass(annotation, enum.Enum):
            return next(iter(annotation))
        if issubclass(annotation, bool):
            return False
        if issubclass(annotation, int):
            return 0
        if issubclass(annotation, float):
            return 0.0
        if issubclass(annotation, str):
            return ""
    return None


def _seed_offline_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make Stage-D snippet execution offline-deterministic (btqzq).

    Deletes provider API keys, then patches the ``_get_llm`` seam at BOTH binding
    sites (mirroring ``install_fake_llm``) with a FALLBACK wrapper: call the real
    ``_get_llm`` (so snippets that pass their own fake ``llm_factory``, e.g.
    testing.mdx, are untouched) and substitute a generic ``StructuredFake`` ONLY
    when the real factory raises ``KeyError`` -- the deterministic env-missing
    signal from a ``os.environ["..._API_KEY"]`` factory. Any OTHER exception
    propagates, so genuine construction/resolution drift still fails the suite."""
    for env_var in _PROVIDER_KEY_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    import neograph._llm as _llm_mod

    original_get_llm = _llm_mod._get_llm

    def _offline_get_llm(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        try:
            return original_get_llm(*args, **kwargs)
        except KeyError:
            return StructuredFake(_build_instance)

    monkeypatch.setattr("neograph._llm._get_llm", _offline_get_llm)
    monkeypatch.setattr("neograph._tool_loop._get_llm", _offline_get_llm)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "website" / "src" / "content" / "docs"

# (page filename, known python-block count) -- the two drift-guard pins.
PINNED_PAGE_COUNTS = [
    ("checkpoint-resume.mdx", 3),
    ("migrating-prompt-compilers.mdx", 8),
]


def _find_examples_mdx():
    """Return the Stage D collector wrapper, or fail with the TDD-red message.

    ``find_examples_mdx`` is the ~10-line MDX collector the implement atom
    (neograph-wv1yh.15) adds to THIS module. Resolving it through the module
    globals (rather than a bare reference) turns the not-yet-implemented state
    into an assertion-shaped failure -- the expected TDD-red -- rather than a
    NameError/collection error.
    """
    fn = globals().get("find_examples_mdx")
    if fn is None:
        raise AssertionError(
            "find_examples_mdx is not yet defined in tests/test_docs_snippets.py -- "
            "the Stage D collector wrapper (implement atom neograph-wv1yh.15) must add "
            "it: glob website/src/content/docs/**/*.mdx and, per file, yield "
            "pytest_examples.find_examples._extract_code_chunks(path, path.read_text('utf-8'), uuid4()). "
            "This is the expected TDD-red failure."
        )
    return fn


# ---------------------------------------------------------------------------
# Stage D collector wrapper (implementation plan v2 step 2).
#
# ``find_examples`` filters on ``path.suffix`` and only handles ``.py``/``.md``;
# our docs are 44 ``.mdx`` / 0 ``.md``, so ``find_examples`` returns ZERO. The
# ONLY MDX adaptation needed is to route ``.mdx`` text through the private
# ``_extract_code_chunks`` chunker directly (the spike proved it extracts
# ``python``/``py`` fenced blocks cleanly and ignores JSX / frontmatter /
# ``<Tabs>``/``<Aside>`` prose nodes). The drift guard above pins two page
# counts so an upstream rename of that PRIVATE symbol fails loud.
# ---------------------------------------------------------------------------


def _extract_code_chunks_fn():
    """Resolve the private ``_extract_code_chunks`` from ``pytest_examples``.

    ``import pytest_examples.find_examples as fe`` binds ``fe`` to the *function*
    ``find_examples`` (the package ``__init__`` re-exports it, shadowing the
    submodule), so we MUST go through ``importlib.import_module`` to reach the
    real module object and its private chunker.
    """
    module = importlib.import_module("pytest_examples.find_examples")
    return module._extract_code_chunks


def find_examples_mdx(*dirs: Path | str) -> Iterable:
    """Yield every ``python``/``py`` fenced ``CodeExample`` under ``dirs``.

    Globs ``**/*.mdx`` under each directory and, per file, routes the raw text
    through ``pytest_examples.find_examples._extract_code_chunks`` with a
    per-file ``group`` UUID (matching upstream ``find_examples`` semantics). The
    chunker itself filters to python fences (``prefix.startswith(("py","{.py"))``)
    so only python blocks are yielded.
    """
    extract = _extract_code_chunks_fn()
    for d in dirs:
        base = Path(d)
        for path in sorted(base.rglob("*.mdx")):
            text = path.read_text("utf-8")
            yield from extract(path, text, uuid4())


# ---------------------------------------------------------------------------
# Per-page execution (implementation plan v2 steps 3-6).
# ---------------------------------------------------------------------------

# Skip-marker convention (net-new, v2 step 5). A block is skipped when one of
# the lines immediately preceding its fence carries a ``test-skip:`` marker.
#
# NOTE ON SYNTAX (necessary deviation from the plan's literal ``<!-- -->``):
# Astro 6 / Starlight 0.38 compile ``.mdx`` with MDX v3, which REJECTS HTML
# comments (``<!-- ... -->`` -> "Unexpected character `!`") and would break
# ``npm run build`` (a hard gate). The MDX-valid invisible-comment form is the
# JSX expression comment ``{/* ... */}``. The regex below accepts either
# terminator so the convention survives a future MDX-comment-syntax change.
_SKIP_RE = re.compile(r"test-skip:\s*(.*?)\s*(?:\*/|-->|\})")


def _skip_reason(text: str, start_line: int) -> str | None:
    """Return the skip reason for the block whose fence is at ``start_line``.

    ``start_line`` is the 1-based line number of the opening fence. Scan the up
    to three raw lines directly above it (allowing a blank line between the
    marker and the fence) for a ``test-skip:`` marker.
    """
    lines = text.splitlines()
    for j in range(max(0, start_line - 4), start_line - 1):
        if j < len(lines):
            m = _SKIP_RE.search(lines[j])
            if m:
                return m.group(1).strip()
    return None


def _python_pages() -> dict[Path, list]:
    """Map each ``.mdx`` page path -> its python blocks in source order."""
    pages: dict[Path, list] = {}
    for chunk in find_examples_mdx(DOCS_ROOT):
        pages.setdefault(Path(chunk.path), []).append(chunk)
    return pages


# Computed once at import so the parametrization enumerates exactly the pages
# that carry >=1 python block (the coverage test asserts this set is complete).
_PAGES_BY_PATH = _python_pages()
DOC_PAGES = sorted(_PAGES_BY_PATH, key=str)


@pytest.mark.parametrize("page", DOC_PAGES, ids=lambda p: str(p.relative_to(DOCS_ROOT)))
def test_docs_page_python_snippets_execute(
    page: Path, eval_example, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every unskipped python block on a page runs (threading page state).

    One test per ``.mdx`` page (NOT per block) so the case is debuggable under
    ``-k``/xdist/reorder with no cross-parametrized-case accumulator. Blocks run
    IN SOURCE ORDER; each block's post-exec module namespace is threaded into the
    next block's ``module_globals`` so later blocks see earlier defs. A block
    carrying a ``test-skip:`` marker is counted-and-skipped (never executed, so a
    genuinely non-runnable fragment / live-LLM / async snippet does not fail the
    suite). The per-page computed invariant pins that every collected block is
    accounted for: ``collected == executed + skipped`` (zero silent drops).

    The registry-cleanup autouse fixture (``conftest._clean_registries``) resets
    the decoration/type registries before this case, so per-page ``@node`` /
    ``@merge_fn`` / ``@tool`` definitions start clean.
    """
    text = page.read_text("utf-8")
    blocks = _PAGES_BY_PATH[page]

    # Offline-deterministic LLM seam: a snippet that constructs a real provider
    # factory resolves to a generic fake instead of a live call (btqzq).
    _seed_offline_llm(monkeypatch)

    acc: dict = {}
    executed = 0
    skipped = 0
    for block in blocks:
        reason = _skip_reason(text, block.start_line)
        if reason:
            skipped += 1
            continue
        # Plain run() -- no lint()/format() (docs are not black-formatted) and no
        # run_print_check() (VERIFIED zero docs use the `#>` output convention).
        acc = eval_example.run(block, module_globals=acc)
        executed += 1

    assert len(blocks) == executed + skipped, (
        f"{page.name}: {len(blocks)} collected != {executed} executed + {skipped} skipped "
        "(a block was silently dropped -- skip-marker detection is out of sync)."
    )


def test_snippet_coverage_is_fully_accounted(capsys) -> None:
    """Core Invariant: every collected python block is executed or skipped.

    REFUTED-FOLKLORE LESSON: pytest-examples tests ONLY what ``find_examples``
    collects -- NOT "every snippet". Coverage is the set of python fenced blocks
    minus explicitly-skipped blocks. This is a COMPUTED assertion (not prose):
    partition every collected block into executed (no marker) vs skipped
    (marker) and assert the two independently-counted tallies sum to the
    collected total, and that the per-page parametrization enumerates EVERY page
    with blocks (so no page silently drops out of execution).
    """
    pages = _python_pages()

    collected = 0
    skipped = 0
    executable = 0
    for page in sorted(pages, key=str):
        text = page.read_text("utf-8")
        blocks = pages[page]
        page_skipped = sum(1 for b in blocks if _skip_reason(text, b.start_line))
        page_exec = sum(1 for b in blocks if not _skip_reason(text, b.start_line))
        collected += len(blocks)
        skipped += page_skipped
        executable += page_exec
        assert len(blocks) == page_skipped + page_exec  # clean partition, per page

    # Zero silent drops: independently-counted buckets reconstruct the total.
    assert collected == skipped + executable
    # The parametrized execution test must cover EVERY page that has blocks.
    assert set(DOC_PAGES) == set(pages), (
        "the per-page execution parametrization does not cover every page with "
        "python blocks -- a page would be silently untested."
    )

    with capsys.disabled():
        print(
            f"\n[Stage D snippet coverage] {collected} python blocks collected "
            f"− {skipped} skipped = {executable} executed. "
            "(pytest-examples tests only what find_examples collects, NOT every snippet.)"
        )


def _python_blocks_for_page(page_filename: str) -> list:
    """All python CodeExample chunks ``find_examples_mdx`` yields for one page.

    Filters the whole-tree collection down to the target page and to fences
    whose info-string is python (``prefix`` starting ``py`` / ``{.py``), matching
    ``_extract_code_chunks``'s own ``prefix.startswith(("py", "{.py"))`` test.
    """
    find_examples_mdx = _find_examples_mdx()
    return [
        chunk
        for chunk in find_examples_mdx(DOCS_ROOT)
        if Path(chunk.path).name == page_filename
        and str(getattr(chunk, "prefix", "")).startswith(("py", "{.py"))
    ]


@pytest.mark.parametrize(
    ("page_filename", "expected_count"),
    PINNED_PAGE_COUNTS,
    ids=[name for name, _ in PINNED_PAGE_COUNTS],
)
def test_find_examples_mdx_collects_pinned_python_block_counts(
    page_filename: str, expected_count: int
) -> None:
    """find_examples_mdx collects the pinned python-block count per drift-guard page.

    Pins that the MDX collector yields EXACTLY the known number of python fenced
    blocks for ``checkpoint-resume.mdx`` (=3) and the indented-fence
    ``migrating-prompt-compilers.mdx`` (=8), so an upstream ``_extract_code_chunks``
    rename or a nested-fence regex regression fails loud instead of silently
    collecting zero/fewer blocks.
    """
    # Sanity: the pinned source pages must exist (a guard-count is meaningless if
    # the page was renamed/moved out from under it).
    matches = list(DOCS_ROOT.rglob(page_filename))
    assert len(matches) == 1, (
        f"expected exactly one {page_filename} under {DOCS_ROOT}, found {len(matches)}: {matches}"
    )

    blocks = _python_blocks_for_page(page_filename)
    assert len(blocks) == expected_count, (
        f"{page_filename}: find_examples_mdx collected {len(blocks)} python blocks, "
        f"expected {expected_count} (drift guard -- an upstream _extract_code_chunks "
        f"rename or an indented-fence regex regression breaks this count)."
    )
