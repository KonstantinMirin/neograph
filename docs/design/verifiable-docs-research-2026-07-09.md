# Compile-time-verified, auto-cross-linked documentation from library code: research report

Date: 2026-07-09
Scope: prior art for code-as-source-of-truth verified docs; snippet verification; Astro/Starlight
implementation paths; recommendation for neograph.
Method: multi-source research with 3-vote adversarial claim verification. 24 claims survived
(23 unanimous, 1 at 2-1); 1 claim was refuted. Findings below cite only surviving claims;
sections marked **[synthesis]** are design reasoning built on those findings plus repo context,
not independently verified claims.

---

## Executive summary

Across ecosystems, teams achieve verified docs through the same three-part pattern:
(1) symbol references in prose resolve against a machine-readable model of the code at
build time, with strictness as an explicit opt-in (rustdoc's `#![deny(rustdoc::broken_intra_doc_links)]`
— warn-by-default otherwise; Sphinx's `-n -W`); (2) code snippets are executed as part of the
project's ordinary test suite rather than by the docs toolchain (Pydantic's pytest-examples,
Sybil, pytest doctest-glob, phmdoctest), with output-level assertions and explicit skip
annotations for non-runnable blocks; (3) authoring syntax splits into two camps — name-resolved
references where the symbol name itself is the link (rustdoc) versus explicit markers
(Sphinx roles, mkdocs-autorefs brackets) — with no verified mainstream tool auto-detecting
bare backticked code spans the way RAMP does. For neograph, the verified evidence supports a
staged plan: a committed Python-generated API manifest guarded by a pytest freshness test
(sidestepping the Amplify no-Python constraint), a remark plugin over that manifest for
symbol validation/autolinking (RAMP's proven architecture), and snippet testing via
pytest-examples or Sybil wired into the existing suite.

---

## 1. Prior art: symbol references verified against code

### 1.1 Rust / rustdoc intra-doc links — the (qualified) gold standard

Rustdoc intra-doc links let authors link to items — functions, types, and more — **by name**,
instead of a hard-coded URL; the symbol name itself is the link target.[^rust-blog] The
stabilization post is explicit that the motivation was structural drift-prevention: hard-coded
URLs "could silently break — the documentation would work when you built it, but any user of
your API could re-export your types and cause the links to be broken."[^rust-blog] Because
resolution follows the item, not the path the author typed, links stay accurate "even if your
types are re-exported in a different module or crate"[^rust-blog] — the property that matters
most for a library like neograph whose public contract is a re-export facade
(`neograph/__init__.py.__all__`).

**Critical correction to the common framing**: the `broken_intra_doc_links` lint only
**warns by default** — a broken symbol link is *not* a hard build error out of the box.[^rust-lints]
Verified live against the rustdoc book: "This lint warns by default," with the `Nonexistent`
example rendered as `warning: unresolved link`. Teams opt into build-failure behavior with a
one-line crate attribute:

```rust
#![deny(rustdoc::broken_intra_doc_links)] // error if there are broken intra-doc links
```

— which fails the *docs* build (`cargo doc` exits nonzero), not `rustc` compilation.[^rust-lints]
So the "gold standard" is really: zero-annotation name resolution + a deliberate one-line
strictness opt-in. That is exactly RAMP's fail-the-build design, and it means RAMP is not more
aggressive than the ecosystem norm — it just ships with the deny bit pre-set.

Prose syntax: `[Item]` / `[path::to::Item]` — auto-resolved names inside standard Markdown
link syntax. Verified: existence + resolution across re-exports. Failure point: docs-build
time, warn unless denied.

### 1.2 Sphinx — explicit roles + two flags to make it strict

Sphinx's `-n` (nitpicky) mode "generates warnings for all missing references," and `-W`
"turn[s] warnings into errors... sphinx-build exits with exit status 1 if any warnings are
generated during the build."[^sphinx-build] Combined, a broken `:py:func:`/`:py:class:` role
reference fails the build rather than passing silently. Same shape as rustdoc: strictness is
opt-in, and mature projects opt in. (Since Sphinx 8.1, `--keep-going` is always on, so the
build runs to completion before exiting 1 — reporting granularity changed, the contract did
not.) `nitpick_ignore` provides the escape hatch — the analogue of RAMP's ignore file, though
Sphinx's is not self-cleaning.

Prose syntax: explicit roles (`:py:func:\`compile\``). Verified: target existence in the
object inventory. Failure point: build time, only under `-n -W`.

### 1.3 mkdocs-autorefs — path-free anchors, explicit markers

mkdocs-autorefs resolves cross-references by heading/anchor identifier at build time, so
authors can "link to a heading from any other page on the site *without* needing to know the
path."[^autorefs] It sits firmly on the **explicit-marker** side of the syntax question: linking
requires reference-link brackets (`[text][identifier]`, `[identifier][]`, or the
code-span-in-brackets form ``[`identifier`][]``); no zero-annotation mode that auto-detects
bare backticked code spans is documented, confirmed through the full changelog to v1.4.4
(2026-02).[^autorefs] Making unresolved refs fail the build additionally requires
`mkdocs build --strict` (the mkdocs analogue of `-W`).

**[synthesis]** The mkdocstrings ecosystem (autodoc-from-docstrings feeding autorefs anchors)
is the closest Python analogue to the full pipeline neograph wants, but it is MkDocs-native;
none of its machinery transfers to Astro. What transfers is the *architecture*: a symbol
inventory + an anchor-identity contract between generated reference content and prose links.

### 1.4 Verification matrix

| System | Prose syntax | Verified | Fails when |
|---|---|---|---|
| rustdoc intra-doc | `[Name]` — name auto-resolved | item existence, re-export-safe resolution | docs build, **only under `deny`** [^rust-lints] |
| Sphinx | explicit roles `:py:func:` | inventory existence | build, **only under `-n -W`** [^sphinx-build] |
| mkdocs-autorefs | explicit `[text][id]` brackets | anchor existence site-wide | build under `--strict` [^autorefs] |
| RAMP (baseline) | backticked tokens, shape-detected | descriptor existence + anchor liveness + stale ignores | build, always |

The convergent lesson: **every mainstream system makes strictness a deliberate switch, and the
projects with reputations for verified docs are simply the ones that flipped it.**

---

## 2. Verifying code snippets against the real library

### 2.1 pytest-examples — the Pydantic team's tool

`pytest-examples` is a pytest plugin maintained by the Pydantic organization for "testing
Python code examples in docstrings and markdown files."[^pytest-examples] It plugs into an
existing suite as an ordinary parametrized test — no separate tool or CI job:

```python
@pytest.mark.parametrize('example', find_examples('foo_dir', 'bar_file.py'), ids=str)
def test_docstrings(example: CodeExample, eval_example: EvalExample):
    eval_example.lint(example)
    eval_example.run(example)
```

Granularity goes beyond "does it run": `eval_example.lint()` checks examples with ruff and
black; `run()` executes; `run_print_check()` asserts printed output against inline `#>`
expected-output comments — output-asserting snippet tests, not import smoke tests.[^pytest-examples]

Pydantic's own wiring lives in a dedicated `tests/test_docs.py`, and supports an auto-fix
mode: `pytest tests/test_docs.py --update-examples` both runs the tests and rewrites example
formatting/expected output in place (verified in the repo: the flag triggers
`eval_example.format(example)` and `run_print_update(...)`).[^pydantic-contrib] The
contributing page does not name the underlying tool; the attribution comes from the repo's
`tests/test_docs.py` imports.

**Refuted claim, kept for honesty**: "Pydantic's CI verifies *every* code example in the
documentation" did **not** survive verification (0-3). The infrastructure tests the examples
it collects; a totality guarantee is not established. Any neograph adoption should state
coverage in terms of what `find_examples` is pointed at, not "all snippets."

### 2.2 Sybil — Markdown-native, richest skip/setup model

Sybil "check[s] examples in your code and documentation by parsing them from their source and
evaluating the parsed examples as part of your normal test run"[^sybil-home], with integration
for pytest and unittest (a `conftest.py` `pytest_collect_file` hook — doc examples become
collected pytest items).[^sybil-home] It handles three markup formats — ReST, Markdown, and
MyST — so plain Markdown/CommonMark sources work without Sphinx.[^sybil-home] Markdown-specific
parsers include `PythonCodeBlockParser` for doctest examples inside `python` fenced
blocks.[^sybil-md]

Two features matter especially for docs-site snippets:

- **Invisible code blocks** via HTML comments (`<!-- invisible-code-block: python ... -->`)
  let setup/boilerplate execute during testing without rendering on the page.[^sybil-md]
- **Skip directives** in Markdown comments mark non-runnable or environment-dependent blocks:
  `<!-- skip: next -->`, ranged `<!-- skip: start -->` / `<!-- skip: end -->`, and conditional
  `<!-- skip: next if(condition, reason="...") -->`.[^sybil-md]

### 2.3 pytest doctest-glob and phmdoctest — the simpler tiers

Plain pytest can collect doctests from text/markup files with `--doctest-glob="*.rst"` (the
flag repeats for multiple patterns); empirically reproduced with pytest 9.0.2 on `.rst` and
`.md` files.[^pytest-doctest] Doctest failures behave as ordinary test failures (verified:
`FAILED mod.py::mod.add`, exit code 1), so a stale snippet fails CI — but **only `>>>`-style
doctest examples execute; plain fenced ```python blocks are not run by this mechanism**, which
is the disqualifier for Starlight `.mdx` pages full of fenced blocks.[^pytest-doctest]

phmdoctest takes the generate-don't-test-in-place approach: it "creates a pytest Python module
that tests Python examples in README and other Markdown files"[^phmdoctest], at two
granularities — doctest sessions (run via `--doctest-modules`) and source blocks whose actual
output is compared against a paired expected-output block via capsys capture.[^phmdoctest]
Non-runnable blocks use HTML-comment directives (`<!--phmdoctest-skip-->`, plus label /
setup / teardown / share-names) — an explicit opt-out annotation model.[^phmdoctest] Caveat:
last release v1.4.0 (2022-03); the author's successor tool is phmutest. The *mechanism* is
verified; the tool as a 2026 recommendation is dated.

### 2.4 Granularity in practice **[synthesis over verified mechanisms]**

The verified tools stack four levels: (1) lint/format the snippet (pytest-examples);
(2) execute it (all); (3) assert printed output (`#>` in pytest-examples, expected-output
blocks in phmdoctest, doctest `>>>` everywhere); (4) auto-repair drift (`--update-examples`).
Every tool that tests Markdown in place converged on **HTML-comment annotations for skips and
setup** (Sybil, phmdoctest) — invisible in rendered output, greppable in source. That
convention ports directly to MDX.

---

## 3. Astro/Starlight specifics

No claims about Starlight-ecosystem plugins (starlight-typedoc, starlight-links-validator) or
Python→JSON→remark precedents survived the verification gate, so this section is
**[synthesis]** grounded in the verified prior art plus first-hand knowledge of the RAMP
implementation (which is itself an existence proof for the architecture).

- **The remark-plugin-over-a-descriptor architecture works.** RAMP resolves dotted
  inline-code tokens against a protobuf descriptor at Astro build time, fails the build on
  high-confidence misses, slugs actual reference-page headings for dead-anchor prevention,
  and self-cleans its ignore file. Nothing in that design is proto-specific: it needs only
  (a) a machine-readable symbol inventory available at Node build time and (b) an anchor
  contract with the reference pages. The rustdoc/Sphinx/autorefs evidence confirms each design
  choice independently (name resolution, build-failure strictness, anchor-existence checking).
- **The non-JS source-of-truth bridge is a committed artifact.** The Amplify build image
  cannot be assumed to have Python or neograph installed, so build-time Python execution
  inside the Astro build is out. The standard workaround shape: generate a JSON manifest from
  Python introspection, commit it, and let the remark plugin (and optionally Astro content
  collections with a Zod schema over the manifest) consume pure JSON. Freshness is then a
  *test-suite* problem, not a *docs-build* problem — which is precisely where neograph's
  structural-guard culture lives.
- **Anchor identity is the fragile seam.** rustdoc and autorefs both own the slugger and the
  content generator, so anchors cannot drift. In a split system (Python generates content,
  Astro slugs headings), the manifest generator must emit anchors using the same slug
  algorithm Starlight uses (github-slugger), or — RAMP's stronger move — the plugin must slug
  the actual rendered reference headings and validate against those, not against assumed slugs.

---

## 4. The syntax/DX question

Verified positions on the spectrum:

- **Zero-annotation, name-resolved**: rustdoc — the symbol name is the link; no per-reference
  ceremony; the compiler's own name resolution supplies the confidence, so false positives are
  structurally impossible.[^rust-blog]
- **Explicit markers**: Sphinx roles and mkdocs-autorefs brackets — autorefs documents *only*
  explicit `[text][id]` forms; no auto-detection of bare code spans exists in the tool through
  v1.4.4.[^autorefs] Cost: authoring friction on every reference; benefit: zero false
  positives without heuristics, trivially greppable.
- **Shape-detected backticks** (RAMP): zero annotation burden, but requires confidence
  heuristics plus an ignore list because a bare `` `foo.bar` `` is ambiguous in a way
  `[foo::bar]` inside link syntax is not.

**[synthesis]** No verified mainstream tool occupies RAMP's exact position; the ecosystem
splits between rustdoc's "language-integrated so detection is exact" and everyone else's
"explicit markers." For an existing 45-page site, shape-detection is the only approach with
incremental-adoption economics — explicit markers require touching every page to get any
coverage, while a detector delivers site-wide verification on day one and the ignore file
absorbs the false-positive tail. Two RAMP mitigations are load-bearing and should be treated
as requirements, not options: **tiered confidence** (dotted refs like `Node.inputs` are
high-confidence and may fail the build; bare single tokens like `compile` only autolink when
they exactly match a manifest symbol, and never fail the build) and the **self-cleaning ignore
file** (a stale ignore entry fails the build), which is the same fail-loud discipline as
neograph's structural guards. An escape hatch both camps provide (nitpick_ignore, autorefs'
explicit anchors) is non-negotiable.

---

## 5. Recommendation for neograph **[synthesis]**

Staged so each stage is independently shippable and catches a class of drift the previous
stage cannot. The ordering front-loads the piece every later stage depends on (the manifest)
and defers the highest-effort piece (generated reference content) until the validator exists
to keep it honest.

### Stage A — `api-manifest.json` + pytest freshness guard (foundation; ~1-2 days)

A Python introspection script (`scripts/gen_api_manifest.py`) walks `neograph.__all__` and
emits a committed `website/src/data/api-manifest.json`:

- symbols with kind (function/class/modifier/exception), signature string, one-line docstring;
- Pydantic model fields (name, rendered annotation, required/default) — `describe_type`
  already knows how to render these;
- the exception hierarchy under `NeographError`;
- lint issue kinds (the `lint()` check-category identifiers);
- per-symbol stable anchor ids, computed with github-slugger-compatible slugging.

A structural guard test (matching the existing `test_guards_*.py` culture) regenerates the
manifest in-memory and diffs it against the committed file; any API change without a manifest
commit fails `pytest`. This is the same freshness contract as rustdoc's "docs build from the
code" — displaced into the test suite because Amplify can't run Python. Failure mode:
**staleness window** — the manifest is only as fresh as the last green test run; a direct push
that skips CI ships a stale manifest. Mitigation: the guard runs in the default suite, and the
remark plugin (Stage B) hard-fails on refs to symbols missing from the manifest, so drift
surfaces at the next docs build regardless.

### Stage B — remark plugin: validate + autolink symbol refs (~2-3 days, ports from RAMP)

Port RAMP's plugin shape, swapping the proto descriptor for the manifest: scan inline-code
nodes in all `.mdx` pages; tiered confidence (dotted `Type.member` → resolve or **fail the
Astro build** if the type exists but the member doesn't; exact bare-name matches → autolink
only); autolink resolved refs to `reference/api#<anchor>`; validate anchors against the actual
reference page headings (slug the real headings, RAMP-style, until Stage C makes anchors
manifest-owned); self-cleaning ignore file. Catches: renamed/removed symbols referenced in
prose, dead anchors — across all 45 existing pages with zero page edits. Failure modes:
MDX inline-code false positives (English words that collide with short symbol names — keep
bare-token matching exact-match-only and never build-failing) and anchor drift against the
hand-written `reference/api.mdx` (mitigated by slugging real headings, eliminated by Stage C).

### Stage C — generated reference sections from the manifest (~2-4 days)

Replace the hand-written body of `reference/api.mdx` incrementally with sections rendered from
the manifest (an Astro component or a small codegen step emitting MDX partials; content
collections with a Zod schema over the manifest give type-safe access). Anchors become
manifest-owned, closing the Stage B anchor-drift seam — the same anchor-identity property
rustdoc and autorefs get from owning both ends.[^rust-blog][^autorefs] Catches: signature and
field-table drift in reference content itself, which Stages A-B only catch when prose
*references* the changed symbol. Failure mode: generated MDX styling/regression churn — keep
generation at the section level, not whole-page, so prose narrative stays hand-written.

### Stage D — snippet testing with pytest-examples (or Sybil) (~2-3 days)

Add `tests/test_docs_snippets.py` on the Pydantic pattern: `find_examples('website/src/content/docs')`
parametrized over `EvalExample`, with `run()` for runnable blocks and `run_print_check()`
where output is shown[^pytest-examples][^pydantic-contrib]; adopt HTML-comment skip markers
for non-runnable blocks (API-key examples 07/11, pseudo-code) following the
Sybil/phmdoctest convention[^sybil-md][^phmdoctest]. Choose **pytest-examples** if the
`--update-examples` auto-repair loop is wanted (it is a strong fit for a sole-maintainer
repo); choose **Sybil** if invisible setup blocks and conditional skips matter more.[^sybil-md]
Plain `--doctest-glob` is insufficient: it executes only `>>>`-style examples, not the fenced
blocks Starlight pages actually contain.[^pytest-doctest] Catches: behavioral drift —
snippets that still *name* real symbols but no longer *run* or no longer print what the page
claims — invisible to Stages A-C. Failure modes: MDX blocks needing framework context
(fake-LLM setup — reuse `tests/fakes.py` via invisible/setup blocks), runtime cost
(examples that compile+run pipelines; keep them on fakes), and the totality trap — report
coverage as "all collected `python` blocks minus N skipped," never "all snippets"
(the refuted-claim lesson from Pydantic).

### What each stage catches that the previous doesn't

| Stage | Drift class caught | New failure surface |
|---|---|---|
| A manifest + guard | API changed at all (any public-surface delta fails pytest) | manifest staleness window |
| B remark validation | prose references to dead/renamed symbols; dead anchors | inline-code false positives |
| C generated reference | signature/field tables wrong in reference content | generated-MDX churn, styling |
| D snippet tests | snippets that resolve but don't run / wrong output | context setup, runtime cost |

---

## Caveats

- The rustdoc "broken link = build error" folklore is wrong as stated: warn-by-default, deny
  is opt-in.[^rust-lints] Every strictness comparison in this report is normalized to that.
- The claim that Pydantic verifies *every* docs example was refuted (0-3); pytest-examples
  coverage is what `find_examples` collects, no more.
- One doctest claim (failures = ordinary failures) passed 2-1: the behavior was empirically
  reproduced (pytest 9.0.2, exit 1), but part of its supporting quote was paraphrase, not
  verbatim, and doctest collection requires explicit opt-in flags.
- No Starlight-ecosystem claims (starlight-typedoc, starlight-links-validator, Python→JSON
  precedents) survived verification; Section 3 and the recommendation are design synthesis
  anchored in verified cross-ecosystem patterns plus the RAMP implementation as first-hand
  precedent.
- Tool-currency: phmdoctest is effectively frozen (last release 2022-03; successor phmutest);
  pytest-examples latest is v0.0.18 (2025-05) — pre-1.0 but actively used by Pydantic;
  autorefs verified through v1.4.4 (2026-02); Sybil docs at v10.x.

## Open questions

1. Does starlight-links-validator (or any Starlight plugin) already validate intra-site
   anchors well enough to replace the Stage B anchor-liveness check, or does its
   page-link-only scope miss heading anchors on generated sections? (No claim survived; needs
   direct evaluation against the plugin repo.)
2. Is there working precedent for pytest-examples against **MDX** (vs plain Markdown) — do
   JSX component islands inside `.mdx` break its fenced-block extraction, or does it skip
   non-code nodes cleanly? A 15-minute spike on one neograph page would settle it.
3. Should the manifest include *behavioral* metadata (mode-inference table, lint issue kinds
   with messages) beyond the symbol surface — i.e., where is the line between "API manifest"
   and "second implementation of the docs"?
4. phmutest (phmdoctest's successor) was not verified: does it supersede the
   pytest-examples/Sybil choice for Markdown-with-expected-output testing, or is it niche?

---

## Sources

[^rust-lints]: rustdoc book, Lints — `broken_intra_doc_links`. https://doc.rust-lang.org/rustdoc/lints.html — "This lint **warns by default**"; `#![deny(rustdoc::broken_intra_doc_links)] // error if there are broken intra-doc links`. (Verified live 2026-07-09.)
[^rust-blog]: Inside Rust blog, "Stabilizing intra-doc links" (2020-09-17). https://blog.rust-lang.org/inside-rust/2020/09/17/stabilizing-intra-doc-links.html — name-resolved links; silent-breakage motivation; re-export accuracy. (Quotes verified verbatim.)
[^sphinx-build]: Sphinx, sphinx-build man page. https://www.sphinx-doc.org/en/master/man/sphinx-build.html — `-n` "generates warnings for all missing references"; `-W` "exits with exit status 1 if any warnings are generated". (Verified live.)
[^autorefs]: mkdocstrings/autorefs. https://github.com/mkdocstrings/autorefs — path-free heading links via explicit `[text][identifier]`; no bare-code-span auto-detection through v1.4.4.
[^pytest-examples]: pydantic/pytest-examples. https://github.com/pydantic/pytest-examples — "Pytest plugin for testing Python code examples in docstrings and markdown files"; `find_examples`/`EvalExample`; `lint`/`run`/`run_print_check` with `#>`.
[^pydantic-contrib]: Pydantic contributing docs + repo. https://docs.pydantic.dev/latest/contributing/ and pydantic/pydantic `tests/test_docs.py` — `pytest tests/test_docs.py --update-examples`; repo imports `pytest_examples` and calls `format`/`run_print_update` under the flag.
[^sybil-home]: Sybil documentation. https://sybil.readthedocs.io/en/latest/ — examples evaluated "as part of your normal test run"; pytest/unittest integration; ReST, Markdown, MyST.
[^sybil-md]: Sybil, Markdown support. https://sybil.readthedocs.io/en/latest/markdown.html — `PythonCodeBlockParser`; invisible code blocks via HTML comments; `skip: next` / ranged / conditional skips.
[^pytest-doctest]: pytest, doctest integration. https://docs.pytest.org/en/stable/how-to/doctest.html — `--doctest-glob` (repeatable); empirically reproduced on `.rst`/`.md` with pytest 9.0.2; only `>>>`-style examples run.
[^phmdoctest]: tmarktaylor/phmdoctest. https://github.com/tmarktaylor/phmdoctest — generates a pytest module from Markdown; doctest sessions + capsys expected-output comparison; `<!--phmdoctest-skip-->` and directive family. Last release v1.4.0 (2022-03); successor: phmutest.
