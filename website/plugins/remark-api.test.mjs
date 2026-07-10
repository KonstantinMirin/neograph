// Fail-path + positive-path coverage for the build-gating api-symbol guard. Without
// this, a refactor that silently stops the guard from biting (an over-broad allow, a
// regex that no longer matches, a swallowed throw, a stale-check that never fires) would
// pass CI. These tests assert it still bites AND still links.
//
// The guard has TWO tiers (the mandatory tiered-confidence contract from
// docs/design/verifiable-docs-research-2026-07-09.md section 4):
//   HARD — a dotted `Type.member` ref whose owner has declared fields resolves or FAILS
//          the Astro build (member-existence is pure manifest validation; full coverage).
//   SOFT — a bare token (compile, Node, Oracle) autolinks on EXACT manifest match with a
//          live reference-page heading, or stays inert. NEVER build-failing.
//
// Cases (a)-(f) pin the plan step-9 required behavior. (g)+(h) pin the self-cleaning
// ignore file with the M1 predicate-based wouldResolve stale check (NOT RAMP's verbatim
// symbols[k] lookup — neograph's symbol map is NAME-keyed, so a dotted ignore entry's
// stale state must be re-derived, not looked up). (i) pins the kind-namespaced anchor
// scheme (node/Node and tool/Tool bare-slug collisions are disambiguated to
// node-function/node-model etc., so a bare Node never mis-links to the `node` fn); (j)
// pins the visitor guard that prevents nested <a> tags when inlineCode sits inside a link.
//
// All assertions run against the REAL manifest (api-manifest.json + api-manifest-mcp.json)
// and the REAL reference-page headings (reference/api.mdx) — no mocks. The only file the
// tests manage is the ignore file (api-symbols-ignore.json), and only for (g)+(h), because
// a stale ignore entry cannot live in production config (it would break every build).
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { visit } from 'unist-util-visit';
import { writeFileSync, unlinkSync, existsSync, copyFileSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import remarkApi from './remark-api.mjs';

const run = (tree) => { remarkApi()(tree, { path: 'test.mdx' }); return tree; };
const para = (value) => ({
  type: 'root',
  children: [{ type: 'paragraph', children: [{ type: 'inlineCode', value }] }],
});
// inlineCode wrapped in an existing markdown link — the L2 visitor guard must leave it.
const paraInLink = (value, url) => ({
  type: 'root',
  children: [{ type: 'paragraph', children: [{ type: 'link', url, children: [{ type: 'inlineCode', value }] }] }],
});
const has = (tree, pred) => { let f = false; visit(tree, (n) => { if (pred(n)) f = true; }); return f; };

// --- ignore-file management for the stale/ignore tests (g, h) ---
// The real ignore file cannot carry a stale entry (it would break every Amplify build),
// so these two tests write a fixture ignore file and import the plugin FRESH (cache-bust
// query) so setup() re-reads it. The plugin caches per module-instance (the ported RAMP
// setup() pattern), so a unique query string yields a new `let state`. before/after save
// and restore the production ignore file so the suite is hermetic.
const ignorePath = fileURLToPath(new URL('../api-symbols-ignore.json', import.meta.url));
const backupPath = ignorePath + '.test-backup';
let ignoreExisted = false;
let importCounter = 0;

async function runWithIgnore(ignoreEntries, tree) {
  writeFileSync(ignorePath, JSON.stringify({ _comment: 'test fixture', ignore: ignoreEntries }));
  // cache-bust: a fresh module URL -> fresh module-level state -> setup() re-reads the ignore file
  const fresh = new URL(`./remark-api.mjs?t=${++importCounter}`, import.meta.url);
  const mod = await import(fresh.href);
  mod.default()(tree, { path: 'test.mdx' });
  return tree;
}

before(() => {
  ignoreExisted = existsSync(ignorePath);
  if (ignoreExisted) copyFileSync(ignorePath, backupPath);
});

after(() => {
  if (ignoreExisted) {
    copyFileSync(backupPath, ignorePath);
    unlinkSync(backupPath);
  } else if (existsSync(ignorePath)) {
    unlinkSync(ignorePath);
  }
});

// (a) HARD tier positive: resolved fielded-member autolinks to reference/api#anchor.
test('resolved fielded-member reference is autolinked to its anchor', () => {
  // LintIssue is a fielded dataclass; `kind` is a declared field; `lintissue` is a live
  // reference-page heading. The dotted ref must resolve AND autolink.
  const tree = run(para('LintIssue.kind'));
  assert.ok(
    has(tree, (n) => n.type === 'link' && n.url.includes('#lintissue')),
    'a real fielded-owner.member should autolink to the reference page anchor',
  );
});

// (a2) HARD tier depth (neograph-0rwuh): a dotted field ref autolinks to the
// FIELD anchor `#{typeAnchor}-{member}`, NOT merely the type-section anchor.
// `outputs` is a declared field of the `Node` model (anchor node-model), and
// node-model is a live reference-page heading -> the ref must land on the field
// row (#node-model-outputs), not the top of the section (#node-model).
// FAILS today: the plugin emits `#node-model` (the type anchor) for every dotted
// field ref, dropping the reader at the section heading instead of the field.
test('dotted field ref autolinks to the field anchor, not the type anchor', () => {
  const manifest = JSON.parse(
    readFileSync(fileURLToPath(new URL('../src/data/api-manifest.json', import.meta.url)), 'utf8'),
  );
  const node = manifest.symbols.find((s) => s.name === 'Node');
  assert.ok(node && node.fields, 'Node model with fields expected in the manifest');
  assert.equal(node.anchor, 'node-model', 'Node model anchor must be node-model');
  assert.ok(
    node.fields.some((f) => f.name === 'outputs'),
    '`outputs` must be a declared field of the Node model',
  );
  const tree = run(para('Node.outputs'));
  // POSITIVE: lands on the exact field anchor.
  assert.ok(
    has(tree, (n) => n.type === 'link' && n.url.includes('#node-model-outputs')),
    'a dotted Type.field ref must autolink to the per-field anchor #node-model-outputs',
  );
  // NEGATIVE: must NOT stop at the bare type-section anchor.
  assert.ok(
    !has(tree, (n) => n.type === 'link' && /#node-model(?!-)/.test(n.url)),
    'must not drop the reader at the type-section anchor #node-model when the member is a field',
  );
});

// (a3) HARD tier preserved (neograph-0rwuh): a NON-field member of a fielded owner
// still FAILS the build -- there is NO type-anchor fallback for a bad member (the
// docs-lie gate). GUARDS EXISTING BEHAVIOR: this PASSES today and must keep passing
// after the field-anchor change lands (the refine note: no type-anchor fallback).
test('non-field member of a fielded owner still HARD-fails the build (no fallback)', () => {
  assert.throws(
    () => run(para('Node.bogus_not_a_field_zzz')),
    /unknown api reference|api-symbols-ignore|unresolved/i,
    'a dotted ref to a non-existent member must still build-fail; no type-anchor fallback',
  );
});

// (b) HARD tier negative: unknown member of a fielded type FAILS the build.
test('unknown member of a fielded type FAILS the build', () => {
  // Node is a fielded owner (18 declared fields); bogus_zzz is not among them. This is the
  // build-gate: the reference silently lies about a field that does not exist.
  assert.throws(
    () => run(para('Node.bogus_zzz')),
    /unknown api reference|api-symbols-ignore|unresolved/i,
    'a dotted ref whose owner has fields but member is absent must throw',
  );
});

// (c) HARD tier inert: dotted ref whose owner is a non-fielded symbol is left alone.
test('dotted ref whose owner is a non-fielded symbol is left alone (no false positive)', () => {
  // FromInput is a manifest symbol WITHOUT a fields array -> owner-gate excludes it from
  // HARD validation (no member list to validate against) -> INERT, never a false positive.
  assert.doesNotThrow(() => run(para('FromInput.bogus')));
  const tree = run(para('FromInput.bogus'));
  assert.ok(!has(tree, (n) => n.type === 'link'), 'must not autolink an unvalidatable dotted ref');
  assert.ok(has(tree, (n) => n.type === 'inlineCode' && n.value === 'FromInput.bogus'), 'left as plain inlineCode');
});

// (d) SOFT tier positive: bare exact-match token autolinks.
test('bare exact-match symbol token is autolinked', () => {
  // `compile` is a manifest function with a live `compile` heading. SOFT tier never
  // throws, but an exact name match WITH a live heading autolinks.
  const tree = run(para('compile'));
  assert.ok(
    has(tree, (n) => n.type === 'link' && n.url.includes('#compile')),
    'a bare token exactly matching a manifest name with a live heading should autolink',
  );
});

// (e) SOFT tier inert: bare token not in the manifest is left alone (English-collision safe).
test('bare token not in the manifest is left inert (never throws)', () => {
  // The whole reason SOFT never fails: bare tokens collide with English. A non-symbol
  // snake_case token must stay plain inlineCode, no throw.
  assert.doesNotThrow(() => run(para('nonsymbol_token')));
  const tree = run(para('nonsymbol_token'));
  assert.ok(!has(tree, (n) => n.type === 'link'), 'must not autolink a non-manifest bare token');
  assert.ok(has(tree, (n) => n.type === 'inlineCode' && n.value === 'nonsymbol_token'), 'left as plain inlineCode');
});

// (f) SOFT tier heading-gate: resolved symbol with no live heading is NOT linked.
test('resolved symbol without a live heading is not linked (heading-gating)', () => {
  // `NeographError` is a real manifest symbol (kind=exception) but it has NO per-symbol
  // reference heading — exceptions live in the fenced error-hierarchy tree (neograph-uorb4),
  // which the harvester skips, so no `#neographerror` anchor is emitted. SOFT tier: resolve
  // (no throw) but stay plain inlineCode rather than emit a dead anchor. This is the
  // heading-gate decline path (remark-api.mjs:161). (Post-Stage-C, exception symbols are the
  // sole resolves-but-no-heading class — every other symbol now has a manifest-generated
  // heading, pinned by TestCrossLinkCoverageCapstone.)
  const tree = run(para('NeographError'));
  assert.ok(!has(tree, (n) => n.type === 'link'), 'must not link to a nonexistent reference-page anchor');
  assert.ok(has(tree, (n) => n.type === 'inlineCode' && n.value === 'NeographError'), 'left as plain inlineCode');
});

// (g) Self-cleaning ignore: a historical-contrast ref to a removed field does not throw.
test('ignore-listed historical reference does not fail', async () => {
  // Node.removed_legacy_x: owner Node is fielded, member does NOT exist -> would throw,
  // UNLESS listed in the ignore file. The test seeds the ignore file with the token and
  // asserts the build does not fail.
  await assert.doesNotReject(
    () => runWithIgnore(['Node.removed_legacy_x'], para('Node.removed_legacy_x')),
    'an ignore-listed dotted ref to a removed field must not fail the build',
  );
});

// (h) Self-cleaning ignore STALE check: an entry that now resolves THROWS.
// M1 (CRITICAL): this is predicate-based wouldResolve, NOT RAMP's verbatim symbols[k]
// lookup. neograph's symbol map is NAME-keyed, so symbols['Node.inputs'] is always
// undefined; the stale check must re-run resolution per entry (dotted -> owner in
// fieldedOwners AND member in owner.fields) to honor self-cleaning for dotted entries.
test('stale ignore entry that now resolves FAILS the build', async () => {
  // Node.inputs IS a declared field -> wouldResolve('Node.inputs') is true -> the stale
  // check must throw at setup() time. This is the structural-guard discipline: the ignore
  // file is self-cleaning, so a symbol that comes back (or was listed by mistake) fails
  // loudly instead of silently suppressing a now-valid reference.
  await assert.rejects(
    () => runWithIgnore(['Node.inputs'], para('Node.inputs')),
    /stale|now resol|remove/i,
    'an ignore entry that now resolves must fail the build (self-cleaning, predicate-based)',
  );
});

// (i) Kind-namespaced anchors (Stage C / neograph-kec0k, closing rfl7b DECISION 1):
// the former node/Node (and tool/Tool) bare-slug COLLISION is gone. The Python
// generator kind-namespaces every colliding symbol's anchor: `node` (fn) ->
// `node-function`, `Node` (model) -> `node-model` (likewise tool/Tool). Stage C
// (kec0k) now RENDERS the kind-namespaced headings, so a bare `Node` resolves via
// NAME-keyed lookup to its OWN `node-model` anchor AND autolinks there — while
// NEVER mis-linking to the decorator's `node-function` section. This is the exact
// silent-inert -> resolving flip neograph-wqzel makes permanent.
test('bare Node resolves to node-model and never mis-links to node-function (disambiguated)', () => {
  const manifest = JSON.parse(
    readFileSync(fileURLToPath(new URL('../src/data/api-manifest.json', import.meta.url)), 'utf8'),
  );
  const byName = Object.fromEntries(manifest.symbols.map((s) => [s.name, s]));
  // The collision the old scheme had is resolved: distinct, kind-namespaced anchors.
  assert.equal(byName.node.anchor, 'node-function', 'node fn anchor must be kind-namespaced');
  assert.equal(byName.Node.anchor, 'node-model', 'Node model anchor must be kind-namespaced');
  assert.notEqual(byName.node.anchor, byName.Node.anchor, 'node/Node anchors must be distinct');
  const tree = run(para('Node'));
  // Positive: bare Node now autolinks to its OWN node-model anchor (Stage C heading).
  assert.ok(
    has(tree, (n) => n.type === 'link' && n.url.includes('#node-model')),
    'bare Node must autolink to its node-model anchor now that Stage C renders the heading',
  );
  // Negative: it must NOT cross-link to the decorator's node-function anchor.
  assert.ok(
    !has(tree, (n) => n.type === 'link' && n.url.includes('#node-function')),
    'bare Node must not autolink to the @node decorator (node-function) anchor',
  );
});

// (i2) The 4 collision-pair refs each RESOLVE to their kind-namespaced anchor now
// that Stage C (kec0k) renders the headings — the motivating defect (silent-inert
// node/Node/tool/Tool refs) is closed, pinned here at the plugin layer.
test('all four collision symbols autolink to their kind-namespaced anchors', () => {
  const cases = [
    ['node', '#node-function'],
    ['Node', '#node-model'],
    ['tool', '#tool-function'],
    ['Tool', '#tool-model'],
  ];
  for (const [token, anchor] of cases) {
    const tree = run(para(token));
    assert.ok(
      has(tree, (n) => n.type === 'link' && n.url.includes(anchor)),
      `bare \`${token}\` must autolink to ${anchor} (Stage C kind-namespaced heading)`,
    );
  }
});

// (j) L2 nested-link-guard: inlineCode inside an existing link is NOT re-wrapped.
test('inlineCode inside an existing link is not re-wrapped (no nested anchors)', () => {
  // Without the visitor guard `parent.type === 'link' -> return`, an inlineCode span
  // inside an existing markdown link ([`compile`](url)) would get re-wrapped in a second
  // link, producing nested <a> tags — invalid HTML, browser-rendered as broken. The guard
  // must skip inlineCode whose direct parent is already a link. `compile` is chosen because
  // it WOULD autolink if it weren't already inside a link, so this proves the guard bites.
  const tree = run(paraInLink('compile', 'https://example.com/custom'));
  let linkCount = 0;
  visit(tree, 'link', () => { linkCount += 1; });
  assert.equal(linkCount, 1, 'must not introduce a second (nested) link around inlineCode in a link');
});
