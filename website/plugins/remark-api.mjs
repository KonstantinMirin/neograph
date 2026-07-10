// remark-api autolinks + validates backticked API-symbol references against the
// introspection-generated manifest (scripts/gen_api_manifest.py -> api-manifest.json).
// Tiered confidence is the contract (verifiable-docs Stage B,
// docs/design/verifiable-docs-research-2026-07-09.md section 4):
//   HARD — a dotted `Type.member` whose owner has declared fields resolves or FAILS
//          the Astro build. Member-existence is pure manifest validation, so HARD has
//          full coverage and does not depend on the reference page having a heading.
//   SOFT — a bare token (compile, Node, Oracle) autolinks on EXACT manifest match WITH
//          a live reference-page heading, or stays inert. NEVER build-failing — bare
//          tokens collide with English ("run", "node", "tool").
//
// Ports the autolink+validate half of RAMP's remark-proto.mjs. The directive half
// (::proto-enum / ::proto-message tables) is deliberately dropped: neograph's reference
// content stays hand-written until Stage C generates it from the manifest. starlight-
// links-validator was evaluated and declined — it validates rendered <a> anchors, not
// backticked inlineCode source or member-existence, so it cannot replace this pass.
//
// Anchors are validated by slugging the REAL reference-page headings (RAMP-style) until
// Stage C makes anchors manifest-owned. The partial SOFT coverage (compile/node link;
// run/Oracle/Each stay inert because hand-written headings embed signatures) is the
// accepted Stage B seam Stage C eliminates.
import { visit, SKIP } from 'unist-util-visit';
import GithubSlugger from 'github-slugger';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const REF_PAGE = '/reference/api/';
const manifestFiles = [
  new URL('../src/data/api-manifest.json', import.meta.url),
  new URL('../src/data/api-manifest-mcp.json', import.meta.url),
];
const refPageFile = new URL('../src/content/docs/reference/api.mdx', import.meta.url);
const ignoreFile = new URL('../api-symbols-ignore.json', import.meta.url);

// HARD: `Type.member` where Type is PascalCase and member starts lowercase/underscore.
const RE_DOTTED = /^[A-Z][A-Za-z0-9]+\.[a-z_][A-Za-z0-9_]*$/;
// SOFT: a bare token — PascalCase (Node, Oracle) or snake_case (compile, run).
const RE_BARE_PASCAL = /^[A-Z][A-Za-z0-9]+$/;
const RE_BARE_SNAKE = /^[a-z][a-z0-9_]+$/;

let state;
function setup() {
  if (state) return state;

  // name -> { name, kind, anchor, fields? }; fieldedOwners = names with a non-empty
  // fields array (the HARD-tier validation set). Core + mcp manifests merge cleanly
  // (no name collisions across the two packages).
  const symbols = new Map();
  const fieldedOwners = new Set();
  for (const url of manifestFiles) {
    let payload;
    try {
      payload = JSON.parse(readFileSync(fileURLToPath(url), 'utf8'));
    } catch {
      continue; // mcp manifest may be absent in a core-only checkout
    }
    for (const s of payload.symbols ?? []) {
      symbols.set(s.name, s);
      if (Array.isArray(s.fields) && s.fields.length) fieldedOwners.add(s.name);
    }
  }

  // Live reference-page heading slugs gate autolinking (no dead anchors). A fresh
  // GithubSlugger dedups repeats (a second heading slugging to X becomes X-1). The
  // manifest anchors are now COLLISION-FREE: names are unique but a few bare slugs
  // collided (node/Node, tool/Tool -> one slug each), so the Python generator
  // kind-namespaces every colliding symbol's anchor to `${slug(name)}-${kindTag}`
  // (e.g. node-function / node-model). Stage C renders each symbol under a heading
  // whose text reproduces that exact anchor via github-slugger, so slugger dedup
  // never fires and the heading slug == the manifest anchor. This plugin keys
  // autolinks by symbol NAME -> anchor, so it needs no logic change.
  const slugger = new GithubSlugger();
  const headings = new Set();
  let fence = false;
  for (const ln of readFileSync(fileURLToPath(refPageFile), 'utf8').split('\n')) {
    if (ln.trimStart().startsWith('```')) { fence = !fence; continue; }
    if (fence) continue;
    const m = /^#{1,6}\s+(.+?)\s*$/.exec(ln);
    if (m) headings.add(slugger.slug(m[1]));
  }

  // Self-cleaning ignore file (empty by default; grows only for genuine historical-
  // contrast refs to removed fields).
  let ignoreEntries = [];
  try {
    ignoreEntries = JSON.parse(readFileSync(fileURLToPath(ignoreFile), 'utf8')).ignore ?? [];
  } catch {
    // no ignore file = empty ignore
  }
  const ignore = new Set(ignoreEntries);

  const ctx = { symbols, fieldedOwners, headings, ignore };

  // Stale check (self-cleaning). M1: predicate-based wouldResolve, NOT RAMP's verbatim
  // `symbols[k]` — neograph's map is name-keyed, so a dotted ignore entry's stale state
  // must be re-derived by re-running resolution, not looked up.
  const stale = [...ignore].filter((k) => wouldResolve(k, ctx));
  if (stale.length) {
    throw new Error(
      `api-symbols-ignore.json: stale entries now resolve in the manifest — remove: ${stale.sort().join(', ')}`,
    );
  }

  state = ctx;
  return ctx;
}

// Does a token now resolve cleanly (autolink or HARD-pass without throwing)? Drives the
// self-cleaning stale check: an ignore entry that resolves is stale and must be removed.
function wouldResolve(token, { symbols, fieldedOwners, headings }) {
  const dot = token.indexOf('.');
  if (dot > 0) {
    const owner = token.slice(0, dot);
    const member = token.slice(dot + 1);
    if (!fieldedOwners.has(owner)) return false; // inert owner: never resolved, never thrown
    const fields = symbols.get(owner)?.fields ?? [];
    return fields.some((f) => f.name === member);
  }
  const sym = symbols.get(token);
  return Boolean(sym) && headings.has(sym.anchor);
}

export default function remarkApi() {
  return (tree, file) => {
    const { symbols, fieldedOwners, headings, ignore } = setup();
    const where = file?.path ?? 'doc';
    const unresolved = new Set();

    visit(tree, 'inlineCode', (node, index, parent) => {
      // L2 guard: an inlineCode span already inside a markdown link must not be
      // re-wrapped (nested <a> tags = invalid HTML).
      if (!parent || index == null || parent.type === 'link') return;

      const t = node.value;
      const dot = t.indexOf('.');

      // HARD tier: dotted Type.member where the owner has declared fields.
      if (dot > 0 && RE_DOTTED.test(t)) {
        const owner = t.slice(0, dot);
        const member = t.slice(dot + 1);
        if (!fieldedOwners.has(owner)) return; // non-fielded owner: cannot validate -> inert
        const fields = symbols.get(owner)?.fields ?? [];
        if (fields.some((f) => f.name === member)) {
          const anchor = symbols.get(owner).anchor;
          if (headings.has(anchor)) {
            parent.children[index] = { type: 'link', url: `${REF_PAGE}#${anchor}`, children: [node] };
            return [SKIP, index + 1];
          }
          return; // resolved but undocumented on the reference page -> not linked (no dead anchor)
        }
        // Member missing from a fielded owner: the reference silently lies -> build-gate,
        // unless the token is ignore-listed (historical-contrast ref to a removed field).
        if (!ignore.has(t)) unresolved.add(t);
        return;
      }

      // SOFT tier: bare token, exact manifest match WITH a live heading -> autolink.
      // No match / no heading -> inert. NEVER throws (bare tokens collide with English).
      if (RE_BARE_PASCAL.test(t) || RE_BARE_SNAKE.test(t)) {
        const sym = symbols.get(t);
        if (sym && headings.has(sym.anchor)) {
          parent.children[index] = { type: 'link', url: `${REF_PAGE}#${sym.anchor}`, children: [node] };
          return [SKIP, index + 1];
        }
      }
      // everything else stays inert
    });

    if (unresolved.size) {
      throw new Error(
        `${where}: unknown api reference(s) — ${[...unresolved].sort().join(', ')}. ` +
          `Fix the member name, or if it references a removed/renamed field, add it to website/api-symbols-ignore.json.`,
      );
    }
  };
}
