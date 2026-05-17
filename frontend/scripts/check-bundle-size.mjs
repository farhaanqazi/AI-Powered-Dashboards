#!/usr/bin/env node
/**
 * Bundle-size guardrail — the "weight limit" so the app can't silently
 * bloat back to the 1.9 MB monolith.
 *
 * It parses dist/index.html for the scripts the browser loads on FIRST paint
 * (the entry + anything eagerly referenced), gzips them, and fails the build
 * if their combined transfer size exceeds the budget. Lazy route chunks
 * (dashboard/plotly/export) are NOT counted — they load on demand.
 *
 * Run: npm run check:size   (or npm run build:checked)
 */
import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import { join, resolve, basename } from 'node:path';

const DIST = resolve(process.cwd(), 'dist');
const INDEX = join(DIST, 'index.html');

// Budgets (gzipped transfer bytes).
const INITIAL_BUDGET = 320 * 1024;   // everything loaded on first paint
const ANY_CHUNK_BUDGET = 650 * 1024; // any single chunk (catches full-plotly regressions)

if (!existsSync(INDEX)) {
  console.error(`✗ ${INDEX} not found — run "vite build" first.`);
  process.exit(1);
}

const html = readFileSync(INDEX, 'utf8');
// First-paint JS = the entry <script> PLUS anything index.html eagerly
// modulepreloads (Vite preloads static-graph chunks). Lazy route chunks are
// not referenced here, so they are correctly excluded.
const scriptSrcs = [
  ...[...html.matchAll(/<script[^>]+src="([^"]+\.js)"/g)].map((m) => m[1]),
  ...[...html.matchAll(/<link[^>]+rel="modulepreload"[^>]+href="([^"]+\.js)"/g)].map((m) => m[1]),
];

const toDistPath = (src) => {
  const clean = src.replace(/^\.?\//, '');           // "./assets/..." | "/assets/..."
  return join(DIST, clean);
};

const gz = (p) => gzipSync(readFileSync(p)).length;
const kb = (n) => (n / 1024).toFixed(1) + ' KB';

let initial = 0;
const rows = [];
for (const src of scriptSrcs) {
  const p = toDistPath(src);
  if (!existsSync(p)) continue;
  const size = gz(p);
  initial += size;
  rows.push([basename(p), kb(size), 'initial']);
}

// Largest individual chunk across the whole build (lazy ones included).
let worst = { name: '', size: 0 };
const assetsRoot = join(DIST, 'assets');
if (existsSync(assetsRoot)) {
  for (const dir of readdirSync(assetsRoot)) {
    const sub = join(assetsRoot, dir);
    for (const f of readdirSync(sub)) {
      if (!f.endsWith('.js')) continue;
      const size = gz(join(sub, f));
      if (size > worst.size) worst = { name: f, size };
    }
  }
}

console.log('\nBundle size (gzipped):');
for (const [n, s, tag] of rows) console.log(`  [${tag}] ${n}  ${s}`);
console.log(`\n  Initial JS (first paint): ${kb(initial)}  / budget ${kb(INITIAL_BUDGET)}`);
console.log(`  Largest single chunk:     ${worst.name} ${kb(worst.size)}  / budget ${kb(ANY_CHUNK_BUDGET)}\n`);

let failed = false;
if (initial > INITIAL_BUDGET) {
  console.error(`✗ Initial JS ${kb(initial)} exceeds budget ${kb(INITIAL_BUDGET)}.`);
  failed = true;
}
if (worst.size > ANY_CHUNK_BUDGET) {
  console.error(`✗ Chunk "${worst.name}" ${kb(worst.size)} exceeds per-chunk budget ${kb(ANY_CHUNK_BUDGET)}.`);
  failed = true;
}
if (failed) process.exit(1);
console.log('✓ Bundle within budget.\n');
