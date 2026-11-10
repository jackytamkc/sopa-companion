// The shared golden fixture of Go deeper's scoring (fixtures/qc-scoring.json): fills in each case's `expect` from the
// page's own code. engine/qc_scoring.py, the same scoring in Python (the engine fixes a test's suspects with it), must
// reproduce every case exactly (tests/test_qc_scoring.py), and qc-scoring.test.mjs recomputes every one from
// qc-deeper.js, so a change on either side is seen. The cases' inputs come from tests/make_qc_scoring_inputs.py; then,
// from the app's folder:  env -u DISPLAY node frontend/test/make-qc-scoring.mjs --write
// It writes only when asked (--write) and never under node --test: a bare `node --test` runs every .mjs file under
// test/, this one too, and would silently rebuild the golden from the page as it is.
//
// expect = {params, scores, kept}, the JSON of:
//   params: what the page makes of the worker answer for the cell, nothing shown before: normalise, explain (its labels
//     cell + 1), the chips by rankOrder, sourceRows' rows, whereRequest: {state, none, chips, rows: [{label, cell, share,
//     same, key}], req: {key, params, labels} | null};
//   scores: what it makes of the where reply to that request (normaliseWhere against req, suspectScores on the
//     normalised answer: P = w x pi; the points at P >= GLOW_MIN, whoever's outline they lie in, with samePlace's rank
//     and of, x and y to 2 decimals as keepSuspects sends them; sourceSupport by source): {count, points: [{i, x, y,
//     gene, owner, near, w, pi, P, rank?, of?}], genes, support: [[k, support]] by k}, null where it takes no reply;
//   kept: the suspects a test of a setting starts with, the window's keepSuspects then the start's own checks
//     (testSuspects): [{x, y, gene, p, rank?, of?}], null where it takes no reply.
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import * as Q from '../src/qc-deeper.js';

export const FIXTURE = new URL('./fixtures/qc-scoring.json', import.meta.url);
const TEST_SUSPECTS = 400, GENE_CHARS = 64;   // a start's suspects at most, a gene name's characters at most

/** The page's parameters for worker answer `result` on cell `c` (label - 1), as Go deeper's window works them out. */
export function params(result, c) {
  const r = Q.normalise(result);
  if (!r) return { state: null, none: false, chips: [], rows: [], req: null };
  if (r.state !== 'ready') return { state: r.state, none: false, chips: [], rows: [], req: null };
  if (!Number.isInteger(c) || c < 0) throw new Error('cell must be a cell number (label - 1)');
  const m = Q.explain(r, { label: x => x + 1 });
  const nb = new Map(m.chips.map(k => [k.key, k.nb]));
  const chips = Q.rankOrder([], m.chips.map(k => k.key), k => nb.get(k));
  const rows = Q.sourceRows(m, []).rows;
  const req = Q.whereRequest(r, { cell: c + 1, none: m.none, rows, chips });
  return { state: r.state, none: !!m.none, chips,
    rows: rows.map(row => ({ label: row.label, cell: row.cell, share: row.share, same: !!row.same, key: row.key })),
    req: req ? { key: req.key, params: req.params, labels: req.labels } : null };
}

/** The where reply `raw` to request `req`, scored on the answer: its glowing points (any owner) and the sources'
 *  support. */
export function scores(result, req, raw) {
  const reply = Q.normaliseWhere(raw, req ?? null);
  if (!reply) return null;
  const s = Q.suspectScores(reply, Q.normalise(result));
  const idx = [];
  for (let i = 0; i < reply.count; i++) if (s.p[i] >= Q.GLOW_MIN) idx.push(i);
  const same = Q.samePlace(reply, idx);
  const points = idx.map(i => ({ i, x: +reply.x[i].toFixed(2), y: +reply.y[i].toFixed(2), gene: reply.genes[reply.gene[i]],
    owner: reply.owner[i], near: reply.near[i], w: reply.w[i], pi: s.pi[i], P: s.p[i], ...(same.get(i) ?? {}) }));
  const support = [...Q.sourceSupport(reply, s)].sort((a, b) => a[0] - b[0]);
  return { count: reply.count, points, genes: s.genes, support };
}

/** The window's keepSuspects (QcDeeper, qc-deeper.js) on a reply and its scores: its list. */
export function keepSuspects(reply, scores) {
  const list = [];
  for (let i = 0; i < reply.count; i++) {
    const P = scores.p[i];
    if (P >= Q.GLOW_MIN) list.push({ i, p: +P.toFixed(3) });
  }
  list.sort((a, b) => b.p - a.p);
  const top = list.slice(0, Q.SUSPECTS_SENT), shared = Q.samePlace(reply, top.map(s => s.i));
  return top.map(({ i, p }) => ({ x: +reply.x[i].toFixed(2), y: +reply.y[i].toFixed(2), gene: reply.genes[reply.gene[i]], p,
    ...shared.get(i) }));
}

/** The suspects a start carries (the panel's testSuspects): what the engine takes, the likeliest first. */
export function testSuspects(list) {
  const finite = Number.isFinite, round = (x, k) => Math.round(x * k) / k;
  return (Array.isArray(list) ? list : []).filter(s => s && finite(s.x) && finite(s.y) && finite(s.p) && s.p >= 0 && s.p <= 1
    && typeof s.gene === 'string' && s.gene.length >= 1 && s.gene.length <= GENE_CHARS).sort((a, b) => b.p - a.p).slice(0, TEST_SUSPECTS)
    .map(s => ({ x: round(s.x, 100), y: round(s.y, 100), gene: s.gene, p: round(s.p, 1000),
      ...(Number.isInteger(s.of) && s.of > 1 && Number.isInteger(s.rank) && s.rank >= 0 && s.rank < s.of ? { rank: s.rank, of: s.of } : {}) }));
}

/** The kept list for the reply to `req`, scored on the answer; null where the page takes no reply. */
export function kept(result, req, raw) {
  const reply = Q.normaliseWhere(raw, req ?? null);
  return reply ? testSuspects(keepSuspects(reply, Q.suspectScores(reply, Q.normalise(result)))) : null;
}

/** A case's expect, as its JSON holds it. */
export function expectOf(c) {
  const p = params(c.result, c.cell);
  return JSON.parse(JSON.stringify({ params: p, scores: scores(c.result, p.req, c.reply), kept: kept(c.result, p.req, c.reply) }));
}

/** One case per line. */
export const text = cases => `[\n${cases.map(c => JSON.stringify(c)).join(',\n')}\n]\n`;

/** Whether this file runs as the main module (`argv` as process.argv), never imported. */
export const isMain = (argv = process.argv) => !!argv[1] && import.meta.url === pathToFileURL(argv[1]).href;
/** Whether a run writes the fixture: this file as the main module, asked with --write, and not under node --test (its
 *  runs carry NODE_TEST_CONTEXT). */
export const writes = (argv = process.argv, env = process.env) => isMain(argv) && argv.includes('--write') && !env.NODE_TEST_CONTEXT;

if (writes()) {
  const cases = JSON.parse(readFileSync(FIXTURE, 'utf8'))
    .map(({ name, result, cell, reply }) => ({ name, result, cell, reply, expect: expectOf({ result, cell, reply }) }));
  const out = text(cases);
  writeFileSync(FIXTURE, out);
  console.log(`${cases.length} cases, ${Math.round(Buffer.byteLength(out) / 1024)} KB: ${FIXTURE.pathname}`);
} else if (isMain() && !process.env.NODE_TEST_CONTEXT) {
  console.log('Nothing written: the fixture is rewritten only with --write (node frontend/test/make-qc-scoring.mjs --write).');
}
