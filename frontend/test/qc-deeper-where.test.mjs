import test from 'node:test';
import assert from 'node:assert/strict';
import {
  QcDeeper, normalise, explain, sourceRows, whereRequest, normaliseWhere, assignSlots, wherePoints, whereView, whereFree, whereModel, whereDensity,
  suspectScores, sourceSupport, glowPoints, glowBoxOf, sourceCentre, pairBox, startPlaces, startIndex, samePlace,
  WHERE_COLORS, WHERE_ALPHA, WHERE_GENES, WHERE_OWN, WHERE_REFRESH_MS, WHERE_WAIT_MS, RETRY_MS, WORKING_MS, COUNTERS, WIDTH,
  GLOW_MIN, LIKELY_MIN, LIKELY_W, SUSPECTS_SENT, PAIR_MARGIN_UM, MATCH_UM, CHIPS, PLACES, HOLD, WHERE_V, HELD, WHERE_FEW,
} from '../src/qc-deeper.js';
import { DEEP_TEXT, allTexts, startTip, glowTip, noOutlineLine } from '../src/qc-deeper-copy.mjs';
import { bannedWords } from '../src/qc-copy.mjs';

// Go deeper's "Show on the image" and "Where they sit": the
// request the window makes, the reply it takes (v 3: each point's proseg share w, and where the shares came from), the
// suspects it scores (P = w x pi, on any owner) and the glow it hands the stage, where they sat at the start
// (startPlaces) and the line that says so, with the start's hold in its tooltip and the caveat of approximate shares,
// the dots, the camera's move (clear of the window and the test corner), the words per source; the strip case
// (pink dots near none of the neighbour that would be named: a cell with no outline here, no side to ripple in from, the
// camera on the cell and its glow); then the window itself in a small DOM with a virtual clock and a scripted engine (the
// endpoint faked: engine/deeper_where.py answers it). Last, suspects that share a place carry their rank by depth, and
// the glow and the suspects follow a replaced result of the same frame.

// ---- a small DOM and a virtual clock (as qc-deeper.test.mjs) ----------------------------------------------------------
function dom() {
  const doc = { activeElement: null, hidden: false };
  const camel = s => s.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
  class Node {
    constructor(tag) {
      Object.assign(this, { tagName: tag.toUpperCase(), nodeType: tag === '#text' ? 3 : 1, childNodes: [], parentElement: null, attributes: new Map(),
        listeners: new Map(), style: {}, dataset: {}, hidden: false, _text: '', className: '', ownerDocument: doc, title: '' });
      const self = this, set = () => new Set(self.className.split(/\s+/).filter(Boolean));
      this.classList = {
        add: (...c) => { const s = set(); c.forEach(x => s.add(x)); self.className = [...s].join(' '); },
        remove: (...c) => { const s = set(); c.forEach(x => s.delete(x)); self.className = [...s].join(' '); },
        toggle: (c, on) => { const s = set(), want = on === undefined ? !s.has(c) : !!on; if (want) s.add(c); else s.delete(c); self.className = [...s].join(' '); return want; },
        contains: c => set().has(c),
      };
    }
    get children() { return this.childNodes.filter(n => n.nodeType === 1); }
    get textContent() { return this._text + this.childNodes.map(c => c.textContent).join(''); }
    set textContent(v) { this._text = String(v ?? ''); for (const c of this.childNodes) c.parentElement = null; this.childNodes = []; }
    append(...nodes) { for (let n of nodes) { if (typeof n !== 'object') n = doc.createTextNode(n); n.remove(); n.parentElement = this; this.childNodes.push(n); } }
    replaceChildren(...nodes) { for (const c of this.childNodes) c.parentElement = null; this.childNodes = []; this._text = ''; this.append(...nodes); }
    remove() { const p = this.parentElement; if (p) { p.childNodes = p.childNodes.filter(n => n !== this); this.parentElement = null; } }
    get isConnected() { let n = this; while (n.parentElement) n = n.parentElement; return n === doc.body; }
    setAttribute(k, v) { this.attributes.set(k, String(v)); if (k === 'id') this.id = String(v); }
    getAttribute(k) { return k === 'title' ? this.title || null : this.attributes.has(k) ? this.attributes.get(k) : null; }
    addEventListener(t, fn) { this.listeners.set(t, [...(this.listeners.get(t) || []), fn]); }
    removeEventListener(t, fn) { this.listeners.set(t, (this.listeners.get(t) || []).filter(f => f !== fn)); }
    emit(t, extra = {}) { const e = { type: t, target: this, button: 0, preventDefault() {}, stopPropagation() {}, ...extra }; for (const fn of this.listeners.get(t) || []) fn(e); return e; }
    click() { return this.onclick?.({ target: this, stopPropagation() {}, preventDefault() {} }); }
    contains(n) { while (n) { if (n === this) return true; n = n.parentElement; } return false; }
    closest(tag) { let n = this; while (n) { if (n.tagName === tag.toUpperCase()) return n; n = n.parentElement; } return null; }
    focus() { doc.activeElement = this; }
    getContext() { return new Proxy({}, { get: () => () => ({ width: 30 }) }); }
    matches(sel) {
      const m = /^([a-z0-9]*)((?:\.[\w-]+)*)((?:\[[^\]]+\])*)$/i.exec(sel.trim());
      if (!m) throw new Error(`selector ${sel}`);
      if (m[1] && this.tagName !== m[1].toUpperCase()) return false;
      for (const c of m[2].match(/\.[\w-]+/g) || []) if (!this.classList.contains(c.slice(1))) return false;
      for (const a of m[3].match(/\[[^\]]+\]/g) || []) {
        const [k, v] = a.slice(1, -1).split('='), name = k.trim();
        const value = name.startsWith('data-') ? this.dataset[camel(name.slice(5))] : name === 'id' ? this.id : this.getAttribute(name);
        if (value == null) return false;
        if (v != null && String(value) !== v.replace(/^["']|["']$/g, '')) return false;
      }
      return true;
    }
    all(sel) { const out = []; const walk = n => { for (const c of n.children) { if (c.matches(sel)) out.push(c); walk(c); } }; walk(this); return out; }
    one(sel) { return this.all(sel)[0] ?? null; }
    visibleText() { return this.hidden ? '' : this._text + this.childNodes.map(c => c.nodeType === 3 ? c.textContent : c.visibleText()).join(' '); }
  }
  doc.createElement = tag => new Node(tag);
  doc.createTextNode = text => { const n = new Node('#text'); n._text = String(text); return n; };
  doc.body = new Node('body');
  const host = new Node('main'); doc.body.append(host);
  return { doc, host };
}
function clock() {
  const c = { t: 1000, timers: [], rafs: [], id: 0 };
  c.now = () => c.t;
  c.setTimeout = (fn, ms) => { const id = ++c.id; c.timers.push({ id, fn, at: c.t + Math.max(0, ms) }); return id; };
  c.clearTimeout = id => { c.timers = c.timers.filter(x => x.id !== id); };
  c.requestAnimationFrame = fn => { const id = ++c.id; c.rafs.push({ id, fn }); return id; };
  c.cancelAnimationFrame = id => { c.rafs = c.rafs.filter(x => x.id !== id); };
  c.advance = ms => {
    const end = c.t + ms;
    for (let guard = 0; guard < 10000; guard++) {
      if (c.rafs.length) { const list = c.rafs; c.rafs = []; list.forEach(r => r.fn(c.t)); continue; }
      const due = c.timers.filter(x => x.at <= end).sort((a, b) => a.at - b.at || a.id - b.id)[0];
      if (!due) break;
      c.t = Math.max(c.t, due.at); c.timers = c.timers.filter(x => x !== due); due.fn();
    }
    c.t = end;
  };
  return c;
}
const settle = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise(resolve => setImmediate(resolve)); };

// ---- results (qc-deeper.test.mjs's cells) and the endpoint's replies ---------------------------------------------------
const CELLS = ['c0', 'ffaaaaaa-1', 'ffbbbbbb-1', 'ggcccccc-1', 'ggaaaaaa-1', 'ggbbbbbb-1', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11'];
/** ffaaaaaa-1 at iteration 1201 (cell 1): 46 %, GXPN 31 of 39, its neighbour of another kind ffbbbbbb-1 (one source row). */
function mixed(extra = {}) {
  return { v: 1, run: 'r1', seq: 12, iter: 1201, cell: 1, model: 'm1', state: 'ready', transcripts: 682, outside: 3, code: 116,
    parts: [0.539, 0.46, 0.001], rough: false,
    genes: [['GXPN', 39, 7.9, 31.1, 1, 0], ['GYL', 11, 5.1, 5.9, 0, 0], ['GXC', 10, 1.6, 8.4, 1, 0], ['GYHNN', 9, 5.5, 3.5, 0], ['GXDN', 7, 2.3, 4.7, 0],
      ['GYDNNN', 4, 0.7, 3.3, 1], ['GYJ', 4, 0.9, 3.1, 1], ['GYCNNN', 3, 0.5, 2.5, 1], ['GXT', 3, 0.7, 2.3, 1], ['GXANN', 2, 0.3, 1.7, 1],
      ['GXRN', 2, 0.35, 1.65, 1], ['GXEN', 4, 4, 0, 0], ['GZCNN', 4, 4, 0, 0], ['GYSNN', 1, 1, 0, 0]],
    kind: ['GYSNN', 'GYRN'], own_genes: [['GYHNN', 5.5, 9], ['GXEN', 4, 4], ['GZCNN', 4, 4], ['GYWNN', 3, 3]], same_kind_share: 0.02,
    neighbours: [
      { cell: 2, um: 25.2, tx: 300, genes: [['GXPN', 31.1], ['GXC', 8.4]], kind: ['GXPN', 'GXDN', 'GXT'], similar: 0.31, known: true },
      ...[6, 7, 8, 9, 10, 11, 3, 4, 5].map((cell, k) => ({ cell, um: 30 + k, tx: 0.4, genes: [['GYHNN', 0.2]], kind: ['GXHNN'], similar: 0.5, known: true })),
    ],
    learned: 'yes', start_change: { points: 44.1, band: 3, draws: 8 }, ms: 1.4, ...extra };
}
/** Little from neighbours (under 5 %): no source rows, so nothing to show where. */
function little(extra = {}) {
  return mixed({ code: 2, parts: [0.989, 0.01, 0.001], genes: [['GXDN', 15, 14.8, 0.2, 0]], neighbours: mixed().neighbours.map(n => ({ ...n, tx: 0.5 })), ...extra });
}
/** Four source rows (two of them the same kind), three chips. */
function lookalike(extra = {}) {
  return mixed({ code: 143, parts: [0.429, 0.57, 0.001], transcripts: 1396, same_kind_share: 0.81,
    genes: [['GXPN', 29, 10.8, 18.2, 0], ['GYPNNN', 5, 0.1, 4.9, 1, 2], ['GZCNN', 4, 0.9, 3.1, 1, 2], ['GYF', 3, 0.2, 2.8, 1, 2]],
    own_genes: [['GXPN', 10.8, 29]],
    neighbours: [
      { cell: 2, um: 17, tx: 460, genes: [['GXPN', 12]], kind: ['GZA'], similar: 0.95, known: true },
      { cell: 3, um: 20, tx: 180, genes: [['GXPN', 5]], kind: ['GXHNN'], similar: 0.93, known: true },
      { cell: 4, um: 19, tx: 125, genes: [['GYPNNN', 4.9], ['GZCNN', 3.1], ['GYF', 2.8]], kind: ['GYPNNN', 'GYF'], similar: 0.4, known: true },
      { cell: 5, um: 15, tx: 30, genes: [['GZDNN', 1]], kind: ['GYXN'], similar: 0.5, known: false },
      ...[6, 7, 8, 9, 10, 11].map(cell => ({ cell, um: 30, tx: 0.5, genes: [], kind: ['GXHNN'], similar: 0.96, known: true })),
    ], ...extra });
}
const ready = r => ({ state: 'ready', seq: r.seq, cell: r.cell, model: 'm1', deep: true, result: r });
/** The endpoint's reply to `params` (version 3 of its answer):
 *  GXPN at the shared edge with ffbbbbbb-1, GXC as common around, the others too few; GYHNN found inside the source.
 *  `points`: [x, y, gene index, owner, near, w] (owner 0 none, 1 the cell, 2 + k source k, 255 another cell; near 1: a
 *  "from" gene's point of the cell within 2 µm of its giver, or a source's own-gene point within 2 µm of the cell; w,
 *  proseg's share of the point for the cell: by default all of it inside the cell's outline, none elsewhere); `held`
 *  where the shares came from. */
function whereReply(params, { points, stats, into, box = [100, 100, 180, 160], total = 685, truncated = false, v = 3, held = 'exact' } = {}) {
  const genes = [...params.genes, ...params.own], roles = [...params.genes.map(() => 'from'), ...params.own.map(() => 'own')];
  const own = genes.indexOf('GYHNN');
  const pts = points ?? [[110, 120, 0, 1, 1], [111, 121, 0, 1, 0], [140, 125, 0, 2, 0], [170, 150, 0, 255, 0], [112, 128, 1, 0, 0], [113, 129, 1, 1, 0],
    [150, 130, own, 2, 1], [114, 131, own, 1, 0], [175, 155, own, 0, 0]];
  return { v, run: 'r1', seq: params.seq, iter: 1201, cell: params.cell, sources: params.sources, basis: 'column', held, held_seq: held === 'exact' ? params.seq : null,
    near_um: 2, box, genes, roles,
    points: { x: pts.map(p => p[0]), y: pts.map(p => p[1]), z: pts.map(() => 0.5), gene: pts.map(p => p[2]), owner: pts.map(p => p[3]),
      near: pts.map(p => p[4] ?? 0), w: pts.map(p => p[5] ?? (p[3] === 1 ? 1 : 0)) },
    cell_total: total, cell_area_um2: 134, source_area_um2: params.sources.map(() => 150), truncated,
    gene_stats: stats ?? [
      { gene: 'GXPN', role: 'from', giver: 0, giver_by: 'result', in_cell: 39, near_giver: 31, own_near: [21, 80], density: { cell: 29.1, giver: 40.6, around: 0.4 }, word: 'edge' },
      { gene: 'GXC', role: 'from', giver: 0, giver_by: 'result', in_cell: 10, near_giver: 3, own_near: [21, 80], density: { cell: 7.5, giver: 9.2, around: 5.1 }, word: 'layer' },
      { gene: 'GYDNNN', role: 'from', giver: null, giver_by: null, in_cell: 3, near_giver: null, own_near: null, density: { cell: 3, giver: null, around: 0 }, word: 'few' },
      { gene: 'GYHNN', role: 'own', giver: null, giver_by: null, in_cell: 9, near_giver: null, own_near: null, density: { cell: 6.7, giver: null, around: 2 }, word: 'spread' },
    ],
    into: into ?? params.sources.flatMap((s, k) => params.own.map((gene, i) => ({ source: k, gene, inside: gene === 'GYHNN' && k === 0 ? 4 : 0,
      near_cell: gene === 'GYHNN' && k === 0 ? 3 : 0, source_own_near: [30, 400] }))), ms: 12.5 };
}
/** The endpoint's reply to the start line's request (seq "start", the same cell, sources and genes, no own
 *  genes): `points` [gene, x, y, owner] placed by gene name (those of genes not asked left out). By default the cell held
 *  the glowing GXPN transcript at the shared edge at the start (110, 120), not GXC's (113, 129: its neighbour's then). */
function startReply(params, { points = [['GXPN', 110, 120, 1], ['GXPN', 111, 121, 0], ['GXC', 113, 129, 2], ['GXPN', 140, 125, 2], ['GXC', 99, 99, 1]] } = {}) {
  const genes = [...params.genes, ...params.own], roles = [...params.genes.map(() => 'from'), ...params.own.map(() => 'own')];
  const pts = points.filter(p => genes.includes(p[0])).map(([g, x, y, o]) => [x, y, genes.indexOf(g), o]);
  return { v: 3, run: 'r1', seq: params.seq, iter: 0, cell: params.cell, sources: params.sources, basis: 'start', held: 'approx', held_seq: null,
    near_um: 2, box: pts.length ? [90, 90, 150, 140] : null, genes, roles,
    points: { x: pts.map(p => p[0]), y: pts.map(p => p[1]), z: pts.map(() => 0.5), gene: pts.map(p => p[2]), owner: pts.map(p => p[3]), near: pts.map(() => 0),
      w: pts.map(p => (p[3] === 1 ? 1 : 0)) },
    cell_total: 161, truncated: false, gene_stats: [], into: [], ms: 3 };
}
/** The request the window makes for the mixed cell's result (its one source row, its chips, its own genes). */
function mixedRequest(r = normalise(mixed())) {
  const m = explain(r, { name: c => CELLS[c], label: c => c + 1 });
  const { rows } = sourceRows(m);
  return whereRequest(r, { cell: 2, none: m.none, rows, chips: m.chips.map(c => c.gene) });
}

// ---- the rules (pure) ---------------------------------------------------------------------------------------------------
test('the request: the result\'s own frame, the cell and its source rows as label - 1, the chips\' genes (6), its own genes (3)', () => {
  const req = mixedRequest();
  assert.deepEqual(req.params, { seq: 12, cell: 1, sources: [2], genes: ['GXPN', 'GXC', 'GYDNNN', 'GYJ', 'GYCNNN', 'GXT'], own: ['GYHNN', 'GXEN', 'GZCNN'] });
  assert.deepEqual(req.labels, [3]); assert.deepEqual(req.sources.map(s => s.name), ['ffbbbbbb-1']);
  assert.equal(WHERE_GENES, 6); assert.equal(WHERE_OWN, 3);
  // The start line's split asks for the start line; a gene in both lists is asked once, as a neighbour's.
  const start = normalise(mixed({ seq: 'start', own_genes: [['GXPN', 5, 9], ['GYHNN', 5.5, 9]] }));
  const s = whereRequest(start, { cell: 2, rows: [{ label: 3, name: 'ffbbbbbb-1', head: 'x' }], chips: ['GXPN'] });
  assert.equal(s.params.seq, 'start'); assert.deepEqual(s.params.own, ['GYHNN']);
  // Every source row is asked (same kind ones too: they are lit), at most five, in the order shown.
  const a = normalise(lookalike()), am = explain(a, { name: c => CELLS[c], label: c => c + 1 });
  const ar = whereRequest(a, { cell: 2, rows: sourceRows(am).rows, chips: am.chips.map(c => c.gene) });
  assert.deepEqual(ar.params.sources, [2, 3, 4, 5]); assert.deepEqual(ar.params.genes, ['GYPNNN', 'GZCNN', 'GYF']); assert.deepEqual(ar.params.own, ['GXPN']);
  // Nothing to show: under 5 % from neighbours, no source row, no ready result.
  const l = normalise(little()), lm = explain(l, { name: c => CELLS[c], label: c => c + 1 });
  assert.equal(whereRequest(l, { cell: 2, none: lm.none, rows: sourceRows(lm).rows, chips: [] }), null);
  assert.equal(whereRequest(normalise(mixed()), { cell: 2, rows: [], chips: ['GXPN'] }), null);
  assert.equal(whereRequest(normalise({ v: 1, state: 'too_few', seq: 12, cell: 1, transcripts: 61, genes: [] }), { cell: 2, rows: [{ label: 3 }] }), null);
  assert.notEqual(mixedRequest().key, s.key, 'another frame is another request');
});

test('a reply (v 3: its points\' near flags and proseg\'s share of each) is checked against its request and its points become typed arrays; a malformed one is refused', () => {
  const req = mixedRequest(), raw = whereReply(req.params), r = normaliseWhere(raw, req);
  assert.equal(r.count, 9); assert.ok(r.x instanceof Float32Array && r.gene instanceof Uint8Array && r.owner instanceof Uint8Array);
  // Version 3: each point's near flag and proseg's share of it for the cell (the doubles the reply gives), and where
  // those shares came from; an older engine's reply (v 1 or 2, no shares) is not taken.
  assert.deepEqual([...WHERE_V], [3]); assert.deepEqual([...HELD], ['exact', 'approx', 'none']);
  assert.equal(r.v, 3); assert.ok(r.near instanceof Uint8Array); assert.deepEqual([...r.near], [1, 0, 0, 0, 0, 0, 1, 0, 0]);
  assert.ok(r.w instanceof Float64Array); assert.deepEqual([...r.w], [1, 1, 0, 0, 0, 1, 0, 1, 0]);
  assert.deepEqual([r.held, r.held_seq], ['exact', 12]);
  for (const v of [1, 2, '3', undefined, null, 4]) assert.equal(normaliseWhere({ ...raw, v }, req), null, String(v));
  const flat = normaliseWhere({ ...raw, points: { ...raw.points, near: undefined } }, req);
  assert.deepEqual([...flat.near], new Array(9).fill(0), 'without near flags: all 0');
  assert.deepEqual([normaliseWhere({ ...raw, held: 'maybe' }, req).held, normaliseWhere({ ...raw, held_seq: '12' }, req).held_seq], [null, null]);
  const share = normaliseWhere(whereReply(req.params, { points: [[1, 2, 0, 1, 0, 0.375]] }), req);
  assert.equal(share.w[0], 0.375);
  assert.deepEqual(r.genes, [...req.params.genes, ...req.params.own]); assert.equal(r.cell_total, 685);
  assert.equal(normaliseWhere({ ...raw, cell_total: 12.375 }, req).cell_total, 12.375, 'proseg\'s count: any number from 0');
  assert.equal(normaliseWhere({ ...raw, cell_total: -1 }, req).cell_total, null);
  assert.equal(r.stats[0].word, 'edge'); assert.deepEqual(r.stats[0].own_near, [21, 80]); assert.equal(r.stats[2].giver, null);
  assert.equal(r.into[0].gene, 'GYHNN');
  const bad = [
    { ...raw, v: 2 }, { ...raw, cell: 7 }, { ...raw, seq: 13 },
    { ...raw, points: { ...raw.points, w: undefined } }, { ...raw, points: { ...raw.points, w: raw.points.w.slice(1) } },
    { ...raw, points: { ...raw.points, w: raw.points.w.map((v, i) => (i === 3 ? 1.5 : v)) } },
    { ...raw, points: { ...raw.points, w: raw.points.w.map((v, i) => (i === 3 ? -0.25 : v)) } },
    { ...raw, points: { ...raw.points, w: raw.points.w.map((v, i) => (i === 3 ? '0.5' : v)) } },
    { ...raw, points: { ...raw.points, w: raw.points.w.map((v, i) => (i === 3 ? null : v)) } },
    { ...raw, points: { ...raw.points, near: raw.points.near.slice(1) } }, { ...raw, points: { ...raw.points, near: raw.points.near.map(() => 2) } },
    { ...raw, points: { ...raw.points, near: 'all' } }, { ...raw, roles: raw.roles.slice(1) }, { ...raw, roles: raw.roles.map(() => 'other') },
    { ...raw, box: [10, 10, 5, 20] }, { ...raw, points: { ...raw.points, owner: raw.points.owner.slice(1) } },
    { ...raw, points: { ...raw.points, gene: raw.points.gene.map(() => 99) } }, { ...raw, points: { ...raw.points, x: raw.points.x.map(() => NaN) } },
    null, 'x'];
  for (const b of bad) assert.equal(normaliseWhere(b, req), null, JSON.stringify(b)?.slice(0, 80));
  // Neither the cell nor a source holds anything in that frame: no box, no dots, still a reply.
  const empty = normaliseWhere({ ...raw, box: null, points: { x: [], y: [], z: [], gene: [], owner: [], w: [] } }, req);
  assert.equal(empty.box, null); assert.equal(empty.count, 0);
  // An unknown word is shown as no word; extra fields are fine.
  assert.equal(normaliseWhere({ ...raw, gene_stats: [{ ...raw.gene_stats[0], word: 'maybe' }], extra: 1 }, req).stats[0].word, null);
});

test('a gene keeps its colour while the toggle is on: ranks change, colours do not; a leaving gene frees its colour', () => {
  const slots = new Map();
  assignSlots(slots, ['GXPN', 'GXC', 'GYJ'], 6);
  assert.deepEqual([...slots], [['GXPN', 0], ['GXC', 1], ['GYJ', 2]]);
  assignSlots(slots, ['GYJ', 'GXPN', 'GXC'], 6);
  assert.deepEqual([...slots], [['GXPN', 0], ['GXC', 1], ['GYJ', 2]], 'a new rank, the same colours');
  assignSlots(slots, ['GYJ', 'GXT', 'GXC'], 6);
  assert.deepEqual(Object.fromEntries(slots), { GXC: 1, GYJ: 2, GXT: 0 }, 'GXPN left; GXT takes the lowest free colour');
  assignSlots(slots, ['A', 'B', 'C', 'D', 'E', 'F', 'G'], 3);
  assert.equal(slots.size, 3, 'never more genes than colours');
  // The palette: six for the chips' genes, never magenta, the own blue or white; three blues for its own genes.
  assert.equal(WHERE_COLORS.from.length, 6); assert.equal(WHERE_COLORS.own.length, 3); assert.equal(WHERE_COLORS.own[0], '#3987e5');
  for (const c of WHERE_COLORS.from) assert.ok(!['#d55181', '#3987e5', '#ffffff'].includes(c.toLowerCase()), c);
  assert.equal(new Set([...WHERE_COLORS.from, ...WHERE_COLORS.own]).size, 9);
});

test('the dots: four floats each, paler first, the chips\' genes in the cell and its sources on top', () => {
  const req = mixedRequest(), r = normaliseWhere(whereReply(req.params), req);
  const out = wherePoints(r, req.labels.length);
  assert.ok(out instanceof Float32Array); assert.equal(out.length, 4 * r.count);
  const dots = Array.from({ length: r.count }, (_, i) => [...out.slice(4 * i, 4 * i + 4)]);
  const own = r.genes.indexOf('GYHNN');
  // Own genes where expected (the cell, around), then chips' genes around, then own genes inside a source, then the rest.
  assert.deepEqual(dots.map(d => d[3]), [WHERE_ALPHA.expected, WHERE_ALPHA.expected, WHERE_ALPHA.around, WHERE_ALPHA.around,
    WHERE_ALPHA.lit, WHERE_ALPHA.lit, WHERE_ALPHA.lit, WHERE_ALPHA.lit, WHERE_ALPHA.lit].map(Math.fround));
  assert.deepEqual(dots.slice(0, 2).map(d => d[2]), [own, own]); assert.deepEqual(dots[4], [150, 130, own, 1]);
  assert.deepEqual(dots.slice(5).map(d => [d[0], d[1], d[2]]), [[110, 120, 0], [111, 121, 0], [140, 125, 0], [113, 129, 1]]);
  // An owner past the sources asked is "around" (another cell's), never lit.
  const lone = normaliseWhere(whereReply(req.params, { points: [[1, 2, 0, 3]] }), req);
  assert.equal(wherePoints(lone, 1)[3], Math.fround(WHERE_ALPHA.around));
});

test('the camera\'s one move puts the cell and its sources in the part of the stage the window leaves free', () => {
  const stage = { width: 1200, height: 800 }, win = { left: 1200 - WIDTH - 12, top: 52, w: WIDTH, h: 500 };
  const box = [100, 100, 180, 160];
  // stage.fit(rect, animate, 0) as the stage does it: zoom = min(W / w, H / h), centred on the rect.
  const fit = rect => { const zoom = Math.min(stage.width / rect.w, stage.height / rect.h); return { zoom, x: rect.x0 + rect.w / 2, y: rect.y0 + rect.h / 2 }; };
  const toScreen = (cam, x, y) => [(x - cam.x) * cam.zoom + stage.width / 2, (y - cam.y) * cam.zoom + stage.height / 2];
  const cam = fit(whereView(box, stage, win));
  const [ax, ay] = toScreen(cam, box[0], box[1]), [bx, by] = toScreen(cam, box[2], box[3]);
  assert.ok(ax >= 12 - 1e-6 && bx <= win.left - 12 + 1e-6, `left of the window: ${ax}..${bx}`);
  assert.ok(ay >= 12 - 1e-6 && by <= stage.height - 12 + 1e-6, `inside the stage: ${ay}..${by}`);
  assert.ok(Math.abs(bx - ax - (win.left - 24)) < 1e-6 || Math.abs(by - ay - (stage.height - 24)) < 1e-6, 'as large as it fits');
  // A window on the left: the box goes right of it; no window, or one too wide to leave a side: the whole stage.
  const leftCam = fit(whereView(box, stage, { left: 12, top: 52, w: WIDTH, h: 500 }));
  assert.ok(toScreen(leftCam, box[0], box[1])[0] >= 12 + WIDTH + 12 - 1e-6);
  const narrow = { width: 500, height: 800 };
  const whole = whereView(box, narrow, { left: 128, top: 52, w: WIDTH, h: 500 }), full = whereView(box, narrow, null);
  assert.deepEqual(whole, full);
});

test('Where they sit: per source its genes with their word (the whole numbers, x of y, in the line\'s tooltip); its own genes inside it on one line; the rest apart', () => {
  const req = mixedRequest(), r = normaliseWhere(whereReply(req.params), req);
  const colors = r.genes.map((g, i) => (r.roles[i] === 'own' ? WHERE_COLORS.own[i - req.params.genes.length] : WHERE_COLORS.from[i]));
  const m = whereModel(r, req, colors);
  assert.equal(m.sources.length, 1); assert.equal(m.sources[0].name, 'ffbbbbbb-1');
  assert.equal(m.sources[0].head, 'ffbbbbbb-1 · makes GXPN · GXDN · GXT', 'the row\'s own words');
  const [gxpn, gxc] = m.sources[0].genes;
  assert.deepEqual([gxpn.gene, gxpn.word, gxpn.text, gxpn.color], ['GXPN', 'edge', 'GXPN · at the shared edge', WHERE_COLORS.from[0]]);
  // Every number on proseg's count (its shares for the cell summed; the tissue around at the cell's keep).
  assert.deepEqual(gxpn.numbers, ["31 of the 39 proseg counts for this cell lie within 2 µm of it (the rest of this cell's RNA: 21 of 80).",
    "Per 100 µm², in proseg's count: 29 in this cell · 41 in it · under 1 in the cells around."]);
  assert.equal(gxpn.tooltip, `${DEEP_TEXT.deepWhereEdgeTooltip} ${gxpn.numbers.join(' ')}`, 'the word\'s meaning, then its numbers');
  assert.equal(gxc.text, 'GXC · as common around');
  const alone = whereModel(normaliseWhere(whereReply(req.params, { stats: [{ ...whereReply(req.params).gene_stats[0], own_near: [0, 0] }] }), req), req, colors);
  assert.equal(alone.sources[0].genes[0].numbers[0], '31 of the 39 proseg counts for this cell lie within 2 µm of it.', 'no own genes asked: no baseline said');
  // This cell's own genes inside the neighbour's outline: one line of proseg's count of them for this cell (the reply's
  // sums of its shares; the outline only says where), their numbers and what it counts for this cell of the other
  // transcripts inside that outline in its tooltip.
  assert.deepEqual(m.sources[0].into, { genes: ['GYHNN'], color: WHERE_COLORS.own[0], text: "Inside the neighbour's outline, proseg counts for this cell: GYHNN 4",
    tooltip: `${DEEP_TEXT.deepWhereIntoTooltip} GYHNN: 4, 3 of them within 2 µm of this cell. Other transcripts inside the neighbour's outline: proseg counts 400 of them for this cell, 30 within 2 µm of this cell.` });
  assert.doesNotMatch(m.sources[0].into.tooltip, /other RNA|this cell's RNA reaches/);
  assert.deepEqual(m.other.map(g => [g.text, g.numbers]), [['GYDNNN · too few to tell', ['Proseg counts 3 for this cell.']]], 'its neighbour not listed');
  assert.equal(m.truncated, null); assert.equal(m.none, false);
  assert.equal(m.line, null, 'no glow: no line');
  // The glow's line: who the glowing transcripts likely come from, every neighbour's or one row's (its Show), then where
  // they sat at the start once that is known (startPlaces, below); how many sit at the shared edge, in its tooltip.
  const g = whereModel(normaliseWhere(whereReply(req.params), req), req, colors, { count: 3, likely: 2, near: 1, places: null, levels: null, row: null, notNear: null });
  assert.deepEqual(g.line, { text: '2 likely from neighbours', tip: startTip({ count: 3, likely: 2, near: 1 }),
    counts: { likely: 2, own: null, none: null, giver: null, other: null, unknown: null } });
  assert.match(g.line.tip, /At the shared edge, within 2 µm of their neighbour: 1 of the likely ones\.$/);
  assert.deepEqual(g.unlit, { text: 'No edge measured for GYDNNN', tip: DEEP_TEXT.deepGlowUnplacedTooltip }, 'a gene whose neighbour is not listed');
  const row = whereModel(normaliseWhere(whereReply(req.params), req), req, colors, { count: 3, likely: 2, near: 0, row: 3 });
  assert.equal(row.line.text, '2 likely from ffbbbbbb-1'); assert.equal(row.unlit, null, 'one row\'s: what other rows hold is not said');
  assert.equal(row.line.tip, DEEP_TEXT.deepGlowTooltip);
  assert.equal(whereModel(r, req, colors, { count: 4, likely: 0, near: 0, row: null }).line.text, '4 maybe from neighbours');
  const none = whereModel(r, req, colors, { count: 0, likely: 0, near: 0, places: { own: 0, none: 0, giver: 0, other: 0, unknown: 0 }, row: 3 }).line;
  assert.deepEqual([none.text, none.counts], ['None looks likely from ffbbbbbb-1', { likely: 0, own: null, none: null, giver: null, other: null, unknown: null }]);
  // Once known: a part per place that holds some, its numbers for the harness.
  const placed = whereModel(r, req, colors, { count: 3, likely: 2, near: 1, places: { own: 1, none: 0, giver: 1, other: 0, unknown: 0 }, row: null }).line;
  assert.equal(placed.text, "2 likely from neighbours · 1 inside its own starting outline · 1 inside a neighbour's starting outline");
  assert.deepEqual(placed.counts, { likely: 2, own: 1, none: 0, giver: 1, other: 0, unknown: 0 });
  assert.deepEqual([0, 0.4, 0.99, 1.4, 29.6, 1234.4].map(whereDensity), ['0', 'under 1', 'under 1', '1', '30', '1,234']);
  const cut = whereModel(normaliseWhere(whereReply(req.params, { truncated: true }), req), req, colors);
  assert.equal(cut.truncated, 'Only 9 transcripts are drawn: there are more in this area.');
  for (const text of [JSON.stringify(m), cut.truncated]) assert.deepEqual(bannedWords(text), [], text);
  for (const text of allTexts()) assert.deepEqual(bannedWords(text), [], text);
});

// ---- the window ---------------------------------------------------------------------------------------------------------
/** The window with a scripted engine: `where` answers the frames' requests, `start` the start line's (seq "start", design
 *  kept apart, calls.starts), each a reply, an Error to throw or a promise. */
function make({ answer = body => ready(mixed({ seq: body.seq })), where = null, start = null, live = true, seq = 12, run = {}, reduced = false,
  view3d = false } = {}) {
  const { doc, host } = dom(), c = clock(), counters = {};
  const calls = { posts: [], wheres: [], starts: [], spills: [], flies: [], focus: [], logs: [], order: [], glows: [], outlines: [] };
  const harness = { count(name, n = 1) { counters[name] = (counters[name] ?? 0) + n; } };
  const batch = { rides: new Map(), marks: new Map(), ride(k, fn) { this.rides.set(k, fn); }, mark(k, fn) { this.marks.set(k, fn); },
    flush() { const fns = [...this.marks.values(), ...this.rides.values()]; this.rides.clear(); this.marks.clear(); fns.forEach(fn => fn()); } };
  const frames = Array.from({ length: 40 }, (_, i) => ({ seq: i, iter: i * 100 + 1 }));
  const ctx = { run: { id: 'r1', sample: 's1', state: live ? 'running' : 'paused', ...run }, frames, seq, live, selected: 2, cells: CELLS,
    qc: { status: 'live', enabled: true }, startSource: 'xenium', calm: false, view3d, stage: { width: 1200, height: 800 } };
  const engine = where ?? (params => whereReply(params)), atStart = start ?? (params => startReply(params));
  const api = {
    qcDeeper: async (id, body) => { calls.posts.push(body); return answer(body); },
    qcDeeperStop: async () => null, qcDeeperHistory: async () => ({ model: 'm1', start: 2, points: [] }),
    qcDeeperWhere: async (id, params) => {
      const first = params.seq === 'start';
      (first ? calls.starts : calls.wheres).push(params);
      const a = (first ? atStart : engine)(params); if (a instanceof Error) throw a; return a;
    },
  };
  const deeper = new QcDeeper(host, { api, batch, harness, clock: c, storage: null, context: () => ctx, reducedMotion: () => reduced,
    name: label => CELLS[label - 1] ?? `cell ${label - 1}`, group: () => null, swatch: () => '#123456', startCode: () => 2,
    onFocus: f => { calls.focus.push(f); calls.order.push(f ? 'focus' : 'no focus'); },
    onSpill: s => { calls.spills.push(s); calls.order.push(s ? 'dots' : 'no dots'); }, onFly: (rect, animate, view = null) => calls.flies.push([rect, animate, view]),
    onGlow: g => calls.glows.push(g), onStartOutline: o => calls.outlines.push(o),
    log: t => calls.logs.push(t) });
  return { deeper, doc, clock: c, calls, counters, batch, ctx, root: deeper.root };
}
const toggle = w => w.root.one('button[data-action=where]');
const close = (a, b, eps = 1e-4) => Math.abs(a - b) <= eps;
const fail = status => Object.assign(new Error(`HTTP ${status}`), { status });

test('the toggle: under the rows of a ready result with source rows, never greyed out; none for little or too few', async () => {
  const w = make();
  w.deeper.open(2); await settle();
  const b = toggle(w);
  assert.ok(b, 'button[data-action=where]'); assert.equal(b.hidden, false); assert.notEqual(b.disabled, true);
  assert.equal(b.textContent, DEEP_TEXT.deepWhereShow); assert.equal(b.title, DEEP_TEXT.deepWhereShowTooltip);
  assert.equal(b.getAttribute('aria-pressed'), 'false'); assert.ok(w.root.one('section.qc-deeper-from').contains(b), 'in From neighbours');
  assert.equal(w.root.one('section.qc-deeper-where').hidden, true, 'Where they sit only while it is on');
  assert.deepEqual(w.calls.wheres, [], 'nothing is asked before it is pressed');
  for (const COUNTER of ['qcDeeperWhereOn', 'qcDeeperWhereAsked', 'qcDeeperWhereShown']) assert.ok(COUNTERS.includes(COUNTER) && COUNTER in w.counters);
  const l = make({ answer: body => ready(little({ seq: body.seq })) });
  l.deeper.open(2); await settle();
  assert.equal(toggle(l).hidden, true, 'under 5 % from neighbours: no rows, nothing to show');
  const few = make({ answer: () => ({ state: 'ready', seq: 12, cell: 1, model: 'm1', result: { v: 1, state: 'too_few', seq: 12, iter: 1201, cell: 1, transcripts: 61, genes: [] } }) });
  few.deeper.open(2); await settle();
  assert.equal(toggle(few).hidden, true);
});

test('pressed: one request for the result\'s frame; the suspects glow, the dots, the fading and one camera move; the chips wear their colours', async () => {
  const w = make(), measured = [], measure = performance.measure.bind(performance);
  w.deeper.open(2); await settle();
  performance.measure = (name, o) => { measured.push(name); return measure(name, o); };
  try { toggle(w).click(); await settle(); } finally { performance.measure = measure; }
  assert.deepEqual(measured.filter(n => n.startsWith('plv:qc-deeper-where')), ['plv:qc-deeper-where-fetch', 'plv:qc-deeper-where'],
    'the request and the answer\'s apply, timed for the harness');
  assert.deepEqual(w.calls.wheres, [{ seq: 12, cell: 1, sources: [2], genes: ['GXPN', 'GXC', 'GYDNNN', 'GYJ', 'GYCNNN', 'GXT'], own: ['GYHNN', 'GXEN', 'GZCNN'] }]);
  assert.equal(toggle(w).getAttribute('aria-pressed'), 'true');
  const spill = w.calls.spills.at(-1);
  assert.equal(spill.cell, 2); assert.deepEqual(spill.lit, [2, 3], 'the cell and its source stay lit');
  assert.ok(spill.points instanceof Float32Array && spill.points.length === 4 * 9);
  assert.deepEqual(spill.colors.slice(0, 6), WHERE_COLORS.from); assert.deepEqual(spill.colors.slice(6), WHERE_COLORS.own);
  assert.equal(w.calls.flies.length, 1); assert.equal(w.calls.flies[0][1], true, 'a finite move (the stage\'s tween)');
  const rect = w.calls.flies[0][0];
  assert.ok(rect.x0 < 100 && rect.y0 < 100 && rect.x0 + rect.w > 180 && rect.y0 + rect.h > 160, 'the cell and its sources in view');
  // The chips (at most four): each wears its dot's colour; the section names each gene with its word.
  const chips = w.root.all('span.qc-deeper-chip');
  assert.equal(CHIPS, 4);
  assert.deepEqual(chips.map(c => [c.dataset.gene, c.dataset.dot]), ['GXPN', 'GXC', 'GYDNNN', 'GYJ'].map((g, k) => [g, WHERE_COLORS.from[k]]));
  assert.equal(chips[0].style.borderLeftColor, WHERE_COLORS.from[0]);
  const section = w.root.one('section.qc-deeper-where');
  assert.equal(section.hidden, false); assert.equal(section.dataset.state, 'shown');
  const src = section.one('div.qc-deeper-where-source[data-cell=ffbbbbbb-1]');
  assert.ok(src, 'a block per source row');
  const gxpn = src.one('p.qc-deeper-where-gene[data-gene=GXPN]');
  assert.equal(gxpn.dataset.word, 'edge'); assert.equal(gxpn.one('i.qc-deeper-where-dot').style.background, WHERE_COLORS.from[0]);
  assert.match(gxpn.visibleText(), /GXPN · at the shared edge/); assert.match(gxpn.title, /31 of the 39 proseg counts for this cell lie within 2 µm of it/);
  const into = src.one('p.qc-deeper-where-into[data-genes=GYHNN]');
  assert.match(into.visibleText(), /Inside the neighbour's outline, proseg counts for this cell: GYHNN 4/); assert.match(into.title, /GYHNN: 4, 3 of them within 2 µm of this cell/);
  assert.match(section.visibleText(), /Where they sit/); assert.equal(section.one('h3.qc-deeper-label').title, DEEP_TEXT.deepWhereIntro);
  // The glow: the suspects at P >= 0.2, dimmest first, as [x, y, P]; the giver's side to ripple in from.
  // GXPN (s = 31.1 / 39, 31 of its 39 near the neighbour against 21 of 80 of the rest): 0.93 near, 0.28 further in; GXC
  // (s = 0.84, 3 of 10 near: hardly any evidence) 0.83; the transcripts outside the cell and its own genes, none.
  const glow = w.calls.glows.at(-1);
  assert.match(glow.key, /^r1:/); assert.equal(glow.points.length, 9);
  // Each glowing point's depth beside it, in the glow's order (the reply's z: the glow stands there in 3-D).
  assert.ok(glow.depth instanceof Float32Array); assert.deepEqual([...glow.depth], [0.5, 0.5, 0.5]);
  const triples = [0, 1, 2].map(j => [...glow.points.slice(3 * j, 3 * j + 3)]);
  assert.deepEqual(triples.map(t => [t[0], t[1]]), [[111, 121], [113, 129], [110, 120]]);
  assert.ok(close(triples[0][2], 0.277444) && close(triples[1][2], 0.832752) && close(triples[2][2], 0.931628), JSON.stringify(triples));
  assert.deepEqual(glow.giver, [145, 127.5], 'the neighbour\'s transcripts in the reply: where it lies');
  // Where they sat at the start: the start line asked once, for the same cell, sources and genes (sorted, no own
  // genes), the starting outline handed to the page; of the two likely ones, GXPN's at the shared edge sat inside its
  // own starting outline then, GXC's inside the starting outline of the neighbour it seems to come from: one line says
  // both, its numbers on it for the harness, the shared edge in its tooltip; the muted "Most were inside…" is gone.
  assert.deepEqual(w.calls.starts, [{ seq: 'start', cell: 1, sources: [2], genes: ['GXC', 'GXPN', 'GXT', 'GYCNNN', 'GYDNNN', 'GYJ'], own: [] }]);
  assert.deepEqual(w.calls.outlines, [{ cell: 2, name: 'ffaaaaaa-1' }]);
  const legend = section.one('p.qc-deeper-glow-legend');
  assert.equal(legend.hidden, false);
  assert.equal(legend.visibleText().trim(), "2 likely from neighbours · 1 inside its own starting outline · 1 inside a neighbour's starting outline");
  assert.deepEqual({ ...legend.dataset }, { likely: '2', own: '1', none: '0', giver: '1', other: '0', unknown: '0' });
  assert.ok(legend.matches('p.qc-deeper-glow-legend[data-likely][data-own][data-none][data-giver][data-other]'), 'the harness\'s hook (I15)');
  assert.equal(legend.title, `${glowTip(true)} At the shared edge, within 2 µm of their neighbour: 1 of the likely ones.`);
  assert.match(legend.title, /dashed white line is this cell's starting outline/, 'drawn in 2-D: named');
  assert.equal(section.one('p.qc-deeper-glow-start'), null, 'no muted line: the line says where they sat');
  assert.doesNotMatch(w.root.visibleText(), /Most were inside|proseg kept them/);
  const unlit = section.one('p.qc-deeper-glow-unlit');
  assert.equal(unlit.textContent, 'No edge measured for GYDNNN'); assert.equal(unlit.title, DEEP_TEXT.deepGlowUnplacedTooltip);
  // The hook: what was asked and shown, the words, and the invariant's two sides.
  const s = w.deeper.snapshot().where;
  assert.equal(s.on, true); assert.equal(s.state, 'shown'); assert.equal(s.seq, 12); assert.deepEqual(s.sources, ['ffbbbbbb-1']); assert.deepEqual(s.lit, [2, 3]);
  assert.deepEqual(s.asked, w.calls.wheres[0]); assert.equal(s.points, 9);
  assert.deepEqual(s.counts, { GXPN: 4, GXC: 2, GYDNNN: 0, GYJ: 0, GYCNNN: 0, GXT: 0, GYHNN: 3, GXEN: 0, GZCNN: 0 });
  assert.deepEqual(s.words, { GXPN: 'edge', GXC: 'layer', GYDNNN: 'few', GYHNN: 'spread' });
  assert.equal(s.total, 685); assert.equal(s.expected, 682 + 3, 'the result\'s transcripts + outside for that frame');
  assert.equal(s.held, 'exact', 'proseg\'s shares from its export of the box');
  assert.deepEqual(s.tx, { GXPN: 39, GXC: 10, GYDNNN: 4, GYJ: 4, GYCNNN: 3, GXT: 3, GYHNN: 9, GXEN: 4, GZCNN: 4 }, 'each gene\'s row: its in_cell there');
  assert.deepEqual([w.counters.qcDeeperWhereOn, w.counters.qcDeeperWhereAsked, w.counters.qcDeeperWhereShown, w.counters.qcDeeperWhereStartAsked], [1, 1, 1, 1]);
  assert.deepEqual(s.glow, { count: 3, likely: 2, near: 1, start: 1, places: { own: 1, none: 0, giver: 1, other: 0, unknown: 0 }, levels: null, row: null, notNear: null },
    'start: places.own; no start layers here: no levels');
  assert.deepEqual(s.startAsked, w.calls.starts[0]); assert.equal(s.startOutline, true);
  for (const text of w.deeper.texts()) assert.deepEqual(bannedWords(text), [], text);
  w.clock.advance(WORKING_MS); await settle();   // the window's own "Working out…" timer from opening
  assert.equal(w.clock.timers.length + w.clock.rafs.length, 0, 'nothing scheduled once it has landed');
  assert.equal(w.calls.wheres.length, 1, 'and nothing asked again'); assert.equal(w.calls.starts.length, 1);
  // Pressed again: off, and the stage as without it (no glow, no dots, no starting outline).
  const glows = w.calls.glows.length;
  toggle(w).click(); await settle();
  assert.equal(w.calls.glows.length, glows + 1); assert.equal(w.calls.glows.at(-1), null);
  assert.equal(w.calls.spills.at(-1), null); assert.equal(toggle(w).getAttribute('aria-pressed'), 'false');
  assert.deepEqual(w.calls.outlines, [{ cell: 2, name: 'ffaaaaaa-1' }, null]);
  assert.equal(w.deeper.snapshot().where.glow, null); assert.equal(w.deeper.snapshot().where.startOutline, false);
  assert.equal(section.hidden, true); assert.equal(section.dataset.state, 'off');
  assert.equal(legend.hidden, true); assert.deepEqual({ ...legend.dataset }, {}, 'no numbers left on a line not shown');
  assert.ok(w.root.all('span.qc-deeper-chip').every(c => c.dataset.dot === undefined && c.style.borderLeftColor === ''), 'the chips\' own rule back');
  assert.equal(w.deeper.snapshot().where.on, false);
});

test('live: it follows the result shown, asked at most once a second; the camera moves once; colours stay with their genes', async () => {
  const w = make();
  w.deeper.open(2); await settle();
  toggle(w).click(); await settle();
  // A crop's pushes every 200 ms for 6 s; the window applies one a second, and the dots follow those.
  const push = (seq, genes) => w.deeper.event({ ev: 'deeper', run: 'r1', seq, iter: seq * 100 + 1, cell: 1, model: 'm1', result: mixed({ seq, iter: seq * 100 + 1, ...(genes ? { genes } : {}) }) });
  for (let k = 1; k <= 30; k++) { w.clock.advance(200); w.ctx.seq = 12 + k; push(12 + k); w.deeper.changed(); w.batch.flush(); await settle(2); }
  w.clock.advance(1500); w.batch.flush(); await settle();
  const asked = w.calls.wheres.length - 1;
  assert.ok(asked >= 4 && asked <= 7, `${asked} requests in 6 s while live`);
  const seqs = w.calls.wheres.map(p => p.seq);
  assert.deepEqual(seqs, [...seqs].sort((a, b) => a - b), 'never an older frame after a newer one');
  assert.equal(w.deeper.snapshot().where.seq, seqs.at(-1));
  assert.equal(w.calls.flies.length, 1, 'the camera moved once, with the first answer');
  // GXC overtakes GXPN (its rank changes): both keep their colours.
  const genes = mixed().genes.map(g => g[0] === 'GXC' ? ['GXC', 60, 2, 58, 1, 0] : g);
  w.clock.advance(1100); w.ctx.seq = 50; push(50, genes); w.deeper.changed(); w.batch.flush(); await settle();
  w.clock.advance(1100); await settle();
  assert.equal(w.calls.wheres.at(-1).genes[0], 'GXC', 'the chips\' new order is asked');
  const s = w.deeper.snapshot().where;
  assert.equal(s.colors[s.genes.indexOf('GXPN')], WHERE_COLORS.from[0]); assert.equal(s.colors[s.genes.indexOf('GXC')], WHERE_COLORS.from[1]);
  // A new result applied soon after the last request (a push right after a request's own answer): asked a second after it.
  const t = make();
  t.deeper.open(2); await settle();
  t.clock.advance(1100); toggle(t).click(); await settle();
  t.clock.advance(100); t.ctx.seq = 13;
  t.deeper.event({ ev: 'deeper', run: 'r1', seq: 13, iter: 1301, cell: 1, model: 'm1', result: mixed({ seq: 13, iter: 1301 }) }); t.batch.flush(); await settle();
  assert.equal(t.deeper.snapshot().seq, 13, 'the result is shown'); assert.deepEqual(t.calls.wheres.map(p => p.seq), [12], 'its dots wait');
  t.clock.advance(WHERE_REFRESH_MS - 100 - 1); await settle(); assert.equal(t.calls.wheres.length, 1);
  t.clock.advance(1); await settle(); assert.deepEqual(t.calls.wheres.map(p => p.seq), [12, 13], 'a second after the last request');
});

test('paused: each frame asked once (a frame held is not asked again); stepped, the new frame\'s dots replace the old', async () => {
  const w = make({ live: false, seq: 12 });
  w.deeper.open(2); w.clock.advance(200); await settle();
  toggle(w).click(); await settle();
  assert.equal(w.calls.wheres.length, 1);
  w.deeper.rerender(); w.deeper.changed(); await settle();
  assert.equal(w.calls.wheres.length, 1, 'the same frame: held');
  w.ctx.seq = 13; w.deeper.changed(); w.clock.advance(200); await settle();
  assert.deepEqual(w.calls.wheres.map(p => p.seq), [12, 13], 'the stepped frame, at once (no throttle while paused)');
  assert.equal(w.calls.spills.at(-1).key.endsWith(w.deeper.where.key), true);
  assert.equal(w.deeper.snapshot().where.seq, 13); assert.equal(w.calls.flies.length, 1);
  w.ctx.seq = 12; w.deeper.changed(); w.clock.advance(200); await settle();
  assert.deepEqual(w.calls.wheres.map(p => p.seq), [12, 13, 12], 'scrubbed back: that frame again (the engine keeps its answer)');
});

test('off when the window closes, moves to another cell (or a neighbour\'s ›), or the run changes; an answer after that is dropped', async () => {
  for (const how of ['close', 'select', 'neighbour', 'reset']) {
    const w = make();
    w.deeper.open(2); await settle();
    toggle(w).click(); await settle();
    assert.ok(w.calls.spills.at(-1));
    if (how === 'close') w.deeper.close();
    else if (how === 'select') { w.deeper.select(5); await settle(); }
    else if (how === 'neighbour') { w.root.one('li.qc-deeper-source').one('button[data-action=deeper]').click(); await settle(); }
    else w.deeper.reset({ id: 'r2', sample: 's1' });
    assert.equal(w.calls.spills.at(-1), null, how); assert.equal(w.deeper.snapshot().where.on, false, how);
    assert.equal(w.calls.glows.at(-1), null, `${how}: the glow goes with the dots`);
    assert.equal(toggle(w).getAttribute('aria-pressed'), 'false', how);
    // The focus goes first, then the dots: main.js brings the arrows back only for a focus still held (none flash).
    const tail = w.calls.order.slice(w.calls.order.lastIndexOf('dots') + 1);
    assert.deepEqual(tail.slice(0, 2), ['no focus', 'no dots'], `${how}: ${tail}`);
  }
  let release;
  const slow = make({ where: params => new Promise(resolve => { release = () => resolve(whereReply(params)); }) });
  slow.deeper.open(2); await settle();
  toggle(slow).click(); await settle();
  toggle(slow).click(); await settle();   // off while the answer is on its way
  const spills = slow.calls.spills.length;
  release(); await settle();
  assert.equal(slow.calls.spills.length, spills, 'dropped'); assert.equal(slow.calls.spills.at(-1), null);
});

test('a frame not saved yet is asked again a second later; a refusal is said once; a lost connection retries by itself', async () => {
  let n = 0;
  const w = make({ live: false, where: params => (++n <= 2 ? fail(409) : whereReply(params)) });
  w.deeper.open(2); w.clock.advance(200); await settle();
  toggle(w).click(); await settle();
  const section = w.root.one('section.qc-deeper-where');
  assert.equal(section.dataset.state, 'waiting'); assert.match(section.visibleText(), /Waiting for this iteration to be saved…/);
  assert.equal(w.calls.spills.length, 0, 'nothing on the stage yet');
  w.clock.advance(WHERE_WAIT_MS); await settle(); w.clock.advance(WHERE_WAIT_MS); await settle();
  assert.equal(w.calls.wheres.length, 3); assert.equal(section.dataset.state, 'shown');
  const r = make({ live: false, where: () => fail(422) });
  r.deeper.open(2); r.clock.advance(200); await settle();
  toggle(r).click(); await settle();
  assert.equal(r.root.one('section.qc-deeper-where').dataset.state, 'failed');
  assert.match(r.root.one('section.qc-deeper-where').visibleText(), /Couldn't show them on the image for this iteration\./);
  r.clock.advance(60000); r.deeper.rerender(); await settle();
  assert.equal(r.calls.wheres.length, 1, 'not asked again for that frame');
  const net = make({ live: false, where: () => new TypeError('Failed to fetch') });
  net.deeper.open(2); net.clock.advance(200); await settle();
  toggle(net).click(); await settle();
  net.clock.advance(RETRY_MS[0]); await settle();
  assert.equal(net.calls.wheres.length, 2, 'asked again after 2 s');
});

test('no dots there, the 3-D view, reduced motion; "Finding them…" only after 250 ms', async () => {
  const none = make({ where: params => ({ ...whereReply(params, { points: [] }), box: null }) });
  none.deeper.open(2); await settle();
  toggle(none).click(); await settle();
  assert.equal(none.root.one('section.qc-deeper-where').dataset.state, 'none');
  assert.match(none.root.one('section.qc-deeper-where').visibleText(), /None of these genes' transcripts lie here\./);
  assert.equal(none.calls.flies.length, 0, 'no box: no move'); assert.equal(none.calls.spills.at(-1).points.length, 0, 'the fading still shows');
  const flat = make({ view3d: true });
  flat.deeper.open(2); await settle();
  toggle(flat).click(); await settle();
  assert.match(flat.root.one('section.qc-deeper-where').visibleText(), /The genes' coloured dots show in the 2-D view\./, 'the pink dots glow in 3-D too');
  // The cell's dashed starting outline is drawn in 2-D only: the glow's tooltip names it there, never in 3-D.
  const legend3 = () => flat.root.one('p.qc-deeper-glow-legend').title;
  assert.ok(legend3().startsWith(DEEP_TEXT.deepGlowTooltip)); assert.doesNotMatch(legend3(), /dashed/, '3-D: no dashed line drawn, none named');
  // 3-D: one move all the same, the angle staying the user's: the stage fits the box itself into the free part (onFly's
  // view), the 2-D rect beside it for a page that cannot.
  assert.equal(flat.calls.flies.length, 1);
  const [rect3, animate3, view3] = flat.calls.flies[0], box3 = flat.deeper.where.reply.box;
  assert.equal(animate3, true); assert.deepEqual(view3, { box: box3, free: whereFree(box3, flat.ctx.stage, { left: flat.deeper.left, top: flat.deeper.top, w: flat.deeper.size.w, h: flat.deeper.size.h }) });
  assert.deepEqual(rect3, flat.deeper.whereRect(box3));
  flat.ctx.view3d = false; flat.deeper.layout();
  assert.doesNotMatch(flat.root.one('section.qc-deeper-where').visibleText(), /2-D view/, 'back in 2-D: said no more');
  assert.ok(legend3().startsWith(glowTip(true)), 'back in 2-D: the dashed line named again');
  // A run from nuclei only has no starting outline to draw: none named in 2-D either.
  const nuclei = make(); nuclei.ctx.startSource = 'nuclei';
  nuclei.deeper.open(2); await settle();
  toggle(nuclei).click(); await settle();
  assert.deepEqual(nuclei.calls.outlines, []); assert.doesNotMatch(nuclei.root.one('p.qc-deeper-glow-legend').title, /dashed/);
  assert.equal(make().calls.flies.length, 0);
  const calm = make({ reduced: true });
  calm.deeper.open(2); await settle();
  toggle(calm).click(); await settle();
  assert.equal(calm.calls.flies[0][1], false, 'reduced motion: the camera lands at once');
  let release;
  const slow = make({ where: params => new Promise(resolve => { release = () => resolve(whereReply(params)); }) });
  slow.deeper.open(2); await settle();
  toggle(slow).click(); await settle();
  assert.doesNotMatch(slow.root.one('section.qc-deeper-where').visibleText(), /Finding them/);
  slow.clock.advance(WORKING_MS); await settle();
  assert.match(slow.root.one('section.qc-deeper-where').visibleText(), /Finding them on the image…/);
  release(); await settle();
  assert.doesNotMatch(slow.root.one('section.qc-deeper-where').visibleText(), /Finding them/);
});

// ---- the suspect transcripts and their glow -----------------------------------------------------------------------------
/** A reply (as normaliseWhere returns it) of one "from" gene G with giver 0: `near` and `far` of its transcripts in the cell
 *  (near: within 2 µm of the giver), `own_near` the rest of the cell's RNA [near, total]; plus G in the source and around,
 *  and an own gene in the cell (never suspects); `w` proseg's shares (by default all of those in the cell, none
 *  elsewhere). */
function oneGene({ near = 4, far = 6, own_near = [20, 100], giver = 0, stats = null, w = null } = {}) {
  const pts = [...Array.from({ length: near }, (_, i) => [10 + i, 10, 0, 1, 1]), ...Array.from({ length: far }, (_, i) => [20 + i, 20, 0, 1, 0]),
    [40, 40, 0, 2, 0], [60, 60, 0, 0, 0], [61, 61, 0, 255, 0], [15, 15, 1, 1, 0]];
  return normaliseWhere({ v: 3, seq: 12, cell: 1, box: [0, 0, 70, 70], genes: ['G', 'OWN'], roles: ['from', 'own'],
    points: { x: pts.map(q => q[0]), y: pts.map(q => q[1]), gene: pts.map(q => q[2]), owner: pts.map(q => q[3]), near: pts.map(q => q[4]),
      w: w ?? pts.map(q => (q[3] === 1 ? 1 : 0)) },
    gene_stats: stats ?? [{ gene: 'G', role: 'from', giver, in_cell: near + far, near_giver: giver == null ? null : near, own_near: giver == null ? null : own_near,
      density: { cell: 1, giver: 2, around: 0 }, word: 'edge' }], into: [] });
}
const resultOf = (tx, nb) => normalise(mixed({ genes: [['G', tx, tx - nb, nb, 1, 0], ['OWN', 5, 5, 0, 0]] }));
const sum = a => a.reduce((x, y) => x + y, 0);

test('suspectScores: P = w x pi, pi the model\'s per-gene split placed by position, w proseg\'s share, on any owner', () => {
  // pi as before; P scaled by proseg's share of each point for the cell: a point inside the cell's outline it does not
  // count is no suspect; one it moved in from the source's outline, or from none, is.
  const shares = [0.5, 1, 0.25, 0, 1, 1, 1, 1, 1, 1, 0.75, 0.5, 0, 1];
  const reply = oneGene({ w: shares }), { p, pi } = suspectScores(reply, resultOf(10, 6)), plain = suspectScores(oneGene(), resultOf(10, 6));
  assert.ok(p instanceof Float32Array && pi instanceof Float32Array);
  for (let i = 0; i < 13; i++) {
    const v = reply.near[i] ? plain.genes[0].near : plain.genes[0].far;
    if (shares[i] > 0) { assert.equal(pi[i], Math.fround(v), `pi ${i}`); assert.equal(p[i], Math.fround(shares[i] * v), `P ${i}`); }
    else { assert.equal(pi[i], 0); assert.equal(p[i], 0); }
  }
  assert.ok(p[10] > 0 && p[11] > 0 && reply.owner[10] === 2 && reply.owner[11] === 0, 'moved in: suspects outside the outline');
  assert.equal(p[13], 0, 'its own gene: none');
});

test('suspectScores: the model\'s per-gene split placed by position; unclamped, a gene\'s P add up to its transcripts from neighbours', () => {
  // s = 6 / 10; the rest of the cell's RNA: 20 of 100 near the neighbour (f_r = 21 / 102); 4 of G's 10 near (f_g = 0.4).
  const reply = oneGene(), { p, genes } = suspectScores(reply, resultOf(10, 6));
  const g = genes[0], fr = 21 / 102, fs = (0.4 - 0.4 * fr) / 0.6;
  assert.ok(close(g.s, 0.6) && close(g.fr, fr) && close(g.fg, 0.4) && close(g.fs, fs), JSON.stringify(g));
  assert.ok(fs > fr && fs < 1, 'not clamped');
  assert.ok(close(g.near, 0.6 * fs / (0.6 * fs + 0.4 * fr)) && close(g.far, 0.6 * (1 - fs) / (0.6 * (1 - fs) + 0.4 * (1 - fr))));
  const cell = [...p].slice(0, 10);
  assert.ok(close(sum(cell), 6, 1e-4), `${sum(cell)}: tx x s, the model's own number`);
  assert.ok(cell.slice(0, 4).every(v => close(v, g.near)) && cell.slice(4).every(v => close(v, g.far)), 'near first, then further in');
  assert.ok(g.near > 0.6 && g.far < 0.6, 'the near ones get more than the model\'s share, the far ones less');
  assert.deepEqual([...p].slice(10), [0, 0, 0, 0], 'in the source, around, in another cell, or its own gene: no suspect');
  // GXPN of cell 1 (31 of its 39 near against 21 of 80): 0.93 near, 0.28 further in, adding up to its 31.1.
  const gxpn = suspectScores(oneGene({ near: 31, far: 8, own_near: [21, 80] }), resultOf(39, 31.1));
  assert.ok(close(gxpn.genes[0].near, 0.931628) && close(gxpn.genes[0].far, 0.277444));
  assert.ok(close(sum([...gxpn.p].slice(0, 39)), 31.1, 1e-3));
});

test('suspectScores: the two limits (no spatial evidence: all at s; hugging the edge: the near ones get most), s = 0, s = 1, no giver, not in the result', () => {
  // f_g = f_r (3 of 10 near, as the rest of the cell's RNA, 29 + 1 of 98 + 2): position adds nothing, every one gets s.
  const flat = suspectScores(oneGene({ near: 3, far: 7, own_near: [29, 98] }), resultOf(10, 5));
  assert.ok([...flat.p].slice(0, 10).every(v => close(v, 0.5)), [...flat.p].slice(0, 10).join());
  // Every one near (f_g = 1 > what s could explain): f_s clamped at 1, the near ones get s / (s + (1 - s) f_r).
  const edge = suspectScores(oneGene({ near: 10, far: 0, own_near: [10, 100] }), resultOf(10, 5)), fr = 11 / 102;
  assert.equal(edge.genes[0].fs, 1); assert.ok(close(edge.p[0], 0.5 / (0.5 + 0.5 * fr)) && edge.p[0] > 0.9);
  // Most near: those near get far more than those further in.
  const most = suspectScores(oneGene({ near: 9, far: 1, own_near: [10, 100] }), resultOf(10, 5));
  assert.ok(most.p[0] > 0.8 && most.p[9] < 0.2, `${most.p[0]} / ${most.p[9]}`);
  // s = 0: none from neighbours, no suspect; s = 1: every one, near or not.
  assert.ok([...suspectScores(oneGene(), resultOf(10, 0)).p].every(v => v === 0));
  const all = suspectScores(oneGene(), resultOf(10, 10));
  assert.ok([...all.p].slice(0, 10).every(v => close(v, 1)), [...all.p].slice(0, 10).join());
  // A gene whose giver is not among the rows: s for each of its transcripts in the cell (nothing to place them by).
  const none = suspectScores(oneGene({ giver: null }), resultOf(10, 6));
  assert.ok([...none.p].slice(0, 10).every(v => close(v, 0.6))); assert.equal(none.genes[0].giver, null);
  // A gene the result does not hold (or no result, or one not ready): none.
  assert.ok([...suspectScores(oneGene(), normalise(mixed())).p].every(v => v === 0), 'G is not in that result');
  assert.equal(suspectScores(oneGene(), null).p.length, oneGene().count); assert.ok([...suspectScores(oneGene(), null).p].every(v => v === 0));
  assert.deepEqual(suspectScores(null, resultOf(10, 6)).genes, []);
});

test('the glow: the suspects at P >= 0.2 on any owner, dimmest first, brighter the likelier; "likely" at pi >= 0.5 where proseg counts half or more for the cell; one row\'s, or every row\'s with the weightiest giver', () => {
  assert.deepEqual([GLOW_MIN, LIKELY_MIN, LIKELY_W, SUSPECTS_SENT, PAIR_MARGIN_UM], [0.2, 0.5, 0.5, 400, 6]);
  const reply = normaliseWhere({ v: 3, seq: 12, cell: 1, box: [0, 0, 100, 100], genes: ['A', 'B', 'C'], roles: ['from', 'from', 'own'],
    points: { x: [1, 2, 3, 4, 5, 6, 7], y: [1, 2, 3, 4, 5, 6, 7], gene: [0, 0, 1, 1, 0, 2, 1], owner: [1, 1, 1, 1, 2, 1, 3], near: [1, 0, 1, 0, 0, 0, 0],
      w: [1, 1, 1, 1, 1, 1, 0] },
    gene_stats: [{ gene: 'A', role: 'from', giver: 0, in_cell: 2 }, { gene: 'B', role: 'from', giver: 1, in_cell: 2 }] });
  const scores = { p: Float32Array.from([0.9, 0.3, 0.6, 0.1, 0.95, 0, 0]), pi: Float32Array.from([0.9, 0.3, 0.6, 0.1, 0.95, 0, 0]) };
  const all = glowPoints(reply, scores);
  assert.deepEqual([all.count, all.likely, all.near], [4, 3, 2], 'the source\'s point (proseg moved it in) glows too');
  assert.deepEqual([...all.points].filter((_, i) => i % 3 === 0), [2, 3, 1, 5], 'dimmest first: 0.3, 0.6, 0.9, 0.95');
  assert.equal(all.giver, 0, 'giver 0\'s suspects weigh most (0.9 + 0.3 + 0.95)');
  const b = glowPoints(reply, scores, 1);
  assert.deepEqual([b.count, b.likely, b.near, b.giver], [1, 1, 1, 1]); assert.deepEqual([...b.points], [3, 3, Math.fround(0.6)]);
  assert.deepEqual(glowPoints(reply, { p: new Float32Array(7), pi: new Float32Array(7) }),
    { points: new Float32Array(0), depth: new Float32Array(0), count: 0, likely: 0, near: 0, start: null, places: null, levels: null, giver: null, notNear: null });
  // Each glowing point's depth, in the glow's order (none in the reply: NaN, drawn in 2-D only).
  assert.deepEqual([...all.depth], [NaN, NaN, NaN, NaN]);
  const deep3 = normaliseWhere({ ...JSON.parse(JSON.stringify({ v: 3, seq: 12, cell: 1, box: [0, 0, 100, 100], genes: ['A', 'B', 'C'], roles: ['from', 'from', 'own'],
    points: { x: [1, 2, 3, 4, 5, 6, 7], y: [1, 2, 3, 4, 5, 6, 7], z: [0.1, 0.2, 0.3, null, 0.5, 0.6, 0.7], gene: [0, 0, 1, 1, 0, 2, 1], owner: [1, 1, 1, 1, 2, 1, 3],
      near: [1, 0, 1, 0, 0, 0, 0], w: [1, 1, 1, 1, 1, 1, 0] },
    gene_stats: [{ gene: 'A', role: 'from', giver: 0, in_cell: 2 }, { gene: 'B', role: 'from', giver: 1, in_cell: 2 }] })) });
  assert.deepEqual([...glowPoints(deep3, scores).depth].map(z => +z.toFixed(3)), [0.2, 0.3, 0.1, 0.5], 'dimmest first, as the points');
  assert.ok(Number.isNaN(glowPoints(deep3, { p: Float32Array.from([0, 0, 0, 0.9, 0, 0, 0]), pi: new Float32Array(7) }).depth[0]), 'no depth: NaN');
  // Likely needs pi and the share: a point proseg counts under half for the cell glows, but is not likely.
  const half = normaliseWhere({ v: 3, seq: 12, cell: 1, box: [0, 0, 100, 100], genes: ['A'], roles: ['from'],
    points: { x: [1, 2], y: [1, 2], gene: [0, 0], owner: [1, 1], near: [1, 1], w: [0.5, 0.4999] },
    gene_stats: [{ gene: 'A', role: 'from', giver: 0, in_cell: 2 }] });
  const g = glowPoints(half, { p: Float32Array.from([0.45, 0.45]), pi: Float32Array.from([0.9, 0.9]) });
  assert.deepEqual([g.count, g.likely, g.near], [2, 1, 1]);
  // Where those the line counts sat at the start, once that is known (startPlaces' codes): of the likely ones, or of
  // every glowing one when none is likely; `start` is the own ones (the hook's old field, kept).
  assert.deepEqual([all.start, all.places, all.levels], [null, null, null]);
  const code = name => PLACES.indexOf(name) + 1;
  const places = Uint8Array.from([code('own'), code('own'), code('none'), code('giver'), code('other'), 0, 0]);   // 0.3 no likely one; 0.1 no glow
  const placed = glowPoints(reply, scores, null, places);
  assert.deepEqual(placed.places, { own: 1, none: 1, giver: 0, other: 1, unknown: 0 }); assert.equal(placed.start, 1);
  assert.deepEqual(glowPoints(reply, scores, 1, places).places, { own: 0, none: 1, giver: 0, other: 0, unknown: 0 }, 'a row\'s: its own likely ones');
  const faint = { p: Float32Array.from([0.4, 0.3, 0.25, 0.1, 0.45, 0, 0]) };
  const maybe = glowPoints(reply, faint, null, places);
  assert.deepEqual([maybe.likely, maybe.start, maybe.places], [0, 2, { own: 2, none: 1, giver: 0, other: 1, unknown: 0 }], 'none likely: every glowing one');
  // A 0/1 array (the cell's at the start, or not) still counts as own and nothing else.
  assert.equal(glowPoints(reply, scores, null, Uint8Array.from([1, 1, 0, 1, 0, 0, 0])).start, 1);
  // The start's hold of those inside its own outline (HOLD codes per point), by level; null without places or holds.
  const holds = Uint8Array.from([HOLD.nucleus, HOLD.cell, HOLD.cell, 0, 0, 0, 0]);
  assert.deepEqual(glowPoints(reply, scores, null, places, holds).levels, { nucleus: 1, cell: 0, unknown: 0 }, 'the likely own one: its nucleus level');
  assert.deepEqual(glowPoints(reply, faint, null, places, holds).levels, { nucleus: 1, cell: 1, unknown: 0 });
  assert.deepEqual(glowPoints(reply, faint, null, places, new Uint8Array(7)).levels, { nucleus: 0, cell: 0, unknown: 2 });
  assert.equal(glowPoints(reply, scores, null, null, holds).levels, null); assert.equal(glowPoints(reply, scores, null, places, null).levels, null);
  // Where a source lies (the glow's side) and the box around the cell and one source (a row's Show fits the camera to it).
  assert.deepEqual(sourceCentre(reply, 0), [5, 5]); assert.equal(sourceCentre(reply, 4), null);
  assert.deepEqual(pairBox(reply, 0), [1 - 6, 1 - 6, 6 + 6, 6 + 6]); assert.deepEqual(pairBox(reply, 1), [-5, -5, 13, 13]); assert.equal(pairBox(reply, 3), null);
});

/** ggdddddd-1-like (the look-alike cell's four rows: two of its own kind): GYPNNN and GZCNN seem to come from ggaaaaaa-1
 *  (row 3, source 2), GYF from ggbbbbbb-1 (row 4, source 3), its own GXPN inside ffbbbbbb-1. */
function lookalikeReply(params) {
  const pts = [[50, 50, 0, 1, 1], [52, 48, 0, 1, 0], [51, 53, 1, 1, 1], [49, 47, 2, 1, 1], [70, 50, 0, 4, 0], [30, 60, 2, 5, 0], [50, 52, 3, 1, 0], [40, 40, 3, 2, 1]];
  return { v: 3, run: 'r1', seq: params.seq, iter: 1201, cell: params.cell, sources: params.sources, box: [20, 30, 90, 70], held: 'exact', held_seq: params.seq,
    genes: [...params.genes, ...params.own], roles: [...params.genes.map(() => 'from'), ...params.own.map(() => 'own')],
    points: { x: pts.map(q => q[0]), y: pts.map(q => q[1]), z: pts.map(() => 0.5), gene: pts.map(q => q[2]), owner: pts.map(q => q[3]), near: pts.map(q => q[4]),
      w: pts.map(q => (q[3] === 1 ? 1 : 0)) },
    cell_total: 1400, truncated: false,
    gene_stats: [
      { gene: 'GYPNNN', role: 'from', giver: 2, giver_by: 'result', in_cell: 5, near_giver: 4, own_near: [10, 60], density: { cell: 3, giver: 9, around: 0 }, word: 'edge' },
      { gene: 'GZCNN', role: 'from', giver: 2, giver_by: 'result', in_cell: 3, near_giver: 3, own_near: [10, 60], density: { cell: 2, giver: 6, around: 0 }, word: 'few' },
      { gene: 'GYF', role: 'from', giver: 3, giver_by: 'result', in_cell: 3, near_giver: 2, own_near: [12, 60], density: { cell: 2, giver: 5, around: 0 }, word: 'few' },
      { gene: 'GXPN', role: 'own', giver: null, in_cell: 29, near_giver: null, own_near: null, density: { cell: 20, giver: null, around: 5 }, word: 'spread' }],
    into: [{ source: 0, gene: 'GXPN', inside: 1, near_cell: 1, source_own_near: [5, 50] }] };
}

test('a row\'s Show: only that neighbour\'s suspects glow, the camera fits the cell and that neighbour; again: every suspect; a same-kind row explains', async () => {
  // A run whose start line the engine does not answer (404): the line says nothing of the start.
  const w = make({ answer: body => ready(lookalike({ seq: body.seq })), where: params => lookalikeReply(params), start: () => fail(404) });
  w.deeper.open(2); await settle();
  const row = cell => w.root.all('li.qc-deeper-source').find(li => li.dataset.cell === cell);
  const showOf = cell => row(cell).one('button[data-action=show]');
  assert.deepEqual(w.calls.wheres, [], 'nothing asked before a press');
  // ggaaaaaa-1's Show: the view comes on for that row (one request), its three suspects glow, the camera fits the pair.
  showOf('ggaaaaaa-1').click(); await settle();
  assert.equal(w.calls.wheres.length, 1); assert.deepEqual(w.calls.wheres[0].sources, [2, 3, 4, 5]);
  assert.equal(toggle(w).getAttribute('aria-pressed'), 'true'); assert.equal(showOf('ggaaaaaa-1').getAttribute('aria-pressed'), 'true');
  let glow = w.calls.glows.at(-1);
  assert.equal(glow.points.length, 9); assert.deepEqual(glow.giver, [70, 50], 'ggaaaaaa-1\'s side');
  assert.deepEqual(w.deeper.snapshot().where.glow, { count: 3, likely: 3, near: 2, start: null, places: null, levels: null, row: 5, notNear: null });
  const section = w.root.one('section.qc-deeper-where'), legend = section.one('p.qc-deeper-glow-legend');
  assert.equal(legend.visibleText().trim(), '3 likely from ggaaaaaa-1');
  assert.match(legend.title, /At the shared edge, within 2 µm of their neighbour: 2 of the likely ones\.$/);
  assert.deepEqual({ ...legend.dataset }, { likely: '3' }, 'no places without the start line\'s reply');
  assert.equal(section.one('p.qc-deeper-glow-unlit').hidden, true, 'one row\'s: nothing said of the others');
  assert.equal(w.calls.flies.length, 1);
  const [rect] = w.calls.flies[0], box = pairBox(normaliseWhere(lookalikeReply(w.calls.wheres[0]), null), 2);
  assert.deepEqual(box, [43, 41, 76, 59]);
  assert.ok(rect.x0 <= box[0] && rect.y0 <= box[1] && rect.x0 + rect.w >= box[2] && rect.y0 + rect.h >= box[3], 'the cell and that neighbour in view');
  assert.ok(rect.w < 70 * 1.6, 'closer than the view of every row (the reply\'s box is 70 µm wide)');
  assert.equal(w.calls.focus.at(-1).hot, 5, 'its highlight pinned');
  // Another row's Show: its suspects instead, the camera moves to that pair; the first row's Show lets go.
  showOf('ggbbbbbb-1').click(); await settle();
  assert.equal(w.calls.wheres.length, 1, 'the same answer: nothing asked again');
  assert.equal(showOf('ggaaaaaa-1').getAttribute('aria-pressed'), 'false'); assert.equal(showOf('ggbbbbbb-1').getAttribute('aria-pressed'), 'true');
  glow = w.calls.glows.at(-1);
  assert.equal(glow.points.length, 3); assert.deepEqual(glow.giver, [30, 60]); assert.equal(w.calls.flies.length, 2);
  assert.deepEqual(w.deeper.snapshot().where.glow, { count: 1, likely: 1, near: 1, start: null, places: null, levels: null, row: 6, notNear: null });
  // Pressed again: every suspect glows again (the dominant giver's side), the camera stays; what cannot glow is said.
  showOf('ggbbbbbb-1').click(); await settle();
  glow = w.calls.glows.at(-1);
  assert.equal(glow.points.length, 12); assert.deepEqual(glow.giver, [70, 50]); assert.equal(w.calls.flies.length, 2);
  assert.deepEqual(w.deeper.snapshot().where.glow, { count: 4, likely: 4, near: 3, start: null, places: null, levels: null, row: null, notNear: null });
  assert.equal(section.one('p.qc-deeper-glow-legend').visibleText().trim(), '4 likely from neighbours');
  const unlit = section.one('p.qc-deeper-glow-unlit');
  assert.equal(unlit.hidden, false); assert.equal(unlit.textContent, DEEP_TEXT.deepGlowSameKind, 'two rows look like its own kind');
  // A test of a setting starts with the suspects of this cell: the top by P, at most 400; none for another.
  const kept = w.deeper.suspects(2);
  assert.equal(kept.length, 4); assert.deepEqual(kept.map(x => x.gene), ['GYPNNN', 'GYF', 'GZCNN', 'GYPNNN']);
  assert.deepEqual(Object.keys(kept[0]), ['x', 'y', 'gene', 'p']); assert.deepEqual([kept[0].x, kept[0].y], [50, 50]);
  assert.ok(kept.every((x, i) => i === 0 || x.p <= kept[i - 1].p)); assert.ok(kept.every(x => x.p >= GLOW_MIN && x.p <= 1));
  assert.equal(w.deeper.suspects(2, 2).length, 2); assert.deepEqual(w.deeper.suspects(3), []);
  // A same-kind row's Show explains (never greyed out) and leaves the glow as it is.
  const glows = w.calls.glows.length;
  showOf('ffbbbbbb-1').click(); await settle();
  assert.equal(w.root.one('p.qc-deeper-row-note').hidden, false); assert.equal(w.calls.glows.length, glows);
  // The main button: off; the glow and the dots go, no row's Show stays pressed.
  showOf('ggaaaaaa-1').click(); await settle();
  toggle(w).click(); await settle();
  assert.equal(w.calls.glows.at(-1), null); assert.equal(w.calls.spills.at(-1), null);
  assert.ok(['ggaaaaaa-1', 'ggbbbbbb-1'].every(c => showOf(c).getAttribute('aria-pressed') === 'false'));
  assert.equal(w.calls.focus.at(-1).hot, 0, 'the pin lets go');
  for (const text of w.deeper.texts()) assert.deepEqual(bannedWords(text), [], text);
  // With the view on 3-D: the glow is handed over (the stage draws it at its depth), and the camera moves to the pair
  // keeping the user's angle (the view: the pair's box and the free part).
  const flat = make({ answer: body => ready(lookalike({ seq: body.seq })), where: params => lookalikeReply(params), start: () => fail(404), view3d: true });
  flat.deeper.open(2); await settle();
  flat.root.all('li.qc-deeper-source').find(li => li.dataset.cell === 'ggaaaaaa-1').one('button[data-action=show]').click(); await settle();
  assert.equal(flat.calls.flies.length, 1); assert.equal(flat.calls.glows.at(-1).points.length, 9); assert.equal(flat.calls.glows.at(-1).depth.length, 3);
  assert.deepEqual(flat.calls.flies[0][2].box, [43, 41, 76, 59], 'the pair\'s box, as in 2-D');
});

test('live: the glow follows the result shown with the dots; the same answer hands the stage nothing again', async () => {
  const w = make();
  w.deeper.open(2); await settle();
  toggle(w).click(); await settle();
  const first = w.calls.glows.at(-1), n = w.calls.glows.length;
  w.deeper.rerender(); await settle();
  assert.equal(w.calls.glows.length, n, 'unchanged: no new glow');
  w.clock.advance(1100); w.ctx.seq = 13;
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 13, iter: 1301, cell: 1, model: 'm1', result: mixed({ seq: 13, iter: 1301 }) }); w.batch.flush(); await settle();
  w.clock.advance(1100); await settle();
  assert.deepEqual(w.calls.wheres.map(p => p.seq), [12, 13]);
  assert.notEqual(w.calls.glows.at(-1).key, first.key, 'the new frame\'s glow');
  // An older engine's reply (v 2: no shares) is not taken: said as a refusal, nothing glows.
  const old = make({ where: params => whereReply(params, { v: 2 }) });
  old.deeper.open(2); await settle();
  toggle(old).click(); await settle();
  assert.equal(old.root.one('section.qc-deeper-where').dataset.state, 'failed'); assert.equal(old.calls.glows.length, 0);
});

// ---- where they sat at the start, the start's hold, the test corner, the words after the view goes off -------------------
test('where they sat at the start: each glowing transcript\'s point of its gene within 0.01 µm in the start line\'s reply, and whose it was then', () => {
  assert.equal(MATCH_UM, 0.01);
  assert.deepEqual([...PLACES], ['own', 'none', 'giver', 'other', 'unknown']);
  const code = name => PLACES.indexOf(name) + 1;
  const req = mixedRequest(), reply = normaliseWhere(whereReply(req.params), req);
  const params = { seq: 'start', cell: 1, sources: [2], genes: [...req.params.genes].sort(), own: [] };
  const at = (points, p = params) => normaliseWhere(startReply(p, points ? { points } : {}), { params: p });
  const places = (r, start, p = params, frame = req.params) => startPlaces(r, start, frame.sources, p.sources);
  // The reply's points: GXPN at (110, 120) in the cell, the cell's at the start too; GXPN at (111, 121), in no cell then;
  // GXC at (113, 129), inside its neighbour's starting outline then, the neighbour it seems to come from (its giver,
  // source 0); its own gene's point (GYHNN at 114, 131: own genes are never asked at the start, and never glow): not
  // known; the source's and the tissue's points: not the cell's (0).
  assert.deepEqual([...places(reply, at())], [code('own'), code('none'), 0, 0, 0, code('giver'), 0, code('unknown'), 0]);
  // Every transcript proseg counts for the cell (w > 0) is placed, inside its outline or not, as any of them can glow;
  // one inside its outline that proseg does not count for it is not (0).
  const moved = normaliseWhere(whereReply(req.params, { points: [[110, 120, 0, 1, 1, 0], [140, 125, 0, 2, 0, 0.75], [112, 128, 1, 0, 0, 0.5]] }), req);
  assert.deepEqual([...places(moved, at())], [0, code('giver'), code('unknown')]);
  // Within 0.01 µm on each axis (the replies give places to 0.01 µm), not 0.02 (then: not in the start's reply); the
  // same gene; whoever held it then.
  assert.equal(places(reply, at([['GXPN', 110.009, 119.991, 1]]))[0], code('own'));
  assert.equal(places(reply, at([['GXPN', 110.02, 120, 1]]))[0], code('unknown'));
  assert.equal(places(reply, at([['GXC', 110, 120, 1]]))[0], code('unknown'));
  assert.equal(places(reply, at([['GXPN', 110, 120, 0]]))[0], code('none'));
  assert.equal(places(reply, at([['GXPN', 110, 120, 2]]))[0], code('giver'), 'its giver\'s starting outline (source 0 = cell 2)');
  assert.equal(places(reply, at([['GXPN', 110, 120, 255]]))[0], code('other'));
  // The start request lists the neighbours sorted (whereStart): its owner 2 + k names the sorted list's k.
  const two = { ...req.params, sources: [5, 2] }, rp = normaliseWhere(whereReply(two), { params: two });
  const sorted = { ...params, sources: [2, 5] };
  assert.equal(places(rp, at([['GXPN', 110, 120, 3]], sorted), sorted, two)[0], code('giver'), 'GXPN\'s giver is cell 5: the start\'s source 1');
  assert.equal(places(rp, at([['GXPN', 110, 120, 2]], sorted), sorted, two)[0], code('other'), 'cell 2 gives it nothing');
  // Two transcripts of the gene at one place in two layers: the one at the same depth.
  const layered = depths => { const raw = startReply(params, { points: [['GXPN', 110, 120, 0], ['GXPN', 110, 120, 1]] }); raw.points.z = depths; return normaliseWhere(raw, { params }); };
  assert.equal(places(reply, layered([0.9, 0.5]))[0], code('own'), 'the reply\'s GXPN lies at z 0.5');
  assert.equal(places(reply, layered([0.5, 0.9]))[0], code('none'));
  // Far from the origin the places are 32-bit (steps of about 1/1000 µm): still matched to 0.01 µm.
  const far = normaliseWhere(whereReply(req.params, { points: [[5000.37, 7000.41, 0, 1, 1]] }), req);
  assert.equal(places(far, at([['GXPN', 5000.37, 7000.41, 1]]))[0], code('own'));
  assert.equal(places(far, at([['GXPN', 5000.379, 7000.401, 1]]))[0], code('own'));
  assert.equal(places(far, at([['GXPN', 5000.385, 7000.41, 1]]))[0], code('unknown'));
  // No start reply, another cell's, one that lacks a gene that glows, or one for other neighbours: nothing known (null).
  assert.equal(places(reply, null), null); assert.equal(startPlaces(null, at()), null);
  assert.equal(places(reply, { ...at(), cell: 7 }), null);
  assert.equal(places(reply, at(null, { ...params, genes: ['GXPN'] })), null);
  assert.equal(startPlaces(reply, at(), [2], [3]), null);
});

test('where they sat at the start: the start line\'s reply is indexed once, and every frame\'s reply is placed against that one index', () => {
  const code = name => PLACES.indexOf(name) + 1;
  const req = mixedRequest(), params = { seq: 'start', cell: 1, sources: [2], genes: [...req.params.genes].sort(), own: [] };
  const start = normaliseWhere(startReply(params), { params });
  const index = startIndex(start);
  assert.ok(index instanceof Map); assert.equal(startIndex(start), index, 'built once per reply, kept with it');
  // Its points (any owner) by gene and place on the 0.01 µm grid.
  assert.deepEqual([...index.values()].flat().sort((a, b) => a - b), [...Array(start.count).keys()]);
  assert.deepEqual(index.get(`GXPN|${Math.round(110 / MATCH_UM)}|${Math.round(120 / MATCH_UM)}`), [0]);
  // A live view: each frame's reply placed against it, each by its own points (the index holds only the start's).
  const first = normaliseWhere(whereReply(req.params), req);
  const next = normaliseWhere(whereReply(req.params, { points: [[111, 121, 0, 1, 1], [110, 120, 0, 1, 0], [113, 129, 1, 1, 0], [140, 125, 0, 2, 0]] }), req);
  const placed = r => [...startPlaces(r, start, req.params.sources, params.sources)];
  assert.deepEqual(placed(first), [code('own'), code('none'), 0, 0, 0, code('giver'), 0, code('unknown'), 0]);
  assert.deepEqual(placed(next), [code('none'), code('own'), code('giver'), 0]);
  assert.deepEqual(placed(first), [code('own'), code('none'), 0, 0, 0, code('giver'), 0, code('unknown'), 0], 'the same again');
  assert.equal(startIndex(start), index, 'still the one index');
  // The next start reply (another set asked): its own index and its own places.
  const again = normaliseWhere(startReply(params, { points: [['GXPN', 110, 120, 0]] }), { params });
  assert.notEqual(startIndex(again), index);
  assert.deepEqual([...startPlaces(first, again, req.params.sources, params.sources)].slice(0, 2), [code('none'), code('unknown')]);
});

test('the line: how many likely, then where they sat at the start, once the start line\'s reply is in; none asked from nuclei only or at the start line; refusals and lost connections', async () => {
  const push = (w, seq, extra = {}) => { w.deeper.event({ ev: 'deeper', run: 'r1', seq, iter: seq * 100 + 1, cell: 1, model: 'm1', result: mixed({ seq, iter: seq * 100 + 1, ...extra }) }); w.batch.flush(); };
  const lineOf = w => w.root.one('p.qc-deeper-glow-legend'), text = w => lineOf(w).visibleText().trim();
  // Live: the frames asked again once a second, the start line once; a new rank of the chips asks nothing again.
  const w = make();
  w.deeper.open(2); await settle();
  toggle(w).click(); await settle();
  for (let k = 1; k <= 3; k++) { w.clock.advance(1100); w.ctx.seq = 12 + k; push(w, 12 + k); await settle(); w.clock.advance(1100); await settle(); }
  assert.equal(w.deeper.snapshot().where.seq, 15);
  assert.deepEqual(w.deeper.snapshot().where.glow.places, { own: 1, none: 0, giver: 1, other: 0, unknown: 0 }, 'each new frame\'s glow placed against it');
  const genes = mixed().genes.map(g => g[0] === 'GXC' ? ['GXC', 60, 2, 58, 1, 0] : g);
  w.clock.advance(1100); w.ctx.seq = 20; push(w, 20, { genes }); await settle(); w.clock.advance(1100); await settle();
  assert.equal(w.calls.wheres.at(-1).genes[0], 'GXC', 'the chips\' new order asked for the frame');
  assert.ok(w.calls.wheres.length >= 4); assert.equal(w.calls.starts.length, 1, 'the same cell, sources and genes: asked once');
  assert.deepEqual(w.calls.outlines, [{ cell: 2, name: 'ffaaaaaa-1' }], 'the starting outline handed over once');
  // Off and on again: a new view asks again.
  toggle(w).click(); toggle(w).click(); await settle();
  assert.equal(w.calls.starts.length, 2); assert.deepEqual(w.calls.outlines.slice(1), [null, { cell: 2, name: 'ffaaaaaa-1' }]);
  // The start line's answer after the frame's: the line gains its places then (and the harness its numbers).
  let release;
  const slow = make({ live: false, start: params => new Promise(resolve => { release = () => resolve(startReply(params)); }) });
  slow.deeper.open(2); slow.clock.advance(200); await settle();
  toggle(slow).click(); await settle();
  assert.equal(text(slow), '2 likely from neighbours'); assert.deepEqual({ ...lineOf(slow).dataset }, { likely: '2' });
  assert.deepEqual([slow.deeper.snapshot().where.glow.start, slow.deeper.snapshot().where.glow.places], [null, null]);
  release(); await settle();
  assert.equal(text(slow), "2 likely from neighbours · 1 inside its own starting outline · 1 inside a neighbour's starting outline");
  assert.deepEqual({ ...lineOf(slow).dataset }, { likely: '2', own: '1', none: '0', giver: '1', other: '0', unknown: '0' });
  // A start where no likely one was the cell's: GXPN's at the shared edge was its neighbour's then, GXC's (113, 129) is
  // not in that reply (outside the box it drew): said so.
  const none = make({ live: false, start: params => startReply(params, { points: [['GXPN', 110, 120, 2]] }) });
  none.deeper.open(2); none.clock.advance(200); await settle();
  toggle(none).click(); await settle();
  assert.equal(text(none), "2 likely from neighbours · 1 inside a neighbour's starting outline · 1 unknown at the start");
  assert.deepEqual(none.deeper.snapshot().where.glow.places, { own: 0, none: 0, giver: 1, other: 0, unknown: 1 });
  assert.equal(none.deeper.snapshot().where.glow.start, 0);
  // A run from nuclei only has no starting cells: nothing asked, no starting outline, the head alone.
  const nuclei = make({ live: false });
  nuclei.ctx.startSource = 'nuclei';
  nuclei.deeper.open(2); nuclei.clock.advance(200); await settle();
  toggle(nuclei).click(); await settle();
  assert.deepEqual([nuclei.calls.starts, nuclei.calls.outlines], [[], []]);
  assert.equal(text(nuclei), '2 likely from neighbours');
  // The view of the start line itself (frame 0): its own request is the start line's; nothing more is asked.
  const first = make({ live: false, seq: 0, answer: body => ready(mixed({ seq: body.seq, iter: 0 })) });
  first.deeper.open(2); first.clock.advance(200); await settle();
  toggle(first).click(); await settle();
  assert.deepEqual(first.calls.starts.map(p => p.own.length > 0), [true], 'only the view\'s own request (with its own genes)');
  assert.equal(first.deeper.snapshot().where.glow.places, null);
  // Refused (404): not asked again with the next frames; a lost connection: asked again with the next result.
  for (const [failure, again] of [[fail(404), 0], [new TypeError('Failed to fetch'), 1]]) {
    const f = make({ start: () => failure });
    f.deeper.open(2); await settle();
    toggle(f).click(); await settle();
    assert.equal(f.calls.starts.length, 1); assert.equal(f.deeper.snapshot().where.glow.places, null);
    f.clock.advance(1100); f.ctx.seq = 13; push(f, 13); await settle(); f.clock.advance(1100); await settle();
    assert.equal(f.calls.starts.length, 1 + again, String(failure));
    assert.ok(f.calls.logs.some(t => /could not tell what ffaaaaaa-1 held at the start/.test(t)));
    assert.equal(text(f), '2 likely from neighbours');
  }
  for (const v of [w, slow, none]) for (const t of v.deeper.texts()) assert.deepEqual(bannedWords(t), [], t);
});

test('the line: a row\'s Show places that neighbour\'s likely ones; pressed again, every neighbour\'s', async () => {
  // ggaaaaaa-1 (row 3, cell 4) gives GYPNNN and GZCNN, ggbbbbbb-1 (row 4, cell 5) GYF. At the start the cell held
  // GYPNNN's at (50, 50); GYPNNN's at (52, 48) lay in ggaaaaaa-1, GZCNN's at (51, 53) in no cell, GYF's at (49, 47) in
  // ggbbbbbb-1: each inside the starting outline of the neighbour it seems to come from.
  const w = make({ answer: body => ready(lookalike({ seq: body.seq })), where: params => lookalikeReply(params),
    start: params => startReply(params, { points: [['GYPNNN', 50, 50, 1], ['GYPNNN', 52, 48, 4], ['GZCNN', 51, 53, 0], ['GYF', 49, 47, 5]] }) });
  w.deeper.open(2); await settle();
  const showOf = cell => w.root.all('li.qc-deeper-source').find(li => li.dataset.cell === cell).one('button[data-action=show]');
  const text = () => w.root.one('p.qc-deeper-glow-legend').visibleText().trim();
  showOf('ggaaaaaa-1').click(); await settle();
  assert.deepEqual(w.calls.starts.map(p => p.sources), [[2, 3, 4, 5]]);
  assert.deepEqual(w.deeper.snapshot().where.glow, { count: 3, likely: 3, near: 2, start: 1, places: { own: 1, none: 1, giver: 1, other: 0, unknown: 0 },
    levels: null, row: 5, notNear: null });
  assert.equal(text(), "3 likely from ggaaaaaa-1 · 1 inside its own starting outline · 1 where no cell started · 1 inside a neighbour's starting outline");
  showOf('ggbbbbbb-1').click(); await settle();
  assert.deepEqual(w.deeper.snapshot().where.glow.places, { own: 0, none: 0, giver: 1, other: 0, unknown: 0 });
  assert.equal(text(), "1 likely from ggbbbbbb-1 · 1 inside a neighbour's starting outline");
  showOf('ggbbbbbb-1').click(); await settle();
  assert.equal(text(), "4 likely from neighbours · 1 inside its own starting outline · 1 where no cell started · 2 inside a neighbour's starting outline");
  assert.equal(w.root.one('p.qc-deeper-glow-legend').dataset.giver, '2');
});

test('the start\'s hold in the line\'s tooltip: context startHold for the reply\'s points (once per reply), split by level for those inside its own outline', async () => {
  // A detail run: its start layers held GXPN's transcript at the shared edge (110, 120: inside its own starting outline)
  // at the nucleus level.
  const w = make({ live: false }), asked = [];
  w.ctx.startHold = (points, label) => { asked.push([points.count, label, points.x[0], points.z[0]]); return Uint8Array.from([HOLD.nucleus, HOLD.cell, 0, 0, 0, HOLD.cell, 0, 0, 0]); };
  w.deeper.open(2); w.clock.advance(200); await settle();
  toggle(w).click(); await settle();
  assert.deepEqual(asked, [[9, 2, 110, Math.fround(0.5)]], 'the reply\'s own points, for the window\'s cell');
  const glow = w.deeper.snapshot().where.glow;
  assert.deepEqual(glow.levels, { nucleus: 1, cell: 0, unknown: 0 }, 'of the likely ones inside its own outline: GXPN\'s, at the nucleus level');
  const line = w.root.one('p.qc-deeper-glow-legend');
  assert.equal(line.title, startTip(glow, true));
  assert.match(line.title, /Inside its own starting outline, the start held 1 at the nucleus level and 0 at the cell level\. A start from cells only holds the whole starting cell at the nucleus level\.$/);
  assert.equal(line.visibleText().trim(), "2 likely from neighbours · 1 inside its own starting outline · 1 inside a neighbour's starting outline", 'the hold in the tooltip only');
  // The same reply painted again (a row's Show and back, a repaint): not asked again.
  const show = w.root.one('li.qc-deeper-source').one('button[data-action=show]');
  show.click(); await settle(); show.click(); await settle(); w.deeper.rerender(); await settle();
  assert.equal(asked.length, 1);
  // A whole-sample run (no start layers on the page: null), or a page without it: no levels, no split; asked again with
  // the next paint (the start layers may come in later).
  for (const hold of [() => null, undefined, () => { throw new Error('no prior yet'); }, () => new Uint8Array(2)]) {
    const v = make({ live: false }), calls = [];
    if (hold) v.ctx.startHold = (...args) => { calls.push(args); return hold(); };
    v.deeper.open(2); v.clock.advance(200); await settle();
    toggle(v).click(); await settle();
    assert.equal(v.deeper.snapshot().where.glow.levels, null, String(hold));
    assert.doesNotMatch(v.root.one('p.qc-deeper-glow-legend').title, /nucleus level/);
    if (hold) { const n = calls.length; v.deeper.paintGlow(); assert.equal(calls.length, n + 1, 'asked again'); }
  }
  // Nothing inside its own outline (the start line refused): never asked.
  const refused = make({ live: false, start: () => fail(404) }), never = [];
  refused.ctx.startHold = (...args) => { never.push(args); return new Uint8Array(9); };
  refused.deeper.open(2); refused.clock.advance(200); await settle();
  toggle(refused).click(); await settle();
  assert.deepEqual(never, []);
});

test('the camera\'s move keeps clear of the open test corner as well as the window', async () => {
  const stage = { width: 1200, height: 800 }, win = { left: 1200 - WIDTH - 12, top: 52, w: WIDTH, h: 500 };
  const corner = { left: 12, top: 800 - 12 - 380, w: 600, h: 380 }, box = [100, 100, 180, 160];
  const fit = rect => { const zoom = Math.min(stage.width / rect.w, stage.height / rect.h); return { zoom, x: rect.x0 + rect.w / 2, y: rect.y0 + rect.h / 2 }; };
  const toScreen = (cam, x, y) => [(x - cam.x) * cam.zoom + stage.width / 2, (y - cam.y) * cam.zoom + stage.height / 2];
  const cam = fit(whereView(box, stage, win, corner));
  const [ax, ay] = toScreen(cam, box[0], box[1]), [bx, by] = toScreen(cam, box[2], box[3]);
  assert.ok(ax >= 12 - 1e-6 && bx <= win.left - 12 + 1e-6, `left of the window: ${ax}..${bx}`);
  assert.ok(ay >= 12 - 1e-6 && Math.abs(by - (corner.top - 12)) < 1e-6, `above the corner, as large as fits there: ${ay}..${by}`);
  // The corner's own box ({left, bottom, w, h}, top worked out from the stage) is the same place.
  assert.deepEqual(whereView(box, stage, win, { left: 12, bottom: 12, w: 600, h: 380 }), whereView(box, stage, win, corner));
  // Closed (null), or out of the part the window leaves: as before.
  assert.deepEqual(whereView(box, stage, win, null), whereView(box, stage, win));
  assert.deepEqual(whereView(box, stage, win, { left: 900, top: 600, w: 200, h: 150 }), whereView(box, stage, win));
  // A corner that leaves too little room on every side of it (a short stage): the box shows as large as without it.
  const short = { width: 1200, height: 520 };
  assert.deepEqual(whereView(box, short, win, { left: 12, top: 520 - 12 - 380, w: 600, h: 380 }), whereView(box, short, win));
  // No window and a corner at the left: the box goes right of it when that shows it larger than above it.
  const tall = [100, 100, 130, 200], right = fit(whereView(tall, stage, null, corner));
  assert.ok(toScreen(right, tall[0], tall[1])[0] >= 612 + 12 - 1e-6, 'right of the corner');
  // The window takes the corner from the page (context cornerBox); the harness can ask it the very rect it flies to.
  const w = make({ live: false });
  w.ctx.cornerBox = () => ({ ...corner, bottom: 12 });
  w.deeper.open(2); w.clock.advance(200); await settle();
  toggle(w).click(); await settle();
  const [rect] = w.calls.flies[0], reply = w.deeper.snapshot().where.box;
  const placed = { left: w.deeper.left, top: w.deeper.top, w: w.deeper.size.w, h: w.deeper.size.h };
  assert.deepEqual(rect, whereView(reply, w.ctx.stage, placed, corner)); assert.deepEqual(w.deeper.whereRect(reply), rect);
  w.ctx.cornerBox = () => { throw new Error('no corner yet'); };
  assert.deepEqual(w.deeper.whereRect(reply), whereView(reply, w.ctx.stage, placed), 'a page without the corner: the window alone');
});

test('Show on the image, then off: the window shows exactly the words it showed before (the browser run of 2026-10-22 found one more on the page)', async () => {
  const w = make({ live: false });
  w.ctx.cellStats = label => (label === 2 ? { assigned: 452, kept: 287 } : null);
  w.deeper.open(2); w.clock.advance(200); await settle();
  const before = w.deeper.texts(), seen = w.root.visibleText();
  assert.ok(before.some(t => t.includes('less 165 it reads as background')) && before.includes(DEEP_TEXT.deepWhereShow));
  // On (the glow, where they sat at the start, Where they sit), a row's Show pressed and let go, then off.
  toggle(w).click(); await settle();
  const on = w.deeper.texts();
  for (const text of ["2 likely from neighbours · 1 inside its own starting outline · 1 inside a neighbour's starting outline", 'GXPN · at the shared edge',
    startTip(w.deeper.snapshot().where.glow, true)])
    assert.ok(on.includes(text), text);
  const show = w.root.one('li.qc-deeper-source').one('button[data-action=show]');
  show.click(); await settle(); show.click(); await settle();
  toggle(w).click(); await settle();
  assert.deepEqual(w.deeper.texts(), before); assert.equal(w.root.visibleText(), seen);
  // A row's Show turns it on and the main button off: the same again; and once more from the start.
  show.click(); await settle();
  assert.equal(w.deeper.snapshot().where.on, true);
  toggle(w).click(); await settle();
  assert.deepEqual(w.deeper.texts(), before); assert.equal(w.root.visibleText(), seen);
  toggle(w).click(); await settle(); toggle(w).click(); await settle();
  assert.deepEqual(w.deeper.texts(), before);
});

// ---- suspects that share a place, a result replaced for the same frame -----------------------------------------------
test('suspects that share a place: how many points of their gene lie there, any owner, and their rank by depth', () => {
  // The case: two EDGE transcripts at (118.5, 64.25), the neighbour's at z 0.9 and the cell's at z 0.1, with
  // another gene's there too; three EDGE points a hundredth of a µm apart in x (within 0.01 µm: a step of the reply's
  // grid; two steps is too far); two a step apart at one depth (ranked in the reply's order, the run's record order, as
  // the engine ranks them: engine/qc_test.py match); one alone.
  const reply = normaliseWhere({ v: 3, seq: 12, cell: 1, box: [100, 50, 140, 90], genes: ['EDGE', 'OTHER'], roles: ['from', 'from'],
    points: { x: [118.5, 118.5, 118.5, 120, 120.01, 120.02, 125.01, 125, 130], y: [64.25, 64.25, 64.25, 70, 70, 70, 80, 80, 85],
      z: [0.9, 0.1, 0.5, 2, 1, 3, 1.5, 1.5, 1], gene: [0, 0, 1, 0, 0, 0, 0, 0, 0], owner: [2, 1, 1, 1, 255, 1, 1, 0, 1], near: [0, 1, 0, 0, 0, 0, 0, 0, 0],
      w: [0, 1, 1, 1, 0, 1, 1, 0, 1] },
    gene_stats: [], into: [] });
  assert.ok(reply.z instanceof Float32Array && close(reply.z[1], 0.1), 'each point\'s depth kept');
  assert.deepEqual(Object.fromEntries(samePlace(reply, [0, 1, 2, 3, 4, 5, 6, 7, 8])), { 0: { rank: 1, of: 2 }, 1: { rank: 0, of: 2 },
    3: { rank: 1, of: 2 }, 4: { rank: 0, of: 3 }, 5: { rank: 1, of: 2 }, 6: { rank: 0, of: 2 }, 7: { rank: 1, of: 2 } });
  assert.deepEqual([...samePlace(reply, [1]).keys()], [1], 'only the points asked for'); assert.equal(samePlace(reply, []).size, 0);
  // Far from the origin the places are 32-bit: still one step of the grid apart, or the same place.
  const far = normaliseWhere({ v: 3, seq: 12, cell: 1, box: null, genes: ['EDGE'], roles: ['from'],
    points: { x: [20000.01, 20000.02, 20000.04], y: [9000.37, 9000.37, 9000.37], z: [3, 1, 2], gene: [0, 0, 0], owner: [1, 2, 1], near: [0, 0, 0], w: [1, 0, 1] },
    gene_stats: [], into: [] });
  assert.deepEqual(Object.fromEntries(samePlace(far, [0, 2])), { 0: { rank: 1, of: 2 } });
  // A reply without depths (too few of them, or one missing) is still a reply, but can't tell its points apart.
  const flat = { v: 3, seq: 12, cell: 1, box: null, genes: ['EDGE'], roles: ['from'],
    points: { x: [118.5, 118.5], y: [64.25, 64.25], gene: [0, 0], owner: [2, 1], near: [0, 1], w: [0, 1] }, gene_stats: [], into: [] };
  for (const z of [undefined, [0.9], [0.9, null]]) {
    const r = normaliseWhere({ ...flat, points: { ...flat.points, ...(z ? { z } : {}) } });
    assert.ok(r, String(z)); assert.equal(samePlace(r, [1]).size, 0, String(z));
  }
});

test('a test starts with a suspect\'s rank by depth where another transcript of its gene shares its place', async () => {
  // The cell's GXPN at the shared edge (110, 120) lies at z 0.1; the neighbour's GXPN at the same place, deeper (z 0.9).
  const pts = [[110, 120, 0, 1, 1], [111, 121, 0, 1, 0], [140, 125, 0, 2, 0], [170, 150, 0, 255, 0], [112, 128, 1, 0, 0], [113, 129, 1, 1, 0], [110, 120, 0, 2, 0]];
  const w = make({ live: false, where: params => { const r = whereReply(params, { points: pts }); r.points.z = [0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9]; return r; } });
  w.deeper.open(2); await settle();
  toggle(w).click(); await settle();
  const kept = w.deeper.suspects(2);
  assert.deepEqual(kept.map(s => [s.x, s.y, s.gene, s.p]), [[110, 120, 'GXPN', 0.932], [113, 129, 'GXC', 0.833], [111, 121, 'GXPN', 0.277]]);
  assert.deepEqual(kept[0], { x: 110, y: 120, gene: 'GXPN', p: 0.932, rank: 0, of: 2 }, 'the shallower of the two at its place');
  assert.deepEqual(Object.keys(kept[1]), ['x', 'y', 'gene', 'p'], 'alone at its place: neither');
  assert.equal(w.calls.glows.at(-1).points.length, 9, 'the glow as before: the neighbour\'s transcript is no suspect');
});

test('the glow and the suspects follow the result shown: the same frame\'s result from another model, or another answer, scores them again', async () => {
  const w = make({ live: false });
  w.deeper.open(2); await settle();
  toggle(w).click(); await settle();
  const where = () => w.deeper.where, p = () => [...where().scores.p];
  const glowP = () => [...w.calls.glows.at(-1).points].filter((_, i) => i % 3 === 2).map(v => +v.toFixed(4));
  const key = w.deeper.whereWanted.key, glows = w.calls.glows.length;
  assert.ok(close(p()[1], 0.277444)); assert.deepEqual(w.deeper.suspects(2).map(s => s.p), [0.932, 0.833, 0.277]);
  // The sample learned again: the new model reads 38.2 of GXPN's 39 transcripts here as a
  // neighbour's, for the same cell, frame, genes and rows: the where request (and its reply) stays the same.
  const relearned = mixed({ model: 'm2' }); relearned.genes[0] = ['GXPN', 39, 0.8, 38.2, 1, 0];
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 12, cell: 1, model: 'm2', result: relearned }); await settle();
  assert.equal(w.deeper.shown.result.model, 'm2', 'the window shows the new split (paused: at once)');
  assert.equal(where().result, w.deeper.shown.result, 'the view is bound to the result shown');
  assert.deepEqual(p(), [...suspectScores(where().reply, w.deeper.shown.result).p]);
  assert.ok(close(p()[1], 0.926829), `${p()[1]}: GXPN's transcript further in, scored on the new split`);
  assert.equal(w.deeper.whereWanted.key, key); assert.equal(w.calls.wheres.length, 1, 'the same reply: nothing asked again');
  assert.equal(w.calls.glows.length, glows + 1, 'the stage given the new glow'); assert.deepEqual(glowP(), [0.8328, 0.9268, 0.9931]);
  assert.deepEqual(w.deeper.suspects(2).map(s => s.p), [0.993, 0.927, 0.833], 'and a test starts with the new suspects');
  // A repaint of the same result (the run's state changed): nothing scored or handed over again.
  const n = w.calls.glows.length, scored = where().scored;
  w.ctx.run.state = 'running'; w.deeper.changed(); await settle();
  assert.equal(where().scored, scored); assert.equal(w.calls.glows.length, n);
  // Another answer for the frame from the same model: scored again too.
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 12, cell: 1, model: 'm2', result: mixed({ model: 'm2' }) }); await settle();
  assert.equal(where().result, w.deeper.shown.result); assert.ok(close(p()[1], 0.277444));
  assert.equal(w.calls.glows.length, n + 1); assert.deepEqual(w.deeper.suspects(2).map(s => s.p), [0.932, 0.833, 0.277]);
});

// ---- pink dots near none of the neighbour that would be named (the strip case), the shares' caveats -------------
/** cell-a (the window's cell, ffaaaaaa-1, cell 1): 13 of its GZFN, a gene of another kind, lie in a strip along its
 *  far edge; ffbbbbbb-1 (cell 2), the only cell of that kind, 30 µm away on the other side, is the neighbour the model
 *  names for them; ggcccccc-1 (cell 3) is of its own kind. */
function cellpose(extra = {}) {
  return mixed({ code: 40, parts: [0.82, 0.16, 0.02], transcripts: 97, outside: 0, same_kind_share: 0.1,
    genes: [['GZFN', 13, 1.0, 12.0, 1, 0], ['GYHNN', 60, 59.0, 1.0, 0], ['GXEN', 24, 23.5, 0.5, 0]],
    kind: ['GYHNN', 'GXEN'], own_genes: [['GYHNN', 59, 60], ['GXEN', 23.5, 24]],
    neighbours: [{ cell: 2, um: 27.5, tx: 12, genes: [['GZFN', 12]], kind: ['GZFN', 'GZQN', 'GZRN', 'GZSN'], similar: 0.1, known: true },
      { cell: 3, um: 13.5, tx: 2, genes: [['GYHNN', 1]], kind: ['GYHNN'], similar: 0.95, known: true }],
    start_change: null, ...extra });
}
/** The engine's reply for it: the strip (x 130.5) proseg's (w 0.875 each), far from ffbbbbbb-1 (x 160-170), or (`beside`)
 *  right beside it (the strip at x 144.5, the neighbour from x 145: near flags set). */
function cellposeReply(params, { beside = false, held = 'exact' } = {}) {
  const gene = name => [...params.genes, ...params.own].indexOf(name), x = beside ? 144.5 : 130.5, fx = beside ? 150 : 165;
  const pts = [...Array.from({ length: 13 }, (_, i) => [x, 65.5 + i, gene('GZFN'), 1, beside ? 1 : 0, held === 'none' ? 0 : 0.875]),
    ...Array.from({ length: 6 }, (_, i) => [fx, 66 + i, gene('GZFN'), 2, 0, 0]),
    ...Array.from({ length: 6 }, (_, i) => [135 + i, 70, gene('GYHNN'), 1, 0, held === 'none' ? 0 : 1])];
  const near = beside ? 13 * 0.875 : 0;
  return whereReply(params, { points: pts, box: [100, 50, 180, 100], total: held === 'none' ? 0 : 97, held,
    stats: [{ gene: 'GZFN', role: 'from', giver: 0, giver_by: 'result', in_cell: 11.375, near_giver: near, own_near: [0, 80],
      density: { cell: 5.1, giver: 9.4, around: 0 }, word: beside ? 'edge' : 'spread' }],
    into: [] });
}

test('the support of a source: proseg\'s share of the glowing transcripts near it; a named source with none is not where they come from', () => {
  const req = (() => { const r = normalise(cellpose()), m = explain(r, { name: c => CELLS[c], label: c => c + 1 });
    return whereRequest(r, { cell: 2, none: m.none, rows: sourceRows(m).rows, chips: m.chips.map(c => c.gene) }); })();
  assert.deepEqual(req.params, { seq: 12, cell: 1, sources: [2, 3], genes: ['GZFN'], own: ['GYHNN', 'GXEN'] });
  assert.deepEqual(req.sources.map(s => [s.name, s.kind]), [['ffbbbbbb-1', ['GZFN', 'GZQN', 'GZRN', 'GZSN']], ['ggcccccc-1', ['GYHNN']]]);
  const result = normalise(cellpose());
  for (const beside of [false, true]) {
    const reply = normaliseWhere(cellposeReply(req.params, { beside }), req), scores = suspectScores(reply, result);
    const support = sourceSupport(reply, scores);
    assert.ok(support instanceof Map);
    assert.deepEqual([...support], beside ? [[0, 13 * 0.875]] : [], 'w of the glowing points near it');
    const g = glowPoints(reply, scores);
    assert.deepEqual([g.count, g.likely], [13, 13], 'likely: pi 12 / 13 on 0.875 of each for the cell');
    // Far: no giver to ripple in from, the source not named (notNear); beside: as ever.
    assert.deepEqual([g.giver, g.notNear], beside ? [0, null] : [null, 0]);
    assert.deepEqual([glowPoints(reply, scores, 0).giver, glowPoints(reply, scores, 0).notNear], beside ? [0, null] : [null, 0], 'its row\'s Show too');
    // The camera's box then: the cell and its glow, a margin around, never the neighbour 30 µm away.
    const box = glowBoxOf(reply, g.points);
    assert.deepEqual(box, beside ? [135 - 6, 65.5 - 6, 144.5 + 6, 77.5 + 6] : [130.5 - 6, 65.5 - 6, 140 + 6, 77.5 + 6]);
  }
  // A glowing point near its giver without a share is no support (P 0: it does not glow); one under the glow neither.
  const thin = normaliseWhere(whereReply(req.params, { points: [[130.5, 66, 0, 1, 1, 0.1], [130.5, 67, 0, 1, 1, 0]], box: [100, 50, 180, 100],
    stats: cellposeReply(req.params).gene_stats }), req);
  assert.deepEqual([...sourceSupport(thin, suspectScores(thin, result))], []);
  assert.deepEqual([...sourceSupport(null, null)], []); assert.equal(glowBoxOf(null, null), null);
});

test('Where they sit: none of the glow near the neighbour it would name: "Likely from a … cell with no outline here", its tooltip naming that neighbour; approximate shares and none said in the tooltip', () => {
  const r = normalise(cellpose()), m = explain(r, { name: c => CELLS[c], label: c => c + 1 });
  const req = whereRequest(r, { cell: 2, none: m.none, rows: sourceRows(m).rows, chips: m.chips.map(c => c.gene) });
  const reply = normaliseWhere(cellposeReply(req.params), req), g = glowPoints(reply, suspectScores(reply, r));
  const glow = { count: g.count, likely: g.likely, near: g.near, start: g.start, places: g.places, levels: g.levels, row: null, notNear: g.notNear };
  const line = whereModel(reply, req, [], glow).line;
  assert.equal(line.text, 'Likely from a GZFN · GZQN · GZRN cell with no outline here');
  assert.equal(line.tip, `None of these pink dots lies near ffbbbbbb-1: they more likely come from a cell of its kind that has no outline here. ${DEEP_TEXT.deepGlowTooltip}`);
  assert.deepEqual(line.counts, { likely: 13, own: null, none: null, giver: null, other: null, unknown: null });
  assert.deepEqual(noOutlineLine(glow, { name: 'n', kind: [] }).text, DEEP_TEXT.deepSourceNoOutlinePlain, 'its kind not known');
  // Where they sit: GZFN under the same words, never under that neighbour's name; nothing said of an edge it shares
  // with it (there is none) or of its density; proseg's own numbers about the cell and the tissue around stay.
  const blocks = whereModel(reply, req, [], glow).sources;
  assert.deepEqual(blocks.map(b => [b.name, b.away ?? false, b.genes.map(g => g.text)]),
    [['Likely from a GZFN · GZQN · GZRN cell with no outline here', true, ['GZFN']]]);
  assert.equal(blocks[0].head, 'None of these pink dots lies near ffbbbbbb-1: they more likely come from a cell of its kind that has no outline here.');
  const gzfn = blocks[0].genes[0];
  assert.equal(gzfn.word, null); assert.doesNotMatch(gzfn.tooltip, /edge|neighbour|within 2 µm/, gzfn.tooltip);
  assert.deepEqual(gzfn.numbers, ["Per 100 µm², in proseg's count: 5 in this cell · 0 in the cells around."]);
  // No glow (the view's line not drawn yet), or one near the neighbour: under its name, as ever.
  assert.deepEqual(whereModel(reply, req, []).sources.map(b => [b.name, b.genes.map(g => g.text)]), [['ffbbbbbb-1', ['GZFN · spread through the cell']]]);
  // Where they sat at the start follows it, as on the usual line.
  const placed = whereModel(reply, req, [], { ...glow, places: { own: 0, none: 13, giver: 0, other: 0, unknown: 0 } }).line;
  assert.equal(placed.text, 'Likely from a GZFN · GZQN · GZRN cell with no outline here · 13 where no cell started');
  // Beside it: the usual line.
  const near = normaliseWhere(cellposeReply(req.params, { beside: true }), req), gn = glowPoints(near, suspectScores(near, r));
  assert.equal(whereModel(near, req, [], { ...gn, row: null }).line.text, '13 likely from neighbours');
  // Approximate shares (the frame's layers): said after the glow's tooltip; none (no layers here): said too.
  const approx = normaliseWhere(whereReply(mixedRequest().params, { held: 'approx' }), mixedRequest());
  const tip = whereModel(approx, mixedRequest(), [], { count: 3, likely: 2, near: 1, row: null }).line.tip;
  assert.equal(tip, `${startTip({ count: 3, likely: 2, near: 1 })} ${DEEP_TEXT.deepGlowApprox}`);
  // ... and every gene's: its sums of w are placed from the frame's layers, not proseg's count of it.
  const rough = whereModel(approx, mixedRequest(), []);
  for (const g of [...rough.sources.flatMap(b => b.genes), ...rough.other]) assert.ok(g.tooltip.endsWith(` ${DEEP_TEXT.deepGlowApprox}`), g.tooltip);
  const exactRows = whereModel(normaliseWhere(whereReply(mixedRequest().params), mixedRequest()), mixedRequest(), []);
  for (const g of [...exactRows.sources.flatMap(b => b.genes), ...exactRows.other]) assert.doesNotMatch(g.tooltip, /Approximate/, g.tooltip);
  // "Too few to tell" says proseg's own count of the gene (its count, from the frame's counts), not the layers' sum.
  const counted = whereReply(mixedRequest().params, { held: 'approx' });
  // (Under WHERE_FEW, the engine's FEW_IN_CELL: 4 on proseg's count.)
  counted.gene_stats = counted.gene_stats.map(st => st.gene === 'GYDNNN' ? { ...st, in_cell: 1.5, count: WHERE_FEW - 1 } : st);
  const few = whereModel(normaliseWhere(counted, mixedRequest()), mixedRequest(), []).other[0];
  assert.deepEqual([few.text, few.numbers], ['GYDNNN · too few to tell', ['Proseg counts 3 for this cell.']]);
  assert.equal(few.tooltip, `${DEEP_TEXT.deepWhereFewTooltip} Proseg counts 3 for this cell. ${DEEP_TEXT.deepGlowApprox}`);
  // Proseg counts WHERE_FEW or more of it, and fewer are placed here (the engine's "few" for want of placed ones): too
  // few to tell all the same, its tooltip saying why, never that proseg counts fewer than 4.
  const short = whereReply(mixedRequest().params, { held: 'approx' });
  short.gene_stats = short.gene_stats.map(st => st.gene === 'GYDNNN' ? { ...st, in_cell: 3.69, count: WHERE_FEW + 0.06 } : st);
  const placedFew = whereModel(normaliseWhere(short, mixedRequest()), mixedRequest(), []).other[0];
  assert.deepEqual([placedFew.text, placedFew.word, placedFew.numbers], ['GYDNNN · too few to tell', 'few',
    ['Proseg counts 4 for this cell, but fewer than 4 of them could be placed in this view: too few to tell where they sit.']]);
  assert.equal(placedFew.tooltip, `${placedFew.numbers[0]} ${DEEP_TEXT.deepGlowApprox}`);
  assert.doesNotMatch(placedFew.tooltip, /counts fewer than 4/, placedFew.tooltip);
  const none = normaliseWhere(whereReply(mixedRequest().params, { held: 'none', points: [[110, 120, 0, 1, 1, 0], [111, 121, 0, 1, 0, 0]] }), mixedRequest());
  const empty = glowPoints(none, suspectScores(none, normalise(mixed())));
  assert.equal(empty.count, 0, 'no share: nothing glows');
  // Nothing measured: no verdict ("None looks likely" would say what was never measured), a neutral line, the why in
  // its tooltip; the same for one row's Show.
  const said = whereModel(none, mixedRequest(), [], { ...empty, row: null }).line;
  assert.deepEqual([said.text, said.tip], ['No pink dots for this iteration', `${DEEP_TEXT.deepGlowTooltip} ${DEEP_TEXT.deepGlowNone}`]);
  assert.equal(whereModel(none, mixedRequest(), [], { ...empty, row: 3 }).line.text, DEEP_TEXT.deepGlowNoneShort);
  assert.equal(whereModel(approx, mixedRequest(), [], { ...empty, row: null }).line.text, 'None looks likely from neighbours',
    'measured (approximately): its verdict stands');
  for (const text of [line.text, line.tip, tip, said.tip]) assert.deepEqual(bannedWords(text), [], text);
});

test('a run that ended (done, stopped, failed) or saves its results (finishing): approximate or missing shares never say to pause it; said so as soon as it ends with the view on', async () => {
  // Finishing, proseg writes its results: the engine refuses a pause ("This run is not sampling") and a where export
  // (request_where: running or paused only), as for a run that ended.
  const noShare = params => whereReply(params, { held: 'none', points: [[110, 120, 0, 1, 1, 0], [111, 121, 0, 1, 0, 0]] });
  for (const state of ['done', 'stopped', 'failed', 'finishing']) {
    // Placed from the frame's layers: every gene's tooltip and the glow line's say so, and that proseg can't be asked.
    const w = make({ live: false, run: { state }, seq: 39, where: params => whereReply(params, { held: 'approx' }), start: () => fail(404) });
    w.deeper.open(2); w.clock.advance(200); await settle();
    toggle(w).click(); await settle();
    const nodes = [...w.root.all('p.qc-deeper-where-gene'), w.root.one('p.qc-deeper-glow-legend')];
    assert.equal(nodes.length, 4);
    for (const node of nodes) {
      assert.ok(node.title.endsWith(` ${DEEP_TEXT.deepGlowApproxEnded}`), node.title);
      assert.doesNotMatch(node.title, /pause|newest/, node.title);
    }
    // No share held here: each gene alone with the why, which never says to pause it either.
    const none = make({ live: false, run: { state }, seq: 39, where: noShare, start: () => fail(404) });
    none.deeper.open(2); none.clock.advance(200); await settle();
    toggle(none).click(); await settle();
    assert.deepEqual(none.root.all('p.qc-deeper-where-gene').map(p => p.title), [1, 2, 3].map(() => DEEP_TEXT.deepGlowNoneEnded));
    assert.ok(none.root.one('p.qc-deeper-glow-legend').title.endsWith(` ${DEEP_TEXT.deepGlowNoneEnded}`));
    assert.doesNotMatch(none.root.one('p.qc-deeper-glow-legend').title, /pause|newest/);
  }
  // Paused, the view on, then Finish & save (the run saves its results), or the run is stopped: its tooltips follow at
  // once.
  for (const state of ['finishing', 'stopped']) {
    const w = make({ live: false, run: { state: 'paused' }, seq: 39, where: params => whereReply(params, { held: 'approx' }), start: () => fail(404) });
    w.deeper.open(2); w.clock.advance(200); await settle();
    toggle(w).click(); await settle();
    const titles = () => [...w.root.all('p.qc-deeper-where-gene'), w.root.one('p.qc-deeper-glow-legend')].map(p => p.title);
    assert.ok(titles().every(t => t.endsWith(` ${DEEP_TEXT.deepGlowApprox}`)), titles().join('\n'));
    w.ctx.run.state = state; w.deeper.changed(); await settle();
    assert.ok(titles().every(t => t.endsWith(` ${DEEP_TEXT.deepGlowApproxEnded}`)), `${state}\n${titles().join('\n')}`);
  }
  for (const text of [DEEP_TEXT.deepGlowApproxEnded, DEEP_TEXT.deepGlowNoneEnded]) assert.deepEqual(bannedWords(text), [], text);
});

test('Where they sit with no share held (held "none": a tiled frame\'s view window misses the cell, or the frame\'s counts are gone): each gene alone, no word and no number (they would read as proseg\'s count), the why in its tooltip; no unlit line', () => {
  const req = mixedRequest(), points = [[110, 120, 0, 1, 1, 0], [111, 121, 0, 1, 0, 0]];
  // The engine's stats in this state: every sum 0 and no word (until round 3 every gene read "few": "Proseg counts 0 for
  // this cell" beside a card that counts 39 of GXPN); and stats with words and numbers, which no reply of this state
  // carries: none of them is said either.
  const zero = s => ({ ...s, in_cell: 0, near_giver: s.giver == null ? null : 0, own_near: s.giver == null ? null : [0, 0],
    density: { cell: 0, giver: s.giver == null ? null : 0, around: 0 }, word: null });
  const engine = whereReply(req.params).gene_stats.map(zero);
  for (const stats of [engine, engine.map(s => ({ ...s, word: 'few' })), undefined]) {
    const reply = normaliseWhere(whereReply(req.params, { held: 'none', points, stats, total: 0 }), req);
    const glow = glowPoints(reply, suspectScores(reply, normalise(mixed())));
    const m = whereModel(reply, req, [], { ...glow, row: null });
    const rows = [...m.sources.flatMap(s => s.genes), ...m.other];
    assert.deepEqual(rows.map(g => g.gene).sort(), ['GXC', 'GXPN', 'GYDNNN'], 'every "from" gene still listed, by its neighbour');
    assert.deepEqual(m.other.map(g => g.gene), ['GYDNNN']);
    for (const g of rows) assert.deepEqual([g.text, g.word, g.tooltip, g.numbers], [g.gene, null, DEEP_TEXT.deepGlowNone, []], g.gene);
    // No into line either: what proseg counts for this cell inside the neighbour's outline is not known here (and a
    // count of the transcripts inside it would not be proseg's).
    assert.equal(m.sources[0].into, null);
    assert.equal(m.unlit, null, 'nothing lit here, for every gene alike');
    assert.deepEqual([m.line.text, m.line.tip], [DEEP_TEXT.deepGlowNoneShort, `${DEEP_TEXT.deepGlowTooltip} ${DEEP_TEXT.deepGlowNone}`]);
    for (const g of rows) for (const text of [g.text, g.tooltip]) assert.doesNotMatch(text, /\d|too few|counts/i, text);
  }
  // Held (exact here): the same stats' words and numbers stand.
  const held = normaliseWhere(whereReply(req.params), req), m = whereModel(held, req, [], { count: 0, likely: 0, near: 0, row: null });
  assert.deepEqual(m.other.map(g => [g.text, g.numbers]), [['GYDNNN · too few to tell', ['Proseg counts 3 for this cell.']]]);
  assert.equal(m.unlit.text, 'No edge measured for GYDNNN');
});

test('the strip case in the window: the glow names no neighbour, has no side to ripple in from, and the camera fits the cell and its glow; a row\'s Show likewise', async () => {
  for (const beside of [false, true]) {
    const w = make({ live: false, answer: body => ready(cellpose({ seq: body.seq })), where: params => cellposeReply(params, { beside }),
      start: () => fail(404) });
    w.deeper.open(2); w.clock.advance(200); await settle();
    // The gene table open: GZFN's row names the neighbour the model splits most of it to, until the view is on.
    w.deeper.tableBox.open = true; w.deeper.tableBox.emit('toggle');
    const gzfn = () => w.root.all('tr').find(tr => tr.dataset.gene === 'GZFN');
    const named = 'GZFN: 13 transcripts · about 1 its own, 12 from neighbours (most from ffbbbbbb-1) (estimate)';
    assert.equal(gzfn().title, named);
    toggle(w).click(); await settle();
    const legend = w.root.one('p.qc-deeper-glow-legend'), glow = w.calls.glows.at(-1), reply = normaliseWhere(cellposeReply(w.calls.wheres[0], { beside }), null);
    assert.equal(glow.points.length, 3 * 13);
    const [rect] = w.calls.flies[0];
    if (beside) {
      assert.equal(gzfn().title, named, 'its glow near the neighbour: named as ever');
      assert.equal(legend.visibleText().trim(), '13 likely from neighbours');
      assert.deepEqual(glow.giver, [150, 68.5], 'it ripples in from the neighbour beside it');
      assert.deepEqual(w.calls.spills.at(-1).lit, [2, 3, 4]); assert.deepEqual(w.calls.focus.at(-1).lit, [2, 3, 4]);
      assert.equal(w.root.one('div.qc-deeper-where-source').one('p.qc-deeper-where-head').textContent, 'ffbbbbbb-1');
      assert.deepEqual(rect, w.deeper.whereRect(reply.box), 'the cell and its sources');
      assert.equal(w.deeper.snapshot().where.glow.notNear, null);
      continue;
    }
    assert.equal(legend.visibleText().trim(), 'Likely from a GZFN · GZQN · GZRN cell with no outline here');
    assert.match(legend.title, /^None of these pink dots lies near ffbbbbbb-1: they more likely come from a cell of its kind that has no outline here\./);
    assert.equal(glow.giver, null, 'no side to ripple in from: never the far neighbour');
    // Nowhere else is it presented as their source: its row's tooltip says what the line says; Where they sit lists
    // GZFN under the line's words, with no word about an edge it shares with it; it is neither lit, outlined nor
    // ringed (cell 3); the cell and its same-kind neighbour stay lit.
    const far = w.root.all('li.qc-deeper-source').find(li => li.dataset.cell === 'ffbbbbbb-1');
    assert.ok(far.title.endsWith(' None of these pink dots lies near ffbbbbbb-1: they more likely come from a cell of its kind that has no outline here.'), far.title);
    // ... nor the gene table's row (the rows drawn already, and those drawn again as the table scrolls).
    const unnamed = 'GZFN: 13 transcripts · about 1 its own, 12 from neighbours (estimate) None of these pink dots lies near ffbbbbbb-1: they more likely come from a cell of its kind that has no outline here.';
    assert.equal(gzfn().title, unnamed);
    w.deeper.renderTable(true); assert.equal(gzfn().title, unnamed);
    assert.deepEqual(w.root.all('div.qc-deeper-where-source').map(b => [b.dataset.cell, b.one('p.qc-deeper-where-head').textContent,
      b.all('p.qc-deeper-where-gene').map(p => p.visibleText().trim())]), [['', 'Likely from a GZFN · GZQN · GZRN cell with no outline here', ['GZFN']]]);
    assert.doesNotMatch(w.root.one('p.qc-deeper-where-gene').title, /edge|neighbour/);
    assert.deepEqual(w.calls.spills.at(-1).lit, [2, 4]); assert.deepEqual(w.deeper.snapshot().where.lit, [2, 4]);
    const focus = w.calls.focus.at(-1);
    assert.deepEqual([focus.sources.map(f => f.label), focus.lit], [[], [2, 4]]);
    assert.deepEqual(rect, w.deeper.whereRect(glowBoxOf(reply, glow.points)), 'the cell and its glow');
    assert.notDeepEqual(rect, w.deeper.whereRect(reply.box));
    assert.equal(w.deeper.snapshot().where.glow.notNear, 0);
    // That neighbour's row Show: its 13 still near none of it: the same line, and the camera on the cell and its glow.
    w.root.all('li.qc-deeper-source').find(li => li.dataset.cell === 'ffbbbbbb-1').one('button[data-action=show]').click(); await settle();
    assert.equal(legend.visibleText().trim(), 'Likely from a GZFN · GZQN · GZRN cell with no outline here');
    assert.equal(w.calls.glows.at(-1).giver, null);
    assert.deepEqual(w.calls.flies.at(-1)[0], w.deeper.whereRect(glowBoxOf(reply, w.calls.glows.at(-1).points)));
    assert.notDeepEqual(w.calls.flies.at(-1)[0], w.deeper.whereRect(pairBox(reply, 0)));
    for (const text of w.deeper.texts()) assert.deepEqual(bannedWords(text), [], text);
    // The view off: its row is a source row again, lit and outlined, its tooltip its own; the table's row names it again.
    toggle(w).click(); await settle();
    assert.deepEqual(w.calls.focus.at(-1).lit, [2, 3, 4]); assert.deepEqual(w.calls.focus.at(-1).sources.map(f => f.label), [3]);
    assert.doesNotMatch(far.title, /None of these pink dots/);
    assert.equal(gzfn().title, named);
  }
  // Approximate shares: the glow's tooltip says so; none here: nothing glows, its tooltip says why.
  const approx = make({ live: false, where: params => whereReply(params, { held: 'approx' }), start: () => fail(404) });
  approx.deeper.open(2); approx.clock.advance(200); await settle();
  toggle(approx).click(); await settle();
  assert.match(approx.root.one('p.qc-deeper-glow-legend').title, new RegExp(`${DEEP_TEXT.deepGlowApprox.replace(/[.?]/g, '\\$&')}$`));
  assert.equal(approx.deeper.snapshot().where.held, 'approx');
  const none = make({ live: false, where: params => whereReply(params, { held: 'none', points: [[110, 120, 0, 1, 1, 0], [111, 121, 0, 1, 0, 0]] }), start: () => fail(404) });
  none.deeper.open(2); none.clock.advance(200); await settle();
  toggle(none).click(); await settle();
  assert.equal(none.calls.glows.length, 0, 'nothing glows');
  assert.match(none.root.one('p.qc-deeper-glow-legend').title, new RegExp(`${DEEP_TEXT.deepGlowNone.replace(/[.?]/g, '\\$&')}$`));
  assert.equal(none.root.one('p.qc-deeper-glow-legend').visibleText().trim(), DEEP_TEXT.deepGlowNoneShort, 'no verdict on screen');
  // Where they sit: each gene alone, the why in its tooltip; no word, no number, no unlit line.
  assert.deepEqual(none.root.all('p.qc-deeper-where-gene').map(p => [p.visibleText().trim(), p.dataset.word ?? null, p.title]),
    ['GXPN', 'GXC', 'GYDNNN'].map(gene => [gene, null, DEEP_TEXT.deepGlowNone]));
  assert.equal(none.root.one('p.qc-deeper-glow-unlit').hidden, true);
  for (const text of none.deeper.texts()) assert.doesNotMatch(text, /too few to tell|Proseg counts \d|in proseg's count:/, text);
});

// ---- the round of 2026-10-31: the camera's part of the stage, the glow's depth, the exact dots kept while following ------
test('whereFree is the part of the stage whereView centres the box in: the 2-D fit puts the box\'s middle at its middle, the box inside it, as large as it fits', () => {
  const stage = { width: 1600, height: 1000 }, fit = rect => { const zoom = Math.min(stage.width / rect.w, stage.height / rect.h); return { zoom, x: rect.x0 + rect.w / 2, y: rect.y0 + rect.h / 2 }; };
  const toScreen = (cam, x, y) => [(x - cam.x) * cam.zoom + stage.width / 2, (y - cam.y) * cam.zoom + stage.height / 2];
  const cases = [[[1000, 1000, 1060, 1060], { left: 16, top: 16, w: 420, h: 700 }, null], [[0, 0, 120, 60], { left: 1600 - 372, top: 52, w: WIDTH, h: 500 }, null],
    [[0, 0, 60, 120], null, null], [[100, 100, 130, 200], null, { left: 12, top: 1000 - 12 - 380, w: 600, h: 380 }],
    [[5, 5, 205, 55], { left: 16, top: 16, w: 420, h: 700 }, { left: 1000, bottom: 12, w: 500, h: 300 }]];
  for (const [box, win, corner] of cases) {
    const free = whereFree(box, stage, win, corner), cam = fit(whereView(box, stage, win, corner));
    const [ax, ay] = toScreen(cam, box[0], box[1]), [bx, by] = toScreen(cam, box[2], box[3]);
    const name = JSON.stringify([box, win, corner]);
    assert.ok(close((ax + bx) / 2, (free[0] + free[2]) / 2) && close((ay + by) / 2, (free[1] + free[3]) / 2), `${name}: centred`);
    assert.ok(ax >= free[0] - 1e-6 && bx <= free[2] + 1e-6 && ay >= free[1] - 1e-6 && by <= free[3] + 1e-6, `${name}: inside`);
    assert.ok(close(bx - ax, free[2] - free[0]) || close(by - ay, free[3] - free[1]), `${name}: as large as it fits`);
  }
  // Clear of the window beside it (the wider side), WHERE_MARGIN inside the stage.
  assert.deepEqual(whereFree([0, 0, 10, 10], stage, { left: 16, top: 16, w: 420, h: 700 }), [16 + 420 + 12, 12, 1600 - 12, 1000 - 12]);
  assert.deepEqual(whereFree([0, 0, 10, 10], stage, null), [12, 12, 1588, 988]);
});

test('the camera\'s move hands its view only in 3-D; in 2-D the rect alone, as before', async () => {
  const w = make();
  w.deeper.open(2); await settle();
  toggle(w).click(); await settle();
  assert.equal(w.calls.flies.length, 1); assert.equal(w.calls.flies[0][2], null, '2-D: no view');
  const box = w.deeper.where.reply.box;
  assert.deepEqual(w.calls.flies[0][0], w.deeper.whereRect(box)); assert.deepEqual(w.deeper.whereFree(box), whereFree(box, w.ctx.stage,
    { left: w.deeper.left, top: w.deeper.top, w: w.deeper.size.w, h: w.deeper.size.h }));
});

/** The window following a live cell whose where replies are `held` per frame (exact unless said), each reply naming its
 *  frame's iteration. */
function following(held = {}) {
  const w = make({ where: params => ({ ...whereReply(params, { held: held[params.seq] ?? 'exact' }), iter: params.seq * 100 + 1 }), start: () => fail(404) });
  const counted = new Set([12]);
  w.ctx.counted = seq => counted.has(seq);
  const push = seq => {
    counted.add(seq);
    w.clock.advance(1100); w.ctx.seq = seq;
    w.deeper.event({ ev: 'deeper', run: 'r1', seq, iter: seq * 100 + 1, cell: 1, model: 'm1', result: mixed({ seq, iter: seq * 100 + 1 }) });
    w.deeper.changed(); w.batch.flush();
  };
  return { w, counted, push, status: () => w.root.one('section.qc-deeper-where').one('div.qc-deeper-where-status') };
}

test('following a cell as the run plays: an approximate reply right after an exact one of the counted frame before keeps the exact one, said so; an exact one, or one a counted frame on, replaces it', async () => {
  // Resumed after a pause on frame 12 (its shares exact); frame 13's reply comes back approximate (its box was not armed).
  const { w, push, status } = following({ 13: 'approx', 14: 'approx' });
  w.deeper.open(2); await settle();
  toggle(w).click(); await settle();
  const spills = w.calls.spills.length, glows = w.calls.glows.length;
  assert.equal(w.deeper.snapshot().where.snapped, false);
  push(13); await settle(); w.clock.advance(1100); await settle();
  assert.deepEqual(w.calls.wheres.map(p => p.seq), [12, 13]);
  // Kept: the dots and the glow of frame 12 stay on the stage (nothing handed over again), said in the where line.
  assert.equal(w.calls.spills.length, spills); assert.equal(w.calls.glows.length, glows);
  const s = w.deeper.snapshot().where;
  assert.deepEqual([s.snapped, s.seq, s.held], [true, 12, 'exact'], 'the shown reply\'s own seq and held');
  const line = status().all('p').find(p => p.textContent === 'Dots at iteration 1,201');
  assert.ok(line, 'Dots at iteration 1,201');
  assert.equal(line.title, "Proseg's exact shares at iteration 1,201, the iteration it counted before this one: here this iteration's shares would only be approximate.");
  assert.equal(w.counters.qcDeeperWhereSnapped, 1);
  // Not asked again for frame 13.
  w.clock.advance(5000); await settle();
  assert.deepEqual(w.calls.wheres.map(p => p.seq), [12, 13]);
  // Frame 14 is a counted frame further on (13 counted between): its approximate reply shows, the line goes.
  push(14); await settle(); w.clock.advance(1100); await settle();
  assert.deepEqual([w.deeper.snapshot().where.seq, w.deeper.snapshot().where.held, w.deeper.snapshot().where.snapped], [14, 'approx', false]);
  assert.equal(w.calls.spills.length, spills + 1); assert.equal(status().all('p').some(p => /^Dots at iteration/.test(p.textContent)), false);
  // An exact reply replaces a kept one at once.
  const e = following({ 13: 'approx' });
  e.w.deeper.open(2); await settle();
  toggle(e.w).click(); await settle();
  e.push(13); await settle(); e.w.clock.advance(1100); await settle();
  assert.equal(e.w.deeper.snapshot().where.snapped, true);
  e.push(14); await settle(); e.w.clock.advance(1100); await settle();
  assert.deepEqual([e.w.deeper.snapshot().where.seq, e.w.deeper.snapshot().where.held, e.w.deeper.snapshot().where.snapped], [14, 'exact', false]);
  assert.equal(e.status().all('p').some(p => /^Dots at iteration/.test(p.textContent)), false);
  // A counted frame between them (the run counted one the window did not ask for): the approximate reply shows.
  const g = following({ 14: 'approx' });
  g.w.deeper.open(2); await settle();
  toggle(g.w).click(); await settle();
  g.counted.add(13); g.push(14); await settle(); g.w.clock.advance(1100); await settle();
  assert.deepEqual([g.w.deeper.snapshot().where.seq, g.w.deeper.snapshot().where.held, g.w.deeper.snapshot().where.snapped], [14, 'approx', false]);
  // No share at all (held "none") is never kept off for an exact one: it shows (its tooltips say why).
  const n = following({ 13: 'none' });
  n.w.deeper.open(2); await settle();
  toggle(n.w).click(); await settle();
  n.push(13); await settle(); n.w.clock.advance(1100); await settle();
  assert.deepEqual([n.w.deeper.snapshot().where.seq, n.w.deeper.snapshot().where.snapped], [13, false]);
  for (const text of w.deeper.texts()) assert.deepEqual(bannedWords(text), [], text);
});

test('on a tiled or whole run the words\' caveat stands only on a gene line with a word (the threshold is the word\'s); a line with none says nothing of it', async () => {
  const stats = whereReply({ seq: 12, cell: 1, sources: [1], genes: [], own: [] }).gene_stats.map(s => (s.gene === 'GXC' ? { ...s, word: null } : s));
  const w = make({ run: { constants: { mode: 'whole', kept: ['glow', 'where_words'] } }, where: params => whereReply(params, { stats }) });
  w.deeper.open(2); await settle();
  toggle(w).click(); await settle();
  const genes = w.root.all('p.qc-deeper-where-gene'), plain = genes.find(p => p.dataset.gene === 'GXC');
  assert.ok(plain && !plain.dataset.word, 'GXC: no word');
  assert.doesNotMatch(plain.title ?? '', /re-measured/, plain.title);
  assert.ok(genes.filter(p => p.dataset.word).every(p => p.title.endsWith(` ${DEEP_TEXT.deepNotRemeasured}`)));
  assert.ok(w.root.one('p.qc-deeper-glow-legend').title.endsWith(` ${DEEP_TEXT.deepNotRemeasured}`), 'the glow\'s own threshold');
});

test('at rest the dots\' frame is asked for itself: the run paused on a frame whose dots were kept from the counted frame before asks it again and shows its exact shares; paused before proseg acknowledges, an approximate answer is asked again once, when it does', async () => {
  const dots = s => s().all('p').some(p => /^Dots at iteration/.test(p.textContent));
  // Paused on frame 13 (a token counted it: no pause line, its answer unchanged), proseg's pause acknowledged at once.
  const held = { 13: 'approx' }, { w, push, status } = following(held);
  w.deeper.open(2); await settle();
  toggle(w).click(); await settle();
  push(13); await settle(); w.clock.advance(1100); await settle();
  assert.deepEqual([w.deeper.snapshot().where.seq, w.deeper.snapshot().where.snapped], [12, true]);
  held[13] = 'exact';   // paused there, the engine gives proseg's exact shares of frame 13
  Object.assign(w.ctx, { live: false }); Object.assign(w.ctx.run, { state: 'paused', ack: { state: 'paused', seq: 13 } });
  w.deeper.changed(); await settle();
  assert.deepEqual(w.calls.wheres.map(p => p.seq), [12, 13, 13], 'frame 13 asked again');
  let s = w.deeper.snapshot().where;
  assert.deepEqual([s.seq, s.held, s.snapped], [13, 'exact', false]); assert.equal(dots(status), false, 'no line kept from frame 12');
  w.deeper.changed(); w.clock.advance(5000); await settle();
  assert.equal(w.calls.wheres.length, 3, 'once');
  // Pausing first (proseg has not acknowledged yet): the frame asked for itself, its approximate answer shown as such
  // (never kept off at rest); asked once more when proseg acknowledges the pause there, then exact.
  const later = { 13: 'approx' }, f = following(later);
  f.w.deeper.open(2); await settle();
  toggle(f.w).click(); await settle();
  f.push(13); await settle(); f.w.clock.advance(1100); await settle();
  Object.assign(f.w.ctx, { live: false }); Object.assign(f.w.ctx.run, { state: 'paused', ack: { state: 'running', seq: 13 } });
  f.w.deeper.changed(); await settle();
  s = f.w.deeper.snapshot().where;
  assert.deepEqual([f.w.calls.wheres.map(p => p.seq), s.seq, s.held, s.snapped], [[12, 13, 13], 13, 'approx', false]);
  assert.equal(dots(f.status), false);
  later[13] = 'exact'; f.w.ctx.run.ack = { state: 'paused', seq: 13 };
  f.w.deeper.changed(); await settle();
  s = f.w.deeper.snapshot().where;
  assert.deepEqual([f.w.calls.wheres.map(p => p.seq), s.seq, s.held], [[12, 13, 13, 13], 13, 'exact']);
  // Paused and acknowledged while frame 13's answer was on its way (worked out as the run still played): it comes back
  // approximate and is asked again once.
  let release = null;
  const slowHeld = { 13: 'approx' };
  const late = make({ where: params => {
    const reply = { ...whereReply(params, { held: slowHeld[params.seq] ?? 'exact' }), iter: params.seq * 100 + 1 };
    return params.seq === 13 && !release ? new Promise(resolve => { release = () => resolve(reply); }) : reply;
  }, start: () => fail(404) });
  late.ctx.counted = seq => seq <= 13;
  late.deeper.open(2); await settle();
  toggle(late).click(); await settle();
  late.clock.advance(1100); late.ctx.seq = 13;
  late.deeper.event({ ev: 'deeper', run: 'r1', seq: 13, iter: 1301, cell: 1, model: 'm1', result: mixed({ seq: 13, iter: 1301 }) });
  late.deeper.changed(); late.batch.flush(); await settle(); late.clock.advance(1100); await settle();
  assert.ok(release, 'frame 13 asked while the run played');
  Object.assign(late.ctx, { live: false }); Object.assign(late.ctx.run, { state: 'paused', ack: { state: 'paused', seq: 13 } });
  late.deeper.changed(); await settle();
  slowHeld[13] = 'exact'; release(); await settle(); await settle();
  s = late.deeper.snapshot().where;
  assert.deepEqual([late.calls.wheres.map(p => p.seq), s.seq, s.held], [[12, 13, 13], 13, 'exact']);
  // An answer the engine still cannot give exactly is not asked again and again.
  const stuck = following({ 13: 'approx' });
  stuck.w.deeper.open(2); await settle();
  toggle(stuck.w).click(); await settle();
  stuck.push(13); await settle(); stuck.w.clock.advance(1100); await settle();
  Object.assign(stuck.w.ctx, { live: false }); Object.assign(stuck.w.ctx.run, { state: 'paused', ack: { state: 'paused', seq: 13 } });
  for (let k = 0; k < 4; k++) { stuck.w.deeper.changed(); stuck.w.clock.advance(2000); await settle(); }
  assert.deepEqual(stuck.w.calls.wheres.map(p => p.seq), [12, 13, 13], 'snapped, then once at rest');
  assert.deepEqual([stuck.w.deeper.snapshot().where.held, stuck.w.deeper.snapshot().where.snapped], ['approx', false]);
  // Scrubbed back to frame 13 on a paused run whose pause is at frame 20: nothing asked again for the scrub's sake.
  const back = following({ 13: 'approx' });
  back.w.deeper.open(2); await settle();
  toggle(back.w).click(); await settle();
  Object.assign(back.w.ctx, { live: false }); Object.assign(back.w.ctx.run, { state: 'paused', ack: { state: 'paused', seq: 20 } });
  back.push(13); await settle();
  for (let k = 0; k < 3; k++) { back.w.deeper.changed(); back.w.clock.advance(2000); await settle(); }
  assert.deepEqual([back.w.calls.wheres.map(p => p.seq), back.w.deeper.snapshot().where.held], [[12, 13], 'approx']);
  for (const text of [...w.deeper.texts(), ...f.w.deeper.texts()]) assert.deepEqual(bannedWords(text), [], text);
});

test('kept from the counted frame before only for the same request while the run plays: new rows, an earlier frame or another cell show the approximate answer', async () => {
  // The rows change between the exact answer and the approximate one (the cell's RNA from neighbours now comes from
  // ggaaaaaa-1): the approximate answer for the new rows shows, its glow, lit cells and words on the new neighbour.
  const { w, counted } = following({ 13: 'approx' });
  w.deeper.open(2); await settle();
  toggle(w).click(); await settle();
  const other = mixed({ seq: 13, iter: 1301 });
  other.neighbours = [{ cell: 4, um: 20, tx: 300, genes: [['GXPN', 31.1], ['GXC', 8.4]], kind: ['GXPN', 'GXDN', 'GXT'], similar: 0.31, known: true },
    ...mixed().neighbours.slice(1).filter(n => n.cell !== 4), { cell: 2, um: 25.2, tx: 0.4, genes: [['GYHNN', 0.2]], kind: ['GXHNN'], similar: 0.5, known: true }];
  counted.add(13); w.clock.advance(1100); w.ctx.seq = 13;
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 13, iter: 1301, cell: 1, model: 'm1', result: other });
  w.deeper.changed(); w.batch.flush(); await settle(); w.clock.advance(1100); await settle();
  const s = w.deeper.snapshot().where, rows = w.root.all('li.qc-deeper-source').map(li => li.dataset.cell);
  assert.deepEqual(w.calls.wheres.map(p => [p.seq, p.sources]), [[12, [2]], [13, [4]]]);
  assert.deepEqual([s.seq, s.held, s.snapped, s.sources, rows], [13, 'approx', false, ['ggaaaaaa-1'], ['ggaaaaaa-1']]);
  assert.deepEqual(s.lit, w.calls.focus.at(-1).lit, 'the cells the dots keep lit are the ones the focus lights');
  // The rule itself: live, the same request a frame on, no counted frame between; anything else shows the answer.
  const req = (seq, extra = {}) => ({ params: { seq, cell: 1, sources: [1], genes: ['GXPN'], own: ['GYHNN'], ...extra } });
  const exact = { held: 'exact', cell: 1, seq: 12 }, approx = (seq, cell = 1) => ({ held: 'approx', cell, seq });
  const g = make({ seq: 13 }); g.ctx.counted = () => false;
  assert.equal(g.deeper.snappedBy(exact, approx(13), req(12), req(13)), true);
  assert.equal(g.deeper.snappedBy(exact, approx(13), req(12), req(13, { sources: [3] })), false, 'other rows');
  assert.equal(g.deeper.snappedBy(exact, approx(13), req(12), req(13, { genes: ['GXC'] })), false, 'other genes');
  assert.equal(g.deeper.snappedBy(exact, approx(13), req(12), req(13, { own: [] })), false, 'other own genes');
  assert.equal(g.deeper.snappedBy({ ...exact, seq: 14 }, approx(13), req(14), req(13)), false, 'an earlier frame');
  assert.equal(g.deeper.snappedBy(exact, approx(13, 2), req(12), req(13, { cell: 2 })), false, 'another cell');
  assert.equal(g.deeper.snappedBy(exact, approx(13, 2), req(12), req(13)), false, 'another cell answering');
  assert.equal(g.deeper.snappedBy(exact, approx(13), null, req(13)), false, 'no request shown');
  assert.equal(g.deeper.snappedBy(exact, { ...approx(13), held: 'none' }, req(12), req(13)), false, 'no share at all');
  g.ctx.counted = seq => seq === 12;
  assert.equal(g.deeper.snappedBy(exact, approx(14), req(12), req(14)), true, 'the frame between has no count of its own');
  g.ctx.counted = seq => seq === 13;
  assert.equal(g.deeper.snappedBy(exact, approx(14), req(12), req(14)), false, 'a counted frame between');
  g.ctx.live = false;
  assert.equal(g.deeper.snappedBy(exact, approx(13), req(12), req(13)), false, 'never at rest');
});

test('on a tiled or whole run the glow\'s line and the words\' tooltips say their thresholds were not measured again for such runs; a region\'s say nothing', async () => {
  for (const [constants, said] of [[{ mode: 'tiled', kept: ['glow', 'where_words'] }, true], [{ mode: 'region', kept: [] }, false], [null, false]]) {
    const w = make({ run: constants ? { constants } : {} });
    w.deeper.open(2); await settle();
    toggle(w).click(); await settle();
    const legend = w.root.one('p.qc-deeper-glow-legend'), words = w.root.all('p.qc-deeper-where-gene').filter(p => p.dataset.word);
    assert.ok(words.length >= 2);
    assert.equal(legend.title.endsWith(` ${DEEP_TEXT.deepNotRemeasured}`), said, `${JSON.stringify(constants)}: ${legend.title}`);
    for (const p of words) assert.equal(p.title.endsWith(` ${DEEP_TEXT.deepNotRemeasured}`), said, p.title);
    assert.doesNotMatch(w.root.one('p.qc-deeper-where-into').title, /re-measured/, 'the into line has no threshold of these');
  }
});
