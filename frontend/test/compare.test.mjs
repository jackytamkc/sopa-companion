import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  CompareView, decodePoints, normaliseStatus, normaliseList, defaultColumns, fitLayout, placement, dotRadius, stamps, rasterise, stepOf, palettes,
  pointGrid, nearestPoint, fadeProgress, sizeText, binShares, COUNTERS, STATUS_MS, RETRY_MS, FAIL_TRIES, POINT_TRIES, MAX_COLUMNS, RECORD_BYTES, NO_KIND,
  FADE_MS, TICK_MS, FADE_FPS, GHOST_ALPHA, BIN_STEPS, MIX_STEPS, COLORS, STEPS, PLOT_PX, HOVER_PX, MISMATCH_TRIES, FIT_MARGIN, DOT_ALPHA, DIM_ALPHA,
  REPLACED_TEXT, COUNTS, decodeClusters, clusterSlots, gridCell, shareGrid, diffStep, differences, flyPoints, ease, FLY_PLACE, FLY_MS, FLY_FPS,
  FLY_TICK_MS, DIFF_COLORS, DIFF_SAME, DIFF_EDGES, NO_CLUSTER, MAPS, MAP_KEY,
} from '../src/compare.mjs';
import { COMPARE_TEXT as T, COMPARE_TEMPLATES, BIN_LABELS, compare, allTexts } from '../src/compare-copy.mjs';
import { bannedWords, number, QC_TEXT } from '../src/qc-copy.mjs';
import { MIXING_COLORS } from '../src/stage.js';
import { clusterColorCSS, CLUSTER_PALETTE } from '../src/clusters.js';

// Compare: the full-screen view of the run's cells
// and the sample's segmentations side by side. Its pure parts first (the deck, the points, the status, the scale, the
// dots), then the view in a small DOM with a virtual clock, a recording 2-D context (with pixels, so the dots can be
// read back) and a fake engine.

// ---- a small DOM (as qc-test-corner.test.mjs's, with a dialog) --------------------------------------------------------
function dom({ pixels = true } = {}) {
  const doc = { activeElement: null, hidden: false, listeners: new Map(), modals: 0 };
  doc.addEventListener = (t, fn) => doc.listeners.set(t, [...(doc.listeners.get(t) || []), fn]);
  doc.removeEventListener = (t, fn) => doc.listeners.set(t, (doc.listeners.get(t) || []).filter(f => f !== fn));
  doc.setHidden = hidden => { doc.hidden = hidden; for (const fn of doc.listeners.get('visibilitychange') || []) fn(); };
  const camel = s => s.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
  class Node {
    constructor(tag) {
      Object.assign(this, { tagName: tag.toUpperCase(), nodeType: tag === '#text' ? 3 : 1, childNodes: [], parentElement: null, attributes: new Map(),
        listeners: new Map(), style: {}, dataset: {}, hidden: false, _text: '', _html: '', className: '', ownerDocument: doc, title: '', value: '', disabled: false });
      const self = this, set = () => new Set(self.className.split(/\s+/).filter(Boolean));
      this.classList = {
        add: (...c) => { const s = set(); c.forEach(x => s.add(x)); self.className = [...s].join(' '); },
        remove: (...c) => { const s = set(); c.forEach(x => s.delete(x)); self.className = [...s].join(' '); },
        toggle: (c, on) => { const s = set(), want = on === undefined ? !s.has(c) : !!on; if (want) s.add(c); else s.delete(c); self.className = [...s].join(' '); return want; },
        contains: c => set().has(c),
      };
      if (tag === 'dialog') {
        this.open = false;
        this.showModal = () => { this.open = true; this.attributes.set('open', ''); doc.modals++; };
        this.close = () => { if (!this.open) return; this.open = false; this.attributes.delete('open'); this.emit('close'); };
      }
    }
    get children() { return this.childNodes.filter(n => n.nodeType === 1); }
    get textContent() { return this._text + this.childNodes.map(c => c.textContent).join(''); }
    set textContent(v) { this._text = String(v ?? ''); for (const c of this.childNodes) c.parentElement = null; this.childNodes = []; }
    set innerHTML(v) { this._html = String(v); this.textContent = ''; }
    get innerHTML() { return this._html; }
    append(...nodes) { for (let n of nodes) { if (typeof n !== 'object') n = doc.createTextNode(n); n.remove(); n.parentElement = this; this.childNodes.push(n); } }
    replaceChildren(...nodes) { for (const c of this.childNodes) c.parentElement = null; this.childNodes = []; this._text = ''; this.append(...nodes); }
    remove() { const p = this.parentElement; if (p) { p.childNodes = p.childNodes.filter(n => n !== this); this.parentElement = null; } }
    setAttribute(k, v) { this.attributes.set(k, String(v)); if (k === 'id') this.id = String(v); }
    getAttribute(k) { return k === 'title' ? this.title || null : this.attributes.has(k) ? this.attributes.get(k) : null; }
    removeAttribute(k) { this.attributes.delete(k); }
    addEventListener(t, fn) { this.listeners.set(t, [...(this.listeners.get(t) || []), fn]); }
    removeEventListener(t, fn) { this.listeners.set(t, (this.listeners.get(t) || []).filter(f => f !== fn)); }
    emit(t, extra = {}) {
      const e = { type: t, target: this, button: 0, preventDefault() { e.prevented = true; }, stopPropagation() { e.stopped = true; }, ...extra };
      this['on' + t]?.(e);
      for (const fn of this.listeners.get(t) || []) fn(e);
      return e;
    }
    click() { return this.onclick?.({ target: this, stopPropagation() {}, preventDefault() {} }); }
    contains(n) { while (n) { if (n === this) return true; n = n.parentElement; } return false; }
    closest(tag) { let n = this; while (n) { if (n.tagName === tag.toUpperCase()) return n; n = n.parentElement; } return null; }
    focus() { doc.activeElement = this; }
    getContext() { return this.ctx ??= context(pixels); }
    matches(sel) {
      const m = /^([a-z0-9]*)((?:#[\w-]+)?)((?:\.[\w-]+)*)((?:\[[^\]]+\])*)$/i.exec(sel.trim());
      if (!m) throw new Error(`selector ${sel}`);
      if (m[1] && this.tagName !== m[1].toUpperCase()) return false;
      if (m[2] && this.id !== m[2].slice(1)) return false;
      for (const c of m[3].match(/\.[\w-]+/g) || []) if (!this.classList.contains(c.slice(1))) return false;
      for (const a of m[4].match(/\[[^\]]+\]/g) || []) {
        const [k, v] = a.slice(1, -1).split('='), name = k.trim();
        const value = name.startsWith('data-') ? this.dataset[camel(name.slice(5))] : name === 'id' ? this.id : this.getAttribute(name);
        if (value == null) return false;
        if (v != null && String(value) !== v.replace(/^["']|["']$/g, '')) return false;
      }
      return true;
    }
    /** The nodes under this one matching `sel` (a compound selector, or several of them for descendants: ".a .b"). */
    all(sel) {
      const [first, ...rest] = sel.trim().split(/\s+(?![^[]*\])/);
      const under = (n, part) => { const out = []; const walk = m => { for (const c of m.children) { if (c.matches(part)) out.push(c); walk(c); } }; walk(n); return out; };
      let found = under(this, first);
      for (const part of rest) found = [...new Set(found.flatMap(n => under(n, part)))];
      return found;
    }
    one(sel) { return this.all(sel)[0] ?? null; }
    /** What a reader sees: hidden nodes left out. */
    visibleText() { return this.hidden ? '' : this._text + this.childNodes.map(c => c.nodeType === 3 ? c.textContent : c.visibleText()).join(' '); }
  }
  doc.createElement = tag => new Node(tag);
  doc.createTextNode = text => { const n = new Node('#text'); n._text = String(text); return n; };
  doc.body = new Node('body');
  const host = new Node('div'); doc.body.append(host);
  return { doc, host };
}
/** A 2-D context that records every call ([name, ...args]) and every property set (['=', name, value]); with `pixels` it
 *  hands out real pixel buffers (createImageData) and keeps each one put on it. */
function context(pixels) {
  const calls = [], images = [];
  const t = { calls, images };
  if (pixels) {
    t.createImageData = (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) });
    t.putImageData = (img, x, y) => { calls.push(['putImageData', img, x, y]); images.push(img); };
  }
  return new Proxy(t, {
    get: (o, k) => k in o ? o[k] : (...args) => { calls.push([k, ...args]); },
    set: (o, k, v) => { calls.push(['=', k, v]); return true; },
  });
}
/** A virtual clock: timers fire as time advances; animation frames run before the next timer. */
function clock() {
  const c = { t: 1000, timers: [], rafs: [], id: 0 };
  c.now = () => c.t;
  c.setTimeout = (fn, ms) => { const id = ++c.id; c.timers.push({ id, fn, at: c.t + Math.max(0, ms) }); return id; };
  c.clearTimeout = id => { c.timers = c.timers.filter(x => x.id !== id); };
  c.requestAnimationFrame = fn => { const id = ++c.id; c.rafs.push({ id, fn }); return id; };
  c.cancelAnimationFrame = id => { c.rafs = c.rafs.filter(x => x.id !== id); };
  c.advance = ms => {
    const end = c.t + ms;
    for (let guard = 0; guard < 100000; guard++) {
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
/** Advance `ms` in steps, letting the view's promises settle in between (as the browser would). */
async function run(s, ms, step = 25) { for (let t = 0; t < ms; t += step) { s.c.advance(Math.min(step, ms - t)); await settle(); } }

// ---- a sample: three segmentations and the nuclei, two kinds; small columns of points at known places ---------------------
const LIST = [
  { key: 'xenium_cell_boundaries', name: 'Xenium cells', source: 'xenium', model: null, count: 44691, extent: [8, 531, 10788, 10016], table: 'xenium_table' },
  { key: 'xenium_nucleus_boundaries', name: 'xenium_nucleus_boundaries', source: 'other', model: null, count: 44327, extent: [8, 531, 10788, 10016], table: null },
  { key: 'cell_boundaries', name: 'Earlier segmentation (imported)', source: 'imported', model: null, count: 42679, extent: [8, 527, 10618, 9379], table: 'table' },
  { key: 'cellpose_test', name: 'Cellpose', source: 'cellpose', model: 'cp4_20261021', count: 5378, extent: [8.1, 530.6, 2081.6, 1934.8], table: null },
];
const KINDS = [{ index: 0, name: 'GXF · GXLNN · GYENNN', color: '#4e9de0' }, { index: 1, name: 'GXWN · GYSNN · GZENN', color: '#e69f00' }];
const NAMES = { run: 'This run · iteration 450', xenium_cell_boundaries: 'Xenium cells', cell_boundaries: 'Earlier segmentation (imported)', cellpose_test: 'Cellpose',
  xenium_nucleus_boundaries: 'xenium_nucleus_boundaries' };
const FIRST = ['run', 'xenium_cell_boundaries', 'cell_boundaries'];
// Each column's points: [x, y, kind, share code] (share 80 = 32 %: looks mixed; 30 = 12 %: some; 3: little; 255: not judged).
const POINTS = {
  run: [[0, 0, 0, 3], [1, 0, 0, 30], [0, 1, 1, 80], [1, 1, 1, 3], [0.5, 0.5, 0, 3], [0.2, 0.8, 255, 3]],
  xenium_cell_boundaries: [[0, 0, 0, 3], [1, 0, 1, 30], [0.5, 1, 1, 80], [1, 1, 0, 3], [0.4, 0.4, 0, 3], [0.9, 0.2, 1, 3]],
  cell_boundaries: [[0.1, 0.1, 0, 3], [0.9, 0, 0, 30], [0, 0.9, 1, 80], [1, 1, 1, 3], [0.5, 0.6, 0, 3], [0.6, 0.3, 1, 3]],
  cellpose_test: [[0.2, 0.2, 0, 80], [0.8, 0.8, 1, 3], [0.5, 0.5, 0, 30]],
};
const HIST = { run: [2490, 314, 376, 615, 450, 545, 197], xenium_cell_boundaries: [20800, 2700, 3200, 5200, 3700, 4500, 1675],
  cell_boundaries: [20300, 2600, 3100, 5000, 3500, 4400, 1702], cellpose_test: [2400, 320, 380, 640, 460, 600, 237] };
const sum = a => a.reduce((x, y) => x + y, 0);
function encodePoints(list) {
  const b = Buffer.alloc(list.length * RECORD_BYTES);
  list.forEach(([x, y, kind, share], i) => { b.writeFloatLE(x, i * 10); b.writeFloatLE(y, i * 10 + 4); b[i * 10 + 8] = kind; b[i * 10 + 9] = share; });
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
// Which count the engine reads each column on (its `count`): the run's own on proseg's count, Xenium's cells on Xenium's,
// the rest by the transcripts inside their outlines.
const COUNTED = { run: 'proseg', xenium_cell_boundaries: 'xenium', cell_boundaries: 'outlines', cellpose_test: 'outlines', xenium_nucleus_boundaries: 'outlines' };
function column(key, extra = {}) {
  const hist = HIST[key];
  return { key, name: NAMES[key], state: 'ready', count: COUNTED[key], cells_in_area: sum(hist) + 389, judged: sum(hist), mixed_per100: 13.3, like100_per100: 11.2,
    histogram: hist, median_tx: 843, under100_share: 0.063, umap_points: POINTS[key].length, sampled: false, error: null, ...extra };
}
const pending = key => ({ key, name: NAMES[key], state: 'working', cells_in_area: null, judged: null, mixed_per100: null, like100_per100: null, histogram: null,
  median_tx: null, under100_share: null, umap_points: 0, sampled: false, error: null });
const AREA = { box: [8, 531, 2082, 1935], size_um: [2074, 1404], tiles: 211 };
function status(keys, { state = 'ready', kinds = KINDS, columns = null, step = null, progress = null, area = AREA } = {}) {
  return { state, step, progress, model: '6ccbac0d4a1dad54', area, kinds, columns: columns ?? keys.map(k => column(k)) };
}
/** A browser's storage in memory (the Map switch is remembered there). */
function memory(items = {}) {
  const m = new Map(Object.entries(items));
  return { m, getItem: k => (m.has(k) ? m.get(k) : null), setItem: (k, v) => { m.set(k, String(v)); } };
}
/** The view on a fake engine: `reply(n, keys)` the n-th status (keys: the columns last started); the points of the
 *  columns started, by index (`points(key, index, what)`: what 'own' or 'clusters' for a column's own map). `map`: the
 *  Map switch as this browser left it (the tests before Each its own read the one map). */
function setup({ reply = (n, keys) => status(keys), points = null, list = LIST, start = null, reduced = false, learn = null, pixels = true,
  map = 'one' } = {}) {
  const { doc, host } = dom({ pixels });
  const c = clock(), counts = {}, calls = { status: [], points: [], start: [], list: [], learn: [] }, motion = { reduced };
  const hook = { count(name, n = 1) { counts[name] = (counts[name] || 0) + n; } };
  const s = { doc, host, c, counts, calls, motion, keys: [], closed: 0, storage: memory(map ? { 'sopa-companion:compare:map': map } : {}) };
  s.view = new CompareView(host, {
    clock: c, hook, reducedMotion: () => motion.reduced, storage: s.storage,
    fetchStatus: run => { calls.status.push([run, c.t]); return Promise.resolve().then(() => reply(calls.status.length, s.keys)); },
    fetchPoints: (run, index, job, layout, what) => {
      calls.points.push(what ? [run, index, c.t, what] : [run, index, c.t]);
      return Promise.resolve().then(() => (points ? points(s.keys[index], index, what) : encodePoints(POINTS[s.keys[index]])));
    },
    start: (run, columns) => { calls.start.push([run, columns, c.t]); s.keys = columns; return start ? start(columns) : Promise.resolve({ state: 'working', columns }); },
    listSegmentations: sample => { calls.list.push(sample); return Promise.resolve(typeof list === 'function' ? list() : { segmentations: list }); },
    ...(learn ? { learnDetails: sample => { calls.learn.push(sample); return learn(sample); } } : {}),
    onClose: () => { s.closed++; },
  });
  s.root = s.view.root;
  s.open = (args = {}) => s.view.open({ run: 'r1', iteration: 450, sampleId: 'zd02871bc0e93', ...args });
  return s;
}
const visible = node => { for (let n = node; n; n = n.parentElement) if (n.hidden) return false; return true; };
const col = (s, key) => s.root.one(`[data-compare-column=${key}]`);
const ctxOf = (s, key) => col(s, key).one('canvas.compare-umap').getContext();
/** The pixel [r, g, b] at (x, y) of the last picture drawn for a column (its composite's pixels). */
function pixel(s, key, x, y) {
  const img = s.view.cols.get(key).comp.canvas.getContext().images.at(-1), o = (Math.round(y) * img.width + Math.round(x)) * 4;
  return [img.data[o], img.data[o + 1], img.data[o + 2]];
}
const rgb = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
const near = (a, b, tol = 4) => a.every((v, i) => Math.abs(v - b[i]) <= tol);
/** A dot's colour where it is drawn in full: `hex` at `alpha` over the background. */
const dot = (hex, alpha = DOT_ALPHA) => rgb(COLORS.background).map((v, i) => v + (rgb(hex)[i] - v) * alpha);
/** Where point `i` of column `key` sits on its canvas (device px). */
function at(s, key, i) {
  const c = s.view.cols.get(key), p = c.comp.place;
  return [p.ox + p.k * c.points.x[i], p.oy - p.k * c.points.y[i]];
}

// ---- the deck -------------------------------------------------------------------------------------------------------
// the mixing check's honesty rules on what the page may say, Go deeper's plain-word list, the no-area rule of
// 2026-10-22 and Compare's own: no direction words between columns, no "cell type".
const UNFIXABLE = new RegExp([
  String.raw`\b(?:no|none\s+of\s+the|not\s+any)\s+(?:\w+\s+){0,2}settings?\b[^.;:]{0,40}?\b(?:fix|solve|cure)`,
  String.raw`\bsettings?\s+(?:\w+\s+){0,2}(?:are|is|were|was|seem|seems)\s+(?:un|not\s+)likely\s+to\s+(?:fix|solve|help)`,
  String.raw`\b(?:can(?:no|')t|cannot|won'?t|will\s+not|could\s*n[o']t)\s+(?:be\s+)?(?:fix(?:ed)?|solved?|cured?)\b`,
  String.raw`\b(?:unfixable|not\s+fixable|beyond\s+(?:the\s+reach\s+of\s+)?(?:any\s+)?settings?)\b`,
  String.raw`\bnothing\s+(?:\w+\s+){0,3}(?:will|can|could|would)\s+(?:fix|solve)`, String.raw`\bnot\s+the\s+settings?\b`].join('|'), 'i');
// The method may be named (the mixing colours' tooltip says whose model it is); the user's notebooks never.
const NOTEBOOK = /\bnotebook\b|\bpart\s*3\b|\bsame\s+(?:settings|model|recipe)\b/i;
const STANDARDISED = /\badjusted\s+for\s+depth\b|\bdepth[- ]adjusted\b|\bstandardi[sz]ed\b/i;
const DEEPER_PLAIN = /\b(band|depth-matched|compactness|posterior|ambient RNA|contaminat)/i;
const AREAS = /(?:\b[Aa]reas?)\s+[A-Z](?![\w'’])|\b[Cc]ells\s+in\s+[A-Z](?![\w'’])|\bthis\s+area\s+is\b|\bnot\s+just\s+this\s+area\b|\bshows\s+this\s+area\b/i;
const DIRECTION = /\b(?:more|less|fewer|higher|lower|better|worse|increased?|decreased?)\b|[▲△↑⬆▼▽↓⬇→←]/i;
const CELL_TYPE = /\bcell[- ]?types?\b/i;
function plainWords(text, where) {
  assert.deepEqual(bannedWords(text), [], `${where}: ${text}`);
  for (const rule of [UNFIXABLE, NOTEBOOK, STANDARDISED, DEEPER_PLAIN, DIRECTION, CELL_TYPE]) assert.doesNotMatch(text, rule, `${where}: ${text}`);
  if (/[A-Z]/.test(text)) assert.doesNotMatch(text.replace(/(?<=\b[Aa]reas?\s)[a-z]/g, ''), /(?:\b[Aa]reas?)\s+[A-Z](?![\w'’])|\b[Cc]ells\s+in\s+[A-Z](?![\w'’])/, `${where}: ${text}`);
  assert.doesNotMatch(text, /\bthis\s+area\s+is\b|\bnot\s+just\s+this\s+area\b|\bshows\s+this\s+area\b/i, `${where}: ${text}`);
}
// Words that come as data (the engine's and the sample's names, keys, kinds), left out of the words checks.
const DATA = /xenium|cellpose|imported|Earlier segmentation|cell_boundaries|GXF|GXWN|This run · iteration|proseg \(this app\)|^[\d,.\s%–]+$|cp4_/i;

test('the deck: plain words under every rule; no word sets one column against another, no "cell type"; templates fail early', () => {
  for (const [key, text] of Object.entries({ ...T, ...COMPARE_TEMPLATES })) plainWords(text, key);
  for (const text of [...allTexts(), ...CompareView.allTexts(), ...BIN_LABELS]) plainWords(text, 'allTexts');
  assert.deepEqual(Object.keys(T).filter(k => k in COMPARE_TEMPLATES), [], 'a text and a template never share a key');
  assert.deepEqual(CompareView.allTexts(), allTexts());
  for (const [key, pattern] of Object.entries(COMPARE_TEMPLATES)) {
    const names = [...new Set([...pattern.matchAll(/\{([^}]+)\}/g)].map(m => m[1]))];
    const values = Object.fromEntries(names.map((n, i) => [n, `V${i}`]));
    assert.doesNotMatch(compare(key, values), /\{[^}]+\}|undefined|null/, key);
    for (const name of names) { const missing = { ...values }; delete missing[name]; assert.throws(() => compare(key, missing), TypeError, `${key}.${name}`); }
  }
  assert.throws(() => compare('nope'), RangeError);
  // Compare's words.
  assert.equal(compare('mixed', { n: 13 }), '13 in 100 cells look mixed');
  assert.equal(`13 ${T.mixedTail}`, compare('mixed', { n: 13 }));
  assert.equal(compare('like', { n: 11 }), 'At 100 transcripts each: 11 in 100');
  assert.equal(compare('areaAll', { size: '2.1 × 1.4 mm' }), 'On the area all of them cover: 2.1 × 1.4 mm');
  assert.equal(compare('runName', { i: 450 }), 'This run · iteration 450');
  assert.deepEqual(BIN_LABELS, ['0–2', '2–5', '5–10', '10–20', '20–30', '30–50', '50+']);
  // The tooltips carry the counting rule (the run's own: its quality, and the genes it left out when it did), the
  // transfer caveat, the like-for-like rule and the area's.
  // Each column on its own count (proseg's, Xenium's, else the transcripts inside its outlines), the bias of proseg's
  // count in one line, word for word.
  const counting = compare('counting', { q: 20 }), excluded = compare('countingExcluded', { q: 20 });
  assert.equal(counting, "Each column uses its own count: proseg's for proseg's cells, Xenium's for Xenium's cells, and for a segmentation with no count of its own each transcript of quality 20 or above, of the mixing model's genes, inside its outlines (in 2-D).");
  assert.match(excluded, /quality 20 or above, of the mixing model's genes except the ones this run left out, inside its outlines/);
  assert.equal(excluded.replace(" except the ones this run left out", ''), counting, 'the same rule, with the one clause');
  // H9: proseg's count named by whose it is: the run's column averaged or as saved, a table proseg saved as that table.
  assert.deepEqual([T.countProsegAveraged, T.countProsegSaved, T.countProsegTable, T.countXenium, T.countOutlines],
    ["Proseg's count (averaged)", "Proseg's count (as saved)", "Proseg's saved table", "Xenium's count", 'Transcripts inside its outlines']);
  assert.ok(!('countProseg' in T), 'one label for both, gone');
  assert.equal(T.countingBias, "Proseg's count sets background aside, and that alone takes some mixing out of its columns.");
  assert.deepEqual(COUNTS, { xenium: 'countXenium', outlines: 'countOutlines' });
  assert.deepEqual({ countProsegTooltip: T.countProsegTooltip, countProsegSavedTooltip: T.countProsegSavedTooltip, countProsegTableTooltip: T.countProsegTableTooltip,
    likeFromProseg: T.likeFromProseg, likeFromApp: T.likeFromApp }, {
    countProsegTooltip: "This column reads proseg's own count of each cell at this iteration, averaged over its random background draw.",
    countProsegSavedTooltip: "This column reads proseg's own count of each cell at this iteration as proseg saves it: one random background draw, not averaged.",
    countProsegTableTooltip: 'This column reads a table proseg saved in this sample: one random background draw of its count, as proseg saves it.',
    likeFromProseg: "Each cut starts from proseg's own draw at this iteration, the count it saves.",
    likeFromApp: "Proseg kept no draw at this iteration, so each cut starts from a random draw the app makes from proseg's count." });
  const words = { stepCounting: "Reading each column's counts…", columnCounting: 'Reading its counts…',
    statMedianTooltip: "The median of this column's transcripts per cell in the area, in its own count.",
    shownTooltip: 'The UMAPs hold the cells with at least 100 transcripts in their count in the area, 60,000 at the most in all: a column with too many for its share shows an even pick of them.',
    roughTooltip: 'Under 300 cells with at least 100 transcripts in their count in the area, so this number is rough.',
    histTooltip: "How the cells with at least 100 transcripts in their count spread by the share of their RNA the mixing model reads as from neighbours. The colours are the tissue's: under 10 %, 10–30 %, 30 % or above (looks mixed)." };
  for (const [key, text] of Object.entries(words)) assert.equal(T[key], text, key);
  assert.equal(compare('mixedTooltip', { j: '4,987' }), 'Of the 4,987 cells in the area with at least 100 transcripts in their count, those the mixing model reads as holding 30 % or above of their RNA from neighbours.');
  assert.match(compare('likeTooltip', { j: '4,987' }), /^Like for like: each of the 4,987 cells with at least 100 transcripts in their count cut down to exactly 100 of them/);
  assert.doesNotMatch([counting, T.statMedianTooltip, T.stepCounting].join(' '), /counted the same way|No table's own counts|counted as above/, 'no longer one count for every column');
  assert.match(T.transfer, /learned from one/); assert.match(T.transfer, /model rather than the outlines/);
  assert.match(T.sideBySide, /no verdict/);
  const like = compare('likeTooltip', { j: '4,987' });
  for (const want of [/^Like for like/, /exactly 100/, /three times/, /averaged/, /Levels change when cells are cut down/]) assert.match(like, want);
  assert.match(compare('areaTooltip', { n: 211 }), /211 squares of 100 µm that hold at least 3 cells of every column/);
  assert.match(compare('areaTooltip', { n: 211 }), /middle of its outline/);
  // The component has no words of its own: every sentence on screen is the deck's.
  const source = readFileSync(new URL('../src/compare.mjs', import.meta.url), 'utf8').replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gm, '');
  for (const text of [T.mixedTail, T.umapWaiting, T.tryAgain, T.addFull, T.kindsNeeded, 'look mixed', 'transcripts each']) assert.ok(!source.includes(text), text);
});

test('a segmentation the dormant InstanSeg saved earlier reads as a saved stain segmentation: no word of InstanSeg', () => {
  // InstanSeg is offered only behind the engine's internal switch; what it saved before is an ordinary column.
  assert.equal(T.sourceInstanseg, 'Stain segmentation');
  assert.deepEqual(allTexts().filter(text => /instanseg/i.test(text)), []);
  const { view } = setup();
  view.list = { ...view.list, items: [{ key: 'instanseg_boundaries_1', name: 'saved stain segmentation', source: 'instanseg',
    model: 'fluorescence_nuclei_and_cells' }] };
  assert.deepEqual(view.names('instanseg_boundaries_1', null), { name: 'saved stain segmentation',
    sub: 'instanseg_boundaries_1 · Stain segmentation fluorescence_nuclei_and_cells' });
});

// ---- pure helpers ---------------------------------------------------------------------------------------------------
test('points: little-endian records of 10 bytes (f32 x, f32 y, u8 kind, u8 share); anything else refused', () => {
  const list = [[1.5, -2.25, 3, 80], [1e6, 0.1, NO_KIND, 255], [NaN, 4, 0, 0]];
  const p = decodePoints(encodePoints(list));
  assert.equal(p.n, 3); assert.equal(RECORD_BYTES, 10);
  assert.deepEqual([...p.x].slice(0, 2), [1.5, 1e6]); assert.ok(Number.isNaN(p.x[2]));
  assert.deepEqual([...p.y], [-2.25, Math.fround(0.1), 4]); assert.deepEqual([...p.kind], [3, 255, 0]); assert.deepEqual([...p.share], [80, 255, 0]);
  // A view of a larger buffer (as a Buffer's slice is) reads the same; the wrong size, or not bytes, is refused.
  const bigger = Buffer.concat([Buffer.alloc(7), Buffer.from(encodePoints(list))]).subarray(7);
  assert.deepEqual([...decodePoints(bigger).share], [80, 255, 0]);
  assert.equal(decodePoints(encodePoints(list).slice(0, 25)), null);
  for (const bad of [null, 'x', { length: 10 }, [1, 2]]) assert.equal(decodePoints(bad), null);
  assert.equal(decodePoints(new ArrayBuffer(0)).n, 0);
});

test('the status is checked: lenient, a column without a key (or twice) left out, the histogram 7 counts, every kind coloured', () => {
  for (const bad of [null, 'x', [], 7]) assert.equal(normaliseStatus(bad), null);
  const s = normaliseStatus({ state: 'odd', progress: { done: 9, of: 4 }, model: 12, area: { box: [0, 0, 2000, 1000], tiles: 12.4 },
    kinds: [{ index: 0, name: ' A · B ', color: '#ABCDEF' }, { index: 3, name: 'C' }, { index: 255, name: 'none' }, { name: 'x' }],
    columns: [column('run'), { ...column('run'), mixed_per100: 99 }, { name: 'no key' }, { ...column('cell_boundaries'), histogram: [1, 2, 3], under100_share: 2 }] });
  assert.equal(s.state, 'working', 'an unknown state reads as working'); assert.deepEqual(s.progress, { done: 4, of: 4 }); assert.equal(s.model, '12');
  assert.deepEqual(s.area, { box: [0, 0, 2000, 1000], size_um: [2000, 1000], tiles: 12 }, 'the size from the box');
  assert.deepEqual(s.kinds, [{ index: 0, name: 'A · B', color: '#abcdef' }, { index: 3, name: 'C', color: clusterColorCSS(3) }]);
  assert.deepEqual(s.columns.map(c => c.key), ['run', 'cell_boundaries']); assert.equal(s.columns[0].mixed, 13.3, 'the first of a key');
  assert.equal(s.columns[1].histogram, null, 'not 7 counts: none'); assert.equal(s.columns[1].under100, 1);
  assert.deepEqual(Object.keys(s.columns[0]).sort(), ['cells', 'clusters', 'count', 'error', 'form', 'histogram', 'judged', 'key', 'like100', 'likeCells', 'likeFrom',
    'matched', 'medianTx', 'mixed', 'name', 'own', 'points', 'sampled', 'state', 'tiles', 'under100']);
  // Each its own: whether a column has its own map and its clusters (how many, how many match the yardstick's); the own
  // maps' view box and the engine's words for them.
  assert.deepEqual([s.columns[0].own, s.columns[0].clusters, s.columns[0].matched], [false, 0, 0]);
  const own = normaliseStatus({ own_layout: 'umap', own_bounds: { x: [-1, 2], y: [0, 3] }, own_error: null,
    columns: [{ ...column('run'), own: true, clusters: 12.2, clusters_matched: 9 }, { ...column('cell_boundaries'), own: 'yes' }] });
  assert.deepEqual([own.ownLayout, own.ownBounds, own.ownError], ['umap', [-1, 0, 2, 3], null]);
  assert.deepEqual(own.columns.map(c => [c.own, c.clusters, c.matched]), [[true, 12, 9], [false, 0, 0]]);
  assert.deepEqual([s.ownLayout, s.ownBounds, s.ownError], [null, null, null]);
  assert.equal(normaliseStatus({ own_error: 'too_few_cells', own_bounds: [3, 3, 1, 1] }).ownBounds, null);
  // The run column's form (which of proseg's counts it reads) and whose draw its cut starts from: the engine's words the
  // page knows, else none.
  assert.deepEqual(['sampled', 'averaged', 'other', null].map(form => normaliseStatus({ columns: [{ ...column('run'), form }] }).columns[0].form),
    ['sampled', 'averaged', null, null]);
  assert.deepEqual(['proseg', 'app', 'other', undefined].map(like_from => normaliseStatus({ columns: [{ ...column('run'), like_from }] }).columns[0].likeFrom),
    ['proseg', 'app', null, null]);
  assert.deepEqual([s.columns[0].likeCells, normaliseStatus({ columns: [{ ...column('run'), like100_cells: 2986.4 }] }).columns[0].likeCells], [null, 2986],
    'the cells the like-for-like line is over, when the engine says');
  // Which count a column is read on: the engine's word when the page knows it, else none.
  assert.deepEqual(['proseg', 'xenium', 'outlines', 'other', null, 7, 'toString'].map(count => normaliseStatus({ columns: [{ ...column('run'), count }] }).columns[0].count),
    ['proseg', 'xenium', 'outlines', null, null, null, null]);
  assert.equal(s.columns[0].tiles, null, 'the squares a column covers: none before the engine has found the area');
  assert.equal(normaliseStatus({ columns: [{ ...column('run'), tiles: 211.2 }] }).columns[0].tiles, 211);
  assert.equal(normaliseStatus({ kinds: null }).kinds, null, 'no details learned yet');
  assert.equal(normaliseStatus({ area: { box: [1, 2, 1, 5] } }).area, null);
  assert.equal(normaliseStatus({ columns: [{ ...pending('run'), state: 'queued' }] }).columns[0].state, 'queued');
});

test('the segmentations and the first columns: the run, Xenium\'s cells (never the nuclei) and the imported one', () => {
  const items = normaliseList({ segmentations: [...LIST, { key: 'run' }, { key: 'cellpose_test' }, { name: 'no key' }] });
  assert.deepEqual(items.map(i => i.key), LIST.map(i => i.key), "the run's own key and a second one left out");
  assert.deepEqual(items[3], { key: 'cellpose_test', name: 'Cellpose', source: 'cellpose', model: 'cp4_20261021', count: 5378, extent: [8.1, 530.6, 2081.6, 1934.8], table: null });
  assert.deepEqual(normaliseList(LIST).length, 4); assert.equal(normaliseList({}), null); assert.equal(normaliseList(null), null);
  assert.deepEqual(defaultColumns(items), FIRST);
  assert.deepEqual(defaultColumns(items.filter(i => i.source !== 'imported')), ['run', 'xenium_cell_boundaries']);
  const nuclei = [{ key: 'xenium_nucleus_boundaries', source: 'xenium', table: null }, { key: 'xenium_cells', source: 'xenium', table: null }];
  assert.deepEqual(defaultColumns(nuclei), ['run', 'xenium_cells'], 'the cells, not the nuclei');
  assert.deepEqual(defaultColumns([]), ['run']); assert.deepEqual(defaultColumns(null), ['run']);
});

test('one scale for every column: 98 % of the points with a margin, x and y alike, y up; dots the same size in every column', () => {
  const a = { n: 101, x: Float32Array.from({ length: 101 }, (_, i) => i), y: Float32Array.from({ length: 101 }, (_, i) => i / 10) };
  const far = { n: 1, x: Float32Array.of(1e6), y: Float32Array.of(NaN) };
  const b = fitLayout([a, far]);
  assert.ok(Math.abs(b[0] - (1 - 98 * FIT_MARGIN)) < 1e-6 && Math.abs(b[2] - (99 + 98 * FIT_MARGIN)) < 1e-6, `x ${b}`);
  assert.equal(fitLayout([]), null); assert.equal(fitLayout([{ n: 1, x: Float32Array.of(NaN), y: Float32Array.of(0) }]), null);
  const p = placement([0, 0, 10, 5], 220, 220, 10);
  assert.equal(p.k, 20, 'the wider side fills the canvas less its padding');
  assert.deepEqual([p.ox + p.k * 5, p.oy - p.k * 2.5], [110, 110], 'the middle in the middle');
  assert.ok(p.oy - p.k * 5 < p.oy - p.k * 0, 'y up');
  assert.equal(dotRadius(15000, 445, 520), 1.1); assert.ok(dotRadius(1000, 445, 520) > 2); assert.equal(dotRadius(1, 2000, 2000), 2.4);
});

test('dots: soft discs written into the pixels in their colours, over what is there; the lit kind over the dimmed others', () => {
  const w = 40, h = 30, px = new Uint8ClampedArray(w * h * 4);
  for (let o = 0; o < px.length; o += 4) px[o + 3] = 255;
  const pts = { n: 4, x: Float32Array.of(10, 30, NaN, 500), y: Float32Array.of(10, 20, 5, 5), kind: Uint8Array.of(0, 1, 0, 0), share: Uint8Array.of(0, 0, 0, 0) };
  const place = { k: 1, ox: 0, oy: 30, w, h }, pal = palettes(KINDS);
  assert.equal(rasterise(px, w, h, pts, place, pts.kind, pal.kind, { r: 2, alpha: 1 }), 2, 'NaN and off the canvas: not drawn');
  const read = (x, y) => [...px.subarray((y * w + x) * 4, (y * w + x) * 4 + 3)];
  assert.deepEqual(read(10, 19), rgb(KINDS[0].color), 'the middle in full (y up: 30 - 10 = 20, the pixel row above)');
  assert.deepEqual(read(30, 9), rgb(KINDS[1].color));
  assert.deepEqual(read(0, 0), [0, 0, 0], 'the rest untouched');
  const edge = read(12, 19); assert.ok(edge[2] > 0 && edge[2] < rgb(KINDS[0].color)[2], 'a soft edge');
  // A kind lit: the others at DIM, the lit ones over them.
  const both = { n: 2, x: Float32Array.of(10, 10), y: Float32Array.of(10, 10), kind: Uint8Array.of(1, 0), share: Uint8Array.of(0, 0) };
  const q = new Uint8ClampedArray(w * h * 4);
  rasterise(q, w, h, both, place, both.kind, pal.kind, { r: 2, alpha: 1, lit: Uint8Array.of(0, 1), dim: 0.1 });
  assert.deepEqual([...q.subarray((19 * w + 10) * 4, (19 * w + 10) * 4 + 3)], rgb(KINDS[0].color), 'the lit kind on top whatever the order');
  // The stamps: SUB x SUB places, each a disc of about pi r^2 coverage.
  const st = stamps(2);
  assert.equal(st.length, 16);
  for (const s of st) { let cover = 0; for (let e = 2; e < s.length; e += 3) cover += s[e]; assert.ok(Math.abs(cover - Math.PI * 4) < 1.5, `${cover}`); }
  assert.equal(stamps(2), st, 'made once');
});

test('the steps and the palettes: the stage\'s three colours by share, the kinds\' by index, grey for no kind', () => {
  assert.deepEqual([0, 24, 25, 74, 75, 249, 255].map(stepOf), [0, 0, 1, 1, 2, 2, 3]);
  const { kind, mixing } = palettes(KINDS);
  assert.deepEqual(MIX_STEPS.map((s, i) => [...mixing.subarray(i * 3, i * 3 + 3)]), MIX_STEPS.map(s => rgb(MIXING_COLORS[s])));
  assert.deepEqual([...kind.subarray(0, 3)], rgb(KINDS[0].color)); assert.deepEqual([...kind.subarray(3, 6)], rgb(KINDS[1].color));
  assert.deepEqual([...kind.subarray(NO_KIND * 3, NO_KIND * 3 + 3)], rgb(COLORS.none)); assert.deepEqual([...kind.subarray(6, 9)], rgb(COLORS.none));
  assert.deepEqual(BIN_STEPS, ['little', 'little', 'little', 'some', 'some', 'mixed', 'mixed'], 'under 10 %, 10-30 %, 30 % or above');
  assert.deepEqual(sizeText([2074, 1404]), '2.1 × 1.4 mm'); assert.equal(sizeText([640, 410.4]), '640 × 410 µm'); assert.equal(sizeText(null), T.missing);
  assert.deepEqual(binShares([1, 3]), [0.25, 0.75]); assert.equal(binShares([0, 0]), null); assert.equal(binShares(null), null);
  assert.ok(fadeProgress(0) > 0 && fadeProgress(0) < 1); assert.equal(fadeProgress(FADE_MS), 1); assert.equal(TICK_MS, 84); assert.equal(FADE_FPS, 12);
});

test('the nearest dot: through a grid, within the reach only', () => {
  const pts = { n: 3, x: Float32Array.of(1, 2, 10), y: Float32Array.of(1, 1, 10), kind: new Uint8Array(3), share: new Uint8Array(3) };
  const place = placement([0, 0, 12, 12], 120, 120, 0), grid = pointGrid(pts, place, 12);
  const [x, y] = [place.ox + place.k * 2, place.oy - place.k * 1];
  assert.equal(nearestPoint(pts, place, grid, x + 2, y, 7), 1);
  assert.equal(nearestPoint(pts, place, grid, x - 6, y, 7), 0, 'the nearer one');
  assert.equal(nearestPoint(pts, place, grid, 60, 60, 7), null, 'none within reach');
});

// ---- the view ---------------------------------------------------------------------------------------------------------
test('opening: the sample\'s segmentations listed, the first columns started, each column filled as its numbers come, then the UMAPs', async () => {
  // The engine: counting (the run's column first), then every column's numbers, then the layout's points.
  const replies = [
    keys => status(keys, { state: 'working', step: 'counting', progress: { done: 1, of: 5 }, kinds: null, columns: [column('run', { umap_points: 0 }), ...keys.slice(1).map(pending)] }),
    keys => status(keys, { state: 'working', step: 'umap', progress: { done: 4, of: 5 }, kinds: KINDS, columns: keys.map(k => column(k, { umap_points: 0 })) }),
    keys => status(keys),
  ];
  const s = setup({ reply: (n, keys) => replies[Math.min(n, replies.length) - 1](keys) });
  assert.equal(s.root.tagName, 'DIALOG'); assert.equal(s.root.id, 'compare-view'); assert.equal(s.root.open, false); assert.equal(s.root.dataset.state, 'closed');
  assert.ok(s.open());
  assert.equal(s.root.open, true, 'shown as a modal dialog'); assert.equal(s.doc.modals, 1);
  assert.ok(s.root.matches('dialog#compare-view[open][data-state=starting]'));
  assert.equal(s.root.dataset.qcCopy, '', "the harness's copy check reads it");
  await run(s, 10);
  assert.deepEqual(s.calls.list, ['zd02871bc0e93']);
  assert.deepEqual(s.calls.start.map(c => c.slice(0, 2)), [['r1', FIRST]], 'the run, Xenium\'s cells and the imported one');
  assert.deepEqual(s.root.all('[data-compare-column]').map(n => n.dataset.compareColumn), FIRST);
  // The first status: the run's column has its numbers; the others still work.
  assert.equal(s.calls.status.length, 1);
  assert.equal(s.root.dataset.state, 'working');
  const run1 = col(s, 'run');
  assert.match(run1.one('.cmp-mixed').visibleText(), /^13 +in 100 cells look mixed/);
  assert.equal(run1.one('[data-mixed]').dataset.mixed, '13.3'); assert.equal(run1.one('[data-like100]').dataset.like100, '11.2');
  assert.match(run1.one('.cmp-like').visibleText(), /At 100 transcripts each: 11 in 100/);
  assert.equal(run1.one('.cmp-like .cmp-info').title, compare('likeTooltip', { j: number(sum(HIST.run)) }), 'the cells judged: the engine names no other');
  assert.equal(run1.one('[data-hist]').dataset.hist, HIST.run.join(','));
  assert.match(col(s, 'xenium_cell_boundaries').one('.cmp-mixed').visibleText(), /Working…/);
  assert.equal(col(s, 'xenium_cell_boundaries').one('[data-mixed]').dataset.mixed, '');
  assert.equal(s.root.one('.cmp-progress').textContent, "Reading each column's counts… 1 of 5");
  assert.match(s.root.one('[data-compare-area]').visibleText(), /On the area all of them cover: 2\.1 × 1\.4 mm/);
  assert.match(s.root.one('.cmp-area .cmp-info').title, /211 squares/);
  // The names: the engine's, the key and what made it; the run's.
  assert.equal(run1.one('.cmp-col-title').textContent, 'This run · iteration 450'); assert.equal(run1.one('.cmp-col-sub').textContent, T.sourceProseg);
  assert.equal(col(s, 'cell_boundaries').one('.cmp-col-sub').textContent, `cell_boundaries · ${T.sourceImported}`);
  // Every column's numbers, then the layout: the points asked for once per column, by index, and drawn at one scale.
  await run(s, STATUS_MS);
  assert.equal(s.calls.status.length, 2); assert.equal(s.calls.points.length, 0, 'no points before the layout');
  assert.equal(s.root.one('.cmp-progress').textContent, 'Laying out the UMAPs… 4 of 5');
  // The engine's step without counts, an unknown step: its words alone, else "Working…" (never the engine's own words).
  s.view.status = { ...s.view.status, progress: null }; assert.equal(s.view.progressText(), T.stepUmap);
  s.view.status = { ...s.view.status, step: 'gpu_warmup' }; assert.equal(s.view.progressText(), T.working);
  await run(s, STATUS_MS + 50);
  assert.equal(s.calls.status.length, 3); assert.equal(s.root.dataset.state, 'ready');
  assert.deepEqual(s.calls.points.map(c => c[1]).sort(), [0, 1, 2]);
  assert.equal(s.root.one('.cmp-progress').textContent, '', 'nothing said once done');
  const snap = s.view.snapshot();
  assert.deepEqual(snap.columns.map(c => c.points), [6, 6, 6]);
  assert.deepEqual(snap.columns.map(c => c.state), ['ready', 'ready', 'ready']);
  const places = FIRST.map(k => s.view.cols.get(k).comp.place);
  assert.ok(places.every(p => p.k === places[0].k && p.ox === places[0].ox && p.oy === places[0].oy), 'one scale for every column');
  for (const key of FIRST) {
    const c = col(s, key);
    assert.ok(c.one('canvas.compare-umap') && c.one('[data-mixed]') && c.one('[data-like100]') && c.one('[data-hist]'), `${key}: its hooks`);
    assert.equal(visible(c.one('.cmp-wait')), false, `${key}: drawn`);
    assert.equal(c.dataset.points, '6');
  }
  // The dots in the kinds' colours (by kind: the model has kinds), each where the shared scale puts it.
  const [x, y] = at(s, 'run', 2);
  assert.ok(near(pixel(s, 'run', x, y), dot(KINDS[1].color)), `${pixel(s, 'run', x, y)}`);
  assert.equal(s.root.dataset.colourBy, 'kind');
  // Nothing more is asked once the engine is done.
  await run(s, 3000, 100);
  assert.equal(s.calls.status.length, 3); assert.equal(s.calls.points.length, 3);
});

test('the status is asked for at most twice a second while the engine works, one at a time; never hidden, closed or once done', async () => {
  let done = false;
  const s = setup({ reply: (n, keys) => status(keys, done ? {} : { state: 'working', columns: keys.map(pending) }) });
  s.open(); await run(s, 3000);
  const times = s.calls.status.map(a => a[1]);
  assert.ok(times.length >= 6 && times.length <= 7, `${times.length} asks in 3 s`);
  for (const t of times) assert.ok(times.filter(u => u >= t && u < t + 1000).length <= 2, 'never more than 2 in a second');
  assert.equal(s.counts.compareFetches, 1 + times.length, 'the start and every ask');
  // Hidden: none, nothing scheduled; shown again: asked again.
  s.doc.setHidden(true); const hiddenAt = s.calls.status.length; await run(s, 2000);
  assert.equal(s.calls.status.length, hiddenAt); assert.equal(s.c.timers.length, 0, 'nothing scheduled while hidden'); assert.equal(s.c.rafs.length, 0);
  s.doc.setHidden(false); await run(s, 600); assert.ok(s.calls.status.length > hiddenAt);
  // Done: no more asks.
  done = true; await run(s, 600); const at = s.calls.status.length; await run(s, 3000);
  assert.equal(s.calls.status.length, at); assert.equal(s.root.dataset.state, 'ready');
  // A slow answer holds the next ask.
  let release = null, asked = 0;
  const slow = setup({ reply: () => new Promise(r => { asked++; release = r; }) });
  slow.open(); await run(slow, 3000);
  assert.equal(asked, 1, 'one ask at a time');
  release(status(slow.keys, { state: 'working', columns: slow.keys.map(pending) })); await run(slow, 600); assert.equal(asked, 2);
  // Closed: never; an answer that comes after changes nothing.
  slow.view.close(); release(status(slow.keys)); await run(slow, 3000);
  assert.equal(asked, 2); assert.equal(slow.c.timers.length, 0); assert.equal(slow.c.rafs.length, 0);
  assert.equal(slow.view.snapshot().open, false); assert.deepEqual(slow.view.snapshot().columns, []);
});

test('an ask that fails is made again a little later, twice as long each time; then the view says so, and Try again starts again', async () => {
  const s = setup({ reply: () => { throw new Error('the engine is gone'); } });
  s.open(); await run(s, 30000, 50);
  const times = s.calls.status.map(a => a[1]);
  assert.equal(times.length, FAIL_TRIES, 'FAIL_TRIES asks, then none');
  times.slice(1).forEach((t, i) => { const gap = t - times[i], want = RETRY_MS * 2 ** i; assert.ok(gap >= want && gap <= want + 50, `retry ${i + 1}: ${gap} ms`); });
  assert.equal(s.c.timers.length, 0, 'never a loop');
  const alert = s.root.one('.cmp-alert');
  assert.equal(visible(alert), true); assert.equal(alert.one('.cmp-alert-text').textContent, T.unreachable);
  const retry = alert.one('button[data-action=retry]');
  assert.equal(visible(retry), true); assert.equal(retry.disabled, false);
  assert.equal(s.root.one('.cmp-progress').textContent, '');
  retry.click(); await run(s, 10);
  assert.equal(s.calls.start.length, 2, 'Try again starts the compare again (the engine reuses it)'); assert.equal(s.calls.status.length, FAIL_TRIES + 1);
  assert.equal(visible(alert), false);
  // The columns the engine took (its reply names them: here without the nuclei it can't count) are the view's.
  const counted = keys => keys.filter(k => !k.includes('nucleus'));
  const took = setup({ start: columns => Promise.resolve({ state: 'working', columns: counted(columns).map(key => ({ key })) }),
    reply: (n, keys) => status(counted(keys)), points: (key, index) => encodePoints(POINTS[counted(took.keys)[index]]) });
  took.view.chosen.set('zd02871bc0e93', ['run', 'xenium_nucleus_boundaries', 'cell_boundaries']);
  took.open(); await run(took, 600, 50);
  assert.deepEqual(took.root.all('[data-compare-column]').map(n => n.dataset.compareColumn), ['run', 'cell_boundaries']);
  assert.equal(took.root.dataset.state, 'ready'); assert.deepEqual(took.view.chosen.get('zd02871bc0e93'), ['run', 'cell_boundaries']);
  // A status of another set of columns (the engine still on the one before) is not shown, and is not waited on forever.
  const other = setup({ reply: () => status(['run', 'cellpose_test']) });
  other.open(); await run(other, STATUS_MS * (MISMATCH_TRIES + 2));
  assert.equal(other.calls.status.length, MISMATCH_TRIES); assert.equal(other.root.dataset.state, 'failed');
  assert.equal(col(other, 'run').one('[data-mixed]').dataset.mixed, '', 'never another compare\'s number');
});

test('a compare that could not start says why with Try again; a column that failed says why; no button is ever greyed out', async () => {
  let refuse = true;
  const s = setup({
    start: columns => (refuse ? Promise.reject(new Error('The mixing check is off for this run')) : Promise.resolve({ state: 'working', columns })),
    reply: (n, keys) => status(keys, { columns: keys.map(k => (k === 'cell_boundaries' ? { ...pending(k), state: 'failed', error: "Its outlines couldn't be read." } : column(k))) }),
  });
  s.open(); await run(s, 50);
  assert.equal(s.root.dataset.state, 'failed'); assert.equal(s.calls.status.length, 0);
  assert.equal(s.root.one('.cmp-alert-text').textContent, "Couldn't compare: The mixing check is off for this run");
  assert.equal(visible(col(s, 'run').one('.cmp-mixed')), false, 'the columns quiet: the alert says it once');
  refuse = false; s.root.one('button[data-action=retry]').click(); await run(s, 600);
  assert.equal(s.root.dataset.state, 'ready'); assert.equal(visible(s.root.one('.cmp-alert')), false);
  const failed = col(s, 'cell_boundaries');
  assert.equal(failed.dataset.state, 'failed');
  assert.equal(failed.one('.cmp-col-error').textContent, "Its outlines couldn't be read."); assert.equal(visible(failed.one('.cmp-col-error')), true);
  assert.equal(visible(failed.one('.cmp-stats')), false); assert.equal(visible(failed.one('.cmp-wait')), false);
  assert.equal(s.view.snapshot().columns.find(c => c.key === 'cell_boundaries').state, 'failed');
  // Without words of its own, the column says it could not be worked out.
  const quiet = setup({ reply: (n, keys) => status(keys, { columns: keys.map(k => (k === 'run' ? { ...pending(k), state: 'failed' } : column(k))) }) });
  quiet.open(); await run(quiet, 50);
  assert.equal(col(quiet, 'run').one('.cmp-col-error').textContent, T.columnFailed);
  for (const b of [...s.root.all('button'), ...quiet.root.all('button')]) assert.equal(b.disabled, false, b.dataset.action);
  assert.doesNotMatch(readFileSync(new URL('../src/compare.mjs', import.meta.url), 'utf8'), /\.disabled\s*=/, 'never greyed out');
});

test('the histogram: the stage\'s three colours, one scale for every column, each bin\'s share in its tooltip; its bins sum to the judged', async () => {
  const s = setup({ reply: (n, keys) => status(keys, { columns: keys.map(k => column(k, k === 'cell_boundaries' ? { histogram: [10, 0, 0, 0, 0, 0, 90], judged: 100 } : {})) }) });
  s.open(); await run(s, 50);
  const tallest = Math.max(...FIRST.flatMap(k => { const h = k === 'cell_boundaries' ? [10, 0, 0, 0, 0, 0, 90] : HIST[k]; return h.map(v => v / sum(h)); }));
  for (const key of FIRST) {
    const c = col(s, key), hist = key === 'cell_boundaries' ? [10, 0, 0, 0, 0, 0, 90] : HIST[key], bins = c.all('.cmp-bin');
    assert.equal(bins.length, 7);
    assert.equal(sum(c.one('[data-hist]').dataset.hist.split(',').map(Number)), Number(c.dataset.judged), `${key}: the bins sum to the judged cells`);
    bins.forEach((bin, b) => {
      const fill = bin.children[0];
      assert.equal(fill.style.background, MIXING_COLORS[BIN_STEPS[b]], `${key} bin ${b}: the stage's colour`);
      assert.equal(fill.style.height, `${((hist[b] / sum(hist) / tallest) * 100).toFixed(1)}%`);
      assert.equal(bin.dataset.count, String(hist[b]));
    });
    assert.deepEqual(c.all('.cmp-ticks span').map(n => n.textContent), BIN_LABELS);
  }
  assert.equal(col(s, 'cell_boundaries').all('.cmp-bin')[6].children[0].style.height, '100.0%', 'the tallest bin of any column fills its height');
  assert.equal(col(s, 'run').all('.cmp-bin')[3].title, `10–20 %: ${Math.round((615 / sum(HIST.run)) * 100)} in 100 of these cells (615)`);
  // The stats: cells in the area, the median, the share under 100 (whole numbers, as on the panel).
  assert.deepEqual(col(s, 'run').all('.cmp-stat dd').map(n => n.textContent), [String(sum(HIST.run) + 389).replace(/\B(?=(\d{3})+$)/g, ','), '843', '6 %']);
  // Under 300 judged, rough; under 30, no number.
  const few = setup({ reply: (n, keys) => status(keys, { columns: keys.map((k, i) => column(k, i === 1 ? { judged: 120 } : i === 2 ? { judged: 12 } : {})) }) });
  few.open(); await run(few, 50);
  assert.equal(visible(col(few, 'xenium_cell_boundaries').one('.cmp-rough')), true); assert.equal(visible(col(few, 'run').one('.cmp-rough')), false);
  assert.match(col(few, 'cell_boundaries').one('.cmp-mixed').visibleText(), /Too few cells to say \(12 with at least 100 transcripts\)/);
  assert.equal(visible(col(few, 'cell_boundaries').one('.cmp-like')), false);
});

test('the layout comes in once: one cross-fade of at most 12 paints a second, then nothing while still; one paint under reduced motion', async () => {
  const s = setup();
  s.open(); await run(s, 30);
  const d0 = s.counts.compareDraws;
  assert.ok(s.view.snapshot().fading >= 1 || s.counts.compareDraws > 0);
  await run(s, FADE_MS + 200, 10);
  const fadePaints = s.counts.compareDraws - d0;
  assert.ok(fadePaints >= 3 * 3 && fadePaints <= 3 * (Math.ceil(FADE_MS / TICK_MS) + 2), `${fadePaints} paints for 3 columns`);
  assert.equal(s.view.snapshot().fading, 0, 'finite');
  const ctx = ctxOf(s, 'run'), comp = s.view.cols.get('run').comp.canvas;
  const alphas = ctx.calls.filter((c, i) => c[0] === '=' && c[1] === 'globalAlpha' && ctx.calls[i + 1]?.[0] === 'drawImage' && ctx.calls[i + 1][1] === comp).map(c => c[2]);
  assert.ok(alphas.some(a => a > 0 && a < 1) && alphas.at(-1) === 1, `faded in: ${alphas}`);
  // Still: nothing is painted or scheduled.
  const d1 = s.counts.compareDraws; await run(s, 3000, 100);
  assert.equal(s.counts.compareDraws, d1); assert.equal(s.c.timers.length, 0); assert.equal(s.c.rafs.length, 0);
  // Reduced motion: one paint per column for the layout.
  const r = setup({ reduced: true });
  r.open(); await run(r, 2000, 50);
  assert.equal(r.view.snapshot().fading, 0);
  const comps = new Set(FIRST.map(k => r.view.cols.get(k).comp.canvas));
  const layDraws = FIRST.map(k => ctxOf(r, k).calls.filter(c => c[0] === 'drawImage' && comps.has(c[1])).length);
  assert.deepEqual(layDraws, [1, 1, 1], 'the picture laid once');
  assert.ok(FIRST.every(k => !ctxOf(r, k).calls.some(c => c[0] === '=' && c[1] === 'globalAlpha' && c[2] > 0 && c[2] < 1)), 'no part-strength paint');
  // A canvas without pixel access still shows the dots (square, one fill per colour).
  const plain = setup({ pixels: false });
  plain.open(); await run(plain, 1000, 50);
  assert.equal(plain.view.snapshot().columns[0].points, 6);
  assert.ok(plain.view.cols.get('run').comp.canvas.getContext().calls.filter(c => c[0] === 'fillRect').length >= 6);
});

test('the points are asked once per column and layout; one that fails is asked again, then the column says so', async () => {
  let fails = 0;
  const s = setup({ points: (key) => { if (key === 'cell_boundaries') { fails++; throw new Error('no'); } return encodePoints(POINTS[key]); } });
  s.open(); await run(s, 5000, 50);
  assert.equal(fails, POINT_TRIES);
  assert.equal(s.calls.points.filter(c => c[1] === 2).length, POINT_TRIES); assert.equal(s.calls.points.filter(c => c[1] === 0).length, 1);
  const bad = col(s, 'cell_boundaries');
  assert.equal(bad.one('.cmp-wait').textContent, T.umapFailed); assert.equal(visible(bad.one('.cmp-wait')), true);
  assert.deepEqual(s.view.snapshot().columns.map(c => c.points), [6, 6, 0], 'the others drawn');
  assert.equal(s.counts.compareFetches, 1 + 1 + 2 + POINT_TRIES, 'the start, one status, the points');
  // A column with no points says it has too few cells for a UMAP.
  const none = setup({ reply: (n, keys) => status(keys, { columns: keys.map(k => column(k, k === 'run' ? { umap_points: 0 } : {})) }) });
  none.open(); await run(none, 1000, 50);
  assert.equal(col(none, 'run').one('.cmp-wait').textContent, T.umapNone);
  // An even pick of the cells: said under the UMAP.
  const picked = setup({ reply: (n, keys) => status(keys, { columns: keys.map(k => column(k, { sampled: k === 'xenium_cell_boundaries' })) }) });
  picked.open(); await run(picked, 1000, 50);
  const shown = col(picked, 'xenium_cell_boundaries').one('.cmp-shown');
  assert.equal(visible(shown), true); assert.equal(shown.textContent, `6 of ${sum(HIST.xenium_cell_boundaries).toLocaleString('en-GB')} cells shown`);
  assert.equal(shown.title, T.shownTooltip); assert.equal(visible(col(picked, 'run').one('.cmp-shown')), false);
});

test('colour by kind or mixing: one press recolours every column with one cross-fade; Kind without kinds says why, with Learn the details', async () => {
  const s = setup();
  s.open(); await run(s, 1000, 50);
  const kind = s.root.one('button[data-action=kind]'), mixing = s.root.one('button[data-action=mixing]');
  assert.equal(kind.getAttribute('aria-pressed'), 'true'); assert.equal(mixing.getAttribute('aria-pressed'), 'false');
  assert.equal(s.root.one('button[data-colour-by=mixing]'), mixing, 'named for the harness (toggle_js)');
  assert.equal(s.root.one('.cmp-tip').getAttribute('role'), 'tooltip');
  assert.equal(visible(s.root.one('.cmp-key-group[data-by=kind]')), true);
  assert.deepEqual(s.root.all('.cmp-key-group[data-by=kind] li').map(li => li.visibleText().trim()), KINDS.map(k => k.name));
  const [x, y] = at(s, 'run', 2);
  assert.ok(near(pixel(s, 'run', x, y), dot(KINDS[1].color)));
  const d0 = s.counts.compareDraws;
  mixing.click(); await run(s, FADE_MS + 200, 10);
  assert.equal(mixing.getAttribute('aria-pressed'), 'true'); assert.equal(kind.getAttribute('aria-pressed'), 'false');
  assert.equal(s.root.dataset.colourBy, 'mixing'); assert.equal(s.view.snapshot().colourBy, 'mixing');
  assert.ok(near(pixel(s, 'run', x, y), dot(MIXING_COLORS.mixed)), 'share 80: looks mixed');
  const [x1, y1] = at(s, 'run', 1);
  assert.ok(near(pixel(s, 'run', x1, y1), dot(MIXING_COLORS.some)));
  const paints = s.counts.compareDraws - d0;
  assert.ok(paints >= 3 * 2 && paints <= 3 * (Math.ceil(FADE_MS / TICK_MS) + 2), `${paints} paints`);
  assert.equal(visible(s.root.one('.cmp-key-group[data-by=mixing]')), true); assert.equal(visible(s.root.one('.cmp-key-group[data-by=kind]')), false);
  assert.deepEqual(s.root.all('.cmp-key-group[data-by=mixing] li').map(li => li.visibleText().trim()), [T.mixLittle, T.mixSome, T.mixMixed]);
  assert.deepEqual(s.root.all('.cmp-key-group[data-by=mixing] .cmp-swatch').map(n => n.style.background), ['little', 'some', 'mixed'].map(k => MIXING_COLORS[k]));
  kind.click(); await run(s, FADE_MS + 200, 10);
  assert.ok(near(pixel(s, 'run', x, y), dot(KINDS[1].color)), 'back to the kinds');
  // No kinds yet: by mixing; Kind says why (never greyed out) and offers Learn the details.
  let answer = { detail: { state: 'learning' } };
  const n = setup({ reply: (i, keys) => status(keys, { kinds: null }), learn: () => Promise.resolve(answer) });
  n.open(); await run(n, 1000, 50);
  assert.equal(n.view.snapshot().colourBy, 'mixing'); assert.equal(n.root.one('button[data-action=mixing]').getAttribute('aria-pressed'), 'true');
  assert.equal(n.view.snapshot().kinds, null);
  const note = n.root.one('.cmp-key-note');
  assert.equal(visible(note), true); assert.match(note.visibleText(), new RegExp(T.kindsNeeded.slice(0, 30)));
  n.root.one('button[data-action=kind]').click(); await run(n, 50);
  assert.equal(n.view.snapshot().colourBy, 'mixing', 'the dots stay by mixing');
  const learn = n.root.one('button[data-action=learn]');
  assert.equal(visible(learn), true); assert.equal(learn.disabled, false);
  learn.click(); await run(n, 50);
  assert.deepEqual(n.calls.learn, ['zd02871bc0e93']);
  assert.match(note.visibleText(), /Learning the details/); assert.equal(visible(learn), false);
  // Learned at once: the compare starts again, and its kinds colour the dots (Kind was pressed).
  answer = { model: { deep: true } };
  const m = setup({ reply: (i, keys) => status(keys, { kinds: m.calls.start.length > 1 ? KINDS : null }), learn: () => Promise.resolve(answer) });
  m.open(); await run(m, 1000, 50);
  m.root.one('button[data-action=kind]').click(); m.root.one('button[data-action=learn]').click(); await run(m, 1500, 50);
  assert.equal(m.calls.start.length, 2); assert.equal(m.view.snapshot().colourBy, 'kind'); assert.equal(visible(m.root.one('.cmp-key-note')), false);
  // Without learnDetails the note stands alone.
  const plainNote = setup({ reply: (i, keys) => status(keys, { kinds: null }) });
  plainNote.open(); await run(plainNote, 500, 50);
  assert.equal(visible(plainNote.root.one('button[data-action=learn]')), false);
});

test('a kind lit from the key: its cells over the others, dimmed, in every column, one paint each; gone when the pointer leaves', async () => {
  const s = setup({ reduced: true });
  s.open(); await run(s, 1000, 50);
  const item = s.root.all('.cmp-key-group[data-by=kind] button')[0];
  const d0 = s.counts.compareDraws;
  item.emit('pointerenter'); await run(s, 50);
  assert.equal(s.counts.compareDraws - d0, 3, 'one paint per column');
  const [x, y] = at(s, 'run', 2);   // kind 1: dimmed
  assert.ok(near(pixel(s, 'run', x, y), dot(KINDS[1].color, DIM_ALPHA)), `${pixel(s, 'run', x, y)}`);
  const [x0, y0] = at(s, 'run', 0);   // kind 0: lit
  assert.ok(near(pixel(s, 'run', x0, y0), dot(KINDS[0].color)));
  assert.equal(col(s, 'xenium_cell_boundaries').one('canvas').getContext().calls.filter(c => c[0] === 'drawImage').length >= 2, true, 'every column');
  item.emit('pointerleave'); await run(s, 50);
  assert.ok(near(pixel(s, 'run', x, y), dot(KINDS[1].color)), 'all at full strength again');
});

test('hover: the dot under the pointer named with its kind and share; its place ringed in every column; leaving hides it', async () => {
  const s = setup({ reduced: true });
  s.open(); await run(s, 1000, 50);
  const canvas = col(s, 'xenium_cell_boundaries').one('canvas.compare-umap'), [x, y] = at(s, 'xenium_cell_boundaries', 2);
  const d0 = s.counts.compareDraws;
  canvas.emit('pointermove', { offsetX: x + 2, offsetY: y - 1 }); await run(s, 30);
  const tip = col(s, 'xenium_cell_boundaries').one('.cmp-tip');
  assert.equal(visible(tip), true);
  assert.equal(tip.one('.cmp-tip-kind').visibleText().trim(), KINDS[1].name);
  assert.equal(tip.one('.cmp-tip-share').textContent, 'About 32 % of its RNA from neighbours · looks mixed');
  assert.equal(tip.one('.cmp-swatch').style.background, KINDS[1].color);
  assert.equal(s.counts.compareDraws - d0, 3, 'each column painted once: its ring');
  const arcs = FIRST.map(k => ctxOf(s, k).calls.filter(c => c[0] === 'arc').at(-1));
  assert.ok(Math.abs(arcs[1][1] - x) < 1e-6 && Math.abs(arcs[1][2] - y) < 1e-6, 'ringed in its column');
  assert.ok(arcs.every(a => Math.abs(a[1] - x) < 1e-6 && Math.abs(a[2] - y) < 1e-6), 'and at the same place in every other');
  assert.ok(s.view.texts().includes(KINDS[1].name) && s.view.texts().includes('About 32 % of its RNA from neighbours · looks mixed'), 'the harness reads the tip');
  // The same dot again: no paint; another: named.
  const d1 = s.counts.compareDraws;
  canvas.emit('pointermove', { offsetX: x + 1, offsetY: y }); await run(s, 30);
  assert.equal(s.counts.compareDraws, d1);
  const [x3, y3] = at(s, 'xenium_cell_boundaries', 3);
  canvas.emit('pointermove', { offsetX: x3, offsetY: y3 }); await run(s, 30);
  assert.equal(tip.one('.cmp-tip-share').textContent, 'About 1 % of its RNA from neighbours');
  // Far from every dot, or leaving: hidden.
  canvas.emit('pointermove', { offsetX: x3 + HOVER_PX + 30, offsetY: y3 + HOVER_PX + 30 }); await run(s, 30);
  assert.equal(visible(tip), false);
  canvas.emit('pointermove', { offsetX: x, offsetY: y }); await run(s, 30); assert.equal(visible(tip), true);
  canvas.emit('pointerleave'); await run(s, 30);
  assert.equal(visible(tip), false); assert.equal(s.view.hover, null);
  // A dot with no kind, not judged.
  const { view } = s;
  view.setHover({ key: 'run', index: 5, x: 10, y: 10 });
  const runTip = col(s, 'run').one('.cmp-tip');
  assert.equal(runTip.one('.cmp-tip-kind').visibleText().trim(), T.noKind);
  // No kinds learned: the share alone.
  const n = setup({ reduced: true, reply: (i, keys) => status(keys, { kinds: null }),
    points: key => encodePoints(key === 'run' ? POINTS.run.map((p, i) => (i === 5 ? [p[0], p[1], p[2], 255] : p)) : POINTS[key]) });
  n.open(); await run(n, 1000, 50);
  n.view.setHover({ key: 'run', index: 2, x: 10, y: 10 });
  const nTip = col(n, 'run').one('.cmp-tip');
  assert.equal(visible(nTip), true); assert.equal(visible(nTip.one('.cmp-tip-kind')), false);
  assert.equal(nTip.one('.cmp-tip-share').textContent, 'About 32 % of its RNA from neighbours · looks mixed');
  // A share not judged (255): said so.
  n.view.setHover({ key: 'run', index: 5, x: 10, y: 10 });
  assert.equal(nTip.one('.cmp-tip-share').textContent, T.mixUnjudged);
  // A narrow plot (four columns): the tip measured and kept inside it, beside the pointer where it fits.
  const narrow = n.view.cols.get('run');
  narrow.css = [410, 500]; Object.assign(nTip, { offsetWidth: 240, offsetHeight: 44 });
  n.view.setHover({ key: 'run', index: 2, x: 208, y: 120 });
  assert.deepEqual([nTip.style.left, nTip.style.top, nTip.style.right, nTip.style.bottom], ['166px', '134px', 'auto', 'auto']);
  n.view.setHover({ key: 'run', index: 3, x: 380, y: 480 });   // near the bottom right: left of and above the pointer
  assert.deepEqual([nTip.style.left, nTip.style.top], ['126px', '422px']);
});

test('Add lists the segmentations not shown; a fifth column explains; one taken out starts again; the last one explains', async () => {
  const s = setup();
  s.open(); await run(s, 1000, 50);
  const add = s.root.one('button[data-action=add]'), menu = s.root.one('.cmp-menu');
  assert.equal(visible(menu), false);
  add.click();
  assert.equal(visible(menu), true); assert.equal(add.getAttribute('aria-expanded'), 'true');
  const items = menu.all('.cmp-menu-item');
  assert.deepEqual(items.map(b => b.dataset.key), ['xenium_nucleus_boundaries', 'cellpose_test']);
  assert.match(items[1].visibleText(), /Cellpose/); assert.match(items[1].visibleText(), /cellpose_test · Cellpose cp4_20261021 · 5,378/);
  // Added: a new compare of four columns; every number goes (another area), the pictures dim until the new layout.
  const ghost = s.view.cols.get('run').comp.canvas;
  let reply = 'working';
  s.view.o.fetchStatus = run => { s.calls.status.push([run, s.c.t]); return Promise.resolve(reply === 'working'
    ? status(s.keys, { state: 'working', columns: s.keys.map(pending) }) : status(s.keys, { columns: s.keys.map(k => column(k)) })); };
  items[1].click(); await run(s, 30);
  assert.equal(visible(menu), false);
  assert.deepEqual(s.calls.start.at(-1)[1], [...FIRST, 'cellpose_test']);
  assert.deepEqual(s.root.all('[data-compare-column]').map(n => n.dataset.compareColumn), [...FIRST, 'cellpose_test']);
  assert.equal(s.root.one('.cmp-columns').dataset.count, '4');
  assert.equal(col(s, 'run').one('[data-mixed]').dataset.mixed, '', 'the old number went');
  assert.equal(s.view.cols.get('run').ghost, ghost);
  const ctx = ctxOf(s, 'run');
  assert.ok(ctx.calls.some((c, i) => c[0] === 'drawImage' && c[1] === ghost && ctx.calls.slice(0, i).reverse().find(d => d[0] === '=' && d[1] === 'globalAlpha')?.[2] === GHOST_ALPHA),
    'the old picture dimmed');
  // Cellpose covers under a tenth of the others: said on its plot.
  assert.equal(visible(col(s, 'cellpose_test').one('.cmp-col-note')), true); assert.equal(visible(col(s, 'run').one('.cmp-col-note')), false);
  reply = 'ready'; await run(s, 1500, 50);
  assert.equal(s.view.cols.get('run').ghost, null, 'the new layout in'); assert.deepEqual(s.view.snapshot().columns.map(c => c.points), [6, 6, 6, 3]);
  // Four: Add explains, and nothing is listed.
  add.click();
  assert.equal(menu.one('.cmp-menu-note').textContent, T.addFull); assert.equal(menu.all('.cmp-menu-item').length, 0); assert.equal(add.disabled, false);
  s.view.add('xenium_nucleus_boundaries');
  assert.equal(s.root.one('.cmp-alert-text').textContent, T.addFull); assert.equal(s.calls.start.length, 2);
  // Esc closes the menu first, the view stays.
  const esc = s.root.emit('cancel');
  assert.equal(esc.prevented, true); assert.equal(visible(menu), false); assert.equal(s.view.isOpen, true);
  // Taken out: a new compare of the others; the last one explains.
  col(s, 'cellpose_test').one('button[data-action=remove]').click(); await run(s, 30);
  assert.deepEqual(s.calls.start.at(-1)[1], FIRST);
  for (const key of ['xenium_cell_boundaries', 'cell_boundaries']) { col(s, key).one('button[data-action=remove]').click(); await run(s, 30); }
  assert.deepEqual(s.calls.start.at(-1)[1], ['run']);
  col(s, 'run').one('button[data-action=remove]').click(); await run(s, 30);
  assert.equal(s.calls.start.length, 5); assert.equal(s.root.one('.cmp-alert-text').textContent, T.keepOne);
  assert.equal(s.root.one('.cmp-columns').dataset.count, '1');
  assert.match(s.root.one('[data-compare-area]').visibleText(), /On the area it covers/);
  // The run listed when it is not a column; a list that could not be had says so, with Try again.
  let listFails = true;
  const l = setup({ list: () => (listFails ? Promise.reject(new Error('no')) : { segmentations: LIST }) });
  l.view.chosen.set('zd02871bc0e93', ['xenium_cell_boundaries']);
  l.open(); await run(l, 500, 50);
  assert.deepEqual(l.calls.start[0][1], ['xenium_cell_boundaries'], 'the columns last chosen for the sample');
  l.root.one('button[data-action=add]').click();
  assert.equal(l.root.one('.cmp-menu-note').textContent, T.addFailed);
  assert.deepEqual(l.root.all('.cmp-menu-item').map(b => b.dataset.key), ['run']);
  assert.match(l.root.one('.cmp-menu-item').visibleText(), /This run · iteration 450/);
  listFails = false; l.root.one('button[data-action=list-retry]').click(); await run(l, 50);
  assert.deepEqual(l.root.all('.cmp-menu-item').map(b => b.dataset.key), ['run', 'xenium_nucleus_boundaries', 'cell_boundaries', 'cellpose_test']);
  // A click outside the menu closes it.
  l.root.emit('pointerdown', { target: l.root.one('.cmp-columns') });
  assert.equal(visible(l.root.one('.cmp-menu')), false);
});

test('close and re-open: Back, Esc or the browser closing the dialog end everything; re-opened, the columns last chosen start again', async () => {
  const s = setup();
  s.open(); await run(s, 1000, 50);
  assert.equal(s.counts.compareOpened, 1);
  s.root.one('button[data-action=back]').click();
  assert.equal(s.root.open, false); assert.equal(s.root.dataset.state, 'closed'); assert.equal(s.closed, 1, 'the page is told (onClose)');
  assert.equal(s.c.timers.length, 0); assert.equal(s.c.rafs.length, 0);
  assert.deepEqual(s.view.texts(), []); assert.equal(s.view.snapshot().open, false);
  assert.equal(s.root.all('[data-compare-column]').length, 0, 'every column let go');
  const asked = s.calls.status.length, drawn = s.counts.compareDraws; await run(s, 3000, 100);
  assert.equal(s.calls.status.length, asked); assert.equal(s.counts.compareDraws, drawn);
  // Re-opened with another choice kept: those columns start again.
  s.open(); await run(s, 1000, 50);
  assert.equal(s.counts.compareOpened, 2); assert.equal(s.root.open, true); assert.equal(s.root.dataset.state, 'ready');
  col(s, 'cell_boundaries').one('button[data-action=remove]').click(); await run(s, 1000, 50);
  s.root.emit('cancel');   // Esc
  assert.equal(s.root.open, false); assert.equal(s.closed, 2);
  s.open(); await run(s, 1000, 50);
  assert.deepEqual(s.calls.start.at(-1)[1], ['run', 'xenium_cell_boundaries'], 'the columns last chosen for this sample');
  assert.deepEqual(s.view.snapshot().columns.map(c => c.points), [6, 6]);
  // The browser closing the dialog by itself (a second Esc) closes the view; a close event queued before a re-open does not.
  s.root.open = false; s.root.emit('close');
  assert.equal(s.view.isOpen, false); assert.equal(s.closed, 3);
  s.open(); await run(s, 50); s.root.emit('close');
  assert.equal(s.view.isOpen, true, 'the dialog is open: a stale close event');
  // Where the dialog cannot be modal, the Esc key itself: the menu first, then the view.
  const flat = setup();
  flat.root.showModal = undefined;
  flat.open(); await run(flat, 500, 50);
  assert.equal(flat.root.open, true); assert.ok(flat.root.matches('dialog#compare-view[open]'));
  flat.root.one('button[data-action=add]').click();
  const key = flat.root.emit('keydown', { key: 'Escape' });
  assert.equal(key.prevented, true); assert.equal(visible(flat.root.one('.cmp-menu')), false); assert.equal(flat.view.isOpen, true);
  flat.root.emit('keydown', { key: 'Escape' });
  assert.equal(flat.view.isOpen, false); assert.equal(flat.root.open, false); assert.equal(flat.closed, 1);
  // Opened on another run while open: the old one's answers are dropped.
  let late = null;
  const o = setup({ reply: () => new Promise(r => { late = r; }) });
  o.open(); await run(o, 50);
  o.open({ run: 'r2' }); late(status(FIRST)); await run(o, 50);
  assert.equal(o.view.snapshot().state, 'working'); assert.deepEqual(o.calls.start.map(c => c[0]), ['r1', 'r2']);
  assert.equal(o.view.open({}), false, 'no run: not opened');
  // Disposed: its listeners and dialog go.
  o.view.dispose();
  assert.equal(o.host.children.length, 0); assert.equal(o.doc.listeners.get('visibilitychange').length, 0);
});

test('the engine\'s status as engine/compare.py gives it: the start\'s reply, layouts that grow, its view box, its words why', async () => {
  // The start answers with the compare as it stands: a reused one is ready at once, nothing more asked.
  const s = setup({ start: columns => Promise.resolve({ ...status(columns), layout: 1, bounds: { x: [0, 1], y: [0, 1] } }) });
  s.open({ seq: 12 }); await run(s, 1500, 50);
  assert.deepEqual(s.calls.start[0].slice(0, 2), ['r1', FIRST]); assert.equal(s.view.seq, 12);
  assert.equal(s.calls.status.length, 0, 'ready from the start\'s reply'); assert.equal(s.root.dataset.state, 'ready');
  assert.deepEqual(s.view.snapshot().columns.map(c => c.points), [6, 6, 6]);
  // Its view box (2nd-98th percentiles), widened by the margin, is the scale of every column.
  assert.deepEqual(s.view.layout.bounds, [-FIT_MARGIN, -FIT_MARGIN, 1 + FIT_MARGIN, 1 + FIT_MARGIN]);
  // The frame on screen goes with every start (the run's column is that frame).
  const t = setup(), started = [];
  t.view.o.start = (run, columns, seq) => { started.push(seq); return Promise.resolve({ state: 'working', columns }); };
  t.open({ seq: 40 }); await run(t, 50); t.open({}); await run(t, 50);
  assert.deepEqual(started, [40, null]);
  // Layouts that grow (the columns ready first, then all of them in one layout): the points are asked for again.
  let layout = 1;
  const g = setup({ reply: (n, keys) => ({ ...status(keys, { state: layout < 2 ? 'working' : 'ready' }), layout }) });
  g.open(); await run(g, 600, 50);
  assert.equal(g.calls.points.length, 3); const first = g.view.layout.id;
  layout = 2; await run(g, 1200, 50);
  assert.equal(g.calls.points.length, 6, 'every column again for the new layout'); assert.notEqual(g.view.layout.id, first);
  await run(g, 2000, 100); assert.equal(g.calls.points.length, 6);
  // A compare that failed as a whole says the engine's words.
  const f = setup({ reply: (n, keys) => ({ ...status(keys, { state: 'failed', columns: keys.map(pending) }), error: 'These share no tissue: no 100 µm square holds 3 cells of each' }) });
  f.open(); await run(f, 600, 50);
  assert.equal(f.root.dataset.state, 'failed');
  assert.equal(f.root.one('.cmp-alert-text').textContent, "Couldn't compare: These share no tissue: no 100 µm square holds 3 cells of each");
  assert.equal(visible(f.root.one('button[data-action=retry]')), true); const asked = f.calls.status.length; await run(f, 2000, 100);
  assert.equal(f.calls.status.length, asked, 'nothing more asked');
  // Kinds that could not be made (no_source, failed): said, with nothing to press; needs_detail: Learn the details.
  for (const [reason, words, press] of [['failed', T.kindsUnavailable, false], ['no_source', T.kindsUnavailable, false], ['needs_detail', T.kindsNeeded, true]]) {
    const k = setup({ reply: (n, keys) => ({ ...status(keys, { kinds: null }), kinds_reason: reason }), learn: () => Promise.resolve({}) });
    k.open(); await run(k, 600, 50);
    assert.equal(k.root.one('.cmp-key-note span').textContent, words, reason);
    assert.equal(visible(k.root.one('button[data-action=learn]')), press, reason);
  }
  // No picture for a reason other than too few cells: said so; a column still counting says it.
  const l = setup({ reply: (n, keys) => ({ ...status(keys, { columns: keys.map(k => column(k, { umap_points: 0 })) }), layout_error: 'the layout went wrong' }) });
  l.open(); await run(l, 600, 50);
  assert.equal(col(l, 'run').one('.cmp-wait').textContent, T.layoutFailed);
  const c = setup({ reply: (n, keys) => status(keys, { state: 'working', step: 'counting', columns: keys.map(k => ({ ...pending(k), state: k === 'run' ? 'working' : 'counting' })) }) });
  c.open(); await run(c, 100);
  assert.match(col(c, 'cell_boundaries').one('.cmp-mixed').visibleText(), new RegExp(T.columnCounting));
  assert.match(col(c, 'run').one('.cmp-mixed').visibleText(), new RegExp(T.columnWorking));
});

test('a layout that moves on while a column\'s points are on their way: the new layout\'s are asked for and drawn; a download a new compare let go changes nothing', async () => {
  // The engine scores the columns ready first (layout 1: the run's alone), then all of them (layout 2), and publishes
  // layout 2 with "ready": the last status asked for. The run's layout-1 points land after it.
  const s = setup({
    reply: (n, keys) => (n === 1 ? { ...status(keys, { state: 'working', columns: [column('run'), ...keys.slice(1).map(pending)] }), layout: 1 }
      : { ...status(keys), layout: 2 }),
    points: key => (s.calls.points.length === 1 ? new Promise(resolve => s.c.setTimeout(() => resolve(encodePoints(POINTS[key])), 700)) : encodePoints(POINTS[key])),
  });
  s.open(); await run(s, 5000, 50);
  assert.equal(s.calls.status.length, 2, 'ready with layout 2: nothing more asked');
  assert.deepEqual(s.calls.points.map(p => p[1]), [0, 1, 2, 0], "the run's asked again for layout 2 once its layout-1 download landed");
  assert.equal(JSON.parse(s.view.layout?.id ?? 'null')?.[1], 2, 'layout 2 drawn');
  assert.deepEqual(s.view.snapshot().columns.map(c => c.points), [6, 6, 6]);
  assert.equal(visible(col(s, 'run').one('.cmp-wait')), false, 'no "Waiting for the UMAP…" left');
  // A column added while the run's points are on their way (the engine still working: asked every 500 ms): the new
  // compare's download is the column's; the old one, landing meanwhile, neither takes its place nor lets a second start.
  let slow = 800;
  const t = setup({
    reply: (n, keys) => status(keys, { state: 'working' }),
    points: key => (key === 'run' ? new Promise(resolve => t.c.setTimeout(() => resolve(encodePoints(POINTS[key])), slow)) : encodePoints(POINTS[key])),
  });
  t.open(); await run(t, 300, 25);
  slow = 1500; t.view.add('cellpose_test'); await run(t, 3000, 25);
  assert.ok(t.calls.status.length >= 5, 'asked all along');
  assert.equal(t.calls.points.filter(p => p[1] === 0).length, 2, "the run's points once per compare");
  assert.deepEqual(t.view.snapshot().columns.map(c => c.points), [6, 6, 6, 3]);
});

test('"Under a tenth of another column\'s area" reads the squares each column covers (the engine\'s), else the extents cut to the run\'s region', async () => {
  // A region run 500 µm across beside the sample's whole segmentations: the engine reads every column inside the region,
  // where each covers about as much as the run. Before the area is known, the extents cut to the region say as much.
  const tiles = { run: 25, xenium_cell_boundaries: 24, cell_boundaries: 22, cellpose_test: 2 };
  let found = false;
  const s = setup({ reply: (n, keys) => (found ? status(keys, { columns: keys.map(k => column(k, { tiles: tiles[k] })) })
    : status(keys, { state: 'working', columns: keys.map(pending) })) });
  const note = key => visible(col(s, key).one('.cmp-col-note'));
  s.open({ box: [0, 500, 500, 1000] }); await run(s, 600, 50);
  assert.deepEqual(FIRST.map(note), [false, false, false], 'the run\'s region against the others\' parts in it');
  found = true; await run(s, 1000, 50);
  assert.equal(s.view.cols.get('run').data.tiles, 25);
  assert.deepEqual(FIRST.map(note), [false, false, false], 'about as many squares each');
  // Cellpose added: two squares, under a tenth of the run's 25 (said on its plot once the engine has counted them).
  s.view.add('cellpose_test'); await run(s, 1000, 50);
  assert.deepEqual([...FIRST, 'cellpose_test'].map(note), [false, false, false, true]);
  assert.equal(col(s, 'cellpose_test').one('.cmp-col-note').textContent, T.smallCover);
  // The squares decide over the extents: Cellpose's box is under a tenth of Xenium's, its squares are not.
  let known = false;
  const w = setup({ reply: (n, keys) => (known ? status(keys, { columns: keys.map(k => column(k, { tiles: k === 'cellpose_test' ? 211 : 900 })) })
    : status(keys, { state: 'working', columns: keys.map(pending) })) });
  w.open(); await run(w, 30);
  w.view.begin(['xenium_cell_boundaries', 'cellpose_test']); await run(w, 600, 50);
  assert.equal(visible(col(w, 'cellpose_test').one('.cmp-col-note')), true, 'before the area: its extent');
  known = true; await run(w, 1000, 50);
  assert.equal(visible(col(w, 'cellpose_test').one('.cmp-col-note')), false, 'once known: its 211 squares of 900');
});

test('the counting rule in the number\'s ⓘ is the run\'s own, as the engine counted every column: its quality, and the genes it left out', async () => {
  const tip = s => col(s, 'run').one('.cmp-mixed .cmp-info').title;
  const answer = rule => setup({ reply: (n, keys) => ({ ...status(keys), ...(rule === undefined ? {} : { rule }) }) });
  for (const [rule, text] of [[undefined, compare('counting', { q: '20' })], [{ min_qv: 25, exclude: null }, compare('counting', { q: '25' })],
    [{ min_qv: 12.5, exclude: null }, compare('counting', { q: '12.5' })], [{ min_qv: 20, exclude: '^(BLANK|NegControl|Unassigned)' }, compare('countingExcluded', { q: '20' })]]) {
    const s = answer(rule);
    s.open(); await run(s, 600, 50);
    // A column on proseg's count (the run's): the counting rule ends with that count's bias.
    assert.ok(tip(s).includes(`${text} ${T.countingBias}`), `${JSON.stringify(rule)}: ${tip(s)}`);
    assert.ok(tip(s).includes(T.transfer));
    assert.ok(!s.view.texts().some(t => /BLANK|NegControl|\^\(/.test(t)), 'a pattern is no words for the page');
    plainWords(text, 'counting');
  }
  assert.match(compare('counting', { q: '25' }), /quality 25 or above/);
  // No column on proseg's count (the run not among them): the rule alone.
  const none = setup({ reply: (n, keys) => status(keys, { columns: keys.map(k => column(k, { count: k === 'run' ? 'outlines' : COUNTED[k] })) }) });
  none.open(); await run(none, 600, 50);
  assert.ok(tip(none).includes(compare('counting', { q: '20' }))); assert.ok(!tip(none).includes(T.countingBias));
  // As the status gives it (GET /runs/{id}/compare `rule`), checked.
  assert.deepEqual(normaliseStatus({ rule: { min_qv: 30, exclude: ' x ' } }).rule, { minQv: 30, exclude: 'x' });
  assert.deepEqual(normaliseStatus({ rule: { min_qv: 'twenty', exclude: '' } }).rule, { minQv: null, exclude: null });
  assert.equal(normaliseStatus({}).rule, null); assert.equal(normaliseStatus({ rule: [20] }).rule, null);
});

test('kinds the engine could not read while the sample was being saved (busy): said in the key, coloured by mixing, nothing to press', async () => {
  const s = setup({ reply: (n, keys) => ({ ...status(keys, { kinds: null }), kinds_reason: 'busy' }), learn: () => Promise.resolve({}) });
  s.open(); await run(s, 600, 50);
  const note = s.root.one('.cmp-key-note');
  assert.equal(note.one('span').textContent, T.kindsBusy); assert.equal(visible(note), true);
  assert.equal(visible(s.root.one('button[data-action=learn]')), false, 'nothing to learn: a new Compare makes them');
  assert.equal(s.view.snapshot().colourBy, 'mixing'); assert.equal(s.view.snapshot().kinds, null);
  s.root.one('button[data-action=kind]').click(); await run(s, 50);
  assert.equal(note.one('span').textContent, T.kindsBusy, 'Kind pressed: the same why'); assert.equal(s.view.snapshot().colourBy, 'mixing');
  plainWords(T.kindsBusy, 'kindsBusy');
  for (const text of s.view.texts()) if (!DATA.test(text)) plainWords(text, 'on screen');
});

test('the snapshot and the counters', async () => {
  const s = setup();
  assert.deepEqual(COUNTERS, ['compareOpened', 'compareDraws', 'compareFetches']);
  for (const name of COUNTERS) assert.equal(s.counts[name], 0, `${name} listed at 0`);
  let snap = s.view.snapshot();
  assert.deepEqual(snap, { open: false, state: 'closed', columns: [], area: null, colourBy: null, map: null, kinds: null, fading: 0, flying: 0, draws: 0,
    fetches: 0, polls: 0, downloads: 0, opened: 0 });
  s.open(); await run(s, 1000, 50);
  snap = s.view.snapshot();
  assert.equal(snap.open, true); assert.equal(snap.state, 'ready');
  assert.deepEqual(snap.columns, FIRST.map(key => ({ key, state: 'ready', count: COUNTED[key], form: null, like_from: null, mixed_per100: 13.3, like100_per100: 11.2, points: 6,
    own: false })));
  assert.deepEqual(snap.area, AREA); assert.equal(snap.colourBy, 'kind'); assert.equal(snap.kinds, 2);
  assert.equal(snap.fetches, s.counts.compareFetches); assert.equal(snap.draws, s.counts.compareDraws); assert.equal(snap.opened, s.counts.compareOpened);
  assert.equal(snap.polls, s.calls.status.length); assert.equal(snap.downloads, s.calls.points.length);
  assert.equal(snap.fetches, s.calls.start.length + snap.polls + snap.downloads);
  // Before the start answers: the columns are starting.
  const w = setup({ start: () => new Promise(() => {}) });
  w.open(); await run(w, 50);
  assert.deepEqual(w.view.snapshot().columns.map(c => c.state), ['starting', 'starting', 'starting']);
  assert.equal(w.root.one('.cmp-progress').textContent, T.starting);
  assert.equal(w.root.one('[data-compare-area]').visibleText().trim(), T.areaFinding);
});

test('every word the view shows passes the rules, in every state (the engine\'s and the sample\'s names left out)', async () => {
  const seen = new Set();
  const check = s => { for (const t of s.view.texts()) { seen.add(t); if (!DATA.test(t)) plainWords(t, 'on screen'); } };
  const a = setup({ reply: (n, keys) => status(keys, { state: 'working', step: 'counting', progress: { done: 1, of: 4 }, kinds: null, columns: [column('run'), ...keys.slice(1).map(pending)] }) });
  a.open(); await run(a, 50); check(a);
  const b = setup({ reply: (n, keys) => status(keys, { kinds: null }), learn: () => Promise.resolve({}) });
  b.open(); await run(b, 1000, 50); b.root.one('button[data-action=kind]').click(); check(b);
  b.root.one('button[data-action=add]').click(); check(b);
  const c = setup({ reply: (n, keys) => status(keys, { columns: keys.map((k, i) => column(k, i ? { sampled: true, judged: 200 } : { state: 'failed', error: null })) }) });
  c.open(); await run(c, 1000, 50); check(c);
  const [x, y] = at(c, 'xenium_cell_boundaries', 2);
  col(c, 'xenium_cell_boundaries').one('canvas').emit('pointermove', { offsetX: x, offsetY: y }); check(c);
  const d = setup({ reply: () => { throw new Error('gone'); } });
  d.open(); await run(d, 30000, 200); check(d);
  for (const want of [T.unreachable, T.kindsNeeded, T.addTitle, T.statCellsTooltip, T.shownTooltip, T.histTooltip, T.columnFailed])
    assert.ok(seen.has(want), `shown: ${want}`);
  assert.ok([...seen].some(t => t.includes(compare('counting', { q: '20' })) && t.includes(T.transfer)), 'the counting rule and the transfer caveat in a tooltip');
});

test('nothing in the view\'s stylesheet moves by itself; the script names no CSS motion; the card tokens and tabular numbers', () => {
  const css = readFileSync(new URL('../src/compare.css', import.meta.url), 'utf8');
  assert.doesNotMatch(css, /@keyframes|animation|transition|backdrop-filter/i);
  assert.match(css, /font-variant-numeric:tabular-nums/);
  for (const token of ['--card-radius', '--card-border', '--card-surface', '--card-pad']) assert.match(css, new RegExp(token), token);
  assert.match(css, /\.compare-umap\b/); assert.match(css, /\.cmp:not\(\[open\]\) \{ display:none; \}/, 'closed: not shown');
  assert.match(css, /\.cmp-mixing \{[^}]*height:\d+px/, 'a fixed height: every plot the same size');
  for (const name of ['compare.mjs', 'compare-copy.mjs']) {
    const source = readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8');
    assert.doesNotMatch(source, /animation\s*:|transition\s*:|backdrop-filter|requestIdleCallback|getContext\(['"]webgl/i, name);
  }
  assert.ok(STATUS_MS >= 500, 'at most twice a second'); assert.equal(MAX_COLUMNS, 4); assert.deepEqual(PLOT_PX, [360, 300]);
  assert.deepEqual(Object.keys(STEPS).sort(), ['area', 'counting', 'kinds', 'layout', 'mixing', 'outlines', 'queued', 'scoring', 'umap', 'waiting']);
});

// ---- a comparison's points are its own -----------------------------------------------------------------------------------
/** One fake engine with one comparison slot per run, as engine/compare.py keeps: a start replaces the slot (a new job);
 *  points are served for the slot's job and layout only (409 otherwise), each file marked with the column it is for. */
function slotEngine() {
  const engine = { job: null, keys: [], layout: 1, n: 0, asked: [] };
  engine.start = keys => { engine.job = `job${++engine.n}`; engine.keys = [...keys]; engine.layout = 1;
    return { ...status(keys, { state: 'working' }), job: engine.job, layout: engine.layout }; };
  engine.status = () => ({ ...status(engine.keys), job: engine.job, layout: engine.layout });
  engine.points = (index, job, layout) => {
    engine.asked.push([index, job, layout]);
    if (job !== engine.job || layout !== engine.layout) return Promise.reject(Object.assign(new Error('replaced'), { status: 409 }));
    return Promise.resolve(encodePoints(POINTS[engine.keys[index]]));
  };
  return engine;
}
function slotView(engine) {
  const { host } = dom(), c = clock(), s = { c, host };
  s.view = new CompareView(host, { clock: c, reducedMotion: () => true, storage: memory({ 'sopa-companion:compare:map': 'one' }),
    fetchStatus: () => Promise.resolve().then(() => engine.status()),
    fetchPoints: (run, index, job, layout) => Promise.resolve().then(() => engine.points(index, job, layout)),
    start: (run, columns) => Promise.resolve().then(() => engine.start(columns)),
    listSegmentations: () => Promise.resolve({ segmentations: LIST }) });
  s.root = s.view.root;
  s.open = () => s.view.open({ run: 'r1', iteration: 450, sampleId: 'zd02871bc0e93' });
  return s;
}
test('points are asked for with the comparison and layout the status named; another window\'s comparison is said, never drawn', async () => {
  const engine = slotEngine(), a = slotView(engine);
  a.open(); await run(a, 1000, 50);
  assert.equal(a.view.state, 'ready'); assert.equal(a.view.job, 'job1');
  assert.deepEqual(engine.asked, [[0, 'job1', 1], [1, 'job1', 1], [2, 'job1', 1]], 'every download names job and layout');
  assert.deepEqual(a.view.snapshot().columns.map(c => c.points), [6, 6, 6]);
  // A race: A starts a comparison (job2) and reads its status; another window's start (its columns
  // reversed) replaces the run's one comparison (job3) before A's points are asked for.
  const points = engine.points;
  let raced = false;
  engine.points = (index, job, layout) => {
    if (!raced && job === 'job2') { raced = true; engine.start(['cell_boundaries', 'xenium_cell_boundaries', 'run']); }
    return points(index, job, layout);
  };
  a.view.begin(a.view.keys); await run(a, 1000, 50);
  // A's download is refused (409), its status asked for again names job3: A says so and keeps nothing of it.
  assert.equal(engine.job, 'job3'); assert.equal(a.view.job, 'job2');
  assert.equal(a.view.state, 'failed'); assert.equal(a.view.replaced, true);
  assert.equal(a.root.one('.cmp-alert-text').textContent, REPLACED_TEXT); assert.equal(visible(a.root.one('button[data-action=retry]')), true);
  assert.ok(engine.asked.filter(([, job]) => job === 'job3').length === 0, 'A never asked for the other comparison\'s points');
  assert.equal(a.view.layout, null, 'nothing of it drawn under A\'s columns');
  assert.deepEqual(bannedWords(REPLACED_TEXT), []);
  // Try again: A's own columns compared anew (a new job), drawn again.
  a.root.one('button[data-action=retry]').click(); await run(a, 1000, 50);
  assert.equal(engine.job, 'job4'); assert.equal(a.view.job, 'job4'); assert.equal(a.view.state, 'ready'); assert.equal(a.view.replaced, false);
  assert.deepEqual(engine.keys, FIRST); assert.deepEqual(a.view.snapshot().columns.map(c => c.points), [6, 6, 6]);
});
test('points refused for a stale layout keep no bytes: the status is asked for again and the newer layout drawn', async () => {
  const engine = slotEngine(), s = slotView(engine);
  const points = engine.points;
  let moved = false;
  engine.points = (index, job, layout) => {
    // The engine lays the points out once more just as the first download is asked for.
    if (!moved) { moved = true; engine.layout = 2; }
    return points(index, job, layout);
  };
  s.open(); await run(s, 2000, 50);
  assert.equal(s.view.state, 'ready');
  const refused = engine.asked.filter(([, , layout]) => layout === 1), fetched = engine.asked.filter(([, , layout]) => layout === 2);
  assert.ok(refused.length >= 1, 'a layout-1 download was refused');
  assert.deepEqual(fetched.map(([index]) => index).sort(), [0, 1, 2], 'each column downloaded once for layout 2');
  assert.equal(JSON.parse(s.view.layout.id)[1], 2, 'layout 2 drawn');
  assert.deepEqual(s.view.snapshot().columns.map(c => c.points), [6, 6, 6]);
});
test('four columns refused in one layout race count once: four races in a row still end drawn; refusals that never end give up', async () => {
  const four = ['run', 'xenium_cell_boundaries', 'cell_boundaries', 'cellpose_test'];
  const raced = races => {
    const engine = slotEngine(), s = slotView(engine), points = engine.points;
    let left = races;
    engine.points = (index, job, layout) => {
      // The engine lays the points out once more just as a layout's first download is asked for: every column's
      // download of that layout is refused (409).
      if (left > 0 && job === engine.job && layout === engine.layout && engine.keys.length === 4) { left--; engine.layout++; }
      return points(index, job, layout);
    };
    return { engine, s };
  };
  const { engine, s } = raced(4);
  s.open(); await run(s, 1000, 50); s.view.begin(four); await run(s, 3000, 50);
  assert.equal(engine.layout, 5, 'four races');
  assert.ok(engine.asked.filter(([, job, layout]) => job === engine.job && layout < 5).length >= 4 * 3, 'each race refused several columns');
  assert.equal(s.view.state, 'ready', `never gave up after ${4 * 4} refusals in ${4} races (${MISMATCH_TRIES} asks at the most)`);
  assert.deepEqual(s.view.snapshot().columns.map(c => c.points), [6, 6, 6, 3]); assert.equal(JSON.parse(s.view.layout.id)[1], 5);
  // Refusals that never end: the view gives up after MISMATCH_TRIES statuses asked because of them.
  const never = raced(Infinity);
  never.s.open(); await run(never.s, 1000, 50); never.s.view.begin(four); await run(never.s, 20000, 50);
  assert.equal(never.s.view.state, 'failed');
  assert.ok(never.engine.layout - 1 <= MISMATCH_TRIES + 1, `${never.engine.layout - 1} layouts for ${MISMATCH_TRIES} asks (and the first status)`);
});

test("proseg's two columns are told apart: the run's by the count it reads (averaged, or as saved), a table proseg saved in the sample as that table (H9)", async () => {
  // The run's column on an iteration counted for a pause, the hold, a finish or the end reads the count proseg saves
  // there (the engine's form "sampled"); any other column on proseg's count is a table proseg saved (one draw of it).
  // Here the sample's cell_boundaries are a table proseg saved (the engine reads it on proseg's count).
  const reply = form => (n, cols) => status(cols, { columns: cols.map(k => column(k, k === 'run' ? { form } : k === 'cell_boundaries' ? { count: 'proseg' } : {})) });
  for (const [form, words, tip] of [['sampled', T.countProsegSaved, T.countProsegSavedTooltip], ['averaged', T.countProsegAveraged, T.countProsegTooltip],
    [undefined, T.countProsegAveraged, T.countProsegTooltip]]) {
    const s = setup({ reply: reply(form) });
    s.open(); await run(s, 1000, 50);
    const label = key => col(s, key).one('.cmp-col-count');
    assert.equal(label('run').textContent, words, String(form)); assert.equal(label('run').title, `${tip} ${T.countingBias}`, String(form));
    assert.equal(label('cell_boundaries').textContent, T.countProsegTable, String(form));
    assert.equal(label('cell_boundaries').title, `${T.countProsegTableTooltip} ${T.countingBias}`);
    assert.notEqual(label('run').textContent, label('cell_boundaries').textContent, 'never the same label');
    assert.equal(label('xenium_cell_boundaries').textContent, T.countXenium);
    for (const text of s.view.texts()) if (!DATA.test(text)) plainWords(text, 'on screen');
  }
});

test("the like-for-like line's tooltip says whose draw the run's cut starts from (H6), and on a whole sample that its depth is not re-measured", async () => {
  // The engine's like_from: "proseg" when the frame's count file has proseg's own draw (a frame counted for a pause, the
  // hold, a finish or the end), "app" when the app draws from proseg's count; another column says neither.
  for (const [from, words] of [['proseg', T.likeFromProseg], ['app', T.likeFromApp], [undefined, '']]) {
    const s = setup({ reply: (n, cols) => status(cols, { columns: cols.map(k => column(k, k === 'run' ? { like_from: from } : {})) }) });
    s.open(); await run(s, 1000, 50);
    const tip = key => col(s, key).one('.cmp-like .cmp-info').title, base = key => compare('likeTooltip', { j: number(sum(HIST[key])) });
    assert.equal(tip('run'), [base('run'), words].filter(Boolean).join(' '), String(from));
    assert.equal(tip('xenium_cell_boundaries'), base('xenium_cell_boundaries'), 'another column: nothing added');
  }
  // E4: a kind of run whose cut depth is kept from before proseg's count (the run's constants.kept: a whole sample's
  // "like_depth") says so after it, on every column; a region's run nothing.
  const whole = setup({ reply: (n, cols) => status(cols, { columns: cols.map(k => column(k, k === 'run' ? { like_from: 'app' } : {})) }) });
  whole.open({ kept: ['glow', 'where_words', 'start_band', 'min_counts', 'like_depth'] }); await run(whole, 1000, 50);
  const tip = key => col(whole, key).one('.cmp-like .cmp-info').title;
  assert.equal(tip('run'), `${compare('likeTooltip', { j: number(sum(HIST.run)) })} ${T.likeFromApp} ${QC_TEXT.notRemeasuredShort}`);
  assert.equal(tip('xenium_cell_boundaries'), `${compare('likeTooltip', { j: number(sum(HIST.xenium_cell_boundaries)) })} ${QC_TEXT.notRemeasuredShort}`);
  const tiled = setup();
  tiled.open({ kept: ['glow', 'where_words'] }); await run(tiled, 1000, 50);
  assert.ok(!col(tiled, 'run').one('.cmp-like .cmp-info').title.includes(QC_TEXT.notRemeasuredShort), 'a tiled run keeps no like_depth');
  for (const text of whole.view.texts()) if (!DATA.test(text)) plainWords(text, 'on screen');
});

test("the like-for-like line's tooltip names the cells it is over: those cut down to 100 transcripts, which the run's column takes from its sampled count", async () => {
  // The run's column cuts its sampled count (or one draw of it): a cell judged on the averaged count just over 100 can
  // fall short of 100 whole transcripts and is left out of the line (the engine's like100_cells).
  const s = setup({ reply: (n, keys) => status(keys, { columns: keys.map(k => column(k, k === 'run' ? { judged: 3000, like100_cells: 2986 } : {})) }) });
  s.open(); await run(s, 1000, 50);
  assert.equal(col(s, 'run').one('.cmp-like .cmp-info').title, compare('likeTooltip', { j: '2,986' }));
  assert.match(col(s, 'run').one('.cmp-like .cmp-info').title, /^Like for like: each of the 2,986 cells/);
  assert.equal(col(s, 'xenium_cell_boundaries').one('.cmp-like .cmp-info').title, compare('likeTooltip', { j: number(sum(HIST.xenium_cell_boundaries)) }),
    'a column whose engine says none: the cells judged');
});

test("each column says which count it is read on, under its name: proseg's, Xenium's, or the transcripts inside its outlines", async () => {
  const s = setup();
  s.open(); await run(s, 1000, 50);
  const label = key => col(s, key).one('.cmp-col-count');
  assert.equal(label('run').textContent, T.countProsegAveraged, 'an engine that names no form: averaged'); assert.equal(label('xenium_cell_boundaries').textContent, T.countXenium);
  assert.equal(label('cell_boundaries').textContent, T.countOutlines);
  for (const key of FIRST) assert.equal(visible(label(key)), true, key);
  // Its tooltip (the label can be cut short in a narrow column): which count it reads, then the bias of proseg's
  // count in one line while a column is on it.
  assert.equal(label('run').title, `${T.countProsegTooltip} ${T.countingBias}`);
  assert.equal(label('xenium_cell_boundaries').title, `${T.countXeniumTooltip} ${T.countingBias}`);
  assert.equal(label('cell_boundaries').title, `${T.countOutlinesTooltip} ${T.countingBias}`);
  assert.deepEqual([T.countProsegTooltip, T.countXeniumTooltip, T.countOutlinesTooltip], ["This column reads proseg's own count of each cell at this iteration, averaged over its random background draw.",
    "This column reads Xenium's own count of each cell, from its cell table.",
    'This segmentation has no count of its own: this column counts the transcripts inside its outlines.']);
  // No column on proseg's count: no bias line.
  const none = setup({ reply: (n, keys) => status(keys, { columns: keys.map(k => column(k, { count: 'outlines' })) }) });
  none.open(); await run(none, 1000, 50);
  assert.equal(col(none, 'run').one('.cmp-col-count').title, T.countOutlinesTooltip);
  assert.equal(label('run').parentElement, col(s, 'run').one('.cmp-col-names'), 'in its head, under its key and source');
  // A count the engine names that the page has no words for, or none: no label (never a guess).
  const odd = setup({ reply: (n, keys) => status(keys, { columns: keys.map(k => column(k, { count: k === 'run' ? 'something new' : undefined })) }) });
  odd.open(); await run(odd, 1000, 50);
  for (const key of FIRST) {
    assert.equal(col(odd, key).one('.cmp-col-count').textContent, ''); assert.equal(visible(col(odd, key).one('.cmp-col-count')), false);
    assert.equal(col(odd, key).one('.cmp-col-count').title, '', 'no label, no tooltip');
  }
  for (const text of s.view.texts()) if (!DATA.test(text)) plainWords(text, 'on screen');
});

// ---- Each its own and One map ------------------------------------------------------------------------------------------
// Each column's own map: the same points in the same order, at places of its own (here the one map's turned and
// doubled), and its own clusters (u16 per point, NO_CLUSTER for none). The engine's view box of every own map.
const OWN = Object.fromEntries(Object.entries(POINTS).map(([key, list]) => [key, list.map(([x, y, kind, share]) => [y * 2, -x * 2, kind, share])]));
const CLUSTERS = { run: [0, 0, 1, 1, 2, NO_CLUSTER], xenium_cell_boundaries: [0, 1, 1, 0, 3, 3], cell_boundaries: [0, 0, 1, 1, 2, 4], cellpose_test: [1, 0, 0] };
const OWN_BOUNDS = { x: [0, 2], y: [-2, 0] };
function encodeClusters(list) {
  const b = Buffer.alloc(list.length * 2);
  list.forEach((id, i) => b.writeUInt16LE(id, i * 2));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
/** The status of a comparison whose columns each have their own map (`own`: the keys with one; all by default). */
function ownStatus(keys, { own = keys, ...extra } = {}) {
  const s = status(keys, extra);
  s.columns = s.columns.map(c => (own.includes(c.key) ? { ...c, own: true, clusters: 4, clusters_matched: 3 } : c));
  return { ...s, own_layout: 'umap', own_bounds: OWN_BOUNDS, own_error: null };
}
const ownPoints = (key, index, what) => (what === 'own' ? encodePoints(OWN[key]) : what === 'clusters' ? encodeClusters(CLUSTERS[key]) : encodePoints(POINTS[key]));
/** Where point `i` of column `key` sits on its canvas as drawn now (its own map or the one map). */
function drawnAt(s, key, i) {
  const c = s.view.cols.get(key), p = c.comp.place;
  return [p.ox + p.k * c.comp.pts.x[i], p.oy - p.k * c.comp.pts.y[i]];
}

test("each its own's clusters are one u16 per point; Differences are shares of a smoothed grid beside the yardstick's, in five steps; a flight mixes two places", () => {
  assert.deepEqual([...decodeClusters(encodeClusters([1, 2, NO_CLUSTER]), 3)], [1, 2, NO_CLUSTER]);
  assert.equal(decodeClusters(encodeClusters([1, 2]), 3), null, 'not one per point'); assert.equal(decodeClusters(null, 0), null);
  assert.deepEqual([...clusterSlots(Uint16Array.from([0, 254, 255, 300, NO_CLUSTER]))], [0, 254, 255, 255, 255], 'past the palette: grey');
  const pal = palettes(KINDS);
  assert.deepEqual([...pal.clusters.subarray(9, 12)], rgb(CLUSTER_PALETTE[3])); assert.deepEqual([...pal.clusters.subarray(255 * 3)], rgb(COLORS.none));
  assert.deepEqual([...pal.diff], DIFF_COLORS.flatMap(rgb));
  // The five steps: a third or under, a half, the same, twice, three times or over (log2 of the ratio).
  assert.deepEqual(DIFF_EDGES, [-1.5, -0.5, 0.5, 1.5]); assert.equal(DIFF_SAME, 2);
  assert.deepEqual([-3, -1.5, -1, -0.5, 0, 0.49, 0.5, 1.49, 1.5, 9, NaN].map(diffStep), [0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 2]);
  assert.equal(gridCell(0.99, 0.01, [0, 0, 1, 1], 4), 3); assert.equal(gridCell(5, -5, [0, 0, 1, 1], 4), 3, 'beyond: the edge square');
  assert.equal(gridCell(NaN, 0, [0, 0, 1, 1], 4), -1);
  // An even spread over the map, the same one, and one gathered in its left half (twice the share there).
  const set = list => ({ n: list.length, x: Float32Array.from(list, p => p[0]), y: Float32Array.from(list, p => p[1]) });
  const even = [], left = [];
  for (let i = 0; i < 20; i++) for (let j = 0; j < 20; j++) { even.push([(i + 0.5) / 20, (j + 0.5) / 20]); left.push([(i + 0.5) / 40, (j + 0.5) / 20]); }
  const grid = shareGrid(set(even), [0, 0, 1, 1]);
  assert.ok(Math.abs(grid.reduce((a, v) => a + v, 0) - 1) < 1e-9, 'shares sum to 1'); assert.equal(shareGrid(set([[NaN, 1]]), [0, 0, 1, 1]), null);
  const [ref, same, gathered, none] = differences([set(even), set(even), set(left), null], 0, [0, 0, 1, 1]);
  assert.ok(ref.steps.every(v => v === DIFF_SAME) && ref.ratio.every(v => v === 1), 'the yardstick: about the same, ratio 1');
  assert.ok(same.steps.every(v => v === DIFF_SAME), 'the same spread: about the same');
  const steps = [...gathered.steps].sort((a, b) => a - b);
  assert.equal(steps[steps.length >> 1], 3, 'twice the share in the left half: about 2×');
  assert.equal(none, null);
  assert.deepEqual(differences([null, set(even)], 0, [0, 0, 1, 1]), [null, null], 'no yardstick: none');
  // A flight: device px from where `a` lies to where `b` lies; FLY_PLACE draws them there (y negated).
  const a = { n: 1, x: Float32Array.of(0), y: Float32Array.of(0) }, b = { n: 1, x: Float32Array.of(1), y: Float32Array.of(1) };
  const pa = { ox: 10, oy: 20, k: 5 }, pb = { ox: 0, oy: 0, k: 10 };
  const at0 = flyPoints(a, pa, b, pb, 0), at1 = flyPoints(a, pa, b, pb, 1), half = flyPoints(a, pa, b, pb, 0.5, at0);
  assert.equal(half, at0, 'into the buffer given'); assert.deepEqual([half.x[0], FLY_PLACE.oy - FLY_PLACE.k * half.y[0]], [10, 5]);
  assert.deepEqual([at1.x[0], -at1.y[0]], [10, -10]);
  assert.deepEqual([ease(-1), ease(0.5), ease(2)], [0, 0.5, 1]);
  assert.deepEqual(MAPS, ['own', 'one']); assert.equal(MAP_KEY, 'sopa-companion:compare:map');
});

test("each its own (the default): every column on its own map, downloaded with its points; Clusters colour each column's own groups, numbered in the key", async () => {
  const s = setup({ map: null, reduced: true, reply: (n, keys) => ownStatus(keys), points: ownPoints });
  s.open(); await run(s, 1000, 50);
  assert.equal(s.view.snapshot().map, 'own'); assert.equal(s.root.dataset.map, 'own');
  assert.deepEqual(s.view.snapshot().columns.map(c => [c.points, c.own]), [[6, true], [6, true], [6, true]]);
  assert.deepEqual(s.calls.points.map(c => [c[1], c[3] ?? 'points']).sort(), [0, 1, 2].flatMap(i => [[i, 'clusters'], [i, 'own'], [i, 'points']]).sort(),
    "each column's points, own map and clusters, once");
  const own = s.root.one('button[data-map=own]'), one = s.root.one('button[data-map=one]'), group = s.root.one('button[data-action=groups]');
  assert.equal(own.getAttribute('aria-pressed'), 'true'); assert.equal(one.getAttribute('aria-pressed'), 'false');
  assert.equal(own.visibleText().trim(), T.mapOwn); assert.match(own.title, /line up with the run's column, never reshaped/);
  // Its own places, one scale for every own map (the engine's view box of them).
  const run0 = s.view.cols.get('run');
  assert.equal(run0.comp.pts, run0.own); assert.equal(run0.comp.own, true);
  const [x, y] = drawnAt(s, 'run', 2), place = placement([0, -2, 2, 0].map((v, i) => v + [-1, -1, 1, 1][i] * 2 * FIT_MARGIN), 360, 300, 14);
  assert.ok(Math.abs(x - (place.ox + place.k * OWN.run[2][0])) < 1e-6 && Math.abs(y - (place.oy - place.k * OWN.run[2][1])) < 1e-6);
  assert.ok(near(pixel(s, 'run', x, y), dot(KINDS[1].color)), 'by kind, as on one map');
  // Clusters: the third colouring on each its own.
  assert.equal(group.visibleText().trim(), T.byClusters); assert.equal(group.dataset.colourBy, 'clusters');
  group.click(); await run(s, 100, 10);
  assert.equal(s.view.snapshot().colourBy, 'clusters'); assert.equal(group.getAttribute('aria-pressed'), 'true');
  assert.ok(near(pixel(s, 'run', x, y), dot(CLUSTER_PALETTE[1])), "its own cluster's colour");
  const [x5, y5] = drawnAt(s, 'run', 5);
  assert.ok(near(pixel(s, 'run', x5, y5), dot(COLORS.none)), 'no cluster: grey');
  const key = s.root.one('.cmp-key-group[data-by=clusters]');
  assert.equal(visible(key), true); assert.equal(visible(s.root.one('.cmp-key-group[data-by=kind]')), false);
  assert.deepEqual(key.all('li').map(li => li.visibleText().trim()), ['1', '2', '3', '4', '5'], 'every cluster on screen, from 1');
  assert.equal(key.one('li[data-cluster=3] button').title, 'Cluster 4: in Xenium cells');
  assert.match(key.one('li[data-cluster=0] button').title, /^Cluster 1: in This run · iteration 450, Xenium cells, Earlier segmentation/);
  key.one('li[data-cluster=1] button').onpointerenter();
  assert.deepEqual(s.view.lit, { by: 'clusters', value: 1 });
  key.one('li[data-cluster=1] button').onpointerleave();
  // The tip names a dot's cluster.
  const canvas = col(s, 'run').one('canvas.compare-umap');
  canvas.emit('pointermove', { offsetX: x, offsetY: y }); await run(s, 30);
  const tip = col(s, 'run').one('.cmp-tip');
  assert.equal(visible(tip), true); assert.equal(tip.one('.cmp-tip-more').textContent, 'Cluster 2');
  canvas.emit('pointerleave'); await run(s, 30);
  for (const text of s.view.texts()) if (!DATA.test(text)) plainWords(text, 'on screen');
});

test('the Map switch: a press flies every dot to its place on the other map, then rests; remembered; Differences on one map; one paint under reduced motion', async () => {
  const s = setup({ map: null, reply: (n, keys) => ownStatus(keys), points: ownPoints });
  s.open(); await run(s, 1500, 50);
  const one = s.root.one('button[data-map=one]'), own = s.root.one('button[data-map=own]'), group = s.root.one('button[data-action=groups]');
  const d0 = s.counts.compareDraws;
  one.click();
  assert.equal(s.view.snapshot().flying, 3, 'every column flies'); assert.equal(s.view.snapshot().map, 'one');
  assert.equal(s.storage.m.get(MAP_KEY), 'one', 'remembered in this browser');
  await run(s, FLY_MS / 2, 10);
  const mid = s.view.cols.get('run').flight.xy, place = s.view.cols.get('run').comp.place;
  const ownX = place.ox + place.k * OWN.run[1][0];
  assert.ok(mid && mid.x[1] !== ownX, 'on its way');
  await run(s, FLY_MS, 10);
  assert.equal(s.view.snapshot().flying, 0, 'landed');
  const paints = s.counts.compareDraws - d0;
  assert.ok(paints >= 3 * 10 && paints <= 3 * (Math.ceil(FLY_MS / FLY_TICK_MS) + 3), `${paints} paints for 3 columns at ${FLY_FPS} a second at most`);
  const runCol = s.view.cols.get('run');
  assert.equal(runCol.comp.pts, runCol.points, 'on the one map'); assert.equal(runCol.comp.own, false);
  const [x, y] = drawnAt(s, 'run', 2);
  assert.ok(near(pixel(s, 'run', x, y), dot(KINDS[1].color)));
  const d1 = s.counts.compareDraws; await run(s, 2000, 100);
  assert.equal(s.counts.compareDraws, d1, 'then still'); assert.equal(s.c.rafs.length, 0);
  // Differences: the third colouring on one map, the run's column the yardstick (all about the same).
  assert.equal(group.visibleText().trim(), T.byDifferences); assert.equal(group.dataset.colourBy, 'differences');
  assert.match(group.title, /set beside the run's column/);
  group.click(); await run(s, FADE_MS + 200, 10);
  assert.equal(s.view.snapshot().colourBy, 'differences');
  assert.ok(near(pixel(s, 'run', x, y), dot(DIFF_COLORS[DIFF_SAME])), "the yardstick's own dots: about the same");
  const key = s.root.one('.cmp-key-group[data-by=differences]');
  assert.equal(visible(key), true);
  assert.deepEqual(key.all('li').map(li => li.visibleText().trim()), [T.diffThird, T.diffHalf, T.diffSame, T.diffTwice, T.diffThrice]);
  assert.match(key.one('.cmp-info').title, /The dots of the run's column are all grey/);
  const xen = s.view.cols.get('xenium_cell_boundaries'), steps = s.view.diffOf(xen).steps;
  const [xx, xy] = drawnAt(s, 'xenium_cell_boundaries', 0);
  assert.ok(near(pixel(s, 'xenium_cell_boundaries', xx, xy), dot(DIFF_COLORS[steps[0]])), 'another column: its step');
  const canvas = col(s, 'xenium_cell_boundaries').one('canvas.compare-umap');
  canvas.emit('pointermove', { offsetX: xx, offsetY: xy }); await run(s, 30);
  assert.match(col(s, 'xenium_cell_boundaries').one('.cmp-tip-more').textContent, /^Share of cells here: [\d.]+× the run's$/);
  canvas.emit('pointerleave'); await run(s, 30);
  for (const text of s.view.texts()) if (!DATA.test(text)) plainWords(text, 'on screen');
  // Back to each its own: the third colouring is Clusters there.
  own.click(); await run(s, FLY_MS + 200, 10);
  assert.equal(s.view.snapshot().colourBy, 'clusters'); assert.equal(s.view.cols.get('run').comp.own, true);
  // Remembered: a new view in this browser opens on the map last chosen.
  const again = setup({ map: 'one', reply: (n, keys) => ownStatus(keys), points: ownPoints });
  assert.equal(again.view.mapMode, 'one');
  // Reduced motion: no flight, one paint per column.
  const r = setup({ map: null, reduced: true, reply: (n, keys) => ownStatus(keys), points: ownPoints });
  r.open(); await run(r, 1500, 50);
  const r0 = r.counts.compareDraws;
  r.root.one('button[data-map=one]').click();
  assert.equal(r.view.snapshot().flying, 0);
  await run(r, FLY_MS + 200, 10);
  assert.equal(r.counts.compareDraws - r0, 3, 'one paint each');
  assert.equal(r.view.cols.get('run').comp.own, false);
});

test("each its own while the engine works shows the one map; a column with too few cells for its own map says so, as do own maps that couldn't be made", async () => {
  const keysOf = keys => keys;
  // The run's column first (no own maps in that call), then every column with its own map but the imported one's.
  const s = setup({ map: null, reduced: true, points: ownPoints,
    reply: (n, keys) => (n < 3 ? status(keysOf(keys), { state: 'working', columns: keys.map((k, i) => (i === 0 ? column(k) : pending(k))) })
      : ownStatus(keys, { own: keys.filter(k => k !== 'cell_boundaries') })) });
  s.open(); await run(s, 600, 50);
  assert.equal(s.view.state, 'working');
  const first = s.view.cols.get('run');
  assert.equal(first.comp?.own, false, 'meanwhile, the one map'); assert.deepEqual(s.view.snapshot().columns.map(c => c.points), [6, 0, 0]);
  await run(s, 1500, 50);
  assert.equal(s.view.state, 'ready');
  assert.deepEqual(s.view.snapshot().columns.map(c => [c.points, c.own]), [[6, true], [6, true], [0, false]]);
  assert.equal(col(s, 'cell_boundaries').one('.cmp-wait').visibleText().trim(), T.ownNone);
  assert.ok(!s.calls.points.some(c => c[1] === 2 && c[3]), 'nothing asked of a column with no map of its own');
  s.root.one('button[data-map=one]').click(); await run(s, 200, 50);
  assert.deepEqual(s.view.snapshot().columns.map(c => c.points), [6, 6, 6], 'one map: every column');
  assert.equal(visible(col(s, 'cell_boundaries').one('.cmp-wait')), false);
  // Own maps that could not be made this time (the engine's own_error), or one that could not be downloaded.
  const f = setup({ map: null, reduced: true, points: ownPoints,
    reply: (n, keys) => ({ ...status(keys), own_layout: null, own_bounds: null, own_error: 'the GPU worker ran out of memory' }) });
  f.open(); await run(f, 1000, 50);
  assert.deepEqual(f.view.snapshot().columns.map(c => c.points), [0, 0, 0]);
  assert.ok(FIRST.every(k => col(f, k).one('.cmp-wait').visibleText().trim() === T.ownFailed));
  const g = setup({ map: null, reduced: true, reply: (n, keys) => ownStatus(keys),
    points: (key, index, what) => (what === 'clusters' && key === 'run' ? Promise.reject(Object.assign(new Error('gone'), { status: 404 })) : ownPoints(key, index, what)) });
  g.open(); await run(g, 1000, 50);
  assert.deepEqual(g.view.snapshot().columns.map(c => c.own), [false, true, true]);
  assert.equal(col(g, 'run').one('.cmp-wait').visibleText().trim(), T.ownFailed);
  // The engine says ready before the last layout's points have come: the one map stays meanwhile, without a word.
  const held = [];
  const slow = setup({ map: null, reduced: true,
    points: (key, index, what) => (slow.view.state === 'ready' ? new Promise(resolve => held.push(() => resolve(ownPoints(key, index, what)))) : ownPoints(key, index, what)),
    reply: (n, keys) => (n < 3 ? status(keys, { state: 'working', columns: keys.map((k, i) => (i === 0 ? column(k) : pending(k))) }) : ownStatus(keys)) });
  slow.open(); await run(slow, 1500, 50);
  assert.equal(slow.view.state, 'ready'); assert.ok(held.length > 0, 'the last layout on its way');
  assert.deepEqual(slow.view.snapshot().columns.map(c => [c.points, c.own]), [[6, false], [0, false], [0, false]], 'the one map meanwhile');
  assert.equal(visible(col(slow, 'run').one('.cmp-wait')), false, 'no word meanwhile');
  held.forEach(release => release()); await run(slow, 500, 50);
  assert.deepEqual(slow.view.snapshot().columns.map(c => [c.points, c.own]), [[6, true], [6, true], [6, true]]);
});
