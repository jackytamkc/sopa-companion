import test from 'node:test';
import assert from 'node:assert/strict';
import {
  QcDeeper, newestAtOrBefore, rankOrder, motionProgress, normalise, explain, sourceRows, basisTooltip, normaliseWhere, whereModel,
  SPLIT_MIN, WHERE_FEW,
  APPLY_MS, MAX_PAINTS, STEP_MS, WORKING_MS, ASK_MS, RETRY_MS, RANK_RATIO, STORAGE_KEY, COUNTERS, CHIPS,
  FOLLOW_RENEW_MS, GROUPS_LAG, groupsLag, KEEP_CELLS, noStartLine, NO_START_TEXT, NO_START_ENDED_TEXT, onPrior, STALE_MS,
} from '../src/qc-deeper.js';
import { allTexts, barLabels, cantRunText, deep, headlinePercent, DEEP_TEXT } from '../src/qc-deeper-copy.mjs';
import { bannedWords, cardLine, QC_TEXT, oldProsegTooltip } from '../src/qc-copy.mjs';
import { QcPanel } from '../src/qc-panel.js';
import { readFileSync } from 'node:fs';

// Go deeper's window: the rules it shows by (pure), then the window itself
// in a small DOM with a virtual clock, a recording engine and the page's paint batch.

// ---- a small DOM ----------------------------------------------------------------------------------------------------
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
    emit(t, extra = {}) {
      const e = { type: t, target: this, button: 0, preventDefault() { e.prevented = true; }, stopPropagation() { e.stopped = true; }, ...extra };
      for (const fn of this.listeners.get(t) || []) fn(e);
      return e;
    }
    click() { return this.onclick?.({ target: this, stopPropagation() {}, preventDefault() {} }); }
    contains(n) { while (n) { if (n === this) return true; n = n.parentElement; } return false; }
    closest(tag) { let n = this; while (n) { if (n.tagName === tag.toUpperCase()) return n; n = n.parentElement; } return null; }
    focus() { doc.activeElement = this; }
    getContext() { return this.ctx ??= recordingContext(); }
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
    /** The text a reader sees: hidden nodes left out. */
    visibleText() { return this.hidden ? '' : this._text + this.childNodes.map(c => c.nodeType === 3 ? c.textContent : c.visibleText()).join(' '); }
  }
  doc.createElement = tag => new Node(tag);
  doc.createTextNode = text => { const n = new Node('#text'); n._text = String(text); return n; };
  doc.body = new Node('body');
  const host = new Node('main'); doc.body.append(host);
  return { doc, host };
}
function recordingContext() {
  const calls = [];
  return new Proxy({ calls }, { get: (t, k) => k in t ? t[k] : (...args) => { calls.push([k, ...args]); return k === 'measureText' ? { width: 30 } : undefined; }, set: () => true });
}
/** A virtual clock: timers fire as time advances; animation frames run at once. */
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

// ---- results (the worker's shapes; the cells' and genes' names made up) ----------------------------------------------
const CELLS = ['c0', 'ffaaaaaa-1', 'ffbbbbbb-1', 'ggcccccc-1', 'ggaaaaaa-1', 'ggbbbbbb-1', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11'];
/** ffaaaaaa-1 at iteration 1201 (cell 1): 46 %, GXPN 31 of 39, its neighbour of another kind ffbbbbbb-1. */
function mixed(extra = {}) {
  return { v: 1, run: 'r1', seq: 12, iter: 1201, cell: 1, model: 'm1', state: 'ready', transcripts: 682, outside: 0, code: 116,
    parts: [0.539, 0.46, 0.001], rough: false,
    genes: [['GXPN', 39, 7.9, 31.1, 1, 0], ['GYL', 11, 5.1, 5.9, 0, 0], ['GXC', 10, 1.6, 8.4, 1, 0], ['GYHNN', 9, 5.5, 3.5, 0], ['GXDN', 7, 2.3, 4.7, 0],
      ['GYDNNN', 4, 0.7, 3.3, 1], ['GYJ', 4, 0.9, 3.1, 1], ['GYCNNN', 3, 0.5, 2.5, 1], ['GXT', 3, 0.7, 2.3, 1], ['GXANN', 2, 0.3, 1.7, 1],
      ['GXRN', 2, 0.35, 1.65, 1], ['GXEN', 4, 4, 0, 0], ['GZCNN', 4, 4, 0, 0], ['GYSNN', 1, 1, 0, 0],
      // The rest of the RNA from neighbours, spread thinly over genes the cell also makes.
      ...Array.from({ length: 60 }, (_, i) => [`G${String(i).padStart(2, '0')}`, 2, 1, 1, 0])],
    kind: ['GYSNN', 'GYRN', 'GXWN', 'GZBNNNN'], own_genes: [['GYHNN', 5.5, 9], ['GXEN', 4, 4], ['GZCNN', 4, 4], ['GYWNN', 3, 3], ['GYZNN', 3, 3], ['GXS', 3, 3], ['X7', 2, 2]],
    same_kind_share: 0.02,
    neighbours: [
      { cell: 2, um: 25.2, tx: 300, genes: [['GXPN', 31.1], ['GXC', 8.4], ['GYL', 5.9], ['GXDN', 4.7]], kind: ['GXPN', 'GXDN', 'GXT'], similar: 0.31, known: true },
      ...[6, 7, 8, 9, 10, 11, 3, 4, 5].map((cell, k) => ({ cell, um: 30 + k, tx: 0.4, genes: [['GYHNN', 0.2]], kind: ['GXHNN'], similar: 0.5, known: true })),
    ],
    learned: 'yes', start_change: { points: 44.1, band: 3, draws: 8, now_tx: 682, start_tx: 161 }, ms: 1.4, ...extra };
}
/** ffcccccc-1 (cell 3): little or no mixing, the model's floor. */
function little(extra = {}) {
  return mixed({ code: 2, parts: [0.989, 0.01, 0.001], transcripts: 824, genes: [['GXDN', 15, 14.8, 0.2, 0], ['GXPN', 13, 12.8, 0.2, 0]],
    same_kind_share: 0, neighbours: mixed().neighbours.map(n => ({ ...n, tx: 0.5 })), start_change: { points: 0.1, band: 3, draws: 8 }, ...extra });
}
/** ggdddddd-1 (cell 4): 57 %, 81 % of it looks like its own kind; three genes can still be shown. */
function lookalike(extra = {}) {
  return mixed({ code: 143, parts: [0.429, 0.57, 0.001], transcripts: 1396, same_kind_share: 0.81, start_change: { points: 37.5, band: 3.4, draws: 8 },
    genes: [['GXPN', 29, 10.8, 18.2, 0], ['GYPNNN', 5, 0.1, 4.9, 1], ['GZCNN', 4, 0.9, 3.1, 1], ['GYF', 3, 0.2, 2.8, 1]],
    neighbours: [
      { cell: 2, um: 17, tx: 460, genes: [['GXPN', 12]], kind: ['GZA', 'GXHNN', 'GXXNN'], similar: 0.95, known: true },
      { cell: 3, um: 20, tx: 180, genes: [['GXPN', 5]], kind: ['GXHNN', 'GXKN', 'GYAN'], similar: 0.93, known: true },
      { cell: 4, um: 19, tx: 125, genes: [['GYPNNN', 4.9], ['GZCNN', 3.1], ['GYF', 2.8]], kind: ['GYPNNN', 'GYF', 'GXZNNN'], similar: 0.4, known: true },
      { cell: 5, um: 15, tx: 30, genes: [['GZDNN', 1]], kind: ['GYXN', 'GXVNN', 'GZCNN'], similar: 0.5, known: false },
      ...[6, 7, 8, 9, 10, 11].map(cell => ({ cell, um: 30, tx: 0.5, genes: [], kind: ['GXHNN'], similar: 0.96, known: true })),
    ], ...extra });
}

// The results are shaped as real cells' are but name no real gene (a set of genes can tell the tissue a sample came
// from): made-up names keep each name's length and order (GXPN; G00…G59 and X7 fill the rest); the cells' ids are made
// up too (page-privacy.test.mjs).
test('the results name made-up genes only', () => {
  const MADE_UP = /^(?:G[XYZ][A-Z](?:N*|-N+)|G\d\d|X\d)$/;
  const names = [mixed(), little(), lookalike()].flatMap(r => [...[...r.genes, ...r.own_genes].map(g => g[0]), ...r.kind,
    ...r.neighbours.flatMap(n => [...n.genes.map(g => g[0]), ...n.kind])]);
  assert.ok(names.length > 30);
  assert.deepEqual([...new Set(names)].filter(name => !MADE_UP.test(name)), []);
});

// ---- the rules (pure) -------------------------------------------------------------------------------------------------
test('the newest result at or before the frame on screen, never a later one; frame 0 is the start line', () => {
  const keys = ['start', 3, 7, 12];
  assert.equal(newestAtOrBefore(keys, 0), 'start');
  assert.equal(newestAtOrBefore(keys, 2), null, 'the start line never stands for a later frame');
  assert.equal(newestAtOrBefore(keys, 7), 7); assert.equal(newestAtOrBefore(keys, 11), 7); assert.equal(newestAtOrBefore(keys, 99), 12);
  assert.equal(newestAtOrBefore([12], 11), null, 'a later result is never shown');
  assert.equal(newestAtOrBefore([3], 'start'), null); assert.equal(newestAtOrBefore([3], -1), null);
});

test('ranks change only when a value passes 1.25 x the one above; new items enter by value; gone ones leave', () => {
  const v = { a: 10, b: 9, c: 8 };
  assert.deepEqual(rankOrder([], ['a', 'b', 'c'], k => v[k]), ['a', 'b', 'c']);
  Object.assign(v, { b: 12 });   // 12 < 1.25 x 10: b stays below a
  assert.deepEqual(rankOrder(['a', 'b', 'c'], ['a', 'b', 'c'], k => v[k]), ['a', 'b', 'c']);
  Object.assign(v, { b: 12.6 });   // past 1.25 x 10
  assert.deepEqual(rankOrder(['a', 'b', 'c'], ['a', 'b', 'c'], k => v[k]), ['b', 'a', 'c']);
  assert.deepEqual(rankOrder(['a', 'b'], ['b', 'd'], k => ({ b: 5, d: 7 })[k]), ['d', 'b'], 'a new item enters where its value puts it');
  assert.equal(RANK_RATIO, 1.25);
});

test('motion: at most 4 paints an update on the stage\'s 150 / 300 / 450 ms steps; the first one already moves', () => {
  assert.equal(MAX_PAINTS, 4); assert.equal(STEP_MS, 150);
  const count = [0, 150, 300, 450].map(t => motionProgress(t, 450));
  assert.ok(count[0] > 0 && count[0] < count[1] && count[1] < count[2] && count[2] < 1 && count[3] === 1);
  assert.equal(motionProgress(300, 300), 1);
});

test('a v1 result is checked: wrong shapes are refused, a gene\'s loose part is what is left', () => {
  assert.equal(normalise(null), null); assert.equal(normalise({ state: 'ready' }), null); assert.equal(normalise({ ...mixed(), v: 2 }), null);
  assert.equal(normalise({ ...mixed(), code: 255 }), null, 'a ready result has a judged byte');
  const r = normalise(mixed());
  assert.equal(r.genes[0].gene, 'GXPN'); assert.ok(Math.abs(r.genes[0].loose) < 1e-9);
  assert.equal(r.neighbours[0].same, false); assert.equal(normalise(lookalike()).neighbours[0].same, true, 'similar >= 0.9: the same kind');
  assert.equal(normalise({ ...mixed(), neighbours: [{ cell: 2, tx: 1, similar: true }] }).neighbours[0].same, true);
  const few = normalise({ v: 1, state: 'too_few', seq: 'start', cell: 1, transcripts: 61, genes: [['GYVNN', 11], ['GXBNN', 2]] });
  assert.equal(few.seq, 'start'); assert.deepEqual(few.genes, [['GYVNN', 11], ['GXBNN', 2]]);
});

test('the step line says what it compares with the cell\'s own two numbers (start_change now and start: normalise keeps them), what they are in its tooltip (P2)', async () => {
  // A saved iteration: the headline on the count proseg saves (45 %), the small line's start iteration 0's (41 %), the
  // change like for like on the averaged count at the same depth: 38 % now, 43 % at the start.
  const sc = { points: -4.62, band: 3, draws: 0, now_tx: 600, start_tx: 410, direction: 'fewer', now: 38.37, start: 42.99 };
  const r = normalise(mixed({ code: 112, count: 'sampled', start_change: sc }));
  assert.deepEqual(r.start_change, { ...sc, held: false });
  for (const bad of [{ now: 'x', start: 43 }, { now: Number.NaN }, {}]) {
    const kept = normalise(mixed({ start_change: { points: -4.62, band: 3, ...bad } })).start_change;
    assert.deepEqual([kept.now, kept.start], [null, bad.start === 43 ? 43 : null], JSON.stringify(bad));
  }
  const m = explain(r, { name: c => CELLS[c] ?? `cell ${c}`, label: c => c + 1, startCode: 102 });
  assert.equal(m.stepLine, 'Looks mixed · ▼ less, like for like: 38 % now, 43 % at the start');
  assert.equal(m.stepTip, `${DEEP_TEXT.deepStepsTooltip} ${DEEP_TEXT.deepLikeNumbersTooltip} ${DEEP_TEXT.deepStartSaved}`);
  // An answer without them (kept from before them): its line and tooltip as before.
  const old = explain(normalise(mixed({ code: 112, start_change: { points: -4.62, band: 3, direction: 'fewer' } })), { startCode: 102 });
  assert.equal(old.stepLine, 'Looks mixed · less than at the start · like for like'); assert.equal(old.stepTip, DEEP_TEXT.deepStepsTooltip);
  // A word held a few seconds (a followed cell's: its answer's own numbers, 44 % now against 43 %, no longer fit "less"):
  // the words without the numbers, nothing said of numbers not shown; that it is held, and what this answer alone reads
  // (F2), in its tooltip.
  const held = explain(normalise(mixed({ code: 112, start_change: { ...sc, points: 0.8, direction: 'fewer', held: true, now: 43.8, start: 42.99 } })), { startCode: 102 });
  assert.equal(held.stepLine, 'Looks mixed · ▼ less, like for like');
  assert.equal(held.stepTip, `${DEEP_TEXT.deepStepsTooltip} ${deep('deepLikeHeld', { now: DEEP_TEXT.deepLikeSame })}`);
  // In the window: the line and its tooltip.
  const w = make({ live: false, run: { state: 'paused' }, seq: 12, answer: body => ready(mixed({ seq: body.seq, code: 112, count: 'sampled', start_change: sc })) });
  w.deeper.open(2); w.clock.advance(ASK_MS); await settle();
  const step = w.root.one('p.qc-deeper-step');
  assert.match(step.textContent, /Looks mixed · ▼ less, like for like: 38 % now, 43 % at the start/);
  assert.equal(step.title, `${DEEP_TEXT.deepStepsTooltip} ${DEEP_TEXT.deepLikeNumbersTooltip} ${DEEP_TEXT.deepStartSaved}`);
});

test('at a rest frame the step line is that iteration\'s own word with its own two numbers, never one held from earlier answers; a held word while the run plays says it is held (F2)', async () => {
  // The audit's cell at a pause (it. 432, count "sampled"): the engine held "fewer" for the followed cell, with an earlier
  // answer's numbers (26.51 / 32.04), while this answer's own points (-2.09) are inside the band.
  const heldAtPause = { points: -2.09, band: 5, draws: 0, now_tx: 401.3, start_tx: 405.3, direction: 'fewer', held: true, now: 26.51, start: 32.04 };
  const at = (start_change, count = 'sampled') => explain(normalise(mixed({ code: 112, count, start_change })), { startCode: 102 });
  const rest = at(heldAtPause);
  assert.equal(rest.stepLine, 'Looks mixed · same as at the start · like for like', 'its own word, no earlier answer\'s numbers beside it');
  assert.equal(rest.stepTip, `${DEEP_TEXT.deepStepsTooltip} ${DEEP_TEXT.deepStartSaved}`, 'not held: nothing said of a held word');
  // With its own numbers (the engine lets the held word go there): the own word and its two numbers.
  const own = at({ ...heldAtPause, direction: 'same', held: false, now: 29.48, start: 31.57 });
  assert.equal(own.stepLine, 'Looks mixed · about the same, like for like: 29 % now, 32 % at the start');
  assert.equal(own.stepTip, `${DEEP_TEXT.deepStepsTooltip} ${DEEP_TEXT.deepLikeNumbersTooltip} ${DEEP_TEXT.deepStartSaved}`);
  // ... and a word the worker kept near the band (its numbers its own): this iteration's own word with them.
  assert.equal(at({ points: -3.1, band: 5, direction: 'fewer', held: true, now: 28.9, start: 32 }).stepLine,
    'Looks mixed · about the same, like for like: 29 % now, 32 % at the start');
  // While the run plays (the averaged count) a held word stays, without numbers, and its tooltip says it is held and
  // what this answer alone reads.
  const playing = at(heldAtPause, 'averaged');
  assert.equal(playing.stepLine, 'Looks mixed · ▼ less, like for like');
  assert.equal(playing.stepTip, `${DEEP_TEXT.deepStepsTooltip} ${deep('deepLikeHeld', { now: DEEP_TEXT.deepLikeSame })}`);
  assert.match(playing.stepTip, /held from earlier answers.*this answer alone reads about the same, like for like\./);
  const kept = at({ points: 2.4, band: 5, direction: 'more', held: true }, 'averaged');
  assert.equal(kept.stepLine, 'Looks mixed · more than at the start · like for like', 'an answer from before the numbers: its words as before');
  assert.equal(kept.stepTip, `${DEEP_TEXT.deepStepsTooltip} ${deep('deepLikeHeld', { now: DEEP_TEXT.deepLikeSame })}`);
  assert.equal(at({ ...heldAtPause, held: false, points: -5.53 }, 'averaged').stepTip, `${DEEP_TEXT.deepStepsTooltip} ${DEEP_TEXT.deepLikeNumbersTooltip}`,
    'a word that is its own says nothing of holding');
  // ... whichever way this answer alone reads past its band: more than at the start under a held "less", less under a
  // held "more".
  const held = 'This word is held from earlier answers so it doesn\'t flicker; this answer alone reads';
  const up = at({ points: 6.2, band: 5, direction: 'fewer', held: true, now: 30.1, start: 32 }, 'averaged');
  assert.deepEqual([up.stepLine, up.stepTip], ['Looks mixed · ▼ less, like for like', `${DEEP_TEXT.deepStepsTooltip} ${held} more than at the start, like for like.`]);
  const down = at({ points: -6.2, band: 5, direction: 'more', held: true, now: 33.9, start: 32 }, 'averaged');
  assert.deepEqual([down.stepLine, down.stepTip], ['Looks mixed · ▲ more, like for like', `${DEEP_TEXT.deepStepsTooltip} ${held} less than at the start, like for like.`]);
  // In the window, paused on the pause's answer: the line and its tooltip.
  const w = make({ live: false, run: { state: 'paused' }, seq: 12, answer: body => ready(mixed({ seq: body.seq, code: 112, count: 'sampled', start_change: heldAtPause })) });
  w.deeper.open(2); w.clock.advance(ASK_MS); await settle();
  const step = w.root.one('p.qc-deeper-step');
  assert.match(step.textContent, /Looks mixed · same as at the start · like for like/); assert.doesNotMatch(step.textContent, /less/);
  assert.equal(step.title, `${DEEP_TEXT.deepStepsTooltip} ${DEEP_TEXT.deepStartSaved}`);
});

test('the headline equals the cell card for every byte, and the bar\'s whole numbers add up to 100', () => {
  for (let code = 0; code <= 250; code++) {
    const p = headlinePercent(code), card = /(\d+)\s?%/.exec(cardLine(code, null, 'xenium'))?.[1];
    assert.equal(p, card, `byte ${code}`);
    for (const loose of [0.001, 0.004, 0.02]) {
      const share = (code + 0.5) / 250, labels = barLabels(code, [Math.max(0, 1 - share - loose), share, loose]);
      assert.equal(labels.own + labels.nb + labels.loose, 100, `byte ${code}, loose ${loose}`);
      assert.equal(String(labels.nb), p);
      assert.equal(labels.looseUnder, loose * 100 < 1);
    }
  }
  assert.equal(barLabels(116, [0.539, 0.46, 0.001]).texts.loose, 'Loose < 1 %');
  assert.equal(headlinePercent(255), null);
});

test('cell 1 (looks mixed): the copy deck filled with its numbers, the explanations in tooltips', () => {
  const r = normalise(mixed()), name = c => CELLS[c] ?? `cell ${c}`;
  const m = explain(r, { name, label: c => c + 1, startCode: 2 });
  assert.equal(m.number, '46 %');
  // The change since the start is the worker's like-for-like one, and the line says so
  // ("start 1 %" beside the small line can read the other way); how, in its tooltip.
  assert.equal(m.stepLine, 'Looks mixed · more than at the start · like for like');
  // The basis: proseg's count of the cell (its averaged count; the background it sets aside in the tooltip once the
  // frame's statistics are in: basisTooltip).
  assert.equal(m.basis, "Proseg's count (averaged): 682");
  assert.equal(m.basisTip, DEEP_TEXT.deepBasisNoBg);
  assert.equal(m.barLabel, 'Own 54 % · Neighbours 46 % · Loose < 1 %');
  // The neighbour genes' label's tooltip: the old heading with its count, and the spread note (the tagged genes hold less
  // than half of the RNA from neighbours).
  assert.equal(m.chipsTip, `Genes far more typical of a neighbour's kind, so they probably come from neighbours (8 here). ${DEEP_TEXT.deepSpread}`);
  // A chip's first number is rounded down: never more than the estimate (GYCNNN 2.5 of 3 reads "2 of 3").
  assert.deepEqual(m.chips.slice(0, 6).map(c => c.text), ['GXPN 31 of 39', 'GXC 8 of 10', 'GYDNNN 3 of 4', 'GYJ 3 of 4', 'GYCNNN 2 of 3', 'GXT 2 of 3']);
  const { rows, fold } = sourceRows(m);
  // A row's two lines: its name and share rounded down, never above the headline; the genes it seems to give, with
  // their transcripts from it (at most two), and its distance. Its words in full in its tooltip.
  assert.deepEqual(rows.map(x => [x.name, x.pctText, x.sub]), [['ffbbbbbb-1', '43 %', 'GXPN 31 · GXC 8 · 25 µm']]);
  assert.equal(rows[0].pct, 43);
  assert.equal(rows[0].tooltip, "ffbbbbbb-1 · makes GXPN · GXDN · GXT: about 43 % of this cell's RNA seems to come from here (a rough guide).");
  assert.equal(fold.text, '+9 more under 1 %');
  // The learned guide's note is the rows' heading's tooltip, not a line in every window.
  assert.equal(m.learnedNote, DEEP_TEXT.deepLearnedNote); assert.equal(m.sameCount, null);
  // Its own kind, and its top genes here (three, whole numbers): on screen only when no kind is named, else a sentence of
  // the kind's tooltip (fewer words; the table holds every gene).
  assert.equal(m.ownKind, 'Own kind: GYSNN · GYRN · GXWN · GZBNNNN');
  assert.deepEqual(m.ownGenes, { text: 'Top here: GYHNN 6 · GXEN 4 · GZCNN 4', tip: DEEP_TEXT.deepOwnGenesTooltip });
  assert.equal(m.ownTop, 'Its top genes here, mostly its own: GYHNN 6 · GXEN 4 · GZCNN 4.');
  assert.equal(m.loose, null, 'under 1 %: the legend says it');
  const loose = explain(normalise(mixed({ parts: [0.52, 0.46, 0.02] })), { name, label: c => c + 1 }).loose;
  assert.deepEqual(loose, { text: 'Loose RNA · 2 %',
    tip: 'The model puts about 2.0 % here in every cell of this sample, spread almost evenly over all genes, so no genes are listed.' });
  assert.equal(m.tableHeading, 'All 74 genes, split');
  assert.equal(m.table[0].gene, 'GXPN'); assert.equal(m.table[0].pct, '80');
  assert.equal(m.table[0].tooltip, 'GXPN: 39 transcripts · about 8 its own, 31 from neighbours (most from ffbbbbbb-1) (estimate)');
  for (const row of m.table) {   // the tooltip's whole numbers add up to the gene's transcripts
    const [, tx, own, nb] = /: (\d+) transcripts · about (\d+) its own, (\d+) from/.exec(row.tooltip).map(Number);
    assert.ok(own + nb <= tx && tx - own - nb <= 1, row.tooltip);
  }
});

test('cells 3 and 4: the floor and "hardly anything" under 5 %; the look-alike lines and the same-kind count', () => {
  const name = c => CELLS[c] ?? `cell ${c}`;
  const l = explain(normalise(little()), { name, label: c => c + 1, startCode: 2 });
  assert.equal(l.stepLine, 'Little or no mixing seen · same as at the start · like for like');
  assert.ok(l.after.some(n => n.text === DEEP_TEXT.deepFloor && n.tip === DEEP_TEXT.deepFloorTooltip));
  assert.equal(l.none, true); assert.equal(l.noneText, 'Too little from neighbours to list any');
  const a = explain(normalise(lookalike()), { name, label: c => c + 1, startCode: 47 });
  assert.equal(a.stepLine, 'Looks mixed · more than at the start · like for like');
  assert.deepEqual(a.chips.map(c => c.text), ['GYPNNN 4 of 5', 'GZCNN 3 of 4', 'GYF 2 of 3']);
  // Most of its RNA from neighbours looks like its own kind: under the headline in a few words, the different kind's part
  // in its tooltip; the section's own line is then not repeated.
  assert.deepEqual(a.notes, [{ text: "Mostly its own kind: can't be checked gene by gene",
    tip: "Most of the RNA from neighbours looks like this cell's own kind: only about 10 % of its RNA looks like a different kind of cell, which genes can back up. The rest can't be checked." }]);
  assert.equal(a.sameLine, null);
  const quarter = explain(normalise(lookalike({ same_kind_share: 0.3 })), { name, label: c => c + 1 });
  assert.deepEqual(quarter.sameLine, { text: 'About 30 % looks like its own kind', tip: DEEP_TEXT.deepSameKindShareTooltip });
  assert.deepEqual(quarter.notes, []);
  assert.equal(a.sameCount, '8 of its 10 nearest neighbours look the same kind.', 'a sentence for the Neighbours label\'s tooltip');
  const { rows, partly } = sourceRows(a);
  assert.deepEqual(rows.map(x => x.name), ['ffbbbbbb-1', 'ggcccccc-1', 'ggaaaaaa-1', 'ggbbbbbb-1'], 'every neighbour holding 1 % or more, by share');
  assert.equal(partly, '1 of these neighbours is new since the model learned this cell, so its share is a rougher guide.');
  // Nothing tagged and three quarters or more look-alike: the line under the headline says it all (no label, no line).
  const all = explain(normalise(lookalike({ genes: [['GXPN', 29, 10.8, 18.2, 0]] })), { name, label: c => c + 1 });
  assert.equal(all.sameLine, null); assert.equal(all.foreignNone, null); assert.equal(all.chipsTip, null);
  assert.equal(all.notes[0].text, DEEP_TEXT.deepLookAlike);
  const none = explain(normalise(mixed({ genes: [['GXPN', 39, 7.9, 31.1, 0]] })), { name, label: c => c + 1 });
  assert.equal(none.foreignNone, DEEP_TEXT.deepForeignNone);
  const notLearned = explain(normalise(mixed({ learned: 'no' })), { name, label: c => c + 1, startCode: 255 });
  assert.equal(notLearned.learnedNote, DEEP_TEXT.deepNotLearnedNote);
  assert.deepEqual(notLearned.notes, [{ text: DEEP_TEXT.deepStartUnjudged, tip: '' }]); assert.equal(notLearned.stepLine, 'Looks mixed');
  const few = explain(normalise({ v: 1, state: 'too_few', seq: 3, cell: 1, transcripts: 70, genes: [['GXJNN', 3], ['GYBNNN', 2], ['GYF', 2]] }));
  assert.deepEqual(few.lines, ['Too few transcripts to split: proseg counts 70 for this cell in this iteration, in the genes the model knows, and at least 100 are needed.',
    'Its genes so far (not split): GXJNN 3 · GYBNNN 2 · GYF 2', DEEP_TEXT.deepTooFewLater]);
  const rough = explain(normalise(mixed({ rough: true, transcripts: 150, outside: 4 })), { name, label: c => c + 1 });
  assert.deepEqual(rough.after, [
    { text: DEEP_TEXT.deepFew, tip: "With 150 transcripts, single genes hold only one or two each: treat any one gene's split as rough." },
    { text: DEEP_TEXT.deepRough, tip: DEEP_TEXT.deepRoughTooltip },
    { text: '4 transcripts of unknown genes left out', tip: DEEP_TEXT.deepOutsideTooltip }]);
});

test('every word the window can show passes the check\'s banned-word rule; the method is named once, the user\'s files never', () => {
  for (const text of allTexts()) {
    assert.deepEqual(bannedWords(text), [], text);
    if (/resolvi/i.test(text)) assert.ok([DEEP_TEXT.deepHow1, DEEP_TEXT.deepHowCredit].includes(text), text);
    assert.doesNotMatch(text, /notebook|part\s*3/i, text);
  }
});

// ---- the window -------------------------------------------------------------------------------------------------------
function make({ answer = null, qc = { status: 'live', enabled: true }, run = {}, seq = 12, live = true, stage = { width: 1200, height: 800 }, storage = null, reduced = false } = {}) {
  const { doc, host } = dom(), c = clock(), calls = { posts: [], stops: 0, focus: [], selects: [], finds: [], switches: 0, learns: 0, logs: [], boxes: [], changes: 0, retries: 0, tests: [] };
  const counters = {};
  const harness = { count(name, n = 1) { counters[name] = (counters[name] ?? 0) + n; } };
  const batch = { rides: new Map(), marks: new Map(), ride(k, fn) { this.rides.set(k, fn); }, mark(k, fn) { this.marks.set(k, fn); },
    flush() { const fns = [...this.marks.values(), ...this.rides.values()]; this.rides.clear(); this.marks.clear(); fns.forEach(fn => fn()); } };
  const frames = Array.from({ length: 20 }, (_, i) => ({ seq: i, iter: i * 100 + 1 }));
  const ctx = { run: { id: 'r1', sample: 's1', state: 'running', ...run }, frames, seq, live, selected: 2, cells: CELLS,
    qc, startSource: 'xenium', calm: false, view3d: false, stage };
  const api = {
    qcDeeper: async (id, body) => { calls.posts.push(body); const a = typeof answer === 'function' ? answer(body) : answer; if (a instanceof Error) throw a; return a; },
    qcDeeperStop: async () => { calls.stops++; return null; },
    qcDeeperHistory: async () => ({ model: 'm1', start: 2, points: [[1, 2], [601, 3]] }),
    learnSampleDetail: async () => { calls.learns++; return { state: 'ready', model: { id: 'm1', deep: false }, detail: { state: 'queued', progress: null, eta_s: 120 } }; },
  };
  const store = storage ?? (() => { const m = new Map(); return { getItem: k => m.get(k) ?? null, setItem: (k, v) => m.set(k, v), removeItem: k => m.delete(k), m }; })();
  const deeper = new QcDeeper(host, { api, batch, harness, clock: c, storage: store, context: () => ctx, reducedMotion: () => reduced,
    name: label => CELLS[label - 1] ?? `cell ${label - 1}`, group: () => null, swatch: () => '#123456', startCode: () => 2,
    onFocus: f => calls.focus.push(f), onSelect: l => calls.selects.push(l), onFind: l => calls.finds.push(l), onSwitchOn: () => calls.switches++,
    onRetryLearning: () => calls.retries++, onChange: () => calls.changes++, onBox: b => calls.boxes.push(b), onTest: (label, name) => calls.tests.push([label, name]),
    log: t => calls.logs.push(t) });
  return { deeper, doc, host, clock: c, calls, counters, batch, ctx, api, store, root: deeper.root };
}
const ready = r => ({ state: 'ready', seq: r.seq, cell: r.cell, model: 'm1', deep: true, result: r });

test('its button opens the window on its cell; it asks the engine to follow that cell; the result fills it in', async () => {
  const w = make({ answer: body => ready(mixed({ seq: body.seq })) });
  const button = w.doc.createElement('button'); w.doc.body.append(button);
  w.deeper.bindButton(button, 2);
  assert.equal(button.textContent, 'Go deeper ›'); assert.equal(button.dataset.action, 'go-deeper');
  assert.equal(button.title, DEEP_TEXT.deepButtonTooltip);
  button.click(); await settle();
  assert.equal(w.root.hidden, false); assert.equal(w.root.id, 'qc-deeper');
  assert.equal(w.root.getAttribute('role'), 'dialog'); assert.equal(w.root.getAttribute('aria-modal'), 'false');
  assert.deepEqual(w.calls.posts, [{ seq: 12, cell: 1, follow: true }], 'the frame on screen, the cell as label - 1, followed live');
  assert.equal(w.root.dataset.state, 'ready'); assert.equal(w.root.dataset.cell, 'ffaaaaaa-1');
  assert.equal(w.root.one('.qc-deeper-number').textContent, '46 %');
  assert.equal(w.root.one('.qc-deeper-frame').textContent, 'Iteration 1,201 · live');
  // At most four chips, then "+k more".
  assert.deepEqual(w.root.all('span.qc-deeper-chip').map(n => n.dataset.gene), ['GXPN', 'GXC', 'GYDNNN', 'GYJ']);
  assert.equal(w.root.one('button.qc-deeper-more').textContent, '+4 more');
  const rows = w.root.all('li.qc-deeper-source');
  assert.deepEqual(rows.map(r => r.dataset.cell), ['ffbbbbbb-1']);
  assert.ok(rows[0].one('button[data-action=show]') && rows[0].one('button[data-action=deeper]'));
  assert.equal(w.counters.qcDeeperOpened, 1); assert.equal(w.counters.qcDeeperApplied, 1);
  for (const name of COUNTERS) assert.ok(name in w.counters, name);
  // The stage's focus: the cell, its source (a different kind), no same-kind neighbour here; only the cells involved
  // stay lit, the rest greyed out (the cell and its shown rows, dim 1, the stage easing it in).
  const focus = w.calls.focus.at(-1);
  assert.equal(focus.cell, 2); assert.deepEqual(focus.sources.map(s => s.label), [3]); assert.deepEqual(focus.same, []);
  assert.deepEqual(focus.lit, [2, 3]); assert.equal(focus.dim, 1);
  assert.equal(focus.apply, true, "a result painted: the page may draw its arrows with its next paint batch (main.js deeperGeometry)");
  assert.deepEqual(w.deeper.snapshot().focus, { cell: 2, sources: [3], same: [], lit: [2, 3], dim: 1 });
  // The button says the window is open on its cell; pressed again it closes the window.
  w.deeper.bindButton(button, 2); assert.equal(button.textContent, 'Go deeper (open)'); assert.equal(button.getAttribute('aria-pressed'), 'true');
  button.click(); await settle();
  assert.equal(w.root.hidden, true); assert.equal(w.calls.stops, 1, 'the engine stops following'); assert.equal(w.calls.focus.at(-1), null);
  assert.equal(w.doc.activeElement, button, 'focus returns to the button');
  for (const text of w.deeper.texts()) assert.deepEqual(bannedWords(text), [], text);
});

test('the mixing check\'s row for the selected cell: its "Go deeper ›" opens the window on that cell, says so, closes it and gets the keyboard back', async () => {
  const w = make({ answer: body => ready(mixed({ seq: body.seq })) });
  const host = w.doc.createElement('details'); w.doc.body.append(host);
  const panel = new QcPanel(host, { storage: null, bindDeeper: (button, label) => w.deeper.bindButton(button, label) });
  // As main.js updateCellRow: every frame, and whenever the window opens, closes or moves to another cell.
  const row = () => panel.setCell({ label: 2, name: 'ffaaaaaa-1', color: '#123456', line: cardLine(116, 2, 'xenium'), loading: false });
  row();
  const button = panel.deeperButton;
  assert.equal(button.parentElement, panel.cellRow); assert.equal(panel.cellRow.parentElement.parentElement, host, 'in the mixing check');
  assert.equal(button.textContent, 'Go deeper ›'); assert.equal(button.title, DEEP_TEXT.deepButtonTooltip);
  assert.equal(button.dataset.action, 'go-deeper'); assert.equal(button.getAttribute('aria-controls'), 'qc-deeper');
  assert.equal(button.getAttribute('aria-pressed'), 'false');
  button.click(); await settle();
  assert.equal(w.root.hidden, false); assert.equal(w.root.dataset.cell, 'ffaaaaaa-1'); assert.deepEqual(w.calls.posts, [{ seq: 12, cell: 1, follow: true }]);
  row();
  assert.equal(button.textContent, 'Go deeper (open)'); assert.equal(button.getAttribute('aria-pressed'), 'true');
  button.click(); await settle();
  assert.equal(w.root.hidden, true); assert.equal(w.calls.stops, 1);
  assert.equal(w.doc.activeElement, button, 'the keyboard back on the button it was opened with');
  row(); assert.equal(button.textContent, 'Go deeper ›'); assert.equal(button.getAttribute('aria-pressed'), 'false');
  assert.deepEqual(w.calls.logs.filter(t => /Go deeper/.test(t)), ['Go deeper from the mixing check: ffaaaaaa-1', 'Go deeper closed from the mixing check']);
  panel.dispose();
});

test('"Test a setting here…" sits beside Show on the image in one row: shown with it while the page can test the cell, whatever the run\'s state; it calls onTest(label, name)', async () => {
  // A run paused at its end (as a test wants it), the page able to test the selected cell (context testable).
  const w = make({ live: false, run: { state: 'paused' }, answer: body => ready(mixed({ seq: body.seq })) });
  w.ctx.testable = true;
  w.deeper.open(2); w.clock.advance(ASK_MS); await settle();
  const row = w.root.one('div.qc-deeper-where-row'), where = row?.one('button[data-action=where]'), entry = w.root.one('button[data-action=test-open]');
  assert.ok(row && where && entry, 'Show on the image and the entry, the harness\'s button[data-action=test-open]');
  // (Nodes by identity or by their data-action: a failing deepEqual of these nodes prints the whole page.)
  assert.ok(entry.parentElement === row, 'in one row with Show on the image'); assert.ok(w.root.one('section.qc-deeper-from').contains(row));
  assert.deepEqual(row.children.map(b => b.dataset.action), ['where', 'test-open'], 'Show on the image first');
  assert.deepEqual([row.hidden, where.hidden, entry.hidden, entry.disabled ?? false], [false, false, false, false], 'shown, never greyed out');
  assert.equal(entry.textContent, QC_TEXT.testOpen); assert.equal(entry.title, QC_TEXT.testOpenTooltip, 'the mixing check\'s own name and tooltip');
  assert.equal(w.deeper.snapshot().test, true);
  // Pressed: the mixing check's test of a setting on the window's cell, by its label and name.
  entry.click(); await settle();
  assert.deepEqual(w.calls.tests, [[2, 'ffaaaaaa-1']]);
  assert.ok(w.calls.logs.includes('test a setting from Go deeper: ffaaaaaa-1'));
  assert.equal(w.root.hidden, false, 'the window stays open');
  // Whatever the run's state (the form lists what stands in the way): finishing, finished, running.
  for (const state of ['finishing', 'done', 'running']) {
    w.ctx.run.state = state; w.deeper.changed(); await settle();
    assert.equal(entry.hidden, false, state);
  }
  // Only while the page can test this cell: it follows the selection (a cleared one keeps the window's cell, pinned).
  Object.defineProperty(w.ctx, 'testable', { configurable: true, get: () => w.ctx.selected > 0 });
  w.ctx.selected = 0; w.deeper.select(0);
  assert.deepEqual([where.hidden, entry.hidden, row.hidden], [false, true, false], 'Show on the image keeps the row');
  assert.equal(w.deeper.snapshot().test, false);
  w.ctx.selected = 2; w.deeper.select(2);
  assert.equal(entry.hidden, false);
  // No test wired on the page: never shown.
  Object.defineProperty(w.ctx, 'testable', { configurable: true, value: undefined, writable: true });
  w.deeper.changed(); await settle();
  assert.equal(entry.hidden, true);
  for (const t of w.deeper.texts()) assert.deepEqual(bannedWords(t), [], t);
  // It goes with Show on the image: under 5 % from neighbours (no rows), too few transcripts, the check learning the
  // sample (the window shows its own state; the panel's entry stays, its form says why), another cell, a closed window.
  // (The check off: beside the check's switch, the next test.)
  const without = [
    ['little', make({ live: false, answer: body => ready(little({ seq: body.seq })) })],
    ['too few', make({ live: false, answer: () => ({ state: 'ready', seq: 12, cell: 1, model: 'm1', result: { v: 1, state: 'too_few', seq: 12, iter: 1201, cell: 1, transcripts: 61, genes: [] } }) })],
    ['learning', make({ live: false, qc: { status: 'learning', enabled: true, progress: { step: 20, of: 50 }, eta_s: 61 }, answer: body => ready(mixed({ seq: body.seq })) })],
  ];
  for (const [why, v] of without) {
    v.ctx.testable = true;
    v.deeper.open(2); v.clock.advance(ASK_MS); await settle();
    assert.equal(v.root.one('button[data-action=test-open]').hidden, true, why);
    assert.equal(v.root.one('div.qc-deeper-where-row').hidden, true, why);
    assert.equal(v.deeper.snapshot().test, false, why);
  }
  w.deeper.close(); await settle();
  assert.deepEqual([entry.hidden, row.hidden, w.deeper.snapshot().test], [true, true, false]);
  w.ctx.testable = true; w.deeper.open(2); await settle();
  assert.equal(entry.hidden, false, 'open again: back');
  entry.click(); await settle();
  assert.deepEqual(w.calls.tests.at(-1), [2, 'ffaaaaaa-1']);
  // Made smaller (a strip): the row goes with the rest of the sections (qc-deeper.css), and the snapshot no longer
  // counts the entry as offered; made bigger again, it is.
  const css = readFileSync(new URL('../src/qc-deeper.css', import.meta.url), 'utf8');
  const small = /([^}]*)\{\s*display:none!important;\s*\}/g, hidden = [...css.matchAll(small)].map(m => m[1]).join(',');
  assert.match(hidden, /\.qc-deeper-small \.qc-deeper-where-row/); assert.match(hidden, /\.qc-deeper-small \.qc-deeper-test-open/);
  assert.equal(w.deeper.snapshot().test, true);
  w.deeper.setSmall(true);
  assert.deepEqual([w.deeper.snapshot().small, w.deeper.snapshot().test], [true, false], 'made smaller: not offered');
  w.deeper.setSmall(false);
  assert.equal(w.deeper.snapshot().test, true, 'bigger again: offered');
});

test('"Test a setting here…" for a paused run whose check is off: beside the check\'s switch, never greyed out; with the check on, beside Show on the image again', async () => {
  // A run paused at its end, the check off: the window offers the check's switch, and beside it the entry (the page can
  // test the cell; a press of its Run switches the check on first: check_off, a wait in Run's tooltip, since a test
  // starts from the check's suspects).
  let release = null;
  const pending = body => new Promise(resolve => { release = () => resolve(ready(mixed({ seq: body.seq }))); });
  const w = make({ live: false, run: { state: 'paused' }, qc: { status: 'off', enabled: false }, answer: pending });
  w.ctx.testable = true;
  w.deeper.open(2); w.clock.advance(ASK_MS); await settle();
  assert.equal(w.root.dataset.state, 'off'); assert.deepEqual(w.calls.posts, [], 'nothing is asked while the check is off');
  const off = w.root.one('div.qc-deeper-off-row'), entry = w.root.one('button[data-action=test-open]'), row = w.root.one('div.qc-deeper-where-row');
  assert.ok(off && entry && row);
  // Where the entry sits, by name (nodes compared by identity: a failing deepEqual of these nodes prints the whole page).
  const at = (v = w) => { const p = v.root.one('button[data-action=test-open]').parentElement;
    return p?.matches('div.qc-deeper-off-row') ? 'off row' : p?.matches('div.qc-deeper-where-row') ? 'where row' : 'elsewhere'; };
  assert.ok(w.root.one('div.qc-deeper-state').contains(off), 'in the state box, under the words');
  assert.deepEqual(off.children.map(b => b.dataset.action), ['switch-on', 'test-open'], 'the switch first, the entry beside it');
  assert.deepEqual([at(), entry.hidden, entry.disabled ?? false, row.hidden], ['off row', false, false, true], 'shown, never greyed out; no Show on the image');
  assert.equal(w.root.all('button[data-action=test-open]').length, 1, 'one entry, wherever it sits: the harness\'s button[data-action=test-open]');
  assert.equal(entry.textContent, QC_TEXT.testOpen); assert.equal(entry.title, QC_TEXT.testOpenTooltip);
  assert.equal(w.deeper.snapshot().test, true);
  for (const t of w.deeper.texts()) assert.deepEqual(bannedWords(t), [], t);
  entry.click(); await settle();
  assert.deepEqual(w.calls.tests, [[2, 'ffaaaaaa-1']], 'the form for the window\'s cell (its Run switches the check on first)');
  assert.equal(w.calls.switches, 0, 'the entry does not switch the check on');
  // Only while the page can test the cell; never while made smaller (its CSS hides it there too).
  w.ctx.testable = false; w.deeper.changed(); await settle();
  assert.deepEqual([entry.hidden, w.deeper.snapshot().test], [true, false]);
  w.ctx.testable = true; w.deeper.changed(); await settle();
  assert.equal(entry.hidden, false);
  w.deeper.setSmall(true); assert.equal(w.deeper.snapshot().test, false); w.deeper.setSmall(false);
  const css = readFileSync(new URL('../src/qc-deeper.css', import.meta.url), 'utf8');
  const hiddenSmall = [...css.matchAll(/([^}]*)\{\s*display:none!important;\s*\}/g)].map(m => m[1]).join(',');
  assert.match(hiddenSmall, /\.qc-deeper-small \.qc-deeper-test-open/, 'made smaller: hidden wherever it sits');
  // The check switched on: while the answer is on its way the entry is back in its row, hidden (the window says it is
  // working it out) ...
  w.root.one('button[data-action=switch-on]').click(); assert.equal(w.calls.switches, 1);
  w.ctx.qc = { status: 'live', enabled: true }; w.deeper.changed(); w.clock.advance(ASK_MS); await settle();
  assert.notEqual(w.root.dataset.state, 'off');
  assert.deepEqual([at(), entry.hidden, w.deeper.snapshot().test], ['where row', true, false], 'back in the row with Show on the image, hidden until it shows');
  // ... then beside Show on the image, as for any ready result.
  release(); await settle();
  assert.equal(w.root.dataset.state, 'ready');
  const where = row.one('button[data-action=where]');
  assert.deepEqual(row.children.map(b => b.dataset.action), ['where', 'test-open'], 'Show on the image first');
  assert.deepEqual([row.hidden, where.hidden, entry.hidden, w.deeper.snapshot().test], [false, false, false, true]);
  // Off again: beside the switch again (the result and Show on the image go).
  w.ctx.qc = { status: 'off', enabled: false }; w.deeper.changed(); await settle();
  assert.deepEqual([w.root.dataset.state, at(), entry.hidden, row.hidden], ['off', 'off row', false, true]);
  assert.equal(w.root.all('button[data-action=test-open]').length, 1);
  // Closed: hidden, back in its row; opened again with the check off: beside the switch.
  w.deeper.close(); await settle();
  assert.deepEqual([at(), entry.hidden, w.deeper.snapshot().test], ['where row', true, false]);
  w.deeper.open(2); w.clock.advance(ASK_MS); await settle();
  assert.deepEqual([w.root.dataset.state, at(), entry.hidden], ['off', 'off row', false]);
  entry.click(); await settle();
  assert.deepEqual(w.calls.tests.at(-1), [2, 'ffaaaaaa-1']);
  // Switched on while a full-screen view covers the page (Compare's): the window waits, then, uncovered, works the answer
  // out with the entry back in its row (the state box painted again leaves it out), hidden until Show on the image shows.
  const v = make({ live: false, run: { state: 'paused' }, qc: { status: 'off', enabled: false }, answer: pending });
  v.ctx.testable = true;
  v.deeper.open(2); v.clock.advance(ASK_MS); await settle();
  assert.deepEqual([v.root.dataset.state, at(v)], ['off', 'off row']);
  v.deeper.setCovered(true);
  v.ctx.qc = { status: 'live', enabled: true }; v.deeper.changed(); await settle();
  assert.deepEqual([v.root.dataset.state, at(v)], ['off', 'off row'], 'covered: nothing changes');
  v.deeper.setCovered(false); v.clock.advance(ASK_MS); await settle();
  assert.notEqual(v.root.dataset.state, 'off');
  assert.deepEqual([at(v), v.root.one('button[data-action=test-open]').hidden, v.deeper.snapshot().test], ['where row', true, false]);
  release(); await settle();
  assert.deepEqual([v.root.dataset.state, at(v), v.root.one('button[data-action=test-open]').hidden], ['ready', 'where row', false]);
});

test('the full result: headline, bar, legend, sections, the table (virtualised) and "How is this worked out?", the explanations in tooltips', async () => {
  const w = make({ answer: body => ready(mixed({ seq: body.seq })) });
  w.ctx.qc = { ...w.ctx.qc, sampleModel: { id: 'm1', cells: 42000, created: '2026-10-21T10:00:00Z', seconds: 420 } };
  w.deeper.open(2); await settle();
  const text = w.root.visibleText();
  for (const part of ['46 %', 'from neighbours', 'Looks mixed · more than at the start · like for like', "Proseg's count (averaged): 682", 'Own 54 %', 'Neighbours 46 %',
    'Loose < 1 %', 'Neighbour genes', 'Neighbours', 'ffbbbbbb-1', '43 %', 'GXPN 31 · GXC 8 · 25 µm', '+9 more under 1 %', 'Show on the image',
    'Own kind: GYSNN · GYRN · GXWN · GZBNNNN', 'All 74 genes, split', DEEP_TEXT.deepEstimate,
    'How is this worked out?'])
    assert.ok(text.includes(part), part);
  // Gone from the screen: the subtitle, the meaning paragraph, the blind spot, the % headings, the loose
  // RNA section under 1 %, the notebook, a row's third gene and the same-kind count (that, in the label's tooltip).
  for (const gone of [DEEP_TEXT.deepSubtitle, DEEP_TEXT.deepBlindSpot, DEEP_TEXT.deepNumberTooltip, 'From neighbours · 46 %', 'Its own · 54 %',
    'Loose RNA', 'notebook', DEEP_TEXT.deepLearnedNote, 'GYL 5', 'nearest', 'Top here']) assert.ok(!text.includes(gone), gone);
  assert.doesNotMatch(text, /usually a little cleaner|inside its outline|Proseg keeps/, 'no outline count, no second bar, no saved-results line');
  // ... into tooltips: the cell's name, the number's label, the step line, the basis (what proseg's count is),
  // the neighbour genes' and the rows' labels (the learned guide), a row, the estimate's blind spot.
  assert.equal(w.root.one('h2.qc-deeper-cell').title, DEEP_TEXT.deepSubtitle);
  assert.equal(w.root.one('.qc-deeper-number-label').title, DEEP_TEXT.deepNumberTooltip);
  assert.equal(w.root.one('p.qc-deeper-step').title, DEEP_TEXT.deepStepsTooltip);
  assert.equal(w.root.one('.qc-deeper-basis').title, DEEP_TEXT.deepBasisNoBg, 'no statistics yet: its background not said');
  assert.match(w.root.one('h3.qc-deeper-foreign-heading').title, /^Genes far more typical of a neighbour's kind.*\(8 here\)\. The rest of the RNA/);
  assert.equal(w.root.one('h3.qc-deeper-sources-heading').title, `${DEEP_TEXT.deepSourcesGuide} ${DEEP_TEXT.deepLearnedNote}`);
  assert.match(w.root.one('li.qc-deeper-source').title, /about 43 % of this cell's RNA seems to come from here \(a rough guide\)/);
  assert.equal(w.root.one('p.qc-deeper-estimate').title, DEEP_TEXT.deepBlindSpot);
  assert.equal(w.root.one('section.qc-deeper-own').all('p').find(p => !p.hidden).title,
    `${DEEP_TEXT.deepOwnKindTooltip} Its top genes here, mostly its own: GYHNN 6 · GXEN 4 · GZCNN 4.`, 'its top genes in its kind\'s tooltip');
  // The info marks are inline icons, hidden from screen readers: the label holding them is named.
  assert.ok(w.root.all('span.qc-deeper-icon').length >= 5); assert.equal(w.root.one('span.qc-deeper-icon').getAttribute('aria-hidden'), 'true');
  // The headline's meaning is never "another kind of cell" (the card's own meaning).
  assert.doesNotMatch(text, /another kind of cell|different kind of cell/);
  // The bar's widths come from the split; its label doubles as the text view.
  const segs = w.root.all('span.qc-deeper-seg');
  assert.ok(Number(segs[0].style.flexGrow) > Number(segs[1].style.flexGrow) && Number(segs[2].style.flexGrow) > 0);
  assert.equal(w.root.one('.qc-deeper-bar').getAttribute('aria-label'), 'Own 54 % · Neighbours 46 % · Loose < 1 %');
  // The table: collapsed until opened, then rows of its genes by transcripts, tagged rows marked.
  const table = w.root.one('details.qc-deeper-table');
  assert.equal(w.root.all('tr[data-gene]').length, 0);
  table.open = true; table.emit('toggle');
  const trs = w.root.all('tr[data-gene]');
  assert.equal(trs[0].dataset.gene, 'GXPN'); assert.equal(trs[0].dataset.foreign, '1'); assert.equal(trs[1].dataset.foreign, '0');
  assert.match(trs[0].visibleText(), /a neighbour's gene/);
  // Up/Down move only inside the table.
  w.root.emit('keydown', { key: 'ArrowDown', target: trs[0] });
  assert.equal(w.doc.activeElement?.dataset.gene, 'GYL');
  const outside = w.root.emit('keydown', { key: 'ArrowDown', target: w.root });
  assert.equal(outside.prevented, undefined, 'Up/Down outside the table are not the window\'s');
  // How is this worked out?: four short points, the method named only there, then what the sample's model
  // learned from and when; never a notebook.
  const items = w.root.one('details.qc-deeper-how').all('li').map(li => li.textContent);
  assert.deepEqual(items, [DEEP_TEXT.deepHow1, DEEP_TEXT.deepHow2, DEEP_TEXT.deepHow3, DEEP_TEXT.deepHow4, 'Learned from 42,000 cells on 21 Oct 2026, in 7 min.', DEEP_TEXT.deepHowCredit]);
  w.ctx.qc = { ...w.ctx.qc, sampleModel: { provenance: { notebook_equivalent: false, epochs_run: 429, epochs_notebook: 50, differences: ['more_epochs_small_table'] } } };
  w.deeper.rerender();
  assert.equal(w.root.one('details.qc-deeper-how').all('li').length, 5, 'a model that does not say how many cells: the four points and the method\'s authors');
  for (const t of w.deeper.texts()) { assert.deepEqual(bannedWords(t), [], t); assert.doesNotMatch(t, /notebook|part\s*3/i, t); }
  // No kind named for its own RNA: its top genes keep their own line.
  const plain = make({ answer: body => ready(mixed({ seq: body.seq, kind: [] })) });
  plain.deeper.open(2); await settle();
  assert.deepEqual(plain.root.one('section.qc-deeper-own').all('p').filter(p => !p.hidden).map(p => p.textContent), ['Top here: GYHNN 6 · GXEN 4 · GZCNN 4']);
});

// Visible words of a typical ready window (about 95, from 320-460): innerText's reading (hidden nodes and
// a closed <details>' content left out, its summary kept); a word is a token with a letter or digit ("·" and "%" are not).
function visibleWords(node) {
  if (!node || node.hidden) return 0;
  if (node.nodeType === 3) return String(node.textContent).split(/\s+/).filter(t => /[\p{L}\p{N}]/u.test(t)).length;
  const own = String(node._text || '').split(/\s+/).filter(t => /[\p{L}\p{N}]/u.test(t)).length;
  const kids = node.tagName === 'DETAILS' && !node.open ? node.children.filter(c => c.tagName === 'SUMMARY') : node.childNodes;
  return own + kids.reduce((s, c) => s + visibleWords(c), 0);
}
test('a typical ready window shows about 95 words (fails above 105; ui.deeper_words warns above 120)', async t => {
  // Paused, as the harness's deeper_open reads it: the frame's statistics in (the basis's background in its tooltip),
  // and "Test a setting here…" beside Show on the image (a proseg run's selected cell: the page can test it).
  const w = make({ live: false, run: { state: 'paused' }, answer: body => ready(mixed({ seq: body.seq })) });
  w.ctx.cellStats = () => ({ assigned: 452, kept: 287 }); w.ctx.testable = true;
  w.deeper.open(2); await settle();
  assert.match(w.deeper.snapshot().basis.tip, /less 165 it reads as background/); assert.equal(w.deeper.snapshot().test, true);
  const n = visibleWords(w.root);
  t.diagnostic(`cell 1 (one row, the test's entry, paused): ${n} visible words`);
  assert.ok(n <= 105, `${n} visible words`);
  assert.ok(n >= 60, `${n}: the window still says what it shows`);
  // The table and How opened: their words are the reader's choice.
  w.root.one('details.qc-deeper-how').open = true;
  assert.ok(visibleWords(w.root) > n);
  // Four neighbour rows, the look-alike line and the same-kind count in a tooltip (cell 4).
  const a = make({ answer: body => ready(lookalike({ seq: body.seq })) });
  a.ctx.testable = true;
  a.deeper.open(2); await settle();
  t.diagnostic(`cell 4 (four rows, the look-alike line, the test's entry, no statistics): ${visibleWords(a.root)} visible words`);
  assert.ok(visibleWords(a.root) <= 125, `${visibleWords(a.root)} visible words with four rows`);
});

test('the basis is proseg\'s count of the cell; its tooltip says what proseg sets aside as background in the result\'s own frame, once its statistics are in', async () => {
  const w = make({ live: false, run: { state: 'paused' }, answer: body => ready(mixed({ seq: body.seq })) });
  const stats = new Map(), asked = [];
  w.ctx.cellStats = (label, seq) => { asked.push([label, seq]); return label === 2 ? stats.get(seq) ?? null : null; };
  w.deeper.open(2); await settle();
  const basis = w.root.one('.qc-deeper-basis');
  assert.equal(basis.textContent, "Proseg's count (averaged): 682");
  assert.equal(basis.title, DEEP_TEXT.deepBasisNoBg, 'no statistics yet: its background not said');
  assert.deepEqual(asked.at(-1), [2, 12], 'the statistics of the result\'s own frame');
  // The frame's statistics arrive with its counted file (main.js CellStats): proseg places 452 in the cell and keeps
  // 287 of them.
  stats.set(12, { assigned: 452, kept: 287 }); w.deeper.statsChanged();
  assert.equal(basis.title, deep('deepBasis', { bg: '165' }));
  assert.equal(basis.title, 'The transcripts proseg places in this cell, less 165 it reads as background, averaged over its random background draw. The saved table holds one draw of this count.');
  assert.deepEqual(w.deeper.snapshot().basis, { text: "Proseg's count (averaged): 682", tip: basis.title });
  // Stepped on, frame 13 still being worked out: frame 12's count is shown, so is its background, never frame 13's
  // (another frame's background beside this count would not add up to what proseg placed in the cell).
  w.ctx.seq = 13; stats.set(13, { assigned: 905, kept: 681 }); w.deeper.changed();
  assert.equal(basis.textContent, "Proseg's count (averaged): 682"); assert.equal(basis.title, deep('deepBasis', { bg: '165' }));
  assert.deepEqual(asked.at(-1), [2, 12]);
  // Its own result in: its own frame's background; whole numbers; none below 0.
  stats.set(13, { assigned: 460.4, kept: 300.2 }); w.clock.advance(ASK_MS); await settle();
  assert.deepEqual(asked.at(-1), [2, 13]);
  assert.equal(basisTooltip(normalise(mixed()), stats.get(13)), deep('deepBasis', { bg: '160' })); assert.equal(basis.title, deep('deepBasis', { bg: '160' }));
  assert.equal(basisTooltip(normalise(mixed()), { assigned: 300, kept: 310 }), deep('deepBasis', { bg: '0' }));
  // The worker's exact amounts (32nds): a count of 171.46875 reads 171, as the stage's readout says it, and with its
  // background adds up to what proseg places in the cell (282).
  const exact = normalise(mixed({ transcripts: 171.46875, outside: 0 }));
  assert.equal(explain(exact, { name: c => CELLS[c], label: c => c + 1, startCode: 2 }).basis, "Proseg's count (averaged): 171");
  assert.equal(basisTooltip(exact, { assigned: 282, kept: 171.46875 }), deep('deepBasis', { bg: '111' }));
  // A kept count ending in .5 (proseg's count is in 32nds): kept and background add up to what proseg places in it.
  assert.equal(basisTooltip(normalise(mixed()), { assigned: 120, kept: 91.5 }), deep('deepBasis', { bg: '28' }));
  assert.equal(basisTooltip(normalise(mixed()), { assigned: 120, kept: 90.5 }), deep('deepBasis', { bg: '29' }));
  for (const odd of [{ assigned: Number.NaN, kept: 3 }, { assigned: -1, kept: 0 }, { kept: 4 }, null]) assert.equal(basisTooltip(normalise(mixed()), odd), DEEP_TEXT.deepBasisNoBg, JSON.stringify(odd));
  w.ctx.cellStats = () => { throw new Error('stats gone'); }; w.deeper.changed(); assert.equal(basis.title, DEEP_TEXT.deepBasisNoBg);
  // The start line's: proseg's count at iteration 0.
  assert.equal(basisTooltip(normalise(mixed({ seq: 'start', transcripts: 161 })), { assigned: 452, kept: 287 }), "At the start: 161, proseg's count at iteration 0.");
  // Proseg's whole count, the genes the model lacks included (as its background and the stage's readout count them);
  // those genes said apart, the split's own share of the cell (its rows) on the model's genes alone.
  const lacking = explain(normalise(mixed({ outside: 15 })), { name: c => CELLS[c], label: c => c + 1, startCode: 2 });
  assert.equal(lacking.basis, "Proseg's count (averaged): 697");
  assert.ok(lacking.after.some(n => n.text === '15 transcripts of unknown genes left out'));
  assert.equal(sourceRows(lacking).rows[0].pct, 43, 'a row\'s share is of the split\'s 682');
  assert.equal(basisTooltip(normalise(mixed({ seq: 'start', transcripts: 161, outside: 4 }))), "At the start: 165, proseg's count at iteration 0.");
  // No second bar any more: the window shows no outline count of the cell.
  assert.equal(w.root.one('div.qc-deeper-proseg'), null);
  for (const t of w.deeper.texts()) { assert.deepEqual(bannedWords(t), [], t); assert.doesNotMatch(t, /inside its outline|moved out/, t); }
});

test('the check off: the window offers the check\'s own switch; learning: the panel\'s words and bar; failed: Try again', async () => {
  const w = make({ qc: { status: 'off', enabled: false } });
  w.deeper.open(2); await settle();
  assert.equal(w.root.dataset.state, 'off'); assert.deepEqual(w.calls.posts, [], 'nothing is asked while the check is off');
  assert.ok(w.root.visibleText().includes(DEEP_TEXT.deepOff));
  w.root.one('button[data-action=switch-on]').click(); assert.equal(w.calls.switches, 1);
  w.ctx.qc = { status: 'learning', enabled: true, progress: { step: 20, of: 50 }, eta_s: 61 }; w.deeper.changed();
  assert.equal(w.root.dataset.state, 'learning');
  assert.ok(w.root.visibleText().includes("Learning this sample's cell types… 20 of 50 · about 2 min left"));
  assert.equal(w.root.one('progress.qc-progress').value, 20);
  assert.ok(w.root.visibleText().includes(DEEP_TEXT.deepLearningNote));
  w.ctx.qc = { status: 'failed', enabled: true, reason: 'oom' }; w.deeper.changed();
  assert.ok(w.root.visibleText().includes("Couldn't learn this sample: the graphics card ran out of memory."));
  w.root.one('button[data-action=retry-learning]').click(); assert.equal(w.calls.retries, 1);
  w.ctx.qc = { status: 'unavailable', enabled: true, reason: 'no_table' }; w.deeper.changed();
  assert.match(w.root.visibleText(), /Go deeper can't run for this run: the mixing check needs a table of cells/);
  const line = () => w.root.one('.qc-deeper-state').all('p')[0];
  assert.equal(line().title ?? '', '', 'the line says it all');
  // A run without proseg's own count: why a new run has it, in the line's tooltip; an old proseg-live: how to update it
  // (the page's /health proseg_fix, in the file the session reads), as the panel's status line and the clusters' say.
  w.ctx.qc = { status: 'unavailable', enabled: true, reason: 'no_counts' }; w.deeper.changed();
  assert.equal(line().textContent, cantRunText(QC_TEXT.unavailableNoCounts)); assert.equal(line().title, QC_TEXT.unavailableNoCountsTooltip);
  Object.assign(w.ctx, { qc: { status: 'unavailable', enabled: true, reason: 'old_proseg' }, prosegFix: 'build' }); w.deeper.changed();
  assert.equal(line().textContent, cantRunText(QC_TEXT.unavailableOldProseg)); assert.equal(line().title, oldProsegTooltip('build'));
  Object.assign(w.ctx, { prosegFix: 'setting', notebook: true }); w.deeper.changed();
  assert.equal(line().title, oldProsegTooltip('setting', true));
  for (const t of w.deeper.texts()) assert.deepEqual(bannedWords(t), [], t);
});

test('a sample learned before Go deeper: "Learn the details", its progress, then the window fills in by itself', async () => {
  let deep = false;
  const w = make({ answer: body => deep ? ready(mixed({ seq: body.seq })) : { state: 'needs_detail', seq: body.seq, cell: body.cell, model: 'm1', deep: false } });
  w.ctx.qc = { status: 'live', enabled: true, sampleModel: { id: 'm1', seconds: 117 } };
  w.deeper.open(2); await settle();
  assert.equal(w.root.dataset.state, 'needs_detail');
  assert.ok(w.root.visibleText().includes("Learning it again with them takes about 2 min, once; the mixing check's numbers stay the same."));
  w.root.one('button[data-action=learn-detail]').click(); await settle();
  assert.equal(w.calls.learns, 1); assert.equal(w.root.dataset.state, 'detail');
  assert.ok(w.root.visibleText().includes('Learning the details for this sample… 0 of 50 · about 2 min left'));
  w.deeper.event({ ev: 'deeper', sample: 's1', state: 'detail', progress: { step: 30, of: 50 }, eta_s: 40 });
  assert.ok(w.root.visibleText().includes('Learning the details for this sample… 30 of 50 · about 40 s left'));
  deep = true;
  w.deeper.event({ ev: 'deeper', sample: 's1', state: 'ready' }); await settle();
  assert.equal(w.root.dataset.state, 'ready'); assert.equal(w.root.one('.qc-deeper-number').textContent, '46 %');
  // A failed detail job: its reason in plain words, and Try again.
  const f = make({ answer: { state: 'detail_failed', reason: 'detail_mismatch', seq: 12, cell: 1, model: 'm1' } });
  f.deeper.open(2); await settle();
  assert.ok(f.root.visibleText().includes("Couldn't add the details: the new learning didn't match the old one exactly, so nothing was changed."));
});

test('live: results ride the analysed passes; the window applies at most one a second, in the paint batch; never a later frame', async () => {
  const w = make({ answer: { state: 'working', seq: 12, cell: 1, model: 'm1', deep: true } });
  w.deeper.open(2); await settle();
  assert.equal(w.root.dataset.state, 'working'); assert.equal(w.root.one('.qc-deeper-state').visibleText().trim(), '');
  w.clock.advance(WORKING_MS);
  assert.ok(w.root.visibleText().includes('Working out ffaaaaaa-1…'), 'only after 250 ms');
  const push = (seq, code) => w.deeper.event({ ev: 'deeper', run: 'r1', seq, iter: seq * 100 + 1, cell: 1, model: 'm1', result: mixed({ seq, iter: seq * 100 + 1, code }) });
  push(12, 116); w.batch.flush();
  assert.equal(w.root.one('.qc-deeper-number').textContent, '46 %'); assert.equal(w.counters.qcDeeperApplied, 1);
  // A crop's pushes every 200 ms: the window takes the newest at most once a second.
  let applied = w.counters.qcDeeperApplied;
  for (let k = 1; k <= 10; k++) {
    w.clock.advance(200); w.ctx.seq = 12 + k; push(12 + k, 116 + k); w.deeper.changed(); w.batch.flush();
  }
  w.clock.advance(1000); w.batch.flush();
  const perSecond = (w.counters.qcDeeperApplied - applied) / 3;
  assert.ok(perSecond <= 1, `${w.counters.qcDeeperApplied - applied} updates in 3 s`);
  assert.equal(w.deeper.snapshot().seq, 22, 'the newest at or before the frame on screen');
  // A result for a frame not yet on screen is held, never shown.
  push(30, 200); w.clock.advance(2000); w.deeper.changed(); w.batch.flush();
  assert.equal(w.deeper.snapshot().seq, 22); assert.equal(w.deeper.snapshot().code, 116 + 10);
  assert.equal(w.calls.posts.length, 1, 'followed: no request per frame');
});

test('motion: a live update a second after the previous moves in at most 4 paints; paused nothing moves; reduced motion lands at once', async () => {
  const w = make({ answer: body => ready(mixed({ seq: body.seq })) });
  w.deeper.open(2); await settle();
  const frames = () => w.counters.qcDeeperAnimFrames ?? 0;
  w.clock.advance(1100); w.ctx.seq = 13;
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 13, iter: 1301, cell: 1, model: 'm1', result: mixed({ seq: 13, iter: 1301, code: 150, parts: [0.4, 0.6, 0] }) });
  w.batch.flush();
  assert.notEqual(w.root.one('.qc-deeper-number').textContent, '60 %', 'the count starts from the old value');
  w.clock.advance(1000);
  assert.equal(w.root.one('.qc-deeper-number').textContent, '60 %');
  assert.equal(frames(), MAX_PAINTS - 1, 'three more paints after the snapshot\'s own');
  assert.equal(w.clock.timers.length + w.clock.rafs.length, 0, 'nothing is scheduled once it has landed');
  // Paused (not live): a late result for the frame on screen lands in one paint.
  w.ctx.live = false; w.ctx.run.state = 'paused'; const before = frames();
  w.clock.advance(5000);
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 13, iter: 1301, cell: 1, model: 'm1', result: mixed({ seq: 13, iter: 1301, code: 20, parts: [0.92, 0.08, 0] }) });
  w.clock.advance(1000);
  assert.equal(frames(), before); assert.equal(w.root.one('.qc-deeper-number').textContent, '8 %');
  // Reduced motion (or the check's calm motion): every change lands in one paint.
  const r = make({ answer: body => ready(mixed({ seq: body.seq })), reduced: true });
  r.deeper.open(2); await settle(); r.clock.advance(1100); r.ctx.seq = 13;
  r.deeper.event({ ev: 'deeper', run: 'r1', seq: 13, iter: 1301, cell: 1, model: 'm1', result: mixed({ seq: 13, iter: 1301, code: 150 }) });
  r.batch.flush();
  assert.equal(r.root.one('.qc-deeper-number').textContent, '60 %'); assert.equal(r.counters.qcDeeperAnimFrames, 0);
});

test('under Compare (setCovered) the window holds still: a motion on its way lands at once; a result pushed meanwhile is shown once the view closes', async () => {
  const w = make({ answer: body => ready(mixed({ seq: body.seq })) });
  w.deeper.open(2); await settle();
  const frames = () => w.counters.qcDeeperAnimFrames ?? 0, number = () => w.root.one('.qc-deeper-number').textContent;
  const push = (code, parts) => w.deeper.event({ ev: 'deeper', run: 'r1', seq: 13, iter: 1301, cell: 1, model: 'm1', result: mixed({ seq: 13, iter: 1301, code, parts }) });
  // A live update moving when Compare opens lands at once; nothing is left scheduled to paint.
  w.clock.advance(1100); w.ctx.seq = 13; push(150, [0.4, 0.6, 0]); w.batch.flush();
  assert.notEqual(number(), '60 %', 'moving');
  const moved = frames();
  w.deeper.setCovered(true);
  assert.equal(number(), '60 %', 'landed at once'); assert.equal(w.clock.rafs.length, 0);
  w.clock.advance(2000); w.batch.flush(); assert.equal(frames(), moved, 'no frame while covered');
  // Paused under it: a result for the frame on screen, pushed meanwhile, waits for the view to close.
  w.ctx.live = false; w.ctx.run.state = 'paused';
  push(20, [0.92, 0.08, 0]); w.clock.advance(5000);
  assert.equal(number(), '60 %', 'held');
  const applied = w.counters.qcDeeperApplied;
  w.deeper.setCovered(false);
  assert.equal(number(), '8 %', 'shown once uncovered'); assert.equal(w.counters.qcDeeperApplied, applied + 1); assert.equal(frames(), moved);
  w.deeper.setCovered(false); assert.equal(w.counters.qcDeeperApplied, applied + 1, 'once');
  // Uncovered as another run or the picker opens instead (no catch-up): nothing is shown for the run left.
  w.deeper.setCovered(true); push(116, [0.539, 0.46, 0.001]); w.deeper.setCovered(false, false);
  assert.equal(number(), '8 %'); assert.equal(w.counters.qcDeeperApplied, applied + 1);
});

test('paused: the frame on screen is asked for after 150 ms (the newest asked wins); a frame held here is not asked again', async () => {
  const w = make({ live: false, run: { state: 'paused' }, seq: 5, answer: body => ready(mixed({ seq: body.seq, iter: body.seq * 100 + 1 })) });
  w.deeper.open(2); await settle();
  assert.deepEqual(w.calls.posts, [{ seq: 5, cell: 1, follow: true }]);
  w.ctx.seq = 6; w.deeper.changed(); w.ctx.seq = 7; w.deeper.changed();
  assert.equal(w.calls.posts.length, 1, 'debounced');
  w.clock.advance(ASK_MS); await settle();
  assert.deepEqual(w.calls.posts.at(-1), { seq: 7, cell: 1, follow: true }, 'only the newest');
  assert.equal(w.deeper.snapshot().seq, 7);
  w.ctx.seq = 5; w.deeper.changed(); w.clock.advance(ASK_MS * 2); await settle();
  assert.equal(w.calls.posts.length, 2, 'frame 5 is held here: answered from memory');
  assert.equal(w.deeper.snapshot().seq, 5);
  // Frame 0: the start line's split, under "At the start".
  w.ctx.seq = 0; w.deeper.changed(); w.clock.advance(ASK_MS); await settle();
  assert.deepEqual(w.calls.posts.at(-1), { seq: 'start', cell: 1, follow: true });
  assert.equal(w.root.one('.qc-deeper-frame').textContent, "At the start (Xenium's cells)");
  // Iteration 0's words are true of proseg's count there: the cells are those the start drew, and its count already
  // sets part of their RNA aside as background (it holds about 0.6 of the drawn cell, never "each cell as drawn").
  const nuclei = w.root.all('div.qc-deeper-notes').flatMap(n => n.all('p')).find(p => p.textContent === "Cells as Xenium's cells drew them");
  assert.ok(nuclei, 'a few words under the headline');
  assert.equal(nuclei.title, "At the first iteration proseg's cells are those Xenium's cells drew, and its count already sets part of their RNA aside as background.");
  assert.doesNotMatch(nuclei.title, /counts each cell as/);
  // A scrubbed frame still being worked out shows the earlier result, and says so.
  const s = make({ live: false, run: { state: 'paused' }, seq: 5, answer: body => body.seq === 5 ? ready(mixed({ seq: 5, iter: 501 })) : { state: 'working', seq: body.seq, cell: 1, model: 'm1' } });
  s.deeper.open(2); await settle();
  s.ctx.seq = 9; s.deeper.changed(); s.clock.advance(ASK_MS); await settle();
  assert.equal(s.root.one('.qc-deeper-frame').textContent, 'From iteration 501 · working out iteration 901…');
});

test('paused, a window opened on an answer it holds already (pushed for another ask) still asks the engine once to follow its cell, so the box its dots arm waits for Resume (SPEC D.1)', async () => {
  // The page has its run (main.js reset) before any window opens, so it keeps what is pushed for any cell.
  const pushed = w => { w.deeper.reset(w.ctx.run); w.deeper.event({ ev: 'deeper', run: 'r1', seq: 5, iter: 501, cell: 1, model: 'm1', result: mixed({ seq: 5, iter: 501 }), explicit: true }); };
  const w = make({ live: false, run: { state: 'paused' }, seq: 5, answer: body => ready(mixed({ seq: body.seq, iter: body.seq * 100 + 1 })) });
  pushed(w); await settle();
  assert.deepEqual([...w.deeper.results.get(1).keys()], [5], 'held before the window opens');
  w.deeper.open(2); await settle();
  assert.deepEqual(w.calls.posts, [{ seq: 5, cell: 1, follow: true }], 'held, and followed all the same');
  w.deeper.changed(); w.clock.advance(ASK_MS * 4); await settle();
  assert.equal(w.calls.posts.length, 1, 'once: the engine follows it now');
  // Closed (the engine stops following) and opened again on the same frame: asked to follow again, once.
  w.deeper.close(); await settle(); assert.equal(w.calls.stops, 1);
  w.deeper.open(2); await settle();
  assert.deepEqual(w.calls.posts.at(-1), { seq: 5, cell: 1, follow: true }); assert.equal(w.calls.posts.length, 2);
  // A run that ended samples no more: nothing held is asked again.
  const d = make({ live: false, run: { state: 'done' }, seq: 5, answer: body => ready(mixed({ seq: body.seq, iter: body.seq * 100 + 1 })) });
  pushed(d); await settle();
  d.deeper.open(2); await settle(); d.clock.advance(ASK_MS * 4); await settle();
  assert.deepEqual(d.calls.posts, [], 'held, on a run that ended: nothing to follow');
});

test('a follow asked for an answer held already that fails is not the frame\'s failure: the answer stays as it is, nothing says an error, the follow is asked again after its wait', async () => {
  const pushed = w => { w.deeper.reset(w.ctx.run); w.deeper.event({ ev: 'deeper', run: 'r1', seq: 5, iter: 501, cell: 1, model: 'm1', result: mixed({ seq: 5, iter: 501 }), explicit: true }); };
  let fail = Object.assign(new Error('down'), { status: 500 });
  const w = make({ live: false, run: { state: 'paused' }, seq: 5, answer: body => fail ?? ready(mixed({ seq: body.seq, iter: 501 })) });
  pushed(w); await settle();
  w.deeper.open(2); await settle();
  const shown = () => [w.root.dataset.state, w.deeper.stateBox.hidden, w.root.one('.qc-deeper-number').textContent];
  assert.deepEqual(w.calls.posts, [{ seq: 5, cell: 1, follow: true }]);
  assert.deepEqual(shown(), ['ready', true, '46 %'], 'the answer held, as it was');
  assert.match(w.calls.logs.at(-1), /could not ask the engine to follow r1:1: HTTP 500 \(try 1\)/);
  for (let i = 0; i < 3; i++) { w.deeper.changed(); w.clock.advance(ASK_MS * 4); await settle(); }
  assert.equal(w.calls.posts.length, 1, 'not asked again with every repaint');
  w.clock.advance(RETRY_MS[0]); await settle();
  assert.equal(w.calls.posts.length, 2, 'asked again after its wait'); assert.deepEqual(shown(), ['ready', true, '46 %']);
  fail = Object.assign(new Error('offline'), { status: 0 });   // a lost connection: the same
  w.clock.advance(RETRY_MS[1]); await settle();
  assert.equal(w.calls.posts.length, 3); assert.deepEqual(shown(), ['ready', true, '46 %']);
  assert.match(w.calls.logs.at(-1), /no connection \(try 3\)/);
  // The engine back (the socket reconnected): the follow asked at once, followed, then nothing more.
  fail = null; w.deeper.refresh(); await settle();
  assert.equal(w.calls.posts.length, 4); assert.deepEqual(shown(), ['ready', true, '46 %']);
  w.deeper.changed(); w.clock.advance(RETRY_MS.at(-1) * 2); await settle();
  assert.equal(w.calls.posts.length, 4, 'followed: nothing more');
  // A frame asked for (not held) still says its failure.
  const f = make({ live: false, run: { state: 'paused' }, seq: 6, answer: () => Object.assign(new Error('down'), { status: 500 }) });
  f.deeper.open(2); f.clock.advance(ASK_MS); await settle();
  assert.equal(f.root.dataset.state, 'error');
});

test('an answer held on the averaged count of a frame counted again since (a pause, the hold, a finish or the end) is asked for once more, on the count proseg saves there; never again once answered (P3-1)', async () => {
  // The window was closed while the run played; frame 5 was a token's (averaged) and its answer was pushed then; proseg
  // counted it again for the end (context sampled) and nothing was followed, so nothing was pushed. Reopened on that
  // frame of a run that ended (nothing posted for a held answer on its own count), the answer held is asked for again.
  const pushed = w => { w.deeper.reset(w.ctx.run); w.deeper.event({ ev: 'deeper', run: 'r1', seq: 5, iter: 501, cell: 1, model: 'm1', result: mixed({ seq: 5, iter: 501, code: 116 }), explicit: true }); };
  const saved = body => ready(mixed({ seq: body.seq, iter: 501, code: 112, count: 'sampled' }));
  const w = make({ live: false, run: { state: 'done' }, seq: 5, answer: saved });
  w.ctx.sampled = seq => seq === 5;
  pushed(w); await settle();
  assert.equal(w.deeper.results.get(1).get(5).count, 'averaged');
  w.deeper.open(2); await settle();
  assert.deepEqual(w.calls.posts, [{ seq: 5, cell: 1, follow: false }], 'one ask: the run ended, nothing to follow');
  assert.equal(w.deeper.results.get(1).get(5).count, 'sampled', 'the answer on the count proseg saves replaces it');
  assert.equal(w.root.one('.qc-deeper-number').textContent, '45 %', 'its own number (46 % on the averaged count)');
  w.deeper.changed(); w.clock.advance(ASK_MS * 4); await settle(); w.deeper.changed(); await settle();
  assert.equal(w.calls.posts.length, 1, 'answered on its count: nothing more');
  // An engine that still answers the averaged count (its saved count's file gone): asked once, never again.
  const v = make({ live: false, run: { state: 'done' }, seq: 5, answer: body => ready(mixed({ seq: body.seq, iter: 501, code: 116 })) });
  v.ctx.sampled = seq => seq === 5;
  pushed(v); await settle();
  v.deeper.open(2); await settle();
  for (let i = 0; i < 3; i++) { v.deeper.changed(); v.clock.advance(ASK_MS * 4); await settle(); }
  v.deeper.close(); await settle(); v.deeper.open(2); await settle();
  assert.deepEqual(v.calls.posts, [{ seq: 5, cell: 1, follow: false }], 'once per cell, frame and count');
  // Held on the frame's own count: nothing asked (the run ended).
  const u = make({ live: false, run: { state: 'done' }, seq: 5, answer: saved });
  u.ctx.sampled = seq => seq === 5;
  u.deeper.reset(u.ctx.run); u.deeper.event({ ev: 'deeper', run: 'r1', seq: 5, iter: 501, cell: 1, model: 'm1', result: mixed({ seq: 5, iter: 501, count: 'sampled' }), explicit: true });
  await settle(); u.deeper.open(2); await settle(); u.clock.advance(ASK_MS * 4); await settle();
  assert.deepEqual(u.calls.posts, [], 'held on its own count: nothing asked');
  // While the run rests, the ask for the saved count follows the cell (as the held answer's own follow would), once.
  const p = make({ live: false, run: { state: 'paused' }, seq: 5, answer: saved });
  p.ctx.sampled = seq => seq === 5;
  pushed(p); await settle();
  p.deeper.open(2); await settle(); p.deeper.changed(); p.clock.advance(ASK_MS * 4); await settle();
  assert.deepEqual(p.calls.posts, [{ seq: 5, cell: 1, follow: true }], 'asked again on the saved count, and followed');
  // A failed ask is asked again after its wait (no answer came), then not again once answered.
  let fails = 1;
  const f = make({ live: false, run: { state: 'done' }, seq: 5, answer: body => (fails-- > 0 ? Object.assign(new Error('down'), { status: 500 }) : saved(body)) });
  f.ctx.sampled = seq => seq === 5;
  pushed(f); await settle();
  f.deeper.open(2); await settle();
  assert.equal(f.calls.posts.length, 1);
  f.clock.advance(RETRY_MS[0] + 10); await settle();
  assert.equal(f.calls.posts.length, 2, 'asked again after the failure');
  assert.equal(f.deeper.results.get(1).get(5).count, 'sampled');
  f.clock.advance(RETRY_MS.at(-1) * 2); f.deeper.changed(); await settle();
  assert.equal(f.calls.posts.length, 2);
  // An engine that promises the answer (working: its usual reply, as nothing followed the cell there) is asked once,
  // however often the window is painted meanwhile; its result, pushed, answers it for good.
  const working = body => ({ state: 'working', seq: body.seq, cell: body.cell, model: 'm1', deep: true });
  for (const state of ['done', 'paused']) {
    const k = make({ live: false, run: { state }, seq: 5, answer: working });
    k.ctx.sampled = seq => seq === 5;
    pushed(k); await settle();
    k.deeper.open(2); await settle();
    for (let i = 0; i < 5; i++) { k.deeper.changed(); await settle(); }
    assert.deepEqual(k.calls.posts, [{ seq: 5, cell: 1, follow: state === 'paused' }], `${state}: one ask while its answer is promised`);
    k.deeper.event({ ev: 'deeper', run: 'r1', seq: 5, iter: 501, cell: 1, model: 'm1', result: mixed({ seq: 5, iter: 501, code: 112, count: 'sampled' }), explicit: true });
    await settle();
    for (let i = 0; i < 3; i++) { k.deeper.changed(); k.clock.advance(ASK_MS * 4); await settle(); }
    assert.equal(k.calls.posts.length, 1, `${state}: answered`); assert.equal(k.root.one('.qc-deeper-number').textContent, '45 %');
    k.clock.advance(STALE_MS * 2); k.deeper.changed(); await settle();
    assert.equal(k.calls.posts.length, 1, `${state}: never asked again`);
  }
  // A promise that goes stale (no result within STALE_MS) is asked again; one whose computation fails, after its wait.
  const z = make({ live: false, run: { state: 'done' }, seq: 5, answer: working });
  z.ctx.sampled = seq => seq === 5;
  pushed(z); await settle();
  z.deeper.open(2); await settle();
  z.clock.advance(STALE_MS - 100); z.deeper.changed(); await settle();
  assert.equal(z.calls.posts.length, 1);
  z.clock.advance(200); await settle();
  assert.equal(z.calls.posts.length, 2, 'gone stale: asked again');
  z.deeper.event({ ev: 'deeper', run: 'r1', seq: 5, cell: 1, state: 'error', error: 'failed', explicit: true }); await settle();
  z.deeper.changed(); await settle();
  assert.equal(z.calls.posts.length, 2, 'failed: asked again only after its wait');
  z.clock.advance(RETRY_MS[0] + 10); await settle();
  assert.equal(z.calls.posts.length, 3);
});

test('a frame without proseg\'s count of its own is answered as the counted frame the engine snaps it to: shown as that frame, said so, asked once', async () => {
  // Paused on frame 13 (proseg never counted it, or its file went): the engine works it out at frame 12, its answer says
  // so (seq 12), and it pushes frame 12's result.
  const w = make({ live: false, run: { state: 'paused' }, seq: 13, answer: body => ({ state: 'working', seq: body.seq === 13 ? 12 : body.seq, cell: 1, model: 'm1', deep: true }) });
  w.deeper.open(2); await settle();
  assert.deepEqual(w.calls.posts, [{ seq: 13, cell: 1, follow: true }]);
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 12, iter: 1201, cell: 1, model: 'm1', result: mixed({ seq: 12, iter: 1201 }), explicit: true });
  await settle();
  const tag = w.root.one('.qc-deeper-frame');
  assert.equal(tag.textContent, "At iteration 1,201: the nearest with proseg's count", 'never "working out iteration 1,301" for ever');
  assert.equal(w.root.dataset.state, 'ready'); assert.equal(w.deeper.status().pending, false);
  assert.equal(w.deeper.snapshot().asked, null, 'its answer came');
  w.clock.advance(60000); await settle();
  assert.equal(w.calls.posts.length, 1, 'asked once in a minute: the answer held is the frame\'s');
  assert.equal(tag.textContent, "At iteration 1,201: the nearest with proseg's count");
  // Back to frame 12 (its own) and to 13 again: both answered from what is held.
  w.ctx.seq = 12; w.deeper.changed(); w.clock.advance(ASK_MS); await settle();
  assert.equal(tag.textContent, 'Iteration 1,201 (not the newest)');
  w.ctx.seq = 13; w.deeper.changed(); w.clock.advance(ASK_MS); await settle();
  assert.equal(w.calls.posts.length, 1); assert.equal(tag.textContent, "At iteration 1,201: the nearest with proseg's count");
  for (const t of w.deeper.texts()) assert.deepEqual(bannedWords(t), [], t);
  // Kept already: the engine's ready answer names frame 12 too; kept as frame 12's.
  const r = make({ live: false, run: { state: 'paused' }, seq: 13, answer: () => ready(mixed({ seq: 12, iter: 1201 })) });
  r.deeper.open(2); await settle();
  assert.equal(r.root.one('.qc-deeper-frame').textContent, "At iteration 1,201: the nearest with proseg's count");
  assert.deepEqual([...r.deeper.results.get(1).keys()], [12]);
  r.clock.advance(60000); await settle();
  assert.equal(r.calls.posts.length, 1);
  // Frame 13 counted since that answer (a pause's count can come after it): asked again as itself.
  r.ctx.counted = seq => seq === 13;
  r.deeper.changed(); r.clock.advance(ASK_MS); await settle();
  assert.equal(r.calls.posts.length, 2); assert.deepEqual(r.calls.posts.at(-1), { seq: 13, cell: 1, follow: true });
  // A frame counted all along whose file the spool pruned: the engine's answer stands (asked once).
  const p = make({ live: false, run: { state: 'paused' }, seq: 13, answer: () => ready(mixed({ seq: 12, iter: 1201 })) });
  p.ctx.counted = () => true;
  p.deeper.open(2); await settle(); p.clock.advance(60000); await settle();
  assert.equal(p.calls.posts.length, 1); assert.equal(p.root.one('.qc-deeper-frame').textContent, "At iteration 1,201: the nearest with proseg's count");
});

test('the engine failing the counted frame the frame on screen is answered as fails the frame on screen: said, and asked again after 2 s', async () => {
  // Paused on frame 13 (no proseg count of its own): the engine works it out at frame 12, and says so.
  const w = make({ live: false, run: { state: 'paused' }, seq: 13, answer: body => ({ state: 'working', seq: body.seq === 13 ? 12 : body.seq, cell: 1, model: 'm1', deep: true }) });
  w.deeper.open(2); await settle();
  const asked = { seq: 13, cell: 1, follow: true };
  assert.deepEqual(w.calls.posts, [asked]);
  // Another frame's failure is not this one's.
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 11, cell: 1, model: 'm1', state: 'error', error: 'boom' }); await settle();
  assert.equal(w.root.dataset.state, 'working');
  // Frame 12's is: the window says it failed, and asks for the frame on screen again after the first wait.
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 12, cell: 1, model: 'm1', state: 'error', error: 'boom' }); await settle();
  assert.equal(w.root.dataset.state, 'error'); assert.ok(w.root.visibleText().includes(DEEP_TEXT.deepError));
  w.clock.advance(RETRY_MS[0] - 1); await settle(); assert.equal(w.calls.posts.length, 1);
  w.clock.advance(1); await settle(); assert.deepEqual(w.calls.posts, [asked, asked]);
});

test('failures are said in plain words and asked again after 2, 5, 10 and 30 s; a lost connection says it tries by itself', async () => {
  let fail = true;
  const w = make({ live: false, run: { state: 'paused' }, answer: body => fail ? Object.assign(new Error('boom'), { status: 500 }) : ready(mixed({ seq: body.seq })) });
  w.deeper.open(2); await settle();
  assert.equal(w.root.dataset.state, 'error'); assert.ok(w.root.visibleText().includes(DEEP_TEXT.deepError));
  const asks = () => w.calls.posts.length;
  w.clock.advance(RETRY_MS[0] - 1); await settle(); assert.equal(asks(), 1);
  w.clock.advance(1); await settle(); assert.equal(asks(), 2);
  w.clock.advance(RETRY_MS[1]); await settle(); assert.equal(asks(), 3);
  w.clock.advance(RETRY_MS[2]); await settle(); assert.equal(asks(), 4);
  w.deeper.changed(); w.deeper.changed(); await settle(); assert.equal(asks(), 4, 'not with every snapshot');
  fail = false; w.root.one('button[data-action=retry]').click(); await settle();
  assert.equal(w.root.dataset.state, 'ready');
  const net = make({ live: false, run: { state: 'paused' }, answer: () => new TypeError('Failed to fetch') });
  net.deeper.open(2); await settle();
  assert.equal(net.root.dataset.state, 'retrying'); assert.ok(net.root.visibleText().includes(DEEP_TEXT.deepRetrying));
  // An engine without the analysis worker (its 503 says so: reason "analysis"): it can't run for this run, and says
  // that, not the mixing check's packages.
  const none = make({ answer: () => Object.assign(new Error('no worker'), { status: 503, data: { reason: 'analysis' } }) });
  none.deeper.open(2); await settle();
  assert.equal(none.root.dataset.state, 'unavailable'); assert.match(none.root.visibleText(), /Go deeper can't run for this run/);
  assert.ok(none.root.visibleText().includes(cantRunText(QC_TEXT.unavailableAnalysis)), none.root.visibleText());
  assert.ok(!none.root.visibleText().includes('scvi-tools'));
  // Any other 503 (a storage error, the engine shutting down) is a failure like the others: said, and asked again.
  const disk = make({ live: false, run: { state: 'paused' }, answer: () => Object.assign(new Error('Storage error: [Errno 5]'), { status: 503, data: { detail: 'Storage error: [Errno 5]', status: 503 } }) });
  disk.deeper.open(2); await settle();
  assert.equal(disk.root.dataset.state, 'error'); assert.ok(disk.root.visibleText().includes(DEEP_TEXT.deepError));
  assert.ok(!disk.root.visibleText().includes(cantRunText(QC_TEXT.unavailableAnalysis)), disk.root.visibleText());
  disk.clock.advance(RETRY_MS[0]); await settle();
  assert.equal(disk.calls.posts.length, 2, 'asked again after its wait');
});

test('it follows the selection; a cleared selection keeps its cell; a neighbour\'s › moves both, with one step back', async () => {
  const w = make({ answer: body => body.cell === 1 ? ready(mixed({ seq: body.seq })) : { state: 'working', seq: body.seq, cell: body.cell, model: 'm1' } });
  w.deeper.select(2); w.deeper.open(2); await settle();
  w.deeper.select(0);
  assert.equal(w.root.one('p.qc-deeper-pinned').hidden, false, 'it says it shows the last cell clicked');
  assert.equal(w.deeper.snapshot().cell, 'ffaaaaaa-1');
  w.deeper.select(4); await settle();
  assert.equal(w.counters.qcDeeperRetargets, 1); assert.equal(w.root.dataset.cell, 'ggcccccc-1');
  assert.ok(w.root.one('.qc-deeper-body').classList.contains('qc-deeper-dim'), 'the old content dims until the new answer lands');
  assert.equal(w.calls.focus.at(-1), null, 'the old arrows retract');
  w.clock.advance(WORKING_MS);
  assert.ok(w.root.visibleText().includes('Working out ggcccccc-1…'));
  // Back on the mixed cell, its neighbour's ›: the window and the selection move there, with "Back to".
  w.deeper.select(2); await settle();
  w.root.one('li.qc-deeper-source').one('button[data-action=deeper]').click(); await settle();
  assert.deepEqual(w.calls.selects, [3]);
  const back = w.root.one('button.qc-deeper-back');
  assert.equal(back.hidden, false); assert.equal(back.textContent, 'Back to ffaaaaaa-1');
  back.click(); await settle();
  assert.deepEqual(w.calls.selects, [3, 2]); assert.equal(w.root.one('button.qc-deeper-back').hidden, true, 'one step only');
});

test('a row\'s Show pins its neighbour\'s highlight and lights its suspects (the glow: qc-deeper-where.test.mjs); a same-kind row\'s explains instead; hovering lifts a row; the stage tooltip names how much comes from it', async () => {
  const quiet = make({ answer: body => ready(little({ seq: body.seq })) });
  quiet.deeper.open(2); await settle();
  assert.deepEqual([quiet.calls.focus.at(-1).sources, quiet.calls.focus.at(-1).same], [[], []], 'under 5 %: no arrows, no outlines but its own');
  assert.deepEqual(quiet.calls.focus.at(-1).lit, [2], 'under 5 %: only the cell stays lit');
  const w = make({ answer: body => ready(lookalike({ seq: body.seq })) });
  w.deeper.open(2); await settle();
  const focus = w.calls.focus.at(-1);
  assert.deepEqual(focus.sources.map(s => s.label), [5, 6], 'arrows and magenta outlines only for neighbours of a different kind');
  assert.deepEqual(focus.same, [3, 4, 7, 8, 9, 10, 11, 12], 'the same-kind neighbours: grey dotted outlines');
  assert.deepEqual(focus.lit, [2, 3, 4, 5, 6], 'lit: the cell and its shown rows, same-kind ones included (what Show on the image lights)');
  const row = w.root.all('li.qc-deeper-source').find(li => li.dataset.cell === 'ggaaaaaa-1'), show = row.one('button[data-action=show]');
  assert.equal(show.title, DEEP_TEXT.deepSourceShow);
  show.click();
  assert.equal(w.calls.focus.at(-1).hot, 5); assert.equal(show.textContent, DEEP_TEXT.deepSourceShownShort);
  assert.equal(show.getAttribute('aria-pressed'), 'true'); assert.equal(show.title, DEEP_TEXT.deepSourceShown);
  assert.equal(w.deeper.snapshot().where.on, true, 'Show on the image is on, for this row');
  assert.equal(w.calls.focus.at(-1).apply, false, 'a pin: drawn at once');
  w.deeper.hover(6);
  assert.equal(w.calls.focus.at(-1).hot, 6); assert.ok(w.root.all('li.qc-deeper-source').find(li => li.dataset.cell === 'ggbbbbbb-1').classList.contains('qc-deeper-lifted'));
  assert.equal(w.calls.focus.at(-1).apply, false, 'a hover: drawn at once, never with a paint batch a snapshot holds');
  w.deeper.hover(0); assert.equal(w.calls.focus.at(-1).hot, 5, 'the pinned one stays lit');
  assert.equal(w.deeper.tooltipLine(5), "ggaaaaaa-1 · makes GYPNNN · GYF · GXZNNN: about 8 % of ffaaaaaa-1's RNA seems to come from here (estimate)",
    'rounded down as its row');
  // A neighbour of the same kind: a grey row that says its share can't be checked, no genes; its Show is never
  // greyed out and explains instead, leaving the pinned row as it was.
  const same = w.root.all('li.qc-deeper-source').find(li => li.dataset.cell === 'ffbbbbbb-1'), sameShow = same.one('button[data-action=show]');
  assert.equal(same.dataset.same, '1'); assert.match(same.visibleText(), /same kind: can't be checked · 17 µm/);
  assert.match(same.title, /Same kind as this cell: no gene tells its RNA apart, so this share can't be checked\.$/);
  assert.notEqual(sameShow.disabled, true); assert.equal(sameShow.title, DEEP_TEXT.deepShowSameKind);
  const note = w.root.one('p.qc-deeper-row-note');
  assert.equal(note.hidden, true);
  sameShow.click();
  assert.equal(note.hidden, false); assert.equal(note.textContent, DEEP_TEXT.deepShowSameKind); assert.equal(note.getAttribute('role'), 'status');
  assert.equal(sameShow.getAttribute('aria-pressed'), 'false'); assert.equal(w.calls.focus.at(-1).hot, 5, 'the other row stays pinned');
  sameShow.click(); assert.equal(note.hidden, true, 'pressed again: the note goes');
  assert.equal(row.dataset.same, '0'); assert.match(row.visibleText(), /GYPNNN 4 · GZCNN 3 · 19 µm/);
  // The same-kind count is the Neighbours label's tooltip, never a line.
  assert.match(w.root.one('h3.qc-deeper-sources-heading').title, /8 of its 10 nearest neighbours look the same kind\.$/);
  assert.doesNotMatch(w.root.visibleText(), /nearest/);
  // Pressed again: the pin goes (every suspect glows again; the view stays on until its own button turns it off).
  show.click();
  assert.equal(show.getAttribute('aria-pressed'), 'false'); assert.equal(w.calls.focus.at(-1).hot, 0); assert.equal(w.deeper.snapshot().where.on, true);
  w.root.one('button[data-action=where]').click(); assert.equal(w.deeper.snapshot().where.on, false);
  assert.equal(w.deeper.tooltipLine(3), "ffbbbbbb-1 looks like the same kind as ffaaaaaa-1: RNA from it can't be told apart.");
  assert.equal(w.deeper.tooltipLine(2), null);
  for (const line of [w.deeper.tooltipLine(5), w.deeper.tooltipLine(3)]) assert.deepEqual(bannedWords(line), []);
  for (const t of w.deeper.texts()) assert.deepEqual(bannedWords(t), [], t);
});

test('rows change rank only past 1.25 x; chips likewise; a pressed or focused row is never rebuilt under the pointer', async () => {
  const w = make({ answer: body => ready(lookalike({ seq: body.seq })) });
  w.deeper.open(2); await settle();
  const order = () => w.root.all('li.qc-deeper-source').map(li => li.dataset.cell);
  assert.deepEqual(order(), ['ffbbbbbb-1', 'ggcccccc-1', 'ggaaaaaa-1', 'ggbbbbbb-1']);
  const swap = (a, b) => lookalike({ seq: 13, iter: 1301, neighbours: lookalike().neighbours.map(n => n.cell === 3 ? { ...n, tx: a } : n.cell === 4 ? { ...n, tx: b } : n) });
  w.clock.advance(1100); w.ctx.seq = 13;
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 13, iter: 1301, cell: 1, model: 'm1', result: swap(150, 170) }); w.batch.flush(); w.clock.advance(500);
  assert.deepEqual(order(), ['ffbbbbbb-1', 'ggcccccc-1', 'ggaaaaaa-1', 'ggbbbbbb-1'], '170 < 1.25 x 150: no swap');
  w.clock.advance(1100); w.ctx.seq = 14;
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 14, iter: 1401, cell: 1, model: 'm1', result: { ...swap(150, 200), seq: 14, iter: 1401 } }); w.batch.flush(); w.clock.advance(500);
  assert.deepEqual(order(), ['ffbbbbbb-1', 'ggaaaaaa-1', 'ggcccccc-1', 'ggbbbbbb-1'], 'past 1.25 x: they swap');
  // Pressed: the next update changes the rows' numbers only; the order follows once it is let go.
  const list = w.root.one('ol.qc-deeper-sources');
  list.emit('pointerdown');
  w.clock.advance(1100); w.ctx.seq = 15;
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 15, iter: 1501, cell: 1, model: 'm1', result: { ...swap(400, 100), seq: 15, iter: 1501 } }); w.batch.flush(); w.clock.advance(500);
  assert.deepEqual(order(), ['ffbbbbbb-1', 'ggaaaaaa-1', 'ggcccccc-1', 'ggbbbbbb-1']);
  list.emit('pointerup'); w.clock.advance(10);
  assert.deepEqual(order(), ['ffbbbbbb-1', 'ggcccccc-1', 'ggaaaaaa-1', 'ggbbbbbb-1']);
  // "+k more" shows every tagged gene.
  const m = make({ answer: body => ready(mixed({ seq: body.seq })) });
  m.deeper.open(2); await settle();
  assert.equal(m.root.all('span.qc-deeper-chip').length, CHIPS);
  m.root.one('button.qc-deeper-more').click();
  assert.equal(m.root.all('span.qc-deeper-chip').length, 8); assert.equal(m.root.one('button.qc-deeper-more'), null);
});

test('too few transcripts, empty, not checked; a run from nuclei only at its first frame', async () => {
  const few = make({ answer: { state: 'ready', seq: 12, cell: 1, model: 'm1', result: { v: 1, state: 'too_few', seq: 12, iter: 1201, cell: 1, model: 'm1', transcripts: 61, genes: [['GYVNN', 11], ['GXBNN', 2]] } } });
  few.deeper.open(2); await settle();
  assert.equal(few.root.dataset.state, 'too_few');
  assert.ok(few.root.visibleText().includes('Too few transcripts to split: proseg counts 61 for this cell in this iteration, in the genes the model knows, and at least 100 are needed.'));
  assert.ok(few.root.visibleText().includes('Its genes so far (not split): GYVNN 11 · GXBNN 2'));
  const empty = make({ answer: { state: 'ready', seq: 12, cell: 1, model: 'm1', result: { v: 1, state: 'empty', seq: 12, iter: 1201, cell: 1, transcripts: 0 } } });
  empty.deeper.open(2); await settle();
  assert.equal(empty.root.dataset.state, 'empty'); assert.ok(empty.root.visibleText().includes(DEEP_TEXT.deepEmpty));
  const nuclei = make({ seq: 0, live: false, run: { state: 'paused' } });
  nuclei.ctx.startSource = 'nuclei'; nuclei.deeper.open(2); await settle();
  assert.equal(nuclei.root.dataset.state, 'nuclei_only'); assert.deepEqual(nuclei.calls.posts, []);
});

test('a run without a start line (from nuclei only, or cells with their nuclei) shows no start at its first frame: nothing asked, its own words; the engine\'s refusal of a start line says the same and is not asked again', async () => {
  // How the run's cells started (GET /runs/{id}/qc start_kind): only "cells" has a start line (proseg's count at
  // iteration 0); "paired" (Xenium's cells with their nuclei, say) holds only the nuclei there.
  // Paused on its prior (proseg acknowledged the pause, frame 0 its only frame), the run button says Start, Step beside
  // it (main.js runButtons, the context's buttons): the words name them, only while both work.
  const NONE = { start: false, resume: false, step: false, pause: false }, PRIOR = { ...NONE, start: true, step: true };
  const atZero = (kind, answer = body => ready(mixed({ seq: body.seq, iter: 0 })), { prior = true } = {}) => {
    const w = make({ seq: 0, live: false, run: { state: 'paused', ack: { state: 'paused' } }, answer });
    if (prior) { w.ctx.frames = w.ctx.frames.slice(0, 1); w.ctx.buttons = PRIOR; }
    if (kind !== undefined) w.ctx.startKind = kind;
    return w;
  };
  for (const [kind, words] of [['paired', DEEP_TEXT.deepNucleiCellsRun], ['nuclei', DEEP_TEXT.deepNucleiOnlyRun], ['other', DEEP_TEXT.deepNoStartRun]]) {
    const w = atZero(kind);
    w.deeper.open(2); await settle();
    assert.equal(w.root.dataset.state, 'nuclei_only', kind); assert.deepEqual(w.calls.posts, [], kind);
    assert.ok(w.root.visibleText().includes(words), kind); assert.match(words, /press Start or Step\.$/, kind);
  }
  assert.deepEqual([[{ state: 'paused', ack: { state: 'paused' } }, 1], [{ state: 'paused', ack: { state: 'paused' } }, 2],
    [{ state: 'paused', ack: { state: 'running' } }, 1], [{ state: 'paused' }, 1], [{ state: 'running', ack: { state: 'paused' } }, 1],
    [{ state: 'done' }, 1]].map(([run, n]) => onPrior(run, Array.from({ length: n }, (_, seq) => ({ seq })))),
  [true, false, false, false, false, false]);
  // Every other run gets the same words naming no button, and they follow the page's buttons: one that ended, saves
  // its results (finishing) or samples on by itself (none of Start and Step); one paused later with its first frame
  // scrubbed back to (Resume and Step: Resume leaves the view there); one still pausing on its prior (none works yet);
  // and the prior itself while its buttons do not work (the finish prompt open: neither Start nor Step).
  const others = [[{ state: 'done' }, NONE], [{ state: 'stopped' }, NONE], [{ state: 'failed' }, NONE], [{ state: 'finishing' }, NONE],
    [{ state: 'running' }, { ...NONE, pause: true }], [{ state: 'paused', ack: { state: 'paused' } }, { ...NONE, resume: true, step: true }],
    [{ state: 'paused', ack: { state: 'running' }, prior: true }, NONE], [{ state: 'paused', ack: { state: 'paused' }, prior: true }, NONE],
    [{ state: 'paused', ack: { state: 'paused' }, prior: true }, { ...NONE, start: true }], [{ state: 'paused', ack: { state: 'paused' }, prior: true }, null]];
  for (const [{ prior = false, ...run }, buttons] of others) {
    for (const [kind, words] of [['paired', DEEP_TEXT.deepNucleiCellsRunEnded], ['nuclei', DEEP_TEXT.deepNucleiOnlyRunEnded], ['other', DEEP_TEXT.deepNoStartRunEnded]]) {
      const w = make({ seq: 0, live: false, run }), state = `${JSON.stringify(run)} ${JSON.stringify(buttons)}`;
      if (prior) w.ctx.frames = w.ctx.frames.slice(0, 1);
      if (buttons) w.ctx.buttons = buttons;
      w.ctx.startKind = kind;
      w.deeper.open(2); await settle();
      assert.equal(w.root.dataset.state, 'nuclei_only', `${state} ${kind}`); assert.deepEqual(w.calls.posts, [], `${state} ${kind}`);
      assert.ok(w.root.visibleText().includes(words), `${state} ${kind}`);
      assert.doesNotMatch(w.root.visibleText(), /Resume|Start|Step/, `${state} ${kind}`);
    }
  }
  const stopped = atZero('nuclei');
  stopped.deeper.open(2); await settle();
  assert.ok(stopped.root.visibleText().includes(DEEP_TEXT.deepNucleiOnlyRun));
  stopped.ctx.run.state = 'stopped'; stopped.ctx.buttons = NONE; stopped.deeper.changed(); await settle();
  assert.ok(stopped.root.visibleText().includes(DEEP_TEXT.deepNucleiOnlyRunEnded), 'stopped meanwhile: said at once');
  assert.ok(!stopped.root.visibleText().includes(DEEP_TEXT.deepNucleiOnlyRun));
  const cells = atZero('cells');
  cells.deeper.open(2); await settle();
  assert.deepEqual(cells.calls.posts, [{ seq: 'start', cell: 1, follow: true }]); assert.equal(cells.root.dataset.state, 'ready');
  // Its kind not known yet: asked. The engine's refusal (404, reason "no_start" and the kind it names) is the first
  // frame's words, never "Couldn't work out this cell." with Try again, and it is not asked again; later frames as usual.
  const noStart = (data) => body => (body.seq === 'start'
    ? Object.assign(new Error('This run started from cells with their nuclei: it has no start line'), { status: 404, data }) : ready(mixed({ seq: body.seq })));
  const refused = atZero(undefined, noStart({ detail: 'x', status: 404, reason: 'no_start', start_kind: 'paired' }), { prior: false });
  refused.deeper.open(2); await settle();
  assert.equal(refused.root.dataset.state, 'nuclei_only'); assert.ok(refused.root.visibleText().includes(DEEP_TEXT.deepNucleiCellsRunEnded));
  assert.equal(refused.root.one('button[data-action=retry]'), null, 'no Try again');
  refused.clock.advance(RETRY_MS.at(-1) * 2); refused.deeper.changed(); await settle();
  assert.equal(refused.calls.posts.length, 1, 'not asked again');
  refused.ctx.seq = 12; refused.deeper.changed(); refused.clock.advance(ASK_MS); await settle();
  assert.deepEqual(refused.calls.posts.at(-1), { seq: 12, cell: 1, follow: true }); assert.equal(refused.root.dataset.state, 'ready');
  assert.equal(refused.root.one('.qc-deeper-step').textContent, 'Looks mixed', 'no change since a start it does not have');
  refused.ctx.seq = 0; refused.deeper.changed(); await settle();
  assert.equal(refused.root.dataset.state, 'nuclei_only'); assert.equal(refused.calls.posts.length, 2);
  const plain = atZero(undefined, noStart({ reason: 'no_start' }));
  plain.deeper.open(2); await settle();
  assert.ok(plain.root.visibleText().includes(DEEP_TEXT.deepNoStartRun), 'a refusal naming no kind: the plain words');
  // Another refusal of the start line (not no_start) is a failure as before, asked again.
  const other = atZero(undefined, () => Object.assign(new Error('Unknown cell'), { status: 404, data: { detail: 'Unknown cell', status: 404 } }));
  other.deeper.open(2); await settle();
  assert.equal(other.root.dataset.state, 'error');
  // The step line: no change since the start for a run without one (it holds no starting number).
  const name = c => CELLS[c] ?? `cell ${c}`;
  for (const startKind of ['paired', 'nuclei', 'none']) assert.equal(explain(normalise(mixed()), { name, label: c => c + 1, startCode: 2, startKind }).stepLine, 'Looks mixed', startKind);
  assert.equal(explain(normalise(mixed()), { name, label: c => c + 1, startCode: 2, startKind: 'cells' }).stepLine, 'Looks mixed · more than at the start · like for like');
  assert.deepEqual([[null, null], ['nuclei', null], ['xenium', 'nuclei'], ['xenium', 'paired'], ['other', 'cells'], ['xenium', 'none']].map(([s, k]) => noStartLine(s, k)),
    [null, 'nuclei', 'nuclei', 'paired', null, '']);
  for (const key of [...Object.values(NO_START_TEXT), ...Object.values(NO_START_ENDED_TEXT)]) {
    assert.ok(DEEP_TEXT[key], key); assert.deepEqual(bannedWords(DEEP_TEXT[key]), [], key);
  }
});

test('Esc closes only from inside; the window is movable, remembered as fractions of the stage, and put back by a double-click', async () => {
  const w = make({ answer: body => ready(mixed({ seq: body.seq })) });
  w.deeper.open(2); await settle();
  assert.deepEqual([w.root.style.left, w.root.style.top], [`${1200 - 360 - 12}px`, '52px'], 'top right, under Fit');
  const head = w.root.one('header.qc-deeper-head');
  head.emit('pointerdown', { pointerId: 1, clientX: 900, clientY: 60, target: head });
  head.emit('pointermove', { pointerId: 1, clientX: 600, clientY: 260 });
  head.emit('pointerup', { pointerId: 1 });
  assert.deepEqual([w.root.style.left, w.root.style.top], ['528px', '252px']);
  assert.deepEqual(JSON.parse(w.store.getItem(STORAGE_KEY)), { x: 0.44, y: 0.315 });
  w.deeper.close(); w.deeper.open(2); await settle();
  assert.deepEqual([w.root.style.left, w.root.style.top], ['528px', '252px'], 'remembered');
  head.emit('dblclick', { target: head });
  assert.equal(w.root.style.left, `${1200 - 360 - 12}px`); assert.equal(w.store.getItem(STORAGE_KEY), null);
  // Make smaller: a strip that stays live.
  w.root.one('button[data-action=smaller]').click();
  assert.ok(w.root.classList.contains('qc-deeper-small')); assert.equal(w.root.one('button[data-action=smaller]').title, DEEP_TEXT.deepBigger);
  assert.equal(w.deeper.snapshot().small, true);
  // Find flies to the cell.
  w.root.one('button[data-action=find]').click(); assert.deepEqual(w.calls.finds, [2]);
  // Esc from inside closes it.
  const e = w.root.emit('keydown', { key: 'Escape', target: w.root });
  assert.equal(e.stopped, true); assert.equal(w.root.hidden, true);
  // A stage under 480 px tall: it opens shrunk.
  const short = make({ answer: body => ready(mixed({ seq: body.seq })), stage: { width: 900, height: 400 } });
  short.deeper.open(2); await settle();
  assert.ok(short.root.classList.contains('qc-deeper-small'));
  assert.ok(short.calls.boxes.some(b => b && b.w === 360), 'the area chips keep clear of it');
});

test('the harness snapshot: what the window shows, never a later frame\'s', async () => {
  const w = make({ answer: body => ready(mixed({ seq: body.seq })) });
  assert.deepEqual(w.deeper.snapshot().open, false);
  w.deeper.open(2); await settle();
  const s = w.deeper.snapshot();
  assert.equal(s.open, true); assert.equal(s.cell, 'ffaaaaaa-1'); assert.equal(s.seq, 12); assert.equal(s.iter, 1201); assert.equal(s.state, 'ready');
  assert.equal(s.code, 116); assert.equal(s.shown, '46'); assert.deepEqual(s.parts, [0.539, 0.46, 0.001]); assert.deepEqual(s.bar, [54, 46, '<1']);
  assert.equal(s.learned, 'yes'); assert.deepEqual(s.foreign, ['GXPN', 'GXC', 'GYDNNN', 'GYJ']); assert.deepEqual(s.sources, [['ffbbbbbb-1', 44]]);
  assert.deepEqual(s.basis, { text: "Proseg's count (averaged): 682", tip: DEEP_TEXT.deepBasisNoBg }, 'no statistics: its background not said');
  assert.equal(s.small, false); assert.equal(s.rough, false); assert.equal(s.tag, 'Iteration 1,201 · live');
  assert.equal(s.shown, /(\d+)\s?%/.exec(cardLine(s.code, null, 'xenium'))[1], 'the window\'s % = the card line for the same byte');
});

test('another run closes the window; a closed window asks nothing and shows nothing of the old cell when reopened', async () => {
  const w = make({ answer: body => ready(mixed({ seq: body.seq })) });
  w.deeper.open(2); await settle();
  w.deeper.reset({ id: 'r2', sample: 's1' });
  assert.equal(w.root.hidden, true); assert.equal(w.calls.stops, 1);
  w.ctx.run = { id: 'r2', sample: 's1', state: 'running' };
  w.deeper.changed(); await settle();
  assert.equal(w.calls.posts.length, 1, 'closed: nothing asked');
  w.deeper.open(4); await settle();
  assert.equal(w.root.dataset.cell, 'ggcccccc-1');
  // A legacy run (no sample): the window does not open.
  const legacy = make(); legacy.ctx.run = { id: 'old', state: 'done' };
  assert.equal(legacy.deeper.open(2), false); assert.equal(legacy.root.hidden, true);
});

test('the full table keeps a focused row through new results (it follows once focus leaves); the same answer twice is one update', async () => {
  const w = make({ answer: body => ready(mixed({ seq: body.seq })) });
  w.deeper.open(2); await settle();
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 12, iter: 1201, cell: 1, model: 'm1', result: mixed() });   // the push of the same answer
  w.batch.flush();
  assert.equal(w.counters.qcDeeperApplied, 1);
  const table = w.root.one('details.qc-deeper-table'); table.open = true; table.emit('toggle');
  const first = w.root.one('tr[data-gene]');
  table.emit('focusin');
  w.clock.advance(1100); w.ctx.seq = 13;
  const changed = mixed({ seq: 13, iter: 1301 }); changed.genes[0] = ['GXPN', 45, 9, 36, 1];
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 13, iter: 1301, cell: 1, model: 'm1', result: changed }); w.batch.flush(); w.clock.advance(600);
  assert.equal(w.root.one('tr[data-gene]'), first, 'not rebuilt under keyboard focus');
  table.emit('focusout'); w.clock.advance(10);
  assert.notEqual(w.root.one('tr[data-gene]'), first); assert.equal(w.root.one('tr[data-gene]').title.startsWith('GXPN: 45 transcripts'), true);
});

test('P5: the gene table\'s rows are made when the table is first read, never in an apply while it is closed', async () => {
  // A cell's hundreds of rows and tooltips were most of each apply's work; the table is collapsed until opened. Each
  // gene's loose part is read only for its row.
  const counted = () => {
    const r = normalise(mixed()); let reads = 0;
    r.genes = r.genes.map(g => Object.defineProperty({ ...g }, 'loose', { get() { reads++; return g.loose; }, enumerable: true }));
    return { r, reads: () => reads };
  };
  const c = counted(), m = explain(c.r, { name: c => `c${c}`, label: c => c + 1 });
  assert.equal(c.reads(), 0, 'no row made yet');
  assert.equal(typeof Object.getOwnPropertyDescriptor(m, 'table').get, 'function');
  const rows = m.table;
  assert.equal(c.reads(), c.r.genes.length, 'every row made at the first read');
  assert.equal(m.table, rows, 'and kept'); assert.equal(c.reads(), c.r.genes.length);
  assert.equal(rows[0].gene, 'GXPN'); assert.equal(rows.length, c.r.genes.length);
  assert.match(m.tableHeading, new RegExp(`^All ${c.r.genes.length} genes, split$`));
  // In the window: closed, nothing of the table is made; opened, its rows are drawn from the same model.
  const w = make({ answer: body => ready(mixed({ seq: body.seq })) });
  w.deeper.open(2); await settle();
  const shown = w.deeper.shownModel;
  assert.equal(w.root.all('tr[data-gene]').length, 0);
  const table = w.root.one('details.qc-deeper-table'); table.open = true; table.emit('toggle');
  assert.equal(w.deeper.shownModel, shown); assert.equal(w.root.one('tr[data-gene]').dataset.gene, 'GXPN');
});

test('P5: a result painted again with nothing it reads changed is not worked out again; a change of what it reads is', async () => {
  // The frame tag or the run's state changing paints the same result: its model is kept (QcDeeper.modelOf). Each
  // gene's flag is read whenever explain works a result out.
  let groupLabel = 'GXPN · GXDN';
  const w = make({ live: false, run: { state: 'paused' }, answer: body => ready(mixed({ seq: body.seq })) });
  w.deeper.o.group = label => (label === 3 ? { label: groupLabel, color: 2, iter: 1201 } : null);
  w.deeper.open(2); w.clock.advance(ASK_MS); await settle();
  const r = w.deeper.shown.result, model = w.deeper.shownModel;
  assert.equal(model.state, 'ready');
  let reads = 0;
  r.genes = r.genes.map(g => Object.defineProperty({ ...g }, 'flag', { get() { reads++; return g.flag; }, enumerable: true }));
  w.ctx.run = { ...w.ctx.run, state: 'done' }; w.deeper.rerender();
  assert.equal(w.root.one('.qc-deeper-frame').textContent, 'Finished · iteration 1,201', 'painted again: its tag');
  assert.equal(reads, 0, 'the same result: its model kept'); assert.equal(w.deeper.shownModel, model);
  groupLabel = 'GYSNN · GYRN'; w.deeper.rerender();
  assert.ok(reads > 0, "a neighbour's cell group changed: worked out again"); assert.notEqual(w.deeper.shownModel, model);
});

test('P5: a result painted before the run\'s cell names came in names its neighbours by them once painted again', async () => {
  // main.js cellName reads the run's cell list, which arrives after the run opens: until then a cell is "cell N".
  let names = null;
  const w = make({ live: false, run: { state: 'paused' }, answer: body => ready(mixed({ seq: body.seq })) });
  w.deeper.o.name = label => names?.[label - 1] ?? `cell ${label - 1}`;
  w.deeper.open(2); w.clock.advance(ASK_MS); await settle();
  const first = () => w.root.all('li.qc-deeper-source')[0];
  assert.equal(first().dataset.cell, 'cell 2'); assert.ok(first().textContent.startsWith('cell 2'));
  names = CELLS;
  w.ctx.run = { ...w.ctx.run, state: 'done' }; w.deeper.rerender();   // painted again: its tag changed
  assert.equal(first().dataset.cell, 'ffbbbbbb-1'); assert.ok(first().textContent.startsWith('ffbbbbbb-1'));
  assert.match(first().title, /ffbbbbbb-1/);
});

test('a very big cell\'s reply folds its 1-transcript genes (the worker\'s 32 KB cap): the table says how many, never hides them', () => {
  const raw = mixed();
  const listed = raw.genes.length;
  const r = normalise({ ...raw, genes_total: listed + 212, genes_rest: [212, 212, 150.3, 61.0] });
  assert.deepEqual(r.rest, { k: 212, tx: 212, own: 150.3, nb: 61 });
  const m = explain(r, { name: c => `c${c}` });
  assert.match(m.tableHeading, new RegExp(`^All ${(listed + 212).toLocaleString('en-GB')} genes, split$`), 'the heading counts every gene');
  assert.equal(m.tableRest, 'And 212 more genes holding 212 transcripts in all, not listed one by one.', 'never "one transcript each"');
  assert.equal(m.table.length, listed, 'the rows are the genes listed');
  const plain = explain(normalise(raw), { name: c => `c${c}` });
  assert.equal(plain.tableRest, null); assert.match(plain.tableHeading, new RegExp(`^All ${listed} genes`));
  // A reply without genes_total (an older worker) counts what it lists plus the fold.
  assert.equal(normalise({ ...raw, genes_rest: [3, 3, 2, 1] }).genes_total, listed + 3);
});


// ---- following, same-kind rows, chips, givers, the start's count, cell groups -----------------------------------------
test('closing the window always stops the engine following: after "details not kept", during the first answer, after another cell', async () => {
  // The details not kept: the engine keeps the follow request until they are learned; closing must withdraw it.
  const detail = make({ answer: body => ({ state: 'needs_detail', seq: body.seq, cell: body.cell, model: 'm1', deep: false }) });
  detail.deeper.open(2); await settle();
  assert.equal(detail.root.dataset.state, 'needs_detail');
  detail.deeper.close(); assert.equal(detail.calls.stops, 1, 'a kept follow request is withdrawn');
  // Closed (Esc) while the follow request is on its way: its answer must not leave the engine following.
  let release;
  const slow = make({ answer: body => new Promise(resolve => { release = () => resolve(ready(mixed({ seq: body.seq }))); }) });
  slow.deeper.open(2); await settle();
  slow.deeper.close(); assert.equal(slow.calls.stops, 1, 'withdrawn while in flight');
  release(); await settle(); assert.equal(slow.root.hidden, true);
  // Another cell, then closed before its answer: still withdrawn.
  const other = make({ answer: body => body.cell === 1 ? ready(mixed({ seq: body.seq })) : new Promise(() => {}) });
  other.deeper.open(2); await settle();
  other.deeper.select(5); await settle();
  other.deeper.close(); assert.equal(other.calls.stops, 1);
  // A window that never asked to follow (a finished run) has nothing to withdraw.
  const done = make({ answer: body => ready(mixed({ seq: body.seq })), run: { state: 'done' }, live: false });
  done.deeper.open(2); done.clock.advance(ASK_MS + 1); await settle();
  assert.equal(done.calls.posts.at(-1).follow, false);
  done.deeper.close(); assert.equal(done.calls.stops, 0);
});

test('the follow is a lease: while open on a live run it is asked again every 30 s, not with every snapshot', async () => {
  const w = make({ answer: body => ready(mixed({ seq: body.seq })) });
  w.deeper.open(2); await settle();
  assert.equal(w.calls.posts.length, 1);
  for (let i = 0; i < 5; i++) { w.clock.advance(1000); w.deeper.changed(); await settle(); }
  assert.equal(w.calls.posts.length, 1, 'followed: no request per snapshot');
  w.clock.advance(FOLLOW_RENEW_MS); w.deeper.changed(); await settle();
  assert.equal(w.calls.posts.length, 2); assert.equal(w.calls.posts.at(-1).follow, true, 'renewed');
  w.deeper.changed(); await settle(); assert.equal(w.calls.posts.length, 2);
});

test('the worker\'s same-kind class wins over its likeness; a same-kind row is grey and says it can\'t be checked', () => {
  const r = normalise(lookalike({ neighbours: lookalike().neighbours.map((n, k) => k === 0 ? { ...n, similar: 0.895, same: true } : k === 2 ? { ...n, similar: 0.91, same: false } : n) }));
  assert.deepEqual([r.neighbours[0].same, r.neighbours[2].same], [true, false], 'kept as the worker classed them (0.88-0.92)');
  const m = explain(r, { name: c => CELLS[c] ?? `cell ${c}`, label: c => c + 1 });
  assert.equal(m.neighbours[0].genes, DEEP_TEXT.deepSourceSameKind); assert.match(m.neighbours[2].genes, /^GYPNNN 4 · GZCNN 3/);
  // Without the worker's class (an older reply): the likeness at the cut-off decides.
  assert.equal(normalise(lookalike()).neighbours[0].same, true);
});

test('a tagged gene whose rounded-down count is under half its transcripts is no chip; it stays in the heading and the table', () => {
  const r = normalise(mixed({ genes: [['GXPN', 39, 7.9, 31.1, 1], ['GYTNN', 3, 1.3, 1.6, 1], ['GXC', 10, 1.6, 8.4, 1]] }));
  const m = explain(r, { name: c => CELLS[c] ?? `cell ${c}`, label: c => c + 1 });
  assert.deepEqual(m.chips.map(c => c.text), ['GXPN 31 of 39', 'GXC 8 of 10'], '"GYTNN 1 of 3" under a heading of genes mostly from neighbours: not a chip');
  assert.match(m.chipsTip, /^Genes far more typical of a neighbour's kind, so they probably come from neighbours \(3 here\)\./);
  assert.equal(m.table.find(x => x.gene === 'GYTNN').flag, true);
});

test('"most from" only from the worker\'s own giver, never guessed from a neighbour\'s top four', () => {
  const name = c => CELLS[c] ?? `cell ${c}`;
  // GYHNN has no giver field: the window no longer guesses one from the rows' top-four lists (they omit the real giver).
  const r = normalise(mixed({ genes: [['GXPN', 39, 7.9, 31.1, 1, 0], ['GYHNN', 36, 23, 13, 0]],
    neighbours: [{ cell: 2, um: 25, tx: 20, genes: [['GYHNN', 2]], kind: ['A'], similar: 0.3, known: true },
      { cell: 3, um: 26, tx: 280, genes: [['GXPN', 31.1]], kind: ['B'], similar: 0.3, known: true }] }));
  const rows = explain(r, { name, label: c => c + 1 }).table;
  assert.equal(rows.find(x => x.gene === 'GYHNN').tooltip, 'GYHNN: 36 transcripts · about 23 its own, 13 from neighbours (estimate)');
  assert.match(rows.find(x => x.gene === 'GXPN').tooltip, /\(most from ffbbbbbb-1\)/, 'the worker names GXPN\'s giver: row 0');
});

test('too few at the start says the start\'s count; nuclei-only runs and results without a like-for-like change give no direction', () => {
  const few = explain(normalise({ v: 1, state: 'too_few', seq: 'start', cell: 1, transcripts: 59, genes: [['GXJNN', 3]] }), { startSource: 'xenium' });
  assert.deepEqual(few.lines, ['Too few transcripts to split: proseg counts 59 for this cell at iteration 0, in the genes the model knows, and at least 100 are needed.',
    'Its genes so far (not split): GXJNN 3', DEEP_TEXT.deepTooFewLaterStart]);
  const name = c => CELLS[c] ?? `cell ${c}`;
  assert.equal(explain(normalise(mixed()), { name, label: c => c + 1, startCode: 2, startSource: 'nuclei' }).stepLine, 'Looks mixed', 'no starting number for nuclei-only runs');
  assert.equal(explain(normalise(mixed({ start_change: null })), { name, label: c => c + 1, startCode: 2 }).stepLine, 'Looks mixed · 1 % at the start');
  assert.equal(explain(normalise(mixed({ start_change: { points: 2.4, band: 3 } })), { name, label: c => c + 1, startCode: 2 }).stepLine,
    'Looks mixed · same as at the start · like for like', 'a raw rise that depth alone gave: like for like, said');
  // The worker's word, kept near the band as last shown, is what the window says (and normalise keeps it).
  const kept = normalise(mixed({ start_change: { points: 2.4, band: 3, direction: 'more', held: true } }));
  assert.deepEqual([kept.start_change.direction, kept.start_change.held], ['more', true]);
  assert.equal(explain(kept, { name, label: c => c + 1, startCode: 2 }).stepLine, 'Looks mixed · more than at the start · like for like');
  const odd = normalise(mixed({ start_change: { points: 2.4, band: 3, direction: 7, held: 'yes' } }));
  assert.deepEqual([odd.start_change.direction, odd.start_change.held], [null, false]);
});

test('the cell groups\' iteration is named only far from the result\'s or newer; the small lines go with their cells', async () => {
  assert.equal(GROUPS_LAG, 50);
  assert.deepEqual([groupsLag(425, 426), groupsLag(594, 595), groupsLag(300, 400), groupsLag(410, 400), groupsLag(null, 400)], [false, false, true, true, false]);
  const w = make({ answer: body => ready(mixed({ seq: body.seq, cell: body.cell })) });
  w.deeper.open(2); await settle();
  for (const label of [3, 4, 5, 6, 7, 8]) { w.deeper.select(label); await settle(); }
  assert.ok(w.deeper.histories.size <= KEEP_CELLS + 1, `kept ${w.deeper.histories.size}`);
  assert.ok(w.deeper.histories.has(7), 'the window\'s own cell keeps its line');
});

// ---- the round of 2026-10-31: the count proseg saves, the words on proseg's count, the small line --------------------------
test('an answer names the count it reads: "(as saved)" with the saved words on the count proseg saves, the averaged words otherwise', () => {
  assert.deepEqual(['sampled', 'averaged', undefined, 'other', null].map(count => normalise(mixed({ count })).count),
    ['sampled', 'averaged', 'averaged', 'averaged', 'averaged'], 'absent (an older engine): averaged');
  const name = c => CELLS[c] ?? `cell ${c}`, saved = normalise(mixed({ count: 'sampled' })), avg = normalise(mixed());
  const ms = explain(saved, { name, label: c => c + 1, startCode: 2 }), ma = explain(avg, { name, label: c => c + 1, startCode: 2 });
  assert.equal(ms.basis, "Proseg's count (as saved): 682"); assert.equal(ma.basis, "Proseg's count (averaged): 682");
  assert.equal(ms.basisTip, DEEP_TEXT.deepBasisNoBgSaved); assert.equal(ma.basisTip, DEEP_TEXT.deepBasisNoBg);
  assert.equal(basisTooltip(saved, { assigned: 452, kept: 287 }), deep('deepBasisSaved', { bg: '165' }));
  assert.equal(basisTooltip(avg, { assigned: 452, kept: 287 }), deep('deepBasis', { bg: '165' }));
  assert.equal(basisTooltip(normalise(mixed({ seq: 'start', count: 'sampled', transcripts: 161 }))), "At the start: 161, proseg's count at iteration 0.");
  // The change since the start stays like for like on the averaged count: said in the steps' tooltip on such an answer,
  // only while the line shows a like-for-like change.
  assert.equal(ms.stepTip, `${DEEP_TEXT.deepStepsTooltip} ${DEEP_TEXT.deepStartSaved}`); assert.equal(ma.stepTip, DEEP_TEXT.deepStepsTooltip);
  assert.equal(explain(normalise(mixed({ count: 'sampled', start_change: null })), { name, label: c => c + 1, startCode: 2 }).stepTip,
    DEEP_TEXT.deepStepsTooltip, 'the start\'s own number alone: nothing like for like to explain');
  assert.equal(explain(saved, { name, label: c => c + 1, startCode: 2, startKind: 'paired' }).stepTip, DEEP_TEXT.deepStepsTooltip, 'no start line');
  // The start's band kept from other runs' measurement (a tiled or whole run: constants.kept): said there too; a region's
  // nothing.
  assert.equal(explain(avg, { name, label: c => c + 1, startCode: 2, kept: ['glow', 'where_words', 'start_band'] }).stepTip,
    `${DEEP_TEXT.deepStepsTooltip} ${DEEP_TEXT.deepNotRemeasured}`);
  assert.equal(explain(saved, { name, label: c => c + 1, startCode: 2, kept: ['start_band'] }).stepTip,
    `${DEEP_TEXT.deepStepsTooltip} ${DEEP_TEXT.deepStartSaved} ${DEEP_TEXT.deepNotRemeasured}`);
  assert.equal(explain(avg, { name, label: c => c + 1, startCode: 2, kept: ['glow', 'where_words'] }).stepTip, DEEP_TEXT.deepStepsTooltip);
  for (const m of [ms, ma]) for (const t of [m.basis, m.basisTip, m.stepTip]) assert.deepEqual(bannedWords(t), [], t);
});

test('a frame counted again (the hold\'s count after a token\'s) is answered again: the new answer replaces the one held, the window says which count', async () => {
  const w = make({ live: false, run: { state: 'paused', constants: { mode: 'tiled', kept: ['glow', 'where_words', 'start_band'] } },
    answer: body => ready(mixed({ seq: body.seq })) });
  const stats = { 12: { assigned: 452, kept: 287 } };
  w.ctx.cellStats = (label, seq) => stats[seq] ?? null;
  w.deeper.open(2); await settle();
  const basis = w.root.one('.qc-deeper-basis'), step = w.root.one('p.qc-deeper-step');
  assert.equal(basis.textContent, "Proseg's count (averaged): 682"); assert.equal(basis.title, deep('deepBasis', { bg: '165' }));
  assert.equal(step.title, `${DEEP_TEXT.deepStepsTooltip} ${DEEP_TEXT.deepNotRemeasured}`, 'a tiled run: the start\'s band kept');
  const wanted = w.deeper.whereWanted.key;
  // The hold counts the frame again with the count proseg saves (the engine pushes the followed cell's answer again):
  // every number of the same frame now, the same iteration; the cell statistics of that frame are the saved count's too.
  stats[12] = { assigned: 452, kept: 290 };
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 12, iter: 1201, cell: 1, model: 'm1', result: mixed({ seq: 12, count: 'sampled' }) }); await settle();
  assert.equal(basis.textContent, "Proseg's count (as saved): 682"); assert.equal(basis.title, deep('deepBasisSaved', { bg: '162' }));
  assert.equal(step.title, `${DEEP_TEXT.deepStepsTooltip} ${DEEP_TEXT.deepStartSaved} ${DEEP_TEXT.deepNotRemeasured}`);
  assert.equal(w.deeper.results.get(1).get(12).count, 'sampled', 'held: the saved count\'s answer');
  // Show on the image asks again for the same frame on its new count (the twin's request is the same; the window's key
  // names the count).
  assert.notEqual(w.deeper.whereWanted.key, wanted); assert.deepEqual(w.deeper.whereWanted.params.seq, 12);
  // The same answer again is no new update.
  const applied = w.counters.qcDeeperApplied;
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 12, iter: 1201, cell: 1, model: 'm1', result: mixed({ seq: 12, count: 'sampled' }) }); await settle();
  assert.equal(w.counters.qcDeeperApplied, applied);
  for (const t of w.deeper.texts()) assert.deepEqual(bannedWords(t), [], t);
});

test('the small line: an answer\'s byte stands at its iteration, a frame counted again replacing it there; the answer shown draws its own', async () => {
  let release;
  const w = make({ live: false, run: { state: 'paused' }, answer: body => ready(mixed({ seq: body.seq })) });
  w.api.qcDeeperHistory = () => new Promise(resolve => { release = resolve; });
  w.deeper.open(2); await settle();
  // The history comes in: the engine's bytes over the run, iteration 1201's the averaged count's (116).
  release({ model: 'm1', start: 2, points: [[1, 2], [601, 3], [1201, 116]] }); await settle();
  assert.deepEqual(w.deeper.histories.get(1).points, [[1, 2], [601, 3], [1201, 116]]);
  // Counted again for the hold: the same iteration, another byte (the count proseg saves). It replaces the point there.
  w.deeper.event({ ev: 'deeper', run: 'r1', seq: 12, iter: 1201, cell: 1, model: 'm1', result: mixed({ seq: 12, count: 'sampled', code: 120 }) }); await settle();
  assert.deepEqual(w.deeper.histories.get(1).points, [[1, 2], [601, 3], [1201, 120]]);
  assert.deepEqual(w.deeper.sparkModel.points.at(-1), [1201, 120]); assert.equal(w.deeper.sparkModel.points.filter(p => p[0] === 1201).length, 1);
  // A history read before the hold's answer, merged after it (its point at 1201 the averaged one): the answer shown still
  // draws its own byte there.
  let late;
  const v = make({ live: false, run: { state: 'paused' }, answer: body => ready(mixed({ seq: body.seq, count: 'sampled', code: 120 })) });
  v.api.qcDeeperHistory = () => new Promise(resolve => { late = resolve; });
  v.deeper.open(2); await settle();
  late({ model: 'm1', start: 2, points: [[1, 2], [601, 3], [1201, 116]] }); await settle();
  assert.deepEqual(v.deeper.histories.get(1).points.at(-1), [1201, 116], 'the engine\'s history as it read it');
  assert.deepEqual(v.deeper.sparkModel.points.filter(p => p[0] === 1201), [[1201, 120]], 'the answer shown: its own byte');
});

/** A where reply as normaliseWhere returns it (no request to check against): one source; GYHNN (the cell's own) and GXPN
 *  (from that source) with `stats`; `into` as given; `held`, the reply's count (`count`: "sampled" or none). */
function whereOf({ held = 'exact', count = undefined, stats = null, into = [] } = {}) {
  const genes = ['GXPN', 'GXC', 'GYDNNN', 'GYHNN'], roles = ['from', 'from', 'from', 'own'];
  return normaliseWhere({ v: 3, seq: 12, iter: 1201, cell: 1, held, ...(count ? { count } : {}), box: [100, 100, 180, 160], genes, roles,
    points: { x: [110], y: [120], z: [0.5], gene: [0], owner: [1], near: [1], w: [held === 'none' ? 0 : 1] },
    gene_stats: stats ?? [
      { gene: 'GXPN', role: 'from', giver: 0, in_cell: 39, near_giver: 31, own_near: [21, 80], density: { cell: 29.1, giver: 40.6, around: 0.4 }, word: 'edge', count: 39 },
      { gene: 'GXC', role: 'from', giver: 0, in_cell: 10, near_giver: 3, own_near: [21, 80], density: { cell: 7.5, giver: 9.2, around: 5.1 }, word: 'layer', count: 10 },
      { gene: 'GYDNNN', role: 'from', giver: 0, in_cell: 3.6875, near_giver: 1, own_near: [21, 80], density: { cell: 3, giver: 2, around: 0 }, word: 'few', count: 3.6875 }],
    into }, null);
}
const SOURCE = { sources: [{ name: 'ffbbbbbb-1', head: 'ffbbbbbb-1 · makes GXPN', same: false, kind: ['GXPN'] }], labels: [3] };
const GLOW = { count: 3, likely: 2, near: 1, places: null, levels: null, row: null, notNear: null };

test('the cells around in proseg\'s count; too few under its floor; this cell\'s genes inside the neighbour\'s outline as proseg counts them, exact shares only', () => {
  const m = whereModel(whereOf(), SOURCE, [], GLOW);
  const [gxpn, gxc, few] = m.sources[0].genes;
  // H1: the density of the cells around is proseg's count of those cells per their own footprints, said so.
  assert.equal(gxpn.numbers[1], "Per 100 µm², in proseg's count: 29 in this cell · 41 in it · under 1 in the cells around.");
  assert.equal(gxc.tooltip.startsWith(DEEP_TEXT.deepWhereLayerTooltip), true); assert.match(gxc.tooltip, /in the cells around\./);
  // No other cell owns any of the box (around: none): no density line at all.
  const alone = whereModel(whereOf({ stats: [{ gene: 'GXPN', role: 'from', giver: 0, in_cell: 39, near_giver: 31, own_near: [21, 80],
    density: { cell: 29.1, giver: 40.6, around: null }, word: 'edge', count: 39 }] }), SOURCE, [], GLOW);
  assert.equal(alone.sources[0].genes[0].numbers.length, 1); assert.doesNotMatch(alone.sources[0].genes[0].tooltip, /Per 100/);
  // H4: a gene proseg counts 3.6875 of is "too few" (fewer than 4): its number reads 3.6, never 4.
  assert.equal(few.word, 'few'); assert.deepEqual(few.numbers, ['Proseg counts 3.6 for this cell.']); assert.equal(WHERE_FEW, 4);
  // H12: inside the neighbour's outline, proseg's count for this cell (sums of its shares): a gene once it is 1 or
  // more (1.4 reads 1; 0.4 is left out), numbers under 1 "under 1", none "0".
  const into = [{ source: 0, gene: 'GYHNN', inside: 1.4, near_cell: 0.3, source_own_near: [0.6, 12.5] },
    { source: 0, gene: 'GXEN', inside: 0.4, near_cell: 0.4, source_own_near: [0.6, 12.5] }, { source: 0, gene: 'GZCNN', inside: 2, near_cell: 0, source_own_near: null }];
  const exact = whereModel(whereOf({ into }), SOURCE, [], GLOW).sources[0].into;
  assert.deepEqual(exact.genes, ['GZCNN', 'GYHNN']);
  assert.equal(exact.text, "Inside the neighbour's outline, proseg counts for this cell: GZCNN 2 · GYHNN 1");
  assert.equal(exact.tooltip, `${DEEP_TEXT.deepWhereIntoTooltip} GZCNN: 2, 0 of them within 2 µm of this cell. GYHNN: 1, under 1 of them within 2 µm of this cell. `
    + "Other transcripts inside the neighbour's outline: proseg counts 13 of them for this cell, under 1 within 2 µm of this cell.");
  // Approximate shares put none of the cell's share inside a neighbour's outline (by how they are made), and none held
  // holds nothing: no into line on either, whatever their entries say.
  for (const held of ['approx', 'none']) assert.equal(whereModel(whereOf({ held, into }), SOURCE, [], GLOW).sources.find(src => src.into), undefined, held);
  assert.equal(whereModel(whereOf({ into: into.slice(1, 2) }), SOURCE, [], GLOW).sources[0].into, null, 'only under 1: no line');
});

test('read on the count proseg saves: the glow\'s, the genes\' and the into lines\' tooltips end saying so; kept constants say so before it', () => {
  const into = [{ source: 0, gene: 'GYHNN', inside: 4, near_cell: 3, source_own_near: [30, 400] }];
  const saved = whereModel(whereOf({ count: 'sampled', into }), SOURCE, [], GLOW), plain = whereModel(whereOf({ into }), SOURCE, [], GLOW);
  const tips = m => [m.line.tip, ...m.sources[0].genes.map(g => g.tooltip), m.sources[0].into?.tooltip];
  assert.equal(tips(saved).length, 5); for (const tip of tips(saved)) assert.ok(tip.endsWith(` ${DEEP_TEXT.deepWhereSaved}`), tip);
  for (const tip of tips(plain)) assert.doesNotMatch(tip, /count proseg saves/, tip);
  assert.equal(whereOf({ count: 'sampled' }).form, 'sampled'); assert.equal(whereOf().form, 'averaged'); assert.equal(whereOf().count, 1, 'count: its points');
  // After the approximate caveat; with nothing held nothing was read: neither said.
  const approx = whereModel(whereOf({ count: 'sampled', held: 'approx' }), SOURCE, [], GLOW);
  assert.ok(approx.line.tip.endsWith(`${DEEP_TEXT.deepGlowApprox} ${DEEP_TEXT.deepWhereSaved}`));
  const none = whereModel(whereOf({ count: 'sampled', held: 'none' }), SOURCE, [], GLOW);
  for (const tip of [none.line.tip, ...none.sources[0].genes.map(g => g.tooltip)]) assert.doesNotMatch(tip, /count proseg saves|re-measured/, tip);
  // A tiled or whole run (constants.kept): the glow's line and the words' lines say their thresholds were not measured
  // again for such runs, before the saved count's words; a region's run says nothing.
  const kept = whereModel(whereOf({ count: 'sampled', into }), SOURCE, [], GLOW, false, ['glow', 'where_words']);
  assert.ok(kept.line.tip.endsWith(`${DEEP_TEXT.deepNotRemeasured} ${DEEP_TEXT.deepWhereSaved}`));
  for (const g of kept.sources[0].genes) assert.ok(g.tooltip.endsWith(`${DEEP_TEXT.deepNotRemeasured} ${DEEP_TEXT.deepWhereSaved}`), g.gene);
  assert.doesNotMatch(kept.sources[0].into.tooltip, /re-measured/, 'the into line is no where word');
  const glowOnly = whereModel(whereOf(), SOURCE, [], GLOW, false, ['glow']);
  assert.ok(glowOnly.line.tip.endsWith(DEEP_TEXT.deepNotRemeasured)); assert.ok(glowOnly.sources[0].genes.every(g => !/re-measured/.test(g.tooltip)));
  const region = whereModel(whereOf(), SOURCE, [], GLOW, false, []);
  for (const tip of [region.line.tip, ...region.sources[0].genes.map(g => g.tooltip)]) assert.doesNotMatch(tip, /re-measured/, tip);
  for (const m of [saved, kept, approx]) for (const tip of tips(m).filter(Boolean)) assert.deepEqual(bannedWords(tip), [], tip);
});

test('too few to split: a count under the 100 reads under it, never 100; every gene number rounded once', async () => {
  assert.equal(SPLIT_MIN, 100);
  const few = n => explain(normalise({ v: 1, state: 'too_few', seq: 12, cell: 1, transcripts: n, genes: [['GXJNN', 2.469], ['GYBNNN', 1.5]] })).lines;
  assert.equal(few(99.71875)[0], 'Too few transcripts to split: proseg counts 99.7 for this cell in this iteration, in the genes the model knows, and at least 100 are needed.');
  assert.match(few(61.4)[0], /proseg counts 61 for this cell/); assert.match(few(99.4)[0], /proseg counts 99 for this cell/);
  assert.equal(few(61)[1], 'Its genes so far (not split): GXJNN 2 · GYBNNN 2', '2.469 reads 2 (rounded once: the engine sends 3 decimals)');
  assert.match(explain(normalise({ v: 1, state: 'too_few', seq: 'start', cell: 1, transcripts: 99.96875, genes: [] })).lines[0], /proseg counts 99\.9 for this cell at iteration 0/);
  // The table's and the rows' numbers: one rounding of the engine's (2.469 is 2 transcripts, its own part 2).
  const m = explain(normalise(mixed({ genes: [['GXPN', 39, 7.9, 31.1, 1, 0], ['GQQ', 2.469, 2.469, 0, 0]] })), { name: c => CELLS[c], label: c => c + 1 });
  assert.equal(m.table.find(g => g.gene === 'GQQ').tooltip, 'GQQ: 2 transcripts · about 2 its own, 0 from neighbours (estimate)');
  const w = make({ answer: body => ready(mixed({ seq: body.seq, genes: [['GXPN', 39, 7.9, 31.1, 1, 0], ['GQQ', 2.469, 2.469, 0, 0]] })) });
  w.deeper.open(2); await settle();
  const table = w.root.one('details.qc-deeper-table'); table.open = true; table.emit('toggle');
  assert.equal(w.root.all('tr[data-gene]').find(tr => tr.dataset.gene === 'GQQ').all('td')[1].textContent, '2');
});
