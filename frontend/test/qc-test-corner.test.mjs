import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  TestCorner, decodeLabels, normaliseView, shares, outline, geometry, breath, fadeProgress, drawFrame, drawGlow, drawCores, glowBox,
  identityRGB, parseColor, COUNTERS, COLORS, ICONS, ROLES, SIDES, PARTS, WENT, COST_ROWS, ACTIONS, PHASES, VIEW_MS, TICK_MS, GLOW_FPS, FADE_MS, REPLAY_MS,
  OF, MARKS, MARGIN, WIDTH, VIEW_PX, DONE_VIEW_PX, MIN_VIEW_PX, SIZES, CHROME_PX, DONE_WIDTH, VIEW_GAP, FRAME_PX, HALO_ALPHA, HALO, CROWD,
  BREATH_LOW, RETRY_MS, FINAL_TRIES, ANSWER_PX, ANSWER_GAP, WIDE_WIDTH, WIDE_STAGE, FORM_WIDTH, MOVED_BELOW,
} from '../src/qc-test-corner.mjs';
import { CORNER_TEXT, CORNER_TEMPLATES, corner, allTexts } from '../src/qc-test-corner-copy.mjs';
import { bannedWords, QC_TEXT as C, copy as QC_COPY } from '../src/qc-copy.mjs';
import { testReadout, testProgress, testWaiting, TEST_PHASES } from '../src/qc-panel.js';

// The test corner: the view of a test of a setting the panel's state machine drives. Its pure parts first (the deck,
// decoding, shapes, colours, geometry, the glow), then the corner itself in a small DOM with a virtual clock, a recording
// 2-D context and a fake engine view: the form (one setting, the panel's lines and the engine's refusals with their
// hooks, what a press would wait for and where it was not measured in Run's tooltip), a test that waits (one line and
// Cancel), the views of one run as now and one with the value (the engine's A1 and B1), the progress by side with the
// schedule's parts, and the answer part by part (the panel's words for the engine's example answers) with its hooks.
const EXAMPLES = JSON.parse(readFileSync(new URL('./fixtures/qc-test-result.json', import.meta.url), 'utf8'));
const RESULTS = Object.fromEntries(EXAMPLES.map(e => [e.name, e.result]));
const REASONS = JSON.parse(readFileSync(new URL('./fixtures/qc-test-reasons.json', import.meta.url), 'utf8'));

// ---- a small DOM ----------------------------------------------------------------------------------------------------
function dom() {
  const doc = { activeElement: null, hidden: false, listeners: new Map() };
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
    setPointerCapture() {}
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
    /** What a reader sees: hidden nodes left out. */
    visibleText() { return this.hidden ? '' : this._text + this.childNodes.map(c => c.nodeType === 3 ? c.textContent : c.visibleText()).join(' '); }
  }
  doc.createElement = tag => new Node(tag);
  doc.createTextNode = text => { const n = new Node('#text'); n._text = String(text); return n; };
  doc.body = new Node('body');
  const host = new Node('div'); doc.body.append(host);
  return { doc, host };
}
/** A 2-D context that records every call ([name, ...args]) and every property set (['=', name, value]). */
function recordingContext() {
  const calls = [];
  const t = { calls };
  return new Proxy(t, {
    get: (o, k) => k in o ? o[k] : k === 'createRadialGradient' ? (...a) => { calls.push([k, ...a]); return { addColorStop() {} }; }
      : (...args) => { calls.push([k, ...args]); },
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
const settle = async (n = 4) => { for (let i = 0; i < n; i++) await new Promise(resolve => setImmediate(resolve)); };
/** Advance `ms` in steps, letting the view's promises settle in between (as the browser would). */
async function run(s, ms, step = 25) { for (let t = 0; t < ms; t += step) { s.c.advance(Math.min(step, ms - t)); await settle(); } }

// ---- a test around one cell: a 60 µm crop, the cell (7) with a neighbour (9) on its right, another cell (3) ------------
const BOX = [100, 200, 160, 260];
const SUSPECTS = [{ x: 125, y: 225, gene: 'GXPN', p: 0.9 }, { x: 138, y: 230, gene: 'GXPN', p: 0.6 }, { x: 145, y: 230, gene: 'GXDN', p: 0.4 },
  { x: 130, y: 235, gene: 'GXC', p: 0.3 }];
const encode = labels => { const b = Buffer.alloc(labels.length * 4); labels.forEach((v, i) => b.writeUInt32LE(v, i * 4)); return b.toString('base64'); };
function labelsAt(voxel) {
  const w = 60 / voxel, h = 60 / voxel, labels = [];
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    const x = 100 + (i + 0.5) * voxel, y = 200 + (j + 0.5) * voxel;
    labels.push(x >= 120 && x < 140 && y >= 220 && y < 240 ? 7 : x >= 140 && x < 150 && y >= 220 && y < 240 ? 9 : x < 118 && y < 218 ? 3 : 0);
  }
  return { w, h, labels };
}
/** One run of the view reply, as the engine names it: the first run of side A (as now: the control view here) or of side
 *  B (with the value: the test view), role A1 or B1. `held`: the share of each suspect proseg counts for the cell there
 *  (as now: three kept, 3 of 4 added up, the third under MOVED_BELOW; with the value: one, 1 of 4); `by`: how. */
function runReply(role, seq, { iter = seq, voxel = seq > 101 ? 1 : 2, held = role === 'control' ? [1, 0.9, 0.1, 0.6] : [1, 0.15, 0.05, 0.1], by = 'exact', cell = 7, total = role === 'control' ? 519 : 475, parents, unsure } = {}) {
  const { w, h, labels } = labelsAt(voxel), side = SIDES[role];
  return { role: `${side}1`, side, seq, iter, voxel, x0: 100, y0: 200, w, h, labels: encode(labels), cell, sources: [9, null], held, held_by: by,
    cell_total: total, ...(parents ? { parents } : {}), ...(unsure ? { unsure } : {}) };
}
function viewReply({ seq = 5, state = 'running', control = {}, test = {}, testSeq = seq } = {}) {
  return { state, box: BOX, suspects: SUSPECTS, runs: [runReply('control', seq, control), runReply('test', testSeq, test)] };
}
// The panel's model, as qc-panel.js testModel gives it: the form's settings (far drift not tried yet), its lines and
// reasons; the setting under test; the engine's progress; the answer (the panel's testReadout of the engine's examples).
const SETTINGS = [
  { flag: 'cell-compactness', name: 'Cell compactness', now: 0.04, nowText: 'now 0.04', tried: true, note: '', tip: '', min: 0.001, max: 10 },
  { flag: 'prior-weight', name: 'Hold to the starting cells', now: 1, nowText: 'now 1', tried: true, note: '', tip: '', min: 0, max: 1000 },
  { flag: 'diffusion-probability', name: 'Far drift', now: 0.2, nowText: 'now 0.2', tried: false, note: C.testFarDriftNote, tip: C.testFarDriftTip, min: 0.1, max: 0.3 },
];
const POOL_LINE = { state: 'reached', text: 'Nearby cells with the same pattern: 23 cells, 187 suspects',
  tooltip: 'The test was measured with 160 suspects or more. With fewer it runs all the same, and its answer says so.', cells: 23, suspects: 187, need: 160 };
const FORM = (extra = {}) => ({ settings: SETTINGS, flag: 'cell-compactness', value: '', problems: [], notHere: '', field: null, line: '', pool: POOL_LINE,
  area: { rule: 'square', text: 'Area: a 500 × 500 µm square around this cell' }, runs: 'Your run waits · 3 runs as now, 3 with this value · about 45 s',
  how: C.testHow, reasons: [], waits: [], caveats: [], ...extra });
const TESTED = { flag: 'cell-compactness', name: 'Cell compactness', value: 0.02, valueText: '0.02', now: 0.04, nowText: 'now 0.04',
  line: 'Setting: Cell compactness 0.04 → 0.02 (lower = rounder cells)', rule: 'square', area: 'Area: a 500 × 500 µm square around this cell',
  square: [3150, 3150, 3650, 3650], perSide: 3, runs: 6, runsText: 'Your run waits · 3 runs as now, 3 with 0.02 · about 45 s', how: C.testHow,
  cancelTooltip: 'Stops the test and deletes its 6 runs. Your run goes on as it was before the test.' };
const PROGRESS = (extra = {}, state = 'running') => testProgress({ state, progress: { iter: 40, of: 450, runs: 6, done: 0, sides: { A: 40, B: 42 },
  marks: [200, 400], phase: 'sampling', retry: false, ...extra } });
const ANSWERS = { new: { text: C.testNewRun, tooltip: C.testNewRunTooltip }, keep: { text: C.testKeep, tooltip: C.testKeepTooltip },
  again: { text: C.testRunAgain, tooltip: C.testRunAgainTooltip } };
/** The answer the panel hands over for example `name` of the engine's answers. */
const RESULT = (name = 'not_clear_apart', extra = {}) => ({ ...testReadout({ value: RESULTS[name].setting.value, result: RESULTS[name] }), repeat: null,
  ...ANSWERS, ...extra });
/** What a test waits for, as the panel says it (testWaiting) for the engine's waits (codes). */
const WAITING = (...codes) => testWaiting({ waiting: codes.map(code => ({ code })) });
const model = (phase, extra = {}) => ({
  open: true, phase, run: 'r1', cell: { label: 8, name: 'ffaaaaaa-1' }, start: 'paired', hold: null,
  form: phase === 'form' ? FORM() : null, test: phase === 'form' ? null : phase === 'waiting' ? { ...TESTED, rule: null, area: '', square: null,
    cancelTooltip: C.testCancelWaitingTooltip } : TESTED,
  waiting: phase === 'waiting' ? WAITING('not_at_end', 'check_off') : null,
  progress: ['preparing', 'running', 'comparing'].includes(phase) ? PROGRESS(phase === 'preparing' ? { phase: 'preparing', iter: 0, sides: { A: 0, B: 0 } } : {}, phase) : null,
  result: phase === 'done' ? RESULT() : null, message: null, busy: null, busyText: null, ...extra,
});
function setup({ width = 1200, height = 800, reply = () => viewReply(), fetchView = null, reduced = false, calm = false, colorOf } = {}) {
  const { doc, host } = dom(); host.clientWidth = width; host.clientHeight = height;
  const c = clock(), counts = {}, actions = [], boxes = [], asked = [];
  const hook = { count(name, n = 1) { counts[name] = (counts[name] || 0) + n; } };
  const motion = { reduced, calm };
  const corner = new TestCorner(host, {
    clock: c, hook, onAction: (a, p) => actions.push(p === undefined ? [a] : [a, p]), onBox: b => boxes.push(b),
    fetchView: fetchView ?? (run => { asked.push([run, c.t]); return Promise.resolve(reply(asked.length)); }),
    reducedMotion: () => motion.reduced, calmMotion: () => motion.calm, ...(colorOf ? { colorOf } : {}),
  });
  return { doc, host, c, counts, actions, boxes, asked, corner, motion, root: corner.root };
}
const visible = node => { if (!node) return false; for (let n = node; n; n = n.parentElement) if (n.hidden) return false; return true; };
const buttons = (root, action) => root.all(`button[data-action=${action}]`).filter(visible);

// ---- the deck -------------------------------------------------------------------------------------------------------
// The mixing check's honesty rules on what the page may say, and Go deeper's own plain-word list.
const UNFIXABLE = new RegExp([
  String.raw`\b(?:no|none\s+of\s+the|not\s+any)\s+(?:\w+\s+){0,2}settings?\b[^.;:]{0,40}?\b(?:fix|solve|cure)`,
  String.raw`\bsettings?\s+(?:\w+\s+){0,2}(?:are|is|were|was|seem|seems)\s+(?:un|not\s+)likely\s+to\s+(?:fix|solve|help)`,
  String.raw`\b(?:can(?:no|')t|cannot|won'?t|will\s+not|could\s*n[o']t)\s+(?:be\s+)?(?:fix(?:ed)?|solved?|cured?)\b`,
  String.raw`\b(?:unfixable|not\s+fixable|beyond\s+(?:the\s+reach\s+of\s+)?(?:any\s+)?settings?)\b`,
  String.raw`\bnothing\s+(?:\w+\s+){0,3}(?:will|can|could|would)\s+(?:fix|solve)`, String.raw`\bnot\s+the\s+settings?\b`].join('|'), 'i');
const NOTEBOOK = /\bnotebook\b|\bpart\s*3\b|\bresolvi\b|\bsame\s+(?:settings|model|recipe)\b/i;
const STANDARDISED = /\badjusted\s+for\s+depth\b|\bdepth[- ]adjusted\b|\bstandardi[sz]ed\b/i;
const DEEPER_PLAIN = /\b(band|depth-matched|compactness|posterior|ambient RNA|contaminat)/i;
function plainWords(text, where) {
  assert.deepEqual(bannedWords(text), [], `${where}: ${text}`);
  for (const rule of [UNFIXABLE, NOTEBOOK, STANDARDISED, DEEPER_PLAIN]) assert.doesNotMatch(text, rule, `${where}: ${text}`);
}
/** The words the corner shows of its own: those that hold none of the panel's (its model's strings: setting names,
 *  lines, the answer's words, checked by the panel's own tests: qc-test.test.mjs, qc-copy.test.mjs). */
function ownTexts(s) {
  const strings = [];
  const walk = v => { if (typeof v === 'string') { if (v.trim().length >= 3) strings.push(v.trim()); } else if (Array.isArray(v)) v.forEach(walk); else if (v && typeof v === 'object') Object.values(v).forEach(walk); };
  walk(s.corner.model);
  return s.corner.texts().filter(t => !strings.some(x => t.includes(x)));
}

test('the deck: plain words under every rule; templates fill every value and fail early without one', () => {
  for (const [key, text] of Object.entries({ ...CORNER_TEXT, ...CORNER_TEMPLATES })) plainWords(text, key);
  for (const text of [...allTexts(), ...TestCorner.allTexts()]) plainWords(text, 'allTexts');
  assert.deepEqual(Object.keys(CORNER_TEXT).filter(k => k in CORNER_TEMPLATES), []);
  // Every word of the corner is the deck's or the panel's (the views that may be older said from the deck too).
  assert.deepEqual(TestCorner.allTexts(), allTexts());
  for (const key of ['stale', 'staleTooltip', 'unsureTooltip', 'asNow', 'aboutLabel']) assert.ok(allTexts().includes(CORNER_TEXT[key]), key);
  const source = readFileSync(new URL('../src/qc-test-corner.mjs', import.meta.url), 'utf8');
  assert.doesNotMatch(source, /Couldn't get the views|Do they come back|Fewer than their own|What it cost|Where the suspects went|Nearby cells/,
    'no words of its own in the component');
  for (const [key, pattern] of Object.entries(CORNER_TEMPLATES)) {
    const names = [...new Set([...pattern.matchAll(/\{([^}]+)\}/g)].map(m => m[1]))];
    const values = Object.fromEntries(names.map((n, i) => [n, `V${i}`]));
    const text = corner(key, values);
    assert.doesNotMatch(text, /\{[^}]+\}|undefined|null/, key);
    for (const name of names) { const missing = { ...values }; delete missing[name]; assert.throws(() => corner(key, missing), TypeError, `${key}.${name}`); }
  }
  assert.throws(() => corner('nope'), RangeError);
  // The corner's words: the views' counts and titles (as now, with the value: the header names the setting); the cell's
  // number is proseg's count, never the transcripts inside its outline.
  assert.equal(corner('suspectsInside', { a: 12, n: 15 }), 'Suspects kept: 12 of 15');
  assert.equal(corner('transcripts', { t: 519 }), "Proseg's count: 519");
  // Proseg's count of the cell is its averaged count (round of its sum of e), and its tooltip says so.
  assert.equal(CORNER_TEXT.totalTooltip, "Proseg's count of this cell in this run, averaged over its random background draw.");
  // A teal ring is a suspect proseg keeps under a fifth of (MOVED_BELOW) for the cell: "Suspects kept" adds its share in,
  // so its words never say proseg no longer counts it.
  assert.equal(CORNER_TEXT.suspectsTooltip, 'Transcripts proseg counted for this cell that likely came from a neighbour. Glowing: proseg still counts it for this cell in this run (brighter: a larger share). Teal ring: proseg keeps under a fifth of it for this cell here.');
  assert.deepEqual([CORNER_TEXT.keySuspectTooltip, CORNER_TEXT.keyMoved, CORNER_TEXT.keyMovedTooltip], ['Glowing: proseg still counts it for this cell.',
    'Mostly left the cell', 'Teal ring: proseg keeps under a fifth of it for this cell in that run.']);
  assert.equal(MOVED_BELOW, 0.2, 'a fifth');
  for (const text of allTexts()) assert.doesNotMatch(text, /no longer counts/, text);
  assert.equal(CORNER_TEXT.approxTooltip, "Approximate in this view: proseg's exact share comes with a later iteration.");
  assert.equal(CORNER_TEXT.approxDoneTooltip, "Approximate in this view: placed from this run's own layers, as no exact share of proseg's came for it.");
  assert.deepEqual([CORNER_TEXT.notPlaced, CORNER_TEXT.notPlacedTooltip], ['Suspects not placed in this view',
    "Proseg's share of these suspects isn't known in this view's iteration, so none is drawn or counted here."]);
  for (const key of ['approxTooltip', 'approxDoneTooltip', 'notPlaced', 'notPlacedTooltip']) assert.ok(allTexts().includes(CORNER_TEXT[key]), key);
  for (const text of allTexts()) assert.doesNotMatch(text, /inside (?:this|its|the) cell's outline|Its transcripts|Suspects inside/, text);
  assert.equal(CORNER_TEXT.asNow, 'As now'); assert.equal(corner('withValue', { v: '0.08' }), 'With 0.08');
  assert.equal(corner('sub', { setting: 'Far drift', cell: 'ffaaaaaa-1' }), 'Far drift · ffaaaaaa-1');
  // No second setting, no progress or answer rows of its own any more (the panel's words: the engine's progress and answer).
  for (const key of ['titleFormTwo', 'titleRunningTwo', 'addSecond', 'addSecondTooltip', 'dropSecond', 'secondLabel', 'yours', 'yoursLabel', 'suspectsLeft',
    'otherTranscripts', 'allTranscripts', 'settling', 'refining', 'preparing', 'unsureRowTooltip']) assert.ok(!(key in CORNER_TEXT), key);
  for (const key of ['setting', 'pair', 'progress', 'progressTooltip', 'comparing', 'row', 'change', 'suspectsOf']) assert.ok(!(key in CORNER_TEMPLATES), key);
  // The engine's refusals in two groups, each under a short heading with what it means in its tooltip; no lead of the
  // whole list, and "Check these first:" leads only the page's own problems (a refused press says the panel's line).
  // Nothing is refused for not being measured: no heading for it.
  assert.ok(!('reasonsLead' in CORNER_TEXT));
  assert.deepEqual([CORNER_TEXT.reasonsFirst, CORNER_TEXT.reasonsCant], ['First:', "Can't test here:"]);
  for (const key of ['reasonsUnmeasured', 'reasonsUnmeasuredTooltip']) assert.ok(!(key in CORNER_TEXT), key);
  assert.equal(CORNER_TEXT.reasonsFirstTooltip, 'Wait for this to end before the test can run.');
  assert.equal(CORNER_TEXT.reasonsCantTooltip, "The test has nothing to measure for this cell, or it can't run on this run or sample.");
  // The answer's ⓘ that says where the test was not measured when no "fewer" part carries it; the fewer part's, named for
  // both when its tooltip carries both.
  assert.equal(CORNER_TEXT.caveatsLabel, 'Where this test was not measured');
  assert.equal(CORNER_TEXT.aboutCaveatsLabel, 'How often this test is wrong, and where it was not measured');
  assert.ok(CORNER_TEXT.aboutCaveatsLabel.startsWith(CORNER_TEXT.aboutLabel));
  // Whatever is said about where a suspect came from is a likelihood (the model splits per gene); what proseg counts is
  // proseg's.
  assert.match(CORNER_TEXT.suspectsTooltip, /likely came from a neighbour/);
  for (const key of ['suspectsTooltip', 'keySuspectTooltip', 'keyMovedTooltip', 'totalTooltip']) assert.match(CORNER_TEXT[key], /[Pp]roseg/, key);
  // The suspects the runs couldn't tell apart: one muted line, its tooltip saying why and that they are left out.
  assert.equal(corner('unsure', { n: 2 }), "2 couldn't be told apart from another transcript at the same place");
  assert.match(CORNER_TEXT.unsureTooltip, /same gene/); assert.match(CORNER_TEXT.unsureTooltip, /left out/); assert.match(CORNER_TEXT.unsureTooltip, /couldn't tell/);
  // The test's own clash words never appear (each means something else in the app), nor "hold".
  for (const text of allTexts()) assert.doesNotMatch(text, /look-alike|\bsimilar\b|same kind|like for like|\bhold\b/i, text);
});

// ---- pure helpers ---------------------------------------------------------------------------------------------------
test('labels: base64 of little-endian u32, row-major; a wrong size or anything else is refused', () => {
  const values = [0, 1, 7, 255, 256, 65536, 2 ** 31 + 5, 2 ** 32 - 1];
  assert.deepEqual([...decodeLabels(encode(values), values.length)], values);
  assert.equal(decodeLabels(encode(values), values.length + 1), null);
  assert.equal(decodeLabels(encode(values), 0), null);
  assert.equal(decodeLabels(12, 1), null);
  assert.equal(decodeLabels('***', 1), null);
});

test('a frame\'s shapes, cell by cell: the cell, its neighbours and the others; fills merged along rows; each outline on its own side', () => {
  // 5 x 2: [0 7 7 9 3] / [0 7 7 9 9]
  const s = outline(Uint32Array.from([0, 7, 7, 9, 3, 0, 7, 7, 9, 9]), 5, 2, 7, [9, null]);
  assert.deepEqual([...s.ids], [7, 9, 3], 'a slot each, in the order met');
  assert.deepEqual([...s.kinds], [3, 2, 1], 'the cell, a neighbour, another cell');
  assert.deepEqual([...s.fills[0]], [1, 0, 3, 1, 1, 3], 'the cell: one run per row');
  assert.deepEqual([...s.fills[1]], [3, 0, 4, 3, 1, 5]);
  assert.deepEqual([...s.fills[2]], [4, 0, 5]);
  // [x0, y0, x1, y1, side]: side 1 when the cell lies after the line (right of it, below it), -1 before it. Between the cell
  // and the neighbour (x = 3) both have a line, each on its own side; the background has none.
  assert.deepEqual([...s.edges[0]], [1, 0, 1, 2, 1, 3, 0, 3, 2, -1]);
  assert.deepEqual([...s.edges[1]], [3, 0, 3, 2, 1, 4, 0, 4, 1, -1, 4, 1, 5, 1, 1]);
  assert.deepEqual([...s.edges[2]], [4, 0, 4, 1, 1, 4, 1, 5, 1, -1]);
  const none = outline(Uint32Array.from([3, 3, 0, 4]), 2, 2, null, []);
  assert.deepEqual([...none.kinds], [1, 1], 'a cell missing from the run: no white');
});

test('the view reply is checked: the engine\'s A1 and B1 in the control and test views, bad runs dropped, a suspect without a place keeps its index, held lines up, parents read', () => {
  const v = normaliseView({ state: 'running', box: BOX, suspects: [...SUSPECTS, { gene: 'X' }],
    runs: [runReply('control', 3), { ...runReply('test', 3), w: 0 }, { ...runReply('test', 4), role: 'C1', side: 'C' }, { ...runReply('test', 5), held: [1] }] });
  assert.deepEqual(v.box, BOX); assert.equal(v.suspects.length, 5); assert.equal(v.suspects[4].ok, false);
  assert.deepEqual(v.runs.map(r => [r.role, r.run, r.seq]), [['control', 'A1', 3], ['test', 'B1', 5]], 'side A as now, side B with the value');
  assert.ok([...v.runs[0].held].every(Number.isNaN), 'four shares for five suspects: none known');
  assert.deepEqual([...v.runs[0].unsure], [0, 0, 0, 0, 0], 'an engine that names none unsure: none is');
  // Each suspect's share proseg counts for the cell (clamped to 0-1), null where the engine knows none; how it was placed.
  const h = normaliseView({ box: BOX, suspects: SUSPECTS, runs: [runReply('control', 1, { held: [0.5, null, 1.5, -1], by: 'approx' }),
    runReply('test', 1, { by: 'nearby' })] }).runs;
  assert.deepEqual([...h[0].held].map(x => (Number.isNaN(x) ? null : x)), [0.5, null, 1, 0]); assert.equal(h[0].heldBy, 'approx');
  assert.equal(h[1].heldBy, null, 'a way the page does not know: none said');
  assert.deepEqual(shares(h[0].held, 7), { weights: Float64Array.from([0.5, 0, 1, 0]), moved: Uint8Array.from([0, 0, 0, 1]), kept: 2, of: 3 },
    'glowing from 0.2 by its share; under it a teal ring; 1.5 of 3 kept rounds to 2');
  assert.deepEqual(shares(h[0].held, null), { weights: new Float64Array(4), moved: new Uint8Array(4), kept: 0, of: 0 }, 'without the cell: none known');
  assert.equal(MOVED_BELOW, 0.2);
  // By side, else by the role's side; an older engine's roles as they are.
  const by = r => normaliseView({ box: BOX, suspects: [], runs: [{ ...runReply('control', 1), ...r }] }).runs.map(x => x.role);
  assert.deepEqual(by({ role: 'B1', side: undefined }), ['test']); assert.deepEqual(by({ role: 'A2', side: undefined }), ['control']);
  assert.deepEqual(by({ role: 'control', side: undefined }), ['control']); assert.deepEqual(by({ role: 'test', side: undefined }), ['test']);
  assert.deepEqual(by({ role: 'A1', side: 'B' }), ['test'], 'the side says'); assert.deepEqual(by({ role: 'other', side: undefined }), []);
  // runs[k].unsure, parallel to held; one that does not line up says none.
  const u = normaliseView({ box: BOX, suspects: SUSPECTS, runs: [runReply('control', 1, { held: [1, 1, 0, 0], unsure: [0, 1, 1, 0] }),
    runReply('test', 1, { unsure: [1] })] });
  assert.deepEqual([...u.runs[0].unsure], [0, 1, 1, 0]); assert.deepEqual([...u.runs[1].unsure], [0, 0, 0, 0]);
  assert.deepEqual([...u.runs[0].held].map(x => (Number.isNaN(x) ? null : x)), [1, null, null, 0],
    'never a share (the engine never says both; were it to, it stays out)');
  assert.deepEqual(v.runs[0].sources, [9, null]); assert.equal(v.runs[0].cell, 7);
  assert.equal(v.runs[0].parents, null, 'no parents named: none');
  assert.equal(normaliseView(null), null);
  assert.equal(normaliseView({ box: [1, 2, 1, 5], runs: [] }).box, null);
  assert.equal(normaliseView({ runs: [{ ...runReply('control', 1), cell: 0 }] }).runs[0].cell, null);
  // Each cell's label in your run (flat pairs); a pair with a cell your run does not have (0) or not a label is left out.
  const p = normaliseView({ box: BOX, suspects: [], runs: [runReply('control', 1, { parents: [7, 1007, 9, 1009, 3, 0, 'x', 5, 4] })] }).runs[0].parents;
  assert.deepEqual([...p], [[7, 1007], [9, 1009]]);
});

test('geometry, breath and fades: the box centred in the view; halos 6 times the core (24-44 CSS px); 60-100 % over 1.6 s', () => {
  const g = geometry(BOX, 240, 1);
  assert.equal(g.k, 4); assert.equal(g.ox, 0); assert.deepEqual(g.clip, [0, 0, 240, 240]);
  assert.equal(g.rc, 3.5); assert.equal(g.rh, 21); assert.equal(g.ring, 4.5); assert.equal(HALO, 6);
  assert.equal(g.grey, g.rc, 'a suspect that could not be told apart: a grey ring the core\'s size, inside the teal one');
  const wide = geometry([0, 0, 100, 50], 200, 2);
  assert.equal(wide.k, 2); assert.equal(wide.oy, 50); assert.equal(wide.rc, 3.5, '1.75 CSS px at the least'); assert.equal(wide.rh, 24, '24 CSS px across at the least');
  for (const side of [20, 48, 90, 140, 400]) for (const dpr of [1, 2]) {
    const q = geometry([0, 0, side, side], 224 * dpr, dpr), across = (2 * q.rh) / dpr;
    assert.ok(across >= 24 && across <= 44, `${across} px across`); assert.ok(Math.abs(q.rh - Math.min(Math.max(HALO * q.rc, 12 * dpr), 22 * dpr)) < 1e-9);
  }
  assert.equal(BREATH_LOW, 0.6);
  assert.ok(Math.abs(breath(0) - 0.6) < 1e-9); assert.ok(Math.abs(breath(800) - 1) < 1e-9); assert.ok(Math.abs(breath(1600) - 0.6) < 1e-9);
  for (let t = 0; t < 3200; t += 37) assert.ok(breath(t) >= 0.6 - 1e-9 && breath(t) <= 1 + 1e-9);
  assert.ok(fadeProgress(0) > 0 && fadeProgress(0) < 1 && fadeProgress(TICK_MS) < 1);
  assert.equal(fadeProgress(FADE_MS), 1); assert.equal(TICK_MS, 84); assert.equal(GLOW_FPS, 12);
});

test('colours: the stage\'s identity colour of a label (as main.js swatch), and the page\'s CSS colours read', () => {
  // main.js swatch(label), the stage's identityColor in JS (the reference this port must match).
  const swatch = label => {
    const id = label >>> 0;
    let h = Math.imul(id, 2654435761) >>> 0;
    h = (h ^ (h >>> 15)) >>> 0; h = Math.imul(h, 2246822519) >>> 0; h = (h ^ (h >>> 13)) >>> 0;
    const hue = ((id * 0.6180339887) % 1 + (h & 255) / 1400) % 1;
    const sat = 0.52 + 0.33 * ((h >>> 8) & 255) / 255, val = 0.82 + 0.18 * ((h >>> 16) & 255) / 255;
    const k = n => { const x = Math.abs(((hue * 6 + n) % 6) - 3) - 1; return val * (1 - sat + sat * Math.min(1, Math.max(0, x))); };
    return `rgb(${[k(0), k(4), k(2)].map(v => Math.round(v * 255)).join(',')})`;
  };
  for (const label of [1, 2, 3, 7, 9, 1007, 40000, 2 ** 31 + 3]) {
    const c = identityRGB(label);
    assert.equal(`rgb(${c.join(',')})`, swatch(label), `label ${label}`);
    assert.ok(c.every(v => Number.isInteger(v) && v >= 0 && v <= 255));
  }
  assert.deepEqual(parseColor('#ff4fa3'), [255, 79, 163]); assert.deepEqual(parseColor(' #0072B2 '), [0, 114, 178]);
  assert.deepEqual(parseColor('rgb(1, 2, 3)'), [1, 2, 3]); assert.deepEqual(parseColor('rgba(10,20,30,0.5)'), [10, 20, 30]);
  assert.deepEqual(parseColor(swatch(7)), identityRGB(7));
  for (const bad of ['red', '#fff', 'rgb(1,2)', '', null, 7]) assert.equal(parseColor(bad), null, String(bad));
});

/** The paths of a recorded context: each beginPath ... fill/stroke with its rects, moves and style (save and restore
 *  keep and bring back the style, as a context does). */
function paths(calls) {
  const out = [], kept = []; let cur = null, style = {};
  for (const call of calls) {
    if (call[0] === '=') { style[call[1]] = call[2]; continue; }
    if (call[0] === 'save') { kept.push({ ...style }); continue; }
    if (call[0] === 'restore') { style = kept.pop() ?? {}; continue; }
    if (call[0] === 'beginPath') { cur = { rects: [], moves: [], lines: [], arcs: [] }; continue; }
    if (!cur) continue;
    if (call[0] === 'rect') cur.rects.push(call.slice(1));
    else if (call[0] === 'moveTo') cur.moves.push(call.slice(1));
    else if (call[0] === 'lineTo') cur.lines.push(call.slice(1));
    else if (call[0] === 'arc') cur.arcs.push(call.slice(1));
    else if (call[0] === 'fill' || call[0] === 'stroke') out.push({ ...cur, op: call[0], fillStyle: style.fillStyle, strokeStyle: style.strokeStyle, lineWidth: style.lineWidth, shadowBlur: style.shadowBlur });
  }
  return out;
}

test('a frame is drawn at its own grid, by its own origin: cells faintly in their colours, the cell white and glowing, neighbours magenta', () => {
  const g = geometry(BOX, 240, 1), v = normaliseView(viewReply({ seq: 5 }));
  const r = v.runs[0], { labels } = labelsAt(2);
  const shape = outline(Uint32Array.from(labels), r.w, r.h, r.cell, r.sources), frame = { ...r, ...shares(r.held, r.cell), shape };
  const colors = Array.from(shape.ids, l => (l === 7 ? [200, 100, 0] : l === 9 ? [0, 100, 200] : [10, 200, 10]));
  const ctx = recordingContext();
  assert.equal(drawFrame(ctx, frame, g, v.suspects, colors), 1, 'one suspect is outside the cell in this run');
  const p = paths(ctx.calls);
  const cellFill = p.find(x => x.op === 'fill' && x.fillStyle === `rgba(200,100,0,${COLORS.fill[3]})`);
  assert.equal(cellFill.rects.length, 10, 'one run per row of the cell (2 µm: 10 rows)');
  assert.deepEqual(cellFill.rects[0], [80, 80, 80, 8], 'x 120-140 µm, y 220-222 µm at 4 px per µm');
  assert.ok(p.some(x => x.op === 'fill' && x.fillStyle === `rgba(0,100,200,${COLORS.fill[2]})`), 'the neighbour in its colour');
  assert.ok(p.some(x => x.op === 'fill' && x.fillStyle === `rgba(10,200,10,${COLORS.fill[1]})`), 'another cell in its colour');
  assert.ok(COLORS.fill.every(a => a <= 0.3), 'faint: the glow leads');
  const white = p.find(x => x.op === 'stroke' && x.strokeStyle === '#ffffff');
  assert.equal(white.lineWidth, 2); assert.equal(white.shadowBlur, 6, 'a soft glow around it');
  assert.ok(white.moves.some(([x, y]) => x === 81 && y === 80), 'the cell\'s left side at x = 120 µm, its line inside it');
  const magenta = p.find(x => x.op === 'stroke' && x.strokeStyle === '#d55181');
  assert.ok(magenta.moves.some(([x]) => x === 199), 'the neighbour\'s right side at x = 150 µm, its line inside it');
  assert.ok(!p.filter(x => x !== white).some(x => x.shadowBlur > 0), 'only the cell glows');
  // Another cell's outline: its own colour, lighter (35 % toward white), at 55 %.
  assert.ok(p.some(x => x.op === 'stroke' && x.strokeStyle === 'rgba(96,219,96,0.55)'));
  const teal = p.find(x => x.op === 'stroke' && x.strokeStyle === COLORS.moved);
  assert.deepEqual(teal.arcs, [[180, 120, 4.5, 0, 2 * Math.PI]], 'the suspect at (145, 230) µm, proseg no longer counting it for the cell');
  assert.ok(!p.some(x => x.strokeStyle === COLORS.unsure), 'none unsure: no grey ring');
  // A suspect the engine couldn't tell apart: a plain small grey ring, never teal
  // (no share either way) and never glowing (drawGlow and drawCores weigh by the shares, 0 for it); drawn also in a run
  // without the cell.
  const doubt = { ...frame, moved: Uint8Array.from([0, 0, 1, 0]), unsure: Uint8Array.from([0, 0, 1, 0]) };
  const dctx = recordingContext();
  assert.equal(drawFrame(dctx, doubt, g, v.suspects, colors), 0, 'the one outside the cell is unsure: no teal ring');
  const grey = paths(dctx.calls).filter(x => x.op === 'stroke' && x.strokeStyle === COLORS.unsure);
  assert.equal(grey.length, 1); assert.deepEqual(grey[0].arcs, [[180, 120, 3.5, 0, 2 * Math.PI]], 'the core\'s size, smaller than a teal ring');
  assert.equal(grey[0].lineWidth, 1); assert.ok(!paths(dctx.calls).some(x => x.strokeStyle === COLORS.moved));
  assert.equal(COLORS.unsure, '#9aa4b2', 'plain grey: neither the glow\'s pink nor the moved-out teal');
  const gone = recordingContext(); drawFrame(gone, { ...doubt, cell: null }, g, v.suspects, colors);
  assert.equal(paths(gone.calls).filter(x => x.strokeStyle === COLORS.unsure).length, 1, 'whether the cell is in the run or not');
  // Without colours: each cell in the stage's own colour for its label.
  const own = recordingContext(); drawFrame(own, frame, g, v.suspects);
  assert.ok(paths(own.calls).some(x => x.op === 'fill' && x.fillStyle === `rgba(${identityRGB(7).join(',')},${COLORS.fill[3]})`));
  // The fine grid of the same crop: 1 µm, so twice the rows; another origin moves everything with it.
  const fine = normaliseView(viewReply({ seq: 120 })).runs[0], f = labelsAt(1), fs = outline(Uint32Array.from(f.labels), fine.w, fine.h, 7, [9]);
  const ctx2 = recordingContext();
  drawFrame(ctx2, { ...fine, x0: 99, shape: fs }, g, v.suspects, Array.from(fs.ids, l => (l === 7 ? [200, 100, 0] : [1, 2, 3])));
  const fineFill = paths(ctx2.calls).find(x => x.op === 'fill' && x.fillStyle === `rgba(200,100,0,${COLORS.fill[3]})`);
  assert.equal(fineFill.rects.length, 20); assert.deepEqual(fineFill.rects[0], [76, 80, 80, 4], 'placed by its own x0 (99 µm)');
});

test('the glow: halos 6 times the core as bright as the likelihood and the breath, cores bright and still; the patch around them', () => {
  const g = geometry(BOX, 240, 1), v = normaliseView(viewReply()), ctx = recordingContext(), sprite = { sprite: true };
  const weights = Float32Array.from([1, 1, 0, 1]);
  assert.equal(drawGlow(ctx, v.suspects, weights, g, 1, sprite), 3);
  const halos = ctx.calls.filter(c => c[0] === 'drawImage');
  assert.deepEqual(halos[0], ['drawImage', sprite, 79, 79, 42, 42], 'the suspect at (125, 225) µm: 42 px across, 6 times its core');
  const alphas = ctx.calls.filter(c => c[0] === '=' && c[1] === 'globalAlpha').map(c => c[2]);
  assert.ok(Math.abs(alphas[0] - HALO_ALPHA * 0.9) < 1e-6 && Math.abs(alphas[2] - HALO_ALPHA * 0.3) < 1e-6, 'brightness from the likelihood');
  assert.ok(ctx.calls.some(c => c[0] === '=' && c[1] === 'globalCompositeOperation' && c[2] === 'screen'), 'a crowd deepens, never burns out');
  assert.equal(ctx.calls.filter(c => c[0] === 'arc').length, 0, 'halos only: the cores are on the frame');
  const dim = recordingContext();
  drawGlow(dim, v.suspects, weights, g, BREATH_LOW, sprite);
  assert.ok(Math.abs(dim.calls.filter(c => c[0] === '=' && c[1] === 'globalAlpha')[0][2] - HALO_ALPHA * 0.9 * BREATH_LOW) < 1e-6, 'the breath');
  assert.equal(drawGlow(recordingContext(), v.suspects, new Float32Array(4), g, 1, sprite), 0);
  // The cores: a white-pink sprite a little past the core's size, strong even for the least likely.
  const cores = recordingContext(), core = { core: true };
  assert.equal(drawCores(cores, v.suspects, weights, g, core), 3);
  const c0 = cores.calls.filter(c => c[0] === 'drawImage')[0];
  assert.equal(c0[1], core); assert.ok(Math.abs(c0[2] - (100 - 3.5 * 1.3)) < 1e-9 && Math.abs(c0[4] - 2 * 3.5 * 1.3) < 1e-9);
  const ca = cores.calls.filter(c => c[0] === '=' && c[1] === 'globalAlpha').map(c => c[2]);
  assert.ok(Math.abs(ca[0] - (0.55 + 0.45 * 0.9)) < 1e-9 && Math.abs(ca[2] - (0.55 + 0.45 * 0.3)) < 1e-9);
  assert.deepEqual(COLORS.core[0], [0, 'rgba(255,244,250,1)'], 'white-pink in the middle');
  const plainCore = recordingContext(); drawCores(plainCore, v.suspects, weights, g);
  assert.deepEqual(plainCore.calls.filter(c => c[0] === 'arc')[0], ['arc', 100, 100, 3.5, 0, 2 * Math.PI], 'without a sprite: a dot');
  // A crowd (96 glowing, four times CROWD): the cores at half strength and smaller, the halos softer.
  const many = Array.from({ length: 96 }, (_, k) => ({ x: 121 + (k % 12) * 1.5, y: 221 + Math.floor(k / 12) * 2, gene: 'GXPN', p: 1, ok: true }));
  const crowded = recordingContext(), crowdedCores = recordingContext();
  drawGlow(crowded, many, new Float32Array(96).fill(1), g, 1, sprite); drawCores(crowdedCores, many, new Float32Array(96).fill(1), g);
  const a = crowded.calls.filter(c => c[0] === '=' && c[1] === 'globalAlpha').map(c => c[2]);
  assert.ok(Math.abs(a[0] - HALO_ALPHA * 0.75) < 1e-6, 'halo x 0.75');
  assert.ok(Math.abs(crowdedCores.calls.filter(c => c[0] === '=' && c[1] === 'globalAlpha')[0][2] - 0.5) < 1e-6, 'core x 0.5');
  assert.ok(Math.abs(crowdedCores.calls.filter(c => c[0] === 'arc')[0][3] - 3.5 * 0.8) < 1e-9);
  assert.equal(CROWD, 24);
  const b = glowBox(v.suspects, weights, g);
  assert.deepEqual(b, [77, 77, 98, 86], 'the halos of the three inside (x 79-173, y 79-161), 2 px more');
  assert.equal(glowBox(v.suspects, new Float32Array(4), g), null);
});

// ---- the corner ---------------------------------------------------------------------------------------------------------
test('choosing: one setting (far drift marked not tried yet), its value empty beside "now", the panel\'s lines with their hooks; Run never greyed out', () => {
  const s = setup();
  s.corner.show(model('form'));
  assert.equal(s.root.hidden, false); assert.equal(s.root.dataset.phase, 'form');
  assert.ok(s.root.matches('[data-qc-test-corner]') && s.root.matches('[data-qc-copy]'));
  const select = s.root.one('select'), input = s.root.one('input.qc-tc-value'), runButton = buttons(s.root, 'run')[0];
  assert.deepEqual(select.children.map(o => [o.value, o.textContent, o.dataset.tried]), [['cell-compactness', 'Cell compactness', 'true'],
    ['prior-weight', 'Hold to the starting cells', 'true'], ['diffusion-probability', 'Far drift', 'false']]);
  assert.equal(s.root.one('option[value=diffusion-probability][data-tried=false]'), select.children[2], 'the harness\'s hook');
  assert.equal(select.value, 'cell-compactness'); assert.equal(input.value, '', 'nothing prefilled');
  assert.match(s.root.visibleText(), /now 0\.04/); assert.doesNotMatch(s.root.visibleText(), /now now/, 'the panel\'s words as they are');
  assert.equal(s.root.one('.qc-tc-title').textContent, 'Test a setting'); assert.equal(s.root.one('.qc-tc-sub').textContent, 'Cell compactness · ffaaaaaa-1');
  assert.equal(s.doc.activeElement, input, 'the value is ready to type');
  assert.equal(runButton.textContent, 'Run the test'); assert.equal(runButton.disabled, false); assert.equal(runButton.title, CORNER_TEXT.runTooltip);
  assert.equal(buttons(s.root, 'second').length, 0, 'one setting a test'); assert.equal(s.root.one('select[id=qc-tc-setting2]'), null);
  // The pool line with its numbers as data (what a test needs in its tooltip), the area with its rule, the runs.
  const pool = s.root.one('[data-pool][data-cells][data-suspects][data-need]');
  assert.equal(pool, s.root.one('.qc-tc-pool')); assert.deepEqual([pool.textContent, pool.title], [POOL_LINE.text, POOL_LINE.tooltip]);
  assert.doesNotMatch(pool.textContent, /needs/, 'what a test needs: the tooltip only');
  assert.match(readFileSync(new URL('../src/qc-test-corner.css', import.meta.url), 'utf8'), /\.qc-tc-pool\[title\]:not\(\[title=''\]\) \{ cursor:help; \}/);
  assert.deepEqual([pool.dataset.pool, pool.dataset.cells, pool.dataset.suspects, pool.dataset.need], ['reached', '23', '187', '160']);
  assert.deepEqual([s.root.one('.qc-tc-area').textContent, s.root.one('.qc-tc-area').dataset.area], ['Area: a 500 × 500 µm square around this cell', 'square']);
  assert.equal(s.root.one('.qc-tc-runs').textContent, 'Your run waits · 3 runs as now, 3 with this value · about 45 s');
  assert.equal(s.root.one('.qc-tc-runs').title, C.testHow, 'how the runs are made: the runs line\'s tooltip (the panel\'s How line), never a note');
  assert.match(readFileSync(new URL('../src/qc-test-corner.css', import.meta.url), 'utf8'),
    /\.qc-tc-runs\[title\]:not\(\[title=''\]\), \.qc-tc-runs-line\[title\]:not\(\[title=''\]\) \{ cursor:help; \}/, 'its cursor says it has one');
  assert.equal(visible(s.root.one('.qc-tc-reasons')), false, 'no reason: no list');
  // Far drift chosen: its note, what it does in the note's tooltip.
  s.corner.show(model('form', { form: FORM({ flag: 'diffusion-probability' }) }));
  const note = s.root.one('.qc-tc-note');
  assert.deepEqual([visible(note), note.textContent, note.title], [true, C.testFarDriftNote, C.testFarDriftTip]);
  assert.match(s.root.visibleText(), /now 0\.2/);
  s.corner.show(model('form')); assert.equal(visible(s.root.one('.qc-tc-note')), false, 'the other settings carry no note');
  // The hold set live since the run's last iteration (F5): "now" as it is, what the test compares with in its tooltip
  // (the panel's nowTip); none for another setting, nor for a hold unchanged since.
  const nowTip = "The test compares your value with 1, your run's value at its last iteration.";
  const since = SETTINGS.map(x => x.flag === 'prior-weight' ? { ...x, now: 1.5, nowText: 'now 1.5', nowTip, from: 1 } : x);
  s.corner.show(model('form', { form: FORM({ settings: since, flag: 'prior-weight' }) }));
  const now = s.root.one('.qc-tc-now');
  assert.deepEqual([visible(now), now.textContent, now.title], [true, 'now 1.5', nowTip]);
  assert.match(readFileSync(new URL('../src/qc-test-corner.css', import.meta.url), 'utf8'), /\.qc-tc-now\[title\]:not\(\[title=''\]\) \{ cursor:help; \}/,
    'its cursor says it has one');
  s.corner.show(model('form', { form: FORM({ settings: since }) })); assert.deepEqual([now.textContent, now.title], ['now 0.04', '']);
  s.corner.show(model('form', { form: FORM({ flag: 'prior-weight' }) })); assert.deepEqual([now.textContent, now.title], ['now 1', '']);
  s.corner.show(model('form'));
  // While the pool is found: "Finding…" (its numbers not known: no numbers as data); it could not be counted; none.
  s.corner.show(model('form', { form: FORM({ pool: { state: 'asking', text: C.testPoolAsking, cells: null, suspects: null, need: null } }) }));
  assert.deepEqual([pool.textContent, pool.dataset.pool, 'cells' in pool.dataset], [C.testPoolAsking, 'asking', false]);
  assert.equal(s.root.one('[data-pool][data-cells][data-suspects][data-need]'), null);
  s.corner.show(model('form', { form: FORM({ pool: { state: 'failed', text: C.testPoolFailed, cells: null, suspects: null, need: null } }) }));
  assert.deepEqual([pool.textContent, pool.dataset.pool], [C.testPoolFailed, 'failed']);
  s.corner.show(model('form', { form: FORM({ pool: null, area: null, runs: '' }) }));
  assert.deepEqual([visible(pool), 'pool' in pool.dataset, visible(s.root.one('.qc-tc-area')), visible(s.root.one('.qc-tc-runs'))], [false, false, false, false]);
  // Why a test can't run here: the panel's words, each by its code, in their groups; then what a press said.
  const reasons = [{ code: 'ended', text: 'This run ended before its last iteration' }, { code: 'no_suspects', text: 'No suspects to test in this cell: none glows, or none could be placed' }];
  s.corner.show(model('form', { form: FORM({ pool: null, reasons, notHere: C.testNotHereCant }) }));
  assert.deepEqual(s.root.all('li[data-reason]').map(li => [li.dataset.reason, li.textContent]), reasons.map(r => [r.code, r.text]));
  assert.deepEqual(s.root.all('li[data-reason]').map(li => li.title ?? ''), ['', ''], 'no tooltip of its own');
  assert.equal(visible(s.root.one('.qc-tc-reasons')), true);
  // At a run's end counted for the hold, each reason's tooltip is the panel's (testAveraged: the count the test reads),
  // and the list is written again when only it changes.
  const tip = QC_COPY('testAveraged', { i: '450' });
  s.corner.show(model('form', { form: FORM({ pool: null, reasons: reasons.map(r => ({ ...r, tip })), notHere: C.testNotHereCant }) }));
  assert.deepEqual(s.root.all('li[data-reason]').map(li => [li.dataset.reason, li.textContent, li.title]), reasons.map(r => [r.code, r.text, tip]));
  s.corner.show(model('form', { form: FORM({ pool: null, reasons, notHere: C.testNotHereCant }) }));
  assert.deepEqual(s.root.all('li[data-reason]').map(li => li.title ?? ''), ['', '']);
  assert.equal(runButton.disabled, false, 'never greyed out');
  const problems = s.root.one('.qc-tc-problems');
  assert.equal(problems.hidden, false); assert.equal(problems.getAttribute('role'), 'alert');
  assert.deepEqual([problems.all('p').map(p => p.textContent), problems.all('li').length], [[C.testNotHereCant], 0], 'one line, nothing to check');
  // What Run still needs on the page: under "Check these first:", the field marked.
  s.corner.show(model('form', { form: FORM({ value: '', problems: ['Value to test: type a number.'], field: 'value' }) }));
  assert.ok(input.classList.contains('needs-input')); assert.equal(input.getAttribute('aria-invalid'), 'true');
  assert.deepEqual([problems.all('p').map(p => p.textContent), problems.all('li').map(li => li.textContent)],
    [[CORNER_TEXT.problemsLead], ['Value to test: type a number.']]);
  assert.equal(s.root.all('li[data-reason]').length, 0, 'the reasons go with the panel\'s');
  // The presses go to the panel.
  select.value = 'diffusion-probability'; select.onchange();
  input.focus(); input.value = '0.3'; input.oninput();
  runButton.click();
  assert.deepEqual(s.actions, [['setting', 'diffusion-probability'], ['value', '0.3'], ['run']]);
  // The panel's echo never moves the caret while typing; its own change lands when the field is not being typed in.
  s.corner.show(model('form', { form: FORM({ flag: 'diffusion-probability', value: '' }) }));
  assert.equal(input.value, '0.3'); assert.equal(select.value, 'diffusion-probability');
  s.doc.activeElement = null; s.corner.show(model('form', { form: FORM({ flag: 'diffusion-probability', value: '' }) }));
  assert.equal(input.value, '', 'another setting: its field empty again');
  assert.equal(buttons(s.root, 'close').length, 1, 'Close in the header');
  assert.equal(s.counts.qcTestViewFetches, 0); assert.equal(s.counts.qcTestCornerDraws, 0, 'no views while choosing');
  for (const t of ownTexts(s)) plainWords(t, 'form');
});

test('why a test can\'t run here: up to two groups, each under its heading with what it means in a tooltip; waits and caveats only in Run\'s tooltip, their codes as data', () => {
  const s = setup();
  const groups = () => s.root.all('.qc-tc-reason-group').filter(visible).map(g => [g.dataset.group, g.one('.qc-tc-reason-head').textContent,
    g.one('.qc-tc-reason-head').title, g.all('li[data-reason]').map(li => [li.dataset.reason, li.textContent])]);
  const words = kind => REASONS.filter(e => e.kind === kind).map(e => ({ code: e.code, text: e.shown }));
  // Every refusal the engine has (the shared fixture, in its order and the panel's words) and one the page doesn't know
  // (a newer engine's, last): "First:" (a test of this run on its way), then "Can't test here:", each line as it is.
  const all = [...words('refuse'), { code: 'a_new_reason', text: C.testReasonUnknown }];
  s.corner.show(model('form', { form: FORM({ pool: null, reasons: all }) }));
  const shown = groups();
  assert.deepEqual(shown.map(g => g.slice(0, 3)), [['first', 'First:', CORNER_TEXT.reasonsFirstTooltip], ['cant', "Can't test here:", CORNER_TEXT.reasonsCantTooltip]]);
  assert.deepEqual(shown.map(g => g[3].map(([code]) => code)), [['busy'],
    ['ending', 'ended', 'no_counts', 'old_proseg', 'check_unavailable', 'not_judged', 'no_suspects', 'same_type', 'no_primary', 'no_controls',
      'a_new_reason']]);
  for (const [code, text] of [...shown[0][3], ...shown[1][3]]) assert.equal(text, all.find(r => r.code === code).text, code);
  assert.equal(s.root.all('li[data-reason]').length, all.length, 'each refusal once, by its code');
  assert.equal(s.root.all('[data-group=unmeasured]').length, 0, 'nothing listed for not being measured');
  for (const t of ownTexts(s)) plainWords(t, 'reasons');
  // What a press would wait for first and where the test was not measured: never on screen; Run's tooltip says them
  // after what it does, a line each, and carries their codes (the harness reads them).
  const runButton = buttons(s.root, 'run')[0];
  assert.deepEqual([runButton.title, runButton.dataset.waits, runButton.dataset.caveats], [CORNER_TEXT.runTooltip, '', ''], 'none: Run\'s own words, empty hooks');
  s.corner.show(model('form', { form: FORM({ waits: words('wait'), caveats: words('caveat') }) }));
  assert.equal(runButton.title, [CORNER_TEXT.runTooltip, ...words('wait').map(r => r.text), ...words('caveat').map(r => r.text)].join('\n'));
  assert.deepEqual([runButton.dataset.waits, runButton.dataset.caveats], [words('wait').map(r => r.code).join(' '), words('caveat').map(r => r.code).join(' ')]);
  assert.ok(s.root.one('button[data-action=run][data-waits][data-caveats]'), 'the harness\'s hook');
  const seen = s.root.visibleText();
  for (const r of [...words('wait'), ...words('caveat')]) assert.ok(!seen.includes(r.text), `${r.code}: not on screen`);
  assert.equal(visible(s.root.one('.qc-tc-reasons')), false, 'no refusal: no list');
  assert.equal(runButton.disabled, false);
  for (const t of ownTexts(s)) plainWords(t, 'tooltip');
  // A cells-only start on a large region, its run not at its end yet: one wait and one caveat.
  s.corner.show(model('form', { form: FORM({ waits: [{ code: 'not_at_end', text: 'Starts once your run reaches its last iteration' }],
    caveats: [{ code: 'square_not_measured', text: 'Not measured yet for cells-only starts on regions over 500 µm' }] }) }));
  assert.equal(runButton.title, `${CORNER_TEXT.runTooltip}\nStarts once your run reaches its last iteration\nNot measured yet for cells-only starts on regions over 500 µm`);
  assert.deepEqual([runButton.dataset.waits, runButton.dataset.caveats], ['not_at_end', 'square_not_measured']);
  // A group shows only with a reason in it; without any, no list; a press with busy listed says to wait for it.
  s.corner.show(model('form', { form: FORM({ notHere: C.testNotHereFirst, reasons: [{ code: 'busy', text: 'A test of this run is on its way' }] }) }));
  assert.deepEqual(groups().map(g => g[0]), ['first']);
  assert.deepEqual(s.root.one('.qc-tc-problems').all('p').map(p => p.textContent), [C.testNotHereFirst]);
  assert.doesNotMatch(s.root.visibleText(), /Check these first/);
  s.corner.show(model('form'));
  assert.deepEqual([visible(s.root.one('.qc-tc-reasons')), s.root.all('.qc-tc-reason-group').length, visible(s.root.one('.qc-tc-problems'))], [false, 0, false]);
});

test('a test that waits: one line (what it waits for first, every one in its tooltip, their codes as data) and Cancel; no views, nothing asked or painted', async () => {
  assert.deepEqual(PHASES, TEST_PHASES, 'the panel\'s phases');
  assert.deepEqual(PHASES, ['form', 'starting', 'waiting', 'preparing', 'running', 'comparing', 'done', 'failed', 'cancelled']);
  const s = setup();
  s.corner.show(model('form')); s.corner.show(model('waiting')); await run(s, 2000);
  assert.equal(s.root.dataset.phase, 'waiting'); assert.ok(s.root.matches('[data-qc-test-corner][data-phase=waiting]'), 'the harness\'s hook');
  const line = s.root.one('p.qc-tc-waiting[data-waiting]');
  assert.deepEqual([visible(line), line.textContent, line.dataset.waiting, line.getAttribute('role')],
    [true, 'Starts once your run reaches its last iteration', 'not_at_end check_off', 'status']);
  assert.equal(line.title, 'Starts once your run reaches its last iteration\nSwitches the mixing check on first');
  assert.match(readFileSync(new URL('../src/qc-test-corner.css', import.meta.url), 'utf8'), /\.qc-tc-waiting\[title\]:not\(\[title=''\]\) \{ cursor:help; \}/);
  // Cancel (its tooltip: no run of its own yet), the title of a test on its way; no Close (a test on its way stops with
  // Cancel only), no form, no views, no answer.
  assert.deepEqual(buttons(s.root, 'cancel').map(b => [b.textContent, b.title]), [['Cancel', C.testCancelWaitingTooltip]]);
  assert.deepEqual(['close', 'run', 'new', 'keep', 'again'].map(a => buttons(s.root, a).length), [0, 0, 0, 0, 0]);
  assert.equal(s.root.one('.qc-tc-title').textContent, 'Testing a setting'); assert.equal(s.root.one('.qc-tc-sub').textContent, 'Cell compactness · ffaaaaaa-1');
  assert.deepEqual(['.qc-tc-form', '.qc-tc-watch', '.qc-tc-answer', '.qc-tc-ended'].map(sel => visible(s.root.one(sel))), [false, false, false, false]);
  assert.equal(s.root.all('canvas').filter(visible).length, 0, 'no views until its runs start');
  assert.deepEqual([s.asked.length, s.counts.qcTestViewFetches, s.counts.qcTestCornerDraws], [0, 0, 0], 'nothing asked or painted while it waits');
  s.root.emit('keydown', { key: 'Escape' }); assert.deepEqual(s.actions, [], 'Escape does not close it');
  // Small, as a test that stopped; the strip says what it waits for.
  assert.deepEqual([s.corner.box().w, s.corner.box().h], SIZES.waiting); assert.deepEqual(SIZES.waiting, [FORM_WIDTH, SIZES.ended[1]]);
  assert.match(readFileSync(new URL('../src/qc-test-corner.css', import.meta.url), 'utf8'), /\.qc-tc\[data-phase=waiting\][^{]*\{ width:360px; \}/);
  buttons(s.root, 'minimise')[0].click();
  assert.equal(s.root.one('.qc-tc-sub').textContent, 'Starts once your run reaches its last iteration'); buttons(s.root, 'minimise')[0].click();
  // The presses: Cancel goes to the panel once until it answers; a refused one says why beside the line.
  buttons(s.root, 'cancel')[0].click(); buttons(s.root, 'cancel')[0].click();
  assert.deepEqual(s.actions, [['cancel']]);
  s.corner.show(model('waiting', { busy: 'cancel', busyText: C.testCancelling }));
  assert.deepEqual([buttons(s.root, 'cancel')[0].textContent, buttons(s.root, 'cancel')[0].getAttribute('aria-busy')], [C.testCancelling, 'true']);
  s.corner.show(model('waiting', { message: C.testCancelFailed }));
  const alert = s.root.one('.qc-tc-alert');
  assert.deepEqual([visible(alert), alert.textContent, alert.parentElement], [true, C.testCancelFailed, s.root.one('.qc-tc-waiting-box')]);
  // What it waits for changes (the check switched on); then nothing holds: the cell's pool being worked out (the panel's
  // words); a model without words reads "Starting…".
  s.corner.show(model('waiting', { waiting: WAITING('not_at_end') }));
  assert.deepEqual([line.textContent, line.title, line.dataset.waiting], ['Starts once your run reaches its last iteration', '', 'not_at_end']);
  s.corner.show(model('waiting', { waiting: WAITING() }));
  assert.deepEqual([line.textContent, line.dataset.waiting], [C.testPoolAsking, '']);
  s.corner.show(model('waiting', { waiting: { text: '', tooltip: '', codes: [] } }));
  assert.deepEqual([line.textContent, line.dataset.waiting], [CORNER_TEXT.starting, '']);
  for (const t of ownTexts(s)) plainWords(t, 'waiting');
  // Its runs start: the views, the progress and Cancel naming the runs; the line goes, with its hook.
  s.corner.show(model('preparing'));
  assert.deepEqual([visible(line), visible(s.root.one('.qc-tc-watch')), buttons(s.root, 'cancel').map(b => b.title)], [false, true, [TESTED.cancelTooltip]]);
  assert.deepEqual(['waiting' in line.dataset, s.root.all('[data-waiting]').length], [false, 0]);
  await run(s, 1000); assert.ok(s.asked.length > 0, 'the views asked for once it runs');
  // Entered from the form, a waiting test is another test: the frames kept before go.
  const o = setup();
  o.corner.show(model('running')); await run(o, 400); assert.ok(o.corner.snapshot().frames > 0);
  o.corner.show(model('done')); await run(o, 400); o.corner.show(model('form')); o.corner.show(model('waiting'));
  assert.equal(o.corner.snapshot().frames, 0);
});

test('each phase renders from the model: starting, the runs at work (the views by side, the progress by side with the schedule\'s parts, the runs), a test that stopped', async () => {
  const s = setup();
  s.corner.show(model('starting'));
  assert.equal(s.root.dataset.phase, 'starting');
  assert.match(s.root.visibleText(), /Starting…/); assert.equal(buttons(s.root, 'close').length, 0, 'a test on its way stops with Cancel only');
  assert.equal(buttons(s.root, 'cancel').length, 0, 'Cancel once the test exists'); assert.equal(s.root.all('canvas').filter(visible).length, 2);
  s.corner.show(model('preparing'));
  const cancel = buttons(s.root, 'cancel')[0];
  assert.deepEqual([cancel.textContent, cancel.title], ['Cancel', TESTED.cancelTooltip], 'Cancel names its runs');
  assert.match(s.root.visibleText(), /Waiting for the first iteration…/); assert.match(s.root.visibleText(), /Preparing the runs…/);
  s.corner.show(model('running'));
  await run(s, 100);
  const text = s.root.visibleText();
  // The views: one run as now, one with the value, each by its side (the harness's hooks).
  assert.equal(s.corner.parts.control.name.textContent, 'As now'); assert.equal(s.corner.parts.test.name.textContent, 'With 0.02');
  assert.equal(s.root.one('canvas[data-role=control][data-side=A]'), s.corner.parts.control.canvas);
  assert.equal(s.root.one('canvas[data-role=test][data-side=B]'), s.corner.parts.test.canvas);
  assert.equal(s.root.one('.qc-tc-title').textContent, 'Testing a setting'); assert.equal(s.root.one('.qc-tc-sub').textContent, 'Cell compactness · ffaaaaaa-1');
  assert.equal(s.root.one('[data-suspects-in=control]').textContent, 'Suspects kept: 3 of 4');
  assert.equal(s.root.one('[data-suspects-in=test]').textContent, 'Suspects kept: 1 of 4');
  assert.equal(s.root.one('[data-suspects-in=control]').title, CORNER_TEXT.suspectsTooltip, 'exact: the plain tooltip');
  assert.deepEqual([s.root.one('[data-suspects-in=control]').dataset.count, s.root.one('[data-suspects-in=control]').dataset.held], ['3', 'exact']);
  assert.match(text, /Proseg's count: 519/); assert.match(text, /Proseg's count: 475/);
  assert.equal(s.root.one('.qc-tc-total').title, CORNER_TEXT.totalTooltip);
  // The progress: the engine's line (the slowest run's iteration of the end, the schedule's part), a bar per side from
  // the slowest run of each, the schedule's parts marked; the runs.
  assert.match(text, /Iteration 40 of 450 · settling/);
  assert.equal(s.corner.fills.control.style.width, `${(40 / 450 * 100).toFixed(1)}%`); assert.equal(s.corner.fills.test.style.width, `${(42 / 450 * 100).toFixed(1)}%`);
  assert.deepEqual([s.corner.fills.control.dataset.side, s.corner.fills.test.dataset.side], ['A', 'B']);
  assert.deepEqual(s.corner.marks.map(m => m.style.left), [`${(200 / 450 * 100).toFixed(2)}%`, `${(400 / 450 * 100).toFixed(2)}%`]);
  assert.match(s.root.one('.qc-tc-progress-line').title, /settling to iteration 200, refining to 400, then final settling to 450/);
  assert.equal(s.root.one('.qc-tc-runs-line').textContent, TESTED.runsText);
  assert.equal(s.root.one('.qc-tc-runs-line').title, C.testHow, 'the same line, the same tooltip as in the form');
  assert.equal(visible(s.root.one('.qc-tc-retry')), false);
  s.corner.show(model('running', { progress: PROGRESS({ iter: 300, sides: { A: 300, B: 310 } }) }));
  assert.match(s.root.visibleText(), /Iteration 300 of 450 · refining/);
  // Another schedule: its own end and marks.
  s.corner.show(model('running', { progress: PROGRESS({ iter: 100, of: 900, marks: [400, 800], sides: { A: 100, B: 120 } }) }));
  assert.deepEqual(s.corner.marks.map(m => m.style.left), ['44.44%', '88.89%']); assert.equal(s.corner.fills.test.style.width, '13.3%');
  // The runs started again without factorisation: said in a muted line under the progress, which takes its room.
  const h = s.corner.box().h;
  s.corner.show(model('running', { progress: PROGRESS({ retry: true, phase: 'loading', iter: 0, sides: { A: 0, B: 0 } }) }));
  assert.deepEqual([visible(s.root.one('.qc-tc-retry')), s.root.one('.qc-tc-retry').textContent], [true, C.testRetry]);
  assert.equal(s.corner.box().h, h + CHROME_PX.line);
  s.corner.show(model('comparing', { progress: PROGRESS({ phase: 'reading', iter: 450, done: 6, sides: { A: 450, B: 450 } }, 'comparing') }));
  assert.match(s.root.visibleText(), /Reading the runs…/); assert.equal(s.corner.fills.control.style.width, '100.0%');
  // A test that stopped: the message, Close.
  s.corner.show(model('failed', { message: 'Your run ended, so the test stopped.' }));
  assert.match(s.root.visibleText(), /Test stopped/); assert.match(s.root.visibleText(), /Your run ended, so the test stopped\./);
  assert.equal(s.root.one('.qc-tc-watch').hidden, true);
  buttons(s.root, 'close').at(-1).click(); assert.deepEqual(s.actions.at(-1), ['close']);
  s.corner.show(model('cancelled'));
  assert.match(s.root.visibleText(), /The test was cancelled\. Your settings are unchanged\./);
  for (const phase of ['form', 'starting', 'running', 'done', 'failed']) { s.corner.show(model(phase)); for (const t of ownTexts(s)) plainWords(t, phase); }
  s.corner.show({ open: false });
  assert.equal(s.root.hidden, true); assert.deepEqual(s.corner.texts(), []);
});

test('the answer, part by part with its hooks: the engine\'s verdict on its root, each part its verdict has, its word, why, the shrink line, the cost rows, the went bars', async () => {
  const s = setup();
  s.corner.show(model('running')); await run(s, 50);
  for (const { name, result: r } of EXAMPLES) {
    const a = RESULT(name);
    s.corner.show(model('done', { result: a })); await run(s, 30);
    const answer = s.root.one('.qc-tc-answer');
    assert.equal(s.root.one(`[data-verdict=${r.verdict}]`), answer, `${name}: the engine's verdict on the answer's root`);
    const shown = PARTS.filter(k => visible(s.root.one(`section[data-part=${k}]`)));
    assert.deepEqual(shown, PARTS.filter(k => a[k]), name);
    if (a.back) {
      const back = s.root.one('section[data-part=back]');
      assert.equal(back.dataset.word, r.come_back.passed ? 'yes' : 'no', name);
      assert.equal(back.one('.qc-tc-part-title').textContent, C.testBackTitle); assert.equal(back.one('.qc-tc-word').textContent, a.back.answer);
      assert.deepEqual([back.one('.qc-tc-text').textContent, back.one('.qc-tc-text').title], [a.back.text, a.back.tip], name);
    }
    const fewer = s.root.one('section[data-part=fewer]');
    if (a.fewer) {
      assert.deepEqual([fewer.dataset.word, fewer.dataset.why, 'shrink' in fewer.dataset], [a.fewer.word, a.fewer.why, a.fewer.shrink], name);
      assert.deepEqual([fewer.one('.qc-tc-part-title').textContent, fewer.one('.qc-tc-verdict').textContent, fewer.one('.qc-tc-text').textContent,
        fewer.one('.qc-tc-why').textContent], [a.fewer.title, a.fewer.head, a.fewer.text, a.fewer.whyText], name);
      // Its ⓘ: how often the test is wrong, then where it was not measured when the example says (its own caveats).
      const info = fewer.one('.qc-tc-info');
      assert.deepEqual([visible(info), info.title, info.getAttribute('aria-label')],
        [true, a.fewer.tip, a.caveats.length ? CORNER_TEXT.aboutCaveatsLabel : CORNER_TEXT.aboutLabel], name);
      assert.equal(a.fewer.tip, [a.calibration, ...a.caveats.map(x => x.text)].join('\n'), name);
    }
    if (a.cost) {
      const cost = s.root.one('section[data-part=cost]');
      assert.deepEqual(cost.all('[data-row]').map(row => row.dataset.row), COST_ROWS, `${name}: three rows, no suspects row`);
      assert.deepEqual(cost.all('[data-row]').map(row => [row.one('.qc-tc-row-label').textContent, row.one('.qc-tc-row-value').textContent,
        row.one('.qc-tc-row-change').textContent]), a.cost.rows.map(x => [x.label, x.value, x.change]), name);
      assert.equal('shrink' in cost.dataset, a.cost.shrink, name); assert.equal(visible(cost.one('.qc-tc-shrink')), a.cost.shrink, name);
      if (a.cost.shrink) assert.equal(cost.one('.qc-tc-shrink').textContent, C.testShrink);
    }
    if (a.went) {
      for (const side of ['A', 'B']) {
        const bar = s.root.one(`.qc-tc-went[data-side=${side}]`), want = a.went.sides.find(x => x.side === side);
        assert.deepEqual(WENT.map(k => bar.one(`[data-went=${k}]`).style.width), want.parts.map(p => `${p.percent}%`), `${name} ${side}`);
        assert.equal(WENT.reduce((n, k) => n + parseFloat(bar.one(`[data-went=${k}]`).style.width), 0), 100, `${name} ${side}: the whole bar`);
        assert.equal(bar.one('.qc-tc-bar').title, want.text);
        assert.equal(bar.one('.qc-tc-went-side').textContent, side === 'A' ? 'As now' : 'With 0.02');
      }
      assert.deepEqual(s.root.one('.qc-tc-went-key').all('[data-went]').map(i => [i.dataset.went, i.textContent]),
        a.went.sides[0].parts.map(p => [p.key, p.label]));
    }
    if (a.alone) assert.equal(s.root.one('section[data-part=alone]').textContent, a.alone.text);
    if (a.toofew) {
      const toofew = s.root.one('section[data-part=toofew]');
      assert.deepEqual([toofew.textContent, toofew.dataset.why], [a.toofew.text, r.why], name);
    }
    // The answers: New run with this, Keep my setting, Run again; none the main one.
    for (const action of ['new', 'keep', 'again']) {
      const b = buttons(s.root, action)[0];
      assert.deepEqual([b.textContent, b.title], [ANSWERS[action].text, ANSWERS[action].tooltip], `${name} ${action}`);
      assert.equal(b.classList.contains('qc-tc-primary'), false, `${name} ${action}: nothing is suggested`);
    }
    assert.equal(buttons(s.root, 'use').length, 0);
  }
  // The one-sentence shrink case says it in the fewer part only; the suspects-alone rule its own words.
  s.corner.show(model('done', { result: RESULT('clearly_fewer_shrinking') }));
  assert.ok('shrink' in s.root.one('section[data-part=fewer]').dataset); assert.equal('shrink' in s.root.one('section[data-part=cost]').dataset, false);
  s.corner.show(model('done', { result: RESULT('clearly_fewer_suspects_rule') }));
  assert.deepEqual([s.root.one('section[data-part=fewer]').dataset.word, s.root.one('.qc-tc-verdict').textContent], ['fewer', C.testFewerAloneYes]);
  assert.ok('shrink' in s.root.one('section[data-part=cost]').dataset);
  // After Run again: the earlier answer's word and why a repeat is no stronger answer, above the parts.
  s.corner.show(model('done', { result: RESULT('not_clear_apart', { repeat: { text: 'Earlier test with 0.02: clearly fewer.', note: C.testRepeatNote } }) }));
  const repeat = s.root.one('.qc-tc-repeat');
  assert.deepEqual([visible(repeat), repeat.one('.qc-tc-repeat-text').textContent, repeat.one('.qc-tc-repeat-note').textContent],
    [true, 'Earlier test with 0.02: clearly fewer.', C.testRepeatNote]);
  assert.equal(s.root.one('.qc-tc-answer').children.indexOf(repeat), 0, 'first');
  s.corner.show(model('done')); assert.equal(visible(repeat), false);
  // No answer the panel reads: the answers alone, no verdict on the root.
  s.corner.show(model('done', { result: { verdict: null, back: null, fewer: null, cost: null, went: null, alone: null, toofew: null, calibration: '', repeat: null, ...ANSWERS } }));
  assert.equal(s.root.one('[data-verdict]'), null); assert.deepEqual(PARTS.filter(k => visible(s.root.one(`section[data-part=${k}]`))), []);
  assert.equal(buttons(s.root, 'again').length, 1);
  s.corner.show(model('done', { result: null }));
  assert.equal(buttons(s.root, 'new').length, 0, 'no words for it: not offered'); assert.equal(buttons(s.root, 'again').length, 1);
  // Every word of the corner's own in the answer passes the rules; the views keep their size.
  s.corner.show(model('done')); for (const t of ownTexts(s)) plainWords(t, 'done');
  assert.equal(s.root.one('canvas[data-role=control]').width, VIEW_PX, 'the views keep their size at the answer'); assert.equal(DONE_VIEW_PX, VIEW_PX);
});

// The short pools' sentence (E3) the fixture's rows with short_rates (cells, as_is) give a pool of k 160 under too_few.
const SHORT_SENTENCE = 'Tests with fewer than 160 suspects, like this one, were measured apart, on only 8 test cells of one sample: about 3 in 100 '
  + 'tests of identical runs came out "clearly fewer" (4 in 100 at worst), and a drop of the suspects alone by 25 % was caught about 2 times in 10.';

test('where the test was not measured, at the answer: tooltips only (the fewer part\'s ⓘ after how often it is wrong, else the back part\'s own, the too-few line\'s), the codes on the root', async () => {
  const s = setup();
  s.corner.show(model('running')); await run(s, 50);
  const caveats = [{ code: 'whole_sample', words: 'x' }, { code: 'too_few', have: 121, need: 160, words: 'x' }];
  const said = 'Not measured yet for runs of 20,000 cells or more\nIts numbers are for tests with 160 suspects or more: 121 here';
  const answer = s.root.one('.qc-tc-answer'), backInfo = s.root.one('section[data-part=back]').one('.qc-tc-info');
  const fewerInfo = s.root.one('section[data-part=fewer]').one('.qc-tc-info'), toofewText = s.root.one('section[data-part=toofew]').one('.qc-tc-text');
  for (const { name, result } of EXAMPLES) {
    const a = RESULT(name, testReadout({ value: result.setting.value, result: { ...result, caveats } }));
    s.corner.show(model('done', { result: a })); await run(s, 30);
    assert.equal(s.root.one('[data-verdict][data-caveats]'), answer, name); assert.equal(answer.dataset.caveats, 'whole_sample too_few', name);
    if (a.fewer) {
      // E3: a row with short pools' own numbers (cells, as_is) adds them after the too_few caveat.
      const short = result.constants?.short_rates ? SHORT_SENTENCE : '';
      assert.deepEqual([visible(fewerInfo), fewerInfo.title, visible(backInfo)], [true, [a.calibration, said, short].filter(Boolean).join('\n'), false], name);
      assert.equal(fewerInfo.getAttribute('aria-label'), CORNER_TEXT.aboutCaveatsLabel, `${name}: its label names both`);
    } else if (a.back) {
      assert.deepEqual([visible(backInfo), backInfo.title, backInfo.getAttribute('aria-label'), backInfo.tabIndex], [true, said, CORNER_TEXT.caveatsLabel, 0], name);
    }
    if (a.toofew) assert.equal(toofewText.title, said, name);
    for (const words of ['Not measured yet', 'suspects or more']) assert.ok(!s.root.visibleText().includes(words), `${name}: never on screen`);
    for (const t of ownTexts(s)) plainWords(t, name);
  }
  // Measured as it is: the hook empty, no ⓘ of the back part's, no tooltip on the too-few line, the fewer part's ⓘ named
  // for how often the test is wrong alone again.
  s.corner.show(model('done', { result: RESULT('comes_and_goes') }));
  assert.deepEqual([answer.dataset.caveats, visible(backInfo), backInfo.title], ['', false, '']);
  s.corner.show(model('done', { result: RESULT('not_clear_apart') }));
  assert.deepEqual([fewerInfo.title, fewerInfo.getAttribute('aria-label')], [RESULT('not_clear_apart').calibration, CORNER_TEXT.aboutLabel]);
  // ... and, with no measured rates to say how often it is wrong, named for where it was not measured alone.
  const r = RESULTS.not_clear_apart, unrated = { ...r, constants: { ...r.constants, rates: null }, caveats };
  s.corner.show(model('done', { result: RESULT('not_clear_apart', testReadout({ value: r.setting.value, result: unrated })) }));
  assert.deepEqual([visible(fewerInfo), fewerInfo.title, fewerInfo.getAttribute('aria-label')], [true, said, CORNER_TEXT.caveatsLabel]);
  s.corner.show(model('done', { result: RESULT('too_few_no_controls') }));
  assert.equal(toofewText.title, '');
  // The engine's own examples where it was not measured: the codes on the root, the words in the tooltips.
  s.corner.show(model('done', { result: RESULT('too_few_cell_lost') }));
  assert.deepEqual([answer.dataset.caveats, toofewText.title], ['too_few', 'Its numbers are for tests with 160 suspects or more: 0 here']);
  s.corner.show(model('done', { result: RESULT('not_clear_unmeasured') }));
  assert.equal(answer.dataset.caveats, 'whole_sample nuclei_start area_narrow');
  assert.deepEqual([fewerInfo.getAttribute('aria-label'), fewerInfo.title.split('\n').slice(1)], [CORNER_TEXT.aboutCaveatsLabel,
    ['Not measured yet for runs of 20,000 cells or more', 'Not measured yet for runs started from nuclei only',
      'Not measured yet for regions over 500 µm long and under 500 µm wide']]);
  for (const words of ['Not measured yet', 'suspects or more']) assert.ok(!s.root.visibleText().includes(words), 'never on screen');
  // No answer: no hook.
  s.corner.show(model('running')); assert.equal('caveats' in answer.dataset, false);
});

test('the cells in the page\'s colours: the colour of their cell in your run when the view names it (colorOf), else their own; recolor', async () => {
  const asked = [], parents = [7, 1007, 9, 1009, 3, 1003];
  const colorOf = label => { asked.push(label); return label === 1007 ? '#e69f00' : label === 1009 ? 'rgb(0, 114, 178)' : 'nonsense'; };
  const s = setup({ colorOf, reply: () => viewReply({ control: { parents }, test: { parents } }) });
  s.corner.show(model('running')); await run(s, 150);
  const fills = paths(s.corner.frames.control[0].comp.canvas.getContext().calls).filter(p => p.op === 'fill').map(p => p.fillStyle);
  assert.ok(fills.includes(`rgba(230,159,0,${COLORS.fill[3]})`), 'the cell: the colour of its cell in your run');
  assert.ok(fills.includes(`rgba(0,114,178,${COLORS.fill[2]})`), 'the neighbour (rgb() read too)');
  assert.ok(fills.includes(`rgba(${identityRGB(1003).join(',')},${COLORS.fill[1]})`), 'a colour the page could not give: the stage\'s own for that cell');
  assert.deepEqual([...new Set(asked)].sort((a, b) => a - b), [1003, 1007, 1009], 'asked by the labels of your run only');
  assert.equal(s.corner.snapshot().colors, 'stage'); assert.match(s.root.one('.qc-tc-key-cell').title, /colour it has on the image/);
  // The page's colours changed (Colour by, new clusters): drawn again, asked again.
  const before = asked.length, draws = s.counts.qcTestCornerDraws;
  s.corner.recolor(); await run(s, 200);
  assert.ok(asked.length > before && s.counts.qcTestCornerDraws > draws);
  // The view names no cell of your run: each in a colour of its own; colorOf never asked with a label of a test run.
  const o = setup({ colorOf: () => { throw new Error('not asked'); } });
  o.corner.show(model('running')); await run(o, 150);
  const own = paths(o.corner.frames.control[0].comp.canvas.getContext().calls).filter(p => p.op === 'fill').map(p => p.fillStyle);
  assert.ok(own.includes(`rgba(${identityRGB(7).join(',')},${COLORS.fill[3]})`));
  assert.equal(o.corner.snapshot().colors, 'own'); assert.match(o.root.one('.qc-tc-key-cell').title, /colour of its own/);
  // Named, but no colorOf: the stage's own colour of the cell in your run (as the stage colours it by identity).
  const n = setup({ reply: () => viewReply({ control: { parents }, test: { parents } }) });
  n.corner.show(model('running')); await run(n, 150);
  const named = paths(n.corner.frames.control[0].comp.canvas.getContext().calls).filter(p => p.op === 'fill').map(p => p.fillStyle);
  assert.ok(named.includes(`rgba(${identityRGB(1007).join(',')},${COLORS.fill[3]})`)); assert.equal(n.corner.snapshot().colors, 'own');
});

test('the presses go to the panel; one pressed again before the panel answers is not sent twice', async () => {
  const s = setup();
  s.corner.show(model('done')); await run(s, 400);
  const keep = buttons(s.root, 'keep')[0];
  keep.click(); keep.click();
  assert.deepEqual(s.actions, [['keep']]); assert.equal(keep.getAttribute('aria-busy'), 'true');
  s.corner.show(model('done'));   // the panel answered (here: nothing changed)
  assert.equal(keep.getAttribute('aria-busy'), null); keep.click(); assert.equal(s.actions.length, 2);
  s.c.advance(3000); buttons(s.root, 'new')[0].click(); buttons(s.root, 'again')[0].click();
  assert.deepEqual(s.actions.slice(2), [['new'], ['again']]);
  s.corner.show(model('running')); buttons(s.root, 'cancel')[0].click();
  assert.deepEqual(s.actions.at(-1), ['cancel']);
  await run(s, 200);
  // Make smaller is the corner's own: the body folds away, the header stays; nothing goes to the panel.
  const small = buttons(s.root, 'minimise')[0];
  small.click();
  assert.ok(s.root.classList.contains('qc-tc-small')); assert.equal(s.root.one('.qc-tc-body').hidden, true);
  assert.equal(small.getAttribute('aria-pressed'), 'true'); assert.equal(small.title, 'Make bigger');
  assert.equal(s.root.one('.qc-tc-sub').textContent, 'Iteration 40 of 450 · settling', 'the strip says how far the test is');
  small.click(); assert.equal(s.root.one('.qc-tc-body').hidden, false);
  assert.ok(s.actions.every(([a]) => ACTIONS.includes(a)));
  assert.deepEqual(ACTIONS, ['setting', 'value', 'run', 'cancel', 'new', 'keep', 'again', 'close']);
  // Escape closes only where Close is offered.
  s.root.emit('keydown', { key: 'Escape' }); assert.notDeepEqual(s.actions.at(-1), ['close']);
  s.corner.show(model('failed')); s.root.emit('keydown', { key: 'Escape' }); assert.deepEqual(s.actions.at(-1), ['close']);
});

test('the panel\'s own marks: the press on its way, a refused press, the answers\' words', async () => {
  const s = setup();
  s.corner.show(model('running', { busy: 'cancel', busyText: 'Cancelling…' }));
  const cancel = buttons(s.root, 'cancel')[0];
  assert.equal(cancel.textContent, 'Cancelling…'); assert.equal(cancel.getAttribute('aria-busy'), 'true'); assert.equal(cancel.disabled, false);
  cancel.click(); assert.deepEqual(s.actions, [], 'not sent again while it is on its way');
  s.corner.show(model('running', { message: "Couldn't cancel the test. Try again." }));
  assert.equal(cancel.textContent, 'Cancel'); assert.equal(cancel.getAttribute('aria-busy'), null);
  const alert = s.root.one('.qc-tc-alert');
  assert.equal(visible(alert), true); assert.equal(alert.getAttribute('role'), 'alert'); assert.equal(alert.parentElement, s.root.one('.qc-tc-watch'));
  cancel.click(); assert.deepEqual(s.actions, [['cancel']]);
  // The form: the problems are the first alert (the harness reads it).
  s.corner.show(model('form', { form: FORM({ value: 'x', problems: ['Value to test: type a number.'], field: 'value' }) }));
  assert.equal(visible(s.root.one('.qc-tc-alert')), false); assert.equal(s.root.one('[role=alert]'), s.root.one('.qc-tc-problems'));
  s.corner.show(model('form')); const input = s.root.one('input.qc-tc-value');
  assert.equal(input.classList.contains('needs-input'), false); assert.equal(input.getAttribute('aria-invalid'), null);
  // The answer: the panel's words for its answers, Run again on its way; a refused Keep said beside the answer.
  s.corner.show(model('done', { result: RESULT('not_clear_apart', { again: { text: 'Run it again', tooltip: 'The same test once more.' } }),
    busy: 'again', busyText: 'Starting…', message: "Couldn't do that. Try again." }));
  await run(s, 400);
  const again = buttons(s.root, 'again')[0];
  assert.deepEqual([again.textContent, again.title, again.getAttribute('aria-busy')], ['Starting…', 'The same test once more.', 'true']);
  assert.equal(s.root.one('.qc-tc-alert').parentElement, s.root.one('.qc-tc-answer')); assert.match(s.root.one('.qc-tc-answer').visibleText(), /Couldn't do that/);
  // A stopped test says it in its own words, once.
  s.corner.show(model('failed', { message: 'Your run ended, so the test stopped.' }));
  assert.equal(visible(s.root.one('.qc-tc-alert')), false); assert.equal(s.root.visibleText().match(/Your run ended/g).length, 1);
});

test('the view is asked for at most 3 times a second while the test runs, one ask at a time, and once more at the answer', async () => {
  let seq = 0, state = 'running';   // the engine's word for the test: done once it has the answer
  const s = setup({ reply: () => viewReply({ seq: ++seq, state }) });
  s.corner.show(model('form')); await run(s, 1000);
  s.corner.show(model('starting')); await run(s, 1000);
  assert.equal(s.asked.length, 0, 'nothing before the test exists');
  s.corner.show(model('preparing')); await run(s, 3000);
  const times = s.asked.map(a => a[1]);
  assert.ok(times.length >= 8 && times.length <= 9, `${times.length} asks in 3 s`);
  for (const t of times) assert.ok(times.filter(u => u >= t && u < t + 1000).length <= 3, 'never more than 3 in a second');
  assert.ok(s.asked.every(([r]) => r === 'r1')); assert.equal(s.counts.qcTestViewFetches, times.length);
  // Hidden: none; shown again: asked again.
  s.doc.setHidden(true); const hiddenAt = s.asked.length; await run(s, 2000);
  assert.equal(s.asked.length, hiddenAt); assert.equal(s.c.timers.length, 0, 'nothing scheduled while hidden'); assert.equal(s.c.rafs.length, 0);
  s.doc.setHidden(false); await run(s, 400); assert.ok(s.asked.length > hiddenAt);
  // Covered by Compare's full-screen view (main.js): none either; uncovered: asked again.
  s.corner.setCovered(true); const coveredAt = s.asked.length; await run(s, 2000);
  assert.equal(s.asked.length, coveredAt); assert.equal(s.c.timers.length, 0, 'nothing scheduled while covered'); assert.equal(s.c.rafs.length, 0);
  s.corner.setCovered(false); await run(s, 400); assert.ok(s.asked.length > coveredAt);
  // The answer: one more ask (the views at the answer), then none.
  state = 'done'; s.corner.show(model('done')); const before = s.asked.length; await run(s, 3000);
  assert.equal(s.asked.length, before + 1, 'one last ask');
  // A test that stopped, or a closed corner: none.
  state = 'running'; s.corner.show(model('running')); s.corner.show(model('failed')); const stopped = s.asked.length; await run(s, 2000);
  assert.ok(s.asked.length - stopped <= 1, 'at most the ask already on its way');
  s.corner.show(model('running')); s.corner.close(); const closed = s.asked.length; await run(s, 2000);
  assert.equal(s.asked.length, closed, 'never while closed'); assert.equal(s.c.timers.length, 0); assert.equal(s.c.rafs.length, 0);
});

test('one ask at a time: a slow answer holds the next; the engine saying it ended stops the asks until the panel does', async () => {
  let release = null, n = 0;
  const s = setup({ fetchView: () => { n++; return new Promise(resolve => { release = resolve; }); } });
  s.corner.show(model('running')); await run(s, 3000);
  assert.equal(n, 1);
  release(viewReply({ seq: 3 })); await run(s, 400); assert.equal(n, 2);
  release(viewReply({ seq: 4, state: 'done' })); await run(s, 2000); assert.equal(n, 2, 'the engine said done: no more asks');
  s.corner.show(model('done')); await run(s, 1000); assert.equal(n, 3, 'the panel says done: the last ask');
  release(viewReply({ seq: 4, state: 'done' })); await run(s, 1000); assert.equal(n, 3);
  // An answer that arrives after the corner closed changes nothing.
  s.corner.show(model('running', { run: 'r2' })); await run(s, 400); assert.equal(n, 4);
  s.corner.close(); release(viewReply({ seq: 9 })); await run(s, 400);
  assert.equal(s.corner.snapshot().frames, 0); assert.equal(n, 4);
});

test('the views at the answer are asked for until the engine\'s last view comes: again a little later, a few times, then said', async () => {
  // The one ask at the answer once failed and the views stayed at iteration 40 for good, though the engine kept those
  // at 160.
  let asks = 0;
  const s = setup({ fetchView: async () => {
    asks++;
    if (asks === 2) throw new Error('the connection failed once');
    return viewReply({ seq: asks === 1 ? 40 : 160, state: asks === 1 ? 'running' : 'done' });
  } });
  s.corner.show(model('running')); await run(s, 20);
  s.corner.show(model('done')); await run(s, 5000);
  assert.equal(asks, 3, 'asked again after the failure, then no more');
  assert.equal(s.corner.snapshot().shown.control.iter, 160); assert.equal(s.corner.snapshot().shown.test.iter, 160);
  assert.equal(s.corner.snapshot().stale, false); assert.equal(visible(s.root.one('.qc-tc-stale')), false);
  // Every ask at the answer fails: FINAL_TRIES asks there, RETRY_MS after a failure and twice as long each time, then
  // none however long the corner stays open (never a loop); the last frame kept stays and the views say it may be older.
  const times = [];
  const f = setup({ fetchView: async () => {
    times.push(f.c.t);
    if (times.length === 1) return viewReply({ seq: 40 });
    throw new Error('the engine is gone');
  } });
  f.corner.show(model('running')); await run(f, 20);
  f.corner.show(model('done')); await run(f, 9000);
  assert.equal(times.length, 1 + FINAL_TRIES, 'the ask while running, then FINAL_TRIES at the answer');
  times.slice(2).forEach((t, i) => {
    const gap = t - times[i + 1], want = RETRY_MS * 2 ** i;
    assert.ok(gap >= want && gap <= want + 25, `retry ${i + 1}: ${gap} ms after the failed ask`);
  });
  assert.equal(f.counts.qcTestViewFetches, times.length);
  await run(f, 60000, 500);
  assert.equal(times.length, 1 + FINAL_TRIES, 'never again'); assert.equal(f.corner.timers.has('ask'), false);
  const snap = f.corner.snapshot();
  assert.equal(snap.stale, true); assert.equal(snap.shown.control.iter, 40, 'the last frame kept stays on screen');
  const note = f.root.one('.qc-tc-stale');
  assert.equal(visible(note), true); assert.equal(note.parentElement, f.root.one('.qc-tc-watch'), 'under the views');
  assert.equal(note.textContent, CORNER_TEXT.stale); assert.equal(note.title, CORNER_TEXT.staleTooltip);
  assert.ok(f.corner.texts().includes(CORNER_TEXT.stale) && f.corner.texts().includes(CORNER_TEXT.staleTooltip), 'the harness reads its words');
  // Its own spacing (it once sat flush under the views) and its room in the corner's size.
  assert.equal(note.className, 'qc-tc-stale');
  assert.match(readFileSync(new URL('../src/qc-test-corner.css', import.meta.url), 'utf8'), /\.qc-tc \.qc-tc-stale \{ margin:6px 0 0;/);
  assert.equal(f.corner.box().h, SIZES.wide[1] + CHROME_PX.line, 'the answer beside the views, and the note under them');
  for (const t of ownTexts(f)) plainWords(t, 'stale');
  // Run again: another test, nothing said of the old one's views.
  f.corner.show(model('starting'));
  assert.equal(f.corner.snapshot().stale, false); assert.equal(visible(f.root.one('.qc-tc-stale')), false);
  assert.equal(f.corner.box().h, SIZES.watch[1], 'its room goes with it');
  // A reply that is not the engine's last view (not a view, an engine's refusal, one that says the test still runs)
  // counts as a failed ask too and never replaces the frames kept (another list of suspects would have cleared them).
  const replies = [viewReply({ seq: 40 }), 'nonsense', { detail: 'No test for this run' }, viewReply({ seq: 41, state: 'running' }),
    viewReply({ seq: 160, state: 'done' })];
  let k = 0;
  const o = setup({ fetchView: async () => replies[Math.min(k++, replies.length - 1)] });
  o.corner.show(model('running')); await run(o, 20);
  o.corner.show(model('done')); await run(o, 9000);
  assert.equal(k, replies.length, 'three that are not the last view, then the last view: FINAL_TRIES asks at the answer');
  assert.deepEqual(o.corner.snapshot().kept, { control: 2, test: 2 }, 'iterations 40 and 160 only');
  assert.equal(o.corner.snapshot().shown.control.iter, 160); assert.equal(o.corner.snapshot().stale, false);
  await run(o, 20000, 500); assert.equal(k, replies.length);
});

test('frames: each distinct one kept and decoded once; a run that starts again drops what came after; the scrubber replays', async () => {
  const replies = [viewReply({ seq: 3 }), viewReply({ seq: 3 }), viewReply({ seq: 9, testSeq: 8 }), viewReply({ seq: 120, test: { held: [0, 0, 0, 0] } }),
    viewReply({ seq: 161, test: { held: [1, 0, 0, 0] }, state: 'done' })];
  let k = 0, decodes = 0;
  const atob = globalThis.atob;
  globalThis.atob = s => { decodes++; return atob(s); };
  try {
    const s = setup({ reply: () => replies[Math.min(k++, replies.length - 1)] });
    s.corner.show(model('running'));
    await run(s, 1200);
    assert.equal(k, 4, 'asks at 0, 350, 700 and 1050 ms');
    let snap = s.corner.snapshot();
    assert.deepEqual(snap.kept, { control: 3, test: 3 }, 'seq 3 once, 9 / 8, 120');
    assert.equal(decodes, 6, 'six frames, six decodes: the repeated one is not decoded again');
    assert.deepEqual(snap.shown.control, { seq: 120, iter: 120, voxel: 1, kept: 3, of: 4, held_by: 'exact', cell_total: 519, found: true, unsure: 0 });
    assert.equal(snap.shown.test.kept, 0);
    assert.match(s.root.visibleText(), /Iteration 40 of 450 · settling/, 'the engine\'s progress, the panel\'s words');
    s.corner.show(model('done')); await run(s, 800);
    snap = s.corner.snapshot();
    assert.deepEqual(snap.kept, { control: 4, test: 4 }); assert.equal(snap.steps, 4);
    assert.equal(snap.shown.control.seq, 161);
    const scrub = s.root.one('input[data-action=scrub]');
    assert.equal(visible(scrub), true); assert.equal(scrub.max, '3'); assert.equal(scrub.value, '3');
    assert.equal(visible(s.root.one('.qc-tc-progress')), false, 'the scrubber where the progress was');
    assert.match(s.root.visibleText(), /Iteration 161 · 161/);
    scrub.value = '1'; scrub.emit('input');
    snap = s.corner.snapshot();
    assert.equal(snap.scrub, 1); assert.equal(snap.shown.control.seq, 9); assert.equal(snap.shown.test.seq, 8, 'what both views showed after that ask');
    assert.match(s.root.visibleText(), /Iteration 9 · 8/);
    scrub.value = '3'; scrub.emit('input'); assert.equal(s.corner.snapshot().scrub, null, 'the end follows the newest again');
    // Replay: from the first step to the last, then it stops by itself.
    const replay = buttons(s.root, 'replay')[0];
    replay.click(); assert.equal(s.corner.snapshot().scrub, 0); assert.equal(replay.getAttribute('aria-pressed'), 'true');
    await run(s, REPLAY_MS * 4 + 50);
    assert.equal(s.corner.snapshot().scrub, null); assert.equal(replay.getAttribute('aria-pressed'), 'false');
    assert.equal(s.c.timers.filter(t => t.at > s.c.t + 1000).length, 0);
    // A run that starts again (frame numbers back to 2): the frames after it go, the scrubber follows.
    s.corner.keep(normaliseView(viewReply({ seq: 2 })).runs[0]);
    assert.deepEqual(s.corner.frames.control.map(f => f.seq), [2]);
    assert.ok(s.corner.steps.every(st => st.control == null || st.control === 2 || s.corner.frames.control.some(f => f.seq === st.control)));
    // Another test (Run again): nothing of the old one is kept.
    s.corner.show(model('starting'));
    assert.equal(s.corner.snapshot().frames, 0); assert.equal(s.corner.snapshot().steps, 0);
  } finally { globalThis.atob = atob; }
});

test('the views: the crop drawn at each run\'s grid, cores on the frame, halos on their own layer, the ones outside ringed; frames fade in', async () => {
  let seq = 4;
  const s = setup({ reply: () => viewReply({ seq: (seq += 1) }) });
  s.corner.show(model('running')); await run(s, 150);
  const canvas = s.root.one('canvas[data-role=control]'), ctx = canvas.getContext();
  const first = s.corner.frames.control[0];
  assert.ok(first.comp, 'drawn once and kept');
  const comp = first.comp.canvas.getContext(), g = geometry(BOX, VIEW_PX, 1);
  assert.ok(paths(comp.calls).some(p => p.op === 'stroke' && p.strokeStyle === COLORS.moved), 'the suspect outside: a teal ring');
  // The cores of the three inside are part of the frame's drawing (still while the halos breathe).
  const coreSprite = s.corner.sprites.core, cx = (125 - BOX[0]) * g.k - g.rc * 1.3;
  assert.ok(coreSprite && comp.calls.some(c => c[0] === 'drawImage' && c[1] === coreSprite && Math.abs(c[2] - cx) < 1e-9));
  assert.equal(comp.calls.filter(c => c[0] === 'drawImage' && c[1] === coreSprite).length, 3);
  assert.ok(ctx.calls.some(c => c[0] === 'drawImage' && c[1] === first.comp.canvas && c.length === 4), 'the frame on the view');
  // The halos are drawn once per frame on their own layer (one on the suspect inside at (125, 225) µm), then laid on the view.
  const hx = (125 - BOX[0]) * g.k - g.rh;
  assert.ok(first.glow, 'its halos kept');
  assert.ok(first.glow.canvas.getContext().calls.some(c => c[0] === 'drawImage' && c[1] === s.corner.sprites.halo && Math.abs(c[2] - hx) < 1e-9 && Math.abs(c[4] - 2 * g.rh) < 1e-9));
  const laid = ctx.calls.findIndex(c => c[0] === 'drawImage' && c[1] === first.glow.canvas);
  assert.ok(laid > 0 && ctx.calls.slice(0, laid).some(c => c[0] === '=' && c[1] === 'globalCompositeOperation' && c[2] === 'screen'));
  // A new frame fades in: the old one under the new one for a few paints, then the new alone.
  ctx.calls.length = 0;
  await run(s, 360);
  const second = s.corner.frames.control.at(-1);
  assert.notEqual(second, first);
  const under = ctx.calls.findIndex(c => c[0] === 'drawImage' && c[1] === first.comp.canvas);
  assert.ok(under >= 0 && ctx.calls.slice(under).some(c => c[0] === 'drawImage' && c[1] === second.comp.canvas), 'the frame before, under the new one');
  assert.ok(ctx.calls.some(c => c[0] === 'drawImage' && c[1] === first.glow.canvas), 'its halos fading out');
  const fades = [];
  for (let i = 0; i < ctx.calls.length - 1; i++) {
    if (ctx.calls[i][0] === '=' && ctx.calls[i][1] === 'globalAlpha' && ctx.calls[i + 1][0] === 'drawImage' && ctx.calls[i + 1][1] === second.comp.canvas) fades.push(ctx.calls[i][2]);
  }
  assert.ok(fades.some(a => a > 0 && a < 1), 'the new frame at part strength');
  assert.equal(s.corner.fades.control, null, 'the fade is over within the ask');
  // Reduced motion: no fade, one full paint per new frame.
  const r = setup({ reduced: true, reply: () => viewReply({ seq: (seq += 1) }) });
  r.corner.show(model('running')); await run(r, 150);
  const rctx = r.root.one('canvas[data-role=control]').getContext(); rctx.calls.length = 0;
  await run(r, 360);
  const comps = new Set(r.corner.frames.control.map(f => f.comp?.canvas).filter(Boolean));
  const draws = rctx.calls.filter(c => c[0] === 'drawImage' && comps.has(c[1]));
  assert.equal(draws.length, 1, 'the new frame alone');
});

test('the glow breathes at most 12 paints a second, just its patch, only while open, shown and visible; still under reduced or calm motion', async () => {
  const s = setup({ reply: () => viewReply({ seq: 5 }) });   // the same frame each time: only the glow moves
  s.corner.show(model('running')); await run(s, 500);
  const d0 = s.counts.qcTestCornerDraws;
  await run(s, 2000, 50);
  const perView = (s.counts.qcTestCornerDraws - d0) / 2 / 2;
  assert.ok(perView >= 10 && perView <= GLOW_FPS, `${perView} paints a second per view`);
  const snap = s.corner.snapshot();
  assert.equal(snap.glow.on, true); assert.ok(snap.glow.fps >= 10 && snap.glow.fps <= 12, `fps ${snap.glow.fps}`);
  // A breath repaints the glowing patch only (clipped), never the whole view.
  const ctx = s.root.one('canvas[data-role=control]').getContext(); ctx.calls.length = 0;
  await run(s, 300, 50);
  const rects = ctx.calls.filter(c => c[0] === 'rect');
  assert.ok(rects.length >= 2 && rects.every(c => c[3] < VIEW_PX && c[4] < VIEW_PX), 'clipped to the patch');
  assert.ok(ctx.calls.some(c => c[0] === 'clip')); assert.equal(ctx.calls.filter(c => c[0] === 'drawImage' && c.length === 4).length, 0);
  // Two image draws a breath (the kept drawing with its cores and the kept halos, the patch of each), however many suspects.
  const breaths = ctx.calls.filter(c => c[0] === 'clip').length, images = ctx.calls.filter(c => c[0] === 'drawImage').length;
  assert.equal(images, 2 * breaths); assert.equal(ctx.calls.filter(c => c[0] === 'arc').length, 0);
  const levels = ctx.calls.filter((c, i) => c[0] === '=' && c[1] === 'globalAlpha' && ctx.calls[i - 1]?.[2] === 'screen').map(c => c[2]);
  assert.ok(levels.length && levels.every(a => a >= BREATH_LOW - 1e-9 && a <= 1 + 1e-9), 'the halos between 60 and 100 %');
  // Made smaller, hidden, closed: nothing is drawn or scheduled for drawing.
  buttons(s.root, 'minimise')[0].click(); let d = s.counts.qcTestCornerDraws; await run(s, 1000, 50);
  assert.equal(s.counts.qcTestCornerDraws, d); assert.equal(s.corner.snapshot().glow.on, false);
  buttons(s.root, 'minimise')[0].click(); await run(s, 500, 50); assert.ok(s.counts.qcTestCornerDraws > d);
  s.doc.setHidden(true); d = s.counts.qcTestCornerDraws; await run(s, 1000, 50);
  assert.equal(s.counts.qcTestCornerDraws, d); assert.equal(s.c.timers.length + s.c.rafs.length, 0);
  s.doc.setHidden(false); await run(s, 500, 50); assert.ok(s.counts.qcTestCornerDraws > d);
  s.corner.setCovered(true); d = s.counts.qcTestCornerDraws; await run(s, 1000, 50);   // under Compare: as hidden
  assert.equal(s.counts.qcTestCornerDraws, d); assert.equal(s.c.timers.length + s.c.rafs.length, 0);
  s.corner.setCovered(false); await run(s, 500, 50); assert.ok(s.counts.qcTestCornerDraws > d);
  // At the answer it keeps breathing ; closed, it stops at once.
  s.corner.show(model('done')); await run(s, 1000, 50); d = s.counts.qcTestCornerDraws; await run(s, 1000, 50);
  assert.ok(s.counts.qcTestCornerDraws - d >= 18, 'still breathing at the answer');
  s.corner.close(); d = s.counts.qcTestCornerDraws; await run(s, 2000, 50);
  assert.equal(s.counts.qcTestCornerDraws, d); assert.equal(s.c.timers.length + s.c.rafs.length, 0);
  assert.deepEqual(s.corner.snapshot().glow, { on: false, fps: 0, still: false });
  // Reduced or calm motion: one paint per change, a still glow at full strength.
  for (const which of ['reduced', 'calm']) {
    const q = setup({ [which]: true, reply: () => viewReply({ seq: 5 }) });
    q.corner.show(model('running')); await run(q, 500);
    const q0 = q.counts.qcTestCornerDraws; await run(q, 2000, 50);
    assert.equal(q.counts.qcTestCornerDraws, q0, `${which}: no paint while nothing changes`);
    assert.equal(q0, 4, `${which}: one paint per view for each change (opened, the first frame)`);
    const g = q.corner.snapshot().glow; assert.equal(g.on, true); assert.equal(g.fps, 0); assert.equal(g.still, true);
    // Motion back on: the page says so (refresh) and the glow breathes again.
    q.motion[which] = false; q.corner.refresh(); await run(q, 1000, 50);
    assert.ok(q.counts.qcTestCornerDraws - q0 >= 18);
  }
  // No suspect proseg still counts for the cell: nothing to breathe.
  const none = setup({ reply: () => viewReply({ seq: 5, control: { held: [0, 0.1, 0, 0.19] }, test: { held: [0, 0, 0, 0] } }) });
  none.corner.show(model('running')); await run(none, 500);
  const n0 = none.counts.qcTestCornerDraws; await run(none, 1000, 50);
  assert.equal(none.counts.qcTestCornerDraws, n0); assert.equal(none.corner.snapshot().glow.on, false);
});

test('the snapshot and the counters', async () => {
  const s = setup();
  assert.deepEqual(COUNTERS, ['qcTestViewFetches', 'qcTestCornerDraws']);
  for (const name of COUNTERS) assert.equal(s.counts[name], 0, `${name} listed at 0`);
  let snap = s.corner.snapshot();
  assert.equal(snap.open, false); assert.equal(snap.phase, 'closed'); assert.equal(snap.frames, 0); assert.equal(snap.colors, null);
  assert.deepEqual(snap.shown, { control: null, test: null }); assert.deepEqual(snap.glow, { on: false, fps: 0, still: false });
  s.corner.show(model('running')); await run(s, 500);
  snap = s.corner.snapshot();
  assert.equal(snap.open, true); assert.equal(snap.phase, 'running'); assert.equal(snap.frames, 2); assert.equal(snap.suspects, 4);
  assert.deepEqual(Object.keys(snap.shown.control).sort(), ['cell_total', 'found', 'held_by', 'iter', 'kept', 'of', 'seq', 'unsure', 'voxel']);
  assert.equal(snap.shown.control.unsure, 0); assert.equal(snap.beside, false);
  assert.deepEqual([snap.shown.control.kept, snap.shown.control.of, snap.shown.control.held_by], [3, 4, 'exact']);
  assert.equal(snap.shown.test.cell_total, 475);
  assert.equal(snap.fetches, s.counts.qcTestViewFetches); assert.equal(snap.draws, s.counts.qcTestCornerDraws);
  assert.ok(snap.glow.on); assert.equal(snap.colors, 'own');
  // The cell missing from a run: said, never counted as none inside.
  const m = setup({ reply: () => viewReply({ test: { cell: null, total: null } }) });
  m.corner.show(model('running')); await run(m, 100);
  assert.equal(m.root.one('[data-suspects-in=test]').textContent, 'Not in this run');
  assert.equal(m.corner.snapshot().shown.test.found, false); assert.equal(m.corner.snapshot().shown.test.cell_total, null);
  assert.deepEqual([m.corner.snapshot().shown.test.kept, m.corner.snapshot().shown.test.of], [null, null]);
  const missing = m.corner.frames.test[0].comp.canvas.getContext();
  assert.ok(!paths(missing.calls).some(p => p.strokeStyle === COLORS.moved), 'no "moved out" rings for a cell that is not there');
  assert.ok(!paths(missing.calls).some(p => p.strokeStyle === '#ffffff'), 'and no white outline');
  // A suspect without a place counts as the engine says but never glows: nothing breathes for it.
  const odd = setup({ reply: () => ({ ...viewReply({ control: { held: [0, 0, 0, 1] }, test: { held: [0, 0, 0, 1] } }), suspects: [...SUSPECTS.slice(0, 3), { gene: 'X' }] }) });
  odd.corner.show(model('running')); await run(odd, 500);
  assert.equal(odd.corner.snapshot().shown.control.kept, 1); assert.equal(odd.corner.snapshot().glow.on, false);
  const o0 = odd.counts.qcTestCornerDraws; await run(odd, 1000, 50); assert.equal(odd.counts.qcTestCornerDraws, o0);
  // The view's own list is the one counted (the engine's, of the cell): none before the first view, none for an empty one.
  const empty = setup({ reply: () => ({ ...viewReply(), suspects: [] }) });
  empty.corner.show(model('running'));
  assert.equal(empty.root.one('[data-suspects-in=control]').hidden, true, 'before the first view: no count');
  await run(empty, 200); assert.equal(empty.root.one('[data-suspects-in=control]').hidden, true);
});

test('each suspect by proseg\'s share of it: glowing as bright as its likelihood times that share, a teal ring under 0.2, "Suspects kept" its shares added up, approximate said in the tooltip', async () => {
  // As now: shares 1, 0.2 (glows: the edge itself), 0.19 (a teal ring), unknown (null: neither, out of "of n"); with the
  // value: placed approximately (the frame came without proseg's export of the view).
  const s = setup({ reply: () => viewReply({ control: { held: [1, 0.2, 0.19, null] }, test: { held: [0.6, 0.6, 0.6, 0.6], by: 'approx' } }) });
  s.corner.show(model('running')); await run(s, 200);
  const count = role => s.root.one(`[data-suspects-in=${role}]`);
  assert.equal(count('control').textContent, 'Suspects kept: 1 of 3', '1 + 0.2 + 0.19 rounds to 1; the unknown one out of "of"');
  assert.deepEqual([count('control').title, count('control').dataset.held], [CORNER_TEXT.suspectsTooltip, 'exact']);
  assert.equal(count('test').textContent, 'Suspects kept: 2 of 4', '4 x 0.6 rounds to 2');
  assert.deepEqual([count('test').title, count('test').dataset.held], [`${CORNER_TEXT.suspectsTooltip} ${CORNER_TEXT.approxTooltip}`, 'approx']);
  const f = s.corner.frames.control[0];
  assert.deepEqual([...f.weights], [1, 0.2, 0, 0]); assert.deepEqual([...f.moved], [0, 0, 1, 0]); assert.equal(f.lit, 2);
  const rings = paths(f.comp.canvas.getContext().calls).find(x => x.strokeStyle === COLORS.moved);
  assert.equal(rings.arcs.length, 1, 'one teal ring: the 0.19');
  // Its halo as bright as its likelihood times its share (here 0.9 x 1 and 0.6 x 0.2), at the breath's top.
  const g = geometry(BOX, 240, 1), ctx = recordingContext();
  assert.equal(drawGlow(ctx, normaliseView(viewReply()).suspects, f.weights, g, 1, { sprite: true }), 2);
  const alphas = ctx.calls.filter(c => c[0] === '=' && c[1] === 'globalAlpha').map(c => c[2]);
  assert.ok(Math.abs(alphas[0] - HALO_ALPHA * 0.9) < 1e-6 && Math.abs(alphas[1] - HALO_ALPHA * 0.6 * 0.2) < 1e-6, String(alphas));
  // The snapshot: what "Suspects kept" says, and how.
  assert.deepEqual([s.corner.snapshot().shown.control.kept, s.corner.snapshot().shown.control.of, s.corner.snapshot().shown.test.held_by],
    [1, 3, 'approx']);
  // Once the runs ended no later iteration brings proseg's exact share, and the tooltip says what holds: while the
  // answer is worked out (comparing: every run done), and at the answer, the runs gone.
  s.corner.show(model('comparing')); await run(s, 200);
  assert.equal(count('test').title, `${CORNER_TEXT.suspectsTooltip} ${CORNER_TEXT.approxDoneTooltip}`);
  s.corner.show(model('done')); await run(s, 200);
  assert.equal(count('test').title, `${CORNER_TEXT.suspectsTooltip} ${CORNER_TEXT.approxDoneTooltip}`);
  assert.doesNotMatch(count('test').title, /later iteration/);
  // A view whose run holds the cell but placed none of its suspects says so, never nothing.
  const none = setup({ reply: () => viewReply({ test: { held: [null, null, null, null], by: 'approx' } }) });
  none.corner.show(model('running')); await run(none, 200);
  const line = none.root.one('[data-suspects-in=test]');
  assert.deepEqual([line.hidden, line.textContent, line.title], [false, CORNER_TEXT.notPlaced, CORNER_TEXT.notPlacedTooltip]);
  assert.equal(none.root.one('[data-suspects-in=control]').textContent, 'Suspects kept: 3 of 4', 'the other view as ever');
});

test('docked at the bottom left, moved by its header, kept inside the stage; its box goes to onBox, null when closed', () => {
  const s = setup();
  s.corner.show(model('running'));
  const [w, h] = SIZES.watch;
  assert.deepEqual(s.corner.box(), { left: MARGIN, top: 800 - MARGIN - h, bottom: MARGIN, w, h });
  // The stylesheet's width is the sum of the parts (the views with their borders, the gap, padding and borders), the same
  // at the answer (it reads under the views).
  assert.equal(WIDTH, 2 * (VIEW_PX + 2) + VIEW_GAP + FRAME_PX - 4); assert.equal(DONE_WIDTH, WIDTH);
  const sheet = readFileSync(new URL('../src/qc-test-corner.css', import.meta.url), 'utf8');
  assert.match(sheet, new RegExp(`width:${WIDTH}px`)); assert.doesNotMatch(sheet, /\[data-phase=done\][^{]*\{[^}]*width:/, 'no other width at the answer');
  assert.deepEqual(s.boxes.at(-1), s.corner.box());
  assert.equal(s.root.style.left, '12px'); assert.equal(s.root.style.bottom, '12px');
  const head = s.root.one('header');
  head.emit('pointerdown', { pointerId: 1, clientX: 100, clientY: 500 });
  head.emit('pointermove', { pointerId: 1, clientX: 300, clientY: 400 });
  assert.equal(s.corner.box().left, 212); assert.equal(s.corner.box().bottom, 112);
  head.emit('pointermove', { pointerId: 1, clientX: 5000, clientY: -5000 });
  assert.equal(s.corner.box().left, 1200 - WIDTH - MARGIN); assert.equal(s.corner.box().top, MARGIN, 'its header stays inside');
  head.emit('pointerup', { pointerId: 1 });
  head.emit('pointermove', { pointerId: 1, clientX: 0, clientY: 0 }); assert.equal(s.corner.box().top, MARGIN, 'only while dragged');
  head.emit('dblclick'); assert.equal(s.corner.box().left, MARGIN); assert.equal(s.corner.box().bottom, MARGIN);
  // A press on a header button is not a drag.
  head.emit('pointerdown', { pointerId: 2, clientX: 0, clientY: 0, target: s.root.one('button[data-action=minimise]') });
  head.emit('pointermove', { pointerId: 2, clientX: 300, clientY: 0 }); assert.equal(s.corner.box().left, MARGIN);
  // A narrow stage: smaller views; a short one: smaller views at the answer (everything else keeps its room).
  const n = setup({ width: 420 });
  n.corner.show(model('running'));
  assert.ok(n.root.one('canvas[data-role=control]').width < VIEW_PX);
  const t = setup({ width: WIDE_STAGE - 1, height: 620 });   // a stage too narrow for the answer beside the views
  t.corner.show(model('running')); assert.equal(t.root.one('canvas[data-role=control]').width, VIEW_PX);
  // While watching, a short stage gets smaller views (the corner never scrolls then).
  const short = setup({ width: 1200, height: 420 });
  short.corner.show(model('running'));
  const px = short.root.one('canvas[data-role=control]').width;
  assert.equal(px, 420 - 2 * MARGIN - CHROME_PX.watch); assert.ok(px < VIEW_PX && px >= MIN_VIEW_PX);
  assert.equal(short.corner.parts.control.fig.style.width, `${px + 2}px`, 'its title and numbers as wide as the view');
  // At the answer the views keep VIEW_PX, under the answer or beside it, however short the stage: the card's body
  // scrolls, the card never taller than the stage leaves it.
  t.corner.show(model('done'));
  assert.equal(t.root.one('canvas[data-role=control]').width, VIEW_PX); assert.equal(t.corner.box().h, 620 - 2 * MARGIN);
  for (const [w, h] of [[1440, 620], [1440, 400], [WIDE_STAGE - 1, 400]]) {
    const b = setup({ width: w, height: h });
    b.corner.show(model('done')); assert.equal(b.root.one('canvas[data-role=control]').width, VIEW_PX, `${w} x ${h}`);
    assert.ok(b.corner.box().h <= h - 2 * MARGIN, `${w} x ${h}: inside the stage`);
  }
  const css = readFileSync(new URL('../src/qc-test-corner.css', import.meta.url), 'utf8');
  assert.match(css, /\.qc-tc-body \{[^}]*overflow:auto/, 'the body scrolls'); assert.match(css, /max-height:calc\(100% - 24px\)/);
  s.corner.close(); assert.equal(s.boxes.at(-1), null); assert.equal(s.corner.box(), null);
  s.corner.dispose(); assert.equal(s.root.parentElement, null);
});

test('suspects the engine couldn\'t tell apart: a grey ring, out of every count, said once in a muted line', async () => {
  // Suspects 1 and 3 have another transcript of their gene at their place in both runs (runs[k].unsure): no share.
  const unsure = [0, 1, 0, 1];
  const s = setup({ reply: () => viewReply({ control: { held: [1, null, 0, null], unsure }, test: { held: [0, null, 0, null], unsure } }) });
  s.corner.show(model('running')); await run(s, 200);
  const count = role => s.root.one(`[data-suspects-in=${role}]`);
  assert.equal(count('control').textContent, 'Suspects kept: 1 of 2', 'of the 4 sent, the 2 told apart');
  assert.deepEqual([count('control').dataset.count, count('control').dataset.of], ['1', '2']);
  assert.deepEqual([count('test').textContent, count('test').dataset.count], ['Suspects kept: 0 of 2', '0']);
  const views = s.root.one('.qc-tc-watch').one('.qc-tc-unsure');
  assert.equal(visible(views), true); assert.equal(views.textContent, "2 couldn't be told apart from another transcript at the same place");
  assert.equal(views.title, CORNER_TEXT.unsureTooltip); assert.equal(views.children[0].tagName, 'I', 'led by its grey ring, as drawn');
  assert.equal(views.parentElement, s.root.one('.qc-tc-watch'), 'under the views');
  // The snapshot keeps the count kept (the harness compares it with the engine's) and says how many were unsure.
  const snap = s.corner.snapshot();
  assert.deepEqual([snap.shown.control.kept, snap.shown.control.of, snap.shown.control.unsure, snap.shown.test.unsure, snap.suspects], [1, 2, 2, 2, 4]);
  // Drawn as plain grey rings: never teal (no share either way), never glowing (one suspect glows in your run, none in
  // the test's; the teal rings are those proseg no longer counts for the cell).
  const comp = role => paths(s.corner.frames[role][0].comp.canvas.getContext().calls);
  for (const role of ROLES) {
    assert.equal(comp(role).find(p => p.strokeStyle === COLORS.unsure).arcs.length, 2, role);
    assert.equal(comp(role).find(p => p.strokeStyle === COLORS.moved).arcs.length, role === 'control' ? 1 : 2, role);
  }
  assert.deepEqual([s.corner.frames.control[0].lit, s.corner.frames.test[0].lit], [1, 0]);
  // Its line takes its room in the corner's size.
  assert.equal(s.corner.box().h, SIZES.watch[1] + CHROME_PX.line);
  for (const t of ownTexts(s)) plainWords(t, 'unsure');
  // At the answer the views keep their line (the answer says nothing of single suspects: its parts are the pool's).
  s.corner.show(model('done')); await run(s, 400);
  assert.equal(visible(views), true); assert.equal(s.root.one('.qc-tc-answer').one('.qc-tc-unsure'), null);
  for (const t of ownTexts(s)) plainWords(t, 'unsure answer');
  // None unsure: no line anywhere, every suspect counted.
  const n = setup();
  n.corner.show(model('running')); await run(n, 200);
  assert.equal(visible(n.root.one('.qc-tc-watch').one('.qc-tc-unsure')), false); assert.equal(n.root.one('[data-suspects-in=control]').dataset.of, '4');
  assert.equal(n.corner.box().h, SIZES.watch[1]);
});

test('the answer beside the views on a wide stage (783 wide, its column setting the card\'s height), under them on a narrow one; the hooks kept', async () => {
  let seq = 3;   // a frame while it runs, the engine's last at the answer: two steps for the scrubber
  const s = setup({ width: 1440, height: 795, reply: n => viewReply({ seq: (seq += 2), state: n > 1 ? 'done' : 'running' }) });
  s.corner.show(model('running')); await run(s, 100);
  assert.equal(s.root.classList.contains('qc-tc-wide'), false, 'watching: the views side by side, the progress under them');
  assert.deepEqual(s.corner.box(), { left: MARGIN, top: 795 - MARGIN - SIZES.watch[1], bottom: MARGIN, w: WIDTH, h: SIZES.watch[1] });
  s.corner.show(model('done')); await run(s, 400);
  assert.ok(s.root.classList.contains('qc-tc-wide')); assert.equal(s.corner.snapshot().beside, true);
  assert.deepEqual(s.corner.box(), { left: MARGIN, top: 795 - MARGIN - SIZES.wide[1], bottom: MARGIN, w: WIDE_WIDTH, h: SIZES.wide[1] });
  assert.deepEqual(s.boxes.at(-1), s.corner.box(), 'the page hears of it (the area chips, the legend)');
  assert.ok(SIZES.wide[1] < SIZES.done[1] && SIZES.wide[0] < 800, 'beside the views the card is shorter; the views keep 220 px');
  assert.equal(s.corner.watchBox.style.width, `${2 * (VIEW_PX + 2) + VIEW_GAP}px`, 'the views\' column as wide as the views');
  // Every hook the harness reads is there and shown, both columns at once.
  for (const sel of ['canvas[data-role=control][data-side=A]', 'canvas[data-role=test][data-side=B]', '[data-suspects-in=control]', '[data-suspects-in=test]',
    'input[data-action=scrub]', '[data-verdict=not_clear]', 'section[data-part=back][data-word=yes]', 'section[data-part=fewer][data-word=unclear][data-why=apart]',
    'section[data-part=cost]', '[data-row=own]', '[data-row=all]', '[data-row=perCell]', '.qc-tc-went[data-side=A] [data-went=stay]',
    '.qc-tc-went[data-side=B] [data-went=background]', 'section[data-part=went]', 'section[data-part=alone]']) {
    const node = sel.split(' ').reduce((at, part) => at?.one(part) ?? null, s.root);   // a descendant, one step at a time
    assert.equal(visible(node), true, sel);
  }
  for (const action of ['new', 'keep', 'again', 'close', 'minimise', 'replay']) assert.equal(buttons(s.root, action).length, 1, action);
  assert.equal(s.root.one('canvas[data-role=control]').width, VIEW_PX, 'the views keep their size (ui.qc_test_view_px >= 220)');
  assert.equal(s.root.dataset.phase, 'done');
  // Made smaller: the strip, never the wide card; bigger again: beside again.
  buttons(s.root, 'minimise')[0].click();
  assert.equal(s.root.classList.contains('qc-tc-wide'), false); assert.deepEqual([s.corner.box().w, s.corner.box().h], SIZES.small);
  buttons(s.root, 'minimise')[0].click(); assert.ok(s.root.classList.contains('qc-tc-wide'));
  // The stage narrows under WIDE_STAGE: the answer under the views; wide enough again: beside.
  s.host.clientWidth = WIDE_STAGE - 1; s.corner.layout();
  assert.equal(s.root.classList.contains('qc-tc-wide'), false);
  assert.deepEqual([s.corner.box().w, s.corner.box().h], [SIZES.done[0], Math.min(SIZES.done[1], 795 - 2 * MARGIN)], 'never taller than the stage leaves it');
  assert.equal(s.corner.watchBox.style.width, ''); assert.equal(s.corner.snapshot().beside, false);
  s.host.clientWidth = WIDE_STAGE; s.corner.layout(); assert.ok(s.root.classList.contains('qc-tc-wide'));
  // Run again: watching, no answer's column; closed: nothing of it kept.
  s.corner.show(model('starting')); assert.equal(s.root.classList.contains('qc-tc-wide'), false);
  s.corner.show(model('done')); assert.ok(s.root.classList.contains('qc-tc-wide'));
  s.corner.close(); assert.equal(s.root.classList.contains('qc-tc-wide'), false); assert.equal(s.corner.snapshot().beside, false);
  // The numbers the stylesheet holds: 486 + (14 + 1 + 14) + 268 = 783, on a stage of 783 + 2 x 12.
  assert.equal(WIDE_WIDTH, WIDTH + ANSWER_GAP + ANSWER_PX); assert.equal(WIDE_STAGE, WIDE_WIDTH + 2 * MARGIN); assert.equal(ANSWER_GAP, 14 + 1 + 14);
  assert.deepEqual([WIDTH, WIDE_WIDTH, WIDE_STAGE], [486, 783, 807]);
  const css = readFileSync(new URL('../src/qc-test-corner.css', import.meta.url), 'utf8');
  assert.match(css, new RegExp(`\\.qc-tc\\.qc-tc-wide \\{ width:${WIDE_WIDTH}px; \\}`));
  assert.match(css, /\.qc-tc\.qc-tc-wide \.qc-tc-answer \{[^}]*margin:0 0 0 14px; padding:0 0 0 14px;[^}]*border-left:1px solid/);
  assert.match(css, /\.qc-tc\.qc-tc-wide \.qc-tc-body \{ display:flex;/);
});

test('the corner stays inside the stage when the browser reports it bigger (the answer\'s column beside the views)', async () => {
  const observers = [], saved = globalThis.ResizeObserver;
  globalThis.ResizeObserver = class { constructor(fn) { this.fn = fn; observers.push(this); } observe() {} disconnect() {} };
  try {
    const s = setup({ width: 1440, height: 795 });
    const report = (w, h) => observers[0].fn([{ borderBoxSize: [{ inlineSize: w, blockSize: h }] }]);
    s.corner.show(model('running')); await run(s, 100); report(WIDTH, 439);
    const head = s.root.one('header');   // dragged to the stage's right edge
    head.emit('pointerdown', { pointerId: 1, clientX: 0, clientY: 0 }); head.emit('pointermove', { pointerId: 1, clientX: 5000, clientY: 0 });
    head.emit('pointerup', { pointerId: 1 });
    assert.equal(s.corner.box().left, 1440 - WIDTH - MARGIN);
    s.corner.show(model('done')); report(800, 430);   // the answer's column came: the browser reports the wider card
    assert.equal(s.corner.box().left, 1440 - 800 - MARGIN, 'moved back inside'); assert.equal(s.root.style.left, `${1440 - 800 - MARGIN}px`);
    assert.deepEqual([s.boxes.at(-1).left, s.boxes.at(-1).w, s.boxes.at(-1).h], [1440 - 800 - MARGIN, 800, 430]);
    // Closed, it forgets the size it had: opened again it waits for the browser's (else its usual one).
    s.corner.close(); s.corner.show(model('form')); assert.deepEqual([s.corner.box().w, s.corner.box().h], SIZES.form);
  } finally { if (saved) globalThis.ResizeObserver = saved; else delete globalThis.ResizeObserver; }
});

test('nothing in the corner\'s stylesheet moves by itself; the script names no CSS motion', () => {
  const css = readFileSync(new URL('../src/qc-test-corner.css', import.meta.url), 'utf8');
  assert.doesNotMatch(css, /@keyframes|animation|transition|backdrop-filter/i);
  assert.match(css, /\.qc-tc\b/); assert.match(css, /font-variant-numeric:tabular-nums/);
  for (const name of ['qc-test-corner.mjs', 'qc-test-corner-copy.mjs']) {
    const source = readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8');
    assert.doesNotMatch(source, /animation\s*:|transition\s*:|backdrop-filter|requestIdleCallback/i, name);
  }
  assert.ok(ROLES.length === 2 && VIEW_MS >= 334, 'at most 3 asks a second');
  assert.deepEqual([OF, [...MARKS]], [450, [200, 400]], "proseg's default schedule until the engine's progress says");
});

test('a setting offered as values to pick shows them as buttons in the value\'s place, the run\'s own one not to be picked', () => {
  const s = setup();
  const grid = { flag: 'burnin-voxel-size', name: 'Starting grid (µm)', kind: 'choice', now: 2, nowText: 'now 2 µm', tried: false, note: 'Not tried here yet.', tip: '',
    choices: [{ value: 1, text: '1 µm', now: false }, { value: 2, text: '2 µm', now: true }, { value: 3, text: '3 µm', now: false }] };
  const drift = { flag: 'no-diffusion', name: 'Drift (on or off)', kind: 'switch', now: 0, nowText: 'now on', tried: false, note: 'Not tried here yet.', tip: '',
    choices: [{ value: 1, text: 'off', now: false }, { value: 0, text: 'on', now: true }] };
  const layers = { flag: 'voxel-layers', name: 'Depth layers', kind: 'whole', now: 4, nowText: 'now 4', tried: false, note: 'Not tried here yet.', tip: '' };
  s.corner.show(model('form', { form: FORM({ settings: [...SETTINGS, grid, drift, layers], flag: 'burnin-voxel-size', value: '3' }) }));
  const box = s.corner.choiceBox, picks = box.children;
  assert.deepEqual([s.corner.input.hidden, box.hidden, box.dataset.kind], [true, false, 'choice']);
  assert.deepEqual(picks.map(b => [b.textContent, b.dataset.value, b.disabled, b.getAttribute('aria-checked')]),
    [['1 µm', '1', false, 'false'], ['2 µm', '2', true, 'false'], ['3 µm', '3', false, 'true']]);
  assert.equal(box.getAttribute('aria-label'), 'Values');
  box.onclick({ target: picks[0] }); box.onclick({ target: picks[1] });
  assert.deepEqual(s.actions.filter(a => a[0] === 'value'), [['value', '1']], 'a press sends its value; the run\'s own sends nothing');
  s.corner.show(model('form', { form: FORM({ settings: [...SETTINGS, grid, drift, layers], flag: 'no-diffusion', value: '' }) }));
  assert.deepEqual(box.children.map(b => [b.textContent, b.disabled]), [['off', false], ['on', true]]); assert.equal(box.dataset.kind, 'switch');
  s.corner.show(model('form', { form: FORM({ settings: [...SETTINGS, grid, drift, layers], flag: 'voxel-layers', value: '8' }) }));
  assert.deepEqual([s.corner.input.hidden, box.hidden, s.corner.input.step], [false, true, '1'], 'a whole number is typed, a step of 1');
  s.corner.show(model('form', { form: FORM({ settings: [...SETTINGS, grid, drift, layers], flag: 'cell-compactness', value: '0.02' }) }));
  assert.deepEqual([s.corner.input.hidden, box.hidden, s.corner.input.step], [false, true, 'any']);
});
